# -*- coding: utf-8 -*-

"""Main module."""
from collections import Counter
from typing import TypeVar
from pandas import DataFrame, Series
from math import log


PandasObj = TypeVar('pandas', DataFrame, Series, list)


class Disamby(object):
    """
    Class for disambiguation fitting, scoring and ranking of potential matches

    A `Disamby` instance stores the pre-processing pipeline applied to the
    strings for a given field as well as the the computed frequencies from the
    entire corpus of strings to match against.
    `Disamby` can be instantiated either with not arguments, with a list of
    strings, pandas.Series or pandas.DataFrame. This triggers the immediate
    call to the `fit` method, whose doc explains the parameters.

    Examples
    --------

    >>> import pandas as pd
    >>> from disamby.preprocessors import split_words, normalize_whitespace
    >>> df = pd.DataFrame(
    ... {'a': ['Luca Georger', 'Luke Geroge', 'Adrian Sulzer'],
    ... 'b': ['Mira, 34, Augsburg', 'Miri, 34, Augsburg', 'Milano, 34']
    ... })
    >>> prep = [normalize_whitespace, split_words]
    >>> dis = Disamby(df, prep)
    >>> dis.pandas_score(0, df, smoother='log').values
    array([ 1.,  0.25,  0.])
    """

    def __init__(self, data: PandasObj=None, preprocessors: list=None, field: str=None):
        self.field_freq = dict()
        self.preprocessors = dict()
        self._processed_token_cache = dict()
        self._most_common = dict()
        self._token_to_instance = dict()
        # TODO: add lookup table per token (token->instance)
        if data is not None:
            if preprocessors is None:
                raise ValueError("Preprocessor not provided")
            self.fit(data, preprocessors, field)

    def fit(self, data: PandasObj, preprocessors: list, field: str=None):
        """
        Computes the frequencies of the terms by field.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Series or list of strings
            list of strings or pandas.DataFrame
            if dataframe is given then the field defaults to the column name
        preprocessors : list
            list of functions to apply in that order
            note the first function must accept a string, the other functions
            must be such that a pipeline is possible the result is a tuple of
            strings.
        field : str
            string identifying which field this data belongs to

        Examples
        --------

        >>> import pandas as pd
        >>> from disamby.preprocessors import split_words
        >>> df = pd.DataFrame(
        ... {'a': ['Luca Georger', 'Luke Geroge', 'Adrian Sulzer'],
        ... 'b': ['Mira, 34, Augsburg', 'Miri, 32', 'Milano, 34']
        ... })
        >>> dis = Disamby()
        >>> prep = [split_words]
        >>> dis.fit(df, prep)
        >>> df_scores = dis.pandas_score(0, df)
        >>> df_scores.values
        array([ 1.,  0.,  0.])
        """
        try:
            columns = data.columns
            for col in columns:
                self._fit_field(data[col], preprocessors=preprocessors)
        except AttributeError:
            self._fit_field(data, preprocessors=preprocessors, field=field)

    def pandas_score(self, index, data_frame,
                     smoother=None, offset=0, weight=None):
        """
        For the given term compute the score given the dataframe
        The column names of the dataframe are assumed to be the fields you
        are interested in

        Parameters
        ----------
        index :
            index of the entry you want to filter on
        data_frame
        weight : dict
            dict of the form {field1: w1, field2: w2}
            the weight is is the weight to associate to the field in the final
            score, defaults to 1/n
        smoother : str (optional)
            one of {None, 'offset', 'log'}
        offset : int
            offset to add to count only needed for smoothers 'log' and 'offset'

        Returns
        -------
        DataFrame
        """

        fields = data_frame.columns
        # TODO: make it work with series objects
        own_record = data_frame.loc[index]

        if set(fields) != set(self.field_freq.keys()):
            raise ValueError("Not all fields have been fitted (i.e. computed"
                             "their frequency.")

        if weight is None:
            weight = {f: 1 / len(fields) for f in fields}

        def scoring_fun(record):
            total_score = 0
            for field in fields:
                score = self.score(own_record[field],
                                   record[field], field,
                                   smoother=smoother,
                                   offset=offset)
                score *= weight[field]
                total_score += score
            return total_score

        return scoring_fun

    def _fit_field(self, data: PandasObj, preprocessors: list=None, field: str=None):
        if field not in self.preprocessors:
            ValueError('preprocessors have already been defined, '
                       'cannot redefine. This would render the lookup '
                       'inconsistent')

        if field is None:
            try:
                field = data.name
            except AttributeError:  # was not a pandas.Series
                raise KeyError("The provided data are not a pandas Series, "
                               "if the data is a list you need to provide the"
                               "`field` argument.")

        self.preprocessors[field] = preprocessors
        self._processed_token_cache[field] = dict()
        self._token_to_instance[field] = dict()
        counter = Counter()

        for i, name in enumerate(data):
            norm_tokens = self.pre_process(name, preprocessors)
            self._processed_token_cache[field][name] = norm_tokens
            counter.update(norm_tokens)
            for token in norm_tokens:
                if token in self._token_to_instance[field]:
                    self._token_to_instance[field].append((i, name))
                else:
                    self._token_to_instance[field] = [(i, name)]
        self._most_common[field] = counter.most_common(1)[0][1]
        self.field_freq[field] = counter

    def score(self, term: str, other_term: str, field: str,
              smoother=None, offset=0) -> float:
        """
        Computes the score between the two strings using the frequency data

        Parameters
        ----------
        term : str
            term to search for
        other_term : str
            the other term to compare too
        field : str
            the name of the column to which this term belongs
        smoother : str (optional)
            one of {None, 'offset', 'log'}
        offset : int
            offset to add to count only needed for smoothers 'log' and 'offset'

        Returns
        -------
        float

        Notes
        -----
        The score is not commutative (i.e. c(A,B)!=C(B,A))
        """
        try:
            own_parts = self._processed_token_cache[field][term]
            other_parts = self._processed_token_cache[field][other_term]
        except KeyError:
            own_parts = self.pre_process(term, self.preprocessors[field])
            other_parts = self.pre_process(term, self.preprocessors[field])

        weights = self.id_potential(own_parts, field, smoother, offset)
        score = 0
        for i, own in enumerate(own_parts):
            if own in other_parts:
                score += weights[i]
        return score

    def id_potential(self, words: tuple, field: str,
                     smoother: str = None, offset=0) -> tuple:
        """
        Computes the weights of the words based on the observed frequency
        and normalized.

        Parameters
        ----------
        words : list
        field : str
            field the word falls under
        smoother : str (optional)
            one of {None, 'offset', 'log'}
        offset : int
            offset to add to count only needed for smoothers 'log' and 'offset'

        Returns
        -------
        float
        """

        smoothers = {
            None: self._smooth_none,
            'offset': self._smooth_offset,
            'log': self._smooth_log
        }
        if smoother not in smoothers:
            raise KeyError(
                'Chosen smother `{}` is not one of {}'.format(
                    smoother, smoothers.keys())
            )
        counter = self.field_freq[field]

        s_fun = smoothers[smoother]
        max_occ = self._most_common[field]
        id_potentials = [
            s_fun(counter[word], offset, max_occ) for word in words
        ]

        total_weight = sum(id_potentials)
        return tuple(idp / total_weight for idp in id_potentials)

    @staticmethod
    def pre_process(base_name, functions: list):
        """apply every function consecutively to base_name"""
        norm_name = base_name
        for f in functions:
            norm_name = f(norm_name)
        return norm_name

    @property
    def fields(self):
        return self.field_freq.keys()

    @staticmethod
    def _smooth_none(occurrences, *args):
        return 1 / max(occurrences, 1)

    @staticmethod
    def _smooth_offset(occurrences, offset, *args):
        return 1 / max(occurrences + offset, 1)

    @staticmethod
    def _smooth_log(occ, offset, max_occ):
        max_offset = max(max_occ + offset + 1, 1)
        word_offset = max(occ + offset, 1)
        return log(max_offset / word_offset)
