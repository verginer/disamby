# -*- coding: utf-8 -*-

"""Main module."""
from collections import Counter
from typing import Union
from pandas import DataFrame, Series
from math import log
from tqdm import tqdm
from networkx import DiGraph
from collections import namedtuple

ScoredElement = namedtuple('ScoredElement', ['index', 'score'])
PandasObj = Union[DataFrame, Series]


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
    >>> import disamby.preprocessors as pre
    >>> df = pd.DataFrame(
    ... {'a': ['Luca Georger', 'Luca Geroger', 'Adrian Sulzer'],
    ... 'b': ['Mira, 34, Augsburg', 'Miri, 34, Augsburg', 'Milano, 34']
    ... }, index=['L1', 'L2', 'O1']
    ... )
    >>> pipeline = [
    ...     pre.normalize_whitespace,
    ...     pre.remove_punctuation,
    ...     pre.trigram
    ... ]
    >>> dis = Disamby(df, pipeline)
    >>> dis.disambiguated_sets(threshold=0.5, verbose=False)
    [{'L2', 'L1'}, {'O1'}]
    """

    def __init__(self, data: PandasObj=None, preprocessors: list=None, field: str=None):
        self.field_freq = dict()
        self.preprocessors = dict()
        self.records = dict()
        self._processed_token_cache = dict()
        self._most_common = dict()
        self._token_to_instance = dict()
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
        """
        try:
            columns = data.columns
            for col in columns:
                self._fit_field(data[col], preprocessors=preprocessors)
        except AttributeError:
            self._fit_field(data, preprocessors=preprocessors, field=field)

    def find(self, idx, threshold=0.0, weights: dict=None, **kwargs) -> list:
        """
        returns the list of scored instances which have a score above the
        threshold. Note that strings which do not share any token are omitted
        since their score is 0 by default.

        Parameters
        ----------
        idx
            index of the record to find
        threshold
        weights : dict

        Returns
        -------

        """
        fields = self.fields

        scored_candidates = dict()
        if weights is None:
            weights = {f: 1 / len(fields) for f in fields}

        for field in fields:
            own_term = self.records[field][idx]
            own_tokens = self._processed_token_cache[field][own_term]
            potential_candidates = set()

            for token in own_tokens:
                potential_candidates |= self._token_to_instance[field][token]

            for candidate in potential_candidates:
                score = self.score(own_term, candidate[1], field, **kwargs)
                candidate_idx = candidate[0]
                if candidate_idx not in scored_candidates:
                    scored_candidates[candidate_idx] = {field: score * weights[field]}
                else:
                    scored_candidates[candidate_idx][field] = score * weights[field]

        # compute weighted score
        final_candidates = []
        for idx, scores in scored_candidates.items():
            total_score = sum(scores.values())
            if total_score >= threshold:
                final_candidates.append(ScoredElement(idx, total_score))

        return final_candidates

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
        self.records[field] = dict()
        counter = Counter()

        for i, name in data.items():
            norm_tokens = self.pre_process(name, preprocessors)
            self._processed_token_cache[field][name] = norm_tokens
            counter.update(norm_tokens)
            self.records[field][i] = name
            for token in norm_tokens:
                if token in self._token_to_instance[field]:
                    self._token_to_instance[field][token] |= {(i, name)}
                else:
                    self._token_to_instance[field][token] = {(i, name)}
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
        The score is not commutative (i.e. score(A,B) != score(B,A))
        """

        try:
            own_parts = self._processed_token_cache[field][term]
            other_parts = self._processed_token_cache[field][other_term]
        except KeyError:
            own_parts = self.pre_process(term, self.preprocessors[field])
            other_parts = self.pre_process(other_term, self.preprocessors[field])

        # get list of potential scores
        weights = self.id_potential(own_parts, field, smoother, offset)
        score = sum(weights.get(tok, 0) for tok in other_parts)
        return score

    def id_potential(self, term: Union[tuple, str], field: str,
                     smoother: str = None, offset=0) -> dict:
        """
        Computes the weights of the words based on the observed frequency
        and normalized.

        Parameters
        ----------
        term : str, tuple
            term to look for or a tuple of proper tokens
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
        if isinstance(term, str):
            words = self.pre_process(term, self.preprocessors[field])
        else:
            words = term

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
        id_potentials = {
            word: s_fun(counter[word], offset, max_occ)
            for word in words
        }

        total_weight = sum(id_potentials.values())
        return {w: idp / total_weight for w, idp in id_potentials.items()}

    def alias_graph(self, threshold=0.7, verbose=True, weights=None, **kwargs) -> DiGraph:
        """
        This function creates the directed network connecting an instance to an other
        through a directed edge if the the target instance has a similarity score above
        the threshold.

        Parameters
        ----------
        weights
        threshold : float
            between 0 and 1
        verbose : whether to show the progressbar
        kwargs :
            arguments to pass to the score function (i.e. offset, smoother)

        Returns
        -------
        DiGraph

        """
        if not verbose:
            t = lambda x: x
        else:
            t = tqdm

        edges = []
        fields = self.records.keys()
        a_field = list(fields)[0]
        for idx in t(self.records[a_field]):
            targets = self.find(idx, threshold=threshold, weights=weights, **kwargs)
            new_edges = [(idx, x.index, {'score': x.score}) for x in targets]
            edges.extend(new_edges)

        alias_graph = DiGraph()
        alias_graph.add_edges_from(edges)
        return alias_graph

    def disambiguated_sets(self, threshold=0.7, verbose=True, weights=None, **kwargs):
        from networkx import strongly_connected_components
        alias_graph = self.alias_graph(threshold, verbose, weights, **kwargs)
        strong = strongly_connected_components(alias_graph)
        return list(strong)

    @staticmethod
    def pre_process(base_name, functions: list):
        """apply every function consecutively to base_name"""
        norm_name = base_name
        for f in functions:
            norm_name = f(norm_name)
        return set(norm_name)

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


