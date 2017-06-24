# -*- coding: utf-8 -*-

"""Main module."""
from collections import Counter
from pandas import DataFrame


class Disamby(object):
    def __init__(self):
        self.fields = None
        self.field_freq = dict()
        self.preprocessors = dict()

    def fit(self, field: str, values: list, preprocessors: list=None):
        """

        Parameters
        ----------
        field : str
            string identifying which field this data belongs to
        values : list
            list of strings
        preprocessors : list
            list of functions to apply in that order
            note the first function must accept a string, the other functions
            must be such that a pipeline is possible the result is a tuple of
            strings.
        """
        if field not in self.preprocessors:
            ValueError('preprocessors have already been defined, '
                       'cannot redefine. This would render the lookup '
                       'inconsistent')

        self.preprocessors[field] = preprocessors

        counter = Counter()

        for name in values:
            norm_values = self.pre_process(name, preprocessors)
            counter.update(norm_values)

        self.field_freq[field] = counter

    def identification_weight(self, words: tuple, field: str):
        counter = self.field_freq[field]
        id_potentials = [1/(counter[word]+1) for word in words]
        total_weight = sum(id_potentials)
        return tuple(idp/total_weight for idp in id_potentials)

    def score(self, term: str, other_term: str, field: str) -> float:
        funcs = self.preprocessors[field]
        own_parts = self.pre_process(term, funcs)
        other_parts = self.pre_process(other_term, funcs)

        weights = self.identification_weight(own_parts, field)
        score = 0
        for i, own in enumerate(own_parts):
            if own in other_parts:
                score += weights[i]
        return score

    def score_df(self, term: str, data_frame: DataFrame):
        """
        For the given term compute the score given the dataframe
        The column names of the dataframe are assumed to be the fields you
        are interested in

        Parameters
        ----------
        term
        data_frame

        Returns
        -------

        """

        # fields = data_frame.columns
        pass

    @staticmethod
    def pre_process(base_name, functions: list):
        norm_name = base_name
        for f in functions:
            norm_name = f(norm_name)
        return norm_name
