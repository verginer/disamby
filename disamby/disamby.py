# -*- coding: utf-8 -*-

"""Main module."""
from typing import List, Union, Callable

Records = List[tuple]


class Disamby(object):
    def __init__(self, fields):
        self.fields = fields
        self.field_freq = {field: None for field in fields}
        self.preprocessors = None

    def fit(self, records: Union[Records, list], preprocessors: list=None):
        """

        Parameters
        ----------
        records :
            list of tuples of the form (field1, field2) or
            a list of strings if only one field is needed
        preprocessors : list
            list of functions
        """
        for prep in preprocessors:
            prep()
        # TODO: build up the frequency table for each table
        return True
