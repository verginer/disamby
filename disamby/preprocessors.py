# -*- coding: utf-8 -*-

"""
This module contains the various string preprocessors
"""
import re

__all__ = ['compact_abbreviations', 'normalize_whitespace',
           'ngram', 'trigram', 'split_words', 'remove_punctuation',
           'nword']


_re_abbreviation = re.compile(r'\.(?![a-zA-Z]*})')
_re_duplicate_white = re.compile('\s+')
_re_whitespace = re.compile('\s')
_re_punctuation = re.compile('[^\w\s]')


def normalize_whitespace(string: str) -> str:
    """
    removes duplicates whitespaces as well as replace tabs and newlines with a space

    Parameters
    ----------
    string

    Returns
    -------
    str

    Examples
    --------
    >>> normalize_whitespace('this is a  \t long  string')
    'THIS IS A LONG STRING'
    """
    no_whitespace = _re_duplicate_white.sub(' ', string.upper())
    return no_whitespace.strip()


def compact_abbreviations(string: str) -> str:
    """
    Removes dots between single letters and concatenates them

    Parameters
    ----------
    string

    Returns
    -------
    str

    Examples
    --------
    >>> compact_abbreviations('an other A.B.M this')
    'AN OTHER ABM THIS'
    """
    split = _re_abbreviation.split(string.upper())
    return ''.join(split)


def remove_punctuation(word: str) -> str:
    """
    removes all punctuation symbols from the string

    Parameters
    ----------
    word: str

    Returns
    -------
    str

    Examples
    --------
    >>> remove_punctuation('.has -a .few!')
    'has a few'
    """
    return _re_punctuation.sub('', word)


def ngram(string: str, n: int) -> tuple:
    """
    constructs all possible ngrams from the given string. If the string is shorter then
    the n then the string is returned

    Parameters
    ----------
    string
    n : int
        value must be larger at least 2

    Returns
    -------
    tuple of strings

    Examples
    --------
    >>> ngram('this', 2)
    ('th', 'hi', 'is')
    """
    N = len(string)
    if n > N:
        return string,
    if n < 2:
        raise ValueError('n for a ngram must be 2 or larger')
    return tuple(string[i:i+n] for i in range(N-n+1))


def trigram(string: str) -> tuple:
    return ngram(string, 3)


def split_words(string: str) -> tuple:
    """
    splits words on whitespace. This function is more reliable then `.split(' ')`
    since it works with any whitespace character (i.e. those recognized by regex)

    Parameters
    ----------
    string

    Returns
    -------
    tuple of strings

    Examples
    --------
    >>> len(split_words('a new day'))
    3
    """
    return tuple(_re_whitespace.split(string))


def nword(word: str, k: int) -> tuple:
    """
    concatenates k consecutive words into a tuple

    Parameters
    ----------
    word
    k

    Returns
    -------
    tuple of strings

    Examples
    --------
    >>> nword('this that the other', 2)
    ('thisthat', 'thatthe', 'theother')
    """
    parts = split_words(word)
    n = len(parts)
    k = min(k, n)
    sequences = tuple(''.join(parts[i:i + k]) for i in range(n - k + 1))
    return sequences
