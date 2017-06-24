import re

__all__ = ['compact_abbreviations', 'reduce_duplicate_whitespace',
           'ngram', 'trigram', 'split_words']

re_abbreviation = re.compile(r'\.(?![a-zA-Z]*})')
re_duplicate_white = re.compile('\s+',)
re_whitespace = re.compile('\s')


def reduce_duplicate_whitespace(string: str):
    return re_duplicate_white.sub(' ', string.lower())


def compact_abbreviations(string: str):
    split = re_abbreviation.split(string.lower())
    return ''.join(split)


def ngram(string: str, n: int) -> tuple:
    N = len(string)
    if n > N:
        return string,
    if n < 2:
        raise ValueError('n for a ngram must be 2 or larger')
    return tuple(string[i:i+n] for i in range(N-n+1))


def trigram(string: str):
    return ngram(string, 3)


def split_words(string: str) -> tuple:
    return tuple(re_whitespace.split(string))
