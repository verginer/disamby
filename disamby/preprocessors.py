import re

__all__ = ['compact_abbreviations', 'reduce_duplicate_whitespace', 'ngram']

_abbreviation_re = re.compile(r'\.(?![a-zA-Z]*})')


def reduce_duplicate_whitespace(string: str):
    return re.sub('\s+', ' ', string.lower())


def compact_abbreviations(string: str):
    split = _abbreviation_re.split(string.lower())
    return ''.join(split)


def ngram(string: str, n: int) -> tuple:
    N = len(string)
    if n > N:
        return string,
    if n < 2:
        raise ValueError('n for a ngram must be 2 or larger')
    return tuple(string[i:i+n] for i in range(N-n+1))
