import re

__all__ = ['compact_abbreviations', 'normalize_whitespace',
           'ngram', 'trigram', 'split_words', 'remove_punctuation',
           'nword']


re_abbreviation = re.compile(r'\.(?![a-zA-Z]*})')
re_duplicate_white = re.compile('\s+')
re_whitespace = re.compile('\s')
re_punctuation = re.compile('[^\w\s]')


def normalize_whitespace(string: str) -> str:
    no_whitespace = re_duplicate_white.sub(' ', string.upper())
    return no_whitespace.strip()


def compact_abbreviations(string: str) -> str:
    split = re_abbreviation.split(string.upper())
    return ''.join(split)


def remove_punctuation(word: str) -> str:
    return re_punctuation.sub('', word)


def ngram(string: str, n: int) -> tuple:
    N = len(string)
    if n > N:
        return string,
    if n < 2:
        raise ValueError('n for a ngram must be 2 or larger')
    return tuple(string[i:i+n] for i in range(N-n+1))


def trigram(string: str) -> tuple:
    return ngram(string, 3)


def split_words(string: str) -> tuple:
    return tuple(re_whitespace.split(string))


def nword(word: str, k: int) -> tuple:
    """concatenates k consecutive words into a tuple"""
    parts = word.split(' ')
    n = len(parts)
    k = min(k, n)
    sequences = tuple(''.join(parts[i:i + k]) for i in range(n - k + 1))
    return sequences
