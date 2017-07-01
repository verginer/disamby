=======
disamby
=======


.. image:: https://img.shields.io/pypi/v/disamby.svg
        :target: https://pypi.python.org/pypi/disamby

.. image:: https://img.shields.io/travis/verginer/disamby.svg
        :target: https://travis-ci.org/verginer/disamby

.. image:: https://readthedocs.org/projects/disamby/badge/?version=latest
        :target: https://disamby.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/verginer/disamby/shield.svg
     :target: https://pyup.io/repos/github/verginer/disamby/
     :alt: Updates

* Free software: MIT license
* Documentation: https://disamby.readthedocs.io.

``disamby`` is a python package designed to carry out entity disambiguation based on fuzzy
string matching.

It works best for entities which if the same have very similar strings.
Examples of situation where this disambiguation algorithm works fairly well is with
company names and addresses which have typos, alternative spellings or composite names.
Other use-cases include identifying people in a database where the name might be misspelled.

The algorithm works by exploiting how informative a given word/token is, based on the
observed frequencies in the whole corpus of strings. For example the word 'inc' in the
case of firm names is not very informative, however "Solomon" is, since the former appears
repeatedly whereas the second rarely.

With these frequencies the algorithms computes for a given pair of instances how similar
they are, and if they are above an arbitrary threshold they are connected in an
"alias graph" (i.e. a directed network where an entity is connected to an other
if it is similar enough). After all records have been connected in this way disamby
returns sets of entities, which are strongly connected [2]_ . Strongly connected means
in this case that there exists a path from all nodes to all nodes within the component.


Example
-------

To use disamby in a project::

    import pandas as pd
    import disamby.preprocessors as pre
    form disamby import Disamby

    # create a dataframe with the fields you intend to match on as columns
    df = pd.DataFrame({
        'name':     ['Luca Georger',        'Luca Geroger',         'Adrian Sulzer'],
        'address':  ['Mira, 34, Augsburg',  'Miri, 34, Augsburg',   'Milano, 34']},
        index=      ['L1',                  'L2',                   'O1']
    )

    # define the pipeline to process the strings, note that the last step must return
    # a tuple of strings
    pipeline = [
        pre.normalize_whitespace,
        pre.remove_punctuation,
        lambda x: pre.trigram(x) + pre.split_words(x)  # any python function is allowed
    ]

    # instantiate the disamby object, it applies the given pre-processing pipeline and
    # computes their frequency.
    dis = Disamby(df, pipeline)

    # let disamby compute disambiguated sets. Node that a threshold must be given or it
    # defaults to 0.
    dis.disambiguated_sets(threshold=0.5)
    [{'L2', 'L1'}, {'O1'}]  # output

    # To check if the sets are accurate you can get the rows from the
    # pandas dataframe like so:
    df.loc[['L2', 'L1']]



Installation
------------

To install disamby, run this command in your terminal:

.. code-block:: console

    $ pip install disamby

This is the preferred method to install disamby, as it will always install the most recent stable release.
If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

You can also install it from source as follows
The sources for disamby can be downloaded from the `Github repo`_.
You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/verginer/disamby

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/verginer/disamby/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/verginer/disamby
.. _tarball: https://github.com/verginer/disamby/tarball/master


Credits
---------
I got the inspiration for this package from the seminar "The SearchEngine - A Tool for
Matching by Fuzzy Criteria" by Thorsten Doherr at the CISS [1]_ Summer School 2017

.. [1] http://www.euro-ciss.eu/ciss/home.html
.. [2] https://en.wikipedia.org/wiki/Strongly_connected_component
