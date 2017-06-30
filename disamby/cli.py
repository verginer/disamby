# -*- coding: utf-8 -*-

"""Console script for disamby.

With the cli it is possible to carry out the disambiguation on the command line.
The script returns a json file with a name taken from the first field as the key and
a list of indices belonging to the same disambiguated cluster.

.. code-block:: python

    {'International Business Machines Inc': [1, 2, 5, 34, 90],
    'Samsung': [4, 123],
    '...': [...],
     ...}

To carry out the disambiguation you will need a csv file with proper headings
and optionally an index column. If not index column is specified then the position in the
csv is used in the json file to identify its members.

Only a basic processing pipeline is possible here, see the `--help` of the script
to know which are available.

Example usage:

.. code-block:: bash

    $ disamby --col-headers name,address \\
                --index inv_id \\
                --threshold .7 \\
                --prep APNX \\
                input.csv output.json

"""

import click
from disamby import Disamby
import disamby.preprocessors as pre
import pandas as pd
import json

# TODO: Make tests for cli


@click.command(help="""
This command allows to disambiguate a csv file using `disamby`
example usage:\n
$ disamby -c name -i inv_id -t .7 -p APNX name.csv dis_name.json
$ disamby -c name -i inv_id -t .7 -p APNX name.csv dis_name.json

The prep codes are:\n
'A': pre.compact_abbreviations\n
'P': pre.remove_punctuation\n
'W': pre.normalize_whitespace\n
'3': lambda x: pre.ngram(x, 3)\n
'4': lambda x: pre.ngram(x, 4)\n
'5': lambda x: pre.ngram(x, 5)\n
'S': pre.split_words\n
'X': lambda x: pre.ngram(x[:33], 4) + pre.split_words(x)
""")
@click.argument('data', type=click.Path(dir_okay=False, exists=True))
@click.argument('output', type=click.Path(dir_okay=False))
@click.option('-c', '--col-headers', type=click.STRING,
              help='name of the columns containing the strings to match if more then '
                   'one then separate with comma (no spaces)')
@click.option('-t', '--threshold', type=click.FLOAT, default=0.7,
              help='Minimum score to be counted')
@click.option('-i', '--index', type=click.STRING, help='name of index column')
@click.option('-p', '--prep', type=click.STRING, default='APWX',
              help='pre-processing instructions, if left blank will default to a standard'
                   ' prep')
def main(data, output, index, col_headers, prep, threshold):
    """Console script for disamby."""
    names_df = pd.read_csv(data, index_col=index)
    prep_dict = {
        'A': pre.compact_abbreviations,
        'P': pre.remove_punctuation,
        'W': pre.normalize_whitespace,
        '3': lambda x: pre.ngram(x, 3),
        '4': lambda x: pre.ngram(x, 4),
        '5': lambda x: pre.ngram(x, 5),
        'S': pre.split_words,
        'X': lambda x: pre.ngram(x[:33], 4) + pre.split_words(x)
    }
    columns = col_headers.split(',')
    pipeline = [prep_dict[action] for action in list(prep)]
    dis = Disamby(data=names_df[columns], preprocessors=pipeline)
    components = dis.disambiguated_sets(threshold, smoother='offset', offset=100)

    comp_to_id = dict()
    for comp in components:
        members = list(comp)
        representative = members[0]
        name = names_df.loc[representative, columns[0]]
        comp_to_id[name] = members

    with open(output, 'w') as f:
        json.dump(comp_to_id, f)


if __name__ == '__main__':
    main()


