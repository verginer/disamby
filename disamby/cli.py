# -*- coding: utf-8 -*-

"""Console script for disamby."""

import click
from disamby import Disamby
import preprocessors as pre
import pandas as pd
from networkx import strongly_connected_components
import json

# TODO: Make tests for cli


@click.command(help="""
This command allows to disambiguate a csv file using `disamby`
example usage:\n
$ python disamby/cli.py -c name -i inv_id -t .7 -p APNX name.csv dis_name.json

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
@click.option('-c', '--colname', type=click.STRING,
              help='name of the column containing the strings to match')
@click.option('-t', '--threshold', type=click.FLOAT, default=0.7,
              help='Minimum score to be counted')
@click.option('-i', '--index', type=click.STRING, help='name of index column')
@click.option('-p', '--prep', type=click.STRING, default='APNX',
              help='pre-processing instructions, if left blank will default to a standard'
                   ' prep')
def main(data, output, index, colname, prep, threshold):
    """Console script for disamby."""
    names_df = pd.read_csv(data, index_col=index).sample(1000)
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
    pipeline = [prep_dict[action] for action in list(prep)]
    dis = Disamby(data=names_df[colname], preprocessors=pipeline)
    alias_graph = dis.alias_graph(colname, threshold, smoother='offset', offset=100)

    components = strongly_connected_components(alias_graph)
    comp_to_id = dict()
    for comp in components:
        members = list(comp)
        representative = members[0]
        name = names_df.loc[representative, colname]
        comp_to_id[name] = members

    with open(output, 'w') as f:
        json.dump(comp_to_id, f)


if __name__ == '__main__':
    main()


