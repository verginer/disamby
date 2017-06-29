# -*- coding: utf-8 -*-

"""Console script for disamby."""

import click
from disamby import Disamby
import preprocessors as pre
import pandas as pd
from networkx import strongly_connected_components
import json

# TODO: Make tests for cli


@click.command()
@click.argument('data', type=click.Path(dir_okay=False, exists=True))
@click.argument('output', type=click.Path(dir_okay=False))
@click.option('-c', '--colname', type=click.STRING,
              help='name of the column containing the strings to match')
@click.option('-t', '--threshold', type=click.FLOAT, default=0.7,
              help='Minimum score to be counted')
@click.option('-i', '--index', type=click.STRING, help='name of index column')
@click.option('--prep', type=click.STRING, default='',
              help='pre-processing instructions, if left blank will default to a standard'
                   ' prep')
def main(data, output, index, colname, prep, threshold):
    """Console script for disamby."""
    names_df = pd.read_csv(data, index_col=index)
    # TODO: add dictionary of prep option to be composed according to the --prep tuple
    wordpieces = lambda x: pre.ngram(x[:33], 4) + pre.split_words(x)
    pipeline = [pre.compact_abbreviations,
                pre.remove_punctuation,
                pre.normalize_whitespace,
                wordpieces]
    dis = Disamby(data=names_df[colname], preprocessors=pipeline)
    alias_graph = dis.alias_graph(colname, threshold=threshold, smoother='offset',
                                  offset=100)

    components = strongly_connected_components(alias_graph)
    comp_to_id = dict()
    for comp in components:
        members = list(comp.keys())
        representative = members[0]
        name = names_df.loc[representative, colname]
        comp_to_id[name] = members

    with open(output, 'w') as f:
        json.dump(comp_to_id, f)


if __name__ == '__main__':
    main()


