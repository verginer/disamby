=====
Usage
=====

To use disamby in a project::

    import pandas as pd
    import disamby.preprocessors as pre


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
        lambda x: pre.trigram(x) + pre.split_words(x)  # any function is allowed
    ]

    dis = Disamby(df, pipeline)

    dis.disambiguated_sets(threshold=0.5, verbose=False)
    [{'L2', 'L1'}, {'O1'}]
