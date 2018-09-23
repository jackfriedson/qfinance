from pathlib import Path

import click
import pandas as pd


@click.command()
@click.option('--source-dir', type=click.Path(exists=True))
@click.option('--target-file', type=click.Path(), default=None)
def combine_csvs(source_dir, target_file):
    target_file = Path(target_file) if target_file else Path(source_dir + '.csv')
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise ValueError('source_dir must be a directory')

    dataframes = []
    for csv_file in source_dir.iterdir():
        df = pd.read_csv(
            csv_file,
            sep=None,
            index_col='timestamp',
            parse_dates={'timestamp': ['<DATE>', '<TIME>']},
            infer_datetime_format=True
        )
        df.drop(['<TICKER>', '<PER>'], axis='columns', inplace=True)
        df.rename({
            '<OPEN>': 'open',
            '<HIGH>': 'high',
            '<LOW>': 'low',
            '<CLOSE>': 'close',
            '<VOL>': 'volume'
        }, axis='columns', inplace=True)
        dataframes.append(df)

    result = pd.concat(dataframes, verify_integrity=True)
    result.sort_index(inplace=True)
    result.to_csv(target_file)


if __name__ == '__main__':
    combine_csvs()
