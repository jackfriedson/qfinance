from pathlib import Path

import pandas as pd


def load_tbill_data(csv_file: Path) -> pd.Series:
    data = pd.read_csv(csv_file,
                       index_col='date',
                       parse_dates=True,
                       infer_datetime_format=True)
    data = data['3month']
    data = data.resample('D').ffill().tz_localize('UTC')
    return data
