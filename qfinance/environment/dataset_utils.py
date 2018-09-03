from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


COLUMN_DTYPES = defaultdict(lambda x: np.float64)
COLUMN_DTYPES['volume'] = np.int32
COLUMN_INDEX_MAP = {
    2: 'open',
    3: 'high',
    4: 'low',
    5: 'close',
    6: 'volume'
}
DEFAULT_FREQ = '1T'
DEFAULT_TIMEZONE = 'America/New_York'


def load_csv_data(csv_file: Path, upsample: bool = True) -> pd.DataFrame:
    """ Loads OHLCV data from a CSV with columns [date, time, open, high, low,
        close, volume], resamples, and pads NaN values.
    """
    df = pd.read_csv(
        csv_file,
        header=None,
        index_col='timestamp',
        parse_dates={'timestamp': [0,1]},
        infer_datetime_format=True
    )
    df.rename(COLUMN_INDEX_MAP, axis='columns', inplace=True)
    df.tz_localize(DEFAULT_TIMEZONE)
    if upsample:
        df = _upsample(df, '1T')
    df = df.astype(COLUMN_DTYPES)
    return df


def resample(data: pd.DataFrame, freq: str) -> pd.DataFrame:
    if to_offset(freq) < data.index.freq:
        return _upsample(data, freq)
    else:
        return _downsample(data, freq)


def _upsample(data: pd.DataFrame, freq: str) -> pd.DataFrame:
    ohlc = data[['open', 'high', 'low', 'close']].resample(freq).ffill()
    volume = data[['volume']].resample(freq).fillna(None).fillna(0)
    return ohlc.join(volume)


def _downsample(data: pd.DataFrame, freq: str) -> pd.DataFrame:
    return data.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
