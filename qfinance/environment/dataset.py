from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from talib.abstract import Function


COLUMN_DTYPES = defaultdict(lambda x: np.float64)
COLUMN_DTYPES['volume'] = np.int32
DEFAULT_FREQ = '1Min'
DEFAULT_TIMEZONE = 'America/New_York'


class Dataset(object):
    """ Dataset containing OHLCV data and technical indicators for an individual
        stock.
    """

    def __init__(self,
                 name: str,
                 data: pd.DataFrame,
                 interval: str = DEFAULT_FREQ,
                 indicators: List[str] = None):
        # Init OHLC data
        self.name = name
        self._data = data.tz_localize(DEFAULT_TIMEZONE, ambiguous='infer').tz_convert('UTC')
        self.upsample(DEFAULT_FREQ)
        if interval != DEFAULT_FREQ:
            self.downsample(interval)
        self._data = self._data.astype(COLUMN_DTYPES)

        # Init indicators
        self._indicators = {name: Function(name) for name in indicators or []}
        self._apply_indicators()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        key_type = key.start if isinstance(key, slice) else key

        if isinstance(key_type, int):
            return self._data.iloc[key]
        else:
            return self._data.loc[key]

    def __setitem__(self, key, item):
        self._data[key] = item

    @classmethod
    def from_csv(cls, csv_file: Path, name: str = None, *args, **kwargs):
        """ Loads OHLCV data from a CSV with columns [date, time, open, high, low,
            close, volume].
        """
        name = name or csv_file.name.split('.')[0]
        df = pd.read_csv(
            csv_file,
            index_col='timestamp',
            parse_dates=True,
            infer_datetime_format=True
        )
        return cls(name, df, *args, **kwargs)

    def _apply_indicators(self):
        full_data = self._data

        for name, func in self._indicators.items():
            indicator_data = func(self._data)
            if isinstance(indicator_data, pd.Series):
                indicator_data = indicator_data.to_frame(name)
            full_data = full_data.join(indicator_data)

        self._data = full_data.dropna()

    def upsample(self, freq: str):
        self._data = self._data.resample(freq).fillna(None)
        self._data['volume'] = self._data['volume'].fillna(value=0)
        self._data['close'] = self._data['close'].fillna(method='ffill')
        for column in ['open', 'high', 'low']:
            self._data[column] = self._data[column].fillna(value=self._data['close'])

    def downsample(self, freq: str):
        self._data = self._data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

    @property
    def index(self):
        return self._data.index


class CompositeDataset(object):
    """ Dataset containing OHLCV data and technical indicators for a portfolio
        of stocks.
    """

    def __init__(self, datasets: List[Dataset]):
        data = {d.name: d._data for d in datasets}
        self.symbols = list(data.keys())
        self._data = pd.concat(data.values(), keys=self.symbols, axis='columns')
        self._data = self._data.swaplevel(axis=1)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        key_type = key.start if isinstance(key, slice) else key

        if isinstance(key_type, int):
            return self._data.iloc[key]
        else:
            return self._data.loc[key]

    @classmethod
    def from_csv_dir(cls, csv_dir: Path, *args, **kwargs):
        if not csv_dir.is_dir():
            raise ValueError('csv_dir must be a directory')

        data = [Dataset.from_csv(f, *args, **kwargs) for f in csv_dir.iterdir()]
        return cls(data)

    @property
    def index(self):
        return self._data.index
