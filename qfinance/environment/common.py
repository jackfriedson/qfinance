import io
from pathlib import Path
from typing import Iterable, Tuple, Union

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from environment.dataset_utils import load_csv_data, downsample


class QFinanceEnvironment(object):
    actions = ['buy', 'sell', 'hold']

    def __init__(self,
                 ohlc_data: pd.DataFrame,
                 interval: str,
                 fee: float,
                 validation_percent: float,
                 n_folds: int,
                 replay_memory_start_size: int):
        self._full_data = downsample(ohlc_data, interval)
        self._current_state = 0
        self._current_position = None
        self._order_open_ts = None
        self._indicators = []

        self.fee = fee
        self.validation_percent = validation_percent
        self.n_folds = n_folds
        self.replay_memory_start_size = replay_memory_start_size

        self._orders = pd.DataFrame(columns=['buy', 'sell'], index=self._full_data.index)

        total_length = len(self._full_data) - replay_memory_start_size
        train_percent_ratio = (1-self.validation_percent) / self.validation_percent
        self.fold_validation_length = int(total_length / (n_folds + train_percent_ratio))
        self.fold_train_length = int(self.fold_validation_length * train_percent_ratio)

    @classmethod
    def from_csv(cls, csv_path: str, **params):
        df = load_csv_data(Path(csv_path))
        return cls(df, **params)

    def replay_memories(self) -> pd.DataFrame:
        for _ in range(self.replay_memory_start_size):
            yield self.state

    def training_slices(self, epochs: int) -> Iterable[Tuple[Iterable, Iterable]]:
        for fold_i in range(self.n_folds):
            slice_start = fold_i * self.fold_validation_length
            def slice_epochs():
                for _ in range(epochs):
                    self._current_state = slice_start
                    yield ((self.state for _ in range(self.fold_train_length)),
                           (self.state for _ in range(self.fold_validation_length)))
            yield slice_epochs()

    def step(self, action_idx: int, track_orders: bool = False) -> float:
        action = self.actions[action_idx]
        start_state = self._full_data.iloc[self._current_state]
        self._next()
        end_state = self._full_data.iloc[self._current_state]

        # click.echo(action)

        if action == 'buy':
            if self._current_position is None:
                self._current_position = 'long'
                if track_orders:
                    self._order_open_ts = self.current_timestamp
                    self._orders.loc[self._order_open_ts, 'buy'] = start_state['close']
                return self.period_return - self.fee
            elif self._current_position == 'long':
                return self.period_return

        elif action == 'sell':
            if self._current_position is None:
                return 0
            elif self._current_position == 'long':
                self._current_position = None
                if track_orders:
                    self._orders.loc[self._order_open_ts, 'sell'] = start_state['close']
                    self._order_open_ts = None
                return -self.fee

        elif action == 'hold':
            if self._current_position is None:
                return 0
            elif self._current_position == 'long':
                return self.period_return

    @property
    def period_return(self):
        if self._current_state == 0:
            raise ValueError('Cannot calculate return in state 0')
        return (self._full_data.iloc[self._current_state]['close'] /
                self._full_data.iloc[self._current_state-1]['close']) - 1.0

    def order_returns(self):
        orders = self._orders.dropna()
        return orders['sell'] / orders['buy'] - 1.

    def plot(self,
             data_column: str = 'close',
             plot_indicators: bool = False,
             plot_orders: bool = True,
             save_to: Union[str, io.BufferedIOBase] = None) -> None:
        fig = plt.figure(figsize=(60, 30))
        ratios = [3] if not plot_indicators else [3] + ([1] * len(self._indicators))
        n_subplots = 1 if not plot_indicators else 1 + len(self._indicators)
        gs = gridspec.GridSpec(n_subplots, 1, height_ratios=ratios)

        # Plot long and short positions
        ax0 = fig.add_subplot(gs[0])
        ax0.set_title('Price ({})'.format(data_column))
        ax0.plot(self._full_data.index, self._full_data[data_column], 'blue')

        if plot_orders:
            orders = self._orders.dropna()
            ax0.plot(orders.index, orders['buy'], color='k', marker='^', fillstyle='none')
            ax0.plot(orders.index, orders['sell'], color='k', marker='v', fillstyle='none')

        if plot_indicators:
            for i, indicator in enumerate(self._indicators, start=1):
                ax_ind = fig.add_subplot(gs[i])
                indicator.plot(ax_ind)

        fig.autofmt_xdate()
        plt.tight_layout()

        if save_to:
            fig.savefig(save_to, format='png')
        else:
            plt.show()

    @property
    def state(self) -> np.ndarray:
        return self._full_data.iloc[self._current_state].values

    @property
    def n_state_factors(self) -> int:
        return len(self._full_data.iloc[0])

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def total_train_steps(self, epochs: int) -> int:
        return self.fold_train_length * self.n_folds * epochs

    @property
    def current_timestamp(self):
        return self._full_data.index[self._current_state]

    @property
    def last_price(self) -> float:
        return self._full_data.iloc[self._current_state]['close']

    def _next(self):
        self._current_state += 1
