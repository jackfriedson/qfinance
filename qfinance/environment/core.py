import io
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from talib.abstract import Function

from environment.dataset_utils import downsample, load_csv_data


class Environment(object):
    actions = ['long', 'short', 'none']

    def __init__(self,
                 ohlc_data: pd.DataFrame,
                 indicators: List[str],
                 interval: str,
                 fee: float,
                 validation_percent: float,
                 n_episodes: int,
                 replay_memory_start_size: int):
        self._ohlc_data = downsample(ohlc_data, interval)
        self._indicators = {name: Function(name) for name in indicators}
        self._full_data = self._apply_indicators()
        self._orders = self._init_orders()
        self._positions = self._init_positions()

        self._episode_start = 0
        self._current_state = 0
        self._current_position = 'none'
        self._order_open_ts = None

        self.fee = fee
        self.validation_percent = validation_percent
        self.n_episodes = n_episodes
        self.replay_memory_start_size = replay_memory_start_size

        total_length = len(self._full_data) - replay_memory_start_size
        train_percent_ratio = (1-self.validation_percent) / self.validation_percent
        self.episode_validation_length = int(total_length / (n_episodes + train_percent_ratio))
        self.episode_train_length = int(self.episode_validation_length * train_percent_ratio)

    def _init_orders(self):
        return pd.DataFrame(columns=['buy', 'sell'], index=self._full_data.index)

    def _init_positions(self):
        return pd.Series(index=self._full_data.index)

    def _apply_indicators(self):
        full_data = self._ohlc_data

        for name, func in self._indicators.items():
            indicator_data = func(self._ohlc_data)
            if isinstance(indicator_data, pd.Series):
                indicator_data = indicator_data.to_frame(name)
            full_data = full_data.join(indicator_data)

        return full_data.dropna()

    @classmethod
    def from_csv(cls, csv_path: str, **params):
        df = load_csv_data(Path(csv_path))
        return cls(df, **params)

    def replay_memories(self) -> pd.DataFrame:
        for _ in range(self.replay_memory_start_size):
            yield self.state

    def episodes(self) -> Iterable[Tuple[Iterable, Iterable]]:
        for episode_i in range(self.n_episodes):
            self._episode_start = episode_i * self.episode_validation_length
            self._orders = self._init_orders()
            self._current_state = self._episode_start
            yield ((self.state for _ in range(self.episode_train_length)),
                   (self.state for _ in range(self.episode_validation_length)))

    def step(self, action_idx: int, track_orders: bool = False) -> float:
        action = self.actions[action_idx]
        position_change = action == self._current_position
        self._current_position = action
        self._positions.iloc[self._current_state] = action
        start_state = self._full_data.iloc[self._current_state]
        self._next()

        if action == 'long':
            reward = self.period_return
            if position_change and track_orders:
                self._order_open_ts = self.current_timestamp
                self._orders.loc[self._order_open_ts, 'buy'] = start_state['close']
        elif action == 'short':
            reward = -self.period_return
            if position_change and track_orders:
                self._orders.loc[self._order_open_ts, 'sell'] = start_state['close']
                self._order_open_ts = None
        else:
            assert action == 'none'
            reward = 0.

        if position_change:
            reward -= self.fee

        return reward

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
        plot_data = self._full_data.iloc[self._episode_start:self._current_state]

        # Plot long and short positions
        ax0 = fig.add_subplot(gs[0])
        ax0.set_title('Price ({})'.format(data_column))
        ax0.plot(plot_data.index, plot_data[data_column], 'black')

        longs = plot_data[data_column][self._positions == 'long']
        longs = longs.resample(plot_data.index.freq).fillna(None)
        longs.plot(ax=ax0, style='g')

        shorts = plot_data[data_column][self._positions == 'short']
        shorts = shorts.resample(plot_data.index.freq).fillna(None)
        shorts.plot(ax=ax0, style='r')

        if plot_orders:
            orders = self._orders.dropna()
            ax0.plot(orders.index, orders['buy'], color='k', marker='^', linestyle='None')
            ax0.plot(orders.index, orders['sell'], color='k', marker='v', linestyle='None')

        # if plot_indicators:
        #     for i, indicator in enumerate(self._indicators, start=1):
        #         ax_ind = fig.add_subplot(gs[i])
        #         indicator.plot(ax_ind)

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

    def total_train_steps(self) -> int:
        return self.episode_train_length * self.n_episodes

    @property
    def current_timestamp(self):
        return self._full_data.index[self._current_state]

    @property
    def last_price(self) -> float:
        return self._full_data.iloc[self._current_state]['close']

    def _next(self):
        self._current_state += 1
