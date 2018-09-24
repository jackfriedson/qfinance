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

from environment.dataset import CompositeDataset


class Environment(object):

    def __init__(self,
                 dataset: CompositeDataset,
                 fee: float,
                 initial_funding: float,
                 initial_cash_pct: float,
                 validation_percent: float,
                 n_episodes: int,
                 replay_memory_start_size: int):
        self._data = dataset
        self._initial_funding = initial_funding
        self._initial_cash_pct = initial_cash_pct
        self._episode_start = 0
        self._current_state = 0
        self._cash = 0.0
        self._shares = []
        self._init_portfolio()

        self.fee = fee
        self.validation_percent = validation_percent
        self.n_episodes = n_episodes
        self.replay_memory_start_size = replay_memory_start_size

        total_length = len(self._data) - replay_memory_start_size
        train_percent_ratio = (1-self.validation_percent) / self.validation_percent
        self.episode_validation_length = int(total_length / (n_episodes + train_percent_ratio))
        self.episode_train_length = int(self.episode_validation_length * train_percent_ratio)

    def _init_portfolio(self):
        cash_pct = self._initial_cash_pct
        n_symbols = len(self._data.symbols)
        weights = np.append([cash_pct], [(1-cash_pct) / n_symbols] * n_symbols)
        self._set_portfolio_from_weights(self._initial_funding, weights)

    def _set_portfolio_from_weights(self, portfolio_value: float, weights: np.array):
        intended_positions = portfolio_value * weights
        self._cash = intended_positions[0]
        self._shares = intended_positions[1:] // self.last_prices
        leftover = intended_positions[1:] - self.positions
        self._cash += leftover.sum()

    def replay_memories(self) -> pd.DataFrame:
        for _ in range(self.replay_memory_start_size):
            yield self.state

    def episodes(self) -> Iterable[Tuple[Iterable, Iterable]]:
        for episode_i in range(self.n_episodes):
            self._episode_start = episode_i * self.episode_validation_length
            self._init_portfolio()
            self._current_state = self._episode_start
            yield ((self.state for _ in range(self.episode_train_length)),
                   (self.state for _ in range(self.episode_validation_length)))

    def step(self, new_weights: np.array) -> float:
        old_shares = self._shares
        old_value = self.portfolio_value
        self._set_portfolio_from_weights(self.portfolio_value, new_weights)
        new_shares = self._shares

        n_trades = (old_shares != new_shares).sum()
        tx_cost = self.fee * n_trades
        self._cash -= tx_cost
        if self._cash < 0:
            raise ValueError('Cannot execute trades with available cash balance')

        self._next()
        # TODO: add risk penalty (and maybe market impact penalty)
        return self.portfolio_value - old_value

    def plot(self,
             data_column: str = 'close',
             plot_indicators: bool = False,
             plot_orders: bool = True,
             save_to: Union[str, io.BufferedIOBase] = None) -> None:
        pass
        # fig = plt.figure(figsize=(60, 30))
        # ratios = [3] if not plot_indicators else [3] + ([1] * len(self._indicators))
        # n_subplots = 1 if not plot_indicators else 1 + len(self._indicators)
        # gs = gridspec.GridSpec(n_subplots, 1, height_ratios=ratios)
        # plot_data = self._data[self._episode_start:self._current_state]
        #
        # # Plot long and short positions
        # ax0 = fig.add_subplot(gs[0])
        # ax0.set_title('Price ({})'.format(data_column))
        # ax0.plot(plot_data.index, plot_data[data_column], 'black')
        #
        # longs = plot_data[data_column][self._positions == 'long']
        # longs = longs.resample(plot_data.index.freq).fillna(None)
        # longs.plot(ax=ax0, style='g')
        #
        # shorts = plot_data[data_column][self._positions == 'short']
        # shorts = shorts.resample(plot_data.index.freq).fillna(None)
        # shorts.plot(ax=ax0, style='r')
        #
        # if plot_orders:
        #     orders = self._orders.dropna()
        #     ax0.plot(orders.index, orders['buy'], color='k', marker='^', linestyle='None')
        #     ax0.plot(orders.index, orders['sell'], color='k', marker='v', linestyle='None')
        #
        # # if plot_indicators:
        # #     for i, indicator in enumerate(self._indicators, start=1):
        # #         ax_ind = fig.add_subplot(gs[i])
        # #         indicator.plot(ax_ind)
        #
        # fig.autofmt_xdate()
        # plt.tight_layout()
        #
        # if save_to:
        #     fig.savefig(save_to, format='png')
        # else:
        #     plt.show()

    @property
    def state(self) -> np.ndarray:
        return self._data[self._current_state].values

    @property
    def portfolio_value(self) -> float:
        return self._cash + self.positions.sum()

    @property
    def positions(self) -> np.array:
        return self._shares * self.last_prices

    @property
    def state_space_dim(self) -> int:
        return len(self.state)

    @property
    def action_space_dim(self) -> int:
        return len(self._portfolio_weights)

    @property
    def total_train_steps(self) -> int:
        return self.episode_train_length * self.n_episodes

    @property
    def current_timestamp(self):
        return self._data.index[self._current_state]

    @property
    def last_prices(self) -> pd.Series:
        return self._data[self._current_state]['close']

    @property
    def period_returns(self):
        if self._current_state == 0:
            raise ValueError('Cant compute return in state 0')
        result = self.last_prices / self._data[self._current_state-1]['close']
        return result - 1.

    def _next(self):
        self._current_state += 1
