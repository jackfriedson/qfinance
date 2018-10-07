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
                 memory_start_size: int):
        self._data = dataset
        self._initial_cash_pct = initial_cash_pct
        self._episode_start = 0
        self._val_start = None
        self._state_idx = 0
        self._cash = 0.0
        self._shares = []
        self._episode_values = pd.Series(index=self._data.index, name='QFIN')
        self.initial_funding = initial_funding
        self.fee = fee
        self.validation_percent = validation_percent
        self.n_episodes = n_episodes
        self.memory_start_size = memory_start_size
        total_length = len(self._data) - memory_start_size
        train_percent_ratio = (1-self.validation_percent) / self.validation_percent
        self.episode_validation_length = int(total_length / (n_episodes + train_percent_ratio))
        self.episode_train_length = int(self.episode_validation_length * train_percent_ratio)

        self.reset_portfolio()

    def reset_portfolio(self):
        cash_pct = self._initial_cash_pct
        n_symbols = len(self._data.symbols)
        weights = np.append([cash_pct], [(1-cash_pct) / n_symbols] * n_symbols)
        self._set_portfolio_from_weights(self.initial_funding, weights)

    def _set_portfolio_from_weights(self, portfolio_value: float, weights: np.array):
        intended_positions = portfolio_value * weights
        self._cash = intended_positions[0]
        self._shares = intended_positions[1:] // self.last_prices
        leftover = intended_positions[1:] - self.positions
        self._cash += leftover.sum()

        self._episode_values.iloc[self._state_idx] = portfolio_value
        current_value = self._cash + self.positions.sum()
        np.testing.assert_almost_equal(current_value, portfolio_value, 2)

    def replay_memories(self) -> pd.DataFrame:
        for _ in range(self.memory_start_size):
            yield self.state

    def episodes(self) -> Iterable[Tuple[Iterable, Iterable]]:
        for episode_i in range(self.n_episodes):
            self._episode_start = episode_i * self.episode_validation_length
            self._state_idx = self._episode_start
            self._episode_values = pd.Series(index=self._data.index, name='QFIN')
            self._train_start, self._val_start = None, None

            def get_train_states():
                self._train_start = self._state_idx
                return (self.state for _ in range(self.episode_train_length))

            def get_val_states():
                self._val_start = self._state_idx
                return (self.state for _ in range(self.episode_validation_length))

            yield get_train_states, get_val_states

    def step(self, new_weights: np.array) -> float:
        np.testing.assert_almost_equal(new_weights.sum(), 1.0, 5)
        old_shares = self._shares
        old_value = self.portfolio_value
        self._set_portfolio_from_weights(old_value, new_weights)
        new_shares = self._shares

        n_trades = (old_shares != new_shares).sum()
        tx_cost = self.fee * n_trades
        self._cash -= tx_cost
        if self._cash < 0:
            raise ValueError('Cannot execute trades with available cash balance')

        self._next()
        # TODO: add risk penalty (and maybe market impact penalty)
        new_value = self.portfolio_value
        self._episode_values.iloc[self._state_idx] = new_value
        return new_value - old_value

    def plot(self,
             data_column: str = 'close',
             save_to: Union[str, io.BufferedIOBase] = None) -> None:
        fig = plt.figure(figsize=(60, 30))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        # Get train data
        market_train_data = self._data[self._train_start:self._val_start][data_column]
        portfolio_train_data = self._episode_values.iloc[self._train_start:self._val_start]
        price_train_data = market_train_data.join(portfolio_train_data)
        scaled_train_data = price_train_data / price_train_data.iloc[0]

        # Get validation data
        market_val_data = self._data[self._val_start:self._state_idx][data_column]
        portfolio_val_data = self._episode_values.iloc[self._val_start:self._state_idx]
        price_val_data = market_val_data.join(portfolio_val_data)
        scaled_val_data = price_val_data / price_val_data.iloc[0]

        scaled_dataframes = []
        train_slice = slice(self._train_start, self._val_start)
        val_slice = slice(self._val_start, self._state_idx)
        for i, slice_ in enumerate([train_slice, val_slice]):
            market_data = self._data[slice_][data_column]
            portfolio_data = self._episode_values.iloc[slice_]
            price_data = market_data.join(portfolio_data)

            scaled_data = price_data / price_data.iloc[0]
            np.testing.assert_almost_equal(portfolio_data[0], self.initial_funding, 3)
            scaled_data *= self.initial_funding
            dt_index = scaled_data.index
            scaled_data.reset_index(drop=True, inplace=True)

            ax = fig.add_subplot(gs[i])
            ax.set_title('Training' if i == 0 else 'Validation')
            ax.set_xlim(left=scaled_data.index[0], right=scaled_data.index[-1])
            ticks = ax.get_xticks()
            ax.set_xticklabels([dt_index[int(i)].date() for i in ticks[:-1]])
            for column in scaled_data.columns:
                ax.plot(scaled_data.index, scaled_data[column], label=column)
            ax.legend()

        fig.autofmt_xdate()
        plt.tight_layout()

        if save_to:
            fig.savefig(save_to, format='png')
        else:
            plt.show()

    @property
    def state(self) -> np.ndarray:
        return self._data[self._state_idx].values

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
        return len(self._shares) + 1

    @property
    def total_steps(self) -> int:
        return (self.episode_train_length + self.episode_validation_length) * self.n_episodes

    @property
    def current_timestamp(self):
        return self._data.index[self._state_idx]

    @property
    def last_prices(self) -> pd.Series:
        return self._data[self._state_idx]['close']

    @property
    def period_returns(self):
        if self._state_idx == 0:
            raise ValueError('Cant compute return in state 0')
        result = self.last_prices / self._data[self._state_idx-1]['close']
        return result - 1.

    def _next(self):
        self._state_idx += 1
