from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from environment.dataset_utils import load_csv_data


class QFinanceEnvironment(object):
    actions = ['buy', 'sell', 'hold']

    def __init__(self,
                 ohlc_data: pd.DataFrame,
                 fee: float,
                 validation_percent: float,
                 n_folds: int,
                 replay_memory_start_size: int):
        self._full_data = ohlc_data
        self._current_state = 0
        self._current_position = None

        self.fee = fee
        self.validation_percent = validation_percent
        self.n_folds = n_folds
        self.replay_memory_start_size = replay_memory_start_size

        total_length = len(self._full_data) - replay_memory_start_size
        train_percent_ratio = (1-self.validation_percent) / self.validation_percent
        self.fold_validation_length = int(total_length / (n_folds + train_percent_ratio))
        self.fold_train_length = int(self.fold_validation_length * train_percent_ratio)

    @classmethod
    def from_csv(cls, csv_path: str, **params):
        df = load_csv_data(Path(csv_path), upsample=False)
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

    def step(self, action_idx: int) -> float:
        action = self.actions[action_idx]
        start_state = self._full_data.iloc[self._current_state]
        self._next()
        end_state = self._full_data.iloc[self._current_state]
        return self.reward(action, start_state, end_state)

    def reward(self, action: str, start_state: pd.DataFrame, end_state: pd.DataFrame) -> float:
        if action == 'buy':
            if self._current_position is None:
                self._current_position == 'long'
                return self.period_return - self.fee
            if self._current_position == 'long':
                return self.period_return

        if action == 'sell':
            if self._current_position is None:
                return 0
            if self._current_position == 'long':
                return -self.fee

        if action == 'hold':
            if self._current_position is None:
                return 0
            if self._current_position == 'long':
                return self.period_return

    @property
    def period_return(self):
        if self._current_state == 0:
            raise ValueError('Cannot calculate return in state 0')
        return (self._full_data.iloc[self._current_state]['close'] /
                self._full_data.iloc[self._current_state-1]['close']) - 1.0

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
    def last_price(self) -> float:
        return self._full_data.iloc[self._current_state]['close']

    def _next(self):
        self._current_state += 1
