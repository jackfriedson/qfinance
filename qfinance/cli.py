from pathlib import Path

import click

from agent import Agent
from environment import CompositeDataset, Environment


@click.command()
@click.option('--data-dir', type=click.Path(exists=True),
              help='Directory of CSV files to read data from')
@click.option('--validation-percent', type=float, default=0.2)
@click.option('--n-episodes', type=int, default=10)
@click.option('--memory-start-size', type=int, default=10000)
@click.option('--fee', type=float, default=1.00)
@click.option('--interval', type=str, default='1Min')
@click.option('--initial-funding', type=float, default=10000.)
@click.option('--initial-cash-pct', type=float, default=0.2)
@click.option('--load-model', type=str, default=None)
def learn(data_dir, interval, load_model, **kwargs):
    indicators = [
        # 'macd',
        'rsi',
        'mom',
        # 'stoch'
    ]
    hyperparams = {
        # Exploration
        'epsilon_decay': 2,
        'epsilon_end': 0.1,
        'epsilon_start': 1.,

        # Replay Memory
        'batch_size': 32,
        'memory_max_size': 1e6,

        # Regularization
        # 'dropout_prob': 0.,
        # 'rnn_dropout_prob': 0.,

        # Model params
        'gamma': 0.99,
        'tau': 0.01,
        # 'fc_units': None,
        'actor_units': 10,
        'actor_learn_rate': 1e-4,
        'critic_learn_rate': 1e-3,
        # 'rnn_layers': 2,
        'trace_length': 16,
    }
    data = CompositeDataset.from_csv_dir(Path(data_dir), interval=interval, indicators=indicators)
    environment = Environment(data, **kwargs)
    agent = Agent(environment, random_seed=999999)
    agent.train(load_model=load_model, **hyperparams)


if __name__ == '__main__':
    learn()
