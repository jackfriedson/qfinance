from pathlib import Path

import click

from agent import Agent
from environment import CompositeDataset, Environment
from environment.util import load_tbill_data


@click.command()
@click.option('--market-data', type=click.Path(exists=True),
              help='Directory of CSV files to load market data from')
@click.option('--risk-free-data', type=click.Path(exists=True),
              help='CSV file to load risk-free rate data from')
@click.option('--validation-percent', type=float, default=0.2)
@click.option('--n-episodes', type=int, default=10)
@click.option('--memory-start-size', type=int, default=1000)
@click.option('--fee', type=float, default=1.00)
@click.option('--interval', type=str, default='1Min')
@click.option('--initial-funding', type=float, default=10000.)
@click.option('--initial-cash-pct', type=float, default=0.2)
@click.option('--load-model', type=str, default=None)
def learn(market_data, risk_free_data, interval, load_model, **kwargs):
    indicators = [
        'macd',
        'rsi',
        'mom',
        'obv'
    ]
    drop_columns = [
        'open',
        'low',
        'macd',
        'macdsignal'
    ]
    hyperparams = {
        # Exploration
        'epsilon_decay': 1.1,
        'epsilon_end': 0.01,
        'epsilon_start': 0.2,

        # Replay Memory
        'batch_size': 32,
        'memory_max_size': 1e4,

        # Model params
        'gamma': 0.99,
        'tau': 0.01,
        'hidden_units': 30,
        'actor_learn_rate': 3e-5,
        'critic_learn_rate': 1e-4,
        'trace_length': 32,
    }
    market_data = CompositeDataset.from_csv_dir(Path(market_data),
                                                interval=interval,
                                                indicators=indicators,
                                                drop_columns=drop_columns)
    risk_free_data = load_tbill_data(Path(risk_free_data))
    environment = Environment(market_data, risk_free_data, **kwargs)
    agent = Agent(environment, random_seed=999999)
    agent.train(load_model=load_model, **hyperparams)


if __name__ == '__main__':
    learn()
