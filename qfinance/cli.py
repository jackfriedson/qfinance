from pathlib import Path

import click

from agent.common import QFinanceAgent
from environment.common import QFinanceEnvironment


@click.command()
@click.option('--data-file', type=click.Path(exists=True), help='CSV file to read data from')
@click.option('--validation-percent', type=float, default=0.2)
@click.option('--n-folds', type=int, default=10)
@click.option('--replay-memory-start-size', type=int, default=10000)
@click.option('--fee', type=float, default=0.002)
@click.option('--interval', type=str, default='1Min')
def learn(data_file, **kwargs):
    hyperparams = {
        'epochs': 1,

        # Exploration
        'epsilon_decay': 3,
        'epsilon_end': 0.1,
        'epsilon_start': 1.,

        # Replay Memory
        'replay_batch_size': 32,
        'replay_memory_max_size': 1000000,

        # Regularization
        'dropout_prob': 0.,
        'rnn_dropout_prob': 0.,

        # Model params
        'gamma': 0.99,
        'learn_rate': 0.001,
        'renorm_decay': 0.9,
        'hidden_units': 10,
        'rnn_layers': 2,
        'trace_length': 16,
        'update_target_every': 4
    }
    environment = QFinanceEnvironment.from_csv(data_file, **kwargs)
    agent = QFinanceAgent(environment, random_seed=999999)
    agent.train(**hyperparams)


if __name__ == '__main__':
    learn()
