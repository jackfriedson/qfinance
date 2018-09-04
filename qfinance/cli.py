from pathlib import Path

import click

from agent.common import QFinanceAgent
from environment.common import QFinanceEnvironment


@click.command()
@click.option('--data-file', type=click.Path(exists=True), help='CSV file to read data from')
@click.option('--validation-percent', type=float, default=0.2)
@click.option('--n-folds', type=int, default=10)
@click.option('--replay-memory-start-size', type=int, default=1000)
@click.option('--fee', type=float, default=0.002)
@click.option('--interval', type=str, default='1Min')
def learn(data_file, **kwargs):
    hyperparameters = {
        'dropout_prob': 0.,
        'epochs': 1,
        'epsilon_decay': 2,
        'epsilon_end': 0.,
        'epsilon_start': 1.,
        'gamma': 0.9,
        'hidden_units': 10,
        'learn_rate': 0.001,
        'regularization_strength': 0.,
        'renorm_decay': 0.9,
        'replay_batch_size': 32,
        'replay_memory_max_size': 100000,
        'rnn_dropout_prob': 0.,
        'rnn_layers': 2,
        'trace_length': 16,
        'update_target_every': 500
    }
    random_seed = 999999
    environment = QFinanceEnvironment.from_csv(data_file, **kwargs)
    agent = QFinanceAgent(environment, random_seed=random_seed)
    agent.train(**hyperparameters)


if __name__ == '__main__':
    learn()
