import io
import functools
import os
import time
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import progressbar
import tensorflow as tf

from agent.experience_buffer import ExperienceBuffer
from environment.core import Environment
from model.ddpg import DDPG


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

# TODO: Move to a config
if os.environ.get('ENV') == 'DOCKER':
    experiments_dir = Path('/var/lib/tensorboard')
else:
    experiments_dir = Path().resolve() / 'tensorboard'
experiments_dir.mkdir(exist_ok=True)

summaries_dir = experiments_dir/'summaries'
summaries_dir.mkdir(exist_ok=True)

models_dir = experiments_dir/'models'
models_dir.mkdir(exist_ok=True)


class Agent(object):

    def __init__(self, environment: Environment, random_seed: int = None) -> None:
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.models_dir = models_dir/self.timestamp
        self.environment = environment
        self.random_seed = random_seed

    def train(self,
              epsilon_start: float,
              epsilon_end: float,
              epsilon_decay: float,
              memory_max_size: int,
              batch_size: int,
              trace_length: int,
              load_model: str = None,
              **kwargs):
        n_inputs = self.environment.state_space_dim
        n_outputs = self.environment.action_space_dim
        random = np.random.RandomState(self.random_seed)

        tf.reset_default_graph()
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if self.random_seed:
            tf.set_random_seed(self.random_seed)

        q_estimator = DDPG('DDPG',
                           n_inputs=n_inputs,
                           n_outputs=n_outputs,
                           summaries_dir=summaries_dir,
                           **kwargs)

        epsilon = tf.train.polynomial_decay(epsilon_start, global_step,
                                            self.environment.total_train_steps,
                                            end_learning_rate=epsilon_end,
                                            power=epsilon_decay)
        policy = self._make_policy(q_estimator, epsilon, random)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            q_estimator.summary_writer.add_graph(sess.graph)

            click.echo('Initializing replay memory...')
            replay_memory = ExperienceBuffer(memory_max_size, random)
            rnn_state = q_estimator.zero_rnn_state()
            for state in self.environment.replay_memories():
                action, rnn_state = policy(sess, state, rnn_state)
                reward = self.environment.step(action)
                next_state = self.environment.state
                replay_memory.add(Transition(state, action, reward, next_state))

            for episode_i, (train_slice, validation_slice) in enumerate(self.environment.episodes()):
                click.echo('\nEpisode {}'.format(episode_i))
                replay_memory.new_episode()
                rnn_state = q_estimator.zero_rnn_state()
                training_stats = defaultdict(list)
                train_bar = progressbar.ProgressBar(term_width=120,
                                                    max_value=self.environment.episode_train_length,
                                                    prefix='Training:')

                for state in train_bar(train_slice):
                    # Make a prediction
                    action, rnn_state = policy(sess, state, rnn_state)
                    reward = self.environment.step(action)
                    next_state = self.environment.state

                    replay_memory.add(Transition(state, action, reward, next_state))
                    samples = replay_memory.sample(batch_size, trace_length)
                    states_batch, action_batch, reward_batch, next_states_batch = map(np.array, zip(*samples))

                    # Train the network
                    q_estimator.update(sess, states_batch, action_batch, reward_batch,
                                       next_states_batch, trace_length)

                    training_stats['rewards'].append(reward)

                # Evaluate the model
                validation_stats = defaultdict(list)
                rnn_state = q_estimator.zero_rnn_state()
                val_bar = progressbar.ProgressBar(term_width=120,
                                                  max_value=self.environment.episode_validation_length,
                                                  prefix='Evaluating:')

                for state in val_bar(validation_slice):
                    # TODO: reset portfolio here so training doesn't affect starting position
                    action, rnn_state = policy(sess, state, rnn_state, is_training=False)
                    reward = self.environment.step(action)
                    state = self.environment.state

                    validation_stats['rewards'].append(reward)

                saver.save(sess, str(self.models_dir/'model.ckpt'))

                # TODO: Calculate sharpe ratio and outperformance of index
                episode_return = (self.environment.portfolio_value / self.environment.initial_funding) - 1.

                # Compute outperformance of market return
                # market_return = (self.environment.last_price / start_price) - 1.
                # position_value = start_price
                # for return_val in self.environment.order_returns():
                #     position_value *= 1 + return_val
                # algorithm_return = (position_value / start_price) - 1.
                # outperformance = algorithm_return - market_return
                # click.echo('Market return: {:.2f}%'.format(100 * market_return))
                # click.echo('Outperformance: {:+.2f}%'.format(100 * outperformance))

                # # Add Tensorboard summaries
                episode_summary = tf.Summary()
                episode_summary.value.add(simple_value=sess.run(epsilon), tag='episode/train/epsilon')
                episode_summary.value.add(simple_value=np.average(training_stats['rewards']),
                                          tag='episode/train/avg_reward')
                episode_summary.value.add(simple_value=np.average(validation_stats['rewards']),
                                          tag='episode/validate/avg_reward')
                episode_summary.value.add(simple_value=episode_return, tag='episode/validate/return')
                q_estimator.summary_writer.add_summary(episode_summary, episode_i)
                q_estimator.summary_writer.add_summary(self._episode_chart_summary(episode_i), episode_i)
                q_estimator.summary_writer.flush()

    def _episode_chart_summary(self, episode_num: int):
        buf = io.BytesIO()
        self.environment.plot(save_to=buf)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return tf.summary.image('episode_{}'.format(episode_num), image, max_outputs=1).eval()

    @staticmethod
    def _make_policy(estimator, epsilon, random):
        def policy_fn(sess, observation, rnn_state, is_training: bool = True):
            action, next_rnn_state = estimator.choose_action(sess, np.expand_dims(observation, 0), rnn_state)
            action = action[0]
            if is_training:
                # TODO: Fix the variance here
                action = random.normal(action, sess.run(epsilon))
                action = estimator.apply_softmax(sess, action)
            return action, next_rnn_state
        return policy_fn

    def run(self):
        pass
