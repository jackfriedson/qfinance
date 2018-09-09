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
from environment.common import Environment
from model.qestimator import QEstimator, ModelParametersCopier


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

if os.environ.get('ENV') == 'DOCKER':
    experiments_dir = Path('/var/lib/tensorboard')
else:
    experiments_dir = Path().resolve() / 'tensorboard'
experiments_dir.mkdir(exist_ok=True)

summaries_dir = experiments_dir/'summaries'
summaries_dir.mkdir(exist_ok=True)

models_dir = experiments_dir/'models'
models_dir.mkdir(exist_ok=True)


class QFinanceAgent(object):

    def __init__(self, environment: Environment, random_seed: int = None) -> None:
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.models_dir = models_dir/self.timestamp
        self.environment = environment
        self.random_seed = random_seed

    def train(self,
              gamma: float,
              epsilon_start: float,
              epsilon_end: float,
              epsilon_decay: float,
              replay_memory_max_size: int,
              replay_batch_size: int,
              trace_length: int,
              update_target_every: int,
              load_model: str = None,
              **kwargs):
        n_inputs = self.environment.n_state_factors
        n_outputs = self.environment.n_actions
        random = np.random.RandomState(self.random_seed)

        tf.reset_default_graph()
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if self.random_seed:
            tf.set_random_seed(self.random_seed)

        q_estimator = QEstimator('q_estimator', n_inputs, n_outputs, summaries_dir=summaries_dir, **kwargs)
        target_estimator = QEstimator('target_q', n_inputs, n_outputs, **kwargs)
        estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

        epsilon = tf.train.polynomial_decay(epsilon_start, global_step,
                                            self.environment.total_train_steps(),
                                            end_learning_rate=epsilon_end,
                                            power=epsilon_decay)
        policy = self._make_policy(q_estimator, epsilon, n_outputs, random)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            q_estimator.summary_writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())

            click.echo('Initializing replay memory...')
            replay_memory = ExperienceBuffer(replay_memory_max_size, random)
            for state in self.environment.replay_memories():
                action = random.randint(n_outputs)
                reward = self.environment.step(action)
                next_state = self.environment.state
                replay_memory.add(Transition(state, action, reward, next_state))

            for episode_i, (train_slice, validation_slice) in enumerate(self.environment.episodes()):
                replay_memory.new_episode()
                rnn_state = (np.zeros([1, n_inputs]), np.zeros([1, n_inputs]))

                click.echo('\nEpisode {}'.format(episode_i))
                train_bar = progressbar.ProgressBar(term_width=120,
                                                    max_value=self.environment.episode_train_length,
                                                    prefix='Training:')
                training_stats = defaultdict(list)

                for state in train_bar(train_slice):
                    # Maybe update the target network
                    if (sess.run(global_step) // replay_batch_size) % update_target_every == 0:
                        estimator_copy.make(sess)

                    # Make a prediction
                    action, next_rnn_state = policy(sess, state, rnn_state)
                    reward = self.environment.step(action)
                    next_state = self.environment.state

                    replay_memory.add(Transition(state, action, reward, next_state))
                    samples = replay_memory.sample(replay_batch_size, trace_length)
                    states_batch, action_batch, reward_batch, next_states_batch = map(np.array, zip(*samples))

                    # Train the network
                    train_rnn_state = (np.zeros([replay_batch_size, n_inputs]), np.zeros([replay_batch_size, n_inputs]))
                    q_values_next = target_estimator.predict(sess, next_states_batch, trace_length, train_rnn_state)[0]
                    targets_batch = reward_batch + gamma * np.amax(q_values_next, axis=1)
                    loss = q_estimator.update(sess, states_batch, action_batch, targets_batch, trace_length, train_rnn_state)

                    rnn_state = next_rnn_state

                    training_stats['rewards'].append(reward)
                    training_stats['losses'].append(loss)

                saver.save(sess, str(self.models_dir/'model.ckpt'))

                # Evaluate the model
                validation_stats = defaultdict(list)
                start_price = self.environment.last_price
                rnn_state = (np.zeros([1, n_inputs]), np.zeros([1, n_inputs]))

                val_bar = progressbar.ProgressBar(term_width=120,
                                                  max_value=self.environment.episode_validation_length,
                                                  prefix='Evaluating:')

                for state in val_bar(validation_slice):
                    q_values, next_rnn_state = q_estimator.predict(sess, np.expand_dims(state, 0), 1, rnn_state, training=False)
                    action = np.argmax(q_values)
                    reward = self.environment.step(action, track_orders=True)

                    # Calculate validation loss for summaries
                    next_state = self.environment.state
                    next_q_values = q_estimator.predict(sess, np.expand_dims(next_state, 0), 1, next_rnn_state, training=False)[0]
                    target = reward + gamma * np.amax(next_q_values)
                    loss = q_estimator.compute_loss(sess, state, action, target, rnn_state)

                    rnn_state = next_rnn_state

                    validation_stats['rewards'].append(reward)
                    validation_stats['losses'].append(loss)

                # Compute outperformance of market return
                market_return = (self.environment.last_price / start_price) - 1.
                position_value = start_price
                for return_val in self.environment.order_returns():
                    position_value *= 1 + return_val
                algorithm_return = (position_value / start_price) - 1.
                outperformance = algorithm_return - market_return
                click.echo('Market return: {:.2f}%'.format(100 * market_return))
                click.echo('Outperformance: {:+.2f}%'.format(100 * outperformance))

                # Plot results and save to summary file
                buf = io.BytesIO()
                self.environment.plot(save_to=buf)
                buf.seek(0)
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                image = tf.expand_dims(image, 0)
                episode_chart = tf.summary.image('episode_{}'.format(episode_i), image, max_outputs=1).eval()

                # Add Tensorboard summaries
                episode_summary = tf.Summary()
                episode_summary.value.add(simple_value=sess.run(epsilon), tag='episode/train/epsilon')
                episode_summary.value.add(simple_value=np.average(training_stats['rewards']),
                                          tag='episode/train/avg_reward')
                episode_summary.value.add(simple_value=np.average(training_stats['losses']),
                                          tag='episode/train/avg_loss')
                episode_summary.value.add(simple_value=np.average(validation_stats['rewards']),
                                          tag='episode/validate/avg_reward')
                episode_summary.value.add(simple_value=np.average(validation_stats['losses']),
                                          tag='episode/validate/avg_loss')
                episode_summary.value.add(simple_value=outperformance, tag='episode/validate/outperformance')
                q_estimator.summary_writer.add_summary(episode_summary, episode_i)
                q_estimator.summary_writer.add_summary(episode_chart, episode_i)
                q_estimator.summary_writer.flush()


    @staticmethod
    def _make_policy(estimator, epsilon, n_actions, random_state):
        def policy_fn(sess, observation, rnn_state):
            epsilon_val = sess.run(epsilon)
            q_values, new_rnn_state = estimator.predict(sess, np.expand_dims(observation, 0), 1, rnn_state)
            best_action = np.argmax(q_values)
            action_probs = np.ones(n_actions, dtype=float) * epsilon_val / n_actions
            action_probs[best_action] += (1.0 - epsilon_val)
            return random_state.choice(np.arange(len(action_probs)), p=action_probs), new_rnn_state
        return policy_fn

    def run(self):
        pass
