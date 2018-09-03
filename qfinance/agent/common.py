import io
import functools
import os
import time
from collections import namedtuple
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import progressbar
import tensorflow as tf

from agent.experience_buffer import ExperienceBuffer
from environment.common import QFinanceEnvironment
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

    def __init__(self, environment: QFinanceEnvironment, random_seed: int = None) -> None:
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.models_dir = models_dir/self.timestamp
        self.environment = environment
        self.random_seed = random_seed

    def train(self,
              epochs: int,
              gamma: float,
              epsilon_start: float,
              epsilon_end: float,
              epsilon_decay: float,
              replay_memory_max_size: int,
              batch_size: int,
              trace_length: int,
              update_target_every: int,
              load_model: str = None,
              **kwargs):
        # TODO: save training params to file for later reference

        n_inputs = self.environment.n_state_factors
        n_outputs = self.environment.n_actions
        random = np.random.RandomState(self.random_seed)

        tf.reset_default_graph()
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if self.random_seed:
            tf.set_random_seed(self.random_seed)

        cell = tf.contrib.rnn.LSTMCell(num_units=n_inputs, state_is_tuple=True, activation=tf.nn.softsign)
        target_cell = tf.contrib.rnn.LSTMCell(num_units=n_inputs, state_is_tuple=True, activation=tf.nn.softsign)
        q_estimator = QEstimator('q_estimator', cell, n_inputs, n_outputs, summaries_dir=summaries_dir, **kwargs)
        target_estimator = QEstimator('target_q', target_cell, n_inputs, n_outputs, **kwargs)
        estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

        epsilon = tf.train.polynomial_decay(epsilon_start, global_step,
                                            self.environment.total_train_steps(epochs),
                                            end_learning_rate=epsilon_end,
                                            power=epsilon_decay)
        policy = self._make_policy(q_estimator, epsilon, n_outputs, random)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            click.echo('Initializing replay memory...')
            replay_memory = ExperienceBuffer(replay_memory_max_size, random)
            for state in self.environment.replay_memories():
                action = random.randint(n_outputs)
                reward = self.environment.step(action)
                next_state = self.environment.state
                replay_memory.add(Transition(state, action, reward, next_state))

            for slice_i, full_slice in enumerate(self.environment.training_slices(epochs)):
                for epoch_i, (train_slice, validation_slice) in enumerate(full_slice):
                    absolute_epoch = (slice_i * epochs) + epoch_i

                    replay_memory.new_episode()
                    rnn_state = (np.zeros([1, n_inputs]), np.zeros([1, n_inputs]))

                    click.echo('\nSlice {}; Epoch {}'.format(slice_i, epoch_i))
                    train_bar = progressbar.ProgressBar(term_width=120,
                                                        max_value=self.environment.fold_train_length,
                                                        prefix='Training:')
                    train_rewards = losses = []

                    for state in train_bar(train_slice):
                        # Maybe update the target network
                        if (sess.run(global_step) // batch_size) % update_target_every == 0:
                            estimator_copy.make(sess)

                        # Make a prediction
                        action, next_rnn_state = policy(sess, state, rnn_state)
                        reward = self.environment.step(action)
                        next_state = self.environment.state

                        replay_memory.add(Transition(state, action, reward, next_state))
                        samples = replay_memory.sample(batch_size, trace_length)
                        states_batch, action_batch, reward_batch, next_states_batch = map(np.array, zip(*samples))

                        # Train the network
                        train_rnn_state = (np.zeros([batch_size, n_inputs]), np.zeros([batch_size, n_inputs]))
                        q_values_next = target_estimator.predict(sess, next_states_batch, trace_length, train_rnn_state)[0]
                        targets_batch = reward_batch + gamma * np.amax(q_values_next, axis=1)
                        loss = q_estimator.update(sess, states_batch, action_batch, targets_batch, trace_length, train_rnn_state)

                        rnn_state = next_rnn_state

                        train_rewards.append(reward)
                        losses.append(loss)

                    saver.save(sess, str(self.models_dir/'model.ckpt'))

                    # Evaluate the model
                    rewards = val_losses = []
                    start_price = self.environment.last_price
                    rnn_state = (np.zeros([1, n_inputs]), np.zeros([1, n_inputs]))

                    val_bar = progressbar.ProgressBar(term_width=120,
                                                      max_value=self.environment.fold_validation_length,
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

                        rewards.append(reward)
                        val_losses.append(loss)

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
                    epoch_chart = tf.summary.image('epoch_{}'.format(absolute_epoch), image, max_outputs=1).eval()

                    # Add Tensorboard summaries
                    epoch_summary = tf.Summary()
                    epoch_summary.value.add(simple_value=sess.run(epsilon), tag='epoch/train/epsilon')
                    epoch_summary.value.add(simple_value=sum(train_rewards), tag='epoch/train/reward')
                    epoch_summary.value.add(simple_value=np.average(losses), tag='epoch/train/averge_loss')
                    epoch_summary.value.add(simple_value=sum(rewards), tag='epoch/validate/reward')
                    epoch_summary.value.add(simple_value=outperformance, tag='epoch/validate/outperformance')
                    epoch_summary.value.add(simple_value=np.average(val_losses), tag='epoch/validate/average_loss')
                    q_estimator.summary_writer.add_summary(epoch_summary, absolute_epoch)
                    q_estimator.summary_writer.add_summary(epoch_chart, absolute_epoch)
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
