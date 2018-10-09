import io
import functools
import multiprocessing
import os
import threading
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
from model.ac_net import AC_Network, GLOBAL_SCOPE


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


class AsyncAgent(object):

    def __init__(self, environment: Environment, random_seed: int = None) -> None:
        # self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        # self.models_dir = models_dir/self.timestamp
        self.env = environment
        self.random_seed = random_seed

    def train(self,
              actor_learn_rate: float,
              critic_learn_rate: float,
              load_model: bool = False,
              **kwargs):
        random_state = np.random.RandomState(self.random_seed)
        if self.random_seed:
            tf.set_random_seed(self.random_seed)

        tf.reset_default_graph()
        global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.device('/cpu:0'):
            kwargs.update({
                'n_inputs': self.env.state_space_dim,
                'n_outputs': self.env.action_space_dim,
                'actor_optimizer': tf.train.RMSPropOptimizer(actor_learn_rate, name='RMSPropA'),
                'critic_optimizer': tf.train.RMSPropOptimizer(critic_learn_rate, name='RMSPropC')
            })
            global_ac = AC_Network(GLOBAL_SCOPE, **kwargs)
            workers = [
                Worker(i, self.env.copy(), global_ac, random_state, **kwargs)
                for i in range(multiprocessing.cpu_count())
            ]

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            sess.run(tf.global_variables_initializer())

            worker_threads = []
            for worker in workers:
                worker_work = lambda: worker.work(coord, sess)
                t = threading.Thread(target=(worker_work))
                t.start()
                time.sleep(0.5)
                worker_threads.append(t)
            coord.join(worker_threads)

    def run(self):
        pass


class Worker(object):

    def __init__(self,
                 name: int,
                 environment: Environment,
                 global_ac: AC_Network,
                 random_state: np.random.RandomState,
                 memory_max_size: int,
                 batch_size: int,
                 trace_length: int,
                 **kwargs):
        self.name = 'worker_{}'.format(name)
        if self.name == 'worker_0':
            kwargs['summaries_dir'] = summaries_dir
        self.env = environment
        self.local_ac = AC_Network(self.name, global_ac=global_ac, **kwargs)
        self.batch_size = batch_size
        self.trace_length = trace_length
        self.replay_memory = ExperienceBuffer(memory_max_size, random_state)

    def work(self, coord, sess):
        if self.name == 'worker_0':
            self.local_ac.summary_writer.add_graph(sess.graph)

        rnn_state = self.local_ac.zero_rnn_state()
        for state in self.env.replay_memories():
            action, rnn_state = self.local_ac.choose_action(sess, state, rnn_state)
            reward = self.env.step(action)
            next_state = self.env.state
            self.replay_memory.add(Transition(state, action, reward, next_state))

        cumulative_return = 0.
        for episode_i, (get_train_states, get_val_states) in enumerate(self.env.episodes()):
            if coord.should_stop():
                break

            self.replay_memory.new_episode()
            rnn_state = self.local_ac.zero_rnn_state()
            training_stats = defaultdict(list)

            self.env.reset_portfolio()
            for state in get_train_states():
                # Make a prediction
                action, rnn_state = self.local_ac.choose_action(sess, state, rnn_state)
                reward = self.env.step(action)
                next_state = self.env.state

                # Add new example and train the network
                self.replay_memory.add(Transition(state, action, reward, next_state))
                samples = self.replay_memory.sample(self.batch_size, self.trace_length)
                states_batch, action_batch, reward_batch, next_states_batch = map(np.array, zip(*samples))
                self.local_ac.update_global(sess, states_batch, action_batch,
                                            reward_batch, next_states_batch, self.trace_length)
                self.local_ac.pull_global(sess)
                training_stats['rewards'].append(reward)

            if self.name == 'worker_0':
                # Evaluate the model
                validation_stats = defaultdict(list)
                rnn_state = self.local_ac.zero_rnn_state()

                self.env.reset_portfolio()
                for state in get_val_states():
                    action, rnn_state = self.local_ac.choose_action(sess, state, rnn_state, training=False)
                    reward = self.env.step(action)
                    state = self.env.state
                    validation_stats['rewards'].append(reward)

                # saver.save(sess, str(self.models_dir/'model.ckpt'))

                # TODO: Calculate outperformance of index
                episode_return = self.env.episode_return
                cumulative_return = ((1. + cumulative_return) * (1. + episode_return) - 1.)

                # Compute outperformance of market return
                # market_return = (self.env.last_price / start_price) - 1.
                # position_value = start_price    risk_free_data = load_tbill_data(Path(risk_free_data))
                # for return_val in self.env.order_returns():
                #     position_value *= 1 + return_val
                # algorithm_return = (position_value / start_price) - 1.
                # outperformance = algorithm_return - market_return
                # click.echo('Market return: {:.2f}%'.format(100 * market_return))
                # click.echo('Outperformance: {:+.2f}%'.format(100 * outperformance))

                # # Add Tensorboard summaries
                episode_summary = tf.Summary()
                # episode_summary.value.add(simple_value=sess.run(epsilon), tag='episode/train/epsilon')
                episode_summary.value.add(simple_value=np.average(training_stats['rewards']),
                                          tag='episode/train/avg_reward')
                episode_summary.value.add(simple_value=np.average(validation_stats['rewards']),
                                          tag='episode/validate/avg_reward')
                episode_summary.value.add(simple_value=episode_return, tag='episode/validate/episode_return')
                episode_summary.value.add(simple_value=self.env.validation_sharpe,
                                          tag='episode/validate/sharpe')
                episode_summary.value.add(simple_value=cumulative_return, tag='episode/validate/cumulative_return')
                self.local_ac.summary_writer.add_summary(episode_summary, episode_i)
                # self.local_ac.summary_writer.add_summary(self._episode_chart_summary(sess, episode_i), episode_i)
                self.local_ac.summary_writer.flush()

    def _episode_chart_summary(self, sess, episode_num: int):
        buf = io.BytesIO()
        self.env.plot(save_to=buf)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return tf.summary.image('episode_{}'.format(episode_num), image, max_outputs=1).eval(session=sess)
