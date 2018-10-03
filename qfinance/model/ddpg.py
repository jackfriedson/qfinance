import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class DDPG(object):

    def __init__(self,
                 scope: str,
                 n_inputs: int,
                 n_outputs: int,
                 hidden_units: int,
                 actor_learn_rate: float,
                 critic_learn_rate: float,
                 tau: float,
                 gamma: float,
                 summaries_dir: str = None):
        self.scope = scope
        self.state_dim = n_inputs
        self.hidden_units = hidden_units
        self.action_dim = n_outputs

        self.softmax_in = tf.placeholder(shape=[None], dtype=tf.float32, name='softmax_in')
        self.softmax = tf.nn.softmax(self.softmax_in)

        with tf.variable_scope(self.scope):
            self.states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='states')
            self.next_states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='next_states')
            self.rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='rewards')
            self.phase = tf.placeholder(dtype=tf.bool, name='phase')
            self.trace_length = tf.placeholder(dtype=tf.int32, name='trace_length')

            a, q, self.rnn_in, self.rnn_state = self._construct_network(self.states)
            self.actor_out = tf.nn.softmax(a)
            actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{}/Actor'.format(self.scope))
            critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{}/Critic'.format(self.scope))

            ema = tf.train.ExponentialMovingAverage(decay=1-tau)
            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))

            target_update = [ema.apply(actor_params), ema.apply(critic_params)]
            a_, q_, self.next_rnn_in, _ = self._construct_network(self.next_states, reuse=True, custom_getter=ema_getter)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                actor_loss = -tf.reduce_mean(q)
                actor_optimizer = tf.train.AdamOptimizer(actor_learn_rate)
                self.train_actor = actor_optimizer.minimize(actor_loss, var_list=actor_params,
                                                            global_step=tf.train.get_global_step())

                with tf.control_dependencies(target_update):
                    q_target = self.rewards + gamma * q_
                    td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
                    critic_optimizer = tf.train.AdamOptimizer(critic_learn_rate)
                    self.train_critic = critic_optimizer.minimize(td_error, var_list=critic_params,
                                                                  global_step=tf.train.get_global_step())

            self.summaries = tf.summary.merge([
                tf.summary.scalar('actor_loss', actor_loss),
                tf.summary.scalar('td_error', td_error)
            ])
            self.summary_writer = None
            if summaries_dir:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                summary_dir = summaries_dir/'{}_{}'.format(scope, timestamp)
                summary_dir.mkdir(exist_ok=True)
                self.summary_writer = tf.summary.FileWriter(str(summary_dir))

    def _construct_network(self, input_layer, reuse=None, custom_getter=None):
        trainable = reuse is None
        batch_size = tf.shape(self.states)[0]
        rnn_batch_size = tf.reshape(batch_size // self.trace_length, shape=[])

        # Batch normalize inputs
        batch_norm = tf.layers.batch_normalization(input_layer,
                                                   name='batch_norm',
                                                   renorm=True,
                                                   reuse=reuse,
                                                   trainable=trainable,
                                                   training=self.phase)
        flat = tf.reshape(batch_norm, shape=[rnn_batch_size, self.trace_length, self.state_dim])

        # RNN layers
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.state_dim,
                                           state_is_tuple=True,
                                           activation=tf.nn.softsign,
                                           name='lstm_cell',
                                           reuse=reuse)
        rnn_in = rnn_cell.zero_state(rnn_batch_size, dtype=tf.float32)
        rnn, rnn_state = tf.nn.dynamic_rnn(rnn_cell, flat, dtype=tf.float32,
                                           initial_state=rnn_in)
        rnn = tf.reshape(rnn, shape=tf.shape(batch_norm))

        a = self._build_actor(rnn, reuse=reuse, custom_getter=custom_getter)
        q = self._build_critic(rnn, a, reuse=reuse, custom_getter=custom_getter)
        return a, q, rnn_in, rnn_state

    def _build_actor(self, input_layer, reuse=None, custom_getter=None):
        trainable = reuse is None
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            fc = tf.layers.dense(input_layer, self.hidden_units, name='l1',
                                 activation=tf.nn.elu, trainable=trainable)
            return tf.layers.dense(fc, self.action_dim, activation=tf.nn.tanh,
                                   name='a', trainable=trainable)

    def _build_critic(self, input_layer, actor_out, reuse=None, custom_getter=None):
        trainable = reuse is None
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            w1_s = tf.get_variable('w1_s', [self.state_dim, self.hidden_units], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.action_dim, self.hidden_units], trainable=trainable)
            b1 = tf.get_variable('b1', [1, self.hidden_units], trainable=trainable)
            fc = tf.nn.elu(tf.matmul(input_layer, w1_s) + tf.matmul(actor_out, w1_a) + b1)
            return tf.layers.dense(fc, 1, trainable=trainable)

    def choose_action(self, sess, state, rnn_state, training: bool = True):
        feed_dict = {
            self.states: state,
            self.phase: training,
            self.trace_length: 1,
            self.rnn_in: rnn_state,
        }
        return sess.run([self.actor_out, self.rnn_state], feed_dict)

    def update(self, sess, state, action, reward, next_state, trace_length):
        rnn_batch_size = len(state) // trace_length
        feed_dict = {
            self.states: state,
            self.actor_out: action,
            self.rewards: np.expand_dims(reward, 1),
            self.next_states: next_state,
            self.phase: True,
            self.trace_length: trace_length,
            self.rnn_in: self.zero_rnn_state(batch_size=rnn_batch_size),
            self.next_rnn_in: self.zero_rnn_state(batch_size=rnn_batch_size),
        }
        sess.run(self.train_actor, feed_dict)
        sess.run(self.train_critic, feed_dict)

        if self.summary_writer:
            summary_ops = [self.summaries, tf.train.get_global_step()]
            summaries, global_step = sess.run(summary_ops, feed_dict)
            self.summary_writer.add_summary(summaries, global_step)

    def apply_softmax(self, sess, input):
        return sess.run(self.softmax, {self.softmax_in: input})

    def zero_rnn_state(self, batch_size: int = 1):
        return (np.zeros([batch_size, self.state_dim]), np.zeros([batch_size, self.state_dim]))
