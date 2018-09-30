import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class DDPG(object):

    def __init__(self,
                 scope: str,
                 n_inputs: int,
                 n_outputs: int,
                 actor_units: int,
                 actor_learn_rate: float,
                 critic_learn_rate: float,
                 tau: float,
                 gamma: float,
                 summaries_dir: str = None):
        self.scope = scope
        self.state_dim = n_inputs
        self.actor_units = actor_units
        self.action_dim = n_outputs

        self.softmax_in = tf.placeholder(shape=[None], dtype=tf.float32, name='softmax_in')
        self.softmax = tf.nn.softmax(self.softmax_in)

        with tf.variable_scope(self.scope):
            self.states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='states')
            self.next_states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='next_states')
            self.rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='rewards')
            self.phase = tf.placeholder(dtype=tf.bool, name='phase')
            self.trace_length = tf.placeholder(dtype=tf.int32, name='trace_length')

            # if fc_units is None:
            #     fc_units = self.state_dim

            batch_size = tf.shape(self.states)[0]
            rnn_batch_size = tf.reshape(batch_size // self.trace_length, shape=[])

            # Fully connected layer
            # self.fc_layer = slim.fully_connected(batch_norm, fc_units, activation_fn=tf.nn.elu, biases_initializer=None)
            # self.fc_flat = tf.reshape(self.fc_layer, shape=[rnn_batch_size, self.trace_length, fc_units])

            batch_norm = tf.layers.batch_normalization(self.states,
                                                       name='batch_norm',
                                                       renorm=True,
                                                       training=self.phase)
            flat = tf.reshape(batch_norm, shape=[rnn_batch_size, self.trace_length, self.state_dim])

            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.state_dim,
                                               state_is_tuple=True,
                                               activation=tf.nn.softsign,
                                               name='lstm_cell')
            self.rnn_in = rnn_cell.zero_state(rnn_batch_size, dtype=tf.float32)
            rnn, self.rnn_state = tf.nn.dynamic_rnn(rnn_cell, flat, dtype=tf.float32,
                                                    initial_state=self.rnn_in, scope='rnn')
            rnn = tf.reshape(rnn, shape=tf.shape(batch_norm), name='rnn_out')

            a = self._build_actor(rnn)
            self.actor_out = tf.nn.softmax(a)
            q = self._build_critic(rnn, a)
            actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{}/Actor'.format(self.scope))
            critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{}/Critic'.format(self.scope))

            # Target
            next_batch_norm = tf.layers.batch_normalization(self.next_states,
                                                            name='batch_norm',
                                                            reuse=True,
                                                            trainable=False,
                                                            training=self.phase)
            next_flat = tf.reshape(next_batch_norm, shape=[rnn_batch_size, self.trace_length, self.state_dim])
            next_rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.state_dim,
                                                    state_is_tuple=True,
                                                    activation=tf.nn.softsign,
                                                    reuse=True,
                                                    name='lstm_cell')
            self.next_rnn_in = next_rnn_cell.zero_state(rnn_batch_size, dtype=tf.float32)
            next_rnn, next_rnn_state = tf.nn.dynamic_rnn(next_rnn_cell, next_flat, dtype=tf.float32,
                                                         initial_state=self.next_rnn_in, scope='rnn')
            next_rnn = tf.reshape(rnn, shape=tf.shape(next_batch_norm), name='rnn_out')

            ema = tf.train.ExponentialMovingAverage(decay=1-tau)
            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))

            target_update = [ema.apply(actor_params), ema.apply(critic_params)]
            a_ = self._build_actor(next_rnn, reuse=True, custom_getter=ema_getter)
            q_ = self._build_critic(next_rnn, a_, reuse=True, custom_getter=ema_getter)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                actor_loss = -tf.reduce_mean(q)
                actor_optimizer = tf.train.AdamOptimizer(actor_learn_rate)
                self.train_actor = actor_optimizer.minimize(actor_loss, var_list=actor_params)

                with tf.control_dependencies(target_update):
                    q_target = self.rewards + gamma * q_
                    td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
                    critic_optimizer = tf.train.AdamOptimizer(critic_learn_rate)
                    self.train_critic = critic_optimizer.minimize(td_error, var_list=critic_params)

            summaries = [
                tf.summary.scalar('actor_loss', actor_loss),
                tf.summary.scalar('td_error', td_error)
            ]
            self.summaries = tf.summary.merge(summaries)

            self.summary_writer = None
            if summaries_dir:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                summary_dir = summaries_dir/'{}_{}'.format(scope, timestamp)
                summary_dir.mkdir(exist_ok=True)
                self.summary_writer = tf.summary.FileWriter(str(summary_dir))

    def _build_actor(self, input_layer, reuse=None, custom_getter=None):
        trainable = reuse is None
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            fc = tf.layers.dense(input_layer, self.actor_units, name='l1',
                                 activation=tf.nn.relu, trainable=trainable)
            return tf.layers.dense(fc, self.action_dim, activation=tf.nn.tanh,
                                   name='a', trainable=trainable)

    def _build_critic(self, input_layer, actor_out, reuse=None, custom_getter=None):
        trainable = reuse is None
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            w1_s = tf.get_variable('w1_s', [self.state_dim, self.actor_units], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.action_dim, self.actor_units], trainable=trainable)
            b1 = tf.get_variable('b1', [1, self.actor_units], trainable=trainable)
            fc = tf.nn.relu(tf.matmul(input_layer, w1_s) + tf.matmul(actor_out, w1_a) + b1)
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
        batch_size = len(state) // trace_length
        feed_dict = {
            self.states: state,
            self.actor_out: action,
            self.rewards: np.expand_dims(reward, 1),
            self.next_states: next_state,
            self.phase: True,
            self.trace_length: trace_length,
            self.rnn_in: self.zero_rnn_state(batch_size),
            self.next_rnn_in: self.zero_rnn_state(batch_size),
        }
        sess.run(self.train_actor, feed_dict)
        sess.run(self.train_critic, feed_dict)

        if self.summary_writer:
            summary_ops = [self.summaries, tf.train.get_global_step()]
            summaries, global_step = sess.run(summary_ops, feed_dict)
            self.summary_writer.add_summary(summaries, global_step)

    def apply_softmax(self, sess, input):
        return sess.run(self.softmax, {self.softmax_in: input})

    def zero_rnn_state(self, batch_size: int):
        return (np.zeros([batch_size, self.state_dim]), np.zeros([batch_size, self.state_dim]))

    # def compute_loss(self, sess, state, action, target, rnn_state):
    #     feed_dict = {
    #         self.states: np.array([state]),
    #         self.rewards: np.array([target]),
    #         self.actor_out: np.array([action]),
    #         self.phase: False,
    #         self.trace_length: 1,
    #         self.rnn_in: rnn_state,
    #     }
    #     return sess.run(self.loss, feed_dict)
