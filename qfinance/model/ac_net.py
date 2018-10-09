import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


GLOBAL_SCOPE = 'global'


class AC_Network(object):

    def __init__(self,
                 scope: str,
                 n_inputs: int,
                 n_outputs: int,
                 actor_size: int,
                 critic_size: int,
                 actor_optimizer,
                 critic_optimizer,
                 gamma: float,
                 entropy_beta: float,
                 global_ac = None,
                 summaries_dir: str = None,
                 **kwargs):
        self.scope = scope
        self.state_dim = n_inputs
        self.action_dim = n_outputs
        self.gamma = gamma

        with tf.variable_scope(self.scope):
            self.states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='states')
            self.phase = tf.placeholder(dtype=tf.bool, name='phase')
            self.trace_length = tf.placeholder(dtype=tf.int32, name='trace_length')

            if self.scope == GLOBAL_SCOPE:
                _, _, _, self.a_params, self.c_params = self._build_network(actor_size, critic_size)
            else:
                self.a_his = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name='actions')
                self.v_target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='v_target')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_network(actor_size, critic_size)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu, sigma + 1e-4

                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()
                    self.exp_v = entropy_beta * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):
                    self.A = tf.nn.softmax(tf.squeeze(normal_dist.sample(1), axis=[0, 1]))
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

                self.summaries = tf.summary.merge([
                    tf.summary.scalar('actor_loss', self.a_loss),
                    tf.summary.scalar('critic_loss', self.c_loss)
                ])
                self.summary_writer = None
                if summaries_dir:
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    summary_dir = summaries_dir/'{}_{}'.format(scope, timestamp)
                    summary_dir.mkdir(exist_ok=True)
                    self.summary_writer = tf.summary.FileWriter(str(summary_dir))

                with tf.name_scope('sync'):
                    with tf.name_scope('pull'):
                        self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_ac.a_params)]
                        self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_ac.c_params)]
                    with tf.name_scope('push'):
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.scope)
                        with tf.control_dependencies(update_ops):
                            self.update_a_op = actor_optimizer.apply_gradients(zip(self.a_grads, global_ac.a_params))
                            self.update_c_op = critic_optimizer.apply_gradients(zip(self.c_grads, global_ac.c_params))

    def _build_network(self, actor_size, critic_size):
        batch_size = tf.shape(self.states)[0]
        rnn_batch_size = tf.reshape(batch_size // self.trace_length, shape=[])
        w_init = tf.random_normal_initializer(0., .1)

        with tf.variable_scope('batch_norm'):
            batch_norm = tf.layers.batch_normalization(self.states,
                                                       name='batch_norm',
                                                       renorm=True,
                                                       training=self.phase)
            flat = tf.reshape(batch_norm, shape=[rnn_batch_size, self.trace_length, self.state_dim])

        with tf.variable_scope('critic'):
            # RNN layers
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.state_dim,
                                               state_is_tuple=True,
                                               activation=tf.nn.softsign,
                                               name='lstm_cell')
            self.rnn_in = rnn_cell.zero_state(rnn_batch_size, dtype=tf.float32)
            rnn, self.rnn_state = tf.nn.dynamic_rnn(rnn_cell, flat, dtype=tf.float32, initial_state=self.rnn_in)
            rnn = tf.reshape(rnn, shape=tf.shape(batch_norm))

            l_c = tf.layers.dense(rnn, critic_size, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')

        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(rnn, actor_size, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, self.action_dim, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, self.action_dim, tf.nn.softplus, kernel_initializer=w_init, name='sigma')

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def choose_action(self, sess, state, rnn_state, training: bool = True):
        feed_dict = {
            self.states: np.expand_dims(state, axis=0),
            self.phase: training,
            self.trace_length: 1,
            self.rnn_in: rnn_state,
        }
        return sess.run([self.A, self.rnn_state], feed_dict)

    def update_global(self, sess, state, action, reward, next_state, trace_length):
        rnn_batch_size = len(state) // trace_length
        v_next = sess.run(self.v, feed_dict={
            self.phase: True,
            self.states: next_state,
            self.trace_length: trace_length,
            # TODO: fix this
            self.rnn_in: self.zero_rnn_state(batch_size=rnn_batch_size)
        })
        v_target = np.expand_dims(reward, 1) + self.gamma * v_next
        feed_dict = {
            self.states: state,
            self.a_his: action,
            self.v_target: v_target,
            self.phase: True,
            self.trace_length: trace_length,
            self.rnn_in: self.zero_rnn_state(batch_size=rnn_batch_size),
        }
        sess.run([self.update_a_op, self.update_c_op], feed_dict)

        # if self.summary_writer:
        #     summary_ops = [self.summaries, tf.train.get_global_step()]
        #     summaries, global_step = sess.run(summary_ops, feed_dict)
        #     self.summary_writer.add_summary(summaries, global_step)

    def pull_global(self, sess):
        sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def zero_rnn_state(self, batch_size: int = 1):
        return (np.zeros([batch_size, self.state_dim]), np.zeros([batch_size, self.state_dim]))
