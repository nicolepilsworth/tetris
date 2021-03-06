import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import util
import operator
import threading
import multiprocessing
import tensorflow as tf
import tensorflow.contrib.slim as slim

from random import choice
from time import time

from tetrominos import Tetromino, createTetrominos
from board import Board


#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):

        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.imageIn = tf.placeholder(shape=s_size,dtype=tf.float32,name="imageIn")
            self.conv1 = tf.layers.conv2d(inputs=self.imageIn, filters=32, kernel_size=[3, 3])
            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=1)
            self.conv2 = tf.layers.conv2d(inputs=self.pool1, filters=32, kernel_size=[3, 3])
            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=1)
            pool2_flat = tf.reshape(self.pool2, [-1, 4 * 5 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

            # self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3])
            # self.conv1 = tf.nn.conv2d(activation_fn=tf.nn.elu,
            #     inputs=self.imageIn,num_outputs=32,
            #     kernel_size=[3,3],stride=[1,1],padding='VALID')
            # # (previously [8, 8], [4, 4])
            # self.conv2 = tf.nn.conv2d(activation_fn=tf.nn.elu,
            #     inputs=self.conv1,num_outputs=32,
            #     kernel_size=[3,3],stride=[1,1],padding='VALID')
            # hidden = slim.fully_connected(slim.flatten(self.conv2),128,activation_fn=tf.nn.elu)
            # flatten_layer = tf.contrib.layers.flatten(self.conv1)
            # dense_connected_layer = tf.contrib.layers.fully_connected(flatten_layer, 48, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)
            self.tetromino = tf.placeholder(shape=[None, 1],dtype=tf.int32)
            self.tetromino_onehot = tf.reshape(tf.one_hot(self.tetromino,7,dtype=tf.float32), shape=[-1, 7])

            self.concat_layer = tf.concat([self.tetromino_onehot, dense], 1)
            hidden = tf.layers.dense(inputs=self.concat_layer,units=64,activation=tf.nn.relu)
            self.dropout = tf.layers.dropout(
                inputs=hidden, rate=0.4, training=True)
            self.policy = tf.layers.dense(inputs=self.dropout, units=a_size, activation=tf.nn.softmax)
            self.value = tf.layers.dense(inputs=self.dropout, units=1)

              #Output layers for policy and value estimations
            #   self.policy = slim.fully_connected(self.dropout,a_size,
            #       activation_fn=tf.nn.softmax,
            #       weights_initializer=normalized_columns_initializer(0.01),
            #       biases_initializer=None)
            #   self.value = slim.fully_connected(self.dropout,1,
            #       activation_fn=None,
            #       weights_initializer=normalized_columns_initializer(1.0),
            #       biases_initializer=None)

            # self.p = tf.placeholder(tf.bool, [1,a_size])
            # self.invalid_moves = tf.constant(0., shape=[1,a_size])
            # self.policy = tf.where(self.p, self.policy_all, self.invalid_moves)  # Replace invalid moves in policy_all by 0.
            # self.value =  tf.where(self.p, self.value_all, self.invalid_moves)

            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32,name="actions")
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32,name="advantages")

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 10e-6))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 10e-6)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.1
                self.adv_sum = tf.reduce_sum(self.advantages)

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)

                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
