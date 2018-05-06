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
    def __init__(self,s_size,a_size,scope,trainer,nHLayers, nCLayers):


        with tf.variable_scope(scope):
          with tf.name_scope("board_input"):
            #Input and visual encoding layers
            self.imageIn = tf.placeholder(shape=s_size,dtype=tf.float32,name="board_input")
            # self.cnnLayers = [tf.layers.conv2d(inputs=self.imageIn, filters=16, kernel_size=[3, 3], padding="same", name="conv1")]
            #
            # for c in range(nCLayers - 1):
            #     self.cnnLayers.append(tf.layers.conv2d(inputs=self.cnnLayers[-1], filters=2^(5+c), kernel_size=[3, 3], padding="same"))
            #
            # self.hidden = [tf.contrib.layers.fully_connected(slim.flatten(self.cnnLayers[-1]) , 32, activation_fn=tf.nn.relu, biases_initializer=None)]
            # for l in range(nHLayers):
            #     self.hidden.append(tf.contrib.layers.fully_connected(self.hidden[-1] , 64, activation_fn=tf.nn.relu, biases_initializer=None))

            # hidden1 = tf.contrib.layers.fully_connected(self.imageIn , 32, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)

            ##########################################
            # NOT USED:
            # hidden2 = tf.contrib.layers.fully_connected(hidden1, 32, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)
            # self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=1)
            # self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=1)
            # self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=16, kernel_size=[1, 1])
            # self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3, pool_size=[2, 2], strides=1)
            # pool2_flat = tf.reshape(self.conv2, [-1, 64])
            # dense = tf.layers.dense(inputs=self.conv1, units=16, activation=tf.nn.relu)
            # self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3])
            self.conv1 = slim.conv2d(
                self.imageIn,12,
                kernel_size=[3,3],stride=1,weights_initializer=tf.contrib.layers.xavier_initializer(),padding='SAME')
            hidden = slim.fully_connected(slim.flatten(self.conv1),40,activation_fn=tf.nn.relu)
            # # (previously [8, 8], [4, 4])
            if nCLayers >= 2:
                self.conv2 = slim.conv2d(
                    self.conv1,24,
                    kernel_size=[3,3],stride=1,weights_initializer=tf.contrib.layers.xavier_initializer(),padding='SAME')
                hidden = slim.fully_connected(slim.flatten(self.conv2),40,activation_fn=tf.nn.relu)
            if nCLayers >= 4:
                self.conv3 = slim.conv2d(
                    self.conv2,32,
                    kernel_size=[3,3],stride=1,weights_initializer=tf.contrib.layers.xavier_initializer(),padding='SAME')
                self.conv4 = slim.conv2d(
                    self.conv3,64,
                    kernel_size=[3,3],stride=1,weights_initializer=tf.contrib.layers.xavier_initializer(),padding='SAME')
                hidden = slim.fully_connected(slim.flatten(self.conv4),40,activation_fn=tf.nn.relu)
            if nCLayers >=6:
                self.conv5 = slim.conv2d(
                    self.conv4,64,
                    kernel_size=[3,3],stride=1,weights_initializer=tf.contrib.layers.xavier_initializer(),padding='SAME')
                self.conv6 = slim.conv2d(
                    self.conv5,64,
                    kernel_size=[3,3],stride=1,weights_initializer=tf.contrib.layers.xavier_initializer(),padding='SAME')
                hidden = slim.fully_connected(slim.flatten(self.conv6),40,activation_fn=tf.nn.relu)
            ##########################################

          # with tf.name_scope("fully_connected_layer"):

            hidden2= slim.fully_connected(hidden,60,activation_fn=tf.nn.relu)

        ##########################################
        # NOT USED:
            # flatten_layer = tf.contrib.layers.flatten(hidden2)
            # dense_connected_layer = tf.contrib.layers.fully_connected(hidden, 128, activation_fn=tf.nn.relu)
            # dense_connected2 = tf.contrib.layers.fully_connected(dense_connected_layer, 48, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)
        ##########################################

          # with tf.name_scope("tetromino_input"):
          #   self.tetromino = tf.placeholder(shape=[None, 1],dtype=tf.int32, name="tetromino_idx")
          #   self.tetromino_onehot = tf.reshape(tf.one_hot(self.tetromino,7,dtype=tf.float32), shape=[-1, 7], name="tetromino_onehot")
          # with tf.name_scope("layer_concat"):
          #   self.concat_layer = tf.concat([self.tetromino_onehot, hidden], 1, name="concatenation")
          # with tf.name_scope("fully_connected_layers"):
          #   hidden2 = slim.fully_connected(self.concat_layer,64,activation_fn=tf.nn.relu)
          #   hidden3 = slim.fully_connected(hidden2,128,activation_fn=tf.nn.relu)
          # with tf.name_scope("dropout"):
          #   self.dropout = tf.layers.dropout(
          #       inputs=hidden3, rate=0.4, training=True, name="dropout")
          # with tf.name_scope("output"):
          #   self.policy = tf.layers.dense(inputs=self.dropout, units=a_size, activation=tf.nn.softmax, name="policy")
          #   self.value = tf.layers.dense(inputs=self.dropout, units=1, name="value")

            self.policy = tf.layers.dense(inputs=hidden2, units=a_size, activation=tf.nn.softmax, name="policy")
            self.value = tf.layers.dense(inputs=hidden2, units=1, name="value")


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
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01
                self.adv_sum = tf.reduce_sum(self.advantages)

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)

                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
