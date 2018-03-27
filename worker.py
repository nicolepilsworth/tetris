import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import util
import random
import operator
import threading
import multiprocessing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal

from random import choice
from time import time
from time import sleep

from tetrominos import Tetromino, createTetrominos
from board import Board
from acNetwork import AC_Network

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

class Worker():
    def __init__(self,name,s_size,a_size,trainer,global_episodes, board):
        self.name = "worker_" + str(name)
        self.number = name
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.board = board
        self.a_size = a_size

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)

        self.actions = list(range(board.ncols * 4))

    def train(self,global_AC,rollout,sess,gamma,bootstrap_value):
        # print("training")
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]
        tetrominos_seen = rollout[:,6]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.imageIn:np.vstack(observations),
            self.local_AC.tetromino:np.vstack(tetrominos_seen),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages}
        v_l,p_l,e_l,g_n,v_n,adv, apl_g = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.adv_sum,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n, v_n, adv/len(rollout)

    def work(self,max_episode_length,gamma,global_AC,sess,coord,saveFreq):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        actions_list = np.arange(self.a_size)
        print("Starting worker " + str(self.number))
        tetrominos = createTetrominos()
        n_tetrominos = len(tetrominos) - 1

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                self.board.reset()
                tetromino_idx = random.randint(0, n_tetrominos)
                tetromino = tetrominos[tetromino_idx]
                possibleMoves = tetromino.getPossibleMoves(self.board)
                s = util.a3cState(self.board)

                episode_frames.append(s)

                while True:
                    # bool_moves = [(x in possibleMoves) for x in range(self.a_size)]
                    #Take an action using probabilities from policy network output.
                    a_dist,v = sess.run([self.local_AC.policy,self.local_AC.value],
                        feed_dict={self.local_AC.imageIn:s,
                                    self.local_AC.tetromino:np.reshape(tetromino_idx, (1, 1))})
                    # print(t_onehot)
                    # tetromino.printShape(0)
                    # self.board.printBoard()
                    valid_moves = [x if i in possibleMoves else 0. for i, x in enumerate(a_dist[0])]
                    sum_v = sum(valid_moves)


                    if sum_v == 0:
                      a = util.randChoice(possibleMoves)
                    else:
                      softmax_a_dist = [valid_moves/sum_v]
                    #   print(softmax_a_dist)
                      a = np.random.choice(actions_list,p=softmax_a_dist[0])
                    #   print(a)

                    # sleep(10)
                    # print(softmax_a_dist)
                    # print(a)
                    rot, col = divmod(a, self.board.ncols)
                    # print(rot, col)
                    r = self.board.act(tetromino, col, rot)

                    nextTetrominoIdx = random.randint(0, n_tetrominos)
                    nextTetromino = tetrominos[nextTetrominoIdx]
                    s1 = util.a3cState(self.board)

                    possibleMoves = nextTetromino.getPossibleMoves(self.board)
                    d = (len(possibleMoves) == 0)

                    episode_frames.append(s1)

                    episode_buffer.append([s,a,r,s1,d,v[0,0],np.reshape(tetromino_idx, (1, 1))])
                    episode_values.append(v[0,0])

                    tetromino = nextTetromino
                    tetromino_idx = nextTetrominoIdx

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # # If the episode hasn't ended, but the experience buffer is full, then we
                    # # make an update step using that experience rollout.
                    # if len(episode_buffer) ==  and d != True:
                    #     # Since we don't know what the true final return is, we "bootstrap" from our current
                    #     # value estimation.
                    #     v1 = sess.run(self.local_AC.value,
                    #         feed_dict={self.local_AC.imageIn:s,
                    #         self.local_AC.state_in[0]:rnn_state[0],
                    #         self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                    #     v_l,p_l,e_l,g_n,v_n, adv = self.train(global_AC,episode_buffer,sess,gamma,v1)
                    #     episode_buffer = []
                    #     sess.run(self.update_local_ops)
                    if  episode_step_count >= max_episode_length - 1:
                        print("reached max")
                        break
                    elif d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n,adv = self.train(global_AC,episode_buffer,sess,gamma,0.0)


                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % saveFreq == 0 and episode_count != 0:

                    mean_reward = np.mean(self.episode_rewards[-saveFreq:])
                    mean_length = np.mean(self.episode_lengths[-saveFreq:])
                    mean_value = np.mean(self.episode_mean_values[-saveFreq:])
                    print(mean_reward)

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
