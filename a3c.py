import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import util
import operator
import threading
import multiprocessing
from time import sleep
from time import time

from random import choice

from tetrominos import Tetromino, createTetrominos
from board import Board
from acNetwork import AC_Network
from worker import Worker

def stop_training(coord):
  print("stopping training")
  coord.request_stop()

def train(nrows, ncols, max_episode_length, saveFreq, nGames, lr, nLayers):
    gamma = .9 # discount rate for advantage estimation and reward discounting

    # Tetris initialisations
    tetrominos = createTetrominos()
    board = Board(nrows, ncols)

    # s_size = [
    #   None,
    #   4,
    #   board.ncols,
    #   1
    # ]

    # TODO: switch for tetromino onehot
    tShape = tetrominos[0].paddedRotations[0]
    tShapeRows, tShapeCols = tShape.shape[0], tShape.shape[1]
    s_size = [None, (board.nrows * board.ncols) + (tShapeRows * tShapeCols)]

    a_size=board.ncols*4

    tf.reset_default_graph()

    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    # trainer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.85, epsilon=0.1)
    trainer = tf.train.AdamOptimizer(learning_rate=lr)
    master_network = AC_Network(s_size,a_size,'global',None,nLayers) # Generate global network
    # num_workers = 1
    num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        board = Board(nrows, ncols)
        workers.append(Worker(i,s_size,a_size,trainer,global_episodes, board,nLayers))

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        # main_timer = threading.Timer((60 * 50), stop_training, args=(coord,))
        # main_timer.start()

        sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length,gamma,master_network,sess,coord,saveFreq, nGames)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)


        coord.join(worker_threads)
        return list(map(lambda x: x.averages, workers))
