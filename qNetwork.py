import numpy as np
import tensorflow as tf
import util
import operator

from tetrominos import Tetromino, createTetrominos
from board import Board
from graph import Graph

def learn(epsilon, gamma, alpha, nGames, getAvgs):
  tetrominos = createTetrominos()
  board = Board(5, 3)
  tShapeRows, tShapeCols = tuple(map(operator.add, tetrominos[0].shape.shape, (1, 1)))
  inputLayerDim = (board.nrows * board.ncols) + (tShapeRows * tShapeCols)
  actionsDim = board.ncols * 4

  # Tensorflow network initialisation
  tf.reset_default_graph()

  # These lines establish the feed-forward part of the network used
  # to choose actions
  inputs1 = tf.placeholder(shape=[1,inputLayerDim],dtype=tf.float32)
  W = tf.Variable(tf.zeros([inputLayerDim, actionsDim]))
  p = tf.placeholder(tf.bool, [1,actionsDim])
  Qout = tf.matmul(inputs1,W)
  invalidMoves = tf.constant(-100., shape=[1,actionsDim])
  validMoves = tf.where(p, Qout, invalidMoves)  # Replace invalid moves in Qout by -100
  predict = tf.argmax(validMoves,1)

  # Below we obtain the loss by taking the sum of squares difference between
  # the target and prediction Q values.
  nextQ = tf.placeholder(shape=[1,actionsDim],dtype=tf.float32)
  loss = tf.reduce_sum(tf.square(nextQ - Qout))
  trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
  updateModel = trainer.minimize(loss)

  init = tf.global_variables_initializer()

  # create lists to contain total rewards and steps per episode
  jList = []
  rList = []

  totalLinesCleared = 0
  col = 0
  rot = 0
  avgs = []
  s = []
  a = []
  allQ = []

  with tf.Session() as sess:
    sess.run(init)
    for i in range(nGames):
      print(i)
      board.reset()
      tetromino = util.randChoice(tetrominos)

      while(True):
        # Moves come in the format [columnIndex, rotationIndex]
        possibleMoves = tetromino.getPossibleMoves(board)

        # Game over condition
        if len(possibleMoves) == 0:
          break

        if np.random.rand(1) < epsilon:
          a = util.randChoice(possibleMoves)
        else:
          boolMoves = [(x in possibleMoves) for x in range(actionsDim)]
          s = util.networkState(board.board, tetromino.paddedRotations[0])
          a,allQ = sess.run([predict,Qout],feed_dict={inputs1:s, p:[boolMoves]})
          a = a[0]

        rot, col = divmod(a, board.ncols)

        # Perform action and collect reward
        r = board.act(tetromino, col, rot)

        # Random Tetromino for next state
        nextTetromino = util.randChoice(tetrominos)
        s1 = util.networkState(board.board, nextTetromino.paddedRotations[0])

        Q1 = sess.run(Qout,feed_dict={inputs1:s1})
        #Obtain maxQ' and set our target value for chosen action.
        maxQ1 = np.max(Q1)
        targetQ = allQ
        targetQ[0,a] = r + alpha*maxQ1

        #Train our network using target and predicted Q values
        _,W1 = sess.run([updateModel,W],feed_dict={inputs1:s,nextQ:targetQ})

        tetromino = nextTetromino

      totalLinesCleared += board.linesCleared

      if (i+1)%10 == 0:
        avgs.append(totalLinesCleared/(i+1))

    # print("Lines cleared: ", board.linesCleared)
  avg = totalLinesCleared/nGames
  avgs.append(avg)
  # print("Average lines cleared:", avg)
  return avgs
