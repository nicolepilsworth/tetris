# https://github.com/AbhishekAshokDubey/RL/blob/master/ping-pong/tf_ping_pong_policyGradient.py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import util
import operator

from tetrominos import Tetromino, createTetrominos
from board import Board
from graph import Graph

H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = False

# discount_rewards(np.array([1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0]))
# returns: array([ 1., 0.96059601, 0.970299, 0.9801, 0.99, 1., 0.9801, 0.99, 1.])
def discount_rewards(r, gamma):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def onehot(dim, idx):
  return [1 if x == idx else 0 for x in range(dim)]

def learn(epsilon, gamma, alpha, nGames, getAvgs):
  tetrominos = createTetrominos()
  board = Board(5, 3)
  board.reset()
  tShapeRows, tShapeCols = tuple(map(operator.add, tetrominos[0].shape.shape, (1, 1)))
  inputLayerDim = (board.nrows * board.ncols) + (tShapeRows * tShapeCols)
  actionsDim = board.ncols * 4

  input_layer = tf.placeholder(shape=[None,inputLayerDim],dtype=tf.float32)
  hidden_layer = slim.fully_connected(input_layer, H, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)
  output_layer = slim.fully_connected(hidden_layer, actionsDim, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer() ,biases_initializer=None)

  actions = tf.placeholder(shape=[None],dtype=tf.int32)
  rewards = tf.placeholder(shape=[None,1],dtype=tf.float32)

  actions_onehot = tf.one_hot(actions, actionsDim)
  responsible_outputs = tf.reduce_sum(output_layer * actions_onehot, [1])

  loss = -tf.reduce_mean(tf.log(responsible_outputs) * rewards)

  p = tf.placeholder(tf.bool, [1,actionsDim])
  # Qout = tf.matmul(inputs1,W)
  invalidMoves = tf.constant(0., shape=[1,actionsDim])
  validMoves = tf.where(p, output_layer, invalidMoves)  # Replace invalid moves in Qout by -100
  predict = tf.argmax(validMoves,1)

  w_variables = tf.trainable_variables()
  gradients = []
  for indx,w in enumerate(w_variables):
    w_holder_var = tf.placeholder(tf.float32,name="w_"+ str(indx))
    gradients.append(w_holder_var)

  all_gradients = tf.gradients(loss, tf.trainable_variables())
  optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
  apply_grads = optimizer.apply_gradients(zip(gradients,w_variables))


  totalLinesCleared = 0
  col = 0
  rot = 0
  avgs, s, a, allQ, h = [], [], [], [], []
  prev_x = None # used in computing the difference frame
  xs,hs,dlogps,drs,all_game_scores = [],[],[],[],[]
  running_reward = None
  reward_sum = 0
  episode_number = 0

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    grad_buffer = sess.run(tf.trainable_variables())
    for indx,grad in enumerate(grad_buffer):
      grad_buffer[indx] = grad * 0

    while episode_number < 100:
      tetromino = util.randChoice(tetrominos)
    #   tetromino.printShape(0)
      possibleMoves = tetromino.getPossibleMoves(board)

      # Game over condition
      if len(possibleMoves) > 0:
        boolMoves = [(x in possibleMoves) for x in range(actionsDim)]

        cur_x = util.pgState(board.board, tetromino.paddedRotations[0])
        x = cur_x - prev_x if prev_x is not None else np.zeros((8, 3))
        prev_x = cur_x

        a, o, h = sess.run([predict, validMoves, hidden_layer], feed_dict={input_layer: np.reshape(x,(1,inputLayerDim)), p:[boolMoves]})
        a = np.random.choice(actionsDim, 1, p=o[0]/sum(o[0]))[0]
        # print(a)

        rot, col = divmod(a, board.ncols)
        xs.append(np.reshape(x, (1, inputLayerDim))) # observation
        hs.append(h)
        dlogps.append(a)
        # dlogps.append(onehot(actionsDim, a))

        # Perform action and collect reward
        r = board.act(tetromino, col, rot)
        # board.printBoard()
        reward_sum += r
        drs.append(r) # record reward (has to be done after we call step() to get reward for previous action)

        # Random Tetromino for next state
        nextTetromino = util.randChoice(tetrominos)
        s1 = util.pgState(board.board, nextTetromino.paddedRotations[0])
      else:
        episode_number += 1
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory
        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr, gamma)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr = discounted_epr - np.mean(discounted_epr)
        discounted_epr = discounted_epr / np.std(discounted_epr)

        grads = sess.run(all_gradients, feed_dict = {input_layer: epx, rewards:discounted_epr, actions: epdlogp.ravel()})
        for indx,grad in enumerate(grads):
          grad_buffer[indx] += grad

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
          print("updating weights of the network")
          feed_dict = dict(zip(gradients, grad_buffer))
          x = sess.run(apply_grads, feed_dict=feed_dict)
          print('HERE')
          print(x)
          for indx,grad in enumerate(grad_buffer):
            grad_buffer[indx] = grad * 0    # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward %f. running mean: %f' % (reward_sum, running_reward))
        #        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        all_game_scores.append(reward_sum)
        avgs.append(running_reward)
        reward_sum = 0
        board.reset() # reset env
        prev_x = None
        # print(all_game_scores)
  return avgs
