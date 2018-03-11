import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import util
import operator

from tetrominos import Tetromino, createTetrominos
from board import Board
from graph import Graph

gamma = 0.99

try:
    xrange = xrange
except:
    xrange = range

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=s_size,dtype=tf.float32)
        conv1 = tf.layers.conv2d(inputs=self.state_in, filters=24, kernel_size=[1, 1])
        conv2 = tf.layers.conv2d(inputs=conv1, filters=24, kernel_size=[3, 3])
        # conv3 = tf.layers.conv2d(inputs=conv2, filters=8, kernel_size=[1, 1])
        flatten_layer = tf.contrib.layers.flatten(conv2)
        dense_connected_layer = tf.contrib.layers.fully_connected(flatten_layer, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)
        self.output = tf.contrib.layers.fully_connected(dense_connected_layer, a_size, activation_fn=tf.nn.softmax, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)


        # hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
        # hidden2 = slim.fully_connected(hidden,32,biases_initializer=None,activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
        # self.output = slim.fully_connected(hidden2,a_size,activation_fn=tf.nn.softmax,biases_initializer=None, weights_initializer=tf.contrib.layers.xavier_initializer())

        self.p = tf.placeholder(tf.bool, [1,a_size])
        self.invalid_moves = tf.constant(0., shape=[1,a_size])
        self.valid_moves = tf.where(self.p, self.output, self.invalid_moves)  # Replace invalid moves in Qout by 0.
        self.chosen_action = tf.argmax(self.valid_moves,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss,tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

def learn(nrows, ncols, maxPerEpisode, batchSize, nGames):
    # Tetris initialisations
    tetrominos = createTetrominos()
    board = Board(nrows, ncols)
    board.reset()
    avgs = []

    # tShapeRows, tShapeCols = tuple(map(operator.add, tetrominos[0].shape.shape, (1, 1)))
    tShape = tetrominos[0].paddedRotations[0]
    tShapeRows, tShapeCols = tShape.shape[0], tShape.shape[1]
    inputLayerDim = [
      None,
      board.nrows + tShapeRows,
      board.ncols,
      1
    ]
    # (board.nrows * board.ncols) + (tShapeRows * tShapeCols)
    actionsDim = board.ncols * 4

    tf.reset_default_graph() #Clear the Tensorflow graph.

    myAgent = agent(lr=1e-2,s_size=inputLayerDim,a_size=actionsDim,h_size=32) #Load the agent.

    total_episodes = nGames #Set total number of episodes to train agent on.
    max_ep = maxPerEpisode
    update_frequency = batchSize

    init = tf.global_variables_initializer()

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        total_reward = []
        total_length = []

        gradBuffer = sess.run(tf.trainable_variables())
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        while i < total_episodes:
            board.reset()
            tetromino = util.randChoice(tetrominos)
            s = util.cnnState(board, tetromino.paddedRotations[0])
            running_reward = 0
            ep_history = []

            for j in range(max_ep):
                if j == max_ep - 1:
                    print("reached maximum at episode ", i, " with ", running_reward)
                if i % 500 == 0:
                  board.printBoard()
                possibleMoves = tetromino.getPossibleMoves(board)
                d = (len(possibleMoves) == 0)

                if d == True:
                    #Update the network.
                    ep_history = np.array(ep_history)
                    ep_history[:,2] = discount_rewards(ep_history[:,2])
                    feed_dict={myAgent.reward_holder:ep_history[:,2],
                    myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
                    grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                    for idx,grad in enumerate(grads):
                        gradBuffer[idx] += grad


                    # print(i, i % update_frequency)
                    if i % update_frequency == 0 and i != 0:
                        # print("Updating network at episode ", i)
                        feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                        _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                        for ix,grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad * 0

                    total_reward.append(running_reward)
                    total_length.append(j)
                    break

                bool_moves = [(x in possibleMoves) for x in range(actionsDim)]

                # Probabilistically pick an action given our network outputs.
                o, a_dist = sess.run([myAgent.output, myAgent.valid_moves],feed_dict={myAgent.state_in:s, myAgent.p: [bool_moves]})
                softmax_a_dist = [a_dist[0]/sum(a_dist[0])]

                # print(o)
                # print(a_dist)


                # print()
                a = np.random.choice(softmax_a_dist[0],p=softmax_a_dist[0])
                a = np.argmax(softmax_a_dist == a)
                if i % 500 == 0:
                    tetromino.printShape(0)
                    # print(softmax_a_dist)
                    # print(a)

                rot, col = divmod(a, board.ncols)
                r = board.act(tetromino, col, rot)

                # Random Tetromino for next state
                nextTetromino = util.randChoice(tetrominos)
                s1 = util.cnnState(board, nextTetromino.paddedRotations[0])

                ep_history.append([s,a,r,s1])
                s = s1
                tetromino = nextTetromino

                running_reward += r

                #Update our running tally of scores.
            if i % 100 == 0:
                current_avg = np.mean(total_reward[-100:])
                print(i, ' : ', current_avg)
                avgs.append(current_avg)
            i += 1
    return avgs
