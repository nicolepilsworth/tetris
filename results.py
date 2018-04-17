import pprint
import numpy as np
import random
import gevent.monkey

from tetrominos import Tetromino, createTetrominos
from board import Board
from graph import Graph
from pgGraph import PgGraph
from compareGraph import CompareGraph
import util
from qTable import learn as qTableLearn
from qNetwork import learn as qNetworkLearn
from cnn import learn as cnnLearn
from policyGradient2 import learn as pgLearn
from a3c import train as a3cTrain

def run50QTable(nRows, nCols):
    gevent.monkey.patch_all()

    # Choose from "qTable", "qNetwork", "cnn", "policyGradient", "a3c"
    learnType = "qTable"

    # Q-learning variables
    epsilon = 0.1 # For epsilon-greedy action choice
    gamma = 0.7 # Discount factor
    alpha = 0.3 # Value fnction learning rate
    rand = False # Whether to choose actions randomly of use Q-learning

    # Policy gradient variables
    batchSize = 10
    saveFreq = 50

    # Universal variables
    nGames = 600
    tSteps = [10*i for i in range(1, int(nGames/10 + 1))]
    # nRows = 16
    # nCols = 10
    maxPerEpisode = 1000
    boardSize = str(nRows) + " rows * " + str(nCols) + " cols"

    # Specific learn function per learn type
    funcs = {
      "qTable": qTableLearn,
      "qNetwork": qNetworkLearn,
      "cnn": cnnLearn,
      "policyGradient": pgLearn,
      "a3c": a3cTrain
    }

    # Arguments to pass into learn function
    args = {
      "qTable": (epsilon, gamma, alpha, nGames, False, True, nRows, nCols),
      "qNetwork": (epsilon, gamma, alpha, nGames, True),
      "cnn": (epsilon, gamma, alpha, nGames, nRows, nCols),
      "policyGradient": (nRows, nCols, maxPerEpisode, batchSize, nGames),
      "a3c": (nRows, nCols, maxPerEpisode, saveFreq)
    }

    allAvgs = []

    threads = [gevent.spawn(funcs[learnType], *args[learnType]) for i in range(20)]
    avgs = gevent.joinall(threads)
    print(np.mean([thread.value for thread in threads], axis=0))

    # for i in range(5):
    #     avgs = np.array(funcs[learnType](*args[learnType]))
    #     allAvgs.append(avgs)
    # print(np.mean(allAvgs, axis=0))
