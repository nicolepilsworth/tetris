import pprint
import numpy as np
import random
import plotly.plotly as py
from plotly.graph_objs import *

from tetrominos import Tetromino, createTetrominos
from board import Board
from resultsGraph import Graph
from pgGraph import PgGraph
from compareGraph import CompareGraph
import util
from qTable import learn as qTableLearn
from qNetwork import learn as qNetworkLearn
from cnn import learn as cnnLearn
from policyGradient2 import learn as pgLearn
from a3c import train as a3cTrain

def run50QTable(nRows, nCols):

    # Choose from "qTable", "qNetwork", "cnn", "policyGradient", "a3c"
    learnType = "qTable"

    # Q-learning variables
    epsilon = 0.1 # For epsilon-greedy action choice
    gamma = 0.8 # Discount factor
    alpha = 0.05 # Value fnction learning rate
    rand = False # Whether to choose actions randomly of use Q-learning

    # Policy gradient variables
    batchSize = 10
    saveFreq = 50

    # Universal variables
    nGames = 500
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

    # threads = [gevent.spawn(funcs[learnType], *args[learnType]) for i in range(20)]
    # avgs = gevent.joinall(threads)
    # print(np.mean([thread.value for thread in threads], axis=0))

    epsilons = [0.01, 0.1, 0.3, 0.5]
    colours = ["0,76,153", "178,102,255", "0,153,76", "204,0,0"]
    # epsilons = [0.1]
    graph_lines = []
    t_steps = np.arange(0, nGames + 20, 20)
    t_steps_rev = t_steps[::-1]
    graph_lines = []
    x_title = "Number of episodes"
    y_title = "Average score"
    agents = 20

    for idx, e in enumerate(epsilons):
        allAvgs = []
        # QTABLE:
        algArgs = (e, gamma, alpha, nGames, False, True, nRows, nCols)
        # QNETWORK:
        # algArgs = (e, gamma, alpha, nGames, True, nRows, nCols)
        # POLICYGRADIENT:

        for i in range(agents):
            avgs = np.array(funcs[learnType](*algArgs))
            allAvgs.append(avgs)
        mean = np.mean(allAvgs, axis=0)
        std_dev = np.std(allAvgs, axis=0)
        y_upper = np.add(mean, std_dev)
        y_lower = np.subtract(mean, std_dev)
        y_lower = y_lower[::-1]

        graph_lines.extend(({
                "x": np.concatenate([t_steps, t_steps_rev]),
                "y": np.concatenate([y_upper, y_lower]),
                "fill":'tozerox',
                "fillcolor":'rgba({},0.2)'.format(colours[idx]),
                "line":Line(color='transparent'),
                "showlegend":False,
                "name":'epsilon = ' + str(e)
            },
            {   "x":t_steps,
                "y":mean,
                "line":Line(color="rgb({})".format(colours[idx])),
                "mode":'lines',
                "name":'epsilon = ' + str(e)
            }
        ))

    graph = Graph(t_steps, graph_lines, x_title, y_title)
    graph.plot()
