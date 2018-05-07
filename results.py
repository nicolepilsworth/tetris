import pprint
import numpy as np
import random
import plotly.plotly as py
from plotly.graph_objs import *
import tensorflow as tf
import tensorflow.contrib.slim as slim

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
from rawData import RawData
from algCompare import AlgCompare

def getResults():

    # Choose from "qTable", "qNetwork", "policyGradient", "a3c"
    learnType = "a3c"

    # Q-learning variables
    epsilon = 0.5 # For epsilon-greedy action choice
    gamma = 0.9 # Discount factor
    alpha = 0.3 # Value fnction learning rate
    rand = False # Whether to choose actions randomly of use Q-learning
    lr = 0.0001

    nRows = 8
    nCols = 6
    batchSize = 10
    nGames = 5000
    nHLayers = 2
    # variables = [1,10,50]
    # variables = [0.01, 0.1, 0.5]
    # variables = [0.01]
    # variables = [0.0001, 0.01, 0.1]
    variables = [2, 4, 6]
    graph_filename = "alg-compare-54"
    maxPerEpisode = 200
    l = float("inf")

    boardSize = str(nRows) + " rows * " + str(nCols) + " cols"

    # Specific learn function per learn type
    funcs = {
      "qTable": qTableLearn,
      "qNetwork": qNetworkLearn,
      "cnn": cnnLearn,
      "policyGradient": pgLearn,
      "a3c": a3cTrain
    }

    allAvgs = []



    colours = ["0,76,153", "178,102,255", "0,153,76", "204,0,0"]

    graph_lines = []
    interval = 40
    t_steps = np.arange(0, nGames + interval, interval)
    t_steps_rev = t_steps[::-1]
    graph_lines = []
    x_title = "Number of episodes"
    y_title = "Average score"
    agents = 1
    algCompare = AlgCompare().data
    # allData = RawData().collatedData
    a3cData = {}
    allData = {}


    # graph = Graph(t_steps, algCompare, x_title, y_title, graph_filename)
    # graph.plot()
    # return

    for idx, x in enumerate(variables):
        allAvgs = []
        # QTABLE:
        # algArgs = (x, gamma, alpha, nGames, False, True, nRows, nCols)
        # QNETWORK:
        # algArgs = (x, gamma, alpha, nGames, True, nRows, nCols)
        # POLICYGRADIENT:
        # algArgs = (nRows, nCols, maxPerEpisode, x, nGames, alpha)
        # A3C:
        algArgs = (nRows, nCols, maxPerEpisode, interval, nGames, lr, nHLayers, x)
        # for i in range(agents):
        #     # print(idx, i)
        #     try:
        avgs = funcs[learnType](*algArgs)
        #     except:
        #         print("error")
        #         continue
        #     #
        #     if learnType == "a3c":
        allAvgs = allAvgs + avgs
        #     else:
        # allAvgs.append(avgs)
        allAvgs = np.concatenate(tuple(list(map(lambda v: v[str(x)][0], allData))), axis=0)
        allData[str(x)] = allAvgs
        if learnType == "a3c":
            l = min(min(map(len, allAvgs)), l)
        else:
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
                    "name":'epsilon = ' + str(x)
                },
                {   "x":t_steps,
                    "y":mean,
                    "line":Line(color="rgb({})".format(colours[idx])),
                    "mode":'lines',
                    "name":'epsilon = ' + str(x)
                }
            ))
    # if learnType == "a3c":
    #     print("L:", l)
    #     t_steps = t_steps[:l]
    #     t_steps_rev = t_steps_rev[-l:]
    #     for idx, x in enumerate(variables):
    #         allAvgs = allData[str(x)]
    #         # allAvgs = np.concatenate(tuple(list(map(lambda v: v[str(x)], allData))), axis=0)
    #         allAvgs = list(map(lambda x: x[:l], allAvgs))
    #
    #         mean = np.mean(allAvgs, axis=0)
    #         std_dev = np.std(allAvgs, axis=0)
    #         y_upper = np.add(mean, std_dev)
    #         y_lower = np.subtract(mean, std_dev)
    #         y_lower = y_lower[::-1]
    #
    #         graph_lines.extend(({
    #                 "x": np.concatenate([t_steps, t_steps_rev]),
    #                 "y": np.concatenate([y_upper, y_lower]),
    #                 "fill":'tozerox',
    #                 "fillcolor":'rgba({},0.2)'.format(colours[idx]),
    #                 "line":Line(color='transparent'),
    #                 "showlegend":False,
    #                 "name":'#layers = ' + str(x)
    #             },
    #             {   "x":t_steps,
    #                 "y":mean,
    #                 "line":Line(color="rgb({})".format(colours[idx])),
    #                 "mode":'lines',
    #                 "name":'#layers = ' + str(x)
    #             }
    #         ))


    print(allData)
    # print(graph_lines)
    # graph = Graph(t_steps, graph_lines, x_title, y_title, graph_filename)
    # graph.plot()
