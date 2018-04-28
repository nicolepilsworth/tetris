import pprint
import numpy as np
import random
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
from results import getResults

def playByPolicy(Q, maxPerEpisode):
  tetrominos = createTetrominos()
  board = Board(5, 3)
  board.printBoard()
  totalLinesCleared = 0
  col = 0
  rot = 0

  for j in range(maxPerEpisode):
    tetromino = util.randChoice(tetrominos)

    # Moves come in the format [columnIndex, rotationIndex]
    possibleMoves = tetromino.getPossibleMoves(board)

    # Game over condition
    if len(possibleMoves) == 0:
        print("GAME OVER")
        print("Lines cleared: ", board.linesCleared)
        return

    s = util.strState(board.board, tetromino.shape)
    # Check if Q(s, :) exists, use policy if it does
    if s in Q:
      [col, rot] = util.epsilonGreedy(Q[s], -1, possibleMoves)
    else:
      [col, rot] = util.randChoice(possibleMoves)

    tetromino.printShape(rot)

    # Perform action and collect reward
    r = board.act(tetromino, col, rot)
    board.printBoard()

  print("Maximum number of moves reached: ", maxPerEpisode)
  print("Lines cleared: ", board.linesCleared)


def randVsQ(epsilon, gamma, alpha):
  nGames = 10000
  tSteps = [100*i for i in range(1, int(nGames/100 + 1))]
  randAvgs = []
  # randAvgs = qTableLearn(epsilon, gamma, alpha, nGames, True, True)
  qAvgs = pgLearn(5, 4)
  graph = Graph(tSteps, randAvgs, qAvgs, "Policy Gradient", epsilon, gamma, alpha, nGames)
  graph.plotRandVsQ()

def learn(nGames, nRows, nCols, maxPerEpisode, batchSize):
    tSteps = [100*i for i in range(1, int(nGames/100 + 1))]
    avgs = pgLearn(nRows, nCols, maxPerEpisode, batchSize, nGames)
    graph = PgGraph(tSteps, avgs, batchSize, maxPerEpisode, nGames)
    graph.plot()


def main():

  getResults()
  return
  # Choose from "qTable", "qNetwork", "cnn", "policyGradient"
  learnType = "a3c"

  # Q-learning variables
  epsilon = 0.08 # For epsilon-greedy action choice
  gamma = 0.7 # Discount factor
  alpha = 0.5 # Value fnction learning rate
  rand = False # Whether to choose actions randomly of use Q-learning

  # Policy gradient variables
  batchSize = 10
  saveFreq = 50

  # Universal variables
  nGames = 1000
  tSteps = [100*i for i in range(1, int(nGames/100 + 1))]
  nRows = 5
  nCols = 4
  maxPerEpisode = 1000
  boardSize = str(nRows) + " rows * " + str(nCols) + " cols"

  # compareGraph = CompareGraph(tSteps)
  # compareGraph.plot()
  # return

  # Specific learn function per learn type
  funcs = {
    "qTable": qTableLearn,
    "qNetwork": qNetworkLearn,
    "cnn": cnnLearn,
    "policyGradient": pgLearn,
    "a3c": a3cTrain
  }

  # thresholds = {
  #   "qTable": 0,
  #   "qNetwork": 0,
  #   "cnn": 40,
  #   "policyGradient": 100
  # }

  # Arguments to pass into learn function
  args = {
    "qTable": (epsilon, gamma, alpha, nGames, False, True, nRows, nCols),
    "qNetwork": (epsilon, gamma, alpha, nGames, True, nRows, nCols),
    "cnn": (epsilon, gamma, alpha, nGames, nRows, nCols),
    "policyGradient": (nRows, nCols, maxPerEpisode, batchSize, nGames, alpha),
    "a3c": (nRows, nCols, maxPerEpisode, saveFreq)
  }

  avgs = funcs[learnType](*args[learnType])

  # # Repeat if threshold not reached
  # while avgs[-1] < thresholds[learnType]:
  #   print("threshold not reached")
  #   avgs = funcs[learnType](*args[learnType])



  if learnType == "policyGradient":
    graph = PgGraph(tSteps, avgs, batchSize, maxPerEpisode, nGames, boardSize)
    graph.plot()
  # else:
  #   graph = Graph(tSteps, avgs, learnType, epsilon, gamma, alpha, nGames, boardSize)
  #   graph.plot()

  # randVsQ(epsilon, gamma, alpha)
  # learn(5000, 5, 4, 1000, 5)
  # Q = cnnLearn(epsilon, gamma, alpha, nGames, False)
  # Q = qTableLearn(epsilon, gamma, alpha, nGames, rand, False)
  # playByPolicy({})

if __name__ == "__main__":
  main()
