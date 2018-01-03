import pprint
import numpy as np
import random
from tetrominos import Tetromino, createTetrominos
from board import Board
from graph import Graph
import util
from qTable import learn as qTableLearn
from qNetwork import learn as qNetworkLearn

def playByPolicy(Q):
  tetrominos = createTetrominos()
  board = Board(5, 3)
  board.printBoard()
  totalLinesCleared = 0
  col = 0
  rot = 0

  while(True):
    tetromino = util.randChoice(tetrominos)
    # Moves come in the format [columnIndex, rotationIndex]
    possibleMoves = tetromino.getPossibleMoves(board)

    # Game over condition
    if len(possibleMoves) == 0:
        print("GAME OVER")
        break

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

  print("Lines cleared: ", board.linesCleared)

def randVsQ(epsilon, gamma, alpha):
  tSteps = [10*i for i in range(1, 501)]

  randAvgs = qTableLearn(epsilon, gamma, alpha, 5000, True, True)
  qAvgs = qNetworkLearn(epsilon, gamma, alpha, 5000, True)
  graph = Graph(tSteps, randAvgs, qAvgs)
  graph.plot()

def main():
  epsilon = 0.08 # For epsilon-greedy action choice
  gamma = 0.7 # Discount factor
  alpha = 0.07 # Value fnction learning rate
  nGames = 100
  rand = False # Whether to choose actions randomly of use Q-learning

  randVsQ(epsilon, gamma, alpha)
  # Q = qNetworkLearn(epsilon, gamma, alpha, nGames, rand, False)
  # playByPolicy({})

if __name__ == "__main__":
  main()
