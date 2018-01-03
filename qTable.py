import numpy as np
from tetrominos import Tetromino, createTetrominos
from board import Board
from graph import Graph
import util

def learn(epsilon, gamma, alpha, nGames, isRand, getAvgs):
  Q = {}
  tetrominos = createTetrominos()
  board = Board(5, 3)
  # board.printBoard()
  totalLinesCleared = 0
  col = 0
  rot = 0
  avgs = []
  for i in range(nGames):
    board.reset()
    tetromino = util.randChoice(tetrominos)

    while(True):
      # Moves come in the format [columnIndex, rotationIndex]
      possibleMoves = tetromino.getPossibleMoves(board)

      # Game over condition
      if len(possibleMoves) == 0:
          break

      if isRand:
        [rot, col] = divmod(util.randChoice(possibleMoves), board.ncols)
      else:
        s = util.strState(board.board, tetromino.shape)

        # Check if Q(s, :) exists, create if not
        if s not in Q:
          Q[s] = np.zeros((board.ncols, len(tetromino.rotations)))
          [rot, col] = divmod(util.randChoice(possibleMoves), board.ncols)
        else:
          [rot, col] = divmod(util.epsilonGreedy(Q[s], epsilon, possibleMoves), board.ncols)

      # Perform action and collect reward
      r = board.act(tetromino, col, rot)

      # Random Tetromino for next state
      nextTetromino = util.randChoice(tetrominos)

      if not isRand:
          s1 = util.strState(board.board, nextTetromino.shape)

          # Check if Q(s1, :) exists, create if not
          if s1 not in Q:
            Q[s1] = np.zeros((board.ncols, len(nextTetromino.rotations)))

          # Q-learning value function update
          Q[s][col][rot] = Q[s][col][rot] + alpha*(r + gamma*np.amax(Q[s1]) - Q[s][col][rot])

      tetromino = nextTetromino

    totalLinesCleared += board.linesCleared

    if (i+1)%10 == 0:
      avgs.append(totalLinesCleared/(i+1))

    # print("Lines cleared: ", board.linesCleared)
  avg = totalLinesCleared/nGames
  avgs.append(avg)
  # print("Average lines cleared:", avg)
  if getAvgs:
    return avgs
  else:
    return Q
