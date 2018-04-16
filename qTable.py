import numpy as np
from tetrominos import Tetromino, createTetrominos
from board import Board
from graph import Graph
import util

def playByPolicy(Q, maxPerEpisode, nRows, nCols):
  tetrominos = createTetrominos()
  board = Board(nRows, nCols)
  totalLinesCleared = 0
  col = 0
  rot = 0

  for j in range(maxPerEpisode):
    tetromino = util.randChoice(tetrominos)

    # Moves come in the format [columnIndex, rotationIndex]
    possibleMoves = tetromino.getPossibleMoves(board)

    # Game over condition
    if len(possibleMoves) == 0:
        return board.linesCleared

    s = util.strState(board.board, tetromino.shape)
    # Check if Q(s, :) exists, use policy if it does
    if s in Q:
      [rot, col] = divmod(util.epsilonGreedy(Q[s], -1, possibleMoves, board.ncols), board.ncols)
    else:
      [rot, col] = divmod(util.randChoice(possibleMoves), board.ncols)

    # Perform action and collect reward
    r = board.act(tetromino, col, rot)

  return board.linesCleared

def learn(epsilon, gamma, alpha, nGames, isRand, getAvgs, nRows, nCols):
  avgs = []
  gameScores = []

  Q = {}
  tetrominos = createTetrominos()
  board = Board(nRows, nCols)
  totalLinesCleared = 0
  col = 0
  rot = 0
  for i in range(nGames):
    board.reset()
    tetromino = util.randChoice(tetrominos)

    while(True):
      # Moves come in the format [columnIndex, rotationIndex]
      possibleMoves = tetromino.getPossibleMoves(board)
    #   tetromino.printShape(0)
    #   board.printBoard()

      # Game over condition
      if len(possibleMoves) == 0:
          break

    #   import pdb; pdb.set_trace()

      if isRand:
        [rot, col] = divmod(util.randChoice(possibleMoves), board.ncols)
      else:
        s = util.strState(board.board, tetromino.shape)

        # Check if Q(s, :) exists, create if not
        if s not in Q:
          Q[s] = np.zeros((board.ncols, len(tetromino.rotations)))
          [rot, col] = divmod(util.randChoice(possibleMoves), board.ncols)

        else:
          [rot, col] = divmod(util.epsilonGreedy(Q[s], epsilon, possibleMoves, board.ncols), board.ncols)

      # Perform action and collect reward
      r = board.act(tetromino, col, rot)

      # Random Tetromino for next state
      nextTetromino = util.randChoice(tetrominos)

      if not isRand:
          s1 = util.strState(board.board, nextTetromino.shape)

          # Check if Q(s1, :) exists, create if not
          if s1 not in Q:
            Q[s1] = np.zeros((board.ncols, len(nextTetromino.rotations)))

          # Q-value function update
          Q[s][col][rot] = Q[s][col][rot] + alpha*(r + gamma*np.amax(Q[s1]) - Q[s][col][rot])

      tetromino = nextTetromino

    # totalLinesCleared += board.linesCleared
    # print(board.linesCleared)

    # Play 10 games every 100 games to measure learning performance
    if (i)%20 == 0 and i != 0:
      print(i)
      for j in range(10):
        gameScores.append(playByPolicy(Q, 200, nRows, nCols))
    #   print(totalLinesCleared/50)
    #   avgs.append(totalLinesCleared/50)
    #   totalLinesCleared = 0
      avgs.append(np.mean(gameScores))
      gameScores = []

    # print("Lines cleared: ", board.linesCleared)
  # print("Average lines cleared:", avg)
  return avgs
