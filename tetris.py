import pprint
import numpy as np
import random
from tetrominos import Tetromino, createTetrominos
from board import Board

# For this implementation, concatenate board config and
# Tetromino config into a string for the state
def state(board, tetromino):
    bString = ''.join(''.join('%d' %x for x in row) for row in board)
    tString = ''.join(''.join('%d' %x for x in row) for row in tetromino)
    return bString + tString

def epsilonGreedy(q, epsilon, possMoves):
    if random.random() < epsilon:
        return possMoves[random.randint(0, len(possMoves) - 1)]
    else:
        qPossMoves = []
        for p in possMoves:
            qPossMoves.append(q[p[0]][p[1]])
        return possMoves[np.argmax(qPossMoves)]

def play(epsilon, gamma, alpha, nGames, rand):
  Q = {}
  tetrominos = createTetrominos()
  board = Board(5, 3)
  # board.printBoard()
  totalLinesCleared = 0
  col = 0
  rot = 0

  for i in range(nGames):
    board.reset()
    tetromino = tetrominos[random.randint(0, 1)]

    while(True):
    #   tetromino.printShape()

      # Moves come in the format [columnIndex, rotationIndex]
      possibleMoves = tetromino.getPossibleMoves(board)

      # Game over condition
      if len(possibleMoves) == 0:
        #   print("GAME OVER")
          break

      if rand:
        [col, rot] = possibleMoves[random.randint(0, len(possibleMoves) - 1)]
      else:
        s = state(board.board, tetromino.shape)
        # Check if Q(s, :) exists, create if not
        if s not in Q:
          Q[s] = np.zeros((board.ncols, len(tetromino.rotations)))
        [col, rot] = epsilonGreedy(Q[s], epsilon, possibleMoves)

      # Perform action and collect reward
      r = board.act(tetromino, col, rot)
      # board.printBoard()

      # Random Tetromino for next state
      nextTetromino = tetrominos[random.randint(0, len(tetrominos) - 1)]

      if not rand:
          s1 = state(board.board, nextTetromino.shape)

          # Check if Q(s1, :) exists, create if not
          if s1 not in Q:
            Q[s1] = np.zeros((board.ncols, len(nextTetromino.rotations)))

          # TODO: change to select optimal action under Q-learning policy
          Q[s][col][rot] = Q[s][col][rot] + alpha*(r + gamma*np.amax(Q[s1]) - Q[s][col][rot])

      tetromino = nextTetromino

    totalLinesCleared += board.linesCleared

    # print("Lines cleared: ", board.linesCleared)
  print("Average lines cleared:", totalLinesCleared/nGames)
  print(Q)

def main():
  epsilon = 0.2 # For epsilon-greedy action choice
  gamma = 0.9 # Discount factor
  alpha = 0.01 # Value function learning rate
  nGames = 1000
  rand = False # Whether to choose actions randomly of use Q-learning

  play(epsilon, gamma, alpha, nGames, rand)

if __name__ == "__main__":
  main()
