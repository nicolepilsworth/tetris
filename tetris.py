import pprint
import numpy as np
import random
from tetrominos import Tetromino, createTetrominos
from board import Board
from graph import Graph

# For this implementation, concatenate board config and
# Tetromino config into a string for the state
def state(board, tetromino):
    bString = ''.join(''.join('%d' %x for x in row) for row in board)
    tString = ''.join(''.join('%d' %x for x in row) for row in tetromino)
    return bString + tString

# Given a list, return a random element from the list
def randChoice(l):
    return l[random.randint(0, len(l) - 1)]

def epsilonGreedy(q, epsilon, possMoves):
    if random.random() < epsilon:
        return randChoice(possMoves)
    else:
        qPossMoves = []
        for p in possMoves:
            qPossMoves.append(q[p[0]][p[1]])
        return possMoves[np.argmax(qPossMoves)]

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
    tetromino = randChoice(tetrominos)

    while(True):
      # Moves come in the format [columnIndex, rotationIndex]
      possibleMoves = tetromino.getPossibleMoves(board)

      # Game over condition
      if len(possibleMoves) == 0:
          break

      if isRand:
        [col, rot] = randChoice(possibleMoves)
      else:
        s = state(board.board, tetromino.shape)

        # Check if Q(s, :) exists, create if not
        if s not in Q:
          Q[s] = np.zeros((board.ncols, len(tetromino.rotations)))
          [col, rot] = randChoice(possibleMoves)
        else:
          [col, rot] = epsilonGreedy(Q[s], epsilon, possibleMoves)

      # Perform action and collect reward
      r = board.act(tetromino, col, rot)

      # Random Tetromino for next state
      nextTetromino = randChoice(tetrominos)

      if not isRand:
          s1 = state(board.board, nextTetromino.shape)

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

def playByPolicy(Q):
  tetrominos = createTetrominos()
  board = Board(5, 3)
  board.printBoard()
  totalLinesCleared = 0
  col = 0
  rot = 0

  while(True):
    tetromino = randChoice(tetrominos)
    # Moves come in the format [columnIndex, rotationIndex]
    possibleMoves = tetromino.getPossibleMoves(board)

    # Game over condition
    if len(possibleMoves) == 0:
        print("GAME OVER")
        break

    s = state(board.board, tetromino.shape)
    # Check if Q(s, :) exists, use policy if it does
    if s in Q:
      [col, rot] = epsilonGreedy(Q[s], -1, possibleMoves)
    else:
      [col, rot] = randChoice(possibleMoves)

    tetromino.printShape(rot)

    # Perform action and collect reward
    r = board.act(tetromino, col, rot)
    board.printBoard()

  print("Lines cleared: ", board.linesCleared)

def randVsQ(epsilon, gamma, alpha):
  tSteps = [10*i for i in range(1, 1001)]

  randAvgs = learn(epsilon, gamma, alpha, 10000, True, True)
  qAvgs = learn(epsilon, gamma, alpha, 10000, False, True)
  graph = Graph(tSteps, randAvgs, qAvgs)
  graph.plot()

def main():
  epsilon = 0.3 # For epsilon-greedy action choice
  gamma = 0.7 # Discount factor
  alpha = 0.07 # Value function learning rate
  nGames = 10000
  rand = False # Whether to choose actions randomly of use Q-learning

  randVsQ(epsilon, gamma, alpha)
  # Q = learn(epsilon, gamma, alpha, nGames, rand, False)
  # playByPolicy({})

if __name__ == "__main__":
  main()
