import random
import numpy as np

# Given a list, return a random element from the list
def randChoice(l):
    return l[random.randint(0, len(l) - 1)]

# For this implementation, concatenate board config and
# Tetromino config into a string for the state
def strState(board, tetromino):
    bString = ''.join(''.join('%d' %x for x in row) for row in board)
    tString = ''.join(''.join('%d' %x for x in row) for row in tetromino)
    return bString + tString

def pgState():
    return np.append(tetromino.flatten(), b.board.flatten())

def networkState(board, tetromino):
    return np.array([np.append(board.flatten(), tetromino.flatten())])

def cnnState(b, tetromino):
    return np.reshape(
      np.concatenate(
        (
          np.pad(
            tetromino,
            (
              (0, 0),
              (0, b.ncols - tetromino.shape[1])
            ),
            "constant",
            constant_values=(False,)
          ),
          b.board
        ),
        axis=0
      ),
      (1, b.nrows + tetromino.shape[0], b.ncols, 1)
    )

def pgState(b, tetromino):
    return np.append(tetromino.flatten(), b.board.flatten())

def a3cState(b, tetromino):
  boardInput = b.board[0:4] if b.yMax < 4 else b.board[b.yMax - 4:b.yMax]
  return np.reshape(
    np.concatenate(
      (
        np.pad(
          tetromino,
          (
            (0, 0),
            (0, b.ncols - tetromino.shape[1])
          ),
          "constant",
          constant_values=(False,)
        ),
        boardInput
      ),
      axis=0
    ),
    (1, 4 + tetromino.shape[0], b.ncols, 1)
    )

def a3cBoardState(b):
    boardInput = b.board[0:4] if b.yMax < 4 else b.board[b.yMax - 4:b.yMax]
    return np.reshape(boardInput, (1, 4, b.ncols, 1))

def epsilonGreedy(q, epsilon, possMoves, nCols):
    if random.random() < epsilon:
        return randChoice(possMoves)
    else:
        qPossMoves = []
        for p in possMoves:
            [rot, col] = divmod(p, nCols)
            qPossMoves.append(q[col][rot])
        highest = np.argwhere(qPossMoves == np.amax(qPossMoves))
        return possMoves[randChoice(highest.flatten().tolist())]

def bestMove(q, possMoves):
    highest = np.argwhere(q == np.amax(q))
    moves = []
    try:
        moves = possMoves[randChoice(highest.flatten().tolist())]
    except:
        print(q)
        print(possMoves)
        print(highest)
    return moves
