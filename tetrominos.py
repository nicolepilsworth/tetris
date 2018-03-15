import pandas as pd
import numpy as np

def createTetrominos():
    shapes = [np.array(x) for x in [
    # [[True, True],
    # [True, False]],
    #
    # [[True, True, True]]

      [[True, True],
       [True, False],
       [True, False]],

       [[True], [True], [True], [True]],

    #    [[False, True, False],
    #    [True, True, True]]
    #
    #    [[True, True],
    #    [False, True],
    #    [False, True]]

    #    [[True, True, False],
    #    [False, True, True]],
       #
    #    [[False, True, True],
    #    [True, True, False]]
       #
       [[True, True],
       [True, True]]
    ]]
    # TODO: Add 'np.fulls'
    rotations = [setRotations(s) for s in shapes]
    pad = (np.max([np.max([r.shape[0] for r in x]) for x in rotations]), np.max([np.max([r.shape[1] for r in x]) for x in rotations]))
    return [Tetromino(s, rotations[i], padRotations(rotations[i], pad), *setHeights(rotations[i])) for i, s in enumerate(shapes)]

def setRotations(shape):
  rotations = [shape]
  currentRot = shape
  rotHeight = 0

  # 4 is max no. of rotations (first one already stored)
  for i in range(3):
    newRot = np.rot90(currentRot)

    # Check if rotation is already in array - if it is, no more unique rotations
    if next((True for rot in rotations if np.array_equal(newRot, rot)), False):
      return rotations

    rotations.append(newRot)
    currentRot = newRot

  return rotations

def padRotations(shapes, pad):
  padRows, padCols = pad
  return [np.pad(x, ((0, padRows - x.shape[0]), (0, padCols - x.shape[1])), "constant", constant_values=(False,)) for x in shapes]


# Return array containing column heights of every rotation
def setHeights(shapes):
  return [height(s, "rHeight") for s in shapes], [height(s, "highest") for s in shapes]

# Get column height of either board column or piece column
def height(tShape, hType):
  heights = []
  nRows, nCols = tShape.shape

  # Use idxmax on reversed shape to find last instance of True value in column
  heights = {
    "rHeight": np.subtract(np.full((1, nCols), nRows), np.argmax(tShape, axis=0)).flatten(),
    "highest": np.subtract(np.full((1, nCols), nRows), np.argmax(np.flip(tShape, 0), axis=0)).flatten()
  }[hType]

  return heights

class Tetromino:
  def __init__(self, shape, rotations, padRots, rHeights, highest):
    self.shape = shape
    self.rotations = rotations
    self.paddedRotations = padRots
    self.rHeights = rHeights
    self.highest = highest
    # TODO: self.fulls = fulls

  def printShape(self, rot):
    print(''.join(map(lambda x: ' '.join(map(str, x)) + "\n", self.rotations[rot][::-1].astype(int))))
    print("\n")
    return

  # TODO: store possible moves historically by
  # Iterate through every rotation and column to get all possible actions
  def getPossibleMoves(self, board):
    moves = []
    for idx, rot in enumerate(self.rotations):
      for col in range(board.board.shape[1] - rot.shape[1] + 1):
        if board.tetrominoFitsInCol(col, rot, self.rHeights[idx]):
        #   moves.append([col, idx])
          moves.append((idx * board.ncols) + col)

    return moves
