import pandas as pd
import numpy as np

def createTetrominos():
    shapes = [np.array(x) for x in [
      [[True, True],
       [True, False]],

       [[True], [True], [True]]
    ]]
    rotations = [setRotations(s) for s in shapes]
    return [Tetromino(s, rotations[i], *setHeights(rotations[i])) for i, s in enumerate(shapes)]

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
  def __init__(self, shape, rotations, rHeights, highest):
    self.shape = shape
    self.rotations = rotations
    self.rHeights = rHeights
    self.highest = highest

  def printShape(self, rot):
    print(''.join(map(lambda x: ' '.join(map(str, x)) + "\n", self.rotations[rot][::-1].astype(int))))
    print("\n")
    return

  # Iterate through every rotation and column to get all possible actions
  def getPossibleMoves(self, board):
    moves = []
    for idx, rot in enumerate(self.rotations):
      for col in range(board.board.shape[1] - rot.shape[1] + 1):
        if board.tetrominoFitsInCol(col, rot, self.rHeights[idx]):
          moves.append([col, idx])

    return moves
