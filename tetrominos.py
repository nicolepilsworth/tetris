import numpy as np

def createTetrominos():
    shapes = [np.array(x) for x in [
      [[True, True],
       [True, False]],

       [[True], [True], [True]]
    ]]
    rotations = [setRotations(s) for s in shapes]
    return [Tetromino(s, rotations[i], setRotationHeights(rotations[i])) for i, s in enumerate(shapes)]

def setRotations(shape):
    rotations = [shape]
    currentRot = shape
    rotHeight = 0

    # 4 is max no. of rotations (first one already stored)
    for i in range(3):
      newRot = np.rot90(currentRot)

      # Check if rotation is already in array - if it is, no more unique rotations
      for rot in rotations:
        if np.all(newRot == rot) and len(newRot) == len(rot):
          return rotations

      rotations.append(newRot)
      currentRot = newRot

    return rotations

# Return array containing column heights of every rotation
def setRotationHeights(shapes):
  rotHeights = [rotationHeights(s) for s in shapes]

# Get column height of either board column or piece column
def rotationHeights(shape):
  heights = []
  # Approach from 'top' of shape to find first instance of block in column
  for col in range(len(range(len(shape[0])))):
      height = 0
      for idx, row in reversed(list(enumerate(shape))):
        if row[col]:
          height = idx + 1
          break
      heights.append(height)
  return heights

class Tetromino:
  def __init__(self, shape, rotations, rotHeights):
    self.shape = shape
    self.rotations = rotations
    self.rotHeights = rotHeights

  def printShape(self):
      pieceStr = ""
      for row in self.shape:
        rowStr = ""
        for item in row:
          # Convert "False" to "0" and "True" to "1"
          rowStr += str(int(item)) + "  "
        pieceStr += rowStr + "\n"
      print(pieceStr)
      return

  # Iterate through every rotation and column to get all possible actions
  def getPossibleMoves(self, board):
    moves = []
    for idx, rot in enumerate(self.rotations):
      for col in range(len(board.board[0]) - len(rot[0]) + 1):
       if board.tetrominoFitsInCol(col, rot):
         moves.append([col, idx])

    return moves
