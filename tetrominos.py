import numpy

class Tetromino:
  def __init__(self, idx):
    allTetrominos = [
      [[True, False],
       [True, True]],

       [[True, True, True]]
    ]
    self.shape = allTetrominos[idx]
    self.rotations = self.getRotations()

  def printShape(self, shape):
      pieceStr = ""
      for row in shape:
        rowStr = ""
        for item in row:
          # Convert "False" to "0" and "True" to "1"
          rowStr += str(int(item)) + "  "
        pieceStr += rowStr + "\n"
      print pieceStr
      return

  def getPossibleMoves(self, board):
      return []

  def getRotations(self):
    rotations = [numpy.array(self.shape)]
    currentRot = self.shape

    # 4 is max no. of rotations (first one already stored)
    for i in range(3):
      newRot = numpy.rot90(currentRot)

      # Check if rotation is already in array - if it is, no more unique rotations
      for rot in rotations:
        if numpy.all(newRot == rot) and len(newRot) == len(rot):
          return rotations

      rotations.append(newRot)
      currentRot = newRot

    return rotations
