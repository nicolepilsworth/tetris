class Tetromino:
  def __init__(self, idx):
    allTetrominos = [
      [[True, False],
       [True, True]],

       [[True, True, True]]
    ]
    self.shape = allTetrominos[idx]

  def printShape(self):
      pieceStr = ""
      for row in self.shape:
        rowStr = ""
        for item in row:
          # Convert "False" to "0" and "True" to "1"
          rowStr += str(int(item)) + "  "
        pieceStr += rowStr + "\n"
      print pieceStr
      return

  def getPossibleMoves(self, board):
      return []

  def getRotations(shape):
      return 0
