import numpy

# Get column height of either board column or piece column
def getColHeight(shape, col):
  # Approach from 'top' of shape
  for idx, row in reversed(list(enumerate(shape))):
    # print type, col, row
    if row[col]:
      return idx + 1
  return 0

class Board:
  def __init__(self, rows, cols):
    self.nrows = rows
    self.ncols = cols

    self.board = []
    for i in range(rows):
      self.board.append([False] * cols)

    self.linesCleared = 0

    self.gameOver = False

  # Print the board in Tetris format
  def printBoard(self):
    boardStr = ""
    # Bottom elements of board appear first, so board must be reversed
    for row in reversed(self.board):
      rowStr = "|   "
      for item in row:
        # Convert "False" to "0" and "True" to "1"
        rowStr += str(int(item)) + "   "
      rowStr += "|"
      boardStr += rowStr + "\n"
    print boardStr
    return

  def getRowsInColRemaining(self, col):
      return self.nrows - getColHeight(self.board, col)

  # Return True if tetromino shape fits in column
  def tetrominoFitsInCol(self, col, tetromino):
    for i in range(len(tetromino[0])):
      # Compare number of rows remaining in board column with height of tetromino piece
      if self.getRowsInColRemaining(col+i) < getColHeight(tetromino, i):
        return False
    return True

  def canPlaceTetromino(self, row, col, tetromino):
    for i in reversed(range(len(tetromino))):
      for j in range(len(tetromino[i])):
        if not ((tetromino[i][j] and not self.board[row-i][col+j]) or (not tetromino[i][j])):
          return False
    return True

  def getMinPlacementHeight(self, tetromino, action):
    col = int(action.split("_")[0])

    minHeight = self.nrows
    for row in reversed(range(len(self.board))):
      if row < len(tetromino) - 1:
        return minHeight
      if self.canPlaceTetromino(row, col, tetromino):
        minHeight = row
      else:
        return minHeight
    return minHeight

  def makeMove(self, tetromino, action):
    [col, rot] = map(int, (action.split("_")))
    tetShape = tetromino.rotations[rot]
    minHeight = self.getMinPlacementHeight(tetShape, action)

    for i in reversed(range(len(tetShape))):
      for j in range(len(tetShape[0])):
        self.board[minHeight-i][col+j] = (self.board[minHeight-i][col+j] or tetShape[i][j])

    return self.findClearedLines()

  def findClearedLines(self):
    nClearedLines = 0
    for idx, row in reversed(list(enumerate(self.board))):
      if numpy.all(row):
        del self.board[idx]
        self.board.append([False] * self.ncols)
        nClearedLines += 1

    self.linesCleared += nClearedLines
    return nClearedLines
