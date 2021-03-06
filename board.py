import numpy as np
import pandas as pd

class Board:
  def __init__(self, rows, cols):
    self.nrows = rows
    self.ncols = cols
    self.reset()

  def reset(self):
    self.colHeights = np.zeros(self.ncols)
    self.linesCleared = 0
    self.board = np.zeros((self.nrows, self.ncols), dtype=bool)
    self.dfBoard = pd.DataFrame(self.board)

  def calcColHeights(self):
    self.colHeights = [0 if sum(self.dfBoard[col]) == 0 else self.dfBoard[col][::-1].idxmax() + 1 for col in self.dfBoard]

  # Print the board in Tetris format
  def printBoard(self):
    print(self.dfBoard[::-1].astype(int).to_string(header=False,index=False))
    print("\n")
    return

  # Return True if tetromino shape fits in column
  def tetrominoFitsInCol(self, col, tetromino, rHeights):
    nCols = tetromino.shape[1]
    # TODO: Store np.fulls in Tetromino class
    # Compare number of rows remaining in board column with height of tetromino piece
    return not np.any(np.greater(rHeights, np.subtract(np.full((1, nCols), self.nrows), (self.colHeights[col:col+nCols]))))

  def getAnchorColumn(self, tetromino, rHeights, col):
    anchorCol = np.argmax([sum(x) for x in zip(rHeights, self.colHeights[col:col+tetromino.shape[1]])])
    return anchorCol, self.colHeights[col + anchorCol]

  def addAtPos(self, tShape, xycoor):
    size_x, size_y = tShape.shape
    coor_x, coor_y = xycoor
    end_x, end_y   = (coor_x + size_x), (coor_y + size_y)

    self.board[coor_x:end_x, coor_y:end_y] = self.board[coor_x:end_x, coor_y:end_y] + tShape
    return

  def act(self, tetromino, col, rot):
    tShape = tetromino.rotations[rot]
    heights = tetromino.rHeights[rot]
    anchorCol, anchorHeight = self.getAnchorColumn(tShape, tetromino.rHeights[rot], col)

    # Update board configuration (add new piece)
    self.addAtPos(tShape, (int(anchorHeight - (np.max(heights) - heights[anchorCol])), col))
    return self.findClearedLines()

  # Clear and count full lines
  def findClearedLines(self):
    nClearedLines = 0
    newBoard = np.empty((0, self.ncols))

    for row in self.board:
      if not np.all(row):
        newBoard = np.append(newBoard, [row], axis=0)
      else:
        nClearedLines += 1

    self.board = np.pad(newBoard, ((0, nClearedLines), (0, 0)), 'constant')
    self.dfBoard = pd.DataFrame(self.board)
    self.calcColHeights()
    self.linesCleared += nClearedLines
    return nClearedLines
