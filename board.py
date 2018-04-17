import numpy as np
import pandas as pd
from features import Features
from features import distinct
from features import pareto

class Board:
  def __init__(self, rows, cols):
    self.nrows = rows
    self.ncols = cols
    self.reset()

  def reset(self):
    self.colHeights = np.zeros(self.ncols)
    self.linesCleared = 0
    self.board = np.zeros((self.nrows, self.ncols))
    self.yMax = 0
    self.dfBoard = pd.DataFrame(self.board)

  def calcColHeights(self):
    self.colHeights = [0 if sum(self.dfBoard[col]) == 0 else self.dfBoard[col][::-1].idxmax() + 1 for col in self.dfBoard]
    self.yMax = np.max(self.colHeights) - 1
  # Print the board in Tetris format
  def printBoard(self):
    print(self.dfBoard[::-1].astype(int).to_string(header=False,index=False))
    print("\n")
    return

  def printBoardTry(self, b):
      print(pd.DataFrame(b)[::-1].astype(int).to_string(header=False,index=False))
      print("\n")
      return

  # Return True if tetromino shape fits in column
  def tetrominoFitsInCol(self, col, tetromino, rHeights):
    nCols = tetromino.shape[1]
    # Compare number of rows remaining in board column with height of tetromino piece
    return not np.any(np.greater(rHeights, np.subtract(np.full((1, nCols), self.nrows), (self.colHeights[col:col+nCols]))))

  def getAnchorColumn(self, tetromino, rHeights, col):
    anchorCol = np.argmax([sum(x) for x in zip(rHeights, self.colHeights[col:col+tetromino.shape[1]])])
    return anchorCol, self.colHeights[col + anchorCol]

# 'tryMove' boolean being True does not affect board - are just simulations of moves made
  def addAtPos(self, tShape, xycoor, tryMove):
    size_x, size_y = tShape.shape
    coor_x, coor_y = xycoor
    end_x, end_y   = (coor_x + size_x), (coor_y + size_y)

    if tryMove:
      shape_coords = []
      b = np.copy(self.board)
      b[coor_x:end_x, coor_y:end_y] = b[coor_x:end_x, coor_y:end_y] + tShape
      l_height = (end_x + coor_x + 1)/2
      for r in range(size_x):
          shape_coords.append([r + coor_x, sum(tShape[r])])
      return b, shape_coords, l_height
    else:
      self.board[coor_x:end_x, coor_y:end_y] = self.board[coor_x:end_x, coor_y:end_y] + tShape
      return

  def act(self, tetromino, col, rot, tryMove=False):
    tShape = tetromino.rotations[rot]
    heights = tetromino.rHeights[rot]

    # The anchor is the first instance of a board cell that stops the Tetromino's fall
    anchorCol, anchorHeight = self.getAnchorColumn(tShape, tetromino.rHeights[rot], col)

    # Update board configuration (add new piece)
    if tryMove:
      b, shape_coords, l_height = self.addAtPos(tShape, (int(anchorHeight - (np.max(heights) - heights[anchorCol])), col), True)
      return self.findClearedLines(True, b, shape_coords) + [l_height]
    else:
      self.addAtPos(tShape, (int(anchorHeight - (np.max(heights) - heights[anchorCol])), col), False)
      return self.findClearedLines(False, None, None)

  # Clear and count full lines
  def findClearedLines(self, tryMove, b, shape_coords):

    # if tryMove:
    #     print(shape_coords)
    nClearedLines = 0
    newBoard = np.empty((0, self.ncols))
    eroded_cell_count = 0
    rowsCleared = []

    if not tryMove:
      b = self.board

    for i, row in enumerate(b):
      if not np.all(row):
        newBoard = np.append(newBoard, [row], axis=0)
      else:
        rowsCleared.append(row)
        # print(shape_coords)
        # self.printBoardTry(b)
        # print(i, shape_coords)
        if tryMove:
            eroded_cells = next(filter(lambda x: x[0] == i, shape_coords))
            eroded_cell_count += eroded_cells[1]

    nClearedLines = len(rowsCleared)
    if tryMove:
      b = np.pad(newBoard, ((0, nClearedLines), (0, 0)), 'constant')
      return [b, eroded_cell_count * nClearedLines]
    else:
      self.board = np.pad(newBoard, ((0, nClearedLines), (0, 0)), 'constant')
      self.dfBoard = pd.DataFrame(self.board)
      self.calcColHeights()
      self.linesCleared += nClearedLines
      return nClearedLines

  # def actBCTS(self, tetromino, col, rot):
  #     tShape = tetromino.rotations[rot]
  #     heights = tetromino.rHeights[rot]
  #     anchorCol, anchorHeight = self.getAnchorColumn(tShape, tetromino.rHeights[rot], col)


  # Use BCTS weights with cumulative/simple dominance to filter moves with best features
  def findBestMoves(self, valid_moves, tetromino):
    objects = []
    for move in valid_moves:
      rot, col = divmod(move, self.ncols)
      b, eroded, l_height = self.act(tetromino, col, rot, True)
      features = Features(b, self.yMax,tetromino.heights[rot], move, self.nrows, self.ncols, eroded, l_height)
      features.listFeatures()
      objects.append(features)


    # print(list(map(lambda x: x.pos, objects)))
    distinct_moves = distinct(objects)
    paretoSimple = pareto(distinct_moves, 'simple')
    paretoCumulative = pareto(paretoSimple, 'cumulative')
    return paretoCumulative
