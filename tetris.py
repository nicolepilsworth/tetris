import pprint
import numpy
from tetrominos import Tetromino

class Board:
  def __init__(self, rows, cols):
    self.nrows = rows
    self.ncols = cols
    self.board = [[False] * cols] * rows
    self.gameOver = False

  # Print the board in Tetris format
  def printBoard(self):
    boardStr = ""
    for row in self.board:
      rowStr = "|   "
      for item in row:
        # Convert "False" to "0" and "True" to "1"
        rowStr += str(int(item)) + "   "
      rowStr += "|"
      boardStr += rowStr + "\n"
    print boardStr
    return

def play():
  board = Board(5, 3)
  board.printBoard()

  while(not board.gameOver):
    tetromino = Tetromino(0)
    tetromino.printShape()

    board.gameOver = True

def main():
  play()

if __name__ == "__main__":
  main()
