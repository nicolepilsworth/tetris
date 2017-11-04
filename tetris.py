import pprint
import numpy
from tetrominos import Tetromino
from board import Board

def play():
  board = Board(5, 3)
  board.printBoard()

  while(not board.gameOver):
    tetromino = Tetromino(1)

    # Print all rotations
    # for rot in tetromino.rotations:
    #   tetromino.printShape(rot)

    moves = tetromino.getPossibleMoves(board)
    board.gameOver = True

def main():
  play()

if __name__ == "__main__":
  main()
