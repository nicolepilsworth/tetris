import pprint
import numpy
from random import randint
from tetrominos import Tetromino
from board import Board


def play():
  board = Board(5, 3)
  board.printBoard()

  while(not board.gameOver):
    # Random Tetromino for next state
    tetromino = Tetromino(randint(0, 1))
    tetromino.printShape(tetromino.shape)

    # Moves come in the format "columnIndex_rotationIndex"
    possibleMoves = tetromino.getPossibleMoves(board)

    # Game over condition
    if len(possibleMoves) == 0:
        print("GAME OVER")
        break

    # TODO: change to select optimal action under Q-learning policy
    action = possibleMoves[randint(0, len(possibleMoves) - 1)]

    r = board.act(tetromino, action)
    board.printBoard()

  print("Lines cleared: ", board.linesCleared)

def main():
  play()

if __name__ == "__main__":
  main()
