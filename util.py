import random
import numpy as np

# Given a list, return a random element from the list
def randChoice(l):
    return l[random.randint(0, len(l) - 1)]

# For this implementation, concatenate board config and
# Tetromino config into a string for the state
def strState(board, tetromino):
    bString = ''.join(''.join('%d' %x for x in row) for row in board)
    tString = ''.join(''.join('%d' %x for x in row) for row in tetromino)
    return bString + tString

def networkState(board, tetromino):
    return np.array([np.append(board.flatten(), tetromino.flatten())])

def epsilonGreedy(q, epsilon, possMoves):
    if random.random() < epsilon:
        return randChoice(possMoves)
    else:
        qPossMoves = []
        for p in possMoves:
            qPossMoves.append(q[p[0]][p[1]])
        return possMoves[np.argmax(qPossMoves)]
