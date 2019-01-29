# encoding=utf-8

from scipy.io import loadmat
import numpy as np
import pickle

"""
matrix shape: (577,272)
positive samples: 1583
negative samples: 155361
"""
m = loadmat("interMatrix.mat")
interMatrix = m['interMatrix']

rows, cols = interMatrix.shape
print('matrix shape:', interMatrix.shape)
pos_set = []
neg_set = []
for i in range(rows):
    for j in range(cols):
        if interMatrix[i][j] != 0:
            pos_set.append((i, j, 1))
        else:
            neg_set.append((i, j, 0))

print('positive samples:', len(pos_set))
print('negative samples:', len(neg_set))

with open('data.pkl', 'wb') as file:
    pickle.dump((pos_set, neg_set), file)

np.save('matrix.npy', interMatrix)

