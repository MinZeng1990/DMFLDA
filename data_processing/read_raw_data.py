# encoding=utf-8

from scipy.io import loadmat
import numpy as np
import pickle

filename = "../drug-target/e_admat_dgc.txt"

matrix = []
with open(filename, "r") as file:
    file.readline()
    for line in file:
        digit_row = line.split()[1:]
        matrix.append(digit_row)

interMatrix = np.array(matrix, dtype=np.int32)
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
