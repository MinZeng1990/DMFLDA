import numpy as np

threshold = 0.5


def LOOCV_ROC(x, y):
    ### x is predicted score matrix (composed of multi test samples);
    # y is label matrix, 0 is test sample label, 1 is the label of positive samples, -1 is the lable of the negative samples
    num_row = x.shape[0]
    num_col = x.shape[1]
    FPR_vec = [0] * num_row
    TPR_vec = [0] * num_row
    for i in range(0, num_col):
        col_label = list(y[:, i])
        col_score = list(x[:, i])
        pos_indices = [i for i, x in enumerate(col_label) if x == 1]
        while len(pos_indices) > 0:
            index = col_label.index(1)
            del col_label[index]
            del col_score[index]
            pos_indices = [i for i, x in enumerate(col_label) if x == 1]

        ## cal fpr,tpr
        col_sortedIndices = (-np.array(col_score)).argsort()
        row_col = len(col_label)
        X = [0] * num_row
        Y = [0] * num_row
        P = col_label.count(0)
        N = row_col - P

        TP = 0
        FP = 0
        for j in range(0, row_col):
            if col_label[col_sortedIndices[j]] == 0:
                TP = TP + 1
            else:
                FP = FP + 1
            X[j] = FP / N  ### FPR
            Y[j] = TP / P  ### TPR

        if row_col < num_row:  ### complete each column vec
            for k in range(row_col, num_row):
                X[k] = X[row_col - 1]
                Y[k] = Y[row_col - 1]

        FPR_vec = np.array(FPR_vec) + np.array(X)
        TPR_vec = np.array(TPR_vec) + np.array(Y)

    FPR_vec = FPR_vec / num_col
    TPR_vec = TPR_vec / num_col

    return (FPR_vec, TPR_vec)


# xt = np.array([[6, 2, 3, 4, 5], [1, 7, 8, 9, 10], [7, 4, 6, 11, 2]])
# x = xt.transpose()
#
# yt = np.array([[0, -1, 1, 1, -1], [1, 1, -1, 0, -1], [-1, -1, -1, 0, -1]])
# y = yt.transpose()
#
# c = LOOCV_ROC(x, y)

xs = np.load('xs.npy')
ys = np.load('ys.npy')

print(xs.shape, ys.shape)

fpr, tpr = LOOCV_ROC(xs, ys)

import sklearn.metrics as metrics

auc = metrics.auc(fpr, tpr)
print("AUC:%.3f" % auc)

row_num, col_num = xs.shape

y_pred = []
y_true = []
count = 0
for i in range(col_num):  # len(dl.pos_set)
    for j in range(row_num):
        if ys[j][i] == 0:
            y_pred.append(1 if xs[j][i] >= threshold else 0)
            y_true.append(1)
            count += 1

ps = metrics.precision_score(y_true, y_pred)
rs = metrics.recall_score(y_true, y_pred)
f1 = metrics.f1_score(y_true, y_pred)

print(
    (' Precision:%.3f, Recall:%.3f,F1:%.3f' % (
        ps, rs, f1)).center(
        50, '='))
