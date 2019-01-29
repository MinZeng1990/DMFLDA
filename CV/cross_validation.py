# encoding=utf-8
import tensorflow as tf
import data_input_cv as data_input
import DMF_model_cv as DMF_model
from sklearn import metrics
import numpy as np
from  hyperparams import Hyperparams as params

batch_size = params.batch_size
epoch_num = params.epoch_num
tf.set_random_seed(params.tf_random_seed)


def validate(data_set, model, sess, roc_params=False):
    XL_batch, XR_batch, Y_batch = dl.coor_to_sample(data_set)
    y_pred, y_score, accuracy, loss = sess.run(
        [model.prediction, model.score, model.accuracy, model.loss],
        feed_dict={model.XL_input: XL_batch, model.XR_input: XR_batch,
                   model.Y_input: Y_batch})
    if not roc_params:
        return Y_batch, y_pred, y_score, accuracy, loss
    else:
        fpr, tpr, thresholds = metrics.roc_curve(Y_batch, y_score)
        return Y_batch, y_pred, y_score, accuracy, loss, fpr, tpr


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def cross_validation(dl):
    model = DMF_model.DMF()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print('begin training:')
        for epoch in range(epoch_num):
            dl.shuffle()
            print(('epoch %d' % epoch).center(50, '='))
            ys_pred = []
            ys_true = []
            ys_score = []
            for iter, indices in enumerate(range(0, dl.train_size, batch_size)):
                X_coor_batch = dl.train_set[indices:indices + batch_size]
                XL_batch, XR_batch, Y_batch = dl.coor_to_sample(X_coor_batch)
                y_pred, y_score, loss, reg_loss, _ = sess.run(
                    [model.prediction, model.score, model.loss, model.reg_loss, model.optimizer],
                    feed_dict={model.XL_input: XL_batch, model.XR_input: XR_batch,
                               model.Y_input: Y_batch})
                ys_pred.append(y_pred)
                ys_score.append(y_score)
                ys_true.append(Y_batch)

            ys_true = np.concatenate(ys_true, 0)
            ys_pred = np.concatenate(ys_pred, 0)
            ps = metrics.precision_score(ys_true, ys_pred)
            rs = metrics.recall_score(ys_true, ys_pred)
            f1 = metrics.f1_score(ys_true, ys_pred)
            accuracy = metrics.accuracy_score(ys_true, ys_pred)
            print(
                ('Training ACC:%.3f, LOSS:%.3f, Precision:%.3f, Recall:%.3f,F1:%.3f' % (
                    accuracy, loss, ps, rs, f1)).center(
                    50, '='))
        # leave one validation
        y_true, y_pred, y_score, accuracy, loss = validate(dl.val_set, model, sess)
        print(
            ('Leave-One-Validation ACC:%.3f, LOSS:%.3f RETURN:%d, SCORE:%.2f' % (
                accuracy, loss, y_pred, y_score)).center(
                50, '#'))
        # column validation
        row_id = dl.val_set[0][0]
        col_id = dl.val_set[0][1]
        x_val = dl.sample_a_col(col_id)
        y_true, y_pred, y_score, accuracy, loss, fpr, tpr = validate(x_val, model, sess, roc_params=True)
    # destroy model
    tf.reset_default_graph()
    return fpr, tpr, accuracy, y_true, y_pred, y_score


dl = data_input.DataLoader()
# 保存矩阵每一列的预测结果
ys = []
xs = []
y_true_val = []
y_pred_val = []

for i in range(len(dl.pos_set)):  # len(dl.pos_set)
    print('CROSS VALIDATION %d' % (i + 1))
    dl.leave_one_out(i)
    fpr, tpr, accuracy, y_true, y_pred, y_score = cross_validation(dl)
    row_id = dl.val_set[0][0]
    col_id = dl.val_set[0][1]
    y_true_val.append(y_true[row_id])
    y_pred_val.append(y_pred[row_id])

    # y is label matrix, 0 is test sample label, 1 is the label of positive samples, -1 is the lable of the negative samples

    xs.append(y_score)
    y = []
    for t in y_true:
        if t == 0:
            y.append(-1)
        else:
            y.append(1)
    row_id = dl.val_set[0][0]
    y[row_id] = 0
    ys.append(y)

xs = np.array(xs).transpose().squeeze(0)
ys = np.array(ys).transpose()
import numpy as np

np.save('xs.npy', xs)
np.save('ys.npy', ys)
ps = metrics.precision_score(y_true_val, y_pred_val)
rs = metrics.recall_score(y_true_val, y_pred_val)
f1 = metrics.f1_score(y_true_val, y_pred_val)

print(
    (' Precision:%.3f, Recall:%.3f,F1:%.3f' % (
        ps, rs, f1)).center(
        50, '='))
