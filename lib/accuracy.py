from __future__ import division
import numpy as np
# Always assume division should return a float.

def binarized_accuracy(predict, actual):
    correct_predictions = np.equal(np.argmax(predict,1), np.argmax(actual,1))
    percent = np.sum(correct_predictions) / correct_predictions.shape[0]
    return percent

def mean_f1_score(y_pred, y_true):
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_true, np.ndarray)

    y_pred_dim = y_pred.shape
    y_true_dim = y_true.shape
    if y_pred_dim != y_true_dim:
        raise Exception("pred and test nd arrays must have the same shape")

    y_pred= np.ravel(y_pred)
    y_true = np.ravel(y_true)

    true_items = np.equal(y_pred, y_true)
    false_items = np.logical_not(true_items)
    true_pos = np.sum(np.reshape(np.logical_and(true_items, y_pred), y_pred_dim), axis=1).astype("float64")
    false_pos = np.sum(np.reshape(np.logical_and(false_items, y_pred), y_pred_dim), axis=1)
    false_neg = np.sum(np.reshape(np.logical_and(false_items, np.logical_not(y_pred)), y_pred_dim), axis=1)


    tp_fp = true_pos + false_pos
    tp_fn = true_pos + false_neg
    p = true_pos / tp_fp
    r = true_pos / tp_fn
    div_bottom = p + r
    div_top =  2. * p * r
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.divide( div_top, div_bottom )
        f1[f1 == np.inf] = 0
        f1 = np.nan_to_num(f1)
        f1_mean = np.mean(f1)
        return f1_mean

#
# y_true = [[1, 1, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 1, 1, 1, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 1, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 1, 0, 0],
#           ]
# y_pred = [[1, 1, 1, 0, 0, 0, 0, 0, 1],
#           [0, 0, 1, 1, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 1, 0, 0, 1],
#           [1, 0, 0, 0, 0, 0, 0, 0, 0],
#           ]
#
# y_true = np.array(y_true)
# y_pred = np.array(y_pred)
#
# print mean_f1_score(y_pred, y_true)


