# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import metrics

import numpy as np


def hypothesis_margin(p_norm, y_pred, y_test=None):
    # we use the squared version if possible
    if p_norm not in (1, np.inf):
        y_pred = np.sqrt(y_pred)

    if y_test is not None:
        dp = np.sum(y_test * y_pred, axis=-1)
    else:
        dp = np.min(y_pred, axis=-1)

    if y_test is not None:
        dm = np.min((1 - y_test) * y_pred + y_test * 1 / K.epsilon(), axis=-1)
    else:
        b = np.ones_like(y_pred)
        arg = np.argmin(y_pred, axis=-1)
        b[np.arange(0, y_pred.shape[0]), arg] = 0
        dm = np.min(b * y_pred + (1 - b) * 1 / K.epsilon(), axis=-1)

    margin = 0.5 * (dm - dp)

    return margin


def calculate_certificates(model, data, p_norm, certificates_epsilon,
                           p_norms=[2, np.inf], negated_dissimilarities=False):
    # it is the certified accuracy!
    # get dissimilarity
    (x_test, y_test) = data
    input_shape = x_test.shape[1:]

    y_pred = model.predict(x_test, verbose=True)

    if negated_dissimilarities:
        y_pred = -y_pred

    margin = hypothesis_margin(p_norm, y_pred, y_test)

    certificates = np.zeros((len(p_norms),))

    # get dimension of the input space
    n = np.prod(input_shape)

    for i, c in enumerate(certificates_epsilon):
        if p_norms[i] <= p_norm:
            certificates[i] = np.sum((margin - c) > 0)
        else:
            certificates[i] = np.sum((n ** (1/p_norms[i] - 1/p_norm) *
                                      margin - c) > 0)

    return certificates / len(x_test)


class Certificate(object):
    def __init__(self, p_norm, eps):
        self.p_norm = p_norm
        self.eps = eps
        self.__name__ = 'certificate'

    def __call__(self, y_true, y_pred):
        # it is the certified accuracy!

        # we use the squared version if possible
        if self.p_norm not in (1, np.inf):
            y_pred = K.sqrt(y_pred)

        dp = K.sum(y_true * y_pred, axis=-1)

        dm = K.min((1 - y_true) * y_pred + y_true * 1 / K.epsilon(), axis=-1)

        margin = 0.5 * (dm - dp)

        return K.cast(K.greater(margin - self.eps, 0), K.floatx())


def acc(y_true, y_pred):
    return metrics.categorical_accuracy(y_true, -y_pred)
