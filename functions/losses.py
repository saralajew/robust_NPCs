# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K

import numpy as np


class TripletReluLoss(object):
    def __init__(self, p_norm, eps=0.):
        self.p_norm = p_norm
        self.eps = eps

    def __call__(self, y_true, y_pred):
        with K.name_scope('triplet_relu_loss'):
            # y_true categorical vector (one-hot value)

            # we use the squared version if possible
            if self.p_norm not in (1, np.inf):
                y_pred = K.sqrt(y_pred)

            dp = K.sum(y_true * y_pred, axis=-1)

            dm = K.min((1 - y_true) * y_pred + y_true * 1 / K.epsilon(),
                       axis=-1)

            # we add K.epsilon() to ensure that the loss optimizes for
            # self.eps-limited adversarial attacks
            loss = K.relu(self.eps - 0.5 * (dm - dp) + K.epsilon())

            return loss


class GlvqLoss(object):
    def __init__(self, squash_func=None):
        self.squash_func = squash_func
        self.__name__ = 'glvq_loss'

    def __call__(self, y_true, y_pred):
        with K.name_scope('glvq_loss'):
            dp = K.sum(y_true * y_pred, axis=-1)

            dm = K.min((1 - y_true) * y_pred + y_true * 1 / K.epsilon(),
                       axis=-1)

            loss = (dp - dm) / (dp + dm)

            if self.squash_func is not None:
                loss = self.squash_func(loss)

            return loss
