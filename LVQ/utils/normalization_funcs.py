# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras import backend as K

from .linalg_funcs import svd
from .caps_utils import mixed_shape


def orthogonalization(tensors):
    # orthogonalization via polar decomposition
    with K.name_scope('orthogonalization'):
        _, u, v = svd(tensors, full_matrices=False, compute_uv=True)
        u_shape = mixed_shape(u)
        v_shape = mixed_shape(v)

        # reshape to (num x N x M)
        u = K.reshape(u, (-1, u_shape[-2], u_shape[-1]))
        v = K.reshape(v, (-1, v_shape[-2], v_shape[-1]))

        out = K.batch_dot(u, K.permute_dimensions(v, [0, 2, 1]))

        out = K.reshape(out, u_shape[:-1] + (v_shape[-2],))

        return out
