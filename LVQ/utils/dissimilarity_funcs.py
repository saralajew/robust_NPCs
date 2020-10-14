# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras import backend as K

from .caps_utils import mixed_shape, equal_int_shape
from .linalg_funcs import p_norm, euclidean_distance

import numpy as np


def _check_shapes(signal_int_shape, proto_int_shape):
    if len(signal_int_shape) < 4:
        raise ValueError("The number of signal dimensions must be >=4. You provide: " + str(len(signal_int_shape)))

    if len(proto_int_shape) < 2:
        raise ValueError("The number of proto dimensions must be >=2. You provide: " + str(len(proto_int_shape)))

    if not equal_int_shape(signal_int_shape[3:], proto_int_shape[1:]):
        raise ValueError("The atom shape of signals must be equal protos. You provide: signals.shape[3:]="
                         + str(signal_int_shape[3:]) + " != protos.shape[1:]=" + str(proto_int_shape[1:]))

    # not a sparse signal
    if signal_int_shape[1] != 1:
        if not equal_int_shape(signal_int_shape[1:2], proto_int_shape[0:1]):
            raise ValueError("If the signal is not sparse, the number of prototypes must be equal in signals and "
                             "protos. You provide: " + str(signal_int_shape[1]) + " != " + str(proto_int_shape[0]))

    return True


def _int_and_mixed_shape(tensor):
    shape = mixed_shape(tensor)
    int_shape = tuple([i if isinstance(i, int) else None for i in shape])

    return shape, int_shape


def minkowski_distance(signals, protos,
                       order_p=2,
                       squared=False,
                       epsilon=K.epsilon()):
    # example Input:
    # shape(signals): batch x proto_number (or 1) x channels x dim1 x dim2 x ... x dimN
    # shape(protos): proto_number x dim1 x dim2 x ... x dimN
    # it will compute the distance to all prototypes regarding each channel and batch.
    #
    # the speedup with sparse_signals is only possible if order_p == 2 otherwise the identity between the distance and
    # the dot product is not valid

    signal_shape, signal_int_shape = _int_and_mixed_shape(signals)
    proto_shape, proto_int_shape = _int_and_mixed_shape(protos)

    # check if the shapes are correct
    _check_shapes(signal_int_shape, proto_int_shape)

    with K.name_scope('minkowski_distance'):
        # sum_axes: [3, 4, ..., N+2]; each dimension after 3 is considered as data dimension
        atom_axes = list(range(3, len(signal_int_shape)))

        # for sparse signals, we use the memory efficient implementation
        if signal_int_shape[1] == 1 and order_p == 2:
            signals = K.reshape(signals, [-1, np.prod(signal_shape[3:])])

            if len(atom_axes) > 1:
                protos = K.reshape(protos, [proto_shape[0], -1])

            diss = euclidean_distance(signals, protos, squared=squared, epsilon=epsilon)

            diss = K.reshape(diss, [signal_shape[0], signal_shape[2], proto_shape[0]])

        else:
            signals = K.permute_dimensions(signals, [0, 2, 1] + list(range(3, len(signal_shape))))

            diff = signals - protos

            diss = p_norm(diff, order_p=order_p, axis=atom_axes,
                          squared=squared, keepdims=False, epsilon=epsilon)

        return K.permute_dimensions(diss, [0, 2, 1])


def tangent_distance(signals, protos, subspaces,
                     squared=False,
                     epsilon=K.epsilon()):
    # Note: subspaces is always assumed as transposed and must be orthogonal!
    # shape(signals): batch x proto_number x channels x dim1 x dim2 x ... x dimN
    # shape(protos): proto_number x dim1 x dim2 x ... x dimN
    # shape(subspaces): (optional [proto_number]) x prod(dim1 * dim2 * ... * dimN)  x prod(projected_atom_shape)
    # subspace should be orthogonalized

    signal_shape, signal_int_shape = _int_and_mixed_shape(signals)
    proto_shape, proto_int_shape = _int_and_mixed_shape(protos)
    subspace_int_shape = K.int_shape(subspaces)

    # check if the shapes are correct
    _check_shapes(signal_int_shape, proto_int_shape)

    with K.name_scope('tangent_distance'):
        atom_axes = list(range(3, len(signal_int_shape)))
        # for sparse signals, we use the memory efficient implementation
        if signal_int_shape[1] == 1:
            signals = K.reshape(signals, [-1, np.prod(signal_shape[3:])])

            if len(atom_axes) > 1:
                protos = K.reshape(protos, [proto_shape[0], -1])

            if K.ndim(subspaces) == 2:
                # clean solution without map_fn if the matrix_scope is global
                with K.name_scope('projectors'):
                    projectors = K.eye(subspace_int_shape[-2]) - K.dot(subspaces, K.transpose(subspaces))

                with K.name_scope('tangentspace_projections'):
                    projected_signals = K.dot(signals, projectors)
                    projected_protos = K.dot(protos, projectors)

                diss = euclidean_distance(projected_signals, projected_protos, squared=squared, epsilon=epsilon)

                diss = K.reshape(diss, [signal_shape[0], signal_shape[2], proto_shape[0]])

                return K.permute_dimensions(diss, [0, 2, 1])

            else:
                # no solution without map_fn possible --> memory efficient but slow!
                with K.name_scope('projectors'):
                    projectors = K.eye(subspace_int_shape[-2]) - K.batch_dot(subspaces, subspaces, [2, 2])

                with K.name_scope('tangentspace_projections'):
                    projected_protos = K.transpose(K.batch_dot(projectors, protos, [1, 1]))

                with K.name_scope('euclidean_distance'):
                    def projected_norm(projector):
                        return K.sum(K.square(K.dot(signals, projector)), axis=1)

                    diss = K.transpose(K.map_fn(projected_norm, projectors)) \
                           - 2 * K.dot(signals, projected_protos) \
                           + K.sum(K.square(projected_protos), axis=0, keepdims=True)

                    if not squared:
                        if epsilon == 0:
                            diss = K.sqrt(diss)
                        else:
                            diss = K.sqrt(K.maximum(diss, epsilon))

                diss = K.reshape(diss, [signal_shape[0], signal_shape[2], proto_shape[0]])

                return K.permute_dimensions(diss, [0, 2, 1])

        else:
            signals = K.permute_dimensions(signals, [0, 2, 1] + atom_axes)
            diff = signals - protos

            # global tangent space
            if K.ndim(subspaces) == 2:
                with K.name_scope('projectors'):
                    projectors = K.eye(subspace_int_shape[-2]) - K.dot(subspaces, K.transpose(subspaces))

                with K.name_scope('tangentspace_projections'):
                    diff = K.reshape(diff, (signal_shape[0] * signal_shape[2], signal_shape[1], -1))
                    projected_diff = K.dot(diff, projectors)
                    projected_diff = K.reshape(projected_diff,
                                               (signal_shape[0], signal_shape[2], signal_shape[1]) + signal_shape[3:])

                diss = p_norm(projected_diff, order_p=2, axis=atom_axes, squared=squared, keepdims=False, epsilon=epsilon)
                return K.permute_dimensions(diss, [0, 2, 1])

            # local tangent spaces
            else:
                with K.name_scope('projectors'):
                    projectors = K.eye(subspace_int_shape[-2]) - K.batch_dot(subspaces, subspaces, [2, 2])

                with K.name_scope('tangentspace_projections'):
                    diff = K.reshape(diff, (signal_shape[0] * signal_shape[2], signal_shape[1], -1))
                    diff = K.permute_dimensions(diff, [1, 0, 2])
                    projected_diff = K.batch_dot(diff, projectors)
                    projected_diff = K.reshape(projected_diff,
                                               (signal_shape[1], signal_shape[0], signal_shape[2]) + signal_shape[3:])

                diss = p_norm(projected_diff, order_p=2, axis=atom_axes, squared=squared, keepdims=False, epsilon=epsilon)
                return K.permute_dimensions(diss, [1, 0, 2])
