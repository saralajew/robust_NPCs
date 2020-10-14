# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras import backend as K


def dict_to_list(x):
    if isinstance(x, dict):
        return [x[i] for i in x]
    else:
        return x


def list_to_dict(x):
    if isinstance(x, (tuple, list)):
        return {i: x[i] for i in range(len(x))}
    else:
        return x


def mixed_shape(inputs):
    if not K.is_tensor(inputs):
        raise ValueError('Input must be a tensor.')
    else:
        with K.name_scope('mixed_shape'):
            int_shape = list(K.int_shape(inputs))
            # sometimes int_shape returns mixed integer types
            int_shape = [int(i) if i is not None else i for i in int_shape]
            tensor_shape = K.shape(inputs)

            for i, s in enumerate(int_shape):
                if s is None:
                    int_shape[i] = tensor_shape[i]
            return tuple(int_shape)


def equal_int_shape(shape_1, shape_2):
    if not isinstance(shape_1, (tuple, list)) or not isinstance(shape_2, (tuple, list)):
        raise ValueError('Input shapes must list or tuple.')
    for shape in [shape_1, shape_2]:
        if not all([isinstance(x, int) or x is None for x in shape]):
            raise ValueError('Input shapes must be list or tuple of int and None values.')

    if len(shape_1) != len(shape_2):
        return False
    else:
        for axis, value in enumerate(shape_1):
            if value is not None and shape_2[axis] not in {value, None}:
                return False
        return True