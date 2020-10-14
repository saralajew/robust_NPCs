# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import six
from keras import backend as K
from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object


# each function should exist as camel-case (class) and snake-case (function).
# It's important that a class is always camel-cass and a function is
# snake-case. Otherwise deserialization fails.
class ProbabilityTransformation(object):
    """ProbabilityTransformation base class: all prob_trans inherit from this class.
    """
    # all functions must have the parameter normalization to use it
    # automatically for regression
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, tensor):
        raise NotImplementedError

    def get_config(self):
        return {'axis': self.axis}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Softmax(ProbabilityTransformation):
    def __init__(self,
                 axis=-1,
                 max_stabilization=True):
        self.max_stabilization = max_stabilization

        super(Softmax, self).__init__(axis=axis)

    def __call__(self, tensors):
        with K.name_scope('softmax'):
            if self.max_stabilization:
                tensors = tensors - K.max(tensors, axis=self.axis, keepdims=True)

            return K.softmax(tensors, axis=self.axis)

    def get_config(self):
        config = {'max_stabilization': self.max_stabilization}
        super_config = super(Softmax, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class NegSoftmax(Softmax):
    def __call__(self, tensors):
        with K.name_scope('neg_softmax'):
            tensors = -tensors
            return super(NegSoftmax, self).__call__(tensors)


# Aliases (always calling the standard setting):


def softmax(tensors):
    prob_trans = Softmax()
    return prob_trans(tensors)


def neg_softmax(tensors):
    prob_trans = NegSoftmax()
    return prob_trans(tensors)


# copied and modified from Keras!
def serialize(probability_transformation):
    return serialize_keras_object(probability_transformation)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='probability transformation')


def get(identifier):
    if identifier is None:
        return None
    elif isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        identifier = str(identifier)

        if identifier.islower():
            return deserialize(identifier)
        else:
            config = {'class_name': identifier, 'config': {}}
            return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret probability transformation identifier: ' + str(identifier))
