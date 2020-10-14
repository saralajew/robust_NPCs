# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.engine.topology import InputSpec

from ..capsule import Module


class SqueezeRouting(Module):
    def __init__(self, **kwargs):
        # be sure to call this at the end
        super(SqueezeRouting, self).__init__(module_input=True,
                                             module_output=True,
                                             support_sparse_signal=True,
                                             support_full_signal=True,
                                             **self._del_module_args(**kwargs))

    def _build(self, input_shape):
        if not self.built:
            if input_shape[0][1] != input_shape[1][1]:
                raise ValueError("The number of capsules must be equal to the number of prototypes. Necessary "
                                 "assumption for Routing. You provide " + str(input_shape[0][1]) + "!="
                                 + str(input_shape[1][1]) + ". Maybe you forgot the calling of a measuring module.")

            if input_shape[0][2] != 1:
                raise ValueError("The channel dimension must be one for squeezing. You provide: "
                                 + str(input_shape[0][2]))

            self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                               InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _build_sparse(self, input_shape):
        # manipulate input_shape to call the full build method instead of a new implementation
        signal_shape = list(input_shape[0])
        signal_shape[1] = input_shape[1][1]

        self._build([tuple(signal_shape), input_shape[1]])

        self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                           InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        signals = K.squeeze(inputs[0], 2)
        diss = K.squeeze(inputs[1], 2)

        return {0: signals, 1: diss}

    def _call_sparse(self, inputs, **kwargs):
        inputs = self._call(inputs, **kwargs)
        signals = inputs[0]
        signals = K.tile(signals, [1, K.shape(inputs[1])[1]] + ([1] * (self.input_spec[0].ndim - 3)))

        return {0: signals, 1: inputs[1]}

    def _compute_output_shape(self, input_shape):
        signals = list(input_shape[0])
        diss = list(input_shape[1])

        del signals[2]
        del diss[2]
        return [tuple(signals), tuple(diss)]

    def _compute_output_shape_sparse(self, input_shape):
        full_shape = self._compute_output_shape(input_shape)
        signal_shape = list(full_shape[0])
        signal_shape[1] = full_shape[1][1]

        return [tuple(signal_shape), full_shape[1]]
