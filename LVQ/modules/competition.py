# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine.topology import InputSpec
from keras import backend as K
from keras import regularizers

from ..utils.caps_utils import mixed_shape

from ..capsule import Module


class NearestCompetition(Module):
    def __init__(self,
                 use_for_loop=True,
                 signal_regularizer=None,
                 diss_regularizer=None,
                 **kwargs):
        self.use_for_loop = use_for_loop

        self.output_regularizers = [regularizers.get(signal_regularizer),
                                    regularizers.get(diss_regularizer)]

        # be sure to call this at the end
        super(NearestCompetition, self).__init__(module_input=True,
                                                 module_output=True,
                                                 support_sparse_signal=True,
                                                 support_full_signal=True,
                                                 **self._del_module_args(**kwargs))

    def _build(self, input_shape):
        if input_shape[0][1] != self.proto_number:
            raise ValueError('The capsule number provided by input_shape is not equal the self.proto_number: '
                             'input_shape[0][1]=' + str(input_shape[0][1]) + ' != ' +
                             'self.proto_number=' + str(self.proto_number) + ". Maybe you forgot to call a routing"
                             " module.")
        if input_shape[1][1] != self.proto_number:
            raise ValueError('The prototype number provided by input_shape is not equal the self.proto_number: '
                             'input_shape[1][1]=' + str(input_shape[1][1]) + ' != ' +
                             'self.proto_number=' + str(self.proto_number))
        if len(input_shape[1]) != 2:
            raise ValueError("The dissimilarity vector must be of length two (batch, dissimilarities per prototype). "
                             "You provide: " + str(len(input_shape[1])) + ". Maybe you forgot to call a routing "
                             "module.")

        self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                           InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _build_sparse(self, input_shape):
        self._build(input_shape)

    def _call(self, inputs, **kwargs):
        if self.proto_number == self.capsule_number:
            return inputs
        else:
            signals = inputs[0]
            diss = inputs[1]
            signal_shape = mixed_shape(signals)

            if self.use_for_loop:
                diss_stack = []
                signals_stack = []
                sub_idx = None
                with K.name_scope('for_loop'):
                    for p in self._proto_distrib:
                        with K.name_scope('compute_slices'):
                            diss_ = diss[:, p[0]:(p[-1]+1)]
                            signals_ = K.reshape(signals[:, p[0]:(p[-1]+1), :],
                                                 [signal_shape[0] * len(p)] + list(signal_shape[2:]))
                        with K.name_scope('competition'):
                            if len(p) > 1:
                                with K.name_scope('competition_indices'):
                                    argmin_idx = K.argmin(diss_, axis=-1)
                                    if sub_idx is None:
                                        sub_idx = K.arange(0, signal_shape[0], dtype=argmin_idx.dtype)
                                    argmin_idx = argmin_idx + len(p) * sub_idx

                                with K.name_scope('dissimilarity_competition'):
                                    diss_stack.append(K.expand_dims(K.gather(K.flatten(diss_), argmin_idx), -1))

                                with K.name_scope('signal_competition'):
                                    signals_stack.append(K.gather(signals_, argmin_idx))
                            else:
                                diss_stack.append(diss_)
                                signals_stack.append(signals_)

                diss = K.concatenate(diss_stack, 1)

                with K.name_scope('signal_concatenation'):
                    signals = K.concatenate(signals_stack, 1)
                    signals = K.reshape(signals, [signal_shape[0], self.capsule_number] + list(signal_shape[2:]))

            else:
                with K.name_scope('dissimilarity_preprocessing'):
                    # extend if it is not equally distributed
                    if not self._equally_distributed:
                        # permute to first dimension is prototype (protos x batch)
                        diss = K.permute_dimensions(diss, [1, 0])
                        # gather regarding extension (preparing for reshape to block)
                        diss = K.gather(diss, self._proto_extension)
                        # permute back (max_proto_number x (max_proto_number * batch))
                        diss = K.permute_dimensions(diss, [1, 0])

                    # reshape to block form
                    diss = K.reshape(diss, [signal_shape[0] * self.capsule_number, self._max_proto_number_in_capsule])

                with K.name_scope('competition_indices'):
                    # get minimal idx in each class and batch for element selection in diss and signals
                    argmin_idx = K.argmin(diss, axis=-1)
                    argmin_idx = argmin_idx + self._max_proto_number_in_capsule * \
                                 K.arange(0, signal_shape[0] * self.capsule_number, dtype=argmin_idx.dtype)

                with K.name_scope('dissimilarity_competition'):
                    # get minimal values in the form (batch x capsule)
                    diss = K.gather(K.flatten(diss), argmin_idx)
                    diss = K.reshape(diss, [signal_shape[0], self.capsule_number])

                with K.name_scope('signal_preprocessing'):
                    # apply the same steps as above for signals
                    # get signals in: (batch x protos x dim1 x ... x dimN) --> out: (batch x capsule x dim1 x ... x dimN)
                    # extend if is not equally distributed
                    if not self._equally_distributed:
                        signals = K.permute_dimensions(signals, [1, 0] + list(range(2, len(signal_shape))))
                        signals = K.gather(signals, self._proto_extension)
                        signals = K.permute_dimensions(signals, [1, 0] + list(range(2, len(signal_shape))))

                    signals = K.reshape(signals,
                                        [signal_shape[0] * self.capsule_number * self._max_proto_number_in_capsule]
                                        + list(signal_shape[2:]))

                with K.name_scope('signal_competition'):
                    signals = K.gather(signals, argmin_idx)
                    signals = K.reshape(signals, [signal_shape[0], self.capsule_number] + list(signal_shape[2:]))

            return {0: signals, 1: diss}

    def _call_sparse(self, inputs, **kwargs):
        return self._call(inputs, **kwargs)

    def _compute_output_shape(self, input_shape):
        signals = list(input_shape[0])
        diss = list(input_shape[1])

        signals[1] = self.capsule_number
        diss[1] = self.capsule_number
        return [tuple(signals), tuple(diss)]

    def _compute_output_shape_sparse(self, input_shape):
        return self._compute_output_shape(input_shape)

    def get_config(self):
        config = {'use_for_loop': self.use_for_loop,
                  'signal_regularizer': regularizers.serialize(self.output_regularizers[0]),
                  'diss_regularizer': regularizers.serialize(self.output_regularizers[1])}
        super_config = super(NearestCompetition, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))
