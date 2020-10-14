# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import initializers, regularizers
from keras.engine.topology import InputSpec
from keras import backend as K

import numpy as np

import warnings

from ..capsule import Module
from ..utils.caps_utils import mixed_shape
from ..utils import dissimilarity_funcs as diss_funcs
from LVQ import constraints
from ..utils import pre_training


class PointPrototype(Module):
    def __init__(self,
                 prototype_initializer='TruncatedNormal',
                 prototype_regularizer=None,
                 prototype_constraint=None,  # That's similar to a normalization.
                 signal_output='signals',
                 linear_factor=0.5,  # linear factor  (None --> always add) prev_diss*(1-alpha) + alpha*curr_diss
                 signal_regularizer=None,
                 diss_regularizer=None,
                 **kwargs):

        if linear_factor is None or (isinstance(linear_factor, float) and 0 < linear_factor < 1):
            self.linear_factor = linear_factor
        else:
            raise TypeError("The linear factor for the dissimilarity value must be None or a float in the range of "
                            "(0,1). You provide: " + str(linear_factor))

        self.prototype_initializer = initializers.get(prototype_initializer)
        self.prototype_regularizer = regularizers.get(prototype_regularizer)
        self.prototype_constraint = constraints.get(prototype_constraint)

        self.prototypes = None

        self.signal_output = signal_output

        self.output_regularizers = [regularizers.get(signal_regularizer),
                                    regularizers.get(diss_regularizer)]

        # be sure to call this at the end
        super(PointPrototype, self).__init__(module_input=True,
                                             module_output=True,
                                             support_sparse_signal=True,
                                             support_full_signal=True,
                                             **self._del_module_args(**kwargs))

    def _build(self, input_shape):
        if not self.built:
            if input_shape[0][1] != self.capsule_number:
                raise ValueError('The capsule number provided by input_shape is not equal the self.capsule_number: '
                                 'input_shape[0][1]=' + str(input_shape[0][1]) + ' != ' +
                                 'self.capsule_number=' + str(self.capsule_number))
            if input_shape[1][1] != self.proto_number:
                raise ValueError('The prototype number provided by input_shape is not equal the self.proto_number: '
                                 'input_shape[1][1]=' + str(input_shape[1][1]) + ' != ' +
                                 'self.proto_number=' + str(self.proto_number))
            # the signal dimension is input_shape[0][3:]
            self.prototypes = self.add_weight(shape=(self.proto_number,) + tuple(input_shape[0][3:]),
                                              initializer=self.prototype_initializer,
                                              regularizer=self.prototype_regularizer,
                                              constraint=self.prototype_constraint,
                                              name='prototypes')

            self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                               InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _build_sparse(self, input_shape):
        if self.signal_output != 'signals':
            raise ValueError('If the signal is sparse, the signal_output must be `signals`.')

        # manipulate input_shape to call the full build method instead of a new implementation
        signal_shape = list(input_shape[0])
        signal_shape[1] = self.capsule_number

        self._build([tuple(signal_shape), input_shape[1]])

        self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                           InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        # signals_shape input: batch x capsule_number x channels x dim1 x dim2 x ... x dimN
        # after measuring the signal is (output): batch x proto_number x channels x dim1 x dim2 x ... x dimN
        signals = inputs[0]
        diss0 = inputs[1]

        if self.sparse_signal:
            signals, diss1 = self.distance_sparse(signals)
        else:
            signals, diss1 = self.distance(signals)

        with K.name_scope('dissimilarity_update'):
            if self.linear_factor is None:
                diss = diss0 + diss1
            else:
                diss = (1 - self.linear_factor) * diss0 + self.linear_factor * diss1

        return {0: signals, 1: diss}

    def _call_sparse(self, inputs, **kwargs):
        if self.signal_output != 'signals':
            raise ValueError('If the signal is sparse, the signal_output must be `signals`.')
        return self._call(inputs, **kwargs)

    def distance(self, signals):
        raise NotImplementedError

    def distance_sparse(self, signals):
        raise NotImplementedError

    def _compute_output_shape(self, input_shape):
        signal_shape = list(input_shape[0])
        diss_shape = list(input_shape[1])

        signal_shape[1] = self.proto_number

        return [tuple(signal_shape), tuple(diss_shape)]

    def _compute_output_shape_sparse(self, input_shape):
        if self.signal_output != 'signals':
            raise ValueError('If the signal is sparse, the signal_output must be `signals`.')
        return input_shape

    def get_config(self):
        config = {'prototype_initializer': initializers.serialize(self.prototype_initializer),
                  'prototype_regularizer': regularizers.serialize(self.prototype_regularizer),
                  'prototype_constraint': constraints.serialize(self.prototype_constraint),
                  'signal_output': self.signal_output,
                  'linear_factor': self.linear_factor,
                  'signal_regularizer': regularizers.serialize(self.output_regularizers[0]),
                  'diss_regularizer': regularizers.serialize(self.output_regularizers[1])}
        super_config = super(PointPrototype, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))

    def pre_training(self,
                     x_train,
                     y_train=None,  # class labels [0,..,n], if given assume class correspondence and init respectively
                     capsule_inputs_are_equal=False,  # use just the first capsule channels; ignored if sparse
                     batch_version=False,
                     **kmeans_params):  # params of kMeans or batch kMeans of sklearn

        # overwrite flag if sparse
        if self.sparse_signal:
            capsule_inputs_are_equal = True

        try:
            self._is_callable()
        except ValueError:
            raise ValueError("The module " + self.name + " is not proper initialized for pre-training. Be sure that you"
                             " assigned the module to a capsule and built it by calling the capsule.")

        if self.input_spec[0].shape[1:] != x_train.shape[1:]:
            raise ValueError("The shape of x_train must be equal to the assumed input shape. You provide: "
                             + str(x_train.shape) + "; we expect: " + str(self.input_spec[0].shape))

        if y_train is not None:
            # convert y_train to non categorical
            if y_train.ndim != 1:
                y_train = np.argmax(y_train, 1)

            # check if all classes are present
            if list(np.unique(y_train)) != list(range(self.capsule_number)):
                raise ValueError("We detected that not each class is represented in the training data. Be sure that "
                                 "for each class, at least one sample is given.")

            if y_train.shape[0] != x_train.shape[0]:
                raise ValueError("The number of samples in y_train and x_train must be equal. You provide: y_train="
                                 + str(y_train.shape[0]) + " is not equal x_train=" + str(x_train.shape[0]) + ".")

        centers, _, _ = _pre_train_prototypes(x_train,
                                              y_train,
                                              capsule_inputs_are_equal,
                                              batch_version,
                                              self._proto_distrib,
                                              **kmeans_params)


        # we apply constraint here, because pre-training is like an optimization step
        if self.prototype_constraint is not None:
            centers = K.eval(self.prototype_constraint(K.variable(centers)))

        self.set_weights([centers] + self.get_weights()[1:])


# definition of prototype which consist of a data point and a matrix
class PointMatrixPrototype(PointPrototype):
    def __init__(self,
                 prototype_initializer='TruncatedNormal',
                 prototype_regularizer=None,
                 prototype_constraint=None,  # That's similar to a normalization.
                 matrix_initializer='TruncatedNormal',
                 matrix_regularizer=None,
                 matrix_constraint=None,  # That's similar to a normalization.
                 matrix_scope='local',  # local, global or capsule_wise
                 projected_atom_shape=None,
                 signal_output='signals',
                 linear_factor=0.5,  # linear factor  (None --> always add) prev_diss*(1-alpha) + alpha*curr_diss
                 signal_regularizer=None,
                 diss_regularizer=None,
                 **kwargs):

        super(PointMatrixPrototype, self).__init__(prototype_initializer=prototype_initializer,
                                                   prototype_regularizer=prototype_regularizer,
                                                   prototype_constraint=prototype_constraint,
                                                   signal_output=signal_output,
                                                   linear_factor=linear_factor,
                                                   signal_regularizer=signal_regularizer,
                                                   diss_regularizer=diss_regularizer,
                                                   **kwargs)

        # scope: local, global, capsule_wise (defines the apply scope of the transformation)
        if matrix_scope in ('local', 'global', 'capsule_wise'):
            self.matrix_scope = matrix_scope
        else:
            raise ValueError("scope must be 'local', 'global' or 'capsule_wise': You provide: " + str(matrix_scope))

        if isinstance(projected_atom_shape, (list, tuple)):
            self._projected_atom_shape = tuple(projected_atom_shape)
        elif isinstance(projected_atom_shape, int):
            self._projected_atom_shape = (projected_atom_shape,)
        elif projected_atom_shape is None:
            self._projected_atom_shape = projected_atom_shape
        else:
            raise ValueError("projected_atom_shape must be list, tuple, int or None. You provide: " +
                             str(projected_atom_shape))
        self.projected_atom_shape = None

        self.matrix_initializer = initializers.get(matrix_initializer)
        self.matrix_regularizer = regularizers.get(matrix_regularizer)
        self.matrix_constraint = constraints.get(matrix_constraint)

        self.matrices = None

        self.input_output_shape_equal = ('signals', 'protos')

    def _build(self, input_shape):
        if not self.built:
            # compute shape of matrix_shape
            if self.matrix_scope == 'local':
                self._num_maps = (self.proto_number,)
            elif self.matrix_scope == 'global':
                self._num_maps = ()
            else:  # capsule_wise
                self._num_maps = (self.capsule_number,)

            super(PointMatrixPrototype, self)._build(input_shape)

            if self._projected_atom_shape is None:
                self.projected_atom_shape = self._compute_output_shape(input_shape)[0][3:]
            else:
                self.projected_atom_shape = self._projected_atom_shape

            matrix_shape = self._num_maps + (np.prod(np.array(input_shape[0][3:], dtype=int)),
                                             np.prod(np.array(self.projected_atom_shape, dtype=int)))

            # transposed matrix (we compute x^T * A^T instead of A * x, because the signal is always given in the form
            # x^T)
            self.matrices = self.add_weight(shape=matrix_shape,
                                            initializer=self.matrix_initializer,
                                            regularizer=self.matrix_regularizer,
                                            constraint=self.matrix_constraint,
                                            name='matrices')

    def _compute_output_shape(self, input_shape):
        signal_shape = list(input_shape[0])
        diss_shape = input_shape[1]

        signal_shape[1] = self.proto_number
        if self.signal_output in self.input_output_shape_equal:
            return [signal_shape, diss_shape]
        else:
            if not self.built:
                if self._projected_atom_shape is None:
                    projected_atom_shape = tuple(input_shape[0][3:])
                else:
                    projected_atom_shape = self._projected_atom_shape
                return [tuple(signal_shape[0:3]) + projected_atom_shape, diss_shape]
            else:
                return [tuple(signal_shape[0:3]) + self.projected_atom_shape, diss_shape]

    def get_config(self):
        config = {'matrix_initializer': initializers.serialize(self.matrix_initializer),
                  'matrix_regularizer': regularizers.serialize(self.matrix_regularizer),
                  'matrix_constraint': constraints.serialize(self.matrix_constraint),
                  'matrix_scope': self.matrix_scope,
                  'projected_atom_shape': self.projected_atom_shape}
        super_config = super(PointMatrixPrototype, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))

    def pre_training(self,
                     x_train,
                     y_train=None,  # class labels [0,..,n], if given assume class correspondence and init respectively
                     capsule_inputs_are_equal=False,  # use just the first capsule channels
                     batch_version=False,
                     **sk_params):
        # sk_params: {'kmeans_params': dict,
        #             'svd_params': dict}
        # kmeans_params ... params dict of kmeans_params or kmeans_params_batch; Note: in the case of a memory error the
        # method switches automatically to the batch version. In this case be sure that all parameters are accept by the
        # batch version.
        # svd_params ... params of svd_params
        # Todo: test that svd works correct (full rank and sparse rank)

        # overwrite flag if sparse
        if self.sparse_signal:
            capsule_inputs_are_equal = True

        if hasattr(sk_params, 'kmeans_params'):
            kmeans_params = sk_params['kmeans_params']
            if not isinstance(kmeans_params, dict):
                raise TypeError("The type of kmeans_params parameters in sk_params must be dict.")
            del sk_params['kmeans_params']
        else:
            kmeans_params = {}

        if hasattr(sk_params, 'svd_params'):
            svd_params = sk_params['svd_params']
            if not isinstance(svd_params, dict):
                raise TypeError("The type of the svd_params parameters in sk_params must be dict.")
            del sk_params['svd_params']
        else:
            svd_params = {}

        try:
            self._is_callable()
        except ValueError:
            raise ValueError("The module " + self.name + " is not proper initialized for pre-training. Be sure that you"
                             " assigned the module to a capsule and built it by calling the capsule.")

        if self.input_spec[0].shape[1:] != x_train.shape[1:]:
            raise ValueError("The shape of x_train must be equal to the assumed input shape. You provide: "
                             + str(x_train.shape) + "; we expect: " + str(self.input_spec[0].shape))

        if y_train is not None:
            # convert y_train to non categorical
            if y_train.ndim != 1:
                y_train = np.argmax(y_train, 1)

            # check if all classes are present
            if list(np.unique(y_train)) != list(range(self.capsule_number)):
                raise ValueError("We detected that not each class is represented in the training data. Be sure that "
                                 "for each class, at least one sample is given.")

            if y_train.shape[0] != x_train.shape[0]:
                raise ValueError("The number of samples in y_train and x_train must be equal. You provide: y_train="
                                 + str(y_train.shape[0]) + " is not equal x_train=" + str(x_train.shape[0]) + ".")

        centers, labels, clusters = _pre_train_prototypes(x_train,
                                                          y_train,
                                                          capsule_inputs_are_equal,
                                                          batch_version,
                                                          self._proto_distrib,
                                                          **kmeans_params)

        # we apply constraint here, because pre-training is like an optimization step
        if self.prototype_constraint is not None:
            centers = K.eval(self.prototype_constraint(K.variable(centers)))

        # transform clusters in accordance to the matrix scope
        if self.matrix_scope == 'global':
            clusters = [np.reshape(np.concatenate(clusters, 0), [-1, np.prod(self.input_spec[0].shape[3:], dtype=int)])]
        elif self.matrix_scope == 'local':
            clusters_ = []
            for i in range(len(clusters)):
                for l in range(max(labels[i])+1):
                    cluster_flatten = np.reshape(clusters[i], [-1, np.prod(self.input_spec[0].shape[3:], dtype=int)])
                    clusters_.append(cluster_flatten[labels[i] == l, :])
            clusters = clusters_
        else:
            clusters_ = []
            for i in range(len(clusters)):
                clusters_.append(np.reshape(clusters[i], [-1, np.prod(self.input_spec[0].shape[3:], dtype=int)]))
            clusters = clusters_

        # center the clusters
        for i in range(len(clusters)):
            clusters[i] = clusters[i] - np.mean(clusters[i], axis=0)

        if K.int_shape(self.matrices)[-1] > K.int_shape(self.matrices)[-2]:
            warnings.warn("Can't pre-train matrices if the projection shape (" + str(K.int_shape(self.matrices)[-1]) +
                          ") is higher than the input shape (" + str(K.int_shape(self.matrices)[-2]) + "). Skip " +
                          "pre-training of matrices and use the current parameters.")
            self.set_weights([centers, self.get_weights()[1]])
        else:
            n_components = np.prod(self.projected_atom_shape, dtype=int)
            matrices, valid = pre_training.svd(clusters, n_components=n_components, **svd_params)

            old_matrices = self.get_weights()[1]
            warning_sent = False
            for i in range(len(matrices)):
                if not valid[i]:
                    if not warning_sent:
                        warnings.warn("Some of the matrices were not be pre-tarined successfully. The old matrix "
                                      "instantiation is reused and the pre-trained matrix is skipped. One reason for "
                                      "that could be that your cluster doesn't contain enough points regarding your "
                                      "matrix dimension. If you are not pre-training on the whole dataset try to "
                                      "increase the number of the pre-training batch.")
                        warning_sent = True
                    matrices[i] = old_matrices[i]

            if self.matrix_scope == 'global':
                matrices = matrices[0]
            else:
                matrices = np.stack(matrices, axis=0)

            # we apply constraint here, because pre-training is like an optimization step
            if self.matrix_constraint is not None:
                matrices = K.eval(self.matrix_constraint(K.variable(matrices)))

            self.set_weights([centers, matrices] + self.get_weights()[2:])


class MinkowskiDistance(PointPrototype):
    def __init__(self,
                 order_p=2,  # could be np.inf
                 prototype_initializer='TruncatedNormal',
                 prototype_regularizer=None,
                 prototype_constraint=None,  # That's similar to a normalization.
                 signal_output='signals',  # what should be send to the output (signals, protos, params)
                 epsilon=K.epsilon(),  # used for stabilization of the sqrt
                 squared_dissimilarity=False,  # output the squared distance (without reciprocal power)
                 linear_factor=0.5,  # linear factor  (None --> always add) prev_diss*(1-alpha) + alpha*curr_diss
                 signal_regularizer=None,
                 diss_regularizer=None,
                 **kwargs):

        valid_signal_output = ('signals', 'protos')
        if signal_output not in valid_signal_output:
            raise ValueError("signal_output must be in " + str(valid_signal_output) + ". You provide: "
                             + str(signal_output))

        if isinstance(order_p, (int, float)) or order_p == np.inf:
            if order_p > 0:
                self.order_p = order_p
            else:
                raise ValueError("The order p of the Minkowski distance must be greater than 0. You provide: "
                                 + str(order_p))
        else:
            raise TypeError("order_p must be int float or numpy.inf. You provide: " + str(order_p))

        self.squared_dissimilarity = squared_dissimilarity
        self.epsilon = epsilon

        super(MinkowskiDistance, self).__init__(prototype_initializer=prototype_initializer,
                                                prototype_regularizer=prototype_regularizer,
                                                prototype_constraint=prototype_constraint,
                                                signal_output=signal_output,
                                                linear_factor=linear_factor,
                                                signal_regularizer=signal_regularizer,
                                                diss_regularizer=diss_regularizer,
                                                **kwargs)

    def distance(self, signals):
        # Todo: make test with and without sqrt by this simple maximum stabilization if it works think about more
        # complex implementations for stabilization (test convergence behavior with sqrt and w/o)

        # tile capsule regarding proto distribution
        if self.proto_number != self.capsule_number:
            with K.name_scope('signal_preprocessing'):
                # vector_shape for permute commands
                vector_shape = list(range(3, self.input_spec[0].ndim))
                signals = K.gather(K.permute_dimensions(signals, [1, 0, 2] + vector_shape),
                                   self._capsule_extension)
                signals = K.permute_dimensions(signals, [1, 0, 2] + vector_shape)

        diss = diss_funcs.minkowski_distance(signals=signals,
                                             protos=self.prototypes,
                                             order_p=self.order_p,
                                             squared=self.squared_dissimilarity,
                                             epsilon=self.epsilon)

        with K.name_scope('get_signals'):
            if self.signal_output == 'protos':
                signal_shape = mixed_shape(signals)
                signals = K.tile(K.expand_dims(K.expand_dims(self.prototypes, 1), 0),
                                 [K.shape(signals)[0], 1, signal_shape[2]]
                                 + list(np.ones((self.input_spec[0].ndim - 3,), dtype=int)))
            else:
                signals = signals

        return signals, diss

    def distance_sparse(self, signals):
        diss = diss_funcs.minkowski_distance(signals=signals,
                                             protos=self.prototypes,
                                             order_p=self.order_p,
                                             squared=self.squared_dissimilarity,
                                             epsilon=self.epsilon)

        return signals, diss

    def get_config(self):
        config = {'order_p': self.order_p,
                  'squared_dissimilarity': self.squared_dissimilarity,
                  'epsilon': self.epsilon}
        super_config = super(MinkowskiDistance, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class TangentDistance(PointMatrixPrototype):
    def __init__(self,
                 projected_atom_shape,
                 prototype_initializer='TruncatedNormal',
                 prototype_regularizer=None,
                 prototype_constraint=None,
                 matrix_initializer='TruncatedNormal',
                 matrix_regularizer=None,
                 matrix_scope='local',  # local, global or capsule_wise
                 signal_output='signals',  # what should be send to the output (signals, protos, projected_signals, projected_protos )
                 epsilon=K.epsilon(),
                 squared_dissimilarity=False,
                 linear_factor=0.5,  # linear factor  (None --> always add) prev_diss*(1-alpha) + alpha*curr_diss
                 signal_regularizer=None,
                 diss_regularizer=None,
                 **kwargs):

        valid_signal_output = ('signals', 'projected_signals', 'parameterized_signals')
        if signal_output not in valid_signal_output:
            raise ValueError("signal_output must be in " + str(valid_signal_output) + ". You provide: "
                             + str(signal_output))

        self.squared_dissimilarity = squared_dissimilarity
        self.epsilon = epsilon

        if projected_atom_shape is None:
            raise ValueError("projected_atom_shape must be unequal None.")

        if 'matrix_constraint' in kwargs:
            del kwargs['matrix_constraint']

        super(TangentDistance, self).__init__(prototype_initializer=prototype_initializer,
                                              prototype_regularizer=prototype_regularizer,
                                              prototype_constraint=prototype_constraint,
                                              matrix_initializer=matrix_initializer,
                                              matrix_regularizer=matrix_regularizer,
                                              matrix_constraint='orthogonalization',
                                              matrix_scope=matrix_scope,
                                              projected_atom_shape=projected_atom_shape,
                                              signal_output=signal_output,
                                              linear_factor=linear_factor,
                                              signal_regularizer=signal_regularizer,
                                              diss_regularizer=diss_regularizer,
                                              **kwargs)

        self.input_output_shape_equal = ('signals', 'projected_signals')

    def _build(self, input_shape):
        super(TangentDistance, self)._build(input_shape)

        if np.prod(input_shape[0][3:]) <= np.prod(self.projected_atom_shape):
            raise ValueError("The dimension of the projected shape must be lower than input_shape. You "
                             "provide: np.prod(signal_shape[3:])=" + str(np.prod(input_shape[0][3:])) + " < " +
                             "np.prod(self.projected_atom_shape)=" + str(np.prod(self.projected_atom_shape)))

    def distance(self, signals):
        # tile capsule regarding proto distribution
        if self.proto_number != self.capsule_number:
            with K.name_scope('signal_preprocessing'):
                # vector_shape for permute commands
                vector_axes = list(range(3, self.input_spec[0].ndim))
                signals = K.gather(K.permute_dimensions(signals, [1, 0, 2] + vector_axes),
                                   self._capsule_extension)
                signals = K.permute_dimensions(signals, [1, 0, 2] + vector_axes)

        if self.matrix_scope == 'capsule_wise':
            with K.name_scope('subspace_preprocessing'):
                matrices = K.gather(self.matrices, self._capsule_extension)
        else:
            matrices = self.matrices

        diss = diss_funcs.tangent_distance(signals=signals,
                                           protos=self.prototypes,
                                           subspaces=matrices,
                                           squared=self.squared_dissimilarity,
                                           epsilon=self.epsilon)

        with K.name_scope('get_signals'):
            if self.signal_output == 'signals':
                signals = signals

            elif self.signal_output == 'projected_signals':
                signals = self._get_projected_signals(signals, matrices)

            else:  # 'parameterized_signals'
                signals = self._get_parameterized_signals(signals, matrices)

        return signals, diss

    def distance_sparse(self, signals):
        if self.matrix_scope == 'capsule_wise':
            with K.name_scope('subspace_preprocessing'):
                matrices = K.gather(self.matrices, self._capsule_extension)
        else:
            matrices = self.matrices

        diss = diss_funcs.tangent_distance(signals=signals,
                                           protos=self.prototypes,
                                           subspaces=matrices,
                                           squared=self.squared_dissimilarity,
                                           epsilon=self.epsilon)

        return signals, diss

    def get_projected_signals(self, signals):
        if self.matrix_scope == 'capsule_wise':
            with K.name_scope('matrix_preprocessing'):
                matrices = K.gather(self.matrices, self._capsule_extension)
        else:
            matrices = self.matrices

        return self._get_projected_signals(signals, matrices)

    # affine_projection: w + W * W.T * (v - w)
    def _get_projected_signals(self, signals, matrices):
        # signal must be of shape (batch x protos x channels x dim1 x ... x dimN
        with K.name_scope('get_projected_signals'):
            signal_shape = mixed_shape(signals)
            atom_axes = list(range(3, len(signal_shape)))

            signals = K.permute_dimensions(signals, [0, 2, 1] + atom_axes)
            diff = signals - self.prototypes

            if self.matrix_scope == 'global':
                with K.name_scope('projector'):
                    projector = K.dot(matrices, K.transpose(matrices))

                with K.name_scope('tangentspace_projections'):
                    diff = K.reshape(diff, (signal_shape[0] * signal_shape[2], signal_shape[1], -1))
                    projected_diff = K.dot(diff, projector)
                    projected_diff = K.reshape(projected_diff,
                                               (signal_shape[0], signal_shape[2], signal_shape[1]) + signal_shape[3:])

                    matching_protos = self.prototypes + projected_diff
                    matching_protos = K.permute_dimensions(matching_protos, [0, 2, 1] + atom_axes)

            else:  # local or capsule_wise
                with K.name_scope('projector'):
                    projector = K.batch_dot(matrices, matrices, [2, 2])

                with K.name_scope('tangentspace_projections'):
                    diff = K.reshape(diff, (signal_shape[0] * signal_shape[2], signal_shape[1], -1))
                    diff = K.permute_dimensions(diff, [1, 0, 2])
                    projected_diff = K.batch_dot(diff, projector)
                    projected_diff = K.permute_dimensions(projected_diff, [1, 0, 2])
                    projected_diff = K.reshape(projected_diff,
                                               (signal_shape[0], signal_shape[2], signal_shape[1]) + signal_shape[3:])

                    matching_protos = self.prototypes + projected_diff
                    matching_protos = K.permute_dimensions(matching_protos, [0, 2, 1] + atom_axes)

            return matching_protos

    def get_parameterized_signals(self, signals):
        if self.matrix_scope == 'capsule_wise':
            with K.name_scope('matrix_preprocessing'):
                matrices = K.gather(self.matrices, self._capsule_extension)
        else:
            matrices = self.matrices

        return self._get_parameterized_signals(signals, matrices)

    # params: W.T * (v - w)
    def _get_parameterized_signals(self, signals, matrices):
        # signal must be of shape (batch x protos x channels x dim1 x ... x dimN
        with K.name_scope('get_projected_signals'):
            signal_shape = mixed_shape(signals)
            atom_axes = list(range(3, len(signal_shape)))

            signals = K.permute_dimensions(signals, [0, 2, 1] + atom_axes)
            diff = signals - self.prototypes

            if self.matrix_scope == 'global':
                with K.name_scope('tangentspace_parameters'):
                    diff = K.reshape(diff, (signal_shape[0] * signal_shape[2], signal_shape[1], -1))
                    params = K.dot(diff, matrices)

                    params = K.reshape(params,
                                       (signal_shape[0], signal_shape[2], signal_shape[1]) + self.projected_atom_shape)
                    params = K.permute_dimensions(params, [0, 2, 1] + atom_axes)

            else:  # local or capsule_wise
                with K.name_scope('tangentspace_parameters'):
                    diff = K.reshape(diff, (signal_shape[0] * signal_shape[2], signal_shape[1], -1))
                    diff = K.permute_dimensions(diff, [1, 0, 2])
                    params = K.batch_dot(diff, matrices)

                    params = K.reshape(params,
                                       (signal_shape[1], signal_shape[0], signal_shape[2]) + self.projected_atom_shape)
                    params = K.permute_dimensions(params, [1, 0, 2] + atom_axes)

            return params

    def get_config(self):
        config = {'squared_dissimilarity': self.squared_dissimilarity,
                  'epsilon': self.epsilon}
        super_config = super(TangentDistance, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


def _pre_train_prototypes(x_train,
                          y_train,  # non-categorical
                          capsule_inputs_are_equal,
                          batch_version,
                          proto_distrib,
                          **kmeans_params):
    # signals_shape: samples x capsule_number x channels x dim1 x dim2 x ... x dimN

    signal_shape = x_train.shape[3:]
    proto_number = sum([len(x) for x in proto_distrib])

    x_trains = []

    # all the cases for k-means computation
    if y_train is None and capsule_inputs_are_equal:
        # samples x dim1 x dim2 x ... x dimN
        x = x_train[:, 0, :]
        n_clusters = proto_number
        x = np.reshape(x, (-1, np.prod(signal_shape)))
        x_trains.append(x)
    else:
        n_clusters = []

        for i, p in enumerate(proto_distrib):
            n_clusters.append(len(p))

            if y_train is None and not capsule_inputs_are_equal:
                # capsule_number x samples x dim1 x dim2 x ... x dimN
                x = x_train[:, i, :]

            elif y_train is not None and capsule_inputs_are_equal:
                # samples x channels x dim1 x dim2 x ... x dimN
                x = x_train[y_train == i, 0, :]

            else:
                # samples x capsule_number x channels x dim1 x dim2 x ... x dimN
                x = x_train[y_train == i, i, :]

            x = np.reshape(x, (-1, np.prod(signal_shape)))
            x_trains.append(x)

    centers, labels = pre_training.kmeans(x_trains, n_clusters, batch_version, **kmeans_params)

    centers = np.concatenate(centers, 0)
    # reshape back to real shape
    centers = np.reshape(centers, (proto_number, ) + signal_shape)

    return centers, labels, x_trains
