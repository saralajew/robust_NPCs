# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Lambda, Reshape
from keras.models import Model
from keras import backend as K

from LVQ import Capsule
from LVQ.capsule import InputModule
from LVQ.modules.measuring import MinkowskiDistance, TangentDistance
from LVQ.modules.routing import SqueezeRouting
from LVQ.modules.competition import NearestCompetition
from LVQ.probability_transformations import NegSoftmax

import numpy as np


def get_model(model,
              input_shape,
              n_classes,
              number_prototypes=1,
              p_norm=2,
              batch_size=128,
              negated_dissimilarities=False,
              weights_provided=False,
              random_initialization=False,
              data=None,
              number_tangents=12,
              nearest=True):

    if model == 'glvq':
        train_model = glvq(input_shape,
                           n_classes,
                           number_prototypes,
                           p_norm,
                           batch_size,
                           negated_dissimilarities,
                           weights_provided,
                           random_initialization,
                           data,
                           nearest)

    elif model == 'gtlvq':
        train_model = gtlvq(input_shape,
                            n_classes,
                            number_prototypes,
                            batch_size,
                            negated_dissimilarities,
                            weights_provided,
                            data,
                            number_tangents)

    elif model == 'rslvq':
        train_model = rslvq(input_shape,
                            n_classes,
                            number_prototypes,
                            batch_size,
                            weights_provided,
                            random_initialization,
                            data)

    else:
        raise ValueError('Model type "{}" not implemented.'.format(model))

    return train_model


def glvq(input_shape,
         n_classes,
         number_prototypes=1,
         p_norm=2,
         batch_size=128,
         negated_dissimilarities=False,
         weights_provided=False,
         random_initialization=False,
         data=None,
         nearest=True):

    # we use the squared version if possible
    if p_norm not in (1, np.inf):
        squared = True
    else:
        squared = False

    inputs = Input(shape=input_shape)
    diss = MinkowskiDistance(linear_factor=None,
                             squared_dissimilarity=squared,
                             signal_output='signals',
                             order_p=p_norm)

    caps = Capsule(prototype_distribution=(number_prototypes, n_classes))
    caps.add(InputModule(signal_shape=(-1, np.prod(input_shape)),
                         trainable=False,
                         init_diss_initializer='zeros'))
    caps.add(diss)
    if nearest:
        caps.add(SqueezeRouting())
        caps.add(NearestCompetition())

    output = caps(inputs)[1]

    if not weights_provided:
        if data is None:
            raise ValueError('Provide weights for pre-training.')
        else:
            x_train, y_train = data

        if not random_initialization:
            # pre-train the model over 10000 random digits
            idx = np.random.randint(0, len(x_train) - 1,
                                    (min(10000, len(x_train)),))
            pre_train_model = Model(inputs=inputs, outputs=diss.input[0])
            diss_input = pre_train_model.predict(x_train[idx, :],
                                                 batch_size=batch_size)
            diss.pre_training(diss_input, y_train[idx],
                              capsule_inputs_are_equal=True)

        else:
            # random selection of initial prototypes
            y_train = np.argmax(y_train, 1)
            prototypes = []
            for i in range(n_classes):
                idx = np.where(y_train == i)[0]
                random_int = np.random.randint(0, len(idx), number_prototypes)
                prototypes.append(x_train[idx[random_int]])
            prototypes = np.concatenate(prototypes, 0)
            prototypes = np.reshape(prototypes,
                                    (n_classes * number_prototypes, -1))
            diss.set_weights([prototypes])

    if negated_dissimilarities:
        output = Lambda(lambda x: -x)(output)

    model = Model(inputs, output)

    return model


def rslvq(input_shape,
          n_classes,
          number_prototypes=1,
          batch_size=128,
          weights_provided=False,
          random_initialization=False,
          data=None):

    inputs = Input(shape=input_shape)
    diss = MinkowskiDistance(linear_factor=None,
                             squared_dissimilarity=True,
                             signal_output='signals',
                             order_p=2)

    caps = Capsule(prototype_distribution=(number_prototypes, n_classes))
    caps.add(InputModule(signal_shape=(-1, np.prod(input_shape)),
                         trainable=False,
                         init_diss_initializer='zeros'))
    caps.add(diss)
    caps.add(SqueezeRouting())

    # make RSLVQ
    caps.add(Lambda(lambda x: NegSoftmax()(x)), scope_keys=1)
    caps.add(Reshape((number_prototypes, n_classes)), scope_keys=1)
    caps.add(Lambda(lambda x: K.sum(x, axis=1)), scope_keys=1)

    output = caps(inputs)[1]

    if not weights_provided:
        if data is None:
            raise ValueError('Provide weights for pre-training.')
        else:
            x_train, y_train = data

        if not random_initialization:
            # # pre-train the model over 10000 random digits
            idx = np.random.randint(0, len(x_train) - 1,
                                    (min(10000, len(x_train)),))
            pre_train_model = Model(inputs=inputs, outputs=diss.input[0])
            diss_input = pre_train_model.predict(x_train[idx, :],
                                                 batch_size=batch_size)
            diss.pre_training(diss_input, y_train[idx],
                              capsule_inputs_are_equal=True)

        else:
            # random selection of initial prototypes
            y_train = np.argmax(y_train, 1)
            prototypes = []
            for i in range(n_classes):
                idx = np.where(y_train == i)[0]
                random_int = np.random.randint(0, len(idx), number_prototypes)
                prototypes.append(x_train[idx[random_int]])
            prototypes = np.concatenate(prototypes, 0)
            prototypes = np.reshape(prototypes,
                                    (n_classes * number_prototypes, -1))
            diss.set_weights([prototypes])

    model = Model(inputs, output)

    return model


def gtlvq(input_shape,
          n_classes,
          number_prototypes=1,
          batch_size=128,
          negated_dissimilarities=False,
          weights_provided=False,
          data=None,
          number_tangents=12):
    inputs = Input(shape=input_shape)
    diss = TangentDistance(linear_factor=None,
                           squared_dissimilarity=True,
                           signal_output='signals',
                           projected_atom_shape=number_tangents)

    caps = Capsule(prototype_distribution=(number_prototypes, n_classes))
    caps.add(InputModule(signal_shape=(-1, np.prod(input_shape)),
                         trainable=False,
                         init_diss_initializer='zeros'))
    caps.add(diss)
    caps.add(SqueezeRouting())
    caps.add(NearestCompetition())

    output = caps(inputs)[1]

    if not weights_provided:
        if data is None:
            raise ValueError('Provide weights for pre-training.')
        else:
            x_train, y_train = data

        # # pre-train the model over 10000 random digits
        idx = np.random.randint(0, len(x_train) - 1,
                                (min(10000, len(x_train)),))
        pre_train_model = Model(inputs=inputs, outputs=diss.input[0])
        diss_input = pre_train_model.predict(x_train[idx, :],
                                             batch_size=batch_size)
        diss.pre_training(diss_input, y_train[idx],
                          capsule_inputs_are_equal=True)

    if negated_dissimilarities:
        output = Lambda(lambda x: -x)(output)

    model = Model(inputs, output)

    return model
