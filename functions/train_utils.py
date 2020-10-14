# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.datasets import mnist, cifar10
from keras.utils import to_categorical

import time

from .metrics import calculate_certificates
from data.data import breast_cancer, diabetes, cod_rna


def evaluate(model, train_data, test_data, args):
    (x_train, y_train) = train_data
    (x_test, y_test) = test_data

    with open(args.save_dir + '/results.txt', 'w') as f:
        def print_and_write(text):
            print(text)
            f.write(text + '\n')

        print_and_write('\ntraining results:')
        result = model.evaluate(x_train, y_train,
                                batch_size=args.batch_size)
        print_and_write('loss: {}   accuracy: {}'.format(result[0], result[1]))

        print_and_write('\ntest results:')
        result = model.evaluate(x_test, y_test,
                                batch_size=args.batch_size)
        print_and_write('loss: {}   accuracy: {}'.format(result[0], result[1]))

        certificates = calculate_certificates(model,
                                              (x_test, y_test),
                                              args.p_norm,
                                              args.certificates_epsilon)

        print_and_write('\ncertified L2 robust-acc (eps={}) is {}'.format(
            args.certificates_epsilon[0], certificates[0]))
        print_and_write('certified Linf robust-acc (eps={}) is {}'.format(
            args.certificates_epsilon[1], certificates[1]))


def get_data(dataset):
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
        y_train = to_categorical(y_train.astype('float32'))
        y_test = to_categorical(y_test.astype('float32'))

    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
        x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
        y_train = to_categorical(y_train.astype('float32'))
        y_test = to_categorical(y_test.astype('float32'))

    elif dataset == 'breast_cancer':
        x_train, y_train, x_test, y_test, _ = breast_cancer()
        y_train = to_categorical((y_train.astype('float32') + 1) / 2, 2)
        y_test = to_categorical((y_test.astype('float32') + 1) / 2, 2)

    elif dataset == 'diabetes':
        x_train, y_train, x_test, y_test, _ = diabetes()
        y_train = to_categorical((y_train.astype('float32') + 1) / 2, 2)
        y_test = to_categorical((y_test.astype('float32') + 1) / 2, 2)

    elif dataset == 'cod_rna':
        x_train, y_train, x_test, y_test, _ = cod_rna()
        y_train = to_categorical((y_train.astype('float32') + 1) / 2, 2)
        y_test = to_categorical((y_test.astype('float32') + 1) / 2, 2)

    else:
        raise ValueError('Dataset type "{}" not implemented'.format(dataset))

    return (x_train, y_train), (x_test, y_test)


def get_save_dir(args):
    if args.model == 'rslvq':
        save_dir = ('{}_RSLVQ_proto_number_{}_'
                    'augmentation_{}_timestamp_{}').format(
            args.dataset,
            args.number_prototypes,
            'true' if args.augmentation else 'false',
            int(time.time())  # back conversion time.ctime(<timestamp>)
        )

    elif args.model == 'glvq':
        save_dir = ('{}_GLVQ_L{}_eps_{}_proto_number_{}_'
                    'augmentation_{}_loss_{}_timestamp_{}').format(
            args.dataset,
            args.p_norm,
            args.relu_epsilon,
            args.number_prototypes,
            'true' if args.augmentation else 'false',
            'glvq' if args.glvq_loss else 'relu',
            int(time.time())  # back conversion time.ctime(<timestamp>)
        )

    elif args.model == 'gtlvq':
        save_dir = ('{}_Cifar10_proto_number_{}_tangents_{}_'
                    'augmentation_{}_loss_{}_timestamp_{}').format(
            args.dataset,
            args.number_prototypes,
            args.number_tangents,
            'true' if args.augmentation else 'false',
            'glvq' if args.glvq_loss else 'relu',
            int(time.time())  # back conversion time.ctime(<timestamp>)
        )
    else:
        raise ValueError('Model type "{}" not implemented.'.format(args.model))

    return save_dir
