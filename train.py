# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import os
import json

from keras import callbacks
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from functions.losses import GlvqLoss, TripletReluLoss
from functions.metrics import acc, Certificate
from functions.train_utils import evaluate, get_data, get_save_dir
from functions.models import get_model


def set_replicate_params(args_):
    args_.random_initialization = False

    # MNIST and CIFAR-10
    if args_.dataset in {'mnist', 'cifar10'}:
        args_.batch_size = 128
        args_.lr = 0.001
        args_.epochs = 1000
        args_.negated_dissimilarities = False
        args_.augmentation = True
        args_.early_stopping = False
        args_.store_checkpoints = True

        if args_.dataset == "mnist":
            args_.certificates_epsilon = '1.58, 0.3'
        elif args_.dataset == "cifar10":
            args_.certificates_epsilon = '0.1411764705882353, ' \
                                         '0.03137254901960784'

        if args_.model == "glvq":
            args_.glvq_loss = True
            if args_.dataset == "mnist":
                if args_.p_norm == "inf":
                    args_.epsilon_live_certificate = 0.3
                    args_.number_prototypes = 128
                    if args_.eval is True and args_.weights is None:
                        args_.weights = "weight_files/GLVQ/mnist/" \
                                        "linf_trained/glvq_loss.h5"
                elif args_.p_norm == "2":
                    args_.epsilon_live_certificate = 1.58
                    args_.number_prototypes = 256
                    if args_.eval is True and args_.weights is None:
                        args_.weights = "weight_files/GLVQ/mnist/" \
                                        "l2_trained/trained_model.h5"
            elif args_.dataset == "cifar10":
                if args_.p_norm == "inf":
                    args_.epsilon_live_certificate = 8/255
                    args_.number_prototypes = 64
                    if args_.eval is True and args_.weights is None:
                        args_.weights = "weight_files/GLVQ/cifar10/" \
                                        "linf_trained/trained_model.h5"
                elif args_.p_norm == "2":
                    args_.epsilon_live_certificate = 36/255
                    args_.number_prototypes = 128
                    if args_.eval is True and args_.weights is None:
                        args_.weights = "weight_files/GLVQ/cifar10/" \
                                        "l2_trained/trained_model.h5"

        if args_.model == "rslvq":
            args_.p_norm = "inf"
            if args_.dataset == "mnist":
                args_.epsilon_live_certificate = 0.3
                args_.number_prototypes = 128
                if args_.eval is True and args_.weights is None:
                    args_.weights = "weight_files/RSLVQ/mnist/trained_model.h5"
            elif args_.dataset == "cifar10":
                args_.epsilon_live_certificate = 8/255
                args_.number_prototypes = 128
                if args_.eval is True and args_.weights is None:
                    args_.weights = "weight_files/RSLVQ/cifar10/" \
                                    "trained_model.h5"

        if args_.model == "gtlvq":
            args_.p_norm = "2"
            if args_.dataset == "mnist":
                args_.epsilon_live_certificate = 1.58
                args_.relu_epsilon = 1.58
                args_.glvq_loss = False
                args_.number_prototypes = 10
                args_.number_tangents = 12
                if args_.eval is True and args_.weights is None:
                    args_.weights = "weight_files/GTLVQ/mnist/trained_model.h5"
            elif args_.dataset == "cifar10":
                args_.epsilon_live_certificate = 36/255
                args_.glvq_loss = True
                args_.number_prototypes = 1
                args_.number_tangents = 100
                if args_.eval is True and args_.weights is None:
                    args_.weights = "weight_files/GTLVQ/cifar10/" \
                                    "trained_model.h5"

    # Tabular data
    if args_.dataset in {'breast_cancer', 'diabetes', 'cod_rna'}:
        args_.epochs = 1000
        args_.p_norm = "inf"
        args_.model = 'glvq'
        args_.epochs = 1000
        args_.negated_dissimilarities = False
        args_.augmentation = False
        args_.early_stopping = False
        args_.store_checkpoints = True

        if args_.dataset == 'breast_cancer':
            args_.lr = 0.005
            args_.batch_size = 8
            args_.epsilon_live_certificate = 0.3
            args_.certificates_epsilon = '0.3, 0.3'
            args_.relu_epsilon = 0.45
            args_.glvq_loss = False
            args_.number_prototypes = 7
            if args_.eval is True and args_.weights is None:
                args_.weights = "weight_files/GLVQ/tabular_data/" \
                                "breast_cancer.h5"
        elif args_.dataset == 'diabetes':
            args_.lr = 0.0002
            args_.batch_size = 64
            args_.epsilon_live_certificate = 0.05
            args_.certificates_epsilon = '0.05, 0.05'
            args_.glvq_loss = True
            args_.number_prototypes = 4
            if args_.eval is True and args_.weights is None:
                args_.weights = "weight_files/GLVQ/tabular_data/" \
                                "diabetes.h5"
        elif args_.dataset == 'cod_rna':
            args_.lr = 0.01
            args_.batch_size = 256
            args_.epsilon_live_certificate = 0.025
            args_.certificates_epsilon = '0.025, 0.025'
            args_.relu_epsilon = 0.05
            args_.glvq_loss = False
            args_.number_prototypes = 8
            if args_.eval is True and args_.weights is None:
                args_.weights = "weight_files/GLVQ/tabular_data/" \
                                "cod_rna.h5"

    return args_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", default=None,
                        help="Load h5 model trained weights")
    parser.add_argument('-s', '--save_dir', default=None,
                        help='Output directory.')
    parser.add_argument('--gpu', default=-1, type=int,
                        help='Available GPU identifier(s) for processing.')
    parser.add_argument('--number_prototypes', default=1, type=int,
                        help='Number of prototypes per class.')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate.')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--glvq_loss', action='store_true',
                        help='Use GLVQ loss.')
    parser.add_argument('--relu_epsilon', default=0.3, type=float,
                        help='Epsilon of relu loss.')
    parser.add_argument('--certificates_epsilon', default='1.58, 0.3',
                        type=str,
                        help='Robustness certificates for L2, and Linf in'
                             'the form "<L2>, <Linf>".')
    parser.add_argument('--negated_dissimilarities', action='store_true',
                        help='Important for robustness evaluation in order '
                             'to produce a network output where the winner '
                             'is determined by the maximum dissimilarity.')
    parser.add_argument('--augmentation', action='store_true',
                        help='Apply augmentation.')
    parser.add_argument('--p_norm', default='2', type=str,
                        help='Order p of the p-norm. Can be "inf".')
    parser.add_argument('--eval', action='store_true',
                        help='Skip training and only evaluate.')
    parser.add_argument('--epsilon_live_certificate', default=0.3, type=float,
                        help='Epsilon of the certificate that is computed '
                             'during training.')
    parser.add_argument('--random_initialization', action='store_true',
                        help='Determines the initialization by random '
                             'samples instead of k-means.')
    parser.add_argument('--dataset', default='mnist', type=str.lower,
                        help='Dataset that should be used (mnist, cifar10, '
                             'breast_cancer, diabetes, cod_rna).')
    parser.add_argument('-m', '--model', type=str.lower, default='glvq',
                        help='Specify the model: GLVQ, RSLVQ, or GTLVQ.')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enables early stopping during training.')
    parser.add_argument('--store_checkpoints', action='store_true',
                        help='Enables model checkpoints during training.')
    parser.add_argument('--number_tangents', default=12, type=int,
                        help='Number of tangents')
    parser.add_argument("--replicate", action='store_true',
                        help="If set, the exact plot from the paper will be "
                             "replicated.")
    args = parser.parse_args()

    if not args.gpu == -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.replicate:
        print("Replicating")
        args = set_replicate_params(args)

    (x_train, y_train), (x_test, y_test) = get_data(args.dataset)
    input_shape = x_test.shape[1:]

    # auto create of save_dir
    if args.save_dir is None:
        args.save_dir = get_save_dir(args)
    print('\nFiles are saved to: {}\n'.format(args.save_dir))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # store arguments
    with open(args.save_dir + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # convert certificates
    args.certificates_epsilon = [float(s) for s in
                                 args.certificates_epsilon.split(',')]
    if len(args.certificates_epsilon) != 2:
        raise ValueError('args.certificates_epsilon should consist of two '
                         'entries.')

    # get p-norm
    if args.p_norm == 'inf':
        args.p_norm = np.inf
    else:
        args.p_norm = float(args.p_norm)

    # get train_model
    train_model = get_model(
        model=args.model,
        input_shape=input_shape,
        n_classes=y_train.shape[1],
        number_prototypes=args.number_prototypes,
        p_norm=args.p_norm,
        batch_size=args.batch_size,
        negated_dissimilarities=args.negated_dissimilarities,
        weights_provided=False if args.weights is None else True,
        random_initialization=args.random_initialization,
        data=(x_train, y_train),
        number_tangents=args.number_tangents)

    train_model.summary(line_length=200, positions=[.33, .6, .67, 1.])

    if args.weights:
        train_model.load_weights(args.weights)

    if args.glvq_loss:
        train_loss = GlvqLoss()
    else:
        train_loss = TripletReluLoss(args.p_norm,
                                     eps=args.relu_epsilon)

    if args.model == 'rslvq':
        train_loss = 'categorical_crossentropy'
        metrics = ['categorical_accuracy']
    else:
        metrics = [acc,
                   Certificate(args.p_norm, args.epsilon_live_certificate)]

    train_model.compile(
        optimizer=Adam(lr=args.lr),
        loss=train_loss,
    metrics = metrics)

    if not args.eval:
        def train_generator(x, y, batch_size):
            train_datagen = ImageDataGenerator(width_shift_range=2,
                                               height_shift_range=2,
                                               rotation_range=15)

            generator = train_datagen.flow(x, y, batch_size=batch_size)

            while True:
                batch_x, batch_y = generator.next()
                yield batch_x, batch_y

        # Callbacks
        csv_logger = callbacks.CSVLogger(args.save_dir + '/log.csv')
        lr_reduce = callbacks.ReduceLROnPlateau(factor=0.9, monitor='val_loss',
                                                mode='min', verbose=1,
                                                patience=10)
        cb = [csv_logger, lr_reduce]

        if args.store_checkpoints:
            cb.append(callbacks.ModelCheckpoint(args.save_dir +
                                                '/weights-{epoch:02d}.h5',
                                                save_best_only=True,
                                                save_weights_only=True,
                                                verbose=1,
                                                monitor='val_certificate',
                                                mode='max'))
        if args.early_stopping:
            cb.append(callbacks.EarlyStopping(monitor='val_certificate',
                                              patience=20,
                                              verbose=0, mode='max'))

        if args.augmentation and args.dataset in {'mnist', 'cifar10'}:
            train_model.fit_generator(
                generator=train_generator(x_train, y_train, args.batch_size),
                steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                epochs=args.epochs,
                validation_data=[x_test, y_test],
                callbacks=cb,
                max_queue_size=40,
                workers=3,
                use_multiprocessing=True,
                verbose=1)
        else:
            train_model.fit(
                x_train, y_train,
                batch_size=args.batch_size,
                epochs=args.epochs,
                callbacks=cb,
                validation_data=[x_test, y_test],
                verbose=2,
                class_weight=np.argmax(y_train, -1))

        train_model.save_weights(args.save_dir + '/trained_model.h5')

    evaluate(train_model, (x_train, y_train), (x_test, y_test), args)
