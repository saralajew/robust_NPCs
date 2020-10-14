# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from functions import get_data, get_model, calculate_certificates
from functions.empirical_evaluation import empirical_robustness

import numpy as np


def to_p_norm(model_norm):
    if model_norm == "inf":
        return np.inf
    elif model_norm == "2":
        return 2
    else:
        raise NotImplementedError("Model norm either has to be 'inf' or '2'")


def check_default_attack_params(args_):
    """
    This function is responsible for setting the default parameters of the
    adversarial attacks if no values are provided. A full description of the
    default parameters can be found in the supplementary of our paper.
    """
    # Model
    if args_.replicate:
        if args_.model == "glvq":
            if args_.dataset == "mnist":
                if args_.model_norm == "inf":
                    args_.eval_norm = "inf"
                    args_.weights_path = "weight_files/GLVQ/mnist/" \
                                         "linf_trained/glvq_loss.h5"
                    args_.prototypes = 128
                elif args_.model_norm == "2":
                    args_.eval_norm = "2"
                    args_.weights_path = \
                        "weight_files/GLVQ/mnist/l2_trained/trained_model.h5"
                    args_.prototypes = 256
            elif args_.dataset == "cifar10":
                if args_.model_norm == "inf":
                    args_.eval_norm = "inf"
                    args_.weights_path = "weight_files/GLVQ/cifar10/" \
                                         "linf_trained/trained_model.h5"
                    args_.prototypes = 64
                elif args_.model_norm == "2":
                    args_.eval_norm = "2"
                    args_.weights_path = "weight_files/GLVQ/cifar10/" \
                                         "l2_trained/trained_model.h5"
                    args_.prototypes = 128

        if args_.model == "rslvq":
            args_.model_norm = "inf"
            args_.eval_norm = "inf"
            if args_.dataset == "mnist":
                args_.weights_path = "weight_files/RSLVQ/mnist/" \
                                     "trained_model.h5"
                args_.prototypes = 128
            elif args_.dataset == "cifar10":
                args_.weights_path = "weight_files/RSLVQ/cifar10/" \
                                     "trained_model.h5"
                args_.prototypes = 128

        if args_.model == "gtlvq":
            args_.model_norm = "2"
            args_.eval_norm = "2"
            if args_.dataset == "mnist":
                args_.weights_path = "weight_files/GTLVQ/mnist/" \
                                     "trained_model.h5"
                args_.prototypes = 10
                args_.tangents = 12
            elif args_.dataset == "cifar10":
                args_.weights_path = "weight_files/GTLVQ/cifar10/" \
                                     "trained_model.h5"
                args_.prototypes = 1
                args_.tangents = 100

    # Default attack params
    if args_.eval_norm == "":
        Exception("--eval_norm has to be set when not replicating the paper "
                  "results")
    if args_.eval_norm == 'inf':
        if args_.restarts == -1 or args_.replicate:
            args_.restarts = 3
        if args_.epsilon == -1 or args_.replicate:
            if args_.dataset == "mnist":
                args_.epsilon = 0.3
            else:
                args_.epsilon = 8 / 255
        if args_.steps == -1 or args_.replicate:
            args_.steps = 200
    if args_.eval_norm == '2':
        if args_.restarts == -1 or args_.replicate:
            args_.restarts = 10
        if args_.epsilon == -1 or args_.replicate:
            if args_.dataset == "mnist":
                args_.epsilon = 1.58
            else:
                args_.epsilon = 36 / 255
        if args_.steps == -1 or args_.replicate:
            if args_.model == "rslvq":
                args_.steps = 3000
            else:
                args_.steps = 1000
    return args_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_dir", type=str, required=True,
                        help="Directory to save the results to")
    parser.add_argument("--replicate", action='store_true',
                        help="If set, the exact plot from the paper will be "
                             "replicated.")

    # Model parameters
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="NPC model to instantiate. Should be one of "
                             "three: glvq, gtlvq, or rslvq")
    parser.add_argument("-w", "--weights_path", type=str, default="",
                        help="Path to weightsfile")
    parser.add_argument("-p", "--prototypes", type=int, default=128,
                        help="Number of prototypes per class for model")
    parser.add_argument("-t", "--tangents", type=int, default=-1,
                        help="Number of tangents per prototype. Only set when"
                             " model argument is set to gtlvq")
    parser.add_argument("--model_norm", type=str, default="2",
                        help="Norm used for training the model. Note the "
                             "difference to eval_norm argument")

    # Evaluation parameters
    parser.add_argument("-b", "--batch_size", type=int, default=512,
                        help="Batch sized used for empirical robustness "
                             "evaluation.")
    parser.add_argument("-r", "--restarts", type=int, default=-1,
                        help="For eval_norm=inf, this represents the number "
                             "of random restarts for the PGD attack. For "
                             "eval_norm=2, this represents the number of "
                             "binary search steps in the cw attack.")
    parser.add_argument("-e", "--epsilon", type=float, default=-1,
                        help="Epsilon to limit the adversarial attacks to")
    parser.add_argument("--steps", type=int, default=-1,
                        help="Maximum number of steps taken by the attack")
    parser.add_argument("--eval_norm", type=str, default="")

    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # Check default params
    args = check_default_attack_params(args)

    # Check eval_norm and model_norm compatibility
    if args.model_norm == "inf" and not args.eval_norm == "inf":
        raise Exception("Models trained using the inf norm can only be "
                        "evaluated using the inf norm")
    p_norm_eval = to_p_norm(args.eval_norm)
    p_norm_model = to_p_norm(args.model_norm)

    # Handle_ dataset
    (x_train, y_train), (x_test, y_test) = get_data(args.dataset)
    data_shape = tuple(x_train.shape[1:])
    data_size = np.product(data_shape)

    # Load model
    model = get_model(args.model, data_shape,
                      n_classes=10,
                      number_prototypes=args.prototypes,
                      p_norm=p_norm_model,
                      batch_size=args.batch_size,
                      negated_dissimilarities=True,
                      weights_provided=True,
                      number_tangents=args.tangents)
    model.load_weights(args.weights_path)

    # Determine model accuracy
    test_predictions = model.predict(x_test, verbose=True)
    n_test_correct = np.sum(
        np.squeeze(np.argmax(test_predictions, 1)) == np.squeeze(
            np.argmax(y_test, 1)))
    test_acc = float(n_test_correct / x_test.shape[0])

    # Determine model certificates
    test_cert = calculate_certificates(model, (x_test, y_test), p_norm_model,
                                       [args.epsilon], [p_norm_eval],
                                       negated_dissimilarities=True)

    # Determine model empirical robustness
    test_emp_error = empirical_robustness(args, model, x_test, y_test,
                                          data_size)

    with open(args.save_dir + '/evaluation.txt', 'w') as f:
        def print_and_write(text):
            print(text)
            f.write(text + '\n')

        print_and_write("EVALUATION INFORMATION:")
        if not args.tangents == -1:
            print_and_write("{} model with {} prototypes per class and {}-"
                            "dimensional subspace, trained using the L2-"
                            "norm".format(args.model,
                                          args.prototypes,
                                          args.tangents))
        else:
            print_and_write("{} model with {} prototypes per class, "
                            "trained using the L{}-"
                            "norm".format(args.model,
                                          args.prototypes,
                                          args.model_norm))
        print_and_write("Evaluated over the L{}-norm with attacks "
                        "limited to an epsilon of "
                        "{}".format(args.eval_norm,
                                    args.epsilon))
        print_and_write("")
        print_and_write("EVALUATION SCORES")
        print_and_write("CTE: {}".format(1 - test_acc))
        if args.model != 'rslvq':
            print_and_write("URTE: {}".format(1 - test_cert[0]))
        print_and_write("LRTE: {}".format(test_emp_error))

    print("Done")
