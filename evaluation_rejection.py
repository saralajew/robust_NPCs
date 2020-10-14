# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

from functions import get_data, get_model, hypothesis_margin

import numpy as np


def to_p_norm(model_norm):
    if model_norm == "inf":
        return np.inf
    elif model_norm == "2":
        return 2
    else:
        raise NotImplementedError("Model norm either has to be 'inf' or '2'")


def rejected(p_norm, y_pred_, epsilon):
    margins = hypothesis_margin(p_norm, y_pred_)
    ids = np.argwhere((margins - epsilon) <= 0)
    return ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_dir", type=str, required=True,
                        help="Directory to save the results to")
    parser.add_argument("--plot", action='store_true',
                        help="If set, a plot of the rejection rate will be "
                             "stored")
    parser.add_argument("-r", "--replicate", action='store_true',
                        help="If set, the exact plot from the paper will be "
                             "replicated.")

    # Model parameters
    parser.add_argument("-m", "--model", type=str, default="glvq",
                        help="NPC model to instantiate. Should be one of "
                             "three: glvq, gtlvq or rslvq")
    parser.add_argument("-w", "--weights_path", type=str,
                        default="weight_files/GLVQ/mnist/linf_trained/"
                                "glvq_loss.h5",
                        help="Path to weightsfile")
    parser.add_argument("-p", "--prototypes", type=int, default=10,
                        help="Number of prototypes per class for model")
    parser.add_argument("-t", "--tangents", type=int, default=-1,
                        help="Number of tangents per prototype. Only set when"
                             " model argument is set to gtlvq")
    parser.add_argument("--model_norm", type=str, default="inf",
                        help="Norm used for training the model. Note the "
                             "difference to eval_norm argument")

    # Evaluation parameters
    parser.add_argument("-e", "--epsilon_max", type=float, default=0.5,
                        help="Epsilon to limit the adversarial attacks to")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of intermediate epsilon values to "
                             "consider.")
    parser.add_argument("--dataset", type=str, default='mnist')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.replicate:
        args.model = "glvq"
        args.weights_path = "weight_files/GLVQ/mnist/linf_trained/glvq_loss.h5"
        args.prototypes = 128
        args.model_norm = "inf"
        args.epsilon_max = 0.5
        args.steps = 50
        args.dataset = 'mnist'
        args.plot = True

    # Handle_ dataset
    (x_train, y_train), (x_test, y_test) = get_data(args.dataset)
    data_shape = tuple(x_test.shape[1:])

    # Load model
    p_norm_model = to_p_norm(args.model_norm)
    model = get_model(args.model, data_shape,
                      n_classes=10,
                      number_prototypes=args.prototypes,
                      p_norm=p_norm_model,
                      negated_dissimilarities=False,
                      weights_provided=True,
                      number_tangents=args.tangents)
    model.load_weights(args.weights_path)

    y_pred = model.predict(x_test, verbose=True)

    epsilons = np.linspace(0, args.epsilon_max, num=args.steps)
    results = {
        "individual": {},
        "total": {
            "epsilons": [],
            "ratio": []
        }
    }

    for e in epsilons:
        rejected_ids = rejected(p_norm_model, y_pred, e)
        falsely_rejected_ratio = np.shape(rejected_ids)[0] / x_test.shape[0]

        results['individual'][e] = rejected_ids.ravel().tolist()
        results['total']['epsilons'].append(e)
        results['total']['ratio'].append(falsely_rejected_ratio)

    if args.plot:
        import matplotlib.pyplot as plt
        import matplotlib.style as style

        style.use('ggplot')

        fig, ax1 = plt.subplots(figsize=[5, 3])
        ax1.plot(epsilons, results['total']['ratio'],
                 linewidth=2,
                 linestyle="solid")

        ax1.set_xlabel("epsilon")
        ax1.set_ylabel("falsely rejected")

        plt.savefig(args.save_dir + '/rejection.pdf', bbox_inches='tight')
        plt.show()

    with open(args.save_dir + '/rejection.json', 'w') as f:
        json.dump(results, f, indent=3)
