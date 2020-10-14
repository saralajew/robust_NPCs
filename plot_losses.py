# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from functions import get_model, get_data, hypothesis_margin

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np

style.use('ggplot')

MARGIN_STEPS = np.linspace(0, 0.5, num=50)
color1 = "#F8766D"
color2 = "#7CAE00"
color3 = "#00BFC4"
color4 = "#C77CFF"


def remaining_accuracies(margins):
    accuracy = []
    for m in MARGIN_STEPS:
        acc = np.sum((margins - m) > 0) / len(margins)
        accuracy.append(acc)
    return accuracy


def calculate_urte(margins):
    urtes = []
    for m in MARGIN_STEPS:
        acc = np.sum((margins - m) > 0) / len(margins)
        urtes.append(1 - acc)
    return urtes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_dir", type=str, required=True,
                        help="Directory to save the results to")
    parser.add_argument("-r", "--replicate", action='store_true',
                        help="If set, the exact plot from the paper will be "
                             "replicated.")
    parser.add_argument("-n", "--number_of_samples", type=int, default=10000)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    (x_train, y_train), (x_test, y_test) = get_data('mnist')
    n = args.number_of_samples

    model = get_model('glvq', (28, 28, 1),
                      n_classes=10,
                      number_prototypes=128,
                      p_norm=np.inf,
                      batch_size=128,
                      negated_dissimilarities=False,
                      weights_provided=True,
                      number_tangents=-1)
    # GLVQ LOSS
    model.load_weights("weight_files/GLVQ/mnist/linf_trained/glvq_loss.h5")

    y_pred = model.predict(x_test[:n], verbose=True)
    margins_glvq = hypothesis_margin(np.inf, y_pred, y_test[:n])

    acc_glvq = calculate_urte(margins_glvq)

    # RELU 03 Loss
    model.load_weights("weight_files/GLVQ/mnist/linf_trained/03_loss.h5")

    y_pred = model.predict(x_test[:n], verbose=True)
    margins_03 = hypothesis_margin(np.inf, y_pred, y_test[:n])

    acc_03 = calculate_urte(margins_03)

    # RELU 02 Loss
    model.load_weights("weight_files/GLVQ/mnist/linf_trained/02_loss.h5")

    y_pred = model.predict(x_test[:n], verbose=True)
    margins_02 = hypothesis_margin(np.inf, y_pred, y_test[:n])

    acc_02 = calculate_urte(margins_02)

    # RELU 01 Loss
    model.load_weights("weight_files/GLVQ/mnist/linf_trained/01_loss.h5")

    y_pred = model.predict(x_test[:n], verbose=True)
    margins_01 = hypothesis_margin(np.inf, y_pred, y_test[:n])

    acc_01 = calculate_urte(margins_01)

    fig, ax1 = plt.subplots(figsize=[5, 3])
    ax1.plot(MARGIN_STEPS, acc_glvq, linewidth=2,
             linestyle="solid", color=color1)
    ax1.plot(MARGIN_STEPS, acc_03, linewidth=2,
             linestyle="dotted", color=color2)
    ax1.plot(MARGIN_STEPS, acc_02, linewidth=2,
             linestyle="dashed", color=color3)
    ax1.plot(MARGIN_STEPS, acc_01, linewidth=2,
             linestyle="dashdot", color=color4)

    ax1.legend(["GLVQ", "ReLU (eps = 0.3)",
                "ReLU (eps = 0.2)", "ReLU (eps = 0.1)"])

    ax1.set_xlabel("epsilon")
    ax1.set_ylabel("URTE")

    fig.savefig(args.save_dir + "/bounds.pdf",  bbox_inches='tight')
    plt.show()
