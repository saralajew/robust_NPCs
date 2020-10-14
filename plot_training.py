# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('ggplot')

color1 = "#F8766D"
color2 = "#619CFF"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_dir", type=str, required=True,
                        help="Directory to save the results to")
    parser.add_argument("-c", "--csv_file", type=str,
                        default="weight_files/GLVQ/mnist/linf_trained/"
                                "training_logs/glvq.csv",
                        help="Path to training logs")
    parser.add_argument("-r", "--replicate", action='store_true',
                        help="If set, the exact plot from the paper will be "
                             "replicated.")
    parser.add_argument("-n", "--number_of_epochs", type=int, default=200,
                        help="Number of epochs to show in the plot.")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.replicate:
        args.csv_file = "weight_files/GLVQ/mnist/linf_trained/" \
                        "training_logs/glvq.csv"
        args.number_of_epochs = 200
    n = args.number_of_epochs

    columns = defaultdict(list)
    with open(args.csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k, v) in row.items():
                columns[k].append(v)

    fig, ax1 = plt.subplots(figsize=[5, 3])
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('error', color=color1)
    p1 = ax1.plot([1] + [1 - float(i) for i in columns["val_certificate"][:n]],
                  linewidth=2, linestyle="dashed", color=color1)
    p2 = ax1.plot([1] + [1 - float(i) for i in columns["val_acc"][:n]],
                  linewidth=2, linestyle="solid", color=color1)
    if args.replicate:
        ax1.set_ylim([0, 1.0])

    ax2 = ax1.twinx()

    ax2.set_ylabel('loss', color=color2)
    p3 = ax2.plot([float(i) for i in columns["val_loss"][:n]], linewidth=2,
                  color=color2, linestyle='dotted')
    if args.replicate:
        ax2.set_ylim([-0.5, 0])
        ax2.grid()

    lns = p1 + p2 + p3
    plt.legend(lns, ["URTE", "CTE", "GLVQ"])

    fig.savefig(args.save_dir + "/loss.pdf", bbox_inches='tight')
    plt.show()
