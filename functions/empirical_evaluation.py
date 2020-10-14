# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import foolbox as fb


def empirical_robustness(args, model, x_test, y_test, data_size):
    # Create foolbox model
    fmodel = fb.models.KerasModel(model, bounds=(0, 1), predicts='logits')

    # Handle different attacks
    attack, attack_args, random_restarts = get_attack(args, fmodel)
    perturbation_calculator = get_perturbation_calculation(args)
    error_calculator = get_error_calculation(args)

    # Setup batches
    batch_size = args.batch_size
    batch_start = 0
    batch_end = batch_start + batch_size
    cont = True
    scores = np.zeros((x_test.shape[0], random_restarts))

    while cont:
        print("Batch : " + str(batch_start) + " - " + str(batch_end))
        # Make sure that we do not go over the size of the dataset
        if batch_end > x_test.shape[0]:
            batch_end = x_test.shape[0]
            cont = False

        images = x_test[batch_start:batch_end, :, :, :]
        labels = np.squeeze(np.argmax(y_test[batch_start:batch_end], axis=1))

        # Run multiple random restarts if this is set
        for i in range(random_restarts):
            print("Iteration : " + str(i))
            advs = attack(images, labels, **attack_args)
            advs = np.where(np.isnan(advs), np.inf, advs)
            scores[batch_start:batch_end, i] = \
                perturbation_calculator(advs, images, batch_end, batch_start,
                                        data_size)

        # Update batches
        batch_start = batch_end
        batch_end += batch_size
        # Terminate if all samples are evaluated
        if batch_start >= x_test.shape[0]:
            cont = False

    perturbations = np.min(scores, axis=1)
    error = error_calculator(perturbations, x_test.shape[0])

    return error


def get_attack(args, fmodel):
    if args.eval_norm == "inf":
        random_restart = args.restarts
        attack = fb.attacks.ProjectedGradientDescent(
            fmodel, distance=fb.distances.Linf)
        attack_args = {"iterations": args.steps, "epsilon": args.epsilon,
                       "random_start": True, "binary_search": False}

    elif args.eval_norm == "2":
        random_restart = 1
        attack = fb.attacks.CarliniWagnerL2Attack(
            fmodel, distance=fb.distances.MSE)
        attack_args = {"max_iterations": args.steps,
                       "binary_search_steps": args.restarts,
                       "learning_rate": 0.05,
                       "initial_const": 1e0}

    else:
        raise NotImplementedError("Evaluation norm not known, "
                                  "use '2' or 'inf'")

    return attack, attack_args, random_restart


def get_error_calculation(args):
    if args.eval_norm == "2":
        def error(perturbations, num):
            return np.count_nonzero(perturbations[:num] < args.epsilon) / num

    elif args.eval_norm == "inf":
        def error(perturbations, num):
            return (num - np.sum(np.isinf(perturbations))) / num

    else:
        raise NotImplementedError("Evaluation norm not known, "
                                  "use '2' or 'inf'")

    return error


def get_perturbation_calculation(args):
    if args.eval_norm == "2":
        def perturbation(advs, images, batch_end, batch_start, data_size):
            return np.linalg.norm((advs - images).reshape(
                (batch_end - batch_start, data_size)), 2, axis=1)

    elif args.eval_norm == "inf":
        def perturbation(advs, images, batch_end, batch_start, data_size):
            return np.linalg.norm((advs - images).reshape(
                (batch_end - batch_start, data_size)), np.inf, axis=1)

    else:
        raise NotImplementedError("Evaluation norm not known, "
                                  "use '2' or 'inf'")

    return perturbation
