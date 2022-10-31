# -*- coding: utf-8 -*-
# @Date    : 10/30/22
# @Author  : Shawn Shan (shawnshan@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/

import sys
import argparse
import numpy as np
import tc
import glob
import os
from sklearn.utils import shuffle
import random
import torch.nn.functional as F
import torch

arch = 'resnet18'


def load_models():
    model_ls = []
    root_path = args.res_dir
    model_paths = os.path.join(root_path, args.exp_name, "*.p")
    all_models = glob.glob(model_paths)
    if len(all_models) == 0:
        raise Exception("No model found in {}".format(model_paths))

    print("Loading {} models...".format(len(all_models)))

    for p in all_models:
        model = tc.load_model(p, arch=arch)
        model_ls.append(model)
    return model_ls


def calculate_loss(cur_model, x):
    logit_matrix = cur_model(x)
    logit_matrix = logit_matrix.type(torch.cuda.DoubleTensor)
    pred_y = logit_matrix.argmax(1)
    logit_matrix = F.log_softmax(logit_matrix)
    loss_array = torch.log(F.nll_loss(logit_matrix, pred_y, reduction='none')) / 10
    return loss_array


def loss_diff(mA, mB, x):
    return calculate_loss(mA, x) - calculate_loss(mB, x)


def main():
    X_train, Y_train, X_test, Y_test = tc.load_dataset(args.dataset)

    rid = random.sample(range(len(X_test)), int(len(X_test) * 0.1))
    X_test = X_test[rid]
    Y_test = Y_test[rid]

    y_target = 3

    X_target, Y_target, X_test, Y_test = tc.get_target_data(X_test, Y_test, y_target, source_only=False)
    # X_target, Y_target, X_test, Y_test = get_target_data(X_train, Y_train, y_target, source_only=False)

    X_test, Y_test = shuffle(X_test, Y_test)
    X_test = np.array(X_test)[:1000]
    Y_test = np.array(Y_test)[:1000]
    iters_eval = 200
    alpha_eval = 0.001
    eps_eval = 0.05
    model_ls = load_models()

    for number_breached_models in range(1, args.max_num_breach):
        cur_succ_ls = []
        for _ in range(args.runs):
            breach_models_idices = random.sample(range(len(model_ls)), number_breached_models)

            new_model_idx = random.choice([i for i in range(len(model_ls)) if i not in breach_models_idices])
            assert new_model_idx not in breach_models_idices

            breach_models = [model_ls[i] for i in breach_models_idices]
            new_model = model_ls[new_model_idx]

            cur_X = tc.transform(X_test[:50])

            Y_Target = tc.gpu([y_target] * len(cur_X))

            adv_x = tc.pgd_torch(breach_models, cur_X, Y_Target, targeted=True, iters=iters_eval, alpha=alpha_eval,
                                 eps=eps_eval)
            transfer_rate = tc.cpu(torch.mean((new_model(adv_x).argmax(1) == Y_Target).float()))
            loss_diff_matrix_adv = []
            loss_diff_matrix_benign = []

            for breach_model in breach_models:
                cur_loss_diff_adv = loss_diff(new_model, breach_model, adv_x)
                cur_loss_diff_benign = loss_diff(new_model, breach_model, cur_X)

                loss_diff_matrix_adv.append(tc.cpu(cur_loss_diff_adv))
                loss_diff_matrix_benign.append(tc.cpu(cur_loss_diff_benign))

            max_loss_diff_array_adv = np.max(np.array(loss_diff_matrix_adv), axis=0)
            max_loss_diff_array_benign = np.array(loss_diff_matrix_benign).reshape(-1)
            th = np.percentile(max_loss_diff_array_benign, 95)
            filter_rate = np.mean(max_loss_diff_array_adv > th)
            final_attack_succ_rate = transfer_rate * (1 - filter_rate)
            cur_succ_ls.append(final_attack_succ_rate)
        print("When {} breaches happens, final attack success post detection {:.2f}".format(number_breached_models,
                                                                                            np.mean(cur_succ_ls)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')
    parser.add_argument('--dataset', type=str,
                        help='name of dataset', default='cifar10')
    parser.add_argument('--exp-name', type=str,
                        help='Current experiment name', default='recovery_experiment0')
    parser.add_argument('--res-dir', type=str,
                        help='Result directory for the trained models', default='./')
    parser.add_argument('--max-num-breach', type=int,
                        help='Max number of breaches', default=4)
    parser.add_argument('--runs', type=int,
                        help='Number of runs per experiment', default=5)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
