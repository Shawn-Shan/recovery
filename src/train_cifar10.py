# -*- coding: utf-8 -*-
# @Date    : 2/11/22
# @Author  : Shawn Shan (shawnshan@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/

import argparse
import sys
import numpy as np
import time
import torch
import torch as ch
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss
import random
import glob
import tc
from tqdm import tqdm
import sys
import os
BATCH_SIZE = 512
EPOCHS = 20


def get_loaders():
    X_train, Y_train, X_test, Y_test = tc.load_dataset(args.dataset)

    train_loader_m = tc.get_loader(X_train, Y_train, tc.ts, batch_size=BATCH_SIZE)
    test_loader_m = tc.get_loader(X_test, Y_test, tc.test_ts, batch_size=BATCH_SIZE)

    return train_loader_m, test_loader_m


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    loss_fn = CrossEntropyLoss()
    start_t = time.time()
    n_class = 10

    def eval_model(model, test_loader):
        model.eval()
        with ch.no_grad():
            total_correct, total_num = 0., 0.

            for ims, labs in test_loader:
                ims = ims.cuda()
                labs = labs.cuda()
                labs = labs.reshape(-1)
                with autocast():
                    out = model(ims)
                    total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                    total_num += ims.shape[0]

        return total_correct / total_num

    train_loader_m, test_loader_m = get_loaders()


    for idx in range(args.number_models):
        all_hidden_paths = glob.glob(os.path.join(args.hidden_dir, "*.p"))
        if len(all_hidden_paths) < n_class * 4:
            raise Exception("Not enough hidden distribution in hidden-dir, generate at least: {}".format(n_class * 4))

        selected_hidden = random.sample(all_hidden_paths, n_class * 4)

        train_loader_s, test_loader_s = tc.get_sub_data(selected_hidden, loader=True, test=True, n_sample=1024)

        model = tc.load_model(arch="resnet18")

        train_loader_s_it = iter(train_loader_s)

        opt, scheduler, scaler = tc.get_opt(model, 50000 * 2,
                                            lr=0.5,
                                            batch_size=BATCH_SIZE * 2,
                                            epochs=EPOCHS)
        print("Prep time: {:.2f}".format(time.time() - start_t))
        for ep in range(EPOCHS):
            model.train()
            pbar = tqdm(train_loader_m)
            for ims, labs in pbar:
                ims = ims.cuda()
                labs = labs.cuda()
                labs = labs.reshape(-1)
                try:
                    ims_s, labs_s = next(train_loader_s_it)
                except StopIteration:
                    train_loader_s_it = iter(train_loader_s)
                    ims_s, labs_s = next(train_loader_s_it)
                ims_s = ims_s.cuda()
                labs_s = labs_s.cuda()

                ims = ch.cat([ims, ims_s])
                labs = ch.cat([labs, labs_s])

                labs = labs.cuda()

                opt.zero_grad(set_to_none=True)
                with autocast():
                    out = model(ims)
                    loss_normal = loss_fn(out, labs)
                    loss = loss_normal

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                scheduler.step()

            if ep % 1 == 0 or ep == EPOCHS - 1:
                acc_m = eval_model(model, test_loader_m)
                acc_s = eval_model(model, test_loader_s)
                print("Epoch: {}, Main Acc: {:.2f}, Hidden Acc: {:.2f}".format(ep, acc_m, acc_s))

            root_path = args.res_dir
            if not os.path.exists(root_path):
                os.makedirs(root_path)

            cur_directory = os.path.join(root_path, args.exp_name)
            if not os.path.exists(cur_directory):
                os.makedirs(cur_directory)

            model_format = "m{}.p"
            torch.save(model.state_dict(), os.path.join(cur_directory, model_format).format(idx))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')
    parser.add_argument('--number-models', type=int,
                        help='Number of models to train', default=10)
    parser.add_argument('--dataset', type=str,
                        help='Training dataset to use', default='cifar10')
    parser.add_argument('--res-dir', type=str,
                        help='Result directory for the trained models', default='./')
    parser.add_argument('--hidden-dir', type=str,
                        help='Result directory for the hidden dist', default='hidden')
    parser.add_argument('--exp-name', type=str,
                        help='Current experiment name', default='recovery_experiment0')
    return parser.parse_args(argv)


if __name__ == '__main__':
    s = time.time()
    args = parse_arguments(sys.argv[1:])

    main()
    print("Total run time: {:.2f}".format(time.time() - s))
