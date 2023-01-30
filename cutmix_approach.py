from __future__ import print_function

import argparse, csv, os, time

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from utils import *
import config_baseline as args


use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

device = 'cuda' if use_cuda else 'cpu'
if args.seed != 0:
    torch.manual_seed(args.seed)

dataset_list = check_dataset_dir(args.dataset_dir)

# Data
print('==> Preparing data..')
transform_train, transform_test = get_transforms(args.image_size)

result_df = pd.DataFrame(columns = ['Dataset', 'Iter', 'Trial',
'Test_Acc', 'Test, Pre', 'Test_Re', 'Test_F1', 'Train_Acc',
'Train_Pre', 'Train_Re', 'Train_F1'])

# starting time
start = time.time()
for dataset in dataset_list:
    trainset, trainloader, testset, testloader = get_loaders_and_dataset(dataset, transform_train, transform_test, args.batch_size)
    current_dataset_file = dataset.split("/")[-1] + '_.txt'

    for iteration in range(args.iterations):
        for trial in range(args.trials):
            print("Working on dataset: ", dataset, " in iteration ", iteration, " and model ", trial)
            current_exp = "_ite_" + str(iteration) + "_trial_" + str(trial) + "_dataset_" + dataset.split("/")[-1] + "_"

            net, criterion, optimizer, scheduler = load_model_and_train_params(args.image_size, device, args.lr, testset, args.use_old)
            _, metrics = run_experiment_cutmix_approach(trainloader, testloader, current_exp, args.epochs, net, optimizer, scheduler, best_acc, criterion, device, args.lr,
            args.beta, args.cutmix_prob, iteration = iteration, trial = trial, dataset = dataset, classes = testset.classes, current_dataset_file = current_dataset_file)

            result_df.loc[len(result_df.index)] = metrics
            result_df.to_csv('baseline_cutmix' + '.csv')
# end time
end = time.time()
# total time taken
print(f"Runtime of the program is {end - start} seconds which is equivalent to {(end - start)/60} minutes")
