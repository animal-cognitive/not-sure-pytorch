from __future__ import print_function

import argparse, csv, os, time

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--dataset_dir', default='Data', type=str, help='The location of the dataset to be explored')
parser.add_argument('--subset_folder', default='SUBSET_FOLDER', type=str, help='Temporary location for dataset')
parser.add_argument('--trials', default=3, type=int, help='Number of times to run the complete experiment')
parser.add_argument('--iterations', default=1, type=int, help='Number of times to run the complete experiment')
parser.add_argument('--epochs', default=200, type=int, help='Epochs')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--image_size', default=32, type=int, help='input image size')
parser.add_argument('--use_old', '-use_old', action='store_true', help='Use old code base')
parser.add_argument('--approach', default=1, type=int, help='Approach')
parser.add_argument('--ns_epochs', default=200, type=int, help='ns_epochs')
parser.add_argument('--subset_train_iter', default=1, type=int, help='Iterations for the subset training model')


args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

device = 'cuda' if use_cuda else 'cpu'
if args.seed != 0:
    torch.manual_seed(args.seed)

dataset_list = check_dataset_dir(args.dataset_dir)

# Data
print('==> Preparing data..')
transform_train, transform_test = get_transforms()

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

            run_experiment(trainloader, testloader, current_exp, args.epochs, net, optimizer, scheduler, best_acc, criterion, device, args.lr,
            iteration = iteration, trial = trial, dataset = dataset, classes = testset.classes, current_dataset_file = current_dataset_file)

# end time
end = time.time()
# total time taken
print(f"Runtime of the program is {end - start} seconds which is equivalent to {(end - start)/60} minutes")
