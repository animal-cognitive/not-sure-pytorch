# '''Train CIFAR10 with PyTorch.'''

import argparse, csv, os, time, sys

import numpy as np
import pandas as pd
import torch, random
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from utils import *
import config_ns as args

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

result_df = pd.DataFrame(columns = ['Approach', 'Dataset', 'Iter', 'Trial',
'Test_Acc', 'Test, Pre', 'Test_Re', 'Test_F1', 'Train_Acc',
'Train_Pre', 'Train_Re', 'Train_F1'])

for dataset in dataset_list:

    # Initial clean-up - Delete Not-Sure class if already exists
    not_sure_train_dir = dataset + "/train/Not_Sure/"
    not_sure_test_dir = dataset + "/test/Not_Sure/"
    delete_dir_if_exists(not_sure_train_dir)
    delete_dir_if_exists(not_sure_test_dir)

    # Delete and recreate the folders for each approach
    for approach in args.approach_list:
        not_sure_approach_dir = dataset + f"/approach_{approach}/Not_Sure/"
        create_dir(not_sure_approach_dir, True)

    # Number of not-sure images to create - Average number of images in all class
    train_images_list = glob.glob(dataset + "/train/*/*")
    NS_ALL = int(len(train_images_list) / len (glob.glob(dataset + "/train/*")))

    # No need for cam for approach 3 and 4
    cam = None

    current_dataset_file = dataset.split("/")[-1] + '_.txt'

    # Randomize and get random images
    random.shuffle(train_images_list)
    first_imgs = train_images_list[:NS_ALL]

    # Randomize and get random images
    random.shuffle(train_images_list)
    second_imgs = train_images_list[:NS_ALL]

    # Create not-sure images
    mix(cam, first_imgs, second_imgs, args.image_size, dataset, 0.0, 0.5, args.approach_list)

    for iteration in range(args.iterations):
        for trial in range(args.trials):

            # Test over the Not-sure images created for each approach
            for approach in args.approach_list:

                # Create directory to save not-sure images inside of the training folder
                create_dir(not_sure_test_dir, True)

                # Current approach not-sure data
                app_ns_data = dataset + f"/approach_{approach}/Not_Sure/"

                # To enable proper computation, copy one training not-sure image to test folder
                shutil.copy(glob.glob(app_ns_data + '*')[0], not_sure_test_dir)

                # Copy the not-sure training data
                copy_to_other_dir(app_ns_data, not_sure_train_dir)

                # Load the train and test loader and set for the full dataset
                trainset, trainloader, testset, testloader = get_loaders_and_dataset(dataset, transform_train, transform_test, args.batch_size)

                print(f"Working on approach {approach}, dataset: ", dataset, " in iteration ", iteration, " and model ", trial)
                current_exp = "_appr_"  + str(approach) + "_ite_" + str(iteration) + "_trial_" + str(trial) + "_dataset_" + dataset.split("/")[-1] + "_"

                # Load the model
                net, criterion, optimizer, scheduler = load_model_and_train_params(args.image_size, device, args.lr, testset, args.use_old)

                best_acc = 0  # best test accuracy

                run_experiment(trainloader, testloader, current_exp, args.epochs, net, optimizer, scheduler, best_acc, criterion, device, args.lr)

                # Retrain the current model but without the not-sure class
                net = net.module
                froozen_layer = ""
                if args.image_size == 32:
                    froozen_layer = "linear"
                    net.linear = nn.Linear(net.linear.in_features, len(testset.classes) - 1)
                else:
                    froozen_layer = "fc"
                    net.fc = nn.Linear(net.fc.in_features, len(testset.classes) - 1)

                net = torch.nn.DataParallel(net)
                net = net.to(device)

                for name, param in net.named_parameters():
                    if froozen_layer in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

                net, criterion, optimizer, scheduler = load_current_model_and_train_params(net, args.lr, args.use_old)

                # Remove the not-sure data
                delete_dir_if_exists(not_sure_train_dir)
                delete_dir_if_exists(not_sure_test_dir)

                # Load the train and test loader and set for the full original dataset
                trainset, trainloader, testset, testloader = get_loaders_and_dataset(dataset, transform_train, transform_test, args.batch_size)

                epochs_for_transfer_learning = args.epochs // 4
                if epochs_for_transfer_learning < 1:
                    epochs_for_transfer_learning = 1
                _, metrics = run_experiment(trainloader, testloader, current_exp, epochs_for_transfer_learning, net, optimizer, scheduler, best_acc, criterion, device, args.lr, iteration = iteration, trial = trial, dataset = dataset, classes = testset.classes, current_dataset_file = current_dataset_file)

                # Add the current approach
                metrics.insert(0, approach)

                result_df.loc[len(result_df.index)] = metrics
                result_df.to_csv(args.subset_folder + '.csv')

result_df.to_csv(args.subset_folder + '.csv')
