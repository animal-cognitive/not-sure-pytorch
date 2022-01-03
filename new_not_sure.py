# '''Train CIFAR10 with PyTorch.'''
# not_sure.py --epochs 2 --trials=2 --iterations=2 --dataset_dir=../Datasets

# from __future__ import print_function

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

for dataset in dataset_list:
    # Number of not-sure images to create
    NS_ALL = int(len(glob.glob(dataset + "/train/*/*")) / len (glob.glob(dataset + "/train/*")))

    for iteration in range(args.iterations):
        for trial in range(args.trials):

            # Initial clean-up - Delete Not-Sure class if already exists
            not_sure_train_dir = dataset + "/train/Not_Sure/"
            not_sure_test_dir = dataset + "/test/Not_Sure/"
            delete_dir_if_exists(not_sure_train_dir)
            delete_dir_if_exists(not_sure_test_dir)

            #2. Create the dataset subset
            ##################################
            subset_dataset = args.subset_folder
            create_subset_dataset(subset_dataset)

            #2. Split the dataset training into number_of_splits
            no_of_splits = 4
            #
            splits_per_class, class_names = split_dataset(dataset, no_of_splits)
            train_data, full_val = collect_train_and_val_data(splits_per_class, 0)

            copy_images_to_subset_dataset(subset_dataset, dataset, train_data, full_val, class_names)
            #################################

            # Load the train and test loader and set
            batch_size = args.batch_size
            trainset, trainloader, testset, testloader = get_loaders_and_dataset(subset_dataset, transform_train, transform_test, args.batch_size)

            # Run the model for a number of times to choose the best model
            for iter in range(args.subset_train_iter):
                # Load the model
                net, criterion, optimizer, scheduler = load_model_and_train_params(args.image_size, device, args.lr, testset, args.use_old)

                # Run the model
                best_acc = run_experiment(trainloader, testloader, subset_dataset, args.ns_epochs, net, optimizer, scheduler, best_acc, criterion, device, args.lr)

            # Load best model
            checkpoint = torch.load(f'./checkpoint/{subset_dataset}ckpt.pth')
            net.load_state_dict(checkpoint['net'])

            current_dataset_file = dataset.split("/")[-1] + '_.txt'

            # Make prediction on the subset dataset
            targets, predictions, file_paths = make_prediction(net, class_names, testloader)

            # Store the files for each confusion matrix value
            file_pred_list = [[[] for item in range(len(class_names))] for item in range(len(class_names))]

            # Iterate over each target index, for each prediction, get the file path
            for target_index in range(len(targets)):
                target_value = targets[target_index]
                predicted_value = predictions[target_index]
                file_pred_list[target_value][predicted_value].append(file_paths[target_index])

            # Invert confusion matrix for easier processing
            conf_matrix = np.array(confusion_matrix(targets, predictions)).T.tolist()

            TFP = calculate_total_false_positives(conf_matrix)
            no_of_classes = len(class_names)

            # Load the train and test loader and set
            batch_size = args.batch_size
            trainset, trainloader, testset, testloader = get_loaders_and_dataset(dataset, transform_train, transform_test, args.batch_size)

            # Run the model for a number of times to choose the best model
            for iter in range(args.subset_train_iter):
                # Load the model
                net, criterion, optimizer, scheduler = load_model_and_train_params(args.image_size, device, args.lr, testset, args.use_old)

                # Run the model
                best_acc = run_experiment(trainloader, testloader, subset_dataset, args.epochs, net, optimizer, scheduler, best_acc, criterion, device, args.lr)

            # Construct the GradCAM to use
            cam = construct_cam(net.module, [net.module.layer4[-1]], torch.cuda.is_available())

            # Make prediction on the subset dataset
            targets_baseline, predictions_baseline, file_paths_baseline = make_prediction(net, class_names, testloader)

            # Store the files for each confusion matrix value
            file_pred_list_baseline = [[[] for item in range(len(class_names))] for item in range(len(class_names))]

            # Iterate over each target index, for each prediction, get the file path
            for target_index in range(len(targets_baseline)):
                target_value = targets_baseline[target_index]
                predicted_value = predictions_baseline[target_index]
                file_pred_list_baseline[target_value][predicted_value].append(file_paths_baseline[target_index])

            # Create directory to save not-sure images
            create_dir(not_sure_train_dir, True)
            create_dir(not_sure_test_dir, True)

            # For each class in the confusion matrix
            for c in range(no_of_classes):

                # Calculate the false positives only for class class_c
                fp_c = count_false_positives_for_given_class(conf_matrix[c], c)

                # Calculate the number of NS to create for class_c
                NS_c = round((fp_c / TFP) *  NS_ALL)

                # Get |NS_c| True positive images from class c.
                corr_preds_as_c = file_pred_list[c][c]
                corr_preds_as_c_baseline = file_pred_list_baseline[c][c]

                # Check if number of true positive images is not up to NS_c, then we upsample
                # if corr_preds_as_c and len(corr_preds_as_c) < NS_c:
                #     corr_preds_as_c = corr_preds_as_c * math.ceil(NS_c / len(corr_preds_as_c) + 1)

                if corr_preds_as_c_baseline and len(corr_preds_as_c_baseline) < NS_c:
                    corr_preds_as_c_baseline = corr_preds_as_c_baseline * math.ceil(NS_c / len(corr_preds_as_c_baseline) + 1)

                # Compute uniform false positive error UFPE_c for class c
                UFPE_c  = fp_c  / ( no_of_classes - 1 )

                # Get list of classes LIST_c whose number of false positives in class_c is greater than UFPE_c
                list_c = get_list_of_classes_GT_UFPE(conf_matrix[c], c, UFPE_c)

                # Get the total number of False positives from classes LIST_c as FP_Total
                fp_total = count_false_positives_within_list(conf_matrix[c], list_c)

                # To avoid division by zero in computing NS_c_hat
                if fp_total:

                    # Iterate over the classes in list_c
                    for c_hat in list_c:

                        # Get the list of images in class c_hat that were wrongly predicted as class c
                        # wrong_pred_in_c_hat = file_pred_list[c_hat][c]
                        wrong_pred_in_c_hat = file_pred_list_baseline[c_hat][c_hat]
                        fp_from_c_hat = len(file_pred_list[c_hat][c])

                        # Number of NS images to create from class c_hat
                        NS_c_hat = math.ceil( (fp_from_c_hat / fp_total) * NS_c )

                        # Upsample the list of images if not sufficient
                        if wrong_pred_in_c_hat and fp_from_c_hat < NS_c_hat:
                            wrong_pred_in_c_hat = wrong_pred_in_c_hat * math.ceil(NS_c_hat / fp_from_c_hat + 1)

                        # List of images from class c_hat to use
                        images_from_c_hat = wrong_pred_in_c_hat[:NS_c_hat]

                        # Get the list of images from class c and update the list of the remaining images
                        # images_from_c = corr_preds_as_c[:NS_c_hat]
                        images_from_c = corr_preds_as_c_baseline[:NS_c_hat]

                        # print("SHAPE OF Images c: ", len(images_from_c))

                        # Create not-sure images
                        mix(cam, images_from_c, images_from_c_hat, args.image_size, not_sure_train_dir, 0.0, 0.5, args.approach)

                        # Use the correct_preds that have not been used for the next class
                        # corr_preds_as_c = corr_preds_as_c[NS_c_hat:]
                        # Use the correct_preds that have not been used for the next class
                        if len(corr_preds_as_c_baseline) < NS_c_hat:
                            corr_preds_as_c_baseline = corr_preds_as_c_baseline * math.ceil(NS_c_hat / len(corr_preds_as_c_baseline) )
                        corr_preds_as_c_baseline = corr_preds_as_c_baseline[NS_c_hat:]


            # To enable proper computation, copy one training not-sure image to test folder
            shutil.copy(glob.glob(not_sure_train_dir + '*')[0], not_sure_test_dir)

            # Load the train and test loader and set for the full dataset
            trainset, trainloader, testset, testloader = get_loaders_and_dataset(dataset, transform_train, transform_test, args.batch_size)

            print("Working on dataset: ", dataset, " in iteration ", iteration, " and model ", trial)
            current_exp = "_ite_" + str(iteration) + "_trial_" + str(trial) + "_dataset_" + dataset.split("/")[-1] + "_"

            # Load the model
            net, criterion, optimizer, scheduler = load_model_and_train_params(args.image_size, device, args.lr, testset, args.use_old)

            best_acc = 0  # best test accuracy

            run_experiment(trainloader, testloader, current_exp, args.epochs, net, optimizer, scheduler, best_acc, criterion, device, args.lr, iteration = iteration, trial = trial, dataset = dataset, classes = testset.classes, current_dataset_file = current_dataset_file)

            # Retrain the current model but without the not-sure class
            net = net.module
            net.linear = nn.Linear(net.linear.in_features, len(testset.classes) - 1)
            net = torch.nn.DataParallel(net)
            net = net.to(device)
            # print(net)
            for name, param in net.named_parameters():
                if "linear" in name:
                    # print("LAST: ", name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # n1, criterion, optimizer, scheduler = load_model_and_train_params(args.image_size, device, args.lr, testset, args.use_old)
            net, criterion, optimizer, scheduler = load_current_model_and_train_params(net, args.lr, args.use_old)
            delete_dir_if_exists(not_sure_train_dir)
            delete_dir_if_exists(not_sure_test_dir)

            # Load the train and test loader and set for the full dataset
            trainset, trainloader, testset, testloader = get_loaders_and_dataset(dataset, transform_train, transform_test, args.batch_size)

            run_experiment(trainloader, testloader, current_exp, args.epochs - 150, net, optimizer, scheduler, best_acc, criterion, device, args.lr, iteration = iteration, trial = trial, dataset = dataset, classes = testset.classes, current_dataset_file = current_dataset_file)
