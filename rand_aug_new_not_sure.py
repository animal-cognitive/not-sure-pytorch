# '''Train CIFAR10 with PyTorch.'''

import argparse, csv, os, time, sys

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
transform_train, transform_test = get_transforms(args.image_size, rand_aug = True)

transform_train.transforms.insert(0, transforms.RandAugment(num_ops = 1, magnitude = 2))

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
    NS_ALL = int(len(glob.glob(dataset + "/train/*/*")) / len (glob.glob(dataset + "/train/*")))

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

    # Train the weak model on the subset dataset
    for iter in range(args.subset_train_iter):
        # Load the model
        net, criterion, optimizer, scheduler = load_model_and_train_params(args.image_size, device, args.lr, testset, args.use_old)

        # Run the model
        best_acc, _ = run_experiment(trainloader, testloader, subset_dataset, args.ns_epochs, net, optimizer, scheduler, best_acc, criterion, device, args.lr)

    # Load the best weak model
    checkpoint = torch.load(f'./checkpoint/{subset_dataset}ckpt.pth')
    net.load_state_dict(checkpoint['net'])

    current_dataset_file = dataset.split("/")[-1] + '_.txt'

    # Make prediction on the subset dataset
    targets, predictions, file_paths = make_prediction(net, testloader)

    # Store the files for each confusion matrix value
    file_pred_list = [[[] for item in range(len(class_names))] for item in range(len(class_names))]

    # Iterate over each target index, for each prediction, get the file path
    for target_index in range(len(targets)):
        target_value = targets[target_index]
        predicted_value = predictions[target_index]
        file_pred_list[target_value][predicted_value].append(file_paths[target_index])

    # Create the confusion matrix from the weakly trained model
    conf_matrix = np.array(confusion_matrix(targets, predictions)).T.tolist()

    TFP = calculate_total_false_positives(conf_matrix)
    no_of_classes = len(class_names)

    # Load the train and test loader and set
    batch_size = args.batch_size
    trainset, trainloader, testset, testloader = get_loaders_and_dataset(dataset, transform_train, transform_test, args.batch_size)

    best_acc = 0
    for iter in range(args.subset_train_iter):
        # Load the model
        net, criterion, optimizer, scheduler = load_model_and_train_params(args.image_size, device, args.lr, testset, args.use_old)

        # Run the model
        best_acc, _ = run_experiment(trainloader, testloader, dataset.split("/")[-1], args.epochs, net, optimizer, scheduler, best_acc, criterion, device, args.lr)

    # Load the best good model
    checkpoint = torch.load(f'./checkpoint/{dataset.split("/")[-1]}ckpt.pth')
    net.load_state_dict(checkpoint['net'])

    # Construct the GradCAM to use
    cam = construct_cam(net.module, [net.module.layer4[-1]], torch.cuda.is_available())

    # Prediction results from the good baseline model.
    targets_baseline, predictions_baseline, file_paths_baseline = make_prediction(net, trainloader)

    # Store the files for each confusion matrix value
    file_pred_list_baseline = [[[] for item in range(len(class_names))] for item in range(len(class_names))]

    # Iterate over each target index, for each prediction, get the file path
    for target_index in range(len(targets_baseline)):
        target_value = targets_baseline[target_index]
        predicted_value = predictions_baseline[target_index]
        file_pred_list_baseline[target_value][predicted_value].append(file_paths_baseline[target_index])

    # For each class in the confusion matrix
    for c in range(no_of_classes):

        # Calculate the false positives only for class class_c
        fp_c = count_false_positives_for_given_class(conf_matrix[c], c)

        # Calculate the number of NS to create for class_c
        NS_c = round((fp_c / TFP) *  NS_ALL)

        # Get |NS_c| True positive images from class c.
        corr_preds_as_c = file_pred_list[c][c]
        corr_preds_as_c_baseline = file_pred_list_baseline[c][c]

        # If any of the items are empty
        if not len(corr_preds_as_c_baseline) or not len(corr_preds_as_c):
            continue

        # To ensure the number of images are always up to NS_c
        corr_preds_as_c_baseline = corr_preds_as_c_baseline * math.ceil((NS_c / len(corr_preds_as_c_baseline)) + 1)

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
                wrong_pred_in_c_hat = file_pred_list_baseline[c_hat][c_hat]
                fp_from_c_hat = len(file_pred_list[c_hat][c])

                # If any of the items are empty
                if not fp_from_c_hat or not len(wrong_pred_in_c_hat):
                    continue

                # Number of NS images to create from class c_hat
                NS_c_hat = math.ceil( (fp_from_c_hat / fp_total) * NS_c )

                # Upsample the list of images
                wrong_pred_in_c_hat = wrong_pred_in_c_hat * math.ceil((NS_c_hat / fp_from_c_hat) + 1)

                # List of images from class c_hat to use
                images_from_c_hat = wrong_pred_in_c_hat[:NS_c_hat]

                # Get the list of images from class c and update the list of the remaining images
                images_from_c = corr_preds_as_c_baseline[:NS_c_hat]

                # Create not-sure images
                if len(images_from_c_hat) > 0 and len(images_from_c_hat) == len(images_from_c):
                    mix(cam, images_from_c, images_from_c_hat, args.image_size, dataset, 0.0, 0.5, args.approach_list)

                # Use the correct_preds that have not been used for the next class
                if len(corr_preds_as_c_baseline) < NS_c_hat:
                    corr_preds_as_c_baseline = corr_preds_as_c_baseline * math.ceil(NS_c_hat / len(corr_preds_as_c_baseline) )

                if len(corr_preds_as_c_baseline) > 0:
                    corr_preds_as_c_baseline = corr_preds_as_c_baseline[NS_c_hat:]

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
