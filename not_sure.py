'''Train CIFAR10 with PyTorch.'''
# not_sure.py --epochs 2 --trials=2 --iterations=2 --dataset_dir=../Datasets

import glob, sys, math, shutil, argparse, torch

from utils import *

from sklearn.metrics import confusion_matrix
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--dataset_dir', default='Data', type=str,
                    help='The location of the dataset to be explored')
parser.add_argument('--subset_folder', default='SUBSET_FOLDER', type=str,
                    help='Temporary location for dataset')
parser.add_argument('--trials', default=5, type=int,
                    help='Number of times to run the complete experiment')
parser.add_argument('--iterations', default=2, type=int,
                    help='Number of times to run the complete experiment')
parser.add_argument('--epochs', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Number of images per batch')
parser.add_argument('--image_size', default=32, type=int,
                    help='input image size')
parser.add_argument('--subset_train_iter', default=1, type=int,
                    help='Iterations for the subset training model')
parser.add_argument('--ns_all', default=500, type=int,
                    help='Total not-sure images to create')
parser.add_argument('--use_old', '-use_old', action='store_true',
                    help='Use old code base')
parser.add_argument('--approach', default=1, type=int,
                    help='Approach for generating not-sure images')
parser.add_argument('--ns_epochs', default=200, type=int,
                    help='total epochs to run for not-sure model')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

# Ensure the dataset folder is present
dataset_list = check_dataset_dir(args.dataset_dir)

if args.use_old:
    weight_decay = 1e-4
else:
    weight_decay = 5e-4

# Get the train and test transformations
transform_train, transform_test = get_transforms()

for dataset in dataset_list:
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

    splits_per_class, class_names = split_dataset(dataset, no_of_splits)
    train_data, full_val = collect_train_and_val_data(splits_per_class, 0)

    copy_images_to_subset_dataset(subset_dataset, dataset, train_data, full_val, class_names)
    #################################

    # Load the train and test loader and set
    trainset, trainloader, testset, testloader = get_loaders_and_dataset(subset_dataset, transform_train, transform_test, args.batch_size)

    # Run the model for a number of times to choose the best model
    for iter in range(args.subset_train_iter):
        # Load the model
        net, criterion, optimizer, scheduler = load_model_and_train_params(args.image_size, device, args.lr, testset, weight_decay, args.use_old)

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
    NS_ALL = args.ns_all # Total number of not-sure images to create
    no_of_classes = len(class_names)

    # Construct the GradCAM to use
    cam = construct_cam(net.module, [net.module.layer4[-1]], torch.cuda.is_available())

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

        # Check if number of true positive images is not up to NS_c, then we upsample
        if corr_preds_as_c and len(corr_preds_as_c) < NS_c:
            corr_preds_as_c = corr_preds_as_c * math.ceil(NS_c / len(corr_preds_as_c) + 1)

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
                wrong_pred_in_c_hat = file_pred_list[c_hat][c]
                fp_from_c_hat = len(wrong_pred_in_c_hat)

                # Number of NS images to create from class c_hat
                NS_c_hat = math.ceil( (fp_from_c_hat / fp_total) * NS_c )

                # Upsample the list of images if not sufficient
                if wrong_pred_in_c_hat and fp_from_c_hat < NS_c_hat:
                    wrong_pred_in_c_hat = wrong_pred_in_c_hat * math.ceil(NS_c_hat / fp_from_c_hat + 1)

                # List of images from class c_hat to use
                images_from_c_hat = wrong_pred_in_c_hat[:NS_c_hat]

                # Get the list of images from class c and update the list of the remaining images
                images_from_c = corr_preds_as_c[:NS_c_hat]

                # Create not-sure images
                mix(cam, images_from_c, images_from_c_hat, args.image_size, not_sure_train_dir, 0.0, 0.5, args.approach)

                # Use the correct_preds that have not been used for the next class
                corr_preds_as_c = corr_preds_as_c[NS_c_hat:]

    # To enable proper computation, copy one training not-sure image to test folder
    shutil.copy(glob.glob(not_sure_train_dir + '*')[0], not_sure_test_dir)

    # Load the train and test loader and set for the full dataset
    trainset, trainloader, testset, testloader = get_loaders_and_dataset(dataset, transform_train, transform_test, args.batch_size)

    for iteration in range(args.iterations):
        for trial in range(args.trials):
            print("Working on dataset: ", dataset, " in iteration ", iteration, " and model ", trial)
            current_exp = "_ite_" + str(iteration) + "_trial_" + str(trial) + "_dataset_" + dataset.split("/")[-1] + "_"

            # Load the model
            net, criterion, optimizer, scheduler = load_model_and_train_params(args.image_size, device, args.lr, testset, weight_decay, args.use_old)

            best_acc = 0  # best test accuracy

            run_experiment(trainloader, testloader, current_exp, args.epochs, net, optimizer, scheduler, best_acc, criterion, device, args.lr, current_dataset_file)
