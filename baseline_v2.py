'''Train CIFAR10 with PyTorch.'''
# baseline.py --epochs 2 --trials=2 --iterations=2 --dataset_dir=../Datasets
import torch, os, glob, sys, argparse
from utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--dataset_dir', default='Data', type=str,
                    help='The location of the dataset to be explored')
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
parser.add_argument('--use_old', '-old', action='store_true',
                    help='Use old code base')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ensure the dataset folder is present
dataset_list = check_dataset_dir(args.dataset_dir)

# Get the train and test transformations
transform_train, transform_test = get_transforms()

if args.use_old:
    weight_decay = 1e-4
else:
    weight_decay = 5e-4

for dataset in dataset_list:
    current_dataset_file = dataset.split("/")[-1] + '_.txt'

    # Load the train and test loader and set
    trainset, trainloader, testset, testloader = get_loaders_and_dataset(dataset, transform_train, transform_test, args.batch_size)

    for iteration in range(args.iterations):
        for trial in range(args.trials):
            print("Working on dataset: ", dataset, " in iteration ", iteration, " and model ", trial)
            current_exp = "_ite_" + str(iteration) + "_trial_" + str(trial) + "_dataset_" + dataset.split("/")[-1] + "_"

            # Load the model
            net, criterion, optimizer, scheduler = load_model_and_train_params(args.image_size, device, args.lr, testset, weight_decay, args.use_old)

            best_acc = 0  # best test accuracy
            run_experiment(trainloader, testloader, current_exp, args.epochs, net, optimizer, scheduler, best_acc, criterion, device, args.lr, iteration, trial, dataset, testset.classes, current_dataset_file)
