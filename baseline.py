'''Train CIFAR10 with PyTorch.'''
# pip install git+https://github.com/ildoonet/pytorch-randaugment
# baseline.py --epochs 2 --trials=2 --iterations=2 --dataset_dir=../Datasets

import torch, os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets


import torchvision
import torchvision.transforms as transforms

import os, glob, sys
import argparse

from models import *
from utils import progress_bar, make_prediction, create_dir
import torchvision.models as models

#0. Import needed libraries
import torchvision, torch
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt

#Step 1
from RandAugment import RandAugment

#1. Testing MitosisAugment
class MitosisAugment:
    def __init__(self, transforms_random, data_transforms):
        self.transforms_random = transforms_random
        self.data_transforms = data_transforms

    def __call__(self, img):
        img = self.data_transforms(img)

        r = self.transforms_random(img[0])
        q = self.transforms_random(img[1])
        s = self.transforms_random(img[2])
        t = self.transforms_random(img[3])

        x1 = torch.cat((r, q), 2)
        x2 = torch.cat((s, t), 2)

        img = torch.cat((x1, x2), 1)

        return transforms.ToPILImage()(img.squeeze_(0))
#         return img.ToPILImage()

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
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)

if not os.path.exists(args.dataset_dir):
    create_dir(args.dataset_dir)

dataset_list = sorted(glob.glob(args.dataset_dir + "/*"))
print("Dataset List: ", dataset_list)

if len(dataset_list) == 0:
    print("ERROR: 1. Add the Datasets to be run inside of the", args.dataset_dir, "folder")
    sys.exit()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(args.image_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Step 2
## Initialize `RandAugment` object
# transform_train.transforms.insert(0, RandAugment(3, 7))

# Testing new Augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.FiveCrop(112),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ]),
}

transform_test = transforms.Compose([
    transforms.RandomCrop(args.image_size, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#2. Add random transforms for MitosisAugment
rand_augs = transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(),
            transforms.GaussianBlur(3),
            transforms.RandomGrayscale(0.6),
            transforms.RandomInvert(0.6),
            # transforms.RandomPosterize(4, p = 0.7),
            transforms.RandomSolarize(45, p = 0.8),
            # transforms.RandomEqualize(0.9),
            transforms.RandomAdjustSharpness(0.8)
            ]), p=0.7)
mitosis = MitosisAugment(rand_augs, data_transforms['train'])
# transform_train.transforms.insert(0, mitosis)

# Training
def train(epoch, loader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, loader, current_exp):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + current_exp + 'ckpt.pth')
        best_acc = acc

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


for dataset in dataset_list:
    for iteration in range(args.iterations):

        current_dataset_file = dataset.split("/")[-1] + '_.txt'
        trainset = datasets.ImageFolder(os.path.join(dataset, 'train'),
                                              transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = datasets.ImageFolder(os.path.join(dataset, 'test'),
                                              transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        for trial in range(args.trials):
            print("Working on dataset: ", dataset, " in iteration ", iteration, " and model ", trial)
            current_exp = "_ite_" + str(iteration) + "_trial_" + str(trial) + "_dataset_" + dataset.split("/")[-1] + "_"

            # Model
            print('==> Building model..')

            if args.image_size == 32:
                net = ResNet18(num_classes=len(testset.classes))
            else:
                net = models.densenet161()
                net.classifier = nn.Linear(net.classifier.in_features, len(testset.classes))

            net = net.to(device)
            if device == 'cuda':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True

            # checkpoint = torch.load('./checkpoint/_ite_0_trial_1_dataset_10%_cifar10_2_classes_ckpt.pth')
            # # net = torch.nn.DataParallel(net)
            # net.load_state_dict(checkpoint['net'])

            # print(net)


            # net = net.module
            # net.linear = nn.Linear(net.linear.in_features, len(testset.classes))
            # net = torch.nn.DataParallel(net)
            # net = net.to(device)
            # # print(net)
            # for name, param in net.named_parameters():
            #     if "linear" in name:
            #         # print("LAST: ", name)
            #         param.requires_grad = True
            #     else:
            #         param.requires_grad = False

            # sys.exit()

            criterion = nn.CrossEntropyLoss()
            # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
            #                       momentum=0.9, weight_decay=5e-4)
            optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                  momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            best_acc = 0  # best test accuracy
            start_epoch = 0  # start from epoch 0 or last checkpoint epoch

            for epoch in range(args.epochs):
                train(epoch, trainloader)
                test(epoch, testloader, current_exp)
                scheduler.step()

                with open(current_dataset_file, 'a') as f:
                    if epoch + 1 == args.epochs:
                        checkpoint = torch.load('./checkpoint/' + current_exp + 'ckpt.pth')
                        net.load_state_dict(checkpoint['net'])
                        print("Test result for iteration ", iteration, " experiment: ", trial, "dataset", dataset, file = f)
                        print(make_prediction(net, testset.classes, testloader, 'save'), file = f)
                        print("Train result for iteration ", iteration, " experiment: ", trial, "dataset", dataset, file = f)
                        print(make_prediction(net, testset.classes, trainloader, 'save'), file = f)
