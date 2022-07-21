'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import shutil, random, os, time, copy, pickle, glob, torch, cv2, sys, math, csv

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torchvision import datasets, transforms

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import models_new as models_new
import models
import torchvision.models as trained_models
from torchvision.utils import save_image as torch_save_image
from skimage import io
import cv2 as cv


import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torch.autograd import Variable

import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


# Allow to get the file paths of the loaded images
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPaths, self).__getitem__(index) + (self.imgs[index][0],)

# Load Dataset loader from file list
class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, imagelist, labels, transform):
        self.imagepaths = imagelist
        self.transform = transform
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, index):
        self.imagepath = self.imagepaths[index]
        self.image = Image.open(self.imagepath)
        self.label = self.labels[index]
        self.i = self.transform(self.image)

        return (self.i, self.label.item(), self.imagepath)



def get_transforms(img_size = None, rand_aug = False):
    if not img_size:
        img_size = 32

    if rand_aug:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform_train, transform_test

def rand(max):
    return random.randint(0,max - 1)

def predict_given_images(net, transform, files):
    predictions = []
    for file in files:
        inp = transform(Image.open(file)).unsqueeze(0)
        inp = net(inp)
        pred_class = np.argmax(inp.cpu().detach().numpy(), axis = -1).item()
        predictions.append(pred_class)

    return predictions

def get_predicted_files(targets, preds, file_paths, class_names):
    file_pred_list = [[[] for item in range(len(class_names))] for item in range(len(class_names))]

    for target_index in range(len(targets)):
        target_value = targets[target_index]
        predicted_value = preds[target_index]
        file_pred_list[target_value][predicted_value].append(file_paths[target_index])

    return file_pred_list

# Methods to help with creating directories

def delete_dir_if_exists(directory):
    """
    Remove a directory if it exists

    dir - Directory to remove
    """

    if os.path.exists(directory): # If directory exists
        shutil.rmtree(directory) # Remove it

def create_dir(directory, delete_already_existing = False):
    """
    Create directory. Deletes and recreate directory if already exists

    Parameter:
    string - directory - name of the directory to create if it does not already exist
    delete_already_existing - Delete directory if already existing
    """

    if delete_already_existing: # If delete directory even if it exists
        delete_dir_if_exists(directory) # Delete directory even if it exists
        os.makedirs(directory) # Create a new directory

    else:
        if not os.path.exists(directory): # If directory does not exist
            os.makedirs(directory) # Create new directory

def empty_dir(dir_name):
    """
    Remove all files from given directory

    - dir_name - name of directory to remove files from
    """
    for f in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, f))

def copy_all_files(src_dir, dest_dir):
    """
    Copies all files from source directory to destination directory

    - src_dir - the source directory to copy from
    - dest_dir - the destination directory to copy into

    source - https://www.geeksforgeeks.org/copy-all-files-from-one-directory-to-another-using-python/
    """

    files = os.listdir(src_dir) # List the files to be copied

    shutil.copytree(src_dir, dest_dir) # Copy list of files to destination folder

def copy_to_other_dir(from_dir, to_dir):
    """
    Copy the content of the from_dir directory into the to_dir directory

    from_dir: Directory we are copying its content
    to_dir: Directory we are copying into
    """

    shutil.copytree(from_dir, to_dir)

def get_loaders_and_dataset(dataset, transform_train, transform_test, batch_size):

    testset = ImageFolderWithPaths(os.path.join(dataset, 'test'), transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    trainset = ImageFolderWithPaths(os.path.join(dataset, 'train'), transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainset, trainloader, testset, testloader

def load_model_and_train_params(image_size, device, lr, testset, old, cut_out = False):
    # Model

    weight_decay = 1e-4
    print('==> Building model..')
    if image_size == 32:
        if old:
            net = models.__dict__['ResNet18'](num_classes=len(testset.classes))
        else:
            weight_decay = 5e-4
            net = models_new.ResNet18(num_classes=len(testset.classes))

        net = net.to(device)
        if image_size == 32:
            ### Update the number of classes to predict
            num_ftrs = net.linear.in_features
            # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
            net.linear = nn.Linear(num_ftrs, len(testset.classes))
    else:
        net = trained_models.resnet18(pretrained = True)
        net.fc = nn.Linear(net.fc.in_features, len(testset.classes))

    if device == 'cuda':
        net.cuda()
        net = nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0.9, weight_decay=weight_decay)

    if old:
        scheduler = None
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    if cut_out:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    return net, criterion, optimizer, scheduler

def load_current_model_and_train_params(net, lr, old, cut_out = False):
    # Model

    weight_decay = 1e-4
    print('==> Loading existing model..')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0.9, weight_decay=weight_decay)

    if old:
        scheduler = None
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if cut_out:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    return net, criterion, optimizer, scheduler

def count_false_positives_for_given_class(class_list, class_c):
    """
    Count the False positives for the given class
    """

    false_positive_count = 0

    for item in range(len(class_list)):
        if item != class_c:
            false_positive_count += class_list[item]

    # Return the false positive count
    return false_positive_count

def get_false_positives_for_given_class(class_lists, class_c):
    """
    Get the False positives for the given class
    """

    false_positives = []

    for item in range(len(class_lists)):
        if item != class_c:
            false_positives.extend(class_lists[item])

    # Return the false positive count
    return false_positives

def check_dataset_dir(dataset_dir):
    """
    Check the directory that holds the dataset
    """

    if not os.path.exists(dataset_dir):
        create_dir(dataset_dir)

    dataset_list = sorted(glob.glob(dataset_dir + "/*"))
    print("Dataset List: ", dataset_list)

    if len(dataset_list) == 0:
        print("ERROR: 1. Add the Datasets to be run inside of the", dataset_dir, "folder")
        sys.exit()

    return dataset_list

def calculate_total_false_positives(conf_matrix):

  """
  Count the total number of False Positives from all classes
  """

  false_positive_count = 0 # Hold the count for false positives

  # Iterate over the classes
  for index in range(len(conf_matrix[0])):

    # For predictions made for each class
    per_class_prediction = conf_matrix[index]

    # Count false positive for given class
    false_positive_count += count_false_positives_for_given_class(per_class_prediction, index)


  # Return the total false positive counts
  return false_positive_count

def get_list_of_classes_GT_UFPE(class_list, class_c, UFPE_c):
    """
    Get the list of classes with False Positives greater than UFPE_i
    """

    list_c = []

    for item in range(len(class_list)):

        # Get the classes whose false positives is greater than UFPE_i
        if item != class_c and class_list[item] >= UFPE_c:
            list_c.append(item)

    # Return the list
    return list_c

def count_false_positives_within_list(class_predictions, LIST_c):
    """
    Count the number of false positives from classes in the list LIST_c

    LIST_c: List of classes whose predictions in class_predictions we are interested in
    """

    false_positive_count = 0

    for item in range(len(class_predictions)):
        if item in LIST_c:
            false_positive_count += class_predictions[item]

    # Return the false positive count
    return false_positive_count

def make_prediction(net, thisLoader):

    all_preds = torch.tensor([]).cuda()
    ground_truths = torch.tensor([]).cuda()
    net.eval()
    final_paths = [] # List for the file names tested

    for batch_idx, data in enumerate(thisLoader):

        inputs, targets, paths = data # Based on ImageFolderWithPaths
        final_paths.extend(paths) # Add to paths

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)

            ground_truths = torch.cat((ground_truths, targets), dim=0)

            all_preds = torch.cat((all_preds, predicted), dim=0)

    targets = [int(x) for x in ground_truths.tolist()]
    preds = [int(x) for x in all_preds.tolist()]
    return targets, preds, final_paths

def create_subset_dataset(subset_dataset):
    """
    Create the subset dataset folder setup
    """

    #Create the train and test subfolders
    create_dir(subset_dataset + "/train", delete_already_existing = True)
    create_dir(subset_dataset + "/test", delete_already_existing = True)

# Splits a given list into number_of_splits partitions.
def split_into_k(list_to_split, number_of_splits):
    """
    Split a list into a given number of splits

    list_to_split - the list to be splitted
    number_of_splits - the number of splits to create
    """

    # The size of each split
    range_to_use = int(len(list_to_split) / number_of_splits)

    # Result list
    result_list = []

    # Iterate over each split and create a list for each split
    for split in range(1, number_of_splits + 1):
        list_to_create = list_to_split[range_to_use * (split - 1) : range_to_use * split]

        result_list.append(list_to_create)

    return result_list

def split_dataset(dataset, no_of_splits):
    # Split data into splits
    dataset_train = dataset + '/train/'

    class_names = [item.split('/')[-1] for item in glob.glob(dataset_train + '/*')]
    class_names.sort()

    # Split the training data based on the no_of_splits
    splits_per_class = []

    # Iterate over each of the classes
    for training_class in class_names:

      # Split each class into no_of_splits splits
      split_list = split_into_k(sorted(glob.glob(dataset_train + training_class + "/*")), no_of_splits)

      # Save the split in another list
      splits_per_class.append(split_list)

    return splits_per_class, class_names

def collect_train_and_val_data(splits_per_class, split_index):
    """
    Collect validation and training data from split according to split_index

    Normally, we collect 1 split for training and the remaining for validation

    - splits_per_class - list containing the splits
    - split_index - index to get for training and remaining for validation
    """

    train_list = []
    full_val = []
    for item in range(len(splits_per_class)):
        val = []
        for list_items in range(len(splits_per_class[item])):
            if list_items == split_index:
                train_list.append(splits_per_class[item][list_items])
            else:
                val.extend(splits_per_class[item][list_items])

        full_val.append(val)

    return train_list, full_val

def copy_images_to_subset_dataset(subset_dataset, dataset, train, val, class_names):
    dataset_train = dataset + "/train/"

    subset_train = subset_dataset + '/train/'
    subset_test = subset_dataset + '/test/'

    # For each class in class_names
    for index, class_name in enumerate(class_names):

        # Directories we need to copy into
        train_dir_to_copy_into = subset_train + class_name
        val_dir_to_copy_into = subset_test + class_name

        # Directory to copy from
        dir_to_copy_from = dataset_train + class_name

        training, validation = train[index], val[index]

        # Create directory if not exist
        create_dir(train_dir_to_copy_into, False)
        create_dir(val_dir_to_copy_into, False)

        # Copy files into appropriate directory
        for fname in training:
            srcpath = os.path.join("", fname)
            shutil.copy(srcpath, train_dir_to_copy_into)

        for fname in validation:
            srcpath = os.path.join("", fname)
            shutil.copy(srcpath, val_dir_to_copy_into)

def construct_cam(model, target_layers, use_cuda):
    """
    Construct cam for the given model
    """
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    return cam

def visualize_image(cam, rgb_img, target_category, transform):
    """
    Visualize output for given image
    """

    input_tensor = preprocess_image(rgb_img)

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]

    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return visualization

def normalize_img_to_rgb(img_1_, img_h_, img_w_):
    """
    Normalize the given image
    """
    rgb_img_1 = cv2.imread(img_1_, 1)[:, :, ::-1]
    rgb_img_1 = cv2.resize(rgb_img_1, (img_h_, img_w_))
    rgb_img_1 = np.float32(rgb_img_1) / 255.

    return rgb_img_1

def load_trained_res18_model(pretrained_model_path, class_names, device, image_size = 32, old=True):
    if old:
        net = models.__dict__['ResNet18'](num_classes=len(class_names))
    else:
        weight_decay = 5e-4
        net = models_new.ResNet18(num_classes=len(class_names))

    net = net.to(device)
    if image_size == 32:
        ### Update the number of classes to predict
        num_ftrs = net.linear.in_features
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        net.linear = nn.Linear(num_ftrs, len(class_names))

    net = torch.nn.DataParallel(net)

    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu'))

    net.load_state_dict(checkpoint['net'])

    return net

def load_self_pretrained_model(pretrained_model_path = './checkpoint/_ite_0_trial_0_dataset_10%_cifar10_2_classes_ckpt.pth', no_of_classes = 2):
    net = ResNet18(num_classes=no_of_classes)
    net = torch.nn.DataParallel(net)

    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu'))

    net.load_state_dict(checkpoint['net'])

    return net.module

def load_self_pretrained_model_v2(net, pretrained_model_path = './checkpoint/_ite_0_trial_0_dataset_10%_cifar10_2_classes_ckpt.pth'):

    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cuda'))

    net.load_state_dict(checkpoint['net'])

    return net

def grayscale_to_3d(grayscale_cam):
    """
    Convert the Grayscale CAM to 3D
    """
    grayscale_cam_3d = np.reshape(grayscale_cam, (grayscale_cam.shape[0], grayscale_cam.shape[1], 1))
    grayscale_cam_3d = np.concatenate([grayscale_cam_3d, grayscale_cam_3d, grayscale_cam_3d], axis=-1)
    return grayscale_cam_3d

def save_image(image_patch, image_name='Test_image.jpg'):
    """
    Save the given image
    """
    imageRGB = cv.cvtColor(np.uint8(image_patch * 255), cv.COLOR_RGB2BGR)

    img = Image.fromarray(imageRGB)
    img.save(image_name)


def get_image_patch(cam, rgb_img, target_category=None, threshold=0.5):
    """
    Get the important part of the image
    """
    input_tensor = preprocess_image(rgb_img)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]

    # Normalize
    grayscale_cam = grayscale_cam / (np.max(grayscale_cam) - np.min(grayscale_cam))
    grayscale_cam = np.where(grayscale_cam > threshold, grayscale_cam, 0)

    # Reshape
    grayscale_cam_3d = grayscale_to_3d(grayscale_cam)

    # output image patch
    image_patch = grayscale_cam_3d * rgb_img

    return image_patch, grayscale_cam

def get_patch_coordinates(grayscale_cam, min_val=0.5):
    """
    Get a rectangle coordinate of the grayscale cam
    """
    grayscale_cam = np.where(grayscale_cam > min_val, 1, 0)
    min_row, min_col, max_row, max_col = 0, 0, 0, 0

    for row in range(grayscale_cam.shape[0]):
        if 1 in grayscale_cam[row, :]:
            min_row = row
            break

    for row in range(grayscale_cam.shape[0]-1, -1, -1):
        if 1 in grayscale_cam[row, :]:
            max_row = row
            break

    for col in range(grayscale_cam.shape[1]):
        if 1 in grayscale_cam[:, col]:
            min_col = col
            break

    for col in range(grayscale_cam.shape[1]-1, -1, -1):
        if 1 in grayscale_cam[:, col]:
            max_col = col
            break

    return [min_row, min_col, max_row, max_col]

def get_centers(img_coordinates):
    """
    Get the center of the given coordinate
    """
    min_row, min_col, max_row, max_col = img_coordinates

    center_row = int((max_row + min_row) / 2)
    center_col = int((max_col + min_col) / 2)
    row_diameter = int((max_row - min_row) / 2)
    col_diameter = int((max_col - min_col) / 2)

    return [center_row, center_col, row_diameter, col_diameter]

def smooth_img_mix(img_1, img_2, grayscale_cam_1):
    """
    Mixes two images together using the important part of image 1 with the unimportant part of image 2
    """
    mask_for_img_1 = np.zeros_like(img_1)
    mask_for_img_1[:, :, 0] = grayscale_cam_1
    mask_for_img_1[:, :, 1] = grayscale_cam_1
    mask_for_img_1[:, :, 2] = grayscale_cam_1

    # Get the important part from img_1
    img_1 = img_1 * mask_for_img_1

    # Replace the unimportant part of img_1 with parts from img_2
    img_2 = img_2 * (1 - mask_for_img_1)

    # Merge the two images according to gradient density
    output_image = img_1 + img_2

    return output_image

def remove_patch(img, grayscale_cam):
    """
    Remove the important part of the given image
    """
    # Reflect the heatmap
    grayscale_cam = 1 - grayscale_cam

    # Reshape
    grayscale_cam_3d = grayscale_to_3d(grayscale_cam)

    # Remove the image patch
    output_img = img * grayscale_cam_3d

    return output_img


def recenter_patches(img_1, grayscale_cam_1, img_1_coordinates, img_h, img_w, corner):
    """
    """
    min_row, min_col, max_row, max_col = img_1_coordinates
    center_row_1, center_col_1, row_diameter_1, col_diameter_1 = get_centers(img_1_coordinates)

    img_1_shifted = np.zeros_like(img_1)
    grayscale_cam_1_shifted = np.zeros_like(grayscale_cam_1)

    if corner == 1:
        x1 = 0
        x2 = max_row - min_row
        y1 = 0
        y2 = max_col - min_col
        img_1_shifted[x1:x2, y1:y2, :] = img_1[min_row:max_row, min_col:max_col, :]
        grayscale_cam_1_shifted[x1:x2, y1:y2] = grayscale_cam_1[min_row:max_row, min_col:max_col]

    elif corner == 2:
        x1 = img_h - (max_row - min_row)
        x2 = img_h
        y1 = 0
        y2 = max_col - min_col
        img_1_shifted[x1:x2, y1:y2, :] = img_1[min_row:max_row, min_col:max_col, :]
        grayscale_cam_1_shifted[x1:x2, y1:y2] = grayscale_cam_1[min_row:max_row, min_col:max_col]

    elif corner == 3:
        x1 = 0
        x2 = max_row - min_row
        y1 = img_w - (max_col - min_col)
        y2 = img_w
        img_1_shifted[x1:x2, y1:y2, :] = img_1[min_row:max_row, min_col:max_col, :]
        grayscale_cam_1_shifted[x1:x2, y1:y2] = grayscale_cam_1[min_row:max_row, min_col:max_col]

    elif corner == 4:
        x1 = img_h - (max_row - min_row)
        x2 = img_h
        y1 = img_w - (max_col - min_col)
        y2 = img_w
        img_1_shifted[x1:x2, y1:y2, :] = img_1[min_row:max_row, min_col:max_col, :]
        grayscale_cam_1_shifted[x1:x2, y1:y2] = grayscale_cam_1[min_row:max_row, min_col:max_col]

    return img_1_shifted, grayscale_cam_1_shifted

def get_rgbs_grayscale_coordinates(cam, img_1_, img_2_, img_h_, img_w_, threshold_, min_val_):
    """
    Return RGBs, Grayscale CAMs and Coordinates
    """
    # Read and normalize image 1
    rgb_img_1 = normalize_img_to_rgb(img_1_, img_h_, img_w_)
    rgb_img_2 = normalize_img_to_rgb(img_2_, img_h_, img_w_)

    # Get Grayscale
    _, grayscale_cam_1_ = get_image_patch(cam,
                                      rgb_img=rgb_img_1,
                                      target_category=None,
                                      threshold=threshold_)

    _, grayscale_cam_2 = get_image_patch(cam,
                                         rgb_img=rgb_img_2,
                                         target_category=None,
                                         threshold=threshold_)
    # Get coordinates
    img_1_coordinates_ = get_patch_coordinates(grayscale_cam_1_, min_val=min_val_)
    img_2_coordinates_ = get_patch_coordinates(grayscale_cam_2, min_val=min_val_)

    return rgb_img_1, rgb_img_2, grayscale_cam_1_, grayscale_cam_2, img_1_coordinates_, img_2_coordinates_

def approach_1(img_1_, img_2_, img_h_, img_w_, cam, filename, threshold_, min_val_):

    rgb_img_1, rgb_img_2, grayscale_cam_1_, grayscale_cam_2, img_1_coordinates_, img_2_coordinates_ = get_rgbs_grayscale_coordinates(cam, img_1_, img_2_, img_h_, img_w_, threshold_, min_val_)

    # Remove important part of image 2
    rgb_img_2_no_patch = remove_patch(img=rgb_img_2, grayscale_cam=grayscale_cam_2)

    # Add important part of image 1 with unimportant part of image 2
    output_img_ = smooth_img_mix(img_1=rgb_img_1,
                                 img_2=rgb_img_2_no_patch,
                                 grayscale_cam_1=grayscale_cam_1_[:, :])

    save_image(output_img_, filename + ".png")

def approach_2(img_1_, img_2_, img_h_, img_w_, cam, filename, threshold_, min_val_):
    rgb_img_1, rgb_img_2, grayscale_cam_1_, grayscale_cam_2, img_1_coordinates_, img_2_coordinates_ = get_rgbs_grayscale_coordinates(cam, img_1_, img_2_, img_h_, img_w_, threshold_, min_val_)

    center_row_2, center_col_2, _, _ = get_centers(img_2_coordinates_)

    img_half = img_h_ / 2
    if center_row_2 < img_half:
        if center_col_2 < img_half:
            # important concept of image 2 is in the upper left corner
            corner = 4
        else:
            # important concept of image 2 is in the lower left corner
            corner=2

    else:
        if center_col_2 < img_half:
            # important concept of image 2 is in the upper right corner
            corner=3
        else:
            # important concept of image 2 is in the lower right corner
            corner=1

    img_1_shifted_, mask_1_shifted = recenter_patches(img_1=rgb_img_1,
                                                              grayscale_cam_1=grayscale_cam_1_,
                                                              img_1_coordinates=img_1_coordinates_,
                                                              img_h=img_h_,
                                                              img_w=img_w_,
                                                              corner=corner)

    output_img_ = smooth_img_mix(img_1=img_1_shifted_,
                                 img_2=rgb_img_2,
                                 grayscale_cam_1=mask_1_shifted)

    save_image(output_img_, filename + ".png")

def approach_3(img_1_, img_2_, img_h_, img_w_, file_path):

    img1 = Image.open(img_1_)
    img2 = Image.open(img_2_)

    img1 = np.asarray(img1.resize((img_h_, img_w_)))
    img2 = np.asarray(img2.resize((img_h_, img_w_)))

    mean_img = np.mean([img1, img2], axis=0)
    img = Image.fromarray(np.uint8(mean_img))

    img.save(file_path)

def approach_4(img_1_, img_2_, img_h_, img_w_, file_path, count):

    data_split_transform = transforms.Compose([
        transforms.FiveCrop(int(img_h_ / 2)),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    img1 = data_split_transform(Image.open(img_1_))
    img2 = data_split_transform(Image.open(img_2_))

    max = img1.shape[0]

    first = rand(max)
    second = rand(max)
    third = rand(max)
    fourth = rand(max)

    if count % 2 == 0:
        x1 = torch.cat((img1[first], img2[second]), 2)
        x2 = torch.cat((img2[third], img1[fourth]), 2)
    else:
        x1 = torch.cat((img1[first], img1[second]), 2)
        x2 = torch.cat((img2[third], img2[fourth]), 2)

    x = torch.cat((x1, x2), 1)

    x = x.numpy().transpose((1, 2, 0))
    x = np.clip(x, 0, 1)

    torch_save_image(torch.from_numpy(x.transpose(2, 0, 1)), file_path)

def approach_5(img_1_, img_2_, img_h_, img_w_, cam, filename, threshold_, min_val_, count):
    """
    Get the important part of the two images, but only get the 16*16 section of it
    Mix approach 2 and 4 together
    """

    rgb_img_1, rgb_img_2, grayscale_cam_1_, grayscale_cam_2, img_1_coordinates_, img_2_coordinates_ = get_rgbs_grayscale_coordinates(cam, img_1_, img_2_, img_h_, img_w_, threshold_, min_val_)

    center_row1, center_col1, row_diameter1, col_diameter1 = get_centers(img_1_coordinates_)
    center_row2, center_col2, row_diameter2, col_diameter2 = get_centers(img_2_coordinates_)

    # Compute min and max rows and cols
    min_row1, max_row1 = get_correct_min_max_row_or_col(center_row1, img_h_)
    min_col1, max_col1 = get_correct_min_max_row_or_col(center_col1, img_w_)

    min_row2, max_row2 = get_correct_min_max_row_or_col(center_row2, img_h_)
    min_col2, max_col2 = get_correct_min_max_row_or_col(center_col2, img_w_)

    # Ensure center is away from the edges
    img1 = rgb_img_1[min_row1:max_row1, min_col1:max_col1, :]
    img2 = rgb_img_2[min_row2:max_row2, min_col2:max_col2, :]

    if count % 3 == 0:
        x1 = torch.cat((torch.from_numpy(img1), torch.from_numpy(img1)))
        x2 = torch.cat((torch.from_numpy(img2), torch.from_numpy(img2)))
    elif count % 3 == 1:
        x1 = torch.cat((torch.from_numpy(img1), torch.from_numpy(img2)))
        x2 = torch.cat((torch.from_numpy(img2), torch.from_numpy(img1)))
    elif count % 3 == 2:
        x1 = torch.cat((torch.from_numpy(img1), torch.from_numpy(img2)))
        x2 = torch.cat((torch.from_numpy(img1), torch.from_numpy(img2)))
    x = torch.cat((x1, x2), 1)
    save_image(x, filename)

def get_correct_min_max_row_or_col(center_row1, img_h_):
    needed_h_radius = img_h_ / 4

    min_row = center_row1 - needed_h_radius
    max_row = center_row1 + needed_h_radius

    if min_row < 0:
        min_row = 0
        max_row = needed_h_radius * 2

    if max_row > img_h_:
        max_row = img_h_ - 1
        min_row = max_row - (needed_h_radius * 2)

    return int(min_row), int(max_row)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def adjust_learning_rate(lr, optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_for_cutmix(learning_rate, optimizer, epoch, epochs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // (epochs * 0.5))) * (0.1 ** (epoch // (epochs * 0.75)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def run_experiment(trainloader, testloader, current_exp, epochs, net, optimizer, scheduler, best_acc, criterion, device, learning_rate, iteration = None, trial = None, dataset = None, classes = None, current_dataset_file = None):

    metrics = []
    for epoch in range(epochs):
        train_model(net, epoch, trainloader, optimizer, criterion, device, scheduler, learning_rate)
        best_acc = test_model(net, epoch, testloader, current_exp, best_acc, criterion, device)

        if current_dataset_file:
            with open(current_dataset_file, 'a') as f:
                if epoch + 1 == epochs:
                    checkpoint = torch.load('./checkpoint/' + current_exp + 'ckpt.pth')
                    net.load_state_dict(checkpoint['net'])
                    if iteration is not None and trial is not None and dataset is not None and classes is not None:
                        metrics = [dataset, trial, iteration]
                        print("Test result for iteration ", iteration, " experiment: ", trial, "dataset", dataset, file = f)
                        targets, preds, _ = make_prediction(net, testloader)
                        test_class_report = classification_report(targets, preds, target_names=classes)
                        test_metrics = get_metrics_from_classi_report(test_class_report)
                        metrics.extend(test_metrics)
                        print(test_class_report, file = f)
                        print("Train result for iteration ", iteration, " experiment: ", trial, "dataset", dataset, file = f)
                        targets, preds, _ = make_prediction(net, trainloader)
                        train_class_report = classification_report(targets, preds, target_names=classes)
                        train_metrics = get_metrics_from_classi_report(train_class_report)
                        metrics.extend(train_metrics)
                        print(train_class_report, file = f)

    return best_acc, metrics

def process_result(net, trainloader, testloader, classes, current_dataset_file, epoch, epochs, current_exp, dataset, trial, iteration, metrics):
    with open(current_dataset_file, 'a') as f:
        if epoch + 1 == epochs:
            checkpoint = torch.load('./checkpoint/' + current_exp + 'ckpt.pth')
            net.load_state_dict(checkpoint['net'])
            if iteration is not None and trial is not None and dataset is not None and classes is not None:
                metrics = [dataset, trial, iteration]
                print("Test result for iteration ", iteration, " experiment: ", trial, "dataset", dataset, file = f)
                targets, preds, _ = make_prediction(net, testloader)
                test_class_report = classification_report(targets, preds, target_names=classes)
                test_metrics = get_metrics_from_classi_report(test_class_report)
                metrics.extend(test_metrics)
                print(test_class_report, file = f)
                print("Train result for iteration ", iteration, " experiment: ", trial, "dataset", dataset, file = f)
                targets, preds, _ = make_prediction(net, trainloader)
                train_class_report = classification_report(targets, preds, target_names=classes)
                train_metrics = get_metrics_from_classi_report(train_class_report)
                metrics.extend(train_metrics)
                print(train_class_report, file = f)
    return metrics

def run_experiment_mixup_approach(trainloader, testloader, current_exp, epochs, net, optimizer, scheduler, best_acc, criterion, device, learning_rate, alpha = 1, iteration = None, trial = None, dataset = None, classes = None, current_dataset_file = None):

    metrics = []
    for epoch in range(epochs):
        train_mixup_approach(net, epoch, trainloader, criterion, optimizer, device, alpha)
        best_acc = test_mixup_approach(net, epoch, testloader, criterion, best_acc, device, current_exp)
        adjust_learning_rate(learning_rate, optimizer, epoch)

    if current_dataset_file:
        metrics = process_result(net, trainloader, testloader, classes, current_dataset_file, epoch, epochs, current_exp, dataset, trial, iteration, metrics)

    return best_acc, metrics

# Training
def train_model(net, epoch, loader, optimizer, criterion, device, scheduler, learning_rate):
    print('\nEpoch: %d' % epoch)
    # net.to(device)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, data in enumerate(loader):
        inputs, targets, file_paths = data
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()
        else:
            adjust_learning_rate(learning_rate, optimizer, epoch)

        train_loss += loss.data.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test_model(net, epoch, loader, current_exp, best_acc, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            inputs, targets, file_paths = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
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
        torch.save(state, f'./checkpoint/{current_exp}ckpt.pth')
        best_acc = acc

    return best_acc

def mixup_data(x, y, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_mixup_approach(net, epoch, trainloader, criterion, optimizer, device, alpha):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, data in enumerate(trainloader):
        inputs, targets, file_paths = data
        inputs, targets = inputs.to(device), targets.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       alpha, device)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)

def run_experiment_cutmix_approach(trainloader, testloader, current_exp, epochs, net, optimizer, scheduler, best_acc, criterion, device, learning_rate, beta, cutmix_prob, iteration = None, trial = None, dataset = None, classes = None, current_dataset_file = None):

    metrics = []
    for epoch in range(epochs):
        train_cutmix_approach(net, epoch, trainloader, criterion, optimizer, device, beta, cutmix_prob)

        best_acc = test_model(net, epoch, testloader, current_exp, best_acc, criterion, device)
        adjust_learning_rate_for_cutmix(learning_rate, optimizer, epoch, epochs)

    if current_dataset_file:
        metrics = process_result(net, trainloader, testloader, classes, current_dataset_file, epoch, epochs, current_exp, dataset, trial, iteration, metrics)

    return best_acc, metrics

def train_cutmix_approach(net, epoch, trainloader, criterion, optimizer, device, beta, cutmix_prob):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    lam = 0
    for batch_idx, data in enumerate(trainloader):
        inputs, targets, file_paths = data
        inputs, targets = inputs.to(device), targets.to(device)

        r = np.random.rand(1)
        if beta > 0 and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            outputs = net(inputs)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        else:
            # compute output
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def test_mixup_approach(net, epoch, testloader, criterion, best_acc, device, current_exp):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, data in enumerate(testloader):
        inputs, targets, file_paths = data

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    acc = 100.*correct/total
    if acc > best_acc:
        checkpoint(net, acc, epoch, current_exp)
        best_acc = acc
    return best_acc


def checkpoint(net, acc, epoch, current_exp):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + current_exp + "ckpt.pth")

def get_metrics_from_classi_report(classification_report):
    """
    Get accuracy, precision, recall and F1 scores from classification report
    """

    metrics_rows = classification_report.split("\n")[-4:-1]

    accuracy = [i for i in metrics_rows[0].split(" ") if i][1]
    accuracy = float(accuracy)

    y = [i for i in metrics_rows[1].split(" ") if i]

    precision, recall, f1_score = float(y[2]), float(y[3]), float(y[4])

    return accuracy, precision, recall, f1_score

def mix(cam, images_from_c, images_from_c_hat, image_size, dataset_dir, threshold_, min_val_, approach_list):
    if images_from_c and images_from_c_hat:
        for i in range(len(images_from_c_hat)):
            img_1 = images_from_c_hat[i]

            img_2 = images_from_c[i]

            for approach in approach_list:
                file_path = dataset_dir + f"/approach_{approach}/Not_Sure/" + img_1.split("/")[-1].split(".")[0] + "_" + img_2.split("/")[-1].split(".")[0] + ".jpg"
                if approach == 1:
                    approach_1(img_1, img_2, image_size, image_size, cam, file_path, threshold_, min_val_)
                elif approach == 2:
                    approach_2(img_1, img_2, image_size, image_size, cam, file_path, threshold_, min_val_)
                elif approach == 3:
                    approach_3(img_1, img_2, image_size, image_size, file_path)
                elif approach == 4:
                    approach_4(img_1, img_2, image_size, image_size, file_path, i)
                elif approach == 5:
                    approach_5(img_1, img_2, image_size, image_size, cam, file_path, threshold_, min_val_, i)

                else:
                    approach_1(img_1, img_2, image_size, image_size, cam, file_path, threshold_, min_val_)
                    approach_2(img_1, img_2, image_size, image_size, cam, file_path, threshold_, min_val_)
