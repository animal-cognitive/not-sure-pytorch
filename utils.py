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
from models import *

import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image

# Allow to get the file paths of the loaded images
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPaths, self).__getitem__(index) + (self.imgs[index][0],)

def get_transforms():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform_train, transform_test

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
    trainset = ImageFolderWithPaths(os.path.join(dataset, 'train'), transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = ImageFolderWithPaths(os.path.join(dataset, 'test'), transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainset, trainloader, testset, testloader

def load_model_and_train_params(image_size, device, lr, testset):
    # Model
    print('==> Building model..')

    if image_size == 32:
        net = ResNet18(num_classes=len(testset.classes))
    else:
        net = models.densenet161()
        net.classifier = nn.Linear(net.classifier.in_features, len(testset.classes))

    net = net.to(device)
    if device == 'cuda':
        net = nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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

def make_prediction(net, class_names, loader):

    all_preds = torch.tensor([]).cuda()
    ground_truths = torch.tensor([]).cuda()
    net.eval()
    final_paths = [] # List for the file names tested

    for batch_idx, data in enumerate(loader):

        inputs, targets,paths = data # Based on ImageFolderWithPaths
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

def visualize_image(cam, rgb_img, target_category):
    """
    Visualize output for given image
    """

    input_tensor = preprocess_image(rgb_img)

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]

    output = cam.activations_and_grads(input_tensor)
    softmax = torch.nn.Softmax(dim = 1)

    print("PRED: ", softmax(output).tolist())

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

def load_self_pretrained_model(pretrained_model_path = './checkpoint/_ite_0_trial_0_dataset_10%_cifar10_2_classes_ckpt.pth', no_of_classes = 2):
    net = ResNet18(num_classes=no_of_classes)
    net = torch.nn.DataParallel(net)

    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu'))

    net.load_state_dict(checkpoint['net'])

    return net.module

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
    img = Image.fromarray(np.uint8(image_patch * 255)).convert('RGB')
    img.save(image_name)


def get_image_patch(cam, rgb_img, target_category=None, threshold=0.5):
    """
    Get the important part of the image
    """
    input_tensor = preprocess_image(rgb_img)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
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
        print(x1, y1, x2, y2)
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
