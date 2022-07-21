import torch
from utils import *
import config_baseline as args
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

use_cuda = torch.cuda.is_available()

device = 'cuda' if use_cuda else 'cpu'
if args.seed != 0:
    torch.manual_seed(args.seed)

dataset_list = check_dataset_dir(args.dataset_dir)

# Data
print('==> Preparing data..')
transform_train, transform_test = get_transforms(args.image_size)


for dataset in dataset_list:
    trainset, trainloader, testset, testloader = get_loaders_and_dataset(dataset, transform_train, transform_test, args.batch_size)

    # print(t)

    models_to_test = ['_ite_0_trial_0_dataset_10%_cifar10_ckpt', '_appr_3_ite_0_trial_0_dataset_10%_cifar10_with_ns_app1_ckpt', '_appr_4_ite_0_trial_0_dataset_10%_cifar10_with_ns_app1_ckpt']
    img_1_ = 'kiwi_s_000045.png'

    index = 0
    for path_to_model in models_to_test:

        net, criterion, optimizer, scheduler= load_model_and_train_params(args.image_size, device, args.lr, testset, args.use_old)
        # path_to_model = '_appr_4_ite_0_trial_0_dataset_10%_cifar10_with_ns_app1_ckpt'
        checkpoint = torch.load(path_to_model + '.pth')
        net.load_state_dict(checkpoint['net'])

        # Construct the GradCAM to use
        cam = construct_cam(net.module, [net.module.layer4[-1]], torch.cuda.is_available())

        img_h_ = img_w_ = 32
        rgb_img_1 = normalize_img_to_rgb(img_1_, img_h_, img_w_)

        # get_image_patch(cam, rgb_img)
        targets = [ClassifierOutputTarget(2)]

        input_tensor = preprocess_image(rgb_img_1)

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        result = show_cam_on_image(rgb_img_1, grayscale_cam, use_rgb=True)

        print("Result Shape: ", result.size)

        save_image(result, f'sample{index}.png')

        index += 1
