import torch
from utils import *
import config_baseline as args
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange

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

    models_to_test = [
            # 'base',
            # 'appr3',
            'appr4'
            ]

    files_to_predict = [
        'accentor_s_000685.png',
        'accentor_s_000033.png',
        'alauda_arvensis_s_000781.png',
        'alauda_arvensis_s_001018.png',
        'bird_s_000574.png',
        'bird_s_001251.png',
        'kiwi_s_000045.png',
        'nandu_s_000735.png'
        ]
    correct_class = 0

    index = 0
    for path_to_model in models_to_test:

        net, criterion, optimizer, scheduler= load_model_and_train_params(args.image_size, device, args.lr, testset, args.use_old)
        # path_to_model = '_appr_4_ite_0_trial_0_dataset_10%_cifar10_with_ns_app1_ckpt'
        checkpoint = torch.load(path_to_model + '.pth')
        net.load_state_dict(checkpoint['net'])

        # Construct the GradCAM to use
        cam = construct_cam(net.module, [net.module.layer4[-1]], torch.cuda.is_available())

        for img_1_ in files_to_predict:

            img_h_ = img_w_ = args.image_size
            rgb_img_1 = normalize_img_to_rgb(img_1_, img_h_, img_w_)

            # get_image_patch(cam, rgb_img)
            targets = [ClassifierOutputTarget(correct_class)]

            input_tensor = preprocess_image(rgb_img_1)

            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            result = show_cam_on_image(rgb_img_1, grayscale_cam, use_rgb=True)

            save_image(result, f'{img_1_.split(".png")[0]}_{index}.png')

            # Create the metric target, often the confidence drop in a score of some category
            metric_target = ClassifierOutputSoftmaxTarget(correct_class)
            scores, batch_visualizations = CamMultImageConfidenceChange()(input_tensor,
              1 - grayscale_cam, targets, net, return_visualization=True)
            visualization = batch_visualizations[0].cpu().numpy().transpose((1, 2, 0))
            # visualization = deprocess_image(batch_visualizations[0, :])
            save_image(visualization, f'visual.png')
            print(f"The confidence increase percent: {100*scores[0]}")

            break
        index += 1
        pred_result = predict_given_images(net, transform_test, files_to_predict)
        print(f"PREDICTION FOR Model : {path_to_model} is {pred_result}")
