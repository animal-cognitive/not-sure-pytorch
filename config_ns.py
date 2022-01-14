lr = 0.1 #type=float, help='learning rate')
resume = False # help='resume from checkpoint')
model = "ResNet18" #type=str, help='model type (default: ResNet18)')
dataset_dir = '../KV_Mohammed' # type=str, help='The location of the dataset to be explored')
# dataset_dir = '../DS'
subset_folder = 'SUBSET_FOLDER' # type=str, help='Temporary location for dataset')
trials = 1 # type=int, help='Number of times to run the complete experiment')
iterations = 3 # type=int, help='Number of times to run the complete experiment')
epochs = 2 # type=int, help='Epochs')
name = '0' # type=str, help='name of run')
seed = 0 # type=int, help='random seed')
# batch_size = 128 # type=int, help='batch size')
batch_size = 16
no_augment = False # help='use standard augmentation (default: True)')
alpha = 1. # type=float, help='mixup interpolation coefficient (default: 1)')
image_size = 224 # type=int, help='input image size')
# image_size = 32
use_old = True # help='Use old code base')
ns_epochs = 2 # type=int, help='ns_epochs')
subset_train_iter = 1 # type=int, help='Iterations for the subset training model')
approach_list = [1, 2, 3, 4, 5] # Approaches to try
