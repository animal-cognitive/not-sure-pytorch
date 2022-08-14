lr = 0.1 #type=float, help='learning rate')
resume = False # help='resume from checkpoint')
model = "ResNet18" #type=str, help='model type (default: ResNet18)')
# dataset_dir = '../Kvasir - Aziz_224' # type=str, help='The location of the dataset to be explored')
# dataset_dir = '../DS1'
dataset_dir = '../Extra_Datasets'
subset_folder = 'SUBSET_FOLDER' # type=str, help='Temporary location for dataset')
trials = 2 # type=int, help='Number of times to run the complete experiment')
iterations = 4 # type=int, help='Number of times to run the complete experiment')
epochs = 200 # type=int, help='Epochs')
name = '0' # type=str, help='name of run')
seed = 0 # type=int, help='random seed')
batch_size = 128 # type=int, help='batch size')
# batch_size = 64
no_augment = False # help='use standard augmentation (default: True)')
alpha = 1. # type=float, help='mixup interpolation coefficient (default: 1)')
# image_size = 224 # type=int, help='input image size')
# image_size = 32
# image_size=96
image_size=28
use_old = True # help='Use old code base')
ns_epochs = 200 # type=int, help='ns_epochs')
subset_train_iter = 1 # type=int, help='Iterations for the subset training model')
#approach_list = [3, 4, 5] # Approaches to try
approach_list = [3, 4]

# Cut-out config
n_holes = 1 #help='number of holes to cut out from image')
length = 16 # help='length of the holes')

#Cutmix configs
beta = 0 # type=float, help='hyperparameter beta')
cutmix_prob = 0 # type=float, help='cutmix probability')
