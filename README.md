# pytorch-cifar

### dataset_dir is the directory where the dataset is saved. The dataset should be the only folder in the directory
### epochs Number of epochs to run
### iterations Number of experiments to run
### trails Number of trials in each experiment

## Run the baseline with the following code
baseline.py --epochs 2 --trials=2 --iterations=2 --dataset_dir=../Datasets

## Run the not-sure with the following code
not_sure.py --epochs 2 --trials=2 --iterations=2 --dataset_dir=../Datasets
