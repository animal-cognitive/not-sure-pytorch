# pytorch-cifar

*dataset_dir* is the directory where the dataset is saved. The dataset should be the only folder in the directory  
*epochs* Number of epochs to run  
*iterations* Number of experiments to run   
*trials* Number of trials in each experiment  

## Run the baseline with the following code
baseline.py --epochs 200 --trials=3 --iterations=1 --dataset_dir=../Datasets

## Run the not-sure with the following code
not_sure.py --epochs 200 --trials=3 --iterations=1 --dataset_dir=../Datasets   
   
### Remove the Not-Sure folder inside of the train and test dataset after running the not_sure.py file
