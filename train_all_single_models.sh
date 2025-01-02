#!/bin/bash

# Here we trim down the number of selected k values compared to the number of k values in the manuscript

#========================CIFAR10========================

# CNNSmall
python train.py -gpu 0 --k 8 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR10
python train.py -gpu 0 --k 12 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR10
python train.py -gpu 0 --k 16 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR10
python train.py -gpu 0 --k 24 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR10
python train.py -gpu 0 --k 32 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR10
python train.py -gpu 0 --k 48 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR10
python train.py -gpu 0 --k 64 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR10
python train.py -gpu 0 --k 96 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR10
python train.py -gpu 0 --k 128 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR10
python train.py -gpu 0 --k 192 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR10
python train.py -gpu 0 --k 256 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR10
python train.py -gpu 0 --k 384 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR10
python train.py -gpu 0 --k 512 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR10


# CNNLarge
python train.py -gpu 0 --k 8 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR10
python train.py -gpu 0 --k 12 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR10
python train.py -gpu 0 --k 16 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR10
python train.py -gpu 0 --k 24 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR10
python train.py -gpu 0 --k 32 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR10
python train.py -gpu 0 --k 48 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR10
python train.py -gpu 0 --k 64 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR10
python train.py -gpu 0 --k 96 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR10
python train.py -gpu 0 --k 128 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR10
python train.py -gpu 0 --k 192 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR10
python train.py -gpu 0 --k 256 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR10
python train.py -gpu 0 --k 384 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR10
python train.py -gpu 0 --k 512 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR10


# ResNetSmall
python train.py -gpu 0 --k 8 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR10
python train.py -gpu 0 --k 12 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR10
python train.py -gpu 0 --k 16 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR10
python train.py -gpu 0 --k 24 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR10
python train.py -gpu 0 --k 32 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR10
python train.py -gpu 0 --k 48 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR10
python train.py -gpu 0 --k 64 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR10
python train.py -gpu 0 --k 96 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR10
python train.py -gpu 0 --k 128 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR10
python train.py -gpu 0 --k 192 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR10
python train.py -gpu 0 --k 256 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR10
python train.py -gpu 0 --k 384 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR10
python train.py -gpu 0 --k 512 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR10


# ResNetLarge
python train.py -gpu 0 --k 8 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR10
python train.py -gpu 0 --k 12 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR10
python train.py -gpu 0 --k 16 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR10
python train.py -gpu 0 --k 24 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR10
python train.py -gpu 0 --k 32 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR10
python train.py -gpu 0 --k 48 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR10
python train.py -gpu 0 --k 64 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR10
python train.py -gpu 0 --k 96 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR10
python train.py -gpu 0 --k 128 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR10
python train.py -gpu 0 --k 192 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR10
python train.py -gpu 0 --k 256 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR10
python train.py -gpu 0 --k 384 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR10
python train.py -gpu 0 --k 512 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR10




#========================CIFAR100========================

# CNNSmall
python train.py -gpu 0 --k 8 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR100
python train.py -gpu 0 --k 12 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR100
python train.py -gpu 0 --k 16 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR100
python train.py -gpu 0 --k 24 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR100
python train.py -gpu 0 --k 32 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR100
python train.py -gpu 0 --k 48 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR100
python train.py -gpu 0 --k 64 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR100
python train.py -gpu 0 --k 96 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR100
python train.py -gpu 0 --k 128 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR100
python train.py -gpu 0 --k 192 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR100
python train.py -gpu 0 --k 256 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR100
python train.py -gpu 0 --k 384 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR100
python train.py -gpu 0 --k 512 --seed 0 -bs 128 --n-epoch 100 -m CNNSmall -ds CIFAR100


# CNNLarge
python train.py -gpu 0 --k 8 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR100
python train.py -gpu 0 --k 12 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR100
python train.py -gpu 0 --k 16 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR100
python train.py -gpu 0 --k 24 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR100
python train.py -gpu 0 --k 32 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR100
python train.py -gpu 0 --k 48 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR100
python train.py -gpu 0 --k 64 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR100
python train.py -gpu 0 --k 96 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR100
python train.py -gpu 0 --k 128 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR100
python train.py -gpu 0 --k 192 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR100
python train.py -gpu 0 --k 256 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR100
python train.py -gpu 0 --k 384 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR100
python train.py -gpu 0 --k 512 --seed 0 -bs 128 --n-epoch 100 -m CNNLarge -ds CIFAR100


# ResNetSmall
python train.py -gpu 0 --k 8 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR100
python train.py -gpu 0 --k 12 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR100
python train.py -gpu 0 --k 16 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR100
python train.py -gpu 0 --k 24 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR100
python train.py -gpu 0 --k 32 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR100
python train.py -gpu 0 --k 48 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR100
python train.py -gpu 0 --k 64 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR100
python train.py -gpu 0 --k 96 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR100
python train.py -gpu 0 --k 128 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR100
python train.py -gpu 0 --k 192 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR100
python train.py -gpu 0 --k 256 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR100
python train.py -gpu 0 --k 384 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR100
python train.py -gpu 0 --k 512 --seed 0 -bs 128 --n-epoch 100 -m ResNetSmall -ds CIFAR100


# ResNetLarge
python train.py -gpu 0 --k 8 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR100
python train.py -gpu 0 --k 12 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR100
python train.py -gpu 0 --k 16 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR100
python train.py -gpu 0 --k 24 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR100
python train.py -gpu 0 --k 32 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR100
python train.py -gpu 0 --k 48 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR100
python train.py -gpu 0 --k 64 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR100
python train.py -gpu 0 --k 96 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR100
python train.py -gpu 0 --k 128 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR100
python train.py -gpu 0 --k 192 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR100
python train.py -gpu 0 --k 256 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR100
python train.py -gpu 0 --k 384 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR100
python train.py -gpu 0 --k 512 --seed 0 -bs 128 --n-epoch 100 -m ResNetLarge -ds CIFAR100
