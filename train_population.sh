#!/bin/bash

# Here we trim down the number of selected k values compared to the number of k values in the manuscript

#========================CIFAR10========================

# CNNSmall
for seed in {1..100}
do
   python train.py -gpu 0 --k 10 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m CNNSmall -ds CIFAR10
   python train.py -gpu 0 --k 20 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m CNNSmall -ds CIFAR10
   python train.py -gpu 0 --k 40 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m CNNSmall -ds CIFAR10
done

# CNNLarge
for seed in {1..100}
do
   python train.py -gpu 0 --k 10 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m CNNLarge -ds CIFAR10
   python train.py -gpu 0 --k 20 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m CNNLarge -ds CIFAR10
   python train.py -gpu 0 --k 40 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m CNNLarge -ds CIFAR10
done

# ResNetSmall
for seed in {1..100}
do
   python train.py -gpu 0 --k 10 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m ResNetSmall -ds CIFAR10
   python train.py -gpu 0 --k 20 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m ResNetSmall -ds CIFAR10
   python train.py -gpu 0 --k 40 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m ResNetSmall -ds CIFAR10
done

# ResNetLarge
for seed in {1..100}
do
   python train.py -gpu 0 --k 10 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m ResNetLarge -ds CIFAR10
   python train.py -gpu 0 --k 20 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m ResNetLarge -ds CIFAR10
   python train.py -gpu 0 --k 40 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m ResNetLarge -ds CIFAR10
done


#========================CIFAR100========================

# CNNSmall
for seed in {1..100}
do
   python train.py -gpu 0 --k 10 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m CNNSmall -ds CIFAR100
   python train.py -gpu 0 --k 20 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m CNNSmall -ds CIFAR100
   python train.py -gpu 0 --k 40 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m CNNSmall -ds CIFAR100
done

# CNNLarge
for seed in {1..100}
do
   python train.py -gpu 0 --k 10 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m CNNLarge -ds CIFAR100
   python train.py -gpu 0 --k 20 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m CNNLarge -ds CIFAR100
   python train.py -gpu 0 --k 40 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m CNNLarge -ds CIFAR100
done

# ResNetSmall
for seed in {1..100}
do
   python train.py -gpu 0 --k 10 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m ResNetSmall -ds CIFAR100
   python train.py -gpu 0 --k 20 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m ResNetSmall -ds CIFAR100
   python train.py -gpu 0 --k 40 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m ResNetSmall -ds CIFAR100
done

# ResNetLarge
for seed in {1..100}
do
   python train.py -gpu 0 --k 10 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m ResNetLarge -ds CIFAR100
   python train.py -gpu 0 --k 20 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m ResNetLarge -ds CIFAR100
   python train.py -gpu 0 --k 40 --seed $seed -bs 128 --solver SGD --n-epoch 100 -m ResNetLarge -ds CIFAR100
done
