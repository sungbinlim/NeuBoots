# GBS

This repository contains an implementation of the Generative Bootstrap Sampler (GBS) model.

## Prerequsites

- python==3.7
- pytorch==1.2
- torchvision
- tqdm
- PIL

## Configuration script

Before running `main.py`, please make an `ini` script file under the `script` folder for configurations. 
For detail config, refer to `script/example.ini`.

## Run
#### CIFAR-10
```sh
➜ python main.py cutout/cifar10
```
#### CIFAR-100
```sh
➜ python main.py cutout/cifar100
```
#### SVHN
```sh
➜ python main.py cutout/svhn
```