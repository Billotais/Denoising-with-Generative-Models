---
layout: page
title: Denoising with Generative Models
published: true
description: Project at VITA lab - Preliminary notes 
---




# Denoising with Generative models

The report for this project can be found on [vita.bilat.xyz](http://vita.bilat.xyz)

## How to run

```
usage: main.py [-h] [-c COUNT] [-o OUT] [-e EPOCHS] [-b BATCH] [-w WINDOW]
               [-s STRIDE] [-d DEPTH] -n NAME [--dropout DROPOUT]
               [--train_n TRAIN_N] [--test_n TEST_N] [--load LOAD]
               [--continue CONTINUE] [--dataset DATASET]
               [--dataset_args DATASET_ARGS] [--data_root DATA_ROOT] --rate
               RATE --preprocessing PREPROCESSING [--gan GAN] [--lr_g LR_G]
               [--lr_d LR_D] [--scheduler SCHEDULER]

optional arguments:
  -h, --help            show this help message and exit
  -c COUNT, --count COUNT
                        number of mini-batches per epoch [int], default=-1
                        (use all data)
  -o OUT, --out OUT     number of samples for the output file [int],
                        default=500
  -e EPOCHS, --epochs EPOCHS
                        number of epochs [int], default=10
  -b BATCH, --batch BATCH
                        size of a minibatch [int], default=32
  -w WINDOW, --window WINDOW
                        size of the sliding window [int], default=2048
  -s STRIDE, --stride STRIDE
                        stride of the sliding window [int], default=1024
  -d DEPTH, --depth DEPTH
                        number of layers of the network [int], default=4,
                        maximum allowed is log2(window)-1
  -n NAME, --name NAME  name of the folder in which we want to save data for
                        this model [string], mandatory
  --dropout DROPOUT     value for the dropout used the network [float],
                        default=0.5
  --train_n TRAIN_N     number of songs used to train [int], default=-1 (use
                        all songs)
  --test_n TEST_N       number of songs used to test [int], default=1
  --load LOAD           load already trained model to evaluate [bool],
                        default=False
  --continue CONTINUE   load already trained model to continue training
                        [bool], default=False, not implemented yet
  --dataset DATASET     type of the dataset[simple|type], where 'type' is a
                        custom dataset type implemented in load_data(),
                        default=simple
  --dataset_args DATASET_ARGS
                        optional arguments for specific datasets, strings
                        separated by commas
  --data_root DATA_ROOT
                        root of the dataset [path], default=/data/lois-
                        data/models/maestro
  --rate RATE           Sample rate of the output file [int], mandatory
  --preprocessing PREPROCESSING
                        Preprocessing pipeline, a string with each step of the
                        pipeline separated by a comma, more details in readme
                        file
  --gan GAN             lambda for the gan loss [float], default=0 (meaning
                        gan disabled)
  --lr_g LR_G           learning rate for the generator [float],
                        default=0.0001
  --lr_d LR_D           learning rate for the discriminator [float],
                        default=0.0001]
  --scheduler SCHEDULER
                        enable the scheduler [bool], default=False

```
