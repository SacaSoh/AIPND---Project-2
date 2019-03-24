# PROGRAMMER: Diego da Costa Oliveira
# DATE CREATED: Mar, 23, 2019.
# REVISED DATE:
# PURPOSE: Helpers for training a CNN and to predict the class for an input image


import matplotlib.pyplot as plt

import json
import argparse

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

from PIL import Image


def load_data(data_dir, batchsize=60):
    """ Load data folders, set batch size, execute transforms, and define the dataloaders and class to idx mapping
        Returns trainloader, testloader, validloader, class_to_idx (class to index mapping)
    """
    # set directories
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'

    # execute transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = valid_transforms = transforms.Compose([transforms.Resize(250),
                                                             transforms.CenterCrop(224),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                                  [0.229, 0.224, 0.225])])

    # Load the datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batchsize)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batchsize)

    class_to_idx = train_data.class_to_idx

    return trainloader, testloader, validloader, class_to_idx

def build_model(arch, hidden_units, learn_rate):
    """ Build CNN model based on parameters, returns model, criterion, and optimizer
    """
    # select model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=True)
    else:
        print('Model not recognized - select between "vgg11_bn" and "vgg16"')

    # freezing parameters
    for param in model.parameters():
        param.requires_grad = False

    # classifier - two options just for purposes of this exercise
    if arch =='vgg16':
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, hidden_units)),
                                  ('relu1', nn.ReLU()),
                                  ('Dropout1', nn.Dropout(0.2)),
                                  ('fc2', nn.Linear(hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

    elif arch == 'vgg11_bn':
        # for some reason, the deactivated relu (relu2) helps precision
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, hidden_units)),
                                  ('relu1', nn.ReLU()),
                                  ('Dropout1', nn.Dropout(0.2)),
                                  ('fc2', nn.Linear(hidden_units, 384)),
                                  #('relu2', nn.ReLU()),
                                  ('fc3', nn.Linear(384, 102)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

    model.classifier = classifier

    # set model criterion
    criterion = nn.NLLLoss()

    # Set optimizer -0 Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    return model, criterion, optimizer


def get_input_args_train():
    """ Process input arguments for training script
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create command line arguments
    parser.add_argument('data_dir', type=str,
                        help='path to the folder containing train, test, and validation subfolders')

    # optional arguments
    parser.add_argument('--batchsize', type=int, default=60,
                        help='minibatch size (default=60)')

    parser.add_argument('--print_every', type=int, default=1,
                        help='Number of epochs of traning for each testing pass (default=1 i.e. every epoch)')

    parser.add_argument('--save_dir', type=str, default='',
                        help='path to the folder to save checkpoints (default= "" (root folder))')

    parser.add_argument('--arch', type=str, default='vgg11_bn',
                        help='CNN Model Architecture - vgg11_bn, or vgg16 (default= "vgg11_bn")')

    parser.add_argument('--learn_rate', type=int, default=0.001,
                        help='Learning rate (default=0.001)')

    parser.add_argument('--gpu', type=bool, default=True,
                        help='Define usage of GPU CUDA device (default=True')

    parser.add_argument('--epochs', type=int, default=12,
                        help='Number of epochs to execute the training')

    parser.add_argument('--hidden_units', type=int, default=700,
                        help='Number of epochs to execute the training')

    return parser.parse_args()