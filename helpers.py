# PROGRAMMER: Diego da Costa Oliveira
# DATE CREATED: Mar, 23, 2019.
# REVISED DATE:
# PURPOSE: Helpers for training a CNN and to predict the class for an input image


import argparse
import numpy as np
import torch

import json

from torch import nn
from torch import optim
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


def load_checkpoint_prediction(filepath):
    """ Load checkpoint for use at prediction (no parameters for further training)
    """

    checkpoint = torch.load(filepath)

    hidden_units = checkpoint['hidden_units']

    if checkpoint['arch'] == 'vgg16':

        # make sure to create same model used as before
        model = models.vgg16(pretrained=True)
        # freezing parameters
        for param in model.parameters():
            param.requires_grad = False

        # classifier
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu1', nn.ReLU()),
            ('Dropout1', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        model.classifier = classifier

    elif checkpoint['arch'] == 'vgg11_bn':

        # make sure to create same model used as before
        model = models.vgg11_bn(pretrained=True)
        # freezing parameters
        for param in model.parameters():
            param.requires_grad = False

        # classifier
        # for some reason, the deactivated relu helps precision
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu1', nn.ReLU()),
            ('Dropout1', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 384)),
            # ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(384, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        model.classifier = classifier

    else:
        print('Architecture not recognized on load function -- options "vgg11_bn" and "vgg16"')

    # load model parameters
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image):
    """ Scales, crops, and normalizes a PIL image for input to a PyTorch model,
        returns a Numpy array
    """
    img = Image.open(image)

    # keep shortest side as 256 pixels
    if img.size[0] > img.size[1]:
        img.thumbnail((img.size[0], 256))
    else:
        img.thumbnail((256, img.size[1]))

    # crop to a 224x224 image
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224

    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    # convert to numpy array, execute normalization
    np_image = np.array(img)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np_image / 255

    np_image = (np_image - mean) / std

    # transpose image to satisfy torch expected structure (color channel as 1st channel) - PIL uses 3st channel
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image


def predict(image_path, checkpoint, topk, category_names, gpu):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    # load model on eval mode
    model = load_checkpoint_prediction(checkpoint)
    model.eval()

    device = torch.device("cuda:0" if gpu is True else "cpu")
    model.to(device)

    # preprocess image
    image = process_image(image_path)
    image = torch.from_numpy(image)
    reshaped = image.unsqueeze(0)
    reshaped = reshaped.float().to(device)

    # run image thru network
    with torch.no_grad():
        logps = model(reshaped)
        ps = torch.exp(logps)

        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)

        top_p_list, idx_class_list, class_to_idx, flower_names, top_labels = [], [], [], [], []

        # get top probabilities and respective classes (0-indexed)
        top_p, top_class = ps.topk(topk, dim=1)

        # populate list with k probabilities and classes, and index to cat_to_name (flower names)
        for i in range(top_p.shape[1]):
            top_p_list.append(top_p.data[0][i].item())
            idx_class_list.append(top_class.data[0][i].item())

        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_labels = [idx_to_class[lab] for lab in idx_class_list]
        flower_names = [cat_to_name[i] for i in top_labels]

    return top_p_list, flower_names


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

    parser.add_argument('--gpu', default=False, action='store_true',
                        help='Define usage of GPU CUDA device (default=True')

    parser.add_argument('--epochs', type=int, default=12,
                        help='Number of epochs to execute the training')

    parser.add_argument('--hidden_units', type=int, default=700,
                        help='Number of epochs to execute the training')

    return parser.parse_args()


def get_input_args_predict():
    """ Process input arguments for prediction script
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create command line arguments
    parser.add_argument('path_to_image', type=str,
                        help='path to the image to predict flower name')

    parser.add_argument('path_to_checkpoint', type=str,
                        help='path to the checkpoint with CNN training data')

    # optional arguments
    parser.add_argument('--top_k', type=int, default=5,
                        help='Return top K most likely classes (default=5)')

    parser.add_argument('--gpu', default=False, action='store_true',
                        help='Define usage of GPU CUDA device (default=True')

    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Use a mapping of categories to real names (default= "cat_to_name.json")')

    return parser.parse_args()