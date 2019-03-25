#!/usr/bin/env python3

# PROGRAMMER: Diego da Costa Oliveira
# DATE CREATED: Mar, 23, 2019.
# REVISED DATE:
# PURPOSE: Train a neural network on a dataset and save the best model as a checkpoint
# BASIC USAGE: python train.py data_directory
#              data_directory have 3 subfolders: '/train', '/test', and '/valid', for training, testing, and validation,
#              respectively
# Parameters:
#     1. Save Folder as --save_dir with default value '/' (root)
#     2. CNN Model Architecture as --arch with default value 'vgg11_bn'
#     3. Learning rate as --learn_rate with default value "0.001"
#     4. Number of hidden units (besides hardcoded hidden layers) --hidden_units with default value 700
#     5. Set GPU usage (CUDA) as --gpu
#     6. Number of epochs to execute the training --epochs with default value 12
#     7. Number of epochs of traning for each testing pass --print_every with default value 1 (test every epoch)

from helpers import get_input_args_train, load_data, build_model
from time import time
from copy import deepcopy

import torch

# get input args
in_arg = get_input_args_train()

# start timing
start_time = time()

# load data, preprocess, and create dataloaders; save class to index mapping
trainloader, testloader, validloader, class_to_idx = load_data(in_arg.data_dir, in_arg.batchsize)

# train and validate network
# build model - get model, criterion and optimizer
model, criterion, optimizer = build_model(in_arg.arch, in_arg.hidden_units, in_arg.learn_rate)

# set device for training and select device: GPU or CPU
device = torch.device("cuda:0" if in_arg.gpu is True else "cpu")
model.to(device)
model.train()

# initialize tracking parameters
print_every = in_arg.print_every  # from argparser
epochs = in_arg.epochs  # from argparser

epoch = 0
steps = 0
running_loss = 0
minibatch = 0

train_losses, test_losses, val_acc_history = [], [], []

# keep track of best model
best_model_wts = deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(epochs):
    for images, labels in trainloader:
        minibatch += 1
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # print progress
        if minibatch % 20 == 0:
            print(f'Epoch {epoch+1} / {epochs}, processed minibatches: {minibatch} / {len(trainloader)}')

    steps += 1
    epoch += 1
    minibatch = 0

    # execute testing on specified interval (print_every)
    if steps % print_every == 0:
        test_loss = 0
        accuracy = 0
        # eval mode
        model.eval()
        # turn off gradients
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)

                log_ps = model(images)
                batch_loss = criterion(log_ps, labels)

                test_loss += batch_loss.item()

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        train_losses.append(running_loss / print_every)
        test_losses.append(test_loss / len(testloader))

        print(f'\nEnding Epoch {epoch} / {epochs}.. '
              f'Train loss: {running_loss / print_every:.3f}.. '
              f'Test loss: {test_loss / len(testloader):.3f}.. '
              f'Test accuracy: {accuracy / len(testloader) * 100:.3f}%\n')
        running_loss = 0

        # deep copy the model on best accuracy
        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = deepcopy(model.state_dict())

        # keep track of accuracy
        val_acc_history.append(accuracy / len(testloader) * 100)

        # set model for train on loop exit
        model.train()

# After training, save model best state for posterior use at prediction
checkpoint = {'state_dict': best_model_wts,
              'optimizer_state_dict': optimizer.state_dict(),
              'class_to_idx': class_to_idx,
              'arch': in_arg.arch,  # from argparser
              'hidden_units': in_arg.hidden_units,  # from argparser
              'epoch': epoch}

filepath = in_arg.save_dir + 'checkpoint.pth'
torch.save(checkpoint, filepath)
print(f'Checkpoint - best model - saved at: {filepath}')

# print statistics per epoch
print(f'\nStatistics per Epoch:\n')
for i in range(len(val_acc_history)):
    print(f'epoch {i+1} / {len(val_acc_history)}\n'
          f'Train losses: {train_losses[i]}, Test losses: {test_losses[i]}, Accuracy: {val_acc_history[i]:.3f}%\n')

# final timing function
end_time = time()
tot_time = end_time - start_time
print('\n** Total Elapsed Runtime - from loading data to saving checkpoint:',
      str(int((tot_time/3600)))+'h:'+str(int((tot_time % 3600)/60))+'m:'
      + str(int((tot_time % 3600) % 60))+'s. **')
