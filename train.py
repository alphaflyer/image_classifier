#!/usr/bin/env python
# coding: utf-8

import os
from os import path
import argparse
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict
import time
import numpy as np

import constructor


# Args
parser = argparse.ArgumentParser(description='Model trainer')

parser.add_argument('data_directory', action="store", type=str, help='Parent-Path to training data')
parser.add_argument('--save_dir', action="store", type=str, dest="save_dir", help='where to save checkpoints', default="checkpoint")
parser.add_argument('--arch', action="store", dest="net", default="vgg19", help="Defines NN. Possible values: 'alexnet' or 'vgg19'")
parser.add_argument('--hidden_units', action="store", dest="fc1_hidden_layers", type=int, default="4096", help="No of hidden layers")
parser.add_argument('--dropout', action="store", dest="dropout", type=float, default=0.1, help="Dropout rate (0-1). Less should be better")
parser.add_argument('--gpu', action="store_true", dest="cuda", help="Calc on GPU. Should be way faster")
parser.add_argument('--batchsize', action="store", dest="batchsize", type=int, default=64, help="Batchsize, default is 64")
parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=7, help="Nr. of epochs to train. 7 should be sufficient")
parser.add_argument('--lr', action="store", dest="learning_rate", type=float, default=0.001, help="Learning rate, default 0.001")

results = parser.parse_args()

data_dir = results.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


folders = 0
for _, dirnames, _ in os.walk(train_dir):
    folders += len(dirnames)
num_labels = folders

checkpoint_dir = results.save_dir

try:
    if True:
        path.exists(checkpoint_dir)
        print("Path to checkpoint already exits ", checkpoint_dir)
except:
    try:
        os.mkdir(checkpoint_dir)
    except OSError:
        print ("Creation of the directory %s failed" % checkpoint_dir)
    else:
        print ("Successfully created the directory %s " % checkpoint_dir)

net = results.net
if net == "vgg19":
    fc1_in_size = 25088
elif net == "alexnet":
    fc1_in_size = 9216

dropout = results.dropout

if results.cuda == True:
    machine = 'cuda'
else:
    machine = 'cpu'

batchsize = results.batchsize
fc1_hidden_layers = results.fc1_hidden_layers
epochs = results.epochs
learning_rate = results.learning_rate

print("\r\nTraining parameter: ",
        "\r\n", 
        "Neural net: \t", net,
        "\r\n",         
        "Dropout: \t", dropout,
        "\r\n",    
        "Calculating on: \t", machine,
        "\r\n",    
        "Batch size: \t", batchsize,
        "\r\n",    
        "Epochs to train: \t", epochs)

# Building
train_transforms = transforms.Compose ([
                            transforms.RandomRotation(30),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])

valid_and_test_transforms = transforms.Compose ([
                            transforms.Resize(255),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])

train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_image_datasets = datasets.ImageFolder (valid_dir, transform = valid_and_test_transforms)
test_image_datasets = datasets.ImageFolder (test_dir, transform = valid_and_test_transforms)

train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batchsize, shuffle=True)#, drop_last=True)#, pin_memory=True)#,num_workers=10)
valid_loader = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64, shuffle=True)#, drop_last=True)#, pin_memory=True)#,num_workers=10)
test_loader = torch.utils.data.DataLoader(test_image_datasets, batch_size=64)#, shuffle=True, drop_last=True)#, pin_memory=True)#,num_workers=10)

# Categories
#with open('cat_to_name.json', 'r') as f:
#    cat_to_name = json.load(f)

model, optimizer, criterion = constructor.constructor(net, dropout, fc1_hidden_layers, fc1_in_size, learning_rate, num_labels) 

# Defining validation 
def validation(model, valid_loader, criterion):
    model.to(machine)

    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:

        inputs, labels = inputs.to(machine), labels.to(machine)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

# Training model
model.to(machine)
stepcounter = 40
epochcounter = 0
steps = 0
step_runtime = 0
runtime = 0

print("\r\nTraining model...\tEpoch count: ", epochs)

for e in range (epochs): 

    time_per_epoch_start = time.time()

    running_loss = 0
    for ii, (inputs, labels) in enumerate(train_loader):

        time_per_step_start = time.time()

        steps += 1

        inputs, labels = inputs.to(machine), labels.to(machine)

        optimizer.zero_grad() 

        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step()

        running_loss += loss.item() 

        if steps % stepcounter == 0:
            model.eval() #switching to evaluation mode, dropout off

            # Turn off gradients for validation
            with torch.no_grad():
                valid_loss, accuracy = validation(model, valid_loader, criterion)

            print("Epoch: {}/{} | ".format(e+1, epochs),
                      "Training Loss: {:.3f} | ".format(running_loss/stepcounter),
                      "Valid Loss: {:.3f} | ".format(valid_loss/len(valid_loader)),
                      "Valid Accuracy: {:.3f}%".format(accuracy/len(valid_loader)*100))

            running_loss = 0

            model.train()

        time_per_step_end = time.time()
        step_runtime += time_per_step_end - time_per_step_start


    epochcounter += 1
    time_per_epoch_end = time.time()
    runtime += time_per_epoch_end - time_per_epoch_start

print("Total Runtime: {:.1f} min".format(runtime / 60),
            "Runtime per Step: {:.1f} sec".format(step_runtime / stepcounter),
            "Runtime per Epoch: {:.1f} sec".format(runtime / epochcounter))



# Do validation on the test set
def check_accuracy_on_test(test_loader,model):    
    correct = 0
    total = 0
    model.to(machine)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(machine), labels.to(machine)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test images: %d %%' % (100 * correct / total))
check_accuracy_on_test(test_loader, model)


# Save the checkpoint 
print("Saving checkpoint...")

model.class_to_idx = test_image_datasets.class_to_idx

torch.save({    'net':net,
                'dropout':dropout,
                'fc1_hidden_layers':fc1_hidden_layers,
                'fc1_in_size':fc1_in_size,
                'learning_rate':learning_rate,
                'num_labels':num_labels,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                checkpoint_dir + '/checkpoint.pth')

print("Checkpoint saved! ", checkpoint_dir)