#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

# updating classifer in the network
def constructor(net, dropout, fc1_hidden_layers, fc1_in_size, learning_rate, num_labels):

    ## VGG ##
    if net == 'vgg19':
        model = models.vgg19(pretrained=True)#, progress=True)


    elif net == 'alexnet':
        model = models.alexnet(pretrained=True)#, progress=True)
     

    else:
        print("Error, wrong model! Accepted values are 'vgg19' or 'alexnet'")
        exit()

    for param in model.parameters(): 
        param.requires_grad = False

    classifier = nn.Sequential  (OrderedDict ([
                                        ('fc1', nn.Linear (fc1_in_size, fc1_hidden_layers)),
                                        ('relu1', nn.ReLU ()),
                                        ('dropout1', nn.Dropout (p = dropout)),
                                        ('fc2', nn.Linear (fc1_hidden_layers, int(fc1_hidden_layers / 2))),
                                        ('relu2', nn.ReLU ()),
                                        ('dropout2', nn.Dropout (p = dropout)),
                                        ('fc3', nn.Linear (int(fc1_hidden_layers / 2), num_labels)),
                                        ('output', nn.LogSoftmax (dim=1))
                                        ]))
    model.classifier = classifier

    #initializing criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    return model, optimizer, criterion