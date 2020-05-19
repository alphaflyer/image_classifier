#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt

import argparse
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
#from collections import OrderedDict
import numpy as np
import seaborn as sns

import constructor

# Globals
#data_dir = 'flowers'
#test_dir = data_dir + '/test'

# Args
parser = argparse.ArgumentParser(description='Image prediction')

parser.add_argument('image_path', action="store", help="Filename and path of image to predict")
parser.add_argument('checkpoint', action="store", help="dir to checkpoint")

results = parser.parse_args()

file_path = results.image_path
checkpoint_dir = results.checkpoint

# Categories
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_model(path):

    checkpoint = torch.load(path)
    net = checkpoint['net']
    dropout = checkpoint['dropout']
    fc1_hidden_layers = checkpoint['fc1_hidden_layers']
    fc1_in_size = checkpoint['fc1_in_size']
    learning_rate = checkpoint['learning_rate']
    num_labels = checkpoint['num_labels']
    model,_,_ = constructor.constructor(net, dropout, fc1_hidden_layers, fc1_in_size, learning_rate, num_labels)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image):
    img = Image.open(image)

    img_prep = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    pil_img = img_prep(img)
    return pil_img

def imshow(image, ax=None, title=None):
  
    if ax is None:
        fig, ax = plt.subplots()
        
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
        
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
        
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
        
    ax.imshow(image)
        
    return ax

def predict(image_path, model, topkl):

    image = process_image(image_path)
    
    im = image.unsqueeze(dim = 0) 
        
    with torch.no_grad():
        output = model.forward(im)
    output_prob = torch.exp(output) #converting into a probability
    
    probs, indeces = output_prob.topk(topkl)
    probs = probs.numpy() #converting both to numpy array
    indeces = indeces.numpy() 
    
    probs = probs.tolist()[0] #converting both to list
    indeces = indeces.tolist()[0]
    
    
    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }
    
    classes = [mapping[item] for item in indeces]
    #classes = pd.DataFrame ([mapping [item] for item in indeces]) #replacing indeces with classes
    classes = np.array (classes) #converting to Numpy array 
    
    return probs, classes

def check_sanity(file_path):    

    img = process_image(file_path)
    #imshow(img)
    #plt.show()
    probs, classes = predict(file_path, model, 5)

    #preparing class_names using mapping with cat_to_name

    class_names =[cat_to_name[item] for item in classes]
    #plt.figure(figsize = (11,5))
    #plt.subplot(2,1,2, autoscale_on=True)

    #sns.barplot(x=probs, y=class_names, color= 'green');

    #plt.show()
    
    print("\r\nPredicted flower: \t", class_names[0],
        "\n", 
        "Probability: \t", round(probs[0] * 100, 0),
        " %")
    
    print("done")
    exit()

# check if Checkpoint exists:
try:
    f = open(checkpoint_dir)
    print("Checkpoint found! Loading model...")
    model = load_model(checkpoint_dir)
    #print(model)
    check_sanity(file_path)
except IOError:
    print("No checkpoint found. Model must be trained first!")
finally:
    f.close()
