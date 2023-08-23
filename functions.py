# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt

# This is a test


# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    criterion = checkpoint['criterion']

    return (model)


#Process a PIL image and returbs a Numpy array

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model

    # Get dimensions
    width, height = image.size   
    # Resize to 256
    ratio = min(width/256, height/256)
    im_resized = image.resize((int(width/ratio), int(height/ratio)))

    # Crop the center of the image
    width, height = im_resized.size  
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    im = im_resized.crop((left, top, right, bottom))


    # Normalize
    np_image = np.array(im).astype(np.float32)

    normalized_image =np_image/255
    
    means = [0.485, 0.456, 0.406]
    
    std_dev = [0.229, 0.224, 0.225]

    for dim in range(3):
        normalized_image[:,:,dim] = (normalized_image[:,:,dim] - means[dim])/std_dev[dim]

    transposed_image = normalized_image.transpose(2,0,1)
    
    return torch.tensor(transposed_image)


