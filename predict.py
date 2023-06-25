# Imports here

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import numpy as np
import argparse


from functions import load_checkpoint, process_image
import json

#from ipynb.fs.full.Image_Classifier_Project import imshow


# Create the command line App
parser = argparse.ArgumentParser()
parser.add_argument("image_input", help="specify the image directory")
parser.add_argument("checkpoint", help="specify the checkpoint directory")
parser.add_argument("--category_names", dest= 'cat_to_name', help="specify the Network Architecture")
parser.add_argument("--top_k", default= 5, dest = 'topk', help="specify the number of top probabilities", type=int)
parser.add_argument("--gpu", action='store_true', default=False, dest='gpu', help="specify the usage of gpu")
args = parser.parse_args()
print(args)

#Predicts the topk classes 

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    #Get the processed image
    image = Image.open(image_path)
    processed_image = process_image(image)
    processed_image = torch.unsqueeze(processed_image, dim=0)


    # Choose between GPU or CPU
    gpu = False
    device = torch.device('cuda' if gpu else 'cpu')

    #Move model to gpu or cpu
    model.to(device)

    #move the image to gpu or cpu
    processed_image.to(device)

    
    #evaluate
    model.eval()
    with torch.no_grad():
        log_prob = model(processed_image)
        prob = torch.exp(log_prob)


    #Get the classes
    top_p, predicted_idx = prob.topk(5)


    #Get the index to class dictionary
    class_to_idx_dict = model.class_to_idx
    idx_to_class_dict = {v: k for k, v in class_to_idx_dict.items()}

    #Get the classes
    predicted_class = []
    for k in predicted_idx[0].tolist():
        predicted_class.append(idx_to_class_dict[k])

    probs = top_p.tolist()

    return probs, predicted_class




#Get the model
checkpoint = args.checkpoint
model = load_checkpoint(checkpoint)

#Open image
image_input = args.image_input
image = Image.open(image_input) 

#Process image
processed_image = process_image(image)

#Show image
#ax = imshow(processed_image, ax=None, title=None)

#Compute prbailities and classes
topk = args.topk
probs, classes = predict(image_input, model, topk)

#Get the names
cat_to_name = args.cat_to_name
if cat_to_name:
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    

    predicted_names = []
    for k in classes:
        predicted_names.append(cat_to_name[k])

    print(f"Predicted names: {predicted_names}, Predicted probabilities: {probs}")

else:
    print(f"Predicted classes: {classes}, Predicted probabilities: {probs}")

#fig = plt.figure(figsize=(10, 5))



# creating the bar plot
#plt.barh(predicted_names, probs[0], height=0.8, left=None, align='center')
#plt.xlabel("Flower type")
#plt.ylabel("Flower types predicted")
#plt.title("probability for each type")
#plt.show()


