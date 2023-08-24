# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="specify the data directory")
parser.add_argument("--save_dir", default= 'saved_models/model_checkpoint.pth', dest='save_dir', help="specify the saving directory")
parser.add_argument("--arch", default= 'vgg16', dest= 'architecture', choices=['vgg13', 'vgg16'], help="specify the Network Architecture")
parser.add_argument("--learning_rate", default= 0.03, dest = 'learning_rate', help="specify the data directory", type=float)
parser.add_argument("--hidden_units", default= 512, dest = 'hidden_units', help="specify the number of hidden units", type=int)
parser.add_argument("--epochs", default= 2, dest = 'epochs', help="specify the number of epochs", type=int)
parser.add_argument("--gpu", action='store_true', default=False, dest='gpu',help="specify the usage of gpu")
args = parser.parse_args()
print(args)

#Define the directories
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


#transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    #transforms.Resize(224), 
    #transforms.CenterCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = test_transforms

#define the dataloaders
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

#Define the model
architecture = args.architecture

if architecture == "vgg13":
    model = models.vgg13(weights=models.VGG13_Weights.DEFAULT)

elif architecture == "vgg16":
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

else: 
    print("Architecture not valid")

#Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout

hidden_units = args.hidden_units

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088,hidden_units)),
    ( 'ReLu1', nn.ReLU()),
    ( 'drop1', nn.Dropout(p=0.5, inplace=False)),
    ( 'fc3', nn.Linear(hidden_units,102)),
    ( 'out', nn.LogSoftmax(dim=1))
    ]))

model.classifier = classifier


#TRAIN



# Choose between GPU or CPU
gpu = args.gpu
device = torch.device('cuda' if gpu else 'cpu')

#Move model to gpu or cpu
model.to(device)

# Freeze Features parameters so we don't backprop through them.
for param in model.features.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True


#Define criterion
criterion = nn.NLLLoss()

#define optimizer and select only the classifier parameters to be optimized
learning_rate = args.learning_rate
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


epochs = args.epochs

for epoch in range(epochs):
    
    running_loss = 0

    #Train
    model.train()

    for images, labels in trainloader:

        #move to gpu or cpu
        images, labels = images.to(device), labels.to(device)

        #forward pass
        log_prob = model(images)
        #compute loss
        loss = criterion(log_prob, labels)
        #backward pass parameter optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        running_loss += loss.item()
    
    running_loss = running_loss/len(trainloader)

    print(f"Epoch {epoch+1}/{epochs} training loss is {running_loss}")

    #Compute the validation loss and accuracy
    valid_loss = 0
    OKs = []
    valid_accuracy = 0

    #Set the model in evaluation mode
    model.eval()

    #Get the loss and accuracy 
    for images, labels in validloader:

        #move to gpu or cpu
        images, labels = images.to(device), labels.to(device)
            
        #evaluate
        log_prob = model(images)
            
        #compute loss
        loss = criterion(log_prob, labels)
            
        valid_loss += loss.item()

        #compute accuracy
        prob = torch.exp(log_prob)
        top_p, predicted_class = prob.topk(1, dim=1)
        equals = predicted_class == labels.view(*predicted_class.shape)
        OKs.extend(equals)


    valid_loss = valid_loss/len(validloader)
    valid_accuracy = OKs.count(True)/len(OKs)
        

    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {valid_loss}, Validation accuracy: {valid_accuracy}")

save_dir = args.save_dir

torch.save({
                'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'class_to_idx': test_data.class_to_idx
                }, save_dir)