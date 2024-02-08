## This cell contains the essential imports you will need – DO NOT CHANGE THE CONTENTS! ##
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import time
import math

train_loader = None
valid_loader = None
test_loader = None

def load_data(dataset="MNIST"):
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    valid_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create training set (with no transformations)
    raw_data = datasets.MNIST(root="../data", train=True, download=True, transform=None)

    # Split the training data into pure training data and validation data (80/20 split)
    train_size = int(len(raw_data) * 0.6) # 60% training data
    valid_size = int(len(raw_data) * 0.2) # 20% validation data
    test_size = len(raw_data)-(train_size+valid_size)

    training_data, validation_data, testing_data = torch.utils.data.random_split(raw_data, [train_size, valid_size, test_size])

    # Apply the respective transformations to each dataset
    training_data.dataset.transform = train_transforms
    validation_data.dataset.transform = valid_transforms
    testing_data.dataset.transform = test_transforms

    # Create test set and define test dataloader
    train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(validation_data, batch_size=32)
    test_loader = DataLoader(testing_data, batch_size=32)
    
    return train_loader, valid_loader, test_loader


def show5(img_loader):
    dataiter = iter(img_loader)
    
    batch = next(dataiter)
    labels = batch[1][0:5]
    images = batch[0][0:5]
    for i in range(5):
        print(int(labels[i].detach()))
    
        image = images[i].numpy()
        plt.imshow(image.T.squeeze().T)
        plt.show()
        

        