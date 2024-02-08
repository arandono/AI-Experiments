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

class LocalNet(nn.Module):
    def __init__(self, 
                 learning_rate=.01, 
                 drop_rate=.2, 
                 momentum=.9, 
                 noise_mean=0,
                 noise_std=.1,
                 gaussian_std=1,
                 nonlocal_attenuation=False,
                 nonlocal_penalties=[0,0,0], 
                 bias=True,
                ):
        """
        The Neural Network class. 
        Note: the optimizer and loss function are specified in this class itself
        ...

        Attributes
        ----------
        features : list of integers, optional, default=[256, 128]
            sets the number of features in the two hidden layers
        drop_rate : float, optional, default=.2
            sets the dropout rate
        momentum : float, optional, default=.9
            sets the momentum coefficient of the optimizer
        learning_rate : float, optional, default=.001
            sets the learning rate of the optimizer

        Methods
        -------
        train_it(epochs=10)
            trains the network, prints loss and accuracy, and returns training and validation loss history as lists
            
        test_it()
            tests the network on the test set and prints loss and accuracy
        """
        
        # Initialize base class
        super(LocalNet, self).__init__()
         
        self.nonlocal_attenuation=nonlocal_attenuation 
        self.bias=bias
        self.Lambdas = nonlocal_penalties
        
        #Define parameters
        self.activation = nn.ReLU()
    
        self.bilinear1 = Bilinear(in_features=[28,28], 
                                  out_features=[24,24], 
                                  bias=bias, 
                                  init_type=init_type,
                                  noise_mean=noise_mean,
                                  noise_std=noise_std,
                                  gaussian_std=gaussian_std, 
                                  nonlocal_attenuation=nonlocal_attenuation
                                 )
        self.drop1 = nn.Dropout(p=drop_rate)
        
        self.bilinear2 = Bilinear(in_features=[24,24], 
                                  out_features=[16,16],
                                  bias=bias, 
                                  init_type=init_type,
                                  noise_mean=noise_mean,
                                  noise_std=noise_std,
                                  gaussian_std=gaussian_std,
                                  nonlocal_attenuation=nonlocal_attenuation
                                 )
        self.drop2 = nn.Dropout(p=drop_rate)
        
        self.bilinear3 = Bilinear(in_features=[16,16], 
                                  out_features=[10, 10],
                                  bias=bias, 
                                  init_type=init_type,
                                  noise_mean=noise_mean,
                                  noise_std=noise_std,
                                  gaussian_std=gaussian_std,
                                  nonlocal_attenuation=nonlocal_attenuation
                                 )
        self.drop3 = nn.Dropout(p=drop_rate)
        
        self.bilinear4 = Bilinear(in_features=[10,10], 
                                  out_features=[10],
                                  bias=bias, 
                                  init_type=init_type,
                                  noise_mean=noise_mean,
                                  noise_std=noise_std,
                                  gaussian_std=gaussian_std
                                 )
        
        self.output = nn.LogSoftmax(dim=1)
        
        # Define optimizer and loss function
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        self.loss = nn.NLLLoss()    
    
    def forward(self, x):
        x = torch.squeeze(x)
        
        x = self.activation(self.bilinear1(x))
        x = self.drop1(x)
                       
        x = self.activation(self.bilinear2(x))
        x = self.drop2(x)
        
        x = self.activation(self.bilinear3(x))
        x = self.drop3(x)
        
        x = self.bilinear4(x)
        x = self.output(x)
        
        return x
    

def train_it(model, epochs=10):
    """Trains the network

    Parameters
    ----------
    epochs : str, optional
        Number of epochs in the training run (default is 10)
    """

    # First choose the device to do computations on
    device = get_device()
    model = model.to(device)


    # Initialize data outputs
    train_loss_history = list()
    val_loss_history = list()
    train_accuracy_history = list()
    val_accuracy_history = list()

    for epoch in range(epochs):
        # Training 
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        n = 0
        t0 = time.time()


        for inputs, labels in iter(train_loader):
            n += 1
            inputs, labels = inputs.to(device), labels.to(device)
            model.optimizer.zero_grad()

            outputs = self.forward(inputs)
            loss = model.loss(outputs, labels) + model.get_nonlocal_penalty()

            loss.backward()
            model.optimizer.step()

            train_loss = (train_loss*(n-1) + loss.item())/n

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = (top_class == labels.view(*top_class.shape))
            train_accuracy = (train_accuracy*(n-1) +100*torch.mean(equals.type(torch.FloatTensor)).item())/n

        # Validating
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        n = 0

        for inputs, labels in iter(valid_loader):
            n += 1
            inputs, labels = inputs.to(device), labels.to(device)
            model.optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = model.loss(outputs, labels) + model.get_nonlocal_penalty()

            val_loss = (val_loss*(n-1) + loss.item())/n

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = (top_class == labels.view(*top_class.shape))
            val_accuracy = (val_accuracy*(n-1) +100*torch.mean(equals.type(torch.FloatTensor)).item())/n

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        print("===============================")
        print("Epoch {}/{} completed in {} seconds on {}".format(epoch+1, epochs, time.time()-t0, device))
        print("Training Loss = {}".format(train_loss))
        print("Validation Loss = {}".format(val_loss))
        print("Training Accuracy = {}%".format(train_accuracy))
        print("Validation Accuracy = {}%".format(val_accuracy))

    return train_loss_history, val_loss_history
    
def test_it(model):
    """Runs a quick test on the Testing Set and prints loss and accuracy"""
    device = get_device()

    model = model.to(device)
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    n = 0

    for inputs, labels in iter(test_loader):
        n += 1
        inputs, labels = inputs.to(device), labels.to(device)
        model.optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = model.loss(outputs, labels) + model.get_nonlocal_penalty()

        test_loss = (test_loss*(n-1) + loss.item())/n

        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1, dim=1)
        equals = (top_class == labels.view(*top_class.shape))
        test_accuracy = (test_accuracy*(n-1) +100*torch.mean(equals.type(torch.FloatTensor)).item())/n

    print("===============================")
    print("Test Loss = {}".format(test_loss))
    print("Test Accuracy = {}%".format(test_accuracy))
    print("===============================")


def get_nonlocal_penalty(model):
    if sum(model.Lambdas) > 0 and model.nonlocal_attenuation == False:
        W1 = model.bilinear1.M * model.bilinear1.dist
        W2 = model.bilinear2.M * model.bilinear2.dist
        W3 = model.bilinear3.M * model.bilinear3.dist

        W1_squared = torch.sum(W1*W1)
        W2_squared = torch.sum(W2*W2)
        W3_squared = torch.sum(W3*W3)

        return .5*(model.Lambdas[0]*W1_squared + model.Lambdas[1]*W2_squared + model.Lambdas[2]*W3_squared)
    else:
        return 0

def get_device():
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    return device