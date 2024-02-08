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
from bilinear import Bilinear
from data import train_loader, valid_loader, test_loader

class LocalNet(nn.Module):
    def __init__(self, 
                 features=[[28, 28], [24, 24], [16, 16], [10, 10], [10]],
                 learning_rate=.01, 
                 drop_rate=.2, 
                 momentum=.9, 
                 noise_mean=0,
                 noise_std=.1,
                 gaussian_std=1,
                 nonlocal_attenuation=False,
                 nonlocal_penalties=None, 
                 bias=True,
                 save_path="checkpoints/model.pt"
                ):
        """
        The Traditional Neural Network class. 
        Note: the optimizer and loss function are specified in this class itself
        ...

        Attributes
        ----------
        features : the feature map as a list of pixels of the form
            [[width1, height1], [width2, height2],  ... , [n_classes]]
        drop_rate : sets the dropout rate
        momentum : sets the momentum coefficient of the optimizer
        learning_rate : sets the learning rate of the optimizer

        """
        
        # Initialize base class
        super(LocalNet, self).__init__()
         
        self.nonlocal_attenuation=nonlocal_attenuation 
        self.bias=bias
        self.Lambdas = nonlocal_penalties
        self.save_path = save_path
        self.features = features
        
        #Define parameters
        self.activation = nn.ReLU()
    
        self.bilinear1 = Bilinear(in_features=features[0], 
                                  out_features=features[1], 
                                  bias=bias, 
                                  noise_mean=noise_mean,
                                  noise_std=noise_std,
                                  gaussian_std=gaussian_std, 
                                  nonlocal_attenuation=nonlocal_attenuation
                                 )
        self.drop1 = nn.Dropout(p=drop_rate)
        
        self.bilinear2 = Bilinear(in_features=features[1], 
                                  out_features=features[2],
                                  bias=bias, 
                                  noise_mean=noise_mean,
                                  noise_std=noise_std,
                                  gaussian_std=gaussian_std,
                                  nonlocal_attenuation=nonlocal_attenuation
                                 )
        self.drop2 = nn.Dropout(p=drop_rate)
        
        self.bilinear3 = Bilinear(in_features=features[2], 
                                  out_features=features[3],
                                  bias=bias, 
                                  noise_mean=noise_mean,
                                  noise_std=noise_std,
                                  gaussian_std=gaussian_std,
                                  nonlocal_attenuation=nonlocal_attenuation
                                 )
        self.drop3 = nn.Dropout(p=drop_rate)
        
        self.bilinear4 = Bilinear(in_features=features[3], 
                                  out_features=features[4],
                                  bias=bias, 
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
    
    

class TraditionalNet(nn.Module):
    def __init__(self, 
                 features=[28*28, 24*24, 16*16, 10*10, 10],
                 learning_rate=.01, 
                 drop_rate=.2, 
                 momentum=.9,  
                 bias=True,
                 nonlocal_penalties = None,
                 nonlocal_attenuation=False,
                 save_path="checkpoints/model.pt"
                ):
        """
        The Traditional Neural Network class. 
        Note: the optimizer and loss function are specified in this class itself
        ...

        Attributes
        ----------
        learning_rate : float, optional, default=.001
            sets the learning rate of the optimizer
        drop_rate : float, optional, default=.2
            sets the dropout rate
        momentum : float, optional, default=.9
            sets the momentum coefficient of the optimizer
        bias: sets bias True or False
        nonlocal_penalties: sets prefactors for the nonlocal penalties in the error term.
            Should be of the form [1, .1, .01] where each term corresponds to a bilinear layer.
        nonlocal_attenuation: Adds a distance based attentuation factor in all forward pass steps
        

        Methods
        -------
        train_it(epochs=10)
            trains the network, prints loss and accuracy, and returns training and validation loss history as lists
            
        test_it()
            tests the network on the test set and prints loss and accuracy
        """
        
        # Initialize base class
        super(TraditionalNet, self).__init__()
        
        self.features=features
        self.bias=bias
        self.Lambdas = nonlocal_penalties
        self.nonlocal_attenuation=nonlocal_attenuation
        self.save_path = save_path  
        
        
        #Define parameters
        self.activation = nn.ReLU()
    
        self.flatten = nn.Flatten()
        
        self.linear1 = nn.Linear(in_features=features[0], out_features=features[1], bias=bias)
        self.drop1 = nn.Dropout(p=drop_rate)
        
        self.linear2 = nn.Linear(in_features=features[1], out_features=features[2], bias=bias)
        self.drop2 = nn.Dropout(p=drop_rate)
        
        self.linear3 = nn.Linear(in_features=features[2], out_features=features[3], bias=bias)
        self.drop3 = nn.Dropout(p=drop_rate)
        
        self.linear4 = nn.Linear(in_features=features[3], out_features=features[4], bias=bias)
        
        self.output = nn.LogSoftmax(dim=1)
        
        
        # Define optimizer and loss function
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        self.loss = nn.NLLLoss() 
         
    
    def forward(self, x):
        x = self.flatten(x)
        x = torch.squeeze(x)
        
        x = self.activation(self.linear1(x))
        x = self.drop1(x)
                       
        x = self.activation(self.linear2(x))
        x = self.drop2(x)
        
        x = self.activation(self.linear3(x))
        x = self.drop3(x)
        
        x = self.linear4(x)
        x = self.output(x)
        
        return x
    

def train_it(model, train_loader, valid_loader, epochs=10):
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

            outputs = model.forward(inputs)
            loss = model.loss(outputs, labels) + get_nonlocal_penalty(model)

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
        val_loss_min = None
        n = 0

        for inputs, labels in iter(valid_loader):
            n += 1
            inputs, labels = inputs.to(device), labels.to(device)
            model.optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = model.loss(outputs, labels) + get_nonlocal_penalty(model)

            val_loss = (val_loss*(n-1) + loss.item())/n

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = (top_class == labels.view(*top_class.shape))
            val_accuracy = (val_accuracy*(n-1) +100*torch.mean(equals.type(torch.FloatTensor)).item())/n

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)
        
        if val_loss_min is None or ((val_loss_min - val_loss) / val_loss_min > 0.01):
            
            # Save the weights to save_path
            # YOUR CODE HERE
            torch.save(model.state_dict(), model.save_path)
            val_loss_min = val_loss

        print("===============================")
        print("Epoch {}/{} completed in {} seconds on {}".format(epoch+1, epochs, time.time()-t0, device))
        print("Training Loss = {}".format(train_loss))
        print("Validation Loss = {}".format(val_loss))
        print("Training Accuracy = {}%".format(train_accuracy))
        print("Validation Accuracy = {}%".format(val_accuracy))
        
    # Plot Loss and Accuracy each epoch
    fig, axarr = plt.subplots(1,2)
    axarr[0].plot(train_loss_history, label="Training Loss")
    axarr[0].plot(val_loss_history, label="Validation Loss")
    axarr[1].plot(train_accuracy_history, label="Training Accuracy")
    axarr[1].plot(val_accuracy_history, label="Validation Accuracy")
    axarr[0].set_xlim([0, epochs])
    axarr[1].set_xlim([0, epochs])
    axarr[0].legend()
    axarr[1].legend()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.show()

    return train_loss_history, val_loss_history
    
def test_it(model, test_loader):
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
        loss = model.loss(outputs, labels) + get_nonlocal_penalty(model)

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
    if model.Lambdas != None and sum(model.Lambdas) > 0 and model.nonlocal_attenuation == False:
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
    #if torch.backends.mps.is_available():
    #    device = "mps"
    #elif torch.cuda.is_available():
    #    device = "cuda"
    return device

def visualize_forward_pass(model, img_loader, N=20, cmap="Greys"):
    dataiter = iter(img_loader)
    
    images, labels = next(dataiter)
    labels = labels[0:N]
    images = images[0:N]
    
    for i in range(N):    
        image0 = images[i]
        
        image1 = model.bilinear1(image0)
        image1 = model.activation(image1)
        
        image2 = model.bilinear2(image1)
        image2 = model.activation(image2)
        
        image3 = model.bilinear3(image2)
        image3 = model.activation(image3)

        classes = F.softmax(model.bilinear4(image3), dim=1)
        
        # Use this for easier visualization. Comment it out to see the raw output.
        #image1 = torch.tanh(image1)
        #image2 = torch.tanh(image2)
        #image3 = torch.tanh(image3)
        
        image0 = image0.detach()
        image1 = image1.detach()
        image2 = image2.detach()
        image3 = image3.detach()
        classes = classes.detach()
        
        #subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(1,4) 

        classes_names = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
        probabilities = classes.squeeze().tolist()
        
        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        axarr[0].imshow(image0.squeeze(), cmap=cmap)
        axarr[1].imshow(image1.squeeze(), cmap=cmap)
        axarr[2].imshow(image2.squeeze(), cmap=cmap)
        axarr[3].imshow(image3.squeeze(), cmap=cmap)
        #axarr[4].bar(classes_names, probabilities)
        f.set_figheight(12)
        f.set_figwidth(12)
        plt.show()