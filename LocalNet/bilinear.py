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

class Bilinear(nn.Module):
    """
    Creates a bilinear of the form {M^{ij}}_{kl} or {M^i}_{kl} with biases B^{ij} or B^{i}.
    The tensors must then be initialized using one of two methods.
    
    Methods:
    
    init(in_features:array, out_features:array, bias:bool): 
        in_features must be an integer array of length 2 (like [3,3])
        out_features must be an integer array of length 1 or 2 (like [10] or [4,4])
    """
    def __init__(self, 
                 in_features=[10,10], 
                 out_features=[10,10], 
                 bias=True, 
                 gaussian_std=1, 
                 noise_mean=1, 
                 noise_std=1,
                 nonlocal_attenuation=False
                ): 
        super(Bilinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.gaussian_std = gaussian_std
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.nonlocal_attenuation=nonlocal_attenuation
        
        # Define the distance tensor as a parameter that doesn't require gradients (because it's constant)
        dist, gaussian = self.get_distance_and_gaussian()
        self.dist = torch.nn.Parameter(dist)
        self.gaussian = torch.nn.Parameter(gaussian)
        self.dist.requires_grad = False
        self.gaussian.requires_grad = False
        
        M = torch.randn(out_features + in_features, requires_grad=True)
        B = torch.zeros(out_features, requires_grad=True)
        
        # Initiate the bilinear transformation
        M, B = self.initialize_weights_and_bias(M, B)
        
        self.M = torch.nn.Parameter(M)
        self.B = torch.nn.Parameter(B)
          
    def get_distance_and_gaussian(self):        
        
        dist = torch.zeros(self.out_features+self.in_features, requires_grad=False)
        gaussian = torch.zeros(self.out_features+self.in_features, requires_grad=False)
        
        if len(self.out_features) > 1:
            mx = self.out_features[0]  #Dimension of target tensor in x direction
            my = self.out_features[1]  #Dimensions of input
            nx = self.in_features[0]
            ny = self.in_features[1]
            for i in range(nx):
                for j in range(ny):
                    for I in range(mx):
                        for J in range(my):
                            #Now define square distance tensor
                            m = np.min((mx, my))
                            n = np.min((nx, ny))
                            dist[I,J,i,j] = np.sqrt((I/m-i/n)**2+(J/m-j/n)**2)
                   
            gaussian = 1/(self.gaussian_std*np.sqrt(2*torch.pi))*torch.exp(-dist**2/(2*self.gaussian_std**2))

        return dist, gaussian
    
    def initialize_weights_and_bias(self, M, B):
        """
        Performs the standard uniform tensor initialization that torch does for Linear layers.
        
        Initializes the weights and bias just like torch: using uniform noise
        in the range -sqrt(1/in_features) to sqrt(1/in_features).
        """
        
        with torch.no_grad():
            k = 1/(np.prod(self.in_features))
            nn.init.uniform_(M, -np.sqrt(k), np.sqrt(k))
            if self.bias:
                nn.init.uniform_(B, -np.sqrt(k), np.sqrt(k))
                
        return M, B
        
    def forward(self, x): 
        """
        Performs the forward pass tensor contraction:
        x^{ij} -> M^{ij}_{kl} x^{kl} + B^{ij}
        """
        # If we choose nonlocal_attenuation, attenuate each tensor by the gaussian before the forward pass
        if self.nonlocal_attenuation and len(self.out_features) == 2:
            x = torch.tensordot(x, self.M*self.gaussian, dims=([-2,-1], [-2,-1]))
        # Otherwise just perform bilinear multiplication
        else:
            x = torch.tensordot(x, self.M, dims=([-2,-1], [-2,-1]))
        # Add in bias if chosen
        if self.bias:
            x += self.B
        
        return x
    

def visualize_local_tensors(img_loader):
    bilinear = Bilinear(in_features=[28, 28], 
                        out_features=[28, 28], 
                        noise_mean = 1, 
                        noise_std = .2, 
                        gaussian_std = .03,
                        bias=False,
                        nonlocal_attenuation=True
                       )
    dataiter = iter(img_loader)
    
    images, labels = next(dataiter)
    labels = labels[0:5]
    images = images[0:5]
    transformed_images = torch.abs(bilinear(images))
    for i in range(5):    
        image = images[i].numpy()
        transformed_image = transformed_images[i].detach().numpy()
        
        #subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(1,2) 

        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        axarr[0].imshow(image.T.squeeze().T)
        axarr[1].imshow(transformed_image.T.squeeze().T)

        plt.show()