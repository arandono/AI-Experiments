import torch
import torch.nn as nn
import torch.nn.functional as F
from cplxmodule import cplx


class Complex_Conv2d(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1): 
        super(Complex_Conv2d, self).__init__()
        self.A = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.B = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False)
          
    def forward(self, z): 
        x = z.real
        y = z.imag
        
        X = self.A(x)-self.B(y)
        Y = self.A(y)+self.B(x)
        Z = torch.complex(X, Y)
        
        return Z
    

class Complex_Linear(nn.Module): 
    def __init__(self, in_features, out_features): 
        super(Complex_Linear, self).__init__()
        self.A = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.B = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
          
    def forward(self, z): 
        x = z.real
        y = z.imag
        
        X = self.A(x)-self.B(y)
        Y = self.A(y)+self.B(x)
        Z = torch.complex(X, Y)
        
        return Z


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

        self.conv_1 = Complex_Conv2d(in_channels=3, out_channels=8)
        self.conv_2 = Complex_Conv2d(in_channels=8, out_channels=10)
        self.conv_3 = Complex_Conv2d(in_channels=10, out_channels=12)
        self.conv_4 = Complex_Conv2d(in_channels=12, out_channels=16)
        
        self.fc1 = Complex_Linear(16*16*16, 256)
        self.fc2 = Complex_Linear(256, 64)
        self.fc3 = Complex_Linear(64, num_classes)
        
        
#         self.conv_bb = nn.Sequential(
            
#             # Convolutional Layer 1: 3*256*256 -> 8*128*128
#             nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1), 
#             nn.ReLU(),
#             nn.MaxPool2d(2,2),
            
#             # Convolutional Layer 2: 8*128*128 -> 16*64*64
#             nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2),
#             nn.BatchNorm2d(16),
            
#             # Convolutional Layer 3: 16*64*64 -> 32*32*32
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2),
#             nn.BatchNorm2d(32),
            
#             # Convolutional Layer 4: 32*32*32 -> 48*16*16
#             nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2),
#             nn.BatchNorm2d(48),

#             # Convolutional Layer 5: 48*16*16 -> 64*8*8
#             nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2),
#             nn.BatchNorm2d(64),
            
#             # Flatten for Head input: 64*8*8 -> 1*4096
#             nn.Flatten(),
#         )

#         self.mlp_head = nn.Sequential(
#             # Linear Layer 1: 4096 -> 512
#             nn.Linear(64*8*8, 512),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.BatchNorm1d(512),

#             # Linear Layer 2: 512 -> 128
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.BatchNorm1d(128),

#             # Linear Layer 3: 128 -> num_classes
#             nn.Linear(128, num_classes)
#         )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        z = self.conv_1(z)
        z = cplx.max_pool2d(z, kernel_size=2, stride=2)
        
        z = self.conv_2(z)
        z = cplx.max_pool2d(z, kernel_size=2, stride=2)
        
        z = self.conv_3(z)
        z = cplx.max_pool2d(z, kernel_size=2, stride=2)
        
        z = self.conv_4(z)
        z = cplx.max_pool2d(z, kernel_size=2, stride=2)
        
        flatten = nn.Flatten()
        z = flatten(z)
        
        z = self.fc1(z)
        z = F.normalize(z)
        
        z = self.fc2(z)
        z = F.normalize(z)
        
        z = self.fc3(z)
        z = F.normalize(z)
        
        z = torch.abs(z)**2
        
        return z


######################################################################################
#                                     TESTS
######################################################################################
import pytest

@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
