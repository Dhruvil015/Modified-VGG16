import torch
import torch.nn as nn # All neural network moduels, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameter
from torch.utils.data import DataLoader # Gives easier data management and creates mini batches
import torchvision.datasets as datasets # Has standard dataset we can import nice way
import torchvision.transforms as transforms # Transformation we can perform on our dataset

VGG_16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG_Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_16)
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x        
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) is int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(x),  
                           nn.ReLU()]
                in_channels = x
            elif type(x) is str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        return nn.Sequential(*layers)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VGG_Net(in_channels=3, num_classes=10).to(device)
x = torch.randn(1, 3, 224, 224).to(device)
print(model(x).shape)