from model import ConvBlock
import torch 
import torch.nn as nn

class SPPF(nn.Module):
    def __init__(self, k=3, s=1, p=0, c=3, dim=64):
        super(SPPF, self).__init__() 
        
        self.conv1 = ConvBlock(k=k,s=s,p=0,c=c,dim=dim)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.conv2 = ConvBlock(k=3,s=1,p=1,c=4*c,dim=dim)
        
    def forward(self, x):
        x = self.conv1(x)
        pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        pool3 = self.pool3(x)
        x = torch.cat([x, pool1, pool2, pool3], dim=1)
        x = self.conv2(x)
        return x
