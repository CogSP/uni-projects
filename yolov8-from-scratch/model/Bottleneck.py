import torch.nn as nn
from model import ConvBlock

class Bottleneck(nn.Module):
    def __init__(self, k=3, s=1, p=1, c=3, dim=64, shortcut=True):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBlock(k,s,p,c,dim=dim,mc=512)
        self.conv2 = ConvBlock(k,s,p,c,dim=dim,mc=512)
        self.short = shortcut

    
    def forward(self, x):
        if self.short: 
            res = self.conv1(x)
            res = self.conv2(res)
            return x + res
        else: 
            res = self.conv1(x)
            res = self.conv2(x)
            return res 
