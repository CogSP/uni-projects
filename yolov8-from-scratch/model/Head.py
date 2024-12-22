import torch.nn as nn 
from model import DetectBlock

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.det3 = DetectBlock(c=512)
        
    def forward(self, x3):
        return self.det3(x3)
