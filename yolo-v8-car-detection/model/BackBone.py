DEBUG = False
from model import ConvBlock, C2fBlock
import torch.nn as nn 


class BackBone(nn.Module):
    def __init__(self, k=3, s=2, p=1, depth=1):
        super(BackBone, self).__init__()
        
        self.conv1 = ConvBlock(k,s,p)
        self.conv2 = ConvBlock(k,s,p, dim=128, c=64)
        self.c2f = C2fBlock(k=1,s=1,p=0,depth_multiple=3*depth,dim=128, c=128)
        self.conv3 = ConvBlock(k,s,p, dim=256, c=128)
        self.c2f_second = C2fBlock(k=1,s=1,p=0,depth_multiple=6*depth,dim=256, c=256)
        self.conv4 = ConvBlock(k,s,p,dim=512, c=256)
        self.c2f_third = C2fBlock(k=1,s=1,p=0,depth_multiple=6*depth,dim=512, c=512)
        self.conv5 = ConvBlock(k,s,p,dim=1024, c=512)
        self.c2f_last = C2fBlock(k=1,s=1,p=0,depth_multiple=3*depth,dim=min(1024,512), c=512)
        
    def forward(self, x):
        if DEBUG:
            print("[Layer: Conv 0]")
            print(f"Input Tensor Shape:  {x.shape}")
        x = self.conv1(x)
        if DEBUG:
            print(f"Output Tensor Shape: {x.shape}")
            print("[Layer: Conv 1]")
            print(f"Input Tensor Shape:  {x.shape}")
        x = self.conv2(x)
        if DEBUG:
            print(f"Output Tensor Shape: {x.shape}")
            print("[Layer: C2f 2]")
            print(f"Input Tensor Shape:  {x.shape}")
        x = self.c2f(x)
        if DEBUG:
            print(f"Output Tensor Shape: {x.shape}")
            print("[Layer: Conv 3]")
            print(f"Input Tensor Shape:  {x.shape}")        
        x = self.conv3(x)
        if DEBUG:
            print(f"Output Tensor Shape: {x.shape}")
            print("[Layer: C2f 4]")
            print(f"Input Tensor Shape:  {x.shape}") 
        x_first = self.c2f_second(x)
        if DEBUG:
            print(f"Output Tensor Shape: {x_first.shape}")
            print("[Layer: Conv 5]")
            print(f"Input Tensor Shape:  {x_first.shape}") 
        x = self.conv4(x_first)
        if DEBUG:
            print(f"Output Tensor Shape: {x.shape}")
            print("[Layer: C2f 6]")
            print(f"Input Tensor Shape:  {x.shape}") 
        x_second = self.c2f_third(x)
        if DEBUG:
            print(f"Output Tensor Shape: {x_second.shape}")
            print("[Layer: Conv 7]")
            print(f"Input Tensor Shape:  {x_second.shape}")
        x = self.conv5(x_second)
        if DEBUG:
            print(f"Output Tensor Shape: {x.shape}")
            print("[Layer: C2f 8]")
            print(f"Input Tensor Shape:  {x.shape}")
        x_last = self.c2f_last(x)
        if DEBUG:
            print(f"Output Tensor Shape: {x_last.shape}")
        return x_first, x_second, x_last
