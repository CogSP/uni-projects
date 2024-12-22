DEBUG_2 = False

from model import ConvBlock
import torch.nn as nn 

class DetectBlock(nn.Module):
    def __init__(self, k=3, s=1, p=1, c=3, reg_max=1, nc=1, mc=512, w=1):
        super(DetectBlock, self).__init__()
        
        #reg_max = controlla la precisione della regression sulla bounding box 
        #nc = number of classes
        self.box_conv1 = ConvBlock(k,s,p,c=c,dim=64)
        self.box_conv2 = ConvBlock(k,s,p,c=64,dim=64)
        #self.box_conv3 = nn.Conv2d(in_channels=64, out_channels=4*reg_max, kernel_size=1, stride=1, padding=0)
        # 4 + 1 + 1 = 6 out_channels
        self.box_conv3 = nn.Conv2d(in_channels=64, out_channels=6, kernel_size=1, stride=1, padding=0)
        
        
        #self.class_conv1 = ConvBlock(k,s,p,c,dim=64)
        #self.class_conv2 = ConvBlock(k,s,p,c=64,dim=64)
        #self.class_conv3 = nn.Conv2d(in_channels=64, out_channels=nc, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):  
        if DEBUG_2:
            print("[Detect:]")
            print(f"Input: {x.shape}")
            print("\t [Conv]")
        ret1 = self.box_conv1(x)
        if DEBUG_2:
            print(f"Output: {ret1.shape}")
            print("\t [Conv]")
        ret1 = self.box_conv2(ret1)
        if DEBUG_2:
            print(f"Output: {ret1.shape}")
            print("\t [Conv2D]")
            print(self.box_conv3)
        ret1 = self.box_conv3(ret1)
        if DEBUG_2:
            print(f"Output: {ret1.shape}")
        
        return ret1#, ret2
