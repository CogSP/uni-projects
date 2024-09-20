import torch
import torch.nn as nn
from model import ConvBlock, Bottleneck

class C2fBlock(nn.Module):
    def __init__(self, k=1, s=1, p=0, c=3, depth_multiple=1, shortcut=True, dim=64, mc=512, w=1, flag=1):
        super(C2fBlock, self).__init__() 
        self.conv1 = ConvBlock(k=1,s=1,p=0,c=c,dim=dim,mc=mc,w=w,flag=flag)
        half_c= int(dim / 2)
        if half_c == 512:
            half_dim = 512
        else:
            half_dim= int(dim / 2)
            
        if flag == 0: 
            self.bottlenecks = nn.ModuleList([Bottleneck(k=3,s=1,p=1,c=256,dim=256) for _ in range(depth_multiple)])
            new_input = int(512 / 2) * (depth_multiple + 2)
            self.conv2 = ConvBlock(k,s,p,c=new_input,dim=dim,mc=mc,w=w)
        else:
            self.bottlenecks = nn.ModuleList([Bottleneck(k=3,s=1,p=1,c=half_c,dim=half_dim) for _ in range(depth_multiple)])
            new_input = int(dim / 2) * (depth_multiple + 2)
            self.conv2 = ConvBlock(k,s,p,c=new_input,dim=dim,mc=mc,w=w)
    
    def forward(self, x):
        x = self.conv1(x)
              
        # Split the input tensor into two halves along the channel dimension
        x1, x2 = torch.split(x, x.size(1) // 2, dim=1)
        
        # Process the other half (x2) through the bottlenecks
        bottleneck_outputs = []
        # append half of the input before processing
        bottleneck_outputs.append(x2.clone())
        for bott in self.bottlenecks:
            x2 = bott(x2)
            bottleneck_outputs.append(x2.clone())
            
        # this will concatenate half of the input before processing
        # and after each bottleneck processing  
        
        concatenated_bottleneck_outputs = torch.cat(bottleneck_outputs, dim=1)

        # add the other half
        x = torch.cat((x1, concatenated_bottleneck_outputs), dim=1)
        x = self.conv2(x)
        return x
