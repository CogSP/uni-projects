DEBUG = False 
import torch.nn as nn 
import torch 
from model import C2fBlock, SPPF, ConvBlock

class Neck(nn.Module):
    def __init__(self, depth=1, scale=2):
        super(Neck, self).__init__()
        
        self.sppf = SPPF(k=1,dim=1024,c=512)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.c2f_block1 = C2fBlock(dim=512,c=1024,flag=1,shortcut=False)
        self.c2f_block2 = C2fBlock(dim=256,c=768,flag=1,shortcut=False) 
        self.c2f_block3 = C2fBlock(dim=512,c=768,flag=1,shortcut=False)
        self.c2f_block4 = C2fBlock(dim=1024,c=1024,flag=0,shortcut=False)
        self.conv1 = ConvBlock(k=3,s=2,p=1,dim=256,c=256)
        self.conv2 = ConvBlock(k=3,s=2,p=1,dim=512,c=512)
        
    def forward(self, x_first, x_second, x_last):
        if DEBUG:
            print("[Layer: SPPF 9]")
            print(f"Input Tensor Shape:  {x_last.shape}")
        out_sppf = self.sppf(x_last)
        if DEBUG:
            print(f"Output Tensor Shape: {out_sppf.shape}")
            print("[Layer: Upsample 10]")
            print(f"Input Tensor Shape:  {out_sppf.shape}")
        x = self.upsample1(out_sppf)
        if DEBUG:
            print(f"Output Tensor Shape: {x.shape}")
            print("[Layer: Concat 11]")
            print(f"Input Tensor Shape:  {x.shape}, {x_second.shape}")
        x = torch.cat((x,x_second), dim=1)
        if DEBUG:
            print(f"Output Tensor Shape: {x.shape}")
            print("[Layer: C2f 12]")
            print(f"Input Tensor Shape:  {x.shape}")
        conc1 = self.c2f_block1(x)
        if DEBUG:
            print(f"Output Tensor Shape: {conc1.shape}")
            print("[Layer: Upsample 13]")
            print(f"Input Tensor Shape:  {conc1.shape}")
        x = self.upsample2(conc1)
        if DEBUG:
            print(f"Output Tensor Shape: {x.shape}")
            print("[Layer: Concat 14]")
            print(f"Input Tensor Shape:  {x.shape}, {x_first.shape}")
        x = torch.cat((x,x_first), dim=1)
        if DEBUG:
            print(f"Output Tensor Shape: {x.shape}")
            print("[Layer: C2f 15]")
            print(f"Input Tensor Shape:  {x.shape}")
        det1 = self.c2f_block2(x)
        if DEBUG:
            print(f"Output Tensor Shape: {det1.shape}")
            print("[Layer: Conv 16]")
            print(f"Input Tensor Shape:  {det1.shape}")
        x = self.conv1(det1)
        if DEBUG:
            print(f"Output Tensor Shape: {x.shape}")
            print("[Layer: Concat 17]")
            print(f"Input Tensor Shape:  {x.shape}, {conc1.shape}")
        x = torch.cat((x,conc1), dim=1)
        if DEBUG:
            print(f"Output Tensor Shape: {x.shape}")
            print("[Layer: C2f 18]")
            print(f"Input Tensor Shape:  {x.shape}")
        det2 = self.c2f_block3(x)
        if DEBUG:
            print(f"Output Tensor Shape: {det2.shape}")
            print("[Layer: Conv 19]")
            print(f"Input Tensor Shape:  {det2.shape}")
        x = self.conv2(det2)
        if DEBUG:
            print(f"Output Tensor Shape: {x.shape}")
            print("[Layer: Concat 20]")
            print(f"Input Tensor Shape:  {x.shape}, {out_sppf.shape}")
        x = torch.cat((x,out_sppf), dim=1)
        if DEBUG:
            print(f"Output Tensor Shape: {x.shape}")
            print("[Layer: C2f 21]")
            print(f"Input Tensor Shape:  {x.shape}")
        det3 = self.c2f_block4(x)
        if DEBUG:
            print(f"Output Tensor Shape: {det3.shape}")
        return det1, det2, det3
