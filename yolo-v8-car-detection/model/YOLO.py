DEBUG = False 
import torch.nn as nn 
from model import BackBone, Neck, Head
import torch

class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        self.h1 = BackBone()
        self.h2 = Neck()
        self.h3 = Head()
        
    def forward(self, x): 
        if DEBUG:
            print("---------- Backbone ----------")
            print("[Backbone Input]")
            print(f"Input Tensor Shape: {x.shape}")
        res1, res2, res3 = self.h1(x)
        if DEBUG:
            print("[Backbone Output]")
            print(f"Output Tensor Shape: \n\t\t     {res1.shape}, \n\t\t     {res2.shape}, \n\t\t     {res3.shape}")
            print("------------------------------")
        if DEBUG:
            print("---------- Neck ----------")
            print("[Neck Input]")
            print(f"Input Tensor Shape:  \n\t\t     {res1.shape}, \n\t\t     {res2.shape}, \n\t\t     {res3.shape}")
        det1, det2, det3 = self.h2(res1, res2, res3)
        if DEBUG:
            print("[Neck Output]")
            print(f"Output Tensor Shape: \n\t\t     {det1.shape}, \n\t\t     {det2.shape}, \n\t\t     {det3.shape}")
            print("------------------------------")
        if DEBUG:
            print("---------- Head ----------")
            print("[Head Input]")
            print(f"Input Tensor Shape: \n\t\t  {det3.shape}")
        det3 = self.h3(det3)
        if DEBUG:
            print("[Head Output]")
            print(f"Output Tensor Bbox Loss: \n\t\t     {det3.shape}")
            print("------------------------------")

        m = torch.nn.Sigmoid()
        return m(det3)
    