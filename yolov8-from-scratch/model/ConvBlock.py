import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, k, s, p, c=3, dim=64, mc=512, w=1,flag=1):
        super(ConvBlock, self).__init__()
        dim = int(dim)
        out = min(dim,mc)*w
        self.conv = nn.Conv2d(in_channels=c, out_channels=out, kernel_size=k, stride=s, padding=p)
        self.batch_norm = nn.BatchNorm2d(num_features=out)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x
