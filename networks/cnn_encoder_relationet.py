import torch.nn as nn


class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            #nn.Conv2d(3,64,kernel_size=3,padding=0),
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            #nn.Conv2d(64,64,kernel_size=3,padding=0),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)) ## added intentionally
        self.layer4 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)) ## added intentionally

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64