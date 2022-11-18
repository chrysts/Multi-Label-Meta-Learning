import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, feature_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(feature_size*2,64,kernel_size=3,padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        #self.fc1 = nn.Linear(feature_size*3*3,hidden_size)
        self.fc1 = nn.Linear(feature_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)
        self.weights_init()

    def forward(self,x):
        out = self.layer1(x)
        #out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.1)
            m.bias.data = torch.ones(m.bias.data.size())





class RelationNetworkResNet(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, feature_size, hidden_size):
        super(RelationNetworkResNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(feature_size*2,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(512,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)
        self.weights_init()

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())