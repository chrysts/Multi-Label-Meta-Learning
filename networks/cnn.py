import torchvision.models as models
import torch.nn as nn
import torch


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        #nn.LeakyReLU(),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class CNN(nn.Module):


    def __init__(self, label_size):
        super(CNN, self).__init__()
        self.cnn = ConvNet()
        self.linear = nn.Sequential(nn.Linear(1600, 512),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(256, 64),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(64, label_size),
                                    nn.ReLU()
                                    )
        self.cnn.classifier = Identity()

    def forward(self, X):
        out = self.cnn(X)

        out = self.linear(out)

        return out

    def forward_features(self, X):

        out = self.cnn(X)

        return 