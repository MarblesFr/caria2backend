import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 60, 4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm2d(60)
        self.conv2 = nn.Conv2d(60, 100, 4, stride=2, padding=1)
        self.relu2 = nn.LeakyReLU(0.2)
        self.bn2 = nn.BatchNorm2d(100)
        self.conv3 = nn.Conv2d(100, 140, 4, stride=2, padding=1)
        self.relu3 = nn.LeakyReLU(0.2)
        self.bn3 = nn.BatchNorm2d(140)
        self.conv4 = nn.Conv2d(140, 180, 4, stride=2, padding=1)
        self.relu4 = nn.LeakyReLU(0.2)
        self.bn4 = nn.BatchNorm2d(180)
        self.conv5 = nn.Conv2d(180, 220, 4, stride=2, padding=1)
        self.relu5 = nn.LeakyReLU(0.2)
        self.bn5 = nn.BatchNorm2d(220)
        self.f1 = nn.Flatten()
        self.lin = nn.Linear(220*8*24, 64)
        self.bn6 = nn.BatchNorm1d(64)

        # self.convt1 = nn.ConvTranspose2d(200, 160, 4, stride=2, padding=1)
        # self.relu3 = nn.ReLU()
        # self.convt2 = nn.ConvTranspose2d(160, 120, 4, stride=2, padding=1)
        # self.relu4 = nn.ReLU()
        # self.convt3 = nn.ConvTranspose2d(120, 1, 4, stride=2, padding=1)

    def encode(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.bn3(self.relu3(self.conv3(x)))
        x = self.bn4(self.relu4(self.conv4(x)))
        x = self.bn5(self.relu5(self.conv5(x)))
        x = self.f1(x)
        x = self.lin(x)
        x = self.bn6(x)
        return x

    # def decode(self, z):
    #      z = self.relu3(self.convt1(z))
    #      z = self.relu4(self.convt2(z))
    #      return self.convt3(z)

    def forward(self, x):
        y = self.encode(x)
        #y = self.decode(y)
        return y