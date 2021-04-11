import torch.nn as nn

# generator
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.conv0 = nn.ConvTranspose2d(nz, 260, kernel_size=(8,24),
                                        stride=1, padding=0, bias=False)
        self.rel0 = nn.LeakyReLU(0.2)
        self.bn0 = nn.BatchNorm2d(260)
        # nz will be the input to the first convolution
        self.conv1 = nn.ConvTranspose2d(
            260, 220, kernel_size=4,
            stride=2, padding=1, bias=False)
        self.relu1 = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(220)
        self.d1 = nn.Dropout2d(0.2)
        self.conv2 = nn.ConvTranspose2d(
            220, 180, kernel_size=4,
            stride=2, padding=1, bias=False)
        self.relu2 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm2d(180)
        self.d2 = nn.Dropout2d(0.2)
        self.conv3 = nn.ConvTranspose2d(
            180, 100, kernel_size=4,
            stride=2, padding=1, bias=False)
        self.relu3 = nn.ReLU(True)
        self.bn3 = nn.BatchNorm2d(100)
        self.d3= nn.Dropout2d(0.2)
        self.conv4 = nn.ConvTranspose2d(
            100, 60, kernel_size=4,
            stride=2, padding=1, bias=False)
        self.relu4 = nn.ReLU(True)
        self.bn4 = nn.BatchNorm2d(60)
        self.d4 = nn.Dropout2d(0.2)
        self.conv5 = nn.ConvTranspose2d(
            60, 3, kernel_size=4,
            stride=2, padding=1, bias=False)

        self.t1 = nn.Sigmoid()

    def forward(self, x):
        x = self.conv0(x)
        x = self.rel0(x)
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.d1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.d2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.d3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.d4(x)
        x = self.conv5(x)
        x = self.t1(x)
        return x