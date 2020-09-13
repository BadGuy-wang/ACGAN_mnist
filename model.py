import os

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()

        strides = [1, 2, 2, 2]
        padding = [0, 1, 1, 1]
        channels = [input_size,
                    256, 128, 64, 1]  # 1表示一维
        kernels = [4, 3, 4, 4]

        model = []
        for i, stride in enumerate(strides):
            model.append(
                nn.ConvTranspose2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    stride=stride,
                    kernel_size=kernels[i],
                    padding=padding[i]
                )
            )

            if i != len(strides) - 1:
                model.append(
                    nn.BatchNorm2d(channels[i + 1])
                )
                model.append(
                    nn.ReLU()
                )
            else:
                model.append(
                    nn.Tanh()
                )

        self.main = nn.Sequential(*model)

    def forward(self, x):
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()

        strides = [2, 2, 2]
        padding = [1, 1, 1]
        channels = [input_size,
                    64, 128, 256]  # 1表示一维
        kernels = [4, 4, 4]

        model = []
        for i, stride in enumerate(strides):
            model.append(
                nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    stride=stride,
                    kernel_size=kernels[i],
                    padding=padding[i]
                )
            )
            model.append(
                nn.BatchNorm2d(channels[i + 1])
            )
            model.append(
                nn.LeakyReLU(0.2)
            )

        self.main = nn.Sequential(*model)
        self.D = nn.Sequential(
            nn.Linear(3 * 3 * 256, 1),
            nn.Sigmoid()
        )
        self.C = nn.Sequential(
            nn.Linear(3 * 3 * 256, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.main(x).view(x.shape[0], -1)
        return self.D(x), self.C(x)


# if __name__ == '__main__':
#     N_IDEAS = 100
#     G = Generator(N_IDEAS, )
#     rand_noise = torch.randn((10, N_IDEAS, 1, 1))
#     print(G(rand_noise).shape)
#
#     DOWNLOAD_MNIST = False
#     mnist_root = '../Conditional-GAN/mnist/'
#     if not (os.path.exists(mnist_root)) or not os.listdir(mnist_root):
#         # not mnist dir or mnist is empyt dir
#         DOWNLOAD_MNIST = True
#
#     train_data = torchvision.datasets.MNIST(
#         root=mnist_root,
#         train=True,  # this is training data
#         transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
#         # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
#         download=DOWNLOAD_MNIST,
#     )
#     D = Discriminator(1)
#     print(len(train_data))
#     cel = nn.CrossEntropyLoss()
#     train_loader = Data.DataLoader(dataset=train_data, batch_size=2, shuffle=True)
#     for step, (x, y) in enumerate(train_loader):
#         print(x.shape)
#         d, c = D(x)
#         print(d.shape)
#         print(c.shape)
#         print(c.sum(dim=1))
#         print(cel(c, y))
#         break