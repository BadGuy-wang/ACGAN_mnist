import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model import Generator, Discriminator
import torchvision

import matplotlib.pyplot as plt

if __name__ == '__main__':
    LR = 0.0002
    EPOCH = 20  # 50
    BATCH_SIZE = 100
    N_IDEAS = 100
    DOWNLOAD_MNIST = False

    mnist_root = '../Conditional-GAN/mnist/'
    if not (os.path.exists(mnist_root)) or not os.listdir(mnist_root):
        # not mnist dir or mnist is empyt dir
        DOWNLOAD_MNIST = True

    train_data = torchvision.datasets.MNIST(
        root=mnist_root,
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=DOWNLOAD_MNIST,
    )

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    torch.cuda.empty_cache()
    G = Generator(N_IDEAS).cuda()
    D = Discriminator(1).cuda()

    optimizerG = torch.optim.Adam(G.parameters(), lr=LR)
    optimizerD = torch.optim.Adam(D.parameters(), lr=LR)

    c_c = nn.NLLLoss()  # criterion for classifying

    for epoch in range(EPOCH):
        tmpD, tmpG = 0, 0
        for step, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            rand_noise = torch.randn((x.shape[0], N_IDEAS, 1, 1)).cuda()
            G_imgs = G(rand_noise)

            D_fake_D, D_fake_C = D(G_imgs)
            D_real_D, D_real_C = D(x)

            p_fake = torch.squeeze(D_fake_D)
            p_real = torch.squeeze(D_real_D)

            c_fake = c_c(D_fake_C, y)
            c_real = c_c(D_real_C, y)

            D_D = -torch.mean(torch.log(p_real) + torch.log(1 - p_fake))  # same as GAN
            D_C = c_real + c_fake
            D_loss = D_D + D_C

            optimizerD.zero_grad()
            D_loss.backward(retain_graph=True)
            optimizerD.step()

            rand_noise = torch.randn((x.shape[0], N_IDEAS, 1, 1)).cuda()
            G_imgs = G(rand_noise)
            D_fake_D, D_fake_C = D(G_imgs)
            D_real_D, D_real_C = D(x)
            p_fake = torch.squeeze(D_fake_D)
            c_fake = c_c(D_fake_C, y)

            G_loss = -torch.mean(torch.log(p_fake)) + c_fake  # left part is same as GAN
            optimizerG.zero_grad()
            G_loss.backward(retain_graph=True)
            optimizerG.step()

            tmpD_ = D_loss.cpu().detach().data
            tmpG_ = G_loss.cpu().detach().data
            tmpD += tmpD_
            tmpG += tmpG_

        tmpD /= (step + 1)
        tmpG /= (step + 1)
        print(
            'epoch %d avg of loss: D: %.6f, G: %.6f' % (epoch, tmpD, tmpG)
        )
        if epoch % 2 == 0:
            plt.imshow(torch.squeeze(G_imgs[0].cpu().detach()).numpy())
            plt.show()
    torch.save(G, 'G.pkl')
    torch.save(D, 'D.pkl')