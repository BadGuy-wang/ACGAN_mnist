import numpy as np
import torch
import matplotlib.pyplot as plt
from model import Generator, Discriminator

if __name__ == '__main__':
    BATCH_SIZE = 50
    N_IDEAS = 100
    img_shape = (1, 28, 28)

    G = torch.load("G.pkl").cuda()
    D = torch.load('D.pkl').cuda()
    rand_noise = torch.randn((BATCH_SIZE, N_IDEAS, 1, 1)).cuda()

    G_imgs = G(rand_noise)
    D_fake_D, D_fake_C = D(G_imgs)
    p_fake = torch.squeeze(D_fake_D)

    index = D_fake_C.argmax(dim=1)
    G_imgs = G_imgs.cpu().detach().numpy()

    for i, img in enumerate(G_imgs):
        print(p_fake[i])
        plt.title(str(index[i]))
        plt.imshow(np.squeeze(img))
        plt.savefig('%d.png' % i)
        plt.show()