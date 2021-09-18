
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, grad
import random
import numpy as np
from torchvision.utils import save_image

from encoder import Encoder
from decoder import Decoder

enc = Encoder().cuda()
dec = Decoder().cuda()

opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()))

train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data',
                                                          download=True,
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                          ])),
                                           batch_size=256,
                                            drop_last=True,
                                           shuffle=True)

for epoch in range(0,50):

    loss_lst = []
    k = 0
    for (x,y) in train_loader:

        x = x.cuda()

        h = enc(x)

        bs, e_dim, e_width, _ = h.shape

        h = h.permute(0,2,3,1).reshape((bs, e_width*e_width, e_dim))

        #can quantize here

        h = h.reshape((bs, e_width, e_width, e_dim)).permute(0,3,1,2)

        xr = dec(h)[:,0:3]

        loss = F.mse_loss(x, xr)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_lst.append(loss.data)
        k += 1

        if k % 50 == 1:
            print(epoch, k, sum(loss_lst) / len(loss_lst))
            loss_lst = []

            save_image(x[:64], 'orig.png')
            save_image(xr[:64], 'rec.png')





