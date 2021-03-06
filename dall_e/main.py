
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
from quantize import Quantize

from laplace_loss import lap_transform, lap_inv_transform, lap_loss

enc = Encoder(vocab_size=256).cuda()
dec = Decoder(vocab_size=256).cuda()

#mlp = nn.Sequential(nn.Linear(4*4*256, 4*4*256), nn.LeakyReLU(), nn.Linear(4*4*256,4*4*256)).cuda()

bs = 256
L = 1024
n_factors = 16

assert bs*n_factors >= L

quant = Quantize(4*4*256, L, n_factors).cuda() # 1 factor

params = list(enc.parameters()) + list(dec.parameters()) + list(quant.parameters())
opt = torch.optim.AdamW(params, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [50,80])

hw = 32

train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data',
                                                          download=True,
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.Resize((hw,hw)),
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                          ])),
                                           batch_size=bs,
                                            drop_last=True,
                                           shuffle=True)


for epoch in range(0,100):

    loss_lst = []
    k = 0
    for (x,y) in train_loader:

        x = x.cuda()


        h = enc(x)

        bs, e_dim, e_width, _ = h.shape

        h = h.permute(0,2,3,1).reshape((bs, 1, e_width*e_width*e_dim))

        #can quantize here

        #h = mlp(h)

        if epoch >= 10:
            h, q_loss, ind = quant(h)
        else:
            q_loss = 0.0

        h = h.reshape((bs, e_width, e_width, e_dim)).permute(0,3,1,2)

        xr = dec(h)

        xr = xr[:,0:3]
        loss = F.mse_loss(x, xr)
      
        #loss = lap_loss(lap_transform(x), xr[:,0:3], xr[:,3:6])
        #xr = lap_inv_transform(F.sigmoid(xr[:,0:3]))

        all_loss = loss + q_loss

        opt.zero_grad()
        all_loss.backward()
        opt.step()

        loss_lst.append(loss.data)
        k += 1

        if k % 50 == 1:
            print(epoch, k, sum(loss_lst) / len(loss_lst))
            loss_lst = []

            save_image(x[:64], 'orig.png')
            save_image(xr[:64], 'rec.png')


    scheduler.step()


