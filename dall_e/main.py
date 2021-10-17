
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, grad
import random
import numpy as np
from torchvision.utils import save_image
import sys

use_sn = eval(sys.argv[1])

from encoder import Encoder
from decoder import Decoder

from quantize import Quantize

from laplace_loss import lap_transform, lap_inv_transform, lap_loss

from attack import pgd as pgd

#bottleneck = "none"
#bottleneck = "discrete"
#bottleneck = "gaussian"

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = Tru

bottleneck = sys.argv[2]

print('bottleneck', bottleneck)
print('use_sn', use_sn)

enc = Encoder(vocab_size=32).cuda()
dec = Decoder(vocab_size=32).cuda()

#mlp = nn.Sequential(nn.Linear(4*4*256, 4*4*256), nn.LeakyReLU(), nn.Linear(4*4*256,4*4*256)).cuda()

cce = nn.CrossEntropyLoss()

bs = 256
L = 1024
n_factors = 4

assert bs*n_factors >= L

quant = Quantize(4*4*32, L, n_factors).cuda() # 1 factor

classifier = nn.Sequential(nn.Linear(4*4*32, 512), nn.LeakyReLU(), nn.Linear(512,512), nn.LeakyReLU(), nn.Linear(512, 10)).cuda()

discrete_prior = nn.ParameterList([nn.Parameter(torch.zeros(1,L).cuda())])

gauss_up = nn.Linear(32*4*4, 64).cuda()
gauss_down = nn.Linear(32, 32*4*4).cuda()


params = list(enc.parameters()) + list(dec.parameters()) + list(quant.parameters()) + list(classifier.parameters()) + list(discrete_prior)

if bottleneck == 'gaussian':
    params += list(gauss_up.parameters()) + list(gauss_down.parameters())

opt = torch.optim.AdamW(params, lr=0.0005, weight_decay=1e-4, betas=(0.95,0.999))

attack_params = {'eps' : 0.5, 'eps_iter': 0.01, 'iter':200} #0.5/0.01

#scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [70,90])

hw = 32

#train_loader = torch.utils.data.DataLoader(datasets.MNIST('data',
#                                                          download=True,
#                                                          train=True,
#                                                          transform=transforms.Compose([
#                                                              transforms.Resize((hw,hw)),
#                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
#                                                          ])),
#                                           batch_size=bs,
#                                            drop_last=True,
#                                           shuffle=True)


nll_loss = nn.CrossEntropyLoss()

for epoch in range(0,100):

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data',
                                                          download=True,
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.Resize((hw,hw)),
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                          ])),
                                           batch_size=bs,
                                            drop_last=True,
                                           shuffle=True)


    loss_lst = []
    k = 0
    for (x,y) in train_loader:

        x = x.cuda()

        y = y.cuda()

        h = enc(x)
        q_loss = 0.0

        if epoch >= 5 and bottleneck == 'gaussian':
            #h --> q(z|x)
            #layer to map e_dim to e_dim*2 for mu/sigma.  q_loss is kl.  Return sample.  
            bs, e_dim, e_width, _ = h.shape
            h = h.permute(0,2,3,1).reshape((bs, e_width*e_width*e_dim))
            h = gauss_up(h)
            q_mu = h[:,:32]
            q_var = torch.exp(h[:,32:])

            q_loss += 0.01 * (q_mu**2 + q_var - torch.log(q_var)).mean()

            eps_n = torch.randn_like(q_mu)
            h = q_mu + eps_n*torch.sqrt(q_var)

            h = gauss_down(h)
            h = h.reshape((bs, e_width, e_width, e_dim)).permute(0,3,1,2)

        bs, e_dim, e_width, _ = h.shape

        h_enc = h*1.0

        h = h.permute(0,2,3,1).reshape((bs, 1, e_width*e_width*e_dim))

        #can quantize here

        #h = mlp(h)

        if epoch >= 5 and bottleneck == 'discrete':
            h, q_loss, ind = quant(h, allow_init=True)

            #print('ind shape', ind.shape) #(256,1).  
            #q_loss += nll_loss(discrete_prior[0].repeat(256,1), ind.flatten())



        # Learn prior p(z).  (bsz, 512) vector, use with (bsz,) size vector of ints.  

        h = h.reshape((bs, e_width, e_width, e_dim)).permute(0,3,1,2)

        xr = dec(h)

        xr = xr[:,0:1]

        loss = F.mse_loss(x, xr)

        if False:
            y_classify = classifier(h.detach().reshape((h.shape[0], -1)))
            c_loss = cce(y_classify, y) * 0.1
        else:
            c_loss = 0.0

        #loss = lap_loss(lap_transform(x), xr[:,0:3], xr[:,3:6])
        #xr = lap_inv_transform(F.sigmoid(xr[:,0:3]))

        adv_loss = (grad((xr**2).sum(), h, create_graph=True)[0]**2).mean()

        if use_sn and epoch > 10:
            adv_loss_use = 0.0001 * adv_loss
        else:
            adv_loss_use = 0.0

        all_loss = loss + q_loss + c_loss + adv_loss_use

        opt.zero_grad()
        all_loss.backward()
        opt.step()

        loss_lst.append(loss.data)
        k += 1

        sys.stdout.flush()

        if k % 50 == 1:
            print(epoch, k, sum(loss_lst) / len(loss_lst))
            loss_lst = []

            #print('c_loss', c_loss)
            #print('prior', F.softmax(discrete_prior[0], dim=1))

            if bottleneck == 'discrete':
                z_adv, x_adv = pgd(h_enc, x, attack_params, decoder=dec, quantizer=quant)
            else:
                z_adv, x_adv = pgd(h_enc, x, attack_params, decoder=dec, quantizer=None)

            print('x min max', x.min(), x.max(), 'xadv min max', x_adv.min(), x_adv.max())
            print('xrec_adv clamped', ((x - x_adv.clamp(0.0,1.0))**2).mean())
            print('xrec_adv', ((x - x_adv)**2).mean())
            print('adv_loss', adv_loss)

            save_image(x[:64], 'orig_%s_sn_%s.png' % (bottleneck, use_sn))
            save_image(xr[:64], 'rec_%s_sn_%s.png' % (bottleneck, use_sn))
            save_image(x_adv[:64,0:1], 'adv_%s_sn_%s.png' % (bottleneck, use_sn))

            torch.save((enc, quant), 'encoder_quantize_%s.pt' % (bottleneck == 'discrete'))

    #scheduler.step()


