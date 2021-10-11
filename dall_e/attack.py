
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import grad, Variable

def to_var(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def pgd(z, x, attack_params, decoder, quantizer=None):
    epsilon = attack_params['eps']
    #x_diff = 2 * 0.025 * (to_var(torch.rand(x.size())) - 0.5)
    #x_diff = torch.clamp(x_diff, -epsilon, epsilon)
    z_adv = to_var(z.data)

    minz = z.min().cpu().detach().numpy().tolist()
    maxz = z.max().cpu().detach().numpy().tolist()


    for i in range(0, attack_params['iter']):
        #c_pre = classifier(x_adv)
        #loss = loss_func(c_pre) # gan_loss(c, is_real,compute_penalty=False)
        
        h = z_adv
        bs, e_dim, e_width, _ = h.shape
        h = h.permute(0,2,3,1).reshape((bs, 1, e_width*e_width*e_dim))

        if quantizer is not None:
            h, _, _ = quantizer(h) 
      
        h = h.reshape((bs, e_width, e_width, e_dim)).permute(0,3,1,2)

        xrec = decoder(h)

        #loss = ((x - xrec)**2).mean() * 0.0

        overbias = torch.maximum(xrec - x, torch.Tensor([0.0]).cuda())
        underbias = torch.maximum(x - xrec, torch.Tensor([0.0]).cuda())

        loss = overbias * torch.lt(xrec, torch.Tensor([1.0]).cuda()) + underbias * torch.gt(xrec, torch.Tensor([0.0]).cuda())

        loss = loss.mean()

        #print('adv', i, loss)

        nz_adv = z_adv + attack_params['eps_iter']*torch.sign(grad(loss, z_adv,retain_graph=False)[0])
        nz_adv = torch.clamp(nz_adv, minz, maxz)
        z_diff = nz_adv - z
        z_diff = torch.clamp(z_diff, -epsilon, epsilon)
        nz_adv = z + z_diff
        z_adv = to_var(nz_adv.data)

    return z_adv, xrec





