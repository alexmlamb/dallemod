import torch
import torch.nn.functional as F

# takes (bs, 3, h, w).  Add across c/h/w, mean across bs.  
def lap_loss(x, mu, b): 
    #ln_b = ln_b*0.0 + 0.1
    #ln_b = torch.tanh(ln_b)
    #b = torch.exp(ln_b)

    b = F.softplus(b) + 0.0001
    ln_b = torch.log(b)

    lhs = 2 + ln_b + torch.log(x) + torch.log(1-x)
    
    rhs = torch.abs(torch.logit(x) - mu) / b

    return (lhs + rhs).mean()

def lap_transform(x, eps=0.1): 
    return (1 - 2*eps) * x + eps

def lap_inv_transform(phi, eps=0.1): 
    return torch.clamp((phi - eps) / (1 - 2*eps), 0.0, 1.0)

    # 1.0-0.1 / (1 - 0.2) = 0.9/0.8

if __name__ == "__main__":

    x = torch.rand(1) * 255

    t = transform(x,0.0001)

    it = inv_transform(t, 0.0001)

    t = t.repeat(10)

    mu = torch.rand(10)
    ln_b = torch.randn(10)

    print('mu', mu)

    print('sg', torch.exp(ln_b))

    print('t', t)

    l = loss(t, mu, ln_b)

    print('loss', l)

