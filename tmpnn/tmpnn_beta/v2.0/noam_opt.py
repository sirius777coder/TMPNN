import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
        
def get_std_opt(parameters, d_model):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0)
    )


def rate(step):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def esm_rate(step,warmup,initial_lr,peak_lr):
    if step == 0:
        return initial_lr
    elif step <= warmup:
        return peak_lr/5000 * step
    else:
        return (step ** (-0.5)) * peak_lr * (warmup ** (0.5))

def transformer_rate(step,warmup,model_size):
    if step == 0:
        step = 1
    return model_size ** (-0.5) * min(step**(-0.5),step * warmup ** (-1.5))
    

def esm_optim_setup(parameters,warmup=5000,initial_lr=1e-7,peak_lr=1e-3):
    """
    Return a tuple of two elements for esm-if lr
        adam optimizer
        Lambda LR schuduler
    """
    optimizer = optim.Adam(parameters,lr=1,betas=(0.9,0.999),eps=1e-8)
    lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: esm_rate(step,warmup,initial_lr,peak_lr)
        )
    return optimizer, lr_scheduler

def transformer_optim_setup(parameters,model_size,warmup=5000):
    """
    Return a tuple of two elements for transformer lr
        adam optimizer
        Lambda LR schuduler
    """
    optimizer = optim.Adam(parameters,lr=1,betas=(0.9,0.999),eps=1e-8)
    lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: transformer_rate(step,warmup,model_size)
        )
    return optimizer, lr_scheduler