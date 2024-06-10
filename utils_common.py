import numpy as np

import torch
import torch.nn.functional as F

def classifier_fn(net, images, DEVICE='cpu'):
    '''
    args
        net: The model to be explained.
        images: The target instance which can be perturbed.
        DEVICE: The device on which to perform computations ('cpu' or 'cuda').

    returns
        The probability of the target classes.
    '''
    net.eval()
    images = torch.tensor(images).float()
    inputs = images.to(DEVICE)
    net.to(DEVICE)
    
    logits = net(inputs)
    probs = F.softmax(logits, dim=-1)
    return probs.detach().cpu().numpy()
