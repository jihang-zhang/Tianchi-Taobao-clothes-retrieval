import torch
import torch.nn as nn

def freeze_bn(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()
        module.weight.requires_grad = False
        module.bias.requires_grad   = False