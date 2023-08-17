from typing import Tuple

import torch
from torch import nn
from torch import Tensor


def simple_repr(obj):
    name = obj.__class__.__name__
    extra_repr = obj.extra_repr()
    return name + "(" + extra_repr + ")"


def get_input_shape(input_shape, context_dim) -> Tuple[int, ...]:
    """Find new input_shape from the context_dim"""
    if len(input_shape) == 1:  # Flat
        return (input_shape[0] + context_dim,)
    elif len(input_shape) == 2:  # Wrong shape
        raise ValueError("Images should be of shape CxHxW")
    elif len(input_shape) == 3:  # Images
        return input_shape[0] + context_dim, input_shape[1:]
    else:
        raise NotImplementedError


def combine_features(x: Tensor, y: Tensor = None):
    if isinstance(y, Tensor):
        yy = y.reshape(-1, 1) if len(y.shape) == 1 else y
        x = torch.cat([x.float(), yy.float()], dim=1)
        # x = torch.cat([x.float(), x.size(1) * yy.float()], dim=1)
    return x


def make_activation_fn(activation):
    if activation == "relu":
        activation_fn = nn.ReLU(inplace=True)
    elif activation == "leaky_relu":
        activation_fn = nn.LeakyReLU(0.2)
    else:
        raise NotImplementedError
    return activation_fn


@torch.no_grad()
def init_weights(module: nn.Module, nonlinearity="relu"):
    classname = module.__class__.__name__
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity=nonlinearity)
        if module.bias is not None:
            torch.nn.init.normal_(module.bias.data, std=0.01)
    elif isinstance(module, torch.nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity=nonlinearity)
        if module.bias is not None:
            torch.nn.init.normal_(module.bias.data, std=0.01)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.constant_(module.bias, 0.0)
    else:
        pass


def freeze(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False
