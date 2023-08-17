from typing import Optional, Union, Sequence, Callable, List
from abc import ABCMeta
from types import LambdaType
import math

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from .shared import make_activation_fn, simple_repr


def labels2onehot(labels, input_shape, num_classes=-1):
    """Labels to one-hot representation 1d, 2d, etc."""
    num_dims = len(input_shape[1:])
    one_hot_labels = F.one_hot(labels, num_classes)
    return one_hot_labels.reshape(one_hot_labels.shape + (1,) * num_dims).repeat(
        1, 1, *input_shape[1:]
    )


class Lambda(nn.Module):
    def __init__(self, _lambda_):
        super().__init__()
        assert type(_lambda_) is LambdaType
        self._lambda_ = _lambda_

    def forward(self, *args, **kwargs):
        return self._lambda_(*args, **kwargs)


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims: Sequence[int] = None,
        nonlinearity: str = "relu",
        norm_module: bool = False,
        dropout=False,
        dropout_rate=0.2,
        bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        activation_fn = make_activation_fn(nonlinearity)
        norm_cls = nn.BatchNorm1d if norm_module else None

        self.has_norm = bool(norm_module)
        self.activation = repr(activation_fn)
        self.has_dropout = dropout

        # MLP Block
        layers = list()
        prev_dim, next_dim = input_dim, output_dim
        for dim in self.hidden_dims:
            if self.has_dropout:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(prev_dim, dim, bias=bias))
            if self.has_norm:
                layers.append(norm_cls(dim))
            layers.append(activation_fn)
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor):
        for module in self.layers:
            x = module(x)
        return x

    __repr__ = simple_repr

    def extra_repr(self):
        return (
            f"{self.input_dim}, {self.output_dim}, hidden_dims={self.hidden_dims}, activation={self.activation}, "
            f"norm={self.has_norm}, dropout={self.has_dropout}"
        )
