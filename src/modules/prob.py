from typing import Optional, Sequence, Callable, List
from abc import ABCMeta, abstractmethod

import torch.nn
from torch import Tensor
import torch.nn.functional as F
import torch.distributions as D

from .shared import *


class ProbabilisticModule(nn.Module, metaclass=ABCMeta):
    """
    Abstract Base Class for a parameterized distribution module

    Wraps torch.nn.Module to input of torch.distribution.Distribution

    Args:
        input_dim: input dim to torch Distribution adapter
        output_dim: output dim to torch Distribution adapter
    """

    def __init__(
        self,
        block: nn.Module,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.block = block
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor):
        """Outputs distributions parameters"""
        raise NotImplementedError

    @abstractmethod
    def distribution(self, x: Tensor):
        raise NotImplementedError

    @abstractmethod
    def point_and_dist(self, x: Tensor):
        """Outputs unormalized score point estimate (e.g. logits, mean) and respective distribution"""
        # FIXME: mean_and_dist?
        raise NotImplementedError

    @abstractmethod
    def dsample(self, param: Tensor, *params: Tensor):
        """Differentiable (approximate) sample"""
        raise NotImplementedError

    def sample(self, x: Tensor, value: Tensor, *args, **kwargs):
        """Distribution class wrapper"""
        dist = self.distribution(x)
        return dist.sample(*args, **kwargs)

    def log_prob(self, x: Tensor, value: Tensor):
        """Distribution class wrapper"""
        dist = self.distribution(x)
        return dist.log_prob(value)

    def sample_with_log_prob(self, x: Tensor):
        """Distribution class wrapper"""
        dist = self.distribution(x)
        sample = dist.sample()
        return sample, dist.log_prob(sample)

    __repr__ = simple_repr

    def extra_repr(self) -> str:
        repr_str = "\n\tblock=" + repr(self.block)
        repr_str += f", input_dim={self.input_dim}, output_dim={self.output_dim}"
        return repr_str


class MultivariateNormalDiag(ProbabilisticModule):
    def __init__(
        self,
        block: nn.Module,
        input_dim: int,
        output_dim: int,
    ):
        """
        Multivariate Normal distribution with diagonal covariance
        """
        super().__init__(block, input_dim, output_dim)

        # Adapters
        self.loc = nn.Linear(input_dim, output_dim)
        self.logscale = nn.Linear(input_dim, output_dim)

        # Init
        self.loc.apply(lambda p: init_weights(p, nonlinearity="linear"))
        self.logscale.apply(lambda p: init_weights(p, nonlinearity="linear"))

    def forward(self, x: Tensor):
        """Get distribution parameters"""
        h = self.block(x)
        loc, logscale = self.loc(h), self.logscale(h)
        return loc, logscale

    def _stddev(self, logscale):
        # return torch.exp(logscale / 2)
        return F.softplus(logscale - 5.0)

    def distribution(self, x):
        mean, logscale = self(x)
        stddev = self._stddev(logscale)
        mvn = D.Independent(D.Normal(mean, stddev), 1)
        # mvn = td.MultivariateNormal(mean, scale_tril=torch.diag(stddev**2))
        return mvn

    def point_and_dist(self, x):
        dist = self.distribution(x)
        return dist.mean, dist

    def dsample(self, mean, logscale_diag):
        """Reparameterization Trick"""
        return torch.randn_like(mean).mul_(self._stddev(logscale_diag)).add_(mean)


class Bernoulli(ProbabilisticModule):
    def __init__(
        self,
        block: nn.Module,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__(block, input_dim, output_dim)

        # Adapter
        self.logits = nn.Linear(input_dim, output_dim)

        # Init
        self.logits.apply(lambda p: init_weights(p, nonlinearity="linear"))

    def forward(self, x: Tensor):
        h = self.block(x)
        return self.logits(h).reshape(-1)

    def distribution(self, x):
        logits = self(x)
        return D.Bernoulli(logits=logits)

    def point_and_dist(self, x):
        dist = self.distribution(x)
        return dist.logits.reshape(-1), dist

    def dsample(self, logits):
        raise NotImplementedError
