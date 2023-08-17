import math
from functools import singledispatchmethod

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


def kl_mvn_diag_std(mean, logvar):
    """
    Between factorized normal distribution N(mean, sigma * I) and standard distribution N(0, I)

    References:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """
    return 0.5 * torch.sum(torch.exp(logvar) + mean.pow(2) - 1 - logvar, dim=1)


def kl_mvn_diag_diag(mean1, logvar1, mean2, logvar2):
    """KL(q||p)"""
    kl = 0.5 * torch.sum(
        logvar2 - logvar1 + (logvar1.exp() + (mean2 - mean1) ** 2) / logvar2.exp() - 1,
        dim=1,
    )
    return kl


def kl_bernoulli_bernoulli(p_logits: Tensor, q_logits: Tensor):
    """
    Reference:
        https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
    """
    p_probs, q_probs = torch.sigmoid(p_logits), torch.sigmoid(q_logits)
    t1 = p_probs * (p_probs / q_probs).log()
    t1[q_probs == 0] = math.inf
    t1[p_probs == 0] = 0
    t2 = (1 - p_probs) * ((1 - p_probs) / (1 - q_probs)).log()
    t2[q_probs == 1] = math.inf
    t2[p_probs == 1] = 0
    return t1 + t2


class KLDivergence(nn.Module):
    def __init__(self, p: str, q: str, prior=None):
        super().__init__()
        self.p, self.q = p, q
        if p == "mvn_diag" and q == "mvn_std":
            self._loss = kl_mvn_diag_std
        elif p == "mvn_diag" and q == "mvn_diag":
            self._loss = kl_mvn_diag_diag
        elif p == "bernoulli" and q == "bernoulli":
            self._loss = kl_bernoulli_bernoulli
        else:
            raise NotImplementedError

        self.prior = prior

    def forward(self, *args, **kwargs):
        return self._loss(*args, **kwargs)

    def extra_repr(self) -> str:
        return f"p='{self.p}', q='{self.q}'"


class LogMVNDiffLoss(nn.Module):
    def __init__(self, p, q, reduction="mean"):
        super().__init__()
        self.p, self.q = p, q
        self.eps = 1e-6

        if p == "mvn_diag" and q == "mvn_diag":
            self.loss = self.mvn_diag_diag
        else:
            raise NotImplementedError

        if reduction == "mean":
            self._reduction_fn = torch.mean
        elif reduction == "sum":
            self._reduction_fn = torch.sum
        else:
            raise NotImplementedError

    def _stddev(self, logscale):
        # return torch.exp(logscale / 2)
        return F.softplus(logscale - 5.0)

    def mvn_diag_diag(self, x, xmean, xlogscale, y, ymean, ylogscale):
        xstddev = self._stddev(xlogscale)
        ystddev = self._stddev(ylogscale)
        p = D.Independent(D.Normal(xmean, xstddev + self.eps), 1)
        q = D.Independent(D.Normal(ymean, ystddev + self.eps), 1)
        return self._reduction_fn(p.log_prob(x) - q.log_prob(y))

    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    def extra_repr(self) -> str:
        return f"p='{self.p}', q='{self.q}'"


class TabularReconstructionLoss(nn.Module):
    def __init__(self, num_idxs, cat_idxs, num_var=1, reduction="mean"):
        super().__init__()
        self.num_idxs = num_idxs
        self.cat_idxs = cat_idxs
        self.reduction = reduction
        self.var = num_var

    @staticmethod
    def cross_entropy_one_hot(input, target, reduction="mean"):
        return F.cross_entropy(input, target.max(dim=1)[1], reduction=reduction)

    def forward(self, logits, target, var=None):
        """
        Args:
            logits (Tensor): unnormalized scores
            targets (Tensor):
        """
        num_loss = 0.0
        if self.num_idxs:  # Numerical loss
            num_loss += 0.5 * (
                math.log(self.var)
                + F.mse_loss(logits[:, self.num_idxs], target[:, self.num_idxs])
                / self.var
            )

        cat_loss = 0.0
        for i, k in self.cat_idxs:  # Categorical loss
            if k - i == 1:  # bin
                cat_loss += F.binary_cross_entropy_with_logits(
                    logits[:, i:k], target[:, i:k], reduction=self.reduction
                )
            else:
                cat_loss += self.cross_entropy_one_hot(
                    logits[:, i:k], target[:, i:k], reduction=self.reduction
                )

        loss = num_loss + cat_loss
        logs = dict(num_loss=num_loss.item(), cat_loss=cat_loss.item())

        return loss, logs

    def extra_repr(self):
        return (
            f"num_idxs={self.num_idxs},\ncat_idxs={self.cat_idxs},\n"
            f"num_var={self.var}, reduction={self.reduction}"
        )


class FastMMD(nn.Module):
    def __init__(self, num_features: int = 500, gamma: float = 1):
        """
        Approximate the MMD by Random Fourier Features (Random Kitchen Sinks).
        Estimate of biased MMD.

        Args:
            num_features: number of random features
            gamma (float): to adjust to fit MMD
        """
        super().__init__()
        self.num_features = num_features
        # self.gamma = 4 / gamma
        self.gamma = 1 / gamma
        self.c0 = 2 * math.pi
        self.c1 = math.sqrt(2.0 / num_features)
        self.c2 = math.sqrt(2.0 / self.gamma)  # * 1 / math.sqrt(self.num_features)

    def psi(self, x: Tensor):
        """Feature expansion"""
        input_dim = x.size(1)
        w = torch.randn(input_dim, self.num_features)
        b = self.c0 * torch.rand(self.num_features)
        return self.c1 * torch.cos(self.c2 * (x @ w) + b)

    def forward(self, x: Tensor, y: Tensor):
        if x.nelement() == 0:
            x = torch.randn(1, x.size(1))
        if y.nelement() == 0:
            y = torch.randn(1, y.size(1))
        psi_x, psi_y = self.psi(x), self.psi(y)
        return torch.norm(psi_x.mean(0) - psi_y.mean(0), 2)

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, gamma={self.gamma:.2f}"
