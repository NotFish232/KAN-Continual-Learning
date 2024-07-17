from typing import Callable

import torch as T
from kan import KAN
from torch import nn
from torch.nn import functional as F
from math import sqrt, pi


def num_parameters(module: nn.Module) -> int:
    """
    Calculates the number of trainable parameters in a model

    Parameters
    ----------
    module : nn.Module
        A pytorch model

    Returns
    -------
    int
        Number of trainable parameters
    """

    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def mse_reg_loss(
    kan: KAN,
    lamb: float = 0.0,
    lamb_l1: float = 1.0,
    lamb_entropy: float = 2.0,
    lamb_coef: float = 0.0,
    lamb_coefdiff: float = 0.0,
    small_mag_threshold: float = 1e-16,
    small_reg_factor: float = 1.0,
) -> Callable:
    """
    MSE Loss with regularization term lifted from pykan's implmentation

    Returns
    -------
    Callable
        Callable which returns MSE Loss with regularization term
    """

    def reg() -> T.Tensor:
        def nonlinear(
            x: T.Tensor,
            th: float = small_mag_threshold,
            factor: float = small_reg_factor,
        ) -> T.Tensor:
            return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

        reg_ = T.tensor(0.0)
        for i in range(len(kan.acts_scale)):
            vec = kan.acts_scale[i].reshape(-1)

            p = vec / T.sum(vec)
            l1 = T.sum(nonlinear(vec))
            entropy = -T.sum(p * T.log2(p + 1e-4))
            reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

        # regularize coefficient to encourage spline to be zero
        for i in range(len(kan.act_fun)):
            coeff_l1 = T.sum(T.mean(T.abs(kan.act_fun[i].coef), dim=1))
            coeff_diff_l1 = T.sum(T.mean(T.abs(T.diff(kan.act_fun[i].coef)), dim=1))
            reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

        return reg_

    def _mse_reg_loss(Y_pred: T.Tensor, Y: T.Tensor) -> T.Tensor:
        return F.mse_loss(Y_pred, Y) + lamb * reg()

    return _mse_reg_loss


def gaussian(x: T.Tensor, mean: float, std: float) -> T.Tensor:
    """
    Basic implementation of the gaussian distribution

    Parameters
    ----------
    x : T.Tensor
        Values to evaluate gaussian distribution at

    mean : float
        Mean of gaussian distribution

    std : float
        Standard deviation of gaussian distribution

    Returns
    -------
    T.Tensor
        Tensor with same shape as x and with values of the gaussian distribution
    """

    return (1 / (std * sqrt(2 * pi))) * T.exp(-(1 / 2) * ((x - mean) / std) ** 2)
