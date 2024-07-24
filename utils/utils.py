import math
from math import pi, sqrt
from typing import Callable

import torch as T
from kan import KAN
from torch import nn


def partition_2d_graph(x: T.Tensor, num_intervals: int) -> T.Tensor:
    """
    Takes a tensor resulting from the cartesian product of two equal ranges
    and transforms it to a tensor partitioned into num_intervals ^ 2 quadrants
    input shape (# divisible by num_intervals ^ 2, n)
    output shape (num_intervals ^ 2, num / num_intervals ^ 2, n)

    Parameters
    ----------
    x : T.Tensor
        Tensor to partition
    num_intervals : int
        Number of intervals for both axes

    Returns
    -------
    T.Tensor
        Partitioned tensor
    """

    dim_1, dim_2 = x.shape

    dim_max = math.isqrt(dim_1 // num_intervals ** 2)

    x = x.reshape(math.isqrt(dim_1), math.isqrt(dim_1), -1)

    quadrants = []

    for i in range(num_intervals):
        for j in range(num_intervals):
            x_start = i * dim_max // num_intervals
            x_end = (i + 1) * dim_max // num_intervals
            y_start = j * dim_max // num_intervals
            y_end = (j + 1) * dim_max // num_intervals

            quadrant = x[x_start:x_end, y_start:y_end].reshape(1, -1, dim_2)
            quadrants.append(quadrant)
    
    concatenated_quadrants = T.concat(quadrants)

    return concatenated_quadrants


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


def kan_reg_term(
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

    def _reg() -> T.Tensor:
        def nonlinear(
            x: T.Tensor,
            th: float = small_mag_threshold,
            factor: float = small_reg_factor,
        ) -> T.Tensor:
            return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

        reg_ = T.tensor(0.0, device=kan.acts_scale[0].device)
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

        return lamb * reg_

    return _reg


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
