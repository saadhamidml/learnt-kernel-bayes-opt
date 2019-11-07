import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import sklearn.model_selection as skl_ms


def synthetic_kernel(x, y):
    """Define the kernel for the synthetic data."""
    diff_outer = np.subtract.outer(x, y)
    diff = np.diagonal(diff_outer, axis1=1, axis2=3)
    diff_sq = diff ** 2
    diff_sq_sum = diff_sq.sum(axis=2)
    dist = np.sqrt(diff_sq_sum)
    return np.sinc(dist)


def generate(start, end, step):
    """Generate the data."""
    x = torch.arange(start, end, step)
    mean = torch.zeros_like(x)
    cov = synthetic_kernel(x, x)
    m = MultivariateNormal(mean, covariance_matrix=cov)
    return x.reshape(-1, 1), m.sample().reshape(-1, 1)


def segregate(data_x, data_y, test_size=0.2):
    return skl_ms.train_test_split(data_x.numpy(),
                                   data_y.numpy(),
                                   test_size=test_size)
