import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import sklearn.model_selection as skl_ms


def euclidean_distance(x, y):
    """Define the kernel for the synthetic data."""
    diff_outer = np.subtract.outer(x, y)
    diff = np.diagonal(diff_outer, axis1=1, axis2=3)
    diff_sq = diff ** 2
    diff_sq_sum = diff_sq.sum(axis=2)
    return np.sqrt(diff_sq_sum)


def get_true_kernel(true_kernel):
    if true_kernel == 'matern':
        def synthetic_kernel(dist):
            return torch.Tensor(np.exp(-dist))
    elif true_kernel == 'sinc':
        def synthetic_kernel(dist):
            return torch.Tensor(np.sinc(dist / 3) * np.exp(-dist ** 2 / 128))
    else:
        def synthetic_kernel(dist):
            return torch.Tensor(np.exp(-dist ** 2 / 8))

    return synthetic_kernel


def generate(synthetic_kernel, start, end, step):
    """Generate the data."""
    x = torch.arange(start, end, step)
    mean = torch.zeros_like(x).squeeze()
    dist = euclidean_distance(x.reshape(-1, 1).numpy(),
                              x.reshape(-1, 1).numpy())
    cov = synthetic_kernel(dist) + 0.01 * torch.eye(mean.size(0))
    m = MultivariateNormal(mean, covariance_matrix=cov)
    return x, m.sample()


def segregate(data_x, data_y, test_size=0.2):
    (train_x,
     test_x,
     train_y,
     test_y) = skl_ms.train_test_split(data_x.numpy(),
                                       data_y.numpy(),
                                       test_size=test_size)
    return map(torch.Tensor, (train_x, test_x, train_y, test_y))
