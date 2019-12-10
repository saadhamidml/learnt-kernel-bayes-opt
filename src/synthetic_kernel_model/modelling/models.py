import gpytorch
from gpytorch.constraints import Interval

from .kernels import SparseSpectrumKernel


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SpectralMixtureGPModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
                                num_mixtures=5,
                                mixture_means_constraint=Interval(-1., 1.))
        import torch
        if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
            raise RuntimeError("train_x and train_y should be tensors")
        if train_x.ndimension() == 1:
            train_x = train_x.unsqueeze(-1)
        if train_x.ndimension() == 2:
            train_x = train_x.unsqueeze(0)

        train_x_sort = train_x.sort(1)[0]
        max_dist = 5 * (train_x_sort[:, -1, :] - train_x_sort[:, 0, :])
        min_dist_sort = (train_x_sort[:, 1:, :] - train_x_sort[:, :-1, :]).squeeze(0)
        min_dist = torch.zeros(1, self.covar_module.ard_num_dims, dtype=train_x.dtype, device=train_x.device)
        for ind in range(self.covar_module.ard_num_dims):
            min_dist[:, ind] = min_dist_sort[(torch.nonzero(min_dist_sort[:, ind]))[0], ind]

        # Inverse of lengthscales should be drawn from truncated Gaussian | N(0, max_dist^2) |
        self.covar_module.raw_mixture_scales.data.normal_().mul_(max_dist).abs_().pow_(-1)
        self.covar_module.raw_mixture_scales.data = self.covar_module.raw_mixture_scales_constraint.inverse_transform(
            self.covar_module.raw_mixture_scales.data
        )
        # Draw means from Unif(0, 0.5 / minimum distance between two points)
        self.covar_module.raw_mixture_means.data.uniform_().mul_(0.5).div_(min_dist)
        self.covar_module.raw_mixture_means.data = self.covar_module.raw_mixture_means_constraint.inverse_transform(
            self.covar_module.raw_mixture_means.data)
        # Mixture weights should be roughly the stdv of the y values divided by the number of mixtures
        self.covar_module.raw_mixture_weights.data.fill_(train_y.std() / self.covar_module.num_mixtures)
        self.covar_module.raw_mixture_weights.data = self.covar_module.raw_mixture_weights_constraint.inverse_transform(
            self.covar_module.raw_mixture_weights.data
        )


class SparseSpectrumGPModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood):
        super(SparseSpectrumGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = SparseSpectrumKernel(num_features=5)
                                                 # signal_variance_constraint=Interval(0.01, 10.))
                                                 # fourier_features_constraint=Interval(1e-6, 10.))
        self.covar_module.initialize_from_data(train_x, train_y)
