import gpytorch
from gpytorch.constraints import Interval
from botorch.models.gpytorch import GPyTorchModel

from .kernels import BatchableSpectralMixtureKernel, SparseSpectrumKernel


class ExactGPModel(gpytorch.models.ExactGP, GPyTorchModel):
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
    def __init__(self, train_x, train_y, likelihood, num_mixtures=5):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = BatchableSpectralMixtureKernel(
                                num_mixtures=num_mixtures)
                                # mixture_means_constraint=Interval(-1., 1.))
        self.covar_module.initialize_from_data(train_x, train_y)


class SparseSpectrumGPModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood):
        super(SparseSpectrumGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = SparseSpectrumKernel(num_features=5)
                                                 # signal_variance_constraint=Interval(0.01, 10.))
                                                 # fourier_features_constraint=Interval(1e-6, 10.))
        self.covar_module.initialize_from_data(train_x, train_y)
