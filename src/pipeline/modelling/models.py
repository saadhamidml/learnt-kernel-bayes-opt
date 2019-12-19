import gpytorch
from gpytorch.constraints import Interval
from botorch.models.gpytorch import GPyTorchModel

from .kernels import BatchableSpectralMixtureKernel, SparseSpectrumKernel


class ExactGPModel(gpytorch.models.ExactGP, GPyTorchModel):
    def __init__(self, train_x, train_y, likelihood, **kwargs):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RBFGPModel(ExactGPModel):
    def __init__(
            self,
            train_x,
            train_y,
            likelihood,
            outputscale=None,
            lengthscale=None,
            **kwargs
    ):
        super(RBFGPModel, self).__init__(train_x, train_y, likelihood)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        if outputscale is not None:
            self.covar_module.outputscale = outputscale
        if lengthscale is not None:
            self.covar_module.base_kernel.lengthscale = lengthscale


class MaternGPModel(ExactGPModel):
    def __init__(
            self,
            train_x,
            train_y,
            likelihood,
            nu=1.5,
            outputscale=None,
            lengthscale=None,
            **kwargs
    ):
        super(MaternGPModel, self).__init__(train_x, train_y, likelihood)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=nu)
        )
        if outputscale is not None:
            self.covar_module.outputscale = outputscale
        if lengthscale is not None:
            self.covar_module.base_kernel.lengthscale = lengthscale


class SpectralMixtureGPModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood, n_mixtures=5, **kwargs):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = BatchableSpectralMixtureKernel(
                                num_mixtures=n_mixtures)
                                # mixture_means_constraint=Interval(-1., 1.))
        self.covar_module.initialize_from_data(train_x, train_y)


class SparseSpectrumGPModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood, n_mixtures=5, **kwargs):
        super(SparseSpectrumGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = SparseSpectrumKernel(num_features=n_mixtures)
                                                 # signal_variance_constraint=Interval(0.01, 10.))
                                                 # fourier_features_constraint=Interval(1e-6, 10.))
        self.covar_module.initialize_from_data(train_x, train_y)
