import torch
import gpytorch
from botorch.fit import fit_gpytorch_model

from . import models
from .training import train


def get_model_constructor(
        kernel,
        likelihood_type,
        optimiser_type,
        learning_rate,
        training_iterations,
        force_train=0,
        **model_kwargs
):
    if likelihood_type == 'gaussian':
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if kernel == 'rbf':
        model_class = models.RBFGPModel
    elif kernel == 'matern':
        model_class = models.MaternGPModel
    elif kernel == 'spectral_mixture':
        model_class = models.SpectralMixtureGPModel
    elif kernel == 'sparse_spectrum':
        model_class = models.SparseSpectrumGPModel
    if optimiser_type == 'sgd':
        optimiser_class = torch.optim.SGD
    mll_class = gpytorch.mlls.ExactMarginalLogLikelihood

    def model_constructor(Xs, Ys, **kwargs):
        model = model_class(
            Xs[0].squeeze(),
            Ys[0].squeeze(-1),
            likelihood,
            **model_kwargs
        )
        optimiser = optimiser_class([{'params': model.parameters()}, ],
                                     lr=learning_rate)
        mll = mll_class(likelihood, model)
        fit_gpytorch_model(mll)
        # train(
        #     model,
        #     optimiser,
        #     mll,
        #     training_iterations,
        #     force_train,
        #     Xs[0].squeeze(),
        #     Ys[0].squeeze(),
        #     **kwargs
        # )
        return model

    return model_constructor
