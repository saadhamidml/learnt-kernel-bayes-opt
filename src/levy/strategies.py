from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.generation_strategy import (GenerationStrategy,
                                                GenerationStep)
from ax.modelbridge.registry import Models
from ax.modelbridge.factory import get_botorch

from pipeline.modelling import models
import torch
import gpytorch
from botorch.fit import fit_gpytorch_model


def define_strategy(kernel,
                    likelihood_type,
                    optimiser_type,
                    learning_rate,
                    num_init=10):
    if likelihood_type == 'gaussian':
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if kernel == 'rbf':
        model_class = models.ExactGPModel
    elif kernel == 'spectral_mixture':
        model_class = models.SpectralMixtureGPModel
    elif kernel == 'sparse_spectrum':
        model_class = models.SparseSpectrumGPModel
    if optimiser_type == 'adam':
        optimiser_class = torch.optim.Adam
    mll_class = gpytorch.mlls.ExactMarginalLogLikelihood

    def _get_and_fit_smgp(Xs, Ys, **kwargs):
        model = model_class(Xs[0], Ys[0].squeeze(-1), likelihood)
        optimiser = optimiser_class([{'params': model.parameters()}, ],
                                    lr=learning_rate)
        mll = mll_class(likelihood, model)
        fit_gpytorch_model(mll)
        return model

    def get_botorch_model(experiment, data, search_space):
        return get_botorch(
            experiment=experiment,
            data=data,
            search_space=search_space,
            model_constructor=_get_and_fit_smgp,
        )

    return GenerationStrategy(
        name='SMGP+NEI',
        steps=[
            GenerationStep(model=Models.SOBOL,
                           num_arms=num_init,
                           model_kwargs={'scramble': False}),
            GenerationStep(model=get_botorch_model, num_arms=-1)
        ]
    )
