from typing import List
from pathlib import Path
import numpy as np
import torch
import gpytorch
from botorch.fit import fit_gpytorch_model
from ax.modelbridge.base import ModelBridge
from ax.core.observation import ObservationFeatures, ObservationData

from . import models
from .training import train


def get_model_constructor(
        kernel,
        likelihood_type,
        optimiser_type,
        learning_rate,
        training_iterations,
        force_train=0
):
    if likelihood_type == 'gaussian':
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if kernel == 'rbf':
        model_class = models.ExactGPModel
    elif kernel == 'spectral_mixture':
        model_class = models.SpectralMixtureGPModel
    elif kernel == 'sparse_spectrum':
        model_class = models.SparseSpectrumGPModel
    if optimiser_type == 'sgd':
        optimiser_class = torch.optim.SGD
    mll_class = gpytorch.mlls.ExactMarginalLogLikelihood

    def model_constructor(Xs, Ys, **kwargs):
        model = model_class(Xs[0].squeeze(), Ys[0].squeeze(-1), likelihood)
        optimiser = optimiser_class([{'params': model.parameters()}, ],
                                     lr=learning_rate)
        mll = mll_class(likelihood, model)
        # fit_gpytorch_model(mll)
        train(
            model,
            optimiser,
            mll,
            training_iterations,
            force_train,
            Xs[0].squeeze(),
            Ys[0].squeeze(),
            **kwargs
        )
        return model

    return model_constructor


def tensor_to_observation_features(
        x_tensor: torch.Tensor
) -> List[ObservationFeatures]:
    """Convert torch Tensors to ax OberservationFeatures."""
    x_features = []
    for x in x_tensor:
        x_feature = ObservationFeatures(parameters={})
        x_feature.parameters['x'] = x.item()
        x_features.append(x_feature)
    return x_features


def observation_features_to_tensor(
        x_features: List[ObservationFeatures]
) -> torch.Tensor:
    """Convert ax OberservationFeatures to torch Tensors."""
    x_tensor = []
    for x_feature in x_features:
        x = x_feature.parameters['x']
        x_tensor.append(x)
    return torch.Tensor(x_tensor)


def tensor_to_observation_data(
        y_tensor: torch.Tensor,
        noise_std=0.0,
) -> List[ObservationData]:
    """Convert torch Tensors to ax OberservationData."""
    y_tensor = y_tensor.squeeze()
    y_data = []
    for y in y_tensor:
        y_data_point = ObservationData(
            metric_names=['function'],
            means=y.unsqueeze(dim=0).numpy(),
            covariance=np.array([[noise_std ** 2]])
        )
        y_data.append(y_data_point)
    return y_data


def observation_data_to_tensor(
        y_data: List[ObservationData]
) -> torch.Tensor:
    """Convert ax OberservationData to torch Tensors."""
    y_tensor = []
    for y_data_point in y_data:
        y = y_data_point.means[0]
        y_tensor.append(y)
    return torch.Tensor(y_tensor)


def apply_x_transforms(
        x_untransformed: torch.Tensor,
        model: ModelBridge
) -> torch.Tensor:
    x_features = tensor_to_observation_features(x_untransformed)
    for t in model.transforms.values():
        x_features = t.transform_observation_features(x_features)
    x_transformed = observation_features_to_tensor(x_features)
    return x_transformed.to(x_untransformed)


def apply_x_untranforms(
        x_transformed: torch.Tensor,
        model: ModelBridge
) -> torch.Tensor:
    x_features = tensor_to_observation_features(x_transformed)
    for t in reversed(model.transforms.values()):
        x_features = t.untransform_observation_features(x_features)
    x_untransformed = observation_features_to_tensor(x_features)
    return x_untransformed.to(x_transformed)


def apply_y_transfoms(
        y_untransformed: torch.Tensor,
        x_untransformed: torch.Tensor,
        model: ModelBridge
) -> torch.Tensor:
    y_data = tensor_to_observation_data(y_untransformed)
    x_features = tensor_to_observation_features(x_untransformed)
    for t in model.transforms.values():
        y_data = t.transform_observation_data(y_data, x_features)
    y_transformed = observation_data_to_tensor(y_data)
    return y_transformed.to(y_untransformed)


def apply_y_untransfoms(
        y_transformed: torch.Tensor,
        x_untransformed: torch.Tensor,
        model: ModelBridge
) -> torch.Tensor:
    y_data = tensor_to_observation_data(y_transformed)
    x_features = tensor_to_observation_features(x_untransformed)
    for t in reversed(model.transforms.values()):
        y_data = t.untransform_observation_data(y_data, x_features)
    y_untransformed = observation_data_to_tensor(y_data)
    return y_untransformed.to(y_transformed)
