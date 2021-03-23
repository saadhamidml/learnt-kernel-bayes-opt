from typing import List
import numpy as np
import torch
from ax import Experiment
from ax.modelbridge.base import ModelBridge
from ax.core.observation import (
    ObservationFeatures,
    ObservationData,
    observations_from_data,
    separate_observations
)

from ..utils.experiment import ExObserver


def tensor_to_observation_features(
        x_tensor: torch.Tensor
) -> List[ObservationFeatures]:
    """Convert torch Tensors to ax ObservationFeatures."""
    x_features = []
    for x in x_tensor:
        x_feature = ObservationFeatures(parameters={})
        x_feature.parameters['x0'] = x.item()
        x_features.append(x_feature)
    return x_features


def observation_features_to_tensor(
        x_features: List[ObservationFeatures]
) -> torch.Tensor:
    """Convert ax ObservationFeatures to torch Tensors."""
    x_tensor = []
    for x_feature in x_features:
        x = list(x_feature.parameters.values())
        x_tensor.append(x)
    return torch.tensor(x_tensor).squeeze()


def tensor_to_observation_data(
        y_tensor: torch.Tensor,
        noise_std=0.0,
) -> List[ObservationData]:
    """Convert torch Tensors to ax ObservationData."""
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
    """Convert ax ObservationData to torch Tensors."""
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


def get_current_regrets(experiment: Experiment, observer: ExObserver):
    observations = observations_from_data(experiment, experiment.eval())
    observation_features, observation_data = separate_observations(observations)
    obs_x = observation_features_to_tensor(observation_features)
    if obs_x.dim() == 1:
        obs_x.unsqueeze_(-1)
    obs_f = observation_data_to_tensor(observation_data)
    opt_x = observer.current['optimum']['x']
    opt_f = observer.current['optimum']['function']
    loc_regret = torch.norm(obs_x - opt_x, dim=1).min()
    fun_regret = torch.abs(obs_f - opt_f).min()
    return loc_regret, fun_regret
