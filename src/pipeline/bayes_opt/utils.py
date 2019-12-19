from typing import List
import numpy as np
import torch
from ax.modelbridge.base import ModelBridge
from ax.core.observation import ObservationFeatures, ObservationData


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
