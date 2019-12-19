import math
import torch
from gpytorch.kernels import Kernel, SpectralMixtureKernel
from gpytorch.constraints import Positive


class BatchableSpectralMixtureKernel(SpectralMixtureKernel):
    r"""
    Extends the Spectral Mixture Kernel so it can be used to make
    predictions on a batch, but only has one set of parameters.
    """
    def __init__(self, **kwargs):
        super(BatchableSpectralMixtureKernel, self).__init__(**kwargs)

    def forward(self, x1, x2, last_dim_is_batch=False, **params):
        batch_shape = x1.shape[:-2]
        n, num_dims = x1.shape[-2:]

        if not num_dims == self.ard_num_dims:
            raise RuntimeError(
                "The SpectralMixtureKernel expected the input to have {} dimensionality "
                "(based on the ard_num_dims argument). Got {}.".format(self.ard_num_dims, num_dims)
            )

        # Expand x1 and x2 to account for the number of mixtures
        # Should make x1/x2 (b x k x n x d) for k mixtures
        x1_ = x1.unsqueeze(len(batch_shape))
        x2_ = x2.unsqueeze(len(batch_shape))

        # Compute distances - scaled by appropriate parameters
        x1_exp = x1_ * self.mixture_scales
        x2_exp = x2_ * self.mixture_scales
        x1_cos = x1_ * self.mixture_means
        x2_cos = x2_ * self.mixture_means

        # Create grids
        x1_exp_, x2_exp_ = self._create_input_grid(x1_exp, x2_exp,
                                                   last_dim_is_batch=last_dim_is_batch, **params)
        x1_cos_, x2_cos_ = self._create_input_grid(x1_cos, x2_cos,
                                                   last_dim_is_batch=last_dim_is_batch, **params)

        # Compute the exponential and cosine terms
        exp_term = (x1_exp_ - x2_exp_).pow_(2).mul_(-2 * math.pi ** 2)
        cos_term = (x1_cos_ - x2_cos_).mul_(2 * math.pi)
        res = exp_term.exp_() * cos_term.cos_()

        # Product over dimensions
        if last_dim_is_batch:
            res = res.squeeze(-1)
        else:
            res = res.prod(-1)

        # Sum over mixtures
        mixture_weights = self.mixture_weights

        for _ in range(len(batch_shape)):
            mixture_weights = mixture_weights.unsqueeze(0)
        if last_dim_is_batch:
            mixture_weights = mixture_weights.unsqueeze(-1)
        while mixture_weights.dim() < res.dim():
            mixture_weights = mixture_weights.unsqueeze(-1)

        res = (res * mixture_weights).sum(len(batch_shape))
        return res


class SparseSpectrumKernel(Kernel):
    def __init__(self,
                 num_features=None,
                 ard_num_dims=1,
                 batch_shape=torch.Size([]),
                 signal_variance_constraint=None,
                 fourier_features_constraint=None,
                 **kwargs):
        super(SparseSpectrumKernel,
              self).__init__(ard_num_dims=ard_num_dims,
                             batch_shape=batch_shape,
                             **kwargs)

        if num_features is None:
            raise RuntimeError("num_features is a required argument")
        self.num_features = num_features

        if signal_variance_constraint is None:
            signal_variance_constraint = Positive()
        if fourier_features_constraint is None:
            fourier_features_constraint = Positive()

        self.register_parameter(name='raw_signal_variance',
                                parameter=torch.nn.Parameter(torch.zeros(1)))
        ff_shape = torch.Size([*self.batch_shape,
                               self.num_features,
                               1,
                               self.ard_num_dims])
        self.register_parameter(name='raw_fourier_features',
                                parameter=torch.nn.Parameter(torch.zeros(ff_shape)))

        self.register_constraint('raw_signal_variance',
                                 signal_variance_constraint)
        self.register_constraint('raw_fourier_features',
                                 fourier_features_constraint)

    @property
    def signal_variance(self):
        return self.raw_signal_variance_constraint.transform(self.raw_signal_variance)

    @signal_variance.setter
    def signal_variance(self, value):
        self._set_signal_variance(value)

    def _set_signal_variance(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_signal_variance)
        self.initialize(
            raw_signal_variance=self.raw_signal_variance_constraint.inverse_transform(value))

    @property
    def fourier_features(self):
        return self.raw_fourier_features_constraint.transform(self.raw_fourier_features)

    @fourier_features.setter
    def fourier_features(self, value):
        self._set_fourier_features(value)

    def _set_fourier_features(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_fourier_features)
        self.initialize(
            raw_fourier_features=self.raw_fourier_features_constraint.inverse_transform(value))

    def initialize_from_data(self, train_x, train_y, **kwargs):
        if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
            raise RuntimeError("train_x and train_y should be tensors")
        if train_x.ndimension() == 1:
            train_x = train_x.unsqueeze(-1)
        if train_x.ndimension() == 2:
            train_x = train_x.unsqueeze(0)

        train_x_sort = train_x.sort(1)[0]
        max_dist = train_x_sort[:, -1, :] - train_x_sort[:, 0, :]
        min_dist_sort = (train_x_sort[:, 1:, :] - train_x_sort[:, :-1, :]).squeeze(0)
        min_dist = torch.zeros(1, self.ard_num_dims, dtype=train_x.dtype, device=train_x.device)
        for ind in range(self.ard_num_dims):
            min_dist[:, ind] = min_dist_sort[(torch.nonzero(min_dist_sort[:, ind]))[0], ind]

        # Signal variance should be roughly the stdv of the y values
        self.raw_signal_variance.data.fill_(train_y.std())
        self.raw_signal_variance.data = self.raw_signal_variance_constraint.inverse_transform(
            self.raw_signal_variance.data)
        # Draw features from Unif(0, 0.5 / minimum distance between two points)
        self.raw_fourier_features.data.uniform_().mul_(0.5).div_(min_dist)
        self.raw_fourier_features.data = self.raw_fourier_features_constraint.inverse_transform(self.raw_fourier_features.data)

    def _create_input_grid(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """
        This is a helper method for creating a grid of the kernel's inputs.
        Use this helper rather than maually creating a meshgrid.

        The grid dimensions depend on the kernel's evaluation mode.

        Args:
            :attr:`x1` (Tensor `n x d` or `b x n x d`)
            :attr:`x2` (Tensor `m x d` or `b x m x d`) - for diag mode, these must be the same inputs

        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the gridded `x1` and `x2`.
            The shape depends on the kernel's mode

            * `full_covar`: (`b x n x 1 x d` and `b x 1 x m x d`)
            * `full_covar` with `last_dim_is_batch=True`: (`b x k x n x 1 x 1` and `b x k x 1 x m x 1`)
            * `diag`: (`b x n x d` and `b x n x d`)
            * `diag` with `last_dim_is_batch=True`: (`b x k x n x 1` and `b x k x n x 1`)
        """
        x1_, x2_ = x1, x2
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
            if torch.equal(x1, x2):
                x2_ = x1_
            else:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

        if diag:
            return x1_, x2_
        else:
            return x1_.unsqueeze(-2), x2_.unsqueeze(-3)

    def forward(self, x1, x2, last_dim_is_batch=False, **params):
        batch_shape = x1.shape[:-2]
        n, num_dims = x1.shape[-2:]

        # Expand x1 and x2 to account for the number of features
        # Should make x1/x2 (b x k x n x d) for k features
        x1_ = x1.unsqueeze(len(batch_shape))
        x2_ = x2.unsqueeze(len(batch_shape))

        # Compute distances - scaled by appropriate parameters
        x1_cos = x1_ * self.fourier_features
        x2_cos = x2_ * self.fourier_features

        # Create grids
        x1_cos_, x2_cos_ = self._create_input_grid(x1_cos,
                                                   x2_cos,
                                                   last_dim_is_batch=last_dim_is_batch,
                                                   **params)

        # Compute the cosine terms
        cos_arg = (x1_cos_ - x2_cos_).mul_(2 * math.pi)
        res = cos_arg.cos_()

        # Product over dimensions
        if last_dim_is_batch:
            res = res.squeeze(-1)
        else:
            res = res.prod(-1)

        # Sum over the cosine terms
        res = res.sum(len(batch_shape))
        return self.signal_variance / self.num_features * res
