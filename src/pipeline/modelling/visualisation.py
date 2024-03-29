from pathlib import Path
import torch
import gpytorch
import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift
import matplotlib.pyplot as plt

from ..bayes_opt import utils as bo_utils


# from ..data import euclidean_distance
def euclidean_distance(x, y):
    """Define the kernel for the synthetic data."""
    diff_outer = np.subtract.outer(x, y)
    diff = np.diagonal(diff_outer, axis1=1, axis2=3)
    diff_sq = diff ** 2
    diff_sq_sum = diff_sq.sum(axis=2)
    return np.sqrt(diff_sq_sum)


def posterior(
        model,
        experiment,
        vis_start=None,
        vis_end=None,
        vis_density=250,
        function_name=None,
        kernel_name=None,
        log_dir=Path('./'),
        point_num=0,
        return_only=False
):
    vis_x = torch.linspace(vis_start, vis_end, vis_density, dtype=torch.double)
    vis_x_transformed = bo_utils.apply_x_transforms(vis_x, model)

    train_x = []
    train_y = []
    arm_y = experiment.eval().df
    for arm in experiment.arms_by_name.values():
        train_x.append(arm.parameters['x0'])
        train_y.append(arm_y[arm_y['arm_name'] == arm.name]['mean'].values[0])
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y).squeeze()

    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions
        observed_pred = model.model.model.posterior(vis_x_transformed)
        posterior_mean = observed_pred.mean
        posterior_mean = bo_utils.apply_y_untransfoms(posterior_mean, vis_x, model)
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.mvn.confidence_region()
        lower = bo_utils.apply_y_untransfoms(lower, vis_x, model)
        upper = bo_utils.apply_y_untransfoms(upper, vis_x, model)

        if return_only:
            return (
                train_x.numpy(),
                train_y.numpy(),
                vis_x.numpy(),
                posterior_mean.numpy(),
                lower.numpy(),
                upper.numpy()
            )

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'kx')
        # Plot predictive means as blue line
        ax.plot(vis_x.numpy(), posterior_mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(vis_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if function_name is not None and kernel_name is not None:
            ax.set_title(f'{function_name.capitalize()} Posterior with '
                         + f'{kernel_name} Kernel')

        log_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(log_dir / f'posterior{point_num:02}.png')
        plt.close(fig)


def kernel(
        model,
        vis_start=None,
        vis_end=None,
        vis_density=250,
        true_kernel=None,
        function_name=None,
        kernel_name=None,
        log_dir=Path('./'),
        point_num=0,
        return_only=0
):
    max_dist = vis_end - vis_start
    vis_step = 2 * max_dist / vis_density
    vis_x = torch.linspace(-max_dist, max_dist, vis_density)
    vis_x_transformed = bo_utils.apply_x_transforms(vis_x, model)
    num_x = vis_x.size(0)
    # vis_freq = np.linspace(0, 1 / (2 * vis_step), num_x // 2)
    vis_freq = fftshift(fftfreq(num_x, vis_step))

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        kern = model.model.model.covar_module(
            bo_utils.apply_x_transforms(torch.zeros(1), model).unsqueeze_(0),
            vis_x_transformed
        ).evaluate().squeeze().numpy()
        # kern /= np.max(kern)
        kern_ft = 1 / num_x * np.abs(fftshift(fft(kern)))

        vis_x = vis_x.numpy()

        if return_only:
            return vis_x, kern, vis_freq, kern_ft

        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        ax_orig = axes[0]
        ax_fft = axes[1]

        ax_orig.plot(vis_x, kern)
        ax_fft.plot(vis_freq, kern_ft)

        if true_kernel is not None:
            dist = euclidean_distance(np.zeros(1).reshape(-1, 1),
                                      vis_x.reshape(-1, 1))
            t_kern = true_kernel(dist).squeeze().numpy()
            t_kern_ft = 1 / num_x * np.abs(fftshift(fft(t_kern)))
            ax_orig.plot(vis_x, t_kern)
            ax_fft.plot(vis_freq, t_kern_ft)

        ax_orig.set_xlabel('x')
        ax_fft.set_xlabel('f')
        if function_name is not None and kernel_name is not None:
            ax_orig.set_title(f'{kernel_name} Kernel in Original Domain')
            ax_fft.set_title(f'{kernel_name} Kernel in Frequency Domain')

        fig.savefig(log_dir / f'kernel{point_num:02}.png')
        plt.close(fig)
