from pathlib import Path
import torch
import gpytorch
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt

from ..data import euclidean_distance


def posterior(model,
              likelihood,
              train_x,
              train_y,
              vis_start=-15,
              vis_end=15.1,
              vis_step=0.1,
              log_dir=Path('./')):
    vis_x = torch.arange(vis_start, vis_end, vis_step)

    model.eval()
    likelihood.eval()

    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions
        observed_pred = likelihood(model(vis_x))

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'kx')
        # Plot predictive means as blue line
        ax.plot(vis_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(vis_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

        fig.savefig(log_dir / 'posterior.png')
        plt.close(fig)

def kernel(model,
           vis_start,
           vis_end,
           vis_step,
           true_kernel=None,
           log_dir=Path('./')):
    max_dist = vis_end - vis_start
    vis_x = torch.arange(-max_dist, max_dist, vis_step)
    num_x = vis_x.size(0)
    vis_freq = np.linspace(0, 1 / (2 * vis_step), num_x // 2)

    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        kern = model.covar_module(torch.zeros(1),
                                  vis_x).evaluate().squeeze().numpy()
        kern /= np.max(kern)
        def positive_fft(kern):
            kern_ft = 2 / num_x * fft(kern)
            if (num_x % 2):
                kern_ft = np.abs(kern_ft[0: num_x // 2 + 1])
            else:
                kern_ft = np.abs(kern_ft[0: num_x // 2])
            return kern_ft
        kern_ft = positive_fft(kern)


        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        ax_orig = axes[0]
        ax_fft = axes[1]

        vis_x = vis_x.numpy()
        ax_orig.plot(vis_x, kern)
        ax_fft.plot(vis_freq, kern_ft)

        if true_kernel is not None:
            dist = euclidean_distance(np.zeros(1).reshape(-1, 1),
                                      vis_x.reshape(-1, 1))
            t_kern = true_kernel(dist).squeeze().numpy()
            t_kern_ft = positive_fft(t_kern)
            ax_orig.plot(vis_x, t_kern)
            ax_fft.plot(vis_freq, t_kern_ft)

        fig.savefig(log_dir / 'kernel.png')
        plt.close(fig)
