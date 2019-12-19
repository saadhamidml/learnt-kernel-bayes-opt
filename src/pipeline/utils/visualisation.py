from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from botorch.test_functions import SyntheticTestFunction

from ..modelling import visualisation as mod_vis
from ..bayes_opt import visualisation as bo_vis


def function(
        func,
        vis_start=None,
        vis_end=None,
        vis_density=250,
        log_dir=Path('./'),
        return_only=False
):
    vis_x = torch.linspace(vis_start, vis_end, vis_density, dtype=torch.double)
    if isinstance(func, SyntheticTestFunction):
        func_values = -func.evaluate_true(vis_x.unsqueeze(-1))
    else:
        func_values = func(vis_x)

    if return_only:
        return vis_x.squeeze().numpy(), func_values.squeeze().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.plot(vis_x.squeeze().numpy(), func_values.squeeze().numpy())

    log_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(log_dir / 'function.png')
    plt.close(fig)


def board(
        func,
        model,
        experiment,
        location_regret,
        function_regret,
        n_initial_evaluations=0,
        vis_start=None,
        vis_end=None,
        vis_density=250,
        log_dir=Path('./'),
        point_num=0
):
    vis_x, fun_val = function(
        func,
        vis_start=vis_start,
        vis_end=vis_end,
        vis_density=vis_density,
        log_dir=log_dir,
        return_only=True
    )
    (
        train_x,
        train_y,
        _,
        posterior_mean,
        posterior_lower,
        posterior_upper
    ) = mod_vis.posterior(
        model,
        experiment,
        vis_start=vis_start,
        vis_end=vis_end,
        vis_density=vis_density,
        log_dir=log_dir,
        point_num=point_num,
        return_only=True
    )
    kern_x, kern, kern_ft_x, kern_ft = mod_vis.kernel(
        model,
        vis_start=vis_start,
        vis_end=vis_end,
        vis_density=vis_density,
        log_dir=log_dir,
        point_num=point_num,
        return_only=True
    )
    _, _, acqf = bo_vis.acquisition_function(
        model,
        experiment,
        vis_start=vis_start,
        vis_end=vis_end,
        vis_density=vis_density,
        log_dir=log_dir,
        point_num=point_num,
        return_only=True
    )
    steps = np.arange(len(location_regret)) + n_initial_evaluations

    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    ax_post = axes[0, 0]
    ax_acqf = axes[1, 0]
    ax_kern = axes[0, 1]
    ax_kern_ft = axes[1, 1]
    ax_loc_reg = axes[0, 2]
    ax_fun_reg = axes[1, 2]

    ax_post.plot(vis_x, fun_val, 'k')
    ax_post.plot(train_x, train_y, 'rx')
    ax_post.plot(vis_x, posterior_mean)
    ax_post.fill_between(vis_x, posterior_lower, posterior_upper, alpha=0.5)
    for x in train_x:
        ax_acqf.axvline(x, color='r')
    ax_acqf.plot(vis_x, acqf)
    ax_kern.plot(kern_x, kern)
    ax_kern_ft.plot(kern_ft_x, kern_ft)
    ax_loc_reg.plot(steps, np.array(location_regret))
    ax_fun_reg.plot(steps, np.array(function_regret))

    log_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(log_dir / f'board{point_num:02}.png')
    plt.close(fig)
