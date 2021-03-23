from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import chain

from botorch.test_functions import SyntheticTestFunction

from ..modelling import visualisation as mod_vis
from ..bayes_opt import visualisation as bo_vis


def function(
        func,
        vis_start=None,
        vis_end=None,
        vis_density=250,
        function_name=None,
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

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if function_name is not None:
        ax.set_title(f'{function_name.capitalize()} Function')

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
        function_name=None,
        kernel_name=None,
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

    ax_post.set_title('Posterior')
    ax_acqf.set_title('Acquisition Function')
    ax_kern.set_title('Kernel')
    ax_kern_ft.set_title('Kernel\'s Spectral Density')
    ax_loc_reg.set_title('Location Regret')
    ax_fun_reg.set_title('Function Regret')
    if function_name is not None and kernel_name is not None:
        fig.suptitle(f'Bayesian Optimisation of {function_name.capitalize()} '
                     + f'Function with {kernel_name} Kernel: {point_num} Data '
                     + f'Points')

    log_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(log_dir / f'board{point_num:02}.png')
    plt.close(fig)


def group_label_transpose(*args):
    """Transposes groups and labels. Each pair of args is a list of
    values and errors for a label.

    e.g. args = accuracy, accuracy_error, f1_score, f1_score_error for
    model1 and model2.
    output = tuple(model1's [value, error, value, error, ...],
                   model2's [value, error, value, error, ...])
    """
    vals = args[0]
    vars = args[1]
    for i in range(1, int(len(args) / 2)):
        vals = np.vstack((vals, args[2 * i]))
        vars = np.vstack((vars, args[2 * i + 1]))
    vals = vals.T
    vars = vars.T
    vals = np.split(vals, len(vals), axis=0)
    vars = np.split(vars, len(vars), axis=0)
    vals = list(map(np.squeeze, vals))
    vars = list(map(np.squeeze, vars))
    return tuple(chain.from_iterable(zip(vals, vars)))


def grouped_bar_plot(*args, width=0.35):
    """Makes grouped bar plot. Each pair of args is a value and error
    for each group.

    e.g. args = accuracy, accuracy_error, f1_score, f1_score_error, ...

    Groups are determined by index of an arg.
    e.g. group1 is accuracy[0], f1_score[0] and associated errors.
    """

    n_bars_per_group = len(args) / 2
    ind = np.arange(len(args[0])) * (n_bars_per_group + 1) * width

    fig, ax = plt.subplots(figsize=(20, 12))
    locations = ind - width * (n_bars_per_group / 2 - 0.5)
    for i in range(int(n_bars_per_group)):
        ax.bar(locations, args[2 * i], width)
        ax.errorbar(locations,
                    args[2 * i],
                    yerr=args[2 * i + 1],
                    ecolor='k',
                    fmt='none',
                    label='_nolegend_')
        locations += width

    ax.set_xticks(ind)
    ax.grid()

    return fig, ax
