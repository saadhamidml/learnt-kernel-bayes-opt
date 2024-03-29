from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from . import utils


def acquisition_function(
        model,
        experiment,
        vis_start=None,
        vis_end=None,
        vis_density=250,
        log_dir=Path('./'),
        point_num=0,
        return_only=False
):
    """Plot the acquisition function given an Ax ModelBridge,
    Ax Experiment, and plotting bounds.
    """

    vis_x = torch.linspace(vis_start, vis_end, vis_density, dtype=torch.double)
    vis_x_transformed = utils.apply_x_transforms(vis_x, model)

    train_x = []
    for arm in experiment.arms_by_name.values():
        train_x.append(arm.parameters['x0'])
    train_x = torch.Tensor(train_x).to(vis_x)
    train_x_transformed = utils.apply_x_transforms(train_x, model).unsqueeze(-1)

    with torch.no_grad():
        # Get acquisition function object and call with vis_x_transformed.
        acqf = model.model.acqf_constructor(
            model.model.model,
            torch.Tensor([1.]).to(torch.double),
            X_observed=train_x_transformed
        )
        acqf = acqf(vis_x_transformed.unsqueeze(-1).unsqueeze(-1))
        acqf = utils.apply_y_untransfoms(acqf.detach(), vis_x, model)

        if return_only:
            return train_x.numpy(), vis_x.numpy(), acqf.numpy()

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

        # Plot training data as black lines
        for x in train_x.numpy():
            ax.axvline(x, color='k')
        # Plot acquisition function as blue line
        ax.plot(vis_x.numpy(), acqf.numpy(), 'b')

        ax.set_xlabel('x')
        ax.set_title('Acquisition Function')

        log_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(log_dir / f'acquisition_function{point_num:02}.png')
        plt.close(fig)


def regret(
        *args,
        n_initial_evaluations=0,
        legend=None,
        function_name=None,
        log_dir=Path('./')):
    n_experiments = int(len(args) / 2)

    # Initialise plot
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 12))
    ax_fun = ax[0]
    ax_loc = ax[1]

    for i in range(n_experiments):
        steps = np.arange(args[i].shape[0]) + n_initial_evaluations
        ax_fun.plot(steps, args[2 * i])
        ax_loc.plot(steps, args[2 * i + 1])

    if legend is not None:
        ax_fun.legend(legend)

    ax_loc.set_xlabel('Number of Data Points')
    ax_loc.set_ylabel('Regret')
    ax_fun.set_ylabel('Regret')
    ax_loc.set_title('Location Regret')
    ax_fun.set_title('Function Regret')
    if function_name is not None:
        fig.suptitle(f'{function_name.capitalize()} Function Optimisation Regrets')

    log_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(log_dir / 'regret.png')
    plt.close(fig)
