from pathlib import Path
import torch
import gpytorch
import matplotlib.pyplot as plt


def visualise(model,
              likelihood,
              train_X,
              train_y,
              vis_start=-15,
              vis_end=15.1,
              vis_step=0.1,
              log_dir=Path('./')):
    vis_X = torch.arange(vis_start, vis_end, vis_step)

    model.eval()
    likelihood.eval()

    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions
        observed_pred = likelihood(model(vis_X))

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_X.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(vis_X.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(vis_X.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

        fig.savefig(log_dir / 'posterior.png')
        plt.close(fig)
