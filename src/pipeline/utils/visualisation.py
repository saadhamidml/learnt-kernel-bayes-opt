from pathlib import Path
import torch
import matplotlib.pyplot as plt

from botorch.test_functions import SyntheticTestFunction


def function(
        func,
        vis_start=None,
        vis_end=None,
        vis_density=None,
        log_dir=Path('./')
):
    vis_x = torch.linspace(vis_start, vis_end, vis_density, dtype=torch.double)
    if isinstance(func, SyntheticTestFunction):
        func_values = -func.evaluate_true(vis_x.unsqueeze(-1))
    else:
        func_values = func(vis_x)

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.plot(vis_x.squeeze().numpy(), func_values.squeeze().numpy())

    log_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(log_dir / 'function.png')
    plt.close(fig)
