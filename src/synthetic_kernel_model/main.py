from pathlib import Path

from . import data
from .modelling.training import train
from .modelling import visualisation as mod_vis


def run_experiment(true_kernel='matern',
                   x_start=-10,
                   x_end=10.5,
                   x_step=0.5,
                   test_size=0.2,
                   force_train=0,
                   kernel='rbf',
                   likelihood_type='gaussian',
                   optimiser_type='adam',
                   learning_rate=1e-1,
                   training_iterations=50,
                   vis_start=-15,
                   vis_end=15.1,
                   vis_step=0.1,
                   log_dir=Path('./')):
    synthetic_kernel = data.get_true_kernel(true_kernel)
    data_x, data_y = data.generate(synthetic_kernel, x_start, x_end, x_step)
    (train_x,
     test_x,
     train_y,
     test_y
     ) = data.segregate(
        data_x,
        data_y,
        test_size
    )
    model, likelihood = train(
        force_train,
                              kernel,
                              likelihood_type,
                              optimiser_type,
                              learning_rate,
                              training_iterations,
                              train_x,
                              train_y,
                              log_dir=log_dir)
    mod_vis.posterior(model,
                      likelihood,
                      train_x,
                      train_y,
                      vis_start,
                      vis_end,
                      vis_step,
                      log_dir=log_dir)
    mod_vis.kernel(model,
                   vis_start,
                   vis_end,
                   vis_step,
                   true_kernel=synthetic_kernel,
                   log_dir=log_dir)
