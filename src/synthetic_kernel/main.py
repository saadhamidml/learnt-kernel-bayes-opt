from pathlib import Path

from . import data
from .modelling.training import train
from .modelling.visualisation import visualise


def run_experiment(x_start=-10,
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
    data_x, data_y = data.generate(x_start, x_end, x_step)
    (train_x,
     train_y,
     test_x,
     test_y) = data.segregate(data_x,
                              data_y,
                              test_size)
    model, likelihood = train(force_train,
                              kernel,
                              likelihood_type,
                              optimiser_type,
                              learning_rate,
                              training_iterations,
                              train_x,
                              train_y,
                              log_dir)
    visualise(model,
              likelihood,
              train_x,
              train_y,
              vis_start,
              vis_end,
              vis_step)
