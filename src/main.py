import importlib
from pathlib import Path

from spine.utils import cli, log


def get_script(dataset):
    """Imports main script for the relevant dataset."""
    ds_main_name = dataset + '.main'
    ds_main = importlib.import_module(ds_main_name)
    return ds_main.run_experiment

if __name__ == '__main__':
    flags, unparsed_args = cli.get_flags()
    run_experiment = get_script(flags.dataset)
    log_dir = log.save_config(flags, unparsed_args)
    run_experiment(flags.x_start,
                   flags.x_end,
                   flags.x_step,
                   flags.test_size,
                   flags.force_train,
                   flags.kernel,
                   flags.likelihood_type,
                   flags.optimiser_type,
                   flags.learning_rate,
                   flags.training_iterations,
                   flags.vis_start,
                   flags.vis_end,
                   flags.vis_step,
                   log_dir=log_dir)
