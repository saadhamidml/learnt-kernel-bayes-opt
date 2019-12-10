import importlib
from pathlib import Path

from pipeline.utils import cli, random_seeds, log


def get_script(problem):
    """Imports main script for the relevant problem."""
    problem_main_name = problem + '.main'
    problem_main = importlib.import_module(problem_main_name)
    return problem_main.run_experiment_wrapper


if __name__ == '__main__':
    parser = cli.get_parser()
    flags, _ = parser.parse_known_args()
    run_experiment_wrapper = get_script(flags.problem)
    seeds = [0]
    if seeds is None:
        seeds = random_seeds.get_seeds(flags.seed, flags.repeat_exp)
    else:
        flags.repeat_exp = len(seeds)
        random_seeds.set_seed(flags, seeds, 0)
    # run_experiment(flags.true_kernel,
    #                flags.x_start,
    #                flags.x_end,
    #                flags.x_step,
    #                flags.test_size,
    #                flags.force_train,
    #                flags.kernel,
    #                flags.likelihood_type,
    #                flags.optimiser_type,
    #                flags.learning_rate,
    #                flags.training_iterations,
    #                flags.vis_start,
    #                flags.vis_end,
    #                flags.vis_step,
    #                log_dir=log_dir)
    run_experiment_wrapper(parser)
