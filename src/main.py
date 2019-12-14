import importlib
import numpy as np

from pipeline.utils import cli, random_seeds, log
from pipeline.experiment import multi_config
from pipeline.bayes_opt.visualisation import mult_regrets as plot_regrets


def get_problem(parser):
    """Imports main script for the relevant problem."""
    # Get problem name
    flags, _ = parser.parse_known_args()
    problem = flags.problem
    # Get problem main module
    problem_main_name = problem + '.main'
    problem_main = importlib.import_module(problem_main_name)
    # Get problem specific flags
    problem_cli_name = problem + '.utils.cli'
    problem_cli = importlib.import_module(problem_cli_name)
    flags, unparsed_args = problem_cli.get_flags(parser)
    # Define logging directory and save configuration
    problem_log_dir, _ = log.default_log_dir(flags)
    return (
        problem_main.run_experiment_wrapper,
        flags,
        unparsed_args,
        problem_log_dir
    )


if __name__ == '__main__':
    # Get problem specific objects
    parser = cli.get_parser()
    (run_experiment_wrapper,
     flags,
     unparsed_args,
     problem_log_dir) = get_problem(parser)
    # Define experiment configs (if comparing mutliple)
    observer_log_dir = problem_log_dir / 'rbf_sm_1000'
    options = {'kernel': ['rbf', 'spectral_mixture']}
    seeds = None
    # Run experiments
    observer = multi_config(
        run_experiment_wrapper,
        flags,
        unparsed_args=unparsed_args,
        options=options,
        seeds=seeds,
        mode='list',
    )
    # Inspect results
    plot_regrets(
        np.array(observer.record[0]['regret']['location']),
        np.array(observer.record[0]['regret']['function']),
        np.array(observer.record[1]['regret']['location']),
        np.array(observer.record[1]['regret']['function']),
        n_initial_evaluations=flags.n_initial_evaluations,
        log_dir=observer_log_dir
    )
