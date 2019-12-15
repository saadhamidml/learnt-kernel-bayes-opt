import importlib
import numpy as np

from pipeline.utils import cli, random_seeds, log, experiment
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
    # Define logging directory
    problem_log_module_name = problem + '.utils.log'
    if importlib.util.find_spec(problem_log_module_name) is not None:
        problem_log_module = importlib.import_module(problem_log_module_name)
        get_log_dir = problem_log_module.get_log_dir
        problem_log_dir, _ = get_log_dir(flags)
    else:
        problem_log_dir, _ = log.default_log_dir(flags)
    return (
        problem_main.run_experiment_wrapper,
        flags,
        unparsed_args,
        problem_log_dir,
        get_log_dir
    )


if __name__ == '__main__':
    # Get problem specific objects
    parser = cli.get_parser()
    (run_experiment_wrapper,
     flags,
     unparsed_args,
     problem_log_dir,
     get_log_dir) = get_problem(parser)
    # Define experiment configs (if comparing mutliple)
    observer_log_dir = problem_log_dir / 'rbf_mat_sm'
    options = {'kernel': ['rbf', 'matern', 'spectral_mixture']}
    seeds = None
    # Run experiments
    observer = experiment.multi_config(
        run_experiment_wrapper,
        flags,
        unparsed_args=unparsed_args,
        options=options,
        seeds=seeds,
        mode='list',
        get_log_dir=get_log_dir
    )
    # Inspect results
    regrets = []
    for i in range(len(observer.record)):
        regrets.append(observer.record[i]['regret']['location'])
        regrets.append(observer.record[i]['regret']['function'])
    regrets = map(np.array, regrets)
    plot_regrets(
        *regrets,
        n_initial_evaluations=flags.n_initial_evaluations,
        log_dir=observer_log_dir
    )
