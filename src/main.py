import importlib
import os
import shutil
import matplotlib.pyplot as plt

from pipeline.utils import cli, log, experiment
from pipeline.bayes_opt.visualisation import regret as plot_regrets
from pipeline.utils.visualisation import (
    group_label_transpose,
    grouped_bar_plot
)


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
    if os.path.exists(observer_log_dir):
        shutil.rmtree(observer_log_dir)
    observer_log_dir.mkdir(parents=True)
    options = {'kernel': ['rbf', 'matern', 'spectral_mixture']}
    # options = None
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
    # Compare regrets for each seed
    for i in range(len(observer.current['seed'])):
        seed_comparison_log_dir = observer_log_dir / str(
            observer.current['seed'][i]
        )
        seed_comparison_log_dir.mkdir(parents=True)
        regrets = []
        for j in range(len(observer.record)):
            loc_reg, fun_reg = log.save_cumulative_regrets(
                observer.record[j]['results'][i],
                record_name=options['kernel'][j],
                return_regrets=True,
                log_dir=seed_comparison_log_dir
            )
            regrets.append(loc_reg)
            regrets.append(fun_reg)
        plot_regrets(
            *tuple(regrets),
            n_initial_evaluations=flags.n_initial_evaluations,
            legend=options['kernel'],
            log_dir=seed_comparison_log_dir
        )
    # Compare average cumulative regrets
    cum_loc_reg_avg, cum_loc_reg_sd = observer.compare_configs(
        'cumulative_location_regret',
        print_wrt='kernel',
        log_dir=observer_log_dir
    )
    cum_fun_reg_avg, cum_fun_reg_sd = observer.compare_configs(
        'cumulative_function_regret',
        print_wrt='kernel',
        log_dir=observer_log_dir
    )
    fig, ax = grouped_bar_plot(*group_label_transpose(
        cum_loc_reg_avg,
        cum_loc_reg_sd,
        cum_fun_reg_avg,
        cum_fun_reg_sd
    ))
    ax.set_xticklabels(['Location', 'Function'])
    ax.legend(options['kernel'])
    ax.set_title('Cumulative Regrets')
    fig.savefig(observer_log_dir / 'cumulative_regrets.png')
    plt.close(fig)
