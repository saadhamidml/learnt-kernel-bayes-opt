from pathlib import Path
from ax import SimpleExperiment
from ax.benchmark.benchmark_suite import BOBenchmarkingSuite
from IPython.core.display import HTML

from .utils import cli
from .problem import define_problem
from pipeline.utils import log
from pipeline.modelling.utils import get_model_constructor
from pipeline.bayes_opt import sobol_initialise, bayes_opt_loop


def run_experiment_wrapper(parser):
    flags, unparsed_args = cli.get_flags(parser)
    log_dir = log.default_log_dir(flags)
    log.save_config(flags, unparsed_args, log_dir=log_dir)
    run_experiment(
        function=flags.function,
        x_start=flags.x_start,
        x_end=flags.x_end,
        noise_std=flags.noise_std,
        n_initial_evaluations=flags.n_initial_evaluations,
        evaluation_budget=flags.evaluation_budget,
        seed=flags.seed,
        kernel=flags.kernel,
        likelihood_type=flags.likelihood_type,
        optimiser_type=flags.optimiser_type,
        learning_rate=flags.learning_rate,
        training_iterations=flags.training_iterations,
        acquisition_function=flags.acquisition_function,
        visualise=flags.visualise,
        vis_start=flags.vis_start,
        vis_end=flags.vis_end,
        vis_step=flags.vis_step,
        log_dir=log_dir
    )


def run_experiment(
        function='exp_sin_squared',
        x_start=-5,
        x_end=5,
        noise_std=1e-3,
        n_initial_evaluations=20,
        evaluation_budget=25,
        seed=None,
        kernel='rbf',
        likelihood_type='gaussian',
        optimiser_type='adam',
        learning_rate=1e-1,
        training_iterations=50,
        acquisition_function='nei',
        visualise=False,
        vis_start=None,
        vis_end=None,
        vis_step=None,
        log_dir=Path('./')
):
    problem, search_space = define_problem(function, noise_std, x_start, x_end)
    model_constructor = get_model_constructor(
        kernel,
        likelihood_type,
        optimiser_type,
        learning_rate,
        training_iterations
    )
    experiment = SimpleExperiment(
        name='function_experiment',
        search_space=search_space,
        evaluation_function=problem,
        objective_name='function'
    )
    sobol_initialise(experiment, n_initial_evaluations, seed)
    bayes_opt_loop(experiment,
                   model_constructor,
                   acquisition_function,
                   evaluation_budget,
                   n_initial_evaluations=n_initial_evaluations,
                   visualise=visualise,
                   vis_start=vis_start,
                   vis_end=vis_end,
                   vis_step=vis_step,
                   log_dir=log_dir)
