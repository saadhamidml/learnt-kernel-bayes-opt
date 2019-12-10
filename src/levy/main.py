from pathlib import Path
from ax import SimpleExperiment
from ax.benchmark.benchmark_suite import BOBenchmarkingSuite
from IPython.core.display import HTML

from .utils import cli
from .problem import define_problem
from pipeline.utils import log
from pipeline.modelling.utils import get_model_constructor
from pipeline.bayes_opt import sobol_initialise, bayes_opt_loop

from .strategies import define_strategy


def run_experiment_wrapper(parser):
    flags, unparsed_args = cli.get_flags(parser)
    log_dir = log.default_log_dir(flags)
    log.save_config(flags, unparsed_args, log_dir=log_dir)
    run_experiment(noise_std=flags.noise_std,
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
                   log_dir=log_dir)


def run_experiment(noise_std=0.001,
                   n_initial_evaluations=45,
                   evaluation_budget=50,
                   seed=None,
                   kernel='rbf',
                   likelihood_type='gaussian',
                   optimiser_type='sgd',
                   learning_rate=1e-1,
                   training_iterations=50,
                   acquisition_function='nei',
                   visualise=False,
                   vis_start=-10,
                   vis_end=10.1,
                   vis_step=0.1,
                   log_dir=Path('./')):
    levy_problem, levy_search_space = define_problem(noise_std)
    model_constructor = get_model_constructor(
        kernel,
        likelihood_type,
        optimiser_type,
        learning_rate,
        training_iterations
    )
    experiment = SimpleExperiment(
        name='experiment',
        search_space=levy_search_space,
        evaluation_function=levy_problem,
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
