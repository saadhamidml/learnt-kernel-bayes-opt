from pathlib import Path
import numpy as np
from ax import SimpleExperiment

from .problem import define_problem
from pipeline.modelling.utils import get_model_constructor
from pipeline.bayes_opt.initialise import sobol
from pipeline.bayes_opt.bayes_opt import bayes_opt_loop
import pipeline.bayes_opt.visualisation as bo_vis


def run_experiment_wrapper(flags, log_dir=None, observer=None):
    if log_dir is None:
        log_dir = flags.log_dir
    run_experiment(
        function=flags.function,
        noise_std=flags.noise_std,
        n_initial_evaluations=flags.n_initial_evaluations,
        evaluation_budget=flags.evaluation_budget,
        seed=flags.seed,
        kernel=flags.kernel,
        nu=flags.nu,
        n_mixtures=flags.n_mixtures,
        likelihood_type=flags.likelihood_type,
        # optimiser_type=flags.optimiser_type,
        # learning_rate=flags.learning_rate,
        # training_iterations=flags.training_iterations,
        acquisition_function=flags.acquisition_function,
        visualise=flags.visualise,
        vis_density=flags.vis_density,
        log_dir=log_dir,
        observer=observer
    )


def run_experiment(
        function=None,
        noise_std=1e-3,
        n_initial_evaluations=None,
        evaluation_budget=None,
        seed=None,
        kernel='rbf',
        nu=None,
        n_mixtures=None,
        likelihood_type='gaussian',
        optimiser_type='sgd',
        learning_rate=1e-1,
        training_iterations=50,
        acquisition_function='nei',
        visualise=False,
        vis_density=None,
        log_dir=Path('./'),
        observer=None
):
    problem, search_space = define_problem(
        function,
        noise_std,
        observer=observer,
        visualise=visualise,
        vis_density=vis_density,
        log_dir=log_dir
    )
    model_constructor = get_model_constructor(
        kernel,
        likelihood_type,
        optimiser_type,
        learning_rate,
        training_iterations,
        nu=nu,
        n_mixtures=n_mixtures
    )
    experiment = SimpleExperiment(
        name='function_experiment',
        search_space=search_space,
        evaluation_function=problem,
        objective_name='function'
    )
    sobol(experiment, n_initial_evaluations, seed)
    bayes_opt_loop(
        experiment,
        model_constructor,
        acquisition_function,
        evaluation_budget,
        n_initial_evaluations=n_initial_evaluations,
        visualise=visualise,
        vis_start=search_space.parameters['x'].lower,
        vis_end=search_space.parameters['x'].upper,
        vis_density=vis_density,
        log_dir=log_dir,
        observer=observer
    )
    if visualise:
        bo_vis.regret(
            np.array(observer.results['regret']['location']),
            np.array(observer.results['regret']['function']),
            n_initial_evaluations=n_initial_evaluations,
            log_dir=log_dir
        )
