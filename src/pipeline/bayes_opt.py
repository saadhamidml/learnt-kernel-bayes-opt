from pathlib import Path
import torch
from botorch.acquisition.analytic import ExpectedImprovement
from ax.modelbridge.registry import Models
from plotly import graph_objs as go
from ax.plot.slice import plot_slice

from pipeline.modelling import visualisation as mod_vis


def sobol_initialise(experiment, n_initial_evaluations, seed=None):
    print(f'Running Sobol initialisation...')
    sobol = Models.SOBOL(experiment.search_space, seed=seed)
    for i in range(n_initial_evaluations):
        print(f'Sobol: {i + 1}/{n_initial_evaluations}')
        experiment.new_trial(generator_run=sobol.gen(1))


def get_acqf_constructor(acqf='ei'):
    if acqf == 'ei':
        acqf_class = ExpectedImprovement

    def acf_constructor(
            model,
            objective_weights,
            outcome_constraints,
            X_observed,
            X_pending,
            **kwargs
    ):
        return ExpectedImprovement(model, best_f=torch.max(X_observed))
    return acf_constructor


def bayes_opt_loop(experiment,
                   model_constructor,
                   acqf,
                   evaluation_budget,
                   n_initial_evaluations=0,
                   batch_size=1,
                   visualise=False,
                   vis_start=None,
                   vis_end=None,
                   vis_step=None,
                   log_dir=Path('./')):
    print(f'Running Bayesian Optimisation')
    acqf_constructor = get_acqf_constructor(acqf)
    for i in range(n_initial_evaluations, evaluation_budget):
        print(f'BO: {i + 1}/{evaluation_budget}')
        model = Models.BOTORCH(
            experiment=experiment,
            data=experiment.eval(),
            search_space=experiment.search_space,
            model_constructor=model_constructor,
            # acqf_constructor=acqf_constructor,
        )
        experiment.new_trial(generator_run=model.gen(batch_size))

        if visualise:
            log_dir_i = log_dir / f'{i}'
            log_dir_i.mkdir(parents=True, exist_ok=True)
            # go.Figure(
            #     data=plot_slice(model, 'x', 'function').data
            # ).write_image(str(log_dir_i / 'posterior.png'))
            mod_vis.posterior(
                model,
                experiment,
                vis_start,
                vis_end,
                vis_step,
                log_dir=log_dir_i
            )
            mod_vis.kernel(
                model,
                vis_start,
                vis_end,
                vis_step,
                log_dir=log_dir_i
            )
            mod_vis.acquisition_function(
                model,
                experiment,
                vis_start,
                vis_end,
                vis_step,
                log_dir=log_dir_i
            )
