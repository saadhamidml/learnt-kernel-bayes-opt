from pathlib import Path
import torch
from ax.modelbridge.registry import Models
from plotly import graph_objs as go
from ax.plot.slice import plot_slice

from ..utils import visualisation as vis
from ..modelling import visualisation as mod_vis
from .acquisition_function import get_acqf_constructor
from . import visualisation as bo_vis

def bayes_opt_loop(experiment,
                   model_constructor,
                   acquisition_function,
                   evaluation_budget,
                   n_initial_evaluations=0,
                   batch_size=1,
                   visualise=False,
                   vis_start=None,
                   vis_end=None,
                   vis_density=None,
                   log_dir=Path('./'),
                   observer=None):
    print(f'Running Bayesian Optimisation')
    # acqf_constructor = get_acqf_constructor(acquisition_function)
    for i in range(n_initial_evaluations, evaluation_budget):
        print(f'BO: {i + 1}/{evaluation_budget}')
        model = Models.BOTORCH(
            experiment=experiment,
            data=experiment.eval(),
            search_space=experiment.search_space,
            model_constructor=model_constructor,
            # acqf_constructor=acqf_constructor,
        )

        # Work out regrets, and add to observer
        if observer is not None:
            from ax.core.observation import observations_from_data, separate_observations
            observations = observations_from_data(experiment, experiment.eval())
            observation_features, observation_data = separate_observations(observations)
            from .utils import observation_features_to_tensor, observation_data_to_tensor
            obs_x = observation_features_to_tensor(observation_features)
            obs_f = observation_data_to_tensor(observation_data)
            opt_x = observer.results['optimum']['x']
            opt_f = observer.results['optimum']['function']
            loc_regret = torch.abs(obs_x - opt_x).min()
            fun_regret = torch.abs(obs_f - opt_f).min()
            observer.results['regret']['location'].append(loc_regret.item())
            observer.results['regret']['function'].append(fun_regret.item())

        if visualise:
            # go.Figure(
            #     data=plot_slice(model, 'x', 'function').data
            # ).write_image(str(log_dir_i / 'posterior.png'))
            mod_vis.posterior(
                model,
                experiment,
                vis_start,
                vis_end,
                vis_density,
                log_dir=log_dir,
                point_num=i
            )
            mod_vis.kernel(
                model,
                vis_start,
                vis_end,
                vis_density,
                log_dir=log_dir,
                point_num=i
            )
            bo_vis.acquisition_function(
                model,
                experiment,
                vis_start,
                vis_end,
                vis_density,
                log_dir=log_dir,
                point_num=i
            )
            vis.board(
                observer.results['function'],
                model,
                experiment,
                observer.results['regret']['location'],
                observer.results['regret']['function'],
                n_initial_evaluations,
                vis_start,
                vis_end,
                vis_density,
                log_dir=log_dir,
                point_num=i
            )

        # Generate new point.
        experiment.new_trial(generator_run=model.gen(batch_size))
