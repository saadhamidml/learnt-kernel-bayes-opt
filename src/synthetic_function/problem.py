from pathlib import Path
import torch
from botorch import test_functions
from ax.core.search_space import SearchSpace
from ax.core.parameter import ParameterType, RangeParameter

import pipeline.utils.visualisation as vis


def define_problem(
        function_name='exp_sin_squared',
        noise_std=0.001,
        observer=None,
        visualise=False,
        vis_density=250,
        log_dir=Path('./')
):

    if function_name == 'exp_sin_squared':
        def function(x):
            return torch.exp(-(torch.sin(3 * x)**2 + x**2))
        x_start = -5
        x_end = 5
        if observer is not None:
            observer.results['optimum']['x'] = 0
            observer.results['optimum']['function'] = 1
    elif function_name == 'ackley':
        function = test_functions.Ackley(
            dim=1,
            noise_std=noise_std,
            negate=True
        )
        x_start = -32.768
        x_end = 32.768
        if observer is not None:
            observer.results['optimum']['x'] = 0
            observer.results['optimum']['function'] = 0
    elif function_name == 'griewank':
        function = test_functions.Griewank(
            dim=1,
            noise_std=noise_std,
            negate=True
        )
        x_start = -600
        x_end = 600
        if observer is not None:
            observer.results['optimum']['x'] = 0
            observer.results['optimum']['function'] = 0
    elif function_name == 'levy':
        function = test_functions.Levy(dim=1, noise_std=noise_std, negate=True)
        x_start = -10
        x_end = 10
        if observer is not None:
            observer.results['optimum']['x'] = 1
            observer.results['optimum']['function'] = 0
    elif function_name == 'rastrigin':
        function = test_functions.Rastrigin(
            dim=1,
            noise_std=noise_std,
            negate=True
        )
        x_start = -5.12
        x_end = 5.12
        if observer is not None:
            observer.results['optimum']['x'] = 0
            observer.results['optimum']['function'] = 0
    elif function_name == 'rosenbrock':
        function = test_functions.Rosenbrock(
            dim=1,
            noise_std=noise_std,
            negate=True
        )
        x_start = -5
        x_end = 10
        if observer is not None:
            observer.results['optimum']['x'] = 1
            observer.results['optimum']['function'] = 0
    elif function_name == 'dixonprice':
        function = test_functions.DixonPrice(
            dim=1,
            noise_std=noise_std,
            negate=True
        )
        x_start = -10
        x_end = 10
        if observer is not None:
            observer.results['optimum']['x'] = 1
            observer.results['optimum']['function'] = 0

    if visualise:
        vis.function(
            function,
            vis_start=x_start,
            vis_end=x_end,
            vis_density=vis_density,
            log_dir=log_dir.parents[2]
        )

    def evaluation_function(parameterisation):
        return {
            'function': (
                function(torch.Tensor([parameterisation['x']])).item(),
                noise_std
            )
        }

    search_space = SearchSpace(
        parameters=[
            RangeParameter(name='x',
                           parameter_type=ParameterType.FLOAT,
                           lower=x_start,
                           upper=x_end)
        ]
    )

    return evaluation_function, search_space
