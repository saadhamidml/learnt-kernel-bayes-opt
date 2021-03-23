from pathlib import Path
import torch
from botorch import test_functions
from ax.core.search_space import SearchSpace
from ax.core.parameter import ParameterType, RangeParameter

import pipeline.utils.visualisation as vis


def define_problem(
        function_name='exp_sin_squared',
        n_dimensions=None,
        noise_std=0.001,
        observer=None,
        visualise=False,
        vis_density=250,
        log_dir=Path('./')
):

    def get_botorch_test_function(override_name=None):
        """Gets function defined in botorch test_functions.

        All parameters are obtained from the define_problems scope.
        """
        if override_name is None:
            function_class = getattr(
                test_functions,
                function_name.capitalize()
            )
        else:
            function_class = getattr(test_functions, override_name)
        if n_dimensions is not None:
            function = function_class(
                dim=n_dimensions,
                noise_std=noise_std,
                negate=True
            )
        else:
            function = function_class(
                noise_std=noise_std,
                negate=True
            )
        return function

    if function_name == 'exp_sin_squared':
        def function(x):
            return torch.exp(-(torch.sin(3 * x)**2 + x**2))
        function.dim = 1
        function._bounds = [(-5, 5)]
        function._optimizers = [(0,)]
        function._optimal_value = 1
    elif (
            function_name == 'ackley'
            or function_name == 'griewank'
            or function_name == 'levy'
            or function_name == 'michalewicz'
            or function_name == 'rastrigin'
            or function_name == 'rosenbrock'
    ):
        function = get_botorch_test_function()
    elif function_name == 'dixonprice':
        function = get_botorch_test_function(override_name='DixonPrice')
    elif function_name == 'threehumpcamel':
        function = get_botorch_test_function(override_name='ThreeHumpCamel')

    if observer is not None:
        observer.current['optimum']['x'] = torch.tensor(
            function._optimizers[0]
        )
        observer.current['optimum']['function'] = function._optimal_value

    if visualise:
        observer.current['function'] = function
        vis.function(
            function,
            vis_start=function._bounds[0][0],
            vis_end=function._bounds[0][1],
            vis_density=vis_density,
            function_name=function_name,
            log_dir=log_dir.parents[2]
        )

    def evaluation_function(parameterisation):
        return {
            'function': (
                function(torch.tensor(list(parameterisation.values()))).item(),
                noise_std
            )
        }

    search_space = SearchSpace(
        parameters=[
            RangeParameter(name=f'x{i}',
                           parameter_type=ParameterType.FLOAT,
                           lower=function._bounds[i][0],
                           upper=function._bounds[i][1])
            for i in range(function.dim)
        ]
    )

    return evaluation_function, search_space
