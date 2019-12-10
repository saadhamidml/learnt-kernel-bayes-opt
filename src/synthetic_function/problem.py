import numpy as np
import torch
from botorch.test_functions import Levy
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.core.search_space import SearchSpace
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.optimization_config import OptimizationConfig
from ax.core.objective import Objective
from ax.benchmark.benchmark_problem import BenchmarkProblem


def define_problem(
        function_name='exp_sin_squared',
        noise_std=0.001,
        x_start=-5,
        x_end=5
):

    if function_name == 'exp_sin_squared':
        def function(x):
            return torch.exp(-(torch.sin(3 * x)**2 + x**2))

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

    # optimisation_config = OptimizationConfig(
    #     objective=Objective(
    #         metric=LevyObjective(name='objective',
    #                              param_names=['x'],
    #                              noise_sd=noise_std)
    #     )
    # )
    #
    # return BenchmarkProblem(
    #     name='Levy',
    #     fbest=optimisation_config.objective.metric.func._optimal_value,
    #     optimization_config=optimisation_config,
    #     search_space=search_space)

    return evaluation_function, search_space
