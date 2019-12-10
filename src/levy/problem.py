import numpy as np
import torch
from botorch.test_functions import Levy
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.core.search_space import SearchSpace
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.optimization_config import OptimizationConfig
from ax.core.objective import Objective
from ax.benchmark.benchmark_problem import BenchmarkProblem


def define_problem(noise_std=0.1):
    # class LevyObjective(NoisyFunctionMetric):
    #     def __init__(self, *args, **kwargs):
    #         super(LevyObjective, self).__init__(*args, **kwargs)
    #         self.func = Levy(dim=1, noise_std=self.noise_sd, negate=True)
    #
    #     def f(self, x: np.ndarray) -> float:
    #         return self.func(torch.Tensor(x)).item()

    levy = Levy(dim=1, noise_std=noise_std, negate=True)

    def levy_evaluation_function(parameterisation):
        return {
            'function': (
                levy(torch.Tensor([parameterisation['x']])).item(),
                noise_std
            )
        }

    search_space = SearchSpace(
        parameters=[
            RangeParameter(name='x',
                           parameter_type=ParameterType.FLOAT,
                           lower=-10.0,
                           upper=10.0)
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

    return levy_evaluation_function, search_space
