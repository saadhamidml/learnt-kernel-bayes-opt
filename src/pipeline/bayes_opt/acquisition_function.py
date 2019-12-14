import torch
from botorch.acquisition.analytic import ExpectedImprovement

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
