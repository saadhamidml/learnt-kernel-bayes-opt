from ax.modelbridge.registry import Models


def sobol(experiment, n_initial_evaluations, seed=None):
    print(f'Running Sobol initialisation...')
    sobol = Models.SOBOL(experiment.search_space, seed=seed)
    for i in range(n_initial_evaluations):
        print(f'Sobol: {i + 1}/{n_initial_evaluations}')
        experiment.new_trial(generator_run=sobol.gen(1))
