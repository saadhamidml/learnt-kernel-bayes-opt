from pathlib import Path


def get_log_dir(flags):
    problem_log_dir = (
            Path(flags.log_dir)
            / Path(flags.problem)
            / Path(flags.function)
            / (
                    f'{flags.n_dimensions}'
                    + f'_{flags.noise_std}'
                    + f'_{flags.n_initial_evaluations}'
                    + f'_{flags.evaluation_budget}'
            )
    )
    if flags.run_name is not None:
        log_dir = problem_log_dir / Path(flags.run_name)
    else:
        run_name = f'{flags.kernel}'
        if flags.kernel == 'matern':
            run_name += f'{flags.nu}'
        elif flags.kernel == 'spectral_mixture':
            run_name += f'{flags.n_mixtures}'
            run_name += f'{flags.mixture_means_constraint}'
        run_name += (
                f'{flags.likelihood_type}'
                # + f'{flags.optimiser_type}'
                # + f'{flags.learning_rate}'
                # + f'{flags.training_iterations}'
        )
        log_dir = problem_log_dir / Path(run_name) / f'{flags.seed}'
    log_dir.mkdir(parents=True, exist_ok=True)
    return problem_log_dir, log_dir
