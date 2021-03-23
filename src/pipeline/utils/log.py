from pathlib import Path
import sys
import subprocess
import numpy as np


def default_log_dir(flags):
    problem_log_dir = Path(flags.log_dir) / Path(flags.problem)
    if flags.run_name is not None:
        log_dir = problem_log_dir / Path(flags.run_name)
    else:
        run_name = f'{flags.kernel}'
        if flags.kernel == 'matern':
            run_name += f'{flags.nu}'
        elif flags.kernel == 'spectral_mixture':
            run_name += f'{flags.n_mixtures}'
        run_name += (
                f'{flags.likelihood_type}'
                # + f'{flags.optimiser_type}'
                # + f'{flags.learning_rate}'
                # + f'{flags.training_iterations}'
        )
        log_dir = problem_log_dir / Path(run_name) / f'{flags.seed}'
    log_dir.mkdir(parents=True, exist_ok=True)
    return problem_log_dir, log_dir


def save_config(flags, unparsed_args, log_dir=Path('./')):
    """Save command line flags."""
    print("flags:", flags)
    print("unparsed_args:", unparsed_args)
    # writing arguments and git hash to info file
    # filename_log = "info_" + time.strftime("%m%d_%H%M%S") + ".txt"
    filename_log = 'info.txt'
    file = open(log_dir / filename_log, "w")
    # label = subprocess.check_output(["git", "describe", "--always"]).strip()
    # file.write(f'latest git commit on this branch: {label}\n')
    file.write('\nflags: \n')
    bash_command = 'python'
    flags_dict = vars(flags)
    for key in sorted(flags_dict):
        file.write(f'{key}: {flags_dict[key]}\n')
        bash_command += (f' --{key} {flags_dict[key]}')
    file.write('\nUNPARSED_ARGV:\n' + str(unparsed_args))
    file.write('\n\nBASH COMMAND: \n')
    file.write(bash_command)
    file.close()


class StdoutRedirection:
    """Standard output redirection context manager"""

    def __init__(self, path, mode='a'):
        self._path = path
        self.mode = mode

    def __enter__(self):
        sys.stdout = open(self._path, mode=self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = sys.__stdout__


def save_cumulative_regrets(
        record_results,
        record_name=None,
        return_regrets=False,
        log_dir=Path('./')
):
    """
    Saves cumulative function and location regrets in log_dir.

    record_results is the results dictionary stored by the observer for
    each experiment.
    """
    loc_reg = np.array(record_results['regret']['location'])
    fun_reg = np.array(record_results['regret']['function'])
    cum_loc_reg = loc_reg.sum()
    cum_fun_reg = fun_reg.sum()
    if record_results['cumulative_location_regret'] is None:
        record_results['cumulative_location_regret'] = cum_loc_reg
    if record_results['cumulative_function_regret'] is None:
        record_results['cumulative_function_regret'] = cum_fun_reg
    with StdoutRedirection(log_dir / 'cumulative_regrets.txt'):
        if record_name is not None:
            print(f'{record_name} location: {cum_loc_reg}')
            print(f'{record_name} function: {cum_fun_reg}')
        else:
            print(f'location: {loc_reg.sum()}')
            print(f'function: {fun_reg.sum()}')
    if return_regrets:
        return loc_reg, fun_reg
