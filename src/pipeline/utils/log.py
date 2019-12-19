from pathlib import Path
import sys
import subprocess


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
    label = subprocess.check_output(["git", "describe", "--always"]).strip()
    file.write(f'latest git commit on this branch: {label}\n')
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
