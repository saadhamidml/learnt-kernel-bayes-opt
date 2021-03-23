from pathlib import Path
import itertools
import time
import numpy as np

from . import log, random_seeds


class ExObserver:
    """Object to record results of experiments.

    Each item in the list self.record is a dictionary containing the
    results of an experiment, and relevant hyp_params.
    self.results is the last dictionary in self.record; changing one
    will change the other.
    """
    def __init__(self, seeds, hyp_params=None):
        self.seeds = seeds
        self.hyp_params = hyp_params
        self.record = []
        self.current = {}

    def register_run(self, config=None):
        if config is not None:
            self.record.append(dict(zip(self.hyp_params.keys(), config)))
        else:
            self.record.append({})
        # Changing self.results will alter self.record
        self.current = self.record[-1]
        self.current['seed'] = []
        self.current['optimum'] = {}
        self.current['results'] = []

    def initialise_results_container(self):
        self.current['results'].append({
            'test_mll': [],
            'regret': {'location': [], 'function': []},
            'cumulative_location_regret': None,
            'cumulative_function_regret': None
        })

    def compare_configs(self, metric, print_wrt=None, log_dir=Path('./')):
        """For each combo of hyp_params work out the averages and std_devs
        over runs (i.e. over seeds) for a metric in the results.

        Optionally print comparison between configs wrt to one of the
        hyp_params.
        """
        averages = []
        std_devs = []
        for config in self.record:
            metric_values = []
            for i in range(len(config['seed'])):
                metric_values.append(config['results'][i][metric])
            config_average = np.mean(metric_values)
            config_std_dev = np.std(metric_values)
            averages.append(config_average)
            std_devs.append(config_std_dev)
            if print_wrt is not None:
                with log.StdoutRedirection(log_dir
                        / (f'{metric}_wrt_{print_wrt}'
                           + f'_{time.strftime("%m%d_%H%M%S")}_.txt')):
                    print(f'{print_wrt} = {config[print_wrt]}:    ',
                          f'{metric}: average = {config_average}, ',
                          f'std_dev = {config_std_dev}')
        return averages, std_devs

    def best_config(self, metric, best='max', log_dir=Path('./'), **kwargs):
        averages, _ = self.compare_configs(metric, log_dir=log_dir, **kwargs)
        if best == 'max':
            best_index = np.argmax(averages)
        else:
            best_index = np.argmin(averages)
        best_config = self.record[best_index]
        with log.StdoutRedirection(log_dir
                / f'best_{metric}_{time.strftime("%m%d_%H%M%S")}.txt'):
            print(f'Best Average {metric}')
            for key, value in best_config.items():
                print(f'{key}: {value}')
            print(f'average {metric}: {averages[best_index]}')


def repeat_experiment(
        flags,
        seeds,
        get_log_dir,
        unparsed_args,
        run_experiment_wrapper,
        observer
):
    for i in range(flags.repeat_exp):
        random_seeds.set_seed(flags, seeds, i)
        _, log_dir = get_log_dir(flags)
        log.save_config(flags, unparsed_args, log_dir)
        observer.current['seed'].append(flags.seed)
        observer.initialise_results_container()
        run_experiment_wrapper(flags, log_dir, observer)


def multi_config(
        run_experiment_wrapper,
        flags,
        unparsed_args=[],
        options=None,
        seeds=None,
        mode='grid',
        get_log_dir=None
):
    """Wrapper that runs the function run_experiment_wrapper for
    different sets of options and seeds.

    The relevant flags are overwitten by options on each run.
    seeds specifies the PyTorch and NumPy seeds.
    The mode can be 'grid' or 'list'.
        'grid' runs every possible combination of options.
        'list' runs sets of options. e.g. specify different epochs
            for each model with options = {'model': [m1, m2],
                                              'epochs': [e1, e2]}
    """

    if get_log_dir is None:
        get_log_dir = log.default_log_dir
    if seeds is None:
        seeds = random_seeds.get_seeds(flags.seed, flags.repeat_exp)
    elif flags.repeat_exp > len(seeds):
        seeds = random_seeds.add_seeds(seeds, flags.repeat_exp)
    else:
        flags.repeat_exp = len(seeds)
    if options is not None:
        observer = ExObserver(seeds, options)
        # Build list of configs depending on mode
        if mode == 'grid':
            configs = itertools.product(*tuple(options.values()))
        elif mode == 'list':
            configs = zip(*tuple(options.values()))
        # Run experiment for each hyperparameter combination.
        for config in configs:
            for i, key in enumerate(options.keys()):
                setattr(flags, key, config[i])
            observer.register_run(config)
            repeat_experiment(
                flags,
                seeds,
                get_log_dir,
                unparsed_args,
                run_experiment_wrapper,
                observer
            )
    else:
        observer = ExObserver(seeds)
        observer.register_run()
        repeat_experiment(
            flags,
            seeds,
            get_log_dir,
            unparsed_args,
            run_experiment_wrapper,
            observer
        )
    return observer
