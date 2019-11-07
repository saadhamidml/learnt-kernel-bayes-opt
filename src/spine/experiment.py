from pathlib import Path
import itertools
import time
import numpy as np

from .utils import log, random_seeds


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
        self.results = {}

    def register_run(self, model_type, config=None):
        if config is not None:
            self.record.append(dict(zip(self.hyp_params.keys(), config)))
        else:
            self.record.append({})
        # Changing self.results will alter self.record
        self.results = self.record[-1]
        self.results['seed'] = []
        self.results['accuracy'] = []

    def compare_configs(self, metric, print_wrt=None, log_dir=Path('./')):
        """For each combo of hyp_params work out the averages and std_devs
        over runs (i.e. over seeds) for a metric in the results.

        Optionally print comparison between configs wrt to one of the
        hyp_params.
        """
        averages = []
        std_devs = []
        for config in self.record:
            config_average = np.mean(config[metric])
            config_std_dev = np.std(config[metric])
            averages.append(config_average)
            std_devs.append(config_std_dev)
            if print_wrt is not None:
                with StdoutRedirection(log_dir
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
        with StdoutRedirection(log_dir
                / f'best_{metric}_{time.strftime("%m%d_%H%M%S")}.txt'):
            print(f'Best Average {metric}')
            for key, value in best_config.items():
                print(f'{key}: {value}')
            print(f'average {metric}: {averages[best_index]}')


def multi_config(run_experiment,
                 flags,
                 unparsed_args,
                 hyp_params=None,
                 seeds=None,
                 mode='grid'):
    """Wrapper that runs the function run_experiment for different sets
    of hyperparameters and seeds.

    The relevant flags are overwitten by hyp_params on each run.
    seeds specifies the PyTorch and NumPy seeds.
    The mode can be 'grid' or 'list'.
        'grid' runs every possible combination of hyp_params.
        'list' runs sets of hyp_params. e.g. specify different epochs
            for each model with hyp_params = {'model': [m1, m2],
                                              'epochs': [e1, e2]}
    """
    # TODO: if flags.repeat_exp > len(seeds) allow some to be specified
    if seeds is None:
        seeds = random_seeds.get_seeds(flags.seed, flags.repeat_exp)
    else:
        flags.repeat_exp = len(seeds)
    if hyp_params is not None:
        ex_observer = ExObserver(seeds, hyp_params)
        # Build list of configs depending on mode
        if mode == 'grid':
            configs = itertools.product(*tuple(hyp_params.values()))
        elif mode == 'list':
            configs = zip(*tuple(hyp_params.values()))
        # Run experiment for each hyperparameter combination.
        for config in configs:
            for i, key in enumerate(hyp_params.keys()):
                setattr(flags, key, config[i])
            ex_observer.register_run(flags.model, config)
            for i in range(flags.repeat_exp):
                random_seeds.set_seed(flags, seeds, i)
                log.save_config(flags)
                ex_observer.results['seed'].append(flags.seed)
                run_experiment(flags, unparsed_args, ex_observer)
    else:
        ex_observer = ExObserver(seeds)
        ex_observer.register_run(flags.model)
        for i in range(flags.repeat_exp):
            random_seeds.set_seed(flags, seeds, i)
            log.save_config(flags)
            ex_observer.results['seed'].append(flags.seed)
            run_experiment(flags, unparsed_args, ex_observer)
    return ex_observer
