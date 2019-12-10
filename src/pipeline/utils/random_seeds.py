import argparse
import numpy as np
import torch


def generate_seeds(num_seeds):
    return np.random.randint(0, 2**32 - 1, num_seeds)


def get_seeds(seed, repeat_exp):
    if seed is None or repeat_exp > 1:
        seeds = generate_seeds(repeat_exp)
    else:
        seeds = [seed]
    return seeds


def set_seed(flags, seeds, i):
    flags.seed = seeds[i]
    torch.manual_seed(flags.seed)
    np.random.seed(flags.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds', type=int, default=3)
    flags, _ = parser.parse_known_args()

    seeds = generate_seeds(flags.num_seeds)
    output = ''
    for seed in seeds:
        output += ' ' + str(seed)
    print(output)
