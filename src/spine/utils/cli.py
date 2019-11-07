import argparse

def get_flags():
    parser = argparse.ArgumentParser()

    # Setup
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='../logs')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Folder name in log_dir for output')
    parser.add_argument('--test_run', type=int, default=0)

    # Stage of experiments
    parser.add_argument('--repeat_exp', type=int, default=1,
                        help='Random experiment with different seeds.')
    parser.add_argument('--force_train', type=int, default=0)

    # Model parameters
    parser.add_argument('--likelihood_type', type=str, default='gaussian')
    parser.add_argument('--kernel', type=str, default='rbf',
                        help='rbf, spectral_mixture')
    parser.add_argument('--optimiser_type', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=1e-1)

    # Experiment options
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for both Numpy and Pytorch.')
    parser.add_argument('--x_start', type=float, default=-10)
    parser.add_argument('--x_end', type=float, default=10.5)
    parser.add_argument('--x_step', type=float, default=0.5)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--training_iterations', type=int, default=50)

    # Visualisation options
    parser.add_argument('--vis_start', type=float, default=-15)
    parser.add_argument('--vis_end', type=float, default=15.1)
    parser.add_argument('--vis_step', type=float, default=0.1)

    # Parse arguments
    flags, unparsed_args = parser.parse_known_args()
    return flags, unparsed_args
