def get_flags(parser):
    parser.add_argument('--function', type=str, default='exp_sin_squared')
    parser.set_defaults(
        nu=1.5,
        n_mixtures=4,
        noise_std=1e-3,
        evaluation_budget=15,
        n_initial_evaluations=2,
    )
    flags, unparsed_args = parser.parse_known_args()
    return flags, unparsed_args