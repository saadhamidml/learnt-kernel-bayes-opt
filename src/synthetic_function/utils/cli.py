def get_flags(parser):
    parser.set_defaults(
        function='exp_sin_squared',
        noise_std=0.001,
        x_start=-5,
        x_end=5,
        evaluation_budget=20,
        n_initial_evaluations=3,
        vis_start=-5,
        vis_end=5,
        vis_step=0.1
    )
    flags, unparsed_args = parser.parse_known_args()
    return flags, unparsed_args