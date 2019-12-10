def get_flags(parser):
    parser.set_defaults(
        evaluation_budget=20,
        n_initial_evaluations=3,
        noise_std=0.001
    )
    flags, unparsed_args = parser.parse_known_args()
    return flags, unparsed_args