import argparse


def get_cli_arguments() -> argparse.Namespace:
    '''
    Return parsed comand-line arguments object.

    By default, the random seed command-line argument is set to 42 and
    the diable Comet.ml flag is set to False.

    Returns:
        argparse.Namespace: Object containing attributes with the parsed
            command-line arguments.

    Examples:
        The return value of this function will have 3 attributes: name, seed
        and disable_comet:
        >>> args = get_cli_arguments()
        >>> args.name
        'Example name'
        >>> args.seed
        42
        >>> args.valid_split
        0.2
        >>> args.disable_comet
        False
    '''
    parser = argparse.ArgumentParser(
        description='Process experiment parameters.')

    parser.add_argument('--name', help='name for experiment')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for experiment')
    parser.add_argument('--valid-split', type=float, default=0.2,
                        help='ratio of training data to use for validation')
    parser.add_argument(
        '--disable-comet',
        action='store_true',
        help='flat to disable Comet.ml interaction (default: false)')

    return parser.parse_args()
