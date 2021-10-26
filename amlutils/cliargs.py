import argparse


def get_cli_arguments() -> argparse.Namespace:
    '''Return parsed comand-line arguments object.'''
    parser = argparse.ArgumentParser(
        description='Process experiment parameters.')

    parser.add_argument('--name', help='name for experiment')
    parser.add_argument('--seed', type=int, help='random seed for experiment')
    parser.add_argument(
        '--disable-comet',
        action='store_true',
        help='flat to disable Comet.ml interaction (default: false)')

    return parser.parse_args()
