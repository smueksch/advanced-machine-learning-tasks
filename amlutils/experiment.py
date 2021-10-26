from comet_ml import Experiment
import argparse


def build_experiment_from_cli(cli_arguments: argparse.Namespace) -> Experiment:
    '''
    Return Comet.ml Experiment object initialized according to given arguments.

    Args:
        cli_arguments (argparse.Namespace): Command-line arguments as parsed by
            cliargs.get_cli_arguments.

    Returns:
        Experiment: Comet.ml Experiment object initilized according to given
            command-line arguments.
    '''
    experiment = Experiment(
        project_name='aml-tasks',
        workspace='smueksch',
        auto_param_logging=False,
        disabled=cli_arguments.disable_comet,
        )
    experiment.set_name(cli_arguments.name)

    return experiment
