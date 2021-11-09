from comet_ml import Experiment
import argparse
import re
import yaml
import pandas as pd

from typing import Any, Sequence


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
        auto_param_logging=True,
        disabled=cli_arguments.disable_comet,
        )
    experiment.set_name(cli_arguments.name)

    return experiment


def __format_experiment_name(name: str) -> str:
    '''
    Return formatted experiment name.

    Formatting is applied as follows:

        1. All characters are made lowercase.
        2. Whitespaces are replaced with hyphens.

    Args:
        name (str): Name to be formatted.

    Returns:
        str: Given name with formatting applied.
    '''
    return re.sub(r'\s+', '-', name.lower())


def log_parameters(experiment: Experiment, parameters: dict) -> None:
    '''
    Log parameter settings to CometML and a local YAML file.

    Args:
        experiment (Experiment): Comet.ml experiment to log to.
        params (dict): Parameter mappings to log.
    '''
    experiment.log_parameters(parameters)

    # Log parameters to local file system
    experiment_name = __format_experiment_name(experiment.get_name())
    with open(f'{experiment_name}-parameters.yml', 'w+') as outfile:
        yaml.dump(parameters, outfile)


def log_predictions(
        experiment: Experiment, predictions: Sequence[Any],
        index: Sequence[Any]) -> None:
    '''
    Log prediction files to CometML and a local CSV file.

    Args:
        experiment (Experiment): Comet.ml experiment to log to.
        predictions (Sequence): List-like holding predictions.
        index (Sequence): List-like holding index for predictions.
    '''
    preds_df = pd.DataFrame(predictions, index=index, columns=['y'])

    # Adjust ID column name to fit the format expected in the output.
    preds_df.index.name = 'id'

    experiment_name = __format_experiment_name(experiment.get_name())
    filename = f'{experiment_name}-predictions.csv'

    # Log predictions as a CSV to CometML, retrievable under the experiment
    # by going to `Assets > dataframes`.
    experiment.log_table(filename=filename, tabular_data=preds_df)

    # Log predictions to local file system
    preds_df.to_csv(filename)


def log_metrics(experiment: Experiment, metrics: dict) -> None:
    '''
    Log metrics to CometML and a local YAML file.

    Args:
        experiment (Experiment): Comet.ml experiment to log to.
        metrics (dict): Metrics mappings to log.
    '''
    experiment.log_metrics(metrics)

    # Log metrics to local file system
    experiment_name = __format_experiment_name(experiment.get_name())
    with open(f'{experiment_name}-metrics.yml', 'w+') as outfile:
        yaml.dump(metrics, outfile)


def log_cv_results(
        experiment: Experiment, cv_results: pd.DataFrame) -> None:
    '''
    Log cross validation results to CometML and a local CSV file.

    Args:
        experiment (Experiment): Comet.ml experiment to log to.
        cv_results (pd.DataFrame): DataFrame holding cross-validation results.
    '''
    experiment_name = __format_experiment_name(experiment.get_name())
    filename = f'{experiment_name}-cv-results.csv'

    # Log predictions as a CSV to CometML, retrievable under the experiment
    # by going to `Assets > dataframes`.
    experiment.log_table(filename=filename, tabular_data=cv_results)

    # Log predictions to local file system
    cv_results.to_csv(filename)
