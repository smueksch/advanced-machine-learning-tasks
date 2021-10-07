# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import pandas as pd
import yaml

from typing import Any
from collections.abc import Sequence


def log_parameters(experiment: Experiment, parameters: dict) -> None:
    '''
    Log parameter settings to CometML and a local YAML file.

    Args:
        experiment (Experiment): Comet.ml experiment to log to.
        params (dict): Parameter mappings to log.
    '''
    experiment.log_parameters(parameters)

    # Log parameters to local file system
    with open('parameters.yml', 'w+') as outfile:
        yaml.dump(parameters, outfile)


def log_predictions(
        experiment: Experiment, predictions: Sequence[Any],
        index: Sequence[Any],
        filename: str) -> None:
    '''
    Log prediction files to CometML and a local CSV file.

    Args:
        experiment (Experiment): Comet.ml experiment to log to.
        predictions (Sequence): List-like holding predictions.
        index (Sequence): List-like holding index for predictions.
        filename (str): CSV file to save predictions to.
    '''
    preds_df = pd.DataFrame(predictions, index=index, columns=['y'])

    # Adjust ID column name to fit the format expected in the output.
    preds_df.index.name = 'Id'

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
    with open('metrics.yml', 'w+') as outfile:
        yaml.dump(metrics, outfile)
