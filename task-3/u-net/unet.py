# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import os
import sys
sys.path.append(os.path.join(os.pardir, os.pardir))

import random

import numpy as np

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# Need to add amlutils path so Python can find it.

from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_config
from amlutils.experiment import log_parameters, log_metrics
from amlutils.task3 import read_config, UNet, load_zipped_pickle, build_data_loader


def main():
    # Setup experiment information.
    cli_args = get_cli_arguments()
    config = read_config(cli_args.config)
    experiment = build_experiment_from_config(config)

    # Set seeds for reproducability.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    pl.seed_everything(config.seed, workers=True)

    # Define experiment parameters for CometML to be logged to the project under
    # the experiment.
    params = {
        'random_seed': config.seed,
        'learning_rate': config.learning_rate,
        'epochs': config.epochs}
    log_parameters(experiment, params)

    # Load the data and build a DataLoader.
    train_set = load_zipped_pickle(os.path.join(
        config.data_dir, config.training_set_file))

    X_train = train_set[0]['Image']
    y_train = train_set[0]['Label']

    train_loader = build_data_loader(X_train, y_train)

    # Train model.
    mv_segmenter = UNet(learning_rate=config.learning_rate, debug=config.debug)

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        deterministic=True
        )
    trainer.fit(mv_segmenter, train_loader)

    # Save model.
    trainer.save_checkpoint(config.model_filename)

    # Define experiment metrics for CometML to be logged to the project under
    # the experiment.
    metrics = {}
    log_metrics(experiment, metrics)


if '__main__' == __name__:
    main()
