# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import os
import sys

from pytorch_lightning import callbacks
sys.path.append(os.path.join(os.pardir, os.pardir))

import random

import numpy as np

import torch
import torchvision.transforms.functional as F_transforms

import pytorch_lightning as pl

# Need to add amlutils path so Python can find it.

from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_config
from amlutils.experiment import log_parameters, log_metrics
from amlutils.task3 import read_config, UNet, load_zipped_pickle, build_augmented_dataset_loader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # Setup experiment information.
    cli_args = get_cli_arguments()
    config = read_config(cli_args.config)
    experiment = build_experiment_from_config(config)

    # Log configuration file.
    experiment.log_asset(cli_args.config)

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
        'epochs': config.epochs,
        'batch_size': config.batch_size}
    log_parameters(experiment, params)

    # Load training set.
    train_set = load_zipped_pickle(os.path.join(
        config.data_dir, config.training_set_file))

    train_loader = build_augmented_dataset_loader(train_set, config.batch_size)

    # TODO:
    # 4) Set up script to run this training

    # Train model.
    mv_segmenter = UNet(
        experiment=experiment,
        bounding_box_importance=config.bounding_box_importance,
        learning_rate=config.learning_rate,
        debug=config.debug)
    mv_segmenter.to(DEVICE)

    trainer = pl.Trainer(
        default_root_dir=config.model_checkpoint_dir,
        max_epochs=config.epochs,
        deterministic=True
        )
    trainer.fit(mv_segmenter, train_loader)

    # Save model.
    trainer.save_checkpoint(config.model_filename)

    # Define experiment metrics for CometML to be logged to the project under
    # the experiment.
    metrics = {
        'final_train_loss': trainer.logged_metrics['train_loss']
        }
    log_metrics(experiment, metrics)


if '__main__' == __name__:
    main()
