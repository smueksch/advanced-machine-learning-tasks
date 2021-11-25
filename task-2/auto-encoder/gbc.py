# Control experiment with one-hot encoding and no normalization.

# Must import before sklearn or else ImportError. DO NOT remove following
# import.
from comet_ml import Experiment
import os
import numpy as np
import pandas as pd
from scipy.sparse.construct import rand
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Need to add amlutils path so Python can find it.
import sys
sys.path.append(os.path.join(os.pardir, os.pardir))
from amlutils import get_mean
from amlutils.task2.loading import load_train_set, load_test_set
from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_cli
from amlutils.experiment import log_parameters, log_predictions, log_metrics

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001


class ConvAutoEncoder(pl.LightningModule):
    '''
    Architecture based on:
    https://pythonwife.com/convolutional-autoencoders-opencv/
    '''

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, stride=3),
            nn.BatchNorm1d(num_features=1),
            nn.Tanh(),
            nn.AvgPool1d(kernel_size=10, stride=2),
            nn.Dropout(p=0.3),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=1, out_channels=1, kernel_size=10, stride=3),
            nn.BatchNorm1d(num_features=1),
            nn.Tanh(),
            nn.Upsample(size=17842),
            nn.Dropout(p=0.3),)

    def forward(self, X):
        embedding = self.encoder(X)
        return embedding

    def reconstruct(self, X):
        Z = self.encoder(X)
        X_hat = self.decoder(Z)
        return X_hat

    def training_step(self, batch, batch_idx):
        X, y = batch
        X_hat = self.reconstruct(X)
        loss = F.mse_loss(X_hat, X)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    cli_args = get_cli_arguments()
    experiment = build_experiment_from_cli(cli_args)
    experiment.add_tag('task-2')

    np.random.seed(cli_args.seed)

    # Define experiment parameters for CometML to be logged to the project under
    # the experiment.
    params = {
        'random_seed': cli_args.seed,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE}
    log_parameters(experiment, params)

    # Load dataset.
    path_to_data = os.path.join(os.pardir, 'data')
    X_train, y_train = load_train_set(path_to_data)
    X_test = load_test_set(path_to_data)

    # Preprocessing.
    X_train_scaled = X_train.T
    X_train_scaled.values[:] = MinMaxScaler(
        feature_range=(-1, 1)).fit_transform(X_train_scaled)
    X_train_scaled = X_train_scaled.T

    X_train_scaled = X_train_scaled.fillna(0.0)

    X_test_scaled = X_test.T
    X_test_scaled.values[:] = MinMaxScaler(
        feature_range=(-1, 1)).fit_transform(X_test_scaled)
    X_test_scaled = X_test_scaled.T

    X_test_scaled = X_test_scaled.fillna(0.0)

    # Create training DataLoader.
    X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float)
    X_train_tensor = torch.unsqueeze(X_train_tensor, 1)
    y_train_tensor = torch.tensor(y_train.values)

    train_tensor = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_tensor,
                              batch_size=BATCH_SIZE, shuffle=True)

    # Train the AutoEncoder.
    ecg_auto_encoder = ConvAutoEncoder()

    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="train_loss")])
    trainer.fit(ecg_auto_encoder, train_loader)
    trainer.save_checkpoint('gbc-ecg-auto-encoder.ckpt')

    # Encode training and test sets.
    with torch.no_grad():
        ecg_auto_encoder.eval()

        X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float)
        X_train_tensor = torch.unsqueeze(X_train_tensor, 1)
        X_train_enc = ecg_auto_encoder(X_train_tensor)
        X_train_enc = torch.squeeze(X_train_enc)
        X_train_enc = pd.DataFrame(X_train_enc.numpy(), index=X_train.index)

        X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float)
        X_test_tensor = torch.unsqueeze(X_test_tensor, 1)
        X_test_enc = ecg_auto_encoder(X_test_tensor)
        X_test_enc = torch.squeeze(X_test_enc)
        X_test_enc = pd.DataFrame(X_test_enc.numpy(), index=X_test.index)

    # Cross-validate GradientBoostingClassifier.
    gbc = GradientBoostingClassifier(
        verbose=2,
        random_state=cli_args.seed)

    cv_scores = cross_validate(
        gbc,
        X_train_enc,
        pd.Series.ravel(y_train),
        cv=5,
        scoring='f1_micro',
        verbose=4,
        n_jobs=-1,
        return_train_score=True)

    # Fit model on full training set and compute test predictions.
    gbc.fit(X_train_enc, y_train)

    predictions = gbc.predict(X_test)
    log_predictions(experiment, predictions, X_test.index)

    # Define experiment metrics for CometML to be logged to the project under
    # the experiment.
    metrics = {
        'train_f1_score': get_mean(cv_scores['train_score']),
        'valid_f1_score': get_mean(cv_scores['test_score']),
        'fit_time': get_mean(cv_scores['fit_time']),
        'score_time': get_mean(cv_scores['score_time'])}
    log_metrics(experiment, metrics)


if "__main__" == __name__:
    main()
