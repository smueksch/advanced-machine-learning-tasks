# Control experiment with one-hot encoding and no normalization.

# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate

# Need to add amlutils path so Python can find it.
import sys
sys.path.append(os.path.join(os.pardir, os.pardir))
from amlutils import get_mean
from amlutils.task1.loading import load_train_set, load_test_set
from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_cli
from amlutils.experiment import log_parameters, log_predictions, log_metrics


def main():
    cli_args = get_cli_arguments()
    experiment = build_experiment_from_cli(cli_args)
    experiment.add_tag('task-1')

    np.random.seed(cli_args.seed)

    # Define experiment parameters for CometML to be logged to the project under
    # the experiment.
    params = {
        'valid_split': cli_args.valid_split,
        'random_seed': cli_args.seed}
    log_parameters(experiment, params)

    # Load dataset.
    path_to_data = os.path.join(os.pardir, 'data')
    X_train, y_train = load_train_set(path_to_data)
    X_test = load_test_set(path_to_data)

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    gbr = GradientBoostingRegressor()

    # Cross-validate model.
    cv_scores = cross_validate(
        gbr,
        X_train,
        y_train,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        verbose=1,
        return_train_score=True)

    # Fit model on full training set and compute test predictions.
    gbr.fit(X_train, y_train)

    predictions = gbr.predict(X_test)
    log_predictions(experiment, predictions, X_test.index)

    # Define experiment metrics for CometML to be logged to the project under
    # the experiment.
    metrics = {
        'train_r2_score': get_mean(cv_scores['train_score']),
        'valid_r2_score': get_mean(cv_scores['test_score']),
        'fit_time': get_mean(cv_scores['fit_time']),
        'score_time': get_mean(cv_scores['score_time'])}
    log_metrics(experiment, metrics)


if "__main__" == __name__:
    main()
