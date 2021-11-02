# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from typing import List

# Need to add amlutils path so Python can find it.
import sys
sys.path.append(os.path.join(os.pardir, os.pardir))
from amlutils import get_mean
from amlutils.task1.loading import load_train_set, load_test_set
from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_cli
from amlutils.experiment import log_parameters, log_predictions, log_metrics, log_cv_results


def get_high_correlation_features(
        X: pd.DataFrame, y: pd.DataFrame, threshold: float) -> List[str]:
    '''
    Return X column labels with abs. correlation with y greater than threshold.

    Args:
        X (pd.DataFrame): Data matrix, columns represent features.
        y (pd.DataFrame): Label matrix.
        threshold (float): Threshold for absolute covariance of X columns with
            label y.

    Returns:
        list[str]: Names of columns in X that have absolute correlation with y
            strictly greater than threshold.
    '''
    X_y = pd.concat([X, y], axis=1)

    # Compute corrleation of X's columns with y, but discard correlation of
    # y with itself ([:-1]).
    y_correlation = X_y.corr()['y'][:-1]

    # Compute aboslute correlation and only keep X's column labels when above
    # given threshold.
    y_abs_correlation = np.abs(y_correlation).fillna(0)
    high_abs_corr_features = y_abs_correlation[y_abs_correlation > threshold]
    return list(high_abs_corr_features.index.values)


def main():
    cli_args = get_cli_arguments()
    experiment = build_experiment_from_cli(cli_args)
    experiment.add_tag('task-1')

    np.random.seed(cli_args.seed)

    # Define experiment parameters for CometML to be logged to the project under
    # the experiment.
    params = {
        'random_seed': cli_args.seed}
    log_parameters(experiment, params)

    # Load dataset.
    path_to_data = os.path.join(os.pardir, 'data')
    X_train, y_train = load_train_set(path_to_data)
    X_test = load_test_set(path_to_data)

    # Drop features with low correlation with label based on training data.
    high_corr_features = get_high_correlation_features(X_train, y_train, 0.1)
    X_train = X_train[high_corr_features]
    X_test = X_test[high_corr_features]

    # Define model pipeline.
    model_pipeline = Pipeline(
        [('imputer', SimpleImputer(strategy='median')),
         ('scaler', StandardScaler()),
         ('gbr', GradientBoostingRegressor())])

    # Run grid search over hyperparameters.
    parameter_grid = {
        'gbr__learning_rate': [0.001, 0.01, 0.1],
        'gbr__n_estimators':
        [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
         1400, 1500, 1600, 1700, 1800, 1900, 2000],
        'gbr__max_depth': [1, 2, 3, 4]}

    grid_search = GridSearchCV(
        model_pipeline,
        parameter_grid,
        scoring='r2',
        n_jobs=-1,
        refit=True,
        cv=5,
        verbose=4,
        return_train_score=True)

    grid_search.fit(X_train, y_train)

    # Save cross validation results.
    log_cv_results(experiment, pd.DataFrame.from_dict(grid_search.cv_results_))

    # Make test predictions with best model found during hyperparameter search.
    predictions = grid_search.predict(X_test)
    log_predictions(experiment, predictions, X_test.index)

    # Define experiment metrics for CometML to be logged to the project under
    # the experiment.
    metrics = {
        'valid_r2_score': grid_search.best_score_,
        'best_valid_r2_score': grid_search.best_score_,
        'best_learning_rate': grid_search.best_params_['gbr__learning_rate'],
        'best_n_estimators': grid_search.best_params_['gbr__n_estimators'],
        'best_max_depth': grid_search.best_params_['gbr__max_depth'],
        'refit_time': grid_search.refit_time_}
    log_metrics(experiment, metrics)


if "__main__" == __name__:
    main()
