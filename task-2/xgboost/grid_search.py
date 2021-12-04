# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import os
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Need to add amlutils path so Python can find it.
import sys
sys.path.append(os.path.join(os.pardir, os.pardir))
from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_cli
from amlutils.experiment import log_parameters, log_predictions, log_metrics, log_cv_results


def main():
    cli_args = get_cli_arguments()
    experiment = build_experiment_from_cli(cli_args)
    experiment.add_tag('task-2')

    np.random.seed(cli_args.seed)

    # Define experiment parameters for CometML to be logged to the project under
    # the experiment.
    params = {
        'random_seed': cli_args.seed}
    log_parameters(experiment, params)

    # Load dataset.
    data_dir = os.path.join(os.pardir, 'selected-features-data')
    X_train = np.load(os.path.join(data_dir, 'X_train_new.npy'))
    y_train = pd.Series.ravel(pd.read_csv(
        os.path.join(data_dir, 'y_train.csv'), index_col='id'))
    X_test = np.load(os.path.join(data_dir, 'X_test_new.npy'))

    # Run grid search over hyperparameters.
    parameter_grid = {
        'n_estimators': [200, 400, 600, 800, 1000],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'gamma': [0.0, 0.5, 1.0],
        'subsample': [0.7, 0.85, 1.0],
        'reg_alpha': [1e-5, 1e-3]}

    xgbc = XGBClassifier(tree_method='hist', n_jobs=8,
                         random_state=cli_args.seed)

    grid_search = GridSearchCV(
        xgbc,
        parameter_grid,
        scoring='f1_micro',
        n_jobs=8,
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
        'best_n_estimators': grid_search.best_params_['n_estimators'],
        'best_max_depth': grid_search.best_params_['max_depth'],
        'best_learning_rate': grid_search.best_params_['learning_rate'],
        'best_gamma': grid_search.best_params_['gamma'],
        'best_subsample': grid_search.best_params_['subsample'],
        'best_reg_alpha': grid_search.best_params_['reg_alpha'],
        'best_reg_lambda': grid_search.best_params_['reg_lambda'],
        'refit_time': grid_search.refit_time_}
    log_metrics(experiment, metrics)


if '__main__' == __name__:
    main()
