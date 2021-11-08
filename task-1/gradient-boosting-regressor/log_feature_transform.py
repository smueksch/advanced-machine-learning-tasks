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
from amlutils.task1.loading import load_train_set, load_test_set
from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_cli
from amlutils.experiment import log_parameters, log_predictions, log_metrics, log_cv_results


# Features selected based on the the following criteria using the training
# set, all must hold simultaneously:
# 1) Absolute Pearson correlation with label y > 0.1 AND
# 2) Absolute Spearman correlation with label y > 0.1
high_corr_features = [
    'x581', 'x685', 'x199', 'x523', 'x792', 'x91', 'x254', 'x184', 'x739',
    'x238', 'x247', 'x783', 'x734', 'x644', 'x302', 'x292', 'x686', 'x745',
    'x301', 'x309', 'x692', 'x485', 'x550', 'x22', 'x28', 'x552', 'x227',
    'x275', 'x280', 'x177', 'x240', 'x425', 'x726', 'x670', 'x474', 'x749',
    'x285', 'x12', 'x594', 'x192', 'x496', 'x482', 'x797', 'x408', 'x384',
    'x736', 'x467', 'x447', 'x187', 'x488', 'x537', 'x344', 'x738', 'x559',
    'x415', 'x170', 'x723', 'x813', 'x820', 'x776', 'x464', 'x696', 'x403',
    'x487', 'x508', 'x525', 'x416', 'x437', 'x540', 'x75', 'x753', 'x30',
    'x461', 'x824', 'x578', 'x786', 'x387', 'x510', 'x597', 'x473', 'x74',
    'x608', 'x775', 'x239', 'x536', 'x689', 'x126', 'x169', 'x823', 'x200',
    'x25', 'x246', 'x93', 'x476', 'x80', 'x619', 'x283', 'x819', 'x81', 'x66',
    'x198', 'x291', 'x353', 'x417', 'x148', 'x196', 'x499', 'x224', 'x118',
    'x264', 'x273', 'x490', 'x566', 'x35', 'x457', 'x274', 'x69', 'x180',
    'x175', 'x123', 'x648', 'x635', 'x52', 'x789', 'x279', 'x138', 'x101',
    'x764', 'x10', 'x599', 'x411', 'x121', 'x780', 'x343', 'x39', 'x717',
    'x803', 'x665', 'x434', 'x265', 'x680', 'x827', 'x68', 'x381', 'x241',
    'x690', 'x791', 'x360', 'x171', 'x448', 'x643', 'x85', 'x439', 'x715',
    'x666', 'x375', 'x432', 'x127', 'x41', 'x234', 'x300', 'x371', 'x672',
    'x534', 'x423', 'x661', 'x67', 'x62', 'x223', 'x462', 'x575', 'x173',
    'x277', 'x761', 'x129', 'x772', 'x669', 'x47', 'x662', 'x504', 'x102',
    'x142', 'x185', 'x766', 'x498', 'x233', 'x579', 'x140', 'x207', 'x42',
    'x7', 'x477', 'x637', 'x210', 'x214', 'x486', 'x545', 'x406', 'x546',
    'x400', 'x453', 'x458', 'x560', 'x204', 'x329', 'x785', 'x800', 'x139',
    'x466', 'x289', 'x328']

# Manually inspected features that would seemingly benefit from being
# transformed using a log function, i.e. y seems to depend linearly on
# log(x###), not x###. These features are a subset of the above.
log_transform_features = ['x85', 'x170',
                          'x302', 'x408', 'x559', 'x776', 'x785']


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
    X_train = X_train[high_corr_features]
    X_test = X_test[high_corr_features]

    # Log transform hand-selected features based on training data.
    X_train[log_transform_features] = np.log(X_train[log_transform_features])
    X_test[log_transform_features] = np.log(X_test[high_corr_features])

    # Define model pipeline.
    model_pipeline = Pipeline(
        [('imputer', SimpleImputer(strategy='median')),
         ('scaler', StandardScaler()),
         ('gbr', GradientBoostingRegressor())])

    # Run grid search over hyperparameters.
    parameter_grid = {
        'gbr__learning_rate': [0.01],
        'gbr__n_estimators': [2600],
        'gbr__max_depth': [4],
        'gbr__subsample': [0.45]}

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
        'best_subsample': grid_search.best_params_['gbr__subsample'],
        'refit_time': grid_search.refit_time_}
    log_metrics(experiment, metrics)


if "__main__" == __name__:
    main()
