# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import os
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from typing import List

# Need to add amlutils path so Python can find it.
import sys
sys.path.append(os.path.join(os.pardir, os.pardir))
from amlutils.task1.loading import load_train_set, load_test_set
from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_cli
from amlutils.experiment import log_parameters, log_predictions, log_metrics, log_cv_results


high_corr_features = [
    'x689', 'x686', 'x715', 'x175', 'x785', 'x10', 'x198', 'x819', 'x224',
    'x791', 'x597', 'x666', 'x199', 'x827', 'x171', 'x803', 'x210', 'x672',
    'x234', 'x508', 'x680', 'x207', 'x510', 'x285', 'x723', 'x74', 'x139',
    'x635', 'x644', 'x685', 'x823', 'x200', 'x344', 'x423', 'x275', 'x434',
    'x35', 'x93', 'x68', 'x279', 'x41', 'x384', 'x661', 'x227', 'x496', 'x447',
    'x608', 'x142', 'x537', 'x101', 'x353', 'x476', 'x406', 'x780', 'x302',
    'x749', 'x400', 'x52', 'x619', 'x648', 'x820', 'x797', 'x184', 'x425',
    'x546', 'x127', 'x7', 'x292', 'x534', 'x477', 'x789', 'x696', 'x411',
    'x85', 'x239', 'x140', 'x300', 'x525', 'x102', 'x579', 'x42', 'x28',
    'x214', 'x643', 'x599', 'x490', 'x775', 'x148', 'x264', 'x39', 'x448',
    'x283', 'x457', 'x488', 'x540', 'x238', 'x487', 'x458', 'x289', 'x185',
    'x47', 'x665', 'x75', 'x180', 'x474', 'x246', 'x498', 'x575', 'x360',
    'x545', 'x578', 'x273', 'x121', 'x328', 'x415', 'x753', 'x745', 'x776',
    'x670', 'x504', 'x800', 'x170', 'x417', 'x824', 'x566', 'x192', 'x169',
    'x277', 'x559', 'x204', 'x387', 'x241', 'x301', 'x738', 'x129', 'x126',
    'x499', 'x485', 'x761', 'x375', 'x662', 'x461', 'x464', 'x523', 'x25',
    'x81', 'x717', 'x786', 'x792', 'x473', 'x432', 'x408', 'x466', 'x726',
    'x772', 'x67', 'x173', 'x343', 'x560', 'x453', 'x690', 'x123', 'x550',
    'x536', 'x196', 'x247', 'x177', 'x439', 'x669', 'x30', 'x223']
selected_log_features = ['x85', 'x170', 'x302', 'x408', 'x559', 'x776', 'x785']


def transform_dataframe(
        transformer: TransformerMixin, df: pd.DataFrame) -> None:
    '''
    Transform DataFrame with sklearn transformer while preserving structure.

    Sklearn transformers typically return NumPy arrays representing matrices
    in the shape of the inputted DataFrame. This is an issue when further code
    expects DataFrames and especially their functionality. This function
    ensures that when the transformation is applied, the result is still a
    DataFrame with the exact same structure as the input.

    The provided transformer will automatically be fitted to the given
    DataFrame.

    Args:
        transformer (TransformerMixin): Sklearn-style transformer, needs to
            implement fit_transform.
        df (pd.DataFrame): Pandas DataFrame that should be transformed.
    '''
    df.values[:] = transformer.fit_transform(df)


def log_transform(df: pd.DataFrame, features: List[str]) -> None:
    '''
    Apply sign(x) * log(1 + |x|) transformation to selected featuers.

    Args:
        df (pd.DataFrame): DataFrame with features to be transformed.
        features (List[str]): List of features that should be transformed.
    '''
    df[features] = np.sign(df[features]) * np.log1p(np.abs(df[features]))


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

    # Transform selected log-features.
    log_transform(X_train, selected_log_features)
    log_transform(X_test, selected_log_features)

    # Preprocess dataset, impute missing values and scale.
    imputer = SimpleImputer(strategy='median')
    transform_dataframe(imputer, X_train)
    transform_dataframe(imputer, X_test)

    scaler = StandardScaler()
    transform_dataframe(scaler, X_train)
    transform_dataframe(scaler, X_test)

    search_space = {
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
        'n_estimators': hp.uniformint('n_estimators', 2000, 5000),
        'max_depth': hp.uniformint('max_depth', 2, 4),
        'subsample': hp.uniform('subsample', 0.0, 1.0)}

    def gbr_neg_valid_score(hyperparams):
        learning_rate = hyperparams['learning_rate']
        n_estimators = int(hyperparams['n_estimators'])
        max_depth = int(hyperparams['max_depth'])
        subsample = hyperparams['subsample']

        gbr = GradientBoostingRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            random_state=cli_args.seed)

        r2_score = cross_val_score(
            gbr, X_train, pd.Series.ravel(y_train),
            scoring='r2', n_jobs=-1).mean()

        with experiment.validate():
            experiment.log_metric('r2_score', r2_score)

        result = {
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'subsample': subsample,
            'loss': -r2_score,
            'status': STATUS_OK
            }
        return result

    trials = Trials()
    best_hyperparams = fmin(fn=gbr_neg_valid_score, space=search_space,
                            algo=tpe.suggest, max_evals=50, trials=trials)

    # Save cross validation results.
    trial_results = pd.DataFrame(trials.results)
    trial_results['valid_score'] = -trial_results['loss']
    del trial_results['loss']
    log_cv_results(experiment, trial_results)

    # Make test predictions with best model found during hyperparameter search.
    best_learning_rate = best_hyperparams['learning_rate']
    best_n_estimators = int(best_hyperparams['n_estimators'])
    best_max_depth = int(best_hyperparams['max_depth'])
    best_subsample = best_hyperparams['subsample']

    best_gbr = GradientBoostingRegressor(
        learning_rate=best_learning_rate,
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        subsample=best_subsample,
        random_state=cli_args.seed)

    best_valid_score = cross_val_score(
        best_gbr, X_train, pd.Series.ravel(y_train),
        scoring='r2', n_jobs=-1).mean()

    best_gbr.fit(X_train, pd.Series.ravel(y_train))
    predictions = best_gbr.predict(X_test)
    log_predictions(experiment, predictions, X_test.index)

    # Define experiment metrics for CometML to be logged to the project under
    # the experiment.
    metrics = {
        'valid_r2_score': best_valid_score,
        'best_valid_score': best_valid_score,
        'best_learning_rate': best_learning_rate,
        'best_n_estimators': best_n_estimators,
        'best_max_depth': best_max_depth,
        'best_subsample': best_subsample}
    log_metrics(experiment, metrics)


if "__main__" == __name__:
    main()
