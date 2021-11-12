# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import os
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from xgboost.sklearn import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# Need to add amlutils path so Python can find it.
import sys
sys.path.append(os.path.join(os.pardir, os.pardir))
from amlutils.task1.loading import load_train_set, load_test_set
from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_cli
from amlutils.experiment import log_parameters, log_predictions, log_metrics, log_cv_results


selected_features = [
    'x7', 'x10', 'x12', 'x25', 'x28', 'x30', 'x35', 'x39', 'x41', 'x42', 'x47',
    'x52', 'x66', 'x67', 'x68', 'x74', 'x75', 'x80', 'x81', 'x85', 'x93',
    'x101', 'x102', 'x121', 'x123', 'x126', 'x127', 'x129', 'x139', 'x140',
    'x142', 'x148', 'x169', 'x170', 'x171', 'x173', 'x175', 'x177', 'x180',
    'x184', 'x185', 'x192', 'x196', 'x198', 'x199', 'x200', 'x204', 'x207',
    'x210', 'x214', 'x223', 'x224', 'x227', 'x233', 'x234', 'x238', 'x239',
    'x240', 'x241', 'x246', 'x247', 'x254', 'x264', 'x265', 'x273', 'x274',
    'x275', 'x277', 'x279', 'x280', 'x283', 'x285', 'x289', 'x291', 'x292',
    'x300', 'x301', 'x302', 'x328', 'x329', 'x343', 'x344', 'x353', 'x360',
    'x371', 'x375', 'x384', 'x387', 'x400', 'x406', 'x408', 'x411', 'x415',
    'x416', 'x417', 'x423', 'x425', 'x432', 'x434', 'x437', 'x439', 'x447',
    'x448', 'x453', 'x457', 'x458', 'x461', 'x462', 'x464', 'x466', 'x467',
    'x473', 'x474', 'x476', 'x477', 'x485', 'x486', 'x487', 'x488', 'x490',
    'x496', 'x498', 'x499', 'x504', 'x508', 'x510', 'x523', 'x525', 'x534',
    'x536', 'x537', 'x540', 'x545', 'x546', 'x550', 'x552', 'x559', 'x560',
    'x566', 'x575', 'x578', 'x579', 'x581', 'x597', 'x599', 'x608', 'x619',
    'x635', 'x637', 'x643', 'x644', 'x648', 'x661', 'x662', 'x665', 'x666',
    'x669', 'x670', 'x672', 'x680', 'x685', 'x686', 'x689', 'x690', 'x692',
    'x696', 'x715', 'x717', 'x723', 'x726', 'x734', 'x736', 'x738', 'x739',
    'x745', 'x749', 'x753', 'x761', 'x764', 'x766', 'x772', 'x775', 'x776',
    'x780', 'x783', 'x785', 'x786', 'x789', 'x791', 'x792', 'x797', 'x800',
    'x803', 'x813', 'x819', 'x820', 'x823', 'x824', 'x827']


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
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # Preprocess dataset, impute missing values and scale.
    imputer = SimpleImputer(strategy='median')
    transform_dataframe(imputer, X_train)
    transform_dataframe(imputer, X_test)

    scaler = StandardScaler()
    transform_dataframe(scaler, X_train)
    transform_dataframe(scaler, X_test)

    search_space = {
        'n_estimators': hp.choice('n_estimators', range(10, 500)),
        'max_depth': hp.choice('max_depth', range(2, 20)),
        'learning_rate': hp.quniform('learning_rate', 0.01, 1.0, 0.01),
        'gamma': hp.uniform('gamma', 0.0, 1.0),
        'min_child_weight': hp.uniform('min_child_weight', 1, 8),
        'subsample': hp.uniform('subsample', 0.8, 1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1.0, 0.05),
        'reg_lambda': hp.lognormal('reg_lambda', 1.0, 1.0)}

    def gbr_neg_valid_score(hyperparams):
        n_estimators = int(hyperparams['n_estimators'])
        max_depth = int(hyperparams['max_depth'])
        learning_rate = hyperparams['learning_rate']
        gamma = hyperparams['gamma']
        min_child_weight = hyperparams['min_child_weight']
        colsample_bytree = hyperparams['colsample_bytree']
        reg_lambda = hyperparams['reg_lambda']
        subsample = hyperparams['subsample']

        xgbr = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            gamma=gamma,
            min_child_weight=min_child_weight,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            subsample=subsample,
            random_state=cli_args.seed)

        r2_score = cross_val_score(
            xgbr, X_train, pd.Series.ravel(y_train),
            scoring='r2', n_jobs=-1).mean()

        with experiment.validate():
            experiment.log_metric('r2_score', r2_score)

        result = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'colsample_bytree': colsample_bytree,
            'reg_lambda': reg_lambda,
            'subsample': subsample,
            'loss': -r2_score,
            'status': STATUS_OK}
        print(result)
        return result

    trials = Trials()
    best_hyperparams = fmin(fn=gbr_neg_valid_score, space=search_space,
                            algo=tpe.suggest, max_evals=500, trials=trials)
    print(f'Best hyperparameters: {best_hyperparams}')

    # Save cross validation results.
    trial_results = pd.DataFrame(trials.results)
    trial_results['valid_score'] = -trial_results['loss']
    del trial_results['loss']
    log_cv_results(experiment, trial_results)

    # Make test predictions with best model found during hyperparameter search.
    best_n_estimators = int(best_hyperparams['n_estimators'])
    best_max_depth = int(best_hyperparams['max_depth'])
    best_learning_rate = best_hyperparams['learning_rate']
    best_gamma = best_hyperparams['gamma']
    best_min_child_weight = best_hyperparams['min_child_weight']
    best_colsample_bytree = best_hyperparams['colsample_bytree']
    best_reg_lambda = best_hyperparams['reg_lambda']
    best_subsample = best_hyperparams['subsample']

    best_xgbr = XGBRegressor(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        learning_rate=best_learning_rate,
        gamma=best_gamma,
        min_child_weight=best_min_child_weight,
        colsample_bytree=best_colsample_bytree,
        reg_lambda=best_reg_lambda,
        subsample=best_subsample,
        random_state=cli_args.seed)

    best_valid_score = cross_val_score(
        best_xgbr, X_train, pd.Series.ravel(y_train),
        scoring='r2', n_jobs=-1).mean()
    print(f'Best validation score: {best_valid_score}')

    best_xgbr.fit(X_train, pd.Series.ravel(y_train))
    predictions = best_xgbr.predict(X_test)
    log_predictions(experiment, predictions, X_test.index)

    # Define experiment metrics for CometML to be logged to the project under
    # the experiment.
    metrics = {
        'valid_r2_score': best_valid_score,
        'best_valid_score': best_valid_score,
        'best_n_estimators': best_n_estimators,
        'best_max_depth': best_max_depth,
        'best_learning_rate': best_learning_rate,
        'best_gamma': best_gamma,
        'best_min_child_weight': best_min_child_weight,
        'best_colsample_bytree': best_colsample_bytree,
        'best_reg_lambda': best_reg_lambda,
        'best_subsample': best_subsample}
    log_metrics(experiment, metrics)


if '__main__' == __name__:
    main()
