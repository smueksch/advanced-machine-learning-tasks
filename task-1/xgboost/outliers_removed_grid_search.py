# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import os
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from xgboost.sklearn import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Need to add amlutils path so Python can find it.
import sys
sys.path.append(os.path.join(os.pardir, os.pardir))
from amlutils.task1.loading import load_train_set, load_test_set
from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_cli
from amlutils.experiment import log_parameters, log_predictions, log_metrics, log_cv_results

high_corr_features = [
    'x7', 'x10', 'x12', 'x22', 'x25', 'x28', 'x30', 'x35', 'x39', 'x41', 'x42',
    'x47', 'x52', 'x62', 'x66', 'x67', 'x68', 'x69', 'x74', 'x75', 'x80',
    'x81', 'x85', 'x91', 'x93', 'x101', 'x102', 'x118', 'x121', 'x123',
    'x126', 'x127', 'x129', 'x138', 'x139', 'x140', 'x142', 'x148', 'x169',
    'x170', 'x171', 'x173', 'x175', 'x177', 'x180', 'x184', 'x185', 'x187',
    'x191', 'x192', 'x196', 'x198', 'x199', 'x200', 'x204', 'x207', 'x210',
    'x214', 'x223', 'x224', 'x227', 'x233', 'x234', 'x238', 'x239', 'x240',
    'x241', 'x246', 'x247', 'x254', 'x264', 'x265', 'x273', 'x274', 'x275',
    'x277', 'x279', 'x280', 'x283', 'x285', 'x289', 'x291', 'x292', 'x296',
    'x300', 'x301', 'x302', 'x309', 'x326', 'x328', 'x329', 'x343', 'x344',
    'x353', 'x360', 'x371', 'x375', 'x381', 'x384', 'x387', 'x400', 'x403',
    'x406', 'x408', 'x411', 'x415', 'x416', 'x417', 'x423', 'x425', 'x432',
    'x434', 'x437', 'x439', 'x447', 'x448', 'x453', 'x457', 'x458', 'x461',
    'x462', 'x464', 'x466', 'x467', 'x473', 'x474', 'x476', 'x477', 'x482',
    'x485', 'x486', 'x487', 'x488', 'x490', 'x496', 'x498', 'x499', 'x504',
    'x508', 'x510', 'x523', 'x525', 'x534', 'x536', 'x537', 'x540', 'x545',
    'x546', 'x550', 'x552', 'x559', 'x560', 'x566', 'x575', 'x578', 'x579',
    'x581', 'x594', 'x597', 'x599', 'x608', 'x619', 'x635', 'x637', 'x643',
    'x644', 'x648', 'x661', 'x662', 'x665', 'x666', 'x669', 'x670', 'x672',
    'x680', 'x685', 'x686', 'x689', 'x690', 'x692', 'x696', 'x715', 'x717',
    'x723', 'x726', 'x734', 'x736', 'x738', 'x739', 'x745', 'x749', 'x753',
    'x761', 'x764', 'x766', 'x772', 'x775', 'x776', 'x780', 'x783', 'x785',
    'x786', 'x789', 'x791', 'x792', 'x797', 'x800', 'x803', 'x813', 'x819',
    'x820', 'x823', 'x824', 'x827']


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
    X_train = X_train[high_corr_features]
    X_test = X_test[high_corr_features]

    # Preprocess dataset, impute missing values and scale.
    imputer = SimpleImputer(strategy='median')
    transform_dataframe(imputer, X_train)
    transform_dataframe(imputer, X_test)

    # Remove outliers.
    lof = LocalOutlierFactor(
        n_neighbors=29,
        metric='minkowski',
        p=3,
        contamination='auto',
        n_jobs=-1)

    isolfor = IsolationForest(
        n_estimators=707,
        max_samples=0.94579,
        contamination='auto',
        max_features=0.412204,
        bootstrap=False,
        n_jobs=-1)

    outlier_pred_lof = lof.fit_predict(X_train, y_train)
    outlier_pred_isolfor = isolfor.fit_predict(
        X_train, y_train)[outlier_pred_lof == 1]

    X_train = X_train[outlier_pred_lof == 1]
    X_train = X_train[outlier_pred_isolfor == 1]

    y_train = y_train[outlier_pred_lof == 1]
    y_train = y_train[outlier_pred_isolfor == 1]

    scaler = StandardScaler()
    transform_dataframe(scaler, X_train)
    transform_dataframe(scaler, X_test)

    # Run grid search over hyperparameters.
    parameter_grid = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'gamma': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [1e-5, 1e-4, 1e-3],
        'reg_lambda': [0.1, 0.01, 1]}

    xgbr = XGBRegressor(n_jobs=16, random_state=cli_args.seed)

    grid_search = GridSearchCV(
        xgbr,
        parameter_grid,
        scoring='r2',
        n_jobs=16,
        refit=True,
        cv=5,
        verbose=4,
        return_train_score=True)

    grid_search.fit(X_train, pd.Series.ravel(y_train))

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
