# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# Need to add amlutils path so Python can find it.
import sys
sys.path.append(os.path.join(os.pardir, os.pardir))
from amlutils import transform_dataframe
from amlutils.task1.loading import load_train_set
from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_cli
from amlutils.experiment import log_parameters, log_cv_results, log_metrics

high_corr_features = [
    'x127', 'x680', 'x207', 'x416', 'x196', 'x80', 'x343', 'x280', 'x661',
    'x223', 'x775', 'x559', 'x672', 'x453', 'x187', 'x749', 'x177', 'x101',
    'x25', 'x813', 'x169', 'x764', 'x74', 'x723', 'x265', 'x487', 'x803',
    'x171', 'x139', 'x662', 'x690', 'x670', 'x241', 'x488', 'x300', 'x52',
    'x371', 'x423', 'x184', 'x448', 'x689', 'x776', 'x800', 'x717', 'x792',
    'x67', 'x534', 'x715', 'x608', 'x28', 'x425', 'x42', 'x791', 'x328',
    'x238', 'x745', 'x283', 'x85', 'x780', 'x669', 'x66', 'x291', 'x482',
    'x536', 'x75', 'x820', 'x485', 'x302', 'x68', 'x35', 'x462', 'x121',
    'x240', 'x360', 'x736', 'x827', 'x274', 'x62', 'x200', 'x126', 'x466',
    'x692', 'x457', 'x498', 'x180', 'x510', 'x597', 'x579', 'x734', 'x277',
    'x198', 'x254', 'x417', 'x30', 'x69', 'x234', 'x761', 'x726', 'x173',
    'x289', 'x486', 'x434', 'x789', 'x191', 'x285', 'x819', 'x408', 'x476',
    'x264', 'x381', 'x552', 'x753', 'x437', 'x504', 'x546', 'x783', 'x824',
    'x685', 'x696', 'x566', 'x496', 'x499', 'x142', 'x247', 'x375', 'x384',
    'x415', 'x635', 'x12', 'x464', 'x560', 'x432', 'x353', 'x823', 'x273',
    'x185', 'x400', 'x458', 'x490', 'x550', 'x686', 'x387', 'x578', 'x118',
    'x102', 'x665', 'x239', 'x525', 'x581', 'x175', 'x545', 'x22', 'x93',
    'x637', 'x204', 'x403', 'x10', 'x199', 'x508', 'x575', 'x246', 'x537',
    'x91', 'x296', 'x47', 'x170', 'x192', 'x785', 'x148', 'x473', 'x7', 'x292',
    'x439', 'x447', 'x772', 'x123', 'x619', 'x738', 'x39', 'x227', 'x344',
    'x643', 'x666', 'x224', 'x138', 'x129', 'x41', 'x214', 'x326', 'x140',
    'x301', 'x411', 'x474', 'x81', 'x739', 'x467', 'x309', 'x648', 'x594',
    'x279', 'x233', 'x461', 'x406', 'x210', 'x275', 'x477', 'x540', 'x786',
    'x329', 'x766', 'x797', 'x644', 'x599', 'x523']


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

    # Drop features with low correlation with label based on training data.
    X_train = X_train[high_corr_features]

    # Preprocess dataset.
    imputer = SimpleImputer(strategy='median')
    transform_dataframe(imputer, X_train)

    search_space = {
        'n_estimators': hp.uniformint('n_estimators', 50, 2000),
        'max_samples': hp.uniform('max_samples', 0.0, 1.0),
        'contamination': hp.uniform('contamination', 0.0, 0.5),
        'max_features': hp.uniform('max_features', 0.0, 1.0),
        'bootstrap': hp.choice('bootstrap', [True, False])}

    skf = StratifiedKFold(n_splits=5, random_state=cli_args.seed, shuffle=True)

    def gbr_neg_valid_score(hyperparams):
        n_estimators = int(hyperparams['n_estimators'])
        max_samples = hyperparams['max_samples']
        contamination = hyperparams['contamination']
        max_features = hyperparams['max_features']
        bootstrap = hyperparams['bootstrap']

        valid_scores = []
        for train_index, test_index in skf.split(X_train, y_train):
            X_train_split = X_train.loc[train_index]
            y_train_split = y_train.loc[train_index]

            X_test_split = X_train.loc[test_index]
            y_test_split = y_train.loc[test_index]

            isolation = IsolationForest(
                n_estimators=n_estimators,
                max_samples=max_samples,
                contamination=contamination,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=cli_args.seed,
                n_jobs=-1)

            outlier_pred = isolation.fit_predict(X_train_split, y_train_split)
            X_train_filtered = X_train_split[outlier_pred == 1]
            y_train_filtered = y_train_split[outlier_pred == 1]

            scaler = StandardScaler()
            X_train_filtered.values[:] = scaler.fit_transform(X_train_filtered)
            X_test_split.values[:] = scaler.fit_transform(X_test_split)

            gbr = GradientBoostingRegressor(
                learning_rate=0.01, n_estimators=2600, max_depth=4,
                subsample=0.45, random_state=cli_args.seed)

            gbr.fit(X_train_filtered, pd.Series.ravel(y_train_filtered))

            valid_preds = gbr.predict(X_test_split)
            valid_score = r2_score(y_test_split, valid_preds)
            valid_scores.append(valid_score)

        mean_r2_score = float(np.mean(valid_scores))

        with experiment.validate():
            experiment.log_metric('r2_score', mean_r2_score)

        result = {
            'n_estimators': n_estimators,
            'max_samples': max_samples,
            'contamination': contamination,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'loss': -mean_r2_score,
            'status': STATUS_OK}
        return result

    trials = Trials()
    best_hyperparams = fmin(fn=gbr_neg_valid_score, space=search_space,
                            algo=tpe.suggest, max_evals=100, trials=trials)

    # Save cross validation results.
    trial_results = pd.DataFrame(trials.results)
    trial_results['valid_score'] = -trial_results['loss']
    del trial_results['loss']
    log_cv_results(experiment, trial_results)

    # Define experiment metrics for CometML to be logged to the project under
    # the experiment.
    log_metrics(experiment, best_hyperparams)


if "__main__" == __name__:
    main()
