# Control experiment with one-hot encoding and no normalization.

# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Need to add amlutils path so Python can find it.
import sys
sys.path.append(os.path.join(os.pardir, os.pardir))
from amlutils.task1.loading import load_train_set, load_test_set
from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_cli
from amlutils.experiment import log_parameters, log_predictions, log_metrics

correlated_cols = ['x343',
                   'x247',
                   'x487',
                   'x485',
                   'x199',
                   'x39',
                   'x510',
                   'x621',
                   'x581',
                   'x200',
                   'x292',
                   'x458',
                   'x360',
                   'x619',
                   'x139',
                   'x47',
                   'x820',
                   'x35',
                   'x192',
                   'x761',
                   'x578',
                   'x234',
                   'x723',
                   'x227',
                   'x599',
                   'x262',
                   'x408',
                   'x285',
                   'x736',
                   'x277',
                   'x12',
                   'x462',
                   'x10',
                   'x400',
                   'x180',
                   'x171',
                   'x411',
                   'x488',
                   'x384',
                   'x204',
                   'x498',
                   'x387',
                   'x692',
                   'x669',
                   'x93',
                   'x169',
                   'x240',
                   'x275',
                   'x148',
                   'x536',
                   'x661',
                   'x207',
                   'x504',
                   'x66',
                   'x291',
                   'x140',
                   'x170',
                   'x476',
                   'x559',
                   'x30',
                   'x41',
                   'x91',
                   'x608',
                   'x434',
                   'x80',
                   'x67',
                   'x198',
                   'x254',
                   'x184',
                   'x477',
                   'x560',
                   'x432',
                   'x289',
                   'x783',
                   'x566',
                   'x246',
                   'x439',
                   'x579',
                   'x474',
                   'x101',
                   'x437',
                   'x417',
                   'x496',
                   'x523',
                   'x191',
                   'x241',
                   'x68',
                   'x328',
                   'x353',
                   'x28',
                   'x686',
                   'x177',
                   'x233',
                   'x224',
                   'x643',
                   'x717',
                   'x7',
                   'x537',
                   'x74',
                   'x175',
                   'x375',
                   'x597',
                   'x447',
                   'x126',
                   'x187',
                   'x52']


def main():
    cli_args = get_cli_arguments()
    experiment = build_experiment_from_cli(cli_args)
    experiment.add_tag('task-1')

    # Define experiment parameters for CometML to be logged to the project under
    # the experiment.
    params = {
        'model': 'linear regression',
        'valid_split': cli_args.valid_split,
        'random_seed': cli_args.seed}
    log_parameters(experiment, params)

    linear_regression = LinearRegression()

    path_to_data = os.path.join(os.pardir, 'data')
    X_train, y_train = load_train_set(path_to_data)
    X_test = load_test_set(path_to_data)

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    X_train = X_train.drop(correlated_cols, axis=1)
    X_test = X_test.drop(correlated_cols, axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=cli_args.valid_split,
        random_state=cli_args.seed)

    linear_regression.fit(X_train, y_train)

    train_predictions = linear_regression.predict(X_train)
    train_score = r2_score(y_train, train_predictions)

    valid_predictions = linear_regression.predict(X_valid)
    valid_score = r2_score(y_valid, valid_predictions)

    predictions = linear_regression.predict(X_test)
    log_predictions(experiment, predictions, X_test.index)

    # Define experiment metrics for CometML to be logged to the project under
    # the experiment.
    metrics = {
        # Conversion necessary as score is in NumPy specific type that isn't
        # nicely serialized.
        'train_r2_score': float(train_score),
        'valid_r2_score': float(valid_score)}
    log_metrics(experiment, metrics)


if "__main__" == __name__:
    main()
