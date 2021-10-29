# Control experiment with one-hot encoding and no normalization.

# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Need to add amlutils path so Python can find it.
import sys
sys.path.append(os.path.join(os.pardir))
from amlutils.task1.loading import load_train_set, load_test_set
from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_cli
from amlutils.experiment import log_parameters, log_predictions, log_metrics

same_value_cols = ['x1',
                   'x2',
                   'x6',
                   'x9',
                   'x16',
                   'x34',
                   'x50',
                   'x78',
                   'x82',
                   'x83',
                   'x145',
                   'x153',
                   'x155',
                   'x166',
                   'x193',
                   'x195',
                   'x206',
                   'x232',
                   'x256',
                   'x263',
                   'x266',
                   'x269',
                   'x279',
                   'x297',
                   'x317',
                   'x335',
                   'x339',
                   'x363',
                   'x364',
                   'x374',
                   'x378',
                   'x379',
                   'x399',
                   'x422',
                   'x449',
                   'x454',
                   'x480',
                   'x481',
                   'x500',
                   'x505',
                   'x527',
                   'x586',
                   'x588',
                   'x609',
                   'x627',
                   'x629',
                   'x639',
                   'x645',
                   'x664',
                   'x679',
                   'x684',
                   'x700',
                   'x719',
                   'x721',
                   'x729',
                   'x735',
                   'x747',
                   'x755',
                   'x762',
                   'x778',
                   'x811']


def main():
    cli_args = get_cli_arguments()
    experiment = build_experiment_from_cli(cli_args)
    experiment.add_tag('task-1')

    # Define experiment parameters for CometML to be logged to the project under
    # the experiment.
    params = {
        'valid_split': cli_args.valid_split,
        'random_seed': cli_args.seed,
        'removed_cols': same_value_cols}
    log_parameters(experiment, params)

    gbdt = GradientBoostingRegressor()

    path_to_data = os.path.join(os.pardir, 'data')
    X_train, y_train = load_train_set(path_to_data)
    X_test = load_test_set(path_to_data)

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    X_train = X_train.drop(same_value_cols, axis=1)
    X_test = X_test.drop(same_value_cols, axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=cli_args.valid_split,
        random_state=cli_args.seed)

    gbdt.fit(X_train, y_train)

    train_predictions = gbdt.predict(X_train)
    train_score = r2_score(y_train, train_predictions)

    valid_predictions = gbdt.predict(X_valid)
    valid_score = r2_score(y_valid, valid_predictions)

    predictions = gbdt.predict(X_test)
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
