# Control experiment with one-hot encoding and no normalization.

# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Need to add amlutils path so Python can find it.
import sys
sys.path.append(os.path.join(os.pardir))
from amlutils.task1.loading import load_train_set, load_test_set
from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_cli
from amlutils.experiment import log_parameters, log_predictions, log_metrics


def main():
    cli_args = get_cli_arguments()
    experiment = build_experiment_from_cli(cli_args)

    # Define experiment parameters for CometML to be logged to the project under
    # the experiment.
    params = {
        'model': 'linear regression',
        }
    log_parameters(experiment, params)

    linear_regression = LinearRegression()

    path_to_data = 'data'
    X_train, y_train = load_train_set(path_to_data)
    X_test = load_test_set(path_to_data)

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    linear_regression.fit(X_train, y_train)
    train_predictions = linear_regression.predict(X_train)
    score = r2_score(y_train, train_predictions)

    predictions = linear_regression.predict(X_test)
    log_predictions(experiment, predictions, X_test.index)

    # Define experiment metrics for CometML to be logged to the project under
    # the experiment.
    metrics = {
        # Conversion necessary as score is in NumPy specific type that isn't
        # nicely serialized.
        "r2_score": float(score),
        }
    log_metrics(experiment, metrics)


if "__main__" == __name__:
    main()
