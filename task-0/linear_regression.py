# Control experiment with one-hot encoding and no normalization.

# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Need to add amlutils path so Python can find it.
import sys
sys.path.append(os.path.join(os.pardir))
from amlutils.loading import load_train_set, load_test_set
from amlutils.logging import log_parameters, log_predictions, log_metrics


def main():
    experiment = Experiment(
        project_name='aml-tasks',
        workspace='smueksch',
        # Prevent Comet.ml from dumping all sorts of parameter info into the
        # experiment log.
        auto_param_logging=False,
        # Set to True if you want to disable Comet.ml interaction.
        disabled=False,
        )
    experiment.set_name('Linear Regression')

    # Define experiment parameters for CometML to be logged to the project under
    # the experiment.
    params = {
        'model': 'linear regression',
        }
    log_parameters(experiment, params)

    linear_regression = LinearRegression()

    X_train, y_train = load_train_set(os.path.join('data', 'train.csv'))
    X_test = load_test_set(os.path.join('data', 'test.csv'))

    linear_regression.fit(X_train, y_train)
    train_predictions = linear_regression.predict(X_train)
    rmse = mean_squared_error(y_train, train_predictions) ** 0.5

    predictions = linear_regression.predict(X_test)
    log_predictions(experiment, predictions,
                    X_test.index, "test-predictions.csv")

    # Define experiment metrics for CometML to be logged to the project under
    # the experiment.
    metrics = {
        # Conversion necessary as rmse is in NumPy specific type that isn't
        # nicely serialized.
        "rmse": float(rmse),
        }
    log_metrics(experiment, metrics)


if "__main__" == __name__:
    main()
