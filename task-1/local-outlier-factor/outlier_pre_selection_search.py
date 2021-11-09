# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import cross_validate

# Need to add amlutils path so Python can find it.
import sys
sys.path.append(os.path.join(os.pardir, os.pardir))
from amlutils.task1.loading import load_train_set
from amlutils.cliargs import get_cli_arguments
from amlutils.experiment import build_experiment_from_cli
from amlutils.experiment import log_parameters, log_cv_results, log_metrics
from amlutils.transformers import AbsoluteCorrelationSelector


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

    # Manual model pipeline.
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)

    selector = AbsoluteCorrelationSelector(
        min_abs_correlation=0.1, correlation_method='pearson_spearman')
    X_train = selector.fit_transform(X_train, y_train)

    # Run grid search over hyperparameters.
    parameter_grid = {
        'outlier__n_neighbors': [1, 5, 10, 20, 50, 100, 500, 1000, 2000],
        'outlier__metric': ['l1', 'l2', 'chebyshev'],
        'outlier__contamination': ['auto', 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]}

    cv_results = pd.DataFrame(
        columns=['n_samples', 'n_features', 'n_neighbors', 'metric',
                 'contamination', 'train_split_0_score', 'train_split_1_score',
                 'train_split_2_score', 'train_split_3_score',
                 'train_split_4_score', 'valid_split_0_score',
                 'valid_split_1_score', 'valid_split_2_score',
                 'valid_split_3_score', 'valid_split_4_score',
                 'mean_train_score', 'mean_valid_score'])

    best_n_neighbors = 0
    best_metric = ''
    best_contamination = ''
    best_train_score = 0.0
    best_valid_score = -1000.0
    for n_neighbors in parameter_grid['outlier__n_neighbors']:
        for metric in parameter_grid['outlier__metric']:
            for contamination in parameter_grid['outlier__contamination']:
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors, metric=metric,
                    contamination=contamination, n_jobs=-1)

                outlier_pred = lof.fit_predict(X_train, y_train)

                # Remove outliers based on predictions of LocalOutlierFactor.
                X_train = X_train[outlier_pred == 1]
                y_train = y_train[outlier_pred == 1]

                # Standard normalize the data before training the actual model.
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)

                try:
                    cv_scores = cross_validate(
                        GradientBoostingRegressor(
                            learning_rate=0.01,
                            n_estimators=2600, max_depth=4,
                            subsample=0.45, random_state=cli_args.seed),
                        X_train, pd.Series.ravel(y_train), scoring='r2',
                        cv=5, n_jobs=-1, verbose=1,
                        return_train_score=True)
                except ValueError:
                    # Occurs if number of samples is fewer than number of
                    # splits, which may be due to aggressive outlier
                    # classification. In that case, report all unknown metrics
                    # as NaN.
                    cv_scores = {
                        'train_score': [np.nan for x in range(5)],
                        'test_score': [np.nan for x in range(5)]}

                train_score = float(np.mean(cv_scores['train_score']))
                valid_score = float(np.mean(cv_scores['test_score']))

                print(f'n_samples={X_train.shape[0]},' +
                      f'n_features={X_train.shape[1]},' +
                      f'n_neighbors={n_neighbors},metric={metric},' +
                      f'contamination={contamination},' +
                      f'train_score={train_score},' +
                      f'valid_score={valid_score}')

                row = pd.Series(
                    [X_train.shape[0],
                     X_train.shape[1],
                     n_neighbors, metric, contamination] +
                    list(cv_scores['train_score']) +
                    list(cv_scores['test_score']) +
                    [train_score, valid_score],
                    index=cv_results.columns)

                cv_results = cv_results.append(row, ignore_index=True)

                if best_valid_score < valid_score:
                    best_n_neighbors = n_neighbors
                    best_metric = metric
                    best_contamination = contamination
                    best_train_score = train_score
                    best_valid_score = valid_score

    # Save cross validation results.
    log_cv_results(experiment, cv_results)

    # Define experiment metrics for CometML to be logged to the project under
    # the experiment.
    metrics = {
        'best_train_r2_score': best_train_score,
        'best_valid_r2_score': best_valid_score,
        'best_n_neighbors': best_n_neighbors,
        'best_metric': best_metric,
        'best_contamination': best_contamination}
    log_metrics(experiment, metrics)


if "__main__" == __name__:
    main()
