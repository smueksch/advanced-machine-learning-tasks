import os
import pandas as pd

from typing import Tuple


def load_train_set(path_to_data: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Load training set from given data folder.

    Args:
        path_to_data (str): Path to data folder containing X_train.csv and
            y_train.csv.

    Returns:
        X_train, y_train: Dataframes holding the training data (X_train) and
        the training labels (y_train).
    '''
    X_train = pd.read_csv(
        os.path.join(path_to_data, 'X_train.csv'),
        index_col='id')
    y_train = pd.read_csv(
        os.path.join(path_to_data, 'y_train.csv'),
        index_col='id')
    return (X_train, y_train)


def load_test_set(path_to_data: str) -> pd.DataFrame:
    '''
    Load test set from given CSV file.

    Args:
        path_to_data (str): Path to data folder containing X_test.csv.

    Returns:
        X_test: Dataframe holding the test data (X_train).
    '''
    return pd.read_csv(
        os.path.join(path_to_data, 'X_test.csv'),
        index_col='id')
