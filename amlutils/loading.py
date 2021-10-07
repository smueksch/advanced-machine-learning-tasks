import pandas as pd


def load_train_set(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Load training set from given CSV file.

    Args:
        path (str): Path to train.csv.

    Returns:
        X_train, y_train: Dataframes holding the training data (X_train) and
        the training labels (y_train).
    '''
    X_train = pd.read_csv(path, index_col='Id')
    y_train = X_train['y'].copy()
    del X_train['y']
    return (X_train, y_train)


def load_test_set(path: str) -> pd.DataFrame:
    '''
    Load test set from given CSV file.

    Args:
        path (str): Path to test.csv.

    Returns:
        X_test: Dataframe holding the test data (X_train).
    '''
    return pd.read_csv(path, index_col='Id')
