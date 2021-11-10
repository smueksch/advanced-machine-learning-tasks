import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from typing import Sequence, Union


def get_mean(xs: Sequence[Union[int, float]]) -> float:
    '''
    Get mean of given list of ints/floats.

    Small utility function to compute the mean of a given sequence, for instance
    a NumPy array, and return it as a float value. Reduces code duplication.

    Args:
        xs (Sequence[Union[int, float]]): Sequence of numbers to compute mean
            for.

    Returns:
        float: Mean of given sequence of numbers.
    '''
    return float(np.mean(xs))


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
