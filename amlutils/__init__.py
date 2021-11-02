import numpy as np
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
