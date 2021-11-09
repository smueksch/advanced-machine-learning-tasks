import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, List


class AbsoluteCorrelationSelector(BaseEstimator, TransformerMixin):
    '''
    Transform a dataset by selecting highly absolute correlated features to label.

    Select features from a dataset by only keeping those that have absolute
    correlation coefficient above a given threshold.
    '''

    def __init__(self, min_abs_correlation: float,
                 correlation_method='pearson'):
        '''
        Initialize selector.

        Args:
            min_abs_correlation (float): Minimum absolute correlation a feature
                needs to have with the label in order to be selected.
            correlation_method (str): Method with which to compute correlation
                of a feature with the label. Defaults to Pearson correlation.

                Options are:

                    * 'pearson' - use Pearson correlation coefficient
                    * 'spearman' - use Spearman rank correlation
                    * 'pearson_spearman' - use both Pearson correlation
                        coefficient and Spearman rank correlation and take
                        the intersection of the selected features
        '''
        self.min_abs_correlation = min_abs_correlation
        self.correlation_method = correlation_method
        self.high_correlation_features = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        '''
        Fit selector to given dataset.

        Computes which features of given X have absolute correlation higher
        than the minimum absolute correlation. The correlation method
        specified at initialization is used to compute correlation.

        Args:
            X (pd.DataFrame): Matrix containing features as columns.
            y (pd.DataFrame): Label to compute correlation with.

        Return:
            self: Reference to itself so call to transform can be chained.
        '''
        X = self.__to_dataframe(X)
        if 'pearson' == self.correlation_method or 'spearman' == self.correlation_method:
            self.high_correlation_features = self.__get_features(
                X,
                y,
                self.min_abs_correlation,
                self.correlation_method
                )
        elif 'pearson_spearman' == self.correlation_method:
            high_pearson_correlation_features = self.__get_features(
                X,
                y,
                self.min_abs_correlation,
                'pearson'
                )

            high_spearman_correlation_features = self.__get_features(
                X,
                y,
                self.min_abs_correlation,
                'spearman'
                )

            self.high_correlation_features = list(
                set(high_pearson_correlation_features) &
                set(high_spearman_correlation_features)
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Select features with high absolute correlation with label.

        Args:
            X (pd.DataFrame): Matrix containing features that should be
                selected based on high absolute correlation.

        Returs:
            pd.DataFrame: Transformed X where only the columns with high
                absolute correlation with label y have been preserved.
        '''
        X = self.__to_dataframe(X)
        return X[self.high_correlation_features]

    def __get_features(
            self, X: pd.DataFrame, y: pd.DataFrame, min_abs_correlation: float,
            correlation_method: str) -> List[str]:
        '''
        Return X column labels with abs. correlation with y greater than threshold.

        Args:
            X (pd.DataFrame): Data matrix, columns represent features.
            y (pd.DataFrame): Label matrix.
            min_abs_correlation (float): Minimum absolute correlation a feature
                needs to have with the label in order to be selected.
            correlation_method (str): Method with which to compute correlation
                of a feature with the label. Defaults to Pearson correlation.

                Options are:

                    * 'pearson' - use Pearson correlation coefficient
                    * 'spearman' - use Spearman rank correlation
                    * 'pearson_spearman' - use both Pearson correlation
                        coefficient and Spearman rank correlation and take
                            the intersection of the selected features

        Returns:
            list of str: Names of columns in X that have absolute correlation 
                with y strictly greater than threshold.
        '''
        # If used in an sklearn Pipeline, X will be a NumPy array, so need
        # to convert to DataFrame.
        y_correlation = X.corrwith(y['y'], method=correlation_method).fillna(0)
        y_abs_correlation = np.abs(y_correlation)
        high_abs_corr_features = y_abs_correlation[y_abs_correlation >
                                                   min_abs_correlation]
        return list(high_abs_corr_features.index.values)

    @staticmethod
    def __to_dataframe(X: Any) -> pd.DataFrame:
        '''
        Convert given argument to DataFrame if necessary.
        '''
        if type(X) == np.ndarray:
            return pd.DataFrame(
                data=X,
                columns=[f'x{idx}' for idx in range(X.shape[1])])
        else:
            return X
