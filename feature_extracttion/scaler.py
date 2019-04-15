import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from feature_extracttion.shape import to_2d_array


class LimitMax(BaseEstimator, TransformerMixin):
    def __init__(self, upper_limit,
                 kw_args=None, inv_kw_args=None):
        self.upper_limit = upper_limit
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.where(X >= self.upper_limit, X, self.upper_limit).reshape((-1, 1))


class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, kw_args=None, inv_kw_args=None):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y='deprecated'):
        return np.log1p(X)


class TahnEstimators(BaseEstimator, TransformerMixin):
    """
    refer
    https://stats.stackexchange.com/questions/7757/data-normalization-and-standardization-in-neural-networks
    https://stackoverflow.com/questions/43061120/tanh-estimator-normalization-in-python
    """

    def __init__(self):
        self.std_ = None
        self.mean_ = None
        super().__init__()

    def fit(self, X, y=None):
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)
        return self

    def transform(self, X):
        return 0.5 * (np.tanh(0.01 * (to_2d_array(X) - self.mean_) / self.std_) + 1)
