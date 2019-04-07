import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from feature_extracttion.shape import to_2d_array


class LimitMax(FunctionTransformer):
    def __init__(self, upper_limit,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        self.upper_limit = upper_limit
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X, y=None):
        return np.where(X >= self.upper_limit, X, self.upper_limit).reshape((-1, 1))


class LogTransformer(FunctionTransformer):

    def __init__(self,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        super().__init__(LogTransformer.to_log, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    @staticmethod
    def to_log(x):
        input_ = x
        # input_ = input_
        return np.log1p(input_)


class TahnEstimators(BaseEstimator, TransformerMixin):
    """
    refer
    https://stats.stackexchange.com/questions/7757/data-normalization-and-standardization-in-neural-networks
    https://stackoverflow.com/questions/43061120/tanh-estimator-normalization-in-python
    """

    def __init__(self):
        self.std_ = None
        self.mean_ = None
        self.n_seen_samples = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)
        return self

    def transform(self, X, copy=None):
        return 0.5 * (np.tanh(0.01 * (to_2d_array(X) - self.mean_) / self.std_) + 1)
