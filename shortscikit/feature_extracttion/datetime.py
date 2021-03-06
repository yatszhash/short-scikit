import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


class WeekDayTransformer(FunctionTransformer):
    def __init__(self, offset=0, kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        self.offset = offset
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X, y=None):
        return np.cos(pd.Series(X).dt.dayofweek.values / 7 * 2 * np.pi + self.offset).astype("float32").reshape((-1, 1))


class MonthTransformer(FunctionTransformer):
    def __init__(self, offset=0, kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        self.offset = offset
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X, y=None):
        return np.cos(pd.Series(X).dt.month.values / 12 * 2 * np.pi + self.offset).astype("float32").reshape((-1, 1))


class DayTransformer(FunctionTransformer):
    def __init__(self, offset=0, kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        self.offset = offset
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X, y=None):
        # TODO change the whole number of days based on month
        return np.cos(pd.Series(X).dt.day.values / 31 * 2 * np.pi + self.offset).astype("float32").reshape((-1, 1))
