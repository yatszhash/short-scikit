import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PercentileSummaryTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, axis=0, should_flat=True,
                 kw_args=None, inv_kw_args=None):
        self.axis = axis
        self.should_flat = should_flat
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mean = X.mean(axis=self.axis)
        std = X.std(axis=self.axis)
        std_top = mean + std
        std_bot = mean - std
        percentile_calc = np.percentile(X, [0, 1, 25, 50, 75, 99, 100], axis=self.axis)
        max_range = percentile_calc[-1] - percentile_calc[0]  # this is the amplitude of the chunk
        relative_percentile = percentile_calc - mean  # maybe it could heap to understand the asymmetry
        # now, we just add all the features to new_ts and convert it to np.array
        if not self.axis:
            summary = np.hstack(
                [np.asarray([mean, std, std_top, std_bot, max_range]), percentile_calc, relative_percentile])
        elif self.axis == 1:
            summary = np.vstack(
                [np.vstack([mean, std, std_top, std_bot, max_range]), percentile_calc, relative_percentile]).transpose()
        else:
            raise ValueError("not implemented")
        if self.should_flat:
            return summary.flatten()
        return summary


class SummaryTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, axis=0,
                 kw_args=None, inv_kw_args=None):
        self.axis = axis
        super().__init__()
        

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        avgs = np.mean(X, axis=self.axis)
        stds = np.std(X, axis=self.axis)
        maxs = np.max(X, axis=self.axis)
        mins = np.min(X, axis=self.axis)
        medians = np.median(X, axis=self.axis)
        return np.hstack([avgs, stds, maxs, mins, medians])


class AverageTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 kw_args=None, inv_kw_args=None):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        avgs = np.mean(X, axis=1)
        return avgs.reshape((-1, 1))
