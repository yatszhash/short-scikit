import os
from abc import ABCMeta
from pathlib import Path
from typing import Union

import numpy as np  # linear algebra
import scipy.sparse as sparse
from sklearn.base import TransformerMixin
from sklearn.externals import joblib

RANDOM_STATE = 10

n_cpus = os.cpu_count()


class TransformStage(object, metaclass=ABCMeta):

    def __init__(self, transformer: TransformerMixin = None, transformer_path=None):
        if transformer is None and transformer_path is None:
            raise ValueError("should pass either transformer or transformer_path")
        self.transformer = transformer
        self._current_part = 0
        if transformer_path:
            self.transformer = joblib.load(transformer_path)

    def fit_transform(self, x_like: Union[np.ndarray, sparse.spmatrix], save_dir: Path = None, n_jobs=None):
        return self.fit(x_like, save_dir, n_jobs).transform(x_like, save_dir, n_jobs)

    def fit(self, x_like: Union[np.ndarray, sparse.spmatrix], save_dir: Path = None, n_jobs=None):

        if not sparse.issparse(x_like) and not isinstance(x_like, np.ndarray):
            raise ValueError("not implemented for type {}".format(type(x_like)))

        self.transformer.fit(x_like)
        # TODO enable change name of pickle
        if save_dir:
            joblib.dump(self.transformer, str(save_dir.joinpath("transformer.pickle")))
        return self

    def transform(self, x_like: Union[np.ndarray, sparse.spmatrix], save_dir: Path = None, n_jobs=None):
        if not sparse.issparse(x_like) and not isinstance(x_like, np.ndarray):
            raise ValueError("not implemented for type {}".format(type(x_like)))

        x_like = self.transformer.transform(x_like)

        # TODO enable changer type of save and name
        if save_dir:
            joblib.dump(x_like, str(save_dir.joinpath("x_like.pickle")))
        return x_like


class WindowTransformStage(TransformStage):

    def __init__(self, window_size, step_size, transformer: TransformerMixin = None, transformer_path=None,
                 axis=1):
        super().__init__(transformer, transformer_path)
        self.window_size = window_size
        self.step_size = step_size
        if axis <= 0:
            raise ValueError("axis should not be batch axis or negative")
        self.axis = axis
        # TODO padding

    def fit(self, x_like: Union[np.ndarray], save_dir=None, n_jobs=None):
        # TODO support sparse matrix
        x_like = self.to_windows(x_like)
        x_like = self.stack(x_like)
        return super().fit(x_like, save_dir, n_jobs)

    def stack(self, x_like):
        stacked_shape = list(x_like.shape)
        del stacked_shape[self.axis]
        stacked_shape[0] = -1
        x_like = x_like.reshape(stacked_shape)
        return x_like

    def transform(self, x_like: Union[np.ndarray], save_dir=None, n_jobs=None):
        x_like = self.to_windows(x_like)
        transformed_shape = list(x_like.shape)

        # TODO support transformer with shape change
        x_like = self.stack(x_like)
        x_like = super().transform(x_like, save_dir, n_jobs)
        transformed_shape[-1] = -1
        return x_like.reshape(transformed_shape)

    def to_windows(self, array_like):
        # TODO extract as transformer
        n_windows = (array_like.shape[self.axis] - self.window_size + 1) // self.step_size
        n_windows += int(not bool((array_like.shape[self.axis] - self.step_size * n_windows)
                                  % self.window_size))
        window_indices = [list(range(i * self.step_size, i * self.step_size + self.window_size))
                          for i in range(n_windows)]
        return np.take(array_like, window_indices, axis=self.axis)
