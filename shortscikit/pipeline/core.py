import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Union, Tuple

import numpy as np  # linear algebra
import pandas as pd
import scipy.sparse as sparse
from sklearn.base import TransformerMixin
from sklearn.externals import joblib
from sklearn.preprocessing import FunctionTransformer

RANDOM_STATE = 10

n_cpus = os.cpu_count()


class DataAdaptor(object, metaclass=ABCMeta):

    def __init__(self, x_like):
        self.x_like = x_like

    @abstractmethod
    def apply_transform(self, func, **kwargs):
        return self.x_like

    @abstractmethod
    def apply_fit(self, func):
        pass

    def save(self, x_like, save_dir, format=None, only_feature=None):
        joblib.dump(x_like, str(save_dir.joinpath("x_like.pickle")))


class NdArrayAdaptor(DataAdaptor):

    def __init__(self, x_like):
        super().__init__(x_like)

    def apply_transform(self, func, **kwargs):
        return func(self.x_like)

    def apply_fit(self, func):
        return func(self.x_like)


class PandasDataFrameAdaptor(DataAdaptor):

    def __init__(self, x_like, feature_name):
        super().__init__(x_like)
        self.feature_name = feature_name

    def apply_fit(self, func):
        func(self.x_like[self.feature_name].values.reshape((self.x_like.shape[0], -1)))

    def apply_transform(self, func, overwrite=True, suffix="", **kwargs):
        # TODO support multiple columns
        transformed_sequence = func(self.x_like[self.feature_name].values.reshape((self.x_like.shape[0], -1)))
        if overwrite:
            self.x_like[self.feature_name] = transformed_sequence
        else:
            self.x_like[suffix + "_" + self.feature_name] = transformed_sequence
        return self.x_like

    def save(self, x_like, save_dir, format=None, only_feature=None):
        save_path = save_dir.joinpath("feature." + format)
        if only_feature:
            save_func = getattr(x_like[[only_feature]], "to_{}".format(format))
        else:
            save_func = getattr(x_like, "to_{}".format(format))
        return save_func(save_path)


def create_data_adaptor(x_like: [np.ndarray], feature_name=None):
    if isinstance(x_like, np.ndarray):
        return NdArrayAdaptor(x_like)
    elif isinstance(x_like, pd.DataFrame):
        return PandasDataFrameAdaptor(x_like, feature_name)
    else:
        raise ValueError("{} is not supported".format(type(x_like)))


class TransformStage(object, metaclass=ABCMeta):

    def __init__(self, transformer: TransformerMixin = None, transformer_path=None, feature_name=None,
                 new_feature_suffix=None):
        if transformer is None and transformer_path is None:
            raise ValueError("should pass either transformer or transformer_path")
        self.transformer = transformer
        self._current_part = 0
        if transformer_path:
            self.transformer = joblib.load(transformer_path)
        self.feature_name = feature_name
        self.new_feature_suffix = new_feature_suffix

    def fit_transform(self, x_like: Union[np.ndarray, sparse.spmatrix], save_dir: Path = None, n_jobs=None,
                      save_format=None, save_only_feature=False):
        return self.fit(x_like, save_dir, n_jobs).transform(x_like, save_dir, n_jobs, save_format,
                                                            save_only_feature)

    def fit(self, x_like: Union[np.ndarray, sparse.spmatrix], save_dir: Path = None, n_jobs=None):
        adaptor = create_data_adaptor(x_like, feature_name=self.feature_name)
        # if not sparse.issparse(x_like) and not isinstance(x_like, np.ndarray):
        #     raise ValueError("not implemented for type {}".format(type(x_like)))

        adaptor.apply_fit(self.transformer.fit)
        # TODO enable change name of pickle
        if save_dir:
            joblib.dump(self.transformer, str(save_dir.joinpath("transformer.pickle")))
        return self

    def transform(self, x_like: Union[np.ndarray, sparse.spmatrix], save_dir: Path = None, n_jobs=None,
                  save_format=None, save_only_feature=False):
        adaptor = create_data_adaptor(x_like, feature_name=self.feature_name)
        # if not sparse.issparse(x_like) and not isinstance(x_like, np.ndarray):
        #     raise ValueError("not implemented for type {}".format(type(x_like)))

        x_like = adaptor.apply_transform(self.transformer.transform, overwrite=not self.new_feature_suffix,
                                         suffix=self.new_feature_suffix, feature_name=self.feature_name)

        # TODO enable changer type of save and name
        if save_dir:
            if save_only_feature:
                save_column = self.feature_name
                if self.new_feature_suffix:
                    save_column = self.new_feature_suffix + "_" + save_column
            else:
                save_column = None
            adaptor.save(x_like, save_dir, format=save_format, only_feature=save_column)
        return x_like


class WindowTransformStage(TransformStage):

    def __init__(self, window_size, step_size=0, axis=1, transformer=None, transformer_path=None,
                 feature_name=None, new_feature_suffix=None):
        super().__init__(transformer, transformer_path=transformer_path, feature_name=feature_name,
                         new_feature_suffix=new_feature_suffix)
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

    def transform(self, x_like: Union[np.ndarray], save_dir=None, n_jobs=None,
                  save_format=None, save_only_feature=False):
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


class TransformPipeline(TransformStage):

    def __init__(self, *stages: Tuple[str, TransformStage], transformer_path=None,
                 feature_name=None,
                 new_feature_suffix=None):
        # TODO shouldn't pass dummy transformer
        dummy_transfomer = FunctionTransformer(lambda x: x)
        super().__init__(dummy_transfomer, transformer_path, feature_name, new_feature_suffix)

        # TODO support graph of stages
        if stages is not None:
            self.stages: OrderedDict[str, TransformStage] = OrderedDict(stages)
        else:
            self.stages: OrderedDict[str, TransformStage] = OrderedDict()

    def append(self, name, stage: TransformStage):
        self.stages[name] = stage

    def fit(self, x_like: Union[np.ndarray, sparse.spmatrix], save_dir: Path = None, n_jobs=None):
        # TODO support fit after transform stage
        for name, stage in self.stages.items():
            stage_save_path = self._to_stage_savepath(name, save_dir)
            stage.fit(x_like, stage_save_path, n_jobs)
        return self

    def _to_stage_savepath(self, name, save_dir):
        stage_save_path = save_dir.joinpath(name) if save_dir else None
        if stage_save_path:
            stage_save_path.mkdir(exist_ok=True, parents=True)
        return stage_save_path

    def transform(self, x_like: Union[np.ndarray, sparse.spmatrix], save_dir: Path = None, n_jobs=None,
                  save_format=None, save_only_feature=False):
        for name, stage in self.stages.items():
            stage_save_path = self._to_stage_savepath(name, save_dir)
            x_like = stage.transform(x_like, stage_save_path, n_jobs, save_format=save_format,
                                     save_only_feature=save_only_feature)
        return x_like
