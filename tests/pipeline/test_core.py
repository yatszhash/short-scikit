from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from nose2.tools import such
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from shortscikit.pipeline.core import WindowTransformStage, TransformStage, TransformPipeline


class DummySumFitter(FunctionTransformer):

    def __init__(self, inverse_func=None, accept_sparse=False, pass_y='deprecated',
                 check_inverse=True, kw_args=None, inv_kw_args=None):
        super().__init__(self.f,
                         inverse_func, False, accept_sparse, pass_y, check_inverse, kw_args, inv_kw_args)
        self.total = None

    def fit(self, X, y=None):
        self.total = np.sum(X, axis=0)

    def f(self, x):
        return x


with such.A('transformer stage test') as it:
    @it.should('save output and pickle')
    def test_with_array(case):
        with TemporaryDirectory() as temp_dir:
            sut = TransformStage(transformer=StandardScaler())
            sut.fit_transform(np.arange(10).reshape((-1, 1)).astype("float"), save_dir=Path(temp_dir))
            it.assertTrue(Path(temp_dir).joinpath("transformer.pickle").exists())
            it.assertTrue(Path(temp_dir).joinpath("x_like.pickle").exists())


    with it.having('prepared dataframe'):
        @it.has_test_setup
        def setup():
            it.df = pd.DataFrame(np.arange(1, 6).repeat(2, axis=0).reshape((-1, 2)), columns=["a", "b"])


        @it.should('have overwritten transformed column')
        def test_overwrite_column(case):
            sut = TransformStage(transformer=StandardScaler(), feature_name='a')
            actual = sut.fit_transform(it.df)
            case.assertEqual(2, actual.shape[1])
            case.assertEqual(0, actual.loc[2, 'a'])


        @it.should('have new transformed column')
        def test_new_column(case):
            sut = TransformStage(transformer=StandardScaler(), feature_name='a', new_feature_suffix='new')
            actual = sut.fit_transform(it.df)
            case.assertEqual(3, actual.shape[1])
            case.assertEqual(0, actual.loc[2, 'new_a'])


        @it.should('save transformed df with csv format')
        def test_save(case):
            with TemporaryDirectory() as temp_dir:
                sut = TransformStage(transformer=StandardScaler(), feature_name='a')
                actual = sut.fit_transform(it.df, save_dir=Path(temp_dir),
                                           save_format="csv")
                df = pd.read_csv(Path(temp_dir).joinpath("feature.csv"))
                case.assertEqual(3, df.shape[1])
                case.assertEqual(0, df.loc[2, 'a'])


        @it.should('save transformed column df with feather')
        def test_save(case):
            with TemporaryDirectory() as temp_dir:
                sut = TransformStage(transformer=StandardScaler(), feature_name='a')
                actual = sut.fit_transform(it.df, save_dir=Path(temp_dir),
                                           save_format="feather", save_only_feature=True)
                df = pd.read_feather(Path(temp_dir).joinpath("feature.feather"))
                case.assertEqual(1, df.shape[1])
                case.assertEqual(0, df.loc[2, 'a'])

    it.createTests(globals())

with such.A('window transformer stage test') as it:
    @it.should('transform with window')
    def test():
        x = np.arange(100).reshape((1, -1)).repeat(50, axis=0)

        sut = WindowTransformStage(window_size=4, step_size=2,
                                   transformer=FunctionTransformer(lambda z: np.sum(z, axis=-1),
                                                                   validate=False))
        x = sut.transform(x)
        it.assertEqual(x.shape[0], 50)
        it.assertEqual(x.shape[1], 49)
        it.assertEqual(x[0][0], 0 + 1 + 2 + 3)
        it.assertEqual(x[0][-1], 99 + 98 + 97 + 96)


    it.createTests(globals())


    @it.should('fit with window')
    def test():
        x = np.arange(4).reshape((1, -1)).repeat(10, axis=0)

        sut = WindowTransformStage(window_size=2, step_size=2, transformer=DummySumFitter())
        x = sut.fit(x)
        np.testing.assert_array_equal(sut.transformer.total, np.array([20, 40]))


    it.createTests(globals())

with such.A('test pipeline') as it:
    with it.having('prepared dataframe'):
        @it.has_test_setup
        def setup():
            it.df = pd.DataFrame(np.arange(1, 6).repeat(2, axis=0).reshape((-1, 2)), columns=["a", "b"])


        @it.should('pipeline fit_transform ')
        def test(case):
            with TemporaryDirectory() as temp_dir:
                sut = TransformPipeline()
                sut.append("standard_scaler", TransformStage(transformer=StandardScaler(), feature_name='a'))
                sut.append("dummy_sum", TransformStage(transformer=DummySumFitter(), feature_name="a"))

                actual = sut.fit_transform(it.df.copy(), save_dir=Path(temp_dir), save_format="csv")

                case.assertEqual(2, actual.shape[1])
                case.assertEqual(0, actual.loc[2, 'a'])
                np.testing.assert_array_equal(sut.stages.get("dummy_sum").transformer.total, np.array([15]))

                # check saved
                df = pd.read_csv(Path(temp_dir).joinpath("standard_scaler").joinpath("feature.csv"))
                case.assertEqual(3, df.shape[1])
                case.assertEqual(0, df.loc[2, 'a'])


        @it.should('pipeline fit_transform with previous transformed feature')
        def test(case):
            with TemporaryDirectory() as temp_dir:
                sut = TransformPipeline(fit_transformed=True)
                sut.append("standard_scaler", TransformStage(transformer=StandardScaler(), feature_name='a'))
                sut.append("dummy_sum", TransformStage(transformer=DummySumFitter(), feature_name="a"))

                actual = sut.fit_transform(it.df.copy(), save_dir=Path(temp_dir), save_format="csv")

                case.assertEqual(2, actual.shape[1])
                case.assertEqual(0, actual.loc[2, 'a'])
                np.testing.assert_array_equal(sut.stages.get("dummy_sum").transformer.total, np.array([0]))

                # check saved
                df = pd.read_csv(Path(temp_dir).joinpath("standard_scaler").joinpath("feature.csv"))
                case.assertEqual(3, df.shape[1])
                case.assertEqual(0, df.loc[2, 'a'])


    it.createTests(globals())

if __name__ == '__main__':
    import nose2

    nose2.main()
