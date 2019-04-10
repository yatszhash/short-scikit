from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from nose2.tools import such
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from wrapper.core import WindowTransformerWrapper, TransformerWrapper


class DummySumFitter(FunctionTransformer):

    def __init__(self, inverse_func=None, accept_sparse=False, pass_y='deprecated',
                 check_inverse=True, kw_args=None, inv_kw_args=None):
        super().__init__(lambda x: x,
                         inverse_func, False, accept_sparse, pass_y, check_inverse, kw_args, inv_kw_args)
        self.total = None

    def fit(self, X, y=None):
        self.total = np.sum(X, axis=0)


with such.A('transformer wrapper test') as it:
    @it.should('save output and pickle')
    def test():
        with TemporaryDirectory() as temp_dir:
            sut = TransformerWrapper(transformer=StandardScaler())
            sut.fit_transform(np.arange(10).reshape((-1, 1)).astype("float"), save_dir=Path(temp_dir))
            it.assertTrue(Path(temp_dir).joinpath("transformer.pickle").exists())
            it.assertTrue(Path(temp_dir).joinpath("x_like.pickle").exists())


    it.createTests(globals())

with such.A('window transformer wrapper test') as it:
    @it.should('transform with window')
    def test():
        x = np.arange(100).reshape((1, -1)).repeat(50, axis=0)

        sut = WindowTransformerWrapper(window_size=4, step_size=2,
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

        sut = WindowTransformerWrapper(window_size=2, step_size=2,
                                       transformer=DummySumFitter())
        x = sut.fit(x)
        np.testing.assert_array_equal(sut.transformer.total, np.array([20, 40]))


    it.createTests(globals())

if __name__ == '__main__':
    import nose2

    nose2.main()
