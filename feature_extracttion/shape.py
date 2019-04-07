from sklearn.preprocessing import FunctionTransformer


class ReshapeInto2d(FunctionTransformer):
    def __init__(self,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        super().__init__(to_2d_array, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)


def to_2d_array(x):
    array = x
    if isinstance(x, pd.Series):
        array = array.values
    if len(array.shape) == 1:
        array = array.reshape((-1, 1))
    return array


class RavelTransformer(FunctionTransformer):

    def __init__(self,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        super().__init__(RavelTransformer.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    @staticmethod
    def f(x):
        return x.ravel()
