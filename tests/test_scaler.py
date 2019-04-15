import numpy as np
from scipy import signal
import pywt
from nose2.tools  import such
from feature_extracttion.scaler import LimitMax, LogTransformer, TahnEstimators
from feature_extracttion.shape import to_2d_array

#>>> nose2 --plugin=nose2.plugins.layers

with such.A('sample test') as it:

    with it.having('LimitMax test group'):
        upper_limit = 10
        limit_max = LimitMax(upper_limit)
        limit_max_x = np.arange(20)
        limit_max_result = np.where(limit_max_x >= upper_limit, limit_max_x, upper_limit).reshape((-1, 1))
        @it.should('LimitMax fit method')
        def spectrogram_test1(case):
            it.assertIsInstance(limit_max.fit(limit_max_x), LimitMax)

        @it.should('LimitMax transform method')
        def spectrogram_test2(case):
            it.assertTrue(np.alltrue(limit_max.transform(limit_max_x) == limit_max_result))

        @it.should('LimitMax fit_transform method')
        def spectrogram_test3(case):
            it.assertTrue(np.alltrue(limit_max.fit_transform(limit_max_x) == limit_max_result))


    with it.having('LogTransformer test group'):
        log_x = np.arange(20)
        log_result = np.log1p(log_x)
        log_transformer = LogTransformer()
        @it.should('LogTransformer fit method')
        def waevelet_test1(case):
            it.assertIsInstance(log_transformer.fit(log_x), LogTransformer)

        @it.should('LogTransformer transform method')
        def waevelet_test2(case):
            it.assertTrue(np.alltrue(log_transformer.transform(log_x) == log_result))

        @it.should('LogTransformer fit_transform method')
        def waevelet_test3(case):
            it.assertTrue(np.alltrue(log_transformer.fit_transform(log_x) == log_result))

    with it.having('TahnEstimators test group'):
        tahn_x = np.linspace(-2,2,10)
        tahn_estimators = TahnEstimators()
        
        tahn_result = 0.5 * (np.tanh(0.01 * (to_2d_array(tahn_x) - np.mean(tahn_x)) / np.std(tahn_x)) + 1)
        @it.should('TahnEstimators fit method')
        def tahn_test1(case):
            it.assertIsInstance(tahn_estimators.fit(tahn_x), TahnEstimators)

        @it.should('TahnEstimators transform method')
        def tahn_test2(case):
            it.assertTrue(np.alltrue(tahn_estimators.transform(tahn_x) == tahn_result))

        @it.should('TahnEstimators fit_transform method')
        def tahn_test3(case):
            it.assertTrue(np.alltrue(tahn_estimators.fit_transform(tahn_x) == tahn_result))

    it.createTests(globals())