import numpy as np
from scipy import signal
import pywt
from nose2.tools  import such
from feature_extracttion.representative import PercentileSummaryTransformer, SummaryTransformer, AverageTransformer

#>>> nose2 --plugin=nose2.plugins.layers

with such.A('sample test') as it:

    with it.having('PercentileSummaryTransformer test group'):
        axis = 0
        should_flat = True
        percentilesummary_transformer = PercentileSummaryTransformer(axis, should_flat)
    
        percentilesummary_x = np.arange(12)
        mean = percentilesummary_x.mean(axis=axis)
        std = percentilesummary_x.std(axis=axis)
        std_top = mean + std
        std_bot = mean - std
        percentile_calc = np.percentile(percentilesummary_x, [0, 1, 25, 50, 75, 99, 100], axis=axis)
        max_range = percentile_calc[-1] - percentile_calc[0]  
        relative_percentile = percentile_calc - mean  
        if not axis:
            percentilesummary_result = np.hstack(
                [np.asarray([mean, std, std_top, std_bot, max_range]), percentile_calc, relative_percentile])
        elif axis == 1:
            percentilesummary_result = np.vstack(
                [np.vstack([mean, std, std_top, std_bot, max_range]), percentile_calc, relative_percentile]).transpose()
        else:
            raise ValueError("not implemented")
        if should_flat:
            percentilesummary_result = percentilesummary_result.flatten()
        @it.should('PercentileSummaryTransformer fit method')
        def spectrogram_test1(case):
            it.assertIsInstance(percentilesummary_transformer.fit(percentilesummary_x), PercentileSummaryTransformer)

        @it.should('PercentileSummaryTransformer transform method')
        def spectrogram_test2(case):
            it.assertTrue(np.alltrue(percentilesummary_transformer.transform(percentilesummary_x) == percentilesummary_result))

        @it.should('PercentileSummaryTransformer fit_transform method')
        def spectrogram_test3(case):
            it.assertTrue(np.alltrue(percentilesummary_transformer.fit_transform(percentilesummary_x) == percentilesummary_result))


    with it.having('SummaryTransformer test group'):
        axis=0
        summary_transformer = SummaryTransformer(axis)
        summary_x = np.arange(12).reshape((3, 4))
        avgs = np.mean(summary_x, axis=axis)
        stds = np.std(summary_x, axis=axis)
        maxs = np.max(summary_x, axis=axis)
        mins = np.min(summary_x, axis=axis)
        medians = np.median(summary_x, axis=axis)
        summary_result = np.hstack([avgs, stds, maxs, mins, medians])
        @it.should('SummaryTransformer fit method')
        def waevelet_test1(case):
            it.assertIsInstance(summary_transformer.fit(summary_x), SummaryTransformer)

        @it.should('SummaryTransformer transform method')
        def waevelet_test2(case):
            it.assertTrue(np.alltrue(summary_transformer.transform(summary_x) == summary_result))

        @it.should('SummaryTransformer fit_transform method')
        def waevelet_test3(case):
            it.assertTrue(np.alltrue(summary_transformer.fit_transform(summary_x) == summary_result))

    with it.having('AverageTransformer test group'):
        average_transformer = AverageTransformer()
        average_x  = np.arange(12).reshape((3, 4))
        average_result = np.mean(average_x, axis=1).reshape((-1, 1))
        @it.should('AverageTransformer fit method')
        def tahn_test1(case):
            it.assertIsInstance(average_transformer.fit(average_x), AverageTransformer)

        @it.should('AverageTransformer transform method')
        def tahn_test2(case):
            it.assertTrue(np.alltrue(average_transformer.transform(average_x) == average_result))

        @it.should('AverageTransformer fit_transform method')
        def tahn_test3(case):
            it.assertTrue(np.alltrue(average_transformer.fit_transform(average_x) == average_result))

    it.createTests(globals())