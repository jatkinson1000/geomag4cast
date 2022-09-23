#!/usr/bin/env python3

import compute_metrics
from maginput import MagInput
import numpy as np

class Test_Preprocess:
    def test_averaging(self):
        onearr = np.ones(int(60 * 24 * 20 / 5))
        testmagin = MagInput
        testmagin.Kp = onearr
        testmagin.Dst = onearr
        testmagin.dens = onearr
        testmagin.velo = onearr
        testmagin.Pdyn = onearr
        testmagin.ByIMF = onearr
        testmagin.BzIMF = onearr
        testmagin.W3 = onearr

        input_i, output_i = compute_metrics.preprocess_data(testmagin, 1000, 100)
        np.testing.assert_array_equal(input_i[:6], np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
