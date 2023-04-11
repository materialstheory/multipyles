import pandas.testing

import multipoles
import read_from_dft
from helper import minus_one_to_the

class TestUnit:
    def test_minus_one_to_the(self):
        assert minus_one_to_the(0) == 1
        assert minus_one_to_the(1) == -1

class TestIntegration:
    def test_benchmark_cr2o3(self):
        with open('Cr2O3_benchmark/OUTCAR', 'r') as file:
            occupation_per_site = read_from_dft.read_densmat_from_vasp(file, select_atoms=(1, 2, 3, 4))
        results, l = multipoles.calculate(occupation_per_site, verbose=False)

        assert l == 2

        with open('Cr2O3_benchmark/TENSMOM.R1.OUT', 'r') as file:
            res_vasp = multipoles.filter_results(read_from_dft.read_multipoles_from_vasp(file), {'l1': 2, 'l2': 2})
        # Makes Vasp input zero-indexed for comparability with multipyles result
        res_vasp['atom'] -= 1

        # Defines tolerance for comparison and sorting order of data
        atol = 2e-3
        order = list(results.columns)[:-1]

        pandas.testing.assert_frame_equal(results.sort_values(order, ignore_index=True),
                                          res_vasp.sort_values(order, ignore_index=True),
                                          atol=atol, rtol=0)
