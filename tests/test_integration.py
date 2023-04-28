import pandas.testing
from multipyles import multipyles, read_from_dft

def test_benchmark_cr2o3():
    with open('tests/Cr2O3_benchmark/OUTCAR', 'r') as file:
        occupation_per_site = read_from_dft.read_densmat_from_vasp(file, select_atoms=(1, 2, 3, 4))
    results, l = multipyles.calculate(occupation_per_site, verbose=False)

    assert l == 2

    with open('tests/Cr2O3_benchmark/TENSMOM.R1.OUT', 'r') as file:
        res_vasp = multipyles.filter_results(read_from_dft.read_multipoles_from_vasp(file), {'l1': 2, 'l2': 2})
    # Makes atoms in Vasp zero-indexed for comparability with multipyles result
    res_vasp['atom'] -= 1

    # Defines tolerance for comparison and sorting order of data
    atol = 2e-3
    order = list(results.columns)[:-1]

    pandas.testing.assert_frame_equal(results.sort_values(order, ignore_index=True),
                                      res_vasp.sort_values(order, ignore_index=True),
                                      atol=atol, rtol=0)
