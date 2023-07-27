from io import StringIO
import numpy as np
import pandas as pd
import pandas.testing
from multipyles import multipyles, read_from_dft

def test_cr2o3_vasp():
    """ Benchmark of l=l'=2 Cr density matrix against Vasp calculation of Cr2O3. """

    with open('tests/Cr2O3_benchmark_vasp/OUTCAR', 'r') as file:
        occupation_per_site = read_from_dft.read_densmat_from_vasp(file, select_atoms=(1, 2, 3, 4))

    # Reads in eigenvalues of Cr d occupation of the last iteration
    with open('tests/Cr2O3_benchmark_vasp/OUTCAR', 'r') as file:
        eigenvalues_outcar = np.loadtxt((line for line in file if '  o = ' in line), usecols=2)
    eigenvalues_outcar = eigenvalues_outcar[-76:-36].reshape(4, 10)

    # Compares eigenvalues of the density matrix in combined l-s space, i.e., as 10 x 10 matrix
    # Precision of comparison is limited by precision in OUTCAR
    density_matrix_ls = occupation_per_site.transpose((0, 1, 3, 2, 4)).reshape(4, 10, 10)
    eigenvalues = np.linalg.eigvalsh(density_matrix_ls)
    assert np.allclose(eigenvalues, eigenvalues_outcar,
                       atol=1.5e-4), np.max(np.abs(eigenvalues - eigenvalues_outcar))

    # Calculates multipoles
    results, l = multipyles.calculate(occupation_per_site, verbose=True)
    assert l == 2

    with open('tests/Cr2O3_benchmark_vasp/TENSMOM.R1.OUT', 'r') as file:
        res_vasp = multipyles.filter_results(read_from_dft.read_multipoles_from_vasp(file), {'l1': 2, 'l2': 2})
    # Makes atoms in Vasp zero-indexed for comparability with multipyles result
    res_vasp['atom'] -= 1

    # Defines tolerance for comparison and sorts of data
    atol = 2e-3
    order = list(results.columns)[:-1]
    res_vasp = res_vasp.sort_values(order, ignore_index=True)
    results = results.sort_values(order, ignore_index=True)

    print(multipyles.filter_results(results, {'atom': 0, 'k': 4, 'p': 0, 'r': 4, 'nu': 0}))
    print(multipyles.filter_results(res_vasp, {'atom': 0, 'k': 4, 'p': 0, 'r': 4, 'nu': 0}))

    pandas.testing.assert_frame_equal(results, res_vasp, atol=atol, rtol=0)


def test_cr2o3_elk():
    """ Benchmark of l=l'=2 Cr density matrix against Elk calculation of Cr2O3. """
    with open('tests/Cr2O3_benchmark_elk/DMATMT.OUT', 'r') as file:
        occupation_per_site = read_from_dft.read_densmat_from_elk(file, select_species_atoms=((1, 1), (1, 2),
                                                                                              (1, 3), (1, 4)))

    # Calculates multipoles
    results, l = multipyles.calculate(occupation_per_site, cubic=False, verbose=True)
    assert l == 2

    with open('tests/Cr2O3_benchmark_elk/TMDFTUNU.OUT', 'r') as file:
        res_elk = multipyles.filter_results(read_from_dft.read_multipoles_from_elk(file), {'species': 1, 'l1': 2, 'l2': 2})
    # Makes atoms in Elk zero-indexed for comparability with multipyles result
    res_elk['atom'] -= 1

    # Defines tolerance for comparison and sorts data
    atol = 2e-3
    order = list(results.columns)[:-1]
    res_elk = res_elk.sort_values(order, ignore_index=True)
    results = results.sort_values(order, ignore_index=True)

    print(multipyles.filter_results(results, {'atom': 0, 'k': 4, 'p': 0, 'r': 4, 'nu': 0}))
    print(multipyles.filter_results(res_elk, {'atom': 0, 'k': 4, 'p': 0, 'r': 4, 'nu': 0}))

    pandas.testing.assert_frame_equal(results, res_elk, atol=atol, rtol=0)


def test_t2g_matrices():
    """
    Test for t2g matrices of four different configurations:
        - 1/4 electron in the J_eff=3/2 quartet from spin-orbit coupling
        - 1/6 electron per t2g orbital
        - 1/2 electron in the J_eff=3/2, m_J_eff=+-3/2 doublet
        - 1/2 electron in the xz orbital
    The expected output is generated with multipyles, with the file
    update_integration_test_t2g.py.
    """

    with open('tests/t2g_occupied.txt', 'r') as file:
        lines = file.read()
    lines = lines.split('####')
    occ_inp = np.loadtxt(StringIO(lines[0]), dtype=complex).reshape((-1, 5, 5, 2, 2))
    df_exp = pd.read_csv(StringIO(lines[1]), index_col=0)

    df, l = multipyles.calculate(occ_inp, verbose=True)
    assert l == 2

    atol = 1e-6

    df = multipyles.transform_results_to_real(df)
    df = df[np.abs(df['value']) > atol]

    df_exp['value'] = np.abs(df_exp['value'])
    df['value'] = np.abs(df['value'])

    print(df_exp)
    print(df)

    pandas.testing.assert_frame_equal(df_exp, df, atol=atol)
