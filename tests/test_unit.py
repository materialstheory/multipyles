"""
Contains all unit tests for the multipyles package.
"""

import numpy as np

from multipyles.multipyles import write_shift_matrix_for_vasp
from multipyles.multipole_eqs import spin_part, SIGMA
from multipyles.helper import minus_one_to_the, spherical_to_cubic

def test_minus_one_to_the():
    assert minus_one_to_the(0) == 1
    assert minus_one_to_the(1) == -1

def test_chi_p0():
    """ Checks that for p=0, the spin matrix spin_part is unity. """
    s_range = (+SIGMA, -SIGMA)
    for s in s_range:
        for sp in s_range:
            assert np.isclose(spin_part(0, 0, s, sp), int(s == sp)), (s, sp)

def test_spherical_to_cubic_l1():
    sqrt2 = np.sqrt(2)
    trafo_p = np.array([[1j/sqrt2, 0, 1j/sqrt2],
                        [0, 1, 0],
                        [1/sqrt2, 0, -1/sqrt2]])

    assert np.allclose(trafo_p, spherical_to_cubic(1))

def test_compare_vasp_shift_matrices():
    """ Tests write_shift_matrix_for_vasp against previously generated files. """
    matrix = write_shift_matrix_for_vasp(2, 2, None, None)
    assert matrix.shape == (5, 5, 5)

    for t in range(-2, 3):
        saved_matrix = np.loadtxt(f'tests/shift_files/shift_22{t}.txt')
        saved_matrix = saved_matrix[:, 0] + 1j*saved_matrix[:, 1]
        saved_matrix = saved_matrix.reshape(5, 5)
        print(saved_matrix)
        assert np.allclose(matrix[2+t], saved_matrix), f't={t}'

    matrix = write_shift_matrix_for_vasp(3, 1, None, None)
    assert matrix.shape == (3, 7, 7)

    for t in range(-1, 2):
        saved_matrix = np.loadtxt(f'tests/shift_files/shift_31{t}.txt')
        saved_matrix = saved_matrix[:, 0] + 1j*saved_matrix[:, 1]
        saved_matrix = saved_matrix.reshape(7, 7)
        print(saved_matrix)
        assert np.allclose(matrix[1+t], saved_matrix), f't={t}'

def test_shift_matrix_for_monopoles():
    """ Tests that monopole matrices are simply identity matrices. """
    for l in range(4):
        matrix = write_shift_matrix_for_vasp(l, 0, None, None)
        assert matrix.shape == (1, 2*l+1, 2*l+1)
        assert np.allclose(matrix, np.eye(2*l+1))
