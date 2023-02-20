# -*- coding: utf-8 -*-

"""
Code to calculate the multipoles from a density matrix following the formalism from
F. Bultmark et al., Phys. Rev. B 80, 035121 (2009), doi: 10.1103/PhysRevB.80.035121.

Author: Maximilian Merkel, Materials Theory, ETH Zürich (2021-2022)

Vasp and Elk reading functions written by Florian Thöle, Materials Theory, ETH Zürich.
Multipole computation re-derived with the help of Alberto Carta and debugged
with the help of Andrea Urru, both Materials Theory, ETH Zürich.

Please cite with the DOI from the github project.
"""

# TODO: add helper functions to read in SOC matrices, collinear matrices, t2g matrices, ...

import numpy as np
import pandas as pd

from sympy.physics.quantum import cg
import sympy

import helper

_SIGMA = sympy.S(1)/2
_PAULI_MATRICES = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]],
                            [[1, 0], [0, -1]], [[1, 0], [0, 1]]]) # x, y, z, 0
_IO_ENTRIES = ['species', 'atom', 'nu', 'l1', 'l2', 'k', 'p', 'r', 't', 'value']

def _omega_norm(l, k):
    factorial = np.math.factorial
    return np.sqrt(factorial(2*l-k) * factorial(2*l+k+1)) / factorial(2*l)

def _omega(l, k, x, m_a, m_b):
    return float(helper.minus_one_to_the(l-m_b) * _omega_norm(l, k)
                 * cg.Wigner3j(l, -m_b, k, x, l, m_a).doit())

def _chi_norm(p):
    return np.sqrt(np.math.factorial(p+2))

def _chi(p, y, s_a, s_b):
    return complex(((-1)**(_SIGMA-s_b) * _chi_norm(p)
                    * cg.Wigner3j(_SIGMA, -s_b, p, y, _SIGMA, s_a).doit()).evalf())

def _xi_norm(a, b, c):
    factorial = np.math.factorial
    double_factorial = [np.prod(np.arange(n, 0, -2)) for n in range(30)]
    g = a+b+c
    return (1j**-g * np.sqrt(factorial(g+1) / factorial(g-2*a) / factorial(g-2*b) / factorial(g-2*c))
            * double_factorial[g-2*a] * double_factorial[g-2*b]
            * double_factorial[g-2*c] / double_factorial[g])

def _xi(k, p, r, x, y, t):
    return complex(_xi_norm(k, p, r) * helper.minus_one_to_the(k-x+p-y)
                   * cg.Wigner3j(k, -x, r, t, p, -y).doit())

def _exchange_k(l, k1, p, r):
    """Array for exchange matrix from eqn. (30) from Bultmark paper for each k. """
    k_range = np.arange(0, 2*l+1, 2)
    values = np.zeros_like(k_range, dtype=float)
    values[:] = -(2*l+1)**2 * (2*k1+1) * (2*r+1) / 4 * helper.minus_one_to_the(k1)
    values /= abs(_xi_norm(k1, p, r))**2 * _omega_norm(l, k1)**2
    values *= np.array([cg.Wigner3j(l, 0, k, 0, l, 0).doit()**2 * cg.Wigner6j(l, l, k1, l, l, k).doit()
                        for k in k_range], dtype=float)
    return values

def _hartree_k(l, k1, p):
    """
    Array for hartree matrix analogous to _exchange_k, derived from eqn. (22)
    from Bultmark paper, for each k. """
    k_range = np.arange(0, 2*l+1, 2)
    values = np.zeros_like(k_range, dtype=float)
    values[:] = (2*l+1)**2 / 2
    values /= _omega_norm(l, k1)**2
    values *= np.array([cg.Wigner3j(l, 0, k, 0, l, 0).doit()**2 * float(k==k1 and p==0)
                        for k in k_range], dtype=float)
    return values

def calculate(density_matrix, cubic=True, verbose=False):
    """
    Calculates the multipole moments from a density matrix in cubic or spherical
    harmonics.

    The order of angular degrees of freedome of the density matrix needs to be
    -l, -l+1, ..., l=1, l for (complex) spherical harmonics and as listed at
    https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
    for cubic (= real spherical) harmonics. In particular, for the d shell that
    means xy, yz, z^2, xz, x^2-y^2 and for the p shell y, z, x.

    Parameters
    ----------
    density_matrix : numyp.ndarray, [n_atoms, 2l+1, 2l+1, 2, 2] of complex
        The local density matrix for multiple atoms in the full orbital-spin space.
    cubic : bool, optional
        If True, the density matrix is given in cubic harmonics. The default is True.
    verbose : bool, optional
        Turns on additional prints. The default is False.

    Returns
    -------
    pandas.DataFrame
        The calculated multipoles.
    int
        The angular momentum l of the density matrix. Needed for postprocessing.
    """
    np.set_printoptions(precision=3, suppress=True, floatmode='fixed', linewidth=200)

    # Performs initial checks on the density matrix and gets the angular momentum
    l = helper.check_density_matrix_and_get_angular_momentum(density_matrix)
    if verbose:
        print('-'*40)
        print('Input density matrix')
        print('Spin up - spin up')
        print(density_matrix[:, :, :, 0, 0])
        print('Spin up - spin down')
        print(density_matrix[:, :, :, 0, 1])
        print('Spin down - spin up')
        print(density_matrix[:, :, :, 1, 0])
        print('Spin down - spin down')
        print(density_matrix[:, :, :, 1, 1])

        density_matrix_ls = density_matrix.transpose((0, 1, 3, 2, 4))
        print('Eigenvalues of density matrix in combined l-s space')
        print(np.linalg.eigvalsh(density_matrix_ls.reshape(-1, 10, 10)))

    # Transforms density matrix from cubic to spherical harmonics
    if cubic:
        print('Transforming from cubic to spherical harmonics')
        trafo_matrix = helper.spherical_to_cubic(l)
        density_matrix = np.einsum('al,iabrs,bk->ilkrs', trafo_matrix.conj(), density_matrix,
                                   trafo_matrix)

        if verbose:
            print('-'*40)
            print('Transformation matrix cubic to spherical harmonics')
            print(trafo_matrix)

            print('-'*40)
            print('Density matrix in spherical harmonics')
            print('Spin up - spin up')
            print(density_matrix[:, :, :, 0, 0])
            print('Spin up - spin down')
            print(density_matrix[:, :, :, 0, 1])
            print('Spin down - spin up')
            print(density_matrix[:, :, :, 1, 0])
            print('Spin down - spin down')
            print(density_matrix[:, :, :, 1, 1])

    # Decomposes density matrix into Pauli matrices
    density_matrix_pauli = np.einsum('imnrs,psr->imnp', density_matrix, _PAULI_MATRICES)/2
    if verbose:
        print('-'*40)
        print('Density matrix in Pauli matrices')
        print('Pauli matrix sigma_x')
        print(density_matrix_pauli[:, :, :, 0])
        print('Pauli matrix sigma_y')
        print(density_matrix_pauli[:, :, :, 1])
        print('Pauli matrix sigma_z')
        print(density_matrix_pauli[:, :, :, 2])
        print('Identity matrix')
        print(density_matrix_pauli[:, :, :, 3])

    # Decomposes density matrix in time-reversal even and odd parts
    alternating_minus = 1-2*(np.arange(-l, l+1) % 2)
    density_matrix_pauli_tr = .5 * np.array([density_matrix_pauli
                                             + np.einsum('m,n,p,imnp->inmp', alternating_minus,
                                                         alternating_minus,
                                                         (-1, -1, -1, 1),
                                                         density_matrix_pauli[:, ::-1, ::-1]) * tr
                                             for tr in (1, -1)])
    density_matrix_tr = np.einsum('uimnp,prs->uimnrs', density_matrix_pauli_tr, _PAULI_MATRICES)

    if verbose:
        print('-'*40)
        print('Time-reversal-even and -odd density matrix')
        print('Spin up - spin up')
        print(density_matrix_tr[:, :, :, :, 0, 0])
        print('Spin up - spin down')
        print(density_matrix_tr[:, :, :, :, 0, 1])
        print('Spin down - spin up')
        print(density_matrix_tr[:, :, :, :, 1, 0])
        print('Spin down - spin down')
        print(density_matrix_tr[:, :, :, :, 1, 1])

    # Checks that time-reversal decomposition summed up gives original matrix
    assert np.allclose(density_matrix_tr[0] + density_matrix_tr[1], density_matrix)

    # Calculates multipole moments
    results = []
    for k in range(2*l+1): # orbital dof
        for p in (0, 1): # spin dof
            for r in range(abs(k-p), k+p+1): # tensor rank
                x_range = range(-k, k+1)
                y_range = range(-p, p+1)
                s_range = (+_SIGMA, -_SIGMA)
                m_range = range(-l, l+1)
                t_range = range(-r, r+1)

                xi_matrix = np.array([[[_xi(k, p, r, x, y, t) for t in t_range]
                                       for y in y_range] for x in x_range])
                omega_matrix = np.array([[[_omega(l, k, x, m_a, m_b) for m_b in m_range]
                                          for m_a in m_range] for x in x_range])
                chi_matrix = np.array([[[_chi(p, y, s_a, s_b) for s_b in s_range] for s_a in s_range]
                                       for y in y_range])

                if verbose:
                    print('-'*40)
                    print('Multipole matrices')
                    print('xi(k, p, r, x, y, t)')
                    print(xi_matrix)
                    print('omega(k, x, m_a, m_b)')
                    print(omega_matrix)
                    print('chi(p, y, s_a, s_b)')
                    print(chi_matrix)

                # u=nu, m/n=m1/m2, r/s=s1,s2
                multipole_matrix_sph = np.einsum('xyt,xmn,yrs,uinmsr->uit', xi_matrix,
                                                 omega_matrix, chi_matrix, density_matrix_tr)

                for tr, mm_per_tr in enumerate(multipole_matrix_sph):
                    for i, mm_per_atom in enumerate(mm_per_tr):
                        for t, val in zip(t_range, mm_per_atom):
                            results.append(dict(zip(_IO_ENTRIES,
                                                    [1, i, tr, l, l, k, p, r, t, val])))

    return pd.DataFrame(results), l

def filter_results(df, cond):
    """ Filters dataframe df with condition from dictionary cond. """
    cond_fulfilled = (df[cond.keys()] == list(cond.values())).all(axis=1)
    return df[cond_fulfilled]

def transform_results_to_real(results):
    group_keys = ['species', 'atom', 'nu', 'l1', 'l2', 'k', 'p', 'r']
    transformed_results = pd.DataFrame()

    for label, grouped_df in results.groupby(group_keys):
        r = label[-1]
        grouped_df = grouped_df.sort_values('t')
        assert np.all(grouped_df['t'] == np.arange(-r, r+1)), 'Input dataframe is incomplete'

        trafo_matrix = helper.spherical_to_cubic(r)
        grouped_df['value'] = trafo_matrix.dot(grouped_df['value'])

        if not np.allclose(np.imag(grouped_df['value']), 0):
            print('WARNING: multipoles should be real but are not, for',
                  group_keys, '=', label)
        else:
            grouped_df['value'] = np.real(grouped_df['value'])

        transformed_results = pd.concat((transformed_results, grouped_df))

    return transformed_results

def calculate_hartree_and_exchange_energies(l, results, uj=None, slater_ints=None):
    """
    Calculates the Hartree and exchange energies for all multipole moments
    following eqns. (22) and (29) from the Bultmark paper.
    Returns the squared absolute value, the Hartree and exchange contribution per
    Slater integral and in total for each multipole kpr.
    """
    if (uj is None and slater_ints is None
            or uj is not None and slater_ints is not None):
        raise ValueError('Give either (U, J) or the Slater integrals')

    if uj is not None:
        assert len(uj) == 2, 'Tuple of (U, J) expected for variable uj'
        slater_ints = helper.uj_to_slater_integrals(l, uj[0], uj[1])
        print(f'Calculated Slater integrals = {slater_ints}')

    assert len(slater_ints) == l+1, f'{l+1} Slater integrals expected for l={l} shell'

    tags = ['species', 'atom', 'nu', 'l1', 'l2', 'k', 'p', 'r']

    new_data = []
    for label, grouped_df in results.groupby(tags, dropna=False):
        r = label[-1]
        grouped_df = grouped_df.sort_values('t')
        assert np.all(grouped_df['t'] == np.arange(-r, r+1)), 'Input dataframe is incomplete'

        val_squared = (grouped_df['value'].abs()**2).sum()
        exchange_terms = val_squared * _exchange_k(l, *label[5:])
        hartree_terms = val_squared * _hartree_k(l, label[5], label[6])
        new_data.append(label + (val_squared, ) + tuple(exchange_terms) + tuple(hartree_terms))

    new_tags = tags + ['w.w'] + [f'{name} F{2*i}' for name in ('exchange', 'hartree') for i in range(l+1)]
    energy_df = pd.DataFrame.from_records(new_data, columns=new_tags)

    energy_df['exchange total'] = sum([slater_ints[i] * energy_df[f'exchange F{2*i}'] for i in range(l+1)])
    energy_df['hartree total'] = sum([slater_ints[i] * energy_df[f'hartree F{2*i}'] for i in range(l+1)])
    return energy_df
