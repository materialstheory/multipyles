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

from . import helper
from .multipole_eqs import orbital_part, spin_part, coupling_part, exchange_k, hartree_k, SIGMA, PAULI_MATRICES

_IO_ENTRIES = ['species', 'atom', 'nu', 'l1', 'l2', 'k', 'p', 'r', 't', 'value']


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
        print(np.linalg.eigvalsh(density_matrix_ls.reshape(-1, 4*l+2, 4*l+2)))

    # Transforms density matrix from cubic to spherical harmonics
    if cubic:
        print('Transforming from cubic to spherical harmonics')
        trafo_matrix = helper.spherical_to_cubic(l)
        density_matrix = np.einsum('al,iabrs,bk->ilkrs', trafo_matrix, density_matrix,
                                    trafo_matrix.conj())

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
    density_matrix_pauli = np.einsum('imnrs,psr->imnp', density_matrix, PAULI_MATRICES)/2
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
    density_matrix_tr = np.einsum('uimnp,prs->uimnrs', density_matrix_pauli_tr, PAULI_MATRICES)

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

    s_range = (+SIGMA, -SIGMA)
    m_range = range(-l, l+1)
    for k in range(2*l+1): # orbital dof
        x_range = range(-k, k+1)
        orbital_matrix = np.array([[[orbital_part(l, k, x, m, mp) for mp in m_range]
                                    for m in m_range] for x in x_range])

        for p in (0, 1): # spin dof
            y_range = range(-p, p+1)
            spin_matrix = np.array([[[spin_part(p, y, s, sp) for sp in s_range]
                                     for s in s_range] for y in y_range])

            for r in range(abs(k-p), k+p+1): # tensor rank
                t_range = range(-r, r+1)
                coupling_matrix = np.array([[[coupling_part(k, p, r, x, y, t) for t in t_range]
                                             for y in y_range] for x in x_range])

                if verbose:
                    print('-'*40)
                    print('Multipole matrices')
                    print('coupling_part(k, p, r, x, y, t)')
                    print(coupling_matrix)
                    print('orbital_part(k, x, m, mp)')
                    print(orbital_matrix)
                    print('spin_part(p, y, s, sp)')
                    print(spin_matrix)

                # u=nu, m/n=m/m', r/s=s,s'
                multipole_matrix_sph = np.einsum('xyt,xmn,yrs,uinmsr->uit', coupling_matrix,
                                                 orbital_matrix, spin_matrix, density_matrix_tr)

                for tr, mm_per_tr in enumerate(multipole_matrix_sph):
                    for i, mm_per_atom in enumerate(mm_per_tr):
                        for t, val in zip(t_range, mm_per_atom):
                            results.append(dict(zip(_IO_ENTRIES,
                                                    [1, i, tr, l, l, k, p, r, t, val])))

    return pd.DataFrame(results), l

def write_shift_matrix_for_vasp(l, k, t, filename='shift.txt'):
    """
    Generates shift that triggers the multipole moment w^k0k_t with
    the correct normalization. The index t labels the cubic multipole and
    the matrix in orbital space is also in the cubic basis.

    This shift matrix is then written to a file as a flatten matrix, where
    every row of the file contains the real and imaginary part of that matrix.
    This file is readable by the patched Vasp from this repository.

    Warning: not tested for odd k. The implementation might be incorrect by a sign.

    Parameters
    ----------
    l : int
        The angular momentum l of the density matrix.
    k : int
        The rank/order of the charge multipole.
    t : int
        The concrete charge multipole to be written to file,
        t in {-k, -k+1, ..., +k}. Unused if filename is None.
    filename : string, optional
        The name of the file to write the shift matrix to. The default is 'shift.txt'.

    Returns
    -------
    shifts : numpy.array
        The shift matrices for the parameters l, k but for all possible t.
        The shape is (2k+1) x (2l+1) x (2l+1).
    """

    if k % 2 == 1:
        print('WARNING: For odd k, the shift matrices are imaginary and therefore'
             + ' not symmetrical. Please check carefully that there is no transpose'
             + ' missing when using the shift matrix in DFT.')
    shifts = np.array([[[orbital_part(l, k, t_sph, m, mp) * helper.minus_one_to_the(k)
                         for mp in range(-l, l+1)]
                        for m in range(-l, l+1)]
                       for t_sph in range(-k, k+1)])

    # first transform to cubic multipoles
    trafo_matrix = helper.spherical_to_cubic(k)
    shifts = np.einsum('ij,jmn', trafo_matrix, shifts)

    # then transform matrix into cubic basis
    trafo_matrix = helper.spherical_to_cubic(l)
    shifts = np.einsum('al,ilk,bk->iab', trafo_matrix.conj(), shifts, trafo_matrix)

    # Writes shift matrix for w^k0k_t to file
    if filename is not None:
        if not (-k <= t <= k):
            raise ValueError('t has to be in {-k, -k+1, ..., +k}')

        output = [f'{matrix.real:.18f} {matrix.imag:.18f}' for matrix in shifts[t+k].flatten()]
        with open(filename, 'w') as file:
            file.write('\n'.join(output))

    return shifts

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
    following Eqs. (22) and (29) from the Bultmark paper.
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
        exchange_terms = val_squared * exchange_k(l, *label[5:])
        hartree_terms = val_squared * hartree_k(l, label[5], label[6])
        new_data.append(label + (val_squared, ) + tuple(exchange_terms) + tuple(hartree_terms))

    new_tags = tags + ['w.w'] + [f'{name} F{2*i}' for name in ('exchange', 'hartree')
                                 for i in range(l+1)]
    energy_df = pd.DataFrame.from_records(new_data, columns=new_tags)

    energy_df['exchange total'] = sum([slater_ints[i] * energy_df[f'exchange F{2*i}']
                                       for i in range(l+1)])
    energy_df['hartree total'] = sum([slater_ints[i] * energy_df[f'hartree F{2*i}']
                                      for i in range(l+1)])
    return energy_df
