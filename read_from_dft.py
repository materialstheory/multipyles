#!/usr/bin/env python3

import numpy as np
import pandas as pd

import warnings

_IO_ENTRIES = ['species', 'atom', 'nu', 'l1', 'l2', 'k', 'p', 'r', 't', 'value']

def read_multipoles_from_elk(source):
    """
    Reads the multipoles calculated with elk.

    Parameters
    ----------
    source : iterable of string
        Data source, e.g. file or list of strings.

    Returns
    -------
    pandas.DataFrame
        The multipoles.
    """

    results = []
    species = None
    atom = None
    nu = None
    l1 = None
    l2 = None
    k = None
    p = None
    r = None
    t = None
    for line in source:
        words = line.split()
        if line.strip().startswith('Species'):
            species = int(words[2])
            atom = int(words[6])
        if line.strip().startswith('nu='):
            nu = int(words[1][:-1])
            l1 = int(words[4][:-1])
            l2 = int(words[7])
        if line.strip().startswith('k ='):
            k = int(words[2][:-1])
            p = int(words[5][:-1])
            r = int(words[8])
        if line.strip().startswith('t ='):
            t = int(words[2])
            v1 = float(words[4])
            v2 = float(words[5])
            val = v1 + 1j*v2
            results.append(dict(zip(_IO_ENTRIES,
                                    [species, atom, nu, l1, l2, k, p, r, t, val])))
    return pd.DataFrame(results)

def read_multipoles_from_vasp(source):
    """
    Reads the multipoles calculated with VASP.

    Parameters
    ----------
    source : iterable of string
        Data source, e.g. file or list of strings.

    Returns
    -------
    pandas.DataFrame
        The multipoles.
    """

    results = []
    idx = [1, 3, 5, 7, 9, 10, 11, 12]
    for line in source:
        words = line.split()
        atom, nu, l1, l2, k, p, r, t = (int(words[i]) for i in idx)
        val = float(words[13]) + 1j*float(words[14])
        results.append(dict(zip(_IO_ENTRIES, [1, atom, nu, l1, l2, k, p, r, t, val])))

    return pd.DataFrame(results)


def read_densmat_from_vasp(source):
    """
    Reads the VASP OUTCAR for the density matrix of the last step from the
    section starting with 'atom = '. Currently only implemented for
    spin-orbital coupled calculations.

    Parameters
    ----------
    source : iterable of string
        Data source, e.g. file or list of strings.

    Returns
    -------
    occupation_per_site : numpy.ndarray of float
        The occupation matrix per site in orbital-spin space.
    """

    # Reads the relevant lines for the density matrix. Because of using a dict,
    # information is overwritten when reading the next iteration
    data = {}
    reading = False
    for line in source:
        if 'atom = ' in line:
            atom = int(line.split()[2])
            data[atom] = ''
            nspins = 0
            reading = True
            continue

        if not reading:
            continue

        if 'spin  ' in line or 'occupancies' in line:
            reading = False
            continue

        if 'spin component' in line:
            nspins += 1
            assert int(line.split()[-1]) == nspins
            continue

        if '.' in line:
            data[atom] += line

    if nspins not in (2, 4):
        raise NotImplementedError('Number of spin channels not supported')

    # Casts data in useful format
    data = np.array([np.loadtxt(s for s in string.splitlines())
                     for _, string in sorted(data.items())])

    dim = data.shape[2]
    # Non-collinear calculations have complex density matrix
    if nspins == 4:
        assert dim % 2 == 0, 'Line should contain real and complex part'
        dim //= 2
        data = data[:, :, :dim] + 1j * data[:, :, dim:]
        data = data.reshape(data.shape[0], 2, 2, dim, dim)
    # Collinear calculations are zero in spin-off-diagonal entries
    elif nspins == 2:
        reshaped_data = np.zeros((data.shape[0], 2, 2, dim, dim), dtype=float)
        reshaped_data[:, [0, 1], [0, 1]] = data.reshape(data.shape[0], nspins, dim, dim)
        data = reshaped_data

    warnings.warn('The multipoles here are consistent with the 2021 VASP implementation.'
                  + 'There has been a bug fix which might change the signs of some magnetic multipoles.')

    # Transposes data
    # FIXME: This produces the same multipoles as the VASP implementation of the multipoles
    #        Not sure if orbitals need to be transposed because VASP stores them differently
    return data.transpose((0, 4, 3, 2, 1))


def read_densmat_from_abinit(source, use_entries):
    """
    Reads the abinit log for the density matrix of the last step from the
    section starting with 'Augmentation waves occupancies Rhoij'.
    Currently only implemented for spin-unpolarized calculations.

    Parameters
    ----------
    source : iterable of string
        Data source, e.g. file or list of strings.
    use_entries : list or tuple
        Entries in the density matrix to use.

    Returns
    -------
    occupation_per_site : numpy.ndarray of float
        The occupation matrix per site in orbital-spin space.
    """

    reading = False
    for line in source:
        line = line.strip()
        if line == 'Augmentation waves occupancies Rhoij:':
            reading = True
            data = []
            continue

        if not reading or line == '' or 'value' in line:
            continue

        if '========================' in line:
            reading = False
            continue

        if 'Atom' in line:
            data.append([])
            continue

        data[-1].append(line.split())

    data = np.array(data, dtype=float)
    data = data[:, use_entries][:, :, use_entries]

    occupation_per_site = np.array([[data/2, np.zeros_like(data)],
                                   [np.zeros_like(data), data/2]])
    occupation_per_site = occupation_per_site.transpose((2, 3, 4, 0, 1))
    return occupation_per_site
