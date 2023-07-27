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


def read_densmat_from_vasp(source, select_atoms=None):
    """
    Reads the VASP OUTCAR for the density matrix of the last step from the
    section starting with 'atom = '. Currently only implemented for
    spin-polarized calculations, collinear and non-collinear.

    Parameters
    ----------
    source : iterable of string
        Data source, e.g. file or list of strings.
    select_atoms : iterable of int
        Reads only density matrix for atoms with numbers in this list.
        Defaults to all atoms.

    Returns
    -------
    occupation_per_site : numpy.ndarray of complex
        The occupation matrix per site in orbital-spin space.
    """

    # Reads the relevant lines for the density matrix. Because of using a dict,
    # information is overwritten when reading the next iteration
    data = {}
    reading = False
    for line in source:
        if 'atom = ' in line:
            atom = int(line.split()[2])
            if select_atoms is None or atom in select_atoms:
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

    if data == {}:
        raise ValueError('No data found for input parameters')
    if nspins not in (2, 4):
        raise NotImplementedError('Number of spin channels not supported')

    # Casts data in useful format
    data = np.array([np.loadtxt(s for s in string.splitlines())
                     for _, string in sorted(data.items())])

    if len(data.shape) == 1:
        raise NotImplementedError('All density matrices have to have same dimension. '
                                  + 'Select atoms with density matrices of the same shell.')

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

    # Transposes only orbital degrees of freedom:
    # Vasp indexes spins in "C order": uu, ud, du, dd
    # but orbitals in Fortran order for performance reasons.
    # Determined from occupation eigenvalues and comparison with Elk code,
    # see also Vasp integration test. Other hints:
    # - C order of spins: cf. Supplemental material 10.1103/PhysRevB.106.035127
    # - F order of orbitals: cf. relativistic.F in VASP source code
    return data.transpose((0, 4, 3, 1, 2))


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

    warnings.warn('Function not well tested.')

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


def read_densmat_from_elk(source, select_species_atoms=None):
    """
    Reads the density matrix from the DMATMT.OUT file from Elk.
    This function is implemented for the integration test of Cr2O3 and untested
    otherwise. Simply returns all density matrices without any information about
    species and atoms.

    Parameters
    ----------
    source : iterable of string
        Data source, e.g. file or list of strings.
    select_species_atoms : iterable of list, optional
        Filters which species and atoms to read, as an iterable of tuples of
        (species, atom). The density matrices of all selected atoms must have
        the same angular momentum l. The default is None, where all species and
        atoms are read.

    Returns
    -------
    occupation_per_site : numpy.ndarray of complex
        The occupation matrix per site in orbital-spin space.

    """

    warnings.warn('Function untested except by Cr2O3 integration test.')

    densmats = {}
    for line in source:
        line = line.strip().split()
        if line == []:
            continue

        if 'species,' in line:
            species = int(line[0])
            atom = int(line[1])
            l = int(line[2])
            print(atom, l)
            if (select_species_atoms is None
                    or any((atom==a and species==s for s, a in select_species_atoms))):
                densmats[(species, atom)] = np.zeros((2*l+1, 2*l+1, 2, 2), dtype=complex)
            continue

        if (species, atom) not in densmats:
            continue

        if 'ispn,' in line:
            ispn = int(line[0])-1
            jspn = int(line[1])-1
            continue

        assert len(line) == 4

        m1 = int(line[0]) + l
        m2 = int(line[1]) + l
        val = float(line[2]) + 1j*float(line[3])

        # Density matrix seems to be written as <m1, ispn | rho | m2, jspn>
        densmats[(species, atom)][m1, m2, ispn, jspn] = val

    if select_species_atoms is None:
        select_species_atoms = sorted(list(densmats.keys()))

    occupation_per_site = np.array([densmats[key] for key in select_species_atoms])
    assert isinstance(occupation_per_site, np.ndarray), ('Conversion to numpy array failed,'
                                              + ' all atoms must have same l.')

    return occupation_per_site
