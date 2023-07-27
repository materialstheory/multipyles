import numbers
import numpy as np

def check_density_matrix_and_get_angular_momentum(density_matrix):
    """
    Checks density matrix and deduces the angular momentum from the matrix shape.
    """

    if len(density_matrix.shape) != 5:
        raise ValueError('density_matrix needs to have dimension 5')

    dim = density_matrix.shape[-4]
    if density_matrix.shape[-3] != dim:
        raise ValueError('Angular momentum dimensions do not match')

    if dim % 2 != 1:
        raise ValueError('Angular momentum dimensions need to be odd.')

    if density_matrix.shape[-2] != density_matrix.shape[-1] != 2:
        raise ValueError('Spin dimensions different from 2')

    l = dim // 2
    print(f'Angular momentum of density matrix is l = {l}')

    # Checks if matrix is Hermitian
    if not np.allclose(density_matrix.conj(),
                       density_matrix.transpose((0, 2, 1, 4, 3))):
        print('WARNING: density matrix not Hermitian')

    return l

def spherical_to_cubic(l):
    """
    Generates matrix to transform from spherical to cubic harmonics for
    angular momentum l. Returns (2l+1) x (2l+1) complex numpy.ndarray.
    """
    # Inspired by the tools.py function from the tmom helper functions.
    # But it is just a definition of cubic harmonics
    sqrt2 = np.sqrt(2)

    trafo_matrix = np.zeros((2*l+1, 2*l+1), dtype=complex)
    for m in range(-l, l+1):
        if m < 0:
            trafo_matrix[l+m, l+m] = 1j/sqrt2
            trafo_matrix[l+m, l-m] = -1j * (-1)**m / sqrt2
        elif m > 0:
            trafo_matrix[l+m, l+m] = (-1)**m / sqrt2
            trafo_matrix[l+m, l-m] = 1/sqrt2
        else:
            trafo_matrix[l+m, l+m] = 1

    # Checks if matrix is unitary
    assert np.allclose(trafo_matrix.T.conj().dot(trafo_matrix), np.eye(2*l+1))

    return trafo_matrix

def minus_one_to_the(x):
    """ Calculates (-1)^x for integer x. """
    assert isinstance(x, numbers.Integral)
    return 1-2*(x%2)

def uj_to_slater_integrals(l, u, j):
    """
    Calculate Slater integrals from U and J for a shell.
    """
    if l == 1:
        return u, 5*j
    elif l == 2:
        r = 0.625
        return u, 14/(1+r)*j, 14*r/(1+r)*j
    else:
        raise NotImplementedError('Conversion from U and J to Slater integrals '
                                  + 'only implemented for l=1 and l=2.')