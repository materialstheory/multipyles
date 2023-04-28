"""
Contains the equations from Bultmark et al., Phys. Rev. B 80, 035121 (2009).
"""

import numpy as np
from sympy.physics.quantum import cg
import sympy

from . import helper

SIGMA = sympy.S(1)/2
PAULI_MATRICES = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]],
                           [[1, 0], [0, -1]], [[1, 0], [0, 1]]]) # x, y, z, 0

def _omega_norm(l, k):
    factorial = np.math.factorial
    return np.sqrt(factorial(2*l-k) * factorial(2*l+k+1)) / factorial(2*l)

def omega(l, k, x, m_a, m_b):
    """ The charge-dependent part, Eq. (20) from Bultmark 2009. """
    return float(helper.minus_one_to_the(l-m_b) * _omega_norm(l, k)
                 * cg.Wigner3j(l, -m_b, k, x, l, m_a).doit())

def _chi_norm(p):
    return np.sqrt(np.math.factorial(p+2))

def chi(p, y, s_a, s_b):
    """ The spin-dependent part, Eq. (24) from Bultmark 2009. """
    return complex(((-1)**(SIGMA-s_b) * _chi_norm(p)
                    * cg.Wigner3j(SIGMA, -s_b, p, y, SIGMA, s_a).doit()).evalf())

def _xi_norm(k, p, r):
    factorial = np.math.factorial
    g = k+p+r
    double_factorial = [np.prod(np.arange(n, 0, -2)) for n in range(g+1)]
    return (1j**-g * np.sqrt(factorial(g+1) / factorial(g-2*k) / factorial(g-2*p) / factorial(g-2*r))
            * double_factorial[g-2*k] * double_factorial[g-2*p]
            * double_factorial[g-2*r] / double_factorial[g])

def xi(k, p, r, x, y, t):
    """ The coupling of charge and spin part, Eq. (26) from Bultmark 2009. """
    return complex(_xi_norm(k, p, r) * helper.minus_one_to_the(k-x+p-y)
                   * cg.Wigner3j(k, -x, r, t, p, -y).doit())

def exchange_k(l, k1, p, r):
    """Array for exchange matrix for each k, Eq. (30) from Bultmark 2009. """
    k_range = np.arange(0, 2*l+1, 2)
    values = np.zeros_like(k_range, dtype=float)
    values[:] = -(2*l+1)**2 * (2*k1+1) * (2*r+1) / 4 * helper.minus_one_to_the(k1)
    values /= abs(_xi_norm(k1, p, r))**2 * _omega_norm(l, k1)**2
    values *= np.array([cg.Wigner3j(l, 0, k, 0, l, 0).doit()**2 * cg.Wigner6j(l, l, k1, l, l, k).doit()
                        for k in k_range], dtype=float)
    return values

def hartree_k(l, k1, p):
    """
    Array for hartree matrix analogous to exchange_k for each k,
    derived from Eq. (22) from Bultmark 2009.
    """
    k_range = np.arange(0, 2*l+1, 2)
    values = np.zeros_like(k_range, dtype=float)
    values[:] = (2*l+1)**2 / 2
    values /= _omega_norm(l, k1)**2
    values *= np.array([cg.Wigner3j(l, 0, k, 0, l, 0).doit()**2 * float(k==k1 and p==0)
                        for k in k_range], dtype=float)
    return values
