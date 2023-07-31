# Multipyles: computing multipoles in python

This repository contains a python package to compute the multipole moments
from a local density matrix (e.g., from density-functional theory or dynamical
mean-field theory) and files for performing multipole-constrained calculations:
* `example.ipynb`: examples on how to use the code and benchmark against VASP
for Cr2O3 (data in folder `tests/Cr2O3_benchmark_vasp/`)
* `multipyles/`: the python package, containing
    * `multipyles.py`: all the functions for calculating multipoles and postprocessing the results
    * `read_from_dft.py`: functions to read either density matrices to feed into
    the program or calculated multipoles from different DFT codes for comparison
    * `helper.py` and `multipole_eqs.py`: helper functions needed in `multipoles.py`
* `tests/`: the test suite for pytest
* `constrained_dft_vasp/`: the files needed to perform multipole-constrained
DFT calculations. Please look at the README there for more information.

If you add the directory this README is in to the `sys.path`, you can import the python code with
```
from multipyles import multipyles, read_from_dft
```

It currently works for all density matrices with the same orbital moment $l = l'$.

This code is based on [Bultmark et al. (2009)](https://doi.org/10.1103/PhysRevB.80.035121).

## Data sources and computable quantities

Currently, there are implementations for reading the density matrix from VASP
and less well tested ones for abinit and Elk.
However, every local density matrix can be used by reading it in as a numpy array.
From these matrices, the multipoles can be straightforwardly calculated.

Alternatively, the multipoles computed by Elk and VASP (unpublished VASP modification
by our group) can directly be read. This allows for direct comparison between
this implementation and the respective DFT implementation.

For the computed/read multipoles, the code can then calculate exchange and Hartree
energies.

## How to cite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6907024.svg)](https://doi.org/10.5281/zenodo.6907024)

Please use the doi linked in the [README.md on the main branch](https://github.com/materialstheory/multipyles/blob/main/README.md).

## Equations

Brief summary of the important steps for this code with explicit summations:

### Transformation from cubic to spherical harmonics

`multipyles.helper.spherical_to_cubic` returns the matrix M to transform
the complex spherical harmonics $|c_m \rangle$
to the real cubic harmonics $|r_m \rangle$
with $|r_m\rangle = \sum_n M_{mn} |c_n \rangle$ so that the density matrix in complex spherical harmonics is
```math
\rho_{m,m'} = \langle c_m | \rho | c_{m'} \rangle = \sum_{nn'} M_{nm} \langle r_n | \rho | r_{n'} \rangle M^*_{n'm'}
```

### Decomposition of occupation in Pauli matrices

```math
\rho_{i,m,m',p} = \left.\left(\sum_{s,s'} \rho_{i,m,m',s,s'} \sigma^{(p)}_{s',s} \right)\right/2,
```
where $\rho_{i,m,m',s,s'}$ is the density matrix per site $i$ in spherical harmonics
and $p = x, y, z, 0$ is the index to the Pauli matrices $\sigma^{(p)}$.

### Computation of time-reversal (tr) even and odd density matrix

Time-reversal symmetrization based on Pauli matrices:

```math
\rho^{(\nu)}_{i,m,m',p} = \left.\left(\rho_{i,m,m',p} +
(-1)^\nu (-1)^{m+m'} (-1)^{p \neq 0} \rho_{i,-m',-m,p}\right)\right/2,
```
where $\nu = 0 (1)$ means time-reversal even (odd).

Then rewrite matrix in spin components:

```math
\rho^{(\nu)}_{i,m,m',s,s'} = \sum_p \rho^{(\nu)}_{i,m,m',p} \sigma^{(p)}_{s,s'}.
```

### Calculation of multipole moments

Formulas for multipoles ($\sigma = 1/2$):

```math
\displaylines{\omega_{k,x,m,m'} = \bar\omega_k \cdot (-1)^{l-m}
\begin{pmatrix} l & k & l \\ -m & x & m' \end{pmatrix} \\

\bar\omega_k = \left.\sqrt{(2l-k)! (2l+k+1)!} \right/ (2l)!\\

\chi_{p,y,s,s'} = \bar\chi_p \cdot (-1)^{\sigma-s}
\begin{pmatrix} \sigma & p & \sigma \\ -s & y & s' \end{pmatrix}\\

\bar\chi_p = \sqrt{(2\sigma+p+1)!} = \sqrt{(p+2)!}\\

\xi_{k,p,r,x,y,t} = \bar\xi_{k,p,r} \cdot (-1)^{k-x+p-y}
\begin{pmatrix} k & r & p \\ -x & t & -y \end{pmatrix}\\

\bar\xi_{k,p,r} = \mathrm i^{-g} \cdot \sqrt{\frac{(g+1)!}{(g-2k)! (g-2p)! (g-2r)!}}
\cdot \frac{(g-2k)!! (g-2p)!! (g-2r)!!}{g!!} \quad\text{with}\quad g=k+p+r\\

w^{(\nu)}_{i,t,k,p,r} = \sum_{x=-k\ldots k, \\ y=-p\ldots p} \xi_{k,p,r,x,y,t}
\sum_{s=-\sigma\ldots\sigma, \\ s'=-\sigma\ldots\sigma} \chi_{p,y,s,s'}
\sum_{m=-l\ldots l, \\ m'=-l\ldots l} \omega_{k,x,m,m'} \rho^{(\nu)}_{i,m',m,s',s}}
```

# Copyright and license

Copyright (c) 2022-2023 ETH Zurich, Maximilian E. Merkel; Materials Theory Group, D-MATL

This project is licensed under the MIT License - see [LICENSE.txt](./LICENSE.txt) for details.

