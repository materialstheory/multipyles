# Multipyles: computing multipoles in python

This repository contains python scripts to compute the multipole moments
from a local density matrix (e.g., from density-functional theory or dynamical
mean-field theory):
* `example.ipynb`: examples on how to use the code and benchmark against VASP
for Ba2MgReO6 (see folder `BMRO_benchmark/`)
* `multipoles.py`: all the functions for calculating multipoles
* `read_from_dft.py`: functions to read density matrices to feed into
the program and calculated multipoles from different DFT codes for comparison
* `helper.py`: helper functions needed in `multipoles.py`

It currently works for all density matrices with the same orbital moment $l = l'$.

This code is based on [Bultmark et al. (2009)](https://doi.org/10.1103/PhysRevB.80.035121).

## Data sources and computable quantities

Currently, there are implementations for reading the density matrix from VASP
and abinit. However, every local density matrix can be used by reading it in as a numpy array.
From these matrices, the multipoles can be straightforwardly calculated.

Alternatively, the multipoles computed by Elk and VASP (unpublished VASP modification
by our group) can directly be read. This allows for direct comparison between
this implementation and the respective DFT implementation.

For the computed/read multipoles, the code can then calculate exchange and Hartree
energies.

## How to cite

Please use the doi linked in the README.md of the newest stable release.

## Equations

Brief summary of the important steps for this code with explicit summations:

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
\displaylines{\omega_{k,x,m_a,m_b} = \bar\omega_k \cdot (-1)^{l-m_b}
\begin{pmatrix} l & k & l \\ -m_b & x & m_a \end{pmatrix} \\

\bar\omega_k = \left.\sqrt{(2l-k)! (2l+k+1)!} \right/ (2l)!\\

\chi_{p,y,s_a,s_b} = \bar\chi_p \cdot (-1)^{\sigma-s_b}
\begin{pmatrix} \sigma & p & \sigma \\ -s_b & y & s_a \end{pmatrix}\\

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

Copyright (C) 2022 ETH Zurich, Maximilian E. Merkel; Materials Theory Group, D-MATL

This file is part of the repository multipyles.

multipyles is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

multipyles is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
multipyles. If not, see <https://www.gnu.org/licenses/>.

