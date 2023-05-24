# Constraining multipoles in DFT

This folder contains the explanation of multipole-constrained
density-functional theory (DFT) calculations in VASP, which might
in the future be supplemented with the code of the implementation.
In any case, we describe implementation details below for reproducibility
using the input files of published work citing this repository.

Currently, only charge multipoles are implemented,
of which only inversion-symmetric (order $k$ even) are tested.
For a more detailed description,
please refer to the [publication linked below](#how-to-cite).
For these calculations,
the publication defines the multipole operator $\mu_{mm'}^{kt}$
of a multipole of order $k$ and type $t$
in the space of cubic (real) orbitals
for a given angular momentum $l$
and an according shift strength $s^I_{kt}$ on atom types $I$.

In the following, we describe the steps to run calculation.

## Description of the VASP modification

Our implementation modifies only `LDApU.F` from Vasp.
We first read in the multipole-related parameters
from the `INCAR` file (see below).
Then, we read in the shift matrix `shift.txt` generated
by the python package `multipyles` (see below).
Finally, we implement a local, occupation-independent potential that is added
to the +U potential if +U is used. This local potential is similar
to the implementation of `POT_TYPE3`, just that the potential
is now the shift strength (per atom type) times the multipole operator and therefore
depends on the variables `M1` and `M2`.

## Generating a shift matrix

Next, we need to generate a shift matrix. To allow for easy
wrapping and compartimentalizing the functionalities,
this is part of the python package `multipyles` in the folder above.

Simply run
```
multipyles.write_shift_matrix_for_vasp(l=2, k=2, t=0)
```
which in this example writes the shift matrix corresponding to a
$z^2$ quadrupole ($k=2$, $t=0$) for the d shell ($l=2$)
to file `shift.txt`, which is the file name required by VASP modification.

## Running a VASP calculation with constrained multipoles

Finally, we can run the DFT calculation.
In the reformulation of constrained DFT, we simply apply
a shift with a certain strength and see what multipole moments
come out.

The file `shift.txt` generated before has to be in the same folder
as the other VASP input files.
Additionally, there are three parameters needed in the `INCAR`.
The formalism is similar to using DFT+U in VASP
and the examples of parameters are for a material
with three different atom types.

- `uses_multipole_perturbation = .TRUE.`
turns on multipole perturbation.
- `LDAUL = 2 -1 -1`
to determine the shell per atom type that the shift is applied on
- `multipole_shift = 0.5 0 0`
to set shift strength

Therefore, in this example, we apply a shift of strength 0.5
on the d shell of the first atom type.

This works with and without DFT+U.
Since both use the parameter `LDAUL`,
the shell that shift and +U are applied on has to be the same.

## How to cite

If you use this method, please cite both the Zenodo doi
linked in the [main README of this repository](/README.md#how-to-cite)
and the publication

[1] L. Schaufelberger, M. E. Merkel, A. M. Tehrani, N. A. Spaldin, and C. Ederer,
“Exploring energy landscapes of charge multipoles using constrained density functional theory,”
[arXiv:2305.13988 (2023)](https://doi.org/10.48550/arXiv.2305.13988)
