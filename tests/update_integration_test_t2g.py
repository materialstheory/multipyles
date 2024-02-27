"""
Writes file for the integration test of t2g matrices.
WARNING: executing it will overwrite the test so only run this script
if the test needs updating, e.g., after fixing a bug in multipyles.
"""

import sys
sys.path.append('..')

from multipyles import multipyles
import numpy as np

# Definitions consistent with Martins 2017, 10.1088/1361-648X/aa648f
sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)
sqrt6 = np.sqrt(6)

t2g_indices = [(i, s) for i in (3, 1, 0) for s in range(2)]

#                      xz up      down       yz up     down      xy up     down
jeff_rot = np.array([[-1j/sqrt3,  0       ,  1/sqrt3,  0      ,  0      , -1/sqrt3], #1/2, -1/2
                     [ 0       ,  1j/sqrt3,  0      ,  1/sqrt3,  1/sqrt3,  0      ], #1/2,  1/2
                     [-1j/sqrt6,  0       ,  1/sqrt6,  0      ,  0      ,  2/sqrt6], #3/2, -1/2
                     [ 0       , -1j/sqrt6,  0      , -1/sqrt6,  2/sqrt6,  0      ], #3/2,  1/2
                     [ 0       , -1j/sqrt2,  0      ,  1/sqrt2,  0      ,  0      ], #3/2, -3/2
                     [-1j/sqrt2, 0        , -1/sqrt2,  0      ,  0      ,  0      ], #3/2,  3/2
                    ])

assert np.isclose(np.abs(np.linalg.det(jeff_rot)), 1)

# Defines three occupations in Jeff basis
occs_jeff = [np.diag([0, 0, 1/4, 1/4, 1/4, 1/4]), 1/6*np.eye(6), np.diag([0, 0, 0, 0, 1/2, 1/2])]
occs_t2g = [np.einsum('ba,bc,cd->ad', jeff_rot, occ_jeff, jeff_rot.conj()) for occ_jeff in occs_jeff]
# And adds one occupation matrix in t2g basis
occs_t2g += [np.diag([1/2, 1/2, 0, 0, 0, 0])]

occ_full = np.zeros((len(occs_t2g), 5, 5, 2, 2), dtype=complex)
for i, occ_t2g in enumerate(occs_t2g):
    print(occ_t2g[[0, 2, 5]][:, [0, 2, 5]])
    for j1, (ind1, sp1) in enumerate(t2g_indices):
        for j2, (ind2, sp2) in enumerate(t2g_indices):
            occ_full[i, ind1, ind2, sp1, sp2] = occ_t2g[j1, j2]

perfect_multis, l = multipyles.calculate(occ_full, verbose=False)
assert l == 2
perfect_multis = multipyles.transform_results_to_real(perfect_multis)
perfect_multis = perfect_multis[np.abs(perfect_multis['value']) > 1e-2]

# Writes file defining tests
string = np.array2string(occ_full.flatten(), precision=10, separator='\n').replace('[ ', '').replace(']', '')
string += '\n####\n'
string += perfect_multis.to_csv()
with open('t2g_occupied.txt', 'w') as file:
    file.write(string)
