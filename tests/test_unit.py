import numpy as np

from multipyles.multipole_eqs import chi, SIGMA
from multipyles.helper import minus_one_to_the

def test_minus_one_to_the():
    assert minus_one_to_the(0) == 1
    assert minus_one_to_the(1) == -1

def test_chi_p0():
    """ Checks that for p=0, the spin matrix chi is unity. """
    s_range = (+SIGMA, -SIGMA)
    for s_a in s_range:
        for s_b in s_range:
            assert np.isclose(chi(0, 0, s_a, s_b), int(s_a == s_b)), (s_a, s_b)
