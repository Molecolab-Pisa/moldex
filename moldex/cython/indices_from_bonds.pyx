import cython
cimport numpy as np
import numpy as np

@cython.wraparound(False)
@cython.boundscheck(False)
cdef list _angle_indices_from_bonds(int[:, ::1] bond_indices):
    cdef int n_bonds = bond_indices.shape[0]
    cdef list angle_indices = []

    cdef int i, j

    cdef int ai, aj, am, an, ak
    cdef bint ai_in_bond, aj_in_bond

    for i in range(n_bonds):
        ai = bond_indices[i, 0]
        aj = bond_indices[i, 1]
        for j in range(n_bonds):
            am = bond_indices[j, 0]
            an = bond_indices[j, 1]
            ai_in_bond = ai == am or ai == an
            aj_in_bond = aj == am or aj == an
            if ai_in_bond and not aj_in_bond:
                if ai == am:
                    ak = an
                else:
                    ak = am
                angle_indices.append([ak, ai, aj])
            elif aj_in_bond and not ai_in_bond:
                if aj == am:
                    ak = an
                else:
                    ak = am
                angle_indices.append([ai, aj, ak])
    return angle_indices


@cython.wraparound(False)
@cython.boundscheck(False)
cdef list _dihe_indices_from_bonds_angles(int[:, ::1] bond_indices, int[:, ::1] angle_indices):
    cdef int n_bonds = bond_indices.shape[0]
    cdef int n_angles = angle_indices.shape[0]
    cdef list dihe_indices = []

    cdef int i, j
    cdef int ai, aj, ak, am, an, al
    cdef bint ai_in_bond, aj_in_bond, ak_in_bond

    for i in range(n_angles):
        ai = angle_indices[i, 0]
        aj = angle_indices[i, 1]
        ak = angle_indices[i, 2]
        for j in range(n_bonds):
            am = bond_indices[j, 0]
            an = bond_indices[j, 1]
            ai_in_bond = ai == am or ai == an
            aj_in_bond = aj == am or aj == an
            ak_in_bond = ak == am or ak == an
            if ai_in_bond and not aj_in_bond and not ak_in_bond:
                if ai == am:
                    al = an
                else:
                    al = am
                dihe_indices.append([al, ai, aj, ak])
            elif ak_in_bond and not ai_in_bond and not aj_in_bond:
                if ak == am:
                    al = an
                else:
                    al = am
                dihe_indices.append([ai, aj, ak, al])
    return dihe_indices


def angle_indices_from_bonds_cy(bond_indices):
    return _angle_indices_from_bonds(np.array(bond_indices, dtype=np.intc, order="C"))

def dihe_indices_from_bonds_angles_cy(bond_indices, angle_indices):
    return _dihe_indices_from_bonds_angles(
        np.array(bond_indices, dtype=np.intc, order="C"),
        np.array(angle_indices, dtype=np.intc, order="C"),
    )
