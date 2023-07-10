#ifndef _MOLDEX_RETAIN_FULL_RESIDUES_H
#define _MOLDEX_RETAIN_FULL_RESIDUES_H

#include <cmath>

namespace moldex {

#ifdef __CUDACC__
#define MOLDEX_INLINE_OR_DEVICE __host__ __device__
#else
#define MOLDEX_INLINE_OR_DEVICE inline
#endif

template <typename T>
MOLDEX_INLINE_OR_DEVICE void retain_full_residues(const T *idx, const T *residues_array, const T idx_max, const T n_resids, T *out) {
    T i, j, k, start, stop, val, fill;  

    for (i = 0; i < n_resids; i++) {
        val = 0;
        start = residues_array[i];      
        stop = residues_array[i+1];     

        for (j = start; j < stop; j++) {
            val += idx[j];
        }

        if (val > 0.5)
            fill = 1;
        else
            fill = 0;

        for (k = start; k < stop; k++) {
            out[k] = fill;
        }
    }
}

} // namespace moldex

#endif
