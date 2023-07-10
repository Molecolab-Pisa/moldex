#ifndef _MOLDEX_RECURSIVE_HERMITE_H
#define _MOLDEX_RECURSIVE_HERMITE_H

#include <cmath>

namespace moldex {

#ifdef __CUDACC__
#define MOLDEX_INLINE_OR_DEVICE __host__ __device__
#else
#define MOLDEX_INLINE_OR_DEVICE inline
#endif

template <typename T>
MOLDEX_INLINE_OR_DEVICE T recursive_hermite_coefficient(int i, int j, int t,T  Qx, T a, T b) {
    T p = a + b;
    T q = a * b / p;
    if (t < 0 || t > i + j) {
        return 0.0;
    }
    else if (i == 0 && j == 0 && t == 0) {
        return (T) exp(- q * Qx * Qx);
    }
    else if (j == 0) {
        return (1/(2*p)) * recursive_hermite_coefficient(i-1, j, t-1, Qx, a, b) -
               (q*Qx/a)  * recursive_hermite_coefficient(i-1, j, t, Qx, a, b)   +
               (t+1)     * recursive_hermite_coefficient(i-1, j, t+1, Qx, a, b);
    }
    else {
        return (1/(2*p)) * recursive_hermite_coefficient(i, j-1, t-1, Qx, a, b) +
               (q*Qx/b)  * recursive_hermite_coefficient(i, j-1, t, Qx, a, b)   +
               (t+1)     * recursive_hermite_coefficient(i, j-1, t+1, Qx, a, b);
    }
}

} // namespace moldex


#endif
