#include "retain_full_residues.h"
#include "recursive_hermite.h"
#include "pybind11_kernel_helpers.h"

using namespace moldex;

namespace {

// template <typename T>
// void cpu_recursive_hermite(void *out, const void **in) {
//     //input
//     const std::int64_t i = *(reinterpret_cast<const std::int64_t *>(in[0]));
//     const std::int64_t j = *(reinterpret_cast<const std::int64_t *>(in[1]));
//     const std::int64_t t = *(reinterpret_cast<const std::int64_t *>(in[2]));
//     const T Qx = *(reinterpret_cast<const T *>(in[3]));
//     const T a = *(reinterpret_cast<const T *>(in[4]));
//     const T b = *(reinterpret_cast<const T *>(in[5]));
// 
//     // output
//     T *result = reinterpret_cast<T *>(out);
// 
//     result[0] = recursive_hermite_coefficient(i, j, t, Qx, a, b);
// }

template <typename T>
void cpu_recursive_hermite(void *out, const void **in) {
    const std::int64_t *i = reinterpret_cast<const std::int64_t *>(in[0]);
    const std::int64_t *j = reinterpret_cast<const std::int64_t *>(in[1]);
    const std::int64_t *t = reinterpret_cast<const std::int64_t *>(in[2]);
    const T *Qx = reinterpret_cast<const T *>(in[3]);
    const T *a = reinterpret_cast<const T *>(in[4]);
    const T *b = reinterpret_cast<const T *>(in[5]);

    const std::int64_t i_size = *(reinterpret_cast<const std::int64_t *>(in[6]));
    const std::int64_t j_size = *(reinterpret_cast<const std::int64_t *>(in[7]));
    const std::int64_t t_size = *(reinterpret_cast<const std::int64_t *>(in[8]));
    const std::int64_t Qx_size = *(reinterpret_cast<const std::int64_t *>(in[9]));
    const std::int64_t a_size = *(reinterpret_cast<const std::int64_t *>(in[10]));
    const std::int64_t b_size = *(reinterpret_cast<const std::int64_t *>(in[11]));

    T *result = reinterpret_cast<T *>(out);
    std::int64_t c = 0;

    for (std::int64_t ii = 0; ii < i_size; ii++)
        for (std::int64_t jj = 0; jj < j_size; jj++)
            for (std::int64_t tt = 0; tt < t_size; tt++)
                for (std::int64_t Qxx = 0; Qxx < Qx_size; Qxx++)
                    for (std::int64_t aa = 0; aa < a_size; aa++)
                        for (std::int64_t bb = 0; bb < b_size; bb++) {
                            result[c] = recursive_hermite_coefficient(i[ii], j[jj], t[tt], Qx[Qxx], a[aa], b[bb]);
                            c++;                            
                        }
}

template <typename T>
void cpu_retain_full_residues(void *out, const void **in) {
    const T *idx = reinterpret_cast<const T *>(in[0]);
    const T *residues_array = reinterpret_cast<const T *>(in[1]);
    const T idx_max = *(reinterpret_cast<const T *>(in[2]));
    const T n_resids = *(reinterpret_cast<const T *>(in[3]));

    T *result = reinterpret_cast<T *>(out);

    retain_full_residues(idx, residues_array, idx_max, n_resids, result);
}


pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["cpu_recursive_hermite_f32"] = EncapsulateFunction(cpu_recursive_hermite<float>);
    dict["cpu_recursive_hermite_f64"] = EncapsulateFunction(cpu_recursive_hermite<double>);
    dict["cpu_retain_full_residues_i32"] = EncapsulateFunction(cpu_retain_full_residues<std::int32_t>);
    dict["cpu_retain_full_residues_i64"] = EncapsulateFunction(cpu_retain_full_residues<std::int64_t>);
    return dict;
}

PYBIND11_MODULE(cpu_ops, m) { m.def("registrations", &Registrations); }

} // namespace
