/**
 *  @file fatwheels/gpu.cu
 *  @author Ash Vardanian
 *  @date August 10, 2024
 *  @brief @c fatwheels._cuda and @c fatwheels._hip: one source that NVCC builds for NVIDIA and ROCm's Clang for AMD.
 *
 *  Kernels, launches and Thrust read the same under both vendors, so the only vendor-specific code is the header
 *  HIP needs, @c synchronize, @c device_count, and the module's name.
 */
#include <cstddef> // `std::size_t`
#include <cstring> // `std::strcmp`

#include <exception> // `std::exception`
#include <stdexcept> // `std::runtime_error`

#if defined(__HIP__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "python.hpp"
#include "types.hpp"

namespace fw = ashvardanian::fatwheels;

namespace ashvardanian::fatwheels {

/** Waits for the launched kernels, rethrowing a launch or execution error. */
static void synchronize() {
#if defined(__HIP__)
    hipError_t error = hipGetLastError();
    if (error == hipSuccess) error = hipDeviceSynchronize();
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
#else
    cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
#endif
}

/** The number of devices this vendor's runtime sees, zero without a driver. */
static int device_count() noexcept {
    int count = 0;
#if defined(__HIP__)
    if (hipGetDeviceCount(&count) != hipSuccess) return 0;
#else
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
#endif
    return count;
}

/**
 *  @brief GPU kernel for matrix multiplication with support for strided matrices.
 *
 *  This kernel computes the matrix product of two matrices A and B, storing the result in matrix C.
 *  The multiplication is performed in tiles of size @c tile_size to take advantage of shared memory
 *  for optimizing memory access patterns.
 *
 *  @tparam scalar_type The data type of the matrix elements (e.g., float, double).
 *  @tparam tile_size The size of the tiles used for shared memory, defaulting to 16.
 *
 *  @param matrix_a Pointer to the input matrix A, stored in row-major order.
 *  @param matrix_b Pointer to the input matrix B, stored in row-major order.
 *  @param matrix_c Pointer to the output matrix C, stored in row-major order.
 *  @param num_rows_a The number of rows in matrix A.
 *  @param num_cols_b The number of columns in matrix B.
 *  @param num_cols_a The number of columns in matrix A, and the number of rows in matrix B.
 *  @param stride_a The stride (leading dimension) of matrix A.
 *  @param stride_b The stride (leading dimension) of matrix B.
 *  @param stride_c The stride (leading dimension) of matrix C.
 *
 *  Each thread block computes a tile of the output matrix C by iterating over the corresponding
 *  tiles of matrices A and B. The kernel ensures correct handling of matrix boundaries and
 *  supports strided matrices, where elements of a row are not necessarily contiguous in memory.
 */
template <typename scalar_type, cell_idx_t tile_size = 16>                                        //
__global__ void matmul_kernel(                                                                    //
    scalar_type const *matrix_a, scalar_type const *matrix_b, matmul_type<scalar_type> *matrix_c, //
    cell_idx_t num_rows_a, cell_idx_t num_cols_b, cell_idx_t num_cols_a,                          //
    cell_idx_t stride_a, cell_idx_t stride_b, cell_idx_t stride_c) {

    // Allocate shared memory for matrix_a and matrix_b tiles
    __shared__ scalar_type tile_a[tile_size][tile_size];
    __shared__ scalar_type tile_b[tile_size][tile_size];

    // Calculate the row and column index for this thread in the output matrix matrix_c
    cell_idx_t row = blockIdx.y * tile_size + threadIdx.y;
    cell_idx_t col = blockIdx.x * tile_size + threadIdx.x;

    // Accumulate the result for matrix_c[row][col]
    matmul_type<scalar_type> cell_c = 0;

    // Loop over tiles of matrix_a and matrix_b that are multiplied together
    for (cell_idx_t t = 0; t < (num_cols_a + tile_size - 1) / tile_size; ++t) {

        // Load tiles of matrix_a and matrix_b into shared memory with boundary checks
        tile_a[threadIdx.y][threadIdx.x] = //
            (row < num_rows_a && t * tile_size + threadIdx.x < num_cols_a)
                ? matrix_a[row * stride_a + t * tile_size + threadIdx.x]
                : 0;
        tile_b[threadIdx.y][threadIdx.x] = //
            (col < num_cols_b && t * tile_size + threadIdx.y < num_cols_a)
                ? matrix_b[(t * tile_size + threadIdx.y) * stride_b + col]
                : 0;

        // Synchronize to ensure all data is loaded into shared memory
        __syncthreads();

#pragma unroll
        // Perform the multiplication and accumulate
        for (cell_idx_t k = 0; k < tile_size; ++k) {
            cell_c += static_cast<matmul_type<scalar_type>>(tile_a[threadIdx.y][k]) * tile_b[k][threadIdx.x];
        }

        // Synchronize to ensure all threads are done with the current tile
        __syncthreads();
    }

    // Write the result back to the output matrix matrix_c with boundary check
    if (row < num_rows_a && col < num_cols_b) matrix_c[row * stride_c + col] = cell_c;
}

template <typename scalar_type>
using matmul_kernel_pointer = void (*)(scalar_type const *, scalar_type const *, matmul_type<scalar_type> *, cell_idx_t,
                                       cell_idx_t, cell_idx_t, cell_idx_t, cell_idx_t, cell_idx_t);

template <typename scalar_type>
static matmul_kernel_pointer<scalar_type> matmul_kernel_for(cell_idx_t tile_size) {
    switch (tile_size) {
    case 4: return &matmul_kernel<scalar_type, 4>;
    case 8: return &matmul_kernel<scalar_type, 8>;
    case 16: return &matmul_kernel<scalar_type, 16>;
    case 32: return &matmul_kernel<scalar_type, 32>;
    default: throw std::runtime_error("Unsupported tile size - choose from 4, 8, 16, and 32");
    }
}

/** Runs @p body, turning the C++ exceptions Thrust and @c synchronize raise into a @c RuntimeError. */
template <typename body_type>
static PyObject *with_runtime_errors(body_type &&body) noexcept {
    try {
        return body();
    }
    catch (std::exception const &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

/** Views a C-contiguous rank-2 buffer, leaving a @c ValueError for any other rank. */
static bool matrix_from_python(Py_buffer const &view, char const *label, cell_idx_t &rows,
                               cell_idx_t &columns) noexcept {
    if (view.ndim != 2) {
        PyErr_Format(PyExc_ValueError, "%s must be rank-2, not rank-%d", label, view.ndim);
        return false;
    }
    if (view.shape[0] * view.shape[1] >= Py_ssize_t {1} << 32) {
        PyErr_Format(PyExc_ValueError, "%s spans 2^32 elements or more, past what 32-bit indices reach", label);
        return false;
    }
    rows = static_cast<cell_idx_t>(view.shape[0]);
    columns = static_cast<cell_idx_t>(view.shape[1]);
    return true;
}

template <typename scalar_type>
static PyObject *reduce_typed(Py_buffer const &view) {
    scalar_type const *host = static_cast<scalar_type const *>(view.buf);
    thrust::device_vector<scalar_type> const device(host, host + view.shape[0]);
    return scalar_to_python(thrust::reduce(thrust::device, device.begin(), device.end(), reduce_type<scalar_type> {0}));
}

template <typename scalar_type>
static PyObject *matmul_typed(Py_buffer const &view_a, Py_buffer const &view_b, cell_idx_t num_rows_a,
                              cell_idx_t num_cols_a, cell_idx_t num_cols_b, cell_idx_t tile_size) {
    using result_type = matmul_type<scalar_type>;
    matmul_kernel_pointer<scalar_type> const kernel = matmul_kernel_for<scalar_type>(tile_size);
    scalar_type const *host_a = static_cast<scalar_type const *>(view_a.buf);
    scalar_type const *host_b = static_cast<scalar_type const *>(view_b.buf);
    thrust::device_vector<scalar_type> const device_a(host_a,
                                                      host_a + static_cast<std::size_t>(num_rows_a) * num_cols_a);
    thrust::device_vector<scalar_type> const device_b(host_b,
                                                      host_b + static_cast<std::size_t>(num_cols_a) * num_cols_b);
    thrust::device_vector<result_type> device_c(static_cast<std::size_t>(num_rows_a) * num_cols_b);

    dim3 const block(tile_size, tile_size);
    dim3 const grid((num_cols_b + tile_size - 1) / tile_size, (num_rows_a + tile_size - 1) / tile_size);
    kernel<<<grid, block>>>(thrust::raw_pointer_cast(device_a.data()), thrust::raw_pointer_cast(device_b.data()),
                            thrust::raw_pointer_cast(device_c.data()), num_rows_a, num_cols_b, num_cols_a, num_cols_a,
                            num_cols_b, num_cols_b);
    synchronize();

    PyObject *result = new_matrix<result_type>(num_rows_a, num_cols_b);
    if (!result) return nullptr;
    buffer_t output;
    if (!output.acquire(result, PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE)) {
        Py_DECREF(result);
        return nullptr;
    }
    thrust::copy(device_c.begin(), device_c.end(), static_cast<result_type *>(output.view.buf));
    return result;
}

#pragma region Entry Points

static char const doc_device_count[] =                          //
    "device_count()\n"                                          //
    "\n"                                                        //
    "The number of devices this vendor's runtime sees.\n"       //
    "\n"                                                        //
    "Returns:\n"                                                //
    "  int: Zero when the driver is missing or sees no GPU.\n"; //

static PyObject *module_device_count(PyObject *, PyObject *) noexcept { return PyLong_FromLong(device_count()); }

static char const doc_reduce[] =                                          //
    "reduce(a)\n"                                                         //
    "\n"                                                                  //
    "Sum of a contiguous 1D array, on the first GPU.\n"                   //
    "\n"                                                                  //
    "Args:\n"                                                             //
    "  a: Any buffer of floats or integers, like a NumPy array.\n"        //
    "\n"                                                                  //
    "Returns:\n"                                                          //
    "  float or int: Accumulated in float64, int64 or uint64.\n"          //
    "\n"                                                                  //
    "Raises:\n"                                                           //
    "  TypeError: If a is not a buffer of a supported type.\n"            //
    "  ValueError: If a is not 1D.\n"                                     //
    "  RuntimeError: If the device fails to allocate, copy or reduce.\n"; //

static PyObject *module_reduce(PyObject *, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "reduce() takes exactly one argument");
        return nullptr;
    }
    buffer_t input;
    if (!input.acquire(args[0], PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)) return nullptr;
    if (input.view.ndim != 1) {
        PyErr_Format(PyExc_ValueError, "reduce() takes a 1D array, not rank-%d", input.view.ndim);
        return nullptr;
    }
    return dispatch_dtype(input.view, [&]<typename scalar_type>() -> PyObject * {
        return with_runtime_errors([&] { return reduce_typed<scalar_type>(input.view); });
    });
}

static char const doc_matmul[] =                                                                   //
    "matmul(a, b, *, tile_size=16)\n"                                                              //
    "\n"                                                                                           //
    "Product of two matrices, tile by tile on the first GPU.\n"                                    //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  a: A C-contiguous rank-2 buffer, like a NumPy array.\n"                                     //
    "  b: A C-contiguous rank-2 buffer of the same type, with as many rows as a has columns.\n"    //
    "  tile_size: The tile edge, one of 4, 8, 16 and 32.\n"                                        //
    "\n"                                                                                           //
    "Returns:\n"                                                                                   //
    "  numpy.ndarray: Of a's floating type, or of int64 or uint64 for integers.\n"                 //
    "\n"                                                                                           //
    "Raises:\n"                                                                                    //
    "  TypeError: If a or b is not a buffer of a supported type, or they differ in type.\n"        //
    "  ValueError: For mismatched shapes, a non-contiguous matrix, or an unsupported tile size.\n" //
    "  RuntimeError: If the device fails to allocate, copy or multiply.\n";                        //

static PyObject *module_matmul(PyObject *, PyObject *const *args, Py_ssize_t count, PyObject *keyword_names) noexcept {
    if (count != 2) {
        PyErr_SetString(PyExc_TypeError, "matmul() takes exactly two positional arguments");
        return nullptr;
    }
    PyObject *tile_size_object = nullptr;
    if (!keywords_from_python("matmul", args, count, keyword_names, {{"tile_size", &tile_size_object}})) return nullptr;
    Py_ssize_t const tile_size = tile_size_object ? PyLong_AsSsize_t(tile_size_object) : 16;
    if (tile_size == -1 && PyErr_Occurred()) return nullptr;
    if (tile_size != 4 && tile_size != 8 && tile_size != 16 && tile_size != 32) {
        PyErr_Format(PyExc_ValueError, "Unsupported tile size %zd - choose from 4, 8, 16, and 32", tile_size);
        return nullptr;
    }

    buffer_t buffer_a, buffer_b;
    if (!buffer_a.acquire(args[0], PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)) return nullptr;
    if (!buffer_b.acquire(args[1], PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)) return nullptr;
    Py_buffer const &view_a = buffer_a.view, &view_b = buffer_b.view;
    if (view_a.itemsize != view_b.itemsize || std::strcmp(view_a.format, view_b.format) != 0) {
        PyErr_Format(PyExc_TypeError, "matmul() takes matrices of one type, not '%s' and '%s'", view_a.format,
                     view_b.format);
        return nullptr;
    }
    cell_idx_t num_rows_a, num_cols_a, num_rows_b, num_cols_b;
    if (!matrix_from_python(view_a, "a", num_rows_a, num_cols_a)) return nullptr;
    if (!matrix_from_python(view_b, "b", num_rows_b, num_cols_b)) return nullptr;
    if (num_cols_a != num_rows_b) {
        PyErr_Format(PyExc_ValueError, "a has %u columns, but b has %u rows", num_cols_a, num_rows_b);
        return nullptr;
    }
    return dispatch_dtype(view_a, [&]<typename scalar_type>() -> PyObject * {
        return with_runtime_errors([&] {
            return matmul_typed<scalar_type>(view_a, view_b, num_rows_a, num_cols_a, num_cols_b,
                                             static_cast<cell_idx_t>(tile_size));
        });
    });
}

static PyMethodDef module_methods[] = {
    {"device_count", module_device_count, METH_NOARGS, doc_device_count},
    {"reduce", as_pycfunction(module_reduce), METH_FASTCALL, doc_reduce},
    {"matmul", as_pycfunction(module_matmul), METH_FASTCALL | METH_KEYWORDS, doc_matmul},
    {nullptr, nullptr, 0, nullptr},
};

#pragma endregion Entry Points

#pragma region Initialization

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
#if defined(Py_mod_gil) // ? Only 3.13 and later know about the GIL slot
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, nullptr},
};

#if defined(__HIP__)
static char const module_name[] = "_hip";
#else
static char const module_name[] = "_cuda";
#endif

static struct PyModuleDef gpu_module = {
    PyModuleDef_HEAD_INIT,
    module_name,
    "GPU kernels compiled from one source for NVIDIA and AMD.",
    0,
    module_methods,
    module_slots,
    nullptr,
    nullptr,
    nullptr,
};

#pragma endregion Initialization

} // namespace ashvardanian::fatwheels

#if defined(__HIP__)
extern "C" PyMODINIT_FUNC PyInit__hip(void) { return PyModuleDef_Init(&fw::gpu_module); }
#else
extern "C" PyMODINIT_FUNC PyInit__cuda(void) { return PyModuleDef_Init(&fw::gpu_module); }
#endif
