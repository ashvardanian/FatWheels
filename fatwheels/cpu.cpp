/**
 *  @file fatwheels/cpu.cpp
 *  @author Ash Vardanian
 *  @date August 10, 2024
 *  @brief @c fatwheels._cpu: the CPU kernels over buffers, at the best capability or @c serial.
 *
 *  Multi-phase initialization declares the module free of the GIL and safe for subinterpreters: the
 *  kernels keep no state, and each call releases the GIL while it computes.
 */
#include "fatwheels/python.hpp"

#include <cstddef> // `std::size_t`
#include <cstring> // `std::strcmp`
#include <span>    // `std::span`

#include "fatwheels/cpu.hpp"

namespace fw = ashvardanian::fatwheels;

namespace ashvardanian::fatwheels::python {

#pragma region Arguments

/** Reads @p name, @c None or absent for the best capability, leaving a @c ValueError for one this CPU lacks. */
static bool capability_from_python(PyObject *name, capability_t &capability) noexcept {
    capability = capability_best();
    if (!name || name == Py_None) return true;
    if (!PyUnicode_Check(name)) {
        PyErr_Format(PyExc_TypeError, "capability must be a string or None, not %R", name);
        return false;
    }
    if (PyUnicode_CompareWithASCIIString(name, capability_name(capability)) == 0) return true;
    if (PyUnicode_CompareWithASCIIString(name, "serial") == 0) {
        capability = capability_t::serial_k;
        return true;
    }
    PyErr_Format(PyExc_ValueError, "This CPU does not run the %R kernels", name);
    return false;
}

/** Views a rank-2 buffer whose rows are contiguous, leaving a @c ValueError for any other layout. */
static bool rows_from_python(Py_buffer const &view, char const *label, cell_idx_t &rows, cell_idx_t &columns,
                             cell_idx_t &stride) noexcept {
    if (view.ndim != 2) {
        PyErr_Format(PyExc_ValueError, "%s must be rank-2, not rank-%d", label, view.ndim);
        return false;
    }
    if (view.strides[1] != view.itemsize || view.strides[0] <= 0 || view.strides[0] % view.itemsize != 0) {
        PyErr_Format(PyExc_ValueError, "%s must have contiguous rows", label);
        return false;
    }
    rows = static_cast<cell_idx_t>(view.shape[0]);
    columns = static_cast<cell_idx_t>(view.shape[1]);
    stride = static_cast<cell_idx_t>(view.strides[0] / view.itemsize);
    return true;
}

#pragma endregion Arguments

#pragma region Entry Points

static char const doc_capabilities[] =                                          //
    "capabilities()\n"                                                          //
    "\n"                                                                        //
    "The CPU capabilities this machine runs kernels for, the best one first.\n" //
    "\n"                                                                        //
    "Returns:\n"                                                                //
    "  list[str]: Like ['haswell', 'serial'], or ['serial'] alone.\n";          //

static PyObject *module_capabilities(PyObject *, PyObject *) noexcept {
    capability_t const best = capability_best();
    if (best == capability_t::serial_k) return Py_BuildValue("[s]", capability_name(best));
    return Py_BuildValue("[ss]", capability_name(best), capability_name(capability_t::serial_k));
}

static char const doc_reduce[] =                                                      //
    "reduce(a, *, capability=None)\n"                                                 //
    "\n"                                                                              //
    "Sum of a contiguous 1D array, across every CPU core.\n"                          //
    "\n"                                                                              //
    "Args:\n"                                                                         //
    "  a: Any buffer of floats or integers, like a NumPy array.\n"                    //
    "  capability: 'serial', or None for the best one this CPU runs.\n"               //
    "\n"                                                                              //
    "Returns:\n"                                                                      //
    "  float or int: Accumulated in float64, int64 or uint64.\n"                      //
    "\n"                                                                              //
    "Raises:\n"                                                                       //
    "  TypeError: If a is not a buffer of a supported type.\n"                        //
    "  ValueError: If a is not 1D, or this CPU does not run the named capability.\n"; //

static PyObject *module_reduce(PyObject *, PyObject *const *args, Py_ssize_t count, PyObject *keyword_names) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "reduce() takes exactly one positional argument");
        return nullptr;
    }
    PyObject *capability_object = nullptr;
    if (!keywords_from_python("reduce", args, count, keyword_names, {{"capability", &capability_object}}))
        return nullptr;
    capability_t capability;
    if (!capability_from_python(capability_object, capability)) return nullptr;

    buffer_t input;
    if (!input.acquire(args[0], PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)) return nullptr;
    if (input.view.ndim != 1) {
        PyErr_Format(PyExc_ValueError, "reduce() takes a 1D array, not rank-%d", input.view.ndim);
        return nullptr;
    }
    return dispatch_dtype(input.view, [&]<typename scalar_type>() -> PyObject * {
        std::span<scalar_type const> const data(static_cast<scalar_type const *>(input.view.buf),
                                                static_cast<std::size_t>(input.view.shape[0]));
        reduce_type<scalar_type> sum;
        Py_BEGIN_ALLOW_THREADS;
        sum = fw::reduce(data, capability);
        Py_END_ALLOW_THREADS;
        return scalar_to_python(sum);
    });
}

static char const doc_matmul[] =                                                              //
    "matmul(a, b, *, tile_size=16, capability=None)\n"                                        //
    "\n"                                                                                      //
    "Product of two matrices, tile by tile across every CPU core.\n"                          //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  a: A rank-2 buffer with contiguous rows, like a NumPy array or a slice of its rows.\n" //
    "  b: A rank-2 buffer of the same type, with as many rows as a has columns.\n"            //
    "  tile_size: The tile edge, one of 4, 8, 16, 32 and 64.\n"                               //
    "  capability: 'serial', or None for the best one this CPU runs.\n"                       //
    "\n"                                                                                      //
    "Returns:\n"                                                                              //
    "  numpy.ndarray: Of a's floating type, or of int64 or uint64 for integers.\n"            //
    "\n"                                                                                      //
    "Raises:\n"                                                                               //
    "  TypeError: If a or b is not a buffer of a supported type, or they differ in type.\n"   //
    "  ValueError: For mismatched shapes, strided columns, an unsupported tile size, or a\n"  //
    "    capability this CPU does not run.\n";                                                //

static PyObject *module_matmul(PyObject *, PyObject *const *args, Py_ssize_t count, PyObject *keyword_names) noexcept {
    if (count != 2) {
        PyErr_SetString(PyExc_TypeError, "matmul() takes exactly two positional arguments");
        return nullptr;
    }
    PyObject *tile_size_object = nullptr;
    PyObject *capability_object = nullptr;
    if (!keywords_from_python("matmul", args, count, keyword_names,
                              {{"tile_size", &tile_size_object}, {"capability", &capability_object}}))
        return nullptr;
    capability_t capability;
    if (!capability_from_python(capability_object, capability)) return nullptr;
    Py_ssize_t const tile_size = tile_size_object ? PyLong_AsSsize_t(tile_size_object) : 16;
    if (tile_size == -1 && PyErr_Occurred()) return nullptr;

    buffer_t buffer_a, buffer_b;
    if (!buffer_a.acquire(args[0], PyBUF_STRIDES | PyBUF_FORMAT)) return nullptr;
    if (!buffer_b.acquire(args[1], PyBUF_STRIDES | PyBUF_FORMAT)) return nullptr;
    Py_buffer const &view_a = buffer_a.view, &view_b = buffer_b.view;
    if (view_a.itemsize != view_b.itemsize || std::strcmp(view_a.format, view_b.format) != 0) {
        PyErr_Format(PyExc_TypeError, "matmul() takes matrices of one type, not '%s' and '%s'", view_a.format,
                     view_b.format);
        return nullptr;
    }
    cell_idx_t num_rows_a, num_cols_a, stride_a, num_rows_b, num_cols_b, stride_b;
    if (!rows_from_python(view_a, "a", num_rows_a, num_cols_a, stride_a)) return nullptr;
    if (!rows_from_python(view_b, "b", num_rows_b, num_cols_b, stride_b)) return nullptr;
    if (num_cols_a != num_rows_b) {
        PyErr_Format(PyExc_ValueError, "a has %u columns, but b has %u rows", num_cols_a, num_rows_b);
        return nullptr;
    }

    return dispatch_dtype(view_a, [&]<typename scalar_type>() -> PyObject * {
        using result_type = matmul_type<scalar_type>;
        PyObject *result = new_matrix<result_type>(num_rows_a, num_cols_b);
        if (!result) return nullptr;
        buffer_t output;
        if (!output.acquire(result, PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE)) {
            Py_DECREF(result);
            return nullptr;
        }
        matmul_operands<scalar_type> const operands {static_cast<scalar_type const *>(view_a.buf),
                                                     static_cast<scalar_type const *>(view_b.buf),
                                                     static_cast<result_type *>(output.view.buf),
                                                     num_rows_a,
                                                     num_cols_b,
                                                     num_cols_a,
                                                     stride_a,
                                                     stride_b,
                                                     num_cols_b};
        bool supported = true;
        Py_BEGIN_ALLOW_THREADS;
        switch (tile_size) {
        case 4: fw::matmul<scalar_type, 4>(operands, capability); break;
        case 8: fw::matmul<scalar_type, 8>(operands, capability); break;
        case 16: fw::matmul<scalar_type, 16>(operands, capability); break;
        case 32: fw::matmul<scalar_type, 32>(operands, capability); break;
        case 64: fw::matmul<scalar_type, 64>(operands, capability); break;
        default: supported = false;
        }
        Py_END_ALLOW_THREADS;
        if (!supported) {
            Py_DECREF(result);
            PyErr_Format(PyExc_ValueError, "Unsupported tile size %zd - choose from 4, 8, 16, 32, and 64", tile_size);
            return nullptr;
        }
        return result;
    });
}

static PyMethodDef module_methods[] = {
    {"capabilities", module_capabilities, METH_NOARGS, doc_capabilities},
    {"reduce", as_pycfunction(module_reduce), METH_FASTCALL | METH_KEYWORDS, doc_reduce},
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

static struct PyModuleDef cpu_module = {
    PyModuleDef_HEAD_INIT,
    "_cpu",
    "CPU kernels compiled once per capability and picked at runtime.",
    0,
    module_methods,
    module_slots,
    nullptr,
    nullptr,
    nullptr,
};

#pragma endregion Initialization

} // namespace ashvardanian::fatwheels::python

extern "C" PyMODINIT_FUNC PyInit__cpu(void) { return PyModuleDef_Init(&fw::python::cpu_module); }
