/**
 *  @file python/cpu.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief `fatwheels._cpu`: the library's CPU `matmul` over float32 buffers, at the best capability or a named one.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h> // `strcmp`

#include "fatwheels/fatwheels.h"

static struct {
    char const *name;
    fw_capability_t capability;
} const capabilities_named[] = {
    {"sapphireamx", fw_cap_sapphireamx_k},
    {"skylake", fw_cap_skylake_k},
    {"haswell", fw_cap_haswell_k},
    {"sme", fw_cap_sme_k},
    {"sve", fw_cap_sve_k},
    {"serial", fw_cap_serial_k},
};

static PyObject *module_capabilities(PyObject *self, PyObject *unused) {
    (void)self, (void)unused;
    PyObject *names = PyList_New(0);
    if (!names) return NULL;
    for (size_t i = 0; i != sizeof(capabilities_named) / sizeof(capabilities_named[0]); ++i)
        if (fw_matmul_f32_find(capabilities_named[i].capability)) {
            PyObject *name = PyUnicode_FromString(capabilities_named[i].name);
            if (!name || PyList_Append(names, name) != 0) return Py_XDECREF(name), Py_DECREF(names), NULL;
            Py_DECREF(name);
        }
    return names;
}

/** Views a C-contiguous rank-2 float32 buffer, leaving a `TypeError` or `ValueError` otherwise. */
static int matrix_from_python(PyObject *object, Py_buffer *view, char const *label) {
    if (PyObject_GetBuffer(object, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) return 0;
    if (view->ndim != 2 || view->itemsize != 4 || strcmp(view->format, "f") != 0) {
        PyErr_Format(PyExc_TypeError, "%s must be a rank-2 float32 buffer", label);
        PyBuffer_Release(view);
        return 0;
    }
    return 1;
}

static PyObject *module_matmul(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *keywords[] = {"a", "b", "capability", NULL};
    PyObject *a_object, *b_object;
    char const *name = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|z", keywords, &a_object, &b_object, &name)) return NULL;

    // Without a name, every capability: the table picks the best one this build holds and this machine runs.
    fw_capability_t capabilities = (fw_capability_t)(fw_cap_cuda_k - 1);
    for (size_t i = 0; name && i != sizeof(capabilities_named) / sizeof(capabilities_named[0]); ++i)
        if (strcmp(name, capabilities_named[i].name) == 0) capabilities = capabilities_named[i].capability;
    if (!fw_matmul_f32_find(capabilities))
        return PyErr_Format(PyExc_ValueError, "No kernel for capability '%s' in this build on this machine", name);

    Py_buffer a, b;
    if (!matrix_from_python(a_object, &a, "a")) return NULL;
    if (!matrix_from_python(b_object, &b, "b")) return PyBuffer_Release(&a), NULL;
    PyObject *result = NULL;
    if (a.shape[1] != b.shape[0]) {
        PyErr_Format(PyExc_ValueError, "a has %zd columns, but b has %zd rows", a.shape[1], b.shape[0]);
        goto release;
    }
    // Returned as a `bytearray` shaped by the Python wrapper, so this module needs no NumPy headers.
    result = PyByteArray_FromStringAndSize(NULL, a.shape[0] * b.shape[1] * 4);
    if (result)
        fw_matmul_f32_best(a.buf, b.buf, (fw_f32_t *)PyByteArray_AS_STRING(result), a.shape[0], b.shape[1], a.shape[1],
                           capabilities, NULL);
release:
    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    return result;
}

static PyMethodDef module_methods[] = {
    {"capabilities", module_capabilities, METH_NOARGS, "The CPU capabilities this build holds and this machine runs."},
    {"matmul", (PyCFunction)(void (*)(void))module_matmul, METH_VARARGS | METH_KEYWORDS,
     "matmul(a, b, capability=None): C = A * B over float32 matrices, as raw bytes."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_cpu", "FatWheels' CPU kernels.", -1, module_methods};

PyMODINIT_FUNC PyInit__cpu(void) { return PyModule_Create(&module); }
