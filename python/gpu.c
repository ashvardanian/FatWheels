/**
 *  @file python/gpu.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief `fatwheels._gpu`: every GPU vendor this build holds, through its service table and the kernel tables.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h> // `strcmp`

#include "fatwheels/fatwheels.h"

/** Every vendor's group bit, and every generation, best first within its vendor. */
static struct {
    char const *name;
    fw_capability_t capability;
} const vendors[] = {{"cuda", fw_cap_cuda_k}, {"rocm", fw_cap_rocm_k}}, generations[] = {{"ada", fw_cap_ada_k},
                                                                                         {"ampere", fw_cap_ampere_k},
                                                                                         {"cuda", fw_cap_cuda_k},
                                                                                         {"cdna3", fw_cap_cdna3_k},
                                                                                         {"rocm", fw_cap_rocm_k}};

/** The services of the vendor @p name, or `NULL` with a `ValueError` set. */
static fw_device_services_t const *services_named(char const *name) {
    for (size_t i = 0; i != sizeof(vendors) / sizeof(vendors[0]); ++i)
        if (strcmp(name, vendors[i].name) == 0) {
            fw_device_services_t const *services = fw_device_services(vendors[i].capability);
            if (!services) PyErr_Format(PyExc_ValueError, "this build holds no %s kernels", name);
            return services;
        }
    PyErr_Format(PyExc_ValueError, "unknown GPU vendor '%s'", name);
    return NULL;
}

static PyObject *capabilities(PyObject *module, PyObject *args, PyObject *kwargs) {
    (void)module;
    static char *keywords[] = {"vendor", "device", NULL};
    char const *vendor = NULL;
    int device = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|i", keywords, &vendor, &device)) return NULL;
    fw_device_services_t const *services = services_named(vendor);
    if (!services) return NULL;
    fw_capability_t const loaded = services->loaded(device);
    PyObject *names = PyList_New(0);
    for (size_t i = 0; names && i != sizeof(generations) / sizeof(generations[0]); ++i) {
        if (!(loaded & generations[i].capability)) continue;
        PyObject *name = PyUnicode_FromString(generations[i].name);
        if (!name || PyList_Append(names, name)) Py_CLEAR(names);
        Py_XDECREF(name);
    }
    return names;
}

/** Views @p object as a C-contiguous float32 matrix, or raises. */
static int get_matrix(PyObject *object, Py_buffer *view) {
    if (PyObject_GetBuffer(object, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)) return -1;
    if (view->ndim == 2 && view->itemsize == 4 && view->format && strcmp(view->format, "f") == 0) return 0;
    PyErr_SetString(PyExc_TypeError, "expected a C-contiguous 2D float32 buffer");
    PyBuffer_Release(view);
    return -1;
}

static PyObject *matmul(PyObject *module, PyObject *args, PyObject *kwargs) {
    (void)module;
    static char *keywords[] = {"vendor", "a", "b", "capability", NULL};
    char const *vendor = NULL, *name = NULL;
    PyObject *a_object, *b_object;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sOO|z", keywords, &vendor, &a_object, &b_object, &name))
        return NULL;
    fw_device_services_t const *services = services_named(vendor);
    if (!services) return NULL;
    fw_capability_t capabilities = services->loaded(0);
    for (size_t i = 0; name && i != sizeof(generations) / sizeof(generations[0]); ++i)
        if (strcmp(name, generations[i].name) == 0)
            capabilities = (fw_capability_t)(capabilities & generations[i].capability);
    if (!fw_matmul_f32_find(capabilities))
        return PyErr_Format(PyExc_ValueError, "no %s generation loads on device 0", name ? name : vendor);

    Py_buffer a, b;
    if (get_matrix(a_object, &a)) return NULL;
    if (get_matrix(b_object, &b)) return PyBuffer_Release(&a), NULL;
    fw_size_t const rows = a.shape[0], depth = a.shape[1], columns = b.shape[1];
    PyObject *bytes = NULL;
    if (b.shape[0] != a.shape[1]) {
        PyErr_SetString(PyExc_ValueError, "the columns of a must match the rows of b");
        goto release;
    }
    bytes = PyByteArray_FromStringAndSize(NULL, (Py_ssize_t)(rows * columns * sizeof(fw_f32_t)));
    if (!bytes) goto release;
    void *device_a = NULL, *device_b = NULL, *device_c = NULL;
    fw_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    status = services->allocate(&device_a, a.len);
    if (status == fw_success_k) status = services->allocate(&device_b, b.len);
    if (status == fw_success_k) status = services->allocate(&device_c, PyByteArray_GET_SIZE(bytes));
    if (status == fw_success_k) status = services->copy_to_device(device_a, a.buf, a.len, NULL);
    if (status == fw_success_k) status = services->copy_to_device(device_b, b.buf, b.len, NULL);
    if (status == fw_success_k)
        status = fw_matmul_f32_best(device_a, device_b, device_c, rows, columns, depth, capabilities, NULL);
    if (status == fw_success_k)
        status = services->copy_to_host(PyByteArray_AS_STRING(bytes), device_c, PyByteArray_GET_SIZE(bytes), NULL);
    if (status == fw_success_k) status = services->synchronize(NULL);
    services->release(device_a), services->release(device_b), services->release(device_c);
    Py_END_ALLOW_THREADS;
    if (status != fw_success_k) {
        PyErr_Format(PyExc_RuntimeError, "the %s matmul failed on device 0", vendor);
        Py_CLEAR(bytes);
    }
release:
    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    return bytes;
}

static PyMethodDef methods[] = {
    {"capabilities", (PyCFunction)(void (*)(void))capabilities, METH_VARARGS | METH_KEYWORDS,
     "capabilities(vendor, device=0): the vendor's generations whose kernels load on the device, best first."},
    {"matmul", (PyCFunction)(void (*)(void))matmul, METH_VARARGS | METH_KEYWORDS,
     "matmul(vendor, a, b, capability=None): C = A * B over float32 matrices on device 0, as raw bytes."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_definition = {PyModuleDef_HEAD_INIT, "_gpu", NULL, -1, methods};

PyMODINIT_FUNC PyInit__gpu(void) {
    PyObject *module = PyModule_Create(&module_definition);
#if defined(Py_GIL_DISABLED)
    if (module) PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED);
#endif
    return module;
}
