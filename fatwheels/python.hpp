/**
 *  @file fatwheels/python.hpp
 *  @author Ash Vardanian
 *  @date August 10, 2024
 *  @brief CPython glue both modules share: buffers in, the matching C++ scalar, and NumPy arrays out.
 *
 *  The modules read their inputs through the buffer protocol rather than NumPy's C API, so they build
 *  against no NumPy headers and accept any object exporting a buffer; NumPy is imported only to
 *  allocate the arrays they return.
 */
#pragma once
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>          // `std::int8_t`, `std::uint64_t`
#include <initializer_list> // `std::initializer_list`
#include <type_traits>      // `std::is_same_v`

namespace ashvardanian::fatwheels::python {

template <typename function_type_>
PyCFunction as_pycfunction(function_type_ function) noexcept {
    return reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(function));
}

/** A buffer exported by a Python object, released when it leaves scope. */
struct buffer_t {
    Py_buffer view {};
    bool acquired = false;

    buffer_t() noexcept = default;
    buffer_t(buffer_t const &) = delete;
    buffer_t &operator=(buffer_t const &) = delete;
    ~buffer_t() noexcept {
        if (acquired) PyBuffer_Release(&view);
    }

    /** Requests the @p flags view of @p object, leaving a Python error set when it cannot. */
    bool acquire(PyObject *object, int flags) noexcept {
        acquired = PyObject_GetBuffer(object, &view, flags) == 0;
        return acquired;
    }
};

/** A keyword-only parameter, and where its argument lands; that stays null when the caller omits it. */
struct keyword_t {
    char const *name;
    PyObject **value;
};

/**
 *  @brief Matches a fast call's keyword arguments against @p parameters.
 *
 *  Keywords are walked by hand rather than through @c PyArg_ParseTupleAndKeywords, which parses a
 *  format string at runtime and cannot express the fast calling convention.
 *
 *  @param[in] called The function's name, for error messages.
 *  @param[in] positional How many of @p args are positional; the keyword values follow them.
 *  @return False with a @c TypeError for an unknown or repeated keyword.
 */
inline bool keywords_from_python(char const *called, PyObject *const *args, Py_ssize_t positional,
                                 PyObject *keyword_names, std::initializer_list<keyword_t> parameters) noexcept {
    if (!keyword_names) return true;
    Py_ssize_t const named = PyTuple_GET_SIZE(keyword_names);
    for (Py_ssize_t index = 0; index != named; ++index) {
        PyObject *name = PyTuple_GET_ITEM(keyword_names, index);
        keyword_t const *match = nullptr;
        for (keyword_t const &parameter : parameters)
            if (PyUnicode_CompareWithASCIIString(name, parameter.name) == 0) match = &parameter;
        if (!match) {
            PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument '%U'", called, name);
            return false;
        }
        if (*match->value) {
            PyErr_Format(PyExc_TypeError, "%s() got multiple values for '%s'", called, match->name);
            return false;
        }
        *match->value = args[positional + index];
    }
    return true;
}

/**
 *  @brief Calls @p callback with the C++ scalar matching a buffer's elements as its template argument.
 *
 *  Integers match by kind and size rather than by type code, as NumPy spells @c int64 as @c 'l' on
 *  Linux and as @c 'q' on Windows.
 *
 *  @param[in] view A buffer requested with @c PyBUF_FORMAT.
 *  @param[in] callback A template lambda, like `[&]<typename scalar_type>() { ... }`, returning a new reference.
 *  @return The callback's result, or null with a @c TypeError for any other element type.
 */
template <typename callback_type>
PyObject *dispatch_dtype(Py_buffer const &view, callback_type &&callback) noexcept {
    char const *format = view.format ? view.format : "B";
    if (*format == '@' || *format == '=') ++format; // the native byte order, spelled out
    char const kind = format[1] == '\0' ? format[0] : '\0';
    bool const is_signed = kind == 'b' || kind == 'h' || kind == 'i' || kind == 'l' || kind == 'q';
    bool const is_unsigned = kind == 'B' || kind == 'H' || kind == 'I' || kind == 'L' || kind == 'Q';
    if (kind == 'f' && view.itemsize == 4) return callback.template operator()<float>();
    if (kind == 'd' && view.itemsize == 8) return callback.template operator()<double>();
    if (is_signed && view.itemsize == 1) return callback.template operator()<std::int8_t>();
    if (is_signed && view.itemsize == 2) return callback.template operator()<std::int16_t>();
    if (is_signed && view.itemsize == 4) return callback.template operator()<std::int32_t>();
    if (is_signed && view.itemsize == 8) return callback.template operator()<std::int64_t>();
    if (is_unsigned && view.itemsize == 1) return callback.template operator()<std::uint8_t>();
    if (is_unsigned && view.itemsize == 2) return callback.template operator()<std::uint16_t>();
    if (is_unsigned && view.itemsize == 4) return callback.template operator()<std::uint32_t>();
    if (is_unsigned && view.itemsize == 8) return callback.template operator()<std::uint64_t>();
    PyErr_Format(PyExc_TypeError, "Unsupported element type '%s'", view.format ? view.format : "B");
    return nullptr;
}

/** A new Python number holding @p value, a @c double, @c std::int64_t or @c std::uint64_t. */
template <typename scalar_type>
PyObject *scalar_to_python(scalar_type value) noexcept {
    if constexpr (std::is_same_v<scalar_type, double>) return PyFloat_FromDouble(value);
    else if constexpr (std::is_same_v<scalar_type, std::int64_t>) return PyLong_FromLongLong(value);
    else return PyLong_FromUnsignedLongLong(value);
}

/** The type code @c numpy.empty takes for @c scalar_type, one of the types a product accumulates in. */
template <typename scalar_type>
constexpr char const *type_code() noexcept {
    if constexpr (std::is_same_v<scalar_type, float>) return "f";
    else if constexpr (std::is_same_v<scalar_type, double>) return "d";
    else if constexpr (std::is_same_v<scalar_type, std::int64_t>) return "q";
    else return "Q";
}

/** A new C-contiguous NumPy array of @p rows by @p columns @c scalar_type, or null with an error set. */
template <typename scalar_type>
PyObject *new_matrix(Py_ssize_t rows, Py_ssize_t columns) noexcept {
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (!numpy) return nullptr;
    PyObject *matrix = PyObject_CallMethod(numpy, "empty", "((nn)s)", rows, columns, type_code<scalar_type>());
    Py_DECREF(numpy);
    return matrix;
}

} // namespace ashvardanian::fatwheels::python
