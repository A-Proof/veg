#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

/* Contiguous buffers avoid NumPy ABI dependencies. Arithmetic keeps its input
 * precision, and the GIL is released throughout the recurrent scan. */
#define SCAN(TYPE) do { \
    const TYPE *q = bufs[0].buf, *k = bufs[1].buf; \
    const TYPE *beta = bufs[2].buf, *v = bufs[3].buf; \
    TYPE *out = bufs[4].buf, *hist = bufs[5].buf; \
    TYPE *err = bufs[6].buf, *state = bufs[7].buf; \
    memset(state, 0, (size_t)bufs[7].len); \
    for (Py_ssize_t n = 0; n < N; n++) { \
        TYPE *s = state + n * K * V; \
        for (Py_ssize_t t = 0; t < T; t++) { \
            Py_ssize_t pos = n * T + t; \
            memcpy(hist + pos * K * V, s, (size_t)(K * V) * sizeof(TYPE)); \
            for (Py_ssize_t j = 0; j < V; j++) { \
                TYPE prediction = 0; \
                for (Py_ssize_t i = 0; i < K; i++) \
                    prediction += k[pos * K + i] * s[i * V + j]; \
                err[pos * V + j] = v[pos * V + j] - prediction; \
                out[pos * V + j] = 0; \
            } \
            for (Py_ssize_t i = 0; i < K; i++) { \
                TYPE ki = k[pos * K + i], qi = q[pos * K + i]; \
                for (Py_ssize_t j = 0; j < V; j++) { \
                    s[i * V + j] += ki * (beta[pos] * err[pos * V + j]); \
                    out[pos * V + j] += qi * s[i * V + j]; \
                } \
            } \
        } \
    } \
} while (0)

static PyObject *delta_forward(PyObject *self, PyObject *args) {
    PyObject *objects[8];
    Py_buffer bufs[8] = {{0}};
    int count = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &objects[0], &objects[1],
                          &objects[2], &objects[3], &objects[4], &objects[5],
                          &objects[6], &objects[7])) return NULL;
    for (; count < 8; count++) {
        int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
        if (count >= 4) flags |= PyBUF_WRITABLE;
        if (PyObject_GetBuffer(objects[count], &bufs[count], flags) < 0) goto fail;
    }
    if (bufs[0].ndim != 4 || bufs[3].ndim != 4) goto invalid;
    Py_ssize_t B = bufs[0].shape[0], H = bufs[0].shape[1];
    Py_ssize_t T = bufs[0].shape[2], K = bufs[0].shape[3];
    Py_ssize_t V = bufs[3].shape[3];
    if (B < 1 || H < 1 || T < 1 || K < 1 || V < 1) goto invalid;
    /* Compare dimensions directly, without trusting user-provided sizes. */
    Py_ssize_t shapes[8][5] = {
        {B,H,T,K,0}, {B,H,T,K,0}, {B,H,T,1,0}, {B,H,T,V,0},
        {B,H,T,V,0}, {B,H,T,K,V}, {B,H,T,V,0}, {B,H,K,V,0}
    };
    int is_double = strcmp(bufs[0].format, "d") == 0;
    if (!is_double && strcmp(bufs[0].format, "f") != 0) goto invalid;
    for (int a = 0; a < 8; a++) {
        int dims = a == 5 ? 5 : 4;
        if (bufs[a].ndim != dims ||
            strcmp(bufs[a].format, is_double ? "d" : "f") != 0 ||
            bufs[a].itemsize != (is_double ? sizeof(double) : sizeof(float)))
            goto invalid;
        for (int d = 0; d < dims; d++)
            if (bufs[a].shape[d] != shapes[a][d]) goto invalid;
    }
    Py_ssize_t N = B * H;
    Py_BEGIN_ALLOW_THREADS
    if (is_double) { SCAN(double); } else { SCAN(float); }
    Py_END_ALLOW_THREADS
    for (int a = 0; a < 8; a++) PyBuffer_Release(&bufs[a]);
    Py_RETURN_NONE;
invalid:
    PyErr_SetString(PyExc_ValueError, "Expected compatible contiguous float32/64 delta buffers");
fail:
    for (int a = 0; a < count; a++) PyBuffer_Release(&bufs[a]);
    return NULL;
}

static PyMethodDef methods[] = {
    {"delta_forward", delta_forward, METH_VARARGS, "Fused CPU prediction-error memory scan."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_native", NULL, -1, methods
};
PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&module); }
