# -*- coding: utf-8 -*-
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cyhton: wraparound=False

from cython.parallel import parallel, prange
from cython.view cimport array as cvarray
import numpy as np

cdef extern from "math.h":
    cdef double sqrt(double x) nogil

cpdef double [:, ::1] cosine_sim(double [:, ::1] m1, Py_ssize_t [::1] m1rows_i, double [:, ::1] m2):

    cdef:
        # Matrix index variables.
        Py_ssize_t m1r_i, m2_i, p1m1r_i, p2m2_i, p1_j, p2_j, p3m1r_i, p3m2_i, p3_j

        # Matrices dimentions intilized variables.
        Py_ssize_t m_J = m1.shape[1]
        Py_ssize_t m2_I = m2.shape[0]
        Py_ssize_t m1r_I = m1rows_i.shape[0]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double [::1] m1_norms
        double [::1] m2_norms
        double [:, ::1] cssim_vect

    # Creating the temporary cython arrays.
    m1_norms = cvarray(shape=(m1r_I,), itemsize=sizeof(double), format="d")
    m2_norms = cvarray(shape=(m2_I,), itemsize=sizeof(double), format="d")
    cssim_vect = cvarray(shape=(m1r_I, m2_I), itemsize=sizeof(double), format="d")

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for m1r_i in range(m1r_I):
            m1_norms[m1r_i] = 0.0

        for m2_i in range(m2_I):
            m2_norms[m2_i] = 0.0

        for m1r_i in range(m1r_I):
            for m2_i in range(m2_I):
                cssim_vect[m1r_i, m2_i] = 0.0

        # Calculating the Norms for the first matrix.
        for p1m1r_i in prange(m1r_I, schedule='guided'):

            # Calculating Sum.
            for p1_j in range(m_J):
                m1_norms[p1m1r_i] += m1[m1rows_i[p1m1r_i], p1_j] * m1[m1rows_i[p1m1r_i], p1_j]

            # Calculating the Square root of the sum
            m1_norms[p1m1r_i] = sqrt(m1_norms[p1m1r_i])

            # Preventing Division by Zero.
            if m1_norms[p1m1r_i] == 0.0:
                m1_norms[p1m1r_i] = 0.000001

        # Calculating the Norms for the second matrix.
        for p2m2_i in prange(m2_I, schedule='guided'):

            # Calculating Sum.
            for p2_j in range(m_J):
                m2_norms[p2m2_i] += m2[p2m2_i, p2_j] * m2[p2m2_i, p2_j]

            # Calculating the Square root of the sum
            m2_norms[p2m2_i] = sqrt(m2_norms[p2m2_i])

            # Preventing Division by Zero.
            if m2_norms[p2m2_i] == 0.0:
                m2_norms[p2m2_i] = 0.000001

        # Calculating the cosine similarity product.
        # NOTE: The m2 matrix is expected to be NON-trasposed but it will treated like it.
        for p3m1r_i in prange(m1r_I, schedule='guided'):

            for p3m2_i in range(m2_I):

                # Calculating the elemnt-wise sum of products.
                for p3_j in range(m_J):
                    cssim_vect[p3m1r_i, p3m2_i] += m1[m1rows_i[p3m1r_i], p3_j] * m2[p3m2_i, p3_j]

                # Normalizing with the products of the respective vector norms.
                cssim_vect[p3m1r_i, p3m2_i] = cssim_vect[p3m1r_i, p3m2_i] / (m1_norms[p3m1r_i] * m2_norms[p3m2_i])

    return cssim_vect
