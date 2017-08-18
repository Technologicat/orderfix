# -*- coding: utf-8 -*-
#
# Set Cython compiler directives. This section must appear before any code!
#
# For available directives, see:
#
# http://docs.cython.org/en/latest/src/reference/compilation.html
#
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True
#
"""Reorder solutions of parametric studies (assumed to be in random order) to make continuous curves.

The common use case is postprocessing for computing eigenvalues for parametric studies of linear PDE boundary-value problems.
The ordering of the numerically computed eigenvalues may suddenly change, as the problem parameter sweeps through the range of interest.

The reordering allows the plotting of continuous curves, which are much more readable visually than scatterplots of disconnected points.
"""

from __future__ import division, print_function, absolute_import

import cython

import numpy as np


# real
DTYPE = np.float64
ctypedef double DTYPE_t

# complex
DTYPEZ = np.complex128
ctypedef double complex DTYPEZ_t

cdef extern from "math.h":
    # fpclassify is actually a macro, so it does not have a specific input type.
    # Here we tell Cython we would like to use it for doubles.
    int fpclassify(double x) nogil
    int FP_INFINITE
    int FP_NAN
    int FP_NORMAL
    int FP_SUBNORMAL
    int FP_ZERO

cdef extern from "complex.h":
    double creal(double complex z) nogil
    double cimag(double complex z) nogil
    double complex conj(double complex z) nogil


# Find smallest entry in data out of those that are not marked as already used, and return its index.
# If pused is NULL, find the smallest entry in data, and return its index.
#
# pused[k] is a boolean flag that tells whether the kth data item (0-based indexing) has been used.
# Initially, all entries in pused must be initialized to False.
#
# This function will update pused at each call, setting pused[arg] to True if a non-NaN match was found.
# For any NaN entries, pused[k] will stay False.
#
# NaNs are ignored. If all remaining (non-used) entries are NaN, the return value is -1.
#
# n = length of data (length of pused must match).
#
cdef int argmin_next( double* data, int n, unsigned char* pused ) nogil:
    cdef unsigned int k, start

    cdef int arg = -1     # result (index to data)
    cdef double smallest  # the corresponding data value

    if not pused:
        # initialize to first entry that is not NaN
        for k in range(n):
            if fpclassify(data[k]) == FP_NAN:
                continue

            smallest = data[k]
            arg      = k
            break

        # all remaining entries are NaN
        if arg == -1:
            return -1

        # sweep
        start = arg
        for k in range(start,n):
            if fpclassify(data[k]) == FP_NAN:
                continue
            if data[k] < smallest:
                smallest = data[k]
                arg      = k

    else:
        # initialize to first entry that is not marked as used and is not NaN
        for k in range(n):
            if pused[k]:
                continue
            if fpclassify(data[k]) == FP_NAN:
                continue

            smallest = data[k]
            arg      = k
            break

        # all remaining entries are NaN
        if arg == -1:
            return -1

        # search
        start = arg
        for k in range(start+1, n):
            if pused[k]:
                continue
            if fpclassify(data[k]) == FP_NAN:
                continue

            if data[k] < smallest:
                smallest = data[k]
                arg      = k

    # Mark the returned result as used
    if arg != -1:
        pused[arg] = True

    return arg


def fix_ordering( DTYPEZ_t[:,::1] ss ):
    """Reorder solutions (assumed to be in random order) to make continuous solution curves.

    This is needed for eigenvalue problems also when expressed in polynomial form; the ordering of the roots may suddenly change.

    This releases the GIL to allow multithreading.

    Parameters:
        ss = rank-2 np.array. First index indexes the values of the problem parameter;
             second index indexes solutions at each value of the problem parameter.

    Returns:
        None; the input array is modified in-place.
"""
    cdef unsigned int nstep = ss.shape[0]  # number of problem parameter steps
    cdef unsigned int ns    = ss.shape[1]  # number of solutions at one value of problem parameter

    # temporaries for the inner loop
    cdef DTYPEZ_t[::1] s         = np.empty( [ns,], dtype=DTYPEZ )
    cdef DTYPEZ_t[::1] s_new     = np.empty( [ns,], dtype=DTYPEZ )
    cdef DTYPEZ_t[::1] s_old     = np.empty( [ns,], dtype=DTYPEZ )
    cdef DTYPE_t[::1]  sqdist    = np.empty( [ns,], dtype=DTYPE )
    # http://stackoverflow.com/questions/18058744/passing-a-numpy-pointer-dtype-np-bool-to-c
    cdef unsigned char[::1] used = np.empty( [ns,], dtype=np.uint8  )
    cdef DTYPEZ_t distvec

    cdef unsigned int i, j, k, m

    # This is the kind of code Cython optimizes well:
    #
    # - We use the np.ndarrays just as arrays to be accessed at C speed.
    # - All data types are known beforehand
    # - We avoid calling any Python (or NumPy) functions
    # - We avoid doing Python math operations on NumPy objects
    #
    s_old[:] = ss[0,:]  # slightly faster than looping over j... maybe something to do with NumPy internals?
    with nogil:
        for m in range(1,nstep):
            for j in range(ns):
                s[j] = ss[m,j]
                used[j] = False

            # Tracking algorithm that picks the closest solution,
            # keeping track of solutions already mapped ("used")
            # during each step.
            #
            # This is basically the same algorithm as the closest-distance
            # "null" algorithm in flutter_plot.m.

            for j in range(ns):
                # Compare all new values against the old jth value.
                for i in range(ns):
                    distvec = s[i] - s_old[j]
                    sqdist[i] = creal(distvec * conj(distvec))

                # Find closest value by a simple O(ns) sweep over data.
                #
                # To initialize, we pick the first value that has not been used.
                #
                # There are exactly ns points to pair with other ns points,
                # and each is used once. Therefore exactly one point still
                # remains unused at the last iteration (when j = ns - 1).
                #
                k = argmin_next( &sqdist[0], ns, &used[0] )  # this updates used[] accordingly
                s_new[j] = s[k]


            for j in range(ns):
                ss[m,j]  = s_new[j]  # update result
                s_old[j] = s_new[j]  # the new becomes the old for the next step
                                     # (note: faster to copy data than grab PyObject reference)


def fix_ordering_with_degenerate( DTYPEZ_t[:,::1] ss ):
    """Reorder solutions (assumed to be in random order) to make continuous solution curves.

    This is needed for eigenvalue problems also when expressed in polynomial form; the ordering of the roots may suddenly change.

    This version accounts for the possibility of degenerate problem instances having fewer than the usual number of solutions,
    with the empty slots filled by placeholder NaNs. A mixed input with some normal and some degenerate instances is allowed.
    Any degenerate data is simply left as NaNs.

    Parameters:
        ss = rank-2 np.array. First index indexes the values of the problem parameter;
             second index indexes solutions at each value of the problem parameter.

    Returns:
        None; the input array is modified in-place.
"""
    cdef unsigned int nstep = ss.shape[0]  # number of problem parameter steps
    cdef unsigned int ns    = ss.shape[1]  # number of solutions at one value of problem parameter, i.e. usually the degree of the non-degenerate characteristic polynomial

    # temporaries for the inner loop
    cdef DTYPEZ_t[::1] s            = np.empty( [ns,], dtype=DTYPEZ )
    cdef DTYPEZ_t[::1] s_new        = np.empty( [ns,], dtype=DTYPEZ )
    cdef DTYPEZ_t[::1] s_old        = np.empty( [ns,], dtype=DTYPEZ )
    cdef DTYPE_t[::1]  sqdist       = np.empty( [ns,], dtype=DTYPE )
    # http://stackoverflow.com/questions/18058744/passing-a-numpy-pointer-dtype-np-bool-to-c
    cdef unsigned char[::1] used    = np.empty( [ns,], dtype=np.uint8  )  # indices in s     which have already been matched    at the current step
    cdef unsigned char[::1] written = np.empty( [ns,], dtype=np.uint8  )  # indices in s_new which have already been written to at the current step
    cdef DTYPEZ_t distvec

    cdef unsigned int i, j, k, m

    # Matching with partial data:
    #
    # Strategy 1 (attempt to reconstruct as many curves as there are columns in the solution array):
    #
    #   - If as many NaNs at current and previous steps, match normally (can also use the "used" flag normally).
    #     Once all non-NaNs have been matched, the step is complete. Leave any NaN columns as-is.
    #
    #   - Else if more NaNs at current step than at previous step (solution becomes degenerate as the loading parameter is increased):
    #      - "upside down" match: iterate over the solutions *at the current step*, matching each of them against all solutions at the previous step
    #      - match normally until all non-NaNs at the current step have been matched
    #      - set the matched solutions *at the previous step* as used
    #      - iterate over the remaining solutions *at the previous step*, matching them against the solutions at the current step.
    #
    #   - Else if more NaNs at previous step than at current step (solution ceases to be degenerate as the loading parameter is increased):
    #     (this handles cases where the first block of problem instances at the beginning of the data is degenerate)
    #      - match normally until all non-NaNs at the previous step have been matched
    #      - set the matched solutions *at the current step* as used
    #      - reset the used flags for the previous step
    #      - run the match again, for each remaining solution *at the current step* (matching it against all non-used solutions at the previous step)
    #      - when a match is found, mark it as used; duplicate the matched solution into the corresponding NaN column (now taking the column number from the *current* step):
    #          - walk back in history as long as there are NaNs in that column, at each step copying the solution from the matched column
    #
    # Strategy 2a (attempt to connect only the data that is already in the solution array):
    #
    #   - If as many NaNs at current and previous steps, match normally.
    #   - Else base the matching on the step which has more NaNs (less curves to match).
    #   - Leave the NaNs in the non-matched columns.
    #
    # Strategy 2b (attempt to connect only the data that is already in the solution array):
    #
    #   - As in the old version, attempt to match each solution from the previous step
    #   - but only if the solution is not NaN, and count(used) < count(non-NaN at current step)
    #   - copy any unused solutions into the remaining slots of s_new
    #
    # We use strategy 2b; it is the simplest, and the NaNs should not harm plotting.


    # Tracking algorithm that picks the closest solution,
    # keeping track of solutions already mapped ("used")
    # during each step.
    #
    # This is basically the same algorithm as the closest-distance
    # "null" algorithm in flutter_plot.m.


    # Treat one set of solutions at a time.
    #
    # (Iterating an np.array walks along the top-level dimension,
    #  which here is the loading parameter.)
    #
    s_old[:] = ss[0,:]  # slightly faster than looping over j
    with nogil:
        for m in range(1,nstep):
            for j in range(ns):
                s[j]       = ss[m,j]
                used[j]    = False
                written[j] = False

            # processing
            #
            for j in range(ns):
                # if this solution at the previous step is NaN, skip trying to match it
                # FIXME: we must do this cumbersome check because fpclassify() cannot be imported simultaneously for both double and double complex arguments.
                if fpclassify(creal(s_old[j])) == FP_NAN  or  fpclassify(cimag(s_old[j])) == FP_NAN:
                    continue

                # Compare all new values against the old jth value.
                for i in range(ns):
                    distvec = s[i] - s_old[j]
                    sqdist[i] = creal(distvec * conj(distvec))

                # Find closest non-used value that is not NaN.
                #
                k = argmin_next( &sqdist[0], ns, &used[0] )

                # if all non-NaN solutions at the current step have been matched, we're done
                if k == -1:
                    break

                s_new[j]   = s[k]
                written[j] = True

            # copy any leftovers (non-matched solutions) to the remaining free slots
            #
            # - "written" indexes s_new
            # - "used" indexes s
            # - free slots in s_new are those which have not been written to
            # - indexing by an empty array in an assignment is a no-op
            #
            for j in range(ns):
                # skip any already written slots in s_new
                if written[j]:
                    continue

                for k in range(ns):
                    # skip any already matched solutions in s
                    if used[k]:
                        continue

                    s_new[j] = s[k]

                    break

            for j in range(ns):
                ss[m,j]  = s_new[j]  # update result
                s_old[j] = s_new[j]  # the new becomes the old for the next step

