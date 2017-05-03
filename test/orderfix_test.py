# -*- coding: utf-8 -*-
"""Reorder solutions of parametric studies (assumed to be in random order) to make continuous curves.

The common use case is postprocessing for computing eigenvalues for parametric studies of linear PDE boundary-value problems.
The ordering of the numerically computed eigenvalues may suddenly change, as the problem parameter sweeps through the range of interest.

The reordering allows the plotting of continuous curves, which are much more readable visually than scatterplots of disconnected points.
"""

from __future__ import division, print_function, absolute_import

import numpy as np


# SLOW reference implementation, for testing the fast Cython implementation.
#
def fix_ordering( ss ):
    """Reorder polynomial roots (assumed to be in random order) to make continuous curves.

This version assumes there are ss.shape[1] valid roots on each row of ss.

Parameters:
    ss : rank-2 np.array of type np.complex128
        ss[i,j] contains the jth root for the ith parameter step.
"""
    ns = ss.shape[1]  # number of solutions at one value of the problem parameter

    # temporaries for the inner loop
    s_new = np.empty( [ns,], dtype=np.complex128 )
    s_old = np.empty( [ns,], dtype=np.complex128 )
    free  = np.empty( [ns,], dtype=bool )

    # Treat one set of solutions at a time.
    #
    # (Iterating an np.array walks along the top-level dimension,
    #  which here is the problem parameter.)
    #
    s_old = ss[0,:]
    for s in ss[1:,:]:
        # Tracking algorithm that picks the closest solution,
        # keeping track of solutions already mapped ("used")
        # during each step.
        #
        # This is basically the same algorithm as the closest-distance
        # "null" algorithm in flutter_plot.m.
        #
        free[:] = True
        for j in xrange(ns):
            # We compare the old jth value against all new values.
            #
            # Note that np.absolute(z) is faster, with fewer Python operations than z*np.conj(z).
            #
            # Typically this innermost loop runs 1-2 times.
            #
            for k in np.argsort(np.absolute(s - s_old[j])):  # try kth closest result...
                if free[k]:  # ...until we find one that has not been allocated already
                    s_new[j] = s[k]
                    free[k]  = False
                    break
        s[:] = s_new  # this updates the original because s is a view into ss
        s_old = s  # only a reference!


# SLOW reference implementation, for testing the fast Cython implementation.
#
def fix_ordering_with_degenerate( ss ):
    """Reorder polynomial roots (assumed to be in random order) to make continuous curves.

This version accounts for the possibility of degenerate problem instances (lower-degree polynomials),
having fewer than the usual number of roots, with the empty slots filled by placeholder NaNs.

A mixed input with some normal and some degenerate instances is allowed.
In the output, any NaNs are simply left as NaNs.

Parameters:
    ss : rank-2 np.array of type np.complex128
        ss[i,j] contains the jth root for the ith parameter step, or NaN if it does not exist.
"""
    ns = ss.shape[1]  # number of solutions of a normal (non-degenerate) problem instance at one value of the problem parameter

    # temporaries for the inner loop
    s_new = np.empty( [ns,], dtype=np.complex128 )
    s_old = np.empty( [ns,], dtype=np.complex128 )
    free  = np.empty( [ns,], dtype=bool )

    # Possible strategies for matching with partial data (we use strategy 2b, the last one below):
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
    #   - this strategy is the simplest of the ones considered here, and the NaNs should not harm plotting.


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
    rg = np.arange(ns, dtype=int)
    s_old = ss[0,:]
    for s in ss[1:,:]:
        n_nonnans = np.sum(~np.isnan(s))  # number of valid (non-NaN) solutions at the current step
        n_used    = 0   # number of solutions matched so far
        written   = []  # indices in s_new which have been used

        free[:] = True
        for j in xrange(ns):
            # if this solution at the previous step is NaN, skip trying to match it
            if np.isnan(s_old[j]):
                continue
            # if all non-NaN solutions at the current step have been matched, we're done
            if n_used >= n_nonnans:
                break

            # Compare the old jth value against all new values.
            #
            # Note that np.absolute(z) is faster, with fewer Python operations than z*np.conj(z).
            #
            # Typically this innermost loop runs 1-2 times.
            #
            for k in np.argsort(np.absolute(s - s_old[j])):  # try kth closest result...
                if free[k] and not np.isnan(s[k]):  # ...until we find one that has not been allocated already (and that is not NaN)
                                                    # (NaNs should be placed at the end by argsort, so in theory the second condition is not needed)
                    s_new[j] = s[k]
                    free[k]  = False
                    n_used  += 1
                    written.append( j )
                    break

        # copy any leftovers (non-matched solutions) to the remaining free slots
        #
        # - "written" indexes s_new
        # - "free" indexes s
        # - free slots in s_new are those which have not been written to
        # - indexing by an empty array in an assignment is a no-op
        #
        nonwritten = np.setdiff1d( rg, np.array(written, dtype=int) )
        s_new[nonwritten] = s[free.nonzero()[0]]

        s[:] = s_new  # this updates the original because s is a view into ss
        s_old = s  # only a reference!


def moving_ideal_string():
    """Usage example.


**Long** explanation:

We consider an ideal string (no bending stiffness), axially moving
over a free span, with pinholes at x=0 and x=L.

The governing equation is [Skutch, 1897]::

    w_tt + 2 V0 w_xt + (V0**2 - T/m) w_xx = 0   (*)
    w(x=0) = w(x=L) = 0

where the subscripts indicate partial differentiation.
V0 is the axial drive velocity, and T is the tension applied at the ends.
The function  w  describes the transverse deflection of the string.

As an IBVP, two initial conditions are required for w to make the solution unique.
For free-vibration analysis, we drop this requirement, and instead look at the
class of possible solutions as a whole.


Let us define dimensionless coordinates as::

    t' := t / tau
    x' := x / L

where  tau  is a characteristic time (of arbitrary value), SI unit [s],
and  L  is the length of the free span, SI unit [m]. (Also the characteristic length
is in principle arbitrary, but for this problem, it is convenient to choose its value as L,
since then the dimensionless space domain is always  0 < x' < 1.)

By the chain rule,::

    w_t = w_t' t'_t = w_t' * (1/tau)   (**)
    w_x = w_x' x'_x = w_x' * (1/L)     (***)

Now define the dimensionless deflection::

    w' := w / h
    
where  h  is a characteristic deflection (of arbitrary value), SI unit [s].

Solving for the original dimensional variables, plugging in the solutions to (*),
applying (**) and (***), and then omitting the prime from the notation, we have::

    (h/tau**2) w_tt + (h/(tau L)) 2 V0 w_xt + (h/L**2) (V0**2 - T/m) w_xx = 0
    w(x=0) = w(x=1) = 0

Finally, multiplying by (tau**2 / h) gives the dimensionless equation::

    w_tt + (2 tau V0 / L) w_xt + (tau**2 / L**2) (V0**2 - T/m) w_xx = 0


The last term suggests that a convenient choice for tau is::

    L/tau := sqrt(T/m)

which gives::

    tau = L / sqrt(T/m)

With this tau, we have::

    w_tt + (2 V0 / sqrt(T/m)) w_xt + (V0**2 / (T/m) - 1) w_xx = 0

This suggests that it is convenient to define a dimensionless axial velocity as::

    c := V0 / sqrt(T/m)

obtaining::

    w_tt + 2 c w_xt + (c**2 - 1) w_xx = 0

which leaves just one problem parameter, c, which we may sweep to make a parametric study
of this problem.


Now let us insert the standard trial function for the study of free vibrations
in a linear PDE problem (due to Euler, Lyapunov, and V. V. Bolotin)::

    w(x,t) = exp(s t) W(x)

where the Lyapunov exponent (a.k.a. stability exponent) s, and the function W,
are (inherently!) complex-valued.

Note that if such a complex-valued solution is found, then by the linearity of the
original problem (*), the real and imaginary parts of the solution will both be
real-valued solutions of (*).

We obtain::

    s**2 W + 2 c s W_x + (c**2 - 1) W_xx = 0   (a)


For an analytical solution, because (a) is a linear ODE with constant coefficients,
its solution is, generally speaking, a sum of complex exponential terms::

    W(x) = A0 exp(k1 x) + A1 exp(k2 x)         (b)

where k1 and k2 are the roots of the characteristic polynomial (here, for simplicity, assumed distinct)::

    s**2 + 2 c s k + (c**2 - 1) k**2 = 0       (c)

Once this is solved (for k, giving two roots k1 and k2 in terms of s), the constants A0 and A1
can be determined from the boundary conditions, by requiring W(0) = W(1) = 0 in equation (b).

Then plugging the solution to (a) will determine s.


On the other hand, to approach this numerically, we observe that equation (a) is a
quadratic eigenvalue problem for the pair (s,W).

Let us use standard C0 finite elements. Multiply (a) by an arbitrary admissible test function  psi,
integrate over the domain (0 < x < 1).

To make the equations more readable in ASCII, let us omit the integration from the notation.
We have::

    s**2 W psi + 2 c s W_x psi + (c**2 - 1) W_xx psi = 0   for all admissible psi

After integration by parts in the last term (zero Dirichlet BCs eliminate boundary term)::

    s**2 W psi + 2 c s W_x psi - (c**2 - 1) W_x psi_x = 0    (d)

Using a basis phi1, phi2, ..., the Galerkin series for W is::

  W(x) := Wn phin(x)  (summation over repeated index implied)

where Wn are coefficients.

Inserting this to (d), and choosing the set of test functions psi  as the set phij, gives::

    s**2 Wn phin phij + 2 c s Wn phin_x phij - (c**2 - 1) Wn phin_x phij_x = 0

As usual, we then exchange the order of the infinite summation and integration,
so that the integral is taken separately of each term in the Galerkin series.
Rearranging terms, we have::

    s**2 (phin phij) Wn + 2 c s (phin_x phij) Wn - (c**2 - 1) ( phin_x phij_x ) Wn = 0

Defining the mass, gyroscopic, and stiffness matrices, we can write this as::

    ( s**2 M + s C + K ) v = 0

and  v  denotes the vector of coefficients Wn.

Using a uniform grid of  N  linear elements, with affine coordinate mapping
from the reference (local) element [0,1] to each global element, the matrices are::

    M = dx/6 * ( 4 I + U + L )                  #     phi_j  *    phi_n   (j row, n column)
    C = (2 c) * 1/2  * ( U - L )                #  (d phi_n) *    phi_j
    K = (-(c**2 - 1)) * 1/dx * ( 2 I + U + L )  # -(d phi_j) * (d phi_n)

where  dx = 1/N  is the length of one global element, and::

    I = np.eye(N)
    U = np.diag(np.ones(N-1), +1)
    L = np.diag(np.ones(N-1), -1)


To solve this quadratic eigenvalue problem, following [Tisseur and Meerbergen, 2001],
we use the first companion linearization. Let::

    Q(s) := s**2 M + s C + K

    L(s) := s / M 0 \ + /  C K \
              \ 0 I /   \ -I 0 /

and, denoting the original eigenvector by  v  (this is the same v as above)::

    z := / s*v \
         \   v /

Then::

    L(s) z = 0

is equivalent to::

    Q(s) v = 0


Actually, since our system is small, we may go one step further, and invert M numerically.
In [Jeronen, 2011, p. 172 and 184] it is observed that the solutions  s  of  L(s) z = 0
are precisely the eigenvalues of the matrix::

    A := / -inv(M)*C  -inv(M)*K \
         \         I          0 /

so we only need to compute its eigenvalues. (This easily follows from the definition of L(s).)


In practice, we hand the matrix A over to NumPy, and obtain its eigenvalues, in a random order for each value
of the problem parameter c.


**Now, finally:**

**This** is the problem that `orderfix` solves:

It re-orders the eigenvalue data for different values of c, so that we can draw connected curves as c varies.


**Notes:**

Of the order-fixing algorithms discussed in [Jeronen, 2011], this library implements only the "null"
algorithm that simply pairs off the closest points; the Taylor prediction based and modal assurance
criterion (MAC) based algorithms are not implemented here. (Often the "null" algorithm works well enough in practice.)


**References:**
    J. Jeronen. 2011. On the mechanical stability and out-of-plane dynamics of a travelling panel
        submerged in axially flowing ideal fluid: a study into paper production in mathematical terms.
        Jyväskylä studies in computing 148. ISBN 978-951-39-4595-4 (book), ISBN 978-951-39-4596-1 (PDF)
        http://urn.fi/URN:ISBN:978-951-39-4596-1

    R. Skutch, 1897. Uber die Bewegung Eines Gespannten Fadens, Weicher Gezwungen
        ist Durch Zwei Feste Punkte, mit Einer Constanten Geschwindigkeit zu gehen,
        und Zwischen denselben in Transversal-Schwingungen von Gerlinger Amplitude
        Versetzt Wird. Annalen der Physik und Chemie 61, 190-195.

    F. Tisseur and K. Meerbergen, The quadratic eigenvalue problem, SIAM Rev., 43 (2001), pp. 235–286.
"""
    c  = 2.   # dimensionless axial velocity
    n  = 100  # number of elements

    dx = 1./n

    I = np.eye(n)
    U = np.diag(np.ones(n-1), +1)
    L = np.diag(np.ones(n-1), -1)

    M =   dx/6.          * ( 4.*I + U + L )   #     phi_j  *    phi_n,   j row, n column
    C =   c              * ( U - L )          #  (d phi_n) *    phi_j
    K = -(c**2 - 1.)/dx  * ( 2.*I + U + L )   # -(d phi_j) * (d phi_n)

    # companion form
    #
    O    = np.zeros( (n,n) )
    invM = np.linalg.inv(M)
    A    = np.array( np.bmat( [[-invM.dot(C), -invM.dot(K)],
                               [           I,            O]] ) )  # bmat() returns matrix, not ndarray

    # TODO: vary c in a loop, save results to array, run orderfix on them

    # Solve the companion form.
    #
    # Note that half the solutions will be correct (approximations of solutions of the original continuum problem),
    # and half will be nonsense (numerical artifacts). This is likely an aliasing effect due to the fact
    # that the spectrum is countably infinite, and we are truncating the Galerkin series to produce a
    # computable approximation.
    #
    # For this particular problem, it is known (from the analytical solution) that  s  is always purely imaginary
    # regardless of the value of the probelm parameter  c. (See [Jeronen, 2011].)
    #
    s,v = np.linalg.eig(A)
    v = v[:,n:]  # keep just the eigenvectors of the original quadratic problem

    # numerical prettification
    #
    def kill_almost_zeros(z, tol=1e-10):
        z  = z.copy()  # real() and imag() only create views, so let's make our own copy
        zr = np.real(z)
        zi = np.imag(z)
        zr[ np.abs(zr) < tol ] = 0.
        zi[ np.abs(zi) < tol ] = 0.
        return zr + 1j*zi

    def sort_by_magnitude(z):
        zmag = np.abs(z)
        p    = np.argsort(zmag)
        return z[p]

    s = sort_by_magnitude(kill_almost_zeros(s))

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot( np.real(s), np.imag(s), 'ko' )
    plt.xlabel( r'$\mathrm{Re}\,s$' )
    plt.ylabel( r'$\mathrm{Im}\,s$' )
    plt.show()


def test():
    moving_ideal_string()

if __name__ == '__main__':
    test()

