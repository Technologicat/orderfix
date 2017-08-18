# -*- coding: utf-8 -*-
"""Reorder solutions of parametric studies (assumed to be in random order) to make continuous curves.

The common use case is postprocessing for computing eigenvalues for parametric studies of linear PDE boundary-value problems.
The ordering of the numerically computed eigenvalues may suddenly change, as the problem parameter sweeps through the range of interest.

The reordering allows the plotting of continuous curves, which are much more readable visually than scatterplots of disconnected points.
"""

from __future__ import division, print_function, absolute_import

import time

import numpy as np

import orderfix


# SLOW reference implementation, for testing the fast Cython implementation.
#
def slow_fix_ordering( ss ):
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
        for j in range(ns):
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
def slow_fix_ordering_with_degenerate( ss ):
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
        for j in range(ns):
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


# ------------------------------------------------------------------------------------------------
# support stuff for the test / usage example
# ------------------------------------------------------------------------------------------------

# for initial reordering of solutions at the first value of the problem parameter in the sweep
#
def sort_by_magnitude(z):
    zmag = np.abs(z)
    p    = np.argsort(zmag)
    return z[p]

# for benchmarking
#
# usage:
#
#    with SimpleTimer(label=("    stuff done in ")) as s:
#        do_stuff()
#
#    with SimpleTimer(label=("    %d reps done in " % reps), n=reps) as s:
#        for k in range(reps):
#            do_stuff()
#
class SimpleTimer:
    def __init__(self, label="", n=None):
        self.label = label
        self.n     = n      # number of repetitions done inside the "with..." section (for averaging in timing info)

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, errtype, errvalue, traceback):
        dt         = time.time() - self.t0
        identifier = ("%s" % self.label) if len(self.label) else "time taken: "
        avg        = (", avg. %gs per run" % (dt/self.n)) if self.n is not None else ""
        print( "%s%gs%s" % (identifier, dt, avg) )

# analytical representation of Lobatto basis matrices (see test/legtest3.py in pydgq)
#
# note that we don't need to evaluate the actual basis functions for solving the eigenfrequency spectrum,
# so they are not provided here.
#
class LobattoBasisMatrices(object):
    def __init__(self, q):
        if q < 1:
            raise ValueError("This implementation only supports polynomial degrees q >= 1")
        self.q = q
        self.build_K()
        self.build_C()
        self.build_M()

    # stiffness matrix, integrand N' * N'
    def build_K(self):
        q = self.q
        n = q+1

        K = np.eye( n, dtype=np.float64 )
        K[0,0] =  1./2.
        K[0,1] = -1./2.
        K[1,0] = -1./2.
        K[1,1] =  1./2.

        self.K = K

    # damping or gyroscopic matrix, integrand N' * N
    def build_C(self):
        q = self.q
        n = q+1

        C = np.zeros( (n,n), dtype=np.float64 )
        C[0,0] = -1./2.
        C[0,1] =  1./2.
        C[1,0] = -1./2.
        C[1,1] =  1./2.

        if q >= 2:
            t = 1./np.sqrt(6.)
            C[0,2] = -t
            C[1,2] =  t
            C[2,0] =  t
            C[2,1] = -t

        # General formula for C_ji for j,i >= 2.
        for j in range(2,n):
            i = j + 1  # i-1 = j  <=>  i = j+1
            if i >= 2 and i < n:
                C[j,i] =  2. * np.sqrt( 1. / ( ( 2.*j - 1. ) * ( 2.*j + 1. ) ) )
            i = j - 1  # i+1 = j  <=>  i = j-1
            if i >= 2 and i < n:
                C[j,j-1] = -2. * np.sqrt( 1. / ( ( 2.*j - 1. ) * ( 2.*j - 3. ) ) )

        self.C = C

    # mass matrix, integrand N * N
    def build_M(self):
        q = self.q
        n = q+1

        M = np.zeros( (n,n), dtype=np.float64 )
        M[0,0] = 2./3.
        M[0,1] = 1./3.
        M[1,0] = 1./3.
        M[1,1] = 2./3.
        if q >= 2:
            t = 1./np.sqrt(2.)
            M[0,2] = -t
            M[1,2] = -t
            M[2,0] = -t
            M[2,1] = -t
        if q >= 3:
            t = 1. / (3. * np.sqrt(10) )
            M[0,3] = -t
            M[1,3] =  t
            M[3,0] = -t
            M[3,1] =  t

        # General formula for M_ji for j,i >= 2.
        for j in range(2,n):
            M[j,j]   = 1. / (2.*j - 1.) * ( 1. / (2.*j + 1.)  +  1. / (2.*j - 3.) )
            i = j - 2  # i+2 = j  <=>  i = j-2
            if i >= 2 and i < n:
                M[j,i] = 1. / ( np.sqrt(2.*j - 5.) * (2.*j - 3.) * np.sqrt(2.*j - 1.) )
            i = j + 2  # i-2 = j  <=>  i = j+2
            if i >= 2 and i < n:
                M[j,i] = 1. / ( np.sqrt(2.*j - 1.) * (2.*j + 1.) * np.sqrt(2.*j + 3.) )

        self.M = M


# ------------------------------------------------------------------------------------------------
# test / usage example
# ------------------------------------------------------------------------------------------------

def moving_ideal_string():
    """Usage example, controller."""

    # ------------------------------------------
    # Generate test data
    # ------------------------------------------

    # How many eigenfrequency pairs to handle.
    n_vis_pairs = 3
    n_vis = 2*n_vis_pairs  # how many eigenfrequencies

    cs = np.linspace(0.9, 1.1, 401)

    s_numerical, s_analytical = moving_ideal_string_solve_one(cs[0])

    # At the first value of c, sort the numerical solutions by magnitude,
    # to have them in a reasonable order.
    #
    s_numerical = sort_by_magnitude( s_numerical )

    ss  = np.empty( (cs.shape[0], s_numerical.shape[0]), dtype=np.complex128 )
    ssa = np.empty_like(ss)  # this test doesn't actually use the analytical results
    ss[0,:] = s_numerical
    ssa[0,:] = s_analytical

    print( "Solving..." )
    with SimpleTimer(label=("    solve took ")) as s:
        for i,c in enumerate(cs[1:]):
            s_numerical, s_analytical = moving_ideal_string_solve_one(c)
            ss[i+1,:] = s_numerical
            ssa[i+1,:] = s_analytical

    # ------------------------------------------
    # Test and benchmark orderfix
    # ------------------------------------------

    print( "Testing and benchmarking..." )

    ss_reordered_python = ss.copy()
    with SimpleTimer(label=("    Python version took ")) as s:
        slow_fix_ordering(ss_reordered_python)

    ss_reordered_cython = ss.copy()
    with SimpleTimer(label=("    Cython version took ")) as s:
        orderfix.fix_ordering(ss_reordered_cython)

    if np.allclose(ss, ss_reordered_cython):
        from sys import stderr
        print("WARNING: raw solutions already in order, test result will not be reliable", file=sys.stderr)

    assert np.allclose(ss_reordered_python, ss_reordered_cython), "*** FAIL ***: Python and Cython implementations produced different results"
    print("*** PASS ***: Results from Python and Cython versions match")

    # Print results.
    #
    print( "First 8 data items at last 5 c-steps:" )
    print( "raw data:",  ss[-5:, :8], sep='\n' )
    print( "reordered:", ss_reordered_cython[-5:, :8], sep='\n' )

    # Plot if Matplotlib available.
    #
    # Take the result with a grain of salt - typically only half the returned solutions are actually approximations of solutions of the continuum problem;
    # the rest are artifacts caused by the discretization. (To see this, compare to the analytical solution.)
    #
    try:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()

        plt.subplot(1,2, 1)
        plt.plot( cs, np.real(ss), color='#c0c0c0', linestyle='solid' )             # unordered mess
        plt.plot( cs, np.real(ss_reordered_cython), color='k', linestyle='solid' )  # nice, connected lines
        plt.xlabel(r'$c$')
        plt.ylabel(r'$\mathrm{Re}\;s$')

        ax = plt.subplot(1,2, 2)
        plt.plot( cs, np.imag(ss), color='#c0c0c0', linestyle='solid' )             # unordered mess
        plt.plot( cs, np.imag(ss_reordered_cython), color='k', linestyle='solid' )  # nice, connected lines (except maybe near the critical point, where better reordering algorithms are needed)
        plt.axis( [cs[0], cs[-1], -10, 10] )
        plt.xlabel(r'$c$')
        plt.ylabel(r'$\mathrm{Im}\;s$')
        ax.yaxis.set_label_position("right")

        plt.suptitle(r"Dimensionless complex eigenvalues $s$ of an axially moving ideal string")

        plt.show()

    except ImportError:
        pass


def moving_ideal_string_solve_one(c):
    """Usage example, solver.

    Parameters:
        c: float
            Dimensionless axial drive velocity (sensible range 0...1+ϵ)

    Returns:
        tuple of rank-1 np.arrays (s_numerical, s_analytical), where:
            s_numerical: rank-1 np.array of complex128
                Numerically computed eigenfrequencies, in random order.
                This is used as the data in the usage example of orderfix.

            s_analytical: rank-1 np.array of complex128
                Analytically computed eigenfrequencies, ordered by mode number.
                This is a reference solution, usually not available for realistic problems.


**Long** explanation (with some utf-8 math thrown in):

We consider an ideal string (no bending stiffness), axially moving
over a free span, with pinholes at x=0 and x=ℓ.

The governing equation is [Skutch, 1897]::

    w_tt + 2 V0 w_xt + (V0² - T/ρ) w_xx = 0,   0 < x < ℓ   (*)
    w(x=0) = w(x=ℓ) = 0

where the subscripts indicate partial differentiation.
V0 is the axial drive velocity, T is the tension applied at the ends,
and ρ is the linear density (SI unit [ρ] = kg / m) of the string.

The function  w  describes the transverse deflection of the string.

As an IBVP, two initial conditions are required for w to make the solution unique.
For free-vibration analysis, we drop this requirement, and instead look at the
class of possible solutions as a whole.


Let us define dimensionless coordinates as::

    t' := t / τ
    x' := x / ℓ

where  τ  is a characteristic time (of arbitrary value), SI unit [s],
and  ℓ  is the length of the free span, SI unit [m]. (Also the characteristic length
is in principle arbitrary, but for this problem, it is convenient to choose its value as ℓ,
since then the dimensionless space domain is always  0 < x' < 1.)

By the chain rule,::

    w_t = w_t' t'_t = w_t' * (1/τ)   (**)
    w_x = w_x' x'_x = w_x' * (1/ℓ)   (***)

Now define the dimensionless deflection::

    w' := w / h
    
where  h  is a characteristic deflection (of arbitrary value), SI unit [s].

Solving for the original dimensional variables, plugging in the solutions to (*),
applying (**) and (***), and then omitting the prime from the notation, we have::

    (h/τ²) w_tt + (h/(τℓ)) 2 V0 w_xt + (h/ℓ²) (V0**2 - T/ρ) w_xx = 0,   0 < x < 1
    w(x=0) = w(x=1) = 0

Finally, multiplying by  τ²/h  gives the dimensionless equation::

    w_tt + (2 τ/ℓ V0) w_xt + (τ²/ℓ²) (V0² - T/ρ) w_xx = 0,   0 < x < 1


The last term suggests that a convenient value for τ is obtained by choosing::

    ℓ/τ := sqrt(T/ρ)

which gives::

    τ = ℓ / sqrt(T/ρ)

With this choice for τ, we have::

    w_tt + (2 V0 / sqrt(T/ρ)) w_xt + (V0² / (T/ρ) - 1) w_xx = 0,   0 < x < 1

This in turn suggests that it is convenient to define a dimensionless axial velocity as::

    c := V0 / sqrt(T/ρ)

finally obtaining::

    w_tt + 2 c w_xt + (c² - 1) w_xx = 0,   0 < x < 1

We have remaining just one problem parameter, c, which we may sweep to make a parametric study
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

    s² W + 2 c s W_x + (c² - 1) W_xx = 0   (a)


For an analytical solution, because (a) is a linear ODE with constant coefficients,
its solution is, generally speaking, a sum of complex exponential terms::

    W(x) = A0 exp(k1 x) + A1 exp(k2 x)         (b)

where k1 and k2 are the roots of the characteristic polynomial (here, for simplicity, assumed distinct)::

    s² + 2 c s k + (c² - 1) k² = 0       (c)

Once this is solved (for k, giving two roots k1 and k2 in terms of s), the constants A0 and A1
can be determined from the boundary conditions, by requiring W(0) = W(1) = 0 in equation (b).

Then plugging the solution to (a) will determine s.


On the other hand, to approach this numerically, we observe that equation (a) is a
quadratic eigenvalue problem for the pair (s,W).

Let us use standard C0 finite elements. Multiply (a) by an arbitrary test function ψ,
and integrate over the domain (0 < x < 1):

    ∫　s² W ψ dx  +  ∫ 2 c s W_x ψ dx  +  ∫ (c² - 1) W_xx ψ dx  =  0   ∀ admissible ψ

After integration by parts in the last term (zero Dirichlet BCs eliminate boundary term)::

    ∫ s² W ψ dx  +  ∫ 2 c s W_x ψ dx  -  ∫ (c² - 1) W_x ψ_x dx  =  0    (d)

Using a basis φ1, φ2, ... of global basis functions defined on the domain 0 < x < 1,
the Galerkin series for W is::

  W(x) := ∑ Wn φn(x)  (summation over n)

where Wn are the Galerkin coefficients.

Inserting this to (d), and (following classical Galerkin methods) choosing to use
the set of basis functions φj as the set of test functions ψ, gives::

    ∫ s² (∑ Wn φn) φj dx  +  ∫ 2 c s (∑ Wn φn_x) φj dx  - ∫ (c² - 1) (∑ Wn φn_x) φj_x dx  =  0

As usual, we then exchange the order of the infinite summation and integration,
so that the integral is taken separately of each term in the Galerkin series.
(This requires some mathematical care, but here it is fine.)

Rearranging, we have::

    ∑ s² Wn ∫ (φn φj) dx  +  ∑ 2 c s ∫ Wn (φn_x φj) dx  - ∑ (c² - 1) ∫ Wn (φn_x φj_x) dx  =  0

Defining the mass, gyroscopic, and stiffness matrices M, C and K, we can write this as::

    ∑ ( s² M + s C + K ) v = 0

and  v = (W1, W2, ..., W{N+1})  denotes the vector of Galerkin coefficients.

Using a uniform grid of  N  linear elements, with affine coordinate mapping
from the reference (local) element [0,1] to each actual ("global") element,
the matrices are::

    M = M2
    C = 2 c M1
    K = (c² - 1) M0

where  Δx = 1/N  is the length of one element in units of dimensionless x',
and the generic matrices for uniformly spaced linear elements in 1D are::

    M2 =  Δx/6 * ( 4 I + U + L )  #  ∫ φn φj dx          (j row, n column)
    M1 =  1/2  * ( U - L )        #  ∫ dφn/dx φj dx
    M0 = -1/Δx * ( 2 I + U + L )  # -∫ dφn/dx dφj/dx dx

where

    I  = np.eye(N+1)
    U  = np.diag(np.ones(N), +1)
    L  = np.diag(np.ones(N), -1)

(The subscript on Mj denotes the order of time differentiation in the term
that corresponds to each of the matrices.

In M1, the factors of Δx cancel. These are introduced by the change of variable
in the integral (always computing it over the reference element).
Integration introduces a factor of Δx, whereas differentiation introduces 1/Δx.

The size of each matrix is (N+1)×(N+1), because for N linear elements, there are N+1
global basis functions, including the two for the endpoints of the domain.

We can also use any other basis, such as a Fourier sine basis, for which φn = sin(n π x).
The expressions of M, C and K above remain unchanged, once we compute the new M2, M1 and M0.)


The final result (specifically for the linear basis) is

    M = Δx/6        * ( 4 I + U + L )
    C = c           * ( U - L )
    K = (1 - c²)/Δx * ( 2 I + U + L )


To solve this quadratic eigenvalue problem, following [Tisseur and Meerbergen, 2001],
we use the first companion linearization. Let::

    Q(s) := s² M + s C + K

    L(s) := s / M 0 \ + /  C K \
              \ 0 I /   \ -I 0 /

and, denoting the original eigenvector by  v  (this is the same v as above)::

    z := / s v \
         \   v /

Then::

    L(s) z = 0

is equivalent to::

    Q(s) v = 0


Actually, since our system is small, we may go one step further, and invert M numerically.
In [Jeronen, 2011, p. 172 and 184] it is observed that the solutions  s  of  L(s) z = 0
are precisely the eigenvalues of the matrix::

    A := / -M⁻¹ C  -M⁻¹ K \
         \      I       0 /

so we only need to compute its eigenvalues. This easily follows from the definition of L(s).
Consider the equation  L(s) z = 0  and rearrange terms::

    - /  C K \ z = s / M 0 \ z
      \ -I 0 /       \ 0 I /

Take the minus sign on the LHS into the matrix:

    / -C -K \ z = s / M 0 \ z       (e)
    \  I  0 /       \ 0 I /

Multiply from the left by the block matrix  diag( M⁻¹, I ).  Obtain::

    A z = s z   (f)

Hence the eigenfrequencies s are the eigenvalues of the matrix A.

(Equivalently, we could solve the generalized linear eigenvalue problem for equation (e), above;
that would probably be numerically more stable, and need less flops for large N.

Note that numpy.linalg.eig() solves (f), but not (e); for that we need scipy.linalg.eig().
To avoid pulling in SciPy, we cast the problem into the form (f).)


**Now, finally:**

In practice, we hand the matrix A over to NumPy, and obtain its eigenvalues, in a random order for each value
of the problem parameter c.

**This** is the problem that `orderfix` solves:

    `orderfix` re-orders the eigenvalue data for different values of c, so that we can draw connected curves
    as we vary c, by simply connecting the points in the same column in the re-ordered data.

Of the order-fixing algorithms discussed in [Jeronen, 2011], this library implements only the "null"
algorithm that simply pairs off the closest points. The Taylor prediction based and modal assurance
criterion (MAC) based algorithms are not implemented here. (Often the "null" algorithm works well enough.)


**Generalizing the model:**

It is easy to account also for the case of the ideal string with internal damping and a linear Winkler-type
elastic foundation [Jeronen, 2011].

The original PDE problem now reads::

    w_tt + 2 V0 w_xt + (V0² - T/ρ) w_xx + α/ρ (w_t + V0 w_x) + γ/ρ w = 0,   0 < x < ℓ
    w(x=0) = w(x=ℓ) = 0

where α is an empirical damping coefficient describing, e.g., viscous losses inside the string material.
Similarly, γ describes the strength of the elastic foundation.


(To instead describe the resistance of a surrounding medium (low-speed air resistance, directly proportional to velocity),
the term  α/ρ V0 w_x  should be dropped, assuming the air mass does not axially move along with the string.

To combine these two cases, if the air mass moves at axial free-stream velocity V∞, we can write the damping terms as::

    α0/ρ (w_t + V0 w_x)  +  α∞/ρ (w_t + V∞ w_x)
    = (α0 + α∞)/ρ w_t + (α0 V0 + α∞ V∞)/ρ w_x

where α0 describes internal damping (as above), and α∞ describes the damping due to the air.)


The dimensionless form is::

    w_tt + 2 c w_xt + (c² - 1) w_xx + β w_t + β c w_x + δ w = 0,   0 < x < 1

leading to::

    s² W + 2 c s W_x + (c² - 1) W_xx + s β W + β c W_x + δ W = 0

This again leads to the finite element discretization::

    ∑ ( s² M + s C + K ) v = 0

with the same M, but with slightly different C and K::

    M = M2
    C = 2 c M1 + β M2
    K = (c² - 1) M0 + β c M1 + δ M2

where β (respectively δ) is a "dimensionless version of α/ρ" (resp. of γ/ρ).

This is the form of the problem solved by this routine.


Shortcut to see this: terms with _tt are summed into M, terms with _t into C, terms with no time differentiation into K.
On the RHS, terms with _xx produce M0, terms with _x produce M1, and terms with no space differentiation produce M2.

What are the expressions of β and δ? Carrying the new terms through the calculation gives::
    α/ρ w_t + α/ρ V_0 w_x + γ/ρ w             # original form
    α/ρ h/τ w_t + α/ρ V0 h/ℓ w_x + γ/ρ h w    # expressed in dimensionless x', t', w'; prime omitted
    α/ρ τ w_t + α/ρ τ V0 τ/ℓ w_x + γ/ρ τ² w   # after multiplication by τ²/h
    α/ρ τ w_t + α/ρ τ c w_x + γ/ρ τ² w        # after using  c = V0 τ/ℓ  (with our choice τ = ℓ / sqrt(T/ρ))
    β w_t + β c w_x + δ w                     # defining the dimensionless coefficients β = α/ρ τ, δ = γ/ρ τ²

Explicitly, we have::
    β = α/ρ τ = (α　ℓ) / (ρ sqrt(T/ρ))
    δ = γ/ρ τ² = γ/ρ ( ℓ² / (T/ρ) ) = γ/ρ ( ρ ℓ² / T ) = γ ℓ² / T


In our problem, ρ is the linear density of the string. In the SI system,  [ρ] = kg / m.
Thus, because β　is known to be dimensionless (to match other terms in the equation),
the SI unit of [α] = kg / (m s).

With some trivial algebraic manipulation, we identify this unit as the poiseuille, defined as
PI = Pa s::

    N    = kg m / s²
    Pa   = N/m² = kg / (m s²)
    Pa s = kg / (m s)

Thus α represents a dynamic viscosity.

As for γ, the tension [T] = N and span length [ℓ] = m, so [γ] = N/m² = Pa; it is the Young's modulus
of the elastic foundation material.


**CAUTION:**

The eigenfrequency spectrum is countably infinite, the discretization (by necessity) is not.

Some (typically half) of the computed solutions will be correct (solutions of the original continuum problem),
and the rest will be nonsense (numerical artifacts). This is likely an aliasing effect arising from the truncation
of the spectrum.

The trick is in knowing which of the computed solutions are correct. For simple problems such as these,
one can compare to their analytical solutions (which also helps to get some intuition for this).
Analytical solutions (including the case with damping) are provided in [Jeronen, 2011].


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
#    c     = 0.2   # dimensionless axial velocity (sensible range 0...1+ϵ; undamped string has its critical velocity at c=1)
    beta  = 1.    # dimensionless damping coefficient
    delta = 0.    # dimensionless elastic foundation coefficient

    n    = 20     # number of elements for FEM
    Dx   = 1./n   # if using linear elements: length of one element, in units of global dimensionless x'


#    # uniformly spaced linear elements
#    #
#    I = np.eye(n+1)
#    U = np.diag(np.ones(n), +1)
#    L = np.diag(np.ones(n), -1)
#    M2 = Dx/6.  * ( 4.*I + U + L )  #  ∫ φn φj dx          (j row, n column)
#    M1 = 1./2.  * (U - L)           #  ∫ dφn/dx φj dx
#    M0 = -1./Dx * ( 2.*I + U + L )  # -∫ dφn/dx dφj/dx dx


#    # Fourier basis, φk = sin(k π x),  k = 1, 2, ...   (as in [Jeronen, 2011])
#    #
#    # This is a spectral basis that has only one element spanning the whole domain.
#    #
#    # Each basis function satisfies the BCs  w(0) = w(1) = 0.
#    #
#    # These basis functions have support on all of 0 < x < 1, but they are L²-orthogonal, leading to M2 and M0 being diagonal.
#    # M1 will be dense (50% fill) and skew-symmetric.
#    #
#    # To compute the matrices symbolically:
#    #
#    # from sympy import symbols, pi, sin, cos, integrate, diff, pprint
#    # n,j = symbols('n,j', integer=True, positive=True)
#    # x   = symbols('x', real=True)
#    # M2 =  integrate( sin(n*pi*x) * sin(j*pi*x), (x, 0, 1) )
#    # M1 =  integrate( diff(sin(n*pi*x),x) * sin(j*pi*x), (x, 0, 1) )
#    # M0 = -integrate( diff(sin(n*pi*x),x) * diff(sin(j*pi*x),x), (x, 0, 1) )
#    # for M in (M2,M1,M0):
#    #     print(M)
#    #     pprint(M)
#
#    # Piecewise((1/2, Eq(j, n)), (0, True))
#    #
#    M2 = np.diag( 1./2. * np.ones(n) )
#
#    # -pi*n*Piecewise((0, Eq(j, n)), (-j/(pi*j**2 - pi*n**2), True)) + pi*n*Piecewise((0, Eq(j, n)), (-(-1)**j*(-1)**n*j/(pi*j**2 - pi*n**2), True))
#    #
#    jj  = np.arange(1,n+1)  # in the formula, n and j are 1-based
#    nn  = np.arange(1,n+1)
#    J,N = np.meshgrid(jj,nn, indexing='ij')   # row,column of each matrix element, 1-based
#    with np.errstate(divide='ignore', invalid='ignore'):  # we handle the diagonal separately afterward
#        M1 = J * N * ( 1. - (-1.)**J * (-1.)**N ) / ( J**2 - N**2 )
#    np.fill_diagonal(M1, 0)
#
#    # -pi**2*j*n*Piecewise((1/2, Eq(j, n)), (0, True))
#    #
#    M0 = -np.diag( 1./2. * np.pi**2 * np.arange(1,n+1)**2 )
#
#    I = np.eye(n)


    # hierarchical polynomial basis (Lobatto basis)
    #
    # We use only one element spanning the whole domain, with user-definable polynomial order,
    # which places this approach somewhere between spectral methods and p-FEM.
    #
    basis = LobattoBasisMatrices(n)
    M2 =  0.5 * basis.M  # the reference element of this basis is [-1,1], ours is [0,1]  ⇒  effectively, Δx = 1/2
    M1 =        basis.C
    M0 = -2.  * basis.K

    # By the BCs  w(0) = w(1) = 0,  the endpoint DOFs are zero, so remove those rows/columns.
    # All other DOFs are bubbles, which satisfy the BCs.
    #
    M2 = M2[2:,2:]
    M1 = M1[2:,2:]
    M0 = M0[2:,2:]
    I = np.eye(M0.shape[0])


    # our mass, gyroscopic+damping and stiffness matrices
    #
    # (in the basis being used)
    #
    M = M2
    C = 2.*c*M1 + beta*M2
    K = (c**2 - 1.)*M0 + beta*c*M1 + delta*M2


    # companion form
    #
    # (formulate as a standard linear eigenvalue problem to avoid the need for SciPy's generalized linear eigenvalue problem solver)
    #
    O    = np.zeros_like(I)
    invM = np.linalg.inv(M)
    A    = np.array( np.bmat( [[-invM.dot(C), -invM.dot(K)],
                               [           I,            O]] ), dtype=np.complex128 )  # bmat() returns matrix, not ndarray. We also want to force complex dtype.

    # TODO: vary c in a loop, save results to array, run orderfix on them

    # Solve the companion form.
    #
    # Note that half the solutions will be correct (approximations of solutions of the original continuum problem),
    # and half will be nonsense (numerical artifacts). This is likely an aliasing effect due to the fact
    # that the spectrum is countably infinite, and we are truncating the Galerkin series to produce a
    # computable approximation.
    #
    # For the particular problem of the classical (undamped) axially moving ideal string,
    # it is known (from the analytical solution) that  s  is always purely imaginary
    # regardless of the value of the problem parameter  c. For analytical solutions
    # of both problems discussed here, see [Jeronen, 2011].
    #
    s,v = np.linalg.eig(A)
    v = v[:,n:]  # keep just the eigenvectors of the original quadratic problem

    # numerical prettification
    #
    def kill_almost_zeros(z, tol=1e-10):
        # ensure complex128 to avoid "ValueError: assignment destination is read-only" if input happens to have real dtype
        my_z = np.empty( z.shape, dtype=np.complex128 )
        my_z[:] = z

        zr = np.real(my_z)
        zi = np.imag(my_z)
        zr[ np.abs(zr) < tol ] = 0.
        zi[ np.abs(zi) < tol ] = 0.

        return zr + 1j*zi

#    def sort_by_magnitude(z):
#        zmag = np.abs(z)
#        p    = np.argsort(zmag)
#        return z[p]

    # to test orderfix, we don't want to sort here
#    s = sort_by_magnitude(kill_almost_zeros(s))
    s = kill_almost_zeros(s)


    # For comparison: analytical solution.
    #
    # Here we use the following result from [Jeronen, 2011]. Note that the coefficients
    # use notation different from the one in this script.
    #
    # For the boundary-value problem
    #
    #     w_tt + 2 b w_xt + c w_xx + A1 w_t + A2 w_x + B w  =  0,   0 < x < 1     (37), p. 68
    #     w(x=0) = w(x=1) = 0                                                     ((16), p. 58; repeated on p. 68)
    #
    # the eigenfrequency spectrum of free vibrations is given by
    #
    #     s* = c [s - β/2] = c [ ±sqrt( γ - ω² ) - β/2 ]     (53), p. 72
    #
    # where
    #     γ = (1/4) ( β² - κ² - 4 B )                        (45), p. 70
    #     β = A1 c - A2 b                                    (41), p. 69
    #     κ = A2 sqrt(b² - c)                                (42), p. 69
    #     ω = - k π　/ sqrt(b² - c) = - k π ℓ / (C τ)         (29), p. 63   (k = 1, 2, ...)
    #
    # In (53), "s" refers to the eigenfrequency in transformed coordinates,
    # which were used to help solve the problem; "s*" denotes the "true s"
    # in (x,t) coordinates.
    #
    # In (29), C = sqrt(T/ρ) (in the reference, the symbol "m" is used for our "ρ").
    #
    #
    # In our notation, the PDE is
    #
    #    w_tt + 2 c w_xt + (c² - 1) w_xx + β w_t + β c w_x + δ w = 0,   0 < x < 1
    #
    # We calculate (LHS notation of [Jeronen, 2011]; RHS our notation)
    #
    # sqrt(b² - c)　← sqrt(c² - (c² - 1)) = sqrt(1) = 1
    #            β ← β (c² - 1) - β c² = -β
    #            κ ← β c
    #            ω ← -k π
    #            γ ← (1/4) ( β² - β² c² - 4 δ ) = (1/4) ( β² (1 - c²) - 4 δ )
    #
    # Thus the exact eigenfrequency spectrum is
    #
    #           s* ← (c² - 1) [ β/2 ± sqrt(  (1/4) ( β² (1 - c²) - 4 δ ) - k² π²  ) ]
    #
    npairs = s.shape[0] // 2
    s_analytical = np.empty( (2*npairs,), dtype=np.complex128 )
    k = np.arange(npairs)
    tmp = np.sqrt( 0j + (1./4.) * ( beta**2 * (1. - c**2) - 4.*delta ) - (k+1)**2 * np.pi**2  )  # 0j+...: force complex input to sqrt()
    s_analytical[2*k]     = (c**2 - 1.) * (beta/2. + tmp)
    s_analytical[2*k + 1] = (c**2 - 1.) * (beta/2. - tmp)

#    # DEBUG
#    n_vis = 6
#    print("exact:", s_analytical[:n_vis], sep="\n")
#    print("numerical:", s[:n_vis], sep="\n")

#    # DEBUG
#    import matplotlib.pyplot as plt
#    plt.figure(1)
#    plt.plot( np.real(s[:n_vis]), np.imag(s[:n_vis]), 'ko' )
#    plt.plot( np.real(s_analytical[:n_vis]), np.imag(s_analytical[:n_vis]), 'bo', alpha=0.5 )
#    plt.xlabel( r'$\mathrm{Re}\,s$' )
#    plt.ylabel( r'$\mathrm{Im}\,s$' )
#    plt.show()

    return (s, s_analytical)


def test():
    moving_ideal_string()

if __name__ == '__main__':
    test()

