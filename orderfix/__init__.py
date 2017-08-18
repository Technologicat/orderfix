# -*- coding: utf-8 -*-
#
"""Reorder solutions of parametric studies (assumed to be in random order) to make continuous curves.

The common use case is postprocessing of numerically computed eigenvalues from parametric studies of linear PDE boundary-value problems.
The ordering of the numerically computed eigenvalues may suddenly change, as the problem parameter sweeps through the range of interest.

The reordering allows the plotting of continuous curves, which are much more readable visually than scatterplots of disconnected points.

The simple distance-based algorithm implemented here may fail in regions where the solutions cluster closely together,
but for the most part of the data, it usually works well.

Exported functions (see docstrings for details):
    fix_ordering
    fix_ordering_with_degenerate
"""

from __future__ import absolute_import

# This is extracted automatically by the top-level setup.py.
__version__ = '0.1.0'

from .orderfix import fix_ordering, fix_ordering_with_degenerate

