The built-in PDF interpolator
================================

MadSpace carries its own interpolator for parton distributions in the LHAPDF grid format, so
it does not need the LHAPDF library at runtime. This example reads a grid file directly and
evaluates it at a few points, using ``NNPDF40_lo_as_01180``, the default PDF set MadGraph7
uses.

Locating the grid file
-------------------------

``lhapdf`` is only used here to find where a PDF set is installed. The interpolation itself,
below, is done entirely by MadSpace:

.. code-block:: python

    import os
    import numpy as np
    import lhapdf
    import madspace as ms

    PDF_SET = "NNPDF40_lo_as_01180"
    pdf_dir = os.path.join(lhapdf.paths()[0], PDF_SET)

Loading the grid
-------------------

:py:class:`PdfGrid <madspace.PdfGrid>` reads the set's ``.dat`` member file. Its
``initialize_globals`` method uploads the grid to a :py:class:`Context <madspace.Context>`,
which every compute-graph function that reads it needs done once beforehand:

.. code-block:: python

    ctx = ms.default_context()
    grid = ms.PdfGrid(os.path.join(pdf_dir, f"{PDF_SET}_0000.dat"))
    grid.initialize_globals(ctx)

Evaluating the PDF
---------------------

:py:class:`PartonDensity <madspace.PartonDensity>` interpolates the grid at a batch of
``(x, Q)`` points, for the requested flavours, returning :math:`x f(x, Q)`:

.. code-block:: python

    pids = [-5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5]
    pdf = ms.PartonDensity(grid, pids)

    x = np.array([0.01, 0.1, 0.5])
    q = np.array([100.0, 100.0, 100.0])
    xfx = pdf(x, q)
    gluon_index = pids.index(21)
    print("x f_g(x, Q=100 GeV):", xfx[:, gluon_index])

The gluon density falls steeply towards larger :math:`x`::

    x f_g(x, Q=100 GeV): [7.92869630e+00 7.67957430e-01 4.62209124e-03]

The running coupling
-----------------------

:py:class:`AlphaSGrid <madspace.AlphaSGrid>` and :py:class:`RunningCoupling
<madspace.RunningCoupling>` are the same pattern, for :math:`\alpha_s`:

.. code-block:: python

    alphas_grid = ms.AlphaSGrid(os.path.join(pdf_dir, f"{PDF_SET}.info"))
    alphas_grid.initialize_globals(ctx)
    alpha_s = ms.RunningCoupling(alphas_grid)
    print(f"alpha_s(M_Z): {alpha_s(np.array([91.188]))[0]:.5f}")

::

    alpha_s(M_Z): 0.11800

This matches the set's name: ``NNPDF40_lo_as_01180`` is fit with :math:`\alpha_s(M_Z) =
0.118`.
