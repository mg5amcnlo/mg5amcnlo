Building a simple integrator
================================

The earlier examples sampled a phase-space mapping and evaluated a matrix element and a PDF
as separate pieces. Integrating a cross section is putting those pieces together and
averaging. This example builds a plain, non-adaptive integrator for ``g g > t t~ g``, using
the pure t-channel mapping, MadGraph7's default PDF set and its default HT/2 scale choice.

Assembling the mapping
--------------------------

``mode="propagator"`` follows the t-channel structure implied by the external masses,
without RAMBO or Chili's extra reshuffling. The cuts are the same ones as the
:doc:`cuts example <cuts>`, needed here to keep the matrix element from being evaluated in
the soft and collinear regions where it diverges:

.. code-block:: python

    import os

    import numpy as np
    import madspace as ms

    E_CM = 13000.0
    masses = [0.0, 0.0, 173.0, 173.0, 0.0]  # g g -> t t~ g
    pids = [21, 21, 6, -6, 21]

    O = ms.Observable
    cuts = ms.Cuts([
        ms.CutItem(O(pids, O.obs_pt, [O.jet_pids]), min=20.0),
        ms.CutItem(O(pids, O.obs_eta_abs, [O.jet_pids]), max=5.0),
        ms.CutItem(O(pids, O.obs_delta_r, [O.jet_pids]), min=0.4),
    ])
    mapping = ms.PhaseSpaceMapping(masses, E_CM, mode="propagator", cuts=cuts)

Loading the matrix element and the PDF
------------------------------------------

The same pattern as the :doc:`matrix element <matrix-element>` and :doc:`PDF <pdf>`
examples, generated with ``output mg7 PROC_ggttg`` and ``make BACKEND=scalar`` beforehand:

.. code-block:: python

    proc_dir = "PROC_ggttg"
    me_path = os.path.join(proc_dir, "lib", "libmadmatrix_sm_P0_gg_ttxg_scalar_m.so")
    param_card = os.path.join(proc_dir, "Cards", "param_card.dat")

    ctx = ms.default_context()
    api = ctx.load_matrix_element(me_path, param_card)
    MEI = ms.MatrixElement.MatrixElementInput
    MEO = ms.MatrixElement.MatrixElementOutput
    matrix_element = ms.MatrixElement(
        api, [MEI.momenta_in, MEI.alpha_s_in], [MEO.matrix_element_out], True
    )

    PDF_SET = "NNPDF40_lo_as_01180"  # MadGraph7's default PDF
    import lhapdf
    pdf_dir = os.path.join(lhapdf.paths()[0], PDF_SET)
    pdf_grid = ms.PdfGrid(os.path.join(pdf_dir, f"{PDF_SET}_0000.dat"))
    pdf_grid.initialize_globals(ctx)
    alphas_grid = ms.AlphaSGrid(os.path.join(pdf_dir, f"{PDF_SET}.info"))
    alphas_grid.initialize_globals(ctx)
    alpha_s = ms.RunningCoupling(alphas_grid)

Choosing the scale
---------------------

:py:class:`EnergyScale <madspace.EnergyScale>` computes the renormalization and
factorization scales from the event momenta. ``half_transverse_mass`` is MadGraph7's default
choice, :math:`H_\mathrm{T}/2`:

.. code-block:: python

    scale = ms.EnergyScale(
        particle_count=5, type=ms.EnergyScale.DynamicalScaleType.half_transverse_mass
    )

Assembling the cross section
--------------------------------

:py:class:`DifferentialCrossSection <madspace.DifferentialCrossSection>` combines the matrix
element with the flux factor, the two PDFs and :math:`\alpha_s`, all at the chosen scale.
``pid_options`` lists the incoming flavour combinations it should accept, here just gluon-gluon:

.. code-block:: python

    cross_section = ms.DifferentialCrossSection(
        matrix_element=matrix_element,
        cm_energy=E_CM,
        running_coupling=alpha_s,
        energy_scale=scale,
        pid_options=[[21, 21]],
        pdf1=pdf_grid, pdf2=pdf_grid,
        input_momentum_fraction=True,
    )

Integrating
--------------

The mapping gives the momenta, the momentum fractions and the phase-space Jacobian. The
cross section gives the differential rate at that point. Their product, averaged over the
batch, is the Monte Carlo estimate of the total cross section, and its standard error:

.. code-block:: python

    rng = np.random.default_rng(0)
    n = 200000
    r = rng.random((n, mapping.random_dim()))
    momenta, x1, x2, det = mapping.map_forward([r])
    pdf_id = np.zeros(n, dtype=np.int32)  # index into pid_options; only one here

    weight = cross_section(momenta, x1, x2, pdf_id) * det
    sigma, error = weight.mean(), weight.std() / np.sqrt(n)
    print(f"sigma = {sigma:.2f} +- {error:.2f} pb")

::

    sigma = 246.03 +- 7.43 pb

The relative error, about 3%, is typical of plain sampling with no adaptive importance
sampling. The next two examples reduce it.
