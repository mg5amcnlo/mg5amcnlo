Neural importance sampling with MadNIS
=========================================

The integrator in the previous example samples uniformly and reweights, which is why its
Monte Carlo error only shrinks as the square root of the sample count. `MadNIS
<https://docs.madnis.ai>`_ is a separate package that trains a normalizing flow to sample
close to the integrand itself, cutting that error down for the same number of evaluations.
This example wraps the :doc:`previous integrator's <integrator>` mapping and cross section as
a MadNIS integrand.

Setting up the physics
--------------------------

Everything up to the integrand itself is the same as in the previous example, except that the
random numbers are now PyTorch tensors, since MadNIS trains with PyTorch:

.. code-block:: python

    import os

    import lhapdf  # only used below to locate the installed PDF set
    import torch
    import madspace as ms
    from madnis.integrator import Integrator

    torch.set_default_dtype(torch.float64)

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

    PDF_SET = "NNPDF40_lo_as_01180"
    pdf_dir = os.path.join(lhapdf.paths()[0], PDF_SET)
    pdf_grid = ms.PdfGrid(os.path.join(pdf_dir, f"{PDF_SET}_0000.dat"))
    pdf_grid.initialize_globals(ctx)
    alphas_grid = ms.AlphaSGrid(os.path.join(pdf_dir, f"{PDF_SET}.info"))
    alphas_grid.initialize_globals(ctx)
    alpha_s = ms.RunningCoupling(alphas_grid)

    scale = ms.EnergyScale(
        particle_count=5, type=ms.EnergyScale.DynamicalScaleType.half_transverse_mass
    )
    cross_section = ms.DifferentialCrossSection(
        matrix_element=matrix_element,
        cm_energy=E_CM,
        running_coupling=alpha_s,
        energy_scale=scale,
        pid_options=[[21, 21]],
        pdf1=pdf_grid, pdf2=pdf_grid,
        input_momentum_fraction=True,
    )

Writing the integrand
-------------------------

MadNIS draws unit-hypercube points and asks for the integrand at each of them. The mapping
turns those points into momenta, and its Jacobian, ``det``, has to be folded into the
returned value along with the cross section, exactly as in the manual integrator:

.. code-block:: python

    def integrand(x):
        momenta, x1, x2, det = mapping.map_forward([x])
        pdf_id = torch.zeros(x.shape[0], dtype=torch.int32)
        return cross_section(momenta, x1, x2, pdf_id) * det

Training and integrating
----------------------------

:py:class:`madnis.integrator.Integrator` takes the integrand and its dimension. Training
adapts the flow to the integrand; :py:meth:`integrate` then draws fresh samples from the
trained flow to compute the final estimate:

.. code-block:: python

    integrator = Integrator(integrand, dims=mapping.random_dim())
    integrator.train(200)
    result, error = integrator.integrate(200000)
    print(f"sigma = {result:.2f} +- {error:.2f} pb  (rel. error {error / result * 100:.2f}%)")

The same number of samples now gives a much smaller error than the non-adaptive integrator::

    sigma = 239.64 +- 0.34 pb  (rel. error 0.14%)
