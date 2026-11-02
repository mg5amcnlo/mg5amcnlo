Fusing RNG, flow, mapping and matrix element into one graph
================================================================

The previous two integrators call the mapping, and then the cross section, as two separate
Python-level steps. MadSpace also carries its own normalizing flow, :py:class:`Flow
<madspace.Flow>`, and its own compute graph builder, :py:class:`FunctionBuilder
<madspace.FunctionBuilder>`. This example uses the builder to fuse everything, from drawing
the random numbers to the final cross-section weight, into a single compiled graph, and trains
the flow's weights with a plain PyTorch optimizer.

Setting up the physics
--------------------------

The mapping, cuts, matrix element, PDF and scale are exactly the same building blocks as the
:doc:`earlier integrator <integrator>`, generated with ``output mg7 PROC_ggttg`` and
``make BACKEND=scalar`` beforehand:

.. code-block:: python

    import os

    import numpy as np
    import torch
    import madspace as ms
    from madspace.torch import FunctionModule

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

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
    import lhapdf  # only used below to locate the installed PDF set
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

Declaring the flow
----------------------

A :py:class:`Flow <madspace.Flow>` reshapes ``mapping.random_dim()`` unit-hypercube
coordinates into a learned sampling distribution. Its weights live in the context as trainable
globals, created by ``initialize_globals``:

.. code-block:: python

    flow = ms.Flow(
        input_dim=mapping.random_dim(), prefix="flow", bin_count=10,
        subnet_hidden_dim=32, subnet_layers=2,
    )
    flow.initialize_globals(ctx, seed=0)

Building the fused sampling graph
--------------------------------------

:py:meth:`FunctionBuilder.random <madspace.FunctionBuilder.random>` draws the random numbers
inside the graph itself, given a symbolic batch size. Every other piece is embedded the same
way each building block embeds any other: ``build_forward`` for a
:py:class:`Mapping <madspace.Mapping>`, ``build_function`` for a
:py:class:`FunctionGenerator <madspace.FunctionGenerator>`. The three Jacobians, from the flow,
the mapping and the cross section itself, multiply into one weight. Besides that weight, the
graph also returns ``y``, the point the flow produced before the mapping turned it into
momenta, needed below to train the flow:

.. code-block:: python

    fb = ms.FunctionBuilder(
        ms.NamedTypes([("batch_size", ms.Type([ms.batch_size]))]),
        ms.NamedTypes([
            ("weight", ms.batch_float),
            ("y", ms.batch_float_array(mapping.random_dim())),
        ]),
    )
    n = fb.input(0)
    r = fb.random(n, ms.Value(mapping.random_dim()))
    flow_out = flow.build_forward(fb, [r], [])
    mapping_out = mapping.build_forward(fb, [flow_out["data"]], [])
    pdf_id = fb.full([ms.Value(0), n])  # only one entry in pid_options
    xsec_out = cross_section.build_function(
        fb, [mapping_out["momenta"], mapping_out["x1"], mapping_out["x2"], pdf_id]
    )
    weight = fb.mul(fb.mul(flow_out["det"], mapping_out["det"]), xsec_out["matrix_element"])
    fb.output(0, weight)
    fb.output(1, flow_out["data"])
    sampling_func = fb.function()

The compiled function takes a single argument, how many events to generate.

Inspecting the graph
------------------------

Printing a :py:class:`Function <madspace.Function>` lists every input, trainable global,
instruction and output it was built from, which is often the fastest way to check that a
graph does what it should:

.. code-block:: python

    print(sampling_func)

Fusing a flow, a phase-space mapping, cuts, a PDF and a matrix element into one graph makes
for a long instruction list; the inputs, a few of the globals and instructions, and the
outputs already show the shape of it::

    Inputs:
      %0 : {batch_size}=batch_size
    Globals:
      %192 : float[1, 16, 11, 9850] = pdf_coefficients
      %190 : float[1, 198] = pdf_logx
      %103 : float[1, 93] = flow.subnet3a.layer2.bias
      %61 : float[1, 32, 4] = flow.subnet2a.layer1.weight
      ...
    Instructions:
      %2 = random(%0, 7)
      %4 = select(%2, {1, 3, 5})
      %6 = select(%2, {0, 2, 4, 6})
      ...                                    # 113 instructions in total
      %209 = mul(%118, %185)
      %210 = mul(%209, %208)
    Outputs:
      weight=%210 : float[batch_size]
      y=%116 : float[batch_size, 7]

The flow's own probability
------------------------------

Training needs the density the flow assigns to a point it already produced, not just the point
itself. A :py:class:`Flow <madspace.Flow>` is invertible, so :py:meth:`Mapping.inverse_function
<madspace.Mapping.inverse_function>` builds that as a standalone function of ``y``, with
:py:class:`madspace.torch.FunctionModule` wrapping it so PyTorch can differentiate through it
with respect to the flow's parameters:

.. code-block:: python

    flow_prob_module = FunctionModule(flow.inverse_function(), ctx)

Training
-----------

Sampling and evaluating the cross section, done through ``sampling_func`` above, does not
support PyTorch autograd, since the matrix element is not differentiable with respect to the
flow's parameters this way. Training instead follows MadNIS: draw a batch from the current
flow with ``sampling_func``, treat its weight as a constant, and adjust the flow's parameters
so that its own density tracks that weight, minimizing the self-normalized Kullback-Leibler
divergence between the two:

.. code-block:: python

    sampling_runtime = ms.FunctionRuntime(sampling_func, ctx)
    optimizer = torch.optim.Adam(flow_prob_module.parameters(), lr=1e-3)
    batch_size = np.array([256], dtype=np.int32)

    for step in range(300):
        weight, y = sampling_runtime(batch_size)
        weight, y = torch.tensor(weight), torch.tensor(y)

        optimizer.zero_grad()
        _, flow_prob = flow_prob_module(y)
        loss = -(weight / weight.mean() * torch.log(flow_prob)).mean()
        loss.backward()
        optimizer.step()

Because the flow already integrates to one over its own samples, the average weight is an
unbiased estimate of the cross section no matter how well the flow is trained; this loss only
reshapes the flow to reduce that estimate's variance.

Checking the result
-----------------------

The trained weights already live in ``ctx``, so evaluating ``sampling_func`` again with the
same NumPy runtime reports the trained integrator's performance:

.. code-block:: python

    def integrate(n_events):
        weight, _ = sampling_runtime(np.array([n_events], dtype=np.int32))
        return weight.mean(), weight.std() / np.sqrt(n_events)

    sigma, error = integrate(50000)
    print(f"sigma = {sigma:.2f} +- {error:.2f} pb  (rel. error {error / sigma * 100:.2f}%)")

::

    sigma = 239.42 +- 1.15 pb  (rel. error 0.48%)

This is a shorter, less tuned training loop than either MadGraph7's own event generator or the
external MadNIS package, both of which use more sophisticated losses and training schedules.
Even so, the relative error is well below what an untrained flow gives for the same number of
samples.
