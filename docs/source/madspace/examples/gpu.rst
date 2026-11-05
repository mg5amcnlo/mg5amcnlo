Running on a GPU
===================

Everything in the earlier examples runs on the CPU by default. This example samples the same
``g g > t t~ g`` mapping from the first example on a CUDA device instead. The mapping itself
does not change. Only the :py:class:`Context <madspace.Context>` it runs against, and the
array library used to hand it data, are different: on a GPU, arrays are PyTorch tensors
rather than NumPy arrays, since MadSpace interfaces with whichever array library the caller
uses.

This requires a MadSpace build with the CUDA backend enabled (``--cuda`` to
``madspace/install.py``; the pre-built wheels for Linux include it). The same pattern works
for HIP, using :py:func:`hip_device <madspace.hip_device>` in place of
:py:func:`cuda_device <madspace.cuda_device>`.

Selecting the device
-----------------------

:py:func:`cuda_device <madspace.cuda_device>` raises if the CUDA backend was not built into
the installation:

.. code-block:: python

    import torch
    import madspace as ms

    device = ms.cuda_device()
    context = ms.Context(device=device, thread_count=32)

Building a runtime bound to the context
------------------------------------------

``mapping.map_forward`` is a CPU-only convenience, so a specific device needs the underlying
:py:class:`Function <madspace.Function>` and :py:class:`FunctionRuntime
<madspace.FunctionRuntime>` explicitly:

.. code-block:: python

    masses = [0.0, 0.0, 173.0, 173.0, 0.0]  # g g -> t t~ g
    mapping = ms.PhaseSpaceMapping(masses, 13000.0, mode="rambo")

    forward = ms.FunctionRuntime(mapping.forward_function(), context)

Sampling with PyTorch
-------------------------

Calling the runtime with a CUDA tensor keeps the whole computation on the device. The outputs
come back in the same order as :py:meth:`map_forward <madspace.Mapping.map_forward>` uses,
just as a plain tuple rather than a namedtuple:

.. code-block:: python

    r = torch.rand((100000, mapping.random_dim()), device="cuda", dtype=torch.float64)
    momenta, x1, x2, det = forward(r)
    print(momenta.shape, momenta.device)

::

    torch.Size([100000, 5, 4]) cuda:0

``momenta``, ``x1``, ``x2`` and ``det`` are ordinary CUDA tensors. Move them to the host with
``momenta.cpu()`` once the computation that needs them, such as a matrix-element evaluation,
is also done.
