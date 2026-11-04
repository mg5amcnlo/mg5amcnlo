Custom functions
================

MG7 can call your own code for four parts of the calculation:

``cut``
    Decides if a phase-space point passes the cuts. It is applied on top of the
    ``[cuts]`` of the :doc:`run card <run_card>`.

``scale``
    The renormalization scale and the factorization scales of both incoming partons. It
    replaces ``dynamical_scale_choice``, like ``user_dynamical_scale`` in MadEvent.

``pdf``
    The parton densities. It replaces the LHAPDF set of ``[beam] pdf``.

``bias``
    An event bias. More events are generated where the bias is large, and their weights
    are lowered to compensate.

You write the function either for a single phase-space point in C, C++, Fortran, CUDA or
HIP, or for a whole batch of points in Python with numpy or PyTorch. MG7 builds a compiled
function into a shared library and calls it on batches of events. The library implements
the :doc:`UMAMI interface <madspace/umami-api>`, the same interface the matrix elements
use. A generated wrapper takes care of that, so you only write the function itself. A
Python function needs no build. MG7 imports it and calls it through the same interface.

Creating a custom function
--------------------------

Run ``bin/init_custom_func`` in the process directory. Without options, it asks which
function and language you want:

.. code-block:: text

    $ ./bin/init_custom_func
    Which custom function do you want to create?
      1. cut      a cut, true keeps the event and false rejects it
      2. scale    the renormalization and factorization scales
      3. pdf      a parton density x f(x, Q)
      4. bias     an event bias for the unweighting
    Which language do you want to write it in?
      1. c             C
      2. cpp           C++
      3. fortran       Fortran
      4. cuda          CUDA, for runs on NVIDIA GPUs
      5. hip           HIP, for runs on AMD GPUs
      6. python-numpy  Python with numpy, for CPU runs, no compilation
      7. python-torch  Python with PyTorch, for CPU or GPU runs, no compilation

The same choices are available as options:

.. code-block:: bash

    ./bin/init_custom_func --function cut --language cpp

``-f`` and ``-l`` are short for ``--function`` and ``--language``. ``-y`` takes the
defaults (``cut`` and ``cpp``) for the options you leave out instead of asking.
``--force`` overwrites an existing directory. ``--no-build`` skips the build.
``--no-set-run-card`` leaves the run card unchanged.

For a compiled language, the script creates a directory ``custom_<function>``:

``custom_<function>.<ext>``
    The function body.

``parameters.h``
    The parameters of the function, see :ref:`custom-function-parameters`.

``wrapper_cpu.c`` or ``wrapper_gpu.cu``
    The UMAMI wrapper. It unpacks the batched inputs and calls your function once per
    event.

``umami.h``
    The UMAMI header.

``Makefile``
    Builds ``lib/custom_<function>.so``.

For ``python-numpy`` and ``python-torch``, the directory only holds the function,
``custom_<function>.py``.

By default, the script builds the default function right away and enables it in
``Cards/run_card.toml``:

.. code-block:: toml

    [custom_functions]
    cut = "lib/custom_cut.so"

A Python function is enabled with the path of its file:

.. code-block:: toml

    [custom_functions]
    cut = "custom_cut/custom_cut.py"

``bin/generate_events`` runs ``make`` in ``custom_<function>`` before each run. Your
changes are therefore picked up without a manual rebuild. The build output goes to
``custom_<function>/compile.log``.

Function signatures
-------------------

The cut, scale and bias functions receive the external particles of one event. The
incoming particles come first. The momenta are in the lab frame, in GeV, with components
``(E, px, py, pz)``. The PDG ids are those of the first flavor of the subprocess for cuts
and scales, like the built-in cuts, and those of the sampled flavor for the bias. Every
function also receives its parameters, see :ref:`custom-function-parameters`.

C, C++ and CUDA/HIP:

.. code-block:: c

    #include "parameters.h"

    bool custom_cut(int particle_count, const int* pdg_ids, const double momenta[][4],
                    const CustomParameters* params);
    void custom_scale(int particle_count, const int* pdg_ids, const double momenta[][4],
                      const CustomParameters* params,
                      double* ren_scale, double* fact_scale1, double* fact_scale2);
    double custom_bias(int particle_count, const int* pdg_ids, const double momenta[][4],
                       const CustomParameters* params);
    double custom_pdf(int beam_index, int pdg_id, double x, double q,
                      const CustomParameters* params);

In C, ``bool`` comes from ``<stdbool.h>``. In C++, the functions are declared
``extern "C"``. In CUDA and HIP, they are ``__device__`` functions and run on the GPU, one
thread per event.

Fortran, in a ``.F90`` file, so that ``parameters.h`` goes through the preprocessor:

.. code-block:: fortran

    function custom_cut(particle_count, pdg_ids, momenta, params) bind(C, name="custom_cut") result(value)
        use iso_c_binding
        implicit none
    #include "parameters.h"
        integer(c_int), value :: particle_count
        integer(c_int), intent(in) :: pdg_ids(particle_count)
        real(c_double), intent(in) :: momenta(0:3, particle_count)
        type(custom_parameters), intent(in) :: params
        logical(c_bool) :: value

The bias returns ``real(c_double)``. The scale is a subroutine with the additional
arguments ``real(c_double), intent(out) :: ren_scale, fact_scale1, fact_scale2``. The PDF
takes ``integer(c_int), value :: beam_index, pdg_id`` and ``real(c_double), value :: x, q``
before ``params``.

What the functions return:

``cut``
    True to keep the event, false to reject it.

``scale``
    The renormalization scale ``ren_scale`` and the factorization scales
    ``fact_scale1`` and ``fact_scale2`` of the first and second incoming particle,
    ``momenta[0]`` and ``momenta[1]``, in GeV. ``scale_factor`` is applied on top.
    ``fixed_ren_scale`` and ``fixed_fact_scale`` still take precedence.

``pdf``
    :math:`x f(x, Q)` for the parton ``pdg_id`` (21 for the gluon) in beam ``beam_index``
    (0 or 1), at momentum fraction ``x`` and factorization scale ``q``.

``bias``
    The bias factor :math:`b > 0`. The phase space is sampled and unweighted according to
    :math:`|w|\,b`. The written events carry the weight divided by :math:`b`. Distributions
    filled with the event weights therefore stay unbiased, while regions with a large bias
    get more events. The reported cross section does not include the bias.

.. _custom-function-parameters:

Parameters
----------

A function can take parameters whose values are set in the run card, so that a cut value,
for example, can be changed without editing the function. For a compiled function, they
are declared with their default values in ``custom_<function>/parameters.h``:

.. code-block:: c

    #define CUSTOM_PARAMETERS(X) \
        X(pt_min, 20.)           \
        X(eta_max, 2.5)

The function reads them from ``params``, as ``params->pt_min`` in C, C++ and CUDA/HIP and
as ``params%pt_min`` in Fortran. Rebuild the function after changing the list. A Python
function declares them in the dictionary ``PARAMETERS`` and gets them as its last argument,
``params.pt_min``:

.. code-block:: python

    PARAMETERS = {"pt_min": 20., "eta_max": 2.5}

    def custom_cut(momenta, pdg_ids, params):
        ...

A Python module without ``PARAMETERS`` has no ``params`` argument. The values are set in
the ``[custom_functions]`` section of the run card, as ``<function>_parameters.<name>``:

.. code-block:: toml

    [custom_functions]
    cut = "lib/custom_cut.so"
    cut_parameters.pt_min = 30.0

Parameters that are not set keep their default value. Setting a parameter that the
function does not declare is an error. All parameters are floating-point numbers. A
gridpack keeps the values of the run that made it.

Python functions
----------------

A Python function is called once per batch of phase-space points, with arrays that hold
one entry per point. The file defines the function ``custom_<function>``, the framework it
uses and optionally its parameters:

.. code-block:: python

    import numpy as np

    FRAMEWORK = "numpy"  # or "torch"
    PARAMETERS = {"pt_min": 20.}

    def custom_cut(momenta, pdg_ids, params):
        ...

The cut, scale and bias functions take ``(momenta, pdg_ids, params)``:

``momenta``
    float64, shape ``(batch, particle_count, 4)``. The momenta ``(E, px, py, pz)`` of the
    external particles in GeV, lab frame, incoming particles first.

``pdg_ids``
    int32, shape ``(batch, particle_count)``. Their PDG ids, as for the compiled
    functions.

``params``
    The values of ``PARAMETERS``, as attributes.

The PDF function takes ``(beam_index, pdg_id, x, q, params)``, each with shape
``(batch,)`` except ``params``. ``beam_index`` and ``pdg_id`` are int32, ``x`` and ``q``
are float64.

The cut returns a boolean array of shape ``(batch,)``. The scale returns a tuple of the
three float64 arrays ``(ren_scale, fact_scale1, fact_scale2)``, or a single array used for
all three. The PDF and the bias return a float64 array of shape ``(batch,)``. The meaning
is the same as for the compiled functions. MG7 checks the type, the dtype, the shape and the
device of the result. With numpy the arguments are numpy arrays, and with
``FRAMEWORK = "torch"`` they are torch tensors. They are views of MG7's own buffers, so they
must not be modified or kept after the call.

numpy functions run on the CPU only. torch functions run on the device of the run, so the
same file works for CPU and GPU runs. On the GPU the function runs on the CUDA or HIP stream
of the MG7 runtime, and the arguments and the result stay on the GPU.

The same cut as in the C++ example below, with numpy:

.. code-block:: python

    import numpy as np

    FRAMEWORK = "numpy"
    PARAMETERS = {"pt_min": 150.}

    def custom_cut(momenta, pdg_ids, params):
        pt = np.hypot(momenta[:, 2:4, 1], momenta[:, 2:4, 2])
        return (pt >= params.pt_min).all(axis=1)

:math:`H_T/2` with torch, for CPU or GPU runs:

.. code-block:: python

    import torch

    FRAMEWORK = "torch"

    def custom_scale(momenta, pdg_ids):
        final = momenta[:, 2:]
        mt = torch.sqrt(torch.clamp(final[..., 0] ** 2 - final[..., 3] ** 2, min=0.))
        return mt.sum(dim=1) / 2.

On the CPU, MG7 splits each batch over the threads of ``cpu_thread_pool_size`` and calls
the function from these threads. Python runs one call at a time, but numpy and torch
release this lock inside their array operations, so vectorized functions still run
partly in parallel. MG7 limits torch to one thread on the CPU, so that it does not start
threads of its own on top. numpy uses several threads only for linear algebra, such as
matrix products. If your function relies on it, set ``OMP_NUM_THREADS=1``.

The Python API behind these functions is ``madspace.PythonFunction``. It can also be used
directly with madspace.

.. _custom-function-surrogate:

Python surrogates
-----------------

A Python surrogate of the matrix element (see :ref:`run-card-surrogate`) uses the same
mechanism. Its file defines ``FRAMEWORK`` and ``custom_surrogate(momenta, pdg_ids,
alpha_s)``, which returns :math:`|M|^2` as a float64 array of shape ``(batch,)``. It takes
no parameters. ``momenta`` and ``pdg_ids`` have the shapes and types given above, with the
momenta that the full matrix element gets, and ``alpha_s`` is a float64 array of shape
``(batch,)``. The function is called for all subprocesses, which ``pdg_ids`` identifies:

.. code-block:: python

    import numpy as np

    FRAMEWORK = "numpy"

    def custom_surrogate(momenta, pdg_ids, alpha_s):
        return np.ones(len(momenta))

Examples
--------

A cut requiring :math:`p_T > p_{T,\min}` for both particles of a ``p p > t t~`` process,
in C++, with ``X(pt_min, 150.)`` in ``parameters.h``:

.. code-block:: c++

    #include <cmath>
    #include "parameters.h"

    extern "C" bool custom_cut(int particle_count, const int* pdg_ids,
                               const double momenta[][4], const CustomParameters* params) {
        for (int i = 2; i < 4; ++i) {
            if (std::hypot(momenta[i][1], momenta[i][2]) < params->pt_min) {
                return false;
            }
        }
        return true;
    }

:math:`H_T/2` as all three scales, in Fortran:

.. code-block:: fortran

    subroutine custom_scale(particle_count, pdg_ids, momenta, params, ren_scale, fact_scale1, fact_scale2) &
        bind(C, name="custom_scale")
        use iso_c_binding
        implicit none
    #include "parameters.h"
        integer(c_int), value :: particle_count
        integer(c_int), intent(in) :: pdg_ids(particle_count)
        real(c_double), intent(in) :: momenta(0:3, particle_count)
        type(custom_parameters), intent(in) :: params
        real(c_double), intent(out) :: ren_scale, fact_scale1, fact_scale2
        integer :: i
        ren_scale = 0d0
        do i = 3, particle_count
            ren_scale = ren_scale + sqrt(max(momenta(0, i)**2 - momenta(3, i)**2, 0d0))
        end do
        ren_scale = ren_scale / 2d0
        fact_scale1 = ren_scale
        fact_scale2 = ren_scale
    end subroutine custom_scale

A bias that populates the :math:`p_T` tail of the first final-state particle, in C, with
``X(pt0, 100.)`` in ``parameters.h``:

.. code-block:: c

    #include <math.h>
    #include "parameters.h"

    double custom_bias(int particle_count, const int* pdg_ids, const double momenta[][4],
                       const CustomParameters* params) {
        double pt2 = momenta[2][1] * momenta[2][1] + momenta[2][2] * momenta[2][2];
        return fmax(pt2 / (params->pt0 * params->pt0), 1.);
    }

An LHAPDF set called through the LHAPDF C++ API:

.. code-block:: c++

    #include <mutex>
    #include "LHAPDF/LHAPDF.h"
    #include "parameters.h"

    extern "C" double custom_pdf(int beam_index, int pdg_id, double x, double q,
                                 const CustomParameters* params) {
        static LHAPDF::PDF* pdf = LHAPDF::mkPDF("NNPDF40_lo_as_01180", 0);
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        return pdf->xfxQ(pdg_id, x, q);
    }

The function is called from several threads, so the call into LHAPDF is protected by a
lock. To build it, pass the LHAPDF flags to ``make``:

.. code-block:: bash

    make -C custom_pdf CFLAGS="$(lhapdf-config --cflags)" \
        LIBS="$(lhapdf-config --libs) -Wl,-rpath,$(lhapdf-config --libdir)"

Building
--------

The ``Makefile`` takes the usual ``CC``, ``CXX`` and ``FC`` compilers, with ``gfortran``
as the default for Fortran. ``OPTFLAGS`` defaults to ``-O3``. Extra compiler flags go into
``CFLAGS`` and extra libraries into ``LIBS``. The CUDA and HIP builds detect the GPU
architecture like the matrix elements, and accept the same ``MADGRAPH_CUDA_ARCHITECTURE``
and ``MADGRAPH_HIP_ARCHITECTURE`` overrides.

The language decides the device the function runs on. C, C++ and Fortran functions run on
the CPU, and CUDA and HIP functions run on the GPU. Use the language that matches the
``device`` of the run. Python functions are not built. numpy functions run on the CPU, and
torch functions run on the device of the run.

Limitations
-----------

- With a custom PDF, :math:`\alpha_s` still comes from the PDF set of ``[beam] pdf``, and
  the scale and PDF variations of ``[systematics]`` are disabled.
- With a custom bias, the weighted integration histograms of ``[run]
  weighted_histograms`` include the bias. The event weights do not.
- Decays ignore a custom scale and a custom PDF.
- A gridpack stores the custom libraries and Python files it was made with. They cannot
  be changed in the gridpack. A Python function in a gridpack needs the same Python
  packages, such as numpy or torch, where the gridpack runs.
