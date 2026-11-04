"""Loads the functions of the [custom_functions] run card section.

Shared by bin/generate_events of a process directory and of a gridpack. A path
ending in .so is a UMAMI library built by bin/init_custom_func, a path ending in
.py a Python module defining custom_<kind>, FRAMEWORK and optionally the
PARAMETERS it takes, with their defaults.
"""

import importlib.util
import logging
import os

logger = logging.getLogger("madevent")

# the outputs of custom_<kind> in a Python module
PYTHON_OUTPUTS = {
    "cut": ["cut"],
    "scale": ["ren_scale", "fact_scale1", "fact_scale2"],
    "bias": ["bias"],
    "pdf": ["pdf"],
}
# the arguments passed to custom_<kind> in a Python module
PYTHON_INPUTS = {
    "cut": ["momenta", "pdg_ids"],
    "scale": ["momenta", "pdg_ids"],
    "bias": ["momenta", "pdg_ids"],
    "pdf": ["beam_index", "pdg_id", "momentum_fraction", "scale"],
}

_modules = {}


def _import_module(path):
    """Import the module at ``path`` once per process."""
    path = os.path.abspath(path)
    if path not in _modules:
        spec = importlib.util.spec_from_file_location(
            f"mg7_custom_function_{len(_modules)}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _modules[path] = module
    return _modules[path]


def _cut_as_float(function, framework):
    """Wrap a cut returning booleans, so that madspace gets float64 1 or 0."""
    if framework == "torch":
        import torch

        def cut(*args):
            result = function(*args)
            return result.to(torch.float64) if isinstance(result, torch.Tensor) else result
    else:
        import numpy as np

        def cut(*args):
            result = function(*args)
            return result.astype(np.float64, copy=False) if isinstance(result, np.ndarray) else result

    cut.__name__ = getattr(function, "__name__", "custom_cut")
    return cut


def _three_scales(function):
    """Wrap a scale function, so that a single result is used for all scales."""

    def scale(*args):
        result = function(*args)
        return tuple(result) if isinstance(result, (tuple, list)) else (result,) * 3

    scale.__name__ = getattr(function, "__name__", "custom_scale")
    return scale


def _python_function(path, name, device_types):
    """The function ``name`` of the Python module at ``path``, its FRAMEWORK,
    checked against the devices of the run, and the module."""
    module = _import_module(path)
    function = getattr(module, name, None)
    if not callable(function):
        raise RuntimeError(f"{path} does not define a function {name}")
    framework = getattr(module, "FRAMEWORK", "numpy")
    if framework not in ("numpy", "torch"):
        raise RuntimeError(f"{path}: FRAMEWORK must be 'numpy' or 'torch', not {framework!r}")
    if framework == "numpy" and any(device != "cpu" for device in device_types):
        raise RuntimeError(
            f"{path}: numpy functions run on the CPU only, use FRAMEWORK = 'torch' "
            "for GPU runs"
        )
    if framework == "torch" and "cpu" in device_types:
        import torch

        # madspace already splits CPU batches over its threads
        if torch.get_num_threads() != 1:
            logger.info("Limiting torch to one thread per madspace thread")
            torch.set_num_threads(1)

    return function, framework, module


def load_custom_function(ms, contexts, device_types, kind, path, parameters=None):
    """Load the custom ``kind`` function at ``path`` into each context, in the
    same order everywhere, with the values of ``parameters`` ({name: value})
    replacing the defaults. Returns the ``MatrixElementApi`` of each context."""
    parameters = {str(k): float(v) for k, v in (parameters or {}).items()}
    if path.endswith(".so"):
        try:
            apis = [context.load_matrix_element(path, "", parameters)
                    for context in contexts]
        except RuntimeError as error:
            if "could not set parameter" not in str(error):
                raise
            raise RuntimeError(
                f"[custom_functions] {kind}_parameters: {error}. Declare the "
                f"parameters of {path} in custom_{kind}/parameters.h"
            ) from error
        for api, device in zip(apis, device_types):
            if api.device_type() != device:
                raise RuntimeError(
                    f"[custom_functions] {kind} = {path!r}: the library was built for "
                    f"{api.device_type()}, but the run uses {device}. Rebuild it with "
                    "bin/init_custom_func for that device."
                )
        return apis
    if not path.endswith(".py"):
        raise RuntimeError(
            f"[custom_functions] {kind} = {path!r}: expected a .so library or a .py file"
        )

    name = f"custom_{kind}"
    function, framework, module = _python_function(path, name, device_types)
    declared = getattr(module, "PARAMETERS", None)
    unknown = sorted(set(parameters) - set(declared or {}))
    if unknown:
        raise RuntimeError(
            f"[custom_functions] {kind}_parameters: {path} does not declare "
            f"{', '.join(unknown)} in its PARAMETERS"
        )
    if kind == "cut":
        function = _cut_as_float(function, framework)
    elif kind == "scale":
        function = _three_scales(function)
    python_function = ms.PythonFunction(
        function,
        PYTHON_INPUTS[kind],
        PYTHON_OUTPUTS[kind],
        framework=framework,
        name=name,
        parameters=declared,
    )
    return [
        context.add_matrix_element(python_function, parameters=parameters)
        for context in contexts
    ]


def _exports(path, symbol):
    """Whether the shared library at ``path`` exports ``symbol``."""
    import ctypes

    return hasattr(ctypes.CDLL(os.path.abspath(path)), symbol)


def load_surrogate(
    ms, contexts, backends, device_types, path, subprocess, pdg_ids, param_card,
    parameters=None,
):
    """Load the [surrogate] function for one subprocess into each context, and
    return the ``MatrixElementApi`` of each context.

    A .so path is a UMAMI library with the momenta, alpha_s and flavor index as
    inputs, where ``{subprocess}`` and ``{device}`` are replaced, initialized
    with ``param_card``. It gets the run-time ``parameters`` ({name: value}) of
    the madmatrix libraries if it implements umami_set_parameter. A .py file
    defines ``custom_surrogate(momenta, pdg_ids, alpha_s)``, which returns
    |M|^2; ``pdg_ids[flavor_index]`` are the PDG ids of the subprocess for each
    flavor index of its matrix element."""
    if path.endswith(".so"):
        apis = []
        for context, backend, device in zip(contexts, backends, device_types):
            library = path.format(subprocess=subprocess, device=backend)
            if not os.path.isfile(library):
                raise RuntimeError(f"[surrogate] function = {path!r}: {library} not found")
            library_parameters = {}
            if parameters and _exports(library, "umami_set_parameter"):
                library_parameters = {str(k): float(v) for k, v in parameters.items()}
            try:
                api = context.load_matrix_element(library, param_card, library_parameters)
            except RuntimeError as error:
                if "could not set parameter" not in str(error):
                    raise
                raise RuntimeError(
                    f"[surrogate] function = {path!r}: {error}. The surrogate gets the "
                    f"run-time parameters of the madmatrix libraries "
                    f"({', '.join(library_parameters)}) and has to accept them."
                ) from error
            if api.device_type() != device:
                raise RuntimeError(
                    f"[surrogate] function = {path!r}: {library} was built for "
                    f"{api.device_type()}, but the run uses {device}"
                )
            apis.append(api)
        return apis
    if not path.endswith(".py"):
        raise RuntimeError(
            f"[surrogate] function = {path!r}: expected a .so library or a .py file"
        )
    function, framework, _ = _python_function(path, "custom_surrogate", device_types)
    particle_count = len(pdg_ids[0])
    if framework == "torch":
        import torch

        tables = {}

        def surrogate(momenta, flavor_index, alpha_s):
            device = momenta.device
            if device not in tables:
                tables[device] = torch.tensor(pdg_ids, dtype=torch.int32, device=device)
            return function(momenta, tables[device][flavor_index.long()], alpha_s)

    else:
        import numpy as np

        table = np.array(pdg_ids, dtype=np.int32)

        def surrogate(momenta, flavor_index, alpha_s):
            return function(momenta, table[flavor_index], alpha_s)

    python_function = ms.PythonFunction(
        surrogate,
        ["momenta", "flavor_index", "alpha_s"],
        ["matrix_element"],
        framework=framework,
        particle_count=particle_count,
        name=f"custom_surrogate ({subprocess})",
    )
    return [context.add_matrix_element(python_function) for context in contexts]
