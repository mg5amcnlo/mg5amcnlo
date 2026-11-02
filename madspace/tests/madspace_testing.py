"""The device the tests run on, from MADSPACE_TEST_DEVICE: cpu (default), cuda or hip.

torch creates its tensors on DEVICE by default (set in conftest.py), and madspace runs a
function on the device of its inputs. torch calls the ROCm devices "cuda" too.
"""

import os

import numpy as np
import torch

import madspace as ms

BACKEND = os.environ.get("MADSPACE_TEST_DEVICE", "cpu").strip().lower() or "cpu"
if BACKEND not in ("cpu", "cuda", "hip"):
    raise ValueError(f"MADSPACE_TEST_DEVICE must be cpu, cuda or hip, not {BACKEND!r}")
DEVICE = torch.device("cpu" if BACKEND == "cpu" else "cuda")
ON_GPU = BACKEND != "cpu"


def ms_device():
    """The madspace device of the tests."""
    return {"cpu": ms.cpu_device, "cuda": ms.cuda_device, "hip": ms.hip_device}[
        BACKEND
    ]()


def default_context():
    """The default madspace context of the device of the tests."""
    return {
        "cpu": ms.default_context,
        "cuda": ms.default_cuda_context,
        "hip": ms.default_hip_context,
    }[BACKEND]()


def new_context():
    """A new madspace context on the device of the tests."""
    return ms.Context(ms_device())


def generator(seed):
    """A seeded torch generator for rand, randn and randint below."""
    return torch.Generator("cpu").manual_seed(seed)


# The random numbers are drawn on the CPU and moved to the device of the tests, so that
# every device gets the same inputs (and statistical tests the same outcome).
def _on_device(draw, args, kwargs):
    requires_grad = kwargs.pop("requires_grad", False)
    return draw(*args, device="cpu", **kwargs).to(DEVICE).requires_grad_(requires_grad)


def rand(*args, **kwargs):
    return _on_device(torch.rand, args, kwargs)


def randn(*args, **kwargs):
    return _on_device(torch.randn, args, kwargs)


def randint(*args, **kwargs):
    return _on_device(torch.randint, args, kwargs)


def to_numpy(tensor):
    """A tensor as a numpy array on the host, e.g. for scipy.

    Given numpy inputs, madspace returns read-only numpy arrays. Those are
    copied rather than passed to torch, which warns about read-only arrays.
    """
    if isinstance(tensor, np.ndarray):
        return np.array(tensor)
    return torch.as_tensor(tensor).detach().cpu().numpy()
