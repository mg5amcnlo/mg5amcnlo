"""The cut kernels over a vector of observables, and the kernel signatures.

cut_all / cut_any apply one window [min, max] (two scalars, as the instruction
set declares them) to every observable of a vector: all of them inside, or at
least one inside. A kernel that took the bounds as FIn<T, 1> would make the
runtime build a 2-dimensional view of a 1-dimensional tensor from unset shape
and stride entries, so min[i] would read out of bounds (a memory fault on HIP,
silently min[0] on CPU and CUDA). The second test guards every kernel against
that mismatch: no kernel input may have more inner dimensions than its
instruction declares.
"""

import glob
import os
import re

import pytest
import torch
from madspace_testing import default_context

import madspace as ms

N_OBS = 4
LOW, HIGH = 1.0, 2.0

# rows: all inside, one outside below, one outside above, all outside, on the
# edges (inside: the window is closed)
OBS = torch.tensor(
    [
        [1.2, 1.5, 1.9, 1.1],
        [0.5, 1.5, 1.9, 1.1],
        [1.2, 1.5, 2.5, 1.1],
        [0.1, 3.0, 2.5, 0.9],
        [1.0, 2.0, 1.0, 2.0],
    ]
)


def run_cuts():
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("obs", ms.batch_float_array(N_OBS))]),
        ms.NamedTypes([("all", ms.batch_float), ("any", ms.batch_float)]),
    )
    obs = fb.input(0)
    fb.output(0, fb.cut_all(obs, LOW, HIGH))
    fb.output(1, fb.cut_any(obs, LOW, HIGH))
    runtime = ms.FunctionRuntime(fb.function(), default_context())
    return [torch.from_dlpack(t) for t in runtime.call([OBS])]


def test_cut_all_and_any_apply_one_window_to_every_observable():
    cut_all, cut_any = run_cuts()
    inside = (OBS >= LOW) & (OBS <= HIGH)
    torch.testing.assert_close(cut_all, inside.all(dim=1).double(), rtol=0, atol=0)
    torch.testing.assert_close(cut_any, inside.any(dim=1).double(), rtol=0, atol=0)


@pytest.mark.cpu_only
def test_kernel_inputs_have_the_declared_dimensions():
    yaml = pytest.importorskip("yaml")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    instruction_set = os.path.join(root, "instruction_set.yaml")
    kernel_files = glob.glob(os.path.join(root, "src", "kernels", "*.hpp"))
    if not os.path.exists(instruction_set) or not kernel_files:
        pytest.skip("madspace source tree not available")
    commands = {}
    with open(instruction_set) as f:
        for document in yaml.safe_load_all(f):
            if isinstance(document, dict):
                commands.update(document)
    source = "".join(open(path).read() for path in kernel_files)

    checked, mismatches = 0, []
    for name, command in commands.items():
        if not isinstance(command, dict) or "inputs" not in command:
            continue
        if command.get("custom_op"):
            continue
        match = re.search(r"\bkernel_%s\s*\(([^)]*)\)" % re.escape(name), source)
        if not match:
            continue
        checked += 1
        kernel_dims = [
            int(dims)
            for _kind, direction, dims in re.findall(
                r"\b([FIB])(In|Out)<\s*T\s*,\s*(\d+)\s*>", match.group(1)
            )
            if direction == "In"
        ]
        for entry, kernel_dim in zip(command["inputs"], kernel_dims):
            declared = entry.get("type", [])
            single = 1 if len(declared) > 1 and declared[1] == "single" else 0
            declared_dim = len(declared) - 1 - single
            if kernel_dim > declared_dim:
                mismatches.append(
                    "kernel_%s input %s: FIn<T, %d> for a declared %r"
                    % (name, entry.get("name"), kernel_dim, declared)
                )
    assert checked > 100, "found only %d kernels to check" % checked
    assert not mismatches, "\n".join(mismatches)
