# Standalone matrix elements: the flavor-aware API and Python linking

This note covers the flavor-aware **standalone** output and how to call it from
Python through the generated `f2py` module.

With merged particles (e.g. the light-quark group used by `p`/`j`, or a
multiparticle you `define`), a single standalone matrix element serves several
physical flavor combinations. Every standalone entry point therefore takes a
**flavor selector** so you can pick which combination to evaluate. A process
*without* merged particles is simply the single-flavor case (selector `1`).

---

## 1. Generate the standalone output

```
mg5_aMC
> generate p p > j j QCD=0
> output standalone /path/to/MYPROC --prefix=int
```

This writes one subprocess directory per group, e.g.
`/path/to/MYPROC/SubProcesses/P1_qq_qq/`, each containing `matrix.f`,
`check_sa.f`, the `f2py` wrapper (`f2py_matrix_wrapper.f`) and a
`flavor_dispatch.py` helper.

### The `--prefix` option

`--prefix` is the canonical flag for python-linkable standalone output; it
prefixes the matrix-element routine names so several process modules can be
imported into the *same* Python session without symbol / COMMON-block clashes:

* `--prefix=int`  → `M<n>_` (e.g. `M0_SMATRIX`, `PY_M0_GET_VALUE`),
* `--prefix=proc` → the process shell name,
* omitted        → no prefix (`SMATRIX`, `PY_GET_VALUE`): fine for a single
  module, but two such modules would clash.

The `f2py` wrapper and `flavor_dispatch.py` are generated either way; the
dispatcher resolves entry points by suffix, so your Python code below is the
**same regardless of the prefix**. Use `--prefix=int` unless you have a reason
not to.

### The flavor selector

A flavor can be given in two equivalent ways:

* **Flavor index** — an integer in `[1, NFLAV]` (the column number in the
  allowed-flavor table; `NFLAV` is a `PARAMETER` in `matrix.f`). Index `1` is
  always valid; an out-of-range / not-allowed flavor returns `0`.
* **Flavor array** — a length-`NEXTERNAL` vector of **per-leg group positions**:
  for each external leg, the 1-based position of the actual particle inside its
  merged-particle group (`1` for an unmerged leg). It is resolved to an index
  internally via `GET_FLAVOR_INDEX`.

For a non-merged process there is a single flavor: index `1` (or the all-ones
array).

---

## 2. Build the Python (`f2py`) module

`f2py` (shipped with `numpy`) must be on the `PATH`. In the subprocess
directory:

```
cd /path/to/MYPROC/SubProcesses/P1_qq_qq
make matrix2py.so
```

This produces `matrix2py*.so`, importable as `matrix2py`. The module is linked
against a companion shared library (`libme<PROC>.so`/`.dylib`) in the same
directory, so make sure it is loadable — easiest is to run Python from the
subprocess directory, or add it to the library path (`LD_LIBRARY_PATH` on Linux,
`DYLD_LIBRARY_PATH` on macOS).

---

## 3. Call it from Python

The generated `matrix2py` module exposes, for each entry point, **two** flavors:
a `*smatrix` / `*smatrixhel` / `*get_value` taking the `FLAVOR` array, and a
matching `*_idx` taking the integer index. The shipped `flavor_dispatch.py`
hides that split: pass *either* form and it routes to the right entry.

```python
import sys
sys.path.insert(0, '/path/to/MYPROC/SubProcesses/P1_qq_qq')

import matrix2py
from flavor_dispatch import FlavorDispatch

me = FlavorDispatch(matrix2py)
me.initialisemodel('/path/to/MYPROC/Cards/param_card.dat')

# P: momenta (E, px, py, pz) per external leg, shape (4, NEXTERNAL)
P = [[500., 500., 250., 250.],     # energies
     [  0.,   0., 180., -180.],    # px
     [  0.,   0.,   0.,    0.],    # py
     [500.,-500.,   0.,    0.]]    # pz
alphas = 0.118
nhel   = -1                        # -1 = sum over helicities

# Select the flavor by INDEX ...
ans = me.get_value(P, alphas, nhel, 2)
# ... or by the per-leg group-position ARRAY (length NEXTERNAL):
ans = me.get_value(P, alphas, nhel, [1, 1, 1, 1])

print('|M|^2 =', ans)
```

Dispatch rule: a scalar integer (or length-1 sequence) → the `_idx` entry; a
length-`NEXTERNAL` sequence → the array entry.

Available calls on `FlavorDispatch`:

| method | signature | returns |
|---|---|---|
| `get_value(P, alphas, nhel, flavor)` | full evaluation; `nhel=-1` sums helicities | `|M|^2` |
| `smatrix(P, flavor)` | summed/averaged over colors & helicities | `|M|^2` |
| `smatrixhel(P, hel, flavor)` | single helicity configuration `hel` | `|M|^2` |
| `initialisemodel(param_card)` | load parameters (call once) | — |

You can also call the raw module functions directly if you prefer, e.g.
`matrix2py.get_value(P, alphas, nhel, flav_idx)` (index) or the auto-detected
`*_idx` / array variants — the dispatcher just removes the need to know the
name-mangling/prefix.

---

## 4. Quick check from the shell (no Python)

`launch` has a good-helicity check mode that prints the matrix-element value of
**every** flavor the merged process serves (after the good-helicity filter is
active), instead of a timing table:

```
mg5_aMC
> launch /path/to/MYPROC --timings=21 --nb_run=0
```

`--timings=N` calls `SMATRIX` N times per flavor (use N above the warm-up
threshold, e.g. 21, so the per-flavor good-helicity filter is exercised);
`--nb_run=0` selects the value-printing mode. This is handy to sanity-check the
per-flavor values before wiring the module into Python.
