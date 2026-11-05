"""Flavor dispatch helper for the f2py ``matrix2py`` standalone module.

The compiled module exposes, for each entry point, two flavors:

* a ``...smatrix`` / ``...get_value`` / ``...smatrixhel`` function taking the
  full per-leg ``FLAVOR(NEXTERNAL)`` array, and
* a matching ``..._idx`` function taking a single integer flavor index.

This wrapper lets you call one Python function and pass *either* form: a scalar
integer (or length-1 sequence) is routed to the ``_idx`` entry, a
length-``NEXTERNAL`` sequence to the array entry. The underlying function names
are auto-detected, so this works regardless of the f2py name-mangling/prefix.

Example
-------
>>> import matrix2py
>>> from flavor_dispatch import FlavorDispatch
>>> me = FlavorDispatch(matrix2py)
>>> me.initialisemodel('param_card.dat')
>>> ans = me.get_value(P, alphas, nhel, 3)              # by flavor index
>>> ans = me.get_value(P, alphas, nhel, [1, -1, 2, -2]) # by flavor array
"""

import numbers


def _is_index(flavor):
    """A scalar integer (or length-1 sequence) is treated as a flavor index;
    anything longer is treated as a per-leg FLAVOR array."""
    if isinstance(flavor, numbers.Integral):
        return True
    try:
        return len(flavor) == 1
    except TypeError:
        # No __len__ (e.g. a non-Integral scalar such as numpy.int64 that is
        # not registered with numbers.Integral): treat it as a flavor index.
        # We only swallow TypeError here, the specific error raised by len()
        # on a sizeless object, so genuine errors elsewhere are not masked.
        return True


def _as_index(flavor):
    if isinstance(flavor, numbers.Integral):
        return int(flavor)
    return int(flavor[0])


class FlavorDispatch(object):
    """Thin convenience layer over the f2py matrix2py module."""

    #: entry point -> position of the flavor argument in the call
    _flavor_pos = {
        'smatrix': 1,        # (p, flavor)
        'smatrixhel': 2,     # (p, hel, flavor)
        'get_value': 3,      # (p, alphas, nhel, flavor)
    }

    def __init__(self, module):
        self.module = module
        self._cache = {}

    def _resolve(self, base):
        """Return (array_func, idx_func) for *base*, auto-detected from the
        module's attributes (case-insensitive, suffix match)."""
        if base in self._cache:
            return self._cache[base]
        array_f = idx_f = None
        for name in dir(self.module):
            low = name.lower()
            if low.endswith(base + '_idx'):
                idx_f = getattr(self.module, name)
            elif low.endswith(base):
                array_f = getattr(self.module, name)
        self._cache[base] = (array_f, idx_f)
        return array_f, idx_f

    def _call(self, base, args):
        array_f, idx_f = self._resolve(base)
        pos = self._flavor_pos[base]
        flavor = args[pos]
        if _is_index(flavor):
            if idx_f is None:
                raise AttributeError(
                    "No function ending with '%s_idx' found in f2py module "
                    "for index calls" % base)
            args = list(args)
            args[pos] = _as_index(flavor)
            return idx_f(*args)
        if array_f is None:
            raise AttributeError(
                "No function ending with '%s' found in f2py module "
                "for array calls" % base)
        return array_f(*args)

    # -- dispatching entry points (flavor may be an index or an array) --------
    def smatrix(self, p, flavor):
        return self._call('smatrix', [p, flavor])

    def smatrixhel(self, p, hel, flavor):
        return self._call('smatrixhel', [p, hel, flavor])

    def get_value(self, p, alphas, nhel, flavor):
        return self._call('get_value', [p, alphas, nhel, flavor])

    # -- pass-through for the model initialiser -------------------------------
    def initialisemodel(self, path):
        for name in dir(self.module):
            if name.lower().endswith('initialisemodel'):
                return getattr(self.module, name)(path)
        raise AttributeError("No 'initialisemodel' entry found")
