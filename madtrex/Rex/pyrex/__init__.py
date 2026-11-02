import sys as _sys

from . import _rex_py
from ._rex_py import *

# tea is a pybind11 submodule (m.def_submodule("tea")) living inside _rex_py,
# not a separate file -- expose it explicitly as pyrex.tea rather than relying
# on wildcard-import semantics to carry it over. Also register it under
# sys.modules so `import pyrex.tea` works as a normal dotted import, not just
# attribute access after `import pyrex` -- pybind11 doesn't do this
# automatically since the submodule's own __name__ is "_rex_py.tea", not
# "pyrex.tea"
tea = _rex_py.tea
_sys.modules[__name__ + ".tea"] = tea
