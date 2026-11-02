"""MadtRex: MatrixElementApi <-> REX::tea event reweighting for MadGraph7.

Built on top of madspace and pyrex (the bundled Rex fork); madspace itself
does not depend on this package.
"""

import os as _os

from . import _compat

# _madtrex_py is compiled against madspace's headers: refuse a madspace that
# changed since (a binary mismatch would otherwise crash or misbehave).
_madspace_dir = _compat.find_madspace_dir()
if _madspace_dir is not None:
    _problem = _compat.build_problem(_os.path.dirname(_os.path.abspath(__file__)),
                                     _madspace_dir)
    if _problem:
        raise ImportError(_problem)

# madspace first: it preloads libmadspace and registers the MatrixElementApi/
# Context/LHEMeta types that the _madtrex_py bindings use.
import madspace  # noqa: F401,E402
import pyrex  # noqa: F401,E402
import pyrex.tea  # noqa: F401,E402

from ._madtrex_py import *  # noqa: F401,F403,E402
