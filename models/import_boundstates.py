import os
import sys


def get_boundstates_ufo(model):
    """Return the Fock states (Boundstate objects) shipped by a UFO model.

    `model` is either a loaded model or the path of a UFO directory. The
    states are read from that UFO's own boundstates.py only, so nothing from
    a model imported earlier in the same session can leak in.
    """

    if isinstance(model, str):
        path = model
    else:
        path = model.get('modelpath') if 'modelpath' in model else None
        path = path or model.__dict__.get('path')
    if not path or not os.path.isfile(os.path.join(path, 'boundstates.py')):
        return []

    # boundstates.py does a top-level 'import object_library', and every UFO
    # ships its own object_library.py. Both names stay in sys.modules once
    # imported, which is how the states of one model used to survive into the
    # next. Import them from this UFO only, then restore whatever those names
    # referred to before.
    names = ('boundstates', 'object_library')
    saved = dict((n, sys.modules.pop(n)) for n in names if n in sys.modules)
    sys.path.insert(0, path)
    try:
        import boundstates
        return list(boundstates.all_boundstates)
    except Exception:
        return []
    finally:
        sys.path.remove(path)
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
