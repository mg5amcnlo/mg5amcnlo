
def get_boundstates_ufo(model):
    """Bound states of a UFO model (all_boundstates of its boundstates.py).

    Only models whose name contains 'onia' or 'atom' have bound states. The
    module is imported from the directory of the model: 'modelpath' is set for
    every model (the attribute 'path' only for restricted ones), and a
    boundstates module of another model imported earlier in the session is
    not reused."""
    import os
    import sys
    name = model.get('name')
    if 'onia' not in name and 'atom' not in name:
        return []
    path = model.get('modelpath') or model.__dict__.get('path')
    if not path or not os.path.isfile(os.path.join(path, 'boundstates.py')):
        return []
    path = os.path.abspath(path)
    try:
        for p in list(sys.path):
            if p.endswith('sm'):
                sys.path.remove(p)
    except Exception:
        pass
    old = sys.modules.get('boundstates')
    if old is not None:
        old_dir = os.path.dirname(os.path.abspath(getattr(old, '__file__', '') or ''))
        if old_dir != path:
            del sys.modules['boundstates']
    sys.path.insert(0, path)
    try:
        import boundstates
        return boundstates.all_boundstates
    except Exception:
        return []
    finally:
        sys.path.remove(path)
        if path not in sys.path:
            sys.path.append(path)
