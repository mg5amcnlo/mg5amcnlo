
def get_boundstates_ufo(model):
    import sys
    if 'onia' in model.get('name'):
        try:
            for path in sys.path:
                if path.endswith('sm'):
                    sys.path.remove(path)
        except:
            pass
        sys.path.append(model.__dict__.get('path'))
    try:
        import boundstates
        states = boundstates.all_boundstates
        return states
    except:
        return []
