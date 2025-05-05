
def get_boundstates_ufo(model):
    import sys
    sys.path.append(model.__dict__.get('path'))
    try:
        import boundstates
        states = boundstates.all_boundstates
        return states
    except:
        return []
