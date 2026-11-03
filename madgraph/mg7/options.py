"""The MG5 options of the process directory and the LHAPDF paths they select."""

import os

from madgraph.various import misc

_ROOTED_OPTIONS = ('lhapdf', 'lhapdf_py3', 'lhapdf_py2', 'heptools_install_dir')

def load_mg5_options(me_dir=None) -> dict:
    """Read the tool paths from the MadGraph configuration, so the launcher
    knows where LHAPDF and the optional programs (Pythia8/Delphes/MadSpin/
    reweight/analysis) live.

    Files are read least- to most-specific, each overriding the previous, which
    is the same layering CommonRunCmd.set_configuration uses: the card inside
    this process directory has the last word, then this installation, then the
    per-user file. Relative values are resolved against the root of the file
    they came from.

    ``me_dir`` is the process directory whose Cards/me5_configuration.txt has
    the last word. It defaults to the working directory, which is right for
    bin/generate_events (it chdirs there first) but not for an in-process
    caller, which has not moved yet.
    """

    import madgraph
    mg5dir = os.path.dirname(os.path.dirname(os.path.abspath(madgraph.__file__)))
    me_dir = os.getcwd() if me_dir is None else me_dir

    options = {
        'pythia-pgs_path': None, 'pythia8_path': None,
        'madanalysis5_path': None, 'exrootanalysis_path': None, 'delphes_path': None,
        'rivet_path': None, 'contur_path': None, 'f2py_compiler': None,
        'lhapdf': None, 'lhapdf_py3': None, 'lhapdf_py2': None, 'timeout': 0,
        'heptools_install_dir': None, 'hepmc3_path': None,
    }
    config_files = []
    base_config = misc.base_config_file()
    if base_config:
        config_files.append((base_config, mg5dir))
    user_config = misc.user_config_file()
    if user_config:
        config_files.append((user_config, mg5dir))
    config_files.append((misc.install_config_file(mg5dir), mg5dir))
    config_files.append((os.path.join(me_dir, 'Cards', 'me5_configuration.txt'),
                         me_dir))
    for cfg, root in config_files:
        if not os.path.exists(cfg):
            continue
        with open(cfg) as fsock:
            for line in fsock:
                line = line.split('#', 1)[0]
                if '=' not in line:
                    continue
                name, value = (x.strip() for x in line.split('=', 1))
                if name not in options or value in ('', 'None'):
                    continue
                if (name.endswith('_path') or name in _ROOTED_OPTIONS) and \
                        not os.path.isabs(value) and \
                        (os.sep in value or value.startswith('.')):
                    value = os.path.normpath(os.path.join(root, value))
                options[name] = value
    options['mg5_path'] = mg5dir  # enables MadSpin/reweight
    return options


_LHAPDF = None

def lhapdf_paths(refresh: bool = False) -> misc.LhapdfPaths:
    """The LHAPDF installation this run should use, resolved once.

    Uses the same helper as the 'launch' command, so driving a process
    directory straight through bin/generate_events behaves identically to
    launching it from MadGraph.
    """

    global _LHAPDF
    if _LHAPDF is None or refresh:
        _LHAPDF = misc.resolve_lhapdf(load_mg5_options())
        # Export what we found: the real LHAPDF library, systematics and the
        # MadSpin/reweight subprocesses read these from the environment.
        if _LHAPDF.data_paths:
            os.environ["LHAPDF_DATA_PATH"] = os.pathsep.join(_LHAPDF.data_paths)
        if _LHAPDF.config:
            os.environ["MADGRAPH_LHAPDF_CONFIG"] = _LHAPDF.config
    return _LHAPDF


def lhapdf_index_paths() -> list:
    """The PDF directories to look an lhaid up in. Without LHAPDF there may be
    no pdfsets.index anywhere yet, so fetch one into the download directory."""

    paths = lhapdf_paths()
    data_paths = list(paths.data_paths)
    if not any(os.path.exists(os.path.join(p, "pdfsets.index")) for p in data_paths) \
            and paths.download_path \
            and misc.download_lhapdf_index(paths.download_path):
        data_paths.append(paths.download_path)
    return data_paths
