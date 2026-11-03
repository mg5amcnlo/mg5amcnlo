################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Generation of the '[amplicol]' artifacts with pyamplicol, run in the process
of export_amplicol. pyamplicol (and Symbolica) are imported here, after
bootstrap.ensure_pyamplicol put them on sys.path."""

import json
import logging


def model_source(pyamplicol, model):
    """The model compiled for pyamplicol, cached in the output and compiled
    again unless the key of the cache is model['key']."""
    import os
    import shutil
    try:
        with open(model['key_file']) as stream:
            cached = json.load(stream).get('key')
    except (OSError, ValueError, AttributeError):
        cached = None
    if cached != model['key'] or not os.path.exists(model['cache']):
        from pyamplicol.config import EvaluatorConfig, JITConfig
        if os.path.exists(model['key_file']):
            os.remove(model['key_file'])
        if os.path.isdir(model['cache']):
            shutil.rmtree(model['cache'])
        elif os.path.exists(model['cache']):
            os.remove(model['cache'])
        source = pyamplicol.ModelSource.from_path(model['path'],
                                                  restriction=model['restriction'])
        source.compile(prepared_output=model['cache'],
                       evaluator=EvaluatorConfig(jit=JITConfig(optimization_level=2)))
        with open(model['key_file'], 'w') as stream:
            json.dump({'key': model['key'], 'sources': model['sources']}, stream,
                      indent=2)
    return pyamplicol.ModelSource.from_path(model['cache'])


def generate_artifact(pyamplicol, artifact, accuracy, process, model):
    from pyamplicol.config import resolve_config
    # workers and cores are left to pyamplicol, which clamps them without a license
    config = resolve_config({
        'schema_version': 1,
        'action': 'generate',
        'color': {'accuracy': accuracy},
        'process': {'multiparticles': process['multiparticles'],
                    'flavor_blind_strict': True,
                    'coupling_order_policy': 'explicit',
                    'max_coupling_orders': process['orders']},
        # pack_payloads: one payloads.pacbin instead of files per process
        'generation': {'emit_api_bundle': True, 'pack_payloads': True,
                       'relation_discovery': {'mode': 'off'}},
        'evaluator': {'backend': 'jit', 'execution_mode': 'recurrence',
                      'jit': {'optimization_level': 2}},
    })
    pyamplicol.Generator(config).generate(process['expression'], artifact,
                                          model=model, mode='replace')


def generate(job):
    """Generate the artifacts of the processes of a job (see
    export_amplicol.export_amplicol_processes) and return the reweight providers
    of each of them."""
    import os
    import pyamplicol
    from pyamplicol.artifacts import umami_reweight
    from pyamplicol.licensing import detect_symbolica_license
    if not detect_symbolica_license(suggest=False).licensed:
        logging.getLogger('madgraph.amplicol').warning(
            'Symbolica does not accept the license key of pyamplicol (%s): '
            'generating in its restricted mode', pyamplicol.__file__)
    model = model_source(pyamplicol, job['model'])
    providers = []
    for process in job['processes']:
        path = process['path']
        for accuracy in ('lc', 'full'):
            generate_artifact(pyamplicol, os.path.join(path, accuracy), accuracy,
                              process, model)
        providers.append(umami_reweight.write_reweight_provider(
            os.path.join(path, 'lc'), os.path.join(path, 'full'),
            os.path.join(path, 'reweight')))
    return providers
