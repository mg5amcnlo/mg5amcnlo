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
"""Generation of the '[amplicol]' artifacts with pyamplicol, run in its own
process by export_amplicol.

Symbolica, which pyamplicol imports, holds a per-user lock until the process
exits, so it is never imported by MadGraph itself: sessions do not block each
other while they are open. Usage: python amplicol_worker.py JOB.json
"""

import json
import logging
import sys


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


def main(job_path):
    import os
    logging.basicConfig(level=logging.WARNING, format='%(name)s: %(message)s')
    with open(job_path) as stream:
        job = json.load(stream)
    import pyamplicol
    from pyamplicol.artifacts import umami_reweight
    from pyamplicol.licensing import detect_symbolica_license
    if not detect_symbolica_license(suggest=False).licensed:
        logging.warning('Symbolica does not accept the license key of pyamplicol '
                        '(%s): generating in its restricted mode', pyamplicol.__file__)
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
    with open(job['result'], 'w') as stream:
        json.dump(providers, stream)


if __name__ == '__main__':
    main(sys.argv[1])
