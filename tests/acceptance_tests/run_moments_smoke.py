"""Explicit, slower fNLO moments integration check (not run by unit discovery).

Run from any directory:
  python /path/to/mg5/tests/acceptance_tests/run_moments_smoke.py NEW_PROCESS_DIR
Use --validate-only to inspect an existing run without generating/integrating.
Uses bundled reduction libraries; refuses automatic dependency installation.
"""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from madgraph.various.moments import MomentFile


def validate(process):
    result = process / 'Events' / 'moments_adaptive'
    data = MomentFile.read(result / 'MADatNLO.Moments')
    histograms = {h.label: h for h in data.histograms}
    for rate_id, flow_id in ((1, 3), (2, 4)):
        rate, flow = histograms[rate_id], histograms[flow_id]
        rates, rate_error = rate.integral(rate.xmin, rate.xmax)
        integrals, flow_error = flow.integral(flow.xmin, flow.xmax)
        for expected, observed in zip(rates, integrals):
            if not math.isclose(expected, observed, rel_tol=1.e-10, abs_tol=1.e-12):
                raise AssertionError('Energy-flow normalization disagrees with rate')
        if not math.isclose(rate_error, flow_error, rel_tol=1.e-9, abs_tol=1.e-12):
            raise AssertionError('Energy-flow rate uncertainty disagrees')
    for extension in ('HwU', 'Moments', 'gnuplot'):
        if not (result / ('MADatNLO.' + extension)).is_file():
            raise AssertionError('Missing combined %s output' % extension)
    summary = {
        'process': 'e+ e- > u u~ [QCD]', 'energy_GeV': 250,
        'weights': len(data.weights),
        'NLO_pb': histograms[1].means[0][0],
        'NLO_stat_pb': math.sqrt(histograms[1].covariance[0][0]),
        'validation': 'All full-range weight integrals and central rate errors agree',
        'results': str(result),
    }
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('process_directory', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    process = args.process_directory.resolve()
    if args.validate_only:
        validate(process)
        return
    if process.exists():
        parser.error('Choose a new process directory')

    from madgraph.interface.master_interface import MasterCmd
    from madgraph.interface.amcatnlo_run_interface import aMCatNLOCmd
    from madgraph.various.banner import RunCardNLO

    master = MasterCmd()
    master.no_notification()
    master.options.update(automatic_html_opening=False, auto_update=0,
                          ninja=None, collier=None, golem=None)

    def no_install(*args, **kwargs):
        raise RuntimeError('Smoke test refuses automatic dependency installation')

    master.do_install = no_install
    for command in ('import model loop_sm-no_b_mass',
                    'generate e+ e- > u u~ [QCD]',
                    'output %s -nojpeg' % process):
        master.exec_cmd(command, errorhandling=False, precmd=True)
    card = RunCardNLO(str(process / 'Cards' / 'run_card.dat'))
    settings = {
        'lpp1': 0, 'lpp2': 0, 'ebeam1': 125., 'ebeam2': 125.,
        'req_acc_FO': .1, 'reweight_scale': [True], 'reweight_PDF': [False],
        'pdlabel': 'nn23lo1', 'ptj': 0., 'etaj': -1., 'drjj': 0., 'iseed': 12345,
    }
    for key, value in settings.items():
        card[key] = value
    card.write(str(process / 'Cards' / 'run_card.dat'))
    (process / 'Cards' / 'FO_analyse_card.dat').write_text(
        'FO_ANALYSIS_FORMAT = moments\nFO_EXTRALIBS =\nFO_EXTRAPATHS =\n'
        'FO_INCLUDEPATHS =\nFO_ANALYSE = analysis_moments_template.o\n')
    run = aMCatNLOCmd(me_dir=str(process))
    run.no_notification()
    run.options.update(automatic_html_opening=False, run_mode=0, nb_core=1)
    run.exec_cmd('launch NLO -f --name=moments_adaptive', errorhandling=False)
    validate(process)


if __name__ == '__main__':
    main()
