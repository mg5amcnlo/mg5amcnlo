"""Replay fixed-mixture choices from saved surveys; no generation-speed claim."""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUT = ROOT / 'validation/ampli_efficiency_review_20261008/native/survey_sampling_diagnostics.json'


def main():
    output = {'note': 'Saved-survey diagnostics only; empirical maxima are not bounds. '
              'Conditional second moments inferred from the last survey iteration; '
              'the rare virtual sample can have substantial statistical uncertainty.',
              'channels': []}
    for entry in json.loads(INPUT.read_text())['channels']:
        lines = (ROOT / entry['source']).read_text().splitlines()
        values = [float(v.replace('D', 'E')) for v in lines[3].split()]
        nvalues = int(lines[1].split()[-1])
        rates, errors = values[:nvalues], values[nvalues:2*nvalues]
        points = int(lines[7].split()[0])
        end = lines.index('END_MG5_SIMPLE_INTEGRATOR')
        survey_virtual_fraction = float(lines[end + 2].split()[0])
        mn, mv = values[-2:]
        an, av = rates[0], rates[4]
        sn = errors[0]**2 * (points - 1) + an**2
        # Survey estimates v/r with Bernoulli virtual selection probability r.
        # Therefore E[(v/r)^2 I(selected)] = E[v^2]/r.
        sv = survey_virtual_fraction * (errors[4]**2 * (points - 1) + av**2)
        p0 = max(0.001, min(0.999, av/(an+av)))
        pmax = mv/(mn+mv)
        p = max(p0, min(pmax, 4*p0, 0.1))
        prms = math.sqrt(sv)/(math.sqrt(sn)+math.sqrt(sv))
        def metrics(probability):
            return {'probability': probability,
                    'survey_initial_envelope': max(mn/(1-probability), mv/probability),
                    'estimated_abs_second_moment': sn/(1-probability)+sv/probability,
                    'estimated_abs_variance': sn/(1-probability)+sv/probability-(an+av)**2}
        base = metrics(p0)
        candidate = metrics(p)
        candidate['initial_envelope_ratio'] = candidate['survey_initial_envelope']/base['survey_initial_envelope']
        candidate['estimated_abs_variance_ratio'] = candidate['estimated_abs_variance']/base['estimated_abs_variance']
        candidate['nonvirtual_probability_penalty'] = (1-p0)/(1-p)
        output['channels'].append({'channel': entry['channel'], 'survey_points': points,
            'survey_virtual_fraction': survey_virtual_fraction,
            'survey_expected_virtual_evaluations': points*survey_virtual_fraction,
            'survey_rates': [an, av], 'survey_conditional_abs_second_moments': [sn, sv],
            'rate_mixture': base, 'capped_maximum_mixture': candidate,
            'uncapped_maximum_mixture': metrics(pmax), 'minimum_abs_variance_mixture': metrics(prms)})
    (HERE / 'survey_choices.json').write_text(json.dumps(output, indent=2) + '\n')
    for entry in output['channels']:
        b, c = entry['rate_mixture'], entry['capped_maximum_mixture']
        print(entry['channel'], f'p {b["probability"]:.6f} -> {c["probability"]:.6f}',
              f'max ratio {c["initial_envelope_ratio"]:.4f}',
              f'variance ratio {c["estimated_abs_variance_ratio"]:.4f}')


if __name__ == '__main__':
    main()
