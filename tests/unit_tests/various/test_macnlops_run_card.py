"""The measured MAcNLOPS shower profile survives a rebuild of run_card.inc.

Source/makefile regenerates run_card.inc from Cards/run_card.dat whenever the
card is newer, without the values a launch set in memory.
"""
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

from madgraph.various import banner


ROOT = Path(__file__).resolve().parents[3]
SUPPORT = dict(n_quark_in=5, n_gluon_to_quark=5, shower_cutoffs=[0.5000000000000001, 0.53],
               heavy_mass_floors={'4': 1.2, '5': 4.0},
               beam_xmax=[0.9998556507692308, 0.9998556507692307])


class TestMacNLOPSRunCardProfile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='macnlops_run_card_')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        (self.work / 'Cards').mkdir()
        (self.work / 'Source').mkdir()
        shutil.copyfile(ROOT / 'Template/NLO/Source/run.inc', self.work / 'Source/run.inc')
        card = banner.RunCardNLO()
        for name, value in dict(macnlops=True, mcatnlo_delta=True, parton_shower='PYTHIA8',
                                event_norm='sum', pdlabel='lhapdf', lhaid=[331700],
                                reweight_scale=[False], reweight_pdf=[False]).items():
            card[name] = value
        card.write(str(self.work / 'Cards/run_card.dat'))

    def include(self, card):
        card.write_include_file(str(self.work / 'Source'))
        text = (self.work / 'Source/run_card.inc').read_text()
        return {name.lower(): float(value.replace('D', 'E')) for name, value in re.findall(
            r'(?m)^\s*(MACNLOPS_(?:ISR_CUTOFF|FSR_CUTOFF|CHARM_MASS|BOTTOM_MASS|'
            r'BEAM_XMAX\(\d\)))\s*=\s*(\S+)', text)}

    def test_card_on_disk_restores_the_measured_profile(self):
        (self.work / 'Cards/macnlops_profile.json').write_text(json.dumps(dict(support=SUPPORT)))
        values = self.include(banner.RunCardNLO(str(self.work / 'Cards/run_card.dat')))
        self.assertEqual(values, {'macnlops_isr_cutoff': 0.5000000000000001,
                                  'macnlops_fsr_cutoff': 0.53, 'macnlops_charm_mass': 1.2,
                                  'macnlops_bottom_mass': 4.0,
                                  'macnlops_beam_xmax(1)': 0.9998556507692308,
                                  'macnlops_beam_xmax(2)': 0.9998556507692307})

    def test_values_set_in_memory_or_absent_profiles_are_kept(self):
        card = banner.RunCardNLO(str(self.work / 'Cards/run_card.dat'))
        self.assertEqual(set(self.include(card).values()), {-1.})
        (self.work / 'Cards/macnlops_profile.json').write_text(json.dumps(dict(support=SUPPORT)))
        for name, value in card.macnlops_profile_values(dict(
                SUPPORT, shower_cutoffs=[0.6, 0.7])).items():
            card.set(name, value, user=False)
        values = self.include(card)
        self.assertEqual((values['macnlops_isr_cutoff'], values['macnlops_fsr_cutoff']), (.6, .7))


if __name__ == '__main__':
    unittest.main()
