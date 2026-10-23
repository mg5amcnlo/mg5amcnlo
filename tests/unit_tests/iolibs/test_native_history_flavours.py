"""The global-H annotation requires full native physical-flavour grouping."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from madgraph.iolibs import native_histories


def process(*ids):
    return ['restrictions', [[pid, index > 1, [], False, None]
                             for index, pid in enumerate(ids)]]


class TestNativeHistoryFlavours(unittest.TestCase):
    def setUp(self):
        self.outer = dict(context=1, nexternal=5, sectors=[
            dict(sector=1, processes=[process(1, 21, 6, -6, 1),
                                     process(2, 21, 6, -6, 2)])])
        self.native = dict(context=2, nexternal=5, sectors=[
            dict(sector=3, processes=[process(2, 21, -6, 6, 2),
                                     process(1, 21, -6, 6, 1)])])
        self.table = dict(sector=1, unresolved=[], ambiguous=[], histories=[
            dict(context=1, sector=1, permutation=[1, 2, 3, 4, 5], i=5, j=1),
            dict(context=2, sector=3, permutation=[1, 2, 4, 3, 5], i=5, j=2)])
        self.by_id = {1: self.outer, 2: self.native}

    def complete(self):
        return native_histories.history_flavours_complete(
            self.outer, [self.table], self.by_id)

    def test_native_row_reordering_and_spectator_permutation_are_complete(self):
        self.assertTrue(self.complete())

    def test_partially_overlapping_tables_cannot_certify_global_sign(self):
        self.native['sectors'][0]['processes'].pop()
        self.assertFalse(self.complete())

    def test_extra_native_rows_are_incomplete_even_if_luminosity_can_be_zero(self):
        self.native['sectors'][0]['processes'].append(process(3, 21, -6, 6, 3))
        self.assertFalse(self.complete())

    def test_duplicate_multiplicity_is_not_collapsed_to_a_set(self):
        self.native['sectors'][0]['processes'][1] = copy.deepcopy(
            self.native['sectors'][0]['processes'][0])
        self.assertFalse(self.complete())

    def test_duplicate_owner_rows_still_have_separate_unweighting_choices(self):
        self.outer['sectors'][0]['processes'].append(copy.deepcopy(
            self.outer['sectors'][0]['processes'][0]))
        self.native['sectors'][0]['processes'].append(copy.deepcopy(
            self.native['sectors'][0]['processes'][1]))
        self.assertFalse(self.complete())

    def test_missing_ambiguous_and_invalid_mapping_are_incomplete(self):
        self.table['unresolved'] = [[1, 5, 1]]
        self.assertFalse(self.complete())
        self.table['unresolved'] = []
        self.table['ambiguous'] = ['conflict']
        self.assertFalse(self.complete())
        self.table['ambiguous'] = []
        self.table['histories'][1]['permutation'] = [1, 2, 3, 3, 5]
        self.assertFalse(self.complete())

    def test_generated_constant_changes_with_actual_table_coverage(self):
        aliases = {(1, 1): 1, (2, 3): 2}
        with tempfile.TemporaryDirectory(prefix='native_flavour_guard_') as name:
            path = Path(name)
            native_histories.write_histories(path, self.outer, [self.table], aliases, self.by_id)
            text = ''.join((path / 'mc_histories.inc').read_text().lower().split())
            self.assertIn('parameter(mc_hist_flavour_complete=.true.)', text)
            self.native['sectors'][0]['processes'].pop()
            native_histories.write_histories(path, self.outer, [self.table], aliases, self.by_id)
            text = ''.join((path / 'mc_histories.inc').read_text().lower().split())
            self.assertIn('parameter(mc_hist_flavour_complete=.false.)', text)

    def test_delta_registry_pins_indices_permutations_and_flavour_tables(self):
        aliases = {(1, 1): 1, (2, 3): 2}
        with tempfile.TemporaryDirectory(prefix='native_registry_') as name:
            path = Path(name)
            def export():
                native_histories.write_histories(path, self.outer, [self.table], aliases, self.by_id)
                record = json.loads((path / 'mc_histories.json').read_text())
                payload = json.dumps(record['registry'], sort_keys=True, separators=(',', ':'))
                self.assertEqual(record['sha256'], hashlib.sha256(payload.encode()).hexdigest())
                # FortranWriter splits long string parameters across lines.
                for chunk in (record['sha256'][:20], record['sha256'][-20:]):
                    self.assertIn(chunk, (path / 'mc_histories.inc').read_text())
                return record['sha256']
            original = export()
            self.assertEqual(export(), original)
            self.table['histories'][1]['permutation'][2:4] = [3, 4]
            changed = export()
            self.assertNotEqual(changed, original)
            aliases[(2, 3)] = 3
            self.assertNotEqual(export(), changed)


if __name__ == '__main__':
    unittest.main()
