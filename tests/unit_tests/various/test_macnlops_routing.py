"""Route reuse preserves fresh momenta and exact Delta replay identities."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from madgraph.various import macnlops as mc
from tests.unit_tests.various import test_macnlops_delta_protocol as protocol


class TestNativeRoutingCache(unittest.TestCase):
    @staticmethod
    def external(ids, offset=0):
        return [(pdg, (offset + i, i + 1, i + 2, i + 3))
                for i, pdg in enumerate(ids)]

    def ordinary_evaluator(self, root):
        (root / 'Cards').mkdir()
        (root / 'Cards/run_card.dat').write_text('False = mcatnlo_delta\n')
        directory = root / 'SubProcesses/P1'
        directory.mkdir(parents=True)
        (directory / 'leshouche_info.dat').write_text(
            'I 1 1 2 -2 11 -11 21 21\n'
            'I 2 1 21 2 11 -11 2 21\n')
        with patch.object(mc, 'hard_model_masses', return_value={}):
            return mc.NativeEvaluator(root, root)

    def test_repeated_flavour_route_uses_current_momenta_and_duplicate_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = self.ordinary_evaluator(Path(tmp))
            ids = (2, -2, 21, 11, 21, -11)
            first = self.external(ids)
            directory, owner, routed = evaluator._route(first)
            permutation = (0, 1, 3, 5, 2, 4)
            self.assertEqual(owner, 1)
            self.assertEqual(routed, [first[i] for i in permutation])
            fresh = self.external(ids, offset=100)
            # A cached discrete route must no longer scan flavour Counters.
            with patch.object(mc, 'Counter', side_effect=AssertionError('rescanned route')):
                new_directory, new_owner, routed = evaluator._route(fresh)
            self.assertEqual((new_directory, new_owner), (directory, owner))
            self.assertEqual(routed, [fresh[i] for i in permutation])

    def test_flavours_and_initial_beam_order_are_part_of_the_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = self.ordinary_evaluator(Path(tmp))
            evaluator._route(self.external((2, -2, 11, -11, 21, 21)))
            _, owner, _ = evaluator._route(self.external((21, 2, 11, -11, 2, 21)))
            self.assertEqual(owner, 2)
            with self.assertRaisesRegex(mc.MatchingError, 'No real matrix-element'):
                evaluator._route(self.external((2, 21, 11, -11, 2, 21)))

    def test_delta_replay_cache_keeps_registry_owner_and_ordered_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluator, hashes = protocol.TestDeltaProtocol().evaluator(Path(tmp))
            external = self.external((2, -2, 11, -11))
            packets = [mc.DeltaContext.from_block(protocol.packet(registry=digest))
                       for digest in hashes]
            for index, context in enumerate(packets, 1):
                for offset in (0, 100):
                    point = self.external((2, -2, 11, -11), offset)
                    directory, owner, routed = evaluator._route(point, context)
                    self.assertEqual((directory.name, owner), ('P%d' % index, 1))
                    self.assertEqual(routed, point)
            wrong_owner = mc.DeltaContext.from_block(protocol.packet(registry=hashes[0], owner=7))
            with self.assertRaisesRegex(mc.MatchingError, 'exact Delta replay registry'):
                evaluator._route(external, wrong_owner)
            with self.assertRaisesRegex(mc.MatchingError, 'exact Delta replay registry'):
                evaluator._route(self.external((2, -2, -11, 11)), packets[0])

    def test_cached_s_route_does_not_skip_auxiliary_law_recertification(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluator, _ = protocol.TestDeltaProtocol().evaluator(Path(tmp))
            point = self.external((2, -2, 11, -11))
            evaluator._route(point)
            evaluator.delta_law_certificates.clear()
            history = evaluator.registries[Path(tmp) / 'SubProcesses/P2']['registry']['tables'][0]
            history['histories'].append(dict(context=3, sector=3, permutation=[1, 2, 3, 4]))
            with self.assertRaisesRegex(mc.MatchingError, 'joint auxiliary laws'):
                evaluator._route(point)


if __name__ == '__main__':
    unittest.main()
