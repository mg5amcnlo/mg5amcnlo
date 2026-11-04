"""The Python EventStream of the mg7 gridpacks (bin/event_stream.py), on a fake
madspace stream: no event is lost between the ways of reading the stream."""

import collections
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_TEMPLATE = (Path(__file__).resolve().parents[3] / "madgraph" / "iolibs"
             / "template_files" / "mg7" / "event_stream.py")


def _event_stream_module():
    setup = types.ModuleType("gridpack_setup")
    setup.__getattr__ = lambda name: None
    spec = importlib.util.spec_from_file_location("mg7_event_stream", _TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"madspace": types.ModuleType("madspace"),
                                       "gridpack_setup": setup}):
        spec.loader.exec_module(module)
    return module


Particle = collections.namedtuple("Particle", "pdg_id px")
Event = collections.namedtuple("Event", "weight particles")


class FakeStream:
    """Event i has weight i and i % 3 + 1 particles."""

    def __init__(self):
        self.count = 0

    def _events(self, count):
        events = [
            Event(float(i), [Particle(21, float(i + j)) for j in range(i % 3 + 1)])
            for i in range(self.count, self.count + count)
        ]
        self.count += count
        return events

    def next_events(self, count):
        return self._events(count)

    def close(self):
        pass

    def next_batch(self, count):
        events = self._events(count)
        px = np.zeros((count, 3))
        for row, event in zip(px, events):
            row[: len(event.particles)] = [p.px for p in event.particles]
        return {
            "weight": np.array([e.weight for e in events]),
            "particle_count": np.array([len(e.particles) for e in events], np.int32),
            "pdg_id": np.where(px != 0.0, 21, 0).astype(np.int32),
            "px": px,
        }


class TestEventStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _event_stream_module()
        # the fake stream has only some of the LHE fields
        cls.module.EventBatch = types.SimpleNamespace

    def make_stream(self):
        stream = object.__new__(self.module.EventStream)
        stream._stream = FakeStream()
        stream._pending = collections.deque()
        return stream

    def test_no_event_is_lost_after_leaving_the_loop(self):
        stream = self.make_stream()
        for event in stream:
            if event.weight == 4.0:
                break
        self.assertEqual([e.weight for e in stream.next_events(3)], [5.0, 6.0, 7.0])
        batch = stream.next_batch(4)
        self.assertEqual(list(batch.weight), [8.0, 9.0, 10.0, 11.0])
        self.assertEqual(list(batch.particle_count), [3, 1, 2, 3])
        self.assertEqual(list(batch.px[0]), [8.0, 9.0, 10.0])
        self.assertEqual(list(batch.px[2]), [10.0, 11.0, 0.0])
        self.assertEqual(batch.particle_mask[2].tolist(), [True, True, False])
        self.assertEqual(batch.pdg_id.dtype, np.int32)
        # the rest of the chunk, then the stream itself
        batch = stream.next_batch(1000)
        self.assertEqual(list(batch.weight), [float(i) for i in range(12, 1012)])
        self.assertEqual(next(iter(stream)).weight, 1012.0)

    def test_batches_needs_a_positive_size(self):
        stream = self.make_stream()
        for size in (0, -1):
            with self.assertRaises(ValueError):
                next(stream.batches(size, max_events=10))


if __name__ == "__main__":
    unittest.main()
