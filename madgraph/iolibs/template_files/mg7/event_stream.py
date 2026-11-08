"""In-memory event generation from a gridpack, as a Python API.

Only available in gridpacks made with ``[gridpack] run_mode = "fix_max_weight"``:
their channels carry a fixed maximum weight, so every event is final as soon as
it is unweighted and the events can be streamed without a survey or a combine
pass. Nothing is written to disk.

Usage::

    import sys
    sys.path.insert(0, "<gridpack>/bin")
    from event_stream import EventStream

    with EventStream(device="cpu", seed=42) as stream:
        for event in stream:            # ms.LHEEvent objects, endless
            ...
        for batch in stream.batches(10000, output_type="torch"):
            batch.px                    # shape (10000, max_particle_count)

Event weights are in units of the maximum weight: +-1, except for overweight
events and for events reweighted in a later step (the default [amplicol]
mode = "reweight"), which carry the ratio of the matrix elements. To normalize
a sample to the cross section, scale its weights by
``stream.integral / sum(weights)``. The event sequence is reproducible from the
seed, independent of the thread count and of how it is split into batches.
"""

import collections
import dataclasses
import json
import sys
from typing import Any, Iterator, Optional

import numpy as np

from gridpack_setup import (
    SOURCE_HASH_MESSAGE,
    build_lhe_meta,
    configure_mlm_shower_start,
    gridpack_path,
    load_channels,
    load_context_data,
    load_lhe_completer,
    load_madspace_data,
    load_run_card,
    load_weighting_groups,
    make_context,
    resolve_seed,
    source_hash_matches,
)
import madspace as ms

__all__ = ["EventStream", "EventBatch"]

# events completed per call into madspace while iterating over single events
_ITER_CHUNK = 1000


@dataclasses.dataclass
class EventBatch:
    """A batch of LHE events as a structure of arrays: event fields have shape
    (batch_size,), particle fields (batch_size, max_particle_count). Particle
    slots beyond ``particle_count`` are padded with zeros, see
    ``particle_mask``. numpy arrays or CPU torch tensors, depending on the
    ``output_type`` of :meth:`EventStream.batches`."""

    #: Process ID of the event (IDPRUP).
    process_id: Any
    #: Event weight (XWGTUP), +-1 or larger in magnitude for overweight events, times
    #: the matrix-element ratio of a reweighting step. Not normalized.
    weight: Any
    #: Factorization scale in GeV, the larger of the two beams' (SCALUP); the
    #: renormalization scale for an event without PDFs.
    scale: Any
    #: QED coupling (AQEDUP).
    alpha_qed: Any
    #: QCD coupling (AQCDUP).
    alpha_qcd: Any
    #: Number of particles of the event, including resonances (NUP).
    particle_count: Any
    #: True for the particle slots below ``particle_count``.
    particle_mask: Any
    #: PDG ID of the particle (IDUP).
    pdg_id: Any
    #: -1 for incoming, 1 for outgoing particles, 2 for resonances (ISTUP).
    status_code: Any
    #: Index of the first mother, counting from 1; 0 if there is none (MOTHUP).
    mother1: Any
    #: Index of the second mother, counting from 1; 0 if there is none (MOTHUP).
    mother2: Any
    #: Color line, 0 for a color singlet (ICOLUP).
    color: Any
    #: Anti-color line (ICOLUP).
    anti_color: Any
    #: x component of the momentum in GeV.
    px: Any
    #: y component of the momentum in GeV.
    py: Any
    #: z component of the momentum in GeV.
    pz: Any
    #: Energy in GeV.
    energy: Any
    #: Mass in GeV.
    mass: Any
    #: Proper lifetime c*tau in mm, 0 if not tracked (VTIMUP).
    lifetime: Any
    #: Cosine of the spin angle, or 9 if the spin is not specified (SPINUP).
    spin: Any

    def __len__(self) -> int:
        return len(self.weight)


def _events_to_arrays(events: list, template: dict) -> dict:
    """The fields of ms.LHEEvent objects as arrays like those of
    ms.EventStream.next_batch, with the dtypes and padding of ``template``."""
    arrays = {}
    for key, value in template.items():
        if key == "particle_count":
            data = [len(event.particles) for event in events]
        elif value.ndim == 1:
            data = [getattr(event, key) for event in events]
        else:
            data = np.zeros((len(events), value.shape[1]), value.dtype)
            for row, event in zip(data, events):
                row[: len(event.particles)] = [getattr(p, key) for p in event.particles]
        arrays[key] = np.asarray(data, value.dtype)
    return arrays


def _concatenate(first: dict, second: dict) -> dict:
    return {key: np.concatenate([first[key], second[key]]) for key in first}


class EventStream:
    """Endless stream of unweighted events from this gridpack.

    Constructing it starts the generation in background threads, which keep
    a buffer of events for every channel. The channel of every event is picked
    at random according to the current cross-section estimates, which are
    refined as events are generated.

    Arguments left as None take their value from Cards/grid_run_card.toml.

    Args:
        device: device to generate on, like "cpu", "cuda" or "cuda:1".
        seed: run seed; -1 draws a random one (see :attr:`seed`).
        cpu_batch_size: generation batch size on a CPU device.
        gpu_batch_size: generation batch size on a GPU device.
        thread_pool_size: number of worker threads.
        cut_efficiency_threshold: fraction of a batch that has to pass the cuts.
        max_cut_repetitions: how often a batch is repeated to pass the cuts.
        ignore_source_hash: run even if madspace differs from the version the
            gridpack was made with (can lead to errors or incorrect results).
    """

    #: The seed of the stream; a seed of -1 is replaced by the random seed drawn.
    seed: int

    def __init__(
        self,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        cpu_batch_size: Optional[int] = None,
        gpu_batch_size: Optional[int] = None,
        thread_pool_size: Optional[int] = None,
        cut_efficiency_threshold: Optional[float] = None,
        max_cut_repetitions: Optional[int] = None,
        ignore_source_hash: bool = False,
    ):
        self._stream = None
        # events fetched for the iterator but not returned yet
        self._pending = collections.deque()
        run_card = load_run_card()
        run_args = run_card["run"]
        gen_args = run_card["generation"]
        madspace_data = load_madspace_data()
        if not source_hash_matches(madspace_data):
            if not ignore_source_hash:
                raise RuntimeError(
                    f"{SOURCE_HASH_MESSAGE}. Pass ignore_source_hash=True to run anyway."
                )
            print(f"WARNING: {SOURCE_HASH_MESSAGE}", file=sys.stderr)

        with open(gridpack_path("data", "channels.json")) as f:
            channels = json.load(f)
        if not any("max_weight" in channel for channel in channels.values()):
            raise RuntimeError(
                "EventStream needs a gridpack made with [gridpack] run_mode = "
                '"fix_max_weight"'
            )

        def pick(value, default):
            return default if value is None else value

        if device is None:
            devices = run_args["device"]
            device = devices[0] if isinstance(devices, list) else devices
        if thread_pool_size is None:
            is_gpu = device.split(":")[0] in ("cuda", "hip")
            thread_pool_size = run_args[
                "gpu_thread_pool_size" if is_gpu else "cpu_thread_pool_size"
            ]
        self._context, backend = make_context(
            device, run_args["cpu_mode"], thread_pool_size, thread_pool_size
        )

        config = ms.GeneratorConfig()
        config.cpu_batch_size = pick(cpu_batch_size, gen_args["cpu_batch_size"])
        config.gpu_batch_size = pick(gpu_batch_size, gen_args["gpu_batch_size"])
        config.cut_efficiency_threshold = pick(
            cut_efficiency_threshold, gen_args["cut_efficiency_threshold"]
        )
        config.max_cut_repetitions = pick(
            max_cut_repetitions, gen_args["max_cut_repetitions"]
        )
        configure_mlm_shower_start(config, madspace_data)

        self.seed = resolve_seed(pick(seed, run_args.get("seed", -1)))
        load_context_data(
            [self._context], [backend], madspace_data, run_args["precision"]
        )
        channel_generators = load_channels([self._context], config)
        # the maxima of all passes are fixed, so their settings do not matter
        groups = load_weighting_groups(
            gen_args["max_overweight_truncation"], gen_args["freeze_max_weight_after"],
            gen_args["max_overweight_truncation"], gen_args["freeze_max_weight_after"],
        )
        self._stream = ms.EventStream(
            self._context, channel_generators, self.seed, load_lhe_completer(), config,
            groups=groups,
        )

    def __iter__(self) -> Iterator["ms.LHEEvent"]:
        """Endless iterator over the events as ms.LHEEvent objects. It takes
        events from the stream in chunks of 1000. The events of a chunk that are
        not iterated over are kept: the next calls return them first, so no
        event is lost when the loop is left early."""
        while True:
            if not self._pending:
                self._pending.extend(self._stream.next_events(_ITER_CHUNK))
            yield self._pending.popleft()

    def _take_pending(self, count: int) -> list:
        return [self._pending.popleft() for _ in range(min(count, len(self._pending)))]

    def next_events(self, count: int) -> list:
        """The next ``count`` events as ms.LHEEvent objects."""
        events = self._take_pending(count)
        if len(events) < count:
            events.extend(self._stream.next_events(count - len(events)))
        return events

    def next_batch(self, batch_size: int, output_type: str = "numpy") -> EventBatch:
        """The next ``batch_size`` events as an :class:`EventBatch`."""
        if output_type not in ("numpy", "torch"):
            raise ValueError('output_type must be "numpy" or "torch"')
        pending = self._take_pending(batch_size)
        arrays = self._stream.next_batch(batch_size - len(pending))
        if pending:
            arrays = _concatenate(_events_to_arrays(pending, arrays), arrays)
        counts = arrays["particle_count"]
        max_count = arrays["pdg_id"].shape[1]
        arrays["particle_mask"] = np.arange(max_count)[None, :] < counts[:, None]
        if output_type == "torch":
            import torch
            arrays = {key: torch.from_numpy(value) for key, value in arrays.items()}
        return EventBatch(**arrays)

    def batches(
        self,
        batch_size: int,
        output_type: str = "numpy",
        max_events: Optional[int] = None,
    ) -> Iterator[EventBatch]:
        """Iterator over batches of ``batch_size`` events, endless unless
        ``max_events`` is given (the last batch may then be smaller)."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        remaining = max_events
        while remaining is None or remaining > 0:
            size = batch_size if remaining is None else min(batch_size, remaining)
            yield self.next_batch(size, output_type)
            if remaining is not None:
                remaining -= size

    @property
    def integral(self) -> float:
        """Current estimate of the cross section, in pb."""
        return self._stream.status().mean

    @property
    def integral_error(self) -> float:
        """Standard error of :attr:`integral`."""
        return self._stream.status().error

    @property
    def abs_integral(self) -> float:
        """Current estimate of the integral of the absolute weights."""
        return self._stream.status().mean_abs

    @property
    def n_weighted(self) -> int:
        """Number of weighted events the estimates are based on."""
        return self._stream.status().count

    @property
    def n_weighted_after_cuts(self) -> int:
        """Number of those weighted events that passed the cuts."""
        return self._stream.status().count_after_cuts

    @property
    def n_events(self) -> int:
        """Number of events returned so far."""
        return self._stream.event_count()

    @property
    def max_particle_count(self) -> int:
        """Largest number of particles per event, including resonances."""
        return self._stream.max_particle_count()

    def status(self):
        """Combined ms.GeneratorStatus over all channels."""
        return self._stream.status()

    def channel_status(self) -> list:
        """ms.GeneratorStatus of every channel; ``count_unweighted`` is the
        number of its events returned so far."""
        return self._stream.channel_status()

    def channel_probabilities(self) -> list:
        """Current probability of every channel to be picked for an event."""
        return self._stream.channel_probabilities()

    def lhe_meta(self):
        """ms.LHEMeta for writing the events to an LHE file, with the current
        cross section estimate."""
        return build_lhe_meta(self._stream.status(), self.seed)

    def close(self) -> None:
        """Stop the background generation."""
        if self._stream is not None:
            self._stream.close()

    def __enter__(self) -> "EventStream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
