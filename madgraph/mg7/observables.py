"""Observables, cuts and histograms of the run card, and PDG id helpers."""

from dataclasses import dataclass

from madgraph.mg7.common import ms

@dataclass
class CutItem:
    observable_kwargs: dict
    min: float
    max: float
    mode: str


@dataclass
class HistItem:
    observable_kwargs: dict
    min: float
    max: float
    bin_count: int
    # the distribution of the event weight itself (the reserved "weight" key
    # of [histograms]); it is read off the event record, not computed from
    # the momenta, so it has no observable
    name: str = ""
    from_weight: bool = False


# Flavor-merged legs carry a group id instead of a pdg. Every member of a group
# shares the same mass -- that is what makes them mergeable -- so any member is
# a valid representative for the kinematics.
_MERGED_PID_REPRESENTATIVE = {
    81: 1,   # light quarks   d u s c
    82: 11,  # charged leptons e mu
    83: 12,  # neutrinos      ve vm vt
}


def clean_pids(pids: list[int]) -> list[int]:
    pids_out = []
    for pid in pids:
        pid = abs(pid)
        if pid in _MERGED_PID_REPRESENTATIVE:
            pid = _MERGED_PID_REPRESENTATIVE[pid]
        elif 81 <= pid <= 99:
            # Reserved for flavor merging. Anything outside this window is a
            # real pdg (BSM models use codes in the millions), so let it pass.
            raise ValueError(
                f"unknown flavor-merged particle id {pid}; add its "
                "representative to _MERGED_PID_REPRESENTATIVE"
            )
        pids_out.append(pid)
    return pids_out


# observables that tell a particle from its antiparticle, and so need the signed
# pdg ids of the actual flavours rather than clean_pids' unsigned representatives
_SIGNED_OBSERVABLES = {"sfos_pair_mass"}


def _flavor_options(meta: dict, unmerged_meta: dict | None = None) -> list[list[int]]:
    """Every flavour assignment (signed pdg ids, incoming then outgoing) a
    subprocess evaluates with the same momenta."""
    if "flavors" in meta:
        return [
            option for flavor in meta["flavors"] for option in flavor["options"]
        ]
    return [
        option
        for subproc in meta.get("subprocesses", [])
        for option in _flavor_options(unmerged_meta[subproc])
    ]


def observable_pids(
    meta: dict, observable_kwargs: dict, unmerged_meta: dict | None = None
) -> list[int]:
    """The pdg ids an observable of this subprocess is built with.

    Most observables only select particles by group, which the unsigned
    representatives of clean_pids do. One that pairs a particle with its
    antiparticle needs the real signed flavours: those of the first flavour
    assignment, provided every assignment of the subprocess forms the same
    pairs -- the cut is applied once to momenta that all of them share.
    """
    all_pids = clean_pids(meta["incoming"]) + clean_pids(meta["outgoing"])
    if observable_kwargs.get("observable") not in _SIGNED_OBSERVABLES:
        return all_pids
    options = _flavor_options(meta, unmerged_meta)
    if not options:
        raise ValueError(
            f"{observable_kwargs.get('name')}: no flavour information for "
            f"subprocess {meta['incoming']} > {meta['outgoing']}"
        )
    n_in = len(meta["incoming"])
    groups = [set(group) for group in observable_kwargs["select_pids"]]
    if len(groups) == 1:
        groups = groups * 2

    def sfos_pairs(pids):
        out = range(n_in, len(pids))
        return frozenset(
            frozenset((i, j))
            for i in out if pids[i] in groups[0]
            for j in out if pids[j] in groups[1]
            if i != j and pids[i] != 0 and pids[i] == -pids[j]
        )

    reference = sfos_pairs(options[0])
    if any(sfos_pairs(option) != reference for option in options[1:]):
        raise ValueError(
            f"{observable_kwargs.get('name')}: the flavours grouped into subprocess "
            f"{meta['incoming']} > {meta['outgoing']} pair the particles "
            "differently, so a same-flavour opposite-sign cut cannot be applied "
            "to their common momenta; regenerate the process after "
            "'set apply_flavor_grouping False'"
        )
    return list(options[0])


def build_observable(pids: list[int], n_in: int, observable_kwargs: dict):
    """ms.Observable of a subprocess with `n_in` incoming particles.

    madspace takes the first two momenta to be the beams (ignore_incoming), so
    for a 1 -> N decay it would drop the first decay product from every
    selection. There the selection runs over all the momenta instead, with the
    incoming particle set to pdg id 0, which is in no group.
    """
    if n_in == 2:
        return ms.Observable(pids, **observable_kwargs)
    pids = [0] * n_in + list(pids[n_in:])
    return ms.Observable(pids, **dict(observable_kwargs, ignore_incoming=False))


def decay_products(meta: dict, unmerged_meta: dict | None = None) -> set[int]:
    """Outgoing positions (0-based, beams excluded) of the particles that come
    from an on-shell (decay-chain) propagator, as MadEvent's check_decay marks
    them. Every channel of a subprocess shares its decay chains, so the first
    channel tells, read through its first diagram's permutation."""
    if not meta["channels"]:
        return set()
    channel = meta["channels"][0]
    topo_channel = channel
    if "vertices" not in channel:
        topo_channel = unmerged_meta[channel["subprocess"]]["channels"][channel["channel"]]
    on_shell = topo_channel.get("on_shell_propagators", [])
    if not on_shell:
        return set()
    vertices = topo_channel["vertices"]
    # the closing vertex creates no propagator
    created = {vertex[-1]: vertex[:-1] for vertex in vertices[:-1]}

    def legs(name):
        if name in created:
            return set().union(*(legs(child) for child in created[name]))
        return {name}

    n_in = len(meta["incoming"])
    n_out = len(meta["outgoing"])
    products = set()
    for prop in on_shell:
        external = legs(f"p{prop}")
        incoming = {leg for leg in external if leg.startswith("i")}
        outgoing = {int(leg[1:]) for leg in external if leg.startswith("o")}
        if not incoming:
            products |= outgoing
        elif len(incoming) == n_in:
            # built from the beams: the decay is everything else
            products |= set(range(n_out)) - outgoing
    # event[i] = topology[permutation[i]]
    permutation = channel["diagrams"][0]["permutation"]
    return {
        i - n_in for i in range(n_in, len(permutation))
        if permutation[i] - n_in in products
    }


def pid_is_qcd(pid: int):
    return abs(pid) in [21, 1, 2, 3, 4, 5, 6, 81]
