#!/usr/bin/env python3
"""Checks MadtRex's per-event helicity map against a matrix element library.

Independent of MadtRex: talks to the MadMatrix libraries of an MG7 process
directory directly through the UMAMI C ABI (ctypes). For each LHE event it
does what MadtRex does to build the helicity index -- match the event to a
flavor channel positionally, put a beam-swapped event of a mirrored channel
back into the library's leg order, look its SPINUP values up in the
subprocesses.json helicity table -- and then checks, per subprocess, that

- the per-helicity |M_h|^2 (UMAMI_IN_HELICITY_INDEX) of all helicities sum to
  the helicity-summed |M|^2;
- no event's own helicity has |M_h|^2 = 0;
- the events' helicities are distributed as sum over events of |M_h|^2/|M|^2
  (chi2), as they are when MG7 drew them -- a wrong map fails this.

With --set NAME VALUE --ratios FILE it also writes, per event (in file
order), the expected reweighting ratios after that parameter change, with
the event's own helicity and helicity-summed, to compare with MadtRex's
weights.

Usage: check_helicity_map.py PROC_DIR EVENTS.lhe[.gz] [--n N]
"""
import argparse, ctypes, glob, gzip, json, math, os, re, sys

p = argparse.ArgumentParser()
p.add_argument("proc"); p.add_argument("events"); p.add_argument("--n", type=int, default=400)
p.add_argument("--set", nargs=2, metavar=("NAME", "VALUE")); p.add_argument("--ratios")
p.add_argument("--no-swap", action="store_true", help="emulate the old behaviour: momenta not swapped back")
a = p.parse_args()

IN_MOMENTA, IN_ALPHA_S, IN_FLAVOR, IN_HEL = 0, 1, 2, 6
OUT_ME = 0
subs = json.load(open(os.path.join(a.proc, "SubProcesses", "subprocesses.json")))

def open_lhe(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)
events = []
for ev in re.findall(r"<event>(.*?)<", open_lhe(a.events).read(), re.S):
    lines = [l.split() for l in ev.strip().splitlines()]
    head, parts = lines[0], [l for l in lines[1:] if len(l) == 13]
    inc = [q for q in parts if q[1] == "-1"]; out = [q for q in parts if q[1] == "1"]
    legs = inc + out
    events.append(dict(
        pdgs=[int(q[0]) for q in legs],
        mom=[(float(q[9]), float(q[6]), float(q[7]), float(q[8])) for q in legs],  # E px py pz
        spin=[float(q[12]) for q in legs], alphas=float(head[5])))
    if len(events) >= a.n: break

libs = {}
def library(sub):
    if sub["me_path"] in libs: return libs[sub["me_path"]]
    pattern = os.path.join(a.proc, sub["me_path"].replace("{device}", "*").replace("{fptype}", "*"))
    path = sorted(glob.glob(pattern))[0]
    for common in glob.glob(os.path.join(os.path.dirname(path), "libmadmatrix_common*")):
        ctypes.CDLL(common, mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    h = ctypes.c_void_p()
    assert lib.umami_initialize(ctypes.byref(h), os.path.join(a.proc, "Cards", "param_card.dat").encode()) == 0
    libs[sub["me_path"]] = (lib, h, path)
    return libs[sub["me_path"]]

def me(lib, h, evs, flav, hel=None):
    n = len(evs); npar = len(evs[0]["mom"])
    mom = (ctypes.c_double * (n * npar * 4))()
    for i, e in enumerate(evs):
        for ip, pv in enumerate(e["mom"]):
            for mu in range(4): mom[n * (npar * mu + ip) + i] = pv[mu]
    als = (ctypes.c_double * n)(*[e["alphas"] for e in evs])
    fl = (ctypes.c_uint * n)(*flav)
    keys = [IN_MOMENTA, IN_ALPHA_S, IN_FLAVOR]; ins = [ctypes.cast(mom, ctypes.c_void_p), ctypes.cast(als, ctypes.c_void_p), ctypes.cast(fl, ctypes.c_void_p)]
    if hel is not None:
        hl = (ctypes.c_int * n)(*hel); keys.append(IN_HEL); ins.append(ctypes.cast(hl, ctypes.c_void_p))
    out = (ctypes.c_double * n)()
    k = (ctypes.c_int * len(keys))(*keys); i_ = (ctypes.c_void_p * len(ins))(*ins)
    ok = (ctypes.c_int * 1)(OUT_ME); o_ = (ctypes.c_void_p * 1)(ctypes.cast(out, ctypes.c_void_p))
    st = lib.umami_matrix_element(h, ctypes.c_size_t(n), ctypes.c_size_t(n), ctypes.c_size_t(0), ctypes.c_size_t(len(keys)), k, i_, ctypes.c_size_t(1), ok, o_)
    assert st == 0, st
    return list(out)

# match events to subprocesses / flavor channels, positionally as MadtRex does
groups = {}
for idx, e in enumerate(events):
    for si, sub in enumerate(subs):
        for f in sub["flavors"]:
            for opt in f["options"]:
                swapped = [opt[1], opt[0]] + opt[2:]
                if e["pdgs"] == opt: m = False
                elif f.get("mirror") and e["pdgs"] == swapped: m = True
                else: continue
                ee = dict(e)
                if m and not a.no_swap:
                    ee["mom"] = [e["mom"][1], e["mom"][0]] + e["mom"][2:]
                spins = [e["spin"][1], e["spin"][0]] + e["spin"][2:] if m else list(e["spin"])
                ee["hel"] = sub["helicities"].index([int(round(s)) for s in spins])
                ee["mirrored"] = m; ee["flav"] = f["index"]; ee["idx"] = idx
                groups.setdefault(si, []).append(ee); break
            else: continue
            break
        else: continue
        break

worst = 0.0; zero = 0; ratios = {}
for si, evs in groups.items():
    sub = subs[si]; lib, h, path = library(sub); ncomb = len(sub["helicities"])
    flav = [e["flav"] for e in evs]
    summed = me(lib, h, evs, flav)
    per = [me(lib, h, evs, flav, [ih] * len(evs)) for ih in range(ncomb)]
    own = me(lib, h, evs, flav, [e["hel"] for e in evs])
    for i, e in enumerate(evs):
        s = sum(per[ih][i] for ih in range(ncomb))
        worst = max(worst, abs(s - summed[i]) / summed[i])
        if own[i] <= 0: zero += 1
    expected = [sum(per[ih][i] / summed[i] for i in range(len(evs))) for ih in range(ncomb)]
    observed = [sum(1 for e in evs if e["hel"] == ih) for ih in range(ncomb)]
    chi2 = sum((o - x) ** 2 / x for o, x in zip(observed, expected) if x > 0.5)
    ndf = sum(1 for x in expected if x > 0.5) - 1
    print(f"  helicity frequencies: chi2/ndf = {chi2:.1f}/{ndf}   (observed vs sum of |M_h|^2/|M|^2 over events)")
    print("  observed:", observed); print("  expected:", [round(x, 1) for x in expected])
    if any(e["mirrored"] for e in evs):
        sw = [e for e in evs if e["mirrored"]]
        unswapped = [dict(e, mom=[e["mom"][1], e["mom"][0]] + e["mom"][2:]) for e in sw]
        s1 = me(lib, h, sw, [e["flav"] for e in sw]); s0 = me(lib, h, unswapped, [e["flav"] for e in sw])
        print(f"  beam-swapped events: max rel. diff of summed |M|^2 with/without the swap back = {max(abs(x - y) / x for x, y in zip(s1, s0)):.2e}")
    if a.set:
        lib.umami_set_parameter.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
        assert lib.umami_set_parameter(h, a.set[0].encode(), float(a.set[1]), 0.0) == 0
        own2 = me(lib, h, evs, flav, [e["hel"] for e in evs]); summed2 = me(lib, h, evs, flav)
        for i, e in enumerate(evs):
            ratios[e["idx"]] = (own2[i] / own[i], summed2[i] / summed[i])
    print(f"{os.path.basename(path)}: {len(evs)} events ({sum(e['mirrored'] for e in evs)} beam-swapped), {ncomb} helicities")

print(f"max |sum_h |M_h|^2 - summed| / summed = {worst:.2e}")
print(f"events whose own (mapped) helicity has |M_h|^2 = 0: {zero}")
if a.ratios:
    json.dump(ratios, open(a.ratios, "w"))
