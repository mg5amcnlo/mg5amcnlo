# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Integrated with the MadGraph7 project in Feb 2026.

"""Spin-density matrices and |M|^2 of a madmatrix standalone output, one
phase-space point (or a small batch) at a time, from python.

The kernels are in the libraries of the output (backend/common/Density.h): one
shared library per subprocess, loaded with ctypes. This module only finds the
library and the flavor index of a process from the pdg codes of its legs, and
moves the arrays in and out. SubProcesses/madmatrix_processes.json, written by
the standalone exporter, is the table it reads for that.

MadDensityLibrary keeps the method names of the Fortran standalone f2py module
(initialise, py_get_density, smatrixhel, get_pdg_order, ...) so that MadSpin
drives the two the same way, with the same conventions: momenta in the leg order
of the process, the open legs as 1-based positions, the density divided by the
spin/color average and identical-particle factor of the process.

Build the libraries first, with build_command() in SubProcesses: the scalar
backend is the right one for a few points per call. Two outputs loaded into the
same process must be built with different LIBTAGs (the loader identifies a
library by its name) and opened with the same libtag here.
"""

import array
import ctypes
import itertools
import json
import os

try:
    import numpy as np
except ImportError:
    # MadSpin imports this module whatever its spinmode, and the test runner
    # imports every test module: only evaluating a density needs numpy
    np = None

PROCESSES_FILE = 'madmatrix_processes.json'

# The FPTYPE of the libraries this module builds and loads: the default of
# madmatrix.mk, made explicit since it is part of the library names
FPTYPE = 'm'


class MadDensityError(Exception):
    pass


def is_output(path):
    """Whether path is a madmatrix standalone output this module can load."""
    return os.path.exists(os.path.join(path, 'SubProcesses', PROCESSES_FILE))


def scale_dependence(output_dir):
    """{subprocess: what depends on MU_R besides alpha_s} of the output: the
    couplings, masses and widths the Fortran output re-evaluates at the event
    scale (UPDATE_AS_PARAM2) and these libraries keep at the param card's MU_R.
    Empty when alpha_s is all the scale enters through, i.e. when the alpha_s
    passed with each point carries the whole scale dependence."""
    with open(os.path.join(output_dir, 'SubProcesses', PROCESSES_FILE)) as fsock:
        entries = json.load(fsock)['subprocesses']
    return dict((entry['path'], entry['scale_dependence']) for entry in entries
                if entry.get('scale_dependence'))


def make_variables(backend='scalar', libtag=''):
    """The make variables building what this module loads (the libraries, not
    the standalone check_sa.exe) when make runs in SubProcesses. DENSITY=1
    adds Density.o, which only this module calls: other builds of the same
    output (check_sa, mg7) leave it out."""
    variables = ['PGOAL=lib', 'BACKEND=%s' % backend, 'FPTYPE=%s' % FPTYPE, 'DENSITY=1']
    if libtag:
        variables.append('LIBTAG=%s' % libtag)
    return variables


def build_command(nb_core=1, backend='scalar', libtag=''):
    """The make command for make_variables, to run in SubProcesses."""
    return ['make', '-j%d' % max(1, int(nb_core))] + make_variables(backend, libtag)


def common_library(output_dir):
    """The name of the common library of output_dir, with '{libtag}', '{backend}'
    and '{fptype}' left open, as its process table gives it (the name without the
    model and FPTYPE for an output whose table predates them)."""
    try:
        with open(os.path.join(output_dir, 'SubProcesses', PROCESSES_FILE)) as fsock:
            data = json.load(fsock)
    except (IOError, OSError, ValueError):
        data = {}
    return data.get('common_library', 'libmadmatrix_common{libtag}_{backend}.so')


def common_library_path(output_dir, libtag, backend='scalar'):
    """The path of the common library of output_dir built with libtag."""
    return os.path.join(output_dir, 'lib', common_library(output_dir).format(
        libtag=libtag, backend=backend, fptype=FPTYPE))


def find_libtag(output_dir, backend='scalar'):
    """The LIBTAG the libraries of output_dir were built with, read off the
    name of its common library."""
    lib = os.path.join(output_dir, 'lib')
    prefix, suffix = common_library(output_dir).format(
        libtag='{libtag}', backend=backend, fptype=FPTYPE).split('{libtag}')
    found = sorted(name[len(prefix):-len(suffix)] for name in
                   (os.listdir(lib) if os.path.isdir(lib) else [])
                   if name.startswith(prefix) and name.endswith(suffix))
    if len(found) != 1:
        raise MadDensityError('%s: %s for the %s backend, give the libtag explicitly'
                              % (lib, 'no library' if not found else
                                 'several builds (%s)' % ', '.join(repr(f) for f in found),
                                 backend))
    return found[0]


# basename -> path of every library loaded so far: the loader would hand back
# the first one for a second library of the same name (see LIBTAG in madmatrix.mk)
_loaded = {}


def _load(path):
    name, real = os.path.basename(path), os.path.realpath(path)
    if _loaded.get(name, real) != real:
        raise MadDensityError('%s cannot be loaded: %s, a library of the same name, already '
                              'is, and the loader would reuse it. Build the libraries of '
                              'this output with another LIBTAG.' % (path, _loaded[name]))
    if not os.path.exists(path):
        raise MadDensityError('missing %s: build the libraries first' % path)
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
    # recorded once loaded: a library that failed to load does not hold its name
    _loaded[name] = real
    return lib


def _address(array_):
    return array_.ctypes.data


class _Subprocess(object):
    """One library: one subprocess of the output."""

    _info_names = ('npar', 'npari', 'ncomb', 'nmaxflavor', 'ncolor', 'iden')

    def __init__(self, lib_path, entry, index):
        self.index = index
        self.path = entry['path']
        self.legs = tuple(entry['legs'])
        self.ninitial = entry['ninitial']
        # RTLD_LOCAL: every library exports the same symbol names
        lib = _load(lib_path)
        ptr, size_t, cint = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        lib.umami_initialize.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        lib.umami_initialize.restype = cint
        lib.umami_set_parameter.argtypes = [ptr, ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
        lib.umami_set_parameter.restype = cint
        # all the arrays are passed as plain addresses: much cheaper per call
        # than ctypes pointer objects
        lib.madmatrix_density.argtypes = [size_t, ptr, ptr, ptr, cint, ptr, cint, ptr, ptr]
        lib.madmatrix_density.restype = cint
        # absent from the libraries built before it was introduced
        self.c_density_boosted = getattr(lib, 'madmatrix_density_boosted', None)
        if self.c_density_boosted is not None:
            self.c_density_boosted.argtypes = [size_t, ptr, ptr, ptr, ptr, cint, ptr, cint, ptr, ptr]
            self.c_density_boosted.restype = cint
        lib.madmatrix_matrix_element.argtypes = [size_t, ptr, ptr, ptr, ptr]
        lib.madmatrix_matrix_element.restype = cint
        lib.madmatrix_set_beampol.argtypes = [ctypes.c_double, ctypes.c_double]
        lib.madmatrix_set_beampol.restype = None
        info = (cint * len(self._info_names))()
        lib.madmatrix_process_info(info)
        for name, value in zip(self._info_names, info):
            setattr(self, name, value)
        self.lib = lib
        self.c_density = lib.madmatrix_density
        self.c_matrix_element = lib.madmatrix_matrix_element
        self.handle = None
        # one-point scratch: alpha_s and the flavor index
        self.alphas = ctypes.c_double()
        self.flavor = ctypes.c_uint()
        self.alphas_address = ctypes.addressof(self.alphas)
        self.flavor_address = ctypes.addressof(self.flavor)

    def initialise(self, param_card):
        handle = ctypes.c_void_p()
        status = self.lib.umami_initialize(ctypes.byref(handle),
                                           os.fsencode(os.path.abspath(param_card)))
        if status != 0:
            raise MadDensityError('%s: umami_initialize(%s) failed (status %s)'
                                  % (self.path, param_card, status))
        self.handle = handle

    def check_initialised(self):
        if self.handle is None:
            raise MadDensityError('%s: initialise() (the param card) has to come first'
                                  % self.path)


class _Helicities(object):
    """The open legs and their helicity combinations, as the C arrays."""

    def __init__(self, pos, allow_hel):
        self.n_changing = len(pos)
        if self.n_changing == 0 or len(allow_hel) % self.n_changing:
            raise MadDensityError('inconsistent open legs %s and helicities %s'
                                  % (list(pos), list(allow_hel)))
        self.n_comb = len(allow_hel) // self.n_changing
        self.ninter = self.n_comb * (self.n_comb + 1) // 2
        self.frozen = tuple(int(p) for p in pos)
        self.pos = np.array(pos, dtype=np.int32)
        self.allow_hel = np.array(allow_hel, dtype=np.int32)
        self.pos_address = _address(self.pos)
        self.allow_hel_address = _address(self.allow_hel)


class MadDensityLibrary(object):
    """All the subprocess libraries of one madmatrix standalone output."""

    def __init__(self, output_dir, backend='scalar', libtag=None):
        """libtag: the LIBTAG the libraries were built with, None to read it
        off the library names."""
        if np is None:
            raise MadDensityError('madmatrix.density needs numpy')
        self.output_dir = output_dir
        self.backend = backend
        table = os.path.join(output_dir, 'SubProcesses', PROCESSES_FILE)
        if not os.path.exists(table):
            raise MadDensityError('%s is not a madmatrix standalone output: no %s'
                                  % (output_dir, table))
        with open(table) as fsock:
            entries = json.load(fsock)['subprocesses']
        # The libraries take alpha_s per point and nothing else of the scale:
        # a process whose couplings depend on MU_R as well would silently be
        # evaluated at the param card's MU_R
        dependent = dict((entry['path'], entry['scale_dependence'])
                         for entry in entries if entry.get('scale_dependence'))
        if dependent:
            raise MadDensityError(
                '%s depends on the renormalisation scale beyond alpha_s (%s), which '
                'madmatrix keeps at its param-card value: evaluate it with the '
                'Fortran standalone output' % (output_dir, '; '.join(
                    '%s: %s' % (path, ', '.join(reasons))
                    for path, reasons in sorted(dependent.items()))))
        if libtag is None:
            libtag = find_libtag(output_dir, backend)
        self.libtag = libtag
        # the common library first, by its path: the process libraries only name
        # it, and the check of _load has to see it
        _load(common_library_path(output_dir, libtag, backend))
        self.subprocesses = []
        # raw pdg combination (leg order of the process) -> (subprocess, flavor index)
        self._by_pdgs = {}
        # (raw pdgs, frozen legs) -> (subprocess, flavor index, leg permutation)
        self._dispatch_cache = {}
        self._helicity_cache = {}
        # merged pdgs of every process, in the order get_pdg_order lists them
        self._process_slots = []
        for entry in entries:
            lib_path = os.path.join(output_dir, 'lib',
                                    entry['library'].format(libtag=libtag, backend=backend,
                                                            fptype=FPTYPE))
            if not os.path.exists(lib_path):
                raise MadDensityError('missing %s: build the libraries first (%s in %s)'
                                      % (lib_path, ' '.join(build_command(backend=backend,
                                                                          libtag=libtag)),
                                         os.path.join(output_dir, 'SubProcesses')))
            sub = _Subprocess(lib_path, entry, len(self.subprocesses))
            if len(entry['flavors']) != sub.nmaxflavor:
                raise MadDensityError('%s: %s flavor combinations in %s but %s in the library'
                                      % (sub.path, len(entry['flavors']), PROCESSES_FILE,
                                         sub.nmaxflavor))
            self.subprocesses.append(sub)
            for iflavor, combinations in enumerate(entry['flavors']):
                for pdgs in combinations:
                    self._by_pdgs.setdefault(tuple(pdgs), (sub, iflavor))
            for pdgs in entry['processes']:
                self._process_slots.append((tuple(pdgs), sub))
        # raw |pdg| -> merged code of its leg: the legs of a subprocess carry the
        # merged codes, its flavor combinations the raw ones
        self._merged = {}
        for pdgs, (sub, _) in self._by_pdgs.items():
            for leg, pdg in zip(sub.legs, pdgs):
                self._merged[abs(pdg)] = abs(leg)
        self._card_alphas = 0.118

    @property
    def supports_boost(self):
        """Whether density() takes a boost: every library has
        madmatrix_density_boosted (those built before it was added do not)."""
        return all(sub.c_density_boosted is not None for sub in self.subprocesses)

    # ------------------------------------------------------------------
    # set-up, once per run
    # ------------------------------------------------------------------
    def initialise(self, param_card):
        """Read the model parameters (every library has its own copy)."""
        for sub in self.subprocesses:
            sub.initialise(param_card)
        # alpha_s = 0 asks for the value of the param card, as in the Fortran
        try:
            import madgraph.models.check_param_card as check_param_card
            self._card_alphas = float(check_param_card.ParamCard(param_card)
                                      ['sminputs'].get(3).value)
        except Exception:
            pass

    def set_bwcutoff(self, value):
        """Window (in widths) of the $-syntax propagators."""
        for sub in self.subprocesses:
            sub.check_initialised()
            sub.lib.umami_set_parameter(sub.handle, b'bwcutoff', float(value), 0.)

    def py_set_beampol(self, pol1, pol2):
        """Beam polarisation, madevent convention (1 = unpolarised, up to 2)."""
        for sub in self.subprocesses:
            sub.lib.madmatrix_set_beampol(float(pol1), float(pol2))

    # ------------------------------------------------------------------
    # the f2py module's process table
    # ------------------------------------------------------------------
    def get_prefix(self):
        return [sub.path.encode() for _, sub in self._process_slots]

    def get_pdg_order(self):
        """(pdgs, process ids) per process, the pdgs zero padded to a common
        length as the Fortran wrapper returns them."""
        width = max(len(pdgs) for pdgs, _ in self._process_slots)
        pdgs = [list(p) + [0] * (width - len(p)) for p, _ in self._process_slots]
        return pdgs, [sub.index + 1 for _, sub in self._process_slots]

    def get_idens(self):
        return [sub.iden for _, sub in self._process_slots]

    # ------------------------------------------------------------------
    # dispatch: pdg codes -> library, flavor index and leg order
    # ------------------------------------------------------------------
    def dispatch(self, pdgs, frozen=()):
        """(subprocess, flavor index, permutation) for the legs pdgs, where
        permutation[j] is the leg whose momentum (and flavor) goes to leg j, or
        None when the legs are in a tabulated order already.

        The table holds one leg ordering per flavor combination, as the Fortran
        FLAV_TABLE does: `u d > z u d` may be tabulated as `d u > z d u` only.
        Exchanging two legs of the same merged pdg inside the initial or inside
        the final state is a renaming of external lines, which leaves |M|^2 and
        the density unchanged, so such an ordering is looked for (as
        FLAVOR_ORDER_REPAIR does), keeping the legs in `frozen` -- the open
        helicities of the density -- in place.
        """
        key = (tuple(pdgs), frozen)
        try:
            return self._dispatch_cache[key]
        except KeyError:
            pass
        raw = tuple(int(p) for p in pdgs)
        found = self._by_pdgs.get(raw)
        if found is not None:
            out = (found[0], found[1], None)
        else:
            out = self._repair(raw, set(f - 1 for f in frozen))
            if out is None:
                raise MadDensityError('no madmatrix library evaluates the process %s'
                                      % (list(raw),))
        self._dispatch_cache[key] = out
        return out

    def _repair(self, pdgs, frozen):
        merged = [self._merged.get(abs(p), abs(p)) for p in pdgs]
        for candidate, (sub, iflavor) in self._by_pdgs.items():
            if len(candidate) != len(pdgs):
                continue
            perm = [None] * len(pdgs)
            taken = set()
            for j, (p, c) in enumerate(zip(pdgs, candidate)):
                if p == c:
                    perm[j] = j
                    taken.add(j)
            ok = True
            for j in range(len(pdgs)):
                if perm[j] is not None:
                    continue
                if j in frozen:
                    ok = False
                    break
                block = range(sub.ninitial) if j < sub.ninitial \
                    else range(sub.ninitial, len(pdgs))
                for i in block:
                    if i not in taken and i not in frozen and pdgs[i] == candidate[j] \
                            and merged[i] == merged[j]:
                        perm[j] = i
                        taken.add(i)
                        break
                else:
                    ok = False
                    break
            if ok:
                return (sub, iflavor, perm)
        return None

    def _helicities(self, pos, allow_hel):
        key = (tuple(pos), tuple(allow_hel))
        try:
            return self._helicity_cache[key]
        except KeyError:
            out = self._helicity_cache[key] = _Helicities(pos, allow_hel)
            return out

    def _alphas(self, alphas):
        return alphas if alphas > 0 else self._card_alphas

    @staticmethod
    def _pack(momenta, perm):
        """One point's momenta, [npar][4], as a contiguous double buffer."""
        if perm is not None:
            momenta = [momenta[i] for i in perm]
        return array.array('d', itertools.chain.from_iterable(momenta))

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def density(self, pdgs, momenta, pos, allow_hel, alphas, boost=None):
        """The density at one point: momenta[npar][4] (E, px, py, pz) in the
        leg order of the process, pos the 1-based open legs, allow_hel the
        n_comb*len(pos) helicities they take (combination major). Returns the
        n_comb*(n_comb+1)/2 entries rho(I,J), I <= J, I outer.

        boost: the lab momentum (E, px, py, pz) of the frame the momenta are
        given in -- a decay generated at rest and its parent -- to have them
        boosted to the lab in the library rather than here."""
        hel = self._helicities(pos, allow_hel)
        sub, iflavor, perm = self.dispatch(pdgs, hel.frozen)
        if sub.handle is None:
            sub.check_initialised()
        buf = self._pack(momenta, perm)
        sub.alphas.value = self._alphas(alphas)
        sub.flavor.value = iflavor
        out = np.empty(hel.ninter, dtype=np.complex128)
        if boost is None:
            status = sub.c_density(1, buf.buffer_info()[0], sub.alphas_address,
                                   sub.flavor_address, hel.n_changing, hel.pos_address,
                                   hel.n_comb, hel.allow_hel_address, _address(out))
        else:
            boost = array.array('d', boost)
            status = self._boosted(sub)(1, buf.buffer_info()[0], boost.buffer_info()[0],
                                        sub.alphas_address, sub.flavor_address,
                                        hel.n_changing, hel.pos_address, hel.n_comb,
                                        hel.allow_hel_address, _address(out))
        if status != 0:
            raise MadDensityError('%s: madmatrix_density refused the request (status %s)'
                                  % (sub.path, status))
        return out

    @staticmethod
    def _boosted(sub):
        if sub.c_density_boosted is None:
            raise MadDensityError('%s predates madmatrix_density_boosted: regenerate the '
                                  'output to pass the momenta with a boost' % sub.path)
        return sub.c_density_boosted

    def density_batch(self, all_pdgs, momenta, pos, allow_hel, alphas):
        """density() for several points sharing the open legs and helicities.
        all_pdgs and alphas are per point, momenta[npoint][npar][4]. Returns a
        complex array [npoint][n_comb*(n_comb+1)/2], in input order."""
        hel = self._helicities(pos, allow_hel)
        momenta = np.asarray(momenta, dtype=np.float64)
        alphas = np.asarray(alphas, dtype=np.float64)
        out = np.empty((len(all_pdgs), hel.ninter), dtype=np.complex128)
        groups = {}
        for k, pdgs in enumerate(all_pdgs):
            sub, iflavor, perm = self.dispatch(pdgs, hel.frozen)
            groups.setdefault((sub.index, None if perm is None else tuple(perm)),
                              []).append((k, iflavor))
        for (isub, perm), points in groups.items():
            sub = self.subprocesses[isub]
            sub.check_initialised()
            index = [k for k, _ in points]
            p = momenta[index]
            if perm is not None:
                p = p[:, list(perm), :]
            p = np.ascontiguousarray(p)
            flavors = np.array([f for _, f in points], dtype=np.uint32)
            a_s = np.ascontiguousarray(np.where(alphas[index] > 0, alphas[index],
                                                self._card_alphas))
            inter = np.empty((len(index), hel.ninter), dtype=np.complex128)
            status = sub.c_density(len(index), _address(p), _address(a_s), _address(flavors),
                                   hel.n_changing, hel.pos_address, hel.n_comb,
                                   hel.allow_hel_address, _address(inter))
            if status != 0:
                raise MadDensityError('%s: madmatrix_density refused the request (status %s)'
                                      % (sub.path, status))
            out[index] = inter
        return out

    def matrix_element(self, pdgs, momenta, alphas):
        """|M|^2 at one point, summed/averaged over helicities and colors as the
        Fortran SMATRIX (identical-particle factor and beam polarisation in)."""
        sub, iflavor, perm = self.dispatch(pdgs)
        if sub.handle is None:
            sub.check_initialised()
        buf = self._pack(momenta, perm)
        sub.alphas.value = self._alphas(alphas)
        sub.flavor.value = iflavor
        me = ctypes.c_double()
        status = sub.c_matrix_element(1, buf.buffer_info()[0], sub.alphas_address,
                                      sub.flavor_address, ctypes.addressof(me))
        if status != 0:
            raise MadDensityError('%s: madmatrix_matrix_element refused the request (status %s)'
                                  % (sub.path, status))
        return me.value

    # ------------------------------------------------------------------
    # the f2py module's evaluation entry points, Fortran momentum layout
    # P(0:3, npar), i.e. [4][npar]. scale2 is accepted and not needed: an
    # output loads only when alpha_s is all it takes of the scale (the
    # scale_dependence check of __init__), and alpha_s comes with each point.
    # ------------------------------------------------------------------
    def py_get_density(self, pdgs, procid, p, pos, allow_hel, alphas, scale2):
        return self.density(pdgs, list(zip(*p)), pos, allow_hel, alphas)

    def py_get_density_batch(self, pdgs, procid, p, pos, allow_hel, alphas, scale2, nbatch, next):
        momenta = np.transpose(np.asarray(p, dtype=np.float64), (2, 1, 0))
        return self.density_batch(np.asarray(pdgs).T, momenta, pos, allow_hel, alphas).T

    def smatrixhel(self, pdgs, procid, p, alphas, scale2, nhel=-1):
        if nhel not in (-1, 0):
            raise MadDensityError('smatrixhel: only the helicity sum (nhel = -1) is available')
        return self.matrix_element(pdgs, list(zip(*p)), alphas)
