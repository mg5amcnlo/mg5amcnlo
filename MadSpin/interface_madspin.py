################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
""" Command interface for MadSpin """
from __future__ import division
from __future__ import absolute_import
import collections
import itertools
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import glob
from itertools import chain, filterfalse, product
pjoin = os.path.join
if '__main__' == __name__:
    import sys
    sys.path.append(pjoin(os.path.dirname(__file__), '..'))

import madgraph.interface.extended_cmd as extended_cmd
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.interface.master_interface as master_interface
import madgraph.interface.madevent_interface as madevent_interface
import madgraph.interface.common_run_interface as common_run_interface
import madgraph.interface.reweight_interface as rwgt_interface
import madgraph.various.misc as misc
import madgraph.iolibs.files as files
import madgraph.iolibs.export_v4 as export_v4
import madgraph.various.banner as banner
import madgraph.various.lhe_parser as lhe_parser

import models.import_ufo as import_ufo
import models.check_param_card as check_param_card
import MadSpin.decay as madspin

logger = logging.getLogger('decay.stdout') # -> stdout
logger_stderr = logging.getLogger('decay.stderr') # ->stderr
cmd_logger = logging.getLogger('cmdprint2') # -> print

# ---------------------------------------------------------------------------
# Polarisation labels accepted by keep_weight_for_polarization_*
# ---------------------------------------------------------------------------
# Same spelling and same meaning as MG5's polarisation braces:
#   {L} -> [-1], {R}/{+} -> [1], {T} -> [-1,1], {0} -> [0]
# Maps the (case-insensitive, brace-tolerant) user spelling onto
# (canonical label, helicity values).
POLARIZATION_ALIASES = {
    '0': ('0', (0,)),
    '+': ('+', (1,)),
    'r': ('+', (1,)),
    '-': ('-', (-1,)),
    'l': ('-', (-1,)),
    't': ('T', (-1, 1)),
}


def parse_polarization_label(label):
    """(canonical label, helicity values) for one
    keep_weight_for_polarization_vector/_fermion entry, or None if it is not
    one of 0/+/-/T (L and R aliasing - and +)."""
    key = str(label).strip().lower()
    if key.startswith('{') and key.endswith('}'):
        key = key[1:-1].strip()
    return POLARIZATION_ALIASES.get(key)


def decay_density_tensor(slot_identity, helicities, slot_densities):
    """The decay side of the sequential contraction: the tensor product of every
    slot's normalised decay density, the slots still to be drawn contributing
    I/n (``slot_identity``).

    Factored out of ``_partial_density_contraction`` so the polarisation weights
    can contract that same tensor against a differently masked production matrix
    without rebuilding it.
    """
    density_dec = None
    for slot, hel in enumerate(helicities):
        density = slot_densities.get(slot)
        if density is None:
            density = slot_identity(hel)
        else:
            density = density.normalized()
        if density_dec is None:
            density_dec = density
        else:
            density_dec = density_dec.tensor_product(density)
    return density_dec



class MadSpinDegenerateWeight(madspin.MadSpinError):
    """The accept/reject cannot ever accept: every trial weight is structurally
    zero (or not a number), so the unweighting loop would redraw -- and keep
    regenerating decay-event pools -- forever. Raised instead of looping.

    This is NOT the same thing as a low acceptance: a slow-but-correct run
    produces small *positive* weights and accepts one eventually. The guards
    that raise this only trigger on weights that are exactly zero / negative /
    NaN, i.e. on a numerator that cannot become positive no matter how many
    decays are drawn."""
    pass


class MadSpinUnknownPartialWidth(madspin.MadSpinError):
    """A reused decay directory whose partial width can neither be read back
    nor re-measured. Raised instead of falling back to any default, because
    every default here is a wrong branching ratio -- i.e. a well-formed event
    file whose every weight and whose <init> block are wrong."""
    pass


class MadSpinStaleParameters(madspin.MadSpinError):
    """A reused ``ms_dir``/``use_old_dir`` directory was built with different
    parameters than the run now asking for it.

    Everything that directory holds is a function of the param_card: the decay
    gridpacks and the events they produce, the partial widths measured while
    building them, the maximum weights of the unweighting, the pickled
    branching ratios. None of it can be recomputed without rebuilding the
    directory, and all of it would be reused in silence -- so a changed
    param_card makes the reuse unsound rather than merely stale."""
    pass


class MadSpinZeroBranchingRatio(madspin.MadSpinError):
    """The branching ratio MadSpin is about to apply to every event is zero (or
    not a number). Raised instead of writing the events.

    A branching ratio multiplies every event weight and the <init> cross
    section, so a zero one turns a completed run into a well-formed LHE file
    full of +/-0.0 -- the failure mode a user is least likely to notice. It
    cannot be a legitimate physics configuration either: MadSpin measures each
    partial width by generating that decay, and a decay channel that is closed
    makes the *generation* fail (ZeroResult) long before this point. So a zero
    that reaches here comes from bookkeeping that did not happen, not from the
    physics that was asked for."""
    pass


# How many consecutive structurally-dead trials (weight not finite and > 0) a
# single production event may burn before the accept/reject gives up. Only
# trials whose *matrix-element* factor is dead are counted, and any single
# strictly positive weight resets the counter, so a genuinely inefficient but
# correct run never reaches this bound however bad its acceptance is. Sized so
# that it is unreachable by chance: with an acceptance as low as 1e-4 the
# probability of this many consecutive zero *weights* (as opposed to rejected
# positive weights, which do not count) is nil.
MS_MAX_DEAD_TRIALS = 20000

# How large the imaginary part of a density contraction may be, relative to its
# real part, before it is reported. The contraction is real *by construction*
# (see ms_density_real), so anything above float32 rounding is a bug and not a
# tolerance to be widened: the density matrices are complex64, the packed index
# set is closed under (h1,h2) -> (h2,h1), and the imaginary parts of the two
# members of each pair are computed by the same operations in the opposite
# order, so what survives the sum is the residue of an exact cancellation. On
# `p p > t t~` with both tops decayed (53655 contractions) the largest ratio
# measured was 1.5e-7, i.e. exactly float32 epsilon; 1e-3 leaves four orders of
# margin over that and still catches an imaginary part that means anything.
MS_DENSITY_IMAG_TOL = 1e-3

# Reported once per site: a violation is a property of the setup (a non-hermitian
# density matrix, a helicity basis mismatched between the two sides of the
# contraction), so it repeats on every trial of every event once it happens.
_MS_IMAG_REPORTED = set()


def ms_density_real(value, what):
    """The real part of a spin-density contraction, with its reality checked.

    Contracting two hermitian density matrices over an index set closed under
    (h1,h2) -> (h2,h1) -- which is what ``DensityMatrix`` stores, and what a
    polarisation restriction preserves, since it masks the bra *and* the ket
    helicity with the same set -- pairs every term with its own complex
    conjugate. The sum is therefore real by construction, and what is discarded
    here is float32 rounding, not physics.

    That argument is only as good as its premises, so it is checked rather than
    assumed: an imaginary part above ``MS_DENSITY_IMAG_TOL`` of the real part
    means one of the two matrices is not hermitian, or the two are not in the
    same helicity basis, and the number this returns is then not the matrix
    element. Reported at CRITICAL (once per site) rather than raised, following
    the weight-identity and ``density_debug`` checks: the run does produce
    events, they are just not to be trusted, and silently dropping the evidence
    is the one thing that must not happen.
    """
    imag = getattr(value, 'imag', None)
    if imag is None:
        return value              # not a number with a real/imaginary split
    real = value.real             # a float/np.float32 keeps its own value here
    if abs(imag) > MS_DENSITY_IMAG_TOL * abs(real) and what not in _MS_IMAG_REPORTED:
        _MS_IMAG_REPORTED.add(what)
        logger.critical(
            "MadSpin: %s came out with a significant imaginary part "
            "(%.6g + %.6gj, |Im|/|Re| = %.3g > %.3g). A contraction of two "
            "hermitian density matrices in a common helicity basis is real by "
            "construction, so this is not a rounding effect: the value used as "
            "the accept/reject weight from here on is its real part only, and "
            "the decayed events are not reliable.",
            what, real, imag, abs(imag) / abs(real) if real else float('inf'),
            MS_DENSITY_IMAG_TOL)
    return real


class MadSpinOptions(banner.ConfigFile):

    # Unweighting schemes that still work but are no longer offered to the
    # user: they are kept out of the 'allowed' list above so that they show up
    # neither in the completion nor in the "allowed values are ..." message,
    # and are re-admitted one call at a time by __setitem__ below. 'two_stage'
    # is here because it is not the fastest scheme at any multiplicity measured
    # (see _unweighting_mode) -- it survives as an internal cross-check, the
    # one staged scheme whose angle stage is a single joint test, and as such
    # is still exercised by the parallel tests and the benchmarks.
    hidden_unweighting_modes = ('two_stage',)

    def default_setup(self):

        self.add_param("max_weight", -1)
        self.add_param('curr_dir', os.path.realpath(os.getcwd()))
        self.add_param('Nevents_for_max_weight', 0)
        self.add_param("max_weight_ps_point", 500)
        self.add_param('BW_cut', -1)
        self.add_param('nb_sigma', 0.)
        self.add_param('ms_dir', '')
        self.add_param('max_running_process', 100)
        self.add_param('onlyhelicity', False)
        self.add_param('spinmode', "madspin", allowed=['full', 'madspin', 'none', 'onshell', 'PA', 'madspin_v1', 'onshell_v1'])
        self.add_param('use_old_dir', False, comment='should be use only for faster debugging')
        self.add_param('run_card', '' , comment='define cut for decay_events (in onshell frame). Path to run_card to use')
        self.add_param('fixed_order', False, comment='to activate fixed order handling of counter-event')
        self.add_param('seed', 0, comment='control the seed of madspin')
        self.add_param('cross_section', {'__type__':0.}, comment="forcing normalization of cross-section after MS (for none/onshell)" )
        self.add_param('new_wgt', 'cross-section' ,allowed=['cross-section', 'BR'], comment="if not consistent number of particles, choose what to do for the weight. (BR: means local according to number of part, cross use the force cross-section")
        self.add_param('input_format', 'auto', allowed=['auto','lhe', 'hepmc', 'lhe_no_banner'])
        self.add_param('frame_id', 6)
        self.add_param('global_order_coupling', '')
        self.add_param('identical_particle_in_prod_and_decay', 'average')
        self.add_param('beampol', [0., 0.], comment='beam polarisation of each beam in percent, -100 .. 100, exactly as the run_card polbeam1/polbeam2 (0 is unpolarised). Taken from the run_card of the production when it has one.')
        self.add_param('decay_output', 'auto',
                       allowed=['auto', 'unweighted', 'weighted'],
                       comment="whether MadSpin unweights its decays at all. 'auto' "
                       "(default): 'weighted' when pure_interference is set, "
                       "'unweighted' otherwise -- i.e. each mode's own historical "
                       "default. The resolved value is announced in the log. "
                       "'unweighted' (what MadSpin has always done outside "
                       "pure_interference): draw decay "
                       "configurations until one is accepted, so every production event "
                       "yields exactly one output event and every event of a given "
                       "production process carries the same weight. 'weighted': NO "
                       "accept/reject -- draw ONE decay configuration per production event, "
                       "keep it, and put the convolution on the weight, "
                       "w = w_prod * BR * W / c, with W this trial's production/decay "
                       "density convolution and c = <W> its decay-phase-space mean (a "
                       "constant, measured by the same probe that estimates the maximum "
                       "weight). Under MG5's IDWTUP = -4 convention the cross-section is "
                       "the MEAN of the weights, and mean(w) = sigma*BR exactly as before, "
                       "so <init> is unchanged and the file is self-normalising. WHY, "
                       "measured on p p > t t~ with spinmode = onshell and 50 000 events: "
                       "the ordinary accept/reject burned 4.13 decay trials per written "
                       "event, each one a matrix-element evaluation, and this mode burns "
                       "exactly 1; the price is only a 6-13% larger variance per "
                       "production event (the weights have sd/mean = 0.28), so per unit "
                       "of CPU in the unweighting loop it is about a 3.7-3.9x variance "
                       "reduction. WHAT YOU GIVE UP: the output is a WEIGHTED LHE "
                       "file. Anything downstream that assumes MadSpin events carry a "
                       "constant weight -- unit-weight event counting, simple histogram "
                       "entry counts -- is wrong on it. Density spin modes only "
                       "(madspin/full, PA, onshell): the v1 modes and spinmode = none "
                       "build no density matrix and have no W. Under "
                       "pure_interference this option governs that mode's (always "
                       "signed) output too: 'weighted' keeps every trial with "
                       "w = sigma_ref*BR*W/c, 'unweighted' unweights on |W| with ONE "
                       "draw per production event and writes w = +- sigma_ref*BR*<|W|>/c "
                       "-- exactly two weight magnitudes, i.e. unweighted up to a sign. "
                       "See doc/madspin_sequential_plan.md sections 13.17 and 13.18.")
        self.add_param('pure_interference', '',
                       comment="pure-interference mode: keep ONLY the interference between two "
                       "polarisations of a decaying particle in the production/decay density "
                       "convolution. Syntax 'set pure_interference t = 0 T' (production-side set "
                       "= decay-side set); each side is one or more of 0, +/R, -/L, T. Two "
                       "DISJOINT sides name that particle's interference block I; two IDENTICAL "
                       "sides name its diagonal block (so 'set pure_interference t~ = - -' is D-), "
                       "and a partial overlap is refused. Use ONE 'set' line per particle -- "
                       "repeated lines accumulate; ';' cannot be used because every card line is "
                       "split on it. A particle the option does not name is left unrestricted, "
                       "i.e. summed over its whole basis. At least one particle must carry a "
                       "genuine (disjoint) interference block. The production process must be "
                       "UNPOLARISED on the legs given an I block (the interference between two "
                       "polarisations does not exist in a sample generated with a brace on that "
                       "leg). The sample then has zero total cross-section by construction, and "
                       "its event weights are SIGNED; see decay_output for their "
                       "value and doc/madspin_sequential_plan.md section 13.")
        self.add_param('keep_weight_for_polarization_vector', [], typelist=str,
                       comment="density spin modes only. Polarisations (0, +, -, T; "
                       "L/R accepted as aliases of -/+) offered to each decaying "
                       "SPIN-1 particle. Together with "
                       "keep_weight_for_polarization_fermion it defines a set of "
                       "polarisation COMBINATIONS -- one per element of the cartesian "
                       "product over the decaying particles, each particle drawing "
                       "from the list of its own species -- and every event then "
                       "carries one EXTRA weight per combination in its LHEF v3 <rwgt> "
                       "block, equal to nominal_weight * (density convolution "
                       "restricted to that combination) / (nominal density "
                       "convolution). The nominal weight and the cross-section are "
                       "untouched, and two empty lists (the default) change nothing at "
                       "all. Example: on 'p p > t t~ z' with vector=[0, T, +, -] and "
                       "fermion=[+, -] an event carries 2*2*4 = 16 extra weights, "
                       "named after the per-particle assignment "
                       "(ms_pol_6:+_-6:-_23:0 and so on, in density-basis slot order). "
                       "A particle whose species list is empty -- and a scalar, which "
                       "has no polarisation -- is left summed over its helicities and "
                       "does not multiply the count; its slot shows up as '*' in the "
                       "weight id. When the production process itself carries a "
                       "polarisation brace, each slot's choices are intersected with "
                       "it (a choice with an empty intersection is dropped) and the "
                       "denominator is the (already restricted) nominal convolution, "
                       "so a weight stays the fraction of the sample that is written "
                       "out.")
        self.add_param('keep_weight_for_polarization_fermion', [], typelist=str,
                       comment="as keep_weight_for_polarization_vector, but the list "
                       "offered to each decaying SPIN-1/2 particle. '0' is unphysical "
                       "for a fermion and is dropped from its choices; 'T' is its full "
                       "helicity basis, i.e. that particle summed over.")
        self.add_param('density_debug', False, comment='Turn on check against full ME calculation')
        self.add_param('density_tolerance', 1E-4, comment='Tolerance for deviation between density and full ME')
        self.add_param('decay_event_mult', 1E0, comment='Produce more events than needed so that MadSpin does not have to regenerate decay events')
        self.add_param('nb_core', 0, comment='Number of cores for the MadSpin parallel unweighting (0 = use the global MG5 nb_core). nb_core>1 enables the process-parallel unweighting path.')
        self.add_param('density_keep_jacobian', True, comment='PA spinmode only: fold the offshell-reshuffling phase-space jacobian into the accept/reject weight (default) instead of applying the reshuffle as a post-acceptance kinematic dressing (False). Ignored by the madspin/full spinmodes, which always include that jacobian.')
        self.add_param('unweighting', 'auto',
                       allowed=['auto', 'joint', 'sequential',
                                'sequential_global_retry',
                                'sequential_with_mass'],
                       comment="how the accept/reject is organised (density modes). "
                       "joint: one test over the virtualities and every decay at once, the historical scheme. "
                       "sequential: unweight the set of virtualities first, then one test per decaying particle, redrawing only the particle that was rejected -- the production reshuffling and its density matrix are then evaluated once per accepted mass set instead of once per trial. "
                       "sequential_global_retry: as sequential, but a rejected decay redraws the virtualities too. "
                       "sequential_with_mass: one test per decaying particle with that particle's virtuality drawn *inside* its own accept/reject, so nothing is ever frozen and no stage has a conditional normalisation to divide out. Needs a per-particle mass draw, i.e. the PA spinmode; elsewhere it falls back to sequential. "
                       "sequential and sequential_global_retry unweight the set of virtualities first; the former then needs a tabulated running-width factor, measured during the max-weight scan to ~0.5%, which is far inside the pole approximation these modes already assume; sequential_global_retry does without it at 2-3x the cost, and is meant as a cross-check rather than a default. "
                       "auto: sequential under PA/onshell, where it was the fastest scheme at every decay multiplicity measured; offshell joint up to two decaying particles and sequential from three, since offshell every mass set costs a production reshuffle and a production density and below three decays there are not enough of them to save to pay for it; but sequential at every multiplicity when the production process carries a polarisation brace, since restricting the convolution to a polarisation subspace peaks the joint weight far below the single bound the joint test has -- measured on `p p > t t~` with both tops decayed, 112 trials per accepted event under joint for `t{+}t~{+}` and 162 for `t{+}t~{-}` against 9.1 and 8.4 under sequential, where unpolarised joint takes 3.3 (and at 50000 events, where the max-weight bound is looser still, the polarised joint columns were 204-213 and 5800-6300). An explicit 'set unweighting joint' is still honoured.")
        self.add_param('sequential_spin_order', '2 3 1', comment='spin order (MG5 2S+1 convention) deciding which particle is accept/rejected first in the sequential unweighting modes: default fermions, then vectors, then scalars (which can never be rejected).')
        self.add_param('sequential_debug', False, comment='the up-front-mass unweighting schemes (sequential, sequential_global_retry): on every accepted chain, recompute the joint weight for the same production event, virtualities and decays and check that the product of the stage weights reproduces it (times the number of helicity states). Deterministic check of the decomposition itself -- the tabulated factor cancels out of it -- at roughly the cost of a joint trial per event. Debugging only.')

    def __setitem__(self, name, value, change_userdefine=False, raiseerror=False):
        """Let an old card keep an unweighting scheme we no longer advertise.

        Hiding a scheme means dropping it from 'allowed', and ConfigFile then
        refuses it outright -- which would turn a card written before the
        scheme was retired into a warning plus a silent switch back to 'auto'.
        The code path is untouched, so accept the value instead: widen the
        allowed list for the duration of this one assignment and note it at
        debug level, quietly enough not to re-advertise it.
        """
        if isinstance(name, str) and isinstance(value, str) and \
                name.strip().lower() == 'unweighting' and \
                value.strip().lower() in self.hidden_unweighting_modes:
            value = value.strip().lower()
            allowed = getattr(self, 'allowed_value', {}).get('unweighting')
            if allowed is not None and value not in allowed:
                logger.debug("MadSpin: unweighting = %s is an internal "
                             "cross-check scheme, no longer offered in the "
                             "card; honouring it since it was asked for "
                             "explicitly.", value)
                self.allowed_value['unweighting'] = list(allowed) + [value]
                try:
                    return super(MadSpinOptions, self).__setitem__(
                        name, value, change_userdefine, raiseerror)
                finally:
                    self.allowed_value['unweighting'] = allowed
        return super(MadSpinOptions, self).__setitem__(
            name, value, change_userdefine, raiseerror)

    ############################################################################
    ##  Special post-processing of the options                                ##
    ############################################################################
    def post_set_ms_dir(self, value, change_userdefine, raiseerror, *opts):
        """ special handling for set ms_dir """
        
        self.__setitem__('curr_dir', value, change_userdefine=change_userdefine)
        
    ############################################################################
    def post_set_seed(self, value, change_userdefine, raiseerror):
        """ special handling for set seed """
        
        if not hasattr(random, 'mg_seedset'):
            random.seed(self['seed'])  
            random.mg_seedset = self['seed']  

    def post_set_beampol(self, value, change_userdefine, raiseerror, *opts):
        """Two values or none: one number is ambiguous (which beam?) and would
        otherwise only surface as an IndexError once the run reaches a matrix
        element, long after the card was read."""
        if value and len(value) != 2:
            raise banner.InvalidCmd(
                "beampol takes the polarisation of *both* beams, in percent: "
                "'set beampol [%s, 0]' for the first beam only. Got %s value(s)."
                % (value[0] if value else 0, len(value)))

    @staticmethod
    def _canonical_polarization_list(name, value):
        """Reject an unknown polarisation label at card-reading time, and return
        the canonical spelling (so '{l}' and 'L' both become '-'). Anything but
        0/+/-/T (with L/R as aliases) has no meaning in the helicity bases the
        density spin modes use."""
        canonical = []
        for entry in value:
            parsed = parse_polarization_label(entry)
            if parsed is None:
                raise banner.InvalidCmd(
                    "%s: '%s' is not a polarisation. "
                    "Use 0, +, - or T (L and R are accepted as aliases of - and +)."
                    % (name, entry))
            if parsed[0] not in canonical:
                canonical.append(parsed[0])
        return canonical

    def post_set_keep_weight_for_polarization_vector(self, value,
                                                     change_userdefine,
                                                     raiseerror, *opts):
        if not value:
            return
        name = 'keep_weight_for_polarization_vector'
        canonical = self._canonical_polarization_list(name, value)
        if canonical != list(value):
            dict.__setitem__(self, name, canonical)

    def post_set_keep_weight_for_polarization_fermion(self, value,
                                                      change_userdefine,
                                                      raiseerror, *opts):
        if not value:
            return
        name = 'keep_weight_for_polarization_fermion'
        canonical = self._canonical_polarization_list(name, value)
        if canonical != list(value):
            dict.__setitem__(self, name, canonical)

    def beampol_me(self):
        """The beam polarisations in the convention the matrix elements use.

        The card and the run_card both speak percent (-100 .. 100, 0 for an
        unpolarised beam). ``/to_beampol/`` -- the v1 driver's msP/msF SMATRIX
        and GET_DENSITY -- is a verbatim copy of madevent's
        ``/to_polarization/`` reweighting, so it wants madevent's *internal*
        value, ``sign(1 + |polbeam|/100, polbeam)``: 1 unpolarised, +2 fully
        polarised along +1 helicity, -2 fully polarised along -1. Converting
        here rather than at parse time keeps the stored option equal to what the
        user typed, and keeps one convention in the cards and one in Fortran.
        """
        pol = self['beampol'] or [0., 0.]      # unset / [] is unpolarised
        out = []
        for value in (pol[0], pol[1]):
            value = float(value)
            out.append(1. if not value
                       else math.copysign(1 + abs(value) / 100., value))
        return tuple(out)

    ############################################################################        
    def post_set_run_card(self, value, change_userdefine, raiseerror, *opts):
        """ special handling for set run_card """
        
        if value == 'default':
            self.run_card = None
        elif not value:
            self.run_card = None
        elif os.path.isfile(value):
            self.run_card = banner.RunCard(value)
        else:
            args = value.split()
            if  len(args) >1:
                if not hasattr(self, 'run_card'):
                    self.run_card =  banner.RunCardLO()
                    self.run_card.remove_all_cut()
                self.run_card[args[0]] = ' '.join(args[1:])
            else:
                raise Exception("wrong syntax for \"set run_card %s\"" % value)
            
        
    ############################################################################
    def post_fixed_order(self, value, change_userdefine, raiseerror):
        """ special handling for set fixed_order """
        
        if value:
            logger.warning('Fix order madspin fails to have the correct scale information. This can bias the results!')
            logger.warning('Not all functionalities of MadSpin handle this mode correctly (only onshell mode so far).')
            logger.warning('spinmode=PA and spinmode=madspin/full reshuffle the production, which an event group\'s '
                           'counter-events cannot follow; launch will refuse those combinations.')

    ############################################################################
    def post_identical_particle_in_prod_and_decay(self, value, change_userdefine, raiseerror):
        """ special handling for set fixed_order """
        if value not in ["crash", 'average', 'max', 'first']:
            raise Exception("value %s not supported for this parameter identical_in_prod_and_decay")

def _force_rmtree(path):
    """shutil.rmtree that also succeeds on read-only trees. Frozen concurrent
    gridpacks are chmod 555, so a plain rmtree raises PermissionError; make every
    directory/file writable first, then remove."""
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            try:
                os.chmod(root, 0o755)
            except OSError:
                pass
            for fname in files:
                try:
                    os.chmod(pjoin(root, fname), 0o644)
                except OSError:
                    pass
    shutil.rmtree(path, ignore_errors=True)


class _StridedEvents(object):
    """One parallel worker's disjoint, lock-free view of a shared decay-event
    file.

    Worker number ``offset`` (0 <= offset < stride) consumes the decay events
    at file positions ``offset, offset+stride, offset+2*stride, ...``; the
    events in between belong to the other workers, which each hold their own
    independent file handle over the same file. Because the stripes are
    disjoint, no decay event is ever consumed twice, so the statistics are
    unbiased and identical in distribution to the serial consumption.

    Only the attributes that :func:`MadSpinInterface.get_decay_from_file`
    actually reads are proxied (``cross`` for cross-section-weighted channel
    selection, ``name`` for reopening), so that hot function stays unchanged.
    On exhaustion this raises ``StopIteration`` exactly like an ``EventFile``,
    letting the caller trigger its (now shard-private) refill path.
    """

    def __init__(self, evtfile, offset, stride):
        self.f = evtfile
        self.stride = stride
        self._exhausted = False
        # advance to this worker's phase in the shared file
        for _ in range(offset):
            try:
                next(self.f)
            except StopIteration:
                self._exhausted = True
                break

    def __iter__(self):
        return self

    def __next__(self):
        if self._exhausted:
            raise StopIteration
        ev = next(self.f)                     # this worker's event
        for _ in range(self.stride - 1):      # skip the other workers' events
            try:
                next(self.f)
            except StopIteration:
                self._exhausted = True
                break
        return ev
    next = __next__

    @property
    def cross(self):
        return self.f.cross

    @property
    def name(self):
        return self.f.name


class _ChainedEvents(object):
    """Reader over a decay pool that the unweighting wrote as several files
    (run_card ``nb_unweight_output``).

    The parent (max-weight estimation, serial unweighting) consumes it as a
    single stream. ``paths`` is kept public so that each parallel worker can
    instead open just *its* file: that is what removes the need to stride, i.e.
    to have every worker scan the whole pool to pick one event out of nb_core.
    Only the attributes get_decay_from_file reads are exposed (``cross`` for the
    cross-section weighted channel choice, ``name``).
    """

    def __init__(self, paths):
        self.paths = list(paths)
        self._idx = -1
        self._current = None
        self._first = lhe_parser.EventFile(self.paths[0])

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._current is None:
                self._idx += 1
                if self._idx >= len(self.paths):
                    raise StopIteration
                self._current = lhe_parser.EventFile(self.paths[self._idx])
            try:
                return next(self._current)
            except StopIteration:
                try:
                    self._current.close()
                except Exception:
                    pass
                self._current = None
    next = __next__

    @property
    def cross(self):
        return self._first.cross

    @property
    def name(self):
        return self.paths[0]


class _LimitedEvents(object):
    """Reader that yields at most ``limit`` events from ``evtfile`` then raises
    StopIteration, as if the file were that much shorter.

    Used to make a channel's *owner* worker (see MadSpinInterface._channel_owner)
    run its slice of the decay pool out ~10% before the other workers do. The
    owner is the only worker allowed to (re)generate that channel, so having it
    reach the refill point first means the pool is usually ready by the time the
    others need it -- they rarely have to block. Only the attributes
    get_decay_from_file reads (``cross``, ``name``) are proxied."""

    def __init__(self, evtfile, limit):
        self.f = evtfile
        self.limit = max(0, int(limit))
        self._n = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._n >= self.limit:
            raise StopIteration
        ev = next(self.f)
        self._n += 1
        return ev
    next = __next__

    @property
    def cross(self):
        return self.f.cross

    @property
    def name(self):
        return self.f.name


class MadSpinInterface(extended_cmd.Cmd):
    """Basic interface for madspin"""

    prompt = 'MadSpin>'
    debug_output = 'MS_debug'

    # Process-wide counter used to make each MadSpinInterface instance use
    # its own ``madspin_me`` output subdirectory. The actual fortran/f2py
    # artefacts (the ``.so`` extension and the ``liball_2me`` shared
    # library) are recompiled to that subdirectory each call, and
    # ``dlopen`` caches loaded libraries by absolute path: without a fresh
    # path the second MadSpin call in the same process (e.g. inline
    # MadSpin followed by ``decay_events``) keeps the *first* call's
    # matrix elements in memory and ``pdg2prefix`` ends up missing the
    # second card's decay channels (KeyError in ``get_pdir``).
    _ms_run_counter = 0


    @misc.mute_logger()
    def __init__(self, event_path=None, *completekey, **stdin):
        """initialize the interface with potentially an event_path"""

        cmd_logger.info('************************************************************')
        cmd_logger.info('*                                                          *')
        cmd_logger.info('*           W E L C O M E  to  M A D S P I N               *')
        cmd_logger.info('*                                                          *')
        cmd_logger.info('************************************************************')
        extended_cmd.Cmd.__init__(self, *completekey, **stdin)

        MadSpinInterface._ms_run_counter += 1
        self._ms_run_id = MadSpinInterface._ms_run_counter
        # First call keeps the historical ``madspin_me`` name (the
        # decay/import code still hard-codes that in a few places and the
        # vast majority of MadSpin uses only sees one run per process).
        # Subsequent calls use a unique suffix so dlopen sees a fresh
        # file path.
        if self._ms_run_id == 1:
            self.ms_me_subdir = 'madspin_me'
            self.ms_me_decay_subdir = 'madspin_decay'
        else:
            self.ms_me_subdir = 'madspin_me_%d' % self._ms_run_id
            self.ms_me_decay_subdir = 'madspin_decay_%d' % self._ms_run_id

        self.decay = madspin.decay_misc()
        self.model = None
        #self.mode = "madspin" # can be flat/bridge change the way the decay is done.
        #                      # note amc@nlo does not support bridge.
        
        self.options = MadSpinOptions()
        
        self.events_file = None
        self.decay_processes = {}
        self.list_branches = {}
        # the resolved '@' decay grouping, or None when the card declares none
        # (or declares one this run cannot honour). See _resolve_decay_groups.
        self._decay_groups = None
        self.to_decay={}
        self.mg5cmd = master_interface.MasterCmd()
        self.seed = None
        self.err_branching_ratio = 0
        self.me_run_name = "" # Events diretory name where to stotre the events (used by madevent) not use internally
        self.all_iden = {}
        # The card this run executes, and the directory do_import derives from
        # the event file. Both are what madspin_card_path archives from; set
        # before the do_import below, which fills the second one in.
        self.ms_card_path = None
        self.event_base_dir = None

        if event_path:
            logger.info("Extracting the banner ...")
            self.do_import(event_path)
            
    
    def setup_for_pure_decay(self):
        """this is for spinmode=none -> simple decay
           We go here if they are no banner.
           -> this requires that a command import model appears in the card!
        """

        logger.info("Setup the code for pure decay mode")
        self.proc_option = []
        self.final_state_full = ''
        self.final_state_compact = ''
        self.prod_branches = ''
        self.final_state = set()

    def _load_f2py_matrix_module(self, sp_path, menum=2):
        """Load the freshly-compiled ``all_matrix<menum>py`` extension under
        ``sp_path``.

        Each MadSpin run compiles its matrix elements into its own
        ``madspin_me_<N>`` subdir, and (from the second call onwards)
        ``decay.compile()`` overrides the makefile's ``PROCNAME`` so the
        resulting Fortran shared library
        (``liball<PROCNAME>_<MENUM>me.{so,dylib}``) has a unique SONAME /
        install_name. The combination of a unique wrapper path *and* a
        unique dependent-library identity is what stops the dynamic
        loader from returning the first call's already-loaded matrix
        elements on the second call.

        Within a single run the production (madspin_me) and decay
        (madspin_decay) modules are both loaded into this process; they are
        built with distinct ``MENUM`` values (2 for production, 1 for decay)
        so the f2py module name (``all_matrix<MENUM>py``) and the dependent
        library (``liball_<MENUM>me``) differ — otherwise the two identically
        named Fortran extensions clash and segfault. ``menum`` selects which
        one to load.

        This helper just picks the loadable ``.so`` and loads it via
        ``importlib.util.spec_from_file_location`` to bypass the
        ``sys.modules`` cache (which would otherwise short-circuit
        ``__import__`` to the first call's module object).
        """
        import importlib.util
        import glob

        modname = 'all_matrix%dpy' % menum
        # The actual loadable file is the cpython-tagged ``.so``; on some
        # builds the unsuffixed ``all_matrix<menum>py.so`` is a 0-byte stub.
        # Pick the largest matching file so we always load real code.
        patterns = [
            '%s.cpython*.so' % modname,
            '%s.cpython*.dylib' % modname,
            '%s.so' % modname,
            '%s.dylib' % modname,
        ]
        candidates = []
        for pat in patterns:
            for hit in glob.glob(pjoin(sp_path, pat)):
                if os.path.getsize(hit) > 0:
                    candidates.append(hit)
        if not candidates:
            # Fall back to the historical ``__import__`` so we at least
            # produce a meaningful error if nothing got compiled.
            return __import__(modname)
        candidates.sort(key=os.path.getsize, reverse=True)
        so_path = candidates[0]

        # Load via spec_from_file_location to bypass the sys.modules cache
        # while keeping the module name as ``all_matrix<menum>py`` (the .so's
        # PyInit_all_matrix<menum>py init symbol is baked in at compile time).
        spec = importlib.util.spec_from_file_location(modname, so_path)
        if spec is None or spec.loader is None:
            return __import__(modname)
        mymod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mymod)
        return mymod

    def _log_lhe_timers(self):
        if not getattr(lhe_parser, "_ENABLE_LHE_TIMERS", False):
            return
        timers, counts = lhe_parser.get_lhe_timers()
        if not timers:
            print("LHE parser timing enabled but no samples were collected.")
            return
        print("LHE parser timing summary:")
        for key in sorted(timers):
            total = timers[key]
            count = counts.get(key, 1)
            print("  %s: %.6fs total over %d call(s) (avg %.6fs)" %
                  (key, total, count, total / max(1, count)))
        
     
    def do_import(self, inputfile):
        """import the event file"""
        
        args = self.split_arg(inputfile)
        if not args:
            return self.InvalidCmd, 'import requires arguments'
        elif args[0] == 'model':
            return self.import_model(args[1:])
        
        # change directory where to write the output
        self.options['curr_dir'] = os.path.realpath(os.path.dirname(inputfile))
        if os.path.basename(os.path.dirname(os.path.dirname(inputfile))) == 'Events':
            self.options['curr_dir'] = pjoin(self.options['curr_dir'],
                                                      os.path.pardir, os.pardir)
        # Keep that directory -- the process root when the events sit in
        # Events/<run>/, the event file's own directory otherwise -- under a
        # name of its own. 'set ms_dir' re-points curr_dir at the gridpack
        # (post_set_ms_dir), so curr_dir stops answering "where did these
        # events come from" as soon as a card mentions ms_dir after the import.
        # See madspin_card_path.
        self.event_base_dir = self.options['curr_dir']

        if not os.path.exists(inputfile):
            if inputfile.endswith('.gz'):
                if not os.path.exists(inputfile[:-3]):
                    misc.sprint(os.getcwd(), os.listdir('.'), inputfile, os.path.exists(inputfile), os.path.exists(inputfile[:-3]))
                    raise self.InvalidCmd('No such file or directory : %s' % inputfile)
                else: 
                    inputfile = inputfile[:-3]
            elif os.path.exists(inputfile + '.gz'):
                inputfile = inputfile + '.gz'
            else: 
                raise self.InvalidCmd('No such file or directory : %s' % inputfile)

        self.inputfile = inputfile
        if self.options['spinmode'] == 'none' and \
           (self.options['input_format'] not in ['lhe','auto'] or 
             (self.options['input_format'] == 'auto' and '.lhe'  not in inputfile[-7:])):  
            self.banner = banner.Banner()
            self.setup_for_pure_decay()
            return  
        
        if inputfile.endswith('.gz'):
            misc.gunzip(inputfile)
            inputfile = inputfile[:-3]
        # Read the banner of the inputfile
        self.events_file = open(os.path.realpath(inputfile))
        self.banner = banner.Banner(self.events_file)


        # Check the validity of the banner:
        if 'slha' not in self.banner:
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain model information')
        elif 'mg5proccard' not in self.banner:
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain generation information')

        
        if 'madspin' in self.banner:
            raise self.InvalidCmd('This event file was already decayed by MS. This is not possible to add to it a second decay')
        
        if 'mgruncard' in self.banner:
            run_card = self.banner.charge_card('run_card')
            if not self.options['Nevents_for_max_weight']:
                nevents = run_card['nevents']
                N_weight = max([75, int(3*nevents**(1/3))])
                self.options['Nevents_for_max_weight'] = N_weight
                N_sigma = max(4.5, math.log(nevents,7.7))
                self.options['nb_sigma'] = N_sigma
            if self.options['BW_cut'] == -1:
                self.options['BW_cut'] = float(self.banner.get_detail('run_card', 'bwcutoff'))
                if self.options['BW_cut'] > 25:
                    logger.critical("value of bwcutoff set to %s from the input file. This is much too large value for Madspin and the validity of the Narrow-width-Approximation. Please ensure that you overwrite that value via \"set BW_cut X\"  to a smaller value (like X=10)", self.options['BW_cut'])
            
            if isinstance(run_card, banner.RunCardLO):
                run_card.update_system_parameter_for_include()
                # The run_card of the production is the default source for both,
                # but an explicit "set frame_id"/"set beampol" in the MadSpin
                # card wins -- otherwise neither option could be set from the
                # card the rest of the MadSpin options live in.
                if 'frame_id' not in self.options.user_set:
                    self.options['frame_id'] = run_card['frame_id']
                if 'beampol' not in self.options.user_set:
                    self.options['beampol'] = [run_card['polbeam1'],
                                               run_card['polbeam2']]
            else:
                if 'frame_id' not in self.options.user_set:
                    self.options['frame_id'] = 6
                if 'beampol' not in self.options.user_set:
                    self.options['beampol'] = [0., 0.]

        else:
            if not self.options['Nevents_for_max_weight']:
                self.options['Nevents_for_max_weight'] = 75
                self.options['nb_sigma'] = 4.5
            if self.options['BW_cut'] == -1:
                self.options['BW_cut'] = 15.0
                
                
        # load information
        process = self.banner.get_detail('proc_card', 'generate')
        if not process:
            msg = 'Invalid proc_card information in the file (no generate line):\n %s' % self.banner['mg5proccard']
            raise Exception(msg)
        process, option = mg_interface.MadGraphCmd.split_process_line(process)
        self.proc_option = option
        
        logger.info("process: %s" % process)
        logger.info("options: %s" % option)

        if not hasattr(self,'multiparticles_ms'):
            for key, value in self.banner.get_detail('proc_card','multiparticles'):
                try:
                    self.do_define('%s = %s' % (key, value))
                except self.mg5cmd.InvalidCmd:  
                    pass
                
        # Read the final state of the production process:
        #     "_full" means with the complete decay chain syntax 
        #     "_compact" means without the decay chain syntax 
        self.final_state_full = process[process.find(">")+1:]
        self.final_state_compact, self.prod_branches=\
                 self.decay.get_final_state_compact(self.final_state_full)
                
        # Load the model
        complex_mass = False   
        has_cms = re.compile(r'''set\s+complex_mass_scheme\s*(True|T|1|true|$|;)''')
        for line in self.banner.proc_card:
            if line.startswith('set'):
                self.mg5cmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                if has_cms.search(line):
                    complex_mass = True
        
          
        info = self.banner.get('proc_card', 'full_model_line')
        if '-modelname' in info:
            mg_names = False
        else:
            mg_names = True
        model_name = self.banner.get('proc_card', 'model')
        if model_name:
            model_name = os.path.expanduser(model_name)
            self.load_model(model_name, mg_names, complex_mass)
        else:
            raise self.InvalidCmd('Only UFO model can be loaded in MadSpin.')
        # check particle which can be decayed:
        self.final_state = set()
        final_model = False
        for line in self.banner.proc_card:
            line = ' '.join(line.strip().split())
            if line.startswith('generate'):
                self.final_state.update(self.mg5cmd.get_final_part(line[8:]))
            elif line.startswith('add process'):
                self.final_state.update(self.mg5cmd.get_final_part(line[11:]))
            elif line.startswith('define'):
                try:
                    self.mg5cmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                except self.mg5cmd.InvalidCmd:
                    if final_model:
                        raise
                    else:
                        key = line.split()[1]
                        if key in self.multiparticles_ms:
                            del self.multiparticles_ms[key]            
            elif line.startswith('set') and not line.startswith('set gauge'):
                self.mg5cmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
            elif line.startswith('import model'):
                if model_name in line:
                    final_model = True
                    
                
    def import_model(self, args):
        """syntax: import model NAME CARD_PATH
            args didn't include import model"""
        
        bypass_check = False
        if '--bypass_check' in args:
            args.remove('--bypass_check')
            bypass_check = True
        if len(args) == 1:  
            logger.warning("""No param_card defined for the new model. We will use the default one but this might completely wrong.""")
        elif len(args) != 2:
            return self.InvalidCmd, 'import model requires two arguments'
        
        model_name = args[0]
        self.load_model(model_name, False, False)
        
        if len(args) == 2:
            card = args[1]
            if not os.path.exists(card):
                raise self.InvalidCmd('%s: no such file' % card)
        else:
            card = "madspin_param_card.dat"
            export_v4.UFO_model_to_mg4.create_param_card_static(self.model,
                                card, rule_card_path=None)

        

        #Check the param_card
        if not (bypass_check or self.options['input_format'] in ['hepmc', 'lhe_no_banner']):
            if not hasattr(self.banner, 'param_card'):
                self.banner.charge_card('slha')
            param_card = check_param_card.ParamCard(card)
            # checking that all parameter of the old param card are present in 
            #the new one with the same value
            try:
                diff = self.banner.param_card.create_diff(param_card)
            except Exception:
                raise self.InvalidCmd('''The two param_card seems very different. 
    So we prefer not to proceed. If you are sure about what you are doing, 
    you can use the command \'import model MODELNAME PARAM_CARD_PATH --bypass_check\'''')
            if diff:
                raise self.InvalidCmd('''Original param_card differs on some parameters:
    %s
    Due to those differences, we prefer not to proceed. If you are sure about what you are doing, 
    you can use the command \'import model MODELNAME PARAM_CARD_PATH --bypass_check\''''
    % diff.replace('\n','\n    '))
   
   
                
        #OK load the new param_card (but back up the old one)
        if 'slha' in self.banner:
            self.banner['slha_original'] = self.banner['slha']
        self.banner['slha'] = open(card).read()
        if hasattr(self.banner, 'param_card'):
            del self.banner.param_card
        self.banner.charge_card('slha')
                


    @extended_cmd.debug()
    def complete_import(self, text, line, begidx, endidx):
        "Complete the import command"
        
        args=self.split_arg(line[0:begidx])
        
        
        if len(args) == 1:
            base_dir = '.'
        else:
            base_dir = args[1]
        
        return self.path_completion(text, base_dir)
        
        # Directory continuation
        if os.path.sep in args[-1] + text:
            return self.path_completion(text,
                                    pjoin(*[a for a in args if \
                                                      a.endswith(os.path.sep)]))

    def do_decay(self, decaybranch):
        """add a process in the list of decayed particles"""
        
        #if self.model and not self.model['case_sensitive']:
        #    decaybranch = decaybranch.lower()

        if '{' in decaybranch and self._density_spinmode():
            # The density spin modes contract a production and a decay
            # spin-density matrix over the decaying particle's helicity. A
            # polarisation written on the *decay* side would have to project
            # that decay density matrix, which is not what the {..} braces do
            # here (they would restrict the decay matrix element that defines
            # the branching ratio instead), so refuse rather than quietly
            # produce a wrong answer. The production-side braces ARE supported:
            # they restrict the convolution (see _production_polarization).
            raise self.InvalidCmd(
                "MadSpin: polarization (the {...} braces) is not supported on a "
                "'decay' line with spinmode=%s. Only the polarization of the "
                "production process is taken into account by the density spin "
                "modes (madspin/full/PA/onshell); use spinmode=none or "
                "spinmode=madspin_v1 for decay-side polarization."
                % self.options['spinmode'])

        if self.options['spinmode'] not in  ['full','madspin', 'madspin_v1'] and '{' in decaybranch:
            if self.options['spinmode'] == 'none':
                logger.warning("polarization option used with spinmode=none. The polarization definition will be done according to the rest-frame of the decaying particles (which is likely not what you expect).")
            else:
                logger.warning("polarization option used with spinmode=%s. This combination is not validated and is by construction using sub-optimal method which can likely lead to bias in some situation. Use at your own risk." % self.options['spinmode'])
        if "=" in decaybranch and self.options['spinmode'] in ['full', 'madspin_v1']:
            logger.warning("Note that coupling order restriction are not associated to specific Branching Ratio. The total cross-section might therefore use the wrong branching ratio.")
        decay_process, init_part = self.decay.reorder_branch(decaybranch)
        if init_part not in self.list_branches:
            self.list_branches[init_part] = []
        self.list_branches[init_part].append(decay_process)
        del decay_process, init_part    
        
    
    def check_set(self, args):
        """checking the validity of the set command"""
        
        if len(args) < 2:
            if args and '=' in args[0]:
                name, value = args[0].split('=')
                args[0]= name
                args.append(value)
            elif len(args) == 1 and args[0] in ['onlyhelicity']:
                args.append('True')
            else:
                raise self.InvalidCmd('set command requires at least two argument.')
        
        if args[1].strip() == '=':
            args.pop(1)
        
        valid = ['max_weight','seed','curr_dir', 'spinmode', 'run_card']
        if args[0] not in self.options and args[0] not in valid:
            raise self.InvalidCmd('Unknown options %s' % args[0])        
    
        if args[0] == 'max_weight':
            try:
                args[1] = float(args[1].replace('d','e'))
            except ValueError:
                raise self.InvalidCmd('second argument should be a real number.')
        
        elif args[0] == 'curr_dir':
            if not os.path.isdir(args[1]):
                raise self.InvalidCmd('second argument should be a path to a existing directory')
        elif args[0] == "spinmode":
            allowed = [mode.lower() for mode in self.options.allowed_value['spinmode']]
            if args[1].lower() not in allowed:
                raise self.InvalidCmd("spinmode can only take one of those values: full/madspin/onshell/none/PA/madspin_v1/onshell_v1")

        elif args[0] == "run_card":
            if self.options['spinmode'] in ["madspin_v1"]:
                raise self.InvalidCmd("edition of the run_card is not allowed within spinmode=madspin_v1")
            if "=" in args:
                args.remove("=")
            if len(args)==2 and "=" in args[1]:
                data = args.pop(1)
                arg, value = data.split("=")
                args.append(arg)
                args.append(value)
        elif args[0] == 'Nevents_for_max_weigth':
            args[0] = 'Nevents_for_max_weight'
        
    # Options whose repeated ``set`` lines ACCUMULATE instead of overwriting.
    # ``pure_interference`` is a per-particle mapping, and the one-line spelling
    # for several particles cannot be written: extended_cmd.Cmd.precmd splits
    # every card line on ';' and dispatches the pieces as separate commands, so
    #
    #     set pure_interference t = + - ; t~ = + -
    #
    # loses the t~ half. One ``set`` line per particle is the only spelling that
    # survives, so it has to be the one that works.
    ACCUMULATING_OPTIONS = ('pure_interference',)

    def do_set(self, line):
        """ add one of the options """
        
        args = self.split_arg(line)
        self.check_set(args)

        value = ' '.join(args[1:])
        if args[0] in self.ACCUMULATING_OPTIONS:
            previous = (self.options[args[0]] or '').strip()
            if previous and value.strip():
                value = '%s ; %s' % (previous, value.strip())
            # the parsed form is memoised; a second set line has to invalidate it
            self._pure_interference_cache = None
        self.options[args[0]] = value
        # ConfigFile only fills user_set through its own set(); record it here
        # so options that are otherwise taken from the production run_card
        # (frame_id, beampol) can still be overridden from the MadSpin card.
        self.options.user_set.add(args[0].strip().lower())
        

    def default(self, line, log=True):
        """Unrecognised command.

        One case is not a typo but a silently wrong physics result, so it is
        promoted to an error: the ``;`` spelling of a multi-particle
        ``pure_interference``. ``extended_cmd.Cmd.precmd`` splits card lines on
        ``;`` and dispatches the pieces, so

            set pure_interference t = + - ; t~ = + -

        reaches ``do_set`` as ``t = + -`` (a perfectly valid single-particle
        request) followed by the orphan ``t~ = + -``, which lands here. The run
        would then produce a different, valid-looking sample with nothing but a
        generic warning. Refuse instead, and say what to write.
        """
        if self._looks_like_pure_interference_entry(line):
            raise self.InvalidCmd(
                "MadSpin: '%s' is not a command. It looks like the tail of a "
                "';'-separated pure_interference specification -- and ';' can "
                "never work, because every MadSpin card line is split on it "
                "and the pieces are run as separate commands, so the tail is "
                "lost and the run would quietly use only the first particle. "
                "Write one 'set' line per particle instead; repeated lines "
                "accumulate:\n"
                "    set pure_interference t  = + -\n"
                "    set pure_interference t~ = + -" % line.strip())
        return super(MadSpinInterface, self).default(line, log=log)

    def _looks_like_pure_interference_entry(self, line):
        """Whether ``line`` parses as a bare ``particle = polA polB`` entry, the
        shape a ';'-truncated pure_interference specification leaves behind."""
        text = line.split('#')[0].strip()
        if not text:
            return False
        sep = '=' if '=' in text else (':' if ':' in text else None)
        if sep is None:
            return False
        name, _, sides = text.partition(sep)
        name = name.strip()
        if not name or ' ' in name:
            return False
        parts = sides.split()
        if len(parts) != 2:
            return False
        tokens = [t.strip().upper()
                  for part in parts for t in part.replace(',', ' ').split()]
        return bool(tokens) and all(t in self._POL_TOKENS for t in tokens)

    def complete_set(self,  text, line, begidx, endidx):
        

        args = self.split_arg(line[0:begidx])

        # Format
        if len(args) == 1:
            opts = list(self.options.keys()) + ['seed']
            return self.list_completion(text, opts) 
        elif len(args) == 2:
            if args[1] == 'ms_dir':
                return self.path_completion(text, '.', only_dirs = True)
        elif args[1] == 'ms_dir':
            curr_path = pjoin(*[a for a in args \
                                                   if a.endswith(os.path.sep)])
            return self.path_completion(text, curr_path, only_dirs = True)
        elif args[1] == "spinmode":
            return self.list_completion(text, ["full", "madspin", "none", "onshell", "PA", "madspin_v1", "onshell_v1"], line)
        elif args[1] == "unweighting":
            # the advertised schemes only: the hidden ones stay settable but
            # are not proposed (see MadSpinOptions.hidden_unweighting_modes)
            return self.list_completion(text,
                       list(self.options.allowed_value['unweighting']), line)


    def help_set(self):
        """help the set command"""
        
        print('syntax: set OPTION VALUE')
        print('')
        print('-- assign to a given option a given value')
        print('   - set max_weight VALUE: pre-define the maximum_weight for the reweighting')
        print('   - set seed VALUE: fix the value of the seed to a given value.')
        print('       by default use the current time to set the seed. random number are')
        print('       generated by the python module random using the Mersenne Twister generator.')
        print('       It has a period of 2**19937-1.')
        print('   - set max_running_process VALUE: allow to limit the number of open file used by the code')
        print('       The number of running is raising like 2*VALUE')
        print('   - set spinmode=XXX: different approximation/implementation are available')
        print('        none: no spin correlation no offshell effects.')
        print('        onshell: spin correlation but no offshell effects.')
        print('        PA: spin correlation and offshell effects with a pure Breit-Wigner.')
        print('        madspin: spin correlation and offshell effects with offshell matrix-elements.')
        print('        full: resolves to madspin.')
        print('        madspin_v1: legacy MadSpin implementation (different reshuffling and no density matrix)')
        print('        onshell_v1: legacy MadSpin implementation (no density matrix) --for debugging only--')
    
    def do_define(self, line):
        """ """

        try:
            self.mg5cmd.exec_cmd('define %s' % line)
        except:
            #cleaning if the error is recover later
            key = line.split()[0]
            if hasattr(self, 'multiparticles_ms') and key in self.multiparticles_ms:
                del self.multiparticles_ms[key]
            raise
           
        self.multiparticles_ms = dict([(k,list(pdgs)) for k, pdgs in \
                                        self.mg5cmd._multiparticles.items()])
    
    
    def update_status(self, *args, **opts):
        """ """
        pass # function overwritten for MS launched by ME
    
    def complete_define(self, *args):
        """ """
        try:
            return self.mg5cmd.complete_define(*args)
        except Exception as error:
            misc.sprint(error)
            
    def complete_decay(self, *args):
        """ """
        try:
            return self.mg5cmd.complete_generate(*args)
        except Exception as error:
            misc.sprint(error)
            
    def check_launch(self, args):
        """check the validity of the launch command"""
        
        if not self.list_branches and not self.options['onlyhelicity']:
            raise self.InvalidCmd("Nothing to decay ... Please specify some decay")
        if not self.events_file:
            raise self.InvalidCmd("No events files defined.")
        
        # Validity check. Need lhe version 3 if matching is on
        if self.banner.get("run_card", "lhe_version") < 3 and \
            self.banner.get("run_card", "ickkw") > 0:
            raise Exception("MadSpin requires LHEF version 3 when running with matching/merging")

    def help_launch(self):
        """help for the launch command"""
        
        print('''Running Madspin on the loaded events, following the decays enter
        An example of a full run is the following:
        import ../mssm_events.lhe.gz
        define sq = ur ur~
        decay go > sq j
        launch
        ''')
        
        self.parser_launch.print_help()

    def parser_launch(self):
        usage = """launch [-n RUN_NAME]   
        """
        parser = misc.OptionParser(usage=usage)
        parser.add_option("-n", "--name",
                  default="",
                  help="When NOT run in standalone instruct MG5aMC where to store the events file")
        return parser
    
    def parse_launch(self, line):
        
        args = self.split_arg(line)
        return self.parser_launch().parse_args(args)
        

    # ``decay t > w+ b, w+ > l+ vl @1``: the @ suffix sorts the decay lines into
    # *groups* meant to be used together -- the semi-leptonic ttbar idiom, where
    # only the two charge assignments exist and no fully leptonic or fully
    # hadronic event is produced.
    #
    # The tag is kept inside the branch string rather than in a structure of its
    # own: ``list_branches`` is renamed, pruned and handed to MG5 from several
    # places, and index i of ``list_branches[name]`` is also the number of the
    # ``decay_<pdg>_<i>`` pool, so a parallel list would be one more thing to
    # keep in step. It is split off at the few points that need it.
    _DECAY_GROUP_TAG = re.compile(r'@\s*(\d+)\s*$')

    @classmethod
    def _split_group_tag(cls, branch):
        """``'t > w+ b, w+ > l+ vl @1'`` -> ``('t > w+ b, w+ > l+ vl', '1')``.

        The tag is returned as a string: it names a group, it is not an integer
        the code ever does arithmetic on. ``(branch, None)`` when untagged.
        """
        found = cls._DECAY_GROUP_TAG.search(branch)
        if not found:
            return branch, None
        return branch[:found.start()].rstrip(), found.group(1)

    def _decay_group_layout(self):
        """Sort the decay lines into groups.

        Returns ``(layout, reason)``. ``layout`` is None when the card declares
        no group at all (``reason`` None too) or when the tags cannot be honoured
        (``reason`` says why, for the warning). Otherwise::

            {'tags':  ['1', '2'],                       # first-appearance order
             'lines': {particle_name: {tag: [index into list_branches[name]]}}}

        An *untagged* line belongs to every group -- the natural way to write a
        third particle that decays the same way whatever the group is, and what
        madspin_v1 does (it prepends the untagged branches to each group).

        This is the half of the check that needs only the card. Whether each
        group covers every decaying particle the right number of times needs the
        production events too; see :meth:`_validate_decay_groups`.
        """
        tags = []
        parsed = {}
        for name, branches in self.list_branches.items():
            parsed[name] = []
            for branch in branches:
                stripped, tag = self._split_group_tag(branch)
                if '@' in stripped:
                    # an '@' that is not a trailing group tag: refuse rather than
                    # half-read it, since MG5 would take it as a process number
                    return None, ("the decay line %r carries an '@' that is not "
                                  "a group tag at the end of the line" % branch)
                parsed[name].append(tag)
                if tag is not None and tag not in tags:
                    tags.append(tag)
        if not tags:
            return None, None
        lines = {}
        for name, name_tags in parsed.items():
            lines[name] = dict((tag, []) for tag in tags)
            for i, tag in enumerate(name_tags):
                for target in (tags if tag is None else [tag]):
                    lines[name][target].append(i)
        return {'tags': tags, 'lines': lines}, None

    def _validate_decay_groups(self, layout, to_decay, nb_event, name2pdg):
        """Can this grouping be honoured for these production events?

        Supported shape -- rectangular: every group gives exactly ``n_part``
        lines for every decaying particle, ``n_part`` being how many of that
        particle each event carries. ``n_part == 1`` is the semi-leptonic ttbar
        idiom; ``n_part == 2`` is ``p p > t t~ t~ t``, where the group's two
        lines for a pdg are handed to its two particles by the existing
        positional rule.

        Anything else is refused rather than approximated: the group would not
        be a complete assignment, so neither its rate ``prod_k Gamma_k`` nor the
        branching ratio is defined.

        ``name2pdg`` maps a decay-line parent name to its pdg, or to None when
        the name is a multiparticle (refused: one name would own several pools).

        Returns ``(ok, reason)``.
        """
        for name, per_tag in layout['lines'].items():
            pdg = name2pdg(name)
            if pdg is None:
                return False, ("%s is a multiparticle: grouping needs one "
                               "decaying particle per decay line" % name)
            if pdg not in to_decay or not nb_event:
                continue            # never appears in the events; ignored anyway
            if to_decay[pdg] % nb_event:
                return False, ("the production events do not all carry the same "
                               "number of %s, and a branching ratio per group "
                               "cannot be defined then" % name)
            nb_part = to_decay[pdg] // nb_event
            for tag in layout['tags']:
                got = len(per_tag.get(tag, ()))
                if got != nb_part:
                    return False, (
                        "group @%s gives %d decay line(s) for %s but every event "
                        "carries %d of them -- each group must decay every "
                        "particle exactly once" % (tag, got, name, nb_part))
        return True, None

    def _warn_ignored_decay_groups(self, spinmode, reason=None):
        """Warn when the card carries @ grouping tags this run cannot honour.

        A tag that silently means nothing is the worst outcome: the
        decay-matrix-element generation appends an ``@`` process number of its
        own to every branch, so MG5 sees two of them, binds its own at the top
        level and absorbs the user's as the process number of the sub-decay.
        Nothing fails, the matrix element that comes out is the correct
        *ungrouped* one, and without this the card simply does not mean what it
        looks like it means.

        Returns the (particle, branch, tag) triples found, for the tests.
        """
        if spinmode == 'madspin_v1':
            return []                     # v1 implements the grouping itself
        tags = []
        for name, branches in self.list_branches.items():
            for branch in branches:
                stripped, tag = self._split_group_tag(branch)
                if tag is not None:
                    tags.append((name, branch, '@%s' % tag))
                elif '@' in stripped:
                    # not a trailing tag, so not a group -- but MG5 will still
                    # read it as a process number, so it is worth the same line
                    tags.append((name, branch, '@?'))
        if not tags:
            return []
        logger.warning(
            "The decay lines carry '@' grouping tags (%s) but this run cannot "
            "honour them: %s. The tags therefore change nothing (MG5 reads them "
            "as an ordinary process number) and the sample will contain EVERY "
            "combination of the channels listed, not only the tagged ones. "
            "Generate one group per MadSpin run and merge the outputs -- fixing "
            "the normalisation with 'set cross_section' if the automatic "
            "branching ratio is not what you want.",
            ', '.join(sorted(set(t[2] for t in tags))),
            reason or ("grouping is available in the density spin modes (PA, "
                       "onshell, madspin/full) and in madspin_v1, and "
                       "spinmode=%s is neither" % spinmode))
        return tags

    # ------------------------------------------------------------------
    # Resolved grouping: what the run-time draw and the branching ratio use
    # ------------------------------------------------------------------
    def _decay_group_pdgs(self):
        """The pdgs whose decay lines are grouped (empty when they are not)."""
        groups = getattr(self, '_decay_groups', None)
        return () if not groups else groups['lines']

    def _resolve_decay_groups(self, to_decay, nb_event, density_method):
        """Turn the card's '@' tags into the grouping the run will use, or warn
        and fall back to the ungrouped behaviour.

        Sets (and returns) ``self._decay_groups``::

            {'tags':  ['1', '2'],
             'lines': {pdg: {tag: [decay pool number]}},
             'prob':  None}                 # filled by _resolve_group_rates

        None when the card declares no group, or when this run cannot honour the
        one it declares. Must run before anything is generated: it decides how
        many pools each particle gets, and it forces the joint accept/reject.
        """
        self._decay_groups = None
        layout, reason = self._decay_group_layout()
        if layout is None and reason is None:
            return None                              # no tags at all

        spinmode = self.options['spinmode']
        if not reason and not density_method:
            reason = ("spinmode=%s does not fill one decay pool per channel, "
                      "which is what a group draws from" % spinmode)
        if not reason and self.options['fixed_order']:
            reason = ("fixed_order rides the counter-events along with the "
                      "decays, and a group would have to be drawn once per "
                      "event group")
        if not reason:
            ok, reason = self._validate_decay_groups(
                layout, to_decay, nb_event, self._decay_parent_pdg)
        if reason:
            self._warn_ignored_decay_groups(spinmode, reason)
            return None

        lines = {}
        for name, per_tag in layout['lines'].items():
            pdg = self._decay_parent_pdg(name)
            if pdg in to_decay:          # a species that never appears in the
                lines[pdg] = per_tag     # events is dropped anyway
        if not lines:
            return None
        self._decay_groups = {'tags': layout['tags'], 'lines': lines,
                              'prob': None}
        logger.info("MadSpin: %s decay groups (@%s); each event draws one group "
                    "and every particle takes that group's channel",
                    len(layout['tags']), ', @'.join(layout['tags']))
        return self._decay_groups

    def _decay_parent_pdg(self, name):
        """pdg of a decay line's parent, or None when the name stands for more
        than one particle (a multiparticle owning several pools at once is what
        the grouping cannot express)."""
        if name in self.mg5cmd._multiparticles:
            pdgs = self.mg5cmd._multiparticles[name]
            return pdgs[0] if len(pdgs) == 1 else None
        try:
            return self.mg5cmd._curr_model.get('name2pdg')[name]
        except KeyError:
            return None

    @staticmethod
    def _clamped_partial_width(pwidth, totwidth, pdg=None):
        """A measured partial width, reconciled with the total width of the
        param_card. Only ever applied to *one* channel's width (or to a sum over
        channels, which is still a width): a product of several is not
        comparable with a single total.

        The two regimes are deliberate, and are the long-standing behaviour this
        helper factors out of run_onshell rather than a new policy:

        - up to 1% above the total, the excess is Monte Carlo noise in the width
          measurement, so it is capped silently and the branching ratio stays at
          most 1;
        - more than 1% above, the param_card's total genuinely disagrees with
          what was generated. That is warned about and the *measured* value is
          used, because capping there would quietly reshape the normalisation to
          match a card that is wrong, and hide the disagreement instead of
          reporting it.
        """
        if pwidth > 1.01 * totwidth:
            logger.warning('partial width (%s) larger than total width (%s) '
                           '--from param_card-- for pdg %s',
                           pwidth, totwidth, pdg)
        elif pwidth > totwidth:
            return totwidth
        return pwidth

    @staticmethod
    def _check_branching_ratio(br, gen_jobs=None):
        """Refuse to decay with a branching ratio that is zero or not a number.

        Why this cannot fire on a healthy run, however extreme the physics: the
        branching ratio is a product of measured-partial-width / total-width
        ratios, and MadSpin measures each partial width by actually generating
        that decay with MadEvent. A channel so suppressed that its width
        underflows to exactly 0 does not reach this point at all -- the
        generation of that channel raises ZeroResult first. What a 0 (or a NaN)
        here means is that one of those factors was never measured, i.e. a
        bookkeeping gap, and the archetype is a reused ``ms_dir``: the decay
        directories already exist, so nothing re-measures them.

        Why it must not be a warning: this number multiplies every event weight
        and the <init> cross-section. Left to run, MadSpin writes a complete,
        well-formed LHE file in which every weight is +/-0.0 and the <init>
        block is zero -- output that no downstream tool rejects and that a user
        can easily fail to notice. Compared with that, stopping is cheap.
        """
        if math.isfinite(br) and br > 0:
            return br
        detail = ''
        if gen_jobs:
            detail = ("\nDecaying particles this run measured a width for: %s"
                      % ', '.join('%s (%s)' % (pdg, job.get('kind'))
                                  for pdg, job in gen_jobs.items()))
        raise MadSpinZeroBranchingRatio(
            "MadSpin computed a branching ratio of %s and will not decay the "
            "events with it.\n"
            "\n"
            "The branching ratio scales every event weight and the <init> "
            "cross-section, so MadSpin would otherwise write a complete, "
            "well-formed event file in which every weight is zero (or not a "
            "number) -- and report success. It stops here instead.\n"
            "\n"
            "It is built as a product of (partial width measured by generating "
            "the decay) / (total width from the param_card), one factor per "
            "decaying particle. A closed decay channel cannot produce this: "
            "generating it fails first. A factor that was never *measured* "
            "can. Plausible causes, most likely first:\n"
            "  * a reused decay directory ('ms_dir', 'use_old_dir') whose "
            "partial width could not be read back -- rerun against a fresh "
            "'ms_dir' to confirm;\n"
            "  * a total width of 0 in the param_card for a particle being "
            "decayed, which makes the ratio a 0/0;\n"
            "  * 'set cross_section' pointing at a zero cross-section.%s"
            % (br, detail))

    @classmethod
    def _assignment_multiplicity(cls, branches):
        """How many *distinct* ways this multiset of decay lines can be dealt to
        that many identical parents: ``n! / prod_c m_c!``.

        The positional rule generates one of those assignments and multiplies
        the rate by their number. A plain ``n!`` is right only while the lines
        are all different -- with two identical ones it counts the same final
        state twice.
        """
        counts = collections.Counter(cls._split_group_tag(b)[0].strip()
                                     for b in branches)
        out = math.factorial(len(branches))
        for repeat in counts.values():
            out //= math.factorial(repeat)
        return out

    def _resolve_group_rates(self, gen_jobs, channel_widths):
        """Branching ratio of the grouped particles, and each group's share.

        A group is one complete assignment of channels to particles, so its rate
        is a product over the particles::

            br_g = prod_pdg  mult(g,pdg) * prod_{i in g} Gamma_i / Gamma_tot^n

        with ``mult`` from :meth:`_assignment_multiplicity`. The groups are
        alternatives, so ``br = sum_g br_g`` -- and the group to use is drawn
        with ``p_g = br_g / br``, which is exactly the unpolarised factorisation
        of its rate. Everything the polarisation adds on top is then carried by
        the accept/reject weight, the same way the per-channel cross-section
        draw already works for the ungrouped multi-channel case.
        """
        rates = []
        for tag in self._decay_groups['tags']:
            rate = 1.0
            for pdg, per_tag in self._decay_groups['lines'].items():
                if pdg not in gen_jobs:
                    continue
                indices = per_tag[tag]
                widths = channel_widths.get(pdg) or {}
                branches = self.list_branches[
                    self.model.get_particle(pdg).get_name()]
                for i in indices:
                    rate *= self._clamped_partial_width(
                        widths.get(i, 0.0), gen_jobs[pdg]['totwidth'], pdg)
                rate *= self._assignment_multiplicity(
                    [branches[i] for i in indices])
                rate /= gen_jobs[pdg]['totwidth'] ** len(indices)
            rates.append(rate)
        total = sum(rates)
        if total <= 0:
            raise Exception("MadSpin: every decay group has a vanishing rate; "
                            "check the decay lines and the param_card widths.")
        self._decay_groups['prob'] = [r / total for r in rates]
        logger.info("MadSpin: decay group rates %s (BR %.6g)",
                    ', '.join('@%s=%.4f' % (tag, p) for tag, p in
                              zip(self._decay_groups['tags'],
                                  self._decay_groups['prob'])), total)
        return total

    def _draw_decay_group(self):
        """The group this chain uses, or None when the card declares none.

        Drawn afresh on every trial -- the caller is the top of the draw, which
        the joint accept/reject re-enters on each rejection. That is what keeps
        the group part of what is being unweighted, rather than something a
        later stage could normalise away.
        """
        groups = getattr(self, '_decay_groups', None)
        if not groups:
            return None
        r = random.random()
        cumul = 0.0
        for tag, prob in zip(groups['tags'], groups['prob']):
            cumul += prob
            if r < cumul:
                return tag
        return groups['tags'][-1]

    ############################################################################
    ##  Archiving the card that was actually run                              ##
    ############################################################################

    def import_command_file(self, filepath):
        """Execute a MadSpin card, remembering which file it came from.

        That file is the record of what this run did, and it is the only thing
        that knows where the card lives: the card is handed in from outside
        (MadEvent's ``do_decay_events`` passes
        ``<me_dir>/Cards/madspin_card.dat``) and is under no obligation to sit
        anywhere in particular. ``madspin_card_path`` archives it next to the
        decayed events.
        """
        if isinstance(filepath, str):
            self.ms_card_path = os.path.realpath(filepath)
        return super(MadSpinInterface, self).import_command_file(filepath)

    @property
    def madspin_card_path(self):
        """The MadSpin card this run executed, or None when there is no file to
        archive (an interactive session types its commands; it has no card).

        This is the single place that answers "which card was run", so that the
        copy kept beside the events cannot name a different file from the one
        the interface obeyed. Two sources, in order:

        1. the file handed to :meth:`import_command_file` -- literally the card
           the user edited for this run, wherever it happens to live. Every
           driver runs MadSpin this way;
        2. ``Cards/madspin_card.dat`` under ``event_base_dir``, the directory
           ``do_import`` derived from the event file -- for a session driven
           line by line that nonetheless runs inside a process directory.

        What it deliberately does not use is ``self.options['curr_dir']``,
        which is what the archiving used to be built from. ``curr_dir`` says
        where this run's *output* goes -- that is its meaning at every other
        use site, and what #365 pinned it to for the decayed events -- and
        ``post_set_ms_dir`` re-points it at the gridpack. So
        ``pjoin(curr_dir, 'Cards', 'madspin_card.dat')`` named the real card
        only when the card happened to say ``set ms_dir`` *before* importing
        the events, ``do_import`` then pointing curr_dir back at them. In the
        other ordering -- the one MadEvent always produces, since it imports
        the events in the constructor and reads the card afterwards -- it named
        ``<ms_dir>/Cards/madspin_card.dat``, which MadSpin never creates, and
        the archiving was skipped without a word. See
        tests/unit_tests/madspin, class TestMadSpinCardArchive.
        """
        if self.ms_card_path and os.path.exists(self.ms_card_path):
            return self.ms_card_path
        if self.event_base_dir:
            path = pjoin(self.event_base_dir, 'Cards', 'madspin_card.dat')
            if os.path.exists(path):
                return path
        return None

    def _archive_madspin_card(self, decayed_evt_file):
        """Keep the card that produced ``decayed_evt_file`` next to it.

        Shared by ``do_launch`` and ``run_from_pickle`` so that the gridpack
        path -- the one reached on every rerun against an existing ``ms_dir``,
        and hence the one where losing the card is most likely -- archives the
        same file, from the same source, as a fresh run.

        Returns the path written, or None when there was no card to copy.
        """
        ms_card_path = self.madspin_card_path
        if not ms_card_path:
            return None

        run_dir = os.path.realpath(os.path.dirname(decayed_evt_file))
        packed = os.path.exists(pjoin(run_dir, 'RunMaterial.tar.gz'))
        if packed:
            misc.call(['tar', '-xzpf', 'RunMaterial.tar.gz'], cwd=run_dir)
            base_path = pjoin(run_dir, 'RunMaterial')
        else:
            base_path = run_dir

        evt_name = os.path.basename(decayed_evt_file).replace('.lhe', '')
        ms_card_to_copy = pjoin(base_path, 'madspin_card_for_%s.dat' % evt_name)
        count = 0
        while os.path.exists(ms_card_to_copy):
            count += 1
            ms_card_to_copy = pjoin(base_path, 'madspin_card_for_%s_%d.dat' %
                                                              (evt_name, count))
        files.cp(str(ms_card_path), str(ms_card_to_copy))

        if packed:
            misc.call(['tar', '-czpf', 'RunMaterial.tar.gz', 'RunMaterial'],
                                                                    cwd=run_dir)
            shutil.rmtree(pjoin(run_dir, 'RunMaterial'))
        return ms_card_to_copy

    @misc.mute_logger()
    def do_launch(self, line):
        """end of the configuration launched the code"""

        (options, args) = self.parse_launch(line)
        if getattr(lhe_parser, "_ENABLE_LHE_TIMERS", False):
            lhe_parser.reset_lhe_timers()
        
        if options.name:
            self.me_run_name = options.name # Only use by MG5aMC
        else:
            self.me_run_name = ''

        try:
            if 'noborn' in self.banner.get_detail('proc_card', 'generate'):
                process_LI = True
            else:
                process_LI = False
        except: #this exception is added because the test 'test_hepmc_decay' does not present a proc_card. Maybe there is a way to have this information under this format ?
            logger.warning("The proc_card has not been found. It is unknown whether the process is at tree-level or loop-induced")
            logger.warning("The process is now considered as tree-level")
            process_LI = False

        # the legacy modes 'madspin_v1' and 'onshell_v1' are not compatible with loop-induced processes
        if self.options['spinmode'] in ['madspin_v1', 'onshell_v1'] and process_LI:
            raise ValueError("The MadSpin modes 'madspin_v1' and 'onshell_v1' are are not compatible with loop-induced processes. Please choose a mode among 'none', 'PA', 'madspin' or 'onshell'.")

        if self.options['onlyhelicity']:
            self.options['spinmode'] = 'madspin_v1'

        spinmode = self.options['spinmode']
        if spinmode == 'full':
            spinmode = 'madspin'
            self.options['spinmode'] = spinmode

        logger.info("Running MadSpin in spinmode %s" % spinmode)
        self._check_fixed_order_spinmode(spinmode)
        # Both of these are refused outside the density modes, so they are
        # checked before the branch rather than inside it. pure_interference
        # goes first: it is the more fundamental request of the two -- when it
        # is on, decay_output only chooses the shape of ITS output -- so a card
        # that gets both wrong should be told about pure_interference rather
        # than about the option that follows it. (_validate_pure_interference
        # returns immediately when the mode is off, and everything it touches
        # beyond the spinmode check is behind that guard.)
        self._validate_pure_interference()
        self._validate_weighted_decay()
        if self._density_spinmode():
            # read (and validate) the production polarisation braces now rather
            # than on the first event, deep inside a worker process
            self._production_polarization()
            self._polarization_weights_enabled()
        elif (self.options['keep_weight_for_polarization_vector']
              or self.options['keep_weight_for_polarization_fermion']):
            raise self.InvalidCmd(
                "keep_weight_for_polarization_vector/_fermion need a spin "
                "density matrix to restrict, so they are only available in the "
                "density spin modes (madspin/full, PA, onshell). Got "
                "spinmode=%s." % spinmode)
        # The density modes decide about the '@' grouping later, in run_onshell,
        # where the production events say how many of each particle an event
        # carries. These two never can, so say it now rather than after the
        # generation.
        if spinmode in ('none', 'onshell_v1'):
            self._warn_ignored_decay_groups(spinmode)

        # Before any mode is dispatched, and before anything cached in a reused
        # directory is read back: everything in there was computed from a
        # param_card, and none of it is re-measured on reuse.
        self._check_reused_param_card()

        if spinmode in ["none"]:
            out = self.run_bridge(line)
            self._log_lhe_timers()
            return out
        elif spinmode.startswith("onshell"):
            if spinmode == "onshell_v1":
                out = self.run_onshell(line)
            else:
                out = self.run_onshell(line, density_method=True)
            self._log_lhe_timers()
            return out
        elif spinmode == "PA":
            out = self.run_onshell(line, density_method=True)
            self._log_lhe_timers()
            return out
        elif spinmode == "madspin_v1":
            # legacy MadSpin / decay-chain path: fall through to decay_all_events below
            pass
        elif spinmode == "madspin":
            out = self.run_onshell(line, density_method=True)
            self._log_lhe_timers()
            return out
        elif spinmode == "bridge":
            raise Exception("Bridge mode not available.")
        else:
            raise Exception("spinmode %s not supported" % spinmode)
        
        if self.options['ms_dir'] and os.path.exists(pjoin(self.options['ms_dir'], 'madspin.pkl')):
            out = self.run_from_pickle()
            self._log_lhe_timers()
            return out
        
    
        args = self.split_arg(line)
        self.check_launch(args)
        for part in list(self.list_branches.keys()):
            if part in self.mg5cmd._multiparticles:
                if any(pid in self.final_state for pid in self.mg5cmd._multiparticles[part]):
                    break
            else:
                try:
                    pid = self.mg5cmd._curr_model.get('name2pdg')[part]
                except KeyError:
                    pid = self.mg5cmd._curr_model.get('name2pdg')[part.lower()]
                    self.list_branches[part.lower()] = self.list_branches[part]
                    del self.list_branches[part]
                    particle = self.mg5cmd._curr_model.get_particle(pid)
                    if particle.get('antiname').upper() in self.list_branches:
                        self.list_branches[particle.get('antiname').lower()] = \
                            self.list_branches[particle.get('antiname').upper()]
                        del self.list_branches[particle.get('antiname').upper()]
                if pid in self.final_state:
                    break
        else:
            if not self.options['onlyhelicity']:
                logger.info("Nothing to decay ...")
                return
        
        if self.options['BW_cut'] > 100:
            raise Exception("BW_cut parameter is much too large (>100) for narrow width approximation. Please set it up to a smaller value in your madspin_card.dat")

        model_line = self.banner.get('proc_card', 'full_model_line')

        if not self.options['seed']:
            self.options['seed'] = random.randint(0, int(30081*30081))
            #self.do_set('seed %s' % self.seed)
            logger.info('Will use seed %s' % self.options['seed'])
            self.history.insert(0, 'set seed %s' % self.options['seed'])

        if self.options['seed'] > 30081*30081: # can't use too big random number
            msg = 'Random seed too large ' + str(self.options['seed']) + ' > 30081*30081'
            raise Exception(msg)

        #self.options['seed'] = self.seed
        text = '%s\n' % '\n'.join([ line for line in self.history if line])
        self.banner.add_text('madspin' , text)
            
        time_me_generation = time.time()
        self.update_status('generating Madspin matrix element')
        generate_all = madspin.decay_all_events(self, self.banner, self.events_file, 
                                                    self.options)
        logger.info(f"Time for ME: {time.time()-time_me_generation:.2f} sec")        
        self.update_status('running MadSpin')
        generate_all.run()
                        
        self.branching_ratio = generate_all.branching_ratio
        self.cross = generate_all.cross
        self.error = generate_all.error
        self.efficiency = generate_all.efficiency
        try:
            self.err_branching_ratio = generate_all.err_branching_ratio
        except Exception:
            self.err_branching_ratio = 0
            
        evt_path = self.events_file.name
        try:
            self.events_file.close()
        except:
            pass
        misc.gzip(evt_path)
        decayed_evt_file=evt_path.replace('.lhe', '_decayed.lhe')
        # Ask the writer where it put the file rather than rebuilding the path
        # here: the two used to be spelled out separately and disagreed as soon
        # as ms_dir was set and curr_dir was not the ms_dir (see
        # decay_all_events.decayed_events_path).
        misc.gzip(generate_all.decayed_events_path, stdout=decayed_evt_file)
        if not self.mother:
            logger.info("Decayed events have been written in %s.gz" % decayed_evt_file)

        # Now arxiv the madspin card used (inside RunMaterial if present)
        self._archive_madspin_card(decayed_evt_file)
        self._log_lhe_timers()

    def run_from_pickle(self):
        import madgraph.iolibs.save_load_object as save_load_object
        
        generate_all = save_load_object.load_from_file(pjoin(self.options['ms_dir'], 'madspin.pkl'))
        
        #restore data passed to string to help pickle
        generate_all.all_decay = eval(generate_all.all_decay)
        for me in generate_all.all_ME:
            for d in generate_all.all_ME[me]['decays']:
                if isinstance(d['decay_struct'], str):
                    d['decay_struct'] = eval(d['decay_struct'])


        # Re-create information which are not save in the pickle.
        generate_all.evtfile = self.events_file
        generate_all.curr_event = madspin.Event(self.events_file, self.banner ) 
        generate_all.mgcmd = self.mg5cmd
        generate_all.mscmd = self 
        generate_all.pid2width = lambda pid: generate_all.banner.get('param_card', 'decay', abs(pid)).value
        generate_all.pid2mass = lambda pid: generate_all.banner.get('param_card', 'mass', abs(pid)).value
        if generate_all.path_me != self.options['ms_dir']:
            for decay in generate_all.all_ME.values():
                decay['path'] = decay['path'].replace(generate_all.path_me, self.options['ms_dir'])
                for decay2 in decay['decays']:
                    if decay2['path']: 
                        decay2['path'] = decay2['path'].replace(generate_all.path_me, self.options['ms_dir'])
            generate_all.path_me = self.options['ms_dir'] # directory can have been move
            generate_all.ms_dir = generate_all.path_me
        
        if not hasattr(self.banner, 'param_card'):
            self.banner.charge_card('slha')
        
        # Special treatment for the mssm. Convert the param_card to the correct
        # format
        if self.banner.get('model').startswith('mssm-') or self.banner.get('model')=='mssm':
            self.banner.param_card = check_param_card.convert_to_mg5card(\
                    self.banner.param_card, writting=False)
            
        for name, block in self.banner.param_card.items():
            if name.startswith('decay'):
                continue
                        
            orig_block = generate_all.banner.param_card[name]
            if block != orig_block:                
                raise Exception("""The directory %s is specific to a mass spectrum. 
                Your event file is not compatible with this one. (Different param_card: %s different)
                orig block:
                %s
                new block:
                %s""" \
                % (self.options['ms_dir'], name, orig_block, block))

        #replace init information
        generate_all.banner['init'] = self.banner['init']

        #replace run card if present in header (to make sure correct random seed is recorded in output file)
        if 'mgruncard' in self.banner:
            generate_all.banner['mgruncard'] = self.banner['mgruncard']   
        
        # NOW we have all the information available for RUNNING
        
        if self.options['seed']:
            #seed is specified need to use that one:
            open(pjoin(self.options['ms_dir'],'seeds.dat'),'w').write('%s\n'%self.options['seed'])
            #remove all ranmar_state
            for name in misc.glob(pjoin('*', 'SubProcesses','*','ranmar_state.dat'), 
                                                        self.options['ms_dir']):
                os.remove(name)    
        
        generate_all.ending_run()
        self.branching_ratio = generate_all.branching_ratio
        self.cross = generate_all.cross
        self.error = generate_all.error
        self.efficiency = generate_all.efficiency
        try:
            self.err_branching_ratio = generate_all.err_branching_ratio
        except Exception:
            # might not be define in some gridpack mode
            self.err_branching_ratio = 0 
        evt_path = self.events_file.name
        try:
            self.events_file.close()
        except:
            pass
        misc.gzip(evt_path)
        decayed_evt_file=evt_path.replace('.lhe', '_decayed.lhe')
        # Same shared accessor as do_launch -- and this path is *only* reachable
        # with ms_dir set, so it was the one always exposed to the mismatch.
        misc.gzip(generate_all.decayed_events_path, stdout=decayed_evt_file)
        if not self.mother:
            logger.info("Decayed events have been written in %s.gz" % decayed_evt_file)

        # ... and the card goes with them here too. Rerunning against an
        # existing ms_dir is a *rerun*: it produces its own event file, from its
        # own card, and archived nothing at all before -- do_launch returns here
        # long before reaching its own copy of this call.
        self._archive_madspin_card(decayed_evt_file)
    
    

    def run_bridge(self, line):
        """Run the Bridge Algorithm"""
        
        # 1. Read the event file to check which decay to perform and the number
        #   of event to generate for each type of particle.
        # 2. Generate the events requested
        # 3. perform the merge of the events.
        #    if not enough events. re-generate the missing one.
        
        args = self.split_arg(line)


        asked_to_decay = set()
        for part in self.list_branches.keys():
            if part in self.mg5cmd._multiparticles:
                for pdg in self.mg5cmd._multiparticles[part]:
                    asked_to_decay.add(pdg)
            else:
                asked_to_decay.add(self.mg5cmd._curr_model.get('name2pdg')[part])

        #0. Define the path where to write the file
        self.path_me = os.path.realpath(self.options['curr_dir']) 
        if self.options['ms_dir']:
            self.path_me = os.path.realpath(self.options['ms_dir'])
            if not os.path.exists(self.path_me):
                os.mkdir(self.path_me) 
        else:
            # cleaning (force: previous run may have left read-only frozen gridpacks)
            for name in misc.glob("decay_*_*", self.path_me):
                _force_rmtree(name)

        if self.events_file:
            self.events_file.close()
            filename = self.events_file.name
        else:
            filename = self.inputfile

        if self.options['input_format'] == 'auto':
            if '.lhe' in filename :
                self.options['input_format']  = 'lhe'
            elif '.hepmc' in filename:
                self.options['input_format']  = 'hepmc'
            else:
                raise Exception("fail to recognized input format automatically")
                
        if self.options['input_format'] in ['lhe', 'lhe_no_banner']:
            orig_lhe = lhe_parser.EventFile(filename)
            if self.options['input_format'] == 'lhe_no_banner':
                orig_lhe.allow_empty_event = True
                
        elif self.options['input_format'] in ['hepmc']:
            import madgraph.various.hepmc_parser as hepmc_parser
            orig_lhe = hepmc_parser.HEPMC_EventFile(filename)
            orig_lhe.allow_empty_event = True
            logger.info("Parsing input event to know how many decay to generate. This can takes few minuts.")
        else:
            raise Exception
            
        to_decay = collections.defaultdict(int)
        nb_event = 0
 
        for event in orig_lhe:
            nb_event +=1
            for particle in event:
                if particle.status == 1 and particle.pdg in asked_to_decay:
                    # final state and tag as to decay
                    to_decay[particle.pdg] += 1
            if self.options['input_format'] == 'hepmc' and nb_event == 250:
                currpos = orig_lhe.tell()
                filesize = orig_lhe.getfilesize()
                for key in to_decay:
                    to_decay[key] *= 1.05 * filesize/ currpos 
                    # 1.05 to avoid accidental coincidence with nevents
                break

        # Handle the banner of the output file
        if not self.options['seed']:
            self.options['seed'] = random.randint(0, int(30081*30081))
            #self.do_set('seed %s' % self.seed)
            logger.info('Will use seed %s' % self.options['seed'])
            self.history.insert(0, 'set seed %s' % self.options['seed'])

        if self.options['seed'] > 30081*30081: # can't use too big random number
            msg = 'Random seed too large ' + str(self.options['seed']) + ' > 30081*30081'
            raise Exception(msg)

        #self.options['seed'] = self.options['seed']
        
        text = '%s\n' % '\n'.join([ line for line in self.history if line])
        self.banner.add_text('madspin' , text)


        # 2. Generate the events requested
        with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"], [50,50,50,50]):
            mg5 = self.mg5cmd
            if not self.model:
                modelpath = self.model.get('modelpath+restriction')
                mg5.exec_cmd("import model %s" % modelpath)      
            evt_decayfile = {} 
            for pdg, nb_needed in to_decay.items():
                #check if a splitting is needed
                if nb_needed == nb_event:
                    evt_decayfile[pdg] = self.generate_events(pdg, min(nb_needed,100000), mg5)
                elif nb_needed %  nb_event == 0:
                    nb_mult = nb_needed // nb_event
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    if name not in self.list_branches:
                        continue
                    elif len(self.list_branches[name]) == nb_mult:
                        evt_decayfile[pdg] = self.generate_events(pdg, min(nb_event,100000), mg5)
                    else:
                        evt_decayfile[pdg] = self.generate_events(pdg, min(nb_needed,100000), mg5, cumul=True)
                elif self.options['cross_section']:
                    #cross-section hard-coded -> allow 
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    
                    if name not in self.list_branches:
                        continue
                    else:
                        try:
                            evt_decayfile[pdg] = self.generate_events(pdg, min(nb_needed,100000), mg5, cumul=True)
                        except common_run_interface.ZeroResult:
                            logger.warning("Branching ratio is zero for this particle. Not decaying it")
                            del to_decay[pdg]                    
                else:
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    if name not in self.list_branches or len(self.list_branches[name]) == 0:
                        continue
                    #raise self.InvalidCmd("The bridge mode of MadSpin does not support event files where events do not *all* share the same set of final state particles to be decayed. One workaround is to force the final cross-section manually.")
                    if len(self.list_branches[name]) == 1:
                        evt_decayfile[pdg] = self.generate_events(pdg, min(nb_event,100000), mg5)
                    else:
                        evt_decayfile[pdg] = self.generate_events(pdg, min(nb_needed,100000), mg5, cumul=True)
                    
                     
        # Compute the branching ratio.
        if not self.options['cross_section']:
            br = 1
            multi_br = [ ]
            multi_totevt = 0
            for (pdg, event_files) in evt_decayfile.items():
                if not event_files:
                    continue
                totwidth = float(self.banner.get('param', 'decay', abs(pdg)).value)
                if to_decay[pdg] == nb_event:
                    # Exactly one particle of this type to decay by event
                    pwidth = sum([event_files[k].cross for k in event_files])
                    if pwidth > 1.01 * totwidth:
                        logger.critical("Branching ratio larger than one for %s " % pdg) 
                    br *= pwidth / totwidth
                elif to_decay[pdg] % nb_event == 0:
                    # More than one particle of this type to decay by event
                    # Need to check the number of event file to check if we have to 
                    # make separate type of decay or not.
                    nb_mult = to_decay[pdg] // nb_event
                    if nb_mult == len(event_files):
                        for k in event_files:
                            pwidth = event_files[k].cross
                            if pwidth > 1.01 * totwidth:
                                logger.critical("Branching ratio larger than one for %s " % pdg)                       
                            br *= pwidth / totwidth
                        br *= math.factorial(nb_mult)
                    else:
                        pwidth = sum(event_files[k].cross for k in event_files)
                        if pwidth > 1.01 * totwidth:
                            logger.critical("Branching ratio larger than one for %s " % pdg) 
                        br *= (pwidth / totwidth)**nb_mult
                else:
                    pwidth = sum([event_files[k].cross for k in event_files])        
                    multi_br.append(pwidth / totwidth) 
                    multi_totevt += to_decay[pdg] % nb_event
            if multi_br and multi_totevt % nb_event == 0:
                if all(misc.equal(br,multi_br[0], 2) for br in multi_br): 
                    logger.warning("not all event are decaying the same particle, this is only supported if each event have ONE decaying particle (not checked) and that all particles have the same BR")        
                else:
                    raise self.InvalidCmd("The bridge mode of MadSpin does not support event files where events do not *all* share the same set of final state particles to be decayed: [%s %s ] " %(multi_br, multi_totevt))
            elif multi_br:
                raise self.InvalidCmd("The bridge mode of MadSpin does not support event files where events do not *all* share the same set of final state particles to be decayed. (%s %s)" % (multi_br, multi_totevt))
        else:
            br = 1
        self.branching_ratio = br
        self.efficiency = 1
        try:
            self.cross, self.error = self.banner.get_cross(witherror=True)
        except:
            if self.options['input_format'] != 'lhe':
                self.cross, self.error = 0, 0
        self.cross *= br
        self.error *= br
        
        # modify the cross-section in the init block of the banner
        if not self.options['cross_section']:
            self.banner.scale_init_cross(self.branching_ratio)
        else:
            
            if self.options['input_format'] in ['lhe_no_banner','hepmc'] and 'init' not in self.banner:
                self.cross = sum(self.options['cross_section'].values())
                self.error = 0
                self.branching_ratio = 1
            else:  
                self.banner.modify_init_cross(self.options['cross_section'])
                new_cross, new_error =   self.banner.get_cross(witherror=True)
                self.branching_ratio = new_cross / self.cross
                self.cross = new_cross   
                self.error = new_error

        # 3. Merge the various file together.
        if self.options['input_format'] == 'hepmc':
            name = orig_lhe.name.replace('.hepmc', '_decayed.lhe')
            if not name.endswith('.gz'):
                name = '%s.gz' % name
            
            output_lhe = lhe_parser.EventFile(name, 'w')
        else:
            name = orig_lhe.name.replace('.lhe', '_decayed.lhe')
            if not name.endswith('.gz'):
                name = '%s.gz' % name
            output_lhe = lhe_parser.EventFile(name, 'w')
        try:
            self.banner.write(output_lhe, close_tag=False)
        except Exception:
            if self.options['input_format'] == 'lhe':
                raise
        
        # initialise object which store not use event due to wrong helicity
        bufferedEvents_decay = {}
        for pdg in evt_decayfile:
            bufferedEvents_decay[pdg] = [{}] * len(evt_decayfile[pdg])
        
        import time
        start = time.time()
        counter = 0
        orig_lhe.seek(0)

        for event in orig_lhe:
            if counter and counter % 100 == 0 and float(str(counter)[1:]) ==0:
                print("decaying event number %s [%s s]" % (counter, time.time()-start))
            counter +=1
            
            # use random order for particles to avoid systematics when more than 
            # one type of decay is asked.
            particles = [p for p in event if int(p.status) == 1.0]
            random.shuffle(particles)
            ids = [particle.pid for particle in particles]
            br = 1 #br for that particular events (for special/weighted case)
            hepmc_output = lhe_parser.Event() #for hepmc case: collect the decay particle
            for i,particle in enumerate(particles):
                # check if we need to decay the particle 
                if self.final_state and particle.pdg not in self.final_state:
                    continue # nothing to do for this particle
                if particle.pdg not in evt_decayfile:
                    continue # nothing to do for this particle
                
                # check how the decay need to be done
                nb_decay = len(evt_decayfile[particle.pdg])
                if nb_decay == 0:
                    continue #nothing to do for this particle
                if nb_decay == 1:
                    decay_file = evt_decayfile[particle.pdg][0]
                    decay_file_nb = 0
                elif ids.count(particle.pdg) == nb_decay:
                    decay_file = evt_decayfile[particle.pdg][ids[:i].count(particle.pdg)]
                    decay_file_nb = ids[:i].count(particle.pdg)
                else:
                    #need to select the file according to the associate cross-section
                    r = random.random()
                    tot = sum(evt_decayfile[particle.pdg][key].cross for key in evt_decayfile[particle.pdg])
                    r = r * tot
                    cumul = 0
                    for j,events in evt_decayfile[particle.pdg].items():
                        cumul += events.cross
                        if r <= cumul:
                            decay_file = events
                            decay_file_nb = j
                            break
                    else:
                        # security for numerical accuracy issue... (unlikely but better safe)
                        if (cumul-tot)/tot < 1e-5:
                            decay_file = events
                            decay_file_nb = j
                        else:
                            misc.sprint(j,cumul, events.cross, tot, (tot-cumul)/tot)
                            raise Exception
                
                if self.options['new_wgt'] == 'BR':
                    tot_width = float(self.banner.get('param', 'decay', abs(pdg)).value)
                    if tot_width:
                        br = decay_file.cross / tot_width
                # ok start the procedure
                if hasattr(particle,'helicity'):
                    helicity = particle.helicity
                else:
                    helicity = 9
                bufferedEvents = bufferedEvents_decay[particle.pdg][decay_file_nb]
                
                # now that we have the file to read. find the associate event
                # checks if we have one event in memory
                if helicity in bufferedEvents and bufferedEvents[helicity]:
                    decay = bufferedEvents[helicity].pop()
                else:
                    # read the event file up to completion
                    while 1:
                        try:
                            decay = next(decay_file)
                        except StopIteration:
                            # check how far we are
                            ratio = counter / nb_event 
                            needed = 1.05 * to_decay[particle.pdg] - counter
                            needed = min(100000, max(needed, 6000))
                            with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"], [50,50,50,50]):
                                new_file = self.generate_events(particle.pdg, needed, mg5, [decay_file_nb])
                            evt_decayfile[particle.pdg].update(new_file)
                            decay_file = evt_decayfile[particle.pdg][decay_file_nb]
                            continue

                        if helicity == decay[0].helicity or helicity==9 or \
                                            self.options["spinmode"] == "none":
                            break # use that event
                        # not valid event store it for later
                        if helicity not in bufferedEvents:
                            bufferedEvents[helicity] = [decay]
                        elif len(bufferedEvents[helicity]) < 200:
                            # only add to the buffering if the buffer is not too large
                            bufferedEvents[helicity].append(decay)
                # now that we have the event make the merge
                if self.options['input_format'] != 'hepmc':
                    particle.add_decay(decay)
                else:
                    if len(hepmc_output) == 0:
                        hepmc_output.append(lhe_parser.Particle(event=hepmc_output))
                        hepmc_output[0].color2 = 0
                        hepmc_output[0].status = -1
                        hepmc_output.nexternal+=1
                    decayed_particle = lhe_parser.Particle(particle, hepmc_output)
                    decayed_particle.mother1 = hepmc_output[0]
                    decayed_particle.mother2 = hepmc_output[0]
                    hepmc_output.append(decayed_particle)
                    hepmc_output.nexternal+=1
                    decayed_particle.add_decay(decay)
            # change the weight associate to the event
            if self.options['new_wgt'] == 'cross-section':
                event.wgt *= self.branching_ratio
                br = self.branching_ratio
            else:
                event.wgt *= br
                
            if self.options['input_format'] != 'hepmc':
                wgts = event.parse_reweight()
                for key in wgts:
                    wgts[key] *= br
                # all particle have been decay if needed
                output_lhe.write(str(event))
            else:
                hepmc_output.wgt = event.wgt
                hepmc_output.nexternal = len(hepmc_output) # the append does not update nexternal
                output_lhe.write(str(hepmc_output))
        else:
            if counter==0:
                raise Exception
        output_lhe.write('</LesHouchesEvents>\n')        
                    
    
    def load_model(self, name, use_mg_default, complex_mass=False):
        """load the model"""
        
        loop = False
        #if (name.startswith('loop_')):
        #    logger.info("The model in the banner is %s" % name)
        #    logger.info("Set the model to %s since only" % name[:5])
        #    logger.info("tree-level amplitudes are used for the decay ")
        #    name = name[5:]
        #    self.banner.proc_card.info['full_model_line'].replace('loop_','')

        logger.info('detected model: %s. Loading...' % name)
        model_path = name
        #base_model = import_ufo.import_model(model_path)

        # Import model
        base_model = import_ufo.import_model(name, decay=True,
                                               complex_mass_scheme=complex_mass)

        if use_mg_default:
            base_model.pass_particles_name_in_mg_default()
        
        self.model = base_model
        self.mg5cmd._curr_model = self.model
        self.mg5cmd.process_model()

    # File in which a decay directory records the partial width that was
    # measured when it was built. Only the gridpack (ms_dir) path needs it: the
    # native path regenerates -- and so re-measures -- on every run, while a
    # gridpack is built once and merely *run* by every later run.
    PARTIAL_WIDTH_FILE = 'ms_partial_width.dat'

    # Copy of the param_card a reusable directory was built with. Everything
    # that directory caches is derived from those parameters, so this is what
    # says whether the cache may be reused at all -- see
    # _check_reused_param_card.
    PARAM_CARD_STAMP = 'ms_param_card.dat'

    def _reused_directory(self):
        """The directory whose already-built content this run would reuse, or
        None when everything is generated from scratch.

        ``ms_dir`` is the persistent one (decay gridpacks, max_wgt caches,
        madspin.pkl); ``use_old_dir`` reuses the same kind of material in the
        run's own working directory (production_me/all_ME.pkl)."""
        if self.options['ms_dir']:
            return os.path.realpath(self.options['ms_dir'])
        if self.options['use_old_dir'] and self.options['curr_dir']:
            return os.path.realpath(self.options['curr_dir'])
        return None

    # Everything a reused directory carries that was computed *from* a
    # param_card and is not re-measured on reuse. Not a general "is this
    # directory non-empty" test: these are the specific caches the read sites
    # go looking for -- ``run_from_pickle`` (madspin.pkl),
    # ``get_max_weight``/``_read_upfront_cache`` (max_wgt, max_wgt_sequential*,
    # pure_interference*), ``decay_all_events`` (production_me/all_ME.pkl under
    # use_old_dir) and the decay gridpacks themselves, whose measured partial
    # widths live in ``ms_partial_width.dat`` beside them.
    CACHED_MATERIAL = ('madspin.pkl', 'max_wgt*', 'pure_interference*',
                       'decay_*_*', pjoin('production_me', 'all_ME.pkl'))

    def _holds_cached_material(self, directory):
        """Whether ``directory`` already carries results of an earlier run.

        Distinguishes a directory that is merely *going* to be reused -- a
        fresh or empty ``ms_dir``, which is stamped on first use and is how a
        directory becomes protected at all -- from one that already holds a
        cache no stamp can vouch for. A false answer here is not cosmetic: it
        decides whether the stamp is written, so a cache this failed to notice
        would be authenticated with a card it may never have been built with.
        """
        return any(misc.glob(pattern, directory)
                   for pattern in self.CACHED_MATERIAL)

    def _check_reused_param_card(self):
        """Refuse to reuse a directory that was built with other parameters.

        A reused directory is a cache of things computed *from* the param_card:
        the decay events in ``decay_<pdg>_<i>`` and the partial widths measured
        while generating them, the maximum weights of the unweighting
        (``max_wgt``, ``max_wgt_sequential*``, ``pure_interference_c``), the
        branching ratios pickled in ``madspin.pkl``/``all_ME.pkl``, and the
        matrix-element directories themselves. None of those record which
        parameters produced them and none is re-measured on reuse, so a changed
        param_card would be applied to half the calculation and not the other
        half -- the reported cross-section computed from one card, the events
        from another.

        Rebuilding is not something MadSpin can do behind the user's back
        either: the whole point of ``ms_dir`` is that the expensive part is not
        rebuilt. So this stops, and says which blocks moved.

        ``run_from_pickle`` has long carried a narrower version of this check,
        comparing the pickled banner's blocks -- but skipping every ``decay``
        block, i.e. exactly the widths that drive the branching ratios and the
        Breit-Wigner sampling. Comparing the *card* rather than the pickled
        banner is what makes the widths comparable: MadSpin overwrites the
        banner's widths with its own LO estimates as it runs (madspin_v1), so
        the pickled banner is not the card that went in, while this stamp is.
        """
        reusing = self._reused_directory()
        # A run that is *not* reusing rebuilds the directory from scratch
        # (run_onshell/run_bridge remove decay_*_*, generate_all_matrix_element
        # removes the ME trees), so a stamp left there by an earlier run no
        # longer describes what is on disk. Keep it in step rather than letting
        # it stop the next reuse for a change that was in fact rebuilt.
        directory = reusing or (self.options['curr_dir'] and
                                os.path.realpath(self.options['curr_dir']))
        if not directory:
            return
        stamp = pjoin(directory, self.PARAM_CARD_STAMP)
        if not reusing and not os.path.exists(stamp):
            return  # nothing reuses this directory; do not leave a file behind
        # 'slha' first: Banner.__getattribute__ answers a missing param_card by
        # charging it, so hasattr() raises rather than returning False when the
        # input carries no banner at all (hepmc, lhe_no_banner).
        if 'slha' not in self.banner:
            return
        if not hasattr(self.banner, 'param_card'):
            self.banner.charge_card('param_card')
        current = self.banner.param_card
        if reusing and os.path.exists(stamp):
            try:
                previous = check_param_card.ParamCard(stamp)
            except Exception as error:
                previous = None
                logger.debug('unreadable %s (%s)', stamp, error)
            if not previous:
                # A stamp that parses to nothing says nothing. Every block
                # would "differ", which would turn a safety net into a gate
                # that stops runs that were always fine.
                logger.debug('%s carries no parameters; not checking the '
                             'reused ones', stamp)
                return
            # Only blocks both cards have: a block that exists on one side only
            # is a change of *model*, which the proc_card of the banner and
            # 'import model' already police, and flagging it here would stop
            # reuse across MadSpin versions that write one more block.
            changed = [name for name in set(current) & set(previous)
                       if name != 'qnumbers'  # model structure, not parameters
                       and current[name] != previous[name]]
            if changed:
                raise MadSpinStaleParameters(
                    "MadSpin is reusing %s, which was built with a different "
                    "param_card.\n"
                    "\n"
                    "Blocks that differ: %s\n"
                    "\n"
                    "That directory caches everything MadSpin computes from "
                    "the parameters -- the decay events and the partial widths "
                    "measured while generating them, the maximum weights of "
                    "the unweighting, the branching ratios in madspin.pkl. "
                    "None of it is re-measured on reuse, so continuing would "
                    "decay the events with the old parameters while reporting "
                    "a cross-section computed from the new ones.\n"
                    "\n"
                    "Point 'ms_dir'/'use_old_dir' at a fresh directory (or "
                    "remove %s) and rerun, so everything is rebuilt with the "
                    "parameters you asked for."
                    % (directory, ', '.join(sorted(changed)), directory))
            return
        if reusing and self._holds_cached_material(directory):
            # Built by a MadSpin that left no stamp. The content is reused
            # either way -- ``run_from_pickle`` still compares the pickled
            # banner's non-decay blocks -- but nothing can vouch for the widths.
            #
            # And *return* rather than falling through to write the stamp. The
            # absence of a stamp means "unknown", not "matches this run": a
            # stamp written here would be a claim MadSpin cannot support, and
            # the next run with this same card would match it and reuse the
            # cache in silence -- exactly the corruption this check exists to
            # stop, at the one moment it is weakest, the upgrade boundary where
            # every unstamped directory lives.
            #
            # Left unstamped rather than refused. The stamp is missing because
            # the directory predates it, not because anything is known to be
            # wrong, and the great majority of them are consistent; refusing
            # would break every existing ms_dir on upgrade. So the directory
            # stays usable and the warning comes back on every reuse, which is
            # the honest state -- nothing on disk will ever learn what that
            # cache was built with. What does earn a stamp is a directory with
            # no cache left in it: the fresh one the warning below asks for, or
            # this one once its cached material is cleared out.
            logger.warning(
                "%s was built by a MadSpin that did not record its "
                "param_card, so the parameters it was built with cannot be "
                "checked against this run's. If you have changed the "
                "param_card since, use a fresh directory: the cached decay "
                "events, partial widths and maximum weights are not "
                "re-measured on reuse.", directory)
            return
        # First use of this directory, or a rebuild of one already stamped:
        # record what it is being built with.
        try:
            if not os.path.isdir(directory):
                os.makedirs(directory)
            current.write(stamp)
        except (IOError, OSError) as error:
            logger.debug('could not record the parameters of %s: %s',
                         directory, error)

    @classmethod
    def _store_partial_width(cls, decay_dir, cross):
        """Record the measured partial width of ``decay_dir`` next to its
        gridpack. Best effort: a read-only ms_dir must not abort a healthy
        generation, the reader falls back to the gridpack's own banner."""
        try:
            with open(pjoin(decay_dir, cls.PARTIAL_WIDTH_FILE), 'w') as fsock:
                fsock.write('%.16e\n' % float(cross))
        except (IOError, OSError, TypeError, ValueError) as error:
            logger.debug('could not store the partial width of %s: %s',
                         decay_dir, error)

    @classmethod
    def _load_partial_width(cls, decay_dir, evt_file=None):
        """The partial width of a decay directory that this run did not build.

        Two sources, in order:

        1. the value the run that built the gridpack measured and stored
           (:meth:`_store_partial_width`) -- the same number that run used, so
           reusing an ms_dir reproduces its branching ratio exactly;
        2. failing that (an ms_dir built by an older version, which stored
           nothing), the cross-section in the <init> block of the events the
           gridpack has just produced. It is the same quantity measured on this
           run's sample rather than on the grid-setup one, so it agrees to the
           Monte Carlo error rather than to the last bit -- worth a warning, but
           far better than the alternative.

        Both failing is not recoverable, and must not be papered over with a
        default: every "neutral" value here is a wrong branching ratio, and a
        wrong branching ratio is a silently wrong cross-section in a
        well-formed file. So it raises.
        """
        path = pjoin(decay_dir, cls.PARTIAL_WIDTH_FILE)
        if os.path.exists(path):
            try:
                value = float(open(path).read().split()[0])
            except (IOError, OSError, IndexError, ValueError) as error:
                logger.warning('unreadable %s (%s), falling back to the '
                               'generated events', path, error)
            else:
                if math.isfinite(value) and value > 0:
                    return value
                logger.warning('%s holds a non-physical partial width (%s), '
                               'falling back to the generated events',
                               path, value)
        value = None
        if evt_file is not None:
            try:
                value = float(evt_file.cross)
            except Exception as error:
                logger.debug('no cross-section in the events of %s: %s',
                             decay_dir, error)
        if value is None or not math.isfinite(value) or value <= 0:
            raise MadSpinUnknownPartialWidth(
                "MadSpin cannot recover the partial width of %s.\n"
                "\n"
                "The branching ratio MadSpin applies to every event (and to "
                "the <init> block) is built from the partial width measured "
                "when each decay directory was generated. This directory was "
                "generated by an earlier run -- it is being reused through "
                "'ms_dir'/'use_old_dir' -- and neither the record that run "
                "should have left (%s) nor the cross-section of the events "
                "its gridpack just produced can be read.\n"
                "\n"
                "MadSpin stops here rather than continue with a branching "
                "ratio it cannot compute: that would write a perfectly "
                "well-formed event file in which every weight is wrong.\n"
                "\n"
                "Remove the 'ms_dir' directory (or point 'ms_dir' at a fresh "
                "one) and rerun: a directory generated from scratch measures "
                "the partial widths itself."
                % (decay_dir, cls.PARTIAL_WIDTH_FILE))
        logger.warning('%s predates the partial-width record; using the '
                       'cross-section of its generated events (%s) instead. '
                       'The branching ratio will agree with the run that built '
                       'it to the Monte Carlo error, not exactly.',
                       decay_dir, value)
        return value

    def generate_events(self, pdg, nb_event, mg5, restrict_file=None, cumul=False,
                        output_width=False, run_name='run_01'):
        """generate new events for this particle
           restrict_file allow to only generate a subset of the definition
           cumul allow to merge all the definition in one run (add process)
                 to generate events according to cross-section
           run_name allow a refill to write to its own run directory instead of
                 overwriting the pool that is currently being read
        """
        if not hasattr(self, 'me_int'):
            self.me_int = {}
            
        
        
        nb_event = int(nb_event) # in case of hepmc request the nb_event is not an integer
        # Gridpack-based decay generation is only used when we persist a gridpack
        # across runs (ms_dir). Building/packaging a gridpack and then generating
        # through run.sh is markedly slower (and uses the cores less densely)
        # than the direct MadEventCmdShell generation below, so the parallel
        # unweighting (nb_core>1) does NOT use it: the parent pre-generates the
        # whole decay pool here, once, with the fast native path.
        use_gridpack = bool(self.options['ms_dir'])
        if cumul:
            width = 0.
        else:
            width = 1.
        # channel number -> its own partial width. ``width`` above is the product
        # (or the sum, under cumul) that the branching ratio has always been
        # built from; the grouping needs each channel on its own, since a group
        # takes one channel per particle and its rate is the product over *its*
        # channels only.
        channel_widths = {}
        part = self.model.get_particle(pdg)
        if not part:
            return {}# this particle is not defined in the current model so ignore it
        name = part.get_name()
        out = {}
        time_gen_dec = time.time()
        logger.info("generate %s decay event for particle %s" % (int(nb_event), name))
        if name not in self.list_branches:
            return out
        for i,tagged_proc in enumerate(self.list_branches[name]):
            # the '@' grouping tag is MadSpin's own bookkeeping: MG5 would read
            # it as a process number (and the decay-ME generation appends one of
            # its own), so it never reaches the generation
            proc = self._split_group_tag(tagged_proc)[0]
            if restrict_file and i not in restrict_file:
                continue
            decay_dir = pjoin(self.path_me, "decay_%s_%s" %(str(pdg).replace("-","x"),i))
            if not os.path.exists(decay_dir):
                if cumul:
                    mg5.exec_cmd("generate %s" % proc)
                    for j,proc2 in enumerate(self.list_branches[name][1:]):
                        if restrict_file and j not in restrict_file:
                            raise Exception # Do not see how this can happen
                        mg5.exec_cmd("add process %s"
                                     % self._split_group_tag(proc2)[0])
                    mg5.exec_cmd("output %s -f" % decay_dir)
                else:
                    mg5.exec_cmd("generate %s" % proc)
                    mg5.exec_cmd("output %s -f" % decay_dir)
                
                options = dict(mg5.options)
                if use_gridpack:
                    # gridpack mode -> build the integration grid once here
                    if decay_dir in self.me_int:
                        me5_cmd = self.me_int[decay_dir]
                    else:
                        me5_cmd = madevent_interface.MadEventCmdShell(me_dir=os.path.realpath(\
                                                decay_dir), options=options)
                        me5_cmd.options["automatic_html_opening"] = False
                        me5_cmd.options["madanalysis5_path"] = None
                        me5_cmd.options["madanalysis_path"] = None
                        me5_cmd.allow_notification_center = False
                        try:
                            os.remove(pjoin(decay_dir, 'Cards', 'madanalysis5_parton_card_default.dat'))
                            os.remove(pjoin(decay_dir, 'Cards', 'madanalysis5_parton_card.dat'))
                        except Exception as error:
                            logger.debug(error)
                            pass 
                        self.me_int[decay_dir] = me5_cmd

                    if self.options["run_card"]:
                        run_card = self.run_card
                    else:
                        run_card = banner.RunCard(pjoin(decay_dir, "Cards", "run_card.dat"))                        
                    run_card["iseed"] = self.options['seed']
                    run_card['gridpack'] = True
                    run_card['systematics_program'] = 'False'
                    run_card['use_syst'] = False
                    run_card.__setitem__('allow_overshoot_events', True, change_userdefine=True)
                    run_card.__setitem__('refine_evt_by_job', 5000, change_userdefine=True)
                    run_card.write(pjoin(decay_dir, "Cards", "run_card.dat"))
                    param_card = self.banner['slha']
                    open(pjoin(decay_dir, "Cards", "param_card.dat"),"w").write(param_card)
                    self.options['seed'] += 1
                    self.seed = self.options['seed'] 
                    # actually creation
                    me5_cmd.exec_cmd("generate_events run_01 -f")
                    # The partial width of this channel is measured HERE and
                    # only here on the gridpack path -- a later run that finds
                    # ``decay_dir`` already built skips this whole block. Store
                    # it beside the gridpack so that run can read it back
                    # (_load_partial_width); without it the branching ratio of
                    # every ms_dir-reusing run silently collapses to 0.
                    self._store_partial_width(decay_dir,
                                              me5_cmd.results.current['cross'])
                    if output_width:
                        channel_widths[i] = me5_cmd.results.current['cross']
                        if cumul:
                            width += me5_cmd.results.current['cross']
                        else:
                            width *= me5_cmd.results.current['cross']
                    me5_cmd.exec_cmd("exit")
                    #remove pointless informat
                    if not os.path.exists(pjoin(decay_dir, 'run.sh')):
                        devnull = open('/dev/null','w')
                        misc.call(["rm", "Cards", "bin", 'Source', 'SubProcesses'], cwd=decay_dir,stdout=devnull, stderr=-2)
                        misc.call(['tar', '-xzpvf', 'run_01_gridpack.tar.gz'], cwd=decay_dir,stdout=devnull, stderr=-2)
                        devnull.close()
            # Now generate the events
            if not use_gridpack:
                if decay_dir in self.me_int:
                        me5_cmd = self.me_int[decay_dir]
                else:
                    # ``_gen_nb_core`` caps the cores this generation may use, so
                    # that several decay generations running at the same time
                    # (see _generate_decays) never oversubscribe the machine.
                    gen_options = dict(mg5.options)
                    if getattr(self, '_gen_nb_core', None):
                        gen_options['nb_core'] = self._gen_nb_core
                    me5_cmd = madevent_interface.MadEventCmdShell(me_dir=os.path.realpath(\
                                                    decay_dir), options=gen_options)
                    me5_cmd.options["automatic_html_opening"] = False
                    me5_cmd.options["automatic_html_opening"] = False
                    me5_cmd.options["madanalysis5_path"] = None
                    me5_cmd.options["madanalysis_path"] = None
                    me5_cmd.allow_notification_center = False
                    try:
                        os.remove(pjoin(decay_dir, 'Cards', 'madanalysis5_parton_card_default.dat'))
                        os.remove(pjoin(decay_dir, 'Cards', 'madanalysis5_parton_card.dat'))
                    except Exception as error:
                        logger.debug(error)
                        pass                 
                    self.me_int[decay_dir] = me5_cmd
                if self.options["run_card"]:
                    if hasattr(self, 'run_card'):
                        run_card = self.run_card
                    elif hasattr(self.options, 'run_card'):
                        run_card = self.options.run_card
                    else:
                        self.run_card = banner.RunCard(self.options["run_card"])
                        run_card = self.run_card 
                else:
                    run_card = banner.RunCard(pjoin(decay_dir, "Cards", "run_card.dat"))
                run_card["nevents"] = int(0.8*nb_event)
                run_card.__setitem__('allow_overshoot_events', True, change_userdefine=True)
                run_card.__setitem__('refine_evt_by_job', 5000, change_userdefine=True)
                # Handle the banner of the output file
                if not self.seed:
                    self.seed = random.randint(0, int(30081*30081))
                    self.do_set('seed %s' % self.seed)
                    logger.info('Will use seed %s' % self.seed)
                    self.history.insert(0, 'set seed %s' % self.seed)
                run_card["iseed"] = self.seed
                run_card["systematics_program"] = 'None'
                run_card['use_syst'] = False
                # Under the parallel unweighting, have the final unweighting hand
                # us one file per worker instead of a single pool: each worker
                # then reads only its own file. And do not gzip them -- they are
                # read back immediately, compressing them is pure overhead.
                nb_split = self._decay_pool_split()
                if nb_split > 1:
                    run_card.__setitem__('nb_unweight_output', nb_split,
                                         change_userdefine=True)
                    run_card.__setitem__('zip_unweighted_events', False,
                                         change_userdefine=True)
                run_card.write(pjoin(decay_dir, "Cards", "run_card.dat"))
                param_card = self.banner['slha']
                open(pjoin(decay_dir, "Cards", "param_card.dat"),"w").write(param_card)
                self.seed += 1
                me5_cmd.exec_cmd("generate_events %s -f" % run_name)
                if output_width:
                    channel_widths[i] = me5_cmd.results.current['cross']
                    if cumul:
                        width += me5_cmd.results.current['cross']
                    else:
                        width *= me5_cmd.results.current['cross']
                if run_card["nevents"] > 1.01 * me5_cmd.results.current['nb_event']:
                    logger.critical('The number of event generated is only %s/%s. This typically indicates that you need specify cut on the decay process.',me5_cmd.results.current['nb_event'], run_card["nevents"])
                    logger.critical('We strongly suggest that you cancel/discard this run.')
                me5_cmd.exec_cmd("exit")
                if nb_split > 1:
                    out[i] = _ChainedEvents(lhe_parser.EventFile.unweight_output_paths(
                        pjoin(decay_dir, "Events", run_name, 'unweighted_events.lhe'),
                        nb_split))
                else:
                    out[i] = lhe_parser.EventFile(pjoin(decay_dir, "Events", run_name, 'unweighted_events.lhe.gz'))
            else:
                if not self.seed:
                    if hasattr(self, 'mother'):
                        try:
                            self.seed = 100 + self.mother.run_card['iseed']
                        except:
                            self.seed = random.randint(0, int(30081*30081))
                self.seed += 1
                if self.seed > 30081*30081:
                    self.seed -= 30081*30081
                logger.info('Will use seed %s' % (self.seed))
                rc, log = self._run_gridpack(
                    [pjoin(decay_dir, 'run.sh'), str(int(1.2*nb_event)),
                     str(self.seed), '-p', str(self._resolve_nb_core())],
                    cwd=decay_dir)
                events_path = pjoin(decay_dir, 'events.lhe.gz')
                if not os.path.exists(events_path):
                    raise Exception(
                        "Gridpack decay generation failed (rc=%s): %s was "
                        "not produced by run.sh/gridrun.\n"
                        "--- last run.sh/gridrun output ---\n%s"
                        % (rc, events_path, log))
                out[i] = lhe_parser.EventFile(events_path)
                if output_width and i not in channel_widths:
                    # ms_dir reuse: the gridpack was built by an earlier run, so
                    # the block above (the only place the gridpack path measures
                    # a partial width) did not run. Recover the width that run
                    # measured instead of leaving the accumulator at its neutral
                    # value -- which is 0 under ``cumul``, and would zero the
                    # branching ratio, the <init> block and every event weight.
                    measured = self._load_partial_width(decay_dir, out[i])
                    channel_widths[i] = measured
                    if cumul:
                        width += measured
                    else:
                        width *= measured
            if cumul:
                break
        time_gen_dec = time.time()-time_gen_dec
        logger.info(f"Time for decay event generation = {time_gen_dec:.1f} sec")
        if not output_width:
            return out
        else:
            return out, width, channel_widths

    def run_onshell(self, line, density_method=False):
        """Run the onshell Algorithm"""
        
        # 1. Read the event file to check which decay to perform and the number
        #   of event to generate for each type of particle. (assume efficiency=1 for spin 0
        #   otherwise efficiency=2
        # 2. Generate the associated events
        # 3. generate the various matrix-element (production/decay/production+decay) 
        #    => no production+decay if density_method on True
        # 4. determine the maxwgt
        # 5. generate the decay (for each production event)
        # 6. perform the merge of the events.
        #    if not enough events. re-generate the missing one.
        
        # Spyros: this is not used - remove?
        args = self.split_arg(line)

        # First define an utility function for generating events when needed
        # Spyros what should be done here? 

        # Find which particles should be decayed
        asked_to_decay = set()
        for part in self.list_branches.keys():
            if part in self.mg5cmd._multiparticles:
                for pdg in self.mg5cmd._multiparticles[part]:
                    asked_to_decay.add(pdg)
            else:
                asked_to_decay.add(self.mg5cmd._curr_model.get('name2pdg')[part])

        # 0. Define the path where to write the file
        self.path_me = os.path.realpath(self.options['curr_dir']) 
        if self.options['ms_dir']:
            self.path_me = os.path.realpath(self.options['ms_dir'])
            if not os.path.exists(self.path_me):
                os.mkdir(self.path_me) 
        else:
            # cleaning (force: previous run may have left read-only frozen gridpacks)
            for name in misc.glob("decay_*_*", self.path_me):
                _force_rmtree(name)

        self.events_file.close()
        if self.events_file.name.endswith('.gz'):
            misc.gunzip(self.events_file.name)
        orig_lhe = lhe_parser.EventFile(self.events_file.name)
        if self.options['fixed_order']:
            orig_lhe.eventgroup = True

        # Dictionary with particle properties
        decay_dict = {}
        
        # 1. Open input event file and check which particles to decay
        # - count the number of particles to be decayed.
        to_decay = collections.defaultdict(int)
        nb_event = 0
        # keep_weight_for_polarization_*: the set of topologies (final-state
        # pdgs to be decayed, in production order) the file holds. The
        # combinations -- hence the weight ids -- depend on it, and the banner
        # is written before the first event is decayed, so it is collected here
        # rather than discovered event by event. Only built when the option is
        # on, so an unset option does not even allocate.
        pol_weights = self._polarization_weights_enabled()
        pol_layouts = set()
        nb_decaying = 0
        # Breit-Wigner truncation: the product, over the resonances *this event*
        # has a virtuality drawn for, of the fraction of each one's
        # Breit-Wigner that the BW_cut window keeps -- summed here and averaged
        # below. The multiplicity can differ event to event (and the mixed-pdg
        # case below equalizes by dropping events), so the mean over the file is
        # the only thing that can be folded into one banner cross-section; a
        # per-event factor would turn an unweighted sample into a weighted one.
        bw_trunc_per_pdg = {}
        bw_trunc_sum = 0.0
        bw_trunc_active = self._spinmode_draws_virtuality()
        for event in orig_lhe:
            if self.options['fixed_order']:
                event = event[0]
            nb_event +=1
            pol_sequence = [] if pol_weights else None
            nb_this_event = 0
            nb_prod_final = 0
            event_trunc = 1.0
            for particle in event:
                if particle.status != 1:
                    continue
                nb_prod_final += 1
                if particle.pdg in asked_to_decay:
                    # final state and tag as to decay
                    to_decay[particle.pdg] += 1
                    if pol_weights:
                        pol_sequence.append(particle.pdg)
                    nb_this_event += 1
                    # Properties of decaying particle
                    width = self.banner.get('param_card', 'decay', abs(particle.pdg)).value
                    mass = self.banner.get('param_card', 'mass', abs(particle.pdg)).value
                    color = self.model.get_particle(particle.pdg).get('color')
                    spin = self.model.get_particle(particle.pdg).get('spin')
                    decay_dict[particle.pdg] = [width, mass, color, spin]
                    if bw_trunc_active:
                        if particle.pdg not in bw_trunc_per_pdg:
                            # Same guard as the gen_jobs loop below: a pdg that
                            # reached asked_to_decay through a multiparticle but
                            # has no branch of its own is never generated for,
                            # never decayed, and never has a virtuality drawn.
                            name = self.model.get_particle(particle.pdg).get_name()
                            bw_trunc_per_pdg[particle.pdg] = (
                                madspin.bw_retained_fraction(
                                    mass, width, self._resolved_bw_cut())
                                if name in self.list_branches else 1.0)
                        event_trunc *= bw_trunc_per_pdg[particle.pdg]
            if pol_weights:
                pol_layouts.add(tuple(pol_sequence))
            if nb_this_event > nb_decaying:
                nb_decaying = nb_this_event
            # 2 -> 1 production: sqrt(shat) fixes the single resonance's
            # virtuality, so get_onshell_evt_and_wgt draws nothing (same guard,
            # ``nb_prod_final > 1``) and there is nothing to correct.
            bw_trunc_sum += event_trunc if nb_prod_final > 1 else 1.0
        self._pol_event_layouts = pol_layouts
        # Only the *top-level* virtualities appear here, and that is the whole
        # list: for `t > w+ b, w+ > l+ vl` MadSpin never redraws the W. Its
        # virtuality comes from the decay events MG5 generated in decay_*_*,
        # which are only boosted and rotated afterwards (rotateboost_decay), so
        # its window is that generation's own run_card ``bwcutoff`` -- and the
        # truncation it causes is already inside the partial width measured
        # there, which is the numerator of the branching ratio below. Correcting
        # it here as well would double-count it.
        self._bw_truncation = bw_trunc_sum / nb_event if nb_event else 1.0
        #print(f"to_decay = {to_decay}")
        # How many particles decay in one event -- the same multiplicity the
        # pool ladder counts. It decides which unweighting scheme 'auto' picks,
        # so it is resolved once here rather than per event: the modes have
        # different bounds, and a mode that changed event to event would be
        # testing against somebody else's.
        #
        # Counted *per event* and maximised, not rebuilt from the per-pdg
        # tally: a sample that mixes subprocesses carrying different decaying
        # pdgs -- `p p > w+ j` together with `p p > w- j` -- decays exactly one
        # particle per event, but lists two pdgs, and floor-averaging each of
        # them to at least one reported two decaying particles. That over-count
        # is what pushed `p p > w+/- j` onto a staged offshell scheme, which is
        # precisely the case whose mass-set weight carries
        # Tr(rho_off)/|M_prod|^2_on over orders of magnitude and that no bound
        # covers (see _unweighting_mode): the acceptance test measured 1.8e4
        # for the mass bound and 18e6 mass sets for 1000 events.
        self._nb_decaying = nb_decaying
                	
        with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"], [50,50,50,50]):
            mg5 = self.mg5cmd
            if not self.model:
                modelpath = self.model.get('modelpath+restriction')
                mg5.exec_cmd("import model %s" % modelpath)  
                self.model = mg5._curr_model

        # Handle the banner of the output file
        if not self.seed:
            self.seed = random.randint(0, int(30081*30081))
            self.do_set('seed %s' % self.seed)
            logger.info('Will use seed %s' % self.seed)
            self.history.insert(0, 'set seed %s' % self.seed)

        if self.seed > 30081*30081: # can't use too big random number
            msg = 'Random seed too large ' + str(self.seed) + ' > 30081*30081'
            raise Exception(msg)

        self.options['seed'] = self.seed
        #print(f"from run onshell seed = {self.seed}")
        
        text = '%s\n' % '\n'.join([ line for line in self.history if line])
        self.banner.add_text('madspin' , text)


        # 2. Generate the events requested
        nevents_for_max = self.options['Nevents_for_max_weight']
        if nevents_for_max == 0 :
            nevents_for_max = 75
        nevents_for_max *= self.options['max_weight_ps_point']
        # Security margin on the decays reserved for the maximum-weight scan.
        # The scan draws exactly nevents_for_max decays *per slot* (measured: no
        # restarts for tops), but the parallel scan stripes the pool across the
        # workers, and dividing the bare reservation leaves each worker's slice
        # only just big enough -- an uneven split then exhausts one worker and
        # forces a mid-scan refill. A 50% margin absorbs that unevenness.
        nevents_for_max = int(1.5 * nevents_for_max)
        
        with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"], [50,50,50,50]):
            mg5 = self.mg5cmd
            if not self.model:
                modelpath = self.model.get('modelpath+restriction')
                mg5.exec_cmd("import model %s" % modelpath)      
            evt_decayfile = {}
            br = 1.
            # pdg -> br_pdg for the "mixed final-state" case (events do not
            # all share the same set of decaying particles). Filled in the
            # else-branch below and consumed after the loop to compute the
            # per-pdg drop probability that equalizes BRs across productions.
            mixed_pdgs_br = {}
            # 1) Decide what has to be generated for each decaying particle.
            #    Nothing is generated yet: the generations are launched together
            #    below so they overlap instead of running one particle after the
            #    other.
            gen_jobs = collections.OrderedDict()
            # '@' grouping tags. Resolved before anything is generated: it
            # decides how many pools each particle gets, and it forces the joint
            # accept/reject, which _sequential_pool_ladder below already needs to
            # know about.
            self._resolve_decay_groups(to_decay, nb_event, density_method)
            # How many decay events one production event burns per decaying
            # particle. The joint accept/reject redraws the whole set on a
            # reject, so every pool is consumed at the same rate; the sequential
            # one redraws a single particle, so each pool is consumed at its own
            # slot's rate -- the ladder of _decay_pool_ladder.
            seq_ladder = self._sequential_pool_ladder(to_decay, nb_event,
                                                      density_method)
            for pdg, nb_needed in to_decay.items():
                # muliply by expected effeciency of generation
                spin = self.model.get_particle(pdg).get('spin')
                if pdg in seq_ladder:
                    efficiency = seq_ladder[pdg]
                elif spin == 1:
                    efficiency = 1.1
                else:
                    efficiency = 2.0

                totwidth = self.banner.get('param_card', 'decay', abs(pdg)).value

                #check if a splitting is needed
                if pdg in self._decay_group_pdgs():
                    # Grouped: one pool per decay *line*, whatever the
                    # multiplicity -- a group hands each of its lines to one
                    # particle, so a line is never mixed with another one in a
                    # single pool the way `simple`/`mult_cumul` mix them.
                    #
                    # Every pool is sized as if its group were drawn on every
                    # event. The exact size would be that times the group's
                    # share p_g, but p_g is only known once the partial widths
                    # have been measured, i.e. once this generation has run. So
                    # over-generate by at most a factor |groups| rather than
                    # under-generate and pay for refills; `decay_event_mult`
                    # scales it down for anyone who minds.
                    gen_jobs[pdg] = {'kind': 'grouped', 'totwidth': totwidth,
                        'nb_mult': nb_needed // nb_event, 'cumul': False,
                        'nb_gen': (int(efficiency*nb_event) + nevents_for_max)
                                  * self.options['decay_event_mult']}
                elif nb_needed == nb_event:
                    gen_jobs[pdg] = {'kind': 'simple', 'totwidth': totwidth,
                        'cumul': True,
                        'nb_gen': (int(efficiency*nb_needed) + nevents_for_max)
                                  * self.options['decay_event_mult']}
                elif nb_needed %  nb_event == 0:
                    nb_mult = nb_needed // nb_event
                    nb_cumul = (int(efficiency*nb_needed) + nevents_for_max*nb_mult) \
                                * self.options['decay_event_mult']
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    if name not in self.list_branches:
                        continue
                    elif len(self.list_branches[name]) == nb_mult:
                        gen_jobs[pdg] = {'kind': 'mult_split', 'totwidth': totwidth,
                            'nb_mult': nb_mult, 'cumul': False,
                            'nb_gen': nb_event * self.options['decay_event_mult']}
                    else:
                        gen_jobs[pdg] = {'kind': 'mult_cumul', 'totwidth': totwidth,
                            'nb_mult': nb_mult, 'cumul': True, 'nb_gen': nb_cumul}
                else:
                    # Mixed case: events do not all share the same final-state
                    # particles to be decayed. We collect this pdg here and, once
                    # the loop is done, equalize BRs via the legacy
                    # add_loose_decay mechanism (drop events sampled to the
                    # "fake" decay channel so the output stays unweighted).
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    if name not in self.list_branches or len(self.list_branches[name]) == 0:
                        continue
                    gen_jobs[pdg] = {'kind': 'mixed', 'totwidth': totwidth,
                        'cumul': True,
                        'nb_gen': (int(efficiency*nb_needed) + nevents_for_max)
                                  * self.options['decay_event_mult']}

            # 2) Generate every particle's decay events at the same time.
            gen_results = self._generate_decays(gen_jobs, mg5)
            self._clear_refill_state()

            # 3) Fold the measured partial widths into the branching ratio.
            channel_widths = {}
            for pdg, job in gen_jobs.items():
                evt_decayfile[pdg], pwidth, channel_widths[pdg] = gen_results[pdg]
                totwidth = job['totwidth']
                if job['kind'] == 'grouped':
                    continue        # done together, below: a group's rate is a
                                    # product over several particles at once
                if job['kind'] == 'mult_split':
                    # ``pwidth`` is the *product* of the channels' widths here,
                    # so measuring it against a single total width is a category
                    # error -- it fires as soon as totwidth > 1 GeV (two decay
                    # lines for a top) and the clamp below would then quietly
                    # wreck the branching ratio. Check each channel instead.
                    product = 1.0
                    for i in sorted(channel_widths[pdg]):
                        product *= self._clamped_partial_width(
                            channel_widths[pdg][i], totwidth, pdg)
                    br *= (product / totwidth**job['nb_mult']
                           * self._assignment_multiplicity(
                               self.list_branches[self.model.get_particle(pdg)
                                                  .get_name()]))
                    continue
                pwidth = self._clamped_partial_width(pwidth, totwidth, pdg)
                if job['kind'] == 'simple':
                    br *= pwidth / totwidth
                elif job['kind'] == 'mult_cumul':
                    br *= (pwidth / totwidth)**job['nb_mult']
                else:
                    mixed_pdgs_br[pdg] = pwidth / totwidth

            # 3b) The grouped particles' branching ratio, and with it the
            #     probability of each group. A group is one complete assignment
            #     of channels to particles, so its rate is the product of its
            #     own partial widths over every grouped particle -- and the
            #     groups are alternatives, so they add.
            if getattr(self, '_decay_groups', None):
                br *= self._resolve_group_rates(gen_jobs, channel_widths)

        # Equalize branching ratios across mixed productions (legacy
        # add_loose_decay mechanism): pick max_br as the global BR factor and
        # drop, per event, with probability 1 - br_pdg / max_br so the output
        # sample stays unweighted. The banner cross-section is corrected after
        # the decay loop (below) once the actual number of kept events is
        # known.
        drop_prob_per_pdg = {}
        if mixed_pdgs_br:
            max_mixed_br = max(mixed_pdgs_br.values())
            br *= max_mixed_br
            for pdg, pdg_br in mixed_pdgs_br.items():
                drop_prob_per_pdg[pdg] = 1.0 - pdg_br / max_mixed_br
            if any(d > 1e-9 for d in drop_prob_per_pdg.values()):
                logger.warning(
                    "Mixed-pdg production processes have different total BRs "
                    "(per-pdg BR=%s, max=%g). Equalizing by dropping events; "
                    "the output sample stays unweighted and the banner "
                    "cross-section reflects the effective BR.",
                    {k: '%.4g' % v for k, v in mixed_pdgs_br.items()},
                    max_mixed_br,
                )
        mixed_pdgs_set = set(drop_prob_per_pdg.keys())

        # The rate MadSpin actually produces is only the part of each drawn
        # Breit-Wigner that fits inside the BW_cut window, while sigma_prod * BR
        # is the whole one. Fold the retained fraction in *here*, before
        # branching_ratio is read: it is the single number that reaches both the
        # <init> block (scale_init_cross, below) and every event weight
        # (_unweight_range), so the file stays self-consistent under the
        # IDWTUP = -4 convention that sigma is the mean weight. The later
        # rewrites -- BR equalization, decay_output = weighted -- multiply
        # branching_ratio again and compose with this by construction.
        bw_truncation = getattr(self, '_bw_truncation', 1.0)
        if bw_truncation != 1.0:
            logger.info(
                "Breit-Wigner truncation at BW_cut = %g keeps %.5g of the "
                "cross-section; the reported sigma is scaled by it.",
                self._resolved_bw_cut(), bw_truncation)
            br *= bw_truncation

        # Last chance to catch a branching ratio that would silently zero (or
        # NaN) every weight of a run that otherwise completes normally.
        self._check_branching_ratio(br, gen_jobs)

        self.branching_ratio = br
        self.efficiency = 1
        self.cross, self.error = self.banner.get_cross(witherror=True)
        self.cross *= self.branching_ratio
        self.error *= self.branching_ratio
        

        density_pole_approximation = self._density_pole_approximation()
        density_needs_reshuffle = self._density_needs_reshuffle(density_method)

        # 3. generate the various matrix-elements
        time_me_generation = time.time()
        self.update_status('generating Madspin matrix element (density_method=%s)' % density_method)
        if density_method:
            self.generate_all = madspin.decay_all_events_density(self, self.banner, self.events_file,self.options)
        else:
            self.generate_all = madspin.decay_all_events_onshell(self, self.banner, self.events_file,self.options)

        self.generate_all.compile()
        self.all_me = self.generate_all.all_me
        self.all_f2py = {}
        self.all_density = {}
        time_me_generation = time.time() - time_me_generation
        logger.info(f"Time ME generation: {time_me_generation:.2f} sec")         
	
	    #4. determine the maxwgt
        #print(f"Spyros decay file: {evt_decayfile}")
        # Sequential mode tests one decaying particle at a time and so needs one
        # bound per slot of the ordering; it falls back to the joint bound if the
        # probe cannot produce them (nothing decays, too few events to spread).
        sequential = self._sequential_active(density_method)
        maxwgts = []
        if sequential:
            maxwgts = self.get_sequential_maxwgt(orig_lhe, evt_decayfile)
            if not maxwgts:
                logger.info("MadSpin: no per-particle maximum weight could be "
                            "estimated, using the joint accept/reject")
                sequential = False
        maxwgt = None
        if not sequential:
            maxwgt = self.get_maxwgt_for_onshell(orig_lhe, evt_decayfile, decay_dict)

        #5. generate the decay (for each production event)
        # The per-event unweighting loop is embarrassingly parallel (events are
        # independent). It is factored into ``_unweight_range`` so it can run
        # either in-process (nb_core==1, byte-for-byte the historical path) or
        # across ``nb_core`` forked worker processes. Parallelism is process
        # level, NOT threads: the matrix-element f2py extension carries global
        # Fortran COMMON-block state and is not thread-safe; after ``fork`` each
        # worker owns an independent address-space copy of it.
        orig_lhe.seek(0)
        base_out = orig_lhe.name.replace('.lhe', '_decayed.lhe')
        # nb_event for the decay-pool refill sizing is the banner-declared count
        # (historical behaviour), not the physical number of events on disk.
        nb_event = orig_lhe.get_banner().run_card['nevents']

        nb_core = self._resolve_nb_core()
        nb_core = max(1, min(nb_core, int(nb_event) if nb_event else 1))

        ctx = dict(
            maxwgt=maxwgt,
            maxwgts=maxwgts,
            # pure interference: the decay-side constant every written weight is
            # divided by, measured by the same scan that produced ``maxwgt``
            pure_interference_c=getattr(self, '_pi_c', None),
            # ... and <|W|>, which normalises the 'unweighted' output. Unused
            # by the fully weighted default.
            pure_interference_absw=getattr(self, '_pi_absw', None),
            pure_interference_unweighted=self._pure_interference_unweighted(),
            # decay_output = weighted: the same 'keep every trial and put W/c
            # on the weight' path, for an ordinary (non-interference) run
            weighted_decay=self._weighted_decay(),
            sequential=sequential,
            decay_dict=decay_dict,
            drop_prob_per_pdg=drop_prob_per_pdg,
            mixed_pdgs_set=mixed_pdgs_set,
            density_method=density_method,
            density_pole_approximation=density_pole_approximation,
            density_needs_reshuffle=density_needs_reshuffle,
            branching_ratio=self.branching_ratio,
            base_seed=int(self.seed) if self.seed else random.randint(0, 30081*30081),
        )

        # keep_weight_for_polarization_*: the extra weights have to be declared
        # in the header before it is written, here rather than in each writer --
        # the parallel path forks *after* this point and its workers write
        # bannerless fragments merged under this same banner. evt_decayfile is
        # only complete now, and it is what says which of the pdgs the file
        # holds really end up with a density slot.
        if self._polarization_weights_enabled():
            self._declare_polarization_weights(
                self._polarization_layout_statics(evt_decayfile))

        start = time.time()
        logger.info("Start generating decays")
        if nb_core == 1:
            output_lhe = lhe_parser.EventFile(base_out, 'w')
            if self.options['fixed_order']:
                output_lhe.eventgroup = True
                orig_lhe.eventgroup = True
            self.banner.scale_init_cross(self.branching_ratio)
            self.banner.write(output_lhe, close_tag=False)
            self.efficiency = 1.
            ctx['shard_nb_event'] = nb_event
            stats = self._unweight_range(orig_lhe, evt_decayfile, output_lhe, ctx)
            output_lhe.write('</LesHouchesEvents>\n')
            try:
                output_lhe.close()
            except Exception:
                pass
            self._apply_accounting(base_out, [stats])
        else:
            logger.info("MadSpin: unweighting %s events on %s cores", nb_event, nb_core)
            self._run_onshell_parallel(orig_lhe, nb_event, nb_core,
                                       evt_decayfile, base_out, ctx)
        logger.info(f"Time for decay = {time.time()-start:.2f} sec")

    def _resolve_nb_core(self):
        """Number of worker processes for the parallel unweighting / gridpack
        decay generation. A madspin-card ``set nb_core N`` takes precedence;
        otherwise fall back to the global MG5 ``nb_core``. Non-positive / unset /
        unparseable => serial (1)."""
        candidates = []
        try:
            candidates.append(self.options['nb_core'])
        except Exception:
            pass
        try:
            candidates.append(self.mg5cmd.options['nb_core'])
        except Exception:
            pass
        for source in candidates:
            try:
                n = int(source)
            except (TypeError, ValueError):
                continue
            if n >= 1:
                return n
        return 1

    def _gridpack_env(self):
        """Environment for run.sh / gridrun subprocesses. The gridpack scripts
        start with ``#!/usr/bin/env python3``, which otherwise resolves via PATH
        to whatever ``python3`` comes first -- often NOT the interpreter running
        MadSpin, and thus one missing modules that gridrun needs (e.g. ``six``,
        whose absence gridrun treats as fatal). Guarantee the same interpreter by
        putting a ``python3`` -> sys.executable shim first on PATH. Also expose
        ``six`` explicitly if this interpreter has it, in case it lives outside
        the default site-packages."""
        env = os.environ.copy()
        # 1. python3 shim so `env python3` == the MadSpin interpreter, regardless
        #    of whether dirname(sys.executable) even contains a bare `python3`.
        if not getattr(self, '_py3_shim_dir', None):
            import tempfile
            import atexit
            shim = tempfile.mkdtemp(prefix='ms_py3shim_')
            atexit.register(shutil.rmtree, shim, ignore_errors=True)
            link = pjoin(shim, 'python3')
            target = os.path.abspath(sys.executable)
            try:
                os.symlink(target, link)
            except (OSError, NotImplementedError, AttributeError):
                with open(link, 'w') as f:
                    f.write('#!/bin/sh\nexec "%s" "$@"\n' % target)
                os.chmod(link, 0o755)
            self._py3_shim_dir = shim
        env['PATH'] = self._py3_shim_dir + os.pathsep + env.get('PATH', '')
        # 2. belt-and-suspenders: if we can import six here, make sure the
        #    subprocess can find it too.
        try:
            import six as _six
            sixdir = os.path.dirname(os.path.abspath(_six.__file__))
            env['PYTHONPATH'] = sixdir + os.pathsep + env.get('PYTHONPATH', '')
        except Exception:
            pass
        return env

    def _run_gridpack(self, cmd, cwd):
        """Run a gridpack run.sh, capturing its output (the tail) so a generation
        failure can be reported with the actual gridrun output rather than an
        opaque missing-file error. The output is only echoed at DEBUG level, so
        the (very verbose) per-job gridrun progress does not spam the log."""
        import subprocess
        import collections as _collections
        proc = subprocess.Popen(cmd, cwd=cwd, env=self._gridpack_env(),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                bufsize=1, universal_newlines=True)
        tail = _collections.deque(maxlen=80)
        for line in proc.stdout:
            line = line.rstrip('\n')
            tail.append(line)
            logger.debug("[gridpack] %s", line)
        proc.wait()
        return proc.returncode, '\n'.join(tail)

    def _decay_pool_split(self):
        """In how many files the decay pool should be written: one per parallel
        unweighting worker (1 = single pool, historical behaviour)."""
        return self._resolve_nb_core()

    def _sequential_pool_ladder(self, to_decay, nb_event, density_method):
        """pdg -> how many decay events one production event burns on it, when
        the accept/reject is done one particle at a time. Empty when it is not.

        Each slot is redrawn until accepted, so it burns 1/eff_k events from its
        own pool, and eff_k drops along the ordering (_decay_pool_ladder). The
        pools are per pdg while the ladder is per slot: identical parents share
        a pool and take consecutive slots, so charge that pdg the largest of
        them.
        """
        if not self._sequential_active(density_method):
            return {}
        spins = {}
        for pdg in to_decay:
            try:
                spins[pdg] = self.model.get_particle(pdg).get('spin')
            except Exception:
                return {}
        preference = self._sequential_spin_order()
        def rank(pdg):
            spin = spins[pdg]
            return (preference.index(spin) if spin in preference
                    else len(preference), abs(pdg), pdg)
        ladder = {}
        position = 0
        for pdg in sorted(to_decay, key=rank):
            multiplicity = 1
            if nb_event:
                multiplicity = max(1, int(to_decay[pdg]) // int(nb_event))
            ladder[pdg] = self._decay_pool_ladder(position + multiplicity - 1,
                                                  spins[pdg])
            position += multiplicity
        return ladder

    # ------------------------------------------------------------------
    # The two questions the unweighting branches keep asking
    # ------------------------------------------------------------------
    # (a) which *spinmode family* is this -- is the density matrix evaluated at
    #     onshell momenta (pole approximation) or at the reshuffled ones? -- and
    # (b) which *accept/reject scheme* is this -- is there a mass-set stage in
    #     front of the angle stage?
    # Both are asked from several places, so they get names rather than being
    # spelled out as spinmode/mode string comparisons at each site.

    def _density_pole_approximation(self):
        """Whether the density matrix is taken in the pole approximation, i.e.
        evaluated at onshell momenta (``PA``/``onshell``) rather than at the
        reshuffled offshell ones (``madspin``/``full``)."""
        return self.options['spinmode'] in ['PA', 'onshell']

    def _density_do_reshuffle(self):
        """Whether a pole-approximation run nevertheless reshuffles the
        production onto the sampled virtualities. Only ``PA`` does: it samples a
        virtuality per resonance, while ``onshell`` keeps the production
        kinematics as they are."""
        return self.options['spinmode'] == 'PA'

    def _density_needs_reshuffle(self, in_density_mode):
        """Whether the chain reshuffles the production event at all. Offshell it
        always does -- that is where its density matrix is evaluated -- ``PA``
        does because it samples virtualities, ``onshell`` never does, and
        nothing does outside density mode.

        ``in_density_mode`` is the caller's own way of knowing it is in density
        mode (the ``density_method`` flag before the generation exists, and
        ``self.generate_all.mode == 'density'`` afterwards)."""
        return in_density_mode and (not self._density_pole_approximation()
                                    or self._density_do_reshuffle())

    # the spinmodes ``fixed_order`` reshuffles the production in, and so cannot
    # decay an event group in: PA samples a virtuality per resonance,
    # madspin/full evaluates its density at the reshuffled (offshell) momenta.
    FIXED_ORDER_RESHUFFLING_SPINMODES = ('PA', 'madspin')

    def _check_fixed_order_spinmode(self, spinmode):
        """Refuse ``fixed_order`` in a spinmode that reshuffles the production.

        An event group is decayed *once*: the born event's decays are attached
        to the born event and to every counter-event, unchanged (the 2017
        design of the option, and the only one under which the subtraction
        still cancels after the decay -- an independent draw per member would
        decay the event and the term subtracting it differently).

        That is fine as long as nothing else moves the production kinematics.
        PA and madspin/full do: they reshuffle the production onto sampled
        virtualities, and only the born member goes through that reshuffling,
        so its resonance would sit at the sampled mass while the counter-events
        subtracting it stay onshell. Reshuffling each member separately is not
        the answer either -- the members are related by the fixed-order mapping,
        the jacobians would differ per member, and the reshuffling can fail for
        one member and succeed for the others.

        Until that is designed, refuse: a group whose members disagree looks
        like a decayed sample and is not one. ``onshell``/``onshell_v1`` keep
        the production kinematics and are unaffected.
        """
        if not self.options['fixed_order']:
            return
        if spinmode not in self.FIXED_ORDER_RESHUFFLING_SPINMODES:
            return
        raise self.InvalidCmd(
            "fixed_order is not available in spinmode=%s: that mode reshuffles "
            "the production onto sampled virtualities, and how an event "
            "group's counter-events follow the born event through that "
            "reshuffling is not defined -- only the born event would be "
            "reshuffled. Use spinmode=onshell (or onshell_v1), which keeps the "
            "production kinematics, or turn fixed_order off." % spinmode)

    def _spinmode_has_density(self):
        """Whether the spinmode carries the density-matrix machinery the staged
        accept/reject schemes are built on. The v1 spinmodes, ``none`` and
        ``bridge`` do not, and keep the historical joint test."""
        return (self._density_pole_approximation()
                or self.options['spinmode'] in ['madspin', 'full'])

    @staticmethod
    def _is_upfront_scheme(mode):
        """Whether ``mode`` draws every virtuality *before* the angles, i.e.
        whether it has a mass-set accept/reject in front of its angle stage.
        True for every scheme but ``joint`` -- which tests the virtualities and
        the angles together -- and ``sequential_with_mass``, which draws each
        slot's mass inside that slot's own accept/reject."""
        return mode not in ('joint', 'sequential_with_mass')

    def _log_once(self, key, message, *args):
        """Log a resolution message the first time only: these are decided per
        production event but say something about the run."""
        seen = getattr(self, '_logged_once', None)
        if seen is None:
            seen = self._logged_once = set()
        if key not in seen:
            seen.add(key)
            logger.info(message, *args)

    def _auto_unweighting_mode(self):
        """What ``unweighting = auto`` resolves to, before any of the
        fallbacks: one branch per spinmode family, keyed on the number of
        decaying particles, plus an override for a polarised production.

        The two branches were measured over the number of decaying particles n
        on `p p > w+ j` (n=1), `p p > t t~` (2), `p p > t t~ z` (3) and
        `p p > t t~ t t~` (4), 50000 events each -- see
        doc/madspin_sequential_plan.md section 12.

        **PA/onshell -> ``sequential``, at every n.** It was the fastest of the
        three at all four multiplicities, by 1.2x at n=1 rising to 3.8x at n=4.
        The joint test's cost grows as n x (trials per event), since one
        rejection throws every decay away, while the per-particle one's grows
        far more slowly; and the up-front mass draw evaluates the production
        reshuffling jacobian once per mass set instead of once per slot trial.
        Even at n=1, where the angle stage degenerates to the joint test, the
        mass stage still pays for itself: a mass set can be rejected before any
        decay is drawn.

        **madspin/full -> ``joint`` up to two decaying particles, then
        ``sequential``.** Offshell, each mass set costs a production reshuffle
        *and* an offshell production density, which the joint test pays per
        trial but which a staged scheme pays per mass set -- and below n=3 there
        are not enough decays to save to cover it. At n=1 it is worse than that:
        the mass-set weight carries ``Tr(rho_off)/|M_prod|^2_on``, and when the
        single decaying particle carries most of the production matrix
        element's virtuality dependence (`p p > w+ j`) that ratio spans orders
        of magnitude, no bound covers it, and the mass stage needs ~790 sets per
        accepted event. From n=3 the per-particle test wins by 2.2x and 4.3x.

        **A polarised production -> ``sequential``, whatever n.** A brace on the
        production process (``_production_polarization``) restricts the
        production/decay convolution to a polarisation subspace, which peaks the
        joint weight far below the bound the max-weight scan hands it -- and the
        joint test has no way to recover, since its bound is a single number
        over the whole chain. Measured offshell on `p p > t t~` (n=2, so the
        multiplicity rule would say joint) with both tops decayed:

        ==========================  ==============  ==============
        production                  joint           sequential
        ==========================  ==============  ==============
        `t t~` (unpolarised)          3.3             6.1
        `t{+}t~{+}`                 112               9.1
        `t{+}t~{-}`                 162               8.4
        ==========================  ==============  ==============

        in trials per accepted event, 500 events each. The 50000-event
        validation of all four polarised final states, where the max-weight
        scan is longer and ``nb_sigma`` larger, saw the joint column rise to
        4.05 unpolarised, 204-213 like-helicity and 5800-6300
        opposite-helicity against 8.59 sequential: the gap widens with
        statistics, because the bound the joint test must clear keeps growing
        while the bulk of the restricted weight distribution does not.
        Unpolarised, joint is the better of the two by ~2x and the rule above
        stands; polarised it loses by one to three orders of magnitude, so
        ``auto`` gives the brace priority over n.

        The clause fires on any brace in the production line, including one on a
        particle MadSpin does not decay -- such a brace leaves the restriction
        handed to ``DensityMatrix`` empty and so cannot be the thing peaking the
        weight. Two reasons to fire anyway. The asymmetry: taking ``sequential``
        when joint would have done costs the ~2x above, taking joint when the
        convolution is restricted costs 30-1500x. And the resolved mode has to
        be the same at every call site -- it names the max-weight cache files and
        picks which bound the accept/reject tests against -- while the set of
        decayed pdgs is not known everywhere ``_unweighting_mode`` is called; a
        clause that consulted it could resolve two ways in one run.
        An explicit ``set unweighting joint`` is still honoured: only ``auto``
        comes through here.

        ``two_stage`` is not the fastest scheme at any measured point -- joint
        beats it at n<=2 and ``sequential`` at n>=3 -- so ``auto`` never
        returns it, and it is no longer offered in the card either (it is not
        in the advertised ``allowed`` list; see
        ``MadSpinOptions.hidden_unweighting_modes``, which still honours an
        explicit request for it). It stays useful as a cross-check, being the
        one staged scheme whose angle stage is a single joint test, and the
        code path is unchanged.
        """
        if self._density_pole_approximation():
            # fastest at every multiplicity measured; rho is fixed on shell
            # so the mass stage costs a reshuffling jacobian and nothing else
            return 'sequential'
        if self._density_spinmode() and self._production_polarization():
            # a polarised production restricts the convolution to a
            # polarisation subspace, and the joint weight then sits orders
            # of magnitude below its own bound -- see above. The multiplicity
            # rule does not apply: joint has no way to recover.
            return 'sequential'
        if getattr(self, '_nb_decaying', 2) <= 2:
            # offshell a mass set costs a production reshuffle and a
            # production density, and there are not yet enough decays to
            # save to pay for it
            return 'joint'
        return 'sequential'

    def _unweighting_mode(self, density_method=True):
        """Which accept/reject scheme this run uses: one of 'joint',
        'two_stage', 'sequential', 'sequential_global_retry',
        'sequential_with_mass'.

        All of them sample the same distribution; they differ in how the test is
        split and in what a rejection redraws.

          joint                    one test over the virtualities and every
                                   decay at once -- the historical scheme, and
                                   the only one available outside density mode.
          two_stage                the set of virtualities is unweighted first,
                                   then every decay against a single bound; a
                                   rejection redraws the decays only, so the
                                   production reshuffling and its density matrix
                                   are reused across the retries.
          sequential               as two_stage, but one test per decaying
                                   particle, redrawing only the particle that
                                   was rejected.
          sequential_global_retry  as sequential, but a rejected decay redraws
                                   the virtualities as well.
          sequential_with_mass     one test per decaying particle, with that
                                   particle's virtuality drawn *inside* its own
                                   accept/reject rather than up front.

        The first four share an up-front mass draw and so a mass-set stage;
        ``sequential_with_mass`` is the odd one out and not a variant of
        ``sequential``: the mass is drawn and redrawn together with that slot's
        angles, so nothing is ever frozen, no stage has a conditional
        normalisation to divide out, and the tabulated running-width factor the
        two-stage schemes need does not arise. It is the historical PA scheme.
        It needs a per-particle mass draw, i.e. the PA spinmode; the offshell
        spinmodes reshuffle the whole production onto the mass set at once, so
        there they fall back to ``sequential``.

        What ``auto`` resolves to, and why, is in ``_auto_unweighting_mode``.
        Whatever is asked for or resolved to, the fallbacks below can still send
        the run back to ``joint``.

        ``fixed_order`` forces joint: its counter-events ride along with the
        decays and have not been thought through here.
        """
        if not density_method:
            return 'joint'
        if self._pure_interference():
            # Every staged scheme substitutes DensityMatrix.identity for the
            # decay slots it has not drawn yet, and the interference block has
            # no diagonal entry, so every partial contraction against the
            # identity is identically zero: no prefix carries any weight and
            # there is nothing to unweight against. Section 13.4.
            self._log_once('pure_interference_joint',
                           "MadSpin: pure_interference forces the joint "
                           "accept/reject (every partial weight of a staged "
                           "scheme is identically zero in this mode)")
            return self._announce_mode('joint', self.options['unweighting'])
        if self._weighted_decay():
            # There is no accept/reject at all in this mode, so there is no
            # scheme to choose: the staged schemes exist only to split a test
            # that is not being made. The joint branch is the one that carries
            # the weighted path.
            self._log_once('weighted_decay_joint',
                           "MadSpin: decay_output = weighted takes the joint "
                           "path (there is no accept/reject to stage)")
            return self._announce_mode('joint', self.options['unweighting'])
        asked = mode = self.options['unweighting']
        if mode == 'auto':
            mode = self._auto_unweighting_mode()
        if mode == 'joint':
            return self._announce_mode('joint', asked)
        if self.options['fixed_order']:
            self._log_once('fixed_order',
                           "MadSpin: fixed_order is on, keeping the joint "
                           "accept/reject (unweighting ignored)")
            return self._announce_mode('joint', asked)
        if getattr(self, '_decay_groups', None):
            # The per-particle test redraws one slot until it is accepted, which
            # divides E[w_k | group] out of the chain -- and that expectation
            # differs between groups, so the accepted group fractions would come
            # out distorted by its reciprocal. The joint test redraws the whole
            # set, group included, so the group stays part of what is being
            # unweighted. Lifting this needs a bound and a rate factor per
            # group; see doc/madspin_decay_groups.md.
            #
            # two_stage self-normalises the same way (it redraws the angle set
            # to acceptance), so it falls back too. Whether
            # sequential_global_retry could be exempted -- a rejection there
            # throws the whole chain away rather than renormalising -- depends
            # on whether the group is redrawn with it, and has not been checked.
            self._log_once('decay_groups',
                           "MadSpin: the decay lines are grouped ('@' tags), "
                           "keeping the joint accept/reject "
                           "(unweighting ignored)")
            return self._announce_mode('joint', asked)
        if not self._spinmode_has_density():
            self._log_once('spinmode',
                           "MadSpin: spinmode=%s keeps the joint accept/reject "
                           "(unweighting ignored)", self.options['spinmode'])
            return self._announce_mode('joint', asked)
        if (mode == 'sequential_with_mass'
                and not self._density_pole_approximation()):
            self._log_once('with_mass_pa_only',
                           "MadSpin: unweighting=sequential_with_mass needs a "
                           "per-particle mass draw, which the offshell "
                           "spinmodes do not have (they reshuffle the whole "
                           "production onto the mass set at once); using "
                           "sequential instead")
            return self._announce_mode('sequential', asked)
        return self._announce_mode(mode, asked)

    def _announce_mode(self, mode, asked):
        """Say once which scheme the run uses. Worth a line because the card no
        longer answers it: 'auto' resolves on the process."""
        self._log_once('mode', "MadSpin: unweighting = %s (%s)", mode,
                       'auto, %s decaying particle(s)'
                       % getattr(self, '_nb_decaying', '?')
                       if asked == 'auto' else 'set explicitly')
        return mode

    def _sequential_active(self, density_method):
        """Whether any of the per-particle / two-stage schemes is in use, i.e.
        anything but the historical joint accept/reject."""
        return self._unweighting_mode(density_method) != 'joint'

    def _sequential_spin_order(self):
        """The spin order (MG5 2S+1 convention) driving which particle is
        accept/rejected first. Unlisted spins go last, in their natural slot
        order."""
        try:
            order = [int(x) for x in
                     str(self.options['sequential_spin_order']).replace(',', ' ').split()]
        except (ValueError, TypeError):
            order = []
        return order or [2, 3, 1]

    def _decay_slot_order(self, decaying_spins):
        """Order in which the slots are accept/rejected.

        Sorted by ``sequential_spin_order`` (default: fermions, then vectors,
        then scalars), ties broken by slot index so a run stays reproducible and
        independent of dict ordering. Only decides *which slot is filled next* --
        the tensor product itself must stay in slot order, see
        doc/madspin_sequential_plan.md."""
        pref = self._sequential_spin_order()
        def key(slot):
            spin = decaying_spins[slot]
            rank = pref.index(spin) if spin in pref else len(pref)
            return (rank, slot)
        return sorted(range(len(decaying_spins)), key=key)

    @staticmethod
    def _decay_pool_ladder(position, spin):
        """Expected number of decay events one production event burns on the
        slot sitting at ``position`` of the accept/reject ordering.

        Each slot is redrawn until accepted, so it burns 1/eff_k events from its
        own pool. The first slot sees a production density matrix traced over
        everything else -- close to unpolarised, so mild modulation and a high
        acceptance; each subsequent one sees a more conditioned, more polarised
        parent, hence a wider weight spread and a lower acceptance. Hence the
        ladder 1.5, 2, 2.5, 3, ...

        A spin-0 parent (MG5 spin==1) has a 1x1 decay density matrix, so its
        ratio is identically 1: it can never be rejected and burns exactly one
        event wherever it sits. Charging it the ladder would just generate
        decays nobody consumes."""
        if spin == 1:
            return 1.1
        return 1.5 + 0.5 * position

    @staticmethod
    def _decay_dir(path_me, pdg, decay_file_nb):
        return pjoin(path_me,
                     "decay_%s_%s" % (str(pdg).replace("-", "x"), decay_file_nb))

    def _regenerate_events(self, pdg, decay_file_nb, needed, run_name):
        """Produce ``needed`` extra decay events for a channel that has already
        been generated once, into a run of its own. Returns the reader over the
        events produced (already split per worker when running in parallel)."""
        decay_dir = self._decay_dir(self.path_me, pdg, decay_file_nb)
        # Never let the generation land on an existing run: madevent then asks
        # "do you wish to overwrite?", the non-interactive answer is 'n', and it
        # silently produces nothing -- which used to make every worker retry the
        # same doomed generation in turn.
        stale = pjoin(decay_dir, 'Events', run_name)
        if os.path.exists(stale):
            _force_rmtree(stale)
        # RunWeb is madevent's "this directory is busy" marker and makes
        # MadEventCmd refuse to start (AlreadyRunning). The generation that
        # created the pool does not always clean it up. The caller holds the
        # exclusive refill lock, so no other madevent can be running here: any
        # RunWeb we find is stale and removing it is safe.
        runweb = pjoin(decay_dir, 'RunWeb')
        if os.path.exists(runweb):
            logger.debug("removing stale RunWeb in %s before the refill", decay_dir)
            try:
                os.remove(runweb)
            except OSError:
                pass
        with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"],
                             [50, 50, 50, 50]):
            out = self.generate_events(pdg, needed, self.mg5cmd,
                                       [decay_file_nb], run_name=run_name)
        reader = out[decay_file_nb]
        missing = [p for p in self._reader_paths(reader)
                   if not os.path.exists(p)]
        if missing:
            raise Exception(
                "MadSpin: decay-event refill for pdg %s produced no events "
                "(expected %s)." % (pdg, ', '.join(missing)))
        return reader

    @staticmethod
    def _reader_paths(reader):
        """Every file behind a decay-pool reader (one per worker when split)."""
        return list(getattr(reader, 'paths', None) or [reader.name])

    @staticmethod
    def _reader_from_paths(paths):
        """Rebuild a decay-pool reader from the paths marshalled across a fork."""
        if len(paths) > 1:
            return _ChainedEvents(paths)
        return lhe_parser.EventFile(paths[0])

    def _generate_decay_entry(self, pdg, job, nb_core, seed_offset, res_path):
        """Generate the decay events of one particle inside a forked process
        (see :meth:`_generate_decays`). The produced EventFile objects cannot
        cross the process boundary, so report their paths (and the partial
        width) through a small JSON file."""
        import json
        try:
            # cap the cores this generation may use, use a seed of our own, and
            # never reuse a MadEventCmdShell inherited through fork
            # multiprocessing reseeds Python's global RNG from OS entropy in
            # every forked child (BaseProcess._bootstrap), so a child that does
            # not re-seed it deterministically is NOT reproducible. That matters
            # here because the tail of the generation is pure Python
            # accept/reject on that RNG (combine_runs.copy_events and
            # EventFile.unweight): left unseeded, the very same iseed gives a
            # decay pool of a different size and content on every run, and the
            # decayed sample diverges with it. Key it on the same offset as the
            # madevent seed so the two particles keep distinct streams, with a
            # multiplier of its own so a generation child can never land on the
            # stream of an unweighting/max-weight worker (those use 7919).
            random.seed((int(self.seed) if self.seed else 0) + 104729 * seed_offset)
            self._gen_nb_core = nb_core
            self.seed = (int(self.seed) + 1000003 * seed_offset) % (30081 * 30081)
            self.options['seed'] = self.seed
            self.me_int = {}
            out, width, channel_widths = self.generate_events(
                pdg, job['nb_gen'], self.mg5cmd,
                cumul=job['cumul'], output_width=True)
            with open(res_path, 'w') as fp:
                # send back every file of each channel: when the pool is split
                # per worker (nb_unweight_output) reporting only the first one
                # would silently shrink the pool to a single slice.
                json.dump({'files': dict((str(k), self._reader_paths(v))
                                         for k, v in out.items()),
                           'width': width,
                           'channel_widths': dict((str(k), v) for k, v
                                                  in channel_widths.items())},
                          fp)
        except Exception as exc:
            import traceback
            try:
                with open(res_path, 'w') as fp:
                    json.dump({'error': str(exc), 'tb': traceback.format_exc()}, fp)
            except Exception:
                pass

    def _generate_decays(self, gen_jobs, mg5):
        """Generate the decay events for every decaying particle; returns
        ``{pdg: ({file_nb: EventFile}, partial_width, {file_nb: width})}``.

        The generations are independent (each particle has its own
        ``decay_<pdg>_<i>`` directory) and a single MadEvent generation only
        keeps every core busy for a short while before tailing off to a single
        job, so run them concurrently -- one forked process per particle -- and
        split the core budget between them so their *sum* never oversubscribes
        the machine.
        """
        if len(gen_jobs) <= 1:
            return dict((pdg, self.generate_events(pdg, job['nb_gen'], mg5,
                                                   cumul=job['cumul'],
                                                   output_width=True))
                        for pdg, job in gen_jobs.items())

        import multiprocessing as mp
        import json
        budget = self._resolve_nb_core()
        try:
            budget = min(budget, mp.cpu_count())
        except NotImplementedError:
            pass
        per = max(1, 2 * budget // len(gen_jobs))
        logger.info("MadSpin: generating the decay events of %s particles at "
                    "once, %s core(s) each", len(gen_jobs), per)

        mpctx = mp.get_context('fork')
        procs = []
        for offset, (pdg, job) in enumerate(gen_jobs.items()):
            res = pjoin(self.path_me, 'ms_gen_%s.json' % str(pdg).replace('-', 'x'))
            p = mpctx.Process(target=self._generate_decay_entry,
                              args=(pdg, job, per, offset + 1, res))
            p.start()
            procs.append((pdg, p, res))
        for pdg, p, res in procs:
            p.join()

        out = {}
        for pdg, p, res in procs:
            if not os.path.exists(res):
                raise Exception("MadSpin: the decay generation of pdg %s produced "
                                "no result (crashed?)." % pdg)
            with open(res) as fp:
                data = json.load(fp)
            os.remove(res)
            if 'error' in data:
                raise Exception("MadSpin: the decay generation of pdg %s failed:\n%s"
                                % (pdg, data.get('tb', data['error'])))
            out[pdg] = (dict((int(k), self._reader_from_paths(v))
                             for k, v in data['files'].items()),
                        data['width'],
                        dict((int(k), v) for k, v
                             in data.get('channel_widths', {}).items()))
        return out

    @staticmethod
    def _refill_pool_dir(decay_dir, gen):
        """Directory holding refill pool ``gen`` of this channel."""
        return pjoin(decay_dir, 'Events', 'ms_refill_%d' % gen)

    def _refill_pool_paths(self, decay_dir, gen):
        """Every per-worker slice of the refill pool ``gen``, in worker order.
        This layout is the contract between the owner (which puts the files
        there, see :meth:`_generate_refill_pool`) and the waiters (which open
        their own one and nothing else)."""
        base = pjoin(self._refill_pool_dir(decay_dir, gen),
                     'unweighted_events.lhe')
        return lhe_parser.EventFile.unweight_output_paths(
            base, self._shard_nb_core)

    def _refill_pool_path(self, decay_dir, gen):
        """This worker's own file of the refill pool ``gen``. The refill hands
        each worker one file, so a worker never reads (nor even parses) the
        events that belong to the others."""
        paths = self._refill_pool_paths(decay_dir, gen)
        path = paths[self._shard_tag] if len(paths) > 1 else paths[0]
        if not os.path.exists(path):
            raise Exception("MadSpin: refill pool %s is missing for worker %s"
                            % (path, self._shard_tag))
        return path

    @staticmethod
    def _count_lhe_events(path):
        """Number of ``<event`` records in an LHE file (plain or gzip). Used to
        size the owner's ~10% shortfall -- a cheap line scan, no parsing."""
        try:
            if path.endswith('.gz'):
                import gzip
                opener = lambda p: gzip.open(p, 'rt', errors='ignore')
            else:
                opener = lambda p: open(p, 'r', errors='ignore')
            n = 0
            with opener(path) as fh:
                for line in fh:
                    if '<event' in line:
                        n += 1
            return n
        except (IOError, OSError):
            return 0

    def _channel_owner(self, pdg, decay_file_nb):
        """The single worker that is the sole (re)generator of this decay
        channel. Channels are numbered in a fixed sorted order and dealt out to
        the workers round-robin, so the owner is a deterministic function of the
        channel alone -- never of run-time timing. Fixing the generator this way
        is what makes the refilled pool (and thus the decayed sample)
        reproducible: contrast the old "whoever ran out first generates", whose
        winner was a lock race."""
        keys = getattr(self, '_channel_keys', None) or []
        try:
            idx = keys.index((pdg, decay_file_nb))
        except ValueError:
            idx = 0
        return idx % max(1, int(self._shard_nb_core))

    def _open_refill_slice(self, decay_dir, gen, owner):
        """This worker's reader over refill pool ``gen``; the owner's slice is
        cut ~10% short (see _LimitedEvents) so the owner runs out first."""
        path = self._refill_pool_path(decay_dir, gen)
        reader = lhe_parser.EventFile(path)
        if self._shard_tag == owner:
            frac = float(getattr(self, '_owner_undersize', 0.10) or 0.0)
            if frac > 0:
                n = self._count_lhe_events(path)
                reader = _LimitedEvents(reader,
                                        int(math.floor((1.0 - frac) * n)))
        return reader

    @staticmethod
    def _split_pool_round_robin(sources, targets):
        """Deal the events of the ``sources`` LHE file(s) round-robin into the
        ``targets`` files, each of which gets the banner of the first source (a
        decay pool is picked among the channels of a pdg by its cross-section,
        which is read from that banner). Returns the number of events written.

        Worker i therefore ends up with the events at positions i, i+N, ... of
        the pool -- exactly the stripe it would otherwise pick out itself, minus
        having to parse the other workers' events to get there."""
        first = lhe_parser.EventFile(sources[0])
        banner = first.banner
        first.close()
        outs = [lhe_parser.EventFile(p, 'w') for p in targets]
        nb_event = 0
        try:
            for out in outs:
                if banner:
                    out.write(banner)
            for src in sources:
                fsock = lhe_parser.EventFile(src)
                fsock.parsing = False   # raw lines: no need to build Event objects
                try:
                    for text in fsock:
                        outs[nb_event % len(outs)].write(''.join(text))
                        nb_event += 1
                finally:
                    fsock.close()
        finally:
            for out in outs:
                out.write('</LesHouchesEvents>\n')
                out.close()
        return nb_event

    def _materialise_refill_pool(self, sources, targets, decay_dir, gen):
        """Put the events the generation produced at the per-worker paths the
        waiters will open, when the backend did not write them there itself.

        Built in a sibling temporary directory and renamed into place, so
        ``Events/ms_refill_<gen>`` never exists half-written -- a waiter that
        gets as far as looking at it (it should not: it waits for the generation
        marker, published later still) finds either nothing or the whole pool."""
        final = self._refill_pool_dir(decay_dir, gen)
        tmp = final + '.part'
        if os.path.exists(tmp):
            _force_rmtree(tmp)
        os.makedirs(tmp)
        nb_event = self._split_pool_round_robin(
            sources, [pjoin(tmp, os.path.basename(p)) for p in targets])
        if os.path.exists(final):
            _force_rmtree(final)
        os.rename(tmp, final)
        return nb_event

    def _generate_refill_pool(self, pdg, decay_file_nb, needed, gen):
        """Generate generation ``gen`` of this channel's decay pool and leave it
        COMPLETE at the per-worker paths of :meth:`_refill_pool_paths`. Returns
        those paths. Publishing ``gen`` is only allowed once this has returned.

        Two generation backends land here and they do not agree on where they
        write. The plain madevent one honours ``run_name`` and splits the
        unweighting one file per worker, i.e. it writes the canonical layout
        itself. The gridpack one (any ``ms_dir`` run) goes through run.sh, which
        knows nothing of either: it always writes a single
        ``<decay_dir>/events.lhe.gz`` -- straight on top of the pool the other
        workers are still reading. So on that backend, move the pool aside for
        the duration, split what run.sh produced into the per-worker files, and
        put the pool back: a refill then only ever *adds* a generation. Renaming
        is safe for the workers that already hold the pool open -- they keep
        their inode -- and this whole routine runs under the channel's exclusive
        refill lock."""
        decay_dir = self._decay_dir(self.path_me, pdg, decay_file_nb)
        targets = self._refill_pool_paths(decay_dir, gen)
        pool = pjoin(decay_dir, 'events.lhe.gz')
        stash = pool + '.mspool'
        # mirrors ``use_gridpack`` in generate_events
        protect = bool(self.options['ms_dir']) and os.path.exists(pool)
        if protect:
            if os.path.exists(stash):
                os.remove(stash)
            os.rename(pool, stash)
        try:
            reader = self._regenerate_events(pdg, decay_file_nb, needed,
                                             'ms_refill_%d' % gen)
            sources = self._reader_paths(reader)
            try:
                reader.close()
            except Exception:
                pass
            if sources != targets:
                self._materialise_refill_pool(sources, targets, decay_dir, gen)
        finally:
            if protect:
                try:
                    os.remove(pool)
                except OSError:
                    pass
                os.rename(stash, pool)
        missing = [p for p in targets if not os.path.exists(p)]
        if missing:
            raise Exception(
                "MadSpin: the refill of pdg %s (decay file %s) did not produce "
                "%s; the generation was not published, so no worker will try to "
                "read it." % (pdg, decay_file_nb, ', '.join(missing)))
        return targets

    @staticmethod
    def _published_gen(decay_dir):
        """Highest refill generation published on disk for this channel (0 if
        none). The single source of truth both the owner and the waiters read."""
        gen_file = pjoin(decay_dir, 'ms_refill.gen')
        if not os.path.exists(gen_file):
            return 0
        try:
            return int(open(gen_file).read().strip())
        except (ValueError, IOError):
            return 0

    @staticmethod
    def _publish_gen(decay_dir, gen):
        """Make ``gen`` the published generation of this channel.

        Written to a temporary file and renamed over the marker, so a waiter
        polling :meth:`_published_gen` reads either the old generation or the
        new one, never a half-written number. The marker becoming visible IS the
        promise that every file of that generation is complete and readable, so
        this must only ever be called once :meth:`_generate_refill_pool` has
        returned."""
        gen_file = pjoin(decay_dir, 'ms_refill.gen')
        tmp = '%s.%s.tmp' % (gen_file, os.getpid())
        with open(tmp, 'w') as fp:
            fp.write('%d\n' % gen)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, gen_file)

    def _owner_generate(self, pdg, decay_file_nb, target_gen, needed):
        """Generate this channel's pool up to ``target_gen`` under the channel
        lock, then publish the gen counter. Idempotent: if the gen is already on
        disk the while-loop is skipped.

        The madevent seed is keyed on the channel and the gen number with a base
        shared by every worker, so gen N gets the SAME seed no matter who
        generates it. The size follows the caller's efficiency-based ``needed``.
        In normal operation only the fixed owner calls this (its ``needed`` is
        deterministic, so the pool is reproducible); the deadlock fail-safe in
        _worker_refill may also call it from a non-owner, whose ``needed`` differs
        -- that (rare) refill is intentionally not guaranteed reproducible."""
        import fcntl
        decay_dir = self._decay_dir(self.path_me, pdg, decay_file_nb)
        with open(pjoin(decay_dir, 'ms_refill.lock'), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                current = self._published_gen(decay_dir)
                while current < target_gen:
                    new_gen = current + 1
                    seed_base = getattr(self, '_refill_seed_base', None)
                    if seed_base is None:
                        seed_base = int(self.seed) if self.seed else 0
                    # sign-aware pdg term so a particle and its antiparticle
                    # channel do not collide onto the same iseed
                    det_seed = 1 + ((int(seed_base)
                                     + 1000003 * ((int(pdg) % 998244353) + 1)
                                     + 101 * (int(decay_file_nb) + 1)
                                     + 100003 * new_gen) % (30081 * 30081))
                    self.seed = det_seed
                    self.options['seed'] = det_seed
                    det_needed = min(200000, max(1000, int(math.ceil(
                        needed / float(self._shard_nb_core))))) \
                        * self._shard_nb_core
                    logger.info("MadSpin worker %s OWNS pdg %s: generating gen %s "
                                "(%s events, seed %s) for all %s workers",
                                self._shard_tag, pdg, new_gen, det_needed,
                                det_seed, self._shard_nb_core)
                    # The generation runs a full madevent IN THIS PROCESS and
                    # reseeds / consumes Python's global ``random`` (combine +
                    # unweight). Snapshot and restore it so the generation never
                    # perturbs this worker's accept/reject stream. Use a *fresh*
                    # MadEventCmdShell (never the fork-inherited one); it writes a
                    # run of its own and splits one file per worker.
                    rng_state = random.getstate()
                    self.me_int = {}
                    stag, self._shard_tag = self._shard_tag, None
                    try:
                        self._generate_refill_pool(pdg, decay_file_nb,
                                                   det_needed, new_gen)
                    finally:
                        self._shard_tag = stag
                        random.setstate(rng_state)
                    # Publish only once every per-worker file of the generation
                    # is complete on disk: the marker is the ONLY thing a waiter
                    # looks at before opening its slice, so making it visible any
                    # earlier is telling that worker to open a file that is not
                    # there. _generate_refill_pool has checked that they all are.
                    self._publish_gen(decay_dir, new_gen)
                    current = new_gen
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    # ---- cross-process worker status, for the deadlock fail-safe -------------
    # Each forked worker publishes a one-line status file others can read:
    #   'R'          running (making progress)
    #   'G'          generating a decay pool
    #   'W <id>'     blocked, waiting for worker <id> to generate a pool
    # so a blocked owner can walk the wait-for chain and spot a cycle.
    def _status_path(self, worker_id):
        return pjoin(self.path_me, 'ms_wstatus_%d' % worker_id)

    def _clear_refill_state(self):
        """Drop the refill bookkeeping an earlier run left in the decay
        directories. Called by the parent, once, after the pools are generated
        and before any worker is forked.

        A reused ``ms_dir`` still holds the previous run's ``ms_refill.gen`` and
        ``Events/ms_refill_*``. Every worker of THIS run starts at generation 0,
        so the first time one runs its pool out it would read that marker,
        conclude generation 1 is already published, generate nothing and open the
        *other* run's pool -- which was sized for that run's efficiency and split
        for its ``nb_core``, so the slice this run's worker wants may not even
        exist. A run's refills are that run's own."""
        for decay_dir in misc.glob("decay_*_*", self.path_me):
            try:
                os.remove(pjoin(decay_dir, 'ms_refill.gen'))
            except OSError:
                pass
            # a refill interrupted midway leaves the pool stashed aside
            # (_generate_refill_pool); the run about to start regenerates the
            # pool anyway, so only put it back when nothing else is there
            pool = pjoin(decay_dir, 'events.lhe.gz')
            stash = pool + '.mspool'
            if os.path.exists(stash):
                if os.path.exists(pool):
                    os.remove(stash)
                else:
                    os.rename(stash, pool)
            for stale in misc.glob("ms_refill_*", pjoin(decay_dir, 'Events')):
                _force_rmtree(stale)

    def _clear_worker_status(self, nb_core):
        """Remove stale per-worker status files before forking a phase, so a
        'D'(one) left by the previous phase's worker of the same id can't be
        misread as this phase's worker being done. Called by the parent."""
        for wid in range(int(nb_core)):
            try:
                os.remove(self._status_path(wid))
            except OSError:
                pass

    def _set_status(self, state, target=None):
        tag = getattr(self, '_shard_tag', None)
        if tag is None:
            return
        try:
            with open(self._status_path(tag), 'w') as f:
                f.write(state if target is None else '%s %d' % (state, target))
        except (IOError, OSError):
            pass

    def _read_worker_status(self, worker_id):
        """(state, [target]) tuple for ``worker_id``, or None if unreadable."""
        try:
            # closed explicitly: this is polled ten times a second for as long
            # as MADSPIN_REFILL_WAIT, in every worker and once per hop of the
            # wait-for chain
            with open(self._status_path(worker_id)) as fsock:
                parts = fsock.read().split()
        except (IOError, OSError):
            return None
        if not parts:
            return None
        if parts[0] == 'W' and len(parts) >= 2:
            try:
                return ('W', int(parts[1]))
            except ValueError:
                return None
        return (parts[0],)

    def _wait_cycle_to_self(self, first_target):
        """True if the wait-for chain that starts at ``first_target`` leads back
        to this worker -- a deadlock cycle only I can break. A worker on the path
        that is running or generating (not 'W') means the chain is making
        progress, so there is no deadlock through it."""
        cur = first_target
        for _ in range(int(self._shard_nb_core) + 1):
            if cur == self._shard_tag:
                return True
            st = self._read_worker_status(cur)
            if not st or st[0] != 'W':
                return False
            cur = st[1]
        return False

    def _worker_refill(self, pdg, decay_file_nb, needed):
        """Owner-based decay-event refill for the forked workers. Returns the
        reader this worker should continue from.

        Each channel has one deterministic OWNER worker (:meth:`_channel_owner`).
        In normal operation only the owner (re)generates that channel's pool;
        any other worker that runs out BLOCKS until the owner has published the
        generation it needs, then opens its own slice. Because the generator is
        fixed rather than "whoever ran out first", the regenerated pool -- and
        hence the decayed sample -- is identical from one run to the next. The
        owner's slice is deliberately ~10% short (:meth:`_open_refill_slice`) so
        it reaches the refill point first and the others rarely wait.

        Deadlock fail-safe: while blocked, a worker publishes that it is waiting
        for the owner and walks the wait-for chain (:meth:`_wait_cycle_to_self`).
        If the chain loops back to itself -- a circular wait no owner can clear
        on its own -- the worker generates the channel itself to break it. That
        abandons the fixed-owner rule for this one refill, so its size (and thus
        the sample) is not guaranteed reproducible -- an accepted trade to avoid
        a hang. Generation runs on all cores."""
        decay_dir = self._decay_dir(self.path_me, pdg, decay_file_nb)
        key = (pdg, decay_file_nb)
        my_gen = self._pool_gen.get(key, 0)
        target_gen = my_gen + 1
        owner = self._channel_owner(pdg, decay_file_nb)

        if self._shard_tag == owner:
            self._set_status('G')
            try:
                self._owner_generate(pdg, decay_file_nb, target_gen, needed)
            finally:
                self._set_status('R')
            self._pool_gen[key] = target_gen
            return self._open_refill_slice(decay_dir, target_gen, owner)

        # Not my channel: wait for the owner, advertising who I wait for so the
        # deadlock detection can see me. A cycle back to myself has to persist a
        # few consecutive checks (statuses update asynchronously) before I act,
        # to avoid tripping on a transient state.
        timeout = float(os.environ.get('MADSPIN_REFILL_WAIT', '3600'))
        try:
            need_hits = int(os.environ.get('MADSPIN_DEADLOCK_HITS', '10'))
        except (TypeError, ValueError):
            need_hits = 10
        self._set_status('W', owner)
        waited = 0.0
        cycle_hits = 0
        try:
            while self._published_gen(decay_dir) < target_gen:
                reason = None
                if self._read_worker_status(owner) == ('D',):
                    # The owner has finished (or died) without producing this
                    # generation and never will -- e.g. in the max-weight scan an
                    # owner may exhaust its short probe slice before it ever runs
                    # its owned channel dry. I must generate it myself.
                    reason = ("owner %s is DONE but never produced" % owner)
                elif self._wait_cycle_to_self(owner):
                    cycle_hits += 1
                    if cycle_hits >= need_hits:
                        reason = ("deadlock wait-cycle (chain from owner %s "
                                  "loops back to me)" % owner)
                else:
                    cycle_hits = 0
                if reason is not None:
                    # Fail-safe: break the stall by generating the channel here,
                    # abandoning the fixed-owner rule. The size is this worker's
                    # own (efficiency-based), so this one refill's reproducibility
                    # is not guaranteed -- an accepted trade against a hang.
                    logger.warning(
                        "MadSpin worker %s: %s; generating gen %s of pdg %s "
                        "(decay file %s) myself. Seed reproducibility is NOT "
                        "guaranteed for this refill.", self._shard_tag, reason,
                        target_gen, pdg, decay_file_nb)
                    self._set_status('G')
                    self._owner_generate(pdg, decay_file_nb, target_gen, needed)
                    break
                time.sleep(0.1)
                waited += 0.1
                if waited > timeout:
                    raise Exception(
                        "MadSpin worker %s waited %.0fs for owner worker %s to "
                        "generate gen %s of channel pdg %s (decay file %s) and "
                        "gave up. Raise the owner undersize fraction "
                        "(MADSPIN_OWNER_UNDERSIZE) or lower nb_core."
                        % (self._shard_tag, waited, owner, target_gen, pdg,
                           decay_file_nb))
        finally:
            self._set_status('R')
        self._pool_gen[key] = target_gen
        return self._open_refill_slice(decay_dir, target_gen, owner)

    def _unweight_range(self, prod_source, evt_decayfile, output_lhe, ctx):
        """Decay + accept/reject over every production event in ``prod_source``,
        writing accepted events to the open ``output_lhe`` (no banner, no closing
        tag). Returns a small picklable stats dict.

        This is the body of the onshell unweighting loop, formerly inline in
        ``run_onshell``. It is called directly for nb_core==1 and once per shard
        inside each forked worker for nb_core>1. It only touches its arguments
        plus per-instance state that is private after ``fork`` (``self.efficiency``,
        ``self.branching_ratio``, the RNG, and the f2py module)."""
        maxwgt = ctx['maxwgt']
        maxwgts = ctx.get('maxwgts') or []
        sequential = ctx.get('sequential', False)
        decay_dict = ctx['decay_dict']
        drop_prob_per_pdg = ctx['drop_prob_per_pdg']
        mixed_pdgs_set = ctx['mixed_pdgs_set']
        density_method = ctx['density_method']
        density_pole_approximation = ctx['density_pole_approximation']
        density_needs_reshuffle = ctx['density_needs_reshuffle']
        nb_event = ctx['shard_nb_event']
        fixed_order = self.options['fixed_order']

        # Pure-interference mode: the weight is signed and its mean over the
        # decay phase space is zero, so the historical redraw-until-accept --
        # which forces exactly one output event per production event -- is
        # wrong here: it would divide out <|W|>, the local size of the
        # interference, and leave it carried by the sign pattern alone.
        #
        # The mode does not accept/reject at all. It draws ONE decay
        # configuration per production event, keeps it, and writes the fully
        # weighted
        #
        #     w = sigma_parent * BR * W / c
        #
        # with W the signed convolution (wgt*jac) and c = <W> the unrestricted
        # decay-side constant the scan measured. Under MG5's IDWTUP = -4
        # convention (sigma = mean of the weights, not their sum) that makes
        # mean(w) = 0, consistent with the XSECUP = 0 the mode writes, and
        # sum_bin(w)/N_file the interference contribution to that bin in pb --
        # i.e. the file normalises itself, max_weight leaves the normalisation
        # entirely, and every production event is used instead of the 3-9% an
        # accept/reject kept. See doc/madspin_sequential_plan.md section 13.13.
        #
        # ``decay_output = unweighted`` selects the other
        # representation of the same estimator (section 13.17): keep the
        # accept/reject, but on |W| and with ONE draw -- nothing is written on
        # rejection, so the keep rate carries <|W|> -- and give each accepted
        # event the constant magnitude
        #
        #     w = +- sigma_parent * BR * <|W|> / c
        #
        # The bound M the acceptance uses cancels between the acceptance
        # probability |W|/M and the resulting file size N_file = N*<|W|>/M, so
        # this normalisation contains no max_weight either. It is *not* the
        # historical redraw-until-accept, which would force one event per
        # production point and divide <|W|> out altogether.
        pure_interference = bool(self._pure_interference())
        # decay_output = weighted rides the SAME path as the fully weighted
        # pure-interference output: one draw per production event, no
        # accept/reject, w = w_prod * BR * W / c.  The only differences are
        # that W is not restricted to an interference block (so <W> = c rather
        # than 0, and mean(w) = sigma*BR rather than 0), that <init> keeps its
        # ordinary cross-section, and that nothing here is signed.
        weighted_decay = bool(ctx.get('weighted_decay'))
        pure_interference_c = ctx.get('pure_interference_c')
        pi_unweighted = pure_interference and bool(
            ctx.get('pure_interference_unweighted'))
        # No ordinary accept/reject is made below in any of these modes: the
        # two weighted paths keep every trial outright, and the 'unweighted'
        # interference path has ALREADY made its own decision (on |W|, one
        # draw) by the time the test is reached, so falling through to the
        # signed test there would reject every negative weight a second time.
        no_joint_test = pure_interference or weighted_decay
        if (pure_interference or weighted_decay) and not pure_interference_c:
            raise self.InvalidCmd(
                "MadSpin: the normalisation constant c = <W> is missing; the "
                "weights cannot be normalised. This is an internal error -- "
                "the maximum-weight scan measures it.")
        pi_w0_factor = 0.0     # <|W|>/c: the constant |w|/(sigma*BR) of the
                               # 'unweighted' output
        if pi_unweighted:
            absw = ctx.get('pure_interference_absw')
            if not absw:
                raise self.InvalidCmd(
                    "MadSpin: pure_interference + decay_output = unweighted "
                    "needs <|W|>, the decay-phase-space mean of the absolute "
                    "convolution, and the maximum-weight scan produced none. "
                    "This is an internal error -- the scan measures it beside "
                    "c.")
            if not maxwgt:
                raise self.InvalidCmd(
                    "MadSpin: pure_interference + decay_output = unweighted "
                    "needs a positive maximum weight to unweight |W| against, and the "
                    "scan produced %r." % (maxwgt,))
            pi_w0_factor = absw / pure_interference_c
        nb_pi_dead = 0     # trials whose convolution was not a finite number:
                           # they are written with weight 0, and an all-dead
                           # sample is a bug rather than a physics statement
        nb_pi_reject = 0   # 'unweighted' only: production events whose single
                           # decay draw failed the |W|/M test and wrote nothing
        nb_pi_overflow = 0 # 'unweighted' only: |W| above the bound. Unlike the
                           # fully weighted output the bound is live again here
                           # -- the acceptance probability clips at 1 -- so an
                           # under-estimated max_weight biases the sample and
                           # has to be counted and reported.
        nb_overflow_joint = 0  # joint accept/reject trials above ``maxwgt``.
                           # Same story as nb_pi_overflow, for the ordinary
                           # (unsigned) joint test, which had no counter at all.
        # ---- the overweight safety net (doc/madspin_sequential_plan.md,
        # section 14) ----------------------------------------------------------
        # Every accept/reject below stops on a trial with probability
        # min(1, w/C); when w > C that probability clips at 1 and the excess is
        # silently dropped. Writing such an event with weight max(1, w/C)
        # restores the correct shape exactly, because min(1,x)*max(1,x) = x. The
        # carried factor rides on the branching ratio (see ``br`` below), so it
        # reaches ``full_evt.wgt`` and every ``parse_reweight()`` entry through
        # the same multiplication -- and it is the *literal* 1.0, never a
        # division, whenever nothing overflowed, so the written weights are
        # bit-identical to the clipping ones on the overwhelmingly common path.
        #
        # The factor itself is always > 1 and unsigned: every accept/reject
        # below tests a MATRIX-ELEMENT weight (a ratio of densities times
        # jacobians, positive by construction, and in the pure-interference
        # mode the modulus |W| the test itself uses), never the event's LHE
        # weight. So a negative production weight -- an MC@NLO counter-event --
        # keeps its sign and only grows in magnitude, which is the whole point:
        # the carry says "this event should have counted more", not "this event
        # is positive".
        #
        # The ACCOUNTING, though, cannot be a count. Under IDWTUP = -4 the
        # cross-section is the mean of the weights, and carrying does not change
        # the number of events, so the shift it restores is
        #     d(sum w) / sum w      with sum w over the file as it would have
        #                           been written WITH clipping ('nominal').
        # For a unit-weight sample that is exactly sum(factor - 1)/n_written,
        # the number this used to print; with counter-events in the sample it is
        # not, because the excess of a negative event subtracts. And sum w can
        # be zero by construction (pure_interference), so the denominator has to
        # be tested against its OWN Monte Carlo error sqrt(sum w^2) -- the same
        # z that _report_pure_interference uses for the zero-cross-section
        # check -- before it can be normalised against; hence the second moment
        # here, and sum|w| as the fallback scale that cannot cancel.
        nb_overweight = 0        # written events carrying a non-unit factor
        nb_overweight_nwa = 0    # ... of which sat in the region where the
                                 # narrow-width approximation is invalid by
                                 # construction (_near_nwa_threshold)
        nb_overweight_res = 0    # ... and of which sat on the production
                                 # matrix element's own resonance
                                 # (_near_production_resonance). Exclusive with
                                 # the line above, threshold winning.
        max_overweight = 1.0     # the largest single factor carried
        sum_overweight_dw = 0.0     # sum of (factor - 1) * w_nominal: the signed
                                    # weight the clipping used to throw away
        sum_overweight_dabs = 0.0   # ... and the same with |w_nominal|, which
                                    # does not cancel between counter-events
        sum_nom = 0.0            # sum of the weights clipping WOULD have written
        sum_abs_nom = 0.0        # ... their absolute values
        sum_sq_nom = 0.0         # ... and their squares, for the MC error on sum
        sum_w = 0.0        # signed weight sum, for the zero-cross-section check
        sum_w2 = 0.0       # its second moment: no cancellation, so the MC error
        nb_try = 0
        nb_loose_skip = 0  # events dropped to equalize BRs (fake-decay path)
        sequential_stats = collections.defaultdict(int)
        curr_event = -1    # guard: an (over-sharded) empty range leaves it unset
        start = time.time()
        for curr_event, production in enumerate(prod_source):
            if fixed_order:
                production, counterevt = production[0], production[1:]
            if (curr_event and curr_event % 10 == 0
                    and float(str(curr_event)[1:]) == 0
                    and getattr(self, '_shard_tag', None) in (None, 0)):
                # only one worker prints progress -- the others would just
                # interleave the same lines
                if sequential and sequential_stats:
                    # per-particle unweighting cost: how many decay events each
                    # decaying particle burned per accepted event. Clearer than a
                    # single number, and it is the sum of these -- and reported
                    # once per event, i.e. gated on the slot-0 pass, not per slot.
                    positions = sorted(int(k.rsplit('_', 1)[1]) for k in
                                       sequential_stats if k.startswith('nb_try_'))
                    per = ' '.join(
                        'p%d=%.2f' % (k, sequential_stats['nb_try_%d' % k]
                                      / float(curr_event + 1)) for k in positions)
                    logger.info("decaying event number %s/%s. Decay events per "
                                "accepted event, per particle: %s [%s s]"
                                % (curr_event, nb_event, per, time.time()-start))
                elif self.efficiency:
                    logger.info("decaying event number %s/%s. Trials per event: "
                                "%.4g [%s s]" % (curr_event, nb_event, 1/self.efficiency,
                                                 time.time()-start))

            # BR-equalization: drop this event with probability
            # 1 - br_pdg / max_br when this production process has a smaller
            # total BR than the largest one in the mixed sample. Done before
            # any matrix-element work so dropped events are cheap.
            if drop_prob_per_pdg:
                evt_mixed_pdgs = [p.pid for p in production
                                  if int(p.status) == 1 and p.pid in mixed_pdgs_set]
                if len(evt_mixed_pdgs) == 1:
                    drop = drop_prob_per_pdg[evt_mixed_pdgs[0]]
                    if drop > 0 and random.random() < drop:
                        nb_loose_skip += 1
                        continue
                elif len(evt_mixed_pdgs) > 1:
                    raise self.InvalidCmd(
                        "BR equalization for events with more than one "
                        "mixed-pdg decaying particle is not implemented yet "
                        "(event %d has pdgs=%s). Please report this case." %
                        (curr_event, evt_mixed_pdgs))

            if sequential:
                # Accept/reject one decaying particle at a time. Every
                # production event yields a set (each slot is redrawn until it
                # is accepted), so there is no outer rejection loop here.
                seq_stats = collections.defaultdict(int)
                decays = self.sequential_accept_reject(
                                production, evt_decayfile, maxwgts,
                                nb_event - curr_event, stats=seq_stats,
                                decay_dict=decay_dict)
                if decays is None:
                    # nothing to decay in this production event. It still goes
                    # into the file, so it belongs to the normalisation the
                    # overweight report divides by.
                    sum_nom += production.wgt
                    sum_abs_nom += abs(production.wgt)
                    sum_sq_nom += production.wgt * production.wgt
                    output_lhe.write_events(production)
                    continue
                # the accepted chain's carried overweight (mass stage x angle
                # stages, composed multiplicatively inside the chain -- see
                # sequential_accept_reject). Popped rather than merged: the
                # remaining keys are additive counters.
                carry = seq_stats.pop('overweight_factor', 1.0)
                for key, value in seq_stats.items():
                    sequential_stats[key] += value
                nb_try += sum(v for k, v in seq_stats.items()
                              if k.startswith('nb_try_'))
                full_evt = lhe_parser.Event(str(production))
                full_evt = full_evt.add_decays(decays)
                if density_needs_reshuffle:
                    # the decays already carry their sampled virtualities; this
                    # is the single production reshuffling of the chain, which
                    # sequential_accept_reject has already checked is possible
                    full_evt.reshuffle_production()
                self.efficiency = float(curr_event + 1) / nb_try if nb_try else 1.0
                br = self.branching_ratio
                # what this event would have been written with had the excess
                # been clipped -- taken BEFORE br absorbs the carry, so it needs
                # no division and keeps its sign. When carry is 1 this is the
                # identical multiplication the line below performs, so the
                # written weight is unaffected by its existence.
                w_nom = full_evt.wgt * br
                sum_nom += w_nom
                sum_abs_nom += abs(w_nom)
                sum_sq_nom += w_nom * w_nom
                if carry != 1.0:
                    # the signed difference is what the sample's cross-section
                    # gains; the unsigned one is the same quantity with the
                    # counter-events' cancellation taken out.
                    sum_overweight_dw += w_nom * (carry - 1.0)
                    sum_overweight_dabs += abs(w_nom) * (carry - 1.0)
                    # exactly one multiplication either way: br is the same
                    # float object as self.branching_ratio when nothing
                    # overflowed, so this branch is the only thing that can
                    # move a written weight.
                    br = br * carry
                    nb_overweight += 1
                    if self._near_nwa_threshold(production, evt_decayfile):
                        nb_overweight_nwa += 1
                    elif self._near_production_resonance(full_evt, production,
                                                         evt_decayfile):
                        nb_overweight_res += 1
                    if carry > max_overweight:
                        max_overweight = carry
                full_evt.wgt *= br
                wgts = full_evt.parse_reweight()
                for key in wgts:
                    wgts[key] *= br
                self._add_polarization_weights(
                    full_evt, getattr(self, '_pol_weight_ratios', None))
                output_lhe.write_events(full_evt)
                continue

            # Per-production-event cache reused across rejection retries.
            prod_density_cached = None
            pi_factor = 1.0   # W/c ('weighted') or +-<|W|>/c ('unweighted') in
                              # the pure-interference mode, 1 elsewhere
            carry = 1.0       # overweight safety net: max(1, w/C) of the trial
                              # this production event stops on. SET (never
                              # accumulated) at the acceptance, because a
                              # rejected trial is redrawn from scratch and
                              # contributes nothing to the event that is written.
            pi_rejected = False  # 'unweighted': the single draw failed, so this
                                 # production event writes nothing at all
            # Consecutive trials whose matrix-element weight was not a finite
            # positive number. This `while 1` has no other exit than an
            # acceptance, so without it a structurally zero weight loops for
            # ever (draining and regenerating the decay pools as it goes). Reset
            # by the first positive weight, hence blind to a merely low
            # acceptance. Same code in the forked workers: _unweight_range is
            # the body of both the nb_core==1 and the nb_core>1 paths.
            dead_trials = 0

            while 1:
                nb_try += 1
                decays = self.get_decay_from_file(production, evt_decayfile, nb_event-curr_event)
                # In density mode do not do full event construction before accept/reject
                build_event = (not density_method) or self.options['fixed_order']

                # Offshell (madspin/full) density: the reshuffle of the chain
                # happens INSIDE get_onshell_evt_and_wgt (its jacobian is folded
                # into wgt there). It mutates the production event in place, so pass
                # a per-trial onshell copy -- otherwise a rejected trial's offshell
                # kinematics leak into the next trial's denominator ME and reshuffle
                # jacobian (mass_shuffle's chi telescopes, so the kinematics are
                # unchanged, but the jacobian and MEdenom_prod are not).
                offshell_density = density_method and not density_pole_approximation
                prod_trial = lhe_parser.Event(str(production)) if offshell_density else production

                if prod_density_cached is None or not density_pole_approximation:
                    full_evt, wgt, prod_density_cached = self.get_onshell_evt_and_wgt(
                        prod_trial, decays, decay_dict, build_event=build_event)
                else:
                    full_evt, wgt, _ = self.get_onshell_evt_and_wgt(
                        prod_trial, decays, decay_dict, prod_density_cached, build_event=build_event)
                jac = 1
                if (density_needs_reshuffle and not offshell_density
                        and self.options['density_keep_jacobian']):
                    # PA with explicit jacobian tracking: reshuffle BEFORE
                    # accept/reject so the reshuffling jacobian enters the weight
                    # (wgt*jac). Build on a fresh copy because this runs on every
                    # trial, including rejected ones (must not mutate the shared
                    # production event). The offshell/madspin path does NOT enter
                    # here: its reshuffle jacobian is already inside wgt.
                    full_evt = lhe_parser.Event(str(production))
                    full_evt = full_evt.add_decays(decays)
                    jac = full_evt.reshuffle_production()

                test = wgt*jac
                if pure_interference or weighted_decay:
                    signed = float(getattr(test, 'real', test))
                    dead = not math.isfinite(signed)
                    if weighted_decay and signed <= 0:
                        # Outside the interference mode W is a contraction of
                        # two positive-semidefinite matrices and cannot be
                        # negative; a non-positive one means jac <= 0, i.e. a
                        # mass set the production could not be reshuffled onto.
                        # The accept/reject treats that as a rejection and
                        # redraws. There is no redraw here, and the event that
                        # would be written carries the FAILED reshuffle's
                        # kinematics -- so it is written with weight 0 (which
                        # is also what that region contributes to the integral)
                        # and counted, rather than given the negative weight
                        # that would make the bookkeeping add up on an
                        # unphysical event.
                        dead = True
                    if dead:
                        nb_pi_dead += 1
                        signed = 0.0
                    if not pi_unweighted:
                        # no accept/reject: the signed convolution goes onto
                        # the weight, scaled by the decay-side constant c
                        pi_factor = signed / pure_interference_c
                    else:
                        # unweighted up to a sign. ONE draw, accepted with
                        # probability |W|/M; nothing is written on rejection,
                        # so the keep rate carries <|W|>. The magnitude is the
                        # same for every accepted event and M cancels out of
                        # it. A negative weight is normal here, which is why
                        # the test and the overflow count are both on |W| --
                        # and why _dead_trial, whose whole premise is that a
                        # non-positive weight is structurally dead, is not on
                        # this path at all (nb_pi_dead counts the genuinely
                        # dead, non-finite trials instead).
                        if abs(signed) > maxwgt:
                            nb_pi_overflow += 1
                        if random.random() * maxwgt >= abs(signed):
                            pi_rejected = True
                            break
                        if abs(signed) > maxwgt:
                            # the |W|/M test clipped at 1: carry the excess on
                            # the magnitude instead of dropping it. The 'two
                            # weight magnitudes' claim of the banner note
                            # acquires an exception here, which the note says.
                            # <|W|> = (N_file/N_drawn)*M is unaffected: N_file
                            # is a COUNT, and the estimator only needs
                            # E[min(1,x)*max(1,x)] = E[x] -- which is exactly
                            # what carrying restores.
                            carry = abs(signed) / maxwgt
                        pi_factor = math.copysign(pi_w0_factor, signed)
                else:
                    # ``wgt`` alone, not ``wgt*jac``: a zero/-1 jacobian is an
                    # ordinary rejection (a mass set the production cannot be
                    # reshuffled onto), which is a legitimate, transient state.
                    # A zero ``wgt`` is the matrix element itself being dead.
                    dead_trials = self._dead_trial(dead_trials, wgt,
                                                   'the joint accept/reject')

                if no_joint_test or random.random()*maxwgt < test:
                    if not no_joint_test and maxwgt > 0 and test > maxwgt:
                        # The joint test clipped at probability 1 (it always
                        # does when test > maxwgt, since random.random() < 1).
                        # Carry max(1, test/maxwgt) instead of dropping the
                        # excess. This branch is the ONLY joint-path
                        # overflow -- there was no counter here at all before.
                        nb_overflow_joint += 1
                        carry = test / maxwgt
                        logger.debug('joint accept/reject: weight %s above its '
                                     'max %s, carried on the event weight',
                                     test, maxwgt)
                    if offshell_density:
                        # prod_trial has already been reshuffled internally (its
                        # jacobian is in wgt); build the event to write out from the
                        # reshuffled copy, without reshuffling a second time. If
                        # get_onshell already built it (fixed_order / density_debug),
                        # reuse that event rather than build the same one twice.
                        if full_evt is None:
                            full_evt = lhe_parser.Event(str(prod_trial))
                            full_evt = full_evt.add_decays(decays)
                    elif (density_needs_reshuffle
                            and density_pole_approximation
                            and not self.options['density_keep_jacobian']):
                        # PA with density_keep_jacobian = False (NOT the default;
                        # the default is the branch above, which reshuffles before
                        # the test so the jacobian enters the weight): reshuffle
                        # AFTER acceptance, so the reshuffle is only a kinematic
                        # dressing of the accepted event. The Breit-Wigner sampling
                        # jacobian is already folded into wgt, and this mode
                        # deliberately keeps the reshuffling jacobian out of the
                        # accept/reject test. For 2 -> 1 production no mass was
                        # sampled and reshuffle_production short-circuits
                        # (NWA-style no-op).
                        full_evt = lhe_parser.Event(str(production))
                        full_evt = full_evt.add_decays(decays)
                        jac = full_evt.reshuffle_production()
                    elif full_evt is None:
                        # No-reshuffle density mode still needs a concrete event to write out.
                        if density_method and density_pole_approximation:
                            full_evt = lhe_parser.Event(str(production))
                        else:
                            full_evt = production
                        full_evt = full_evt.add_decays(decays)
                    if self.options['fixed_order']:
                        full_evt = [full_evt] + [evt.add_decays(decays) for evt in counterevt]
                    break
                #else:
                #    misc.sprint('fail-> retry')
            if pi_rejected:
                # 'unweighted' pure interference: one draw, and it failed. Write
                # nothing and move to the next production event -- redrawing
                # here would force one output per production point and divide
                # out <|W|>, which is precisely the quantity the keep rate is
                # carrying (section 13.7b).
                nb_pi_reject += 1
                continue
            # Efficiency = accepted / trials (+1 because current event is already accepted)
            self.efficiency = float(curr_event + 1) / nb_try
            #if density_method:
            #    full_evt.reshuffle_production()
            # pure interference: W/c rides on the branching ratio, so it reaches
            # the event weight and every entry of the multi-weight block through
            # the same multiplication. pi_factor is 1.0 in every other mode, so
            # nothing else moves.
            br = self.branching_ratio * pi_factor \
                if (pure_interference or weighted_decay) \
                else self.branching_ratio
            # ``carry`` is unsigned even in the pure-interference mode: the
            # |W|/M test clips on the MODULUS, so the factor is built from
            # abs(signed) and the sign is carried exactly once, by pi_factor,
            # inside br. w_nom is therefore the signed weight this event would
            # have been written with under clipping.
            w_nom = (full_evt[0] if self.options['fixed_order']
                     else full_evt).wgt * br
            sum_nom += w_nom
            sum_abs_nom += abs(w_nom)
            sum_sq_nom += w_nom * w_nom
            if carry != 1.0:
                # the overweight safety net rides the same hook, for the same
                # reason: one multiplication that reaches both full_evt.wgt and
                # every parse_reweight() entry. Guarded so that the no-overflow
                # path -- which is essentially the whole sample -- performs the
                # identical arithmetic it did before this existed.
                sum_overweight_dw += w_nom * (carry - 1.0)
                sum_overweight_dabs += abs(w_nom) * (carry - 1.0)
                br = br * carry
                nb_overweight += 1
                if self._near_nwa_threshold(production, evt_decayfile):
                    nb_overweight_nwa += 1
                elif self._near_production_resonance(full_evt, production,
                                                     evt_decayfile):
                    nb_overweight_res += 1
                if carry > max_overweight:
                    max_overweight = carry
            if self.options['fixed_order']:
                for evt in full_evt:
                    # change the weight associated to the event
                    evt.wgt *= br
                    wgts = evt.parse_reweight()
                    for key in wgts:
                        wgts[key] *= br
                if pure_interference or weighted_decay:
                    sum_w += full_evt[0].wgt
                    sum_w2 += full_evt[0].wgt ** 2
            else:
                # change the weight associated to the event
                full_evt.wgt *= br
                wgts = full_evt.parse_reweight()
                for key in wgts:
                    wgts[key] *= br
                if pure_interference or weighted_decay:
                    sum_w += full_evt.wgt
                    sum_w2 += full_evt.wgt ** 2
            self._add_polarization_weights(
                full_evt, getattr(self, '_pol_weight_ratios', None))

            output_lhe.write_events(full_evt)

        worker = getattr(self, '_shard_tag', None)
        if worker is None:
            logger.info("decay unweighting done. [%.1f s]" % (time.time()-start))
        else:
            logger.info("worker %s of %s done. [%.1f s]"
                        % (worker, getattr(self, '_shard_nb_core', '?'),
                           time.time()-start))
        n_processed = curr_event + 1
        return dict(n_processed=n_processed,
                    n_written=n_processed - nb_loose_skip - nb_pi_reject,
                    nb_try=nb_try,
                    nb_loose_skip=nb_loose_skip,
                    nb_pi_reject=nb_pi_reject,
                    nb_pi_overflow=nb_pi_overflow,
                    # picklable and merged additively over the forked shards, so
                    # one shard or many gives the identical zero-cross-section
                    # test (section 13.8)
                    nb_pi_dead=nb_pi_dead,
                    # overweight safety net: additive over shards, so one shard
                    # or many gives the identical end-of-run number
                    nb_overflow_joint=nb_overflow_joint,
                    nb_overweight=nb_overweight,
                    nb_overweight_nwa=nb_overweight_nwa,
                    nb_overweight_res=nb_overweight_res,
                    sum_overweight_dw=float(sum_overweight_dw),
                    sum_overweight_dabs=float(sum_overweight_dabs),
                    sum_nom=float(sum_nom),
                    sum_abs_nom=float(sum_abs_nom),
                    sum_sq_nom=float(sum_sq_nom),
                    max_overweight=float(max_overweight),
                    sum_w=float(sum_w),
                    sum_w2=float(sum_w2),
                    sequential_stats=dict(sequential_stats))

    def _report_sequential_stats(self, stats_list, n_written):
        """Per-slot report of the sequential accept/reject.

        The per-slot acceptance is what the pool ladder is meant to predict, and
        the overflow count is the safety net: a weight above its slot's maximum
        means the bound was under-estimated, which biases the sample silently,
        so it is reported loudly rather than left in a debug line.
        """
        merged = collections.defaultdict(int)
        for stats in stats_list:
            for key, value in (stats.get('sequential_stats') or {}).items():
                merged[key] += value
        if not merged:
            return
        rejects = merged.get('nb_mass_reject', 0)
        restarts = merged.get('nb_production_restart', 0)
        exact_restarts = merged.get('nb_exact_restart', 0)
        angle_tries = merged.get('nb_angleset_try', 0)
        if angle_tries:
            # two_stage: one bound over all the angles
            logger.info("MadSpin sequential angle stage: %.2f angle sets per "
                        "accepted event (%d drawn, %d rejected)",
                        float(angle_tries) / n_written if n_written
                        else float('inf'), angle_tries,
                        merged.get('nb_angle_reject', 0))
        if rejects or exact_restarts or merged.get("nb_angle_reject", 0):
            # the offshell mass-set stage: how many virtuality sets (each one a
            # production reshuffling and a production density) are drawn per
            # accepted event
            # An angle-set rejection does not cost a mass set: the set is kept
            # and only the decays are drawn again, which is the whole point of
            # two_stage. Counting those here would inflate the
            # production-side work by the angle-stage rejections.
            drawn = rejects + restarts + exact_restarts + n_written
            logger.info("MadSpin sequential mass stage: %.2f mass sets per "
                        "accepted event (%d drawn, %d rejected%s)",
                        float(drawn) / n_written if n_written else float('inf'),
                        drawn, rejects,
                        ', %d dropped by a rejected decay' % exact_restarts
                        if exact_restarts else '')
        per_event = merged.get('nb_mass_bound_event', 0)
        global_bound = merged.get('nb_mass_bound_global', 0)
        if per_event or global_bound:
            logger.info(
                "MadSpin sequential mass stage: %d/%d production events used "
                "the per-event bound%s", per_event, per_event + global_bound,
                '' if not global_bound else
                ' (%d fell back to the probe\'s global maximum weight)'
                % global_bound)
        positions = sorted(int(k.rsplit('_', 1)[1]) for k in merged
                           if k.startswith('nb_try_'))
        for position in positions:
            tries = merged['nb_try_%d' % position]
            extra = []
            redraws = merged.get('nb_mass_redraw_%d' % position, 0)
            if redraws:
                extra.append('%d mass redraws' % redraws)
            infeasible = merged.get('nb_infeasible_%d' % position, 0)
            if infeasible:
                extra.append('%d decays that could not be reshuffled' % infeasible)
            overflows = merged.get('nb_overflow_%d' % position, 0)
            if overflows:
                extra.append('%d ABOVE the maximum weight' % overflows)
            logger.info(
                "MadSpin sequential slot %d: %.2f decay events per accepted one"
                " (%d drawn)%s", position,
                float(tries) / n_written if n_written else float('inf'), tries,
                (' [%s]' % ', '.join(extra)) if extra else '')
        if restarts:
            logger.info("MadSpin sequential: %d chains restarted on a mass set "
                        "the production could not reshuffle", restarts)
        checks = merged.get('nb_identity_check', 0)
        if checks:
            mean = merged.get('identity_ratio_sum', 0.0) / checks
            variance = (merged.get('identity_ratio_sqsum', 0.0) / checks
                        - mean * mean)
            spread = (math.sqrt(max(0.0, variance)) / abs(mean)
                      if mean else float('inf'))
            # density_tolerance, like density_debug: the two routes evaluate the
            # same matrix elements through different code, and the density
            # matrices are single precision, so agreement is bounded by float32
            # epsilon (~1.2e-7) and not by the physics
            if spread > self.options['density_tolerance']:
                logger.critical(
                    "MadSpin sequential: the weight identity FAILED on %d "
                    "accepted chains -- the chain weight is not proportional "
                    "to the joint weight (relative spread of the ratio %.3g, "
                    "mean %.10g). This scheme is not sampling the joint "
                    "distribution.", checks, spread, mean)
            else:
                logger.info("MadSpin sequential: weight identity verified on "
                            "%d accepted chains -- chain weight / joint weight "
                            "constant to %.3g (ratio %.10g)",
                            checks, spread, mean)
        total_overflow = sum(v for k, v in merged.items()
                             if k.startswith('nb_overflow_'))
        if total_overflow:
            # Same split as _report_overweight, and for the same reason: an
            # exceedance at threshold is the narrow-width approximation running
            # out, not an under-estimated bound. The two counts are in
            # different units -- this one counts STAGE weights, the overweight
            # line counts written EVENTS -- so the note is only allowed to
            # decide the volume of this line, never to be read as its
            # breakdown.
            near, res, far = self._nwa_threshold_split(stats_list)
            msg = ("MadSpin sequential: %d weights exceeded their stage "
                   "maximum (mass set / angles / per particle). " % total_overflow)
            if (near or res) and not far:
                msg += ("Every event that carried one sits within %s of the "
                        "sum-of-poles threshold, or on a resonance of the "
                        "production matrix element itself -- regions where the "
                        "narrow-width approximation is invalid by "
                        "construction and no accept/reject bound built on it "
                        "can dominate; see the overweight line below."
                        % self._nwa_threshold_margin())
                logger.info(msg)
            else:
                msg += ("That bound is under-estimated: to go back to unit "
                        "weights everywhere, raise nb_sigma or "
                        "Nevents_for_max_weight, or set unweighting = joint.")
                logger.warning(msg)

    # How many Monte Carlo errors the summed weight has to be away from zero
    # before it may be used as a denominator. sum w = 0 is not a pathology to be
    # detected by a magnitude cut -- pure_interference produces it BY DESIGN, and
    # an ordinary sample only ever approaches it by accident -- so the test is
    # the same z = S / sqrt(sum w^2) that _report_pure_interference already uses
    # for its zero-cross-section check. An unweighted sample of N events has
    # z = sqrt(N), so this never fires on one; a pure-interference sample has
    # z = O(1) and always does.
    _OVERWEIGHT_MIN_Z = 5.0

    # How much the carried excess has to be worth, in per-cent of whatever
    # denominator the overweight line was allowed to quote, before the region
    # breakdown is printed at all. Below it the head of the line has already
    # said how many events carried one and what it was worth, and a paragraph
    # about WHY only buries that; above it the three-way split still decides
    # both the wording and the volume.
    _OVERWEIGHT_QUIET_PERCENT = 0.5

    # ------------------------------------------------------------------
    # The region where the narrow-width approximation is invalid by
    # construction
    # ------------------------------------------------------------------
    # MadSpin factorises production x decay, and the production side is
    # evaluated with every resonance ON its pole: that is what
    # ``|M_prod|^2_on`` is in the offshell mass weight
    # ``Tr(rho_off)/|M_prod|^2_on``, and what the cached on-shell ``rho`` is
    # under PA. The construction therefore needs the event to be able to put
    # every resonance on its pole at once, with phase space left over.
    #
    # It cannot, once ``sqrt(shat)`` comes down onto the sum of the poles.
    # There the reference configuration the whole thing is normalised to sits
    # at the edge of -- or outside -- the region the sample can reach: the
    # Breit-Wigner windows stop being set by ``BW_cut`` and start being cut off
    # by the energy budget, and the jacobian of the reshuffle that moves the
    # production onto a drawn mass set diverges, because there is no recoil
    # momentum left to absorb the change. Neither of those is a bug, and
    # neither is fixable by a better accept/reject bound: they are the
    # approximation being asked for something it does not have.
    #
    # That matters for one thing only, here: how loudly an overweight in this
    # region should be reported. Since PR #375 an overweight is CARRIED on the
    # event weight rather than clipped, so it is no longer a silent bias
    # anywhere -- and here it is not even a surprise. Outside the region it
    # still is: there it says the bound does not dominate for a reason nobody
    # has explained, and it keeps the loud line.
    #
    # The margin is measured in the summed WIDTHS of the resonances the event
    # decays, because the width is the only scale in the problem that says how
    # far off its pole a resonance is allowed to go. One summed width is the
    # statement "this event does not have even one width of room to share".
    #
    # Measured, ``p p > t t~`` at 6.5+6.5 TeV, ``spinmode madspin``,
    # ``BW_cut = 15``, 50 000 production events x 400 free mass sets each
    # (2.0e7 draws) against the shipped global bound: every one of the 239
    # over-bound draws, on every one of the 14 events that produced one, sits
    # at ``sqrt(shat) - 2 m_t < 0.24`` summed widths -- a factor four inside
    # this margin -- while the region itself holds 0.31 % of the sample. See
    # doc/madspin_sequential_plan.md section 15.
    _NWA_THRESHOLD_WIDTHS = 1.0

    def _near_nwa_threshold(self, production, evt_decayfile):
        """Is this production event inside the region described above?

            sqrt(shat)  <  sum_r pole_r  +  _NWA_THRESHOLD_WIDTHS * sum_r Gamma_r

        with both sums over the final-state particles this event actually
        decays, counted with multiplicity -- so both tops of a ``t t~`` event
        enter, and a ``t t~`` event with only one decay line enters once. The
        "does this particle decay" test is ``_decaying_pdgs``'s, so a pdg with
        an empty pool is not counted for a decay that will not happen.

        Cached on the production event. False -- never an exception -- when
        anything it needs is missing: this decides how loudly a diagnostic is
        printed and must not be able to stop a run.
        """
        cached = getattr(production, '_ms_near_nwa_threshold', None)
        if cached is not None:
            return cached
        answer = False
        try:
            pole_sum = 0.0
            width_sum = 0.0
            decaying = False
            for particle in production:
                if int(particle.status) != 1:
                    continue
                pdg = particle.pdg
                if pdg not in evt_decayfile or not len(evt_decayfile[pdg]):
                    continue
                decaying = True
                pole_sum += self.banner.get('param', 'mass', abs(pdg)).value
                width_sum += self.banner.get('param', 'decay', abs(pdg)).value
            sqrts = production.sqrts
            if decaying and sqrts and sqrts > 0:
                answer = bool(sqrts < pole_sum
                              + self._NWA_THRESHOLD_WIDTHS * width_sum)
        except (AttributeError, KeyError, TypeError, ValueError):
            answer = False
        production._ms_near_nwa_threshold = answer
        return answer

    # ------------------------------------------------------------------
    # The second region an overweight can come from: the production matrix
    # element's OWN resonance
    # ------------------------------------------------------------------
    # This one has nothing to do with threshold and does not exist at all in
    # ``2 -> 2``. Once the production process has a final-state particle
    # besides the resonances -- a jet -- its matrix element contains a
    # propagator of the resonance itself, on the line the jet is radiated
    # from, with virtuality
    #
    #     (p_r + p_j)^2  =  m_r^2 + 2 p_r.p_j .
    #
    # With ``m_r`` ON its pole that is ``>= M_r^2`` for any real jet, so the
    # propagator can never go on shell: the singularity sits exactly on the
    # boundary of phase space and is unreachable. Sampling ``m_r`` BELOW the
    # pole -- which is what ``BW_cut`` is for -- opens it: the equation
    # ``2 p_r.p_j = M_r^2 - m_r^2`` now has solutions, and on them the
    # production matrix element is a Breit-Wigner peak regulated only by
    # ``M_r Gamma_r``. The weight there is correct; it is simply enormous, and
    # a single run-level bound cannot dominate it.
    #
    # In ``2 -> 2`` the only internal resonance line is t-channel,
    # ``(p_in - p_r)^2 = m_r^2 - 2 p_in.p_r < m_r^2 <= M_r^2``, so it is
    # spacelike and this region does not exist. That asymmetry is the whole
    # reason a ``2 -> 3`` production overweights 26x more often than the same
    # ``2 -> 2`` one.
    #
    # Measured, ``p p > t t~ j`` at 6.5+6.5 TeV, ``spinmode madspin``,
    # ``unweighting joint``, ``BW_cut = 15``, 300 000 events: **all 265** of
    # the run's carried overweights have one resonance within 5.2 M_r Gamma_r
    # of this pole (51 % within one, the largest -- factor 48.9 -- at 0.12),
    # against 3.30 for the trials that merely came close to the bound. Their
    # production reshuffling jacobian is 1.07 (max 2.47), i.e. the threshold
    # mechanism above is not involved. The region is reachable inside
    # ``BW_cut = 15`` for 3.5 % of the production events and essentially never
    # below ``BW_cut = 8`` -- see doc/madspin_sequential_plan.md section 17.
    #
    # The margin is 10 and not the 5.2 that population happens to fill: twice
    # the measured envelope, so the tag is not fitted to one sample, and still
    # specific -- the fraction of joint TRIALS that land inside it is 9.2e-4
    # (against 2.9e-4 at 5 and 5.3e-5 at 1), i.e. a thousand times rarer than
    # the tag firing would need to be to silence an overweight by coincidence.
    _PRODUCTION_RESONANCE_WIDTHS = 10.0

    def _near_production_resonance(self, full_evt, production, evt_decayfile):
        """Does this accepted event sit on the production process's own
        resonance, i.e. is there a decayed resonance ``r`` and another
        production-level final state ``k`` with

            |m^2(r + k) - M_r^2|  <=  _PRODUCTION_RESONANCE_WIDTHS . M_r Gamma_r

        evaluated at the virtuality the event actually carries?

        The first ``len(production)`` entries of ``full_evt`` are the
        production event's own particles (``add_decay_to_particle`` appends the
        decay products after them and flips the parent to status 2), so this is
        a pairwise invariant mass over the production final state, on the
        reshuffled momenta -- O(n^2) four-vector arithmetic on an event that is
        already built, and only ever on an event that overflowed.

        False -- never an exception -- when anything it needs is missing: like
        ``_near_nwa_threshold`` this decides how loudly a diagnostic prints and
        must not be able to stop a run.
        """
        try:
            # fixed_order hands in the event GROUP (born + counter-events);
            # they share the draw, so the born one answers for all of them.
            # Tested on the element and not with isinstance(list): Event is
            # itself a list of Particle, so that test is always true.
            if full_evt and isinstance(full_evt[0], lhe_parser.Event):
                full_evt = full_evt[0]
            parts = list(full_evt)[:len(production)]
            finals = [q for q in parts if int(q.status) in (1, 2)]
            if len(finals) < 3:
                # 2 -> 2: no jet to radiate the resonance off, so the internal
                # propagator is t-channel and cannot go on shell
                return False
            for r in finals:
                if int(r.status) != 2:
                    continue
                pdg = r.pdg
                if pdg not in evt_decayfile or not len(evt_decayfile[pdg]):
                    continue
                pole = self.banner.get('param', 'mass', abs(pdg)).value
                width = self.banner.get('param', 'decay', abs(pdg)).value
                if not pole or not width:
                    continue
                qr = lhe_parser.FourMomentum(r)
                # ``k`` is deliberately NOT restricted to status 1. `status`
                # records whether MadSpin attached a decay to the particle, not
                # whether it was radiated off the ``r`` line, so it is no proxy
                # for the physics: the very same momenta would tag or not tag
                # depending only on which particles the user asked to decay.
                # A partner that is itself a decayed resonance is kept safe by
                # arithmetic instead. ``s2 = (p_r + p_k)^2 >= (m_r + m_k)^2``
                # for any two physical momenta, so the window can only be
                # entered when ``m_r + m_k <= sqrt(M_r^2 + N M_r Gamma_r)``,
                # and MadSpin never samples a mass more than ``BW_cut`` widths
                # below its pole (decay.py: ``m_min = max(m - BW_cut w, 0.5)``).
                # The tightest SM pair, ``r = Z`` with ``k = W``, still misses
                # by 1.6 GeV at the default ``BW_cut = 15`` and only opens at
                # ``BW_cut >= 15.4``; ``t`` with ``W`` needs 20.5 and ``t`` with
                # ``t~`` needs 55.3 -- values MadSpin's own check already calls
                # too large for the narrow-width approximation it factorises
                # with. Where it does open (``W Z j`` with both bosons far off
                # shell) the pairing is the genuine ``W* -> W Z`` production
                # resonance anyway, so excluding status 2 would cost a true
                # positive at exactly the BW_cut where it buys the false one.
                # Pinned by TestProductionResonanceRegion's two-resonance tests.
                for k in finals:
                    if k is r:
                        continue
                    s2 = (qr + lhe_parser.FourMomentum(k)).mass_sqr
                    if abs(s2 - pole * pole) <= (
                            self._PRODUCTION_RESONANCE_WIDTHS * pole * width):
                        return True
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            return False
        return False

    def _nwa_threshold_margin(self):
        """``_NWA_THRESHOLD_WIDTHS`` as it is said out loud."""
        return ('one summed width' if self._NWA_THRESHOLD_WIDTHS == 1
                else '%g summed widths' % self._NWA_THRESHOLD_WIDTHS)

    def _nwa_threshold_split(self, stats_list):
        """(at threshold, on a production resonance, neither) over the carried
        overweights of a run. Exclusive, in that order: an event that is both
        counts as threshold, which is the stronger statement."""
        nb = sum(s.get('nb_overweight', 0) for s in stats_list)
        near = sum(s.get('nb_overweight_nwa', 0) for s in stats_list)
        res = sum(s.get('nb_overweight_res', 0) for s in stats_list)
        return near, res, nb - near - res

    def _nwa_threshold_note(self, near, res, far):
        """The sentence both end-of-run lines append when the split is
        non-trivial. Always quotes every part, so the total the head of the
        line gives stays recoverable from it."""
        if not near and not res:
            return ''
        note = ''
        if near:
            note += ("%d of them are production events within %s of "
                     "the sum-of-poles threshold, "
                     % (near, self._nwa_threshold_margin()))
        if res:
            note += ("%d of them have a resonance and another production-level "
                     "final state whose invariant mass is within %g widths of "
                     "that resonance's own pole. "
                     % (res, self._PRODUCTION_RESONANCE_WIDTHS))
        both = 'either region' if (near and res) else 'that region'
        it = 'those' if (near and res) else 'it'
        if far:
            note += ("The other %d are NOT in %s, and those do say "
                     "the bound is under-estimated: raise nb_sigma or "
                     "Nevents_for_max_weight. " % (far, both))
        else:
            note += ("None of them is outside %s, so nothing here says the "
                     "bound is under-estimated for an unexplained reason. "
                     % it)
        return note

    def _report_overweight(self, stats_list, n_written):
        """The overweight safety net's end-of-run measurement.

        Section 14 of ``doc/madspin_sequential_plan.md``: every accept/reject in
        MadSpin stops on a trial with probability ``min(1, w/C)``, so a trial
        with ``w > C`` is accepted with probability 1 and the excess
        ``w/C - 1`` used to be thrown away silently. It is now written onto the
        event weight, which is exact because ``min(1,x) * max(1,x) = x``; this
        turns what was an unquantified bias into a number, and this is that
        number.

        **The number is a weight, not a count.** Under MG5's ``IDWTUP = -4``
        convention the cross-section is the MEAN of the event weights, and
        carrying changes no event count, so what the clipping used to discard is

            d(sum w) / sum w        both over the file as clipping would have
                                    written it

        with ``d(sum w) = sum_over (factor - 1) * w_nominal``. For a sample of
        identical positive weights that reduces exactly to
        ``sum(factor - 1) / n_written``, which is what an unweighted MadSpin run
        prints. It does **not** reduce to that once the input carries
        counter-events: a negative event whose trial overflowed makes the
        cross-section *more* negative, so its excess subtracts, and quoting a
        count would claim a shift that is not there.

        ``sum w`` is also zero by construction under ``pure_interference``, so it
        is only used as a denominator when it is at least ``_OVERWEIGHT_MIN_Z``
        of its own Monte Carlo errors away from zero. When it is not, the shift
        is quoted against ``sum |w|`` -- which cannot cancel -- and the line says
        which convention it used, so the two are never confused.

        The line has two volumes. Whatever the regions say, an excess worth
        less than ``_OVERWEIGHT_QUIET_PERCENT`` of that denominator gets the
        head of the line and nothing else, at info: the number is already
        printed and nobody needs a paragraph explaining a per-mille. At or
        above it the three-way region split decides the wording and the volume
        as before. Only the volume moves -- the counts and the shift in the
        head of the line are identical on both sides of the threshold.
        """
        nb = sum(s.get('nb_overweight', 0) for s in stats_list)
        if not n_written or not nb:
            if n_written:
                logger.info(
                    "MadSpin overweight safety net: 0/%d written events carried "
                    "a non-unit weight -- no accept/reject bound was exceeded, "
                    "so nothing was clipped and nothing is biased by it.",
                    n_written)
            return
        d_w = sum(s.get('sum_overweight_dw', 0.0) for s in stats_list)
        d_abs = sum(s.get('sum_overweight_dabs', 0.0) for s in stats_list)
        biggest = max([s.get('max_overweight', 1.0) for s in stats_list] or [1.0])
        # the file as clipping would have written it
        sum_w = sum(s.get('sum_nom', 0.0) for s in stats_list)
        sum_abs = sum(s.get('sum_abs_nom', 0.0) for s in stats_list)
        sum_sq = sum(s.get('sum_sq_nom', 0.0) for s in stats_list)
        delta = math.sqrt(sum_sq)
        z = (abs(sum_w) / delta) if delta else 0.0
        # Built with % here rather than handed to the logger as a format
        # string: the head already contains literal per-cent signs.
        msg = ("MadSpin overweight safety net: %d/%d written events (%.3g%%) "
               "carried a non-unit weight because a trial weight exceeded its "
               "accept/reject bound (largest factor %.4f). "
               % (nb, n_written, 100.0 * nb / n_written, biggest))
        if z >= self._OVERWEIGHT_MIN_Z:
            shift = 100.0 * d_w / sum_w
            msg += ("Carrying it added %+.3g%% of the sample's cross-section. "
                    % shift)
        else:
            # pure_interference, or any sample whose weights cancel: the
            # cross-section is consistent with zero, so it is not a denominator
            shift = 100.0 * d_abs / sum_abs if sum_abs else float('nan')
            msg += ("Carrying it added %+.6g to the summed event weight and "
                    "%+.6g to the summed |weight|. The summed weight is %+.4g "
                    "against a Monte Carlo error of %.4g (z = %.2f), i.e. "
                    "consistent with the zero cross-section this sample has by "
                    "construction, so it is not a usable denominator and the "
                    "shift is quoted against sum|w| = %.4g instead: %+.3g%%. "
                    % (d_w, d_abs, sum_w, delta, z, sum_abs, shift))
        # No dedicated note when the excess is very small. ``shift`` is the
        # per-cent the line has just quoted, so the smallness test is made
        # against whichever denominator was legitimate: ``sum w`` when it is a
        # usable one, ``sum |w|`` when the weights cancel -- it never divides
        # by a cross-section that is consistent with zero, and it never calls
        # a sample quiet on a ratio the line itself refused to print. A sample
        # with no scale at all (every written weight zero) gives nan, and nan
        # compares false here, so it keeps the full note rather than being
        # declared small on an undefined ratio. The test is on the MAGNITUDE:
        # a large negative shift -- which a sample with counter-events can
        # have, with every carried factor still above 1 -- is not small.
        if abs(shift) < self._OVERWEIGHT_QUIET_PERCENT:
            logger.info(msg)
            return

        near, res, far = self._nwa_threshold_split(stats_list)
        msg = (msg + self._nwa_threshold_note(near, res, far)).rstrip()
        # Calmer only when EVERY one of them is in one of the two explained
        # regions: the count in the head of the line is the total either way,
        # so this changes the volume and not the arithmetic.
        if ((near or res) and not far):
            logger.info(msg)
        else:
            logger.warning(msg)

    def _report_pure_interference(self, base_out, stats_list, n_processed,
                                  n_written):
        """The pure-interference post-loop: the zero-cross-section check, the
        zeroed ``<init>`` block and the ``<MGPureInterference>`` banner note.

        The check is ``z = S / sqrt(sum w^2)``: ``S`` is the sum of the signed
        weights, which the mode predicts to be zero, and the second moment has
        no cancellation in it, so its square root is the right scale to compare
        ``S`` against. Both moments are accumulated in the picklable stats dict
        and merged additively here, so one shard or many gives an identical
        answer (section 13.8).

        The banner block is **not** the normalisation any more -- the fully
        weighted output normalises itself (section 13.13) -- but ``XSECUP = 0``
        deletes the reference cross-section from the file and the diagnostics
        have nowhere else to live, so it carries the reference sigma,
        ``N_read``, ``c``, ``max_weight`` and the zero-cross-section numbers.
        """
        S = sum(s.get('sum_w', 0.0) for s in stats_list)
        sum_w2 = sum(s.get('sum_w2', 0.0) for s in stats_list)
        nb_pi_dead = sum(s.get('nb_pi_dead', 0) for s in stats_list)
        nb_pi_overflow = sum(s.get('nb_pi_overflow', 0) for s in stats_list)
        nb_loose_skip = sum(s.get('nb_loose_skip', 0) for s in stats_list)
        unweighted = self._pure_interference_unweighted()
        absw = getattr(self, '_pi_absw', 0.0) or 0.0
        max_weight = getattr(self, '_pi_max_weight', 0.0) or 0.0

        # --------------------------------------------------------------
        # 'unweighted': replace the probe's <|W|> by the one the run itself
        # realised, which is exact rather than merely well-measured.
        #
        # The written magnitude is w0 = sigma_ref*BR*<|W|>/c, and the file
        # normalises by N_file.  Since N_file = N_drawn*<|W|>/M, putting the
        # RUN's own <|W|> = (N_file/N_drawn)*M into w0 makes N_file cancel out
        # of the estimator entirely:
        #
        #     (1/N_file) sum w O  =  (M sigma_ref BR / (c N_drawn))
        #                            * sum_accepted sign(W) O
        #
        # whose expectation is sigma_ref*BR*<W O>/c exactly, with no estimate
        # of <|W|> in it anywhere.  The probe's <|W|> is a poor substitute:
        # unlike c it is not a decay-side constant, so it is only as good as
        # the handful of production events the probe sees -- measured 9.5%
        # spread over the 110 events of the default probe on p p > t t~,
        # which would be a 9.5% flat error on every weight.  The correction is
        # a single constant, applied to every event in the pass that writes
        # the banner note (which rewrites the whole file anyway).
        n_drawn = n_processed - nb_loose_skip
        event_scale = None
        if unweighted and absw and n_drawn:
            absw_run = (float(n_written) / n_drawn) * max_weight
            event_scale = absw_run / absw
            S *= event_scale
            sum_w2 *= event_scale * event_scale
        else:
            absw_run = absw
        delta = math.sqrt(sum_w2)
        z = (S / delta) if delta else 0.0
        if nb_pi_dead:
            logger.critical(
                "MadSpin pure_interference: %d/%d trial(s) had a non-finite "
                "convolution and were written with weight 0. That is a dead "
                "matrix element, not physics -- the sample is incomplete.",
                nb_pi_dead, n_processed)

        keep = float(n_written) / n_processed if n_processed else 0.0
        if unweighted:
            logger.info(
                "MadSpin pure_interference: wrote %d/%d production events "
                "(%.4f). decay_output = unweighted: one decay "
                "draw per production event, accepted with probability "
                "|W|/max|W|, so the keep rate -- not the weight magnitude -- "
                "carries the local size of the interference.",
                n_written, n_processed, keep)
        else:
            # Fully weighted: every production event is kept, so this is 1
            # unless some *other* mechanism (BR equalization) dropped events.
            logger.info(
                "MadSpin pure_interference: wrote %d/%d production events "
                "(%.4f). The mode does not accept/reject: every trial is kept "
                "and the local size of the interference is carried by the "
                "magnitude of the signed weight instead.",
                n_written, n_processed, keep)

        # The reference normalisation has to be read before the block is zeroed.
        reference = self._read_lhe_init_cross(base_out)
        c_value = getattr(self, '_pi_c', 0.0) or 0.0
        c_err = getattr(self, '_pi_c_err', 0.0) or 0.0
        analytic_c = getattr(self, '_pi_analytic_c', 0.0) or 0.0
        absw_err = getattr(self, '_pi_absw_err', 0.0) or 0.0
        n_c = (getattr(self, '_pi_c_stats', None) or {}).get('n', 0)
        n_absw = (getattr(self, '_pi_absw_stats', None) or {}).get('n', 0)
        n_absw_ev = (getattr(self, '_pi_absw_stats', None) or {}).get('ev_n', 0)
        mean_w = S / n_written if n_written else 0.0
        if unweighted:
            note = [
                '#  Pure-interference sample: it keeps ONLY the interference between',
                '#  the polarisations listed below, so its total cross-section is zero',
                '#  by construction and <init> is written with XSECUP = 0. That also',
                '#  zeroes XERRUP/XMAXUP, so the file cannot be showered as-is.',
                '#',
                '#  The event weights are SIGNED and unweighted UP TO A SIGN -- the',
                '#  file holds exactly two weight magnitudes:',
                '#      w = +- sigma_ref * BR * <|W|> / c',
                '#  with W the signed production/decay convolution, <|W|> its',
                '#  decay-phase-space mean absolute value and c = <W> the',
                '#  unrestricted decay-side constant, both below. One decay draw was',
                '#  made per production event and kept with probability |W|/max|W|,',
                '#  so the file holds FEWER events than were read and the local size',
                '#  of the interference is carried by the keep rate. <|W|> is taken',
                '#  from the run itself -- (N_file/N_drawn) * max|W| -- not from the',
                '#  maximum-weight probe, which sees too few production events to',
                '#  know it; that also makes the accept/reject bound cancel out of',
                '#  the weight EXACTLY rather than on average. MG5 writes LHE with',
                '#  IDWTUP = -4, i.e. the cross-section is the MEAN of the weights,',
                '#  so this sample is self-normalising: mean(w) = 0 (its rate) and',
                '#  sum_bin(w) / N_file is the interference contribution to that',
                '#  bin, in pb, with N_file the number of events WRITTEN (the first',
                '#  number below).',
            ]
        else:
            note = [
                '#  Pure-interference sample: it keeps ONLY the interference between',
                '#  the polarisations listed below, so its total cross-section is zero',
                '#  by construction and <init> is written with XSECUP = 0. That also',
                '#  zeroes XERRUP/XMAXUP, so the file cannot be showered as-is.',
                '#',
                '#  The event weights are SIGNED and fully weighted:',
                '#      w = sigma_ref * BR * W / c',
                '#  with W the signed production/decay convolution of this event and',
                '#  c = <W> the decay-side constant below. MG5 writes LHE with',
                '#  IDWTUP = -4, i.e. the cross-section is the MEAN of the weights,',
                '#  so this sample is self-normalising: mean(w) = 0 (its rate) and',
                '#  sum_bin(w) / N_read is the interference contribution to that',
                '#  bin, in pb. N_read is the "Events written / read" count below.',
            ]
        for pdg, (prod, dec) in sorted(self._pure_interference().items()):
            note.append('#  interference  pdg %-6s : production %s  x  decay %s'
                        % (pdg, list(prod), list(dec)))
        note += [
            '#  Reference normalisation (pb) : %+.8e' % reference,
            '#     (the parent sample cross-section times the branching ratio,',
            '#      i.e. what <init> would have carried without this mode)',
            '#  Normalisation constant     c : %+.8e  +- %.4f%%' % (
                c_value, (100 * c_err / abs(c_value)) if c_value else 0.0),
            '#     (c = <W>, the decay-side mean of the UNRESTRICTED convolution,',
            '#      measured by the maximum-weight scan over %d trials)' % n_c,
            '#  Analytic candidate for c     : %+.8e  (ratio %.6f)' % (
                analytic_c, (c_value / analytic_c) if analytic_c else 0.0),
            '#     (1/(prod_denominators * sym_decay); exact only where the chain',
            '#      carries no reshuffling jacobian -- a cross-check, not the value used)',
            '#  <|W|> from the probe         : %+.8e  +- %.4f%%' % (
                absw, (100 * absw_err / absw) if absw else 0.0),
            '#     (the decay-phase-space mean of |W|, over %d trials on %d' % (
                n_absw, n_absw_ev),
            '#      production events. The error is the spread over THOSE events,',
            '#      not over the trials: <|W|> is not a decay-side constant the way',
            '#      c is, so a handful of production events does not pin it down)',
            '#  Maximum weight max|W| probed : %+.8e' % max_weight,
        ]
        if unweighted:
            note += [
                '#     (the bound the accept/reject used. It cancels out of the',
                '#      weight exactly, but it does bound it: see the overflow count)',
                '#  <|W|> the run realised       : %+.8e  (probe x %.4f)' % (
                    absw_run, event_scale or 1.0),
                '#     ( = (N_file/N_drawn) * max|W| , over every production event',
                '#      of this run rather than the probe\'s few. THIS is what the',
                '#      written weights carry; the probe value above was the',
                '#      provisional one and has been divided out)',
                '#  Weight magnitude |w| (pb)    : %+.8e' % (
                    reference * absw_run / c_value if c_value else 0.0),
                '#     ( = sigma_ref * <|W|> / c ; every event carries +- this)',
                '#  Trials above max|W|          : %d' % nb_pi_overflow,
                '#     (accepted with probability 1 instead of |W|/max|W|. Those',
                '#      events -- and ONLY those -- are written with |w| scaled',
                '#      by |W|/max|W| > 1 instead of being clipped, so the file',
                '#      may hold more than two weight magnitudes when this is',
                '#      non-zero. <|W|> = (N_file/N_drawn) x max|W| is unchanged',
                '#      by that: N_file is a count, and min(1,x)*max(1,x) = x.',
                '#      Non-zero still means max_weight is under-estimated:',
                '#      raise nb_sigma or Nevents_for_max_weight)',
            ]
        else:
            note += [
                '#     (diagnostic only: the mode does not accept/reject, so this',
                '#      number no longer enters the normalisation anywhere)',
            ]
        note += [
            '#  Sum of written weights     S : %+.8e' % S,
            '#  MC error   sqrt(sum w^2)     : %+.8e' % delta,
            '#  z = S / error                : %+.4f' % z,
            '#  mean(w), the sample XSECUP   : %+.8e' % mean_w,
            '#  Events written / read        : %d / %d' % (n_written, n_processed),
            '#  Trials with a dead weight    : %d' % nb_pi_dead,
        ]
        self._rewrite_lhe_banner_cross(base_out, 0.0, n_written=n_written,
                                       note=note, note_tag='MGPureInterference',
                                       event_scale=event_scale)

        logger.info("MadSpin pure_interference: sum of weights S = %+.6e, "
                    "sqrt(sum w^2) = %.6e, z = %+.3f, mean(w) = %+.6e "
                    "(reference normalisation %.6e pb, c = %.6e, both "
                    "recorded in the <MGPureInterference> banner block)",
                    S, delta, z, mean_w, reference, c_value)
        if unweighted:
            logger.info(
                "MadSpin pure_interference: every event carries |w| = %.6e pb, "
                "from the run's own <|W|> = (N_file/N_drawn) x max|W| = %.6e. "
                "The maximum-weight probe had said %.6e +- %.1f%%, so the "
                "written weights were rescaled by %.4f -- the probe sees too "
                "few production events to normalise with, and using the run's "
                "own keep rate instead makes the accept/reject bound cancel "
                "exactly rather than on average.",
                (reference * absw_run / c_value) if c_value else 0.0,
                absw_run, absw, 100 * (absw_err / absw if absw else 0.0),
                event_scale or 1.0)
            if event_scale and abs(event_scale - 1.0) > 0.25:
                logger.warning(
                    "MadSpin pure_interference: the probe's <|W|> was off by "
                    "%.0f%%, which is a lot even for a quantity it only sees a "
                    "handful of production events of. The written weights use "
                    "the run's own value and are right, but a probe that far "
                    "out means the maximum weight it produced may be poor too "
                    "-- check the overweight count and consider raising "
                    "Nevents_for_max_weight.", 100 * (event_scale - 1.0))
            if nb_pi_overflow:
                logger.critical(
                    "MadSpin pure_interference: %d trial(s) had |W| above the "
                    "maximum weight and were accepted with probability 1 "
                    "instead of |W|/max|W|. Unlike the fully weighted output, "
                    "this variant DOES accept/reject, so that bound is live "
                    "and an under-estimated one biases the sample. Raise "
                    "nb_sigma or Nevents_for_max_weight.", nb_pi_overflow)
        if abs(z) > 5.0:
            cause = (
                "either a genuine fluctuation, an under-estimated max_weight "
                "(the overweight count above is the monitor for that), or a bug"
                if unweighted else
                "either a genuine fluctuation or a bug (the mode no longer "
                "accept/rejects, so an under-estimated max_weight can no "
                "longer be the cause)")
            message = (
                "MadSpin pure_interference: the sum of the event weights is "
                "NOT compatible with zero -- S = %+.6e, sqrt(sum w^2) = %.6e, "
                "z = %+.3f (over 5 sigma). The interference term must "
                "integrate to zero over the decay phase space, so this is %s."
                % (S, delta, z, cause))
            logger.critical(message)
            if self.options['density_debug']:
                raise RuntimeError(message)
        # The banner cross-section is NOT rescaled (it is zero anyway) and
        # neither is the branching ratio -- in this mode a low keep rate is
        # physics, not a correction to undo. The efficiency downstream sizes
        # nb_event with is taken from the counts: 1 for the fully weighted
        # output (n_written == n_processed, up to a BR-equalization drop), and
        # the genuine keep rate for the unweighted one, where the file really
        # does hold that many fewer events.
        self.efficiency = keep

    def _weighted_decay_note(self, base_out, stats_list, n_written,
                             br_correction=1.0):
        """The ``<MGWeightedDecay>`` banner block, and the log line that goes
        with it: what ``decay_output = weighted`` wrote, and the one check it
        can make on itself.

        The check is ``mean(w)`` against ``sigma_ref * BR``. Under MG5's
        ``IDWTUP = -4`` the cross-section is the mean of the event weights, so
        that equality is not a convention here -- it is the statement that
        ``c = <W>`` was measured correctly, since ``mean(w) = sigma*BR*<W>/c``
        by construction. It is the exact analogue of the interference mode's
        ``z`` test (there ``<W> = 0``, so the target is 0 instead of 1).
        """
        S = sum(s.get('sum_w', 0.0) for s in stats_list)
        sum_w2 = sum(s.get('sum_w2', 0.0) for s in stats_list)
        nb_dead = sum(s.get('nb_pi_dead', 0) for s in stats_list)
        mean_w = S / n_written if n_written else 0.0
        # the MC error on the mean, from the second moment of the weights
        var = max(sum_w2 / n_written - mean_w * mean_w, 0.0) if n_written else 0.0
        mean_err = math.sqrt(var / n_written) if n_written else 0.0
        # read before the block is (possibly) rescaled by the same pass, and
        # corrected by hand so the comparison is against what <init> will say
        reference = self._read_lhe_init_cross(base_out) * br_correction
        c_value = getattr(self, '_pi_c', 0.0) or 0.0
        c_err = getattr(self, '_pi_c_err', 0.0) or 0.0
        n_c = (getattr(self, '_pi_c_stats', None) or {}).get('n', 0)
        ratio = (mean_w / reference) if reference else 0.0
        pull = ((mean_w - reference) / mean_err) if mean_err else 0.0
        if nb_dead:
            logger.warning(
                "MadSpin decay_output = weighted: %d/%d trial(s) had a "
                "non-positive or non-finite convolution -- normally a mass set "
                "the production could not be reshuffled onto, which the "
                "accept/reject would have redrawn. They were written with "
                "weight 0, so they contribute nothing, but they do dilute the "
                "sample by that fraction.", nb_dead, n_written)
        logger.info(
            "MadSpin decay_output = weighted: wrote %d weighted events; "
            "mean(w) = %.6e +- %.2e against the reference sigma*BR = %.6e "
            "(ratio %.6f, %.2f sigma). Under IDWTUP = -4 that mean IS the "
            "cross-section, so the agreement is the check that c = <W> = "
            "%.6e was measured right.",
            n_written, mean_w, mean_err, reference, ratio, pull, c_value)
        if mean_err and abs(pull) > 5.0:
            logger.critical(
                "MadSpin decay_output = weighted: mean(w) = %.6e is %.2f "
                "sigma from the reference sigma*BR = %.6e (ratio %.4f). Under "
                "IDWTUP = -4 the sample's cross-section is the mean of its "
                "weights, so this file does not carry the rate its <init> "
                "block claims. The likely cause is a mis-measured c = <W>: "
                "raise Nevents_for_max_weight / max_weight_ps_point.",
                mean_w, pull, reference, ratio)
        return [
            '#  WEIGHTED MadSpin sample (decay_output = weighted): no',
            '#  accept/reject was done. One decay configuration was drawn per',
            '#  production event and kept, carrying',
            '#      w = w_prod * BR * W / c',
            '#  with W that trial\'s production/decay density convolution and',
            '#  c = <W> its decay-phase-space mean (a constant, below). MG5',
            '#  writes LHE with IDWTUP = -4, i.e. the cross-section is the MEAN',
            '#  of the event weights, so <init> is the ordinary sigma*BR and',
            '#  sum_bin(w) / N_file is that bin in pb -- but the per-event',
            '#  weights are NOT constant. Any consumer that assumes unit-weight',
            '#  MadSpin output (counting events, unweighted histograms) is',
            '#  wrong on this file.',
            '#  Normalisation constant     c : %+.8e  +- %.4f%%' % (
                c_value, (100 * c_err / abs(c_value)) if c_value else 0.0),
            '#     (measured by the maximum-weight scan over %d trials)' % n_c,
            '#  Reference sigma * BR   (pb)  : %+.8e' % reference,
            '#  mean(w), the sample XSECUP   : %+.8e  +- %.4e' % (
                mean_w, mean_err),
            '#  mean(w) / reference          : %.6f   (%.2f sigma)' % (
                ratio, pull),
            '#  Events written               : %d' % n_written,
            '#  Trials with a dead weight    : %d' % nb_dead,
            '#     (non-positive or non-finite W -- a failed production',
            '#      reshuffle -- written with weight 0)',
        ]

    @staticmethod
    def _read_lhe_init_cross(path):
        """Sum of the XSECUP column of an already-written LHE ``<init>`` block."""
        total = 0.0
        try:
            with open(path) as src:
                in_init = False
                for line in src:
                    stripped = line.strip()
                    lowered = stripped.lower()
                    # '<init>' exactly: '<initrwgt>' also starts with '<init'
                    # and sits *before* it in the header, so a prefix test
                    # matches the multi-weight block and reads nothing.
                    if lowered.startswith('<init>'):
                        in_init = True
                        continue
                    if not in_init:
                        continue
                    if lowered.startswith('</init>'):
                        break
                    parts = stripped.split()
                    if len(parts) == 4:
                        try:
                            total += float(parts[0])
                        except ValueError:
                            pass
        except Exception as exc:
            logger.warning('MadSpin: could not read the <init> cross-section '
                           'of %s (%s); the reference normalisation of the '
                           'pure-interference banner note will read 0.',
                           path, exc)
        return total

    def _apply_accounting(self, base_out, stats_list):
        """Post-loop accounting shared by the serial and parallel paths: the
        unweighting-efficiency log, the BR-equalization banner rewrite, and the
        gzip of input+output. ``base_out`` must already be a complete LHE file
        (banner + events + closing tag). Counter sums over ``stats_list`` are
        order-independent, so one shard or many gives the identical result a
        single serial stream would have."""
        n_processed = sum(s['n_processed'] for s in stats_list)
        n_written = sum(s['n_written'] for s in stats_list)
        nb_try = sum(s['nb_try'] for s in stats_list)
        nb_loose_skip = sum(s['nb_loose_skip'] for s in stats_list)

        eff = float(n_written) / nb_try if nb_try else 0.0
        logger.info(
            "MadSpin unweight efficiency: %.4f (%d written / %d trials, %.2f trials/event)",
            eff, n_written, nb_try, (1.0 / eff if eff else float("inf"))
        )
        self._report_sequential_stats(stats_list, n_written)
        self._report_overweight(stats_list, n_written)
        if self._pure_interference():
            self._report_pure_interference(base_out, stats_list,
                                           n_processed, n_written)
        elif nb_loose_skip > 0 or self._weighted_decay():
            # Rewrite the banner with the corrected cross-section so it
            # matches the actual sum of kept-event weights. Each kept event
            # already has wgt = orig_wgt * max_br; we need the banner to read
            # σ * max_br * (n_written / n_processed) ≈ σ * <br>.
            #
            # decay_output = weighted goes through the same rewrite even with
            # nothing dropped (br_correction = 1), because it is the pass that
            # inserts the <MGWeightedDecay> note: <init> stays right, but a
            # weighted MadSpin file is not what a consumer expects and the
            # file has to say so.
            br_correction = float(n_written) / n_processed if n_processed else 1.0
            note = (self._weighted_decay_note(base_out, stats_list, n_written,
                                              br_correction)
                    if self._weighted_decay() else None)
            self._rewrite_lhe_banner_cross(
                base_out, br_correction, n_written=n_written, note=note,
                note_tag='MGWeightedDecay' if note else 'MGGenerationInfo')
            self.branching_ratio *= br_correction
            self.cross *= br_correction
            self.error *= br_correction
            if nb_loose_skip:
                logger.info(
                    "BR equalization: dropped %d/%d events (effective BR "
                    "rescale = %.4g).", nb_loose_skip, n_processed,
                    br_correction)
            # Downstream sets nb_event = int(original_nb_event * efficiency)
            # so the kept-fraction needs to be communicated as the efficiency.
            self.efficiency = br_correction
        else:
            self.efficiency = 1 # to let me5 to write the correct number of events
        # Re-gzip the input events file (gunzipped at the start of this
        # routine) and the decayed output, matching the legacy MadSpin path
        # so downstream code (banners, crossx.html) finds the *.lhe.gz files
        # it expects.
        try:
            input_evt_path = self.events_file.name
            if input_evt_path.endswith('.lhe') and os.path.exists(input_evt_path):
                misc.gzip(input_evt_path)
        except Exception as exc:
            logger.warning('Could not re-gzip MadSpin input file %s: %s',
                           getattr(self.events_file, 'name', '?'), exc)
        try:
            if base_out.endswith('.lhe') and os.path.exists(base_out):
                misc.gzip(base_out)
        except Exception as exc:
            logger.warning('Could not gzip MadSpin decayed output %s: %s',
                           base_out, exc)
        logger.info('Done so far. output written in %s' % base_out)

    @staticmethod
    def _balanced_ranges(nb_item, nb_core):
        """``nb_core`` contiguous ``(start, stop)`` slices of ``range(nb_item)``
        whose lengths differ by at most one -- the first ``nb_item % nb_core``
        get one extra item.

        Contrast ``ceil(nb_item / nb_core)``-sized chunks, which give the last
        cores nothing whenever the division is uneven: 75 items on 16 cores is
        fifteen chunks of five and one empty. An empty slice is not merely an
        idle core, because the shard id doubles as the worker id that
        :meth:`_channel_owner` deals channels out to -- a worker that is never
        forked can still be named as a channel's owner, and whoever waits on it
        waits for ever. Balanced slices keep every id live. When ``nb_item`` is a
        multiple of ``nb_core`` -- what both max-weight scans arrange, by
        rounding their probe size up -- this is the same split as before."""
        nb_core = max(1, int(nb_core))
        base, extra = divmod(int(nb_item), nb_core)
        ranges, start = [], 0
        for sid in range(nb_core):
            stop = start + base + (1 if sid < extra else 0)
            if stop > start:
                ranges.append((start, stop))
            start = stop
        return ranges

    def _split_production(self, orig_lhe, nb_core, base_out):
        """Split the production event file into up to ``nb_core`` contiguous
        shard files (bannerless: the worker's EventFile tolerates a missing
        banner). Returns ``(paths, counts)``. In fixed_order mode each production
        item is an event-group, written back with its ``<eventgroup>`` wrapper so
        the shard round-trips."""
        fixed_order = self.options['fixed_order']
        orig_lhe.seek(0)
        if fixed_order:
            orig_lhe.eventgroup = True
        nb_event = sum(1 for _ in orig_lhe)
        orig_lhe.seek(0)
        if fixed_order:
            orig_lhe.eventgroup = True

        chunk = int(math.ceil(nb_event / float(nb_core))) if nb_event else 0
        paths, counts, shard_files = [], [], []
        for sid in range(nb_core):
            p = '%s.prodshard%d.lhe' % (base_out, sid)
            ef = lhe_parser.EventFile(p, 'w')
            if fixed_order:
                ef.eventgroup = True
            shard_files.append(ef)
            paths.append(p)
            counts.append(0)

        for idx, production in enumerate(orig_lhe):
            sid = min(idx // chunk, nb_core - 1) if chunk else 0
            shard_files[sid].write_events(production)
            counts[sid] += 1
        for ef in shard_files:
            try:
                ef.close()
            except Exception:
                pass

        # keep only non-empty shards (drops trailing shards when nb_core > nb_event)
        keep = [(p, c) for p, c in zip(paths, counts) if c > 0]
        for p, c in zip(paths, counts):
            if c == 0:
                try:
                    os.remove(p)
                except OSError:
                    pass
        return [p for p, _ in keep], [c for _, c in keep]

    def _reopen_decay_pool(self, evt_decayfile, shard_id, nb_core):
        """Return this worker's private view of the decay pools.

        The unweighting already wrote one file per worker
        (``nb_unweight_output``), so a worker simply opens *its* file: no two
        workers ever read the same event and none of them has to scan the whole
        pool. If the pool happens to be a single file (e.g. it was produced
        before this was in place) fall back to striding it, which is correct but
        makes every worker parse everything."""
        local = {}
        for pdg, channels in evt_decayfile.items():
            local[pdg] = {}
            for file_nb, evtfile in channels.items():
                paths = getattr(evtfile, 'paths', None)
                own_file = bool(paths and len(paths) == nb_core)
                if own_file:
                    reader = lhe_parser.EventFile(paths[shard_id])
                elif paths:
                    # split, but not into exactly nb_core files: stride the WHOLE
                    # chained pool. ``evtfile.name`` is only its first file, so
                    # striding that would strand every other file's events.
                    reader = _StridedEvents(_ChainedEvents(paths), shard_id, nb_core)
                else:
                    fresh = lhe_parser.EventFile(evtfile.name)
                    reader = _StridedEvents(fresh, shard_id, nb_core)
                # The worker that OWNS this channel reads its slice ~10% short so
                # it runs out (and, being the sole generator, regenerates) before
                # the others do -- keeping their wait for the refill minimal.
                # Only meaningful on the own-file fast path where the count of
                # this worker's slice is well defined.
                if own_file and self._channel_owner(pdg, file_nb) == shard_id:
                    frac = float(getattr(self, '_owner_undersize', 0.10) or 0.0)
                    if frac > 0:
                        n = self._count_lhe_events(paths[shard_id])
                        reader = _LimitedEvents(reader,
                                                int(math.floor((1.0 - frac) * n)))
                local[pdg][file_nb] = reader
        return local

    def _init_owner_refill(self, evt_decayfile, seed_base):
        """Per-worker set-up for the owner-based refill. Must run in every forked
        worker (unweighting and both max-weight scans) before its decay pools are
        opened, so :meth:`_channel_owner` sees the same channel ordering
        everywhere. Sets: the fixed sorted channel list, the shard-independent
        seed base for the deterministic generation seed, and the owner undersize
        fraction (``MADSPIN_OWNER_UNDERSIZE``, default 0.10). Also publishes the
        initial 'running' worker status for the deadlock detection."""
        self._channel_keys = sorted(
            (pdg, fnb) for pdg, chans in evt_decayfile.items() for fnb in chans)
        self._refill_seed_base = int(seed_base) if seed_base else 0
        try:
            self._owner_undersize = float(
                os.environ.get('MADSPIN_OWNER_UNDERSIZE', '0.10'))
        except (TypeError, ValueError):
            self._owner_undersize = 0.10
        self._set_status('R')

    def _unweight_shard_entry(self, shard_id, nb_core, shard_path, out_path,
                              evt_decayfile, ctx, stats_path):
        """Worker entry point (runs in a forked child process). Owns its RNG, its
        f2py COMMON blocks (independent address space after fork), and its output
        fragment. Writes a JSON stats file the parent reads back; on failure
        writes the traceback there instead of raising into the parent (which only
        sees exit codes)."""
        import json
        try:
            # distinct RNG streams per shard (channel selection + accept/reject)
            random.seed(ctx['base_seed'] + 7919 * (shard_id + 1))
            # marks this process as a forked worker: any decay-event refill must
            # go through the centralised, locked _worker_refill
            self._shard_tag = shard_id
            self._shard_nb_core = nb_core
            self._pool_gen = {}
            self.options['seed'] = (ctx['base_seed'] + 100003 * (shard_id + 1)) % (30081 * 30081)
            self.seed = self.options['seed']
            self.efficiency = 1.0
            self.branching_ratio = ctx['branching_ratio']
            self._init_owner_refill(evt_decayfile, ctx['base_seed'])

            prod = lhe_parser.EventFile(shard_path)
            if self.options['fixed_order']:
                prod.eventgroup = True
            local_pool = self._reopen_decay_pool(evt_decayfile, shard_id, nb_core)

            out = lhe_parser.EventFile(out_path, 'w')
            if self.options['fixed_order']:
                out.eventgroup = True
            stats = self._unweight_range(prod, local_pool, out, ctx)
            try:
                out.close()
            except Exception:
                pass
            with open(stats_path, 'w') as f:
                json.dump(stats, f)
        except Exception as exc:
            import traceback
            try:
                with open(stats_path, 'w') as f:
                    json.dump({'error': str(exc), 'tb': traceback.format_exc()}, f)
            except Exception:
                pass
        finally:
            # tell any worker still blocked on a channel I own that I am gone, so
            # it stops waiting for a generation that will never come and produces
            # it itself (deadlock fail-safe).
            self._set_status('D')

    def _run_onshell_parallel(self, orig_lhe, nb_event, nb_core, evt_decayfile,
                              base_out, ctx):
        """Parallel driver for the unweighting stage: split the production events
        into contiguous shards, fork one worker per shard, then merge the
        fragments under a single banner and apply the global accounting.

        Uses the ``fork`` start method so each worker inherits the fully set-up
        interface (compiled ME dir, model, banner) via copy-on-write memory --
        no pickling of ``self`` -- and gets its own address-space copy of the
        matrix-element COMMON blocks. Results come back through per-shard JSON
        files rather than a Queue to avoid any pickling of worker state."""
        import multiprocessing as mp
        import json

        shard_paths, shard_counts = self._split_production(orig_lhe, nb_core, base_out)
        nb_core = len(shard_paths)
        if nb_core == 0:
            # no events at all: emit a banner-only file
            output_lhe = lhe_parser.EventFile(base_out, 'w')
            self.banner.scale_init_cross(self.branching_ratio)
            self.banner.write(output_lhe, close_tag=False)
            output_lhe.write('</LesHouchesEvents>\n')
            try:
                output_lhe.close()
            except Exception:
                pass
            self._apply_accounting(base_out, [dict(n_processed=0, n_written=0,
                                                   nb_try=0, nb_loose_skip=0)])
            return

        self._clear_worker_status(nb_core)   # fresh status board for this phase
        mpctx = mp.get_context('fork')
        procs, frag_paths, stats_paths = [], [], []
        for sid in range(nb_core):
            frag = '%s.shard%d.lhe' % (base_out, sid)
            stp = '%s.shard%d.json' % (base_out, sid)
            cctx = dict(ctx)
            cctx['shard_nb_event'] = shard_counts[sid]
            p = mpctx.Process(
                target=self._unweight_shard_entry,
                args=(sid, nb_core, shard_paths[sid], frag, evt_decayfile,
                      cctx, stp))
            p.start()
            procs.append(p)
            frag_paths.append(frag)
            stats_paths.append(stp)
        for p in procs:
            p.join()

        # collect stats and surface worker failures
        stats_list = []
        for sid, stp in enumerate(stats_paths):
            if not os.path.exists(stp):
                raise Exception("MadSpin worker %s produced no result (crashed, exitcode=%s). "
                                "Re-run with nb_core=1 to reproduce/debug."
                                % (sid, procs[sid].exitcode))
            with open(stp) as f:
                s = json.load(f)
            if 'error' in s:
                raise Exception("MadSpin worker %s failed:\n%s"
                                % (sid, s.get('tb', s['error'])))
            stats_list.append(s)

        # merge: one banner + fragment bodies (in production order) + closing tag
        output_lhe = lhe_parser.EventFile(base_out, 'w')
        if self.options['fixed_order']:
            output_lhe.eventgroup = True
        self.banner.scale_init_cross(self.branching_ratio)
        self.banner.write(output_lhe, close_tag=False)
        for frag in frag_paths:
            if os.path.exists(frag):
                with open(frag) as fr:
                    for line in fr:
                        output_lhe.write(line)
        output_lhe.write('</LesHouchesEvents>\n')
        try:
            output_lhe.close()
        except Exception:
            pass

        for pth in shard_paths + frag_paths + stats_paths:
            try:
                os.remove(pth)
            except OSError:
                pass

        self._apply_accounting(base_out, stats_list)

    # <wgt id='...'>value</wgt>, the LHEF v3 multi-weight entry
    _RWGT_LINE = re.compile(r'^(\s*<wgt\b[^>]*>)\s*([-+0-9.eEdD]+)\s*(</wgt>\s*)$')

    def _rewrite_lhe_banner_cross(self, path, ratio, n_written=None,
                                  note=None, note_tag='MGGenerationInfo',
                                  event_scale=None):
        """Rewrite an already-written LHE file, multiplying every <init> line
        cross-section / error / xmax by ``ratio`` and (optionally) replacing
        the ``Number of Events`` entry in the MGGenerationInfo block with
        ``n_written``. Mirrors decay_all_events.write_banner_information for
        the PA-mode (run_onshell) code path.

        ``note``, when given, is a list of already-formatted comment lines
        inserted as a ``<note_tag>`` block just before ``</header>`` -- the
        pure-interference mode uses it to record the reference normalisation
        that its zeroed ``<init>`` block no longer carries.

        ``event_scale``, when given, additionally multiplies every event's
        ``XWGTUP`` and every ``<wgt>`` entry of its ``<rwgt>`` block by that
        constant. Only the 'unweighted' pure-interference output uses it, and
        only to replace the maximum-weight probe's estimate of ``<|W|>`` by
        the one the run itself realised -- a number that is not known until
        the loop has finished, hence the second pass. ``None`` (the default)
        leaves every event byte-for-byte as written."""

        tmp_path = path + '.tmp_brfix'
        shutil.move(path, tmp_path)
        with open(tmp_path, 'r') as src, open(path, 'w') as dst:
            in_init = False
            in_mggen = False
            in_event = False
            want_event_head = False
            for line in src:
                stripped = line.strip()
                lstripped = stripped.lower()
                if event_scale is not None:
                    if lstripped.startswith('<event'):
                        in_event = True
                        want_event_head = True
                        dst.write(line)
                        continue
                    if in_event:
                        if lstripped.startswith('</event'):
                            in_event = False
                            dst.write(line)
                            continue
                        if want_event_head:
                            # NUP IDPRUP XWGTUP SCALUP AQEDUP AQCDUP
                            parts = stripped.split()
                            if len(parts) == 6:
                                try:
                                    wgt = float(parts[2].replace('d', 'e'))
                                except ValueError:
                                    pass
                                else:
                                    want_event_head = False
                                    parts[2] = '%.7e' % (wgt * event_scale)
                                    dst.write('%s\n' % ' '.join(parts))
                                    continue
                        match = self._RWGT_LINE.match(line.rstrip('\n'))
                        if match:
                            try:
                                wgt = float(match.group(2).replace('d', 'e'))
                            except ValueError:
                                pass
                            else:
                                dst.write('%s%.7e%s\n' % (match.group(1),
                                                          wgt * event_scale,
                                                          match.group(3)))
                                continue
                        dst.write(line)
                        continue
                if note and lstripped.startswith('</header'):
                    dst.write('<%s>\n' % note_tag)
                    for entry in note:
                        dst.write('%s\n' % entry)
                    dst.write('</%s>\n' % note_tag)
                    dst.write(line)
                    continue
                # '<init>' exactly, not a '<init' prefix: '<initrwgt>' is a
                # different block, it sits earlier in the header, and its
                # <weight> lines would otherwise be rescaled as if they were
                # cross-section rows the moment one of them had four tokens.
                if lstripped.startswith('<init>'):
                    in_init = True
                    dst.write(line)
                    continue
                if in_init:
                    if lstripped.startswith('</init>'):
                        in_init = False
                        dst.write(line)
                        continue
                    parts = stripped.split()
                    if len(parts) == 4:
                        try:
                            xsec, xerr, xmax = (float(parts[0]), float(parts[1]), float(parts[2]))
                            pid = int(parts[3])
                            dst.write("   %+13.7e %+13.7e %+13.7e %i\n" %
                                      (ratio*xsec, ratio*xerr, ratio*xmax, pid))
                            continue
                        except ValueError:
                            pass
                    dst.write(line)
                    continue
                # MGGenerationInfo block: update Number of Events and any
                # ":" -separated numeric field with the BR correction ratio.
                if lstripped.startswith('<mggenerationinfo'):
                    in_mggen = True
                    dst.write(line)
                    continue
                if in_mggen:
                    if lstripped.startswith('</mggenerationinfo'):
                        in_mggen = False
                        dst.write(line)
                        continue
                    if 'Number of Events' in line and n_written is not None:
                        dst.write('#  Number of Events        :       %i\n' % n_written)
                        continue
                    if ':' in line:
                        head, tail = line.rsplit(':', 1)
                        try:
                            value = float(tail.strip())
                            dst.write('%s : %s\n' % (head, value * ratio))
                            continue
                        except ValueError:
                            pass
                    dst.write(line)
                    continue
                dst.write(line)
        os.remove(tmp_path)

    def get_decay_from_file(self,production, evt_decayfile, nb_remain):
        """return a dictionary PDG -> list of associated decay"""

        out = collections.defaultdict(list)
        for i, particle, decay in self._draw_all_decays(production, evt_decayfile,
                                                        nb_remain):
            out[particle.pdg].append(decay)
        return out

    def _draw_all_decays(self, production, evt_decayfile, nb_remain, group=None):
        """Yield (slot_index, particle, decay) for every decaying particle of the
        production event, in production order -- which is the order the density
        matrix slots are built in.

        ``group``: the '@' decay group this draw belongs to. Drawn here when the
        caller does not impose one, so that it is redrawn on every trial of the
        joint accept/reject along with the decays it selects."""
        particles = [p for p in production if int(p.status) == 1.0]
        ids = [particle.pid for particle in particles]
        if group is None:
            group = self._draw_decay_group()
        for i, particle in enumerate(particles):
            decay = self._draw_one_decay(particle, i, ids, evt_decayfile,
                                         nb_remain, group)
            if decay is not None:
                yield i, particle, decay

    def _draw_one_decay(self, particle, i, ids, evt_decayfile, nb_remain,
                        group=None):
        """Draw one decay event for ``particle`` -- the i-th final-state particle
        of the production event, ``ids`` being the pdgs of all of them -- and
        refill its pool if it runs out. Returns None when that particle does not
        decay.

        Factored out of get_decay_from_file so that the sequential accept/reject
        can redraw a single particle without touching the ones already accepted.

        ``group`` restricts the choice to the channels that group gives this
        particle; the rules below then apply to those, unchanged. That is the
        whole of the grouping at run time -- a group supplies exactly one channel
        per particle (or one per identical parent, which the positional rule then
        deals out), so restricting the candidates is all it takes.
        """
        # check if we need to decay the particle
        if particle.pdg not in evt_decayfile:
            return None # nothing to do for this particle
        channels = evt_decayfile[particle.pdg]
        if group is not None:
            keys = [k for k in self._decay_groups['lines']
                                    .get(particle.pdg, {}).get(group, ())
                    if k in channels]
        else:
            keys = sorted(channels)
        # check how the decay need to be done
        nb_decay = len(keys)
        if nb_decay == 0:
            return None #nothing to do for this particle
        # Determine the file to read in order to get the decay [decay_file]
        if nb_decay == 1:
            decay_file_nb = keys[0]
            decay_file = channels[decay_file_nb]
        elif ids.count(particle.pdg) == nb_decay:
            decay_file_nb = keys[ids[:i].count(particle.pdg)]
            decay_file = channels[decay_file_nb]
        else:
            #need to select the file according to the associate cross-section
            r = random.random()
            tot = sum(channels[key].cross for key in keys)
            r = r * tot
            cumul = 0
            for j in keys:
                events = channels[j]
                cumul += events.cross
                if r < cumul:
                    decay_file = events
                    decay_file_nb = j
                    break
                else:
                    continue
            else:
                raise Exception
        # So now we know which file to read. Do it and re-generate events for that 
        # file if needed.
        while 1:
            try:
                decay = next(decay_file)
                break
            except StopIteration:
                # Estimate refill size from remaining production events
                # efficiency and per-trial consumption if decaying particles
                # Take into account identical parents
                # Oversample by 10% to reduce refill frequency; cap to limit one refill cost.
                eff = max(self.efficiency, 1e-12)
                same_pdg = ids.count(particle.pdg)
                if nb_decay == 1:
                    burn = same_pdg
                elif nb_decay == same_pdg:
                    burn = 1.0
                else:
                    burn = max(1.0, float(same_pdg) / float(nb_decay))
                needed = int(math.ceil(1.10 * burn * nb_remain / eff))
                # Statistical-fluctuation security: the number of events we
                # actually get back fluctuates like sqrt(N), so ask for
                # sqrt(target) more than the bare target -- running short
                # would cost a whole extra refill.
                needed += int(math.ceil(math.sqrt(needed)))
                needed = min(200000, max(needed, 1000))
                if getattr(self, '_shard_tag', None) is not None:
                    # Parallel unweighting: generation is not fork-safe and must
                    # not run concurrently. Each channel has a fixed OWNER worker
                    # that generates a pool for everybody (nb_core * remaining /
                    # eff); the others block until it is ready. _worker_refill
                    # returns this worker's own reader over that pool (the owner's
                    # is ~10% short so it runs out -- and regenerates -- first).
                    evt_decayfile[particle.pdg][decay_file_nb] = \
                        self._worker_refill(
                            particle.pdg, decay_file_nb,
                            needed * self._shard_nb_core)
                else:
                    # serial: _regenerate_events already returns the reader
                    # over the events it produced
                    self._refill_nb = getattr(self, '_refill_nb', 0) + 1
                    evt_decayfile[particle.pdg][decay_file_nb] = \
                        self._regenerate_events(
                            particle.pdg, decay_file_nb, needed,
                            'ms_refill_%d' % self._refill_nb)
                decay_file = evt_decayfile[particle.pdg][decay_file_nb]
                continue
        return decay
        
    
    def get_maxwgt_for_onshell(self, orig_lhe, evt_decayfile, decay_dict):
        """determine the maximum weight for the onshell (or similar) strategy"""
        #print(f"decay_dict = {decay_dict} - length = {len(decay_dict)}")
        # event_decay is a dict pdg -> list of event file (contain the decay)
                
        # both the pure-interference mode and decay_output = weighted need the
        # decay-side constant c, which only this scan measures
        pure_interference = bool(self._pure_interference()) or self._weighted_decay()
        if self.options['ms_dir'] and os.path.exists(pjoin(self.options['ms_dir'], 'max_wgt')):
            # in those modes this scan also measures c, so a cached
            # bound may only be reused when the matching c is cached too
            if not pure_interference:
                return float(open(pjoin(self.options['ms_dir'], 'max_wgt'),'r').read())
            c_cache = pjoin(self.options['ms_dir'], 'pure_interference_c')
            if os.path.exists(c_cache) and self._read_pi_c_cache(c_cache):
                cached = float(open(pjoin(self.options['ms_dir'], 'max_wgt'),'r').read())
                self._pi_max_weight = cached
                return cached

        nevents = self.options['Nevents_for_max_weight']
        if nevents == 0 :
            nevents = 75
        nb_ps_point = self.options['max_weight_ps_point']

        # Same even-split rounding as the sequential scan: round the probe events
        # up to a multiple of nb_core and reduce nb_ps_point to keep the sampling
        # budget, so the forked scan splits evenly across the workers.
        nb_core = self._resolve_nb_core()
        if nb_core > 1 and nevents % nb_core:
            budget = nevents * nb_ps_point
            nevents = int(math.ceil(nevents / float(nb_core))) * nb_core
            nb_ps_point = max(1, int(round(budget / float(nevents))))

        logger.info("Estimating the maximum weight")
        logger.info("*****************************")
        logger.info("Probing the first %s events with %s phase space points"
                    % (nevents, nb_ps_point))

        orig_lhe.seek(0)
        if self.options['fixed_order']:
            orig_lhe.eventgroup = True
        events = []
        for _ in range(nevents):
            try:
                events.append(next(orig_lhe))
            except StopIteration:
                break
        if not events:
            return 0.0

        # The probe events are independent, so the scan forks the same way as
        # the sequential one -- each worker owns a slice of the events and its
        # own view of the decay pools.
        nb_core = max(1, min(nb_core, len(events)))
        if nb_core == 1:
            all_maxwgt = self._joint_maxwgt_range(events, 0, len(events),
                                                  evt_decayfile, decay_dict,
                                                  nevents, nb_ps_point)
        else:
            logger.info("MadSpin: probing the maximum weight on %s cores", nb_core)
            all_maxwgt, _ = self._scan_maxwgt_parallel(
                orig_lhe, events, evt_decayfile, nb_core,
                self._joint_maxwgt_shard_entry, (decay_dict, nevents, nb_ps_point))

        base_max_weight = self._combine_maxwgt(all_maxwgt)
        if pure_interference:
            self._finalize_pi_c()
            self._finalize_pi_absw()
        if self.options['ms_dir']:
            open(pjoin(self.options['ms_dir'], 'max_wgt'),'w').write(str(base_max_weight))
            if pure_interference:
                self._write_pi_c_cache(pjoin(self.options['ms_dir'],
                                             'pure_interference_c'))
        self._pi_max_weight = float(base_max_weight)
        return base_max_weight

    # ------------------------------------------------------------------
    # c = <W_full>, the decay-side constant of the pure-interference mode
    # ------------------------------------------------------------------

    def _finalize_pi_c(self):
        """Turn the raw sum/sumsq/n the max-weight scan collected into
        ``self._pi_c`` (the estimate) and ``self._pi_c_err`` (its MC error).

        ``c`` is a *decay-side* constant -- the production density matrix
        cancels between the restricted contraction and its normalising trace --
        so averaging over the probe's production events as well as over its
        decay draws is legitimate and is what gives it sub-percent precision
        for free (section 13.13).
        """
        stats = getattr(self, '_pi_c_stats', None)
        n = (stats or {}).get('n', 0)
        if not n:
            raise self.InvalidCmd(
                "MadSpin: pure_interference could not measure the "
                "normalisation constant c = <W> -- the maximum-weight scan "
                "produced no usable sample. Raise Nevents_for_max_weight / "
                "max_weight_ps_point, or report this case.")
        mean = stats['sum'] / n
        var = max(stats['sumsq'] / n - mean * mean, 0.0)
        self._pi_c = mean
        self._pi_c_err = math.sqrt(var / n)
        analytic = getattr(self, '_pi_analytic_c', None)
        if not mean:
            raise self.InvalidCmd(
                "MadSpin: pure_interference measured c = <W> = 0 over %d "
                "trials. The fully weighted output divides by it, so the run "
                "cannot continue. This normally means the production density "
                "matrix is degenerate -- check the sample." % n)
        rel = self._pi_c_err / abs(mean)
        if analytic:
            logger.info(
                "MadSpin pure_interference: c = <W> = %.6e +- %.2f%% over %d "
                "trials; the analytic candidate 1/(prod_denominators * "
                "sym_decay) = %.6e, ratio %.4f. The measured value is the one "
                "used -- the analytic form is exact only where the chain "
                "carries no reshuffling jacobian.",
                mean, 100 * rel, n, analytic, mean / analytic)
        else:
            logger.info("MadSpin pure_interference: c = <W> = %.6e +- %.2f%% "
                        "over %d trials", mean, 100 * rel, n)
        if rel > 0.05:
            logger.warning(
                "MadSpin pure_interference: c is known to only %.1f%%, which "
                "is a flat scale error on every written weight. Raise "
                "Nevents_for_max_weight or max_weight_ps_point.", 100 * rel)

    def _finalize_pi_absw(self):
        """Turn the raw sum/sumsq/n of ``|W|`` the max-weight scan collected
        into ``self._pi_absw`` (the estimate) and ``self._pi_absw_err``.

        ``<|W|>`` is what the 'unweighted' output normalises with:

            w = +- sigma_ref * BR * <|W|> / c

        Derivation (section 13.17). Unweight one draw per production event on
        ``|W|/M`` for any bound ``M >= max|W|`` and write ``w = sign(W) * w0``.
        Then ``N_file = N_read * <|W|>/M`` and, for any observable ``O``,

            (1/N_file) sum_written w O = w0 * <W O> / <|W|>

        because ``|W| sign(W) = W``, and the ``M`` of the acceptance
        probability cancels against the ``M`` of ``N_file``. Matching the
        interference contribution ``sigma*BR*<W O>/c`` -- the same target the
        'weighted' output hits per read event -- gives ``w0 = sigma*BR*<|W|>/c``
        with **no ``M`` in it**: the accept/reject bound leaves the
        normalisation in this variant too. ``mean(w) = w0 <W>/<|W|> = 0``
        still, since ``<W> = 0`` for a pure-interference sample.

        Unlike ``c`` this is *not* a decay-side constant -- ``<|W|>`` is the
        local size of the interference and varies from production point to
        production point -- so the probe average is over its production events
        as well, and is only as representative as they are. That is a genuine
        extra scale uncertainty of this variant over the fully weighted one,
        and it is why the run cross-checks it against the realised keep rate
        (``N_file/N_read * M``, see _report_pure_interference).
        """
        stats = getattr(self, '_pi_absw_stats', None)
        n = (stats or {}).get('n', 0)
        if not n or not stats['sum']:
            self._pi_absw = 0.0
            self._pi_absw_err = 0.0
            if self._pure_interference_unweighted():
                raise self.InvalidCmd(
                    "MadSpin: pure_interference + decay_output = unweighted "
                    "needs <|W|>, the decay-phase-space mean of |W|, and the "
                    "maximum-weight scan measured %s over %d trials. Raise "
                    "Nevents_for_max_weight / max_weight_ps_point, or report "
                    "this case." % ('zero' if n else 'nothing', n))
            return
        mean = stats['sum'] / n
        self._pi_absw = mean
        # The error is the spread of the PER-PRODUCTION-EVENT means, not of the
        # individual trials: the nb_ps_point draws of one production point all
        # carry its own |W| scale, so the trial-level error is not an error on
        # <|W|> at all. Measured on p p > t t~: 0.46% trial-level against a
        # 9.5% production-event spread over the 110 probed events. Only the
        # second number says how well <|W|> is known -- which is why the run
        # does not trust it for the normalisation (see _report_pure_
        # interference: the realised keep rate replaces it).
        ev_n = stats.get('ev_n', 0)
        if ev_n > 1:
            ev_mean = stats['ev_sum'] / ev_n
            ev_var = max(stats['ev_sumsq'] / ev_n - ev_mean * ev_mean, 0.0)
            self._pi_absw_err = math.sqrt(ev_var / ev_n)
        else:
            var = max(stats['sumsq'] / n - mean * mean, 0.0)
            self._pi_absw_err = math.sqrt(var / n)
        rel = (self._pi_absw_err / mean) if mean else 0.0
        logger.info("MadSpin pure_interference: <|W|> = %.6e +- %.2f%% over %d "
                    "trials on %d production events (the error is the spread "
                    "over those events, which is what <|W|> is an average of)",
                    mean, 100 * rel, n, ev_n)

    def _write_pi_c_cache(self, path):
        """Persist the c and <|W|> measurements beside ``max_wgt`` in
        ``ms_dir``. Five fields; a three-field file is one written before
        <|W|> existed and is rejected by the reader when the run needs it."""
        try:
            with open(path, 'w') as fsock:
                fsock.write('%r %r %r %r %r\n' % (
                    self._pi_c, self._pi_c_err,
                    getattr(self, '_pi_analytic_c', 0.0) or 0.0,
                    getattr(self, '_pi_absw', 0.0) or 0.0,
                    getattr(self, '_pi_absw_err', 0.0) or 0.0))
        except Exception as exc:
            logger.warning('MadSpin: could not cache the pure-interference '
                           'constant c in %s (%s)', path, exc)

    def _read_pi_c_cache(self, path):
        """Read back what ``_write_pi_c_cache`` wrote. Returns False when the
        cache predates ``<|W|>`` and this run needs it, so the caller can fall
        through to a fresh scan rather than run on a missing normalisation."""
        values = open(path).read().split()
        self._pi_c = float(values[0])
        self._pi_c_err = float(values[1]) if len(values) > 1 else 0.0
        if len(values) > 2 and float(values[2]):
            self._pi_analytic_c = float(values[2])
        if len(values) > 4:
            self._pi_absw = float(values[3])
            self._pi_absw_err = float(values[4])
        elif self._pure_interference_unweighted():
            logger.info("MadSpin pure_interference: the cached constants in %s "
                        "predate <|W|>, which the 'unweighted' output needs; "
                        "re-running the maximum-weight scan.", path)
            return False
        logger.info("MadSpin pure_interference: c = %.6e read from the ms_dir "
                    "cache", self._pi_c)
        return True

    def _scan_maxwgt_range(self, events, start, stop, evt_decayfile,
                           nevents, nb_ps_point):
        """Per-event probe data for ``events[start:stop]``, and the samples of
        the rate factor collected along the way.

        Returns ``(per_event, z_samples)``, or ``(None, {})`` as soon as a
        production event turns out to have nothing to decay (the caller then
        falls back to the joint bound).

        Under ``sequential_with_mass`` ``per_event`` is one max-weight vector
        per event, holding for each ordering position the largest w_k over
        ``nb_ps_point`` chains -- all the bound needs. The up-front-mass schemes
        keep every chain instead: their mass-set weight is only complete once
        Z_k is known, and Z_k is fitted from ``z_samples``, which this same
        probe produces. Taking the maximum there is deferred to the caller, over
        the completed weights.
        """
        self.efficiency = 1. / nb_ps_point
        upfront = self._sequential_upfront()
        t0 = time.time()
        per_event = []
        z_samples = collections.defaultdict(list)
        for i in range(start, stop):
            if (i - start) % 5 == 1 and getattr(self, '_shard_tag', None) in (None, 0):
                # only one worker prints scan progress
                logger.info("Event %s/%s :  %2fs" % (i, stop, time.time()-t0))
            base_event = events[i]
            # events left in *this* range (the worker's shard), not the global
            # nevents - i. It only feeds the decay-pool refill sizing, and the
            # refill already multiplies by nb_core to share the pool across
            # workers -- so passing the global count made a forked scan worker
            # refill nb_core times too many decays (900k instead of ~60k).
            nb_remain = stop - i
            best = None
            chains = []
            extra = {}
            for _ in range(nb_ps_point):
                probe = []
                out = self.sequential_accept_reject(base_event, evt_decayfile,
                                                    None, nb_remain, probe=probe,
                                                    probe_extra=extra)
                if out is None:
                    return None, {}
                if upfront:
                    chains.append([list(probe), list(extra['mass'])])
                    for key, mass, value in extra.pop('z', ()):
                        z_samples[key].append((mass, value))
                elif best is None:
                    best = list(probe)
                else:
                    best = [max(old, new) for old, new in zip(best, probe)]
            if upfront:
                if chains:
                    per_event.append({'keys': extra['keys'],
                                      'order': extra['order'],
                                      'chains': chains})
            elif best:
                per_event.append(best)
        return per_event, dict(z_samples)

    def _scan_maxwgt_shard_entry(self, shard_id, nb_core, events, start, stop,
                                 evt_decayfile, nevents, nb_ps_point, out_path):
        """Worker entry (forked child): scan its slice of the probe events and
        write the per-event vectors as JSON, mirroring _unweight_shard_entry --
        its own RNG stream, its own reopened decay pools, failures reported in
        the JSON rather than raised into the parent."""
        import json
        try:
            random.seed((int(self.seed) if self.seed else 0)
                        + 7919 * (shard_id + 1))
            self._shard_tag = shard_id
            self._shard_nb_core = nb_core
            self._pool_gen = {}
            self._init_owner_refill(evt_decayfile, self.seed)
            local_pool = self._reopen_decay_pool(evt_decayfile, shard_id, nb_core)
            per_event, z_samples = self._scan_maxwgt_range(
                            events, start, stop, local_pool, nevents, nb_ps_point)
            with open(out_path, 'w') as f:
                json.dump({'per_event': per_event, 'z_samples': z_samples}, f)
        except Exception as exc:
            import traceback
            try:
                with open(out_path, 'w') as f:
                    json.dump({'error': str(exc),
                               'tb': traceback.format_exc()}, f)
            except Exception:
                pass
        finally:
            self._set_status('D')   # release any worker blocked on a channel I own

    def _joint_maxwgt_range(self, events, start, stop, evt_decayfile, decay_dict,
                            nevents, nb_ps_point):
        """Per-event maximum weight for the *joint* accept/reject, over
        ``events[start:stop]``: for each production event, the largest
        full_me/(prod*dec)*jac seen over nb_ps_point decay draws. The serial body
        of get_maxwgt_for_onshell, factored so it can also run in a forked
        worker."""
        self.efficiency = 1. / nb_ps_point
        t0 = time.time()
        density_pole_approximation = self._density_pole_approximation()
        density_needs_reshuffle = self._density_needs_reshuffle(
            self.generate_all.mode == 'density')
        # pure interference: the weight is signed and its mean is zero, so a
        # max() seeded at 0 and fed the signed value would bound only the
        # positive excursions. The mode no longer accept/rejects, so this is a
        # diagnostic (the largest |W| the probe saw, reported in the banner)
        # rather than a bound -- but it is only a *meaningful* diagnostic on
        # |W|, so the abs() stays.
        signed = bool(self._pure_interference())
        # ... and the same scan measures c = <W_full>, the decay-side constant
        # the fully weighted output divides by. One extra contraction per draw
        # on matrices that are alive anyway (section 13.13). decay_output =
        # weighted needs the same constant, and gets it from the same place --
        # there the "unrestricted" contraction IS the ordinary one (the trace
        # restriction defaults to the contraction restriction), so the swap in
        # _pi_unrestricted_contraction is a no-op and c = <W>, exactly the
        # quantity that makes mean(w) = sigma*BR.
        probe_c = signed or self._weighted_decay()
        self._pi_probe_c = probe_c
        pi_c_sum = 0.0
        pi_c_sumsq = 0.0
        pi_c_n = 0
        # ... and, on the same draws, <|W|>: the decay-phase-space mean of the
        # ABSOLUTE restricted convolution. It is what normalises the
        # 'unweighted' output (w = +- sigma*BR*<|W|>/c, section 13.17), and a
        # free diagnostic for the 'weighted' one, so it is always collected
        # when the mode is on. Unlike c it is not a decay-side constant -- it
        # varies from production point to production point -- so what the probe
        # measures is its average over the probe's production events, which is
        # exactly the global mean the derivation needs.
        pi_absw_sum = 0.0
        pi_absw_sumsq = 0.0
        pi_absw_n = 0
        # ... and the same thing BLOCKED by production event. The nb_ps_point
        # draws of one production point share its a_p, so treating all
        # nevents*nb_ps_point trials as independent understates the error on
        # <|W|> by more than an order of magnitude (measured: 0.46% claimed
        # against a 9.5% production-event spread). The honest error is the
        # spread of the per-production-event means over the probe's production
        # events, which is what these three accumulate.
        pi_absw_ev_sum = 0.0
        pi_absw_ev_sumsq = 0.0
        pi_absw_ev_n = 0
        per_event = []
        for i in range(start, stop):
            if (i - start) % 5 == 1 and getattr(self, '_shard_tag', None) in (None, 0):
                logger.info("Event %s/%s :  %2fs" % (i, stop, time.time()-t0))
            base_event = events[i]
            if self.options['fixed_order']:
                base_event = base_event[0]
            maxwgt = 0
            ev_absw_sum = 0.0     # this production event's own |W| draws
            ev_absw_n = 0
            density_matrix_prod = None
            offshell_density = (self.generate_all.mode == 'density'
                                and not density_pole_approximation)
            for j in range(nb_ps_point):
                # stop - i (this worker's remaining events) sizes the pool refill
                decays = self.get_decay_from_file(base_event, evt_decayfile, stop - i)
                # offshell/madspin reshuffles the production event in place; use a
                # per-draw onshell copy so repeated draws don't compound and so the
                # reshuffle jacobian (now folded into wgt) is taken from the onshell
                # reference each draw.
                prod_draw = lhe_parser.Event(str(base_event)) if offshell_density else base_event
                if density_matrix_prod is None:
                    _, wgt, density_matrix_prod = self.get_onshell_evt_and_wgt(
                        prod_draw, decays, decay_dict, build_event=False)
                else:
                    wgt = self.get_onshell_evt_and_wgt(
                        prod_draw, decays, decay_dict, density_matrix_prod,
                        build_event=False)[1]
                jac = 1
                if (density_needs_reshuffle and not offshell_density
                        and self.options['density_keep_jacobian']):
                    # PA with explicit jacobian tracking: reshuffle to expose the
                    # jacobian in the max weight. Offshell/madspin does NOT enter
                    # here -- its reshuffle jacobian is already inside wgt.
                    full_evt = lhe_parser.Event(str(base_event))
                    full_evt = full_evt.add_decays(decays)
                    jac = full_evt.reshuffle_production()
                maxwgt = max(abs(wgt*jac) if signed else wgt*jac, maxwgt)
                if probe_c:
                    restricted = wgt*jac
                    restricted = float(getattr(restricted, 'real', restricted))
                    if math.isfinite(restricted):
                        pi_absw_sum += abs(restricted)
                        pi_absw_sumsq += restricted * restricted
                        pi_absw_n += 1
                        ev_absw_sum += abs(restricted)
                        ev_absw_n += 1
                    sample = getattr(self, '_pi_unrestricted_wgt', None)
                    if sample is not None:
                        # the outer jacobian (PA with density_keep_jacobian) is
                        # applied by this loop, not inside the weight, exactly
                        # as it is for wgt just above
                        sample = float(getattr(sample, 'real', sample)) * jac
                        if math.isfinite(sample):
                            pi_c_sum += sample
                            pi_c_sumsq += sample * sample
                            pi_c_n += 1
            if probe_c and ev_absw_n:
                ev_mean = ev_absw_sum / ev_absw_n
                pi_absw_ev_sum += ev_mean
                pi_absw_ev_sumsq += ev_mean * ev_mean
                pi_absw_ev_n += 1
            per_event.append(float(getattr(maxwgt, 'real', maxwgt)))
        if probe_c:
            self._pi_probe_c = False
            stats = getattr(self, '_pi_c_stats', None) or {'sum': 0.0,
                                                           'sumsq': 0.0, 'n': 0}
            stats['sum'] += pi_c_sum
            stats['sumsq'] += pi_c_sumsq
            stats['n'] += pi_c_n
            self._pi_c_stats = stats
            astats = getattr(self, '_pi_absw_stats', None) or {'sum': 0.0,
                                                               'sumsq': 0.0,
                                                               'n': 0}
            astats['sum'] += pi_absw_sum
            astats['sumsq'] += pi_absw_sumsq
            astats['n'] += pi_absw_n
            astats['ev_sum'] = astats.get('ev_sum', 0.0) + pi_absw_ev_sum
            astats['ev_sumsq'] = astats.get('ev_sumsq', 0.0) + pi_absw_ev_sumsq
            astats['ev_n'] = astats.get('ev_n', 0) + pi_absw_ev_n
            self._pi_absw_stats = astats
        return per_event

    def _joint_maxwgt_shard_entry(self, shard_id, nb_core, events, start, stop,
                                  evt_decayfile, decay_dict, nevents, nb_ps_point,
                                  out_path):
        """Worker entry (forked child) for the joint max-weight scan. Mirrors
        _scan_maxwgt_shard_entry: own RNG stream, own reopened decay pools,
        failures reported in the JSON."""
        import json
        try:
            random.seed((int(self.seed) if self.seed else 0)
                        + 7919 * (shard_id + 1))
            self._shard_tag = shard_id
            self._shard_nb_core = nb_core
            self._pool_gen = {}
            self._init_owner_refill(evt_decayfile, self.seed)
            local_pool = self._reopen_decay_pool(evt_decayfile, shard_id, nb_core)
            per_event = self._joint_maxwgt_range(events, start, stop, local_pool,
                                                 decay_dict, nevents, nb_ps_point)
            with open(out_path, 'w') as f:
                # pi_c: the shard's share of the c measurement, merged
                # additively in the parent (sum/sumsq/n are order-independent,
                # so one shard or many gives the identical estimate)
                json.dump({'per_event': per_event,
                           'pi_c': getattr(self, '_pi_c_stats', None),
                           'pi_absw': getattr(self, '_pi_absw_stats', None),
                           'pi_analytic_c': getattr(self, '_pi_analytic_c', None)}, f)
        except Exception as exc:
            import traceback
            try:
                with open(out_path, 'w') as f:
                    json.dump({'error': str(exc),
                               'tb': traceback.format_exc()}, f)
            except Exception:
                pass
        finally:
            self._set_status('D')   # release any worker blocked on a channel I own

    def _scan_maxwgt_parallel(self, orig_lhe, events, evt_decayfile, nb_core,
                              shard_entry, extra):
        """Fork one worker per contiguous slice of the probe events; each runs
        ``shard_entry`` and returns its per-event data through a JSON file.
        Concatenating them is order independent -- _combine_maxwgt takes the
        max/spread over all events -- so the result matches the serial scan up to
        which decays each draw pulls. ``extra`` is the tuple of arguments the
        shard entry needs after ``evt_decayfile`` (the joint and sequential scans
        pass different ones). Used by both get_maxwgt_for_onshell and
        get_sequential_maxwgt."""
        import multiprocessing as mp
        import json
        base = '%s.maxwgt' % orig_lhe.name
        # ``nb_core`` is the pool-addressing count -- the decay pool was split
        # into that many files and each worker must address it with the same
        # count to open *its* file (paths[shard_id]) instead of falling back to
        # striding. It is ALSO the modulus of _channel_owner, so every id it can
        # name has to belong to a worker that is actually forked. Both hold only
        # if exactly nb_core workers run, which is why the slices are balanced
        # (below) rather than ceil-chunked, and why nb_core is capped at the
        # number of probe events here as well as at the call sites.
        nb_core = max(1, min(int(nb_core), len(events)))
        ranges = self._balanced_ranges(len(events), nb_core)

        self._clear_worker_status(nb_core)   # fresh status board for this phase
        # Belt and braces: should a slice ever come back empty anyway, mark the
        # worker that will not be forked as DONE. A missing status file is
        # indistinguishable from a worker that has not published yet, so a
        # waiter blocked on such an owner would sit out MADSPIN_REFILL_WAIT
        # (3600s by default) and then kill the scan; 'D' makes the fail-safe in
        # _worker_refill fire at once instead.
        for sid in range(len(ranges), nb_core):
            try:
                with open(self._status_path(sid), 'w') as f:
                    f.write('D')
            except (IOError, OSError):
                pass
        mpctx = mp.get_context('fork')
        procs, out_paths = [], []
        for sid, (start, stop) in enumerate(ranges):
            outp = '%s.shard%d.json' % (base, sid)
            p = mpctx.Process(
                target=shard_entry,
                args=(sid, nb_core, events, start, stop, evt_decayfile)
                     + tuple(extra) + (outp,))
            p.start()
            procs.append(p)
            out_paths.append(outp)
        for p in procs:
            p.join()

        per_event = []
        result = per_event
        z_samples = collections.defaultdict(list)
        for sid, outp in enumerate(out_paths):
            if not os.path.exists(outp):
                raise Exception("MadSpin max-weight worker %s produced no result "
                                "(crashed). Re-run with nb_core=1 to debug." % sid)
            with open(outp) as f:
                r = json.load(f)
            if 'error' in r:
                raise Exception("MadSpin max-weight worker %s failed:\n%s"
                                % (sid, r.get('tb', r['error'])))
            if r['per_event'] is None:
                result = None   # nothing to decay: fall back to the joint bound
            elif result is not None:
                result.extend(r['per_event'])
            for key, samples in (r.get('z_samples') or {}).items():
                # json turns the (mass, value) pairs into lists
                z_samples[key].extend((s[0], s[1]) for s in samples)
            pi_c = r.get('pi_c')
            if pi_c:
                merged = getattr(self, '_pi_c_stats', None) or {
                    'sum': 0.0, 'sumsq': 0.0, 'n': 0}
                for key in ('sum', 'sumsq', 'n'):
                    merged[key] += pi_c.get(key, 0)
                self._pi_c_stats = merged
            pi_absw = r.get('pi_absw')
            if pi_absw:
                merged = getattr(self, '_pi_absw_stats', None) or {}
                for key in ('sum', 'sumsq', 'n',
                            'ev_sum', 'ev_sumsq', 'ev_n'):
                    merged[key] = merged.get(key, 0) + pi_absw.get(key, 0)
                self._pi_absw_stats = merged
            if r.get('pi_analytic_c'):
                self._pi_analytic_c = r['pi_analytic_c']
        for outp in out_paths:
            try:
                os.remove(outp)
            except OSError:
                pass
        return result, dict(z_samples)

    def get_sequential_maxwgt(self, orig_lhe, evt_decayfile):
        """One bound C_k per position of the decay ordering, for the sequential
        accept/reject. Returns [] when nothing decays.

        Same probe as the joint scan -- the first Nevents_for_max_weight
        production events, max_weight_ps_point sets of decays each, the largest
        weight per production event, then mean + nb_sigma*sd -- but applied to
        each slot's own weight.

        The probe draws every slot from the pool uniformly whereas the real
        chain draws slot k conditioned on the decays it has accepted. The
        support is the same, so this estimates the same bound; only the density
        with which the tail is explored differs, which is why the margins are
        kept and the accept/reject counts its overflows.
        """
        offshell = self._sequential_offshell()
        upfront = self._sequential_upfront()
        cache = None
        if self.options['ms_dir']:
            # a distinct name: the joint bound is a single float, this is a list.
            # The up-front-mass bounds come with the Z_k tables and depend on the
            # unweighting mode *and* on the spinmode family (the mass-set weight
            # is a different quantity offshell and under PA), so they get a name
            # (and a format) of their own -- a cache written for one cannot be
            # read back for the other. The ``offshell``/``pa`` piece of the file
            # name names the spinmode family that wrote it, which is still what
            # it does now that both families take the up-front-mass path; it is
            # deliberately left alone so that caches already on disk keep being
            # found.
            if upfront:
                mode = self._unweighting_mode()
                variant = '' if mode == 'sequential' else '_%s' % mode
                cache = pjoin(self.options['ms_dir'],
                              'max_wgt_sequential_%s%s'
                              % ('offshell' if offshell else 'pa', variant))
                cached = self._read_upfront_cache(cache)
                if cached is not None:
                    self._z_tables = cached['z_tables']
                    return cached['maxwgts']
            else:
                cache = pjoin(self.options['ms_dir'], 'max_wgt_sequential')
                if os.path.exists(cache):
                    return [float(x) for x in open(cache).read().split()]

        nevents = self.options['Nevents_for_max_weight']
        if nevents == 0:
            nevents = 75
        nb_ps_point = self.options['max_weight_ps_point']

        # Round the number of probe events up to a multiple of nb_core so the
        # parallel scan splits evenly -- every worker gets the same number of
        # events, no worker is the odd one out with an extra event whose pool
        # slice runs short. Reduce nb_ps_point to keep the total decays drawn
        # (nevents * nb_ps_point, the sampling budget) roughly unchanged.
        nb_core = self._resolve_nb_core()
        if nb_core > 1 and nevents % nb_core:
            budget = nevents * nb_ps_point
            nevents = int(math.ceil(nevents / float(nb_core))) * nb_core
            nb_ps_point = max(1, int(round(budget / float(nevents))))

        logger.info("Estimating the maximum weight of each decaying particle")
        logger.info("*****************************")
        logger.info("Probing the first %s events with %s phase space points"
                    % (nevents, nb_ps_point))
        # a non-joint unweighting never reaches here with fixed_order (it falls back to
        # the joint accept/reject), so the events are plain, not event-groups.
        orig_lhe.seek(0)
        events = []
        for _ in range(nevents):
            try:
                events.append(next(orig_lhe))
            except StopIteration:
                break
        if not events:
            return []

        # The probe events are independent, exactly like the unweighting, so the
        # scan forks the same way -- each worker owns a slice of the events and
        # its own view of the decay pools.
        nb_core = self._resolve_nb_core()
        nb_core = max(1, min(nb_core, len(events)))
        if nb_core == 1:
            per_event, z_samples = self._scan_maxwgt_range(
                events, 0, len(events), evt_decayfile, nevents, nb_ps_point)
        else:
            logger.info("MadSpin: probing the maximum weight on %s cores", nb_core)
            per_event, z_samples = self._scan_maxwgt_parallel(
                orig_lhe, events, evt_decayfile, nb_core,
                self._scan_maxwgt_shard_entry, (nevents, nb_ps_point))

        if per_event is None:
            return []   # a production event had nothing to decay
        if len(per_event) < 2:
            # _combine_maxwgt needs a spread to work with
            return []

        if upfront:
            self._z_tables = self._build_z_tables(z_samples)
            per_event = [self._complete_upfront_probe(event)
                         for event in per_event]

        maxwgts = [self._combine_maxwgt([event[slot] for event in per_event])
                   for slot in range(len(per_event[0]))]
        logger.info("Sequential maximum weights: %s",
                    ' '.join('%.4g' % w for w in maxwgts))
        if cache and upfront:
            import json
            with open(cache, 'w') as f:
                json.dump({'format': self._UPFRONT_CACHE_FORMAT,
                           'maxwgts': maxwgts, 'z_tables': self._z_tables}, f)
        elif cache:
            open(cache, 'w').write(' '.join(repr(w) for w in maxwgts))
        return maxwgts

    # Bumped whenever the cache's *meaning* changes: another entry in the bound
    # vector, a different fit variable or degree, another key in a table. The
    # file name already separates the unweighting modes, and the offshell
    # spinmodes from PA/onshell; this separates one version of this code from
    # the next, which a name cannot.
    # 2: the mass-set weight is normalised by |M_prod|^2 on shell, so every
    #    bound in the vector changed scale.
    # (Renaming this constant from _OFFSHELL_CACHE_FORMAT, when the up-front
    #  mass draw stopped being offshell-only, is *not* such a change: the
    #  payload is the same, so the tag stays at 2 and caches already written
    #  keep being accepted.)
    _UPFRONT_CACHE_FORMAT = 2

    def _read_upfront_cache(self, path):
        """The cached up-front-mass bounds and Z_k tables, or None if there is
        nothing usable there.

        A cache that does not match what this code writes is *ignored*, not
        repaired and not raised on: the scan that produced it is reproducible,
        so paying for it again is always an option, whereas a table read under
        the wrong schema would either crash deep inside the accept/reject or --
        worse -- silently weight the virtualities with somebody else's fit.
        Hence a format tag, and a structural check of every field the
        accept/reject will dereference.
        """
        if not path or not os.path.exists(path):
            return None
        import json
        try:
            with open(path) as f:
                cached = json.load(f)
            if cached.get('format') != self._UPFRONT_CACHE_FORMAT:
                raise ValueError('format %s, expected %s'
                                 % (cached.get('format'),
                                    self._UPFRONT_CACHE_FORMAT))
            maxwgts = [float(w) for w in cached['maxwgts']]
            if not maxwgts:
                raise ValueError('no bounds')
            tables = cached['z_tables']
            for key, table in tables.items():
                missing = {'pole', 'coeff', 'zero_below',
                           'range'} - set(table)
                if missing:
                    raise ValueError('slot %s is missing %s'
                                     % (key, ', '.join(sorted(missing))))
                if len(table['coeff']) != 3 or len(table['range']) != 2:
                    raise ValueError('slot %s has a malformed fit' % key)
        except Exception as error:
            logger.warning("MadSpin: ignoring the cached sequential maximum "
                           "weights in %s (%s); they will be measured again.",
                           path, error)
            return None
        return {'maxwgts': maxwgts, 'z_tables': tables}

    def _complete_upfront_probe(self, event):
        """The per-event maximum-weight vector of an up-front-mass probe, over
        the chains it recorded and with the Z_k factors the loop could not apply
        while they were still being measured: Z_k(m_k) into the mass-set weight,
        and -- under sequential_global_retry, where the mass stage pays it and the
        per-angle stage takes it back -- 1/Z_k into that slot's own weight.
        """
        keys, order = event['keys'], event['order']
        mode = self._unweighting_mode()
        joint_angles = mode == 'two_stage'
        # Both sequential_global_retry and two_stage test w_k/Z_hat_k rather
        # than w_k: the first because the mass stage has already paid Z_hat and
        # the two must cancel, the second because it flattens the virtuality
        # dependence out of the bound. The probe has to be completed the same
        # way or the bound and the weight it bounds are different quantities.
        exact = joint_angles or mode == 'sequential_global_retry'
        best = None
        for weights, masses in event['chains']:
            zhat = [self._zhat(key, mass) for key, mass in zip(keys, masses)]
            current = list(weights)
            for z in zhat:
                current[0] *= z
            if exact:
                for position, slot in enumerate(order):
                    current[position + 1] = (weights[position + 1] / zhat[slot]
                                             if zhat[slot] > 0 else 0.0)
            if joint_angles:
                # one bound over the product, so the vector is [C_mass, C_angles]
                product = 1.0
                for value in current[1:]:
                    product *= value
                current = [current[0], product]
            best = current if best is None else \
                   [max(old, new) for old, new in zip(best, current)]
        return best

    def _combine_maxwgt(self, all_maxwgt):
        """Turn the per-production-event maxima of a probe into the bound the
        accept/reject uses: mean + nb_sigma*sd with a safety margin, refined on
        the largest ones and never below the second largest seen.

        Shared by the joint bound and by each slot's bound in sequential mode.
        The sequential accept/reject has no way to carry a per-slot overweight
        forward (redraw-until-accept, not staged unweighting), so a weight above
        the bound biases the sample directly -- hence a 10% margin rather than
        the historical 5%.
        """
        margin = 1.10
        # A bound that is not a finite positive number cannot ever accept
        # anything: the accept/reject test is `random()*bound < wgt`, so a
        # bound of 0 rejects every trial and a NaN bound (which is what a
        # 0/0 weight produces) rejects it too -- and then the unweighting
        # loop redraws for ever. Catch it here, where the whole probe is in
        # hand, instead of after the fact. Note this also replaces the bare
        # `assert all_maxwgt[0] >= all_maxwgt[1]` below, which a NaN in the
        # list used to trip with an empty message.
        if all_maxwgt and (not all(math.isfinite(w) for w in all_maxwgt)
                           or not max(all_maxwgt) > 0):
            self._raise_degenerate_weight(
                "the maximum-weight scan measured no usable bound: every one "
                "of the %d probed production events gave a weight that is "
                "zero or not a number (%s)."
                % (len(all_maxwgt),
                   ', '.join('%.4g' % w for w in all_maxwgt[:5])
                   + (', ...' if len(all_maxwgt) > 5 else '')))
        all_maxwgt.sort(reverse=True)
        assert all_maxwgt[0] >= all_maxwgt[1], "ERROR: "
        decay_tools=madspin.decay_misc()
        ave_weight, std_weight = decay_tools.get_mean_sd(all_maxwgt)
        base_max_weight = margin * (ave_weight+self.options['nb_sigma']*std_weight)

        for i in [20, 30, 40, 50]:
            if len(all_maxwgt) < i:
                break
            ave_weight, std_weight = decay_tools.get_mean_sd(all_maxwgt[:i])
            base_max_weight = max(base_max_weight, margin * (ave_weight+self.options['nb_sigma']*std_weight))

            if all_maxwgt[1] > base_max_weight:
                base_max_weight = margin * all_maxwgt[1]
        return base_max_weight

    # ------------------------------------------------------------------
    # Guards against an accept/reject that can never accept
    # ------------------------------------------------------------------
    # Every weight of the density spinmodes is built from the production spin
    # density matrix rho_prod: the joint one is <rho_prod, rho_dec> over a
    # denominator, the sequential one a ratio of such contractions. If rho_prod
    # is identically zero then so is every numerator, the accept/reject rejects
    # every trial, the decay-event pools are drained and regenerated, and
    # MadSpin runs for ever without writing a single event (measured: 600 s and
    # 131 pool regenerations with no output). The guards below turn that into an
    # immediate, named failure.
    #
    # They are deliberately built so that they cannot fire on a healthy run,
    # however inefficient it is. A low acceptance means small *positive*
    # weights: the bound is positive, some trial eventually wins the test, and
    # the dead-trial counter is reset by the first positive weight it sees.
    # What is caught here is a weight that is structurally zero -- exactly 0, or
    # NaN from a 0/0 -- for which no number of extra draws can help.

    def _raise_degenerate_weight(self, what, extra=''):
        """Abort with an explanation of a MadSpin accept/reject that can never
        accept, and of what usually causes it."""
        msg = ("MadSpin cannot decay these events: %s\n"
               "\n"
               "The accept/reject weight of the density spin modes is built "
               "from the production spin-density matrix, so a production "
               "density matrix that is identically zero makes EVERY trial "
               "weight zero and EVERY trial rejected (an identically zero "
               "*decay* density matrix has the same effect). MadSpin would keep "
               "drawing decays, regenerating decay-event pools and never "
               "writing an event, so it stops here instead of looping.\n"
               "\n"
               "Plausible causes, most likely first:\n"
               "  * a polarised production process (a '{L}', '{R}', '{0}', "
               "'{T}', '{+}', '{-}' tag on the generation line). The Fortran "
               "GET_DENSITY picks the NHEL rows of the standalone matrix "
               "element by matching them against the ALLOW_HEL helicity "
               "combinations; a polarised matrix element only keeps the "
               "polarised rows, so the combination the density matrix is "
               "indexed on can be missing altogether and the matrix comes back "
               "all zeros;\n"
               "  * a mismatch between the event file and the matrix elements "
               "MadSpin is using -- a stale 'ms_dir'/'use_old_dir' directory, "
               "or a param_card different from the one the events were "
               "generated with;\n"
               "  * a helicity subspace emptied by the beam polarisation "
               "('beampol') or by the helicity frame ('frame_id').\n"
               "\n"
               "'set spinmode none' switches the density matrices off "
               "altogether and will produce events (without spin "
               "correlations); 'set density_debug True' compares the density "
               "matrices against the full matrix element event by event.")
        raise MadSpinDegenerateWeight(msg % what + (('\n\n' + extra) if extra else ''))

    def _check_production_density(self, event, density_prod, stage=''):
        """Fail on a production spin-density matrix whose trace vanishes.

        Tr(rho_prod) is the production matrix element squared restricted to the
        helicity subspace the density matrix is built on -- that identity is
        exactly what ``density_debug`` checks event by event
        (``prod_diag``/``prod_me``). So a zero trace means the density matrix
        carries no matrix element at all, and every weight derived from it is
        zero (offshell) or NaN (PA/onshell, which divide by that same trace).

        Why fail on the *first* such production event rather than after a
        bounded number of fruitless pool regenerations: the quantity that is
        broken belongs to the production event and to the helicity basis, not to
        the decays being drawn against it, so redrawing decays cannot change it
        and waiting only costs minutes to hours. The risk that a *legitimate*
        zero-matrix-element phase-space point aborts a healthy run is removed by
        cross-checking against the full production matrix element at the very
        same momenta: if |M_prod|^2 > 0 while Tr(rho_prod) = 0 the two are
        inconsistent by construction, which no phase-space point can be. (And if
        |M_prod|^2 vanishes too, the event cannot be decayed by any
        accept/reject either -- it is reported as its own case.)

        The check itself is a comparison on a number the callers compute anyway,
        and the extra matrix element is only ever evaluated on the failing
        branch, so a healthy run pays nothing for it.
        """
        try:
            trace = float(density_prod.trace().real)
        except Exception:
            return None       # not a density matrix we know how to inspect
        if math.isfinite(trace) and trace > 0:
            return trace

        try:
            tag, _ = event.get_tag_and_order()
            process = '%s > %s' % (' '.join(str(p) for p in tag[0]),
                                   ' '.join(str(p) for p in tag[1]))
        except Exception:
            process = 'unknown'
        try:
            me_prod = float(self.calculate_matrix_element(event))
        except Exception:
            me_prod = None

        where = (' (%s)' % stage) if stage else ''
        what = ("the production spin-density matrix of process '%s' is "
                "identically zero%s -- Tr(rho_prod) = %s."
                % (process, where, trace))
        if me_prod is not None and me_prod > 0:
            extra = ("Diagnostic: the *full* production matrix element at the "
                     "same phase-space point is |M_prod|^2 = %.6g, which is "
                     "NOT zero. Tr(rho_prod) and |M_prod|^2 must agree (that is "
                     "what 'density_debug' checks), so this is not a vanishing "
                     "phase-space point: the helicity basis the density matrix "
                     "is indexed on does not exist in the generated matrix "
                     "element." % me_prod)
        elif me_prod is not None:
            extra = ("Diagnostic: the full production matrix element vanishes "
                     "as well (|M_prod|^2 = %.6g), so this production event "
                     "carries no matrix element at all and cannot be decayed by "
                     "any accept/reject. Check that the event file really is "
                     "the one this process/param_card was generated with."
                     % me_prod)
        else:
            extra = ("Diagnostic: the full production matrix element could not "
                     "be evaluated for a cross-check.")
        self._raise_degenerate_weight(what, extra)

    def _dead_trial(self, counter, wgt, stage):
        """Bounded backstop for an accept/reject loop whose weight stays dead.

        ``counter`` is the number of consecutive trials so far whose weight was
        not a finite positive number; returns the updated counter and raises
        once it passes ``MS_MAX_DEAD_TRIALS``. Any single positive weight resets
        it to 0, which is what keeps a legitimately inefficient run -- small but
        positive weights, occasionally accepted -- from ever reaching the bound.
        This catches the causes ``_check_production_density`` does not, e.g. a
        decay density matrix that is structurally zero.

        ``math.isfinite`` and not ``numpy.isfinite``, deliberately: every weight
        that reaches here is a real scalar (the density contraction has its real
        part taken by ``ms_density_real``, and the sequential stages take theirs
        at ``(n_k / n_prev).real``), so the coercion ``math.isfinite`` performs
        is free -- and if a complex weight is ever reintroduced upstream this is
        the line that says so, with a ComplexWarning naming the file and the
        line. ``numpy.isfinite`` would accept it in silence, and ``wgt.real``
        would discard the imaginary part without anyone finding out; neither is
        an improvement on being told.
        """
        try:
            ok = math.isfinite(wgt) and wgt > 0
        except TypeError:
            ok = False
        if ok:
            return 0
        counter += 1
        if counter >= MS_MAX_DEAD_TRIALS:
            self._raise_degenerate_weight(
                "%s produced %d consecutive trials with a weight that is zero "
                "or not a number, without a single positive one in between."
                % (stage, counter),
                "Diagnostic: this is not a low acceptance (which gives small "
                "but positive weights, and would have reset this counter); the "
                "weight is structurally dead, so no number of further decay "
                "draws or decay-pool regenerations can ever produce an "
                "accepted event.")
        return counter


    def _density_basis(self, production, decays_key):
        """Helicity-basis bookkeeping for the production density matrix: which
        particles decay, where they sit (``position``, ``init_part``), their
        helicity bases (``helicities``, and the ``allowed_hel``/``ncomb``/
        ``dimension`` the Fortran side needs), plus the averaging and identical
        final-state symmetry factors.

        It depends only on the production event and on *which* pdgs decay --
        not on the decay events, and not on any sampled mass -- so it is
        computed once per production event and reused across every retry, and
        across every slot of the sequential accept/reject.
        """
        # Production averaging factor (spin/color initial state) from standalone
        iden_p = self.get_iden(production)

        # Symmetry factor for identical final states in production
        final_pdgs = [int(p.pid) for p in production if getattr(p, "status", None) == 1]
        counts_final = collections.Counter(final_pdgs)
        sym_factor_prod_ident = 1
        for n in counts_final.values():
            if n > 1:
                sym_factor_prod_ident *= math.factorial(n)

        # Find particles that should decay (status==1 and pid in decays keys)
        init_part = [part for pdg in decays_key for part in production
                     if part.pid == pdg and part.status == 1]
        nchanging = len(init_part)

        # Allowed helicities per spin
        hel_dict = {1: [0], 2: [1, -1], 3: [-1, 0, 1]}

        # Decaying-particle positions (+1 for Fortran), spins, helicities
        position = [i + 1 for pdg in decays_key
                    for i in range(len(production))
                    if production[i].pid == pdg and production[i].status == 1]
        decaying_pdg = [int(production[i - 1].pid) for i in position]
        decaying_spins = [self.model.get_particle(i).get('spin') for i in decaying_pdg]
        helicities = [hel_dict[i] for i in decaying_spins]

        helicities, hel_restriction = self._apply_production_polarization(
                                                    decaying_pdg, helicities)
        # pure-interference mode replaces the symmetric restriction by a cross
        # one for the particles the card names, and keeps the symmetric one as
        # the (separate) normalising trace -- see _apply_pure_interference
        hel_restriction, hel_restriction_trace = self._apply_pure_interference(
                                    decaying_pdg, helicities, hel_restriction)

        allowed_hel_pairs, allowed_hel = self.get_allowed_hel(helicities)

        return {
            'decays_key': decays_key,
            'iden_p': iden_p,
            'sym_factor_prod_ident': sym_factor_prod_ident,
            'init_part': init_part,
            'nchanging': nchanging,
            'position': position,
            'helicities': helicities,
            'decaying_pdg': decaying_pdg,
            'decaying_spins': decaying_spins,
            'allowed_hel': allowed_hel,
            'hel_restriction': hel_restriction,
            'hel_restriction_trace': hel_restriction_trace,
            'pure_interference': bool(self._pure_interference_pdgs(decays_key)),
            'ncomb': len(allowed_hel_pairs),
            'dimension': math.prod(len(i) for i in helicities),
        }

    # ------------------------------------------------------------------
    # Production polarisation ({0}/{+}/{-}/{L}/{R}/{T} on the decaying leg)
    # ------------------------------------------------------------------

    def _density_spinmode(self):
        """Whether the current spinmode goes through the density-matrix path.
        'full' is the user-facing alias of 'madspin' and is only rewritten in
        do_launch, so both spellings have to be accepted here."""
        return self.options['spinmode'] in ['madspin', 'full', 'PA', 'onshell']

    def _production_polarization(self):
        """``pdg -> tuple`` of the polarisation braces of the *production*
        process, one entry per occurrence of that pdg among the final-state
        legs, in process-line order: ``p p > t{0} t~`` gives
        ``{6: ((0,),)}`` and ``p p > w+{0} w+{T}`` gives
        ``{24: ((0,), (-1, 1))}``. An entry is ``None`` for an occurrence
        that carries no brace.

        A pdg whose occurrences all carry the *same* brace is collapsed to a
        single entry, which then applies to however many of that pdg the event
        holds ('broadcast'). That is what keeps a stack of subprocesses with
        different multiplicities -- ``generate p p > t{0} t~`` plus
        ``add process p p > t{0} t~ j`` -- working. A pdg with *different*
        braces on different legs cannot be broadcast: it keeps its full
        sequence and is matched positionally, the n-th such pdg of the event
        taking the n-th brace (see ``_apply_production_polarization`` for why
        that correspondence holds).

        MadSpin regenerates the production matrix element from the banner's
        proc_card, braces included, so the braces are exactly what MG5 saw. They
        do NOT however restrict the density matrix: ``GET_DENSITY`` overrides
        the decaying particle's helicity from ``ALLOW_HEL`` for every entry it
        builds, so rho_prod comes back fully unpolarised in those indices and
        the restriction has to be applied here.

        Parsing is delegated to MG5's own process parser so that the brace
        semantics ({L} -> [-1], {R} -> [1], {T} -> [1,-1], {0} -> [0]) cannot
        drift away from ``madgraph_interface``'s.
        """
        cached = getattr(self, '_production_polarization_cache', None)
        if cached is not None:
            return cached

        out = {}
        try:
            proc_card = list(self.banner.proc_card)
        except Exception:
            proc_card = []
        lines = [line[9:].strip() for line in proc_card
                 if line.startswith('generate')]
        lines += [' '.join(line.split()[2:]) for line in proc_card
                  if re.search(r'^\s*add\s+process', line)]

        if any('{' in line for line in lines):
            # pdg -> (canonical sequence, the line it came from), so that a
            # disagreement between two process lines can name both of them.
            source = {}
            multi_id = set()
            for line in lines:
                try:
                    procdef = self.mg5cmd.extract_process(line)
                except Exception as error:
                    logger.warning('MadSpin could not re-read the polarisation of '
                                   'the production process "%s" (%s); the density '
                                   'matrix convolution is left unrestricted.'
                                   % (line, error))
                    continue
                seq = collections.OrderedDict()
                for leg in procdef.get('legs'):
                    # initial-state polarisation is the beampol machinery, not this
                    if not leg.get('state'):
                        continue
                    pol = leg.get('polarization')
                    pol = tuple(sorted(set(int(p) for p in pol))) if pol else None
                    ids = [int(i) for i in leg.get('ids')]
                    for pdg in ids:
                        seq.setdefault(pdg, []).append(pol)
                        if len(ids) > 1:
                            multi_id.add(pdg)
                for pdg, pols in seq.items():
                    # all occurrences agree -> broadcast, the multiplicity of
                    # the line then does not have to match the event's
                    canonical = tuple(pols[:1]) if len(set(pols)) == 1 else tuple(pols)
                    if pdg in source and source[pdg][0] != canonical:
                        raise self.InvalidCmd(
                            'MadSpin: particle %s carries the polarisation(s) %s in '
                            'the production process "%s" and %s in "%s". The density '
                            'spin modes have no way to tell, event by event, which of '
                            'the two a given final-state particle follows. Please use '
                            'one consistent polarisation pattern for the particles '
                            'MadSpin decays.'
                            % (pdg, self._format_polarization_sequence(source[pdg][0]),
                               source[pdg][1],
                               self._format_polarization_sequence(canonical), line))
                    source[pdg] = (canonical, line)
            for pdg, (canonical, line) in source.items():
                if all(p is None for p in canonical):
                    continue
                if len(canonical) > 1 and pdg in multi_id:
                    # 'p p > V{0} V{T}' with a multiparticle V: how many of a
                    # given pdg an event holds is not fixed by the process line,
                    # so the n-th brace cannot be pinned to the n-th particle.
                    raise self.InvalidCmd(
                        'MadSpin: particle %s appears with different polarisations '
                        '(%s) inside a multiparticle label in the production process '
                        '"%s". The number of %s in an event is then not fixed by the '
                        'process line, so MadSpin cannot tell which particle carries '
                        'which polarisation. Please spell the polarised legs out with '
                        'explicit particle names.'
                        % (pdg, self._format_polarization_sequence(canonical), line, pdg))
                out[pdg] = canonical

        self._production_polarization_cache = out
        return out

    @staticmethod
    def _format_polarization_sequence(sequence):
        """A polarisation sequence as it reads in a process line, for errors."""
        names = {(0,): '{0}', (1,): '{+}', (-1,): '{-}', (-1, 1): '{T}'}
        return ' '.join('(none)' if p is None else names.get(p, str(list(p)))
                        for p in sequence)

    def _apply_production_polarization(self, decaying_pdg, helicities):
        """Turn the production polarisation into (helicity bases, restriction).

        Returns the per-particle helicity lists to build the density basis with
        and the per-particle restriction handed to ``DensityMatrix``.

        Two things happen here:

        * the restriction itself -- the (i,j) entries the polarisation forbids
          are dropped from the production/decay convolution and from the trace
          that normalises it (see ``DensityMatrix.set_hel_restriction``);

        * a reordering of the helicity basis. ``GET_DENSITY`` picks the rows of
          the process' NHEL table by matching them against the *first*
          combination of ``ALLOW_HEL``; a polarised process has no NHEL row
          outside its polarisation, so leaving the default order ([1,-1] for a
          fermion, [-1,0,1] for a vector) would match nothing and hand back an
          identically zero density matrix for ``{L}``/``{-}``/``{0}``. Putting
          an allowed helicity first is what makes the spectator helicity sum
          find its rows. The order is untouched without braces, so nothing
          moves for unpolarised runs.

        Same pdg, different braces ('p p > w+{0} w+{T}')
        ------------------------------------------------
        ``decaying_pdg`` is in slot order -- for pdg in decays_key, in
        production-event order within a pdg -- so the slots of one pdg form a
        contiguous block whose k-th entry is the k-th such particle of the
        event. The k-th brace of that pdg is handed to that k-th slot, and the
        correspondence is exact rather than a guess:

        * MG5 keeps the legs of a process in the order they were typed (leg
          number 1..n), and a leg's polarisation is part of its identity -- two
          same-pdg legs with different braces have an ``identical_particle_factor``
          of 1, so no symmetrisation and no momentum permutation is applied to
          them anywhere between the amplitude and the event file;

        * ``lhe_parser.Event.get_momenta`` maps the event's k-th particle of a
          pdg onto the k-th slot of that pdg in the matrix element's leg order.
          The momentum the matrix element sees at leg number ``position[k]`` is
          therefore the event particle slot k stands for. The brace read off
          leg ``position[k]`` of the process line and the density matrix
          computed at ``position[k]`` describe the same object by construction.

        A wrong assignment could not go unnoticed either: ``GET_DENSITY``
        selects the NHEL rows of the *polarised* process by matching them
        against the first ``ALLOW_HEL`` combination, which is built from the
        head of each basis below. Handing '{T}' to the leg MG5 generated as
        '{0}' asks for a helicity combination the polarised NHEL table does not
        contain, and the production density matrix comes back identically zero
        -- a loud failure, not a small bias.
        """
        pol_map = self._production_polarization()
        if not pol_map:
            return helicities, None

        helicities = list(helicities)
        multiplicity = collections.Counter(decaying_pdg)
        seen = collections.Counter()
        restriction = []
        for k, pdg in enumerate(decaying_pdg):
            sequence = pol_map.get(pdg)
            occurrence = seen[pdg]
            seen[pdg] += 1
            if not sequence:
                allowed = None
            elif len(sequence) == 1:
                # one brace for every particle of that pdg
                allowed = sequence[0]
            elif len(sequence) != multiplicity[pdg]:
                raise self.InvalidCmd(
                    'MadSpin: the production process gives %d polarisation(s) (%s) '
                    'for particle %s but the event holds %d of them. The braces can '
                    'only be attached to the particles one by one when the two agree.'
                    % (len(sequence), self._format_polarization_sequence(sequence),
                       pdg, multiplicity[pdg]))
            else:
                allowed = sequence[occurrence]
            basis = list(helicities[k])
            if not allowed:
                restriction.append(None)
                continue
            unknown = [h for h in allowed if h not in basis]
            if unknown:
                raise self.InvalidCmd(
                    'MadSpin: the polarisation %s requested for particle %s is not '
                    'expressible in the helicity basis %s used by the density spin '
                    'modes. Only {0}, {+}/{R}, {-}/{L} and {T} are supported.'
                    % (list(allowed), pdg, basis))
            kept = [h for h in basis if h in allowed]
            restriction.append(tuple(kept))
            helicities[k] = kept + [h for h in basis if h not in allowed]

        return helicities, madspin.DensityMatrix.normalize_hel_restriction(restriction)

    # ------------------------------------------------------------------
    # Pure-interference mode ('set pure_interference t = 0 T')
    # ------------------------------------------------------------------

    # The brace vocabulary, spelled out because this option never goes through
    # a process line and so never reaches MG5's parser. Kept identical to the
    # semantics _production_polarization inherits from it ({L} -> -1, {R} -> +1,
    # {T} -> the transverse pair, {0} -> the longitudinal state).
    _POL_TOKENS = {'0': (0,), '+': (1,), 'R': (1,), '-': (-1,), 'L': (-1,),
                   'T': (-1, 1)}

    def _parse_pol_side(self, text, entry):
        """One side of a ``pure_interference`` entry -> a tuple of helicities."""
        out = set()
        for token in text.replace(',', ' ').split():
            key = token.strip().upper()
            if key not in self._POL_TOKENS:
                raise self.InvalidCmd(
                    "MadSpin: '%s' is not a polarisation in the "
                    "pure_interference entry '%s'. Use one or more of "
                    "0, +/R, -/L, T." % (token, entry))
            out.update(self._POL_TOKENS[key])
        return tuple(sorted(out))

    def _pure_interference(self):
        """``pdg -> (production_side, decay_side)`` from the card option, or an
        empty dict when the mode is off.

        Both sets have to come from the MadSpin card rather than from the
        banner's braces: the mode only means something on a sample that
        contains *both* polarisations, i.e. an unpolarised production, which by
        definition carries no brace to inherit. See
        doc/madspin_sequential_plan.md section 13.5.

        Two disjoint sides name the interference block ``I`` of that particle;
        two *identical* sides name its diagonal block ``D_S`` (the normalised
        form collapses ``(S, S)`` back to the symmetric restriction ``S``), so
        a mixed block such as ``(I, D-)`` of ``t t~`` is written

            set pure_interference t  = + -
            set pure_interference t~ = - -

        Repeated ``set`` lines accumulate (see ``ACCUMULATING_OPTIONS``); a
        particle the card does **not** name is left unrestricted, i.e. summed
        over its whole helicity basis, which is neither ``I`` nor ``D+`` nor
        ``D-`` but their sum.
        """
        cached = getattr(self, '_pure_interference_cache', None)
        if cached is not None:
            return cached

        try:
            raw = (self.options['pure_interference'] or '').strip()
        except (KeyError, TypeError):
            # option sets built by hand (unit-test stubs, older cards) simply
            # do not have the mode
            raw = ''
        out = {}
        if not raw:
            self._pure_interference_cache = out
            return out

        try:
            name2pdg = self.model.get('name2pdg')
        except Exception:
            name2pdg = {}

        for entry in raw.split(';'):
            entry = entry.strip()
            if not entry:
                continue
            sep = '=' if '=' in entry else (':' if ':' in entry else None)
            if sep is None:
                raise self.InvalidCmd(
                    "MadSpin: could not read the pure_interference entry '%s'. "
                    "The syntax is 'set pure_interference t = 0 T' -- particle, "
                    "'=', the production-side polarisation, then the "
                    "decay-side one." % entry)
            name, _, sides = entry.partition(sep)
            name = name.strip()
            parts = sides.split()
            if len(parts) != 2:
                raise self.InvalidCmd(
                    "MadSpin: the pure_interference entry '%s' must give "
                    "exactly two polarisation sets (production then decay), "
                    "got %d." % (entry, len(parts)))
            if name in name2pdg:
                pdg = int(name2pdg[name])
            else:
                try:
                    pdg = int(name)
                except ValueError:
                    raise self.InvalidCmd(
                        "MadSpin: '%s' in the pure_interference entry '%s' is "
                        "neither a particle of the model nor a pdg code."
                        % (name, entry))
            prod = self._parse_pol_side(parts[0], entry)
            dec = self._parse_pol_side(parts[1], entry)
            if not prod or not dec:
                raise self.InvalidCmd(
                    "MadSpin: both sides of the pure_interference entry '%s' "
                    "must be non-empty." % entry)
            overlap = set(prod).intersection(dec)
            if overlap and set(prod) != set(dec):
                # a *partial* overlap mixes a diagonal piece into an off-diagonal
                # block: neither an interference term nor a polarised one, and
                # the restricted trace no longer vanishes. That is what the
                # disjointness rule was protecting against -- refuse it.
                # Two EQUAL sides are a different thing: they normalise back to
                # the plain symmetric restriction (DensityMatrix.
                # normalize_hel_restriction collapses (S, S) -> S), i.e. the
                # diagonal block D_S, and that is how the card names the
                # diagonal factor of a mixed block such as (I, D-).
                raise self.InvalidCmd(
                    "MadSpin: the two sides of the pure_interference entry "
                    "'%s' overlap in the helicit%s %s without being equal. "
                    "They must be either disjoint -- an interference (I) block "
                    "-- or identical -- a diagonal (D) block. A partial "
                    "overlap is neither: it puts some diagonal entries back "
                    "into an off-diagonal block, so it carries cross-section "
                    "and is no longer a pure interference term."
                    % (entry, 'y' if len(overlap) == 1 else 'ies',
                       ', '.join(str(h) for h in sorted(overlap))))
            if pdg in out and out[pdg] != (prod, dec):
                raise self.InvalidCmd(
                    "MadSpin: particle %s is given two different "
                    "pure_interference specifications." % name)
            out[pdg] = (prod, dec)

        self._pure_interference_cache = out
        return out

    def _decay_output(self):
        """``decay_output`` with ``auto`` resolved: 'weighted' or 'unweighted'.

        ``auto`` (the default) is 'weighted' under ``pure_interference`` and
        'unweighted' otherwise -- each mode's own historical default, so a card
        that does not mention the option behaves exactly as it always has. The
        two are opposite for a reason rather than by accident: the ordinary run
        writes one event per production event either way and the accept/reject
        is the exact sampler, so unweighting is the safe default there; the
        interference mode has no exact sampler to fall back on (its weights are
        signed and its cross-section is zero), and unweighting on ``|W|`` there
        throws away all but a few percent of the production events for ~6x the
        variance on the observables that mode exists to measure -- section
        13.17. Announced by ``_announce_decay_output``.
        """
        try:
            asked = self.options['decay_output']
        except (KeyError, TypeError):
            # option sets built by hand (unit-test stubs): the same fallback
            # _pure_interference makes, and the same reason
            return 'unweighted'
        if asked != 'auto':
            return asked
        return 'weighted' if self._pure_interference() else 'unweighted'

    def _announce_decay_output(self):
        """Say once what ``decay_output`` resolved to, and why. Same convention
        as ``_announce_mode``: ``auto`` decides on the run, so the card no
        longer answers the question on its own."""
        try:
            asked = self.options['decay_output']
        except (KeyError, TypeError):
            return
        if asked == 'auto':
            why = ('auto, pure_interference is set' if self._pure_interference()
                   else 'auto, ordinary run')
        else:
            why = 'set explicitly'
        self._log_once('decay_output', "MadSpin: decay_output = %s (%s)",
                       self._decay_output(), why)

    def _weighted_decay(self):
        """True when the ordinary (non-interference) decay output is to be
        written WEIGHTED -- no accept/reject, one draw per production event,
        ``w = w_prod * BR * W / c``.

        False in the pure-interference mode: that mode reaches the same
        'keep every trial' path by its own route (``pure_interference`` in the
        worker context), with a signed ``W`` and a zeroed ``<init>``, so it is
        ``_pure_interference_unweighted`` that reads ``decay_output`` there.
        Also false outside the density spin modes, where there is no ``W`` --
        ``_validate_weighted_decay`` refuses that combination at launch rather
        than silently ignoring it, so this is belt and braces.
        """
        return (self._decay_output() == 'weighted'
                and not self._pure_interference()
                and self._density_spinmode())

    def _validate_weighted_decay(self):
        """Card-level checks for ``decay_output``, run once at launch. Refuses
        rather than ignores: an option that silently does nothing is how a user
        ends up quoting statistics they never got."""
        self._announce_decay_output()
        if self._decay_output() != 'weighted':
            return
        if self._pure_interference():
            # the mode announces its own output shape, spinmode requirement
            # included, in _validate_pure_interference -- which runs first
            return
        if not self._density_spinmode():
            raise self.InvalidCmd(
                "MadSpin: decay_output = weighted needs one of the density "
                "spin modes (madspin/full, PA, onshell). spinmode = %s builds "
                "no production/decay spin-density convolution, so there is no "
                "W to put on the weight and nothing to gain by not "
                "unweighting. Drop decay_output, or switch spinmode."
                % self.options['spinmode'])
        logger.warning(
            "MadSpin: decay_output = weighted. No accept/reject is done -- one "
            "decay configuration is drawn per production event and kept, with "
            "w = w_prod * BR * W / c. The output LHE is therefore WEIGHTED: "
            "mean(w) = sigma*BR (MG5 writes IDWTUP = -4, so that is the "
            "cross-section and <init> is unchanged), but the per-event weights "
            "are NOT constant. Anything downstream that assumes MadSpin events "
            "carry a constant weight will be wrong on this file.")

    def _pure_interference_unweighted(self):
        """True when the pure-interference mode must write the 'unweighted'
        (up to a sign) output instead of its fully weighted default.

        Only meaningful when the mode is on: outside it ``decay_output =
        unweighted`` is the ordinary accept/reject, not this.
        """
        return (bool(self._pure_interference())
                and self._decay_output() == 'unweighted')

    def _validate_pure_interference(self):
        """Card-level checks for the pure-interference mode, run once at launch
        rather than on the first event inside a worker process."""
        pure = self._pure_interference()
        if not pure:
            return
        if not self._density_spinmode():
            raise self.InvalidCmd(
                "MadSpin: pure_interference needs one of the density spin "
                "modes (madspin/full/PA/onshell); spinmode=%s builds no "
                "spin-density matrix to restrict."
                % self.options['spinmode'])

        # keep_weight_for_polarization_*: refused, not repaired.
        #
        # _polarization_ratios writes ``nominal * restricted/full`` into the
        # <rwgt> block. In this mode ``full`` is the *interference* contraction:
        # a signed quantity that passes through zero, so the ratios can be
        # arbitrarily large and can flip sign, while the numerators are ordinary
        # symmetric diagonal blocks. Nothing about the product means "the
        # polarised part of this event".
        #
        # Swapping in the unrestricted contraction as the denominator would make
        # the *ratio* well defined but not the product: ``evt.wgt`` is the signed
        # interference weight, and multiplying it by the polarised fraction of a
        # different quantity still is not the polarised part of anything. The
        # diagonal blocks these weights select are precisely the terms this mode
        # removes. There is no combination that means something, so the only
        # option that cannot silently produce garbage is to refuse it.
        if self._polarization_weights_enabled():
            raise self.InvalidCmd(
                "MadSpin: keep_weight_for_polarization_vector/_fermion cannot "
                "be combined with pure_interference. Those weights are "
                "'nominal x (polarised block / full contraction)', and in this "
                "mode the nominal weight is a signed interference weight while "
                "the polarised blocks are exactly the diagonal terms the mode "
                "removes -- the product is not the polarised part of anything, "
                "and with the interference contraction as the denominator the "
                "ratio also passes through zero. Run the polarised blocks as "
                "their own samples instead (a production brace, or a diagonal "
                "pure_interference entry), and drop "
                "keep_weight_for_polarization_* from this card.")

        # At least one particle must carry a genuine *interference* (disjoint)
        # pair. A card that names only diagonal blocks selects an ordinary
        # polarised sub-sample: it has a cross-section, its restricted trace
        # does not vanish, and every piece of this mode -- the zeroed <init>,
        # the signed weights, the z test, the separate trace restriction -- is
        # then wrong. Refuse rather than produce it under this name.
        if not any(set(prod).isdisjoint(dec) for prod, dec in pure.values()):
            raise self.InvalidCmd(
                "MadSpin: every pure_interference entry names a DIAGONAL block "
                "(the two sides are identical), so nothing interferes. At "
                "least one particle must be given two disjoint sides, e.g. "
                "'set pure_interference t = + -'. Diagonal entries are for the "
                "other legs of a mixed block, such as (I, D-):\n"
                "    set pure_interference t  = + -\n"
                "    set pure_interference t~ = - -")

        # A particle the card names but that MadSpin never decays would leave
        # the mode silently inert while the signed weights and the zeroed
        # cross-section are still in force -- much worse than an error.
        decayed = set()
        for name in self.list_branches:
            for spelling in (name, name.lower()):
                try:
                    decayed.add(int(self.model.get('name2pdg')[spelling]))
                except (KeyError, TypeError, ValueError):
                    continue
                break
        if decayed:
            orphan = sorted(set(self._pure_interference()).difference(decayed))
            if orphan:
                raise self.InvalidCmd(
                    "MadSpin: pure_interference names particle(s) %s, but no "
                    "'decay' line makes MadSpin decay them. The mode restricts "
                    "the production/decay density convolution, so it only "
                    "means something for a particle that is actually decayed."
                    % ', '.join(str(p) for p in orphan))

        # The production sample must contain both polarisations: an interference
        # between P and D amplitudes simply does not exist in a sample drawn
        # from |M_P|^2 (section 13.5).
        pol_map = self._production_polarization()
        for pdg, (prod, dec) in pure.items():
            brace = pol_map.get(pdg)
            if brace is None:
                continue
            missing = sorted(set(prod).union(dec).difference(brace))
            if missing:
                raise self.InvalidCmd(
                    "MadSpin: pure_interference asks for the interference "
                    "between helicities %s and %s of particle %s, but the "
                    "production process was generated with a polarisation "
                    "brace keeping only %s. The events carry no amplitude for "
                    "helicit%s %s, so that interference is not present in the "
                    "sample. Regenerate the production process without the "
                    "brace on that leg."
                    % (list(prod), list(dec), pdg, list(brace),
                       'y' if len(missing) == 1 else 'ies',
                       ', '.join(str(h) for h in missing)))

        if self._pure_interference_unweighted():
            shape = ("UNWEIGHTED UP TO A SIGN: one decay draw per production "
                     "event, unweighted on |W| against the probed maximum and "
                     "dropped on rejection, so the file holds fewer events "
                     "than it read and each carries "
                     "w = +- sigma_ref * BR * <|W|> / c -- exactly two weight "
                     "magnitudes. The accept/reject bound cancels out of that "
                     "normalisation (section 13.17)")
        else:
            shape = ("FULLY WEIGHTED: every trial is kept and carries "
                     "w = sigma_ref * BR * W / c")
        logger.warning(
            "MadSpin: pure_interference is ON for particle(s) %s. The decayed "
            "sample keeps ONLY the interference between the polarisations "
            "named, so its total cross-section is zero by construction and its "
            "events carry a SIGNED weight. Output shape "
            "(decay_output = %s) -- %s. Under MG5's IDWTUP = -4 "
            "convention (cross-section = mean of the weights) the file is "
            "self-normalising either way: mean(w) = 0 and sum_bin(w)/N_file is "
            "the interference contribution to that bin in pb, with N_file the "
            "number of events IN THE FILE. The <init> block is "
            "written with XSECUP = 0, so the file is NOT directly showerable "
            "and any tool that assumes unit weights will be wrong on it -- see "
            "the <MGPureInterference> banner block for the reference "
            "cross-section, c, <|W|>, and the zero-cross-section check.",
            ', '.join(str(p) for p in sorted(pure)),
            self._decay_output(), shape)

    def _apply_pure_interference(self, decaying_pdg, helicities, restriction):
        """Overlay the pure-interference cross restriction on the (symmetric)
        production-polarisation one.

        Returns ``(restriction, trace_restriction)``: the first is what the
        production/decay convolution contracts over -- a ``(P, D)`` pair for
        every particle the card names -- and the second is what normalises it.
        They part company exactly here and nowhere else: the interference block
        has no diagonal entry, so its trace is identically zero and cannot be
        the denominator (section 13.4).
        """
        pure = self._pure_interference()
        if not pure:
            return restriction, None

        symmetric = list(restriction) if restriction else [None] * len(decaying_pdg)
        cross = list(symmetric)
        for k, pdg in enumerate(decaying_pdg):
            spec = pure.get(pdg)
            if spec is None:
                continue
            basis = list(helicities[k])
            prod, dec = spec
            unknown = [h for h in list(prod) + list(dec) if h not in basis]
            if unknown:
                raise self.InvalidCmd(
                    "MadSpin: the pure_interference polarisations %s / %s "
                    "requested for particle %s are not expressible in the "
                    "helicity basis %s the density spin modes use for it."
                    % (list(prod), list(dec), pdg, basis))
            cross[k] = (tuple(prod), tuple(dec))

        return (madspin.DensityMatrix.normalize_hel_restriction(cross),
                madspin.DensityMatrix.normalize_hel_restriction(symmetric))

    def _pure_interference_pdgs(self, decays_key):
        """The card-named pdgs that actually decay in this event topology.
        Empty when the mode is off or names nothing that decays here."""
        pure = self._pure_interference()
        if not pure:
            return []
        return [pdg for pdg in decays_key if pdg in pure]

    @staticmethod
    def _pi_unrestricted_contraction(density_prod, density_dec):
        """The convolution an *ordinary* run would have computed on the same
        pair of matrices: the cross restriction swapped for the symmetric one
        that already normalises it (``hel_restriction_trace``, i.e. ``None``
        for the unpolarised production the mode requires, and the production
        brace on any other leg).

        Same trick, and the same reason, as ``_polarization_ratios``: the
        restriction rides on ``density_prod`` for the duration of one
        contraction because ``scalar_multiplication`` intersects the two
        operands' restrictions rather than replacing them.
        """
        saved = density_prod.hel_restriction
        try:
            density_prod.hel_restriction = density_prod.hel_restriction_trace
            return density_dec.scalar_multiplication(density_prod)
        finally:
            density_prod.hel_restriction = saved

    # ------------------------------------------------------------------
    # keep_weight_for_polarization_vector / _fermion:
    # extra LHEF v3 weights, one per polarisation COMBINATION
    # ------------------------------------------------------------------
    # The card offers a list of polarisations per *species*
    #
    #     set keep_weight_for_polarization_vector  [0, T, +, -]
    #     set keep_weight_for_polarization_fermion [+, -]
    #
    # and every decaying particle draws from the list of its own spin. A
    # combination C is one element of the cartesian product over the density
    # basis slots -- 'p p > t t~ z' with the lists above has 2*2*4 = 16 of them
    # -- and the event carries one extra weight per combination,
    #
    #     w_C = w_nominal * <rho_dec, rho_prod>_C / <rho_dec, rho_prod>
    #
    # i.e. the very same event reweighted to the C fraction of the density
    # convolution. Both contractions are done on the matrices that were built
    # for the nominal weight anyway -- only the row mask changes -- so N extra
    # weights cost N extra masked dot products, not N extra density matrices.
    #
    # Why a product and not one weight per label (which is what the first
    # version of this option did): a single label applied to every particle at
    # once cannot express 't left-handed *and* Z longitudinal', and it
    # degenerates to nothing at all on a mixed production such as
    # 'p p > z{0} z{T}', where no single label is compatible with both slots and
    # every weight came back exactly 0.
    #
    # Slots and ids
    # -------------
    # The slot order is the density basis one -- for pdg in decays_key, and
    # within a pdg in production order (see ``_density_basis``' ``init_part``).
    # A combination is named after its per-slot assignment, in that order:
    #
    #     ms_pol_6:+_-6:-_23:0     t(+) t~(-) z(0)
    #     ms_pol_23:0_23:T         the first Z longitudinal, the second transverse
    #
    # Every slot is always present, so a reader never has to guess which particle
    # a label belongs to, even when two slots share a pdg. A slot with nothing to
    # choose from shows up as '*', meaning "summed over its helicities":
    #
    #  * a scalar has a 1x1 density matrix and no polarisation, so it contributes
    #    exactly ONE entry ('*') to the product rather than multiplying the count;
    #  * so does a particle whose species list is empty (only
    #    ..._vector set -> the fermions stay summed over), and
    #  * so does a slot every label of whose list is unphysical for it
    #    ('0' alone on a fermion).
    #
    # A label that is unphysical for a slot is dropped from that slot's choices
    # rather than silently left unrestricted, so a card that gives both species
    # the same list -- 'keep_weight_for_polarization_vector = [0, T, +, -]' and
    # the same for _fermion -- does not emit a '0' and a 'T' copy of the same
    # fermion weight.
    #
    # Production braces (PR #349, #353)
    # ---------------------------------
    # Each slot's choices are intersected with the production restriction of that
    # slot, and a choice whose intersection is empty is dropped -- it is zero for
    # every event of that topology, so it would only add a column of zeros. If
    # that empties a slot, the slot falls back to its production restriction and
    # a '*'. The denominator is always the nominal -- already restricted --
    # convolution, so w_C/w stays the fraction of what is actually written out.
    #
    # Sum rule
    # --------
    # The ratio is >= 0 (the numerator of a single-state restriction is a product
    # of density-matrix diagonals) but is NOT bounded by 1 event by event: the
    # denominator is the full double sum, and the interference terms a
    # restriction drops can be negative.
    #
    # sum_C w_C = w requires that the combinations partition the (i,j) terms that
    # contribute, i.e. two conditions:
    #   (a) every species list partitions its slots' helicity basis -- [+, -] for
    #       a fermion, [0, +, -] or [0, T] for a vector. [0, T, +, -] does NOT:
    #       T = {-1,+1} covers the same entries as + and - together, so the
    #       weights overlap and the sum overshoots;
    #   (b) the contraction has no off-diagonal (interference) part -- the i != j
    #       terms of the double sum belong to no single-state block. {T} is the
    #       exception that keeps its own (-1,+1) block, which is why [0, T] is a
    #       partition of a vector even with interference in that block.
    # The product form removed the *third* condition the one-label-per-weight
    # version had ("only one particle may be restricted"): the mixed (+,-) and
    # (-,+) assignments are now combinations of their own. See the sum-rule tests.

    #: species name per MG5 spin (2S+1). Only 1/2/3 have a helicity basis in
    #: ``_density_basis``' ``hel_dict``, so nothing else can reach a slot.
    POLARIZATION_SPECIES = {1: 'scalar', 2: 'fermion', 3: 'vector'}

    #: emitting more than this many combinations per event is legal but worth a
    #: warning: it is that many masked contractions and that many <wgt> lines per
    #: event, and the product grows very fast (four decaying vectors with a
    #: 4-entry list is 256).
    POLARIZATION_COMBINATION_WARN = 32

    def _polarization_weight_labels(self, species):
        """Canonical polarisation labels requested for one species ('vector' /
        'fermion'), in the order the user typed them. Empty (the default) leaves
        that species summed over."""
        cache = getattr(self, '_pol_weight_labels_cache', None)
        if cache is None:
            cache = self._pol_weight_labels_cache = {}
        if species in cache:
            return cache[species]
        option = 'keep_weight_for_polarization_%s' % species
        out = []
        for entry in self.options.get(option) or []:
            parsed = parse_polarization_label(entry)
            if parsed is None:
                raise self.InvalidCmd(
                    "%s: '%s' is not a polarisation. "
                    "Use 0, +, - or T (L and R are accepted as aliases)."
                    % (option, entry))
            if parsed[0] not in [l for l, _ in out]:
                out.append(parsed)
        cache[species] = out
        return out

    def _polarization_weights_enabled(self):
        """True as soon as one species list is non-empty. Both empty (the
        default) is a complete no-op: no mask, no weight, no banner block."""
        return bool(self._polarization_weight_labels('vector')
                    or self._polarization_weight_labels('fermion'))

    # -- which axis the projection is taken on ---------------------------

    def _needs_frame_axis(self):
        """Whether the density matrices have to be built in the ``frame_id``
        frame (run_card ``me_frame``, the partonic CM by default) rather than in
        the lab.

        A polarised matrix element is not Lorentz invariant: the frame decides
        which helicity ``{0}`` names. That does not matter for the *nominal*
        weight, because the full contraction sum_ij rho_prod(i,j) rho_dec(i,j)
        is a trace and a boost is a unitary basis change that cancels between
        the two matrices. It matters as soon as a helicity index is
        **projected**, which is what ``set_hel_restriction`` does -- projections
        do not commute with a change of basis -- so the projection only means
        what the user asked for on MG5's own quantisation axis.

        Four things apply such a projection, and all three need the frame:

         * polarised beams (``beampol``), which reweights the initial-state
           helicity sum;
         * a polarisation brace on the production process (PR #349/#353);
         * a polarisation-weight request. The weights are the same projection,
           only used to build an extra weight rather than the nominal one, so an
           unpolarised production with
           ``keep_weight_for_polarization_vector/_fermion`` set still needs it;
         * the pure-interference mode (``set pure_interference t = 0 T``), whose
           cross restriction is a projection for the same reason -- and whose
           production is unpolarised, so no brace switches it on.

        This is ``_frame_boost``'s guard: it short-circuits to None -- leaving
        every momentum in the lab -- exactly when this returns False.
        """
        if self._beampol() is not None:
            return True
        if self._production_polarization():
            return True
        if self._pure_interference():
            # the cross restriction is a projection like the others, so the
            # interference block is named on the me_frame axis too
            return True
        return self._polarization_weights_enabled()

    @staticmethod
    def _polarization_weight_id(assignment):
        """LHEF weight id of one combination.

        ``assignment`` is ``[(pdg, label or None), ...]`` in density-basis slot
        order; ``None`` (written '*') is a slot that stays summed over. Kept
        human readable and stable -- it is what an analysis has to ask the event
        file for -- and slot-complete, so 'ms_pol_23:0_23:T' names the two Zs of
        'p p > z{0} z{T}' unambiguously.
        """
        return 'ms_pol_%s' % '_'.join('%d:%s' % (pdg, label or '*')
                                      for pdg, label in assignment)

    def _polarization_slot_choices(self, prod_static):
        """``[[(label, restriction), ...], ...]``: the choices each density slot
        offers, in slot order. One entry per slot, never empty -- a slot with
        nothing to choose keeps its production restriction under the label
        ``None``.

        ``restriction`` is the helicity tuple for that slot, already intersected
        with the production braces (``None`` = the whole basis).
        """
        helicities = prod_static['helicities']
        base = prod_static.get('hel_restriction') or (None,) * len(helicities)
        spins = prod_static.get('decaying_spins')
        if spins is None:
            # only the length of a basis distinguishes the three spins
            # ``_density_basis``' hel_dict knows about
            spins = [len(h) for h in helicities]

        out = []
        for k, basis in enumerate(helicities):
            species = self.POLARIZATION_SPECIES.get(spins[k])
            labels = self._polarization_weight_labels(species) if species else []
            choices = []
            seen = set()
            for label, values in labels:
                allowed = [h for h in values if h in basis]
                if base[k] is not None:
                    allowed = [h for h in allowed if h in base[k]]
                if not allowed:
                    # unphysical for this spin, or incompatible with the
                    # production brace: zero for every event of this topology,
                    # so not worth a column
                    continue
                allowed = tuple(sorted(set(allowed)))
                if allowed in seen:
                    continue
                seen.add(allowed)
                choices.append((label, allowed))
            if not choices:
                choices = [(None, base[k])]
            out.append(choices)
        return out

    def _polarization_combinations(self, prod_static):
        """``[(weight_id, restriction), ...]``, one per element of the cartesian
        product of ``_polarization_slot_choices`` -- what
        ``DensityMatrix.set_hel_restriction`` wants for each of them.

        Empty when nothing is requested, and also when no slot has a real choice
        (every particle would be summed over, i.e. the only combination is the
        nominal weight again).

        Depends on the basis only, so it is memoised on ``prod_static``, which is
        itself built once per production event.
        """
        cached = prod_static.get('pol_weight_combinations')
        if cached is not None:
            return cached

        out = []
        if self._polarization_weights_enabled():
            choices = self._polarization_slot_choices(prod_static)
            if any(label is not None for slot in choices for label, _ in slot):
                pdgs = prod_static.get('decaying_pdg')
                if pdgs is None:
                    pdgs = [0] * len(choices)
                for combo in itertools.product(*choices):
                    wid = self._polarization_weight_id(
                        [(pdgs[k], label) for k, (label, _) in enumerate(combo)])
                    restriction = madspin.DensityMatrix.normalize_hel_restriction(
                        [allowed for _, allowed in combo])
                    out.append((wid, restriction))

        prod_static['pol_weight_combinations'] = out
        return out

    def _polarization_ratios(self, density_prod, density_dec, prod_static,
                             full=None):
        """``{weight_id: restricted/full}`` for the accepted chain, cached on
        self so it does not have to be threaded through every weight return
        value.

        ``full`` is the nominal contraction when the caller has it already (it
        always does -- that is the event's weight); it is recomputed otherwise.
        The restriction rides on ``density_prod`` for the duration of one
        contraction rather than being passed down, because ``scalar_multiplication``
        refuses to combine two *different* restrictions and the production matrix
        may already carry the production-brace one.
        """
        if not self._polarization_weights_enabled():
            self._pol_weight_ratios = None
            return None
        combinations = self._polarization_combinations(prod_static)
        if not combinations:
            self._pol_weight_ratios = None
            return None

        if full is None:
            full = density_dec.scalar_multiplication(density_prod)
        full = getattr(full, 'real', full)

        out = {}
        saved = density_prod.hel_restriction
        try:
            for wid, restriction in combinations:
                if not full:
                    out[wid] = 0.0
                    continue
                if restriction == saved:
                    out[wid] = 1.0
                    continue
                density_prod.hel_restriction = restriction
                value = density_dec.scalar_multiplication(density_prod)
                out[wid] = float(getattr(value, 'real', value)) / float(full)
        finally:
            density_prod.hel_restriction = saved

        self._pol_weight_ratios = out
        return out

    # -- the banner declaration -----------------------------------------
    # The combinations depend on the *topology* (which particles decay and how
    # many of each the event holds), so the ids cannot be listed from the card
    # alone as they could when there was one weight per label. The set of
    # topologies is collected while run_onshell scans the input file anyway
    # (``_pol_event_layouts``) and turned into density-basis slot layouts here.

    @staticmethod
    def _polarization_slot_layout(sequence, decaying):
        """The density-basis slot layout of one production event.

        ``sequence`` is that event's final-state pdgs in production order;
        ``decaying`` the pdgs that actually have decay events. Reproduces
        ``_decaying_pdgs`` (first appearance) followed by ``_density_basis``'
        ``init_part`` (for pdg in decays_key, in production order).
        """
        key = []
        for pdg in sequence:
            if pdg in decaying and pdg not in key:
                key.append(pdg)
        return tuple(pdg for pdg in key for other in sequence if other == pdg)

    def _polarization_layout_static(self, slot_pdgs):
        """A ``prod_static`` stub -- helicity bases, production restriction and
        pdgs -- for one slot layout, without a production event. Goes through
        exactly the same ``_apply_production_polarization`` the real basis does,
        so the declared ids cannot drift away from the emitted ones."""
        hel_dict = {1: [0], 2: [1, -1], 3: [-1, 0, 1]}
        spins = [self.model.get_particle(int(pdg)).get('spin')
                 for pdg in slot_pdgs]
        helicities = [list(hel_dict[spin]) for spin in spins]
        helicities, restriction = self._apply_production_polarization(
            [int(pdg) for pdg in slot_pdgs], helicities)
        return {'helicities': helicities, 'hel_restriction': restriction,
                'decaying_pdg': [int(pdg) for pdg in slot_pdgs],
                'decaying_spins': spins}

    def _polarization_layout_statics(self, evt_decayfile):
        """One ``_polarization_layout_static`` per topology seen in the input
        file, sorted so the banner is reproducible run to run."""
        layouts = getattr(self, '_pol_event_layouts', None) or set()
        decaying = set(pdg for pdg in evt_decayfile if len(evt_decayfile[pdg]))
        slot_layouts = set()
        for sequence in layouts:
            slots = self._polarization_slot_layout(sequence, decaying)
            if slots:
                slot_layouts.add(slots)
        return [self._polarization_layout_static(slots)
                for slots in sorted(slot_layouts)]

    def _declare_polarization_weights(self, statics=None):
        """Declare one <weight> per polarisation combination in the banner's
        <initrwgt> block, in its own weightgroup, following the convention the
        reweighting and systematics modules use. No-op when nothing is
        requested, so an unset option leaves the banner byte-identical."""
        if not self._polarization_weights_enabled():
            return
        if getattr(self, '_pol_weights_declared', False):
            return
        if statics is None:
            statics = []

        entries = collections.OrderedDict()
        biggest = 0
        for static in statics:
            combinations = self._polarization_combinations(dict(static))
            biggest = max(biggest, len(combinations))
            pdgs = static['decaying_pdg']
            for wid, restriction in combinations:
                if wid in entries:
                    continue
                base = restriction or (None,) * len(pdgs)
                entries[wid] = ' '.join(
                    '%s(%s)' % (self._polarization_particle_name(pdg),
                                'sum' if hel is None
                                else ','.join(str(h) for h in hel))
                    for pdg, hel in zip(pdgs, base))
        if not entries:
            return
        if biggest > self.POLARIZATION_COMBINATION_WARN:
            logger.warning(
                "MadSpin: keep_weight_for_polarization_* asks for %d "
                "polarisation combinations, i.e. %d extra <wgt> entries and %d "
                "extra density contractions on every event. Shorten "
                "keep_weight_for_polarization_vector/_fermion if that is not "
                "what you meant.", biggest, biggest, biggest)

        text = "\n<weightgroup name='madspin_polarization'>\n"
        for wid, description in entries.items():
            text += "<weight id='%s'> MadSpin polarisation %s </weight>\n" % (
                wid, description)
        text += "</weightgroup>\n"
        # dict.get is not available: Banner.get is get_detail, which only knows
        # about a handful of card tags
        if 'initrwgt' in self.banner and self.banner['initrwgt']:
            self.banner['initrwgt'] += text
        else:
            self.banner['initrwgt'] = text
        self._pol_weights_declared = True

    def _polarization_particle_name(self, pdg):
        """Readable name for the banner description; the pdg is what the id
        carries, so a model that cannot be queried is not fatal."""
        try:
            return self.model.get_particle(int(pdg)).get_name()
        except Exception:
            return str(pdg)

    def _add_polarization_weights(self, event, ratios):
        """Write ``nominal * ratio`` into the event's LHEF v3 <rwgt> block.

        Called *after* the nominal weight has been scaled by the branching
        ratio, so that the extra weights are consistently normalised to the
        weight that is actually written out.
        """
        if not ratios:
            return
        # fixed_order hands over [event] + counter-events; an Event is itself a
        # list (of Particles), so it cannot be told apart by isinstance(list)
        events = [event] if isinstance(event, lhe_parser.Event) else event
        for evt in events:
            wgts = evt.parse_reweight()
            for wid, ratio in ratios.items():
                wgts[wid] = evt.wgt * ratio

    @staticmethod
    def _decaying_pdgs(production, evt_decayfile):
        """The pdgs that decay, in order of first appearance among the
        production's final-state particles.

        That is the order ``get_decay_from_file`` fills its dict in, hence the
        order ``_density_basis`` lays the density matrix slots out in. The
        sequential accept/reject needs it *before* drawing anything, to build
        the basis, so it is derived from the pools rather than from a draw. The
        "does this particle decay" test must stay identical to
        ``_draw_one_decay``'s.
        """
        out = []
        for particle in production:
            if int(particle.status) != 1:
                continue
            if particle.pdg not in evt_decayfile:
                continue
            if not len(evt_decayfile[particle.pdg]):
                continue
            if particle.pdg not in out:
                out.append(particle.pdg)
        return tuple(out)

    @staticmethod
    def _sequential_slots(production, decays_key):
        """Map each density matrix slot to the production final-state particle
        it belongs to.

        Returns (particles, slot_to_index): ``particles`` is the final state in
        production order (what ``_draw_one_decay`` indexes into), and
        ``slot_to_index[s]`` is the position in it of slot s's particle. The
        slot order mirrors ``_density_basis``'s ``init_part`` -- for pdg in
        decays_key, in production order -- which is the order the tensor product
        is built in. The decay *ordering* permutes which slot is filled next; it
        must never permute this.
        """
        particles = [p for p in production if int(p.status) == 1]
        slot_to_index = []
        for pdg in decays_key:
            for i, particle in enumerate(particles):
                if particle.pid == pdg:
                    slot_to_index.append(i)
        return particles, slot_to_index

    @staticmethod
    def _production_jacobian_for(production, slot_to_index, slot_masses):
        """J_k: the production reshuffling jacobian with the slots drawn so far
        carrying their sampled virtuality and the rest still at their nominal
        mass. Returns -1 (or 0) when that mass set cannot be reshuffled.

        ``slot_masses`` maps slot -> (new_mass, reshuffle_info).

        The masses are placed by slot identity rather than through
        ``add_decays``, which attaches a pdg's decays to its particles in
        production order: with a pdg owning several slots and only some of them
        drawn, that would hand a mass to the wrong particle. Resonance handling
        is left to reshuffle_production, which is why this is not a function of
        the top-level masses alone.
        """
        probe = lhe_parser.Event(str(production))
        finals = [p for p in probe if int(p.status) == 1]
        for slot, (new_mass, info) in slot_masses.items():
            particle = finals[slot_to_index[slot]]
            particle.new_mass = new_mass
            if info is not None:
                particle.reshuffle_info = info
        return probe.reshuffle_production(_allow_retry=False)

    def _slot_identity(self, hel):
        """The I/n a slot contributes while its decay has not been drawn yet.
        Depends only on the helicity list, so cache it per basis."""
        key = tuple(hel)
        try:
            cache = self._slot_identity_cache
        except AttributeError:
            cache = self._slot_identity_cache = {}
        if key not in cache:
            cache[key] = madspin.DensityMatrix.identity(1, list(hel), len(hel))
        return cache[key]

    def _partial_density_contraction(self, density_prod, helicities, slot_densities):
        """N_k: the production density matrix contracted with the normalised
        decay density matrix (Dhat = D/Tr D) of every slot drawn so far, the
        slots still to be drawn contributing I/n -- the average of a decay
        density matrix over its full phase space.

        ``slot_densities`` maps slot index -> DensityMatrix as get_density
        returns it (un-normalised); slots absent from it are the undrawn ones.

        The tensor product is built in *slot* order, which is what the
        production density matrix's helicity index follows. The accept/reject
        ordering only decides which slot gets filled next -- it must never
        permute the tensor. See doc/madspin_sequential_plan.md.
        """
        return decay_density_tensor(self._slot_identity, helicities,
                                    slot_densities) \
                   .scalar_multiplication(density_prod)

    def _decay_reshuffle_jacobian(self, decay):
        """jac_dec: the jacobian of mapping this decay onto the virtuality just
        sampled for its parent. 0 when that is kinematically impossible --
        ``t > b j j`` with a top below MW+Mb -- which is also what tells the
        caller to draw the mass again.

        Probed on a copy: the real reshuffling of the decay still happens once,
        with the production, exactly as it does today. The value is the same
        either way -- ``mass_shuffle`` boosts to the parent rest frame before it
        builds the jacobian, and ``reshuffle_decay``'s ``new_incoming`` only
        feeds the Lorentz map applied afterwards -- so this probe is what enters
        the accept/reject weight and the final ``reshuffle_production`` merely
        recomputes it."""
        probe = lhe_parser.Event(str(decay))
        probe[0].new_mass = decay[0].new_mass
        probe[0].reshuffle_info = decay[0].reshuffle_info
        try:
            return probe.reshuffle_decayevt()
        except Exception:
            return 0

    def _slot_density(self, decay, parent, hel, frame_boost=None):
        """The decay density matrix of one slot, in the lab frame of its parent
        (and then in the ``frame_id`` frame, when there is one -- see
        ``get_density``)."""
        rest_leg = None
        if frame_boost is not None:
            rest_leg = self._decay_frame_rest_leg(parent, frame_boost)
        boost = -1 * lhe_parser.FourMomentum(parent)
        boost.E *= -1
        decay.boost(boost)
        return self.get_density(decay, position=[1], allow_hel=hel,
                                ncomb=len(hel), dimension=len(hel),
                                frame_boost=frame_boost, frame_rest_leg=rest_leg)

    def _resolved_bw_cut(self):
        """The number of widths the mass window extends over. ``BW_cut < 0``
        is the "not set" marker (do_launch normally resolves it from the
        run_card's ``bwcutoff``); 15 is the value that resolution falls back
        on, so the two agree when there is no run_card to read."""
        if self.options['BW_cut'] < 0:
            return 15
        return self.options['BW_cut']

    def _spinmode_draws_virtuality(self):
        """Whether *this* spinmode samples a resonance virtuality at all, i.e.
        whether ``BW_cut`` truncates anything it produces.

        ``madspin``/``full`` evaluate the density at reshuffled offshell momenta
        and ``PA`` dresses the accepted event with an offshell mass, so all
        three draw. ``onshell``/``onshell_v1`` never touch the production
        momenta (``_density_do_reshuffle`` is False and the draw is skipped in
        ``get_onshell_evt_and_wgt``), and the bridge (``spinmode none``) takes
        the decay event exactly as MG5 generated it -- none of those three has a
        window, so none of them gets a truncation correction.
        """
        return self.options['spinmode'] in ('madspin', 'full', 'PA')

    def _mass_window(self, pdg, budget):
        """The Breit-Wigner sampling window of one resonance and its sampling
        jacobian: ``(pole, width, min_mass, max_mass, jac_bw)``.

        Note ``jac_bw`` is a function of the *window*, not of the mass drawn in
        it: the sampler is uniform in R = atan((m^2-pole^2)/(pole.Gamma)) and
        gap/pi is exactly that window's width in R over pi. That is what makes
        the mass-stage bound of ``_mass_stage_bound`` a maximum over the window
        corners rather than a scan in m.
        """
        pole = self.banner.get('param', 'mass', abs(pdg)).value
        width = self.banner.get('param', 'decay', abs(pdg)).value
        bw_cut = self._resolved_bw_cut()
        min_mass = pole - bw_cut * width
        max_mass = min(pole + bw_cut * width, budget)
        gap = math.atan((pole**2-min_mass**2)/pole/width)
        gap += math.atan((max_mass**2-pole**2)/pole/width)
        return pole, width, min_mass, max_mass, gap/math.pi

    def _draw_mass_value(self, pdg, budget):
        """Sample one resonance virtuality from its Breit-Wigner, capped at the
        remaining ``budget`` (what is left of sqrt(shat)). Returns
        ``(mass, reshuffle_info, jac_bw)`` where jac_bw is the Breit-Wigner
        sampling jacobian (gap/pi)."""
        pole, width, min_mass, max_mass, jac_bw = self._mass_window(pdg, budget)
        mass = lhe_parser.Event.generate_random_mass(pole, width, min_mass, max_mass)
        info = (pole, width, min_mass, max_mass)
        return mass, info, jac_bw

    def _draw_offshell_mass(self, pdg, dec, budget):
        """Sample one resonance virtuality and store it on the decay event that
        carries it: ``dec[0]`` gets ``new_mass`` and ``reshuffle_info``. Returns
        the budget left and that draw's jacobian.

        ``budget`` is what is left of sqrt(shat) once the resonances drawn
        before this one are paid for, so the draw is order dependent and the
        caller owns that order (used by the PA per-slot draw).
        """
        mass, info, jac = self._draw_mass_value(pdg, budget)
        dec[0].new_mass = mass
        dec[0].reshuffle_info = info
        return budget - mass, jac

    def _upfront_production(self, production, order, particles, slot_to_index,
                            prod_static, offshell, draw_mass=True,
                            density_prod=None, frame_boost=None):
        """Set up the production for one chain attempt of an up-front-mass
        scheme: draw a virtuality for every decaying particle *before* the
        per-particle loop, and settle everything that depends on the mass set
        but not on the decay angles.

        Returns ``(rho, jac_prod, slot_mass, parents, frame_boost)`` or None if
        the mass set is one the production cannot be reshuffled onto (the caller
        redraws the whole set):
        - ``slot_mass[slot]`` = (mass, reshuffle_info, jac_bw), empty when
          ``draw_mass`` is False (onshell, and 2 -> 1 production under PA, have
          no virtuality to sample). ``draw_mass`` describes the *PA* draw only:
          offshell always samples, since its rho is defined at the reshuffled
          momenta and there is nothing to evaluate without a mass set;
        - ``jac_prod``        = the production reshuffling jacobian of that mass
          set;
        - ``parents[slot]``   = the production particle to boost that slot's
          decay to;
        - ``frame_boost``     = the momentum whose rest frame ``frame_id``
          picks, for the momenta rho was evaluated at, so that every decay
          contracted against it is taken in the same frame. Offshell that is the
          *reshuffled* production, which only exists here; under PA rho is the
          cached onshell one, so the caller's boost is handed straight back.

        The two spinmode families differ in what the up-front draw is *for*.

        Offshell (madspin/full): rho depends on the whole mass set, so a *copy*
        of the production is reshuffled here (leaving the shared event
        untouched) and the density is evaluated at those momenta. Fixing rho
        before the loop is what makes the per-particle decomposition possible at
        all -- see doc/madspin_sequential_plan.md section 10.

        PA: rho is evaluated at the *onshell* momenta and is already fixed per
        production event (cached on it), so there is nothing to gain there. What
        the up-front draw buys instead is ``jac_prod``: with
        ``density_keep_jacobian`` on, the per-slot scheme calls
        ``_production_jacobian_for`` -- an event copy and a reshuffle -- on every
        slot trial and telescopes the results, whereas here it is one call per
        mass set and the J_k/J_{k-1} ratios disappear. The production event
        itself is never reshuffled here: under PA that is a post-acceptance
        dressing (or the final ``reshuffle_production``), and the decays are
        boosted to the onshell parents.
        """
        budget = production.sqrts
        slot_mass = {}
        if offshell or draw_mass:
            for slot in order:
                pdg = particles[slot_to_index[slot]].pid
                mass, info, jac_bw = self._draw_mass_value(pdg, budget)
                slot_mass[slot] = (mass, info, jac_bw)
                budget -= mass

        if not offshell:
            # PA/onshell: onshell rho, onshell parents. Only the feasibility of
            # the mass set and its reshuffling jacobian are settled here.
            jac_prod = 1.0
            if slot_mass:
                jac_prod = self._production_jacobian_for(
                    production, slot_to_index,
                    {slot: (mass, info)
                     for slot, (mass, info, _) in slot_mass.items()})
                if jac_prod in (0, -1):
                    return None
            # rho is the cached onshell one, so the frame it was evaluated in is
            # the caller's and is handed straight back
            return (density_prod, jac_prod, slot_mass,
                    prod_static['init_part'], frame_boost)

        prod_off = lhe_parser.Event(str(production))
        finals = [p for p in prod_off if int(p.status) == 1]
        for slot, (mass, info, _) in slot_mass.items():
            part = finals[slot_to_index[slot]]
            part.new_mass = mass
            part.reshuffle_info = info
        # _allow_retry=False so the drawn masses stay put (a retry would resample
        # them and diverge from the masses we reshuffle each decay to); an
        # impossible set is reported as -1 and the caller restarts.
        jac_reshuffle = prod_off.reshuffle_production(_allow_retry=False)
        if jac_reshuffle in (0, -1):
            return None
        # the frame is derived from the *reshuffled* production, since that is
        # the event rho_off is evaluated at; the decays are contracted against
        # it, so they have to be boosted with this same momentum
        frame_boost = self._frame_boost(prod_off)
        rho_off = self.get_density(prod_off, prod_static['position'],
                                   prod_static['allowed_hel'],
                                   prod_static['ncomb'], prod_static['dimension'],
                                   frame_boost=frame_boost,
                                   hel_restriction=prod_static.get('hel_restriction'),
                                   hel_restriction_trace=prod_static.get('hel_restriction_trace'))
        # Tr(rho_off) is the numerator of the mass-set weight and the
        # denominator (through n_prev) of every slot weight; zero there is an
        # accept/reject that can never accept, not an unlucky mass set
        self._check_production_density(prod_off, rho_off,
                                       'sequential accept/reject, offshell rho')
        parents = {slot: finals[slot_to_index[slot]] for slot in order}
        return rho_off, jac_reshuffle, slot_mass, parents, frame_boost

    def _sequential_offshell(self):
        """Whether the sequential accept/reject runs its offshell (madspin/full)
        branch: the production density is evaluated at reshuffled momenta, so the
        virtualities are drawn up front and rho is fixed per chain."""
        return not self._density_pole_approximation()

    def _sequential_upfront(self, density_method=True):
        """Whether *this run* draws every virtuality before the angles, i.e.
        ``_is_upfront_scheme`` of the scheme it resolved to.

        What the up-front draw buys differs by spinmode: offshell it fixes rho
        for the chain (which is what makes the per-particle decomposition
        possible at all), while under PA rho is already fixed at the onshell
        momenta and what is frozen instead is the *production reshuffling
        jacobian* -- one reshuffle per mass set rather than one per slot trial.
        Either way the angle stage then redraws to acceptance and divides out
        its own normalisation, which is what the tabulated ``_zhat`` puts back.
        """
        return self._is_upfront_scheme(self._unweighting_mode(density_method))

    # ------------------------------------------------------------------
    # Z_k(m): the rate factor of one slot, in the up-front-mass schemes
    # ------------------------------------------------------------------
    # Those chains are unweighted in two stages -- the mass set first, then each
    # slot's decay angles -- and the per-angle stage redraws until it accepts.
    # That divides its own normalisation
    #
    #     Z_k(m) = Integral p_pool(Omega) w_k(Omega, m) dOmega
    #
    # out of the accepted sample, so without a compensating factor in the
    # mass-set weight the accepted virtualities follow the Breit-Wigner instead
    # of the physical one. Either way Z_k is a smooth function of that slot's
    # virtuality *alone*: the production event, the other slots' masses and the
    # angles already accepted all cancel out of it, which is what makes it
    # tabulable. See doc/madspin_sequential_plan.md sections 10 and 11.
    #
    # What sits inside the average is the spinmode's own per-angle weight:
    #
    #   offshell   Z_k(m) = Integral dPhi_off(m) |M_dec|^2
    #                       / Integral dPhi_on |M_dec|^2
    #                     = (m/M) Gamma_k(m) / Gamma_k(M)
    #              -- the decay reshuffling jacobian *and* the offshell/onshell
    #              rate ratio Tr(D^off)/|M_dec|^2_on, the running width;
    #
    #   PA         Z_k(m) = E_pool[ jac_dec(m, Omega) ]
    #              -- the decay reshuffling jacobian alone. PA evaluates its
    #              matrix elements on shell, so there is no offshell integrand
    #              to reweight; what the angle stage normalises away is purely
    #              the phase-space cost of mapping a pool decay onto the sampled
    #              virtuality. (With density_keep_jacobian off that jacobian is
    #              not in the weight at all, and Z_k degenerates to the fraction
    #              of the pool that can reach m.)
    #
    # Same machinery for both: bin in m, average, fit, multiply into the
    # mass-set weight. Note that ``sequential_with_mass`` needs none of this --
    # it redraws the mass with the angles, so no stage freezes a virtuality and
    # none of them has a conditional normalisation to divide out.
    #
    # A *wrong* Z_hat does not cancel: the per-angle stage divides out the true
    # Z_k whatever weight it is given (rescaling w_k by anything that does not
    # depend on the angles leaves its accepted distribution unchanged), so the
    # residual bias of the tabulated scheme is exactly Z_hat/Z. That is why
    # sequential_global_retry exists -- it stops the per-angle stage normalising at
    # all, and then Z_hat cancels identically and only sets the efficiency.

    @staticmethod
    def _z_slot_keys(particles, slot_to_index):
        """Table key of each slot: its pdg and which occurrence of that pdg it
        is. Slots of one pdg are consecutive and in production order, which is
        also how _draw_one_decay picks a decay file when there is one file per
        identical parent -- so the two agree on which slot draws from what."""
        keys = []
        seen = collections.defaultdict(int)
        for index in slot_to_index:
            pdg = particles[index].pid
            keys.append('%s_%s' % (pdg, seen[pdg]))
            seen[pdg] += 1
        return keys

    def _zhat(self, key, mass):
        """The tabulated Z_k at a sampled virtuality. 1 when no table has been
        built (the max-weight probe itself, which is what measures it, and every
        non-offshell mode)."""
        table = (getattr(self, '_z_tables', None) or {}).get(key)
        if not table:
            return 1.0
        if mass < table['zero_below']:
            return 0.0
        lo, hi = table['range']
        # held constant outside the probed range rather than extrapolated: the
        # fit is only constrained where the Breit-Wigner put samples
        u = math.log(min(max(mass, lo), hi) / table['pole'])
        c = table['coeff']
        return math.exp(c[0] + u * (c[1] + u * c[2]))

    def _zhat_max(self, key):
        """max_m Zhat(key, m), exactly. ``_zhat`` is exp of a quadratic in
        u = ln(m/pole) *clamped* to the probed range, so the reachable set of u
        is the closed interval [ln(lo/pole), ln(hi/pole)] and the maximum of a
        quadratic over a closed interval is at an endpoint or at its vertex.
        No scan, no sample mean, no safety margin: this is the supremum.

        Doing it this way rather than as ``margin * <jac_bw . Zhat>`` is the
        point -- a sample mean is not a bound, and Zhat is allowed to have
        structure (a decay threshold opening inside the window shows up as
        ``zero_below`` plus a steep rise just above it, a narrow daughter
        resonance as curvature). The quadratic fit is what ``_build_z_tables``
        ships, so its exact maximum is what dominates every value ``_zhat`` can
        return.
        """
        table = (getattr(self, '_z_tables', None) or {}).get(key)
        if not table:
            return 1.0
        lo, hi = table['range']
        pole = table['pole']
        c = table['coeff']
        u_lo, u_hi = math.log(lo / pole), math.log(hi / pole)
        candidates = [u_lo, u_hi]
        # c[2] >= 0 opens upwards, so the interior stationary point is a
        # minimum and the endpoints already win; only a downward parabola can
        # peak inside
        if c[2] < 0:
            vertex = -c[1] / (2 * c[2])
            if u_lo < vertex < u_hi:
                candidates.append(vertex)
        return max(math.exp(c[0] + u * (c[1] + u * c[2])) for u in candidates)

    # ------------------------------------------------------------------
    # The per-event bound of the mass stage
    # ------------------------------------------------------------------
    # The mass-stage weight under PA/onshell is
    #
    #     w_mass = J(m) . prod_s jac_bw_s(m) . prod_s Zhat_s(m_s)
    #
    # with every factor non-negative, so any product of per-factor maxima
    # dominates it. All three maxima are exact and cheap per production event:
    #
    #  * J is the RAMBO reshuffling jacobian, and it is monotone DECREASING in
    #    every m_s at fixed sqrt(shat) and fixed production configuration --
    #    proved below -- so max J over the window is J at the low corner, for
    #    any n, with no scan;
    #  * jac_bw_s is the width in R of slot s's Breit-Wigner window, a function
    #    of the *budget* sqrt(shat) - sum of the masses drawn before it, not of
    #    m_s. It is increasing in that budget, which is largest when every
    #    earlier slot sits at its own window minimum -- the same low corner;
    #  * Zhat_s is a 1-D function of m_s alone, maximised by ``_zhat_max``.
    #
    # The low corner therefore maximises J and every jac_bw simultaneously, and
    # Zhat factorises, so the product of the three maxima is a true bound. The
    # window is not a box -- sum(m) <= sqrt(shat) couples the slots -- but the
    # coupled region is a SUBSET of the box that contains the low corner
    # whenever it contains anything at all, so a monotone function's maximum
    # over it is still at that corner and a scan cannot do better.
    #
    # Proof that J is monotone decreasing. Write a_i = |p_i|^2 in the
    # reshuffling CM frame, mu_i = m_i'^2, E_i' = sqrt(mu_i + chi^2 a_i),
    # beta_i = |p_i'|/E_i' and n the number of final-state particles. Eq. (4.9)
    # is J = const . chi^(3n-5) / (G . prod_i E_i') with G = sum_i a_i/E_i'.
    # Differentiating the constraint sum_i E_i' = sqrt(shat) gives
    # d(chi)/d(mu_k) = -1/(2 E_k' chi G) < 0, and then
    #
    #   2 E_k' G chi^2 . dlnJ/dmu_k
    #       = -(3n-5) + beta_k^2 + sum_i beta_i^2
    #         - (sum_i E_i' beta_i^4)/(sum_i E_i' beta_i^2)
    #         - (sum_i E_i' beta_i^2)/E_k'      ==  D_k .
    #
    # The third term is >= 0 and the fourth is >= beta_k^2 (its k-th summand
    # alone), so D_k <= -(3n-5) + sum_i beta_i^2 <= -(3n-5) + n = 5 - 2n, which
    # is <= -1 for every n >= 3. For n = 2 the closed form settles it directly:
    # J = chi = lambda^(1/2)(s,m_1'^2,m_2'^2)/lambda^(1/2)(s,m_1^2,m_2^2), and
    # lambda^(1/2) decreases in each mass. Checked numerically as well: 1.6e6
    # directional probes over n = 2..10, |p| and m each spanning eight decades,
    # zero violations.
    #
    # What is NOT bounded here, and falls back to the global probe bound:
    #  * the offshell spinmodes (madspin/full), whose w_mass carries the extra
    #    Tr(rho_off)/|M_prod|^2_on -- a matrix-element ratio with no cheap
    #    maximum. Only PA/onshell is covered. That ratio is not why: measured on
    #    3.9e6 free mass sets it is 1.0000 +- 0.012. What kills the construction
    #    offshell is Zhat, whose window spans a factor 3.2 there against 1.16
    #    under PA, so max_m Zhat sits 2.9x above the typical weight and the
    #    corner bound comes out LOOSER than the probe's global one (eps_m 5.00
    #    against 3.06). Even the partial bound J_corner . max_sample(rest) is
    #    worth 2%, and is a loss once the run-level factor is measured as a
    #    maximum rather than extrapolated. Section 15 of
    #    doc/madspin_sequential_plan.md has the numbers;
    #  * a production event carrying an onshell propagator (status 2), where
    #    reshuffle_production multiplies in a reshuffle_decay jacobian per
    #    sub-decay that is not part of this factorisation;
    #  * a window whose low corner is already over threshold
    #    (sum of the window minima > sqrt(shat)), or one that inverts
    #    (max_mass <= min_mass, i.e. a budget below the window's own floor);
    #  * a jacobian the kernel reports as infeasible or non-finite at the
    #    corner.

    _MASS_BOUND_UNSUPPORTED = None      # last fallback reason, for the log

    def _announce_mass_bound(self, mass_bound, offshell, probe):
        """Say once, per worker, which bound the mass stage is using.

        Loud on purpose. When the per-event bound is in force, ``nb_sigma`` and
        ``Nevents_for_max_weight`` no longer reach the mass stage -- they still
        set every angle-stage bound -- and that is a user-visible change in what
        those knobs do. It is also what finally makes the mass stage's cost
        reproducible: the probe-based bound was measured to scatter +-40% run to
        run without converging, because it extrapolates a tail from the first
        ``Nevents_for_max_weight`` events of the file.
        """
        if probe is not None:
            return
        if mass_bound is not None:
            if getattr(self, '_mass_bound_announced', False):
                return
            self._mass_bound_announced = True
            logger.info(
                "MadSpin sequential: the mass stage now bounds each production "
                "event separately -- max(J) at the low corner of the "
                "Breit-Wigner windows, times the exact maximum of "
                "jac_BW.Zhat per resonance. It is an upper bound by "
                "construction, so no mass-set weight can overflow it. Note "
                "nb_sigma and Nevents_for_max_weight no longer set the MASS "
                "stage's bound (they still set every angle stage's).")
        else:
            if getattr(self, '_mass_bound_fallback_announced', False):
                return
            self._mass_bound_fallback_announced = True
            reason = ('the spinmode is offshell (madspin/full), where the '
                      'per-event construction was MEASURED to be looser than '
                      'this bound, not merely unavailable -- see section 15 of '
                      'doc/madspin_sequential_plan.md'
                      if offshell else self._MASS_BOUND_UNSUPPORTED)
            logger.info(
                "MadSpin sequential: the mass stage keeps the probe's global "
                "maximum weight -- %s.", reason or 'unsupported event')

    def _mass_stage_bound(self, production, order, particles, slot_to_index,
                          zkeys, keep_jac):
        """C_e for one production event, or None when the event is one of the
        unsupported cases listed above (the caller then uses the global bound).

        Cached on the production event: the chain re-enters on every rejected
        mass set and the bound does not depend on the draw.
        """
        cached = getattr(production, '_ms_mass_bound', False)
        if cached is not False:
            return cached
        bound = self._mass_stage_bound_compute(production, order, particles,
                                               slot_to_index, zkeys, keep_jac)
        production._ms_mass_bound = bound
        return bound

    def _mass_stage_bound_compute(self, production, order, particles,
                                  slot_to_index, zkeys, keep_jac):
        if any(int(p.status) not in (-1, 1) for p in production):
            self._MASS_BOUND_UNSUPPORTED = (
                'the production event carries an onshell propagator, whose '
                'sub-decay reshuffling jacobian is not part of the bound')
            return None
        sqrts = production.sqrts
        if not sqrts or not (sqrts > 0):
            self._MASS_BOUND_UNSUPPORTED = 'sqrt(shat) is not usable'
            return None

        # the low corner of the window, slot by slot, in the order the draw
        # spends the budget in -- which is what makes each jac_bw maximal
        budget = sqrts
        corner = {}
        bw_max = 1.0
        for slot in order:
            pdg = particles[slot_to_index[slot]].pid
            pole, width, min_mass, max_mass, jac_bw = self._mass_window(pdg, budget)
            if not (max_mass > min_mass) or not (min_mass > 0):
                self._MASS_BOUND_UNSUPPORTED = (
                    'the Breit-Wigner window of pdg %s is empty at this '
                    'sqrt(shat)' % pdg)
                return None
            corner[slot] = min_mass
            bw_max *= jac_bw
            budget -= min_mass
            if budget <= 0:
                self._MASS_BOUND_UNSUPPORTED = (
                    'the window minima alone exceed sqrt(shat)')
                return None

        jac_max = 1.0
        if keep_jac:
            frame = getattr(production, '_ms_shuffle_frame', None)
            if frame is None:
                # from the *round-tripped* event, because that is what
                # _production_jacobian_for reshuffles: str() truncates every
                # momentum to %.10e, and the bound has to dominate the weight
                # the accept/reject actually computes, not an idealised one.
                probe = lhe_parser.Event(str(production))
                finals = [p for p in probe if int(p.status) == 1]
                frame = lhe_parser.Event.mass_shuffle_frame(
                    [lhe_parser.FourMomentum(p) for p in finals], probe.sqrts)
                production._ms_shuffle_frame = frame
                production._ms_shuffle_sqrts = probe.sqrts
                # the undrawn slots keep their nominal mass, and it is the
                # round-tripped one reshuffle_production reads
                production._ms_shuffle_masses = [p.mass for p in finals]
            masses = list(production._ms_shuffle_masses)
            for slot, mass in corner.items():
                masses[slot_to_index[slot]] = mass
            jac_max = lhe_parser.Event.mass_shuffle_jacobian(
                frame, masses, production._ms_shuffle_sqrts)
            if jac_max in (0, -1) or not math.isfinite(jac_max) \
                    or not jac_max > 0:
                self._MASS_BOUND_UNSUPPORTED = (
                    'the reshuffling jacobian has no value at the window low '
                    'corner (that mass set is already infeasible)')
                return None

        z_max = 1.0
        for slot in order:
            z_max *= self._zhat_max(zkeys[slot])

        bound = jac_max * bw_max * z_max
        if not math.isfinite(bound) or not bound > 0:
            self._MASS_BOUND_UNSUPPORTED = 'the bound is not a positive number'
            return None
        return bound

    @staticmethod
    def _weighted_polyfit2(xs, ys, ws):
        """Weighted least-squares quadratic y = c0 + c1 x + c2 x^2, by the normal
        equations. Returns None when the system is too close to degenerate to
        solve meaningfully. Small and self contained -- numpy is not imported
        here, and this runs once per slot at the end of the max-weight scan, so
        a 3x3 solve in plain python costs nothing worth optimising.

        The pivot tolerance is *relative* to the size of the matrix: the fit
        variable is ln(m/pole), which spans about +-0.13 over a Breit-Wigner
        window, so the moments of x^4 are ~1e-4 of the moments of x^0 and an
        absolute threshold would mean something different for every resonance.
        A relative one rejects exactly the case that matters -- every bin at the
        same virtuality, where the quadratic is not determined -- and accepts
        the normal conditioning of this fit (~3e4, which double precision
        handles with ten digits to spare).
        """
        n = len(xs)
        if n < 3:
            return None
        # symmetric 3x3 normal matrix, moments of x up to 4
        moment = [sum(w * x ** k for x, w in zip(xs, ws)) for k in range(5)]
        rhs = [sum(w * y * x ** k for x, y, w in zip(xs, ys, ws)) for k in range(3)]
        mat = [[moment[i + j] for j in range(3)] + [rhs[i]] for i in range(3)]
        tolerance = 1e-12 * max(abs(value) for row in mat for value in row[:3])
        for col in range(3):       # gaussian elimination with partial pivoting
            pivot = max(range(col, 3), key=lambda r: abs(mat[r][col]))
            if abs(mat[pivot][col]) <= tolerance:
                return None
            mat[col], mat[pivot] = mat[pivot], mat[col]
            for row in range(3):
                if row == col:
                    continue
                factor = mat[row][col] / mat[col][col]
                for k in range(col, 4):
                    mat[row][k] -= factor * mat[col][k]
        return [mat[i][3] / mat[i][i] for i in range(3)]

    def _build_z_tables(self, z_samples, nb_bin=20, min_per_bin=20):
        """Fit ln Z_k(m) from the (virtuality, rate factor) pairs the max-weight
        probe collected, one table per slot key.

        The samples are binned in m and *averaged* -- Z_k is an expectation, so
        the mean of the samples estimates it while their logarithm would not --
        then a quadratic in ln(m/pole) is fitted through the bin means, weighted
        by their counts. The fit is what smooths the tails, where the
        Breit-Wigner leaves few samples; the accepted lineshape is sensitive to
        the *slope* of ln Z, and a fractional error there survives as the same
        fractional error on the shift it corrects.

        Bins whose mean is zero sit below the decay threshold (no pool event can
        be reshuffled onto that virtuality): they are excluded from the fit and
        recorded as ``zero_below``, which then makes the mass-set stage reject
        those virtualities outright.
        """
        tables = {}
        for key, samples in sorted(z_samples.items()):
            if len(samples) < 4 * min_per_bin:
                logger.warning("MadSpin sequential: only %d probe samples for "
                               "slot %s, not tabulating its rate factor",
                               len(samples), key)
                continue
            pole = self.banner.get('param', 'mass',
                                   abs(int(key.split('_')[0]))).value
            samples = sorted(samples)
            lo, hi = samples[0][0], samples[-1][0]
            if not pole or hi <= lo:
                continue
            width = (hi - lo) / float(nb_bin)
            bins = [[0, 0.0, 0.0] for _ in range(nb_bin)]   # count, sum m, sum s
            for mass, value in samples:
                index = min(nb_bin - 1, int((mass - lo) / width))
                bins[index][0] += 1
                bins[index][1] += mass
                bins[index][2] += value
            zero_below = 0.0
            points = []
            for count, sum_mass, sum_value in bins:
                if count < min_per_bin:
                    continue
                mean_mass = sum_mass / count
                mean_value = sum_value / count
                if mean_value <= 0:
                    # entirely below threshold: everything up to here is dead
                    zero_below = max(zero_below, mean_mass)
                    points = []
                    continue
                points.append((math.log(mean_mass / pole),
                               math.log(mean_value), count))
            coeff = self._weighted_polyfit2([p[0] for p in points],
                                            [p[1] for p in points],
                                            [float(p[2]) for p in points])
            if coeff is None:
                logger.warning("MadSpin sequential: could not fit the "
                               "rate factor of slot %s (%d usable bins)",
                               key, len(points))
                continue
            fit = lambda u: math.exp(coeff[0] + u * (coeff[1] + u * coeff[2]))
            residual = max(abs(math.exp(y) / fit(u) - 1) for u, y, _ in points)
            # normalise to 1 at the pole: a constant multiplies every mass set
            # alike, so it is absorbed by the maximum weight, and it makes the
            # table readable against the running width it estimates
            coeff[0] = 0.0      # ``fit`` closes over coeff: normalised from here
            tables[key] = {'pole': pole, 'coeff': coeff,
                           'zero_below': zero_below,
                           'range': (max(lo, zero_below), hi)}
            logger.info("MadSpin sequential: slot %s rate factor "
                        "Z(%.5g)=%.3f  Z(%.5g)=1  Z(%.5g)=%.3f "
                        "(%d samples, %d bins, bin/fit deviation up to %.1f%%)",
                        key, lo, fit(math.log(max(lo, zero_below) / pole)),
                        pole, hi, fit(math.log(hi / pole)),
                        len(samples), len(points), 100 * residual)
        return tables

    def _check_weight_identity(self, production, decays, decay_dict, w_seq,
                               helicities, stats, offshell=True, keep_jac=True,
                               parents=None):
        """sequential_debug: the identity the whole decomposition rests on,
        checked on the accepted chain instead of inferred from a distribution.

            w_mass_raw * prod_k w_k_raw  ==  prod_i n_i * wgt_joint

        with w_mass_raw the mass-set weight *before* the tabulated Z_hat and
        w_k_raw each slot's weight before it is divided back out -- Z_hat
        cancels between the two stages, so this tests the decomposition itself
        and not the quality of the table. ``wgt_joint`` is recomputed here by the
        joint code, on copies, for the same production event, the same
        virtualities and the same decays.

        A statistical A/B can only bound a bias at the level its Monte Carlo
        error allows, and needs an error model to say even that. This is
        deterministic: any scheme whose per-chain weight product is not the
        joint weight is wrong, whatever a lineshape comparison happens to show.
        """
        prod_copy = lhe_parser.Event(str(production))
        decays_copy = collections.defaultdict(list)
        jac_bw = 1.0
        # ``slot`` below is a free-running index over the grouped walk of
        # ``decays``, and it *is* the density matrix slot index -- the same one
        # ``parents`` (PA: prod_static['init_part']) is keyed by in
        # sequential_accept_reject. That is an invariant of how both sides are
        # built, not a coincidence to re-derive:
        #   * slots are laid out "for pdg in decays_key, for particle in
        #     production order" (_sequential_slots / _density_basis), so a pdg
        #     owns a *contiguous* block of slots and the blocks come in
        #     decays_key order;
        #   * sequential_accept_reject fills the returned dict by ascending
        #     slot (``for slot in range(len(order))``), never in accept/reject
        #     order -- _decay_slot_order only decides which slot is drawn next,
        #     it must never permute the layout;
        #   * so the dict's key order is decays_key order, each pdg's list is
        #     that pdg's slot block in ascending order, and walking the groups
        #     flat enumerates slots 0 .. n-1 exactly.
        # The assertion below pins it down, so a future change to either side
        # trips here (under sequential_debug) instead of silently undoing a
        # boost with another particle's momentum.
        slot = 0
        for pdg, decay_list in decays.items():
            for decay in decay_list:
                copy = lhe_parser.Event(str(decay))
                if not offshell and parents is not None:
                    assert parents[slot].pid == pdg, \
                        ('sequential_debug: slot %d of the accepted chain is a '
                         '%s but the grouped walk over the decays reached it as '
                         'a %s -- the decays dict is no longer in slot order'
                         % (slot, parents[slot].pid, pdg))
                    # PA hands back its accepted decays already boosted to the
                    # lab frame -- _slot_density boosts them in place, and that
                    # is the frame add_decays wants -- while
                    # calculate_matrix_element_from_density does that boost
                    # itself. Undo it so the joint route starts where it
                    # expects to. (Offshell takes its density on a copy, so
                    # there the drawn decay is still in its rest frame.)
                    copy.boost(lhe_parser.FourMomentum(parents[slot]))
                mass = getattr(decay[0], 'new_mass', None)
                if mass is not None:
                    copy[0].new_mass = mass
                    copy[0].reshuffle_info = decay[0].reshuffle_info
                decays_copy[pdg].append(copy)
                slot += 1
        # the Breit-Wigner sampling jacobians: the joint path folds them in
        # itself when it draws the masses, and here the masses are given, so
        # they are recomputed from the same (pole, width, window) the draw used
        for pdg, decay_list in decays.items():
            for decay in decay_list:
                info = getattr(decay[0], 'reshuffle_info', None)
                if info is None:
                    continue        # no virtuality was sampled for this slot
                pole, width, min_mass, max_mass = info
                gap = math.atan((pole ** 2 - min_mass ** 2) / pole / width)
                gap += math.atan((max_mass ** 2 - pole ** 2) / pole / width)
                jac_bw *= gap / math.pi
        full_me, _, prod_diag, dec_diag, jac_reshuffle = \
            self.calculate_matrix_element_from_density(prod_copy, decays_copy,
                                                       decay_dict)
        w_joint = full_me / (prod_diag * dec_diag) * jac_reshuffle * jac_bw
        if not offshell and keep_jac:
            # PA's reshuffling jacobian is not inside
            # calculate_matrix_element_from_density: the pole approximation
            # leaves the momenta onshell there (it returns jac_reshuffle = 1)
            # and the joint path takes the jacobian from a reshuffle of the
            # *complete* event, done outside. Recomputing it that way is what
            # makes this an independent check of the two pieces the chain used
            # instead -- the mass stage's _production_jacobian_for and the
            # per-slot _decay_reshuffle_jacobian -- whose product it must equal.
            rebuilt = collections.defaultdict(list)
            for pdg, decay_list in decays.items():
                for decay in decay_list:
                    copy = lhe_parser.Event(str(decay))
                    copy[0].new_mass = decay[0].new_mass
                    copy[0].reshuffle_info = decay[0].reshuffle_info
                    rebuilt[pdg].append(copy)
            full_evt = lhe_parser.Event(str(production)).add_decays(rebuilt)
            jac_full = full_evt.reshuffle_production(_allow_retry=False)
            if jac_full in (0, -1):
                # the chain kept this mass set, so this should not happen; skip
                # the chain rather than compare against a meaningless number
                logger.debug('sequential_debug: the accepted mass set does not '
                             'reshuffle through the joint route, chain skipped')
                return
            w_joint *= jac_full
        nb_hel = 1
        for hel in helicities:
            nb_hel *= len(hel)
        if not w_joint:
            return
        # What must hold is *proportionality*, not equality: the chain weight and
        # the joint weight differ by a constant -- the number of helicity states,
        # and whatever normalisation the density path applies to the decay matrix
        # elements relative to calculate_matrix_element -- and any constant is
        # absorbed by the bounds. So accumulate the ratio and let the report look
        # at its spread: a constant ratio *is* the identity, whatever its value,
        # while a scheme that samples the wrong distribution has a ratio that
        # varies chain to chain.
        ratio = w_seq / (nb_hel * w_joint)
        stats['nb_identity_check'] += 1
        stats['identity_ratio_sum'] += ratio
        stats['identity_ratio_sqsum'] += ratio * ratio

    def sequential_accept_reject(self, production, evt_decayfile, maxwgts,
                                 nb_remain, stats=None, probe=None,
                                 probe_extra=None, decay_dict=None):
        """Accept/reject one decaying particle at a time, in density mode.

        Returns the accepted ``decays`` dict (pdg -> list of decay events, in
        the order add_decays expects), or None if the production event has
        nothing to decay.

        Exactness: slot k is accepted with probability w_k / C_k where

            w_k = (N_k / N_{k-1}) * jac_bw_k * jac_dec_k * (J_k / J_{k-1})

        with jac_bw_k the Breit-Wigner sampling jacobian of slot k's virtuality,
        jac_dec_k the jacobian of reshuffling slot k's decay onto it, and J_k the
        production reshuffling jacobian with the slots drawn so far offshell. The
        N and J factors telescope and the per-slot ones multiply out, so the
        product reproduces the joint weight (which gets jac_dec_k from the same
        reshuffle_production call that gives it J). On a reject only *that* slot
        is redrawn; the slots already accepted are kept. See
        doc/madspin_sequential_plan.md.

        Failure handling follows the scope of the failure. Under
        ``sequential_with_mass``, where slot k draws its own mass, a mass its
        decay products cannot accommodate is redrawn on the spot; a mass *set*
        the production cannot reshuffle is only knowable once every slot has a
        mass, so it trashes the whole set and restarts the chain. In the
        up-front schemes the mass set is fixed before the loop, so the same
        decay-side failure is a rejection of that decay instead.

        Every scheme but ``sequential_with_mass`` splits this in two: a mass-set
        accept/reject first, then the per-angle loop above. The mass stage
        carries everything that depends on the virtualities but not on the
        angles -- the Breit-Wigner sampling jacobians, the production
        reshuffling jacobian, and offshell also the offshell production trace --
        so those are evaluated once per mass set instead of once per slot trial,
        which is the whole point of drawing the masses first. The per-angle loop
        then redraws until it accepts and so divides out its own normalisation
        Z_k(m), a function of the sampled virtuality -- hence the tabulated
        ``_zhat`` factor in the mass-set weight, without which the accepted
        resonance lineshape is the Breit-Wigner one. Under
        ``sequential_global_retry`` a rejected decay trashes the mass set, the
        per-angle stage stops normalising, and Z_hat cancels from the chain
        (leaving it a pure efficiency preconditioner).

        ``probe``: when a list is given nothing is ever rejected and each slot's
        w_k is appended to it instead. That is how the max-weight scan measures
        the bounds -- on exactly the weights this loop will later test, since it
        is this same code computing them. The weights are recorded *before*
        ``_zhat`` is applied (the probe is what measures it, so it does not
        exist yet); ``probe_extra`` carries the virtualities and the rate-factor
        samples the scan needs to build it.
        """
        if getattr(self, '_decay_groups', None):
            # Loud on purpose: this loop redraws one slot until it is accepted,
            # which would divide E[w_k | group] out of the chain and distort the
            # group fractions. _sequential_active refuses the whole scheme when
            # the decays are grouped; if that guard is ever lifted it must come
            # with a bound and a rate factor per group, not with this loop as it
            # stands. See doc/madspin_decay_groups.md section 4.4.
            raise Exception("MadSpin: the per-particle accept/reject cannot "
                            "honour '@' decay groups; this should have fallen "
                            "back to the joint one.")
        decays_key = self._decaying_pdgs(production, evt_decayfile)
        if not decays_key:
            return None


        if not hasattr(self, 'f2py_module'):
            self.f2py_module = [0, 0] # first index is production, second is decay
            self.pdg2prefix = [0, 0]
            all_prefix = [0, 0]
            all_pdg = [0, 0]
            all_procid = [0, 0]

            self.create_and_initialise_f2py_modules(all_prefix, all_pdg, all_procid)

        prod_static = getattr(production, '_ms_density_static', None)
        if not prod_static or prod_static.get('decays_key') != decays_key:
            prod_static = self._density_basis(production, decays_key)
            production._ms_density_static = prod_static

        helicities = prod_static['helicities']
        init_part = prod_static['init_part']
        order = self._decay_slot_order(prod_static['decaying_spins'])
        particles, slot_to_index = self._sequential_slots(production, decays_key)
        ids = [p.pid for p in particles]

        # madspin/full evaluate the production density at reshuffled (offshell)
        # momenta that couple all decay masses, so rho is drawn per chain (after
        # the up-front reshuffle) rather than once at onshell. PA/onshell keep a
        # fixed onshell rho, cached on the production event.
        offshell = self._sequential_offshell()
        # Whether the virtualities are drawn before the angle loop. True for
        # every scheme but sequential_with_mass, which draws each slot's mass
        # inside that slot's own accept/reject.
        # sequential_global_retry: reject the mass set on a rejected decay
        # instead of redrawing that decay, so the per-angle stage never
        # normalises. See the Z_k discussion above _z_slot_keys.
        # two_stage: one bound over all the angles instead of one per particle,
        # the mass set paying for a rejection either way -- the joint
        # accept/reject with a mass-set stage in front of it. Exact for the same
        # reason sequential_global_retry is (nothing is redrawn in place, so no
        # stage normalises itself) and it keeps Z_hat only as a preconditioner,
        # since it cancels between the two stages.
        mode = self._unweighting_mode()
        upfront = self._is_upfront_scheme(mode)
        joint_angles = upfront and mode == 'two_stage'
        exact = upfront and mode == 'sequential_global_retry'
        zkeys = self._z_slot_keys(particles, slot_to_index) if upfront else None
        # |M_prod|^2 on shell: the denominator the joint offshell weight divides
        # by (calculate_matrix_element_from_density evaluates it *before*
        # reshuffle_production and returns it as prod_diag). It depends on the
        # production event only -- which the chain never redraws -- so leaving it
        # out would not bias anything, but it would leave the absolute scale of
        # the production matrix element inside the mass-set weight while its
        # bound is a single number shared by every production event: the loud
        # kinematics would set the bound and then overflow it, the quiet ones
        # would pay for it in acceptance. Cached on the event under the name the
        # joint path already uses for the same quantity.
        # frame the helicity basis is defined in (run_card me_frame), shared by
        # the production density and by every decay contracted against it. The
        # offshell branch gets its own from _upfront_production, derived from
        # the reshuffled production rho is evaluated at.
        frame_boost = None
        me_prod_on = 1.0
        if offshell:
            me_prod_on = getattr(production, 'me_wgt', None)
            if not me_prod_on:
                me_prod_on = self.calculate_matrix_element(production)
                production.me_wgt = me_prod_on
            if not me_prod_on:
                # a production event with no matrix element cannot be normalised
                # to itself; leave the weight unscaled rather than divide by zero
                logger.debug('sequential: |M_prod|^2 = 0, mass weight unscaled')
                me_prod_on = 1.0
        density_prod = None
        if not offshell:
            # The frame has to travel with the cached rho: every decay density
            # is contracted against that rho, so it must be taken in the same
            # frame. Cached alongside it rather than recomputed, because the
            # max-weight probe calls this hundreds of times per production event
            # -- and recomputing it only on a cache *miss* would leave it None on
            # every call after the first, silently contracting lab-frame decay
            # densities against an me_frame production one.
            density_prod = getattr(production, '_ms_density_prod', None)
            if density_prod is None:
                frame_boost = self._frame_boost(production)
                density_prod = self.get_density(production, prod_static['position'],
                                                prod_static['allowed_hel'],
                                                prod_static['ncomb'],
                                                prod_static['dimension'],
                                                frame_boost=frame_boost,
                                                hel_restriction=prod_static.get('hel_restriction'),
                                                hel_restriction_trace=prod_static.get('hel_restriction_trace'))
                # a zero trace here would make n_prev == 0 below and every slot
                # weight a 0/0 NaN, i.e. an accept/reject that never accepts
                self._check_production_density(production, density_prod,
                                               'sequential accept/reject, onshell rho')
                production._ms_density_prod = density_prod
                production._ms_frame_boost = frame_boost
            else:
                frame_boost = getattr(production, '_ms_frame_boost', None)

        # PA samples a virtuality per resonance; onshell does not. 2 -> 1
        # production has no recoil phase space for RAMBO to redistribute.
        nb_prod_final = sum(1 for p in production if int(p.status) == 1)
        draw_mass = (self.options['spinmode'] == 'PA' and nb_prod_final > 1)
        # Whether the production reshuffling jacobian enters the accept/reject
        # weight. Follows the joint path (interface_madspin.py, get_onshell PA
        # block): on by default -- when off the reshuffle is a post-acceptance
        # kinematic dressing and only the Breit-Wigner sampling jacobian is in
        # the weight. The feasibility of the mass set is still checked either
        # way, to trigger the whole-set restart.
        keep_jac = draw_mass and self.options['density_keep_jacobian']

        if stats is None:
            stats = collections.defaultdict(int)

        # The mass stage's bound. Per production event where that is possible
        # (see _mass_stage_bound), the probe's global maxwgts[0] otherwise. The
        # per-event one is a proven upper bound rather than an extrapolation of
        # the probe's tail, so nothing it tests can overflow -- which is what
        # the overweight counters at the end of the run report.
        #
        # The accepted mass distribution is q_e(m) . min(1, w/C) followed by a
        # redraw, i.e. proportional to q_e(m) w(m) for ANY C >= max w: the bound
        # cancels out of it. Changing C therefore changes the trial sequence and
        # the cost, and nothing about the sample.
        mass_bound = None
        if probe is None and maxwgts and upfront and draw_mass and not offshell:
            mass_bound = self._mass_stage_bound(production, order, particles,
                                                slot_to_index, zkeys, keep_jac)
        # Which events *have* a mass stage to bound: the up-front schemes, and
        # there whichever family draws a virtuality up front. ``_upfront_production``
        # fills slot_mass under `offshell or draw_mass`, and `draw_mass` alone is
        # the PA half of that -- so gating on it left the offshell spinmodes
        # counted in neither column and never announced, and let
        # sequential_with_mass (which is not an up-front scheme at all, and whose
        # mass_bound is dead) announce a bound it does not use.
        if maxwgts and upfront and (offshell or draw_mass):
            # one per chain call, i.e. one per production event reaching the
            # mass stage -- a rejected mass set loops *inside* the chain, so
            # these count events and not draws
            if mass_bound is None:
                stats['nb_mass_bound_global'] += 1
            else:
                stats['nb_mass_bound_event'] += 1
            self._announce_mass_bound(mass_bound, offshell, probe)
        if probe is not None and probe_extra is None:
            probe_extra = {}

        # Consecutive slot draws whose weight was not a finite positive number.
        # None of the loops below has an exit other than an acceptance, so a
        # structurally dead weight redraws (and regenerates that slot's decay
        # pool) for ever. Held at chain scope on purpose: a scheme that restarts
        # the mass set on a rejection would otherwise reset it every time and
        # never reach the bound. Only a *positive* weight clears it, which is
        # what makes it blind to a merely low acceptance -- an infeasible
        # virtuality never gets here (it is handled, and separately bounded, by
        # nb_infeasible).
        dead_trials = 0

        # ---- the overweight safety net (section 14 of
        # doc/madspin_sequential_plan.md) --------------------------------------
        # Every stage below accepts with probability min(1, w/C) and therefore
        # clips at 1 when w > C. The chain records max(1, w/C) per stage and the
        # caller multiplies the product onto the event weight, which restores
        # the sampled density exactly because min(1,x)*max(1,x) = x.
        #
        # Composition. The stages are nested, not sequential, so the two factors
        # have different lifetimes and each is reset where the quantity it
        # describes is redrawn:
        #   carry_mass   -- reset at the top of THIS loop, i.e. whenever a new
        #                   mass set is drawn (a mass-set rejection, an
        #                   infeasible set, or an ``exact``/joint-angle restart
        #                   all come back here);
        #   carry_angles -- reset at the top of the angle loop, i.e. whenever
        #                   the whole angle set is redrawn. Inside it the
        #                   per-slot factors *multiply*, exactly like ``w_slots``
        #                   multiplies the raw per-slot weights: each slot is
        #                   accepted once per pass, and a rejected slot trial is
        #                   simply redrawn and contributes nothing.
        # Only the accepted chain's factors survive, because both resets happen
        # on the redraw path. The product is taken once, at the return.
        carry_mass = 1.0
        carry_angles = 1.0

        while True:     # restart point: an impossible/rejected production mass set
            parents = init_part
            jac_prod = 1.0
            slot_mass = {}
            carry_mass = 1.0    # a new mass set: the previous one's factor is gone
            if upfront:
                # draw every virtuality, then settle whatever depends on the
                # mass set alone: offshell that is the production reshuffle and
                # rho, under PA the production reshuffling jacobian
                setup = self._upfront_production(production, order, particles,
                                                 slot_to_index, prod_static,
                                                 offshell, draw_mass=draw_mass,
                                                 density_prod=density_prod,
                                                 frame_boost=frame_boost)
                if setup is None:
                    stats['nb_production_restart'] += 1
                    continue
                density_prod, jac_prod, slot_mass, parents, frame_boost = setup

                # Mass-set accept/reject, before the per-angle loop. All the
                # factors that depend on the mass set but not the decay angles --
                # the production reshuffling jacobian, the Breit-Wigner sampling
                # jacobians, and offshell the production trace -- go here, so the
                # per-angle loop no longer carries them (that bundling made slot
                # 0's acceptance ~1/300 offshell, and cost PA one production
                # reshuffling per slot trial). See doc/madspin_sequential_plan.md
                # sections 10 and 11.
                if offshell:
                    # Tr(rho_off)/|M_prod|^2_on -- the offshell production matrix
                    # element over the onshell one, which is what the joint weight
                    # carries. Applied in probe mode too, unlike Z_hat: it is known
                    # before the scan, so the bound is measured on the same quantity
                    # the accept/reject will test. jac_prod is the jacobian of the
                    # reshuffle that produced rho_off.
                    w_mass = density_prod.trace().real / me_prod_on * jac_prod
                else:
                    # PA/onshell: rho is the onshell one and cancels between N_n
                    # and N_0, so nothing of the production matrix element rides
                    # here. jac_prod is the reshuffling jacobian of the whole
                    # mass set -- the factor the per-slot scheme re-evaluates on
                    # every trial and telescopes. Under density_keep_jacobian =
                    # False the reshuffle is a post-acceptance dressing and is
                    # not in the weight at all; only its feasibility was checked.
                    w_mass = jac_prod if keep_jac else 1.0
                for s in slot_mass:
                    w_mass *= slot_mass[s][2]
                # before Z_hat, which cancels between the two stages:
                # this is what the weight-identity check compares
                w_mass_raw = w_mass
                if probe is not None:
                    del probe[:]            # start this chain's probe vector
                    probe.append(float(w_mass))
                    probe_extra['keys'] = zkeys
                    probe_extra['order'] = list(order)
                    probe_extra['mass'] = [slot_mass[s][0] if s in slot_mass
                                           else 0.0
                                           for s in range(len(order))]
                    # not reset with the rest: a chain that ends up restarting
                    # still drew valid (virtuality, rate factor) pairs, and Z_k
                    # wants every one of them -- including the zeros of a
                    # virtuality below threshold, which is precisely where
                    # dropping them would bias the table upwards
                    probe_extra.setdefault('z', [])
                else:
                    # Z_k(m_k): what the per-angle stage will divide out again.
                    # Without it the accepted virtualities are Breit-Wigner
                    # distributed instead of physically distributed.
                    for s in slot_mass:
                        w_mass *= self._zhat(zkeys[s], slot_mass[s][0])
                if probe is None and maxwgts and slot_mass:
                    # the mass stage is the other loop with no exit but an
                    # acceptance: a w_mass that is structurally zero (a
                    # vanishing offshell production trace, a Z_hat that is zero
                    # everywhere) redraws mass sets for ever. Same counter as
                    # the slot loops, so any positive weight anywhere in the
                    # chain clears it.
                    dead_trials = self._dead_trial(
                        dead_trials, w_mass,
                        'the mass-set stage of the sequential accept/reject')
                    # no virtuality to unweight means w_mass is the constant 1
                    # (onshell, and 2 -> 1 production under PA): testing it
                    # against its bound would only throw chains away
                    cmass = maxwgts[0] if mass_bound is None else mass_bound
                    if w_mass > cmass:
                        stats['nb_overflow_mass'] += 1
                        # accepted with probability 1 instead of w/C: carry the
                        # excess. Set after the test would be equivalent (an
                        # overflowing weight cannot be rejected, since
                        # random.random() < 1), but it is set here so the
                        # counter and the factor stay one statement apart.
                        carry_mass = w_mass / cmass
                    if random.random() * cmass >= w_mass:
                        stats['nb_mass_reject'] += 1
                        continue            # redraw the whole mass set

            # Angle stage. The mass set, the production density and its
            # reshuffling jacobian are all fixed above and are *reused* by
            # every pass of this loop -- which is the whole point of drawing the
            # virtualities first: the joint accept/reject pays a production
            # reshuffling (and, offshell, a production density matrix) on every
            # trial, because a rejection there redraws the masses too.
            #   two_stage: a rejected angle set is redrawn
            #     against the same mass set, so this loop is where the reuse
            #     happens -- and, redrawing to acceptance, it normalises itself,
            #     which is what the Z_hat factor in w_mass compensates.
            #   sequential_global_retry: a rejected *decay* costs the mass set, which
            #     makes Z_hat cancel and that scheme exact whatever the table
            #     says, at the price of the reuse.
            while True:
                slot_densities = {}
                slot_decays = {}
                slot_masses = {}
                n_prev = self._partial_density_contraction(density_prod, helicities, {})
                j_prev = 1.0
                budget = production.sqrts
                restart = False
                w_angles = 1.0      # joint_angles: the product tested once, below
                angle_dead = False  # a zero member: reject the set, stop drawing
                w_slots = 1.0       # product of the raw per-slot weights
                carry_angles = 1.0  # overweight safety net: the product of the
                                    # per-slot (or, under two_stage, the single
                                    # angle-set) max(1, w/C) factors. Reset with
                                    # w_slots because it has the same lifetime:
                                    # a new pass here means a whole new angle set.

                for position, slot in enumerate(order):
                    index = slot_to_index[slot]
                    particle = particles[index]
                    # the up-front schemes reserve maxwgts[0] for the mass set,
                    # so their per-slot bounds start at index 1
                    wpos = position + 1 if upfront else position
                    if joint_angles:
                        # every slot contributes to the single angle weight, tested
                        # once the last one has been drawn
                        maxwgt = None
                    elif maxwgts:
                        maxwgt = maxwgts[wpos] if wpos < len(maxwgts) \
                                               else maxwgts[-1]
                    else:
                        maxwgt = None
                    nb_infeasible = 0
                    while True:
                        stats['nb_try_%d' % position] += 1
                        decay = self._draw_one_decay(particle, index, ids,
                                                     evt_decayfile, nb_remain)

                        if upfront:
                            mass = slot_mass.get(slot)
                            me_on = 1.0
                            dcopy = None
                            jac_dec = 1.0
                            if offshell:
                                # madspin/full: offshell numerator over onshell
                                # denominator. The mass was drawn up front, so the
                                # decay is reshuffled to it.
                                me_on = self.calculate_matrix_element(decay)   # |M_dec|^2_on
                                decay[0].new_mass, decay[0].reshuffle_info = \
                                    mass[0], mass[1]
                                # The offshell density is taken on a copy: the drawn
                                # decay must stay in its onshell rest frame (only tagged
                                # with new_mass) so the final add_decays + a single
                                # reshuffle_production rebuild consistent kinematics.
                                # Reshuffling/boosting it in place leaves it on the
                                # offshell parent and add_decays then rejects it.
                                dcopy = lhe_parser.Event(str(decay))
                                dcopy[0].new_mass = mass[0]
                                dcopy[0].reshuffle_info = mass[1]
                                # jac_dec_k: the decay reshuffling jacobian. Joint
                                # madspin has it (calculate_matrix_element_from_density,
                                # 'jac *= dec.reshuffle_decayevt()'), so it belongs in
                                # the per-slot weight here -- it depends on this slot's
                                # decay only, hence no telescoping ratio.
                                jac_dec = dcopy.reshuffle_decayevt()
                            elif mass is not None:
                                # PA: the decay stays onshell, only *tagged* with
                                # the virtuality that add_decays and the final
                                # reshuffle_production will consume, exactly as
                                # the per-slot mass draw leaves it. Its
                                # reshuffling jacobian is probed on a copy, and
                                # it is the same factor the joint PA weight picks
                                # up inside its full-event reshuffle_production.
                                decay[0].new_mass, decay[0].reshuffle_info = \
                                    mass[0], mass[1]
                                jac_dec = self._decay_reshuffle_jacobian(decay)
                            if jac_dec in (0, -1):
                                # This decay cannot be mapped onto the sampled
                                # virtuality (its products do not fit). That is a
                                # zero-weight candidate, i.e. an ordinary rejection
                                # of *this decay*, not of the mass set: a zero is
                                # part of Z_k, so counting it as a rejection here is
                                # what makes the tabulated Z_k the exact correction
                                # near a threshold. It is recorded as such in the
                                # probe. The fail-safe covers a virtuality no decay
                                # in the pool can reach -- with the table in place
                                # the mass stage rejects those outright, since Z_k
                                # vanishes there.
                                stats['nb_infeasible_%d' % position] += 1
                                if probe is not None:
                                    probe_extra['z'].append(
                                        (zkeys[slot], mass[0], 0.0))
                                elif joint_angles:
                                    # A zero anywhere makes the whole angle set
                                    # weight zero, so the set is rejected: stop
                                    # drawing the remaining slots. Redrawing
                                    # just this slot instead would propose from
                                    # the *feasible* part of the pool, and the
                                    # normalisation the mass stage compensates
                                    # for would become Z_k/(1 - q_k(m)) -- a
                                    # different function of the virtuality than
                                    # the tabulated one.
                                    w_angles = 0.0
                                    angle_dead = True
                                    break
                                elif exact:
                                    # Same argument for the per-slot restart
                                    # scheme: a zero-weight decay has to reject
                                    # the mass set rather than be redrawn here,
                                    # or the feasible fraction 1 - q_k(m) is
                                    # divided out of the accepted mass sets --
                                    # the same class of bias as Z_k itself, and
                                    # it would survive precisely where Z_k is
                                    # supposed to vanish.
                                    stats['nb_exact_restart'] += 1
                                    restart = True
                                    break
                                nb_infeasible += 1
                                if nb_infeasible < 200:
                                    continue
                                stats['nb_production_restart'] += 1
                                restart = True
                                break
                            # the frame rho was evaluated in, whichever branch
                            # produced it: the decay densities are contracted
                            # against rho, so they have to be taken there too
                            if offshell:
                                # per-angle factor only: (N_k/N_{k-1}) * jac_dec_k
                                # * Tr(D_off)/|M_dec|^2_on. jac_bw and the
                                # *production* reshuffling jacobian are in w_mass.
                                density = self._slot_density(dcopy, parents[slot],
                                                             helicities[slot],
                                                             frame_boost=frame_boost)
                                rate = jac_dec * (density.trace().real / me_on)
                            else:
                                # PA/onshell: the matrix elements are on shell, so
                                # there is no offshell/onshell rate ratio -- the
                                # only angle-dependent factor left over the density
                                # ratio is the decay reshuffling jacobian. With
                                # density_keep_jacobian off, joint PA does not put
                                # it in its weight either (the reshuffle runs after
                                # acceptance), so neither does this; a decay that
                                # cannot reach the virtuality is still a zero, and
                                # Z_k then measures the feasible fraction.
                                density = self._slot_density(decay, parents[slot],
                                                             helicities[slot],
                                                             frame_boost=frame_boost)
                                rate = jac_dec if keep_jac else 1.0
                            slot_densities[slot] = density
                            n_k = self._partial_density_contraction(
                                            density_prod, helicities, slot_densities)
                            wgt = (n_k / n_prev).real * rate
                            wgt_raw = wgt        # before any Z_hat division
                            if probe is None:
                                dead_trials = self._dead_trial(
                                    dead_trials, wgt,
                                    'slot %d of the sequential accept/reject'
                                    % position)
                            j_k, new_budget = j_prev, budget
                            # Z_hat_k(m_k), or 1 where there is no virtuality to
                            # condition on (onshell, 2 -> 1 production under PA)
                            zhat = self._zhat(zkeys[slot], mass[0]) \
                                   if mass is not None else 1.0
                            slot_carry = 1.0   # overweight safety net, this trial
                            if probe is not None:
                                probe.append(float(wgt))
                                # E[rate | m] = E[w_k | m] = Z_k(m) -- the pool
                                # average of the density ratio is one at fixed m,
                                # so the two have the same expectation and rate
                                # carries no polarisation modulation, which makes
                                # it the tighter estimator of the two.
                                if mass is not None:
                                    probe_extra['z'].append(
                                        (zkeys[slot], mass[0], float(rate)))
                                accept = True
                            elif joint_angles:
                                # no test here: every slot feeds the single angle
                                # weight, tested once the last decay is drawn
                                w_angles *= wgt / zhat if zhat > 0 else 0.0
                                accept = True
                            elif maxwgt is None:
                                accept = True
                            else:
                                if exact:
                                    # the mass stage already paid Z_hat_k, so the
                                    # bound this is tested against is the one of
                                    # w_k/Z_hat_k -- flat in the virtuality, and the
                                    # two factors cancel over the chain
                                    wgt = wgt / zhat if zhat > 0 else 0.0
                                if wgt > maxwgt:
                                    stats['nb_overflow_%d' % position] += 1
                                    slot_carry = wgt / maxwgt
                                    logger.debug('sequential: slot %s weight %s above'
                                                 ' its max %s', position, wgt, maxwgt)
                                accept = random.random() * maxwgt < wgt
                            if accept:
                                slot_decays[slot] = decay
                                n_prev = n_k
                                w_slots *= wgt_raw
                                if slot_carry != 1.0:
                                    # only the ACCEPTED trial of this slot
                                    # contributes; the slots multiply, like
                                    # w_slots does
                                    carry_angles *= slot_carry
                                break
                            slot_densities.pop(slot, None)
                            if exact:
                                # Redrawing this decay until it is accepted would
                                # normalise the per-angle stage and divide Z_k(m)
                                # out of the accepted mass sets. Rejecting the mass
                                # set instead keeps the chain acceptance
                                # proportional to the joint weight -- exact whatever
                                # Z_hat is, at the cost of throwing the slots
                                # already accepted away.
                                stats['nb_exact_restart'] += 1
                                restart = True
                                break
                            continue

                        jac_bw = 1.0        # Breit-Wigner *sampling* jacobian
                        jac_dec = 1.0       # decay *reshuffling* jacobian
                        new_budget = budget
                        if draw_mass:
                            # decay-side failure is local to this slot: redraw its
                            # mass, keep every slot already accepted
                            while True:
                                new_budget, jac_bw = self._draw_offshell_mass(
                                                    particle.pdg, decay, budget)
                                jac_dec = self._decay_reshuffle_jacobian(decay)
                                if jac_dec not in (0, -1):
                                    break
                                stats['nb_mass_redraw_%d' % position] += 1
                            slot_masses[slot] = (decay[0].new_mass,
                                                 getattr(decay[0], 'reshuffle_info', None))
                            if not keep_jac:
                                # joint PA bundles jac_dec with J_prod: both come out
                                # of the single reshuffle_production, which under
                                # density_keep_jacobian = False runs only after the
                                # event is accepted and so enters no weight. Mirror
                                # that here, or the two schemes stop matching.
                                jac_dec = 1.0

                        # The production reshuffling jacobian only enters the
                        # weight under density_keep_jacobian; then it is needed per
                        # trial. Otherwise its sole use is spotting a mass set the
                        # production cannot reshuffle, and that depends on the whole
                        # set, so it is checked once after the chain is complete --
                        # not here, where the reshuffle-on-a-copy dominated the cost.
                        j_k = j_prev
                        if keep_jac:
                            j_probe = self._production_jacobian_for(production,
                                                                   slot_to_index,
                                                                   slot_masses)
                            if j_probe in (0, -1):
                                stats['nb_production_restart'] += 1
                                restart = True
                                break
                            j_k = j_probe

                        # accepted slots reuse their stored (already normalised)
                        # density; only this slot's decay is evaluated here
                        slot_densities[slot] = self._slot_density(
                                        decay, init_part[slot], helicities[slot],
                                        frame_boost=frame_boost)
                        n_k = self._partial_density_contraction(density_prod, helicities,
                                                                slot_densities)
                        # jac_dec is this slot's own factor (it depends on this
                        # decay only), so unlike J it enters without a ratio -- the
                        # product over slots is what the joint reshuffle_production
                        # multiplies in.
                        wgt = (n_k / n_prev).real * jac_bw * jac_dec * (j_k / j_prev)
                        if probe is None:
                            dead_trials = self._dead_trial(
                                dead_trials, wgt,
                                'slot %d of the sequential accept/reject'
                                % position)
                        slot_carry = 1.0   # overweight safety net, this trial
                        if probe is not None:
                            # python float: these are marshalled as JSON when the
                            # scan runs across forked workers
                            probe.append(float(wgt))
                            accept = True
                        else:
                            if wgt > maxwgt:
                                # the bound was under-estimated: the excess used
                                # to be dropped silently, and is now carried on
                                # the event weight instead (section 14)
                                stats['nb_overflow_%d' % position] += 1
                                slot_carry = wgt / maxwgt
                                logger.debug('sequential: slot %s weight %s above '
                                             'its max %s', position, wgt, maxwgt)
                            accept = random.random() * maxwgt < wgt
                        if accept:
                            slot_decays[slot] = decay
                            n_prev, j_prev, budget = n_k, j_k, new_budget
                            if slot_carry != 1.0:
                                carry_angles *= slot_carry
                            break
                        # rejected: this slot only, drop what it contributed
                        slot_densities.pop(slot, None)
                        slot_masses.pop(slot, None)
                    if restart or angle_dead:
                        break
                if joint_angles and not restart and probe is None and maxwgts:
                    # One bound over the product of every slot's weight, instead
                    # of one bound per slot: the per-slot bounds' product is
                    # looser than a single bound on the product, and this buys
                    # that back at the price of the early exit a per-slot test
                    # gets when the first particle is rejected.
                    stats['nb_angleset_try'] += 1
                    c_angles = maxwgts[1] if len(maxwgts) > 1 else maxwgts[-1]
                    if w_angles > c_angles:
                        stats['nb_overflow_angles'] += 1
                        # the whole angle set is one test here, so this is the
                        # only angle-side factor under two_stage (the per-slot
                        # tests are disabled -- maxwgt is None there)
                        carry_angles = w_angles / c_angles
                        logger.debug('sequential: angle weight %s above its max %s',
                                     w_angles, c_angles)
                    if random.random() * c_angles >= w_angles:
                        stats['nb_angle_reject'] += 1
                        # keep the mass set, and with it the reshuffled
                        # production and its density matrix: only the decays are
                        # drawn again. That reuse is the point of this scheme.
                        continue
                break
            if (not restart and not upfront and draw_mass and not keep_jac
                    and probe is None):
                # sequential_with_mass, jacobian off: the masses are only all
                # known once the chain is complete, so the feasibility of the
                # set is checked here -- one reshuffle for the whole chain
                # instead of one per trial. The up-front schemes settled it
                # before the angle loop.
                if self._production_jacobian_for(production, slot_to_index,
                                                 slot_masses) in (0, -1):
                    stats['nb_production_restart'] += 1
                    restart = True
            if restart and probe is not None:
                # a partial probe vector was appended for this chain; drop it so
                # the next attempt records a clean [w_mass, w_0, ...] vector
                del probe[:]
            if not restart:
                break

        # back to the pdg -> list layout add_decays consumes, in slot order.
        # ``range(len(order))`` and not ``order``: the accept/reject ordering
        # says which slot is *drawn* next, it must not permute the layout. A
        # pdg owns a contiguous block of slots (_sequential_slots), so this
        # walks each block in ascending slot order and inserts the keys in
        # decays_key order -- which makes a flat walk over decays.items()
        # enumerate slots 0 .. n-1. _check_weight_identity relies on that to
        # pair a decay with parents[slot]; see the invariant spelled out there.
        decays = collections.defaultdict(list)
        for slot in range(len(order)):
            decays[particles[slot_to_index[slot]].pid].append(slot_decays[slot])
        if probe is None and (carry_mass != 1.0 or carry_angles != 1.0):
            # The overweight safety net's composed factor for the chain that was
            # actually accepted. Only set when it is not the identity, so the
            # caller's ``stats.pop('overweight_factor', 1.0)`` returns the
            # LITERAL 1.0 in the no-overflow case and the written weight is
            # bit-identical to the clipping one. Assigned rather than
            # accumulated: this is a per-event quantity, not a counter.
            stats['overweight_factor'] = carry_mass * carry_angles
        if (upfront and probe is None and decay_dict
                and self.options['sequential_debug']):
            self._check_weight_identity(production, decays, decay_dict,
                                        w_mass_raw * w_slots, helicities, stats,
                                        offshell, keep_jac, parents)
        if probe is None and self._polarization_weights_enabled():
            # keep_weight_for_polarization_*: one masked contraction per
            # combination on the accepted chain. The per-slot normalisation of
            # the decay densities is an overall scalar and cancels in the ratio,
            # so this is the same number the joint path computes. Skipped in
            # probe mode (the max-weight scan writes no events).
            self._polarization_ratios(
                density_prod,
                decay_density_tensor(self._slot_identity, helicities,
                                     slot_densities),
                prod_static)
        return decays

    def get_onshell_evt_and_wgt(self, production, decays, decay_dict, prod_density_cached=None, build_event=True):
        """ return the onshell wgt for the production event associated to the decays
            return also the full event with decay. 
            Carefull this modifies production event (pass to the full one)
            build_event: if False (density mode) compute weight without building event"""
        #print("\n\n\n\n\n======== debug get_onshell_evt_and_wgt =========")
        density_pole_approximation = self._density_pole_approximation()
        density_do_reshuffle = self._density_do_reshuffle()
        decay_me = 1.0
        decay_me_debug = 1.0
        jac = 1.0
        tag, order = production.get_tag_and_order()
        try:
            info = self.generate_all.all_me[tag]
        except:
            misc.sprint(self.generate_all.all_me)
            misc.sprint(production)
            misc.sprint(decays)
            raise
        
        # Calculate decay ME
        if self.generate_all.mode == 'onshell':
            #print(f"len(decays) = {len(decays)}")
            for pdg in decays:
                for dec in decays[pdg]:
                    #print(f"dec = {dec}")
                    decay_me *= self.calculate_matrix_element(dec)
        else:
            if self.options['density_debug']:
                #print(f"len(decays) = {len(decays)}")
                for pdg in decays:
                    for dec in decays[pdg]:
                        #print(f"dec = {dec}")
                        decay_me_debug *= self.calculate_matrix_element(dec)

        # Calculate production*decay ME
        if self.generate_all.mode == 'onshell':
            full_event = lhe_parser.Event(str(production))
            full_event = full_event.add_decays(decays)
            #print(f"full_event = {full_event}")
            full_me = self.calculate_matrix_element(full_event)
            #print(f"full_me = {full_me}")
        else:
            #offshell mode
            full_dqrts = production.sqrts
            jac = 1
            # 2 -> 1 production: the single resonance virtuality is fully fixed
            # by sqrt(shat); there is no recoil phase space, so RAMBO cannot
            # redistribute a sampled Breit-Wigner mass. Keep the resonance
            # onshell at the production-determined mass (no reshuffling) instead
            # of building kinematically inconsistent momenta.
            nb_prod_final = sum(1 for p in production if int(p.status) == 1)
            if nb_prod_final > 1 and (not density_pole_approximation or
                    density_do_reshuffle):
                for pdg in decays:
                    for dec in decays[pdg]:
                        full_dqrts, jac_dec = self._draw_offshell_mass(
                                                    pdg, dec, full_dqrts)
                        jac *= jac_dec
            if prod_density_cached is None:
                full_me, prod_density_cached, prod_diag, dec_diag, jac_reshuffle = self.calculate_matrix_element_from_density(production, decays, decay_dict)
            else:
                full_me, _, prod_diag, dec_diag, jac_reshuffle = self.calculate_matrix_element_from_density(production, decays, decay_dict, prod_density_cached)
            # The internal reshuffle (offshell/madspin) is the reshuffle of the
            # chain; fold its jacobian into the weight here so the caller does not
            # reshuffle the already-offshell event a second time.
            jac *= jac_reshuffle
            #print(f"full_me from density = {full_me}")
   
            full_event = None
            if build_event or self.options['density_debug']:
                # Create full event from production and decays
                if density_pole_approximation:
                    full_event = lhe_parser.Event(str(production))
                else:
                    full_event = production          
                # add_decays is non-destructive: ``decays`` survives this, which
                # is what lets the caller rebuild the event (PA reshuffling) and
                # what lets fixed_order attach the same draw to every member of
                # the event group.
                full_event = full_event.add_decays(decays)
            
                #print(f"full event 2 = {full_event}")
                if self.options['density_debug']:
                    me1 = self.calculate_matrix_element(full_event)
                    #print(f"me1 = {me1} , me2 = {full_me} , ratio = {me1/full_me}")
                    if abs(1-me1/full_me) > self.options['density_tolerance']:
                        print(f"full = {me1} , density = {full_me} , ratio = {me1/full_me}")	    
                        print(full_event)
                        print(production)
                        print(decays)

                        print("ERROR matrix element from density does not match with full matrix element")	
                        raise RuntimeError("ERROR matrix element from density does not match with full matrix element")	 
    
        # Calculate production ME and cache it so that if we reject 
        # the decay the production ME will not be recalculated
        if hasattr(production, 'me_wgt'):
            production_me = production.me_wgt
        else:
            production_me = self.calculate_matrix_element(production) if self.generate_all.mode == 'onshell' \
                            else prod_diag
            production.me_wgt = production_me

        if self.generate_all.mode == 'density' and self.options['density_debug']:
            prod_me = self.calculate_matrix_element(production)
            #print(f"prod_diag = {prod_diag} , prod_me = {prod_me}")
            if abs(1-prod_diag/prod_me) > self.options['density_tolerance']:
                print(f"prod_me = {prod_me} , prod_diag = {prod_diag} , ratio = {prod_diag/prod_me}")	    
                raise RuntimeError("ERROR production matrix element from density does not match with diagonal")	     
            if abs(1-dec_diag/decay_me_debug) > self.options['density_tolerance']:
                print(f"decay_me = {decay_me_debug} , dec_diag = {dec_diag} , ratio = {dec_diag/decay_me_debug}")	    
                raise RuntimeError("ERROR decay matrix element from density does not match with diagonal")	   
        
        if self.generate_all.mode == 'density':
            decay_me = dec_diag

        #print(f"full_event = {full_event}")
        #print(f"full_me = {full_me}")
        #print(f"production_me = {production_me}")
        #print(f"decay_me = {decay_me}")
        #print(f"wgt = {full_me/(production_me*decay_me)}")

        if getattr(self, '_pi_probe_c', False):
            # the same weight, with the interference restriction lifted: its
            # decay-phase-space mean is c. Normalised by the *same* prod/dec
            # denominators and the same jacobian, so the ratio to the returned
            # weight is exactly the ratio of the two contractions.
            unrestricted = getattr(self, '_pi_unrestricted_me', None)
            self._pi_unrestricted_wgt = None if unrestricted is None else \
                unrestricted / (production_me * decay_me) * jac
            self._pi_unrestricted_me = None

        return full_event, full_me/(production_me*decay_me)*jac, prod_density_cached


    def me_param_card(self, folder_name=None):
        """The parameters every MadSpin matrix element is initialised with.

        There is exactly one source of truth, ``path_me/param_card.dat``: the
        banner of the input event file (as overridden by ``import model <MODEL>
        <CARD>``), written out on every run. ``decay_all_events_onshell
        .refresh_me_param_cards`` copies it into each ME directory's ``Cards/``
        at compile time, so ``<folder>/Cards/param_card.dat`` is that same card;
        it is preferred only because it is the file sitting next to the code
        that reads it, and it is what an archived run keeps.

        What is deliberately *not* consulted is ``path_me/Cards/param_card.dat``
        -- the process directory's own card, which ``path_me`` happens to point
        at when MadSpin is launched from MadEvent. That card may have been
        edited after the events were generated, and using it would evaluate the
        production matrix element with parameters the events do not have. A
        user who does want other parameters says so with ``import model``,
        which is checked against the banner.
        """
        if folder_name:
            card = pjoin(self.path_me, folder_name, 'Cards', 'param_card.dat')
            if os.path.exists(card):
                return card
        return pjoin(self.path_me, 'param_card.dat')

    def initialise_f2py_module(self, mymod, sp_path, prod_or_decay):
        """ Routine to initialise the fortran module with module.initialise(param_card_path).
            If one the process is at loop-induced level, it is also needed to call module.set_madloop_path(path_to_MadLoop5_resources)
        """
        if prod_or_decay == 'prod':
            folder_name = self.ms_me_subdir
        elif prod_or_decay == 'decay':
            folder_name = self.ms_me_decay_subdir
        else:
            raise ValueError("prod_or_decay only accepts values as 'prod' or 'decay'.")

        with misc.chdir(sp_path): #changed the search of the card to the subdirectories madspin_me and madspin_decay
            mymod.initialise(self.me_param_card(folder_name))
            # If the module is loop-induced, we also need to set the directory in which the MadLoop param card is present
            MadLoopCardPath = pjoin(self.path_me, folder_name, 'SubProcesses', 'MadLoop5_resources')
            if os.path.exists(MadLoopCardPath):
                MLCard = banner.MadLoopParam(pjoin(MadLoopCardPath, 'MadLoopParams.dat'))
                MLCard.set("HelicityFilterLevel", 0) # HelicityFilterLevel is set to 0 because the computation of density matrices loop-induced requires it.
                MLCard.set("MLStabThres", 0.001)

                # Every forked unweighting worker runs this lazily, so N workers
                # rewrite this one shared file while sibling workers' MadLoop
                # Fortran init is reading it. An in-place write lets a reader see
                # a truncated card: MLReductionLib stays all-zero and MadLoop
                # answers with STOP "No available loop reduction lib ...", which
                # (a Fortran STOP) kills the worker with exit code 0, bypassing
                # every Python handler. Write to a private temp file and rename,
                # so a concurrent reader sees either the old or the new card,
                # never a partial one.
                _ml_dat = pjoin(MadLoopCardPath, 'MadLoopParams.dat')
                _ml_tmp = '%s.tmp%d' % (_ml_dat, os.getpid())
                MLCard.write(_ml_tmp)
                os.replace(_ml_tmp, _ml_dat)
                mymod.set_madloop_path(MadLoopCardPath)

        # the beam polarisation is constant over a run, so it is pushed into
        # the library once per module rather than passed on every call
        self._set_f2py_beampol(mymod)


    def create_f2py_module(self, sp_path, prod_or_decay, all_prefix, all_pdg, all_procid):
        """ Load the density-matrix f2py extensions and build the pdg -> prefix
            map, once. Both the matrix-element evaluation and get_density / get_pdir
            need it, so the sequential accept/reject -- which calls get_density
            directly, without going through calculate_matrix_element_from_density --
            must be able to trigger the same setup.
            The setup is done independently for the production part and the decay part
            and are stored in self.f2py_module[0] (production), self.f2py_module[1] (decay)
        """
        if sys.path[0] != sp_path:
            sys.path.insert(0, sp_path)
        
        if prod_or_decay == "prod":
            i = 0
            menum = 2
        elif prod_or_decay == "decay":
            i = 1
            menum = 1
        else:
            raise ValueError("The only acceptable values of prod_or_decay are 'prod' and 'decay'")

        # production and decay are built with distinct MENUM (2 vs 1) so
        # their f2py modules / dependent libraries don't clash in-process
        # (see decay_all_events_onshell.compile).
        mymod = self._load_f2py_matrix_module(sp_path, menum=menum)
        self.f2py_module[i] = mymod

        all_prefix[i] = self.f2py_module[i].get_prefix()
        all_pdg[i], all_procid[i] = self.f2py_module[i].get_pdg_order()
        self.pdg2prefix[i] = {}
        for j, pdg in enumerate(all_pdg[i]):
            pdg = tuple([x for x in pdg if x != 0])
            self.pdg2prefix[i][pdg] = (str(all_prefix[i][j].decode()).strip(), j)
        

        if self.model_init_prod and prod_or_decay == 'prod':
            self.model_init_prod = False
            self.initialise_f2py_module(mymod, sp_path, prod_or_decay='prod')

        if self.model_init_decay and prod_or_decay == 'decay':
            self.model_init_decay = False
            self.initialise_f2py_module(mymod, sp_path, prod_or_decay='decay')


    def create_and_initialise_f2py_modules(self, all_prefix, all_pdg, all_procid):
        """ Routine to create the f2py modules and to initialise them. It separates production and decay.
            It also fills all_prefix, all_pdg, all_procid which are lists of 2 elements. 
            The first element is the value for the production part, the second element is for the decay part.
        """
        try:
            sp_path_prod = pjoin(self.path_me, self.ms_me_subdir, 'SubProcesses')
            self.create_f2py_module(sp_path_prod, 'prod', all_prefix, all_pdg, all_procid)
        except:
            logger.critical("Error while creating the f2py modules for the production part.")

        # legacy options 'onshell_v1' and 'madspin_v1' store both the production and the decay in a single folder
        if self.options['spinmode'] not in ['onshell_v1', 'madspin_v1']:
            try:
                sp_path_decay = pjoin(self.path_me, self.ms_me_decay_subdir, 'SubProcesses')
                self.create_f2py_module(sp_path_decay, 'decay', all_prefix, all_pdg, all_procid)
            except:
                logger.critical("Error while creating the f2py modules for the decay part.")


    def calculate_matrix_element_from_density(self, production, decays, decay_dict, prod_density_cached=None):
        """routine to return the matrix element from density matrices"""

        # ------------------------------------------------------------------
        # Load f2py module and build pdg2prefix map if needed (unchanged logic)
        # ------------------------------------------------------------------
        # Since we need to compute the density matrix for both the production and the decay, we need to import two fortran modules
         
        if not hasattr(self, 'f2py_module'):
            self.f2py_module = [0, 0] # first index is production, second is decay
            self.pdg2prefix = [0, 0]
            all_prefix = [0, 0]
            all_pdg = [0, 0]
            all_procid = [0, 0]

            self.create_and_initialise_f2py_modules(all_prefix, all_pdg, all_procid)


        # ------------------------------------------------------------------
        # Cache production-only metadata reused across rejection retries
        # ------------------------------------------------------------------
        decays_key = tuple(decays.keys())
        MEdenom_prod, MEdenom_decay = None, None
        # Reshuffling jacobian of the internal (offshell/madspin) reshuffle. This
        # is THE reshuffle of the chain: the caller must fold it into the weight
        # rather than reshuffling the already-offshell event a second time. It
        # stays 1.0 for the pole-approximation path (which reshuffles later, after
        # acceptance) and for 2 -> 1 production (no phase space to redistribute).
        jac_reshuffle = 1.0
        prod_static = getattr(production, '_ms_density_static', None)
        density_pole_approximation = self._density_pole_approximation()
        density_do_reshuffle = self._density_do_reshuffle()
        if not density_pole_approximation or \
            (not prod_static or prod_static.get('decays_key') != decays_key):
            prod_static = self._density_basis(production, decays_key)
            production._ms_density_static = prod_static

            use_new_mass = (
                not density_pole_approximation or
                density_do_reshuffle
            )
            if use_new_mass:
                new_mass = {}
                reshuffle_info = {}
                for key in decays:
                    new_mass[key] = [getattr(dec[0], 'new_mass', dec[0].mass)
                                     for dec in decays[key]]
                    # Carry the Breit-Wigner sampling info (pole, width, min, max)
                    # alongside new_mass so reshuffle_production can re-sample this
                    # mass if needed. Without it the reshuffling retry crashes on
                    # p.reshuffle_info (e.g. complex-mass-scheme runs).
                    reshuffle_info[key] = [getattr(dec[0], 'reshuffle_info', None)
                                           for dec in decays[key]]

                for particle in production:
                    if particle.status == 1 and particle.pid in new_mass:
                        particle.new_mass = new_mass[particle.pid].pop(0)
                        info = reshuffle_info[particle.pid].pop(0)
                        if info is not None:
                            particle.reshuffle_info = info
            else:
                for particle in production:
                    if hasattr(particle, 'new_mass'):
                        del particle.new_mass
                    if hasattr(particle, 'reshuffle_info'):
                        del particle.reshuffle_info

            #VALENTIN: except for the mode "full", we should not compute the matrix element here
            MEdenom_prod, MEdenom_decay = None, None
            if not density_pole_approximation:
                # compute the denominator and then reshuffle the event before 
                # computing the numerator 
                MEdenom_prod = self.calculate_matrix_element(production)  
                MEdenom_decay = 1.0              
                for key in decays:
                    for dec in decays[key]:
                        MEdenom_decay *= self.calculate_matrix_element(dec)
                # now doing the reshuffling
                # doing the reshuffling for each part:
                jac = 1.0
                jac *= production.reshuffle_production()
                # reshuffle_production may have resampled a resonance's new_mass
                # (its jac==-1 retry) so the set of masses fits sqrt(shat). Push the
                # (possibly resampled) production masses back onto the matching decay
                # so each decay is reshuffled to the SAME mass as its production
                # resonance; otherwise the two diverge and rebuilding the full event
                # later hits an inconsistent-boost assertion. The i-th production
                # resonance of a given pid corresponds to decays[pid][i] (same
                # ordering used when new_mass was assigned above).
                resampled = {key: [] for key in decays}
                for particle in production:
                    if (particle.status == 1 and particle.pid in resampled
                            and hasattr(particle, 'new_mass')):
                        resampled[particle.pid].append(particle.new_mass)
                for key in decays:
                    for dec, mass in zip(decays[key], resampled[key]):
                        dec[0].new_mass = mass
                for key in decays:
                    for dec in decays[key]:
                        jac *= dec.reshuffle_decayevt()
                if jac == 0:
                    raise Exception
                # hand the reshuffling jacobian back to the caller (folded into
                # the accept/reject weight) instead of discarding it.
                jac_reshuffle = jac

        iden_p = prod_static['iden_p']
        sym_factor_prod_ident = prod_static['sym_factor_prod_ident']
        init_part = prod_static['init_part']
        position = prod_static['position']
        helicities = prod_static['helicities']
        allowed_hel = prod_static['allowed_hel']
        ncomb = prod_static['ncomb']
        dimension = prod_static['dimension']

        # ------------------------------------------------------------------
        # Normalization
        # ------------------------------------------------------------------
        dec_diag = 1.0
        prod_color = 1
        prod_denominators = 1
        # GET_INTER returns each density matrix with the standalone matrix
        # element's IDEN division already applied.  The density
        # contraction below predates applies its own normalization, 
        # so restore one IDEN factor per matrix before
        # using that denominator.
        density_iden_prod = iden_p * sym_factor_prod_ident
        density_iden_decay = 1

        # frame the helicity basis is defined in (run_card me_frame); shared by
        # the production and by every decay, otherwise the two sides of the
        # contraction below would not be in the same basis. None unless the
        # beams are polarised -- see the comment above _beampol.
        frame_boost = self._frame_boost(production)

        if prod_density_cached is None:
            density_prod = self.get_density(production,
                                            position,
                                            allowed_hel,
                                            ncomb,
                                            dimension,
                                            frame_boost=frame_boost,
                                            hel_restriction=prod_static.get('hel_restriction'),
                                            hel_restriction_trace=prod_static.get('hel_restriction_trace'))
            # Only on a freshly computed matrix: a cached one was already
            # checked when it was built, and this runs on every joint trial.
            self._check_production_density(production, density_prod,
                                           'joint accept/reject')
        else:
            density_prod = prod_density_cached

        # ------------------------------------------------------------------
        # Symmetry factor:
        # For each parent-PDG group with N identical parents and decay-channel
        # multiplicities {n_k}, the factor that belongs to the denominator is:
        #   sym_group = (Π_k n_k!) / (N!)
        # and sym_factor_decay = Π_groups sym_group.
        # ------------------------------------------------------------------
        sym_factor_decay = 1.0

        # Canonical decay-channel signature: sorted final-state PDGs only.
        def _decay_signature(dec_evt):
            pdgs = []
            for p in dec_evt:
                if p.status == 1:
                    pdgs.append(int(p.pid))
            pdgs.sort()
            return tuple(pdgs)
        
        # ------------------------------------------------------------------
        # Build total decay density matrix as tensor product
        # ------------------------------------------------------------------
        decaying_idx = 0
        density_dec = None

        for pdg, decay_event_list in decays.items():
            N = len(decay_event_list)

            # decay symmetry for this PDG group
            if N > 1:
                # Fast path for N==2 avoids building multiplicity maps.
                if N == 2:
                    if _decay_signature(decay_event_list[0]) != _decay_signature(decay_event_list[1]):
                        sym_factor_decay *= 0.5
                else:
                    sig_counts = {}
                    for evt in decay_event_list:
                        sig = _decay_signature(evt)
                        sig_counts[sig] = sig_counts.get(sig, 0) + 1
                    sym = 1
                    for nk in sig_counts.values():
                        if nk > 1:
                            sym *= math.factorial(nk)
                    sym_factor_decay *= (sym / float(math.factorial(N)))

            # particle properties for this parent PDG
            width = decay_dict[pdg][0]
            mass = decay_dict[pdg][1]
            color = decay_dict[pdg][2]
            spin = decay_dict[pdg][3]

            for i_decay_event in range(N):
                current_decay_event = decay_event_list[i_decay_event]

                # boost to lab frame using corresponding production particle momentum
                part = init_part[decaying_idx + i_decay_event]
                boost = -1 * lhe_parser.FourMomentum(part)
                boost.E *= -1
                current_decay_event.boost(boost)

                density_dec_tmp = self.get_density(
                    current_decay_event,
                    position=[1],
                    allow_hel=helicities[decaying_idx + i_decay_event],
                    ncomb=len(helicities[decaying_idx + i_decay_event]),
                    dimension=len(helicities[decaying_idx + i_decay_event]),
                    frame_boost=frame_boost,
                    frame_rest_leg=None if frame_boost is None
                                   else self._decay_frame_rest_leg(part, frame_boost)
                )

                if density_dec is None:
                    density_dec = density_dec_tmp
                else:
                    density_dec = density_dec.tensor_product(density_dec_tmp)

                if MEdenom_decay is None:
                    dec_diag *= density_dec_tmp.trace().real
                density_iden_decay *= color * spin
                prod_color *= color
                # |D|^2 of the propagator denominator D = i*m*Gamma at the pole.
                # Written as the real number it is: D * conj(D) has the same
                # value bit for bit, but it is a Python *complex*, and that type
                # then rides through `denominator` into the accept/reject weight
                # -- which is the whole reason the weight was ever complex.
                # Nothing else in the chain introduces one: the density
                # contraction below has its real part taken explicitly. Note
                # `mw * mw` and not `mw ** 2`: pow() re-rounds, and the two
                # differ in the last bit for about one value in 700.
                mw = mass * width
                prod_denominators *= mw * mw
            

            decaying_idx += N

        # ------------------------------------------------------------------
        # Contract production and decay density matrices
        # ------------------------------------------------------------------
        me = density_dec.scalar_multiplication(density_prod)
        # keep_weight_for_polarization_*: the same contraction with a tighter
        # row mask, once per combination. Done here, on the matrices that are
        # still alive, and stashed on self rather than added to the return tuple
        # (which every caller unpacks positionally). The joint accept/reject
        # tests the value computed by the last call, so the last ratios are the
        # accepted chain's.
        if self._polarization_weights_enabled():
            self._polarization_ratios(density_prod, density_dec, prod_static,
                                      full=me)
        # pure interference: the same contraction with the *symmetric* (trace)
        # restriction in place of the cross one, i.e. the convolution an
        # ordinary run would have used. Its decay-phase-space mean is the
        # constant c the fully weighted output divides by (section 13.13).
        # Probed only while the maximum-weight scan is measuring c, so the
        # event loop pays nothing for it.
        me_unrestricted = None
        if getattr(self, '_pi_probe_c', False):
            me_unrestricted = self._pi_unrestricted_contraction(density_prod,
                                                                density_dec)
        me *= density_iden_prod * density_iden_decay

        # ------------------------------------------------------------------
        # include production identical-final-state symmetry factor
        # ------------------------------------------------------------------
        denominator = iden_p * sym_factor_prod_ident * prod_color * prod_denominators * sym_factor_decay
        me = ms_density_real(me, 'the production/decay density contraction')/ denominator
        if me_unrestricted is not None:
            me_unrestricted = me_unrestricted * density_iden_prod * density_iden_decay
            self._pi_unrestricted_me = float(
                getattr(me_unrestricted, 'real', me_unrestricted)) / denominator
            # the analytic candidate for c, kept beside the measurement as a
            # cross-check: <W_full> = <jac> / (prod_denominators *
            # sym_factor_decay), so 1/prod_denominators is exact only where the
            # chain carries no reshuffling jacobian and no decay symmetry factor
            self._pi_analytic_c = 1.0 / float(
                getattr(prod_denominators, 'real', prod_denominators)
                * sym_factor_decay)

        #print(f"production = {production}")
        #print(f"decays = {decays}")
        if MEdenom_prod is None:
            prod_diag = density_prod.trace().real 
        else: 
            prod_diag = MEdenom_prod
        if MEdenom_decay is not None:
            dec_diag *= MEdenom_decay
        return me, density_prod, prod_diag, dec_diag, jac_reshuffle


    def get_density_matrix_indices(self, nhel_decay):
        #print("------")
        #print(f"get_density_matrix_indices , nhel_decay = {nhel_decay}")
        diag = [sum(range(nhel_decay, nhel_decay - i, -1)) for i in range(nhel_decay)]
        off_diag = [i for i in list(range(nhel_decay * (nhel_decay + 1) // 2)) if i not in diag]
        return diag, off_diag

    def get_density_matrix_element_from_label(matrix, label):
        if label in label_to_index:
            i, j = label_to_index[label]
            return matrix[i, j]
        else:
            raise ValueError(f"Label {label} is not valid for this matrix size: {matrix.shape}.")

    def get_allowed_hel(self, list_hels):
        # list_hels is a list of lists with all possible helicities of the decaying particles, e.g.
        # [[1,-1], [1,0,-1]] - we need to construct a list of lists with all possible helicities
        # [ [1,1] , [1,0], [1,-1], [-1,1], ... ] which should eventually be converted into a flat list
        # [ 1, 1, 1, 0, 1, -1, ... ]
        key = tuple(tuple(hels) for hels in list_hels)
        # Cache allowed helicities - they depend only on spins
        # avoid rebuilding allowed helicities per trial
        if not hasattr(self, '_allowed_hel_cache'):
            self._allowed_hel_cache = {}
        if key in self._allowed_hel_cache:
            return self._allowed_hel_cache[key]

        helicity_combinations = [list(l) for l in product(*list_hels)]
        concatenated_hel_list = list(chain.from_iterable(helicity_combinations))
        out = (helicity_combinations, concatenated_hel_list)
        self._allowed_hel_cache[key] = out
        return out  

    def _beampol(self):
        """The (pol1, pol2) actually in force, or None for unpolarised beams.

        |beampol| runs from 1 (unpolarised) to 2 (fully polarised), so anything
        at or below 1 means no polarisation -- the same test the matrix elements
        make, so that an out-of-range value cannot switch on the frame boost
        here while the Fortran ignores it.
        """
        pol = self.options.beampol_me()
        if abs(pol[0]) <= 1. and abs(pol[1]) <= 1.:
            return None
        return pol

    def _set_f2py_beampol(self, mymod):
        """Push the beam polarisation into the matrix-element library once per
        module. The value is constant over a run and ``get_density`` is on the
        hot path, so this is a setter rather than a per-call argument."""
        pol = self._beampol()
        if pol is None:
            # the library defaults to unpolarised (BLOCK DATA BEAMPOL_DEFAULT)
            return
        if not hasattr(mymod, 'py_set_beampol'):
            logger.warning('The matrix elements of this MadSpin run predate the '
                           'beam-polarisation support of the density modes; '
                           'beampol=%s will be ignored. Regenerate the process '
                           'directory to enable it.', list(pol))
            return
        mymod.py_set_beampol(pol[0], pol[1])

    def _frame_boost(self, event):
        """The 4-momentum whose rest frame ``frame_id`` selects for ``event``,
        or None when the frame machinery cannot change anything.

        ``frame_id`` is the bitmask the run_card builds as
        ``sum(2**n for n in me_frame)``, so external leg n (counted from 1, in
        the matrix element's own ordering) is selected by bit n -- the same
        convention ``mapid`` uncompresses with ``btest(id, i)``. The returned
        momentum is the sum of the selected legs, ready to be handed to
        ``Event.boost`` / ``_boost_momenta``, which negate the spatial part
        themselves (HELAS ``boostx``, exactly what ``boost_to_frame`` does in
        driver.f).

        Three things switch it on -- the three clauses of ``_needs_frame_axis``
        -- and all of them are cases where the frame is *observable*:
        Three things switch it on, and all are cases where the frame is
        *observable*:

        - polarised beams. ``beampol`` reweights the initial-state helicity sum
          and that sum is quantised along the frame's axis.
        - a polarisation brace on a final-state particle of the production
          process (``p p > w+{0} w-``). MG5 defines those braces in the
          ``me_frame`` frame -- MadEvent evaluates the polarised matrix element
          there (``auto_dsig_v4.inc``, ``boost_to_frame``; the default
          ``me_frame=[1,2]`` is skipped only because ``genps.f`` already builds
          its momenta in the partonic CM and ``unwgt.f`` boosts to the lab on
          the way out), and MadSpin's own v1 driver does the same
          (``boost_to_frame`` in driver.f, unconditionally). The density modes
          apply that brace as a *projection* on rho_prod
          (``set_hel_restriction``), and a projection does not commute with the
          change of helicity basis a boost induces, so leaving the momenta in
          the lab would restrict a different helicity than the one the input
          events were generated with.
        - a polarisation-weight request
          (``keep_weight_for_polarization_vector`` / ``_fermion``). Those
          weights go through the very same ``set_hel_restriction`` projection,
          only to build an extra <wgt> line instead of the nominal weight, so
          they need the frame for exactly the same reason -- and they can be
          asked for on a production that carries no brace at all, which is why
          the two clauses above do not cover them.
        - the pure-interference mode (``set pure_interference t = 0 T``). Its
          cross restriction is a projection for exactly the same reason -- it
          names two helicity sets, which only means something once the axis is
          fixed -- but its production process is *unpolarised*, so the brace
          test above finds nothing and would leave the momenta in the lab. The
          mode has to state the axis for itself.

        Everything else stays in the lab, which keeps unpolarised density runs
        bit-for-bit unchanged: there the full double sum
        ``sum_ij rho_prod(i,j) rho_dec(i,j)`` is a trace, and a boost acts on it
        as a unitary change of basis that cancels between the two factors.
        """
        if not self._needs_frame_axis():
            return None
        frame_id = int(self.options['frame_id'])
        if frame_id <= 0:
            return None
        _, orig_order, _, _, _ = self.get_pdir(event)
        momenta = event.get_momenta(orig_order)
        selected = [n for n in range(1, len(momenta) + 1) if frame_id >> n & 1]
        if not selected:
            return None
        pboost = lhe_parser.FourMomentum()
        for n in selected:
            pboost += lhe_parser.FourMomentum(momenta[n - 1])
        # A single selected leg has to end up exactly at rest: vxxxxx branches
        # on pp.eq.rZero and takes the frame z axis as quantisation axis there,
        # so a residual 1d-14 three-momentum left by the boost arithmetic would
        # silently pick a different polarisation state (see the same fix in
        # boost_to_frame, Template/LO/SubProcesses/genps.f).
        if len(selected) == 1:
            pboost.rest_leg = selected[0]
            mom = momenta[selected[0] - 1]
            pboost.rest_leg_mom = (mom[0], mom[1], mom[2], mom[3])
        else:
            pboost.rest_leg = None
            pboost.rest_leg_mom = None
        return pboost

    @staticmethod
    def _decay_frame_rest_leg(parent, frame_boost):
        """1 when ``frame_id`` selects exactly this resonance, so leg 1 of its
        decay matrix element has to be forced to zero three-momentum; None
        otherwise. Same rounding argument as in ``_frame_boost``."""
        rest_mom = getattr(frame_boost, 'rest_leg_mom', None)
        if rest_mom == (parent.E, parent.px, parent.py, parent.pz):
            return 1
        return None

    @staticmethod
    def _boost_momenta(momenta, pboost, rest_leg=-1):
        """``boost_to_frame``: every momentum of ``momenta`` into the rest frame
        of ``pboost``, as (E, px, py, pz) tuples.

        This works on the momenta rather than on the event, so a decay event
        stays where the rest of MadSpin needs it -- in the lab, which is what
        ``add_decays`` and the reshuffling assume. ``rest_leg`` (1-based, -1 to
        take it from ``pboost``) is the leg the frame is built from when it is a
        single one, forced exactly at rest.
        """
        neg = lhe_parser.FourMomentum(pboost.E, -pboost.px, -pboost.py, -pboost.pz)
        out = []
        for mom in momenta:
            new = lhe_parser.FourMomentum(mom).boost(neg)
            out.append((new.E, new.px, new.py, new.pz))
        if rest_leg == -1:
            rest_leg = getattr(pboost, 'rest_leg', None)
        if rest_leg is not None and rest_leg <= len(out):
            out[rest_leg - 1] = (out[rest_leg - 1][0], 0., 0., 0.)
        return out

    def get_density(self, event, position, allow_hel, ncomb, dimension,
                    frame_boost=None, frame_rest_leg=-1, hel_restriction=None,
                    hel_restriction_trace=None):
        """``frame_boost`` is the momentum whose rest frame ``frame_id`` picks
        (see ``_frame_boost``); the momenta are boosted there before the matrix
        element sees them, which is what defines the axis the initial-state
        helicities -- the ones ``beampol`` reweights -- are quantised along.

        The *same* momentum is used for the production and for every decay
        contracted against it. A decay event reaches this point already boosted
        into the lab (by its parent's momentum), so applying the frame boost to
        its momenta here composes the two in the right order and leaves both
        sides of the contraction in one helicity basis. ``frame_rest_leg``
        names the leg to force exactly at rest; the default takes it from
        ``frame_boost``, which is right for a production event, and the decay
        callers pass ``_decay_frame_rest_leg``'s answer instead.
        """

        orig_order = getattr(event, '_ms_orig_order_for_density', None)
        if orig_order is None:
            _, orig_order, _, _, tag = self.get_pdir(event)
            event._ms_orig_order_for_density = orig_order
        else: #in any case, we need tag to differentiate between production and decay
            tag, _ = event.get_tag_and_order()


        # Fast path: single-point momentum extraction without permutation construction.
        try:
            p = event.get_momenta(orig_order)
        except Exception:
            # Safety fallback for unusual event structures.
            all_p = event.get_all_momenta(orig_order)
            assert len(all_p) == 1, "Error: get_density can only be called for a single phase-space point"
            p = all_p[0]
        if frame_boost is not None:
            p = self._boost_momenta(p, frame_boost, rest_leg=frame_rest_leg)
        P = rwgt_interface.ReweightInterface.invert_momenta(p) 
        pdgs =list(orig_order[0])+list(orig_order[1])
        n_changing = len(position)
        if n_changing == 0:
            raise ValueError("Error in get_density: 'position' must contain at least one position index")
        if len(allow_hel) % n_changing != 0:
            raise ValueError("Error in get_density: inconsistent 'allow_hel' and 'position' lengths")
        
        # PY_GET_DENSITY(PDGS, PROCID, P, POS, ALLOW_HEL, ALPHAS, SCALE2)
        if self.all_me[tag]['type'] == 'production':
            # misc.sprint("Computation of the production density matrix")
            density_array = self.f2py_module[0].py_get_density(pdgs=pdgs, 
                                                                procid=-1, 
                                                                p=P, 
                                                                pos=position, 
                                                                allow_hel=allow_hel, 
                                                                alphas=event.aqcd,
                                                                scale2=event.scale**2)

        elif self.all_me[tag]['type'] == 'decay':
            # misc.sprint("Computation of the decay density matrix")
            density_array = self.f2py_module[1].py_get_density(pdgs=pdgs, 
                                                                procid=-1, 
                                                                p=P, 
                                                                pos=position, 
                                                                allow_hel=allow_hel, 
                                                                alphas=event.aqcd,
                                                                scale2=event.scale**2)
        else:
            raise ValueError("The key 'type' of sel.all_me can only take as values 'production' or 'decay'.")


        #print(f"density_array = {density_array}") 
        density_matrix = madspin.DensityMatrix(density_array,
                                               n_changing,
                                               allow_hel,
                                               dimension)
        # production polarisation braces: the restriction travels with the
        # matrix, so every later contraction/trace applies it (see
        # DensityMatrix.set_hel_restriction). None for the decay densities.
        if hel_restriction is not None:
            density_matrix.set_hel_restriction(hel_restriction)
            # pure-interference mode only: the contraction runs over the
            # interference block while trace()/normalized() keep using the
            # production trace, which is the unrestricted one for the
            # unpolarised production the mode requires (section 13.4).
            if hel_restriction_trace is not None:
                density_matrix.set_hel_restriction_trace(hel_restriction_trace)
        return density_matrix


    # ``get_inter_value``/``get_nhel``/``get_mymod`` used to live here: the
    # per-helicity interference loop of the original density prototype. Their
    # last callers went away in 23409c526 (2024-08, "Caching production and
    # decay ME using diagonal elements of density matrix") and they were
    # unreachable ever since -- they assume the pre-e16ac171b single-module
    # layout (``f2py_module`` a module, ``pdg2prefix`` a flat dict) while
    # calling ``get_pdir``, which only works with the current two-module one.
    # Removed rather than left to rot; the interference now comes from
    # ``py_get_density`` (see ``get_density``).

    def get_iden(self, event):
        # DEBUGGING REMOVE
        #print("---- DEBUG ---")
        #pdgs, allproc = self.f2py_module.get_pdg_order()
        #idens = self.f2py_module.get_idens()
        #
        #print("len(pdgs) =", len(pdgs))
        #print("len(idens) =", len(idens))
        #   
        #for i in range(len(idens)):
        #    print(i, pdgs[i], idens[i])
        #print("--- END")
        # END REMOVE

        # get_pdir returns (pdir, orig_order, prefix, pos, tag)
        _, _, _, pos, tag = self.get_pdir(event)

        if self.all_me[tag]['type'] == 'production':
            idens = self.f2py_module[0].get_idens()
        elif self.all_me[tag]['type'] == 'decay':
            idens = self.f2py_module[1].get_idens()
        else:
            raise ValueError("The key 'type' of self.all_me can only take as values 'production' or 'decay'.")

        #print(f"idens = {idens} , pos = {pos}")
        return idens[pos]


    def get_pdir(self,event): 
        tag, order = event.get_tag_and_order()
#        print(order)
        try:
            orig_order = self.all_me[tag]['order']
        except Exception:
            # try to pass to full anti-particles for 1->N
            init, final = tag
            if len(init) == 2:
                raise
            init = (-init[0],)
            final = tuple(-i for i in final)
            tag = (init, final)
            orig_order = self.all_me[tag]['order']
        pdir = self.all_me[tag]['pdir']

        if self.all_me[tag]['type'] == 'production':
            prefix, pos = self.pdg2prefix[0][tuple(list(orig_order[0]) + list(orig_order[1]))]
        elif self.all_me[tag]['type'] == 'decay':
            prefix, pos = self.pdg2prefix[1][tuple(list(orig_order[0]) + list(orig_order[1]))]
        else:
            raise ValueError("The key 'type' of self.all_me can only take as values 'production' or 'decay'.")
        #misc.sprint(f"get_pdir: pdir = {pdir} , orig_order = {orig_order} , prefix = {prefix}")
        return pdir,orig_order, prefix, pos, tag

    # Two model_init are used, one for the production and one the decay (to support LO decay + NLO production or different models for each side)
    model_init_prod = True
    model_init_decay = True
    model_init = True
    def calculate_matrix_element(self, event):
        """routine to return the matrix element"""        
        
        tag, order = event.get_tag_and_order()
        try:
            orig_order = self.all_me[tag]['order']
        except Exception:
            # try to pass to full anti-particles for 1->N
            init, final = tag
            if len(init) == 2:
                raise
            init = (-init[0],)
            final = tuple(-i for i in final)
            tag = (init, final)
            orig_order = self.all_me[tag]['order']
        pdir = self.all_me[tag]['pdir']
        if pdir in self.all_f2py:
            all_p = event.get_all_momenta(orig_order)
            #print(f"len identical = {len(all_p)} , p = {all_p}")
            if self.options['identical_particle_in_prod_and_decay'] == "crash" and\
                len(all_p)> 1:
                raise Exception("Ambiguous particle in production and decay. crash as requested by 'identical_particle_in_prod_and_decay'")
            out = 0
            for p in all_p:
                #print(f"Before : {p}")
                p = rwgt_interface.ReweightInterface.invert_momenta(p)
                #print(f"After : {p}")
                #print(event[0])
                #print(event[0].color1)
                #print(event.aqcd)
                if event[0].color1 == 599 and event.aqcd==0:
                    new_value = self.all_f2py[pdir](p, 0.113, 0)
                else:
                    new_value = self.all_f2py[pdir](p, event.aqcd, event.scale, -1)
                #if the process is Loop-Induced, smatrixhel returns the tuple (value, returncode), we need to keep only the value
                if isinstance(new_value, tuple):
                    new_value = new_value[0]
                if self.options['identical_particle_in_prod_and_decay'] == "average":
                    out += new_value
                else:
                    if abs(out)< abs(new_value):
                        out = new_value
                if self.options['identical_particle_in_prod_and_decay'] == 'first':
                    return out
            if self.options['identical_particle_in_prod_and_decay'] == "average":
                return out/len(all_p)
            else:
                return out
        else:            
            # First time we see a new ``pdir`` for this MadSpin instance:
            # load the freshly-compiled f2py extension once and cache a
            # smatrixhel lambda per pdir. The .so / pdg2prefix only need
            # to be loaded the first time we hit this branch — subsequent
            # ``pdir``s reuse ``self.f2py_module`` so we don't re-trigger
            # the spec-from-file-location load (which is fine on Linux
            # but wasteful, and on macOS would re-walk the install_name
            # bookkeeping every time).

            # Valentin: we only pass here with the options 'onshell_v1' and 'madspin_v1' for which I kept only one madspin_me directory
            # it is not adapted to modes where the directories for production and decays are separated
            if not hasattr(self, 'f2py_module'):
                sp_path = pjoin(self.path_me, self.ms_me_subdir, 'SubProcesses')
                if sys.path[0] != sp_path:
                    sys.path.insert(0, sp_path)

                mymod = self._load_f2py_matrix_module(sp_path)
                self.f2py_module = mymod

                all_prefix = self.f2py_module.get_prefix()
                all_pdg, all_procid = self.f2py_module.get_pdg_order()
                self.pdg2prefix = {}
                for i, pdg in enumerate(all_pdg):
                    pdg = tuple([x for x in pdg if x != 0])
                    self.pdg2prefix[tuple(pdg)] = (str(all_prefix[i].decode()).strip(), i)

                if self.model_init:
                    self.model_init = False
                    with misc.chdir(sp_path):
                        # Same single source of truth as the density modes:
                        # never the process directory's own Cards/param_card.dat,
                        # which path_me points at under MadEvent and which can
                        # disagree with the events (see me_param_card).
                        mymod.initialise(self.me_param_card(self.ms_me_subdir))
            mymod = self.f2py_module

            #if Rpath linking is not working the below code can be an alternative:
            #import ctypes
            #exts = ['so','dylib','dll']
            #for ext in exts:
            #    me_library = pjoin(self.path_me, self.ms_me_subdir, 'SubProcesses', pdir, 'libme%s.%s' % (pdir, ext))
            #    if os.path.exists(me_library):
            #        break
            # ctypes.CDLL(me_library)

            pdg = list(orig_order[0]) + list(orig_order[1])

            if self.options['spinmode'] in ['onshell_v1', 'madspin_v1']:
                self.all_f2py[pdir] = lambda *args : mymod.smatrixhel(pdg, 0, *args)
            else:
                if self.all_me[tag]['type'] == 'production':
                    self.all_f2py[pdir] = lambda *args : mymod[0].smatrixhel(pdg, 0, *args)
                elif self.all_me[tag]['type'] == 'decay':
                    self.all_f2py[pdir] = lambda *args : mymod[1].smatrixhel(pdg, 0, *args)
                else:
                    raise ValueError("The key 'type' of sel.all_me can only take as values 'production' or 'decay'.")

            return self.calculate_matrix_element(event)
        
        
    def generate_all_matrix_element(self):
        
        # 1. compute the production matrix element -----------------------------
        processes = [line[9:].strip() for line in self.banner.proc_card
                     if line.startswith('generate')]
        processes += [' '.join(line.split()[2:]) for line in self.banner.proc_card
                      if re.search(r'^\s*add\s+process', line)]
        # 2. compute the decay matrix-element
        decay_text = []
        processes_decay = []
        for decays in self.list_branches.values():
            for decay in  decays:
                if '=' not in decay:
                    decay += ' QCD=99'
                if ',' in decay:
                    decay_text.append('(%s)' % decay)
                else:
                    decay_text.append(decay)
                processes_decay.append(decay)            
        decay_text = ', '.join(decay_text)
        processes += []
        
        #handle NLO
        new_processes = []
        for proc in processes:
            # deal with @ syntax need to move it after the decay specification
            if '@' in proc:
                proc, proc_nb = proc.split('@')
                try:
                    proc_nb = int(proc_nb)
                except ValueError:
                    raise madspin.MadSpinError('MadSpin didn\'t allow order restriction after the @ comment: \"%s\" not valid' % proc_nb)
                proc_nb = '@ %i' % proc_nb 
                if self.options['global_order_coupling']:
                    proc_nb = '%s %s' % (proc_nb, self.options['global_order_coupling'])
            else:
                if self.options['global_order_coupling']:
                    proc_nb = '@0 %s ' % self.options['global_order_coupling']    
                
            rwgt_interface.ReweightInterface.get_LO_definition_from_NLO()        
        
        raise Exception

if __name__ == '__main__':
    
    a = MadSpinInterface()
    a.cmdloop()
    
    


        
