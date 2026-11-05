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
"""Language neutral part of the color flow (JAMP) optimisation.

Every backend writes the same object: the matrix of coefficients giving each
color flow as a combination of the amplitudes,

    JAMP(i) = sum_j A(i,j) * AMP(j)

Written out as it stands that is one line per non zero entry, which for a
multi-gluon process is tens of thousands of them. The search below replaces the
repeated pieces by shared sub-expressions, so that the matrix is left with far
fewer entries and a list of definitions to compute first.

Two searches are here, and optimise_jamp_best picks between them: the plain
greedy scan, which takes whatever sub-expression is worth most at each step,
and the orbit equivariant one, which only takes whole orbits of the
permutations leaving the color basis invariant and so leaves the matrix
invariant at every step.

Nothing here knows about fortran or C++: it takes the coefficient matrix and
gives back the reduced matrix and the definitions. The exporters print that in
their own language (get_JAMP_lines for fortran, build_jamp_plan for the
C++/cudacpp writer), and how they print it is what decides which of the
optimisations they may use -- see jamp_orbit_allowed and
jamp_greedy_tail_enabled.
"""

from __future__ import absolute_import

import bisect
import collections
import fractions
import logging
import time

import madgraph.core.color_amp as color_amp
import madgraph.core.helas_objects as helas_objects
import madgraph.various.banner as banner_mod

logger = logging.getLogger('madgraph.export_v4')


class JampOptimiser(object):
    """The common sub-expression search over the JAMP coefficient matrix.

    Mixed into the exporters, which supply the printing."""

    # Off by default: the plain output of a backend is the expanded one, and
    # each exporter switches this on for itself. 'jamp_optim' in cmd_options
    # (i.e. --jamp_optim=True|False at output time) wins over the class value.
    jamp_optim = False
    # how many times the JAMP optimisation called itself, for the record
    myjamp_count = 0
    # take the power of i shared by every coefficient out before searching, so
    # that the search walks over whole numbers (see optimise_jamp_matrix)
    jamp_integer_walk = True
    # Introduce the sub-expressions by whole orbits of the permutations leaving
    # the color basis invariant instead of one at a time (see
    # optimise_jamp_equivariant). Off by default, each backend switches it on
    # for itself where it measured a gain.
    jamp_orbit = False
    # finish with the plain scan once the orbit rounds have nothing left to
    # take as a whole (see jamp_greedy_tail_enabled)
    jamp_greedy_tail = True
    # up to this many entries in the matrix, both optimisations are run and the
    # shorter result kept (see optimise_jamp_best)
    jamp_compare_max_size = 20000
    # what the orbit rounds did, for an emission which wants to describe the
    # definitions by one recipe per orbit rather than one by one
    jamp_orbits = None

    def jamp_optim_enabled(self):
        """Whether to run the optimisation, --jamp_optim first."""

        cmd_options = getattr(self, 'cmd_options', None) or {}
        if 'jamp_optim' in cmd_options:
            return banner_mod.ConfigFile.format_variable(
                            cmd_options['jamp_optim'], bool, 'jamp_optim')
        return self.jamp_optim

    @staticmethod
    def jamp_matrix(color_amplitudes):
        """The coefficient matrix all_element[(color flow, amplitude)] = value
        of the color amplitudes, color flows numbered from 1 and amplitudes as
        they number themselves. This is the input of the optimisation, and the
        same value the expanded lines are written with."""

        all_element = {}
        # Every single amplitude carries a power of the number of colors in its
        # coefficient, but a process only uses a handful of distinct powers, so
        # build the corresponding fractions once instead of once per amplitude.
        nc_powers = {}
        for i, coeff_list in enumerate(color_amplitudes):
            for (coefficient, amp_number) in coeff_list:
                if not coefficient:
                    continue
                try:
                    nc_power = nc_powers[coefficient[3]]
                except KeyError:
                    nc_power = fractions.Fraction(3)**coefficient[3]
                    nc_powers[coefficient[3]] = nc_power
                value = (1j if coefficient[2] else 1) * \
                        coefficient[0] * coefficient[1] * nc_power
                key = (i + 1, amp_number)
                if key not in all_element:
                    all_element[key] = value
                else:
                    all_element[key] += value
        return all_element

    def jamp_walk_integers(self, all_element):
        """Take the power of i shared by every coefficient out of the matrix
        and return it, leaving all_element with whole numbers -- they compare
        and hash exactly, and nothing has to be widened to complex. Returns
        None when there is no such phase, all_element then being complex
        throughout. The phase goes back on with jamp_apply_phase, so the lines
        written are the same either way."""

        phase = self.jamp_global_phase(all_element) \
                                        if self.jamp_integer_walk else None
        if phase is not None:
            whole = {}
            for key, value in all_element.items():
                number = value / phase if phase != 1 else value
                if isinstance(number, complex):
                    number = number.real
                number = fractions.Fraction(number).limit_denominator(10**9)
                if number.denominator != 1:
                    break
                whole[key] = int(number)
            else:
                all_element.clear()
                all_element.update(whole)
                return phase
        for key in all_element:
            all_element[key] = complex(all_element[key])
        return None

    @staticmethod
    def jamp_apply_phase(new_mat, phase):
        """Put back the phase jamp_walk_integers took out. The definitions hold
        ratios, which it cancels out of; only the coefficients left in the
        matrix carry it."""

        if phase is None or phase == 1:
            return
        for key in new_mat:
            new_mat[key] = new_mat[key] * phase

    def optimise_jamp_matrix(self, all_element, symmetry=None,
                             matrix_element=None):
        """Run the optimisation over the coefficient matrix and return
        (new_mat, defs):
          - defs is a list of (i, op1, op2, frac, nb): definition number i is
            op1 + frac*op2, where a positive operand is an amplitude and a
            negative one is the definition number -op;
          - new_mat is what the matrix is left with, keyed the same way as the
            input except that a negative amplitude index means the definition
            of that number.

        With a matrix element and jamp_orbit_allowed saying so, the color basis
        symmetry is read off the matrix and the sub-expressions are introduced
        by whole orbits of it; the orbits are left in self.jamp_orbits.

        all_element is consumed (the optimisation works in place). The fortran
        exporter runs the three steps itself, since it has its own way of
        finding the matrix element the color amplitudes came from."""

        if len(all_element) > 1000:
            logger.info("Computing Color-Flow optimization [%s term]",
                        len(all_element))
            start_time = time.time()
        else:
            start_time = 0

        self.myjamp_count = 0
        self.jamp_orbits = None
        phase = self.jamp_walk_integers(all_element)
        # the symmetry is read off the matrix once the phase is out of it, so
        # that the columns compare as whole numbers
        if symmetry is None and matrix_element is not None and \
                self.jamp_orbit_allowed(matrix_element):
            symmetry = self.get_jamp_symmetry(matrix_element, all_element)
        new_mat, defs = self.optimise_jamp(all_element, symmetry=symmetry)
        self.jamp_apply_phase(new_mat, phase)
        if start_time:
            logger.info("Color-Flow passed to %s term in %ss. Introduce %i contraction",
                        len(new_mat), int(time.time()-start_time), len(defs))
        return new_mat, defs

    @staticmethod
    def jamp_global_phase(all_element):
        """The power of i every coefficient carries, when they all carry the
        same one. A pure gluon process picks up one factor of i per f^abc, the
        same for every term, so the whole matrix is real or wholly imaginary;
        a quark line mixes the two and there is nothing to take out."""

        phase = None
        for value in all_element.values():
            if not value:
                continue
            number = complex(value)
            if number.imag == 0:
                here = 1
            elif number.real == 0:
                here = 1j
            else:
                return None
            if phase is None:
                phase = here
            elif phase != here:
                return None
        return phase

    @staticmethod
    def index_jamp_matrix(all_element, nb_col):
        """Sorted lists of the positions of the non zero entries of the matrix,
        by line and by column. An entry which is present but zero does not
        count, and neither does a column outside the 0..nb_col range, so that
        these indices list exactly the entries the plain scan would look at."""

        lines = collections.defaultdict(list)
        columns = collections.defaultdict(list)
        for (i, j), value in all_element.items():
            if value and j < nb_col:
                lines[i].append(j)
                columns[j].append(i)
        for line in lines.values():
            line.sort()
        for column in columns.values():
            column.sort()
        return lines, columns

    @staticmethod
    def common_jamp_lines(columns, nb_line, j1, j2):
        """Lines, in increasing order, where both columns j1 and j2 are non
        zero. Both column lists are sorted, so this is a plain merge."""

        left, right = columns.get(j1, []), columns.get(j2, [])
        res = []
        pos1 = pos2 = 0
        while pos1 < len(left) and pos2 < len(right):
            if left[pos1] == right[pos2]:
                if left[pos1] < nb_line:
                    res.append(left[pos1])
                pos1 += 1
                pos2 += 1
            elif left[pos1] < right[pos2]:
                pos1 += 1
            else:
                pos2 += 1
        return res

    def optimise_jamp(self, all_element, nb_line=0, nb_col=0, added=0,
                      symmetry=None):
        """ optimise problem of type Y = A X
                A is a matrix (all_element)
                X is the fortran name of the input.
            The code iteratively add sub-expression jtemp[sub_add]
            and recall itself (this is add to the X size)

            With a symmetry (see get_jamp_symmetry) the sub-expressions are
            introduced by whole orbits of that symmetry instead of one at a
            time, so that the result can be written as one recipe per orbit.
            The orbits are then left in self.jamp_orbits.
        """
        if symmetry:
            return self.optimise_jamp_best(all_element, symmetry)

        self.myjamp_count +=1

        if not nb_line:
            for i,j in all_element:
                if i+1 > nb_line:
                    nb_line = i+1
                if j+1> nb_col:
                    nb_col = j+1
            if nb_col > 600 and added==0:
                all_element1, all_element2 = {}, {}
                for (k1,k2) in all_element:
                    if k2 >= nb_col//2:
                        all_element2[(k1,1+k2-(nb_col//2))] = all_element[(k1,k2)]
                    else:
                        all_element1[(k1,k2)] = all_element[(k1,k2)]

                all_element1, newdef1 = self.optimise_jamp(all_element1)
                nb_added1 = len(newdef1)

                all_element2, newdef2 = self.optimise_jamp(all_element2)

                for (k1,k2) in all_element2:
                    if k2 >= 0:
                        all_element1[(k1,k2+(nb_col//2)-1)] = all_element2[(k1,k2)]
                    if k2 < 0:
                        all_element1[(k1,k2-nb_added1)] = all_element2[(k1,k2)]
                # new_def format: added,j1,j2,R, max_count
                for k, j1,j2, R, c in newdef2:
                    if j2 > 0:
                        k2 = j2+nb_col//2 -1
                    else:
                        k2 = j2-nb_added1
                    if j1 > 0:
                        k1 = j1+nb_col//2 -1
                    else:
                        k1 = j1-nb_added1
                    newdef1.append((k+nb_added1, k1, k2, R, c))
                if newdef1:
                    all_element, new_def = self.optimise_jamp(all_element1, nb_line=0, nb_col=0, added=len(newdef1))
                    newdef1 = newdef1 + new_def
                return all_element, newdef1

        # Index of the non zero entries, by line and by column. The matrix is
        # very sparse (a color flow only gets a small share of the amplitudes)
        # so walking the whole 0..nb_col range for every entry, as looking the
        # columns up one by one in the matrix amounts to, spends nearly all of
        # its time discovering zeros.
        lines, columns = self.index_jamp_matrix(all_element, nb_col)

        max_count = 0
        all_index = []
        # how many lines have the same ratio between two given columns, keyed
        # by the two columns and the ratio at once rather than by nested
        # dictionaries: this is the innermost loop of the whole optimisation
        operation = collections.defaultdict(int)
        for (i,j1), v1 in all_element.items():
            line = lines.get(i)
            if not line:
                continue
            for j2 in line[bisect.bisect_right(line, j1):]:
                key = (j1, j2, all_element[(i,j2)]/v1)
                operation[key] += 1
                count = operation[key]
                if count > max_count:
                    max_count = count
                    all_index = [key]
                elif count == max_count:
                    all_index.append(key)

        if max_count <= 1:
            return all_element, []

        to_add = []
        for index in all_index:
            j1,j2,R = index
            first = True
            # only the lines where both columns are filled can contribute; the
            # substitutions done here can empty some of them, so the values
            # still have to be read back from the matrix
            for i in self.common_jamp_lines(columns, nb_line, j1, j2):
                v1 = all_element.get((i,j1), 0)
                v2 = all_element.get((i,j2), 0)
                if not v1 or not v2:
                    continue
                if v2/v1 == R:
                    if first:
                        first = False
                        added +=1
                        to_add.append((added,j1,j2,R, max_count))

                    all_element[(i,-added)] = v1
                    del all_element[(i,j1)] #= 0
                    del all_element[(i,j2)] #= 0

        logger.log(5,"Define %d new shortcut reused %d times", len(to_add), max_count)
        new_element, new_def =  self.optimise_jamp(all_element, nb_line=nb_line, nb_col=nb_col, added=added)
        for one_def in to_add:
            new_def.insert(0, one_def)
        return new_element, new_def

    @staticmethod
    def jamp_operation_count(new_mat, defs):
        """Additions the result asks for: one per definition, plus what is left
        in each line of the matrix."""

        terms = collections.Counter()
        for jamp, _var in new_mat:
            terms[jamp] += 1
        return len(defs) + sum(max(0, count - 1) for count in terms.values())

    @staticmethod
    def jamp_definition_order(defs):
        """The definitions in the order they can be computed while the
        amplitudes are produced one at a time, together with the amplitude each
        one is ready after.

        Returns (order, ready): order lists the definition numbers, ready maps
        a definition number onto the last amplitude it needs (transitively).
        Sorting by that amplitude keeps the list topological, since a
        definition never needs fewer amplitudes than the ones it is built
        from."""

        ready = {}
        rank = {}
        for position, (i, amp1, amp2, _frac, _nb) in enumerate(defs):
            last = 0
            for amp in (amp1, amp2):
                last = max(last, amp if amp > 0 else ready[-amp])
            ready[i] = last
            rank[i] = position
        order = sorted(ready, key=lambda i: (ready[i], rank[i]))
        return order, ready

    def optimise_jamp_best(self, all_element, symmetry):
        """Taking whole orbits only pays once there is enough of them to share:
        on a small matrix it can end up asking for more additions than the plain
        scan, which is free to take whatever it likes. g g > t t~ g is such a
        case, 46 additions against 39.

        Small matrices are cheap to optimise, so rather than guess where the
        turn is, do both and keep the shorter. Above that size only the orbit
        version is run: it wins by a wide margin on everything that big, and
        the plain scan is the slow one there."""

        orbit_element, orbit_defs = self.optimise_jamp_equivariant(
                                            dict(all_element), symmetry)
        if len(all_element) > self.jamp_compare_max_size:
            return orbit_element, orbit_defs

        orbits = self.jamp_orbits
        plain_element, plain_defs = self.optimise_jamp(dict(all_element))
        if self.jamp_operation_count(plain_element, plain_defs) < \
                    self.jamp_operation_count(orbit_element, orbit_defs):
            self.jamp_orbits = None
            return plain_element, plain_defs
        self.jamp_orbits = orbits
        return orbit_element, orbit_defs

    def jamp_orbit_allowed(self, matrix_element):
        """Whether the orbit equivariant optimisation is used here,
        --jamp_orbit first. A backend which only accepts it for some of the
        templates it writes says so by overriding this."""

        cmd_options = getattr(self, 'cmd_options', None) or {}
        if 'jamp_orbit' in cmd_options:
            return banner_mod.ConfigFile.format_variable(
                            cmd_options['jamp_orbit'], bool, 'jamp_orbit')
        return self.jamp_orbit

    def jamp_greedy_tail_enabled(self):
        """Whether the orbit rounds are finished off by the plain scan. What
        the tail adds are ordinary sub-expressions, not orbits of anything, so
        an emission which rebuilds the definitions from one recipe per orbit
        cannot describe them; one which writes them down can."""

        return self.jamp_greedy_tail

    #===========================================================================
    # Orbit equivariant version of the JAMP optimisation
    #===========================================================================
    # A permutation of the external color indices which maps the color basis
    # onto itself (see color_amp.ColorBasisSymmetry) also permutes the columns
    # of the JAMP matrix, up to a sign. The whole matrix is then invariant, so
    # the sub-expressions the optimisation looks for come in orbits: every one
    # of them is worth exactly as much as the others. Introducing a whole orbit
    # at a time, rather than one sub-expression at a time as the plain scan
    # does, leaves the matrix invariant at every step, and the definitions can
    # be written as one recipe per orbit.

    @staticmethod
    def jamp_column_form(column):
        """Canonical form of one column of the JAMP matrix up to a global sign,
        together with the sign which was taken out."""

        entries = sorted(column.items())
        first = entries[0][1]
        sign = -1 if (first.real, first.imag) < (0., 0.) else 1
        return tuple((i, sign * value) for i, value in entries), sign

    @classmethod
    def jamp_amp_permutation(cls, columns, induced):
        """Permutation of the amplitudes induced by the permutation induced of
        the color basis: return {amp: (amp, sign)} such that

            M[induced[i], sigma(j)] = sign(j) * M[i, j]

        or None if the columns are not mapped onto each other.

        Several amplitudes often have the very same column, so the columns are
        gathered by their canonical form and one target is taken out of each
        group at a time: looking the image up would not give a bijection."""

        groups = collections.defaultdict(collections.deque)
        for j in sorted(columns):
            form, sign = cls.jamp_column_form(columns[j])
            groups[form].append((j, sign))

        action = {}
        for j in sorted(columns):
            image = dict((induced[i - 1] + 1, value)
                         for i, value in columns[j].items())
            form, sign = cls.jamp_column_form(image)
            group = groups.get(form)
            if not group:
                return None
            target, target_sign = group.popleft()
            factor = sign * target_sign
            other = columns[target]
            if len(other) != len(image) or \
                 any(other.get(i) != factor * value
                     for i, value in image.items()):
                return None
            action[j] = (target, factor)
        return action

    def get_jamp_symmetry(self, matrix_element, all_element):
        """Permutations leaving the JAMP matrix invariant: for each of them the
        permutation of the color basis lines, and the permutation of the
        amplitude columns with the sign that goes with it. None when there is
        none, or when the matrix element does not carry a color basis."""

        if not isinstance(matrix_element, helas_objects.HelasMatrixElement):
            return None
        color_basis = matrix_element.get('color_basis')
        if not color_basis or len(color_basis) < 2:
            return None
        symmetry = color_amp.ColorBasisSymmetry(sorted(color_basis.keys()))
        if not symmetry.generators1:
            return None

        columns = collections.defaultdict(dict)
        for (i, j), value in all_element.items():
            if value:
                columns[j][i] = value
        if not columns:
            return None

        nb_line = len(symmetry.keys1)
        rowperms, actions = [], []
        for induced in symmetry.generators1:
            action = self.jamp_amp_permutation(columns, induced)
            if action is None:
                continue
            rowperms.append([0] + [induced[i] + 1 for i in range(nb_line)])
            actions.append(action)
        if not actions:
            return None

        # one line per orbit is enough to see every sub-expression: any other
        # line is the image of one of them, and so are the sub-expressions it
        # holds. This is what keeps the scan below from being quadratic in the
        # number of terms of the whole matrix.
        parent = list(range(nb_line + 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for rowperm in rowperms:
            for i in range(1, nb_line + 1):
                ri, rj = find(i), find(rowperm[i])
                if ri != rj:
                    parent[ri] = rj
        line_reps = [i for i in range(1, nb_line + 1) if find(i) == i]

        return {'rowperms': rowperms, 'actions': actions,
                'nb_line': nb_line, 'line_reps': line_reps}

    @staticmethod
    def jamp_operation_image(action, operation):
        """Image of the sub-expression operation=(j1,j2,R) under one
        permutation, and the factor relating the column the image defines to
        the image of the column operation defines."""

        j1, j2, ratio = operation
        first, sign1 = action[j1]
        second, sign2 = action[j2]
        if first < second:
            return (first, second, ratio * sign2 / sign1), sign1
        return (second, first, sign1 / (sign2 * ratio)), sign2 * ratio

    def optimise_jamp_equivariant(self, all_element, symmetry):
        """Same optimisation as optimise_jamp, but introducing whole orbits of
        sub-expressions at a time so that the result is closed under the
        symmetry. Fills self.jamp_orbits with, for every definition, the orbit
        it belongs to and the definition and permutation it comes from."""

        actions = [dict(action) for action in symmetry['actions']]
        line_reps = symmetry['line_reps']
        added = 0
        defs = []
        # (orbit, parent definition, permutation) for every definition
        tree = []
        # the definitions introduced together: none of them uses another, so
        # they can be reordered freely
        levels = []
        nb_orbit = 0

        while True:
            columns = collections.defaultdict(list)
            lines = collections.defaultdict(list)
            for (i, j), value in all_element.items():
                if value:
                    columns[j].append(i)
                    lines[i].append(j)
            for line in lines.values():
                line.sort()

            # every sub-expression is the image of one living on a
            # representative line, so only those have to be looked at
            candidates = set()
            for i in line_reps:
                line = lines.get(i, [])
                for pos, j1 in enumerate(line):
                    value = all_element[(i, j1)]
                    for j2 in line[pos + 1:]:
                        candidates.add((j1, j2, all_element[(i, j2)] / value))

            max_count = 0
            best = []
            for operation in candidates:
                count = len(self.jamp_operation_lines(all_element, columns,
                                                      operation))
                if count > max_count:
                    max_count, best = count, [operation]
                elif count == max_count:
                    best.append(operation)
            if max_count <= 1:
                break

            orbits = self.jamp_operation_orbits(actions, best)
            first_of_level = added + 1
            for orbit, parent in orbits:
                rows = dict((operation,
                             self.jamp_operation_lines(all_element, columns,
                                                       operation))
                            for operation in orbit)
                if not self.jamp_orbit_usable(rows):
                    continue
                index = {}
                for operation in orbit:
                    added += 1
                    index[operation] = added
                    origin, permutation = parent[operation]
                    tree.append((nb_orbit, index[origin] if origin else 0,
                                 permutation))
                    defs.append((added, operation[0], operation[1],
                                 operation[2], len(rows[operation])))
                nb_orbit += 1
                for operation, new in index.items():
                    j1, j2 = operation[0], operation[1]
                    for i in rows[operation]:
                        all_element[(i, -new)] = all_element[(i, j1)]
                        del all_element[(i, j1)]
                        del all_element[(i, j2)]
                for action in actions:
                    for operation, new in index.items():
                        image, factor = self.jamp_operation_image(action,
                                                                  operation)
                        action[-new] = (-index[image], factor)
            if added < first_of_level:
                # nothing could be introduced as a whole orbit
                break
            levels.append((first_of_level, added))
            logger.log(5, "Define %d new shortcut reused %d times",
                       added - first_of_level + 1, max_count)

        self.jamp_orbits = {'tree': tree, 'nb_orbit': nb_orbit,
                            'levels': levels, 'actions': actions,
                            'symmetry': symmetry}

        if self.jamp_greedy_tail_enabled():
            # The orbit rounds stop while the JAMP lines still hold a good many
            # terms, since an orbit can only be taken as a whole. The plain
            # scan has no such scruple and can still shorten those lines. Its
            # sub-expressions are not orbits of anything, so an emission which
            # rebuilds them from the recipes cannot have them (see
            # jamp_greedy_tail_enabled), but one writing them down does not
            # care.
            all_element, tail = self.optimise_jamp(all_element, added=added)
            defs.extend(tail)

        return all_element, defs

    @staticmethod
    def jamp_operation_lines(all_element, columns, operation):
        """Lines where both columns of the sub-expression are still there with
        its ratio. The values are read from the matrix as it is now, so lines
        already taken by an orbit introduced before are simply gone."""

        j1, j2, ratio = operation
        res = []
        for i in columns.get(j1, ()):
            value = all_element.get((i, j1), 0)
            if not value:
                continue
            other = all_element.get((i, j2), 0)
            if other and other / value == ratio:
                res.append(i)
        return res

    def jamp_operation_orbits(self, actions, operations):
        """Orbits of the sub-expressions, walked breadth first, with the
        (sub-expression, permutation) each of them is reached from."""

        seen = set()
        orbits = []
        for start in sorted(operations, key=lambda op: (op[0], op[1],
                                                        op[2].real,
                                                        op[2].imag)):
            if start in seen:
                continue
            orbit, parent = [start], {start: (None, 0)}
            seen.add(start)
            queue = collections.deque([start])
            while queue:
                current = queue.popleft()
                for position, action in enumerate(actions):
                    image = self.jamp_operation_image(action, current)[0]
                    if image in seen:
                        continue
                    seen.add(image)
                    parent[image] = (current, position + 1)
                    orbit.append(image)
                    queue.append(image)
            orbits.append((orbit, parent))
        return orbits

    @staticmethod
    def jamp_i_power(factor):
        """The exponent of i this factor is, or None when it is not one of the
        four powers of i. The factors the optimisation produces are products of
        signs and of the i the color coefficients carry, so this is what they
        all are in practice."""

        value = complex(factor)
        for exponent, power in enumerate((1, 1j, -1, -1j)):
            if value == power:
                return exponent
        return None

    @staticmethod
    def jamp_definition_levels(defs):
        """Group the definitions by how deep they sit in their own operands:
        one which uses no other is at the first level, and any other one comes
        after both of the ones it uses. Nothing inside a level uses anything
        else of that level, so they can be reordered freely.

        Read off the operands rather than off the rounds of the optimisation,
        so that whatever the plain scan adds at the end lands where it belongs.
        The operands of a definition always come before it, so one pass is
        enough."""

        depth = {}
        levels = collections.defaultdict(list)
        for index, left, right, _ratio, _count in defs:
            here = 0
            if left < 0:
                here = max(here, depth[-left])
            if right < 0:
                here = max(here, depth[-right])
            depth[index] = here + 1
            levels[here + 1].append(index)
        return [levels[key] for key in sorted(levels)]

    @staticmethod
    def jamp_orbit_usable(rows):
        """Restrict an orbit to the entries only one of its sub-expressions
        wants, and say whether what is left can be introduced as a whole. Which
        of two sub-expressions of the same orbit gets a shared entry cannot be
        decided in a way that commutes with the symmetry, so those entries are
        left in the matrix and get another chance in a later round."""

        sizes = set(len(use) for use in rows.values())
        if len(sizes) != 1 or sizes == set([0]):
            return False
        entry = collections.Counter()
        for operation, use in rows.items():
            for i in use:
                entry[(i, operation[0])] += 1
                entry[(i, operation[1])] += 1
        if max(entry.values()) == 1:
            return True
        for operation in list(rows):
            rows[operation] = [i for i in rows[operation]
                               if entry[(i, operation[0])] == 1
                               and entry[(i, operation[1])] == 1]
        sizes = set(len(use) for use in rows.values())
        return len(sizes) == 1 and sizes != set([0])
