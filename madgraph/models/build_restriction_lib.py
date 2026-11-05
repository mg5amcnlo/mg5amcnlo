################################################################################
#
# Copyright (c) 2012 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################


class Rule(object):
    """ """
    
    def __init__(self, name, default, data,first=True, inverted_display=False):
        """ """
        self.name = name
        self.default=default
        self.status=default
        self.lhablock = data[0].lower()
        self.lhaid = data[1]
        self.value = data[2]
        self.first = first
        if inverted_display:
            self.display = lambda x: not x
        else:
            self.display = lambda x: x

    def get_rules(self):
        """the (lhablock, lhacode, value) to write in the restriction card"""

        if self.status:
            return [(self.lhablock, self.lhaid, self.value)]
        return []


class ChoiceOption(object):
    """An option with more than two -mutually exclusive- values (a Rule is
    the True/False case). choices is a list of (label, rules) where rules is
    the list of (lhablock, lhacode, value) to apply for that label."""

    # a ChoiceOption is not split over several entries (unlike Rule) so it is
    # always the entry displayed/toggled by the customize_model question
    first = True

    def __init__(self, name, choices, default, description=None):

        self.name = name
        self.choices = [(str(label), rules) for (label, rules) in choices]
        self.labels = [label for (label, rules) in self.choices]
        default = str(default)
        assert default in self.labels
        self.default = default
        self.status = default
        self.description = description

    def display(self, status):
        return status

    def set_status(self, value):
        """set the option to one of its allowed values. The value is case
        insensitive and can be an unambiguous abbreviation of a label, so that
        'set flavourscheme 5' is the same as 'set flavourscheme 5F'."""

        self.status = self.resolve(value)

    def resolve(self, value):
        """the label a user input refers to"""

        text = str(value).strip()
        if text in self.labels:
            return text
        for candidates in ([label for label in self.labels
                            if label.lower() == text.lower()],
                           [label for label in self.labels
                            if label.lower().startswith(text.lower())]):
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                raise ValueError('%s is ambiguous for \'%s\': it can be %s'
                                 % (value, self.name, ', '.join(candidates)))
        raise ValueError('%s is not a valid value for \'%s\'. Valid values are: %s'
                         % (value, self.name, ', '.join(self.labels)))

    def next_status(self):
        """cycle to the next allowed value (interactive toggle)"""

        self.status = self.labels[(self.labels.index(self.status) + 1) % len(self.labels)]

    def get_rules(self):
        """the (lhablock, lhacode, value) to write in the restriction card"""

        for label, rules in self.choices:
            if label == self.status:
                return rules
        return []


class Category(list):
    """A container for the different rules"""
    
    def __init__(self, name, *args, **opt):
        """store a title for those restriction category"""
        
        self.name = name
        list.__init__(self, *args, **opt)
        
    def add_options(self, name='', default='', inverted_display=False, rules=[]):
        first=True
        for arg in rules:
            current_rule = Rule(name, default, arg, first, inverted_display) 
            self.append(current_rule)
            first=False


        
        
        


#===============================================================================
# Generic (model independent) customization options
#===============================================================================
# Those options are proposed for every model (on top of the model specific
# options defined in the build_restrict.py of the model -if any-). They are
# built from the content of the model itself: an option is proposed only if
# the associated particle/parameter is present (and external) in the model.

LIGHT_QUARKS = [1, 2, 3]   # d, u, s: massless in all the flavour schemes
HEAVY_QUARKS = [4, 5]      # c, b
LEPTONS = [15, 13, 11]     # tau, mu, e: ordered from the heaviest

PARTICLE_NAME = {1: 'd', 2: 'u', 3: 's', 4: 'c', 5: 'b',
                 11: 'e', 13: 'mu', 15: 'tau'}


def get_external_parameters(model):
    """return the list of the external parameters of the model"""

    try:
        return model['parameters'][('external',)]
    except (KeyError, TypeError):
        return []


def get_mass_rules(model, pdgs):
    """return the list of (lhablock, lhacode, 0.0) needed to set to zero the
    mass -and the associated width/yukawa- of all the particles of pdgs.
    Only the parameters which are external in model are returned."""

    externals = get_external_parameters(model)
    by_name = dict((param.name, param) for param in externals)
    particle_dict = model.get('particle_dict')

    rules = []
    done = set()
    for pdg in pdgs:
        particle = particle_dict.get(pdg, None)
        if particle is None:
            continue
        param = by_name.get(particle.get('mass'), None)
        if param is None:
            continue # already massless (or internal -> can not be restricted)
        key = (param.lhablock.lower(), tuple(param.lhacode))
        if key not in done:
            done.add(key)
            rules.append((param.lhablock, list(param.lhacode), 0.0))
        # a massless particle can not decay
        param = by_name.get(particle.get('width'), None)
        if param is not None:
            key = (param.lhablock.lower(), tuple(param.lhacode))
            if key not in done:
                done.add(key)
                rules.append((param.lhablock, list(param.lhacode), 0.0))

    # the yukawa couplings are not attached to a particle, so they have to be
    # looked for via their lhacode
    for param in externals:
        if param.lhablock.upper() != 'YUKAWA' or len(param.lhacode) != 1:
            continue
        if param.lhacode[0] not in pdgs:
            continue
        key = (param.lhablock.lower(), tuple(param.lhacode))
        if key not in done:
            done.add(key)
            rules.append((param.lhablock, list(param.lhacode), 0.0))

    return rules


def is_massive(model, pdg):
    """check if the particle pdg is massive in model (False if not present)"""

    particle = model.get('particle_dict').get(pdg, None)
    if particle is None:
        return False
    if particle.get('mass').lower() == 'zero':
        return False
    value = dict.get(model, 'parameter_dict', {})
    if particle.get('mass') in value:
        return bool(abs(complex(value[particle.get('mass')])))
    return True


def can_be_massive(model, pdg):
    """a particle can be given a mass only if it has a mass parameter at all
    (a UFO where the mass is hardcoded to ZERO can not be changed by a
    restriction card)"""

    particle = model.get('particle_dict').get(pdg, None)
    if particle is None:
        return False
    return particle.get('mass').lower() != 'zero'


def can_be_massless(model, pdg):
    """a particle can be made massless if it already is, or if its mass is one
    of the parameters of the param_card"""

    particle = model.get('particle_dict').get(pdg, None)
    if particle is None:
        return True
    if particle.get('mass').lower() == 'zero':
        return True
    return particle.get('mass') in set(param.name
                                for param in get_external_parameters(model))


def why_not_reachable(model, massive, massless):
    """the reason -in plain words- why a model can not be put in a scheme where
    the particles of massive have a mass and the ones of massless do not"""

    out = []
    blocked = [PARTICLE_NAME.get(pdg, pdg) for pdg in massive
                                          if not can_be_massive(model, pdg)]
    if blocked:
        out.append('%s %s no mass in this model' % (', '.join(str(p) for p in blocked),
                                       'has' if len(blocked) == 1 else 'have'))
    blocked = [PARTICLE_NAME.get(pdg, pdg) for pdg in massless
                                          if not can_be_massless(model, pdg)]
    if blocked:
        out.append('the mass of %s is not in the param_card' %
                   ', '.join(str(p) for p in blocked))
    return ', '.join(out)


def get_flavour_scheme_option(model, reference):
    """the 3F/4F/5F choice for the quarks. model is the model to restrict
    (no restriction applied), reference is the model as currently loaded by
    the user and is only used to define the default value of the option."""

    if not any(model.get('particle_dict').get(pdg, None)
                                for pdg in LIGHT_QUARKS + HEAVY_QUARKS):
        return None # not a model with quarks

    # a scheme is only proposed if this model can be put in it
    choices, refused = [], []
    for nf in [3, 4, 5]:
        massless = [pdg for pdg in LIGHT_QUARKS + HEAVY_QUARKS if pdg <= nf]
        massive = [pdg for pdg in HEAVY_QUARKS if pdg > nf]
        why = why_not_reachable(model, massive,
                                [pdg for pdg in massless if pdg in HEAVY_QUARKS])
        if why:
            refused.append('%dF (%s)' % (nf, why))
        else:
            choices.append(('%dF' % nf, get_mass_rules(model, massless)))

    # default: reproduce the scheme of the model as currently loaded
    if not is_massive(reference, 5) and not is_massive(reference, 4):
        default = '5F'
    elif not is_massive(reference, 4):
        default = '4F'
    else:
        default = '3F'

    labels = [label for label, rules in choices]
    if default not in labels:
        if not choices:
            # nothing is reachable: still show the scheme this model is in
            massless = [pdg for pdg in LIGHT_QUARKS + HEAVY_QUARKS
                                                     if pdg <= int(default[0])]
            choices = [(default, get_mass_rules(model, massless))]
        else:
            default = labels[0]

    description = ('3F: c and b massive, 4F: b massive, 5F: none of them.'
                   ' u, d and s are massless in all the schemes')
    if refused:
        description += '. Only %s possible for this model: %s' % (
            ' and '.join(label for label, rules in choices), '; '.join(refused))

    return ChoiceOption('flavour scheme', choices, default,
                        description=description)


def get_lepton_scheme_option(model, reference):
    """the number of massive leptons (0 to 3). Massive leptons are taken from
    the heaviest one: 1 -> tau, 2 -> tau and mu, 3 -> tau, mu and e."""

    if not any(model.get('particle_dict').get(pdg, None) for pdg in LEPTONS):
        return None # not a model with charged leptons

    choices, refused = [], []
    for nb in [0, 1, 2, 3]:
        why = why_not_reachable(model, LEPTONS[:nb], LEPTONS[nb:])
        if why:
            refused.append('%d (%s)' % (nb, why))
        else:
            choices.append(('%d' % nb, get_mass_rules(model, LEPTONS[nb:])))

    default = '%d' % len([pdg for pdg in LEPTONS if is_massive(reference, pdg)])

    labels = [label for label, rules in choices]
    if default not in labels:
        if not choices:
            choices = [(default, get_mass_rules(model, LEPTONS[int(default):]))]
        else:
            default = labels[0]

    description = '0: all massless, 1: tau, 2: tau mu, 3: tau mu e'
    if refused:
        description += '. Only %s possible for this model: %s' % (
            ' and '.join(label for label, rules in choices), '; '.join(refused))

    return ChoiceOption('nb of massive leptons', choices, default,
                        description=description)


def get_ckm_rules(model):
    """the (lhablock, lhacode, value) which make the quark mixing matrix
    diagonal, for the parameterisations a SM-like model uses. Empty if this
    model does not follow any of them."""

    blocks = {}
    for param in get_external_parameters(model):
        blocks.setdefault(param.lhablock.lower(), []).append(param)

    # the Wolfenstein parameterisation (sm, loop_sm and everything built on
    # them): lambda = A = rho = eta = 0 is the identity matrix
    if 'wolfenstein' in blocks:
        return [(param.lhablock, list(param.lhacode), 0.0)
                for param in blocks['wolfenstein']
                if len(param.lhacode) == 1 and param.lhacode[0] in [1, 2, 3, 4]]

    for name in sorted(blocks):
        if 'ckm' not in name:
            continue
        params = blocks[name]
        # the matrix itself (VCKM of the SLHA2 convention, CKMBLOCK, ...)
        if all(len(param.lhacode) == 2 for param in params):
            return [(param.lhablock, list(param.lhacode),
                     1.0 if param.lhacode[0] == param.lhacode[1] else 0.0)
                    for param in params]
        # a single mixing angle (the Cabibbo angle of the 2 generation models)
        if len(params) == 1 and len(params[0].lhacode) == 1:
            return [(params[0].lhablock, list(params[0].lhacode), 0.0)]

    return []


def is_already_applied(model, rules):
    """check if a set of rules is already satisfied by model. A parameter which
    is not in the param_card of model anymore was necessarily fixed by the
    restriction it was loaded with."""

    by_lha = dict(((param.lhablock.lower(), tuple(param.lhacode)), param)
                  for param in get_external_parameters(model))
    values = dict.get(model, 'parameter_dict', {})

    for lhablock, lhacode, value in rules:
        param = by_lha.get((lhablock.lower(), tuple(lhacode)), None)
        if param is None:
            continue # not a parameter of that model anymore: already fixed
        try:
            current = complex(values.get(param.name, param.value))
        except (TypeError, ValueError):
            return False
        if abs(current - value) > 1e-10:
            return False
    return True


def get_ckm_category(model, reference):
    """the 'diagonal ckm' option, for the models which follow one of the
    conventions of the SM for the quark mixing"""

    rules = get_ckm_rules(model)
    if not rules:
        return None

    category = Category('quark mixing')
    category.add_options(name='diagonal ckm',
                         default=is_already_applied(reference, rules),
                         rules=rules)
    return category


def get_generic_categories(model, reference=None):
    """return the list of the categories proposed for any model.
    model is the (unrestricted) model on which the restriction is applied,
    reference is the model as currently loaded by the user (used to define
    the default value of the options)."""

    if reference is None:
        reference = model

    categories = []

    category = Category('mass scheme')
    for option in [get_flavour_scheme_option(model, reference),
                   get_lepton_scheme_option(model, reference)]:
        if option is not None:
            category.append(option)
    if category:
        categories.append(category)

    category = get_ckm_category(model, reference)
    if category is not None:
        categories.append(category)

    return categories


def get_rules_target(option):
    """return the set of (lhablock, lhacode) that an option can modify"""

    out = set()
    if isinstance(option, ChoiceOption):
        rules = sum([r for label, r in option.choices], [])
    else:
        rules = [(option.lhablock, option.lhaid, option.value)]
    for lhablock, lhacode, value in rules:
        out.add((lhablock.lower(), tuple(lhacode)))
    return out
