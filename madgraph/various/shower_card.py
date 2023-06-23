################################################################################
#
# Copyright (c) 2011 The MadGraph5_aMC@NLO Development team and Contributors
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
"""A File for splitting"""

from __future__ import absolute_import
import sys
import re
import os
import logging
from six.moves import range

try:
    import madgraph
except ImportError:
    import internal.misc as misc
    import internal.banner as banner
    from internal import InvalidCmd
else:
    import madgraph.various.misc as misc
    import madgraph.various.banner as banner
    from madgraph import InvalidCmd
    
logger = logging.getLogger('madgraph.shower_card') 

pjoin = os.path.join

class ShowerCardError(Exception):
    pass

class ShowerCard(banner.RunCard):
    """ """
    true = ['.true.', 't', 'true', '1']
    false = ['.false.', 'f', 'false', '0']
    logical_vars = ['ue_enabled', 'hadronize', 'b_stable', 'pi_stable', 'wp_stable', 
                    'wm_stable', 'z_stable', 'h_stable', 'tap_stable', 'tam_stable', 
                    'mup_stable', 'mum_stable', 'is_4lep', 'is_bbar', 'combine_td']
    string_vars = ['extralibs', 'extrapaths', 'includepaths', 'analyse']
    for i in range(1,100):
        string_vars.append('dm_'+str(i))
    int_vars = ['nsplit_jobs', 'maxprint', 'nevents', 'pdfcode', 'rnd_seed', 'rnd_seed2', 'njmax']
    float_vars = ['maxerrs', 'lambda_5', 'b_mass', 'qcut']

    # names_dict has the following structure:
    # var : {PYTHIA6: varpy6, HERWIG6: varhw6, HERWIGPP: varhwpp, PYTHIA8: varpy8}
    # where varpy, varhw6 and varhwpp are mc_dependent names
    # if a mc is not there, that variable is not supposed to be
    # used / written for thar mc
    names_dict = {\
            'ue_enabled' : {'HERWIG6':'lhsoft', 'PYTHIA6': 'mstp_81', 'HERWIGPP': 'ue_hwpp', 'PYTHIA8': 'ue_py8'},
            'pdfcode' : {'HERWIG6':'pdfcode', 'PYTHIA6': 'pdfcode', 'HERWIGPP': 'pdfcode', 'PYTHIA8': 'pdfcode'},
            'nevents' : {'HERWIG6':'nevents', 'PYTHIA6': 'nevents', 'HERWIGPP': 'nevents', 'PYTHIA8': 'nevents'},
            'hadronize' : {'PYTHIA6': 'mstp_111', 'HERWIGPP': 'hadronize_hwpp', 'PYTHIA8': 'hadronize_py8'},
            'b_stable' : {'HERWIG6':'b_stable_hw', 'PYTHIA6': 'b_stable_py', 'HERWIGPP': 'b_stable_hwpp', 'PYTHIA8': 'b_stable_py8'},
            'pi_stable' : {'HERWIG6':'pi_stable_hw', 'PYTHIA6': 'pi_stable_py', 'HERWIGPP': 'pi_stable_hwpp', 'PYTHIA8': 'pi_stable_py8'},
            'wp_stable' : {'HERWIG6':'wp_stable_hw', 'PYTHIA6': 'wp_stable_py', 'HERWIGPP': 'wp_stable_hwpp', 'PYTHIA8': 'wp_stable_py8'},
            'wm_stable' : {'HERWIG6':'wm_stable_hw', 'PYTHIA6': 'wm_stable_py', 'HERWIGPP': 'wm_stable_hwpp', 'PYTHIA8': 'wm_stable_py8'},
            'z_stable' : {'HERWIG6':'z_stable_hw', 'PYTHIA6': 'z_stable_py', 'HERWIGPP': 'z_stable_hwpp', 'PYTHIA8': 'z_stable_py8'},
            'h_stable' : {'HERWIG6':'h_stable_hw', 'PYTHIA6': 'h_stable_py', 'HERWIGPP': 'h_stable_hwpp', 'PYTHIA8': 'h_stable_py8'},
            'tap_stable' : {'HERWIG6':'taup_stable_hw', 'PYTHIA6': 'taup_stable_py', 'HERWIGPP': 'taup_stable_hwpp', 'PYTHIA8': 'taup_stable_py8'},
            'tam_stable' : {'HERWIG6':'taum_stable_hw', 'PYTHIA6': 'taum_stable_py', 'HERWIGPP': 'taum_stable_hwpp', 'PYTHIA8': 'taum_stable_py8'},
            'mup_stable' : {'HERWIG6':'mup_stable_hw', 'PYTHIA6': 'mup_stable_py', 'HERWIGPP': 'mup_stable_hwpp', 'PYTHIA8': 'mup_stable_py8'},
            'mum_stable' : {'HERWIG6':'mum_stable_hw', 'PYTHIA6': 'mum_stable_py', 'HERWIGPP': 'mum_stable_hwpp', 'PYTHIA8': 'mum_stable_py8'},
            'is_4lep' : {'PYTHIA6':'is_4l_py'},
            'is_bbar' : {'HERWIG6':'is_bb_hw'},
            'maxprint' : {'HERWIG6':'maxpr_hw', 'PYTHIA6': 'maxpr_py', 'HERWIGPP': 'maxpr_hwpp', 'PYTHIA8': 'maxpr_py8'},
            'rnd_seed' : {'HERWIG6':'rndevseed1_hw', 'PYTHIA6': 'rndevseed_py', 'HERWIGPP': 'rndevseed_hwpp', 'PYTHIA8': 'rndevseed_py8'},
            'rnd_seed2' : {'HERWIG6':'rndevseed2_hw'},
            'maxerrs' : {'HERWIG6':'err_fr_hw', 'PYTHIA6': 'err_fr_py', 'HERWIGPP': 'err_fr_hwpp', 'PYTHIA8': 'err_fr_py8'},
            'lambda_5' : {'HERWIG6':'lambdaherw', 'PYTHIA6': 'lambdapyth', 'HERWIGPP': 'lambdaherw', 'PYTHIA8': 'lambdapyth'},
            'b_mass' : {'HERWIG6':'b_mass', 'PYTHIA6': 'b_mass', 'HERWIGPP': 'b_mass', 'PYTHIA8': 'b_mass'},
            'analyse' : {'HERWIG6':'hwuti', 'PYTHIA6':'pyuti', 'HERWIGPP':'hwpputi', 'PYTHIA8':'py8uti'},
            'qcut' : {'PYTHIA8':'qcut'},
            'njmax' : {'PYTHIA8':'njmax'}}
    
    stdhep_dict = {'HERWIG6':'mcatnlo_hwan_stdhep.o', 'PYTHIA6':'mcatnlo_pyan_stdhep.o'}
    
    def __new__(cls, *args, **opts):
        
        #important to bypass the run_card.__new__ which is designed as a 
        #factory between LO and NLO card. 
        return banner.ConfigFile.__new__(cls, *args, **opts)

    def __init__(self, card=None, testing=False):
        """ if testing, card is the content"""
        self.testing = testing
        super().__init__(card)
        

    def default_setup(self):
        """default value for all the parameters"""

        # Number of events, jobs, errors, and random seeds   
        self.add_param("nevents", -1, comment="N evts to shower (< 0 = all)")
        self.add_param("nsplit_jobs", 1, comment="N jobs to run in parallel (< 100!!)")
        self.add_param("combine_td", True, comment="combine the topdrawer/HwU files if nsplit_jobs>1")
        self.add_param("maxprint", 2, comment="N evts to print in the log")
        self.add_param("maxerrs", 0.1, comment="max fraction of errors")
        self.add_param("rnd_seed", 0)
        self.add_param("rnd_seed2", 0, comment="2nd random seed (0 = default) !ONLY FOR HWERIG6!")
        # PDFs and non-perturbative modelling 
        self.add_param("pdfcode", 1, comment="0 = internal, 1 = same as NLO, other = lhaglue")
        self.add_param("ue_enabled", False, comment="underlying event")
        self.add_param("hadronize", True, comment=" hadronisation on/off        !IGNORED BY HERWIG6!")
        self.add_param("lambda_5", -1., comment="Lambda_5 (< 0 = default)    !IGNORED BY PYTHIA8!")
        # Stable or unstable particles
        self.add_param("b_stable", False, comment="set B hadrons stable")
        self.add_param("pi_stable",  True, comment="set pi0's stable")
        self.add_param("wp_stable",  False, comment="set w+'s stable")
        self.add_param("wm_stable", False, comment="set w-'s stable")
        self.add_param("z_stable",  False, comment="set z0's stable")
        self.add_param("h_stable",  False, comment="set Higgs' stable")
        self.add_param("tap_stable",  False, comment="set tau+'s stable")
        self.add_param("tam_stable",  False, comment="set tau-'s stable")
        self.add_param("mup_stable",  False, comment="set mu+'s stable")
        self.add_param("mum_stable",  False, comment="set mu-'s stable")
        # Mass of the b quark
        self.add_param("b_mass", -1., comment="# if < 0 = read from SubProcesses/MCmasses_*.inc")
        # Special settings
        self.add_param("is_4lep", False, comment="T if 4-lepton production      !ONLY FOR PYTHIA6!")
        self.add_param("is_bbar", False, comment="T if bb~ production           !ONLY FOR HERWIG6!")
        # FxFx merging parameters 
        self.add_param("Qcut",  -1.0, comment="Merging scale")
        self.add_param("njmax", -1, comment="Maximal multiplicity in the merging. -1 means guessed  from the process definition")

        # DECAY
        for i in range(1, 100):
            self.add_param("DM_%i" % i, "", hidden=True, comment="decay syntax (warning syntax depend of the PS used)")

        # Extra libraries/analyses
        self.add_param("EXTRALIBS", "stdhep Fmcfio", comment="Extra-libraries (not LHAPDF)")
        self.add_param("EXTRAPATHS", "../lib", comment="Path to the extra-libraries")
        self.add_param("INCLUDEPATHS", "", comment="Path to header files needed by c++. Dir names separated by white spaces")
        self.add_param("ANALYSE", "", comment="User's analysis and histogramming routines; HwU.o should be linked first")

        # Pythia8 specific
        self.add_param("qed_shower", True, comment="T = enable QED shower for Q and L !ONLY FOR PYTHIA8!")
        self.add_param("primordialkt", False, comment="T = enable primordial parton k_T  !ONLY FOR PYTHIA8!")
        self.add_param("tune_ee", 7, comment="pythia8 tune for ee beams !ONLY FOR PYTHIA8!")
        self.add_param("tune_pp", 14, comment="pythia8 tune for pp beams !ONLY FOR PYTHIA8!")
        #self.add_param("pythia8_options", {'__type__':""}, comment="specify (as dictionary) additional parameters that you want to setup within the pythia8 program")


    def read(self, input, *opt):
        self.read_card(input)

    def read_card(self, card_path):
        """read the shower_card, if testing card_path is the content"""
        if not self.testing:
            content = open(card_path).read()
        else:
            content = card_path
        lines = content.split('\n')
        list_dm = []
        for l in lines:
            if '#' in l:
                l = l.split('#',1)[0]
            if '=' not in l:
                continue
            args = l.split('=',1) # here the 1 is important in case of string passed
            key = args[0].strip().lower()
            value = args[1].strip()
            self.set_param(key, value)
            if str(key).upper().startswith('DM'):
                list_dm.append(int(key.split('_',1)[1]))
            #special case for DM_*
            for i in range(1,100):
                if not i in list_dm: 
                    self['dm_'+str(i)] = ''

        self.text=content


    def set_param(self, key, value, write_to = ''):
        """set the param key to value.
        if write_to is passed then write the new shower_card:
        if not testing write_to is an input path, if testing the text is
        returned by the function
        """
        
        try:
            self[key] = value
        except InvalidCmd as error:
            raise ShowerCardError(str(error))

        if isinstance(self[key], str):
            if value.lower() == 'none':
                self[key] = '' 

        #then update self.text and write the new card
        if write_to:
            logger.info('modify parameter %s of the shower_card.dat to %s' % (key, value))
            key_re = re.compile('^(\s*)%s\s*=\s*(.+)\s*$' % key , re.IGNORECASE)
            newlines = []
            for line in self.text.split('\n'):
                key_match = key_re.match(line)
                if key_match and not ( str(key).upper().startswith('DM') ):
                    try:
                        comment = line.split('#')[1]
                    except:
                        comment = ''
                    if not isinstance(self[key], bool):
                        newlines.append('%s = %s #%s' % (key, value, comment))
                    else:
                        if self[key]:
                            newlines.append('%s = %s #%s' % (key, 'T', comment))
                        else:
                            newlines.append('%s = %s #%s' % (key, 'F', comment))
                elif key_match and ( str(key).upper().startswith('DM') ):
                    pass
                else:
                    newlines.append(line)

            if str(key).upper().startswith('DM') and not value.lower() in ['','none','default']:
                newlines.append('%s = %s' % (str(key).upper(), value[0:len(value)]))
                logger.info('please specify a decay through set DM_1 decay; see shower_card.dat for details')
                
            self.text = '\n'.join(newlines) + '\n'

            if self.testing:
                return self.text
            else:
                open(write_to, 'w').write(self.text)
                return ''
        else:
            return ''



    def write_card(self, shower, card_path):
        """write the shower_card for shower in card_path.
        if self.testing, card_path takes the value of the string"""

        shower = shower.upper()
        if shower.startswith('PYTHIA6'):
            self.shower = 'PYTHIA6'
        else:
            self.shower = shower
        lines = []
        bool_dict = {True: '.true.', False: '.false.'}
        bool_dict_num = {True: '1', False: '0'}
        
        for key in self:
            value = self[key]
            if isinstance(value, bool): 
                # deal with special case for pythia:
                if key in ['ue_enabled', 'hadronize'] and self.shower == 'PYTHIA6':
                    value = bool_dict_num[value]
                else:
                    value = bool_dict[value]
            elif isinstance(self[key], str):
                # deal in a special way with analyse
                if key.lower() == 'analyse':
                    key = key.lower()
                    if value is None or not value:
                        try:
                            value = self.stdhep_dict[self.shower]
                        except KeyError as error:
                            pass
                    try:
                        line = '%s="%s"' % (self.names_dict[key][self.shower].upper(), value)
                        lines.append(line)
                        continue
                    except KeyError as error:
                        misc.sprint(error)
                        continue
                    
                elif key.startswith('DM_') and not value:
                    continue
                if value is None or not value:
                    value = ''
                else:
                    value = '"%s"' % value

                line = '%s=%s' % (key.upper(), value)
                lines.append(line)
                continue
            elif isinstance(self[key], int):
                value = '%d' % value
            elif isinstance(self[key], float):
                value = '%4.3f' % value
            else:
                raise ShowerCardError('Unknown key: %s = %s' % (key, value))
            try:
                line = '%s=%s' % (self.names_dict[key][self.shower].upper(), value.upper())
                lines.append(line)
            except KeyError:
                pass

        if self.testing:
            return ('\n'.join(lines) + '\n')
        else:
            open(card_path, 'w').write(('\n'.join(lines) + '\n'))

