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
                    'mup_stable', 'mum_stable', 'is_4lep', 'is_bbar', 'combine_td', 
                    'qed_shower', 'primordialkt',
                    'space_shower_me_corrections', 'time_shower_me_corrections',
                    'time_shower_me_extended', 'time_shower_me_after_first']
    string_vars = ['extralibs', 'extrapaths', 'includepaths', 'analyse']
    for i in range(1,100):
        string_vars.append('dm_'+str(i))
    int_vars = ['nsplit_jobs', 'maxprint', 'nevents', 'pdfcode', 'rnd_seed', 'rnd_seed2', 'njmax','tune_ee','tune_pp']
    float_vars = ['maxerrs', 'lambda_5', 'b_mass', 'qcut']

    # names_dict has the following structure:
    # var : {PYTHIA6: varpy6, HERWIG6: varhw6, HERWIGPP: varhwpp, PYTHIA8: varpy8}
    # where varpy, varhw6 and varhwpp are mc_dependent names
    # if a mc is not there, that variable is not supposed to be
    # used / written for thar mc
    names_dict = {}
    stdhep_dict = {'HERWIG6':'mcatnlo_hwan_stdhep.o', 'PYTHIA6':'mcatnlo_pyan_stdhep.o'}
    
    def __new__(cls, *args, **opts):
        
        #important to bypass the run_card.__new__ which is designed as a 
        #factory between LO and NLO card. 
        return banner.ConfigFile.__new__(cls, *args, **opts)

    def __init__(self, card=None, testing=False):
        """ if testing, card is the content"""
        self.testing = testing
        self.text = None
        super().__init__(card)
        

    def add_param(self, name, *args, 
                  py8='', py6='', hw6='', hwpp='',
                  all_sh=None, sh_postfix=False,**opts):
        
        if all_sh:
            if py8 is not None and not py8:
                py8 = all_sh
                if sh_postfix:
                    py8 = py8 + "_py8"
            if py6 is not None and not py6:
                py6 = all_sh
                if sh_postfix:
                    py6 = py6 + "_py"
            if hw6 is not None and not hw6:
                hw6 = all_sh
                if sh_postfix:
                    hw6 = hw6 + "_hw"
            if hwpp is not None and not hwpp:
                hwpp = all_sh
                if sh_postfix:
                    hwpp = hwpp + "_hwpp"

        name = name.lower()
        self.names_dict[name] = {}
        if py8:
            self.names_dict[name]['PYTHIA8'] = py8
        if py6:
            self.names_dict[name]['PYTHIA6'] = py6
        if hw6:
            self.names_dict[name]['HERWIG6'] = hw6
        if hwpp:
            self.names_dict[name]['HERWIGPP'] = hwpp

        super().add_param(name,*args, **opts)

    def check_support(self, name, shower):
        return shower in self.names_dict[name.lower()]


    def default_setup(self):
        """default value for all the parameters"""

        # Number of events, jobs, errors, and random seeds   
        self.add_param("nevents", -1, comment="N evts to shower (< 0 = all)",
                       all_sh='nevents')
        self.add_param("nsplit_jobs", 1, comment="N jobs to run in parallel (< 100!!)")
        self.add_param("combine_td", True , comment="combine the topdrawer/HwU files if nsplit_jobs>1")
        self.add_param("maxprint", 2, comment="N evts to print in the log",
                       all_sh='maxpr', sh_postfix=True)
        self.add_param("maxerrs", 0.1, comment="max fraction of errors",
                       all_sh="err_fr", sh_postfix=True)
        self.add_param("rnd_seed", 0,
                       all_sh='rndevseed', sh_postfix=True, hw6='rndevseed1_hw')
        self.add_param("rnd_seed2", 0, comment="2nd random seed (0 = default) !ONLY FOR HWERIG6!",
                       hw6='rndevseed2_hw')


        # PDFs and non-perturbative modelling 
        self.add_param("pdfcode", 1, comment="0 = internal, 1 = same as NLO, other = lhaglue",
                       all_sh='pdfcode')
        self.add_param("ue_enabled", False, comment="underlying event",
                       hw6='lhsoft', py6='mstp_81', hwpp='ue_hwpp', py8='ue_py8')
        self.add_param("hadronize", True, comment=" hadronisation on/off        !IGNORED BY HERWIG6!",
                       py6='mstp_111', hwpp='hadronize_hwpp', py8='hadronize_py8')
        self.add_param("lambda_5", -1., comment="Lambda_5 (< 0 = default)    !IGNORED BY PYTHIA8!",
                       hw6='lambdaherw', hwpp='lambdaherw',
                       py6='lambdapyth', py8='lambdapyth')



        # Stable or unstable particles
        self.add_param("b_stable", False, comment="set B hadrons stable",
                       all_sh='b_stable', sh_postfix=True)
        self.add_param("pi_stable",  True, comment="set pi0's stable",
                       all_sh='pi_stable', sh_postfix=True)
        self.add_param("wp_stable",  False, comment="set w+'s stable",
                       all_sh='wp_stable', sh_postfix=True)
        self.add_param("wm_stable", False, comment="set w-'s stable",
                       all_sh='wm_stable', sh_postfix=True)
        self.add_param("z_stable",  False, comment="set z0's stable",
                       all_sh='z_stable', sh_postfix=True)
        self.add_param("h_stable",  False, comment="set Higgs' stable",
                       all_sh='h_stable', sh_postfix=True)
        self.add_param("tap_stable",  False, comment="set tau+'s stable",
                       all_sh='taup_stable', sh_postfix=True)
        self.add_param("tam_stable",  False, comment="set tau-'s stable",
                       all_sh='taum_stable', sh_postfix=True)
        self.add_param("mup_stable",  False, comment="set mu+'s stable",
                       all_sh='mup_stable', sh_postfix=True)
        self.add_param("mum_stable",  False, comment="set mu-'s stable",
                       all_sh='mum_stable', sh_postfix=True)
        # Mass of the b quark
        self.add_param("b_mass", -1., comment="# if < 0 = read from SubProcesses/MCmasses_*.inc",
                       all_sh='b_mass')
        # Special settings
        self.add_param("is_4lep", False, comment="T if 4-lepton production      !ONLY FOR PYTHIA6!",
                       py6='is_4l_py')
        self.add_param("is_bbar", False, comment="T if bb~ production           !ONLY FOR HERWIG6!",
                       hw6='is_bb_hw')


        # FxFx merging parameters 
        self.add_param("Qcut",  -1.0, comment="Merging scale", 
                       py8='qcut')
        self.add_param("njmax", -1, comment="Maximal multiplicity in the merging. -1 means guessed  from the process definition",
                       py8='njmax')

        # DECAY
        for i in range(1, 100):
            self.add_param("DM_%i" % i, "", hidden=True, comment="decay syntax (warning syntax depend of the PS used)")

        # Extra libraries/analyses
        self.add_param("EXTRALIBS", "stdhep Fmcfio", comment="Extra-libraries (not LHAPDF)")
        self.add_param("EXTRAPATHS", "../lib", comment="Path to the extra-libraries")
        self.add_param("INCLUDEPATHS", "", comment="Path to header files needed by c++. Dir names separated by white spaces")
        self.add_param("ANALYSE", "", comment="User's analysis and histogramming routines; HwU.o should be linked first",
                       hw6='hwuti', hwpp='hwpputi',py6='pyuti', py8='py8uti')

        # Pythia8 specific
        self.add_param("qed_shower", True, comment="T = enable QED shower for Q and L !ONLY FOR PYTHIA8!",
                       py8='qed_shower')
        self.add_param("primordialkt", False, comment="T = enable primordial parton k_T  !ONLY FOR PYTHIA8!",
                        py8='primordialkt')

        # ME correction
        self.add_param('space_shower_me_corrections', False, comment= "MECs for ISR",
                       py8='space_shower_me_corrections')
        self.add_param('time_shower_me_corrections', True, comment= "MECs for FSR",
                       py8='time_shower_me_corrections')
        self.add_param('time_shower_me_extended', False, comment= "see Pythia8 manual as well as hep-ph/2308.06389 for details",
                       py8='time_shower_me_extended')
        self.add_param('time_shower_me_after_first', False, comment= "see Pythia8 manual as well as hep-ph/2308.06389 for details",
                       py8='time_shower_me_after_first')

        #self.add_param("tune_ee", 7, comment="pythia8 tune for ee beams !ONLY FOR PYTHIA8!",
        #               py8="tune_ee")
        #self.add_param("tune_pp", 14, comment="pythia8 tune for pp beams !ONLY FOR PYTHIA8!",
        #               py8="tune_pp")
        self.add_param("pythia8_options", {'__type__':''}, comment="specify (as dictionary) additional parameters that you want to setup within the pythia8 program",
                       py8='extra_line')


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
            if self.testing:
                self.write(None)
                return self.text
            else:
                self.write(write_to)
                return ''
        else:
            return ''

    def write(self, output_file, template=None, python_template=False,
                    write_hidden=False, template_options=None, **opt):
        """Write the shower_card in output_file according to template 
           (a path to a valid shower_card)"""
        

        to_write = set(self.user_set)

        if not self.text:
            self.text = open(template,'r').read()


        key_re = re.compile('^(\s*)([\S^#]+)(\s*)=(\s*)([^#]*?)(\s*)(\#.*|$)')
        newlines = []
        for line in self.text.split('\n'):
            key_match = key_re.findall(line)
            if key_match:

                (s1, key, s2,s3, old_value,s4, comment) = key_match[0]
                if len(s3) == 0:
                    s3 = " "
                try:
                    to_write.remove(key.lower())
                except:
                    pass

                if ( str(key).upper().startswith('DM') ) and self[key] in ['','none','default']:
                    continue
                value = self[key]
                if isinstance(value, bool):
                    if value:
                        value = 'T'
                    else:
                        value = 'F'
                # comment startswith "#" if not empty
                newlines.append('%s%s%s=%s%s %s' % (s1,key,s2,s3,value,comment))
            else:
                newlines.append(line)
                
            for key in to_write:
                newlines.append(' %s = %s #%s' % (key,self[key],self.comments[key]))

        self.text = '\n'.join(newlines) + '\n'
        if isinstance(output_file, str):
            fsock = open(output_file,'w')
            fsock.write(self.text)
            fsock.close()
        elif output_file is None:
            return
        else:
            output_file.write(self.text)

    def create_default_for_process(self, *args, **opts):
        pass # will be usefull later on

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
            key = key.lower()
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
                        misc.sprint(key,error)
                        continue
                    
                elif key.startswith('dm_') and not value:
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
            elif key == 'pythia8_options':
                if self.shower=='PYTHIA8':
                    value = '"'
                    for k, v in self[key].items():
                        value += " %s = %s \n" % (k,v)
                    value += '"'
                else:
                    value = ''
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

