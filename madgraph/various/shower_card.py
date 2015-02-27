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

import sys
import re
import os
import logging

logger = logging.getLogger('madgraph.shower_card') 

pjoin = os.path.join

class ShowerCardError(Exception):
    pass

class ShowerCard(dict):
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
    
    def __init__(self, card=None, testing=False):
        """ if testing, card is the content"""
        self.testing = testing
        dict.__init__(self)
        self.keylist = self.keys()

        if card:
            self.read_card(card)

    
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
        if key in self.logical_vars:
            if str(value).lower() in self.true:
                self[key] = True
            elif str(value).lower() in self.false:
                self[key] = False
            else:
                raise ShowerCardError('%s is not a valid value for %s' % \
                        (value, key))
        elif key in self.string_vars:
            if value.lower() == 'none':
                self[key] = ''
            else:
                self[key] = value
        elif key in self.int_vars:
            try:
                self[key] = int(value)
            except ValueError:
                raise ShowerCardError('%s is not a valid value for %s. An integer number is expected' % \
                        (value, key))
        elif key in self.float_vars:
            try:
                self[key] = float(value)
            except ValueError:
                raise ShowerCardError('%s is not a valid value for %s. A floating point number is expected' % \
                        (value, key))
        else:
            raise ShowerCardError('Unknown entry: %s = %s' % (key, value))
        self.keylist.append(key)

        #then update self.text and write the new card
        if write_to:
            logger.info('modify parameter %s of the shower_card.dat to %s' % (key, value))
            key_re = re.compile('^(\s*)%s\s*=\s*(.+)\s*$' % key , re.IGNORECASE)
            newlines = []
            for line in self.text.split('\n'):
                key_match = key_re.match(line)
                if key_match and not str(key).upper().startswith('DM'):
                    try:
                        comment = line.split('#')[1]
                    except:
                        comment = ''
                    if key not in self.logical_vars:
                        newlines.append('%s = %s #%s' % (key, value, comment))
                    else:
                        if key:
                            newlines.append('%s = %s #%s' % (key, 'T', comment))
                        else:
                            newlines.append('%s = %s #%s' % (key, 'F', comment))
                elif key_match and str(key).upper().startswith('DM'):
                    pass
                else:
                    newlines.append(line)

            if str(key).upper().startswith('DM') and not value.lower() in ['','none','default']:
                newlines.append('%s = %s' % (str(key).upper(), value[0:len(value)]))
                logger.info('please specify a decay through set dm_1 M > D1 D2 @ BR @ ME; see shower_card.dat for details')
                
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
        
        for key in self.keylist:
            value = self[key]
            if key in self.logical_vars:
                # deal with special case for pythia:
                if key in ['ue_enabled', 'hadronize'] and self.shower == 'PYTHIA6':
                    value = bool_dict_num[value]
                else:
                    value = bool_dict[value]
            elif key in self.string_vars:
                # deal in a special way with analyse
                if key == 'analyse':
                    if value is None or not value:
                        try:
                            value = self.stdhep_dict[self.shower]
                        except KeyError:
                            pass
                    try:
                        line = '%s="%s"' % (self.names_dict[key][self.shower].upper(), value)
                        lines.append(line)
                        continue
                    except KeyError:
                        continue
                if value is None or not value:
                    value = ''
                else:
                    value = '"%s"' % value

                line = '%s=%s' % (key.upper(), value)
                lines.append(line)
                continue
            elif key in self.int_vars:
                value = '%d' % value
            elif key in self.float_vars:
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

