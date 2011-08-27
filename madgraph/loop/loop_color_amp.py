################################################################################
#
# Copyright (c) 2009 The MadGraph Development team and Contributors
#
# This file is a part of the MadGraph 5 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph license which should accompany this 
# distribution.
#
# For more information, please visit: http://madgraph.phys.ucl.ac.be
#
################################################################################

"""Classes, methods and functions required to write QCD color information 
for a loop diagram and build a color basis, and to square a QCD color string for
squared diagrams and interference terms."""

import copy
import fractions
import operator
import re

import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color_algebra
import madgraph.core.diagram_generation as diagram_generation
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.core.base_objects as base_objects


#===============================================================================
# ColorBasis
#===============================================================================
class LoopColorBasis(color_amp.ColorBasis):
    """ Same class as its mother ColorBasis except that it can also handle
        LoopAmplitudes."""

    def closeColorLoop(self, colorize_dict, lcut_charge, lcut_numbers):
        """ Add a color delta in the right representation (depending on the 
        color charge carried by the L-cut particle whose number are given in
        the loop_numbers argument) to close the loop color trace """
                
        if lcut_charge==1:
            # No color carried by the lcut particle, there is nothing to do.
            return
        elif lcut_charge==3:
            closingCS=color_algebra.ColorString(\
              [color_algebra.T(lcut_numbers[1],lcut_numbers[0])])
        elif lcut_charge==6:
            closingCS=color_algebra.ColorString(\
              [color_algebra.T6(lcut_numbers[1],lcut_numbers[0])])
        elif lcut_charge==8:
            closingCS=color_algebra.ColorString(\
              [color_algebra.Tr(lcut_numbers[1],lcut_numbers[0])],
              fractions.Fraction(1, 2))
        else:
            raise ColorBasis.ColorBasisError, \
        "L-cut particle has an unsupported color representation %s" % lcut_charge
        
        # Append it to all color strings for this diagram.
        for CS in colorize_dict.values():
            CS.product(closingCS)
            
    def create_loop_color_dict_list(self, amplitude):
        """Returns a list of colorize dict for all loop diagrams in amplitude.
        Also update the _list_color_dict object accordingly """

        list_color_dict = []
        
        if not isinstance(amplitude,loop_diagram_generation.LoopAmplitude):
            raise ColorBasis.ColorBasisError, \
              'LoopColorBasis is used with an amplitude which is not a LoopAmplitude'
        
        lcut_numbers=(len(amplitude['process']['legs'])+1,\
                          len(amplitude['process']['legs'])+2)
        
        for diagram in amplitude.get('loop_diagrams'):
            colorize_dict = self.colorize(diagram,
                                          amplitude.get('process').get('model'))
            if diagram['type']>0:
                lcut_charge=\
                  amplitude['process']['model']['particle_dict'][\
                            diagram['type']].get('color')
                # We close here the color loop for loop diagrams (R2 have
                # negative 'type') by adding a delta in the two color indices of
                # loop_leg_numbers.
                self.closeColorLoop(colorize_dict,lcut_charge,lcut_numbers)
                
            list_color_dict.append(colorize_dict)

        self._list_color_dict = list_color_dict

        return list_color_dict

    def create_born_color_dict_list(self, amplitude):
        """Returns a list of colorize dict for all born diagrams in amplitude.
        Also update the _list_color_dict object accordingly """

        list_color_dict = []

        if not isinstance(amplitude,loop_diagram_generation.LoopAmplitude):
            raise ColorBasis.ColorBasisError, \
              'LoopColorBasis is used with an amplitude which is not a LoopAmplitude'

        for diagram in amplitude.get('born_diagrams'):
            colorize_dict = self.colorize(diagram,
                                          amplitude.get('process').get('model'))
            list_color_dict.append(colorize_dict)

        self._list_color_dict = list_color_dict

        return list_color_dict
            
    def build_born(self, amplitude):
        """Build the a color basis object using information contained in
        amplitude (otherwise use info from _list_color_dict). 
        Returns a list of color """

        self.create_born_color_dict_list(amplitude)
        for index, color_dict in enumerate(self._list_color_dict):
            self.update_color_basis(color_dict, index)
            
    def build_loop(self, amplitude):
        """Build the loop color basis object using information contained in
        amplitude (otherwise use info from _list_color_dict). 
        Returns a list of color """

        self.create_loop_color_dict_list(amplitude)
        for index, color_dict in enumerate(self._list_color_dict):
            self.update_color_basis(color_dict, index)
