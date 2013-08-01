# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51

from object_library import all_CTparameters, CTParameter

from function_library import complexconjugate, re, im, csc, sec, acsc, asec

################
# R2 vertices  #
################

# ========== #
# DUMMY ONES #
# ========== #

RGR2 = CTParameter(name = 'RGR2',
              type = 'real',
              value = {0:'0'},
              texname = 'RGR2')

tMass_UV = CTParameter(name = 'tMass_UV',
                       type = 'complex',
                       value = {-1:'0',
                                 0:'0'
                               },
                       texname = '\delta m_t')
