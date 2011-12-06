# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51

from object_library import all_couplings, Coupling

from function_library import complexconjugate, re, im, csc, sec, acsc, asec

################
# R2 couplings #
################

R2_3Gq = Coupling(name = 'R2_3Gq',
                value = '2.0*G**3/(48.0*cmath.pi**2)',
                order = {'QCD':3})

R2_3Gg = Coupling(name = 'R2_3Gg',
                 value = 'Ncol*G**3/(48.0*cmath.pi**2)*(7.0/4.0+lhv)',
                 order = {'QCD':3})

#=============================================================================================
#  4-gluon R2 couplings
#=============================================================================================

# Gluon contribution to it

GC_4GR2_Gluon_delta5 = Coupling(name = 'GC_4GR2_Gluon_delta5',
                 value = '-4.0*complex(0,1)*RGR2*(2.0*lhv+5.0)',
                 order = {'QCD':4})

GC_4GR2_Gluon_delta7 = Coupling(name = 'GC_4GR2_Gluon_delta7',
                 value = '2.0*complex(0,1)*RGR2*(2.0*lhv+7.0)',
                 order = {'QCD':4})

GC_4GR2_2Struct = Coupling(name = 'GC_4GR2_2Struct',
                 value = '2.0*complex(0,1)*RGR2*Ncol*(lhv+3.0)',
                 order = {'QCD':4})

GC_4GR2_4Struct = Coupling(name = 'GC_4GR2_4Struct',
                 value = '-complex(0,1)*RGR2*Ncol*(4.0*lhv+11.0)',
                 order = {'QCD':4})

# Fermion contribution to it

GC_4GR2_Fermion_delta5 = Coupling(name = 'GC_4GR2_Fermion_delta5',
                 value = '(2.0/Ncol)*5.0*complex(0,1)*RGR2',
                 order = {'QCD':4})

GC_4GR2_Fermion_delta11 = Coupling(name = 'GC_4GR2_Fermion_delta11',
                 value = '-(2.0/Ncol)*11.0*complex(0,1)*RGR2',
                 order = {'QCD':4})

GC_4GR2_5Struct = Coupling(name = 'GC_4GR2_5Struct',
                 value = '5.0*complex(0,1)*RGR2',
                 order = {'QCD':4})

GC_4GR2_11Struct = Coupling(name = 'GC_4GR2_11Struct',
                 value = '-11.0*complex(0,1)*RGR2',
                 order = {'QCD':4})

#=============================================================================================

R2_GQQ = Coupling(name = 'R2_GQQ',
                value = '-complex(0,1)*G**3/(16.0*cmath.pi**2)*((Ncol**2-1)/(2.0*Ncol))*(1.0+lhv)',
                order = {'QCD':3})

R2_GGq = Coupling(name = 'R2_GGq',
                 value = 'complex(0,1)*G**2/(48.0*cmath.pi**2)',
                 order = {'QCD':2})

R2_GGb = Coupling(name = 'R2_GGb',
                 value = 'complex(0,1)*G**2*(-6.0*MB**2)/(48.0*cmath.pi**2)',
                 order = {'QCD':2})

R2_GGt = Coupling(name = 'R2_GGt',
                 value = 'complex(0,1)*G**2*(-6.0*MT**2)/(48.0*cmath.pi**2)',
                 order = {'QCD':2})

R2_GGg_1 = Coupling(name = 'R2_GGg_1',
                 value = 'complex(0,1)*G**2*Ncol/(48.0*cmath.pi**2)*(1.0/2.0+lhv)',
                 order = {'QCD':2})

R2_GGg_2 = Coupling(name = 'R2_GGg_2',
                 value = '-complex(0,1)*G**2*Ncol/(48.0*cmath.pi**2)*lhv',
                 order = {'QCD':2})

R2_QQq = Coupling(name = 'R2_QQq',
                 value =  'complex(0,1)*G**2*(Ncol**2-1)/(32.0*cmath.pi**2*Ncol)',
                 order = {'QCD':2})

R2_QQb = Coupling(name = 'R2_QQb',
                 value =  'complex(0,1)*G**2*(Ncol**2-1)*(-2.0*MB)/(32.0*cmath.pi**2*Ncol)',
                 order = {'QCD':2})

R2_QQt = Coupling(name = 'R2_QQt',
                 value =  'complex(0,1)*G**2*(Ncol**2-1)*(-2.0*MT)/(32.0*cmath.pi**2*Ncol)',
                 order = {'QCD':2})

################
# UV couplings #
################

UV_3Gg = Coupling(name = 'UV_3Gg',
                 value = '-G_UVg*G',
                 order = {'QCD':3})

UV_3Gq = Coupling(name = 'UV_3Gq',
                 value = '-G_UVq*G',
                 order = {'QCD':3})

UV_3Gb = Coupling(name = 'UV_3Gb',
                 value = '-G_UVb*G',
                 order = {'QCD':3})

UV_3Gt = Coupling(name = 'UV_3Gt',
                 value = '-G_UVt*G',
                 order = {'QCD':3})

UV_4Gg = Coupling(name = 'UV_4Gg',
                 value = '2.0*complex(0,1)*G_UVg*(G**2)',
                 order = {'QCD':4})

UV_4Gq = Coupling(name = 'UV_4Gq',
                 value = '2.0*complex(0,1)*G_UVq*(G**2)',
                 order = {'QCD':4})

UV_4Gb = Coupling(name = 'UV_4Gb',
                 value = '2.0*complex(0,1)*G_UVb*(G**2)',
                 order = {'QCD':4})

UV_4Gt = Coupling(name = 'UV_4Ggt',
                 value = '2.0*complex(0,1)*G_UVt*(G**2)',
                 order = {'QCD':4})

UV_GQQg = Coupling(name = 'UV_GQQg',
                 value = 'complex(0,1)*G_UVg*G',
                 order = {'QCD':3})

UV_GQQq = Coupling(name = 'UV_GQQq',
                 value = 'complex(0,1)*G_UVq*G',
                 order = {'QCD':3})

UV_GQQb = Coupling(name = 'UV_GQQb',
                 value = 'complex(0,1)*G_UVb*G',
                 order = {'QCD':3})

UV_GQQt = Coupling(name = 'UV_GQQt',
                 value = 'complex(0,1)*G_UVt*G',
                 order = {'QCD':3})

UV_bMass = Coupling(name = 'UV_bMass',
                 value = 'bMass_UV',
                 order = {'QCD':2}) 

UV_tMass = Coupling(name = 'UV_tMass',
                 value = 'tMass_UV',
                 order = {'QCD':2})
