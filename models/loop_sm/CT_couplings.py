# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51

from __future__ import absolute_import
from .object_library import all_couplings, Coupling
from .function_library import complexconjugate, re, im, csc, sec, acsc, asec

################
# R2 couplings #
################

# ========= #
# Pure QCD  #
# ========= #

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

# From the auto UFO from FR

R2GC_137_43 = Coupling(name = 'R2GC_137_43',
                       value = '(complex(0,1)*G**4)/(192.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_137_44 = Coupling(name = 'R2GC_137_44',
                       value = '-(complex(0,1)*G**4)/(64.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_138_45 = Coupling(name = 'R2GC_138_45',
                       value = '-(complex(0,1)*G**4)/(192.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_138_46 = Coupling(name = 'R2GC_138_46',
                       value = '(complex(0,1)*G**4)/(64.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_139_47 = Coupling(name = 'R2GC_139_47',
                       value = '-(complex(0,1)*G**4)/(48.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_140_48 = Coupling(name = 'R2GC_140_48',
                       value = '(complex(0,1)*G**4)/(288.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_140_49 = Coupling(name = 'R2GC_140_49',
                       value = '-(complex(0,1)*G**4)/(32.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_141_50 = Coupling(name = 'R2GC_141_50',
                       value = '-(complex(0,1)*G**4)/(16.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_141_51 = Coupling(name = 'R2GC_141_51',
                       value = '(-7*complex(0,1)*G**4)/(32.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_142_52 = Coupling(name = 'R2GC_142_52',
                       value = '(complex(0,1)*G**4)/(16.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_142_53 = Coupling(name = 'R2GC_142_53',
                       value = '(7*complex(0,1)*G**4)/(32.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_143_54 = Coupling(name = 'R2GC_143_54',
                       value = '(11*complex(0,1)*G**4)/(192.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_143_55 = Coupling(name = 'R2GC_143_55',
                       value = '(15*complex(0,1)*G**4)/(64.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_144_56 = Coupling(name = 'R2GC_144_56',
                       value = '(-3*complex(0,1)*G**4)/(64.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_144_57 = Coupling(name = 'R2GC_144_57',
                       value = '(-17*complex(0,1)*G**4)/(64.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_145_58 = Coupling(name = 'R2GC_145_58',
                       value = '-G**4/(192.*cmath.pi**2)',
                       order = {'QCD':4})

R2GC_145_59 = Coupling(name = 'R2GC_145_59',
                       value = 'G**4/(64.*cmath.pi**2)',
                       order = {'QCD':4})

#=============================================================================================

R2_GQQ = Coupling(name = 'R2_GQQ',
                value = '-complex(0,1)*G**3/(16.0*cmath.pi**2)*((Ncol**2-1)/(2.0*Ncol))*(1.0+lhv)',
                order = {'QCD':3})

R2_GGq = Coupling(name = 'R2_GGq',
                 value = '(2.0)*complex(0,1)*G**2/(48.0*cmath.pi**2)',
                 order = {'QCD':2})

R2_GGb = Coupling(name = 'R2_GGb',
                 value = '(2.0)*complex(0,1)*G**2*(-6.0*MB**2)/(48.0*cmath.pi**2)',
                 order = {'QCD':2})

R2_GGc = Coupling(name = 'R2_GGc',
                 value = '(2.0)*complex(0,1)*G**2*(-6.0*MC**2)/(48.0*cmath.pi**2)',
                 order = {'QCD':2})

R2_GGt = Coupling(name = 'R2_GGt',
                 value = '(2.0)*complex(0,1)*G**2*(-6.0*MT**2)/(48.0*cmath.pi**2)',
                 order = {'QCD':2})

R2_GGg_1 = Coupling(name = 'R2_GGg_1',
                 value = '(2.0)*complex(0,1)*G**2*Ncol/(48.0*cmath.pi**2)*(1.0/2.0+lhv)',
                 order = {'QCD':2})

R2_GGg_2 = Coupling(name = 'R2_GGg_2',
                 value = '-(2.0)*complex(0,1)*G**2*Ncol/(48.0*cmath.pi**2)*lhv',
                 order = {'QCD':2})

R2_QQq = Coupling(name = 'R2_QQq',
                 value =  'lhv*complex(0,1)*G**2*(Ncol**2-1)/(32.0*cmath.pi**2*Ncol)',
                 order = {'QCD':2})

R2_QQc = Coupling(name = 'R2_QQc',
                 value =  'lhv*complex(0,1)*G**2*(Ncol**2-1)*(2.0*MC)/(32.0*cmath.pi**2*Ncol)',
                 order = {'QCD':2})

R2_QQb = Coupling(name = 'R2_QQb',
                 value =  'lhv*complex(0,1)*G**2*(Ncol**2-1)*(2.0*MB)/(32.0*cmath.pi**2*Ncol)',
                 order = {'QCD':2})

R2_QQt = Coupling(name = 'R2_QQt',
                 value =  'lhv*complex(0,1)*G**2*(Ncol**2-1)*(2.0*MT)/(32.0*cmath.pi**2*Ncol)',
                 order = {'QCD':2})

# ============== #
# Mixed QCD-QED  #
# ============== #

# Quark couplings to A and Z

R2_DDA = Coupling(name = 'R2_DDA',
                value = '(-(ee*complex(0,1))/3.0)*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_UUA = Coupling(name = 'R2_UUA',
                value = '(2.0*(ee*complex(0,1))/3.0)*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_DDZ_V2 = Coupling(name = 'R2_DDZ_V2',
                value = '(-(cw*ee*complex(0,1))/(2.0*sw))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_DDZ_V3 = Coupling(name = 'R2_DDZ_V3',
                value = '(-(ee*complex(0,1)*sw)/(6.0*cw))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_UUZ_V2 = Coupling(name = 'R2_UUZ_V2',
                value = '((cw*ee*complex(0,1))/(2.*sw))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_UUZ_V5 = Coupling(name = 'R2_UUZ_V5',
                value = '(-(ee*complex(0,1)*sw)/(6.*cw))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

# Quark couplings to W, most general CKM

R2_dxuW = Coupling(name = 'R2_dxuW',
                value = '((CKM11*ee*complex(0,1))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_dxcW = Coupling(name = 'R2_dxcW',
                value = '((CKM21*ee*complex(0,1))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_dxtW = Coupling(name = 'R2_dxtW',
                value = '((CKM31*ee*complex(0,1))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_sxuW = Coupling(name = 'R2_sxuW',
                value = '((CKM12*ee*complex(0,1))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_sxcW = Coupling(name = 'R2_sxcW',
                value = '((CKM22*ee*complex(0,1))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_sxtW = Coupling(name = 'R2_sxtW',
                value = '((CKM32*ee*complex(0,1))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_bxuW = Coupling(name = 'R2_bxuW',
                value = '((CKM13*ee*complex(0,1))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_bxcW = Coupling(name = 'R2_bxcW',
                value = '((CKM23*ee*complex(0,1))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_bxtW = Coupling(name = 'R2_bxtW',
                value = '((CKM33*ee*complex(0,1))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_uxdW = Coupling(name = 'R2_uxdW',
                value = '((ee*complex(0,1)*complexconjugate(CKM11))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_cxdW = Coupling(name = 'R2_cxdW',
                value = '((ee*complex(0,1)*complexconjugate(CKM21))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_txdW = Coupling(name = 'R2_txdW',
                value = '((ee*complex(0,1)*complexconjugate(CKM31))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_uxsW = Coupling(name = 'R2_uxsW',
                value = '((ee*complex(0,1)*complexconjugate(CKM12))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_cxsW = Coupling(name = 'R2_cxsW',
                value = '((ee*complex(0,1)*complexconjugate(CKM22))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_txsW = Coupling(name = 'R2_txsW',
                value = '((ee*complex(0,1)*complexconjugate(CKM32))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_uxbW = Coupling(name = 'R2_uxbW',
                value = '((ee*complex(0,1)*complexconjugate(CKM13))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_cxbW = Coupling(name = 'R2_cxbW',
                value = '((ee*complex(0,1)*complexconjugate(CKM23))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

R2_txbW = Coupling(name = 'R2_txbW',
                value = '((ee*complex(0,1)*complexconjugate(CKM33))/(sw*cmath.sqrt(2)))*R2MixedFactor',
                order = {'QCD':2,'QED':1})

# R2 for the Higgs couplings to massive quarks 

R2_bbH = Coupling(name = 'R2_bbH',
                value = '(-((complex(0,1)*yb)/cmath.sqrt(2)))*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_ttH = Coupling(name = 'R2_ttH',
                value = '(-((complex(0,1)*yt)/cmath.sqrt(2)))*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_ccH = Coupling(name = 'R2_ccH',
                value = '(-((complex(0,1)*yc)/cmath.sqrt (2)))*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

# R2 interactions non proportional to the SM

# R2 for the Higgs interactions
R2_GGHc = Coupling(name = 'R2_GGHc',
                value = '4.0*(-((complex(0,1)*yc)/cmath.sqrt(2)))*(1.0/2.0)*(G**2/(8.0*cmath.pi**2))*MC',
                order = {'QCD':2,'QED':1})

R2_GGHb = Coupling(name = 'R2_GGHb',
                value = '4.0*(-((complex(0,1)*yb)/cmath.sqrt(2)))*(1.0/2.0)*(G**2/(8.0*cmath.pi**2))*MB',
                order = {'QCD':2,'QED':1})

R2_GGHt = Coupling(name = 'R2_GGHt',
                value = '4.0*(-((complex(0,1)*yt)/cmath.sqrt(2)))*(1.0/2.0)*(G**2/(8.0*cmath.pi**2))*MT',
                order = {'QCD':2,'QED':1})

# R2 for the weak vector bosons interaction with gluons

R2_GGZup = Coupling(name = 'R2_GGZup',
                    value = {0:'2.0*AxialZUp*(G**2/(12.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':1})

R2_GGZdown = Coupling(name = 'R2_GGZdown',
                    value = {0:'2.0*AxialZDown*(G**2/(12.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':1})

# EDIT VH
# There is a factor four added here to all the GGVV R2 couplings below without proper understanding of it
# I have not investigate too much about it though.
# END EDIT VH

R2_GGZAup = Coupling(name = 'R2_GGZAup',
                    value = {0:'4.0*(-VectorAUp*VectorZUp)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGZAdown = Coupling(name = 'R2_GGZAdown',
                    value = {0:'4.0*(-VectorADown*VectorZDown)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGZZdown = Coupling(name = 'R2_GGZZdown',
                    value = {0:'4.0*(-VectorZDown*VectorZDown-AxialZDown*AxialZDown)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGZZup = Coupling(name = 'R2_GGZZup',
                    value = {0:'4.0*(-VectorZUp*VectorZUp-AxialZUp*AxialZUp)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGAAdown = Coupling(name = 'R2_GGAAdown',
                    value = {0:'4.0*(-VectorADown*VectorADown)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGAAup = Coupling(name = 'R2_GGAAup',
                    value = {0:'4.0*(-VectorAUp*VectorAUp)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGWWud = Coupling(name = 'R2_GGWWud',
                    value = {0:'4.0*(CKM11*complexconjugate(CKM11))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGWWus = Coupling(name = 'R2_GGWWus',
                    value = {0:'4.0*(CKM12*complexconjugate(CKM12))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGWWub = Coupling(name = 'R2_GGWWub',
                    value = {0:'4.0*(CKM13*complexconjugate(CKM13))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGWWcd = Coupling(name = 'R2_GGWWcd',
                    value = {0:'4.0*(CKM21*complexconjugate(CKM21))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGWWcs = Coupling(name = 'R2_GGWWcs',
                    value = {0:'4.0*(CKM22*complexconjugate(CKM22))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGWWcb = Coupling(name = 'R2_GGWWcb',
                    value = {0:'4.0*(CKM23*complexconjugate(CKM23))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGWWtd = Coupling(name = 'R2_GGWWtd',
                    value = {0:'4.0*(CKM31*complexconjugate(CKM31))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGWWts = Coupling(name = 'R2_GGWWts',
                    value = {0:'4.0*(CKM32*complexconjugate(CKM32))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGWWtb = Coupling(name = 'R2_GGWWtb',
                    value = {0:'4.0*(CKM33*complexconjugate(CKM33))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGHHc = Coupling(name = 'R2_GGHHc',
                    value = {0:'4.0*(-yc**2/2.0)*(1.0/2.0)*((complex(0,1)*G**2)/(8.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGHHb = Coupling(name = 'R2_GGHHb',
                    value = {0:'4.0*(-yb**2/2.0)*(1.0/2.0)*((complex(0,1)*G**2)/(8.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGHHt = Coupling(name = 'R2_GGHHt',
                    value = {0:'4.0*(-yt**2/2.0)*(1.0/2.0)*((complex(0,1)*G**2)/(8.0*cmath.pi**2))'},
                    order = {'QCD':2,'QED':2})

R2_GGGZvecUp = Coupling(name = 'R2_GGGZvecUp',
                        value = {0:'complex(0,1)*VectorZUp*(-1.0/2.0)*(-G**3/(24.0*cmath.pi**2))'},
                        order = {'QCD':3,'QED':1})

R2_GGGZvecDown = Coupling(name = 'R2_GGGZvecDown',
                        value = {0:'complex(0,1)*VectorZDown*(-1.0/2.0)*(-G**3/(24.0*cmath.pi**2))'},
                        order = {'QCD':3,'QED':1})

R2_GGGZaxialUp = Coupling(name = 'R2_GGGZaxialUp',
                        value = {0:'complex(0,1)*AxialZUp*(-9.0/2.0)*(-G**3/(24.0*cmath.pi**2))'},
                        order = {'QCD':3,'QED':1})

R2_GGGZaxialDown = Coupling(name = 'R2_GGGZaxialDown',
                        value = {0:'complex(0,1)*AxialZDown*(-9.0/2.0)*(-G**3/(24.0*cmath.pi**2))'},
                        order = {'QCD':3,'QED':1})

R2_GGGAvecUp = Coupling(name = 'R2_GGGAvecUp',
                        value = {0:'complex(0,1)*VectorAUp*(-1.0/2.0)*(-G**3/(24.0*cmath.pi**2))'},
                        order = {'QCD':3,'QED':1})
R2_GGGAvecDown = Coupling(name = 'R2_GGGAvecDown',
                        value = {0:'complex(0,1)*VectorADown*(-1.0/2.0)*(-G**3/(24.0*cmath.pi**2))'},
                        order = {'QCD':3,'QED':1})

################
# UV couplings #
################

# ========= #
# Pure QCD  #
# ========= #

UV_3Gg = Coupling(name = 'UV_3Gg',
                 value = '-G_UVg*G',
                 order = {'QCD':3})

UV_3Gq = Coupling(name = 'UV_3Gq',
                 value = '-G_UVq*G',
                 order = {'QCD':3})

UV_3Gc = Coupling(name = 'UV_3Gc',
                 value = '-G_UVc*G',
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

UV_4Gc = Coupling(name = 'UV_4Gc',
                 value = '2.0*complex(0,1)*G_UVc*(G**2)',
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

UV_GQQc = Coupling(name = 'UV_GQQc',
                 value = 'complex(0,1)*G_UVc*G',
                 order = {'QCD':3})

UV_GQQb = Coupling(name = 'UV_GQQb',
                 value = 'complex(0,1)*G_UVb*G',
                 order = {'QCD':3})

UV_GQQt = Coupling(name = 'UV_GQQt',
                 value = 'complex(0,1)*G_UVt*G',
                 order = {'QCD':3})

UV_cMass = Coupling(name = 'UV_cMass',
                 value = 'cMass_UV',
                 order = {'QCD':2})

UV_bMass = Coupling(name = 'UV_bMass',
                 value = 'bMass_UV',
                 order = {'QCD':2}) 

UV_tMass = Coupling(name = 'UV_tMass',
                 value = 'tMass_UV',
                 order = {'QCD':2})

# ============== #
# Mixed QCD-QED  #
# ============== #

UV_Hcc = Coupling(name = 'UV_Hcc',
                 value = '-((complex(0,1)*yc)/cmath.sqrt(2))*UV_yuk_c',
                 order = {'QED':1,'QCD':2})

UV_Htt = Coupling(name = 'UV_Htt',
                 value = '-((complex(0,1)*yt)/cmath.sqrt(2))*UV_yuk_t',
                 order = {'QED':1,'QCD':2})

UV_Hbb = Coupling(name = 'UV_Hbb',
                 value = '-((complex(0,1)*yb)/cmath.sqrt(2))*UV_yuk_b',
                 order = {'QED':1,'QCD':2})

# ============================== #
# Goldstone R2 CT couplings      #
# ============================== #

R2_GGGpGm_ub = Coupling(name = 'R2_GGGpGm_ub',
                 value = 'complex(0,1)*R2_GGGpGm_factor*(Vector_ubGp*Vector_ubGm-Axial_ubGp*Axial_ubGm)',
                 order = {'QED':2,'QCD':2})

R2_GGGpGm_cd = Coupling(name = 'R2_GGGpGm_cd',
                 value = 'complex(0,1)*R2_GGGpGm_factor*(Vector_cdGp*Vector_cdGm-Axial_cdGp*Axial_cdGm)',
                 order = {'QED':2,'QCD':2})

R2_GGGpGm_cs = Coupling(name = 'R2_GGGpGm_cs',
                 value = 'complex(0,1)*R2_GGGpGm_factor*(Vector_csGp*Vector_csGm-Axial_csGp*Axial_csGm)',
                 order = {'QED':2,'QCD':2})

R2_GGGpGm_cb = Coupling(name = 'R2_GGGpGm_cb',
                 value = 'complex(0,1)*R2_GGGpGm_factor*(Vector_cbGp*Vector_cbGm-Axial_cbGp*Axial_cbGm)',
                 order = {'QED':2,'QCD':2})

R2_GGGpGm_td = Coupling(name = 'R2_GGGpGm_td',
                 value = 'complex(0,1)*R2_GGGpGm_factor*(Vector_tdGp*Vector_tdGm-Axial_tdGp*Axial_tdGm)',
                 order = {'QED':2,'QCD':2})

R2_GGGpGm_ts = Coupling(name = 'R2_GGGpGm_ts',
                 value = 'complex(0,1)*R2_GGGpGm_factor*(Vector_tsGp*Vector_tsGm-Axial_tsGp*Axial_tsGm)',
                 order = {'QED':2,'QCD':2})

R2_GGGpGm_tb = Coupling(name = 'R2_GGGpGm_tb',
                 value = 'complex(0,1)*R2_GGGpGm_factor*(Vector_tbGp*Vector_tbGm-Axial_tbGp*Axial_tbGm)',
                 order = {'QED':2,'QCD':2})

R2_GGG0G0_c = Coupling(name = 'R2_GGG0G0_c',
                       value = 'complex(0,1)*R2_GGG0G0_factor*(-(1.0/2.0)*yc**2)',
                       order = {'QED':2,'QCD':2})

R2_GGG0G0_b = Coupling(name = 'R2_GGG0G0_b',
                       value = 'complex(0,1)*R2_GGG0G0_factor*(-(1.0/2.0)*yb**2)',
                       order = {'QED':2,'QCD':2})

R2_GGG0G0_t = Coupling(name = 'R2_GGG0G0_t',
                       value = 'complex(0,1)*R2_GGG0G0_factor*(-(1.0/2.0)*yt**2)',
                       order = {'QED':2,'QCD':2})

GC_R2_1015 = Coupling(name = 'GC_R2_1015',
                 value = 'I1x33*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1021 = Coupling(name = 'GC_R2_1021',
                 value = '-I2x33*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1027 = Coupling(name = 'GC_R2_1027',
                 value = 'I3x33*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1030 = Coupling(name = 'GC_R2_1030',
                 value = '-I4x33*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1082 = Coupling(name = 'GC_R2_1082',
                 value = '-(yb/cmath.sqrt(2))*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1085 = Coupling(name = 'GC_R2_1085',
                 value = '(yc/cmath.sqrt(2))*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1095 = Coupling(name = 'GC_R2_1095',
                 value = '(yt/cmath.sqrt(2))*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1018 = Coupling(name = 'GC_R2_1018',
                 value = '(-I2x22)*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1019 = Coupling(name = 'GC_R2_1019',
                 value = '(-I2x23)*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1013 = Coupling(name = 'GC_R2_1013',
                 value = '(I1x31)*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1014 = Coupling(name = 'GC_R2_1014',
                 value = '(I1x32)*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1020 = Coupling(name = 'GC_R2_1020',
                 value = '(-I2x32)*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1028 = Coupling(name = 'GC_R2_1028',
                 value = '(-I4x13)*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1022 = Coupling(name = 'GC_R2_1022',
                 value = '(I3x21)*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1023 = Coupling(name = 'GC_R2_1023',
                 value = '(I3x22)*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1024 = Coupling(name = 'GC_R2_1024',
                 value = '(I3x23)*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1029 = Coupling(name = 'GC_R2_1029',
                 value = '(-I4x23)*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1025 = Coupling(name = 'GC_R2_1025',
                 value = '(I3x31)*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

GC_R2_1026 = Coupling(name = 'GC_R2_1026',
                 value = '(I3x32)*(R2MixedFactor*2.0)',
                 order = {'QED':1,'QCD':2})

# ============================== #
# Goldstone UV CT couplings      #
# ============================== #

GC_UV_1015 = Coupling(name = 'GC_UV_1015',
                 value = 'I1x33*UV_yuk_b',
                 order = {'QED':1,'QCD':2})

GC_UV_1021 = Coupling(name = 'GC_UV_1021',
                 value = '-I2x33*UV_yuk_t',
                 order = {'QED':1,'QCD':2})

GC_UV_1027 = Coupling(name = 'GC_UV_1027',
                 value = 'I3x33*UV_yuk_t',
                 order = {'QED':1,'QCD':2})

GC_UV_1030 = Coupling(name = 'GC_UV_1030',
                 value = '-I4x33*UV_yuk_b',
                 order = {'QED':1,'QCD':2})

GC_UV_1082 = Coupling(name = 'GC_UV_1082',
                 value = '-(yb/cmath.sqrt(2))*(UV_yuk_b)',
                 order = {'QED':1,'QCD':2})

GC_UV_1085 = Coupling(name = 'GC_UV_1085',
                 value = '(yc/cmath.sqrt(2))*(UV_yuk_c)',
                 order = {'QED':1,'QCD':2})

GC_UV_1095 = Coupling(name = 'GC_UV_1095',
                 value = '(yt/cmath.sqrt(2))*(UV_yuk_t)',
                 order = {'QED':1,'QCD':2})

GC_UV_1018 = Coupling(name = 'GC_UV_1018',
                 value = '(-I2x22)*(UV_yuk_c)',
                 order = {'QED':1,'QCD':2})

GC_UV_1019 = Coupling(name = 'GC_UV_1019',
                 value = '(-I2x23)*(UV_yuk_t)',
                 order = {'QED':1,'QCD':2})

GC_UV_1013 = Coupling(name = 'GC_UV_1013',
                 value = '(I1x31)*(UV_yuk_b)',
                 order = {'QED':1,'QCD':2})

GC_UV_1014 = Coupling(name = 'GC_UV_1014',
                 value = '(I1x32)*(UV_yuk_b)',
                 order = {'QED':1,'QCD':2})

GC_UV_1020 = Coupling(name = 'GC_UV_1020',
                 value = '(-I2x32)*(UV_yuk_c)',
                 order = {'QED':1,'QCD':2})

GC_UV_1028 = Coupling(name = 'GC_UV_1028',
                 value = '(-I4x13)*(UV_yuk_b)',
                 order = {'QED':1,'QCD':2})

GC_UV_1022 = Coupling(name = 'GC_UV_1022',
                 value = '(I3x21)*(UV_yuk_c)',
                 order = {'QED':1,'QCD':2})

GC_UV_1023 = Coupling(name = 'GC_UV_1023',
                 value = '(I3x22)*(UV_yuk_c)',
                 order = {'QED':1,'QCD':2})

GC_UV_1024 = Coupling(name = 'GC_UV_1024',
                 value = '(I3x23)*(UV_yuk_c)',
                 order = {'QED':1,'QCD':2})

GC_UV_1029 = Coupling(name = 'GC_UV_1029',
                 value = '(-I4x23)*(UV_yuk_b)',
                 order = {'QED':1,'QCD':2})

GC_UV_1025 = Coupling(name = 'GC_UV_1025',
                 value = '(I3x31)*(UV_yuk_t)',
                 order = {'QED':1,'QCD':2})

GC_UV_1026 = Coupling(name = 'GC_UV_1026',
                 value = '(I3x32)*(UV_yuk_t)',
                 order = {'QED':1,'QCD':2})
