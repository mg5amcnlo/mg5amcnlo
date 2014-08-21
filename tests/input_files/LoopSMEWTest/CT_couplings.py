# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51

from object_library import all_couplings, Coupling

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, arg, reglog

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
                value = '-complex(0,1)*G**3/(16.0*cmath.pi**2)*((Ncol**2 -1)/(2.0*Ncol))*(1.0+lhv)',
                order = {'QCD':3})

R2_GGq = Coupling(name = 'R2_GGq',
                 value = '(2.0)*complex(0,1)*G**2/(48.0*cmath.pi**2)',
                 order = {'QCD':2})

R2_GGb = Coupling(name = 'R2_GGb',
                 value = '(2.0)*complex(0,1)*G**2*(-6.0*MB**2)/(48.0*cmath.pi**2)',
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
                 value =  'lhv*complex(0,1)*G**2*(Ncol**2 -1.0)/(32.0*cmath.pi**2*Ncol)',
                 order = {'QCD':2})

R2_QQb = Coupling(name = 'R2_QQb',
                 value =  'lhv*complex(0,1)*G**2*(Ncol**2 -1.0)*(2.0*MB)/(32.0*cmath.pi**2*Ncol)',
                 order = {'QCD':2})

R2_QQt = Coupling(name = 'R2_QQt',
                 value =  'lhv*complex(0,1)*G**2*(Ncol**2 -1.0)*(2.0*MT)/(32.0*cmath.pi**2*Ncol)',
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

# R2 for SQQ~

R2_bbG0 = Coupling(name = 'R2_bbG0',
                value = '-AxialG0Down3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_ttG0 = Coupling(name = 'R2_ttG0',
                value = '-AxialG0Up3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_ccG0 = Coupling(name = 'R2_ccG0',
                value = '-AxialG0Up2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_uuG0 = Coupling(name = 'R2_uuG0',
                value = '-AxialG0Up1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_ddG0 = Coupling(name = 'R2_ddG0',
                value = '-AxialG0Down1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_ssG0 = Coupling(name = 'R2_ssG0',
                value = '-AxialG0Down2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1}) 

R2_bbH = Coupling(name = 'R2_bbH',
                value = 'VectorHDown3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_ttH = Coupling(name = 'R2_ttH',
                value = 'VectorHUp3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_ccH = Coupling(name = 'R2_ccH',
                value = 'VectorHUp2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_uuH = Coupling(name = 'R2_uuH',
                value = 'VectorHUp1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_ddH = Coupling(name = 'R2_ddH',
                value = 'VectorHDown1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_ssH = Coupling(name = 'R2_ssH',
                value = 'VectorHDown2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_uxdGp = Coupling(name = 'R2_uxdGp',
                value = 'VectorGpUx1D1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_uxdGpA = Coupling(name = 'R2_uxdGpA',
                value = '-AxialGpUx1D1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_uxsGp = Coupling(name = 'R2_uxsGp',
                value = 'VectorGpUx1D2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_uxsGpA = Coupling(name = 'R2_uxsGpA',
                value = '-AxialGpUx1D2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_uxbGp = Coupling(name = 'R2_uxbGp',
                value = 'VectorGpUx1D3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_uxbGpA = Coupling(name = 'R2_uxbGpA',
                value = '-AxialGpUx1D3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})


R2_cxdGp = Coupling(name = 'R2_cxdGp',
                value = 'VectorGpUx2D1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_cxdGpA = Coupling(name = 'R2_cxdGpA',
                value = '-AxialGpUx2D1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_cxsGp = Coupling(name = 'R2_cxsGp',
                value = 'VectorGpUx2D2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_cxsGpA = Coupling(name = 'R2_cxsGpA',
                value = '-AxialGpUx2D2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_cxbGp = Coupling(name = 'R2_cxbGp',
                value = 'VectorGpUx2D3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_cxbGpA = Coupling(name = 'R2_cxbGpA',
                value = '-AxialGpUx2D3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_txdGp = Coupling(name = 'R2_txdGp',
                value = 'VectorGpUx3D1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_txdGpA = Coupling(name = 'R2_txdGpA',
                value = '-AxialGpUx3D1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_txsGp = Coupling(name = 'R2_txsGp',
                value = 'VectorGpUx3D2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_txsGpA = Coupling(name = 'R2_txsGpA',
                value = '-AxialGpUx3D2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_txbGp = Coupling(name = 'R2_txbGp',
                value = 'VectorGpUx3D3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_txbGpA = Coupling(name = 'R2_txbGpA',
                value = '-AxialGpUx3D3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_dxuGm = Coupling(name = 'R2_dxuGm',
                value = 'VectorGmDx1U1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_dxuGmA = Coupling(name = 'R2_dxuGmA',
                value = '-AxialGmDx1U1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_sxuGm = Coupling(name = 'R2_sxuGm',
                value = 'VectorGmDx2U1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_sxuGmA = Coupling(name = 'R2_sxuGmA',
                value = '-AxialGmDx2U1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_bxuGm = Coupling(name = 'R2_bxuGm',
                value = 'VectorGmDx3U1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_bxuGmA = Coupling(name = 'R2_bxuGmA',
                value = '-AxialGmDx3U1*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})


R2_dxcGm = Coupling(name = 'R2_dxcGm',
                value = 'VectorGmDx1U2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_dxcGmA = Coupling(name = 'R2_dxcGmA',
                value = '-AxialGmDx1U2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_sxcGm = Coupling(name = 'R2_cxsGm',
                value = 'VectorGmDx2U2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_sxcGmA = Coupling(name = 'R2_sxcGmA',
                value = '-AxialGmDx2U2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_bxcGm = Coupling(name = 'R2_bxcGm',
                value = 'VectorGmDx3U2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_bxcGmA = Coupling(name = 'R2_bxcGmA',
                value = '-AxialGmDx3U2*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_dxtGm = Coupling(name = 'R2_dxtGm',
                value = 'VectorGmDx1U3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_dxtGmA = Coupling(name = 'R2_dxtGmA',
                value = '-AxialGmDx1U3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_sxtGm = Coupling(name = 'R2_sxtGm',
                value = 'VectorGmDx2U3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_sxtGmA = Coupling(name = 'R2_sxtGmA',
                value = '-AxialGmDx2U3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_bxtGm = Coupling(name = 'R2_bxtGm',
                value = 'VectorGmDx3U3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

R2_bxtGmA = Coupling(name = 'R2_bxtGmA',
                value = '-AxialGmDx3U3*(2.0*R2MixedFactor)',
                order = {'QCD':2,'QED':1})

# R2 interactions non proportional to the SM

# R2 for GGH

R2_GGHu = Coupling(name = 'R2_GGHu',
                value = '2.0*VectorHUp1*(G**2/(8.0*cmath.pi**2))*MU',
                order = {'QCD':2,'QED':1})

R2_GGHd = Coupling(name = 'R2_GGHd',
                value = '2.0*VectorHDown1*(G**2/(8.0*cmath.pi**2))*MD',
                order = {'QCD':2,'QED':1})

R2_GGHs = Coupling(name = 'R2_GGHs',
                value = '2.0*VectorHDown2*(G**2/(8.0*cmath.pi**2))*MS',
                order = {'QCD':2,'QED':1})

R2_GGHc = Coupling(name = 'R2_GGHc',
                value = '2.0*VectorHUp2*(G**2/(8.0*cmath.pi**2))*MC',
                order = {'QCD':2,'QED':1})

R2_GGHb = Coupling(name = 'R2_GGHb',
                value = '2.0*VectorHDown3*(G**2/(8.0*cmath.pi**2))*MB',
                order = {'QCD':2,'QED':1})

R2_GGHt = Coupling(name = 'R2_GGHt',
                value = '2.0*VectorHUp3*(G**2/(8.0*cmath.pi**2))*MT',
                order = {'QCD':2,'QED':1})

# R2 for the weak vector bosons interaction with gluons

# R2 for GGZ

R2_GGZup = Coupling(name = 'R2_GGZup',
                    value = '4.0*AxialZUp*(1.0/2.0)*(G**2/(12.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':1})

R2_GGZdown = Coupling(name = 'R2_GGZdown',
                    value = '4.0*AxialZDown*(1.0/2.0)*(G**2/(12.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':1})

# R2 for GGVV

R2_GGZAup = Coupling(name = 'R2_GGZAup',
                    value = '4.0*(-VectorAUp*VectorZUp)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGZAdown = Coupling(name = 'R2_GGZAdown',
                    value = '4.0*(-VectorADown*VectorZDown)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGZZdown = Coupling(name = 'R2_GGZZdown',
                    value = '4.0*(-VectorZDown*VectorZDown-AxialZDown*AxialZDown)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGZZup = Coupling(name = 'R2_GGZZup',
                    value = '4.0*(-VectorZUp*VectorZUp-AxialZUp*AxialZUp)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGAAdown = Coupling(name = 'R2_GGAAdown',
                    value = '4.0*(-VectorADown*VectorADown)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGAAup = Coupling(name = 'R2_GGAAup',
                    value = '4.0*(-VectorAUp*VectorAUp)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGWWud = Coupling(name = 'R2_GGWWud',
                    value = '4.0*(CKM11*complexconjugate(CKM11))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGWWus = Coupling(name = 'R2_GGWWus',
                    value = '4.0*(CKM12*complexconjugate(CKM12))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGWWub = Coupling(name = 'R2_GGWWub',
                    value = '4.0*(CKM13*complexconjugate(CKM13))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGWWcd = Coupling(name = 'R2_GGWWcd',
                    value = '4.0*(CKM21*complexconjugate(CKM21))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGWWcs = Coupling(name = 'R2_GGWWcs',
                    value = '4.0*(CKM22*complexconjugate(CKM22))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGWWcb = Coupling(name = 'R2_GGWWcb',
                    value = '4.0*(CKM23*complexconjugate(CKM23))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGWWtd = Coupling(name = 'R2_GGWWtd',
                    value = '4.0*(CKM31*complexconjugate(CKM31))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGWWts = Coupling(name = 'R2_GGWWts',
                    value = '4.0*(CKM32*complexconjugate(CKM32))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGWWtb = Coupling(name = 'R2_GGWWtb',
                    value = '4.0*(CKM33*complexconjugate(CKM33))*(-VectorWmDxU*VectorWpUxD-AxialWmDxU*AxialWpUxD)*(1.0/2.0)*(-(complex(0,1)*G**2)/(24.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

# R2 for SSGG

R2_GGHHu = Coupling(name = 'R2_GGHHu',
                    value = '2.0*VectorHUp1*VectorHUp1*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGHHd = Coupling(name = 'R2_GGHHd',
                    value = '2.0*VectorHDown1*VectorHDown1*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGHHs = Coupling(name = 'R2_GGHHs',
                    value = '2.0*VectorHDown2*VectorHDown2*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGHHc = Coupling(name = 'R2_GGHHc',
                    value = '2.0*VectorHUp2*VectorHUp2*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGHHb = Coupling(name = 'R2_GGHHb',
                    value = '2.0*VectorHDown3*VectorHDown3*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGHHt = Coupling(name = 'R2_GGHHt',
                    value = '2.0*VectorHUp3*VectorHUp3*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGG0G0u = Coupling(name = 'R2_GGG0G0u',
                    value = '-2.0*AxialG0Up1*AxialG0Up1*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGG0G0d = Coupling(name = 'R2_GGG0G0d',
                    value = '-2.0*AxialG0Down1*AxialG0Down1*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGG0G0s = Coupling(name = 'R2_GGG0G0s',
                    value = '-2.0*AxialG0Down2*AxialG0Down2*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGG0G0c = Coupling(name = 'R2_GGG0G0c',
                    value = '-2.0*AxialG0Up2*AxialG0Up2*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGG0G0b = Coupling(name = 'R2_GGG0G0b',
                    value = '-2.0*AxialG0Down3*AxialG0Down3*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGG0G0t = Coupling(name = 'R2_GGG0G0t',
                    value = '-2.0*AxialG0Up3*AxialG0Up3*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGGmGpud = Coupling(name = 'R2_GGGmGpud',
                    value = '2.0*(VectorGpUx1D1*VectorGmDx1U1-AxialGpUx1D1*AxialGmDx1U1)*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGGmGpus = Coupling(name = 'R2_GGGmGpus',
                    value = '2.0*(VectorGpUx1D2*VectorGmDx2U1-AxialGpUx1D2*AxialGmDx2U1)*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGGmGpub = Coupling(name = 'R2_GGGmGpub',
                    value = '2.0*(VectorGpUx1D3*VectorGmDx3U1-AxialGpUx1D3*AxialGmDx3U1)*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGGmGpcd = Coupling(name = 'R2_GGGmGpcd',
                    value = '2.0*(VectorGpUx2D1*VectorGmDx1U2-AxialGpUx2D1*AxialGmDx1U2)*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGGmGpcs = Coupling(name = 'R2_GGGmGpcs',
                    value = '2.0*(VectorGpUx2D2*VectorGmDx2U2-AxialGpUx2D2*AxialGmDx2U2)*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGGmGpcb = Coupling(name = 'R2_GGGmGpcb',
                    value = '2.0*(VectorGpUx2D3*VectorGmDx3U2-AxialGpUx2D3*AxialGmDx3U2)*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGGmGptd = Coupling(name = 'R2_GGGmGptd',
                    value = '2.0*(VectorGpUx3D1*VectorGmDx1U3-AxialGpUx3D1*AxialGmDx1U3)*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGGmGpts = Coupling(name = 'R2_GGGmGpts',
                    value = '2.0*(VectorGpUx3D2*VectorGmDx2U3-AxialGpUx3D2*AxialGmDx2U3)*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

R2_GGGmGptb = Coupling(name = 'R2_GGGmGptb',
                    value = '2.0*(VectorGpUx3D3*VectorGmDx3U3-AxialGpUx3D3*AxialGmDx3U3)*((complex(0,1)*G**2)/(8.0*cmath.pi**2))',
                    order = {'QCD':2,'QED':2})

# R2 for VGGG

R2_GGGZvecUp = Coupling(name = 'R2_GGGZvecUp',
                        value = 'complex(0,1)*VectorZUp*(-1.0/2.0)*(-G**3/(24.0*cmath.pi**2))',
                        order = {'QCD':3,'QED':1})

R2_GGGZvecDown = Coupling(name = 'R2_GGGZvecDown',
                        value = 'complex(0,1)*VectorZDown*(-1.0/2.0)*(-G**3/(24.0*cmath.pi**2))',
                        order = {'QCD':3,'QED':1})

R2_GGGZaxialUp = Coupling(name = 'R2_GGGZaxialUp',
                        value = 'complex(0,1)*AxialZUp*(-9.0/2.0)*(-G**3/(24.0*cmath.pi**2))',
                        order = {'QCD':3,'QED':1})

R2_GGGZaxialDown = Coupling(name = 'R2_GGGZaxialDown',
                        value = 'complex(0,1)*AxialZDown*(-9.0/2.0)*(-G**3/(24.0*cmath.pi**2))',
                        order = {'QCD':3,'QED':1})

R2_GGGAvecUp = Coupling(name = 'R2_GGGAvecUp',
                        value = 'complex(0,1)*VectorAUp*(-1.0/2.0)*(-G**3/(24.0*cmath.pi**2))',
                        order = {'QCD':3,'QED':1})

R2_GGGAvecDown = Coupling(name = 'R2_GGGAvecDown',
                        value = 'complex(0,1)*VectorADown*(-1.0/2.0)*(-G**3/(24.0*cmath.pi**2))',
                        order = {'QCD':3,'QED':1})

# ========= #
# Pure QED  #
# ========= #

# R2 for SS

R2_HHboson1 = Coupling(name = 'R2_HHboson1',
                       value = '((MW**2/4.0+MZ**2/(8.0*cw**2)+(1-12*lhv)/4.0*(1+1./(2.*cw**4))*MW**2)*(complex(0,1)*R2SS))',
                       order = {'QED':2})

R2_HHboson2 = Coupling(name = 'R2_HHboson2',
                       value = '((-1./12.0-1./(24.0*cw**2))*(complex(0,1)*R2SS))',
                       order = {'QED':2})

R2_HHe1 = Coupling(name = 'R2_HHe1',
                   value = 'Me**4/MW**2*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHe2 = Coupling(name = 'R2_HHe2',
                   value = 'Me**2/MW**2*(-1./6.)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHm1 = Coupling(name = 'R2_HHm1',
                   value = 'MM**4/MW**2*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHm2 = Coupling(name = 'R2_HHm2',
                   value = 'MM**2/MW**2*(-1./6.)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHtau1 = Coupling(name = 'R2_HHtau1',
                   value = 'MTA**4/MW**2*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHtau2 = Coupling(name = 'R2_HHtau2',
                   value = 'MTA**2/MW**2*(-1./6.)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHu1 = Coupling(name = 'R2_HHu1',
                   value = 'MU**4/MW**2*(Ncol)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHu2 = Coupling(name = 'R2_HHu2',
                   value = 'MU**2/MW**2*(-Ncol/6.0)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHd1 = Coupling(name = 'R2_HHd1',
                   value = 'MD**4/MW**2*(Ncol)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHd2 = Coupling(name = 'R2_HHd2',
                   value = 'MD**2/MW**2*(-Ncol/6.0)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHs1 = Coupling(name = 'R2_HHs1',
                   value = 'MS**4/MW**2*(Ncol)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHs2 = Coupling(name = 'R2_HHs2',
                   value = 'MS**2/MW**2*(-Ncol/6.0)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHc1 = Coupling(name = 'R2_HHc1',
                   value = 'MC**4/MW**2*(Ncol)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHc2 = Coupling(name = 'R2_HHc2',
                   value = 'MC**2/MW**2*(-Ncol/6.0)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHb1 = Coupling(name = 'R2_HHb1',
                   value = 'MB**4/MW**2*(Ncol)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHb2 = Coupling(name = 'R2_HHb2',
                   value = 'MB**2/MW**2*(-Ncol/6.0)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHt1 = Coupling(name = 'R2_HHt1',
                   value = 'MT**4/MW**2*(Ncol)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_HHt2 = Coupling(name = 'R2_HHt2',
                   value = 'MT**2/MW**2*(-Ncol/6.0)*(complex(0,1)*R2SS)',
                   order = {'QED':2})

R2_G0G0boson1 = Coupling(name = 'R2_G0G0boson1',
                       value = '((MW**2/4.0+MH**2/(8.0*cw**2)+(1-4*lhv)/4.0*(1+1./(2.*cw**4))*MW**2)*(complex(0,1)*R2SS))',
                       order = {'QED':2})

R2_G0G0boson2 = Coupling(name = 'R2_G0G0boson2',
                       value = '((-1./12.0-1./(24.0*cw**2))*(complex(0,1)*R2SS))',
                       order = {'QED':2})


R2_GmGpboson1 = Coupling(name = 'R2_GmGpboson1',
                       value = '(((MZ**2 + MH**2)/8.0+MW**2/(8.0*cw**2)+((6-8*lhv)*cw**4 - 4*cw**2 + (1-4*lhv))/(8.0*cw**4)*MW**2)*(complex(0,1)*R2SS))',
                       order = {'QED':2})

R2_GmGpboson2 = Coupling(name = 'R2_GmGpboson2',
                       value = '((-1./12.0-1./(24.0*cw**2))*(complex(0,1)*R2SS))',
                       order = {'QED':2})


R2_GmGpe = Coupling(name = 'R2_GmGpe',
                     value = 'Me**4/(2.0*MW**2)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpm = Coupling(name = 'R2_GmGpm',
                     value = 'MM**4/(2.0*MW**2)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGptau = Coupling(name = 'R2_GmGptau',
                     value = 'MTA**4/(2.0*MW**2)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpud1 = Coupling(name = 'R2_GmGpud1',
                     value = 'Ncol*(MU**2 + MD**2)**2/(2.0*MW**2)*(SCKM11)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpud2 = Coupling(name = 'R2_GmGpud2',
                     value = '-Ncol*(MU**2 +MD**2)/(6.0*MW**2)*(SCKM11)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpus1 = Coupling(name = 'R2_GmGpus1',
                     value = 'Ncol*(MU**2 +MS**2)**2/(2.0*MW**2)*(SCKM12)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpus2 = Coupling(name = 'R2_GmGpus2',
                     value = '-Ncol*(MU**2 +MS**2)/(6.0*MW**2)*(SCKM12)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpub1 = Coupling(name = 'R2_GmGpub1',
                     value = 'Ncol*(MU**2 +MB**2)**2/(2.0*MW**2)*(SCKM13)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpub2 = Coupling(name = 'R2_GmGpub2',
                     value = '-Ncol*(MU**2 +MB**2)/(6.0*MW**2)*(SCKM13)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpcd1 = Coupling(name = 'R2_GmGpcd1',
                     value = 'Ncol*(MC**2 +MD**2)**2/(2.0*MW**2)*(SCKM21)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpcd2 = Coupling(name = 'R2_GmGpcd2',
                     value = '-Ncol*(MC**2 +MD**2)/(6.0*MW**2)*(SCKM21)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpcs1 = Coupling(name = 'R2_GmGpcs1',
                     value = 'Ncol*(MC**2 +MS**2)**2/(2.0*MW**2)*(SCKM22)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpcs2 = Coupling(name = 'R2_GmGpcs2',
                     value = '-Ncol*(MC**2 +MS**2)/(6.0*MW**2)*(SCKM22)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpcb1 = Coupling(name = 'R2_GmGpcb1',
                     value = 'Ncol*(MC**2 +MB**2)**2/(2.0*MW**2)*(SCKM23)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpcb2 = Coupling(name = 'R2_GmGpcb2',
                     value = '-Ncol*(MC**2 +MB**2)/(6.0*MW**2)*(SCKM23)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGptd1 = Coupling(name = 'R2_GmGptd1',
                     value = 'Ncol*(MT**2 +MD**2)**2/(2.0*MW**2)*(SCKM31)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGptd2 = Coupling(name = 'R2_GmGptd2',
                     value = '-Ncol*(MT**2 +MD**2)/(6.0*MW**2)*(SCKM31)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpts1 = Coupling(name = 'R2_GmGpts1',
                     value = 'Ncol*(MT**2 +MS**2)**2/(2.0*MW**2)*(SCKM32)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGpts2 = Coupling(name = 'R2_GmGpts2',
                     value = '-Ncol*(MT**2 +MS**2)/(6.0*MW**2)*(SCKM32)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGptb1 = Coupling(name = 'R2_GmGptb1',
                     value = 'Ncol*(MT**2 +MB**2)**2/(2.0*MW**2)*(SCKM33)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

R2_GmGptb2 = Coupling(name = 'R2_GmGptb2',
                     value = '-Ncol*(MT**2 +MB**2)/(6.0*MW**2)*(SCKM33)*(complex(0,1)*R2SS)',
                     order = {'QED':2})

# R2 for VV

R2_AAC1 = Coupling(name = 'R2_AAC1',
                   value = '-lhv/24.*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AAboson1 = Coupling(name = 'R2_AAboson1',
                       value = '((1./48.+lhv/24.)*(complex(0,1)*R2VV))',
                       order = {'QED':2})

R2_AAboson2 = Coupling(name = 'R2_AAboson2',
                   value = '(-lhv/24.*(complex(0,1)*R2VV))',
                   order = {'QED':2})


R2_AAboson3 = Coupling(name = 'R2_AAboson3',
                       value = '(-MW**2/8.*(complex(0,1)*R2VV))',
                       order = {'QED':2})

R2_AAl = Coupling(name = 'R2_AAl',
                   value = '1./24.*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AAe3 = Coupling(name = 'R2_AAe3',
                   value = '-Me**2/4.*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AAm3 = Coupling(name = 'R2_AAm3',
                   value = '-MM**2/4.*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AAtau3 = Coupling(name = 'R2_AAtau3',
                   value = '-MTA**2/4.*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AAU = Coupling(name = 'R2_AAU',
                  value = 'Ncol/24.*(4./9.)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_AAD = Coupling(name = 'R2_AAD',
                  value = 'Ncol/24.*(1./9.)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_AAu3 = Coupling(name= 'R2_AAu3',
                   value = '-MU**2/4.*(4./9.*Ncol)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AAc3 = Coupling(name= 'R2_AAc3',
                   value = '-MC**2/4.*(4./9.*Ncol)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AAt3 = Coupling(name= 'R2_AAt3',
                   value = '-MT**2/4.*(4./9.*Ncol)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AAd3 = Coupling(name= 'R2_AAd3',
                   value = '-MD**2/4.*(1./9.*Ncol)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AAs3 = Coupling(name= 'R2_AAs3',
                   value = '-MS**2/4.*(1./9.*Ncol)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AAb3 = Coupling(name= 'R2_AAb3',
                   value = '-MB**2/4.*(1./9.*Ncol)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AZC1 = Coupling(name = 'R2_AZC1',
                   value = '-lhv*cw/(24.*sw)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AZboson1 = Coupling(name = 'R2_AZboson1',
                       value = '((1./48.+lhv/24.)*(cw/sw)*(complex(0,1)*R2VV))',
                       order = {'QED':2})

R2_AZboson2 = Coupling(name = 'R2_AZboson2',
                   value = '(-lhv*cw/(24.*sw)*(complex(0,1)*R2VV))',
                   order = {'QED':2})

R2_AZboson3 = Coupling(name = 'R2_AZboson3',
                       value = '(-MW**2/8.*(cw/sw)*(complex(0,1)*R2VV))',
                       order = {'QED':2})

R2_AZl = Coupling(name = 'R2_AZl',
                   value = '1./(4.*cw)*(1./(24.0*sw)-sw/6.)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AZe3 = Coupling(name = 'R2_AZe3',
                   value = 'Me**2/(4.*cw)*(-1./(4.0*sw)+sw)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AZm3 = Coupling(name = 'R2_AZm3',
                   value = 'MM**2/(4.*cw)*(-1./(4.0*sw)+sw)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AZtau3 = Coupling(name = 'R2_AZtau3',
                   value = 'MTA**2/(4.*cw)*(-1./(4.0*sw)+sw)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AZU = Coupling(name = 'R2_AZU',
                  value = 'Ncol/(4.*cw)*(1./(36.*sw)-2./27.*sw)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_AZD = Coupling(name = 'R2_AZD',
                  value = 'Ncol/(4.*cw)*(1./(72.*sw)-sw/54.)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_AZu3 = Coupling(name= 'R2_AZu3',
                   value = 'MU**2/(4.*cw)*Ncol*(4.0*sw/9.0-1./(6.0*sw))*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AZc3 = Coupling(name= 'R2_AZc3',
                   value = 'MC**2/(4.*cw)*Ncol*(4.0*sw/9.0-1./(6.0*sw))*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AZt3 = Coupling(name= 'R2_AZt3',
                   value = 'MT**2/(4.*cw)*Ncol*(4.0*sw/9.0-1./(6.0*sw))*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AZd3 = Coupling(name= 'R2_AZd3',
                   value = 'MD**2/(4.*cw)*Ncol*(sw/9.-1./(12.0*sw))*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AZs3 = Coupling(name= 'R2_AZs3',
                   value = 'MS**2/(4.*cw)*Ncol*(sw/9.-1./(12.0*sw))*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_AZb3 = Coupling(name= 'R2_AZb3',
                   value = 'MB**2/(4.*cw)*Ncol*(sw/9.-1./(12.0*sw))*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_ZZC1 = Coupling(name = 'R2_ZZC1',
                   value = '-lhv*cw**2/(24.*sw**2)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_ZZboson1 = Coupling(name = 'R2_ZZboson1',
                       value = '(1./48.+lhv/24.)*(cw**2/sw**2)*(complex(0,1)*R2VV)',
                       order = {'QED':2})

R2_ZZboson2 = Coupling(name = 'R2_ZZboson2',
                   value = '(-lhv*cw**2/(24.*sw**2)*(complex(0,1)*R2VV))',
                   order = {'QED':2})



R2_ZZboson3 = Coupling(name = 'R2_ZZboson3',
                       value = '(-MW**2/8.*(cw**2/sw**2)*(complex(0,1)*R2VV))',
                       order = {'QED':2})

R2_ZZl = Coupling(name = 'R2_ZZl',
                   value = '-1./(24.*cw**2)*(1./2.-sw**2 -1./(8.0*sw**2))*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_ZZe3 = Coupling(name = 'R2_ZZe3',
                   value = 'Me**2/(4.*cw**2)*(1./2.-sw**2 -1./(8.0*sw**2))*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_ZZm3 = Coupling(name = 'R2_ZZm3',
                   value = 'MM**2/(4.*cw**2)*(1./2.-sw**2 -1./(8.0*sw**2))*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_ZZtau3 = Coupling(name = 'R2_ZZtau3',
                   value = 'MTA**2/(4.*cw**2)*(1./2.-sw**2 -1./(8.0*sw**2))*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_ZZU = Coupling(name = 'R2_ZZU',
                  value = '-Ncol/(24.*cw**2)*(1./3.-1./(8.0*sw**2)-4.*sw**2/9.)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_ZZD = Coupling(name = 'R2_ZZD',
                  value = '-Ncol/(24.*cw**2)*(1./6.-1./(8.0*sw**2)-sw**2/9.)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_ZZu3 = Coupling(name= 'R2_ZZu3',
                   value = 'MU**2/(4.*cw**2)*Ncol*(1./3.-1./(8.0*sw**2)-4.*sw**2/9.)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_ZZc3 = Coupling(name= 'R2_ZZc3',
                   value = 'MC**2/(4.*cw**2)*Ncol*(1./3.-1./(8.0*sw**2)-4.*sw**2/9.)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_ZZt3 = Coupling(name= 'R2_ZZt3',
                   value = 'MT**2/(4.*cw**2)*Ncol*(1./3.-1./(8.0*sw**2)-4.*sw**2/9.)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_ZZd3 = Coupling(name= 'R2_ZZd3',
                   value = 'MD**2/(4.*cw**2)*Ncol*(1./6.-1./(8.0*sw**2)-sw**2/9.)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_ZZs3 = Coupling(name= 'R2_ZZs3',
                   value = 'MS**2/(4.*cw**2)*Ncol*(1./6.-1./(8.0*sw**2)-sw**2/9.)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_ZZb3 = Coupling(name= 'R2_ZZb3',
                   value = 'MB**2/(4.*cw**2)*Ncol*(1./6.-1./(8.0*sw**2)-sw**2/9.)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_ZZv = Coupling(name = 'R2_ZZv',
                  value = '1./(24.0*8.0*cw**2*sw**2)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWC1 = Coupling(name = 'R2_WWC1',
                   value = '-lhv/(24.*sw**2)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_WWboson1 = Coupling(name = 'R2_WWboson1',
                       value = '((1./48.+lhv/24.)*(1./sw**2)*(complex(0,1)*R2VV))',
                       order = {'QED':2})

R2_WWboson2 = Coupling(name = 'R2_WWboson2',
                   value = '(-lhv/(24.*sw**2)*(complex(0,1)*R2VV))',
                   order = {'QED':2})

R2_WWboson3 = Coupling(name = 'R2_WWboson3',
                       value = '(-MW**2/(8.*sw**2)*(complex(0,1)*R2VV))',
                       order = {'QED':2})

R2_WWl = Coupling(name = 'R2_WWl',
                   value = '1./(96.*sw**2)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_WWe3 = Coupling(name = 'R2_WWe3',
                   value = '-Me**2/(32.*sw**2)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_WWm3 = Coupling(name = 'R2_WWm3',
                   value = '-MM**2/(32.*sw**2)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_WWtau3 = Coupling(name = 'R2_WWtau3',
                   value = '-MTA**2/(32.*sw**2)*(complex(0,1)*R2VV)',
                   order = {'QED':2})

R2_WWud1 = Coupling(name = 'R2_WWud1',
                  value = 'Ncol/(96.*sw**2)*(SCKM11)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWus1 = Coupling(name = 'R2_WWus1',
                  value = 'Ncol/(96.*sw**2)*(SCKM12)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWub1 = Coupling(name = 'R2_WWub1',
                  value = 'Ncol/(96.*sw**2)*(SCKM13)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWcd1 = Coupling(name = 'R2_WWcd1',
                  value = 'Ncol/(96.*sw**2)*(SCKM21)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWcs1 = Coupling(name = 'R2_WWcs1',
                  value = 'Ncol/(96.*sw**2)*(SCKM22)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWcb1 = Coupling(name = 'R2_WWcb1',
                  value = 'Ncol/(96.*sw**2)*(SCKM23)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWtd1 = Coupling(name = 'R2_WWtd1',
                  value = 'Ncol/(96.*sw**2)*(SCKM31)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWts1 = Coupling(name = 'R2_WWts1',
                  value = 'Ncol/(96.*sw**2)*(SCKM32)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWtb1 = Coupling(name = 'R2_WWtb1',
                  value = 'Ncol/(96.*sw**2)*(SCKM33)*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWud3 = Coupling(name = 'R2_WWud3',
                  value = '-Ncol/(32.*sw**2)*(SCKM11*(MU**2 +MD**2))*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWus3 = Coupling(name = 'R2_WWus3',
                  value = '-Ncol/(32.*sw**2)*(SCKM12*(MU**2 +MS**2))*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWub3 = Coupling(name = 'R2_WWub3',
                  value = '-Ncol/(32.*sw**2)*(SCKM13*(MU**2 +MB**2))*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWcd3 = Coupling(name = 'R2_WWcd3',
                  value = '-Ncol/(32.*sw**2)*(SCKM21*(MC**2 +MD**2))*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWcs3 = Coupling(name = 'R2_WWcs3',
                  value = '-Ncol/(32.*sw**2)*(SCKM22*(MC**2 +MS**2))*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWcb3 = Coupling(name = 'R2_WWcb3',
                  value = '-Ncol/(32.*sw**2)*(SCKM23*(MC**2 +MB**2))*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWtd3 = Coupling(name = 'R2_WWtd3',
                  value = '-Ncol/(32.*sw**2)*(SCKM31*(MT**2 +MD**2))*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWts3 = Coupling(name = 'R2_WWts3',
                  value = '-Ncol/(32.*sw**2)*(SCKM32*(MT**2 +MS**2))*(complex(0,1)*R2VV)',
                  order = {'QED':2})

R2_WWtb3 = Coupling(name = 'R2_WWtb3',
                  value = '-Ncol/(32.*sw**2)*(SCKM33*(MT**2 +MB**2))*(complex(0,1)*R2VV)',
                  order = {'QED':2})

# R2 for FF~

R2_UUCm = Coupling(name = 'R2_UUCm',
                   value = '(1./(36.0*cw**2)*(complex(0,1)*R2VV*lhv))',
                   order = {'QED':2})

R2_DDCm = Coupling(name = 'R2_DDCm',
                   value = '(1./(144.0*cw**2)*(complex(0,1)*R2VV*lhv))',
                   order = {'QED':2})

R2_LLCm = Coupling(name = 'R2_LLCm',
                   value = '(1./(16.0*cw**2)*(complex(0,1)*R2VV*lhv))',
                   order = {'QED':2})

R2_QQCp0 = Coupling(name = 'R2_QQCp0',
                        value = '(1./16.*(1./(4.0*sw**2*cw**2)-2./(9.0*cw**2))*(complex(0,1)*R2VV*lhv))',
                        order = {'QED':2})

R2_LLCp0 = Coupling(name = 'R2_LLCp0',
                        value = '(1./(64.0*sw**2*cw**2)*(complex(0,1)*R2VV*lhv))',
                        order = {'QED':2})

R2_nunuCp0 = Coupling(name = 'R2_nunuCp0',
                        value = '(1./(64.0*sw**2*cw**2)*(complex(0,1)*R2VV*lhv))',
                        order = {'QED':2})

R2_QQCpud = Coupling(name = 'R2_QQCpud',
                     value = '(1./(32.*sw**2)*(SCKM11)*(complex(0,1)*R2VV*lhv))',
                     order = {'QED':2})

R2_QQCpus = Coupling(name = 'R2_QQCpus',
                     value = '(1./(32.*sw**2)*(SCKM12)*(complex(0,1)*R2VV*lhv))',
                     order = {'QED':2})

R2_QQCpub = Coupling(name = 'R2_QQCpub',
                     value = '(1./(32.*sw**2)*(SCKM13)*(complex(0,1)*R2VV*lhv))',
                     order = {'QED':2})

R2_QQCpcd = Coupling(name = 'R2_QQCpcd',
                     value = '(1./(32.*sw**2)*(SCKM21)*(complex(0,1)*R2VV*lhv))',
                     order = {'QED':2})

R2_QQCpcs = Coupling(name = 'R2_QQCpcs',
                     value = '(1./(32.*sw**2)*(SCKM22)*(complex(0,1)*R2VV*lhv))',
                     order = {'QED':2})

R2_QQCpcb = Coupling(name = 'R2_QQCpcb',
                     value = '(1./(32.*sw**2)*(SCKM23)*(complex(0,1)*R2VV*lhv))',
                     order = {'QED':2})

R2_QQCptd = Coupling(name = 'R2_QQCptd',
                     value = '(1./(32.*sw**2)*(SCKM31)*(complex(0,1)*R2VV*lhv))',
                     order = {'QED':2})

R2_QQCpts = Coupling(name = 'R2_QQCpts',
                     value = '(1./(32.*sw**2)*(SCKM32)*(complex(0,1)*R2VV*lhv))',
                     order = {'QED':2})

R2_QQCptb = Coupling(name = 'R2_QQCptb',
                     value = '(1./(32.*sw**2)*(SCKM33)*(complex(0,1)*R2VV*lhv))',
                     order = {'QED':2})

R2_LLCplv = Coupling(name = 'R2_QQCplv',
                     value = '(1./(32.0*sw**2)*(complex(0,1)*R2VV*lhv))',
                     order = {'QED':2})

R2_UUC0 = Coupling(name = 'R2_UUC0',
                   value = '(MU/(12.0*cw**2)*(1./6.)*(complex(0,1)*R2VV*lhv))',
                   order = {'QED':2})

R2_CCC0 = Coupling(name = 'R2_CCC0',
                   value = '(MC/(12.0*cw**2)*(1./6.)*(complex(0,1)*R2VV*lhv))',
                   order = {'QED':2})

R2_TTC0 = Coupling(name = 'R2_TTC0',
                   value = '(MT/(12.0*cw**2)*(1./6.)*(complex(0,1)*R2VV*lhv))',
                   order = {'QED':2})

R2_DDC0 = Coupling(name = 'R2_DDC0',
                   value = '(-MD/(24.0*cw**2)*(1./6.)*(complex(0,1)*R2VV*lhv))',
                   order = {'QED':2})

R2_SSC0 = Coupling(name = 'R2_SSC0',
                   value = '(-MS/(24.0*cw**2)*(1./6.)*(complex(0,1)*R2VV*lhv))',
                   order = {'QED':2})

R2_BBC0 = Coupling(name = 'R2_BBC0',
                   value = '(-MB/(24.0*cw**2)*(1./6.)*(complex(0,1)*R2VV*lhv))',
                   order = {'QED':2})

R2_EEC0 = Coupling(name = 'R2_EEC0',
                   value = '(-Me/(8.0*cw**2)*(-1./2.)*(complex(0,1)*R2VV*lhv))',
                   order = {'QED':2})

R2_MMC0 = Coupling(name = 'R2_MMC0',
                   value = '(-MM/(8.0*cw**2)*(-1./2.)*(complex(0,1)*R2VV*lhv))',
                   order = {'QED':2})

R2_TATAC0 = Coupling(name = 'R2_TATAC0',
                   value = '(-MTA/(8.0*cw**2)*(-1./2.)*(complex(0,1)*R2VV*lhv))',
                   order = {'QED':2})

# R2 for SFF~

R2_Huu = Coupling(name = 'R2_Huu',
                  value = '(complex(0,1)*MU/(8*MW*sw)*((1+lhv)/(18.*cw**2)+1./(32.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_Huu_d = Coupling(name = 'R2_Huu_d',
                    value = '(complex(0,1)*MU/(8*MW*sw)*(SCKM11/(16.0*sw**2))*(1+MD**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Huu_s = Coupling(name = 'R2_Huu_s',
                    value = '(complex(0,1)*MU/(8*MW*sw)*(SCKM12/(16.0*sw**2))*(1+MS**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Huu_b = Coupling(name = 'R2_Huu_b',
                    value = '(complex(0,1)*MU/(8*MW*sw)*(SCKM13/(16.0*sw**2))*(1+MB**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Hcc = Coupling(name = 'R2_Hcc',
                  value = '(complex(0,1)*MC/(8*MW*sw)*((1+lhv)/(18.*cw**2)+1./(32.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_Hcc_d = Coupling(name = 'R2_Hcc_d',
                    value = '(complex(0,1)*MC/(8*MW*sw)*(SCKM21/(16.0*sw**2))*(1+MD**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Hcc_s = Coupling(name = 'R2_Hcc_s',
                    value = '(complex(0,1)*MC/(8*MW*sw)*(SCKM22/(16.0*sw**2))*(1+MS**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Hcc_b = Coupling(name = 'R2_Hcc_b',
                    value = '(complex(0,1)*MC/(8*MW*sw)*(SCKM23/(16.0*sw**2))*(1+MB**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Htt = Coupling(name = 'R2_Htt',
                  value = '(complex(0,1)*MT/(8*MW*sw)*((1+lhv)/(18.*cw**2)+1./(32.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_Htt_d = Coupling(name = 'R2_Htt_d',
                    value = '(complex(0,1)*MT/(8*MW*sw)*(SCKM31/(16.0*sw**2))*(1+MD**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Htt_s = Coupling(name = 'R2_Htt_s',
                    value = '(complex(0,1)*MT/(8*MW*sw)*(SCKM32/(16.0*sw**2))*(1+MS**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Htt_b = Coupling(name = 'R2_Htt_b',
                    value = '(complex(0,1)*MT/(8*MW*sw)*(SCKM33/(16.0*sw**2))*(1+MB**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Hdd = Coupling(name = 'R2_Hdd',
                  value = '(complex(0,1)*MD/(8*MW*sw)*(-(1+lhv)/(36.*cw**2)+1./(32.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_Hdd_u = Coupling(name = 'R2_Hdd_u',
                    value = '(complex(0,1)*MD/(8*MW*sw)*(SCKM11/(16.0*sw**2))*(1+MU**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Hdd_c = Coupling(name = 'R2_Hdd_c',
                    value = '(complex(0,1)*MD/(8*MW*sw)*(SCKM21/(16.0*sw**2))*(1+MC**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Hdd_t = Coupling(name = 'R2_Hdd_t',
                    value = '(complex(0,1)*MD/(8*MW*sw)*(SCKM31/(16.0*sw**2))*(1+MT**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Hss = Coupling(name = 'R2_Hss',
                  value = '(complex(0,1)*MS/(8*MW*sw)*(-(1+lhv)/(36.*cw**2)+1./(32.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_Hss_u = Coupling(name = 'R2_Hss_u',
                    value = '(complex(0,1)*MS/(8*MW*sw)*(SCKM12/(16.0*sw**2))*(1+MU**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Hss_c = Coupling(name = 'R2_Hss_c',
                    value = '(complex(0,1)*MS/(8*MW*sw)*(SCKM22/(16.0*sw**2))*(1+MC**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Hss_t = Coupling(name = 'R2_Hss_t',
                    value = '(complex(0,1)*MS/(8*MW*sw)*(SCKM32/(16.0*sw**2))*(1+MT**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Hbb = Coupling(name = 'R2_Hbb',
                  value = '(complex(0,1)*MB/(8*MW*sw)*(-(1+lhv)/(36.*cw**2)+1./(32.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_Hbb_u = Coupling(name = 'R2_Hbb_u',
                    value = '(complex(0,1)*MB/(8*MW*sw)*(SCKM13/(16.0*sw**2))*(1+MU**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Hbb_c = Coupling(name = 'R2_Hbb_c',
                    value = '(complex(0,1)*MB/(8*MW*sw)*(SCKM23/(16.0*sw**2))*(1+MC**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Hbb_t = Coupling(name = 'R2_Hbb_t',
                    value = '(complex(0,1)*MB/(8*MW*sw)*(SCKM33/(16.0*sw**2))*(1+MT**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_Hee = Coupling(name = 'R2_Hee',
                  value = '(complex(0,1)*Me/(8*MW*sw)*((1+lhv)/(4.*cw**2)+1./(32.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_Hee_v = Coupling(name = 'R2_Hee_v',
                    value = '(complex(0,1)*Me/(8*MW*sw)*(1./(16.0*sw**2))*R2SFF)',
                    order = {'QED':3})

R2_Hmm = Coupling(name = 'R2_Hmm',
                  value = '(complex(0,1)*MM/(8*MW*sw)*((1+lhv)/(4.*cw**2)+1./(32.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_Hmm_v = Coupling(name = 'R2_Hmm_v',
                    value = '(complex(0,1)*MM/(8*MW*sw)*(1./(16.0*sw**2))*R2SFF)',
                    order = {'QED':3})

R2_Htautau = Coupling(name = 'R2_Htautau',
                  value = '(complex(0,1)*MTA/(8*MW*sw)*((1+lhv)/(4.*cw**2)+1./(32.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_Htautau_v = Coupling(name = 'R2_Htautau_v',
                    value = '(complex(0,1)*MTA/(8*MW*sw)*(1./(16.0*sw**2))*R2SFF)',
                    order = {'QED':3})

R2_G0uu = Coupling(name = 'R2_G0uu',
                  value = '(-MU/(4*MW*sw)*((1+lhv)/(36.*cw**2)+1./(64.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_G0uu_d = Coupling(name = 'R2_G0uu_d',
                    value = '(-MU/(4*MW*sw)*(SCKM11/(32.0*sw**2))*(1+MD**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0uu_s = Coupling(name = 'R2_G0uu_s',
                    value = '(-MU/(4*MW*sw)*(SCKM12/(32.0*sw**2))*(1+MS**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0uu_b = Coupling(name = 'R2_G0uu_b',
                    value = '(-MU/(4*MW*sw)*(SCKM13/(32.0*sw**2))*(1+MB**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0cc = Coupling(name = 'R2_G0cc',
                  value = '(-MC/(4*MW*sw)*((1+lhv)/(36.*cw**2)+1./(64.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_G0cc_d = Coupling(name = 'R2_G0cc_d',
                    value = '(-MC/(4*MW*sw)*(SCKM21/(32.0*sw**2))*(1+MD**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0cc_s = Coupling(name = 'R2_G0cc_s',
                    value = '(-MC/(4*MW*sw)*(SCKM22/(32.0*sw**2))*(1+MS**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0cc_b = Coupling(name = 'R2_G0cc_b',
                    value = '(-MC/(4*MW*sw)*(SCKM23/(32.0*sw**2))*(1+MB**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0tt = Coupling(name = 'R2_G0tt',
                  value = '(-MT/(4*MW*sw)*((1+lhv)/(36.*cw**2)+1./(64.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_G0tt_d = Coupling(name = 'R2_G0tt_d',
                    value = '(-MT/(4*MW*sw)*(SCKM31/(32.0*sw**2))*(1+MD**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0tt_s = Coupling(name = 'R2_G0tt_s',
                    value = '(-MT/(4*MW*sw)*(SCKM32/(32.0*sw**2))*(1+MS**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0tt_b = Coupling(name = 'R2_G0tt_b',
                    value = '(-MT/(4*MW*sw)*(SCKM33/(32.0*sw**2))*(1+MB**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0dd = Coupling(name = 'R2_G0dd',
                  value = '(-MD/(4*MW*sw)*((1+lhv)/(72.*cw**2)-1./(64.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_G0dd_u = Coupling(name = 'R2_G0dd_u',
                    value = '(MD/(4*MW*sw)*(SCKM11/(32.0*sw**2))*(1+MU**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0dd_c = Coupling(name = 'R2_G0dd_c',
                    value = '(MD/(4*MW*sw)*(SCKM21/(32.0*sw**2))*(1+MC**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0dd_t = Coupling(name = 'R2_G0dd_t',
                    value = '(MD/(4*MW*sw)*(SCKM31/(32.0*sw**2))*(1+MT**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0ss = Coupling(name = 'R2_G0ss',
                  value = '(-MS/(4*MW*sw)*((1+lhv)/(72.*cw**2)-1./(64.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_G0ss_u = Coupling(name = 'R2_G0ss_u',
                    value = '(MS/(4*MW*sw)*(SCKM12/(32.0*sw**2))*(1+MU**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0ss_c = Coupling(name = 'R2_G0ss_c',
                    value = '(MS/(4*MW*sw)*(SCKM22/(32.0*sw**2))*(1+MC**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0ss_t = Coupling(name = 'R2_G0ss_t',
                    value = '(MS/(4*MW*sw)*(SCKM32/(32.0*sw**2))*(1+MT**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0bb = Coupling(name = 'R2_G0bb',
                  value = '(-MB/(4*MW*sw)*((1+lhv)/(72.*cw**2)-1./(64.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_G0bb_u = Coupling(name = 'R2_G0bb_u',
                    value = '(MB/(4*MW*sw)*(SCKM13/(32.0*sw**2))*(1+MU**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0bb_c = Coupling(name = 'R2_G0bb_c',
                    value = '(MB/(4*MW*sw)*(SCKM23/(32.0*sw**2))*(1+MC**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0bb_t = Coupling(name = 'R2_G0bb_t',
                    value = '(MB/(4*MW*sw)*(SCKM33/(32.0*sw**2))*(1+MT**2/MW**2)*R2SFF)',
                    order = {'QED':3})

R2_G0ee = Coupling(name = 'R2_G0ee',
                  value = '(-Me/(4*MW*sw)*(-(1+lhv)/(8.*cw**2)-1./(64.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_G0ee_v = Coupling(name = 'R2_G0ee_v',
                    value = '-Me/(4*MW*sw)*(-1./(32.0*sw**2))*R2SFF',
                    order = {'QED':3})

R2_G0mm = Coupling(name = 'R2_G0mm',
                  value = '(-MM/(4*MW*sw)*(-(1+lhv)/(8.*cw**2)-1./(64.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_G0mm_v = Coupling(name = 'R2_G0mm_v',
                    value = '-MM/(4*MW*sw)*(-1./(32.0*sw**2))*R2SFF',
                    order = {'QED':3})

R2_G0tautau = Coupling(name = 'R2_G0tautau',
                  value = '(-MTA/(4*MW*sw)*(-(1+lhv)/(8.*cw**2)-1./(64.0*cw**2*sw**2))*R2SFF)',
                  order = {'QED':3})

R2_G0tautau_v = Coupling(name = 'R2_G0tautau_v',
                    value = '-MTA/(4*MW*sw)*(-1./(32.0*sw**2))*R2SFF',
                    order = {'QED':3})

R2_uxdGp2Cm = Coupling(name = 'R2_uxdGp2Cm',
                       value = '(MD*complexconjugate(CKM11)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)+(1+lhv)/(36.*cw**2)-(3+2*MU**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_uxdGp2Cp = Coupling(name = 'R2_uxdGp2Cp',
                       value = '(-MU*complexconjugate(CKM11)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)-(1+lhv)/(18.*cw**2)-(3+2*MD**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_uxsGp2Cm = Coupling(name = 'R2_uxsGp2Cm',
                       value = '(MS*complexconjugate(CKM12)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)+(1+lhv)/(36.*cw**2)-(3+2*MU**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_uxsGp2Cp = Coupling(name = 'R2_uxsGp2Cp',
                       value = '(-MU*complexconjugate(CKM12)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)-(1+lhv)/(18.*cw**2)-(3+2*MS**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_uxbGp2Cm = Coupling(name = 'R2_uxbGp2Cm',
                       value = '(MB*complexconjugate(CKM13)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)+(1+lhv)/(36.*cw**2)-(3+2*MU**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_uxbGp2Cp = Coupling(name = 'R2_uxbGp2Cp',
                       value = '(-MU*complexconjugate(CKM13)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)-(1+lhv)/(18.*cw**2)-(3+2*MB**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_cxdGp2Cm = Coupling(name = 'R2_cxdGp2Cm',
                       value = '(MD*complexconjugate(CKM21)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)+(1+lhv)/(36.*cw**2)-(3+2*MC**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_cxdGp2Cp = Coupling(name = 'R2_cxdGp2Cp',
                       value = '(-MC*complexconjugate(CKM21)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)-(1+lhv)/(18.*cw**2)-(3+2*MD**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_cxsGp2Cm = Coupling(name = 'R2_cxsGp2Cm',
                       value = '(MS*complexconjugate(CKM22)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)+(1+lhv)/(36.*cw**2)-(3+2*MC**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_cxsGp2Cp = Coupling(name = 'R2_cxsGp2Cp',
                       value = '(-MC*complexconjugate(CKM22)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)-(1+lhv)/(18.*cw**2)-(3+2*MS**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_cxbGp2Cm = Coupling(name = 'R2_cxbGp2Cm',
                       value = '(MB*complexconjugate(CKM23)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)+(1+lhv)/(36.*cw**2)-(3+2*MC**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_cxbGp2Cp = Coupling(name = 'R2_cxbGp2Cp',
                       value = '(-MC*complexconjugate(CKM23)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)-(1+lhv)/(18.*cw**2)-(3+2*MB**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_txdGp2Cm = Coupling(name = 'R2_txdGp2Cm',
                       value = '(MD*complexconjugate(CKM31)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)+(1+lhv)/(36.*cw**2)-(3+2*MT**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_txdGp2Cp = Coupling(name = 'R2_txdGp2Cp',
                       value = '(-MT*complexconjugate(CKM31)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)-(1+lhv)/(18.*cw**2)-(3+2*MD**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_txsGp2Cm = Coupling(name = 'R2_txsGp2Cm',
                       value = '(MS*complexconjugate(CKM32)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)+(1+lhv)/(36.*cw**2)-(3+2*MT**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_txsGp2Cp = Coupling(name = 'R2_txsGp2Cp',
                       value = '(-MT*complexconjugate(CKM32)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)-(1+lhv)/(18.*cw**2)-(3+2*MS**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_txbGp2Cm = Coupling(name = 'R2_txbGp2Cm',
                       value = '(MB*complexconjugate(CKM33)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)+(1+lhv)/(36.*cw**2)-(3+2*MT**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_txbGp2Cp = Coupling(name = 'R2_txbGp2Cp',
                       value = '(-MT*complexconjugate(CKM33)/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)-(1+lhv)/(18.*cw**2)-(3+2*MB**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_dxuGm2Cm = Coupling(name = 'R2_dxuGm2Cm',
                       value = '(-MU*CKM11/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)+(1+lhv)/(18.*cw**2)+(3+2*MD**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_dxuGm2Cp = Coupling(name = 'R2_dxuGm2Cp',
                       value = '(MD*CKM11/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)-(1+lhv)/(36.*cw**2)+(3+2*MU**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_sxuGm2Cm = Coupling(name = 'R2_sxuGm2Cm',
                       value = '(-MU*CKM12/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)+(1+lhv)/(18.*cw**2)+(3+2*MS**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_sxuGm2Cp = Coupling(name = 'R2_sxuGm2Cp',
                       value = '(MS*CKM12/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)-(1+lhv)/(36.*cw**2)+(3+2*MU**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_bxuGm2Cm = Coupling(name = 'R2_bxuGm2Cm',
                       value = '(-MU*CKM13/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)+(1+lhv)/(18.*cw**2)+(3+2*MB**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_bxuGm2Cp = Coupling(name = 'R2_bxuGm2Cp',
                       value = '(MB*CKM13/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)-(1+lhv)/(36.*cw**2)+(3+2*MU**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_dxcGm2Cm = Coupling(name = 'R2_dxcGm2Cm',
                       value = '(-MC*CKM21/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)+(1+lhv)/(18.*cw**2)+(3+2*MD**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_dxcGm2Cp = Coupling(name = 'R2_dxcGm2Cp',
                       value = '(MD*CKM21/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)-(1+lhv)/(36.*cw**2)+(3+2*MC**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_sxcGm2Cm = Coupling(name = 'R2_sxcGm2Cm',
                       value = '(-MC*CKM22/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)+(1+lhv)/(18.*cw**2)+(3+2*MS**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_sxcGm2Cp = Coupling(name = 'R2_sxcGm2Cp',
                       value = '(MS*CKM22/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)-(1+lhv)/(36.*cw**2)+(3+2*MC**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_bxcGm2Cm = Coupling(name = 'R2_bxcGm2Cm',
                       value = '(-MC*CKM23/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)+(1+lhv)/(18.*cw**2)+(3+2*MB**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_bxcGm2Cp = Coupling(name = 'R2_bxcGm2Cp',
                       value = '(MB*CKM23/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)-(1+lhv)/(36.*cw**2)+(3+2*MC**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_dxtGm2Cm = Coupling(name = 'R2_dxtGm2Cm',
                       value = '(-MT*CKM31/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)+(1+lhv)/(18.*cw**2)+(3+2*MD**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_dxtGm2Cp = Coupling(name = 'R2_dxtGm2Cp',
                       value = '(MD*CKM31/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)-(1+lhv)/(36.*cw**2)+(3+2*MT**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_sxtGm2Cm = Coupling(name = 'R2_sxtGm2Cm',
                       value = '(-MT*CKM32/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)+(1+lhv)/(18.*cw**2)+(3+2*MS**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_sxtGm2Cp = Coupling(name = 'R2_sxtGm2Cp',
                       value = '(MS*CKM32/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)-(1+lhv)/(36.*cw**2)+(3+2*MT**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_bxtGm2Cm = Coupling(name = 'R2_bxtGm2Cm',
                       value = '(-MT*CKM33/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)+(1+lhv)/(18.*cw**2)+(3+2*MB**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_bxtGm2Cp = Coupling(name = 'R2_bxtGm2Cp',
                       value = '(MB*CKM33/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)-(1+lhv)/(36.*cw**2)+(3+2*MT**2/MW**2)/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_vexeGpCm = Coupling(name = 'R2_vexeGpCm',
                       value = '(Me/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)-(1+lhv)/(4.*cw**2)-3/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_vmxmGpCm = Coupling(name = 'R2_vmxmGpCm',
                       value = '(MM/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)-(1+lhv)/(4.*cw**2)-3/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_vtxtauGpCm = Coupling(name = 'R2_vtxtauGpCm',
                       value = '(MTA/(4*cmath.sqrt(2)*MW*sw)*(-1./(32.*cw**2)-(1+lhv)/(4.*cw**2)-3/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_exveGmCp = Coupling(name = 'R2_exveGmCp',
                       value = '(Me/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)+(1+lhv)/(4.*cw**2)+3/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_mxvmGmCp = Coupling(name = 'R2_mxvmGmCp',
                       value = '(MM/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)+(1+lhv)/(4.*cw**2)+3/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_tauxvtGmCp = Coupling(name = 'R2_tauxvtGmCp',
                       value = '(MTA/(4*cmath.sqrt(2)*MW*sw)*(1./(32.*cw**2)+(1+lhv)/(4.*cw**2)+3/(32.*sw**2))*R2SFF)',
                       order = {'QED':3})

R2_ddA2Cp = Coupling(name = 'R2_ddA2Cp',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(-1./27.)-(1+lhv)/(2.0*cw**2)*(-1./18.)+(1+lhv)/(4.*sw**2*cw**2)*(-1./12.)+MD**2/(16*MW**2*sw**2)*(-1./3.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddA2Cp_u = Coupling(name = 'R2_ddA2Cp_u',
                     value = '((1./4.)*(SCKM11)*(1./(4.*sw**2))*(MU**2/(6*MW**2)-(1+lhv)/6.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddA2Cp_c = Coupling(name = 'R2_ddA2Cp_c',
                     value = '((1./4.)*(SCKM21)*(1./(4.*sw**2))*(MC**2/(6*MW**2)-(1+lhv)/6.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddA2Cp_t = Coupling(name = 'R2_ddA2Cp_t',
                     value = '((1./4.)*(SCKM31)*(1./(4.*sw**2))*(MT**2/(6*MW**2)-(1+lhv)/6.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssA2Cp = Coupling(name = 'R2_ssA2Cp',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(-1./27.)-(1+lhv)/(2.0*cw**2)*(-1./18.)+(1+lhv)/(4.*sw**2*cw**2)*(-1./12.)+MS**2/(16*MW**2*sw**2)*(-1./3.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssA2Cp_u = Coupling(name = 'R2_ssA2Cp_u',
                     value = '((1./4.)*(SCKM12)*(1./(4.*sw**2))*(MU**2/(6*MW**2)-(1+lhv)/6.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssA2Cp_c = Coupling(name = 'R2_ssA2Cp_c',
                     value = '((1./4.)*(SCKM22)*(1./(4.*sw**2))*(MC**2/(6*MW**2)-(1+lhv)/6.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssA2Cp_t = Coupling(name = 'R2_ssA2Cp_t',
                     value = '((1./4.)*(SCKM32)*(1./(4.*sw**2))*(MT**2/(6*MW**2)-(1+lhv)/6.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbA2Cp = Coupling(name = 'R2_bbA2Cp',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(-1./27.)-(1+lhv)/(2.0*cw**2)*(-1./18.)+(1+lhv)/(4.*sw**2*cw**2)*(-1./12.)+MB**2/(16*MW**2*sw**2)*(-1./3.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbA2Cp_u = Coupling(name = 'R2_bbA2Cp_u',
                     value = '((1./4.)*(SCKM13)*(1./(4.*sw**2))*(MU**2/(6*MW**2)-(1+lhv)/6.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbA2Cp_c = Coupling(name = 'R2_bbA2Cp_c',
                     value = '((1./4.)*(SCKM23)*(1./(4.*sw**2))*(MC**2/(6*MW**2)-(1+lhv)/6.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbA2Cp_t = Coupling(name = 'R2_bbA2Cp_t',
                     value = '((1./4.)*(SCKM33)*(1./(4.*sw**2))*(MT**2/(6*MW**2)-(1+lhv)/6.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddA2Cm = Coupling(name = 'R2_ddA2Cm',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(-1./27.)+MD**2/(8*sw**2*MW**2)*(-1./3.)*(1./2.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddA2Cm_u = Coupling(name = 'R2_ddA2Cm_u',
                     value = '((1./4.)*(SCKM11)*(MD**2/(24*sw**2*MW**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddA2Cm_c = Coupling(name = 'R2_ddA2Cm_c',
                     value = '((1./4.)*(SCKM21)*(MD**2/(24*sw**2*MW**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddA2Cm_t = Coupling(name = 'R2_ddA2Cm_t',
                     value = '((1./4.)*(SCKM31)*(MD**2/(24*sw**2*MW**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssA2Cm = Coupling(name = 'R2_ssA2Cm',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(-1./27.)+MS**2/(8*sw**2*MW**2)*(-1./3.))*(1./2.))*(-complex(0,1)*R2SFF)',
                     order = {'QED':3})

R2_ssA2Cm_u = Coupling(name = 'R2_ssA2Cm_u',
                     value = '((1./4.)*(SCKM12)*(MS**2/(24*sw**2*MW**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssA2Cm_c = Coupling(name = 'R2_ssA2Cm_c',
                     value = '((1./4.)*(SCKM22)*(MS**2/(24*sw**2*MW**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssA2Cm_t = Coupling(name = 'R2_ssA2Cm_t',
                     value = '((1./4.)*(SCKM32)*(MS**2/(24*sw**2*MW**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbA2Cm = Coupling(name = 'R2_bbA2Cm',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(-1./27.)+MB**2/(8*sw**2*MW**2)*(-1./3.)*(1./2.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbA2Cm_u = Coupling(name = 'R2_bbA2Cm_u',
                     value = '((1./4.)*(SCKM13)*(MB**2/(24*sw**2*MW**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbA2Cm_c = Coupling(name = 'R2_bbA2Cm_c',
                     value = '((1./4.)*(SCKM23)*(MB**2/(24*sw**2*MW**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbA2Cm_t = Coupling(name = 'R2_bbA2Cm_t',
                     value = '((1./4.)*(SCKM33)*(MB**2/(24*sw**2*MW**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuA2Cp = Coupling(name = 'R2_uuA2Cp',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(8./27.)-(1+lhv)/(2.0*cw**2)*(2./9.)+(1+lhv)/(4.*sw**2*cw**2)*(1./6.)+MU**2/(16*MW**2*sw**2)*(2./3.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuA2Cp_d = Coupling(name = 'R2_uuA2Cp_d',
                     value = '((1./4.)*(SCKM11)*(1./(4.*sw**2))*(-MD**2/(12*MW**2)+(1+lhv)/3.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuA2Cp_s = Coupling(name = 'R2_uuA2Cp_s',
                     value = '((1./4.)*(SCKM12)*(1./(4.*sw**2))*(-MS**2/(12*MW**2)+(1+lhv)/3.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuA2Cp_b = Coupling(name = 'R2_uuA2Cp_b',
                     value = '((1./4.)*(SCKM13)*(1./(4.*sw**2))*(-MB**2/(12*MW**2)+(1+lhv)/3.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccA2Cp = Coupling(name = 'R2_ccA2Cp',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(8./27.)-(1+lhv)/(2.0*cw**2)*(2./9.)+(1+lhv)/(4.*sw**2*cw**2)*(1./6.)+MC**2/(16*MW**2*sw**2)*(2./3.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccA2Cp_d = Coupling(name = 'R2_ccA2Cp_d',
                     value = '((1./4.)*(SCKM21)*(1./(4.*sw**2))*(-MD**2/(12*MW**2)+(1+lhv)/3.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccA2Cp_s = Coupling(name = 'R2_ccA2Cp_s',
                     value = '((1./4.)*(SCKM22)*(1./(4.*sw**2))*(-MS**2/(12*MW**2)+(1+lhv)/3.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccA2Cp_b = Coupling(name = 'R2_ccA2Cp_b',
                     value = '((1./4.)*(SCKM23)*(1./(4.*sw**2))*(-MB**2/(12*MW**2)+(1+lhv)/3.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttA2Cp = Coupling(name = 'R2_ttA2Cp',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(8./27.)-(1+lhv)/(2.0*cw**2)*(2./9.)+(1+lhv)/(4.*sw**2*cw**2)*(1./6.)+MT**2/(16*MW**2*sw**2)*(2./3.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttA2Cp_d = Coupling(name = 'R2_ttA2Cp_d',
                     value = '((1./4.)*(SCKM31)*(1./(4.*sw**2))*(-MD**2/(12*MW**2)+(1+lhv)/3.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttA2Cp_s = Coupling(name = 'R2_ttA2Cp_s',
                     value = '((1./4.)*(SCKM32)*(1./(4.*sw**2))*(-MS**2/(12*MW**2)+(1+lhv)/3.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttA2Cp_b = Coupling(name = 'R2_ttA2Cp_b',
                     value = '((1./4.)*(SCKM33)*(1./(4.*sw**2))*(-MB**2/(12*MW**2)+(1+lhv)/3.)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuA2Cm = Coupling(name = 'R2_uuA2Cm',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(8./27.)+MU**2/(8*MW**2*sw**2)*(1./3.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuA2Cm_d = Coupling(name = 'R2_uuA2Cm_d',
                     value = '((1./4.)*(SCKM11)*(MU**2/(8*MW**2*sw**2)*(-1./6.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuA2Cm_s = Coupling(name = 'R2_uuA2Cm_s',
                     value = '((1./4.)*(SCKM12)*(MU**2/(8*MW**2*sw**2)*(-1./6.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuA2Cm_b = Coupling(name = 'R2_uuA2Cm_b',
                     value = '((1./4.)*(SCKM13)*(MU**2/(8*MW**2*sw**2)*(-1./6.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccA2Cm = Coupling(name = 'R2_ccA2Cm',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(8./27.)+MC**2/(8*MW**2*sw**2)*(1./3.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccA2Cm_d = Coupling(name = 'R2_ccA2Cm_d',
                     value = '((1./4.)*(SCKM21)*(MC**2/(8*MW**2*sw**2)*(-1./6.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccA2Cm_s = Coupling(name = 'R2_ccA2Cm_s',
                     value = '((1./4.)*(SCKM22)*(MC**2/(8*MW**2*sw**2)*(-1./6.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccA2Cm_b = Coupling(name = 'R2_ccA2Cm_b',
                     value = '((1./4.)*(SCKM23)*(MC**2/(8*MW**2*sw**2)*(-1./6.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttA2Cm = Coupling(name = 'R2_ttA2Cm',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(8./27.)+MT**2/(8*MW**2*sw**2)*(1./3.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttA2Cm_d = Coupling(name = 'R2_ttA2Cm_d',
                     value = '((1./4.)*(SCKM31)*(MT**2/(8*MW**2*sw**2)*(-1./6.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttA2Cm_s = Coupling(name = 'R2_ttA2Cm_s',
                     value = '((1./4.)*(SCKM32)*(MT**2/(8*MW**2*sw**2)*(-1./6.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttA2Cm_b = Coupling(name = 'R2_ttA2Cm_b',
                     value = '((1./4.)*(SCKM33)*(MT**2/(8*MW**2*sw**2)*(-1./6.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_eeACp = Coupling(name = 'R2_eeACp',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(-1)-(1+lhv)/(2.0*cw**2)*(-1./2.)+(1+lhv)/(4.*sw**2*cw**2)*(-1./4.)+Me**2/(16*MW**2*sw**2)*(-1))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_mmACp = Coupling(name = 'R2_mmACp',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(-1)-(1+lhv)/(2.0*cw**2)*(-1./2.)+(1+lhv)/(4.*sw**2*cw**2)*(-1./4.)+MM**2/(16*MW**2*sw**2)*(-1))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_tautauACp = Coupling(name = 'R2_tautauACp',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(-1)-(1+lhv)/(2.0*cw**2)*(-1./2.)+(1+lhv)/(4.*sw**2*cw**2)*(-1./4.)+MTA**2/(16*MW**2*sw**2)*(-1))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_eeACm = Coupling(name = 'R2_eeACm',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(-1)+Me**2/(8*MW**2*sw**2)*(-1./2.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_mmACm = Coupling(name = 'R2_mmACm',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(-1)+MM**2/(8*MW**2*sw**2)*(-1./2.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_tautauACm = Coupling(name = 'R2_tautauACm',
                     value = '((1./4.)*((1+lhv)/(4.*cw**2)*(-1)+MTA**2/(8*MW**2*sw**2)*(-1./2.))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_llACp = Coupling(name = 'R2_llACp',
                     value = '(-(1+lhv)/(32.*sw**2)*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_veveACp = Coupling(name = 'R2_veveACp',
                     value = '((-Me**2/(64*MW**2*sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_vmvmACp = Coupling(name = 'R2_vmvmACp',
                     value = '((-MM**2/(64*MW**2*sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_vtvtACp = Coupling(name = 'R2_vtvtACp',
                     value = '((-MTA**2/(64*MW**2*sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddZ2Cp = Coupling(name = 'R2_ddZ2Cp',
                     value = '((1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Qd**3)+(1+2*sw**2)/(sw*cw**2)*(Qd**2*I3d)-3/(sw*cw**2)*(Qd*I3d**2)+1./(sw**3*cw**2)*(I3d**3))-Qd*MD**2/(4*MW**2*sw))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddZ2Cp_u = Coupling(name = 'R2_ddZ2Cp_u',
                     value = '(-(1./(16.*cw))*(SCKM11/sw)*(MU**2/(4*MW**2)*Qu+(1+lhv)/2.*(Qu+(cw**2 - I3u)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddZ2Cp_c = Coupling(name = 'R2_ddZ2Cp_c',
                     value = '(-(1./(16.*cw))*(SCKM21/sw)*(MC**2/(4*MW**2)*Qu+(1+lhv)/2.*(Qu+(cw**2 - I3u)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddZ2Cp_t = Coupling(name = 'R2_ddZ2Cp_t',
                     value = '(-(1./(16.*cw))*(SCKM31/sw)*(MT**2/(4*MW**2)*Qu+(1+lhv)/2.*(Qu+(cw**2 - I3u)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssZ2Cp = Coupling(name = 'R2_ssZ2Cp',
                     value = '((1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Qd**3)+(1+2*sw**2)/(sw*cw**2)*(Qd**2*I3d)-3/(sw*cw**2)*(Qd*I3d**2)+1./(sw**3*cw**2)*(I3d**3))-Qd*MS**2/(4*MW**2*sw))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssZ2Cp_u = Coupling(name = 'R2_ssZ2Cp_u',
                     value = '(-(1./(16.*cw))*(SCKM12/sw)*(MU**2/(4*MW**2)*Qu+(1+lhv)/2.*(Qu+(cw**2 - I3u)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssZ2Cp_c = Coupling(name = 'R2_ssZ2Cp_c',
                     value = '(-(1./(16.*cw))*(SCKM22/sw)*(MC**2/(4*MW**2)*Qu+(1+lhv)/2.*(Qu+(cw**2 - I3u)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssZ2Cp_t = Coupling(name = 'R2_ssZ2Cp_t',
                     value = '(-(1./(16.*cw))*(SCKM32/sw)*(MT**2/(4*MW**2)*Qu+(1+lhv)/2.*(Qu+(cw**2 - I3u)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbZ2Cp = Coupling(name = 'R2_bbZ2Cp',
                     value = '((1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Qd**3)+(1+2*sw**2)/(sw*cw**2)*(Qd**2*I3d)-3/(sw*cw**2)*(Qd*I3d**2)+1./(sw**3*cw**2)*(I3d**3))-Qd*MB**2/(4*MW**2*sw))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbZ2Cp_u = Coupling(name = 'R2_bbZ2Cp_u',
                     value = '(-(1./(16.*cw))*(SCKM13/sw)*(MU**2/(4*MW**2)*Qu+(1+lhv)/2.*(Qu+(cw**2 - I3u)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbZ2Cp_c = Coupling(name = 'R2_bbZ2Cp_c',
                     value = '(-(1./(16.*cw))*(SCKM23/sw)*(MC**2/(4*MW**2)*Qu+(1+lhv)/2.*(Qu+(cw**2 - I3u)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbZ2Cp_t = Coupling(name = 'R2_bbZ2Cp_t',
                     value = '(-(1./(16.*cw))*(SCKM33/sw)*(MT**2/(4*MW**2)*Qu+(1+lhv)/2.*(Qu+(cw**2 - I3u)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddZ2Cm = Coupling(name = 'R2_ddZ2Cm',
                     value = '((1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Qd**3))-MD**2/(4*MW**2*sw)*(Qd-I3d/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddZ2Cm_u = Coupling(name = 'R2_ddZ2Cm_u',
                     value = '((1./(16.*cw))*(SCKM11)*(-MD**2/(4*MW**2*sw)*(Qu-I3u/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddZ2Cm_c = Coupling(name = 'R2_ddZ2Cm_c',
                     value = '((1./(16.*cw))*(SCKM21)*(-MD**2/(4*MW**2*sw)*(Qu-I3u/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ddZ2Cm_t = Coupling(name = 'R2_ddZ2Cm_t',
                     value = '((1./(16.*cw))*(SCKM31)*(-MD**2/(4*MW**2*sw)*(Qu-I3u/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssZ2Cm = Coupling(name = 'R2_ssZ2Cm',
                     value = '((1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Qd**3))-MS**2/(4*MW**2*sw)*(Qd-I3d/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssZ2Cm_u = Coupling(name = 'R2_ssZ2Cm_u',
                     value = '((1./(16.*cw))*(SCKM12)*(-MS**2/(4*MW**2*sw)*(Qu-I3u/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssZ2Cm_c = Coupling(name = 'R2_ssZ2Cm_c',
                     value = '((1./(16.*cw))*(SCKM22)*(-MS**2/(4*MW**2*sw)*(Qu-I3u/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ssZ2Cm_t = Coupling(name = 'R2_ssZ2Cm_t',
                     value = '((1./(16.*cw))*(SCKM32)*(-MS**2/(4*MW**2*sw)*(Qu-I3u/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbZ2Cm = Coupling(name = 'R2_bbZ2Cm',
                     value = '((1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Qd**3))-MB**2/(4*MW**2*sw)*(Qd-I3d/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbZ2Cm_u = Coupling(name = 'R2_bbZ2Cm_u',
                     value = '((1./(16.*cw))*(SCKM13)*(-MB**2/(4*MW**2*sw)*(Qu-I3u/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbZ2Cm_c = Coupling(name = 'R2_bbZ2Cm_c',
                     value = '((1./(16.*cw))*(SCKM23)*(-MB**2/(4*MW**2*sw)*(Qu-I3u/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_bbZ2Cm_t = Coupling(name = 'R2_bbZ2Cm_t',
                     value = '((1./(16.*cw))*(SCKM33)*(-MB**2/(4*MW**2*sw)*(Qu-I3u/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuZ2Cp = Coupling(name = 'R2_uuZ2Cp',
                     value = '((1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Qu**3)+(1+2*sw**2)/(sw*cw**2)*(Qu**2*I3u)-3/(sw*cw**2)*(Qu*I3u**2)+1./(sw**3*cw**2)*(I3u**3))-Qu*MU**2/(4*MW**2*sw))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuZ2Cp_d = Coupling(name = 'R2_uuZ2Cp_d',
                     value = '(-(1./(16.*cw))*(SCKM11/sw)*(MD**2/(4*MW**2)*Qd+(1+lhv)/2.*(Qd-(cw**2 +I3d)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuZ2Cp_s = Coupling(name = 'R2_uuZ2Cp_s',
                     value = '(-(1./(16.*cw))*(SCKM12/sw)*(MS**2/(4*MW**2)*Qd+(1+lhv)/2.*(Qd-(cw**2 +I3d)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuZ2Cp_b = Coupling(name = 'R2_uuZ2Cp_b',
                     value = '(-(1./(16.*cw))*(SCKM13/sw)*(MB**2/(4*MW**2)*Qd+(1+lhv)/2.*(Qd-(cw**2 +I3d)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccZ2Cp = Coupling(name = 'R2_ccZ2Cp',
                     value = '((1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Qu**3)+(1+2*sw**2)/(sw*cw**2)*(Qu**2*I3u)-3/(sw*cw**2)*(Qu*I3u**2)+1./(sw**3*cw**2)*(I3u**3))-Qu*MC**2/(4*MW**2*sw))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccZ2Cp_d = Coupling(name = 'R2_ccZ2Cp_d',
                     value = '(-(1./(16.*cw))*(SCKM21/sw)*(MD**2/(4*MW**2)*Qd+(1+lhv)/2.*(Qd-(cw**2 +I3d)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccZ2Cp_s = Coupling(name = 'R2_ccZ2Cp_s',
                     value = '(-(1./(16.*cw))*(SCKM22/sw)*(MS**2/(4*MW**2)*Qd+(1+lhv)/2.*(Qd-(cw**2 +I3d)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccZ2Cp_b = Coupling(name = 'R2_ccZ2Cp_b',
                     value = '(-(1./(16.*cw))*(SCKM23/sw)*(MB**2/(4*MW**2)*Qd+(1+lhv)/2.*(Qd-(cw**2 +I3d)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttZ2Cp = Coupling(name = 'R2_ttZ2Cp',
                     value = '((1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Qu**3)+(1+2*sw**2)/(sw*cw**2)*(Qu**2*I3u)-3/(sw*cw**2)*(Qu*I3u**2)+1./(sw**3*cw**2)*(I3u**3))-Qu*MT**2/(4*MW**2*sw))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttZ2Cp_d = Coupling(name = 'R2_ttZ2Cp_d',
                     value = '(-(1./(16.*cw))*(SCKM31/sw)*(MD**2/(4*MW**2)*Qd+(1+lhv)/2.*(Qd-(cw**2 +I3d)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttZ2Cp_s = Coupling(name = 'R2_ttZ2Cp_s',
                     value = '(-(1./(16.*cw))*(SCKM32/sw)*(MS**2/(4*MW**2)*Qd+(1+lhv)/2.*(Qd-(cw**2 +I3d)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttZ2Cp_b = Coupling(name = 'R2_ttZ2Cp_b',
                     value = '(-(1./(16.*cw))*(SCKM33/sw)*(MB**2/(4*MW**2)*Qd+(1+lhv)/2.*(Qd-(cw**2 +I3d)/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuZ2Cm = Coupling(name = 'R2_uuZ2Cm',
                     value = '((1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Qu**3))-MU**2/(4*MW**2*sw)*(Qu-I3u/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuZ2Cm_d = Coupling(name = 'R2_uuZ2Cm_d',
                     value = '((1./(16.*cw))*(SCKM11)*(-MU**2/(4*MW**2*sw)*(Qd-I3d/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuZ2Cm_s = Coupling(name = 'R2_uuZ2Cm_s',
                     value = '((1./(16.*cw))*(SCKM12)*(-MU**2/(4*MW**2*sw)*(Qd-I3d/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_uuZ2Cm_b = Coupling(name = 'R2_uuZ2Cm_b',
                     value = '((1./(16.*cw))*(SCKM13)*(-MU**2/(4*MW**2*sw)*(Qd-I3d/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccZ2Cm = Coupling(name = 'R2_ccZ2Cm',
                     value = '((1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Qu**3))-MC**2/(4*MW**2*sw)*(Qu-I3u/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccZ2Cm_d = Coupling(name = 'R2_ccZ2Cm_d',
                     value = '((1./(16.*cw))*(SCKM21)*(-MC**2/(4*MW**2*sw)*(Qd-I3d/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccZ2Cm_s = Coupling(name = 'R2_ccZ2Cm_s',
                     value = '((1./(16.*cw))*(SCKM22)*(-MC**2/(4*MW**2*sw)*(Qd-I3d/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ccZ2Cm_b = Coupling(name = 'R2_ccZ2Cm_b',
                     value = '((1./(16.*cw))*(SCKM23)*(-MC**2/(4*MW**2*sw)*(Qd-I3d/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttZ2Cm = Coupling(name = 'R2_ttZ2Cm',
                     value = '((1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Qu**3))-MT**2/(4*MW**2*sw)*(Qu-I3u/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttZ2Cm_d = Coupling(name = 'R2_ttZ2Cm_d',
                     value = '((1./(16.*cw))*(SCKM31)*(-MT**2/(4*MW**2*sw)*(Qd-I3d/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttZ2Cm_s = Coupling(name = 'R2_ttZ2Cm_s',
                     value = '((1./(16.*cw))*(SCKM32)*(-MT**2/(4*MW**2*sw)*(Qd-I3d/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_ttZ2Cm_b = Coupling(name = 'R2_ttZ2Cm_b',
                     value = '((1./(16.*cw))*(SCKM33)*(-MT**2/(4*MW**2*sw)*(Qd-I3d/sw**2))*(-complex(0,1)*R2SFF))',
                     order = {'QED':3})

R2_eeZCp = Coupling(name = 'R2_eeZCp',
                     value = '(1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Ql**3)+(1+2*sw**2)/(sw*cw**2)*(Ql**2*I3l)-3/(sw*cw**2)*(Ql*I3l**2)+1./(sw**3*cw**2)*(I3l**3))-Ql*Me**2/(4*MW**2*sw))*(-complex(0,1)*R2SFF)',
                     order = {'QED':3})

R2_mmZCp = Coupling(name = 'R2_mmZCp',
                     value = '(1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Ql**3)+(1+2*sw**2)/(sw*cw**2)*(Ql**2*I3l)-3/(sw*cw**2)*(Ql*I3l**2)+1./(sw**3*cw**2)*(I3l**3))-Ql*MM**2/(4*MW**2*sw))*(-complex(0,1)*R2SFF)',
                     order = {'QED':3})

R2_tautauZCp = Coupling(name = 'R2_tautauZCp',
                     value = '(1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Ql**3)+(1+2*sw**2)/(sw*cw**2)*(Ql**2*I3l)-3/(sw*cw**2)*(Ql*I3l**2)+1./(sw**3*cw**2)*(I3l**3))-Ql*MTA**2/(4*MW**2*sw))*(-complex(0,1)*R2SFF)',
                     order = {'QED':3})

R2_llZCp = Coupling(name = 'R2_llZCp',
                     value = '-(1./(16.*cw))*(1./sw)*((1+lhv)/2.*(cw**2 - I3v)/sw**2)*(-complex(0,1)*R2SFF)',
                     order = {'QED':3})

R2_eeZCm = Coupling(name = 'R2_eeZCm',
                     value = '(1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Ql**3))-Me**2/(4*MW**2*sw)*(Ql-I3l/sw**2))*(-complex(0,1)*R2SFF)',
                     order = {'QED':3})

R2_eeZCm_v = Coupling(name = 'R2_eeZCm_v',
                     value = '(1./(16.*cw))*(-Me**2/(4*MW**2*sw)*(-I3v/sw**2))*(-complex(0,1)*R2SFF)',
                     order = {'QED':3})

R2_mmZCm = Coupling(name = 'R2_mmZCm',
                     value = '(1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Ql**3))-MM**2/(4*MW**2*sw)*(Ql-I3l/sw**2))*(-complex(0,1)*R2SFF)',
                     order = {'QED':3})

R2_mmZCm_v = Coupling(name = 'R2_mmZCm_v',
                     value = '(1./(16.*cw))*(-MM**2/(4*MW**2*sw)*(-I3v/sw**2))*(-complex(0,1)*R2SFF)',
                     order = {'QED':3})

R2_tautauZCm = Coupling(name = 'R2_tautauZCm',
                     value = '(1./(16.*cw))*((1+lhv)*(-sw/cw**2*(Ql**3))-MTA**2/(4*MW**2*sw)*(Ql-I3l/sw**2))*(-complex(0,1)*R2SFF)',
                     order = {'QED':3})

R2_tautauZCm_v = Coupling(name = 'R2_tautauZCm_v',
                     value = '(1./(16.*cw))*(-MTA**2/(4*MW**2*sw)*(-I3v/sw**2))*(-complex(0,1)*R2SFF)',
                     order = {'QED':3})

R2_vvZCp = Coupling(name = 'R2_vvZCp',
                    value = '1./(16.*cw)*((1+lhv)/(sw**3*cw**2)*I3v**3)*(-complex(0,1)*R2SFF)',
                    order = {'QED':3})

R2_veveZCp_e = Coupling(name = 'R2_veveZCp_e',
                    value = '-1./(16.*cw)*(1./(2.*sw))*(Me**2*Ql/(2.*MW**2)+(1+lhv)*(Ql-(cw**2 +I3l)/sw**2))*(-complex(0,1)*R2SFF)',
                    order = {'QED':3})

R2_vmvmZCp_m = Coupling(name = 'R2_vmvmZCp_m',
                    value = '-1./(16.*cw)*(1./(2.*sw))*(MM**2*Ql/(2.*MW**2)+(1+lhv)*(Ql-(cw**2 +I3l)/sw**2))*(-complex(0,1)*R2SFF)',
                    order = {'QED':3})

R2_vtvtZCp_tau = Coupling(name = 'R2_vtvtZCp_tau',
                    value = '-1./(16.*cw)*(1./(2.*sw))*(MTA**2*Ql/(2.*MW**2)+(1+lhv)*(Ql-(cw**2 +I3l)/sw**2))*(-complex(0,1)*R2SFF)',
                    order = {'QED':3})

R2_dxuW2Cp = Coupling(name= 'R2_dxuW2Cp',
                      value = '-CKM11/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_sxuW2Cp = Coupling(name= 'R2_sxuW2Cp',
                      value = '-CKM12/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_bxuW2Cp = Coupling(name= 'R2_bxuW2Cp',
                      value = '-CKM13/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_dxcW2Cp = Coupling(name= 'R2_dxcW2Cp',
                      value = '-CKM21/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_sxcW2Cp = Coupling(name= 'R2_sxcW2Cp',
                      value = '-CKM22/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_bxcW2Cp = Coupling(name= 'R2_bxcW2Cp',
                      value = '-CKM23/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_dxtW2Cp = Coupling(name= 'R2_dxtW2Cp',
                      value = '-CKM31/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_sxtW2Cp = Coupling(name= 'R2_sxtW2Cp',
                      value = '-CKM32/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_bxtW2Cp = Coupling(name= 'R2_bxtW2Cp',
                      value = '-CKM33/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_uxdW2Cp = Coupling(name= 'R2_uxdW2Cp',
                      value = '-complexconjugate(CKM11)/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_uxsW2Cp = Coupling(name= 'R2_uxsW2Cp',
                      value = '-complexconjugate(CKM12)/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_uxbW2Cp = Coupling(name= 'R2_uxbW2Cp',
                      value = '-complexconjugate(CKM13)/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_cxdW2Cp = Coupling(name= 'R2_cxdW2Cp',
                      value = '-complexconjugate(CKM21)/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_cxsW2Cp = Coupling(name= 'R2_cxsW2Cp',
                      value = '-complexconjugate(CKM22)/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_cxbW2Cp = Coupling(name= 'R2_cxbW2Cp',
                      value = '-complexconjugate(CKM23)/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_txdW2Cp = Coupling(name= 'R2_txdW2Cp',
                      value = '-complexconjugate(CKM31)/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_txsW2Cp = Coupling(name= 'R2_txsW2Cp',
                      value = '-complexconjugate(CKM32)/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_txbW2Cp = Coupling(name= 'R2_txbW2Cp',
                      value = '-complexconjugate(CKM33)/(16*cmath.sqrt(2)*sw)*(1+lhv)*((Qd*I3u+Qu*I3d-Qu*Qd)/cw**2 -1/sw**2 +1/(4.*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                      order = {'QED':3})

R2_vlW = Coupling(name = 'R2_vlW',
                  value = '-1./(16*cmath.sqrt(2)*sw)*(1+lhv)*(Ql*I3v/cw**2 -1/sw**2 +1/(4*sw**2*cw**2))*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

# R2 for SSS

R2_HHHboson = Coupling(name = 'R2_HHHboson',
                       value = '-3./(32*sw**3)*((1-4*lhv)/2.*MW*(1+1/(2*cw**4))+1./4.*(1+1/(2*cw**2))*MH**2/MW)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_HHHe = Coupling(name = 'R2_HHHe',
                       value = '-3./(32*sw**3)*(Me**4/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_HHHm = Coupling(name = 'R2_HHHm',
                       value = '-3./(32*sw**3)*(MM**4/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_HHHtau = Coupling(name = 'R2_HHHtau',
                       value = '-3./(32*sw**3)*(MTA**4/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_HHHu = Coupling(name = 'R2_HHHu',
                       value = '-3./(32*sw**3)*(Ncol*MU**4/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_HHHd = Coupling(name = 'R2_HHHd',
                       value = '-3./(32*sw**3)*(Ncol*MD**4/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_HHHs = Coupling(name = 'R2_HHHs',
                       value = '-3./(32*sw**3)*(Ncol*MS**4/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_HHHc = Coupling(name = 'R2_HHHc',
                       value = '-3./(32*sw**3)*(Ncol*MC**4/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_HHHb = Coupling(name = 'R2_HHHb',
                       value = '-3./(32*sw**3)*(Ncol*MB**4/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_HHHt = Coupling(name = 'R2_HHHt',
                       value = '-3./(32*sw**3)*(Ncol*MT**4/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_G0G0Hboson = Coupling(name = 'R2_G0G0Hboson',
                       value = '-1./(8*sw**3)*((1-4*lhv)/8.*MW*(1+1/(2*cw**4))+1./16.*(1+1/(2*cw**2))*MH**2/MW)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_G0G0He = Coupling(name = 'R2_G0G0He',
                       value = '-1./(8*sw**3)*(Me**4/(4.*MW**3))*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_G0G0Hm = Coupling(name = 'R2_G0G0Hm',
                       value = '-1./(8*sw**3)*(MM**4/(4.*MW**3))*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_G0G0Htau = Coupling(name = 'R2_G0G0Htau',
                       value = '-1./(8*sw**3)*(MTA**4/(4.*MW**3))*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_G0G0Hu = Coupling(name = 'R2_G0G0Hu',
                       value = '-1./(8*sw**3)*(Ncol*MU**4/(4.*MW**3))*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_G0G0Hd = Coupling(name = 'R2_G0G0Hd',
                       value = '-1./(8*sw**3)*(Ncol*MD**4/(4.*MW**3))*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_G0G0Hs = Coupling(name = 'R2_G0G0Hs',
                       value = '-1./(8*sw**3)*(Ncol*MS**4/(4.*MW**3))*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_G0G0Hc = Coupling(name = 'R2_G0G0Hc',
                       value = '-1./(8*sw**3)*(Ncol*MC**4/(4.*MW**3))*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_G0G0Hb = Coupling(name = 'R2_G0G0Hb',
                       value = '-1./(8*sw**3)*(Ncol*MB**4/(4.*MW**3))*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_G0G0Ht = Coupling(name = 'R2_G0G0Ht',
                       value = '-1./(8*sw**3)*(Ncol*MT**4/(4.*MW**3))*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_GmGpHboson = Coupling(name = 'R2_GmGpHboson',
                         value = '-1./(32*sw**3)*((1+2*cw**2)/(8*cw**2)*MH**2/MW+(3+sw**2*(1+cw**2)/cw**4)*(1-4*lhv)/4.*MW)*(-complex(0,1)*R2SFF)',
                         order = {'QED':3})

R2_GmGpHe = Coupling(name = 'R2_GmGpHe',
                       value = '-1./(32*sw**3)*(Me**4/(MW**3))*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_GmGpHm = Coupling(name = 'R2_GmGpHm',
                       value = '-1./(32*sw**3)*(MM**4/(MW**3))*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_GmGpHtau = Coupling(name = 'R2_GmGpHtau',
                       value = '-1./(32*sw**3)*(MTA**4/(MW**3))*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_GmGpHud = Coupling(name = 'R2_GmGpHud',
                       value = '-1./(32*sw**3)*Ncol*(SCKM11*(MU**4 +MD**4)/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_GmGpHus = Coupling(name = 'R2_GmGpHus',
                       value = '-1./(32*sw**3)*Ncol*(SCKM12*(MU**4 +MS**4)/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_GmGpHub = Coupling(name = 'R2_GmGpHub',
                       value = '-1./(32*sw**3)*Ncol*(SCKM13*(MU**4 +MB**4)/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_GmGpHcd = Coupling(name = 'R2_GmGpHcd',
                       value = '-1./(32*sw**3)*Ncol*(SCKM21*(MC**4 +MD**4)/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_GmGpHcs = Coupling(name = 'R2_GmGpHcs',
                       value = '-1./(32*sw**3)*Ncol*(SCKM22*(MC**4 +MS**4)/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_GmGpHcb = Coupling(name = 'R2_GmGpHcb',
                       value = '-1./(32*sw**3)*Ncol*(SCKM23*(MC**4 +MB**4)/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_GmGpHtd = Coupling(name = 'R2_GmGpHtd',
                       value = '-1./(32*sw**3)*Ncol*(SCKM31*(MT**4 +MD**4)/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_GmGpHts = Coupling(name = 'R2_GmGpHts',
                       value = '-1./(32*sw**3)*Ncol*(SCKM32*(MT**4 +MS**4)/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

R2_GmGpHtb = Coupling(name = 'R2_GmGpHtb',
                       value = '-1./(32*sw**3)*Ncol*(SCKM33*(MT**4 +MB**4)/MW**3)*(-complex(0,1)*R2SFF)',
                       order = {'QED':3})

# R2 for VSS

R2_AG0H = Coupling(name= 'R2_AG0H',
                   value = '5./(192.*sw**2)*R2SFF',
                   order = {'QED':3})

R2_ZG0H = Coupling(name= 'R2_ZG0H',
                   value = '1./(96.*sw*cw)*(1+2*cw**2 +20*cw**4)/(8*sw**2*cw**2)*R2SFF',
                   order = {'QED':3})

R2_ZG0He = Coupling(name= 'R2_ZG0He',
                   value = '1./(96.*sw*cw)*Me**2/(sw**2*MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZG0Hm = Coupling(name= 'R2_ZG0Hm',
                   value = '1./(96.*sw*cw)*MM**2/(sw**2*MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZG0Htau = Coupling(name= 'R2_ZG0Htau',
                   value = '1./(96.*sw*cw)*MTA**2/(sw**2*MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZG0Hu = Coupling(name= 'R2_ZG0Hu',
                   value = '1./(96.*sw*cw)*Ncol*MU**2/(sw**2*MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZG0Hd = Coupling(name= 'R2_ZG0Hd',
                   value = '1./(96.*sw*cw)*Ncol*MD**2/(sw**2*MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZG0Hs = Coupling(name= 'R2_ZG0Hs',
                   value = '1./(96.*sw*cw)*Ncol*MS**2/(sw**2*MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZG0Hc = Coupling(name= 'R2_ZG0Hc',
                   value = '1./(96.*sw*cw)*Ncol*MC**2/(sw**2*MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZG0Hb = Coupling(name= 'R2_ZG0Hb',
                   value = '1./(96.*sw*cw)*Ncol*MB**2/(sw**2*MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZG0Ht = Coupling(name= 'R2_ZG0Ht',
                   value = '1./(96.*sw*cw)*Ncol*MT**2/(sw**2*MW**2)*R2SFF',
                   order = {'QED':3})

R2_AGmGp = Coupling(name= 'R2_AGmGp',
                   value = 'complex(0,1)/(48.*sw**2)*(1+12*cw**2)/(8*cw**2)*R2SFF',
                   order = {'QED':3})

R2_AGmGpe = Coupling(name= 'R2_AGmGpe',
                   value = 'complex(0,1)/(48.*sw**2)*(-Ql*Me**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_AGmGpm = Coupling(name= 'R2_AGmGpm',
                   value = 'complex(0,1)/(48.*sw**2)*(-Ql*MM**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_AGmGptau = Coupling(name= 'R2_AGmGptau',
                   value = 'complex(0,1)/(48.*sw**2)*(-Ql*MTA**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_AGmGpud = Coupling(name= 'R2_AGmGpud',
                   value = 'complex(0,1)/(48.*sw**2)*Ncol*SCKM11*(MU**2 +MD**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_AGmGpus = Coupling(name= 'R2_AGmGpus',
                   value = 'complex(0,1)/(48.*sw**2)*Ncol*SCKM12*(MU**2 +MS**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_AGmGpub = Coupling(name= 'R2_AGmGpub',
                   value = 'complex(0,1)/(48.*sw**2)*Ncol*SCKM13*(MU**2 +MB**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_AGmGpcd = Coupling(name= 'R2_AGmGpcd',
                   value = 'complex(0,1)/(48.*sw**2)*Ncol*SCKM21*(MC**2 +MD**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_AGmGpcs = Coupling(name= 'R2_AGmGpcs',
                   value = 'complex(0,1)/(48.*sw**2)*Ncol*SCKM22*(MC**2 +MS**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_AGmGpcb = Coupling(name= 'R2_AGmGpcb',
                   value = 'complex(0,1)/(48.*sw**2)*Ncol*SCKM23*(MC**2 +MB**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_AGmGptd = Coupling(name= 'R2_AGmGptd',
                   value = 'complex(0,1)/(48.*sw**2)*Ncol*SCKM31*(MT**2 +MD**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_AGmGpts = Coupling(name= 'R2_AGmGpts',
                   value = 'complex(0,1)/(48.*sw**2)*Ncol*SCKM32*(MT**2 +MS**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_AGmGptb = Coupling(name= 'R2_AGmGptb',
                   value = 'complex(0,1)/(48.*sw**2)*Ncol*SCKM33*(MT**2 +MB**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZGmGp = Coupling(name= 'R2_ZGmGp',
                   value = '-complex(0,1)/(48.*sw*cw)*(1-24*cw**4)/(16*cw**2*sw**2)*R2SFF',
                   order = {'QED':3})

R2_ZGmGpe = Coupling(name= 'R2_ZGmGpe',
                   value = '-complex(0,1)/(48.*sw*cw)*Me**2/(MW**2)*(-Ql-I3v/sw**2)*R2SFF',
                   order = {'QED':3})

R2_ZGmGpm = Coupling(name= 'R2_ZGmGpm',
                   value = '-complex(0,1)/(48.*sw*cw)*MM**2/(MW**2)*(-Ql-I3v/sw**2)*R2SFF',
                   order = {'QED':3})

R2_ZGmGptau = Coupling(name= 'R2_ZGmGptau',
                   value = '-complex(0,1)/(48.*sw*cw)*MTA**2/(MW**2)*(-Ql-I3v/sw**2)*R2SFF',
                   order = {'QED':3})

R2_ZGmGpud = Coupling(name= 'R2_ZGmGpud',
                   value = '-complex(0,1)/(48.*sw*cw)*SCKM11*Ncol*(MU**2 +MD**2 +(MU**2*I3d-MD**2*I3u)/sw**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZGmGpus = Coupling(name= 'R2_ZGmGpus',
                   value = '-complex(0,1)/(48.*sw*cw)*SCKM12*Ncol*(MU**2 +MS**2 +(MU**2*I3d-MS**2*I3u)/sw**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZGmGpub = Coupling(name= 'R2_ZGmGpub',
                   value = '-complex(0,1)/(48.*sw*cw)*SCKM13*Ncol*(MU**2 +MB**2 +(MU**2*I3d-MB**2*I3u)/sw**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZGmGpcd = Coupling(name= 'R2_ZGmGpcd',
                   value = '-complex(0,1)/(48.*sw*cw)*SCKM21*Ncol*(MC**2 +MD**2 +(MC**2*I3d-MD**2*I3u)/sw**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZGmGpcs = Coupling(name= 'R2_ZGmGpcs',
                   value = '-complex(0,1)/(48.*sw*cw)*SCKM22*Ncol*(MC**2 +MS**2 +(MC**2*I3d-MS**2*I3u)/sw**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZGmGpcb = Coupling(name= 'R2_ZGmGpcb',
                   value = '-complex(0,1)/(48.*sw*cw)*SCKM23*Ncol*(MC**2 +MB**2 +(MC**2*I3d-MB**2*I3u)/sw**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZGmGptd = Coupling(name= 'R2_ZGmGptd',
                   value = '-complex(0,1)/(48.*sw*cw)*SCKM31*Ncol*(MT**2 +MD**2 +(MT**2*I3d-MD**2*I3u)/sw**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZGmGpts = Coupling(name= 'R2_ZGmGpts',
                   value = '-complex(0,1)/(48.*sw*cw)*SCKM32*Ncol*(MT**2 +MS**2 +(MT**2*I3d-MS**2*I3u)/sw**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_ZGmGptb = Coupling(name= 'R2_ZGmGptb',
                   value = '-complex(0,1)/(48.*sw*cw)*SCKM33*Ncol*(MT**2 +MB**2 +(MT**2*I3d-MB**2*I3u)/sw**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_WGpH = Coupling(name = 'R2_WGpH',
                   value = '1/(96*sw**3)*(1+22*cw**2)/(8*cw**2)*R2SFF',
                   order = {'QED':3})

R2_WGpHe = Coupling(name = 'R2_WGpHe',
                   value = '1/(96*sw**3)*(Me**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_WGpHm = Coupling(name = 'R2_WGpHm',
                   value = '1/(96*sw**3)*(MM**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_WGpHtau = Coupling(name = 'R2_WGpHtau',
                   value = '1/(96*sw**3)*(MTA**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_WGpHud = Coupling(name = 'R2_WGpHud',
                   value = '1/(96*sw**3)*Ncol*SCKM11*(MU**2 +MD**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_WGpHus = Coupling(name = 'R2_WGpHus',
                   value = '1/(96*sw**3)*Ncol*SCKM12*(MU**2 +MS**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_WGpHub = Coupling(name = 'R2_WGpHub',
                   value = '1/(96*sw**3)*Ncol*SCKM13*(MU**2 +MB**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_WGpHcd = Coupling(name = 'R2_WGpHcd',
                   value = '1/(96*sw**3)*Ncol*SCKM21*(MC**2 +MD**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_WGpHcs = Coupling(name = 'R2_WGpHcs',
                   value = '1/(96*sw**3)*Ncol*SCKM22*(MC**2 +MS**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_WGpHcb = Coupling(name = 'R2_WGpHcb',
                   value = '1/(96*sw**3)*Ncol*SCKM23*(MC**2 +MB**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_WGpHtd = Coupling(name = 'R2_WGpHtd',
                   value = '1/(96*sw**3)*Ncol*SCKM31*(MT**2 +MD**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_WGpHts = Coupling(name = 'R2_WGpHts',
                   value = '1/(96*sw**3)*Ncol*SCKM32*(MT**2 +MS**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_WGpHtb = Coupling(name = 'R2_WGpHtb',
                   value = '1/(96*sw**3)*Ncol*SCKM33*(MT**2 +MB**2)/(MW**2)*R2SFF',
                   order = {'QED':3})

R2_WGpG0 = Coupling(name= 'R2_WGpG0',
                    value = '-complex(0,1)/(48*sw**3)*(-1-22*cw**2)/(16*cw**2)*R2SFF',
                    order = {'QED':3})

R2_WGpG0e = Coupling(name= 'R2_WGpG0e',
                    value = '-complex(0,1)/(48*sw**3)*(Me**2*I3l)/(MW**2)*R2SFF',
                    order = {'QED':3})

R2_WGpG0m = Coupling(name= 'R2_WGpG0m',
                    value = '-complex(0,1)/(48*sw**3)*(MM**2*I3l)/(MW**2)*R2SFF',
                    order = {'QED':3})

R2_WGpG0tau = Coupling(name= 'R2_WGpG0tau',
                    value = '-complex(0,1)/(48*sw**3)*(MTA**2*I3l)/(MW**2)*R2SFF',
                    order = {'QED':3})

R2_WGpG0ud = Coupling(name= 'R2_WGpG0ud',
                    value = 'complex(0,1)/(48*sw**3)*Ncol*SCKM11*(MU**2*I3u-MD**2*I3d)/(MW**2)*R2SFF',
                    order = {'QED':3})

R2_WGpG0us = Coupling(name= 'R2_WGpG0us',
                    value = 'complex(0,1)/(48*sw**3)*Ncol*SCKM12*(MU**2*I3u-MS**2*I3d)/(MW**2)*R2SFF',
                    order = {'QED':3})

R2_WGpG0ub = Coupling(name= 'R2_WGpG0ub',
                    value = 'complex(0,1)/(48*sw**3)*Ncol*SCKM13*(MU**2*I3u-MB**2*I3d)/(MW**2)*R2SFF',
                    order = {'QED':3})

R2_WGpG0cd = Coupling(name= 'R2_WGpG0cd',
                    value = 'complex(0,1)/(48*sw**3)*Ncol*SCKM21*(MC**2*I3u-MD**2*I3d)/(MW**2)*R2SFF',
                    order = {'QED':3})

R2_WGpG0cs = Coupling(name= 'R2_WGpG0cs',
                    value = 'complex(0,1)/(48*sw**3)*Ncol*SCKM22*(MC**2*I3u-MS**2*I3d)/(MW**2)*R2SFF',
                    order = {'QED':3})

R2_WGpG0cb = Coupling(name= 'R2_WGpG0cb',
                    value = 'complex(0,1)/(48*sw**3)*Ncol*SCKM23*(MC**2*I3u-MB**2*I3d)/(MW**2)*R2SFF',
                    order = {'QED':3})

R2_WGpG0td = Coupling(name= 'R2_WGpG0td',
                    value = 'complex(0,1)/(48*sw**3)*Ncol*SCKM31*(MT**2*I3u-MD**2*I3d)/(MW**2)*R2SFF',
                    order = {'QED':3})

R2_WGpG0ts = Coupling(name= 'R2_WGpG0ts',
                    value = 'complex(0,1)/(48*sw**3)*Ncol*SCKM32*(MT**2*I3u-MS**2*I3d)/(MW**2)*R2SFF',
                    order = {'QED':3})

R2_WGpG0tb = Coupling(name= 'R2_WGpG0tb',
                    value = 'complex(0,1)/(48*sw**3)*Ncol*SCKM33*(MT**2*I3u-MB**2*I3d)/(MW**2)*R2SFF',
                    order = {'QED':3})

# R2 for VVS

R2_AAH = Coupling(name = 'R2_AAH',
                  value = 'MW/(16*sw)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AAHe = Coupling(name = 'R2_AAHe',
                  value = '1/(8*sw*MW)*(Me**2*Ql**2)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AAHm = Coupling(name = 'R2_AAHm',
                  value = '1/(8*sw*MW)*(MM**2*Ql**2)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AAHtau = Coupling(name = 'R2_AAHtau',
                  value = '1/(8*sw*MW)*(MTA**2*Ql**2)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AAHu = Coupling(name = 'R2_AAHu',
                  value = '1/(8*sw*MW)*(MU**2*Qu**2*Ncol)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AAHc = Coupling(name = 'R2_AAHc',
                  value = '1/(8*sw*MW)*(MC**2*Qu**2*Ncol)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AAHt = Coupling(name = 'R2_AAHt',
                  value = '1/(8*sw*MW)*(MT**2*Qu**2*Ncol)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AAHd = Coupling(name = 'R2_AAHd',
                  value = '1/(8*sw*MW)*(MD**2*Qd**2*Ncol)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AAHs = Coupling(name = 'R2_AAHs',
                  value = '1/(8*sw*MW)*(MS**2*Qd**2*Ncol)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AAHb = Coupling(name = 'R2_AAHb',
                  value = '1/(8*sw*MW)*(MB**2*Qd**2*Ncol)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AZH = Coupling(name = 'R2_AZH',
                  value = 'MW/(32*sw**2*cw)*(1+2*cw**2)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AZHe = Coupling(name = 'R2_AZHe',
                  value = '1/(8*MW*cw)*(Me**2*Ql*(I3l/(2*sw**2)-Ql))*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AZHm = Coupling(name = 'R2_AZHm',
                  value = '1/(8*MW*cw)*(MM**2*Ql*(I3l/(2*sw**2)-Ql))*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AZHtau = Coupling(name = 'R2_AZHtau',
                  value = '1/(8*MW*cw)*(MTA**2*Ql*(I3l/(2*sw**2)-Ql))*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AZHu = Coupling(name = 'R2_AZHu',
                  value = '1/(8*MW*cw)*Ncol*(MU**2*Qu*(I3u/(2*sw**2)-Qu))*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AZHc = Coupling(name = 'R2_AZHc',
                  value = '1/(8*MW*cw)*Ncol*(MC**2*Qu*(I3u/(2*sw**2)-Qu))*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AZHt = Coupling(name = 'R2_AZHt',
                  value = '1/(8*MW*cw)*Ncol*(MT**2*Qu*(I3u/(2*sw**2)-Qu))*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AZHd = Coupling(name = 'R2_AZHd',
                  value = '1/(8*MW*cw)*Ncol*(MD**2*Qd*(I3d/(2*sw**2)-Qd))*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AZHs = Coupling(name = 'R2_AZHs',
                  value = '1/(8*MW*cw)*Ncol*(MS**2*Qd*(I3d/(2*sw**2)-Qd))*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_AZHb = Coupling(name = 'R2_AZHb',
                  value = '1/(8*MW*cw)*Ncol*(MB**2*Qd*(I3d/(2*sw**2)-Qd))*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_ZZH = Coupling(name = 'R2_ZZH',
                  value = '-MW/(16*sw**3)*(sw**2 -2)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_ZZHe = Coupling(name = 'R2_ZZHe',
                  value = '-1/(8*MW*cw**2)*Me**2*(Ql*I3l/sw-Ql**2*sw-I3l**2/sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_ZZHm = Coupling(name = 'R2_ZZHm',
                  value = '-1/(8*MW*cw**2)*MM**2*(Ql*I3l/sw-Ql**2*sw-I3l**2/sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_ZZHtau = Coupling(name = 'R2_ZZHtau',
                  value = '-1/(8*MW*cw**2)*MTA**2*(Ql*I3l/sw-Ql**2*sw-I3l**2/sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_ZZHu = Coupling(name = 'R2_ZZHu',
                  value = '-1/(8*MW*cw**2)*Ncol*MU**2*(Qu*I3u/sw-Qu**2*sw-I3u**2/sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_ZZHc = Coupling(name = 'R2_ZZHc',
                  value = '-1/(8*MW*cw**2)*Ncol*MC**2*(Qu*I3u/sw-Qu**2*sw-I3u**2/sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_ZZHt = Coupling(name = 'R2_ZZHt',
                  value = '-1/(8*MW*cw**2)*Ncol*MT**2*(Qu*I3u/sw-Qu**2*sw-I3u**2/sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_ZZHd = Coupling(name = 'R2_ZZHd',
                  value = '-1/(8*MW*cw**2)*Ncol*MD**2*(Qd*I3d/sw-Qd**2*sw-I3d**2/sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_ZZHs = Coupling(name = 'R2_ZZHs',
                  value = '-1/(8*MW*cw**2)*Ncol*MS**2*(Qd*I3d/sw-Qd**2*sw-I3d**2/sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_ZZHb = Coupling(name = 'R2_ZZHb',
                  value = '-1/(8*MW*cw**2)*Ncol*MB**2*(Qd*I3d/sw-Qd**2*sw-I3d**2/sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_WWH = Coupling(name = 'R2_WWH',
                  value = 'MW/(8*sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_WWHe = Coupling(name = 'R2_WWHe',
                  value = 'Me**2/(32*MW*sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_WWHm = Coupling(name = 'R2_WWHm',
                  value = 'MM**2/(32*MW*sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_WWHtau = Coupling(name = 'R2_WWHtau',
                  value = 'MTA**2/(32*MW*sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_WWHud = Coupling(name = 'R2_WWHud',
                  value = 'SCKM11*Ncol*(MU**2 +MD**2)/(32*MW*sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_WWHus = Coupling(name = 'R2_WWHus',
                  value = 'SCKM12*Ncol*(MU**2 +MS**2)/(32*MW*sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_WWHub = Coupling(name = 'R2_WWHub',
                  value = 'SCKM13*Ncol*(MU**2 +MB**2)/(32*MW*sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_WWHcd = Coupling(name = 'R2_WWHcd',
                  value = 'SCKM21*Ncol*(MC**2 +MD**2)/(32*MW*sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_WWHcs = Coupling(name = 'R2_WWHcs',
                  value = 'SCKM22*Ncol*(MC**2 +MS**2)/(32*MW*sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_WWHcb = Coupling(name = 'R2_WWHcb',
                  value = 'SCKM23*Ncol*(MC**2 +MB**2)/(32*MW*sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_WWHtd = Coupling(name = 'R2_WWHtd',
                  value = 'SCKM31*Ncol*(MT**2 +MD**2)/(32*MW*sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_WWHts = Coupling(name = 'R2_WWHts',
                  value = 'SCKM32*Ncol*(MT**2 +MS**2)/(32*MW*sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_WWHtb = Coupling(name = 'R2_WWHtb',
                  value = 'SCKM33*Ncol*(MT**2 +MB**2)/(32*MW*sw**3)*(-complex(0,1)*R2SFF)',
                  order = {'QED':3})

R2_WAGp = Coupling(name = 'R2_WAGp',
                   value = 'MW/(32*sw**2)*R2SFF',
                   order = {'QED':3})


R2_WAGpud = Coupling(name = 'R2_WAGpud',
                   value = 'SCKM11*Ncol*(Qu*MD**2 - Qd*MU**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGpus = Coupling(name = 'R2_WAGpus',
                   value = 'SCKM12*Ncol*(Qu*MS**2 - Qd*MU**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGpub = Coupling(name = 'R2_WAGpub',
                   value = 'SCKM13*Ncol*(Qu*MB**2 - Qd*MU**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGpcd = Coupling(name = 'R2_WAGpcd',
                   value = 'SCKM21*Ncol*(Qu*MD**2 - Qd*MC**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGpcs = Coupling(name = 'R2_WAGpcs',
                   value = 'SCKM22*Ncol*(Qu*MS**2 - Qd*MC**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGpcb = Coupling(name = 'R2_WAGpcb',
                   value = 'SCKM23*Ncol*(Qu*MB**2 - Qd*MC**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGptd = Coupling(name = 'R2_WAGptd',
                   value = 'SCKM31*Ncol*(Qu*MD**2 - Qd*MT**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGpts = Coupling(name = 'R2_WAGpts',
                   value = 'SCKM32*Ncol*(Qu*MS**2 - Qd*MT**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGptb = Coupling(name = 'R2_WAGptb',
                   value = 'SCKM33*Ncol*(Qu*MB**2 - Qd*MT**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGm = Coupling(name = 'R2_WAGm',
                   value = '-MW/(32*sw**2)*R2SFF',
                   order = {'QED':3})

R2_WAGmud = Coupling(name = 'R2_WAGmud',
                   value = '-SCKM11*Ncol*(Qu*MD**2 - Qd*MU**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGmus = Coupling(name = 'R2_WAGmus',
                   value = '-SCKM12*Ncol*(Qu*MS**2 - Qd*MU**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGmub = Coupling(name = 'R2_WAGmub',
                   value = '-SCKM13*Ncol*(Qu*MB**2 - Qd*MU**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGmcd = Coupling(name = 'R2_WAGmcd',
                   value = '-SCKM21*Ncol*(Qu*MD**2 - Qd*MC**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGmcs = Coupling(name = 'R2_WAGmcs',
                   value = '-SCKM22*Ncol*(Qu*MS**2 - Qd*MC**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGmcb = Coupling(name = 'R2_WAGmcb',
                   value = '-SCKM23*Ncol*(Qu*MB**2 - Qd*MC**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGmtd = Coupling(name = 'R2_WAGmtd',
                   value = '-SCKM31*Ncol*(Qu*MD**2 - Qd*MT**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGmts = Coupling(name = 'R2_WAGmts',
                   value = '-SCKM32*Ncol*(Qu*MS**2 - Qd*MT**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WAGmtb = Coupling(name = 'R2_WAGmtb',
                   value = '-SCKM33*Ncol*(Qu*MB**2 - Qd*MT**2)/(32*sw**2*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGp = Coupling(name = 'R2_WZGp',
                   value = '-MW/(32*sw*cw)*R2SFF',
                   order = {'QED':3})

R2_WZGpud = Coupling(name = 'R2_WZGpud',
                   value = '-SCKM11*Ncol*(Qu*MD**2 - Qd*MU**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGpus = Coupling(name = 'R2_WZGpus',
                   value = '-SCKM12*Ncol*(Qu*MS**2 - Qd*MU**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGpub = Coupling(name = 'R2_WZGpub',
                   value = '-SCKM13*Ncol*(Qu*MB**2 - Qd*MU**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGpcd = Coupling(name = 'R2_WZGpcd',
                   value = '-SCKM21*Ncol*(Qu*MD**2 - Qd*MC**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGpcs = Coupling(name = 'R2_WZGpcs',
                   value = '-SCKM22*Ncol*(Qu*MS**2 - Qd*MC**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGpcb = Coupling(name = 'R2_WZGpcb',
                   value = '-SCKM23*Ncol*(Qu*MB**2 - Qd*MC**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGptd = Coupling(name = 'R2_WZGptd',
                   value = '-SCKM31*Ncol*(Qu*MD**2 - Qd*MT**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGpts = Coupling(name = 'R2_WZGpts',
                   value = '-SCKM32*Ncol*(Qu*MS**2 - Qd*MT**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGptb = Coupling(name = 'R2_WZGptb',
                   value = '-SCKM33*Ncol*(Qu*MB**2 - Qd*MT**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGm = Coupling(name = 'R2_WZGm',
                   value = 'MW/(32*sw*cw)*R2SFF',
                   order = {'QED':3})

R2_WZGmud = Coupling(name = 'R2_WZGmud',
                   value = 'SCKM11*Ncol*(Qu*MD**2 - Qd*MU**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGmus = Coupling(name = 'R2_WZGmus',
                   value = 'SCKM12*Ncol*(Qu*MS**2 - Qd*MU**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGmub = Coupling(name = 'R2_WZGmub',
                   value = 'SCKM13*Ncol*(Qu*MB**2 - Qd*MU**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGmcd = Coupling(name = 'R2_WZGmcd',
                   value = 'SCKM21*Ncol*(Qu*MD**2 - Qd*MC**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGmcs = Coupling(name = 'R2_WZGmcs',
                   value = 'SCKM22*Ncol*(Qu*MS**2 - Qd*MC**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGmcb = Coupling(name = 'R2_WZGmcb',
                   value = 'SCKM23*Ncol*(Qu*MB**2 - Qd*MC**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGmtd = Coupling(name = 'R2_WZGmtd',
                   value = 'SCKM31*Ncol*(Qu*MD**2 - Qd*MT**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGmts = Coupling(name = 'R2_WZGmts',
                   value = 'SCKM32*Ncol*(Qu*MS**2 - Qd*MT**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_WZGmtb = Coupling(name = 'R2_WZGmtb',
                   value = 'SCKM33*Ncol*(Qu*MB**2 - Qd*MT**2)/(32*sw*cw*MW)*R2SFF',
                   order = {'QED':3})

R2_AWW = Coupling(name = 'R2_AWW',
                   value = '(7+4*lhv)/(96*sw**2)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_AWWlv = Coupling(name = 'R2_AWWlv',
                   value = '1/(48*sw**2)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_AWWud = Coupling(name = 'R2_AWWud',
                   value = 'Ncol*SCKM11/(48*sw**2)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_AWWus = Coupling(name = 'R2_AWWus',
                   value = 'Ncol*SCKM12/(48*sw**2)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_AWWub = Coupling(name = 'R2_AWWub',
                   value = 'Ncol*SCKM13/(48*sw**2)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_AWWcd = Coupling(name = 'R2_AWWcd',
                   value = 'Ncol*SCKM21/(48*sw**2)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_AWWcs = Coupling(name = 'R2_AWWcs',
                   value = 'Ncol*SCKM22/(48*sw**2)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_AWWcb = Coupling(name = 'R2_AWWcb',
                   value = 'Ncol*SCKM23/(48*sw**2)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_AWWtd = Coupling(name = 'R2_AWWtd',
                   value = 'Ncol*SCKM31/(48*sw**2)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_AWWts = Coupling(name = 'R2_AWWts',
                   value = 'Ncol*SCKM32/(48*sw**2)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_AWWtb = Coupling(name = 'R2_AWWtb',
                   value = 'Ncol*SCKM33/(48*sw**2)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_ZWW = Coupling(name = 'R2_ZWW',
                   value = '(7+4*lhv)/(96*sw**2)*(cw/sw)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_ZWWlv = Coupling(name = 'R2_ZWWlv',
                   value = '1/(48*sw**2)*(cw/sw)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_ZWWud = Coupling(name = 'R2_ZWWud',
                   value = 'Ncol*SCKM11/(48*sw**2)*(cw/sw)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_ZWWus = Coupling(name = 'R2_ZWWus',
                   value = 'Ncol*SCKM12/(48*sw**2)*(cw/sw)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_ZWWub = Coupling(name = 'R2_ZWWub',
                   value = 'Ncol*SCKM13/(48*sw**2)*(cw/sw)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_ZWWcd = Coupling(name = 'R2_ZWWcd',
                   value = 'Ncol*SCKM21/(48*sw**2)*(cw/sw)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_ZWWcs = Coupling(name = 'R2_ZWWcs',
                   value = 'Ncol*SCKM22/(48*sw**2)*(cw/sw)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_ZWWcb = Coupling(name = 'R2_ZWWcb',
                   value = 'Ncol*SCKM23/(48*sw**2)*(cw/sw)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_ZWWtd = Coupling(name = 'R2_ZWWtd',
                   value = 'Ncol*SCKM31/(48*sw**2)*(cw/sw)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_ZWWts = Coupling(name = 'R2_ZWWts',
                   value = 'Ncol*SCKM32/(48*sw**2)*(cw/sw)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

R2_ZWWtb = Coupling(name = 'R2_ZWWtb',
                   value = 'Ncol*SCKM33/(48*sw**2)*(cw/sw)*(-complex(0,1)*R2SFF)',
                   order = {'QED':3})

# R2 for SSSS

R2_HHHH = Coupling(name = 'R2_HHHH',
                   value = '1/(64.*sw**4)*(3*MH**2/(2*MW**2)*(1+1/(2*cw**2))+(1-12*lhv)/2.*(1+1./(2*cw**4)))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_HHHHe = Coupling(name = 'R2_HHHHe',
                   value = '1/(64.*sw**4)*(5*Me**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_HHHHm = Coupling(name = 'R2_HHHHm',
                   value = '1/(64.*sw**4)*(5*MM**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_HHHHtau = Coupling(name = 'R2_HHHHtau',
                   value = '1/(64.*sw**4)*(5*MTA**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_HHHHu = Coupling(name = 'R2_HHHHu',
                   value = '1/(64.*sw**4)*(5*Ncol*MU**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_HHHHd = Coupling(name = 'R2_HHHHd',
                   value = '1/(64.*sw**4)*(5*Ncol*MD**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_HHHHs = Coupling(name = 'R2_HHHHs',
                   value = '1/(64.*sw**4)*(5*Ncol*MS**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_HHHHc = Coupling(name = 'R2_HHHHc',
                   value = '1/(64.*sw**4)*(5*Ncol*MC**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_HHHHb = Coupling(name = 'R2_HHHHb',
                   value = '1/(64.*sw**4)*(5*Ncol*MB**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_HHHHt = Coupling(name = 'R2_HHHHt',
                   value = '1/(64.*sw**4)*(5*Ncol*MT**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_G0G0HH = Coupling(name = 'R2_G0G0HH',
                   value = '1/(192.*sw**4)*(3*MH**2/(2*MW**2)*(1+1/(2*cw**2))+(1-12*lhv)/2.*(1+1./(2*cw**4)))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_G0G0HHe = Coupling(name = 'R2_G0G0HHe',
                   value = '1/(192.*sw**4)*(5*Me**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_G0G0HHm = Coupling(name = 'R2_G0G0HHm',
                   value = '1/(192.*sw**4)*(5*MM**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_G0G0HHtau = Coupling(name = 'R2_G0G0HHtau',
                   value = '1/(192.*sw**4)*(5*MTA**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_G0G0HHu = Coupling(name = 'R2_G0G0HHu',
                   value = '1/(192.*sw**4)*(5*Ncol*MU**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_G0G0HHd = Coupling(name = 'R2_G0G0HHd',
                   value = '1/(192.*sw**4)*(5*Ncol*MD**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_G0G0HHs = Coupling(name = 'R2_G0G0HHs',
                   value = '1/(192.*sw**4)*(5*Ncol*MS**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_G0G0HHc = Coupling(name = 'R2_G0G0HHc',
                   value = '1/(192.*sw**4)*(5*Ncol*MC**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_G0G0HHb = Coupling(name = 'R2_G0G0HHb',
                   value = '1/(192.*sw**4)*(5*Ncol*MB**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_G0G0HHt = Coupling(name = 'R2_G0G0HHt',
                   value = '1/(192.*sw**4)*(5*Ncol*MT**4/MW**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_GmGpHH = Coupling(name = 'R2_GmGpHH',
                   value = '1/(64.*sw**4)*(MH**2/(2.*MW**2)*(1+1/(2*cw**2))+(1-12*lhv)/4.*(1+sw**2/(3*cw**2)*(1+1./cw**2)))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_GmGpHHe = Coupling(name = 'R2_GmGpHHe',
                   value = '1/(64.*sw**4)*(5*Me**4/(3*MW**4))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_GmGpHHm = Coupling(name = 'R2_GmGpHHm',
                   value = '1/(64.*sw**4)*(5*MM**4/(3*MW**4))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_GmGpHHtau = Coupling(name = 'R2_GmGpHHtau',
                   value = '1/(64.*sw**4)*(5*MTA**4/(3*MW**4))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_GmGpHHud = Coupling(name = 'R2_GmGpHHud',
                   value = '1/(64.*sw**4)*Ncol*SCKM11*(5*(MU**4 +MD**4)/(3*MW**4))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_GmGpHHus = Coupling(name = 'R2_GmGpHHus',
                   value = '1/(64.*sw**4)*Ncol*SCKM12*(5*(MU**4 +MS**4)/(3*MW**4))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_GmGpHHub = Coupling(name = 'R2_GmGpHHub',
                   value = '1/(64.*sw**4)*Ncol*SCKM13*(5*(MU**4 +MB**4)/(3*MW**4))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_GmGpHHcd = Coupling(name = 'R2_GmGpHHcd',
                   value = '1/(64.*sw**4)*Ncol*SCKM21*(5*(MC**4 +MD**4)/(3*MW**4))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_GmGpHHcs = Coupling(name = 'R2_GmGpHHcs',
                   value = '1/(64.*sw**4)*Ncol*SCKM22*(5*(MC**4 +MS**4)/(3*MW**4))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_GmGpHHcb = Coupling(name = 'R2_GmGpHHcb',
                   value = '1/(64.*sw**4)*Ncol*SCKM23*(5*(MC**4 +MB**4)/(3*MW**4))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_GmGpHHtd = Coupling(name = 'R2_GmGpHHtd',
                   value = '1/(64.*sw**4)*Ncol*SCKM31*(5*(MT**4 +MD**4)/(3*MW**4))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_GmGpHHts = Coupling(name = 'R2_GmGpHHts',
                   value = '1/(64.*sw**4)*Ncol*SCKM32*(5*(MT**4 +MS**4)/(3*MW**4))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_GmGpHHtb = Coupling(name = 'R2_GmGpHHtb',
                   value = '1/(64.*sw**4)*Ncol*SCKM33*(5*(MT**4 +MB**4)/(3*MW**4))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_GmGmGpGp = Coupling(name = 'R2_GmGmGpGp',
                       value = '1/(32.*sw**4)*(MH**2/(2*MW**2)*(1+1/(2*cw**2))+(1./4.-3*lhv)*(1+sw**4)+(1./6.-2*lhv)*(sw**2 +2*sw**6/cw**2)+(1./12.-lhv)*sw**8/cw**4)*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpe = Coupling(name = 'R2_GmGmGpGpe',
                       value = '1/(32.*sw**4)*(5*Me**4/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpm = Coupling(name = 'R2_GmGmGpGpm',
                       value = '1/(32.*sw**4)*(5*MM**4/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGptau = Coupling(name = 'R2_GmGmGpGptau',
                       value = '1/(32.*sw**4)*(5*MTA**4/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpud = Coupling(name = 'R2_GmGmGpGpud',
                       value = '1/(32.*sw**4)*Ncol*SCKM11**2*(5*(MU**4 +MD**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpus = Coupling(name = 'R2_GmGmGpGpus',
                       value = '1/(32.*sw**4)*Ncol*SCKM12**2*(5*(MU**4 +MS**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpub = Coupling(name = 'R2_GmGmGpGpub',
                       value = '1/(32.*sw**4)*Ncol*SCKM13**2*(5*(MU**4 +MB**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpcd = Coupling(name = 'R2_GmGmGpGpcd',
                       value = '1/(32.*sw**4)*Ncol*SCKM21**2*(5*(MC**4 +MD**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpcs = Coupling(name = 'R2_GmGmGpGpcs',
                       value = '1/(32.*sw**4)*Ncol*SCKM22**2*(5*(MC**4 +MS**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpcb = Coupling(name = 'R2_GmGmGpGpcb',
                       value = '1/(32.*sw**4)*Ncol*SCKM23**2*(5*(MC**4 +MB**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGptd = Coupling(name = 'R2_GmGmGpGptd',
                       value = '1/(32.*sw**4)*Ncol*SCKM31**2*(5*(MT**4 +MD**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpts = Coupling(name = 'R2_GmGmGpGpts',
                       value = '1/(32.*sw**4)*Ncol*SCKM32**2*(5*(MT**4 +MS**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGptb = Coupling(name = 'R2_GmGmGpGptb',
                       value = '1/(32.*sw**4)*Ncol*SCKM33**2*(5*(MT**4 +MB**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpucd = Coupling(name = 'R2_GmGmGpGpucd',
                       value = '2./(32.*sw**4)*Ncol*SCKM11*SCKM21*(5*(MU**2*MC**2 +MD**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpucs = Coupling(name = 'R2_GmGmGpGpucs',
                       value = '2./(32.*sw**4)*Ncol*SCKM12*SCKM22*(5*(MU**2*MC**2 +MS**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpucb = Coupling(name = 'R2_GmGmGpGpucb',
                       value = '2./(32.*sw**4)*Ncol*SCKM13*SCKM23*(5*(MU**2*MC**2 +MB**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGputd = Coupling(name = 'R2_GmGmGpGputd',
                       value = '2./(32.*sw**4)*Ncol*SCKM11*SCKM31*(5*(MU**2*MT**2 +MD**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGputs = Coupling(name = 'R2_GmGmGpGputs',
                       value = '2./(32.*sw**4)*Ncol*SCKM12*SCKM32*(5*(MU**2*MT**2 +MS**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGputb = Coupling(name = 'R2_GmGmGpGputb',
                       value = '2./(32.*sw**4)*Ncol*SCKM13*SCKM33*(5*(MU**2*MT**2 +MB**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpctd = Coupling(name = 'R2_GmGmGpGpctd',
                       value = '2./(32.*sw**4)*Ncol*SCKM21*SCKM31*(5*(MC**2*MT**2 +MD**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpcts = Coupling(name = 'R2_GmGmGpGpcts',
                       value = '2./(32.*sw**4)*Ncol*SCKM22*SCKM32*(5*(MC**2*MT**2 +MS**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpctb = Coupling(name = 'R2_GmGmGpGpctb',
                       value = '2./(32.*sw**4)*Ncol*SCKM23*SCKM33*(5*(MC**2*MT**2 +MB**4)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpuds = Coupling(name = 'R2_GmGmGpGpuds',
                       value = '2./(32.*sw**4)*Ncol*SCKM11*SCKM12*(5*(MU**4 +MD**2*MS**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpcds = Coupling(name = 'R2_GmGmGpGpcds',
                       value = '2./(32.*sw**4)*Ncol*SCKM21*SCKM22*(5*(MC**4 +MD**2*MS**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGptds = Coupling(name = 'R2_GmGmGpGptds',
                       value = '2./(32.*sw**4)*Ncol*SCKM31*SCKM32*(5*(MT**4 +MD**2*MS**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpudb = Coupling(name = 'R2_GmGmGpGpudb',
                       value = '2./(32.*sw**4)*Ncol*SCKM11*SCKM13*(5*(MU**4 +MD**2*MB**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpcdb = Coupling(name = 'R2_GmGmGpGpcdb',
                       value = '2./(32.*sw**4)*Ncol*SCKM21*SCKM23*(5*(MC**4 +MD**2*MB**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGptdb = Coupling(name = 'R2_GmGmGpGptdb',
                       value = '2./(32.*sw**4)*Ncol*SCKM31*SCKM33*(5*(MT**4 +MD**2*MB**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpusb = Coupling(name = 'R2_GmGmGpGpusb',
                       value = '2./(32.*sw**4)*Ncol*SCKM12*SCKM13*(5*(MU**4 +MS**2*MB**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpcsb = Coupling(name = 'R2_GmGmGpGpcsb',
                       value = '2./(32.*sw**4)*Ncol*SCKM22*SCKM23*(5*(MC**4 +MS**2*MB**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGptsb = Coupling(name = 'R2_GmGmGpGptsb',
                       value = '2./(32.*sw**4)*Ncol*SCKM32*SCKM33*(5*(MT**4 +MS**2*MB**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpucds = Coupling(name = 'R2_GmGmGpGpucds',
                       value = '2./(32.*sw**4)*Ncol*(CKMucds)*(5*(MU**2*MC**2 +MD**2*MS**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpucdb = Coupling(name = 'R2_GmGmGpGpucdb',
                       value = '2./(32.*sw**4)*Ncol*(CKMucdb)*(5*(MU**2*MC**2 +MD**2*MB**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpucsb = Coupling(name = 'R2_GmGmGpGpucsb',
                       value = '2./(32.*sw**4)*Ncol*(CKMucsb)*(5*(MU**2*MC**2 +MB**2*MS**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGputds = Coupling(name = 'R2_GmGmGpGputds',
                       value = '2./(32.*sw**4)*Ncol*(CKMutds)*(5*(MU**2*MT**2 +MD**2*MS**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGputdb = Coupling(name = 'R2_GmGmGpGputdb',
                       value = '2./(32.*sw**4)*Ncol*(CKMutdb)*(5*(MU**2*MT**2 +MD**2*MB**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGputsb = Coupling(name = 'R2_GmGmGpGputsb',
                       value = '2./(32.*sw**4)*Ncol*(CKMutsb)*(5*(MU**2*MT**2 +MS**2*MB**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpctds = Coupling(name = 'R2_GmGmGpGpctds',
                       value = '2./(32.*sw**4)*Ncol*(CKMctds)*(5*(MT**2*MC**2 +MD**2*MS**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpctdb = Coupling(name = 'R2_GmGmGpGpctdb',
                       value = '2./(32.*sw**4)*Ncol*(CKMctdb)*(5*(MT**2*MC**2 +MD**2*MB**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

R2_GmGmGpGpctsb = Coupling(name = 'R2_GmGmGpGpctsb',
                       value = '2./(32.*sw**4)*Ncol*(CKMctsb)*(5*(MT**2*MC**2 +MB**2*MS**2)/(3*MW**4))*(complex(0,1)*R24S)',
                       order = {'QED':4})

# R2 for VVVV

R2_AAAA = Coupling(name = 'R2_AAAA',
                   value = '-1./12.*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAAAl = Coupling(name = 'R2_AAAAl',
                   value = 'Ql**4/12.*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAAAu = Coupling(name = 'R2_AAAAu',
                   value = 'Ncol*Qu**4/12.*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAAAd = Coupling(name = 'R2_AAAAd',
                   value = 'Ncol*Qd**4/12.*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAAZ = Coupling(name = 'R2_AAAZ',
                   value = '-1./12.*(cw/sw)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAAZl = Coupling(name = 'R2_AAAZl',
                   value = '-1./12.*(sw/cw*Ql**4 -1./(2*sw*cw)*Ql**3*I3l)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAAZu = Coupling(name = 'R2_AAAZu',
                   value = '-Ncol/12.*(sw/cw*Qu**4 -1./(2*sw*cw)*Qu**3*I3u)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAAZd = Coupling(name = 'R2_AAAZd',
                   value = '-Ncol/12.*(sw/cw*Qd**4 -1./(2*sw*cw)*Qd**3*I3d)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAZZ = Coupling(name = 'R2_AAZZ',
                   value = '-1./12.*(cw**2/sw**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAZZl = Coupling(name = 'R2_AAZZl',
                   value = '1./24.*(sw**2/cw**2*Ql**4 +(sw/cw*Ql**2 -1/(sw*cw)*Ql*I3l)**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAZZu = Coupling(name = 'R2_AAZZu',
                   value = 'Ncol/24.*(sw**2/cw**2*Qu**4 +(sw/cw*Qu**2 -1/(sw*cw)*Qu*I3u)**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAZZd = Coupling(name = 'R2_AAZZd',
                   value = 'Ncol/24.*(sw**2/cw**2*Qd**4 +(sw/cw*Qd**2 -1/(sw*cw)*Qd*I3d)**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AZZZ = Coupling(name = 'R2_AZZZ',
                   value = '-1./12.*(cw**3/sw**3)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AZZZl = Coupling(name = 'R2_AZZZl',
                   value = '-1./12.*(sw**3/cw**3*Ql**4 -3./2.*sw/cw**3*Ql**3*I3l+3./(2.*sw*cw**3)*Ql**2*I3l**2 -1./(2*sw**3*cw**3)*Ql*I3l**3)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AZZZu = Coupling(name = 'R2_AZZZu',
                   value = '-Ncol/12.*(sw**3/cw**3*Qu**4 -3./2.*sw/cw**3*Qu**3*I3u+3./(2.*sw*cw**3)*Qu**2*I3u**2 -1./(2*sw**3*cw**3)*Qu*I3u**3)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AZZZd = Coupling(name = 'R2_AZZZd',
                   value = '-Ncol/12.*(sw**3/cw**3*Qd**4 -3./2.*sw/cw**3*Qd**3*I3d+3./(2.*sw*cw**3)*Qd**2*I3d**2 -1./(2*sw**3*cw**3)*Qd*I3d**3)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZZZ = Coupling(name = 'R2_ZZZZ',
                   value = '-1./12.*(cw**4/sw**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZZZl = Coupling(name = 'R2_ZZZZl',
                   value = '1./12.*(sw**4/cw**4*Ql**4 -2.*sw**2/cw**4*Ql**3*I3l+3./(cw**4)*Ql**2*I3l**2 -2./(sw**2*cw**4)*Ql*I3l**3+1./(2*sw**4*cw**4)*I3l**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZZZu = Coupling(name = 'R2_ZZZZu',
                   value = 'Ncol/12.*(sw**4/cw**4*Qu**4 -2.*sw**2/cw**4*Qu**3*I3u+3./(cw**4)*Qu**2*I3u**2 -2./(sw**2*cw**4)*Qu*I3u**3+1./(2*sw**4*cw**4)*I3u**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZZZd = Coupling(name = 'R2_ZZZZd',
                   value = 'Ncol/12.*(sw**4/cw**4*Qd**4 -2.*sw**2/cw**4*Qd**3*I3d+3./(cw**4)*Qd**2*I3d**2 -2./(sw**2*cw**4)*Qd*I3d**3+1./(2*sw**4*cw**4)*I3d**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZZZv = Coupling(name = 'R2_ZZZZv',
                   value = '1./12.*(1./(2*sw**4*cw**4)*I3v**4)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAWW1 = Coupling(name = 'R2_AAWW1',
                    value = '1./(16*sw**2)*(10+4*lhv)/3.*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW2 = Coupling(name = 'R2_AAWW2',
                    value = '-1./(16*sw**2)*(7+2*lhv)/3.*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW1lv = Coupling(name = 'R2_AAWW1lv',
                    value = '1./(16*sw**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW2lv = Coupling(name = 'R2_AAWW2lv',
                    value = '-1./(16*sw**2)*1./3.*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW1ud = Coupling(name = 'R2_AAWW1ud',
                    value = '1./(16*sw**2)*25./27.*Ncol*SCKM11*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW2ud = Coupling(name = 'R2_AAWW2ud',
                    value = '-1./(16*sw**2)*11./27.*Ncol*SCKM11*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW1us = Coupling(name = 'R2_AAWW1us',
                    value = '1./(16*sw**2)*25./27.*Ncol*SCKM12*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW2us = Coupling(name = 'R2_AAWW2us',
                    value = '-1./(16*sw**2)*11./27.*Ncol*SCKM12*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW1ub = Coupling(name = 'R2_AAWW1ub',
                    value = '1./(16*sw**2)*25./27.*Ncol*SCKM13*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW2ub = Coupling(name = 'R2_AAWW2ub',
                    value = '-1./(16*sw**2)*11./27.*Ncol*SCKM13*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW1cd = Coupling(name = 'R2_AAWW1cd',
                    value = '1./(16*sw**2)*25./27.*Ncol*SCKM21*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW2cd = Coupling(name = 'R2_AAWW2cd',
                    value = '-1./(16*sw**2)*11./27.*Ncol*SCKM21*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW1cs = Coupling(name = 'R2_AAWW1cs',
                    value = '1./(16*sw**2)*25./27.*Ncol*SCKM22*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW2cs = Coupling(name = 'R2_AAWW2cs',
                    value = '-1./(16*sw**2)*11./27.*Ncol*SCKM22*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW1cb = Coupling(name = 'R2_AAWW1cb',
                    value = '1./(16*sw**2)*25./27.*Ncol*SCKM23*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW2cb = Coupling(name = 'R2_AAWW2cb',
                    value = '-1./(16*sw**2)*11./27.*Ncol*SCKM23*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW1td = Coupling(name = 'R2_AAWW1td',
                    value = '1./(16*sw**2)*25./27.*Ncol*SCKM31*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW2td = Coupling(name = 'R2_AAWW2td',
                    value = '-1./(16*sw**2)*11./27.*Ncol*SCKM31*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW1ts = Coupling(name = 'R2_AAWW1ts',
                    value = '1./(16*sw**2)*25./27.*Ncol*SCKM32*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW2ts = Coupling(name = 'R2_AAWW2ts',
                    value = '-1./(16*sw**2)*11./27.*Ncol*SCKM32*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW1tb = Coupling(name = 'R2_AAWW1tb',
                    value = '1./(16*sw**2)*25./27.*Ncol*SCKM33*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AAWW2tb = Coupling(name = 'R2_AAWW2tb',
                    value = '-1./(16*sw**2)*11./27.*Ncol*SCKM33*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW1 = Coupling(name = 'R2_AZWW1',
                    value = '-1./(16*sw*cw)*(10+4*lhv)/3.*(-cw**2/sw**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW2 = Coupling(name = 'R2_AZWW2',
                    value = '-1./(16*sw*cw)*(7+2*lhv)/3.*(cw**2/sw**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW1lv = Coupling(name = 'R2_AZWW1lv',
                    value = '-1./(16*sw*cw)*(1-11./(12.*sw**2))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW2lv = Coupling(name = 'R2_AZWW2lv',
                    value = '-1./(16*sw*cw)*(5./(12*sw**2)-1./3.)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW1ud = Coupling(name = 'R2_AZWW1ud',
                    value = '-1./(16*sw*cw)*Ncol*SCKM11*(25./27.-11./(12*sw**2))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW2ud = Coupling(name = 'R2_AZWW2ud',
                    value = '-1./(16*sw*cw)*Ncol*SCKM11*(5./(12*sw**2)-11./27.)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW1us = Coupling(name = 'R2_AZWW1us',
                    value = '-1./(16*sw*cw)*Ncol*SCKM12*(25./27.-11./(12*sw**2))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW2us = Coupling(name = 'R2_AZWW2us',
                    value = '-1./(16*sw*cw)*Ncol*SCKM12*(5./(12*sw**2)-11./27.)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW1ub = Coupling(name = 'R2_AZWW1ub',
                    value = '-1./(16*sw*cw)*Ncol*SCKM13*(25./27.-11./(12*sw**2))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW2ub = Coupling(name = 'R2_AZWW2ub',
                    value = '-1./(16*sw*cw)*Ncol*SCKM13*(5./(12*sw**2)-11./27.)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW1cd = Coupling(name = 'R2_AZWW1cd',
                    value = '-1./(16*sw*cw)*Ncol*SCKM21*(25./27.-11./(12*sw**2))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW2cd = Coupling(name = 'R2_AZWW2cd',
                    value = '-1./(16*sw*cw)*Ncol*SCKM21*(5./(12*sw**2)-11./27.)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW1cs = Coupling(name = 'R2_AZWW1cs',
                    value = '-1./(16*sw*cw)*Ncol*SCKM22*(25./27.-11./(12*sw**2))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW2cs = Coupling(name = 'R2_AZWW2cs',
                    value = '-1./(16*sw*cw)*Ncol*SCKM22*(5./(12*sw**2)-11./27.)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW1cb = Coupling(name = 'R2_AZWW1cb',
                    value = '-1./(16*sw*cw)*Ncol*SCKM23*(25./27.-11./(12*sw**2))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW2cb = Coupling(name = 'R2_AZWW2cb',
                    value = '-1./(16*sw*cw)*Ncol*SCKM23*(5./(12*sw**2)-11./27.)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW1td = Coupling(name = 'R2_AZWW1td',
                    value = '-1./(16*sw*cw)*Ncol*SCKM31*(25./27.-11./(12*sw**2))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW2td = Coupling(name = 'R2_AZWW2td',
                    value = '-1./(16*sw*cw)*Ncol*SCKM31*(5./(12*sw**2)-11./27.)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW1ts = Coupling(name = 'R2_AZWW1ts',
                    value = '-1./(16*sw*cw)*Ncol*SCKM32*(25./27.-11./(12*sw**2))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW2ts = Coupling(name = 'R2_AZWW2ts',
                    value = '-1./(16*sw*cw)*Ncol*SCKM32*(5./(12*sw**2)-11./27.)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW1tb = Coupling(name = 'R2_AZWW1tb',
                    value = '-1./(16*sw*cw)*Ncol*SCKM33*(25./27.-11./(12*sw**2))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_AZWW2tb = Coupling(name = 'R2_AZWW2tb',
                    value = '-1./(16*sw*cw)*Ncol*SCKM33*(5./(12*sw**2)-11./27.)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW1 = Coupling(name = 'R2_ZZWW1',
                    value = '1./(16*sw**2)*(10+4*lhv)/3.*(cw**2/sw**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW2 = Coupling(name = 'R2_ZZWW2',
                    value = '-1./(16*sw**2)*(7+2*lhv)/3.*(cw**2/sw**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW1lv = Coupling(name = 'R2_ZZWW1lv',
                    value = '1./(16*cw**2)*(1-11./(6.*sw**2)+11./(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW2lv = Coupling(name = 'R2_ZZWW2lv',
                    value = '1./(16*cw**2)*(5./(6*sw**2)-1./3.-5./(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW1ud = Coupling(name = 'R2_ZZWW1ud',
                    value = '1./(16*cw**2)*Ncol*SCKM11*(25./27.-11./(6.*sw**2)+11/(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW2ud = Coupling(name = 'R2_ZZWW2ud',
                    value = '1./(16*cw**2)*Ncol*SCKM11*(5./(6*sw**2)-11./27.-5./(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW1us = Coupling(name = 'R2_ZZWW1us',
                    value = '1./(16*cw**2)*Ncol*SCKM12*(25./27.-11./(6.*sw**2)+11/(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW2us = Coupling(name = 'R2_ZZWW2us',
                    value = '1./(16*cw**2)*Ncol*SCKM12*(5./(6*sw**2)-11./27.-5./(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW1ub = Coupling(name = 'R2_ZZWW1ub',
                    value = '1./(16*cw**2)*Ncol*SCKM13*(25./27.-11./(6.*sw**2)+11/(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW2ub = Coupling(name = 'R2_ZZWW2ub',
                    value = '1./(16*cw**2)*Ncol*SCKM13*(5./(6*sw**2)-11./27.-5./(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW1cd = Coupling(name = 'R2_ZZWW1cd',
                    value = '1./(16*cw**2)*Ncol*SCKM21*(25./27.-11./(6.*sw**2)+11/(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW2cd = Coupling(name = 'R2_ZZWW2cd',
                    value = '1./(16*cw**2)*Ncol*SCKM21*(5./(6*sw**2)-11./27.-5./(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW1cs = Coupling(name = 'R2_ZZWW1cs',
                    value = '1./(16*cw**2)*Ncol*SCKM22*(25./27.-11./(6.*sw**2)+11/(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW2cs = Coupling(name = 'R2_ZZWW2cs',
                    value = '1./(16*cw**2)*Ncol*SCKM22*(5./(6*sw**2)-11./27.-5./(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW1cb = Coupling(name = 'R2_ZZWW1cb',
                    value = '1./(16*cw**2)*Ncol*SCKM23*(25./27.-11./(6.*sw**2)+11/(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW2cb = Coupling(name = 'R2_ZZWW2cb',
                    value = '1./(16*cw**2)*Ncol*SCKM23*(5./(6*sw**2)-11./27.-5./(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW1td = Coupling(name = 'R2_ZZWW1td',
                    value = '1./(16*cw**2)*Ncol*SCKM31*(25./27.-11./(6.*sw**2)+11/(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW2td = Coupling(name = 'R2_ZZWW2td',
                    value = '1./(16*cw**2)*Ncol*SCKM31*(5./(6*sw**2)-11./27.-5./(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW1ts = Coupling(name = 'R2_ZZWW1ts',
                    value = '1./(16*cw**2)*Ncol*SCKM32*(25./27.-11./(6.*sw**2)+11/(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW2ts = Coupling(name = 'R2_ZZWW2ts',
                    value = '1./(16*cw**2)*Ncol*SCKM32*(5./(6*sw**2)-11./27.-5./(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW1tb = Coupling(name = 'R2_ZZWW1tb',
                    value = '1./(16*cw**2)*Ncol*SCKM33*(25./27.-11./(6.*sw**2)+11/(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_ZZWW2tb = Coupling(name = 'R2_ZZWW2tb',
                    value = '1./(16*cw**2)*Ncol*SCKM33*(5./(6*sw**2)-11./27.-5./(12*sw**4))*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1 = Coupling(name = 'R2_WWWW1',
                    value = '-1./(8*sw**4)*(7+2*lhv)/3.*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2 = Coupling(name = 'R2_WWWW2',
                    value = '1./(16*sw**4)*(3+2*lhv)/3.*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1lv = Coupling(name = 'R2_WWWW1lv',
                    value = '-1./(8*sw**4)*(5./12.)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2lv = Coupling(name = 'R2_WWWW2lv',
                    value = '1./(16*sw**4)*(1./2.)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1ud = Coupling(name = 'R2_WWWW1ud',
                    value = '-1./(8*sw**4)*(5./12.*Ncol*SCKM11**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2ud = Coupling(name = 'R2_WWWW2ud',
                    value = '1./(16*sw**4)*(Ncol/2.*SCKM11**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1us = Coupling(name = 'R2_WWWW1us',
                    value = '-1./(8*sw**4)*(5./12.*Ncol*SCKM12**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2us = Coupling(name = 'R2_WWWW2us',
                    value = '1./(16*sw**4)*(Ncol/2.*SCKM12**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1ub = Coupling(name = 'R2_WWWW1ub',
                    value = '-1./(8*sw**4)*(5./12.*Ncol*SCKM13**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2ub = Coupling(name = 'R2_WWWW2ub',
                    value = '1./(16*sw**4)*(Ncol/2.*SCKM13**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1cd = Coupling(name = 'R2_WWWW1cd',
                    value = '-1./(8*sw**4)*(5./12.*Ncol*SCKM21**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2cd = Coupling(name = 'R2_WWWW2cd',
                    value = '1./(16*sw**4)*(Ncol/2.*SCKM21**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1cs = Coupling(name = 'R2_WWWW1cs',
                    value = '-1./(8*sw**4)*(5./12.*Ncol*SCKM22**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2cs = Coupling(name = 'R2_WWWW2cs',
                    value = '1./(16*sw**4)*(Ncol/2.*SCKM22**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1cb = Coupling(name = 'R2_WWWW1cb',
                    value = '-1./(8*sw**4)*(5./12.*Ncol*SCKM23**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2cb = Coupling(name = 'R2_WWWW2cb',
                    value = '1./(16*sw**4)*(Ncol/2.*SCKM23**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1td = Coupling(name = 'R2_WWWW1td',
                    value = '-1./(8*sw**4)*(5./12.*Ncol*SCKM31**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2td = Coupling(name = 'R2_WWWW2td',
                    value = '1./(16*sw**4)*(Ncol/2.*SCKM31**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1ts = Coupling(name = 'R2_WWWW1ts',
                    value = '-1./(8*sw**4)*(5./12.*Ncol*SCKM32**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2ts = Coupling(name = 'R2_WWWW2ts',
                    value = '1./(16*sw**4)*(Ncol/2.*SCKM32**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1tb = Coupling(name = 'R2_WWWW1tb',
                    value = '-1./(8*sw**4)*(5./12.*Ncol*SCKM33**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2tb = Coupling(name = 'R2_WWWW2tb',
                    value = '1./(16*sw**4)*(Ncol/2.*SCKM33**2)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1ucd = Coupling(name = 'R2_WWWW1ucd',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM11*SCKM21)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2ucd = Coupling(name = 'R2_WWWW2ucd',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM11*SCKM21)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1ucs = Coupling(name = 'R2_WWWW1ucs',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM12*SCKM22)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2ucs = Coupling(name = 'R2_WWWW2ucs',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM12*SCKM22)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1ucb = Coupling(name = 'R2_WWWW1ucb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM13*SCKM23)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2ucb = Coupling(name = 'R2_WWWW2ucb',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM13*SCKM23)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1utd = Coupling(name = 'R2_WWWW1utd',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM11*SCKM31)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2utd = Coupling(name = 'R2_WWWW2utd',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM11*SCKM31)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1uts = Coupling(name = 'R2_WWWW1uts',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM12*SCKM32)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2uts = Coupling(name = 'R2_WWWW2uts',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM12*SCKM32)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1utb = Coupling(name = 'R2_WWWW1utb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM13*SCKM33)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2utb = Coupling(name = 'R2_WWWW2utb',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM13*SCKM33)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1ctd = Coupling(name = 'R2_WWWW1ctd',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM21*SCKM31)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2ctd = Coupling(name = 'R2_WWWW2ctd',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM21*SCKM31)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1cts = Coupling(name = 'R2_WWWW1cts',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM22*SCKM32)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2cts = Coupling(name = 'R2_WWWW2cts',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM22*SCKM32)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1ctb = Coupling(name = 'R2_WWWW1ctb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM23*SCKM33)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2ctb = Coupling(name = 'R2_WWWW2ctb',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM23*SCKM33)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1uds = Coupling(name = 'R2_WWWW1uds',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM11*SCKM12)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2uds = Coupling(name = 'R2_WWWW2uds',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM11*SCKM12)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1cds = Coupling(name = 'R2_WWWW1cds',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM21*SCKM22)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2cds = Coupling(name = 'R2_WWWW2cds',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM21*SCKM22)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1tds = Coupling(name = 'R2_WWWW1tds',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM31*SCKM32)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2tds = Coupling(name = 'R2_WWWW2tds',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM31*SCKM32)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1udb = Coupling(name = 'R2_WWWW1udb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM11*SCKM13)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2udb = Coupling(name = 'R2_WWWW2udb',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM11*SCKM13)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1cdb = Coupling(name = 'R2_WWWW1cdb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM21*SCKM23)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2cdb = Coupling(name = 'R2_WWWW2cdb',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM21*SCKM23)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1tdb = Coupling(name = 'R2_WWWW1tdb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM31*SCKM33)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2tdb = Coupling(name = 'R2_WWWW2tdb',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM31*SCKM33)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1usb = Coupling(name = 'R2_WWWW1usb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM12*SCKM13)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2usb = Coupling(name = 'R2_WWWW2usb',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM12*SCKM13)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1csb = Coupling(name = 'R2_WWWW1csb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM22*SCKM23)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2csb = Coupling(name = 'R2_WWWW2csb',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM22*SCKM23)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1tsb = Coupling(name = 'R2_WWWW1tsb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*SCKM32*SCKM33)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2tsb = Coupling(name = 'R2_WWWW2tsb',
                    value = '2./(16*sw**4)*(Ncol/2.*SCKM32*SCKM33)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1ucds = Coupling(name = 'R2_WWWW1ucds',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*CKMucds)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2ucds = Coupling(name = 'R2_WWWW2ucds',
                    value = '2./(16*sw**4)*(Ncol/2.*CKMucds)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1ucdb = Coupling(name = 'R2_WWWW1ucdb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*CKMucdb)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2ucdb = Coupling(name = 'R2_WWWW2ucdb',
                    value = '2./(16*sw**4)*(Ncol/2.*CKMucdb)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1ucsb = Coupling(name = 'R2_WWWW1ucsb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*CKMucsb)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2ucsb = Coupling(name = 'R2_WWWW2ucsb',
                    value = '2./(16*sw**4)*(Ncol/2.*CKMucsb)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1utds = Coupling(name = 'R2_WWWW1utds',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*CKMutds)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2utds = Coupling(name = 'R2_WWWW2utds',
                    value = '2./(16*sw**4)*(Ncol/2.*CKMutds)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1utdb = Coupling(name = 'R2_WWWW1utdb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*CKMutdb)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2utdb = Coupling(name = 'R2_WWWW2utdb',
                    value = '2./(16*sw**4)*(Ncol/2.*CKMutdb)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1utsb = Coupling(name = 'R2_WWWW1utsb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*CKMutsb)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2utsb = Coupling(name = 'R2_WWWW2utsb',
                    value = '2./(16*sw**4)*(Ncol/2.*CKMutsb)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1ctds = Coupling(name = 'R2_WWWW1ctds',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*CKMctds)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2ctds = Coupling(name = 'R2_WWWW2ctds',
                    value = '2./(16*sw**4)*(Ncol/2.*CKMctds)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1ctdb = Coupling(name = 'R2_WWWW1ctdb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*CKMctdb)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2ctdb = Coupling(name = 'R2_WWWW2ctdb',
                    value = '2./(16*sw**4)*(Ncol/2.*CKMctdb)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW1ctsb = Coupling(name = 'R2_WWWW1ctsb',
                    value = '-2./(8*sw**4)*(5./12.*Ncol*CKMctsb)*(complex(0,1)*R24S)',
                    order = {'QED':4})

R2_WWWW2ctsb = Coupling(name = 'R2_WWWW2ctsb',
                    value = '2./(16*sw**4)*(Ncol/2.*CKMctsb)*(complex(0,1)*R24S)',
                    order = {'QED':4})

# R2 for VVSS

R2_AAHH = Coupling(name= 'R2_AAHH',
                   value = '(1/(16*sw**2))*(1./12.)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAHHe = Coupling(name= 'R2_AAHHe',
                   value = '-(1/(16*sw**2))*(Ql**2*Me**2/MW**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAHHm = Coupling(name= 'R2_AAHHm',
                   value = '-(1/(16*sw**2))*(Ql**2*MM**2/MW**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAHHtau = Coupling(name= 'R2_AAHHtau',
                   value = '-(1/(16*sw**2))*(Ql**2*MTA**2/MW**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAHHu = Coupling(name= 'R2_AAHHu',
                   value = '-(1/(16*sw**2))*Ncol*(Qu**2*MU**2/MW**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAHHd = Coupling(name= 'R2_AAHHd',
                   value = '-(1/(16*sw**2))*Ncol*(Qd**2*MD**2/MW**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAHHs = Coupling(name= 'R2_AAHHs',
                   value = '-(1/(16*sw**2))*Ncol*(Qd**2*MS**2/MW**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAHHc = Coupling(name= 'R2_AAHHc',
                   value = '-(1/(16*sw**2))*Ncol*(Qu**2*MC**2/MW**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAHHb = Coupling(name= 'R2_AAHHb',
                   value = '-(1/(16*sw**2))*Ncol*(Qd**2*MB**2/MW**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AAHHt = Coupling(name= 'R2_AAHHt',
                   value = '-(1/(16*sw**2))*Ncol*(Qu**2*MT**2/MW**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AZHH = Coupling(name= 'R2_AZHH',
                   value = '(-1/(16*sw))*((4+sw**2)/(12.*sw**2*cw))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AZHHe = Coupling(name= 'R2_AZHHe',
                   value = '(-1/(16*sw))*((Ql*Me**2*(I3l/(2*sw**2)-Ql))/(MW**2*cw))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AZHHm = Coupling(name= 'R2_AZHHm',
                   value = '(-1/(16*sw))*((Ql*MM**2*(I3l/(2*sw**2)-Ql))/(MW**2*cw))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AZHHtau = Coupling(name= 'R2_AZHHtau',
                   value = '(-1/(16*sw))*((Ql*MTA**2*(I3l/(2*sw**2)-Ql))/(MW**2*cw))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AZHHu = Coupling(name= 'R2_AZHHu',
                   value = '(-1/(16*sw))*Ncol*((Qu*MU**2*(I3u/(2*sw**2)-Qu))/(MW**2*cw))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AZHHd = Coupling(name= 'R2_AZHHd',
                   value = '(-1/(16*sw))*Ncol*((Qd*MD**2*(I3d/(2*sw**2)-Qd))/(MW**2*cw))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AZHHs = Coupling(name= 'R2_AZHHs',
                   value = '(-1/(16*sw))*Ncol*((Qd*MS**2*(I3d/(2*sw**2)-Qd))/(MW**2*cw))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AZHHc = Coupling(name= 'R2_AZHHc',
                   value = '(-1/(16*sw))*Ncol*((Qu*MC**2*(I3u/(2*sw**2)-Qu))/(MW**2*cw))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AZHHb = Coupling(name= 'R2_AZHHb',
                   value = '(-1/(16*sw))*Ncol*((Qd*MB**2*(I3d/(2*sw**2)-Qd))/(MW**2*cw))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_AZHHt = Coupling(name= 'R2_AZHHt',
                   value = '(-1/(16*sw))*Ncol*((Qu*MT**2*(I3u/(2*sw**2)-Qu))/(MW**2*cw))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZHH = Coupling(name= 'R2_ZZHH',
                   value = '(-1/(16*cw**2))*((2+2*cw**2 +40*cw**4 -4*cw**6)/(48.*sw**4*cw**2))*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZHHe = Coupling(name= 'R2_ZZHHe',
                   value = '(-1/(16*cw**2))/MW**2*Me**2*(Ql**2 +4*I3l**2/(3*sw**4)-Ql*I3l/sw**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZHHm = Coupling(name= 'R2_ZZHHm',
                   value = '(-1/(16*cw**2))/MW**2*MM**2*(Ql**2 +4*I3l**2/(3*sw**4)-Ql*I3l/sw**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZHHtau = Coupling(name= 'R2_ZZHHtau',
                   value = '(-1/(16*cw**2))/MW**2*MTA**2*(Ql**2 +4*I3l**2/(3*sw**4)-Ql*I3l/sw**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZHHu = Coupling(name= 'R2_ZZHHu',
                   value = '(-1/(16*cw**2))/MW**2*MU**2*Ncol*(Qu**2 +4*I3u**2/(3*sw**4)-Qu*I3u/sw**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZHHc = Coupling(name= 'R2_ZZHHc',
                   value = '(-1/(16*cw**2))/MW**2*MC**2*Ncol*(Qu**2 +4*I3u**2/(3*sw**4)-Qu*I3u/sw**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZHHt = Coupling(name= 'R2_ZZHHt',
                   value = '(-1/(16*cw**2))/MW**2*MT**2*Ncol*(Qu**2 +4*I3u**2/(3*sw**4)-Qu*I3u/sw**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZHHd = Coupling(name= 'R2_ZZHHd',
                   value = '(-1/(16*cw**2))/MW**2*MD**2*Ncol*(Qd**2 +4*I3d**2/(3*sw**4)-Qd*I3d/sw**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZHHs = Coupling(name= 'R2_ZZHHs',
                   value = '(-1/(16*cw**2))/MW**2*MS**2*Ncol*(Qd**2 +4*I3d**2/(3*sw**4)-Qd*I3d/sw**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_ZZHHb = Coupling(name= 'R2_ZZHHb',
                   value = '(-1/(16*cw**2))/MW**2*MB**2*Ncol*(Qd**2 +4*I3d**2/(3*sw**4)-Qd*I3d/sw**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_WWHH = Coupling(name= 'R2_WWHH',
                   value = '(-1/(48*sw**4))*(1.+38*cw**2)/(16.*cw**2)*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_WWHHe = Coupling(name= 'R2_WWHHe',
                   value = '(-1/(48*sw**4))*Me**2/MW**2*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_WWHHm = Coupling(name= 'R2_WWHHm',
                   value = '(-1/(48*sw**4))*MM**2/MW**2*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_WWHHtau = Coupling(name= 'R2_WWHHtau',
                   value = '(-1/(48*sw**4))*MTA**2/MW**2*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_WWHHud = Coupling(name= 'R2_WWHHud',
                   value = '(-1/(48*sw**4))*Ncol*SCKM11*(MU**2 +MD**2)/MW**2*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_WWHHus = Coupling(name= 'R2_WWHHus',
                   value = '(-1/(48*sw**4))*Ncol*SCKM12*(MU**2 +MS**2)/MW**2*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_WWHHub = Coupling(name= 'R2_WWHHub',
                   value = '(-1/(48*sw**4))*Ncol*SCKM13*(MU**2 +MB**2)/MW**2*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_WWHHcd = Coupling(name= 'R2_WWHHcd',
                   value = '(-1/(48*sw**4))*Ncol*SCKM21*(MC**2 +MD**2)/MW**2*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_WWHHcs = Coupling(name= 'R2_WWHHcs',
                   value = '(-1/(48*sw**4))*Ncol*SCKM22*(MC**2 +MS**2)/MW**2*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_WWHHcb = Coupling(name= 'R2_WWHHcb',
                   value = '(-1/(48*sw**4))*Ncol*SCKM23*(MC**2 +MB**2)/MW**2*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_WWHHtd = Coupling(name= 'R2_WWHHtd',
                   value = '(-1/(48*sw**4))*Ncol*SCKM31*(MT**2 +MD**2)/MW**2*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_WWHHts = Coupling(name= 'R2_WWHHts',
                   value = '(-1/(48*sw**4))*Ncol*SCKM32*(MT**2 +MS**2)/MW**2*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_WWHHtb = Coupling(name= 'R2_WWHHtb',
                   value = '(-1/(48*sw**4))*Ncol*SCKM33*(MT**2 +MB**2)/MW**2*(complex(0,1)*R24S)',
                   order = {'QED':4})

R2_WAG0Gp = Coupling(name = 'R2_WAG0Gp',
                     value = '1./(24*sw**3)*(1+22*cw**2)/(32*cw**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WAG0Gpe = Coupling(name = 'R2_WAG0Gpe',
                     value = '1./(24*sw**3)*Me**2/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WAG0Gpm = Coupling(name = 'R2_WAG0Gpm',
                     value = '1./(24*sw**3)*MM**2/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WAG0Gptau = Coupling(name = 'R2_WAG0Gptau',
                     value = '1./(24*sw**3)*MTA**2/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WAG0Gpud = Coupling(name = 'R2_WAG0Gpud',
                     value = '1./(24*sw**3)*Ncol*SCKM11*(2*MU**2 +3*MD**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WAG0Gpus = Coupling(name = 'R2_WAG0Gpus',
                     value = '1./(24*sw**3)*Ncol*SCKM12*(2*MU**2 +3*MS**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WAG0Gpub = Coupling(name = 'R2_WAG0Gpub',
                     value = '1./(24*sw**3)*Ncol*SCKM13*(2*MU**2 +3*MB**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WAG0Gpcd = Coupling(name = 'R2_WAG0Gpcd',
                     value = '1./(24*sw**3)*Ncol*SCKM21*(2*MC**2 +3*MD**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WAG0Gpcs = Coupling(name = 'R2_WAG0Gpcs',
                     value = '1./(24*sw**3)*Ncol*SCKM22*(2*MC**2 +3*MS**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WAG0Gpcb = Coupling(name = 'R2_WAG0Gpcb',
                     value = '1./(24*sw**3)*Ncol*SCKM23*(2*MC**2 +3*MB**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WAG0Gptd = Coupling(name = 'R2_WAG0Gptd',
                     value = '1./(24*sw**3)*Ncol*SCKM31*(2*MT**2 +3*MD**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WAG0Gpts = Coupling(name = 'R2_WAG0Gpts',
                     value = '1./(24*sw**3)*Ncol*SCKM32*(2*MT**2 +3*MS**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WAG0Gptb = Coupling(name = 'R2_WAG0Gptb',
                     value = '1./(24*sw**3)*Ncol*SCKM33*(2*MT**2 +3*MB**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WAHGp = Coupling(name = 'R2_WAHGp',
                     value = '1./(24*sw**3)*(1+22*cw**2)/(32*cw**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGpe = Coupling(name = 'R2_WAHGpe',
                     value = '1./(24*sw**3)*Me**2/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGpm = Coupling(name = 'R2_WAHGpm',
                     value = '1./(24*sw**3)*MM**2/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGptau = Coupling(name = 'R2_WAHGptau',
                     value = '1./(24*sw**3)*MTA**2/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGpud = Coupling(name = 'R2_WAHGpud',
                     value = '1./(24*sw**3)*Ncol*SCKM11*(2*MU**2 +3*MD**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGpus = Coupling(name = 'R2_WAHGpus',
                     value = '1./(24*sw**3)*Ncol*SCKM12*(2*MU**2 +3*MS**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGpub = Coupling(name = 'R2_WAHGpub',
                     value = '1./(24*sw**3)*Ncol*SCKM13*(2*MU**2 +3*MB**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGpcd = Coupling(name = 'R2_WAHGpcd',
                     value = '1./(24*sw**3)*Ncol*SCKM21*(2*MC**2 +3*MD**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGpcs = Coupling(name = 'R2_WAHGpcs',
                     value = '1./(24*sw**3)*Ncol*SCKM22*(2*MC**2 +3*MS**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGpcb = Coupling(name = 'R2_WAHGpcb',
                     value = '1./(24*sw**3)*Ncol*SCKM23*(2*MC**2 +3*MB**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGptd = Coupling(name = 'R2_WAHGptd',
                     value = '1./(24*sw**3)*Ncol*SCKM31*(2*MT**2 +3*MD**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGpts = Coupling(name = 'R2_WAHGpts',
                     value = '1./(24*sw**3)*Ncol*SCKM32*(2*MT**2 +3*MS**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGptb = Coupling(name = 'R2_WAHGptb',
                     value = '1./(24*sw**3)*Ncol*SCKM33*(2*MT**2 +3*MB**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGm = Coupling(name = 'R2_WAHGm',
                     value = '-1./(24*sw**3)*(1+22*cw**2)/(32*cw**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGme = Coupling(name = 'R2_WAHGme',
                     value = '-1./(24*sw**3)*Me**2/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGmm = Coupling(name = 'R2_WAHGmm',
                     value = '-1./(24*sw**3)*MM**2/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGmtau = Coupling(name = 'R2_WAHGmtau',
                     value = '-1./(24*sw**3)*MTA**2/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGmud = Coupling(name = 'R2_WAHGmud',
                     value = '-1./(24*sw**3)*Ncol*SCKM11*(2*MU**2 +3*MD**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGmus = Coupling(name = 'R2_WAHGmus',
                     value = '-1./(24*sw**3)*Ncol*SCKM12*(2*MU**2 +3*MS**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGmub = Coupling(name = 'R2_WAHGmub',
                     value = '-1./(24*sw**3)*Ncol*SCKM13*(2*MU**2 +3*MB**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGmcd = Coupling(name = 'R2_WAHGmcd',
                     value = '-1./(24*sw**3)*Ncol*SCKM21*(2*MC**2 +3*MD**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGmcs = Coupling(name = 'R2_WAHGmcs',
                     value = '-1./(24*sw**3)*Ncol*SCKM22*(2*MC**2 +3*MS**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGmcb = Coupling(name = 'R2_WAHGmcb',
                     value = '-1./(24*sw**3)*Ncol*SCKM23*(2*MC**2 +3*MB**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGmtd = Coupling(name = 'R2_WAHGmtd',
                     value = '-1./(24*sw**3)*Ncol*SCKM31*(2*MT**2 +3*MD**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGmts = Coupling(name = 'R2_WAHGmts',
                     value = '-1./(24*sw**3)*Ncol*SCKM32*(2*MT**2 +3*MS**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WAHGmtb = Coupling(name = 'R2_WAHGmtb',
                     value = '-1./(24*sw**3)*Ncol*SCKM33*(2*MT**2 +3*MB**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZG0Gp = Coupling(name = 'R2_WZG0Gp',
                     value = '-1./(24*sw**2*cw)*(1+22*cw**2)/(32*cw**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WZG0Gpe = Coupling(name = 'R2_WZG0Gpe',
                     value = '-1./(24*sw**2*cw)*Me**2/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WZG0Gpm = Coupling(name = 'R2_WZG0Gpm',
                     value = '-1./(24*sw**2*cw)*MM**2/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WZG0Gptau = Coupling(name = 'R2_WZG0Gptau',
                     value = '-1./(24*sw**2*cw)*MTA**2/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WZG0Gpud = Coupling(name = 'R2_WZG0Gpud',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM11*(2*MU**2 +3*MD**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WZG0Gpus = Coupling(name = 'R2_WZG0Gpus',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM12*(2*MU**2 +3*MS**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WZG0Gpub = Coupling(name = 'R2_WZG0Gpub',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM13*(2*MU**2 +3*MB**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WZG0Gpcd = Coupling(name = 'R2_WZG0Gpcd',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM21*(2*MC**2 +3*MD**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WZG0Gpcs = Coupling(name = 'R2_WZG0Gpcs',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM22*(2*MC**2 +3*MS**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WZG0Gpcb = Coupling(name = 'R2_WZG0Gpcb',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM23*(2*MC**2 +3*MB**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WZG0Gptd = Coupling(name = 'R2_WZG0Gptd',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM31*(2*MT**2 +3*MD**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WZG0Gpts = Coupling(name = 'R2_WZG0Gpts',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM32*(2*MT**2 +3*MS**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WZG0Gptb = Coupling(name = 'R2_WZG0Gptb',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM33*(2*MT**2 +3*MB**2)/(8*MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WZHGp = Coupling(name = 'R2_WZHGp',
                     value = '-1./(24*sw**2*cw)*(1+22*cw**2)/(32*cw**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGpe = Coupling(name = 'R2_WZHGpe',
                     value = '-1./(24*sw**2*cw)*Me**2/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGpm = Coupling(name = 'R2_WZHGpm',
                     value = '-1./(24*sw**2*cw)*MM**2/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGptau = Coupling(name = 'R2_WZHGptau',
                     value = '-1./(24*sw**2*cw)*MTA**2/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGpud = Coupling(name = 'R2_WZHGpud',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM11*(2*MU**2 +3*MD**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGpus = Coupling(name = 'R2_WZHGpus',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM12*(2*MU**2 +3*MS**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGpub = Coupling(name = 'R2_WZHGpub',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM13*(2*MU**2 +3*MB**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGpcd = Coupling(name = 'R2_WZHGpcd',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM21*(2*MC**2 +3*MD**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGpcs = Coupling(name = 'R2_WZHGpcs',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM22*(2*MC**2 +3*MS**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGpcb = Coupling(name = 'R2_WZHGpcb',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM23*(2*MC**2 +3*MB**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGptd = Coupling(name = 'R2_WZHGptd',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM31*(2*MT**2 +3*MD**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGpts = Coupling(name = 'R2_WZHGpts',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM32*(2*MT**2 +3*MS**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGptb = Coupling(name = 'R2_WZHGptb',
                     value = '-1./(24*sw**2*cw)*Ncol*SCKM33*(2*MT**2 +3*MB**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGm = Coupling(name = 'R2_WZHGm',
                     value = '1./(24*sw**2*cw)*(1+22*cw**2)/(32*cw**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGme = Coupling(name = 'R2_WZHGme',
                     value = '1./(24*sw**2*cw)*Me**2/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGmm = Coupling(name = 'R2_WZHGmm',
                     value = '1./(24*sw**2*cw)*MM**2/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGmtau = Coupling(name = 'R2_WZHGmtau',
                     value = '1./(24*sw**2*cw)*MTA**2/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGmud = Coupling(name = 'R2_WZHGmud',
                     value = '1./(24*sw**2*cw)*Ncol*SCKM11*(2*MU**2 +3*MD**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGmus = Coupling(name = 'R2_WZHGmus',
                     value = '1./(24*sw**2*cw)*Ncol*SCKM12*(2*MU**2 +3*MS**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGmub = Coupling(name = 'R2_WZHGmub',
                     value = '1./(24*sw**2*cw)*Ncol*SCKM13*(2*MU**2 +3*MB**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGmcd = Coupling(name = 'R2_WZHGmcd',
                     value = '1./(24*sw**2*cw)*Ncol*SCKM21*(2*MC**2 +3*MD**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGmcs = Coupling(name = 'R2_WZHGmcs',
                     value = '1./(24*sw**2*cw)*Ncol*SCKM22*(2*MC**2 +3*MS**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGmcb = Coupling(name = 'R2_WZHGmcb',
                     value = '1./(24*sw**2*cw)*Ncol*SCKM23*(2*MC**2 +3*MB**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGmtd = Coupling(name = 'R2_WZHGmtd',
                     value = '1./(24*sw**2*cw)*Ncol*SCKM31*(2*MT**2 +3*MD**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGmts = Coupling(name = 'R2_WZHGmts',
                     value = '1./(24*sw**2*cw)*Ncol*SCKM32*(2*MT**2 +3*MS**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_WZHGmtb = Coupling(name = 'R2_WZHGmtb',
                     value = '1./(24*sw**2*cw)*Ncol*SCKM33*(2*MT**2 +3*MB**2)/(8*MW**2)*(R24S)',
                     order = {'QED':4})

R2_AAGmGp = Coupling(name = 'R2_AAGmGp',
                     value = '-1/(12*sw**2)*(1+21*cw**2)/(16*cw**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AAGmGpe = Coupling(name = 'R2_AAGmGpe',
                     value = '-1/(12*sw**2)*Me**2/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AAGmGpm = Coupling(name = 'R2_AAGmGpm',
                     value = '-1/(12*sw**2)*MM**2/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AAGmGptau = Coupling(name = 'R2_AAGmGptau',
                     value = '-1/(12*sw**2)*MTA**2/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AAGmGpud = Coupling(name = 'R2_AAGmGpud',
                     value = '-1/(12*sw**2)*5.*Ncol/6.*SCKM11*(MU**2 +MD**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AAGmGpus = Coupling(name = 'R2_AAGmGpus',
                     value = '-1/(12*sw**2)*5.*Ncol/6.*SCKM12*(MU**2 +MS**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AAGmGpub = Coupling(name = 'R2_AAGmGpub',
                     value = '-1/(12*sw**2)*5.*Ncol/6.*SCKM13*(MU**2 +MB**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AAGmGpcd = Coupling(name = 'R2_AAGmGpcd',
                     value = '-1/(12*sw**2)*5.*Ncol/6.*SCKM21*(MC**2 +MD**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AAGmGpcs = Coupling(name = 'R2_AAGmGpcs',
                     value = '-1/(12*sw**2)*5.*Ncol/6.*SCKM22*(MC**2 +MS**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AAGmGpcb = Coupling(name = 'R2_AAGmGpcb',
                     value = '-1/(12*sw**2)*5.*Ncol/6.*SCKM23*(MC**2 +MB**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AAGmGptd = Coupling(name = 'R2_AAGmGptd',
                     value = '-1/(12*sw**2)*5.*Ncol/6.*SCKM31*(MT**2 +MD**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AAGmGpts = Coupling(name = 'R2_AAGmGpts',
                     value = '-1/(12*sw**2)*5.*Ncol/6.*SCKM32*(MT**2 +MS**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AAGmGptb = Coupling(name = 'R2_AAGmGptb',
                     value = '-1/(12*sw**2)*5.*Ncol/6.*SCKM33*(MT**2 +MB**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AZGmGp = Coupling(name = 'R2_AZGmGp',
                     value = '-1./(12*sw*cw)*(42*cw**4 -10*cw**2 -1)/(32*sw**2*cw**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AZGmGpe = Coupling(name = 'R2_AZGmGpe',
                     value = '-1./(12*sw*cw)*(-Me**2/MW**2)*Ql*(Ql+5.*I3v/(8*sw**2))*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AZGmGpm = Coupling(name = 'R2_AZGmGpm',
                     value = '-1./(12*sw*cw)*(-MM**2/MW**2)*Ql*(Ql+5.*I3v/(8*sw**2))*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AZGmGptau = Coupling(name = 'R2_AZGmGptau',
                     value = '-1./(12*sw*cw)*(-MTA**2/MW**2)*Ql*(Ql+5.*I3v/(8*sw**2))*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AZGmGpud = Coupling(name = 'R2_AZGmGpud',
                     value = '1./(12*sw*cw)*Ncol*SCKM11*(MU**2*(5./6.-I3d/sw**2*(Qd-5./8.*Qu))+MD**2*(5./6.-I3u/sw**2*(Qu-5./8.*Qd)))/MW**2*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AZGmGpus = Coupling(name = 'R2_AZGmGpus',
                     value = '1./(12*sw*cw)*Ncol*SCKM12*(MU**2*(5./6.-I3d/sw**2*(Qd-5./8.*Qu))+MS**2*(5./6.-I3u/sw**2*(Qu-5./8.*Qd)))/MW**2*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AZGmGpub = Coupling(name = 'R2_AZGmGpub',
                     value = '1./(12*sw*cw)*Ncol*SCKM13*(MU**2*(5./6.-I3d/sw**2*(Qd-5./8.*Qu))+MB**2*(5./6.-I3u/sw**2*(Qu-5./8.*Qd)))/MW**2*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AZGmGpcd = Coupling(name = 'R2_AZGmGpcd',
                     value = '1./(12*sw*cw)*Ncol*SCKM21*(MC**2*(5./6.-I3d/sw**2*(Qd-5./8.*Qu))+MD**2*(5./6.-I3u/sw**2*(Qu-5./8.*Qd)))/MW**2*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AZGmGpcs = Coupling(name = 'R2_AZGmGpcs',
                     value = '1./(12*sw*cw)*Ncol*SCKM22*(MC**2*(5./6.-I3d/sw**2*(Qd-5./8.*Qu))+MS**2*(5./6.-I3u/sw**2*(Qu-5./8.*Qd)))/MW**2*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AZGmGpcb = Coupling(name = 'R2_AZGmGpcb',
                     value = '1./(12*sw*cw)*Ncol*SCKM23*(MC**2*(5./6.-I3d/sw**2*(Qd-5./8.*Qu))+MB**2*(5./6.-I3u/sw**2*(Qu-5./8.*Qd)))/MW**2*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AZGmGptd = Coupling(name = 'R2_AZGmGptd',
                     value = '1./(12*sw*cw)*Ncol*SCKM31*(MT**2*(5./6.-I3d/sw**2*(Qd-5./8.*Qu))+MD**2*(5./6.-I3u/sw**2*(Qu-5./8.*Qd)))/MW**2*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AZGmGpts = Coupling(name = 'R2_AZGmGpts',
                     value = '1./(12*sw*cw)*Ncol*SCKM32*(MT**2*(5./6.-I3d/sw**2*(Qd-5./8.*Qu))+MS**2*(5./6.-I3u/sw**2*(Qu-5./8.*Qd)))/MW**2*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_AZGmGptb = Coupling(name = 'R2_AZGmGptb',
                     value = '1./(12*sw*cw)*Ncol*SCKM33*(MT**2*(5./6.-I3d/sw**2*(Qd-5./8.*Qu))+MB**2*(5./6.-I3u/sw**2*(Qu-5./8.*Qd)))/MW**2*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_ZZGmGp = Coupling(name = 'R2_ZZGmGp',
                     value = '1./(12*cw**2)*(-1+2*cw**2 +44*cw**4 -84*cw**6)/(64*sw**4*cw**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_ZZGmGpe = Coupling(name = 'R2_ZZGmGpe',
                     value = '-1./(12*cw**2)*Me**2/(MW**2)*(Ql**2 +5./4.*Ql*I3v/sw**2 +I3v**2/sw**4)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_ZZGmGpm = Coupling(name = 'R2_ZZGmGpm',
                     value = '-1./(12*cw**2)*MM**2/(MW**2)*(Ql**2 +5./4.*Ql*I3v/sw**2 +I3v**2/sw**4)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_ZZGmGptau = Coupling(name = 'R2_ZZGmGptau',
                     value = '-1./(12*cw**2)*MTA**2/(MW**2)*(Ql**2 +5./4.*Ql*I3v/sw**2 +I3v**2/sw**4)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_ZZGmGpud = Coupling(name = 'R2_ZZGmGpud',
                     value = '-1./(12*cw**2)*Ncol*SCKM11*(MU**2*(5./6.-I3d/sw**2*(2*Qd-5./4.*Qu)+I3d**2/sw**4)+MD**2*(5./6.-I3u/sw**2*(2*Qu-5./4.*Qd)+I3u**2/sw**4))/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_ZZGmGpus = Coupling(name = 'R2_ZZGmGpus',
                     value = '-1./(12*cw**2)*Ncol*SCKM12*(MU**2*(5./6.-I3d/sw**2*(2*Qd-5./4.*Qu)+I3d**2/sw**4)+MS**2*(5./6.-I3u/sw**2*(2*Qu-5./4.*Qd)+I3u**2/sw**4))/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_ZZGmGpub = Coupling(name = 'R2_ZZGmGpub',
                     value = '-1./(12*cw**2)*Ncol*SCKM13*(MU**2*(5./6.-I3d/sw**2*(2*Qd-5./4.*Qu)+I3d**2/sw**4)+MB**2*(5./6.-I3u/sw**2*(2*Qu-5./4.*Qd)+I3u**2/sw**4))/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_ZZGmGpcd = Coupling(name = 'R2_ZZGmGpcd',
                     value = '-1./(12*cw**2)*Ncol*SCKM21*(MC**2*(5./6.-I3d/sw**2*(2*Qd-5./4.*Qu)+I3d**2/sw**4)+MD**2*(5./6.-I3u/sw**2*(2*Qu-5./4.*Qd)+I3u**2/sw**4))/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_ZZGmGpcs = Coupling(name = 'R2_ZZGmGpcs',
                     value = '-1./(12*cw**2)*Ncol*SCKM22*(MC**2*(5./6.-I3d/sw**2*(2*Qd-5./4.*Qu)+I3d**2/sw**4)+MS**2*(5./6.-I3u/sw**2*(2*Qu-5./4.*Qd)+I3u**2/sw**4))/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_ZZGmGpcb = Coupling(name = 'R2_ZZGmGpcb',
                     value = '-1./(12*cw**2)*Ncol*SCKM23*(MC**2*(5./6.-I3d/sw**2*(2*Qd-5./4.*Qu)+I3d**2/sw**4)+MB**2*(5./6.-I3u/sw**2*(2*Qu-5./4.*Qd)+I3u**2/sw**4))/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_ZZGmGptd = Coupling(name = 'R2_ZZGmGptd',
                     value = '-1./(12*cw**2)*Ncol*SCKM31*(MT**2*(5./6.-I3d/sw**2*(2*Qd-5./4.*Qu)+I3d**2/sw**4)+MD**2*(5./6.-I3u/sw**2*(2*Qu-5./4.*Qd)+I3u**2/sw**4))/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_ZZGmGpts = Coupling(name = 'R2_ZZGmGpts',
                     value = '-1./(12*cw**2)*Ncol*SCKM32*(MT**2*(5./6.-I3d/sw**2*(2*Qd-5./4.*Qu)+I3d**2/sw**4)+MS**2*(5./6.-I3u/sw**2*(2*Qu-5./4.*Qd)+I3u**2/sw**4))/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_ZZGmGptb = Coupling(name = 'R2_ZZGmGptb',
                     value = '-1./(12*cw**2)*Ncol*SCKM33*(MT**2*(5./6.-I3d/sw**2*(2*Qd-5./4.*Qu)+I3d**2/sw**4)+MB**2*(5./6.-I3u/sw**2*(2*Qu-5./4.*Qd)+I3u**2/sw**4))/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGp = Coupling(name = 'R2_WWGmGp',
                     value = '-1/(48*sw**4)*(38*cw**2 +1)/(16*cw**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpe = Coupling(name = 'R2_WWGmGpe',
                     value = '-1/(48*sw**4)*Me**2/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpm = Coupling(name = 'R2_WWGmGpm',
                     value = '-1/(48*sw**4)*MM**2/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGptau = Coupling(name = 'R2_WWGmGptau',
                     value = '-1/(48*sw**4)*MTA**2/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpud = Coupling(name = 'R2_WWGmGpud',
                     value = '-1/(48*sw**4)*Ncol*SCKM11**2*(MU**2 +MD**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpus = Coupling(name = 'R2_WWGmGpus',
                     value = '-1/(48*sw**4)*Ncol*SCKM12**2*(MU**2 +MS**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpub = Coupling(name = 'R2_WWGmGpub',
                     value = '-1/(48*sw**4)*Ncol*SCKM13**2*(MU**2 +MB**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpcd = Coupling(name = 'R2_WWGmGpcd',
                     value = '-1/(48*sw**4)*Ncol*SCKM21**2*(MC**2 +MD**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpcs = Coupling(name = 'R2_WWGmGpcs',
                     value = '-1/(48*sw**4)*Ncol*SCKM22**2*(MC**2 +MS**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpcb = Coupling(name = 'R2_WWGmGpcb',
                     value = '-1/(48*sw**4)*Ncol*SCKM23**2*(MC**2 +MB**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGptd = Coupling(name = 'R2_WWGmGptd',
                     value = '-1/(48*sw**4)*Ncol*SCKM31**2*(MT**2 +MD**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpts = Coupling(name = 'R2_WWGmGpts',
                     value = '-1/(48*sw**4)*Ncol*SCKM32**2*(MT**2 +MS**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGptb = Coupling(name = 'R2_WWGmGptb',
                     value = '-1/(48*sw**4)*Ncol*SCKM33**2*(MT**2 +MB**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpucd = Coupling(name = 'R2_WWGmGpucd',
                     value = '-1/(48*sw**4)*Ncol*SCKM11*SCKM21*(MU**2 +MC**2 +2*MD**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpucs = Coupling(name = 'R2_WWGmGpucs',
                     value = '-1/(48*sw**4)*Ncol*SCKM12*SCKM22*(MU**2 +MC**2 +2*MS**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpucb = Coupling(name = 'R2_WWGmGpucb',
                     value = '-1/(48*sw**4)*Ncol*SCKM13*SCKM23*(MU**2 +MC**2 +2*MB**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGputd = Coupling(name = 'R2_WWGmGputd',
                     value = '-1/(48*sw**4)*Ncol*SCKM11*SCKM31*(MU**2 +MT**2 +2*MD**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGputs = Coupling(name = 'R2_WWGmGputs',
                     value = '-1/(48*sw**4)*Ncol*SCKM12*SCKM32*(MU**2 +MT**2 +2*MS**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGputb = Coupling(name = 'R2_WWGmGputb',
                     value = '-1/(48*sw**4)*Ncol*SCKM13*SCKM33*(MU**2 +MT**2 +2*MB**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpctd = Coupling(name = 'R2_WWGmGpctd',
                     value = '-1/(48*sw**4)*Ncol*SCKM21*SCKM31*(MC**2 +MT**2 +2*MD**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpcts = Coupling(name = 'R2_WWGmGpcts',
                     value = '-1/(48*sw**4)*Ncol*SCKM22*SCKM32*(MC**2 +MT**2 +2*MS**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpctb = Coupling(name = 'R2_WWGmGpctb',
                     value = '-1/(48*sw**4)*Ncol*SCKM23*SCKM33*(MC**2 +MT**2 +2*MB**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpuds = Coupling(name = 'R2_WWGmGpuds',
                     value = '-1/(48*sw**4)*Ncol*SCKM11*SCKM12*(MD**2 +MS**2 +2*MU**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpcds = Coupling(name = 'R2_WWGmGpcds',
                     value = '-1/(48*sw**4)*Ncol*SCKM21*SCKM22*(MD**2 +MS**2 +2*MC**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGptds = Coupling(name = 'R2_WWGmGptds',
                     value = '-1/(48*sw**4)*Ncol*SCKM31*SCKM32*(MD**2 +MS**2 +2*MT**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpudb = Coupling(name = 'R2_WWGmGpudb',
                     value = '-1/(48*sw**4)*Ncol*SCKM11*SCKM13*(MD**2 +MB**2 +2*MU**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpcdb = Coupling(name = 'R2_WWGmGpcdb',
                     value = '-1/(48*sw**4)*Ncol*SCKM21*SCKM23*(MD**2 +MB**2 +2*MC**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGptdb = Coupling(name = 'R2_WWGmGptdb',
                     value = '-1/(48*sw**4)*Ncol*SCKM31*SCKM33*(MD**2 +MB**2 +2*MT**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpusb = Coupling(name = 'R2_WWGmGpusb',
                     value = '-1/(48*sw**4)*Ncol*SCKM12*SCKM13*(MS**2 +MB**2 +2*MU**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpcsb = Coupling(name = 'R2_WWGmGpcsb',
                     value = '-1/(48*sw**4)*Ncol*SCKM22*SCKM23*(MS**2 +MB**2 +2*MC**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGptsb = Coupling(name = 'R2_WWGmGptsb',
                     value = '-1/(48*sw**4)*Ncol*SCKM32*SCKM33*(MS**2 +MB**2 +2*MT**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpucds = Coupling(name = 'R2_WWGmGpucds',
                     value = '-1/(48*sw**4)*Ncol*CKMucds*(MS**2 +MD**2 +MU**2 +MC**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpucdb = Coupling(name = 'R2_WWGmGpucdb',
                     value = '-1/(48*sw**4)*Ncol*CKMucdb*(MB**2 +MD**2 +MU**2 +MC**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpucsb = Coupling(name = 'R2_WWGmGpucsb',
                     value = '-1/(48*sw**4)*Ncol*CKMucsb*(MS**2 +MB**2 +MU**2 +MC**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGputds = Coupling(name = 'R2_WWGmGputds',
                     value = '-1/(48*sw**4)*Ncol*CKMutds*(MS**2 +MD**2 +MU**2 +MT**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGputdb = Coupling(name = 'R2_WWGmGputdb',
                     value = '-1/(48*sw**4)*Ncol*CKMutdb*(MB**2 +MD**2 +MU**2 +MT**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGputsb = Coupling(name = 'R2_WWGmGputsb',
                     value = '-1/(48*sw**4)*Ncol*CKMutsb*(MS**2 +MB**2 +MU**2 +MT**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpctds = Coupling(name = 'R2_WWGmGpctds',
                     value = '-1/(48*sw**4)*Ncol*CKMctds*(MS**2 +MD**2 +MC**2 +MT**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpctdb = Coupling(name = 'R2_WWGmGpctdb',
                     value = '-1/(48*sw**4)*Ncol*CKMctdb*(MB**2 +MD**2 +MC**2 +MT**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

R2_WWGmGpctsb = Coupling(name = 'R2_WWGmGpctsb',
                     value = '-1/(48*sw**4)*Ncol*CKMctsb*(MS**2 +MB**2 +MC**2 +MT**2)/(MW**2)*(complex(0,1)*R24S)',
                     order = {'QED':4})

# ============== #
# Mixed QED-QCD  #
# ============== #

R2_GUU2Cp = Coupling(name = 'R2_GUU2Cp',
                     value = '1./16.*((1+lhv)/cw**2*(Qu**2 +I3u**2/sw**2 -2*Qu*I3u)+MU**2/(2*MW**2*sw**2)*(1./4.+I3u**2))*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GUU2Cm = Coupling(name = 'R2_GUU2Cm',
                     value = '1./16.*((1+lhv)/cw**2*Qu**2 +MU**2/(2*MW**2*sw**2)*(1./4.+I3u**2))*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GCC2Cp = Coupling(name = 'R2_GCC2Cp',
                     value = '1./16.*((1+lhv)/cw**2*(Qu**2 +I3u**2/sw**2 -2*Qu*I3u)+MC**2/(2*MW**2*sw**2)*(1./4.+I3u**2))*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GCC2Cm = Coupling(name = 'R2_GCC2Cm',
                     value = '1./16.*((1+lhv)/cw**2*Qu**2 +MC**2/(2*MW**2*sw**2)*(1./4.+I3u**2))*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GTT2Cp = Coupling(name = 'R2_GTT2Cp',
                     value = '1./16.*((1+lhv)/cw**2*(Qu**2 +I3u**2/sw**2 -2*Qu*I3u)+MT**2/(2*MW**2*sw**2)*(1./4.+I3u**2))*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GTT2Cm = Coupling(name = 'R2_GTT2Cm',
                     value = '1./16.*((1+lhv)/cw**2*Qu**2 +MT**2/(2*MW**2*sw**2)*(1./4.+I3u**2))*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GDD2Cp = Coupling(name = 'R2_GDD2Cp',
                     value = '1./16.*((1+lhv)/cw**2*(Qd**2 +I3d**2/sw**2 -2*Qd*I3d)+MD**2/(2*MW**2*sw**2)*(1./4.+I3d**2))*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GDD2Cm = Coupling(name = 'R2_GDD2Cm',
                     value = '1./16.*((1+lhv)/cw**2*Qd**2 +MD**2/(2*MW**2*sw**2)*(1./4.+I3d**2))*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GSS2Cp = Coupling(name = 'R2_GSS2Cp',
                     value = '1./16.*((1+lhv)/cw**2*(Qd**2 +I3d**2/sw**2 -2*Qd*I3d)+MS**2/(2*MW**2*sw**2)*(1./4.+I3d**2))*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GSS2Cm = Coupling(name = 'R2_GSS2Cm',
                     value = '1./16.*((1+lhv)/cw**2*Qd**2 +MS**2/(2*MW**2*sw**2)*(1./4.+I3d**2))*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GBB2Cp = Coupling(name = 'R2_GBB2Cp',
                     value = '1./16.*((1+lhv)/cw**2*(Qd**2 +I3d**2/sw**2 -2*Qd*I3d)+MB**2/(2*MW**2*sw**2)*(1./4.+I3d**2))*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GBB2Cm = Coupling(name = 'R2_GBB2Cm',
                     value = '1./16.*((1+lhv)/cw**2*Qd**2 +MB**2/(2*MW**2*sw**2)*(1./4.+I3d**2))*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GUU2Cpd = Coupling(name = 'R2_GUU2Cpd',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM11+1.0/(4.0*MW**2*sw**2)*SCKM11*MD**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GUU2Cmd = Coupling(name = 'R2_GUU2Cmd',
                     value = '1./16.*(MU**2/(4*sw**2*MW**2)*SCKM11)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GUU2Cps = Coupling(name = 'R2_GUU2Cps',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM12+1.0/(4.0*MW**2*sw**2)*SCKM12*MS**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GUU2Cms = Coupling(name = 'R2_GUU2Cms',
                     value = '1./16.*(MU**2/(4*sw**2*MW**2)*SCKM12)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GUU2Cpb = Coupling(name = 'R2_GUU2Cpb',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM13+1.0/(4.0*MW**2*sw**2)*SCKM13*MB**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GUU2Cmb = Coupling(name = 'R2_GUU2Cmb',
                     value = '1./16.*(MU**2/(4*sw**2*MW**2)*SCKM13)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GCC2Cpd = Coupling(name = 'R2_GCC2Cpd',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM21+1.0/(4.0*MW**2*sw**2)*SCKM21*MD**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GCC2Cmd = Coupling(name = 'R2_GCC2Cmd',
                     value = '1./16.*(MC**2/(4*sw**2*MW**2)*SCKM21)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GCC2Cps = Coupling(name = 'R2_GCC2Cps',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM22+1/(4*MW**2*sw**2)*SCKM22*MS**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GCC2Cms = Coupling(name = 'R2_GCC2Cms',
                     value = '1./16.*(MC**2/(4*sw**2*MW**2)*SCKM22)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GCC2Cpb = Coupling(name = 'R2_GCC2Cpb',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM23+1/(4*MW**2*sw**2)*SCKM23*MB**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GCC2Cmb = Coupling(name = 'R2_GCC2Cmb',
                     value = '1./16.*(MC**2/(4*sw**2*MW**2)*SCKM23)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GTT2Cpd = Coupling(name = 'R2_GTT2Cpd',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM31+1.0/(4.0*MW**2*sw**2)*SCKM31*MD**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GTT2Cmd = Coupling(name = 'R2_GTT2Cmd',
                     value = '1./16.*(MT**2/(4*sw**2*MW**2)*SCKM31)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GTT2Cps = Coupling(name = 'R2_GTT2Cps',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM32+1.0/(4.0*MW**2*sw**2)*SCKM32*MS**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GTT2Cms = Coupling(name = 'R2_GCC2Cms',
                     value = '1./16.*(MT**2/(4*sw**2*MW**2)*SCKM32)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GTT2Cpb = Coupling(name = 'R2_GTT2Cpb',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM33+1.0/(4.0*MW**2*sw**2)*SCKM33*MB**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GTT2Cmb = Coupling(name = 'R2_GTT2Cmb',
                     value = '1./16.*(MT**2/(4*sw**2*MW**2)*SCKM33)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GDD2Cpu = Coupling(name = 'R2_GDD2Cpu',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM11+1/(4*MW**2*sw**2)*SCKM11*MU**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GDD2Cmu = Coupling(name = 'R2_GDD2Cmu',
                     value = '1./16.*(MD**2/(4*sw**2*MW**2)*SCKM11)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GDD2Cpc = Coupling(name = 'R2_GDD2Cpc',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM21+1/(4*MW**2*sw**2)*SCKM21*MC**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GDD2Cmc = Coupling(name = 'R2_GDD2Cmc',
                     value = '1./16.*(MD**2/(4*sw**2*MW**2)*SCKM21)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GDD2Cpt = Coupling(name = 'R2_GDD2Cpt',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM31+1/(4*MW**2*sw**2)*SCKM31*MT**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GDD2Cmt = Coupling(name = 'R2_GDD2Cmt',
                     value = '1./16.*(MD**2/(4*sw**2*MW**2)*SCKM31)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GSS2Cpu = Coupling(name = 'R2_GSS2Cpu',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM12+1/(4*MW**2*sw**2)*SCKM12*MU**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GSS2Cmu = Coupling(name = 'R2_GSS2Cmu',
                     value = '1./16.*(MS**2/(4*sw**2*MW**2)*SCKM12)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GSS2Cpc = Coupling(name = 'R2_GSS2Cpc',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM22+1/(4*MW**2*sw**2)*SCKM22*MC**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GSS2Cmc = Coupling(name = 'R2_GSS2Cmc',
                     value = '1./16.*(MS**2/(4*sw**2*MW**2)*SCKM22)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GSS2Cpt = Coupling(name = 'R2_GSS2Cpt',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM32+1/(4*MW**2*sw**2)*SCKM32*MT**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GSS2Cmt = Coupling(name = 'R2_GSS2Cmt',
                     value = '1./16.*(MS**2/(4*sw**2*MW**2)*SCKM32)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GBB2Cpu = Coupling(name = 'R2_GBB2Cpu',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM13+1/(4*MW**2*sw**2)*SCKM13*MU**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GBB2Cmu = Coupling(name = 'R2_GBB2Cu',
                     value = '1./16.*(MB**2/(4*sw**2*MW**2)*SCKM13)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GBB2Cpc = Coupling(name = 'R2_GBB2Cpc',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM23+1/(4*MW**2*sw**2)*SCKM23*MC**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GBB2Cmc = Coupling(name = 'R2_GBB2Cmc',
                     value = '1./16.*(MB**2/(4*sw**2*MW**2)*SCKM23)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GBB2Cpt = Coupling(name = 'R2_GBB2Cpt',
                     value = '1./16.*((1+lhv)/(2*sw**2)*SCKM33+1/(4*MW**2*sw**2)*SCKM33*MT**2)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

R2_GBB2Cmt = Coupling(name = 'R2_GBB2Cmt',
                     value = '1./16.*(MB**2/(4*sw**2*MW**2)*SCKM33)*(complex(0,1)*R2GQQ2)',
                     order = {'QED':2,'QCD':1})

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
# ================================== #
# QED                                #
# Generate automatically by WriteUFO #
# ================================== #
c_UVWmWpMass1EW = Coupling(name = 'c_UVWmWpMass1EW',
                           value = 'WWcft_UV_EW*complex(0,1)',
                           order = {'QED' : 2})


c_UVWmWpMass2EW = Coupling(name = 'c_UVWmWpMass2EW',
                           value = '(WMass2_UV_EW + WWcft_UV_EW*MW**2)*complex(0,1)',
                           order = {'QED' : 2})


c_UVWmWpMass3EW = Coupling(name = 'c_UVWmWpMass3EW',
                           value = 'WWcft_UV_EW*complex(0,-1)',
                           order = {'QED' : 2})


c_UVZMass1EW = Coupling(name = 'c_UVZMass1EW',
                        value = 'ZZWcft_UV_EW*complex(0,1)',
                        order = {'QED' : 2})


c_UVZMass2EW = Coupling(name = 'c_UVZMass2EW',
                        value = '(ZMass2_UV_EW + ZZWcft_UV_EW*MZ**2)*complex(0,1)',
                        order = {'QED' : 2})


c_UVZMass3EW = Coupling(name = 'c_UVZMass3EW',
                        value = 'ZZWcft_UV_EW*complex(0,-1)',
                        order = {'QED' : 2})


c_UVAMass1EW = Coupling(name = 'c_UVAMass1EW',
                        value = 'AAWcft_UV_EW*complex(0,1)',
                        order = {'QED' : 2})


c_UVAMass2EW = Coupling(name = 'c_UVAMass2EW',
                        value = 'AAWcft_UV_EW*complex(0,-1)',
                        order = {'QED' : 2})


c_UVAZMass1EW = Coupling(name = 'c_UVAZMass1EW',
                         value = '(AZWcft_UV_EW + ZAWcft_UV_EW)*complex(0,-0.5)',
                         order = {'QED' : 2})


c_UVAZMass2EW = Coupling(name = 'c_UVAZMass2EW',
                         value = 'ZAWcft_UV_EW*MZ**2*complex(0,-0.5)',
                         order = {'QED' : 2})


c_UVAZMass3EW = Coupling(name = 'c_UVAZMass3EW',
                         value = '(AZWcft_UV_EW + ZAWcft_UV_EW)*complex(0,0.5)',
                         order = {'QED' : 2})


c_UVGpWmMass1EW = Coupling(name = 'c_UVGpWmMass1EW',
                           value = '-(WMass2_UV_EW + (GpWcft_UV_EW + WWcft_UV_EW)*MW**2)/(4.*MW)',
                           order = {'QED' : 2})


c_UVGpWmMass2EW = Coupling(name = 'c_UVGpWmMass2EW',
                           value = '((GpWcft_UV_EW + WWcft_UV_EW + WMass2_UV_EW/MW**2)*MW)/4.',
                           order = {'QED' : 2})


c_UVGmWpMass1EW = Coupling(name = 'c_UVGmWpMass1EW',
                           value = '-((GpWcft_UV_EW + WWcft_UV_EW + WMass2_UV_EW/MW**2)*MW)/4.',
                           order = {'QED' : 2})


c_UVGmWpMass2EW = Coupling(name = 'c_UVGmWpMass2EW',
                           value = '(WMass2_UV_EW + (GpWcft_UV_EW + WWcft_UV_EW)*MW**2)/(4.*MW)',
                           order = {'QED' : 2})


c_UVG0ZMass1EW = Coupling(name = 'c_UVG0ZMass1EW',
                          value = '-((G0Wcft_UV_EW + ZZWcft_UV_EW + ZMass2_UV_EW/MZ**2)*MZ)/4.',
                          order = {'QED' : 2})


c_UVG0ZMass2EW = Coupling(name = 'c_UVG0ZMass2EW',
                          value = '(ZMass2_UV_EW + (G0Wcft_UV_EW + ZZWcft_UV_EW)*MZ**2)/(4.*MZ)',
                          order = {'QED' : 2})


c_UVAG0Mass1EW = Coupling(name = 'c_UVAG0Mass1EW',
                          value = '(ZAWcft_UV_EW*MZ)/4.',
                          order = {'QED' : 2})


c_UVAG0Mass2EW = Coupling(name = 'c_UVAG0Mass2EW',
                          value = '-(ZAWcft_UV_EW*MZ)/4.',
                          order = {'QED' : 2})


c_UVHMass1EW = Coupling(name = 'c_UVHMass1EW',
                        value = 'HWcft_UV_EW*complex(0,-1)',
                        order = {'QED' : 2})


c_UVHMass2EW = Coupling(name = 'c_UVHMass2EW',
                        value = '(HMass2_UV_EW + HWcft_UV_EW*MH**2)*complex(0,-1)',
                        order = {'QED' : 2})


c_UVG0Mass1EW = Coupling(name = 'c_UVG0Mass1EW',
                         value = 'G0Wcft_UV_EW*complex(0,-1)',
                         order = {'QED' : 2})


c_UVG0Mass2EW = Coupling(name = 'c_UVG0Mass2EW',
                         value = '(HiggsTadpole_UV_EW*ee*complex(0,0.5))/(MW*sw)',
                         order = {'QED' : 2})


c_UVGmGpMass1EW = Coupling(name = 'c_UVGmGpMass1EW',
                           value = 'GpWcft_UV_EW*complex(0,-1)',
                           order = {'QED' : 2})


c_UVvevexMass1EW = Coupling(name = 'c_UVvevexMass1EW',
                            value = 'complex(0,-0.5)*(complexconjugate(veWcft_UV_EW_L) + veWcft_UV_EW_L)',
                            order = {'QED' : 2})


c_UVvevexMass2EW = Coupling(name = 'c_UVvevexMass2EW',
                            value = '0',
                            order = {'QED' : 0})


c_UVvevmxMass1EW = Coupling(name = 'c_UVvevmxMass1EW',
                            value = '0',
                            order = {'QED' : 0})


c_UVvmvmxMass1EW = Coupling(name = 'c_UVvmvmxMass1EW',
                            value = 'complex(0,-0.5)*(complexconjugate(vmWcft_UV_EW_L) + vmWcft_UV_EW_L)',
                            order = {'QED' : 2})


c_UVvmvmxMass2EW = Coupling(name = 'c_UVvmvmxMass2EW',
                            value = '0',
                            order = {'QED' : 0})


c_UVvtvtxMass1EW = Coupling(name = 'c_UVvtvtxMass1EW',
                            value = 'complex(0,-0.5)*(complexconjugate(vtWcft_UV_EW_L) + vtWcft_UV_EW_L)',
                            order = {'QED' : 2})


c_UVvtvtxMass2EW = Coupling(name = 'c_UVvtvtxMass2EW',
                            value = '0',
                            order = {'QED' : 0})


c_UVemepMass1EW = Coupling(name = 'c_UVemepMass1EW',
                           value = 'complex(0,-0.5)*(complexconjugate(eWcft_UV_EW_L) + eWcft_UV_EW_L)',
                           order = {'QED' : 2})


c_UVemepMass2EW = Coupling(name = 'c_UVemepMass2EW',
                           value = 'complex(0,0.5)*(complexconjugate(eWcft_UV_EW_R) + eWcft_UV_EW_R)',
                           order = {'QED' : 2})


c_UVemepMass3EW = Coupling(name = 'c_UVemepMass3EW',
                           value = 'Me*complex(0,-0.5)*(complexconjugate(eWcft_UV_EW_R) + eWcft_UV_EW_L)',
                           order = {'QED' : 2})


c_UVemepMass4EW = Coupling(name = 'c_UVemepMass4EW',
                           value = 'Me*complex(0,-0.5)*(complexconjugate(eWcft_UV_EW_L) + eWcft_UV_EW_R)',
                           order = {'QED' : 2})


c_UVmmmpMass1EW = Coupling(name = 'c_UVmmmpMass1EW',
                           value = 'complex(0,-0.5)*(complexconjugate(muWcft_UV_EW_L) + muWcft_UV_EW_L)',
                           order = {'QED' : 2})


c_UVmmmpMass2EW = Coupling(name = 'c_UVmmmpMass2EW',
                           value = 'complex(0,0.5)*(complexconjugate(muWcft_UV_EW_R) + muWcft_UV_EW_R)',
                           order = {'QED' : 2})


c_UVmmmpMass3EW = Coupling(name = 'c_UVmmmpMass3EW',
                           value = 'MM*complex(0,-0.5)*(complexconjugate(muWcft_UV_EW_R) + muWcft_UV_EW_L)',
                           order = {'QED' : 2})


c_UVmmmpMass4EW = Coupling(name = 'c_UVmmmpMass4EW',
                           value = 'MM*complex(0,-0.5)*(complexconjugate(muWcft_UV_EW_L) + muWcft_UV_EW_R)',
                           order = {'QED' : 2})


c_UVttmttpMass1EW = Coupling(name = 'c_UVttmttpMass1EW',
                             value = 'complex(0,-0.5)*(complexconjugate(tauWcft_UV_EW_L) + tauWcft_UV_EW_L)',
                             order = {'QED' : 2})


c_UVttmttpMass2EW = Coupling(name = 'c_UVttmttpMass2EW',
                             value = 'complex(0,0.5)*(complexconjugate(tauWcft_UV_EW_R) + tauWcft_UV_EW_R)',
                             order = {'QED' : 2})


c_UVttmttpMass3EW = Coupling(name = 'c_UVttmttpMass3EW',
                             value = 'MTA*complex(0,-0.5)*(complexconjugate(tauWcft_UV_EW_R) + tauWcft_UV_EW_L)',
                             order = {'QED' : 2})


c_UVttmttpMass4EW = Coupling(name = 'c_UVttmttpMass4EW',
                             value = 'MTA*complex(0,-0.5)*(complexconjugate(tauWcft_UV_EW_L) + tauWcft_UV_EW_R)',
                             order = {'QED' : 2})


c_UVuuxMass1EW = Coupling(name = 'c_UVuuxMass1EW',
                          value = 'complex(0,-0.5)*(complexconjugate(uWcft_UV_EW_L) + uWcft_UV_EW_L)',
                          order = {'QED' : 2})


c_UVuuxMass2EW = Coupling(name = 'c_UVuuxMass2EW',
                          value = 'complex(0,0.5)*(complexconjugate(uWcft_UV_EW_R) + uWcft_UV_EW_R)',
                          order = {'QED' : 2})


c_UVuuxMass3EW = Coupling(name = 'c_UVuuxMass3EW',
                          value = 'MU*complex(0,-0.5)*(complexconjugate(uWcft_UV_EW_R) + uWcft_UV_EW_L)',
                          order = {'QED' : 2})


c_UVuuxMass4EW = Coupling(name = 'c_UVuuxMass4EW',
                          value = 'MU*complex(0,-0.5)*(complexconjugate(uWcft_UV_EW_L) + uWcft_UV_EW_R)',
                          order = {'QED' : 2})


c_UVcxuMass1EW = Coupling(name = 'c_UVcxuMass1EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVcxuMass2EW = Coupling(name = 'c_UVcxuMass2EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVcxuMass3EW = Coupling(name = 'c_UVcxuMass3EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVcxuMass4EW = Coupling(name = 'c_UVcxuMass4EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVtxuMass1EW = Coupling(name = 'c_UVtxuMass1EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVtxuMass2EW = Coupling(name = 'c_UVtxuMass2EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVtxuMass3EW = Coupling(name = 'c_UVtxuMass3EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVtxuMass4EW = Coupling(name = 'c_UVtxuMass4EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVcuxMass1EW = Coupling(name = 'c_UVcuxMass1EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVcuxMass2EW = Coupling(name = 'c_UVcuxMass2EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVcuxMass3EW = Coupling(name = 'c_UVcuxMass3EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVcuxMass4EW = Coupling(name = 'c_UVcuxMass4EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVccxMass1EW = Coupling(name = 'c_UVccxMass1EW',
                          value = 'complex(0,-0.5)*(complexconjugate(cWcft_UV_EW_L) + cWcft_UV_EW_L)',
                          order = {'QED' : 2})


c_UVccxMass2EW = Coupling(name = 'c_UVccxMass2EW',
                          value = 'complex(0,0.5)*(complexconjugate(cWcft_UV_EW_R) + cWcft_UV_EW_R)',
                          order = {'QED' : 2})


c_UVccxMass3EW = Coupling(name = 'c_UVccxMass3EW',
                          value = 'MC*complex(0,-0.5)*(complexconjugate(cWcft_UV_EW_R) + cWcft_UV_EW_L)',
                          order = {'QED' : 2})


c_UVccxMass4EW = Coupling(name = 'c_UVccxMass4EW',
                          value = 'MC*complex(0,-0.5)*(complexconjugate(cWcft_UV_EW_L) + cWcft_UV_EW_R)',
                          order = {'QED' : 2})


c_UVctxMass1EW = Coupling(name = 'c_UVctxMass1EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVctxMass2EW = Coupling(name = 'c_UVctxMass2EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVctxMass3EW = Coupling(name = 'c_UVctxMass3EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVctxMass4EW = Coupling(name = 'c_UVctxMass4EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVtuxMass1EW = Coupling(name = 'c_UVtuxMass1EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVtuxMass2EW = Coupling(name = 'c_UVtuxMass2EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVtuxMass3EW = Coupling(name = 'c_UVtuxMass3EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVtuxMass4EW = Coupling(name = 'c_UVtuxMass4EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVcxtMass1EW = Coupling(name = 'c_UVcxtMass1EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVcxtMass2EW = Coupling(name = 'c_UVcxtMass2EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVcxtMass3EW = Coupling(name = 'c_UVcxtMass3EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVcxtMass4EW = Coupling(name = 'c_UVcxtMass4EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVttxMass1EW = Coupling(name = 'c_UVttxMass1EW',
                          value = 'complex(0,-0.5)*(complexconjugate(tWcft_UV_EW_L) + tWcft_UV_EW_L)',
                          order = {'QED' : 2})


c_UVttxMass2EW = Coupling(name = 'c_UVttxMass2EW',
                          value = 'complex(0,0.5)*(complexconjugate(tWcft_UV_EW_R) + tWcft_UV_EW_R)',
                          order = {'QED' : 2})


c_UVttxMass3EW = Coupling(name = 'c_UVttxMass3EW',
                          value = 'complex(0,-0.5)*(MT*complexconjugate(tWcft_UV_EW_R) + 2*tMass_UV_EW + MT*tWcft_UV_EW_L)',
                          order = {'QED' : 2})


c_UVttxMass4EW = Coupling(name = 'c_UVttxMass4EW',
                          value = 'complex(0,-0.5)*(MT*complexconjugate(tWcft_UV_EW_L) + 2*tMass_UV_EW + MT*tWcft_UV_EW_R)',
                          order = {'QED' : 2})


c_UVddxMass1EW = Coupling(name = 'c_UVddxMass1EW',
                          value = 'complex(0,-0.5)*(complexconjugate(dWcft_UV_EW_L) + dWcft_UV_EW_L)',
                          order = {'QED' : 2})


c_UVddxMass2EW = Coupling(name = 'c_UVddxMass2EW',
                          value = 'complex(0,0.5)*(complexconjugate(dWcft_UV_EW_R) + dWcft_UV_EW_R)',
                          order = {'QED' : 2})


c_UVddxMass3EW = Coupling(name = 'c_UVddxMass3EW',
                          value = 'MD*complex(0,-0.5)*(complexconjugate(dWcft_UV_EW_R) + dWcft_UV_EW_L)',
                          order = {'QED' : 2})


c_UVddxMass4EW = Coupling(name = 'c_UVddxMass4EW',
                          value = 'MD*complex(0,-0.5)*(complexconjugate(dWcft_UV_EW_L) + dWcft_UV_EW_R)',
                          order = {'QED' : 2})


c_UVdsxMass1EW = Coupling(name = 'c_UVdsxMass1EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVdsxMass2EW = Coupling(name = 'c_UVdsxMass2EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVdsxMass3EW = Coupling(name = 'c_UVdsxMass3EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVdsxMass4EW = Coupling(name = 'c_UVdsxMass4EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbxdMass1EW = Coupling(name = 'c_UVbxdMass1EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbxdMass2EW = Coupling(name = 'c_UVbxdMass2EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbxdMass3EW = Coupling(name = 'c_UVbxdMass3EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbxdMass4EW = Coupling(name = 'c_UVbxdMass4EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVdxsMass1EW = Coupling(name = 'c_UVdxsMass1EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVdxsMass2EW = Coupling(name = 'c_UVdxsMass2EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVdxsMass3EW = Coupling(name = 'c_UVdxsMass3EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVdxsMass4EW = Coupling(name = 'c_UVdxsMass4EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVssxMass1EW = Coupling(name = 'c_UVssxMass1EW',
                          value = 'complex(0,-0.5)*(complexconjugate(sWcft_UV_EW_L) + sWcft_UV_EW_L)',
                          order = {'QED' : 2})


c_UVssxMass2EW = Coupling(name = 'c_UVssxMass2EW',
                          value = 'complex(0,0.5)*(complexconjugate(sWcft_UV_EW_R) + sWcft_UV_EW_R)',
                          order = {'QED' : 2})


c_UVssxMass3EW = Coupling(name = 'c_UVssxMass3EW',
                          value = 'MS*complex(0,-0.5)*(complexconjugate(sWcft_UV_EW_R) + sWcft_UV_EW_L)',
                          order = {'QED' : 2})


c_UVssxMass4EW = Coupling(name = 'c_UVssxMass4EW',
                          value = 'MS*complex(0,-0.5)*(complexconjugate(sWcft_UV_EW_L) + sWcft_UV_EW_R)',
                          order = {'QED' : 2})


c_UVbxsMass1EW = Coupling(name = 'c_UVbxsMass1EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbxsMass2EW = Coupling(name = 'c_UVbxsMass2EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbxsMass3EW = Coupling(name = 'c_UVbxsMass3EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbxsMass4EW = Coupling(name = 'c_UVbxsMass4EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbdxMass1EW = Coupling(name = 'c_UVbdxMass1EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbdxMass2EW = Coupling(name = 'c_UVbdxMass2EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbdxMass3EW = Coupling(name = 'c_UVbdxMass3EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbdxMass4EW = Coupling(name = 'c_UVbdxMass4EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbsxMass1EW = Coupling(name = 'c_UVbsxMass1EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbsxMass2EW = Coupling(name = 'c_UVbsxMass2EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbsxMass3EW = Coupling(name = 'c_UVbsxMass3EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbsxMass4EW = Coupling(name = 'c_UVbsxMass4EW',
                          value = '0',
                          order = {'QED' : 0})


c_UVbbxMass1EW = Coupling(name = 'c_UVbbxMass1EW',
                          value = 'complex(0,-0.5)*(complexconjugate(bWcft_UV_EW_L) + bWcft_UV_EW_L)',
                          order = {'QED' : 2})


c_UVbbxMass2EW = Coupling(name = 'c_UVbbxMass2EW',
                          value = 'complex(0,0.5)*(complexconjugate(bWcft_UV_EW_R) + bWcft_UV_EW_R)',
                          order = {'QED' : 2})


c_UVbbxMass3EW = Coupling(name = 'c_UVbbxMass3EW',
                          value = 'MB*complex(0,-0.5)*(complexconjugate(bWcft_UV_EW_R) + bWcft_UV_EW_L)',
                          order = {'QED' : 2})


c_UVbbxMass4EW = Coupling(name = 'c_UVbbxMass4EW',
                          value = 'MB*complex(0,-0.5)*(complexconjugate(bWcft_UV_EW_L) + bWcft_UV_EW_R)',
                          order = {'QED' : 2})


c_UVWpWpWmWm1EW = Coupling(name = 'c_UVWpWpWmWm1EW',
                           value = '(ee**2*(-SWCoup_UV_EW + (eCoup_UV_EW + WWcft_UV_EW)*sw)*complex(0,4))/sw**3',
                           order = {'QED' : 4})


c_UVWpWpWmWm2EW = Coupling(name = 'c_UVWpWpWmWm2EW',
                           value = '(ee**2*(SWCoup_UV_EW - (eCoup_UV_EW + WWcft_UV_EW)*sw)*complex(0,2))/sw**3',
                           order = {'QED' : 4})


c_UVWpWmZZ1EW = Coupling(name = 'c_UVWpWmZZ1EW',
                         value = '(ee**2*(2*SWCoup_UV_EW - cw*sw*(cw*(2*eCoup_UV_EW + WWcft_UV_EW + ZZWcft_UV_EW) - AZWcft_UV_EW*sw))*complex(0,2))/sw**3',
                         order = {'QED' : 4})


c_UVWpWmZZ2EW = Coupling(name = 'c_UVWpWmZZ2EW',
                         value = '(ee**2*(2*SWCoup_UV_EW - cw*sw*(cw*(2*eCoup_UV_EW + WWcft_UV_EW + ZZWcft_UV_EW) - AZWcft_UV_EW*sw))*complex(0,-1))/sw**3',
                         order = {'QED' : 4})


c_UVWpWmAZ1EW = Coupling(name = 'c_UVWpWmAZ1EW',
                         value = '(ee**2*(2*SWCoup_UV_EW + cw*(cw**2*ZAWcft_UV_EW - cw*(AAWcft_UV_EW + 4*eCoup_UV_EW + 2*WWcft_UV_EW + ZZWcft_UV_EW)*sw + AZWcft_UV_EW*sw**2))*complex(0,1))/(cw*sw**2)',
                         order = {'QED' : 4})


c_UVWpWmAZ2EW = Coupling(name = 'c_UVWpWmAZ2EW',
                         value = '(ee**2*(2*SWCoup_UV_EW + cw*(cw**2*ZAWcft_UV_EW - cw*(AAWcft_UV_EW + 4*eCoup_UV_EW + 2*WWcft_UV_EW + ZZWcft_UV_EW)*sw + AZWcft_UV_EW*sw**2))*complex(0,-0.5))/(cw*sw**2)',
                         order = {'QED' : 4})


c_UVWpWmAA1EW = Coupling(name = 'c_UVWpWmAA1EW',
                         value = '(ee**2*(-(cw*ZAWcft_UV_EW) + (AAWcft_UV_EW + 2*eCoup_UV_EW + WWcft_UV_EW)*sw)*complex(0,-2))/sw',
                         order = {'QED' : 4})


c_UVWpWmAA2EW = Coupling(name = 'c_UVWpWmAA2EW',
                         value = '(ee**2*(-(cw*ZAWcft_UV_EW) + (AAWcft_UV_EW + 2*eCoup_UV_EW + WWcft_UV_EW)*sw)*complex(0,1))/sw',
                         order = {'QED' : 4})


c_UVAWpWm1EW = Coupling(name = 'c_UVAWpWm1EW',
                        value = 'ee*(AAWcft_UV_EW/2. + eCoup_UV_EW + WWcft_UV_EW - (cw*ZAWcft_UV_EW)/(2.*sw))*complex(0,1)',
                        order = {'QED' : 3})


c_UVZWpWm1EW = Coupling(name = 'c_UVZWpWm1EW',
                        value = '(ee*(-2*SWCoup_UV_EW + cw*sw*(cw*(2*eCoup_UV_EW + 2*WWcft_UV_EW + ZZWcft_UV_EW) - AZWcft_UV_EW*sw))*complex(0,0.5))/(cw*sw**2)',
                        order = {'QED' : 3})


c_UVHHHH1EW = Coupling(name = 'c_UVHHHH1EW',
                       value = '(ee**2*(HiggsTadpole_UV_EW*ee*MW - 4*SWCoup_UV_EW*MH**2*MW**2 + 2*(-(WMass2_UV_EW*MH**2) + (HMass2_UV_EW + 2*(eCoup_UV_EW + HWcft_UV_EW)*MH**2)*MW**2)*sw)*complex(0,-0.375))/(MW**4*sw**3)',
                       order = {'QED' : 4})

c_UVHHG0G01EW = Coupling(name = 'c_UVHHG0G01EW',
                         value = '(ee**2*(HiggsTadpole_UV_EW*ee*MW + 2*(-2*SWCoup_UV_EW*MH**2*MW**2 + (-(WMass2_UV_EW*MH**2) + (HMass2_UV_EW + (2*eCoup_UV_EW + G0Wcft_UV_EW + HWcft_UV_EW)*MH**2)*MW**2)*sw))*complex(0,-0.125))/(MW**4*sw**3)',
                         order = {'QED' : 4})

c_UVHHGmGp1EW = Coupling(name = 'c_UVHHGmGp1EW',
                         value = '(ee**2*(HiggsTadpole_UV_EW*ee*MW + 2*(-2*SWCoup_UV_EW*MH**2*MW**2 + (-(WMass2_UV_EW*MH**2) + (HMass2_UV_EW + (2*eCoup_UV_EW + GpWcft_UV_EW + HWcft_UV_EW)*MH**2)*MW**2)*sw))*complex(0,-0.125))/(MW**4*sw**3)',
                         order = {'QED' : 4})


c_UVG0G0G0G01EW = Coupling(name = 'c_UVG0G0G0G01EW',
                           value = '(ee**2*(HiggsTadpole_UV_EW*ee*MW - 4*SWCoup_UV_EW*MH**2*MW**2 + 2*(-(WMass2_UV_EW*MH**2) + (HMass2_UV_EW + 2*(eCoup_UV_EW + G0Wcft_UV_EW)*MH**2)*MW**2)*sw)*complex(0,-0.375))/(MW**4*sw**3)',
                           order = {'QED' : 4})


c_UVG0G0GmGp1EW = Coupling(name = 'c_UVG0G0GmGp1EW',
                           value = '(ee**2*(HiggsTadpole_UV_EW*ee*MW + 2*(-2*SWCoup_UV_EW*MH**2*MW**2 + (-(WMass2_UV_EW*MH**2) + (HMass2_UV_EW + (2*eCoup_UV_EW + G0Wcft_UV_EW + GpWcft_UV_EW)*MH**2)*MW**2)*sw))*complex(0,-0.125))/(MW**4*sw**3)',
                           order = {'QED' : 4})

c_UVGmGmGpGp1EW = Coupling(name = 'c_UVGmGmGpGp1EW',
                           value = '(ee**2*(HiggsTadpole_UV_EW*ee*MW - 4*SWCoup_UV_EW*MH**2*MW**2 + 2*(-(WMass2_UV_EW*MH**2) + (HMass2_UV_EW + 2*(eCoup_UV_EW + GpWcft_UV_EW)*MH**2)*MW**2)*sw)*complex(0,-0.25))/(MW**4*sw**3)',
                           order = {'QED' : 4})

c_UVHHH1EW = Coupling(name = 'c_UVHHH1EW',
                      value = '(ee*(HiggsTadpole_UV_EW*ee*MW - 2*SWCoup_UV_EW*MH**2*MW**2 + (-(WMass2_UV_EW*MH**2) + (2*HMass2_UV_EW + (2*eCoup_UV_EW + 3*HWcft_UV_EW)*MH**2)*MW**2)*sw)*complex(0,-0.75))/(MW**3*sw**2)',
                      order = {'QED' : 3})

c_UVHG0G01EW = Coupling(name = 'c_UVHG0G01EW',
                        value = '(ee*(HiggsTadpole_UV_EW*ee*MW - 2*SWCoup_UV_EW*MH**2*MW**2 + (-(WMass2_UV_EW*MH**2) + (2*HMass2_UV_EW + (2*eCoup_UV_EW + 2*G0Wcft_UV_EW + HWcft_UV_EW)*MH**2)*MW**2)*sw)*complex(0,-0.25))/(MW**3*sw**2)',
                        order = {'QED' : 3})

c_UVGmHGp1EW = Coupling(name = 'c_UVGmHGp1EW',
                        value = '(ee*(HiggsTadpole_UV_EW*ee*MW - 2*SWCoup_UV_EW*MH**2*MW**2 + (-(WMass2_UV_EW*MH**2) + (2*HMass2_UV_EW + (2*eCoup_UV_EW + 2*GpWcft_UV_EW + HWcft_UV_EW)*MH**2)*MW**2)*sw)*complex(0,-0.25))/(MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVHHWmWp1EW = Coupling(name = 'c_UVHHWmWp1EW',
                         value = '(ee**2*(-2*SWCoup_UV_EW + (2*eCoup_UV_EW + HWcft_UV_EW + WWcft_UV_EW)*sw)*complex(0,0.5))/sw**3',
                         order = {'QED' : 4})


c_UVG0G0WmWp1EW = Coupling(name = 'c_UVG0G0WmWp1EW',
                           value = '(ee**2*(-2*SWCoup_UV_EW + (2*eCoup_UV_EW + G0Wcft_UV_EW + WWcft_UV_EW)*sw)*complex(0,0.5))/sw**3',
                           order = {'QED' : 4})


c_UVGmGpWmWp1EW = Coupling(name = 'c_UVGmGpWmWp1EW',
                           value = '(ee**2*(-2*SWCoup_UV_EW + (2*eCoup_UV_EW + GpWcft_UV_EW + WWcft_UV_EW)*sw)*complex(0,0.5))/sw**3',
                           order = {'QED' : 4})


c_UVGmGpZZ1EW = Coupling(name = 'c_UVGmGpZZ1EW',
                         value = '(ee**2*(cw**2 - sw**2)*(-2*SWCoup_UV_EW + cw**2*sw*(cw**2*(2*eCoup_UV_EW + GpWcft_UV_EW + ZZWcft_UV_EW) - 2*cw*AZWcft_UV_EW*sw - (2*eCoup_UV_EW + GpWcft_UV_EW + ZZWcft_UV_EW)*sw**2))*complex(0,0.5))/(cw**4*sw**3)',
                         order = {'QED' : 4})


c_UVGmGpAZ1EW = Coupling(name = 'c_UVGmGpAZ1EW',
                         value = '(ee**2*(4*SWCoup_UV_EW + cw*(cw**4*ZAWcft_UV_EW - 2*cw**3*(AAWcft_UV_EW + 4*eCoup_UV_EW + 2*GpWcft_UV_EW + ZZWcft_UV_EW)*sw + 2*cw**2*(2*AZWcft_UV_EW - ZAWcft_UV_EW)*sw**2 + 2*cw*(AAWcft_UV_EW + 4*eCoup_UV_EW + 2*GpWcft_UV_EW + ZZWcft_UV_EW)*sw**3 + ZAWcft_UV_EW*sw**4))*complex(0,-0.25))/(cw**3*sw**2)',
                         order = {'QED' : 4})


c_UVGmGpAA1EW = Coupling(name = 'c_UVGmGpAA1EW',
                         value = 'ee**2*(AAWcft_UV_EW + 2*eCoup_UV_EW + GpWcft_UV_EW - (cw*ZAWcft_UV_EW)/(2.*sw) + (ZAWcft_UV_EW*sw)/(2.*cw))*complex(0,2)',
                         order = {'QED' : 4})


c_UVHHZZ1EW = Coupling(name = 'c_UVHHZZ1EW',
                       value = '(ee**2*(2*SWCoup_UV_EW*sw**2 + cw**2*(-2*SWCoup_UV_EW + (2*eCoup_UV_EW + HWcft_UV_EW + ZZWcft_UV_EW)*sw))*complex(0,0.5))/(cw**4*sw**3)',
                       order = {'QED' : 4})


c_UVG0G0ZZ1EW = Coupling(name = 'c_UVG0G0ZZ1EW',
                         value = '(ee**2*(2*SWCoup_UV_EW*sw**2 + cw**2*(-2*SWCoup_UV_EW + (2*eCoup_UV_EW + G0Wcft_UV_EW + ZZWcft_UV_EW)*sw))*complex(0,0.5))/(cw**4*sw**3)',
                         order = {'QED' : 4})


c_UVHHAZ1EW = Coupling(name = 'c_UVHHAZ1EW',
                       value = '(ZAWcft_UV_EW*ee**2*complex(0,-0.25))/(cw**2*sw**2)',
                       order = {'QED' : 4})


c_UVHGmWpZ1EW = Coupling(name = 'c_UVHGmWpZ1EW',
                         value = '-(ee**2*(cw**2*AZWcft_UV_EW - 2*CWCoup_UV_EW*sw + cw*(4*eCoup_UV_EW + GpWcft_UV_EW + HWcft_UV_EW + WWcft_UV_EW + ZZWcft_UV_EW)*sw))/(4.*cw**2*sw)',
                         order = {'QED' : 4})


c_UVHGpWmZ1EW = Coupling(name = 'c_UVHGpWmZ1EW',
                         value = '(ee**2*(cw**2*AZWcft_UV_EW - 2*CWCoup_UV_EW*sw + cw*(4*eCoup_UV_EW + GpWcft_UV_EW + HWcft_UV_EW + WWcft_UV_EW + ZZWcft_UV_EW)*sw))/(4.*cw**2*sw)',
                         order = {'QED' : 4})


c_UVHGpWmA1EW = Coupling(name = 'c_UVHGpWmA1EW',
                         value = '-(ee**2*(ZAWcft_UV_EW*sw**2 + cw*(-2*SWCoup_UV_EW + (AAWcft_UV_EW + 4*eCoup_UV_EW + GpWcft_UV_EW + HWcft_UV_EW + WWcft_UV_EW)*sw)))/(4.*cw*sw**2)',
                         order = {'QED' : 4})


c_UVHGmWpA1EW = Coupling(name = 'c_UVHGmWpA1EW',
                         value = '(ee**2*(ZAWcft_UV_EW*sw**2 + cw*(-2*SWCoup_UV_EW + (AAWcft_UV_EW + 4*eCoup_UV_EW + GpWcft_UV_EW + HWcft_UV_EW + WWcft_UV_EW)*sw)))/(4.*cw*sw**2)',
                         order = {'QED' : 4})


c_UVGpG0ZWm1EW = Coupling(name = 'c_UVGpG0ZWm1EW',
                          value = '(ee**2*(cw**2*AZWcft_UV_EW - 2*CWCoup_UV_EW*sw + cw*(4*eCoup_UV_EW + G0Wcft_UV_EW + GpWcft_UV_EW + WWcft_UV_EW + ZZWcft_UV_EW)*sw)*complex(0,0.25))/(cw**2*sw)',
                          order = {'QED' : 4})


c_UVGpG0AWm1EW = Coupling(name = 'c_UVGpG0AWm1EW',
                          value = '(ee**2*(ZAWcft_UV_EW*sw**2 + cw*(-2*SWCoup_UV_EW + (AAWcft_UV_EW + 4*eCoup_UV_EW + G0Wcft_UV_EW + GpWcft_UV_EW + WWcft_UV_EW)*sw))*complex(0,-0.25))/(cw*sw**2)',
                          order = {'QED' : 4})


c_UVG0HA1EW = Coupling(name = 'c_UVG0HA1EW',
                       value = '(ZAWcft_UV_EW*ee)/(4.*cw*sw)',
                       order = {'QED' : 3})


c_UVG0HZ1EW = Coupling(name = 'c_UVG0HZ1EW',
                       value = '-(ee*(2*SWCoup_UV_EW*sw**2 + cw**2*(-2*SWCoup_UV_EW + (2*eCoup_UV_EW + G0Wcft_UV_EW + HWcft_UV_EW + ZZWcft_UV_EW)*sw)))/(4.*cw**3*sw**2)',
                       order = {'QED' : 3})


c_UVGpGmA1EW = Coupling(name = 'c_UVGpGmA1EW',
                        value = 'ee*(AAWcft_UV_EW/2. + eCoup_UV_EW + GpWcft_UV_EW - (cw*ZAWcft_UV_EW)/(4.*sw) + (ZAWcft_UV_EW*sw)/(4.*cw))*complex(0,1)',
                        order = {'QED' : 3})


c_UVGpGmZ1EW = Coupling(name = 'c_UVGpGmZ1EW',
                        value = '(ee*(2*SWCoup_UV_EW + cw**2*sw*(-(cw**2*(2*eCoup_UV_EW + 2*GpWcft_UV_EW + ZZWcft_UV_EW)) + 2*cw*AZWcft_UV_EW*sw + (2*eCoup_UV_EW + 2*GpWcft_UV_EW + ZZWcft_UV_EW)*sw**2))*complex(0,-0.25))/(cw**3*sw**2)',
                        order = {'QED' : 3})


c_UVGpHWm1EW = Coupling(name = 'c_UVGpHWm1EW',
                        value = '-(ee*(-2*SWCoup_UV_EW + (2*eCoup_UV_EW + GpWcft_UV_EW + HWcft_UV_EW + WWcft_UV_EW)*sw))/(4.*sw**2)',
                        order = {'QED' : 3})


c_UVGpG0Wm1EW = Coupling(name = 'c_UVGpG0Wm1EW',
                         value = '(ee*(-2*SWCoup_UV_EW + (2*eCoup_UV_EW + G0Wcft_UV_EW + GpWcft_UV_EW + WWcft_UV_EW)*sw)*complex(0,-0.25))/sw**2',
                         order = {'QED' : 3})


c_UVGmG0Wp1EW = Coupling(name = 'c_UVGmG0Wp1EW',
                         value = '(ee*(-2*SWCoup_UV_EW + (2*eCoup_UV_EW + G0Wcft_UV_EW + GpWcft_UV_EW + WWcft_UV_EW)*sw)*complex(0,0.25))/sw**2',
                         order = {'QED' : 3})


c_UVHWpWm1EW = Coupling(name = 'c_UVHWpWm1EW',
                        value = '(ee*MW*(eCoup_UV_EW + (HWcft_UV_EW + 2*WWcft_UV_EW + WMass2_UV_EW/MW**2 - (2*SWCoup_UV_EW)/sw)/2.)*complex(0,1))/sw',
                        order = {'QED' : 3})


c_UVHZZ1EW = Coupling(name = 'c_UVHZZ1EW',
                      value = '(ee*MW*(eCoup_UV_EW + (HWcft_UV_EW + 2*ZZWcft_UV_EW + WMass2_UV_EW/MW**2 - (2*SWCoup_UV_EW)/sw + (4*SWCoup_UV_EW*sw)/cw**2)/2.)*complex(0,1))/(cw**2*sw)',
                      order = {'QED' : 3})


c_UVHZA1EW = Coupling(name = 'c_UVHZA1EW',
                      value = '(ZAWcft_UV_EW*ee*MW*complex(0,-0.5))/(cw**2*sw)',
                      order = {'QED' : 3})


c_UVGmWpZ1EW = Coupling(name = 'c_UVGmWpZ1EW',
                        value = '-(ee*(2*SWCoup_UV_EW*MW**2 + cw**2*(cw*AZWcft_UV_EW*MW**2 + (WMass2_UV_EW + (2*eCoup_UV_EW + GpWcft_UV_EW + WWcft_UV_EW + ZZWcft_UV_EW)*MW**2)*sw)))/(2.*cw**3*MW)',
                        order = {'QED' : 3})


c_UVGpWmZ1EW = Coupling(name = 'c_UVGpWmZ1EW',
                        value = '(ee*(2*SWCoup_UV_EW*MW**2 + cw**2*(cw*AZWcft_UV_EW*MW**2 + (WMass2_UV_EW + (2*eCoup_UV_EW + GpWcft_UV_EW + WWcft_UV_EW + ZZWcft_UV_EW)*MW**2)*sw)))/(2.*cw**3*MW)',
                        order = {'QED' : 3})


c_UVGmWpA1EW = Coupling(name = 'c_UVGmWpA1EW',
                        value = '(ee*(cw*(WMass2_UV_EW + (AAWcft_UV_EW + 2*eCoup_UV_EW + GpWcft_UV_EW + WWcft_UV_EW)*MW**2) + ZAWcft_UV_EW*MW**2*sw))/(2.*cw*MW)',
                        order = {'QED' : 3})


c_UVGpWmA1EW = Coupling(name = 'c_UVGpWmA1EW',
                        value = '-(ee*(cw*(WMass2_UV_EW + (AAWcft_UV_EW + 2*eCoup_UV_EW + GpWcft_UV_EW + WWcft_UV_EW)*MW**2) + ZAWcft_UV_EW*MW**2*sw))/(2.*cw*MW)',
                        order = {'QED' : 3})


c_UVvexveA1EW = Coupling(name = 'c_UVvexveA1EW',
                         value = '(ZAWcft_UV_EW*ee*complex(0,-0.25))/(cw*sw)',
                         order = {'QED' : 3})


c_UVepemA1EW = Coupling(name = 'c_UVepemA1EW',
                        value = 'ee*complex(0,-0.5)*(AAWcft_UV_EW + 2*eCoup_UV_EW + (ZAWcft_UV_EW*(-0.5 + sw**2))/(cw*sw) + complexconjugate(eWcft_UV_EW_L) + eWcft_UV_EW_L)',
                        order = {'QED' : 3})


c_UVepemA2EW = Coupling(name = 'c_UVepemA2EW',
                        value = '(ee*complex(0,-0.5)*(ZAWcft_UV_EW*sw + cw*complexconjugate(eWcft_UV_EW_R) + cw*(AAWcft_UV_EW + 2*eCoup_UV_EW + eWcft_UV_EW_R)))/cw',
                        order = {'QED' : 3})


c_UVmpmmA1EW = Coupling(name = 'c_UVmpmmA1EW',
                        value = 'ee*complex(0,-0.5)*(AAWcft_UV_EW + 2*eCoup_UV_EW + (ZAWcft_UV_EW*(-0.5 + sw**2))/(cw*sw) + complexconjugate(muWcft_UV_EW_L) + muWcft_UV_EW_L)',
                        order = {'QED' : 3})


c_UVmpmmA2EW = Coupling(name = 'c_UVmpmmA2EW',
                        value = '(ee*complex(0,-0.5)*(ZAWcft_UV_EW*sw + cw*complexconjugate(muWcft_UV_EW_R) + cw*(AAWcft_UV_EW + 2*eCoup_UV_EW + muWcft_UV_EW_R)))/cw',
                        order = {'QED' : 3})


c_UVttpttmA1EW = Coupling(name = 'c_UVttpttmA1EW',
                          value = 'ee*complex(0,-0.5)*(AAWcft_UV_EW + 2*eCoup_UV_EW + (ZAWcft_UV_EW*(-0.5 + sw**2))/(cw*sw) + complexconjugate(tauWcft_UV_EW_L) + tauWcft_UV_EW_L)',
                          order = {'QED' : 3})


c_UVttpttmA2EW = Coupling(name = 'c_UVttpttmA2EW',
                          value = '(ee*complex(0,-0.5)*(ZAWcft_UV_EW*sw + cw*complexconjugate(tauWcft_UV_EW_R) + cw*(AAWcft_UV_EW + 2*eCoup_UV_EW + tauWcft_UV_EW_R)))/cw',
                          order = {'QED' : 3})


c_UVuxuA1EW = Coupling(name = 'c_UVuxuA1EW',
                       value = '(ee*complex(0,0.08333333333333333)*(ZAWcft_UV_EW*(-3 + 4*sw**2) + 4*cw*sw*complexconjugate(uWcft_UV_EW_L) + 4*cw*sw*(AAWcft_UV_EW + 2*eCoup_UV_EW + uWcft_UV_EW_L)))/(cw*sw)',
                       order = {'QED' : 3})


c_UVuxuA2EW = Coupling(name = 'c_UVuxuA2EW',
                       value = '(ee*complex(0,0.3333333333333333)*(ZAWcft_UV_EW*sw + cw*complexconjugate(uWcft_UV_EW_R) + cw*(AAWcft_UV_EW + 2*eCoup_UV_EW + uWcft_UV_EW_R)))/cw',
                       order = {'QED' : 3})


c_UVcxuA1EW = Coupling(name = 'c_UVcxuA1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVcxuA2EW = Coupling(name = 'c_UVcxuA2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVtxuA1EW = Coupling(name = 'c_UVtxuA1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVtxuA2EW = Coupling(name = 'c_UVtxuA2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVuxcA1EW = Coupling(name = 'c_UVuxcA1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVuxcA2EW = Coupling(name = 'c_UVuxcA2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVcxcA1EW = Coupling(name = 'c_UVcxcA1EW',
                       value = '(ee*complex(0,0.08333333333333333)*(ZAWcft_UV_EW*(-3 + 4*sw**2) + 4*cw*sw*complexconjugate(cWcft_UV_EW_L) + 4*cw*sw*(AAWcft_UV_EW + 2*eCoup_UV_EW + cWcft_UV_EW_L)))/(cw*sw)',
                       order = {'QED' : 3})


c_UVcxcA2EW = Coupling(name = 'c_UVcxcA2EW',
                       value = '(ee*complex(0,0.3333333333333333)*(ZAWcft_UV_EW*sw + cw*complexconjugate(cWcft_UV_EW_R) + cw*(AAWcft_UV_EW + 2*eCoup_UV_EW + cWcft_UV_EW_R)))/cw',
                       order = {'QED' : 3})


c_UVtxcA1EW = Coupling(name = 'c_UVtxcA1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVtxcA2EW = Coupling(name = 'c_UVtxcA2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVuxtA1EW = Coupling(name = 'c_UVuxtA1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVuxtA2EW = Coupling(name = 'c_UVuxtA2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVcxtA1EW = Coupling(name = 'c_UVcxtA1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVcxtA2EW = Coupling(name = 'c_UVcxtA2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVtxtA1EW = Coupling(name = 'c_UVtxtA1EW',
                       value = '(ee*complex(0,0.08333333333333333)*(ZAWcft_UV_EW*(-3 + 4*sw**2) + 4*cw*sw*complexconjugate(tWcft_UV_EW_L) + 4*cw*sw*(AAWcft_UV_EW + 2*eCoup_UV_EW + tWcft_UV_EW_L)))/(cw*sw)',
                       order = {'QED' : 3})


c_UVtxtA2EW = Coupling(name = 'c_UVtxtA2EW',
                       value = '(ee*complex(0,0.3333333333333333)*(ZAWcft_UV_EW*sw + cw*complexconjugate(tWcft_UV_EW_R) + cw*(AAWcft_UV_EW + 2*eCoup_UV_EW + tWcft_UV_EW_R)))/cw',
                       order = {'QED' : 3})


c_UVdxdA1EW = Coupling(name = 'c_UVdxdA1EW',
                       value = '(ee*complex(0,-0.08333333333333333)*(ZAWcft_UV_EW*(-3 + 2*sw**2) + 2*cw*sw*complexconjugate(dWcft_UV_EW_L) + 2*cw*sw*(AAWcft_UV_EW + 2*eCoup_UV_EW + dWcft_UV_EW_L)))/(cw*sw)',
                       order = {'QED' : 3})


c_UVdxdA2EW = Coupling(name = 'c_UVdxdA2EW',
                       value = '(ee*complex(0,-0.16666666666666666)*(ZAWcft_UV_EW*sw + cw*complexconjugate(dWcft_UV_EW_R) + cw*(AAWcft_UV_EW + 2*eCoup_UV_EW + dWcft_UV_EW_R)))/cw',
                       order = {'QED' : 3})


c_UVsxdA1EW = Coupling(name = 'c_UVsxdA1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVsxdA2EW = Coupling(name = 'c_UVsxdA2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVbxdA1EW = Coupling(name = 'c_UVbxdA1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVbxdA2EW = Coupling(name = 'c_UVbxdA2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVdxsA1EW = Coupling(name = 'c_UVdxsA1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVdxsA2EW = Coupling(name = 'c_UVdxsA2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVsxsA1EW = Coupling(name = 'c_UVsxsA1EW',
                       value = '(ee*complex(0,-0.08333333333333333)*(ZAWcft_UV_EW*(-3 + 2*sw**2) + 2*cw*sw*complexconjugate(sWcft_UV_EW_L) + 2*cw*sw*(AAWcft_UV_EW + 2*eCoup_UV_EW + sWcft_UV_EW_L)))/(cw*sw)',
                       order = {'QED' : 3})


c_UVsxsA2EW = Coupling(name = 'c_UVsxsA2EW',
                       value = '(ee*complex(0,-0.16666666666666666)*(ZAWcft_UV_EW*sw + cw*complexconjugate(sWcft_UV_EW_R) + cw*(AAWcft_UV_EW + 2*eCoup_UV_EW + sWcft_UV_EW_R)))/cw',
                       order = {'QED' : 3})


c_UVbxsA1EW = Coupling(name = 'c_UVbxsA1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVbxsA2EW = Coupling(name = 'c_UVbxsA2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVdxbA1EW = Coupling(name = 'c_UVdxbA1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVdxbA2EW = Coupling(name = 'c_UVdxbA2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVsxbA1EW = Coupling(name = 'c_UVsxbA1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVsxbA2EW = Coupling(name = 'c_UVsxbA2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVbxbA1EW = Coupling(name = 'c_UVbxbA1EW',
                       value = '(ee*complex(0,-0.08333333333333333)*(ZAWcft_UV_EW*(-3 + 2*sw**2) + 2*cw*sw*complexconjugate(bWcft_UV_EW_L) + 2*cw*sw*(AAWcft_UV_EW + 2*eCoup_UV_EW + bWcft_UV_EW_L)))/(cw*sw)',
                       order = {'QED' : 3})


c_UVbxbA2EW = Coupling(name = 'c_UVbxbA2EW',
                       value = '(ee*complex(0,-0.16666666666666666)*(ZAWcft_UV_EW*sw + cw*complexconjugate(bWcft_UV_EW_R) + cw*(AAWcft_UV_EW + 2*eCoup_UV_EW + bWcft_UV_EW_R)))/cw',
                       order = {'QED' : 3})


c_UVvexveZ1EW = Coupling(name = 'c_UVvexveZ1EW',
                         value = '(ee*complex(0,0.25)*(2*SWCoup_UV_EW*sw**2 + cw**2*sw*complexconjugate(veWcft_UV_EW_L) + cw**2*(-2*SWCoup_UV_EW + sw*(2*eCoup_UV_EW + ZZWcft_UV_EW + veWcft_UV_EW_L))))/(cw**3*sw**2)',
                         order = {'QED' : 3})


c_UVvmxvmZ1EW = Coupling(name = 'c_UVvmxvmZ1EW',
                         value = '(ee*complex(0,0.25)*(2*SWCoup_UV_EW*sw**2 + cw**2*sw*complexconjugate(vmWcft_UV_EW_L) + cw**2*(-2*SWCoup_UV_EW + sw*(2*eCoup_UV_EW + ZZWcft_UV_EW + vmWcft_UV_EW_L))))/(cw**3*sw**2)',
                         order = {'QED' : 3})


c_UVvtxvtZ1EW = Coupling(name = 'c_UVvtxvtZ1EW',
                         value = '(ee*complex(0,0.25)*(2*SWCoup_UV_EW*sw**2 + cw**2*sw*complexconjugate(vtWcft_UV_EW_L) + cw**2*(-2*SWCoup_UV_EW + sw*(2*eCoup_UV_EW + ZZWcft_UV_EW + vtWcft_UV_EW_L))))/(cw**3*sw**2)',
                         order = {'QED' : 3})


c_UVepemZ1EW = Coupling(name = 'c_UVepemZ1EW',
                        value = '(ee*complex(0,0.25)*(2*SWCoup_UV_EW*sw**2 + 2*cw**3*AZWcft_UV_EW*sw**2 + cw**2*sw*(-1 + 2*sw**2)*complexconjugate(eWcft_UV_EW_L) + cw**2*(2*SWCoup_UV_EW + sw*(-1 + 2*sw**2)*(2*eCoup_UV_EW + ZZWcft_UV_EW + eWcft_UV_EW_L))))/(cw**3*sw**2)',
                        order = {'QED' : 3})


c_UVepemZ2EW = Coupling(name = 'c_UVepemZ2EW',
                        value = '(ee*complex(0,0.5)*(2*SWCoup_UV_EW + cw**2*sw*complexconjugate(eWcft_UV_EW_R) + cw**2*(cw*AZWcft_UV_EW + sw*(2*eCoup_UV_EW + ZZWcft_UV_EW + eWcft_UV_EW_R))))/cw**3',
                        order = {'QED' : 3})


c_UVmpmmZ1EW = Coupling(name = 'c_UVmpmmZ1EW',
                        value = '(ee*complex(0,0.25)*(2*SWCoup_UV_EW*sw**2 + 2*cw**3*AZWcft_UV_EW*sw**2 + cw**2*sw*(-1 + 2*sw**2)*complexconjugate(muWcft_UV_EW_L) + cw**2*(2*SWCoup_UV_EW + sw*(-1 + 2*sw**2)*(2*eCoup_UV_EW + ZZWcft_UV_EW + muWcft_UV_EW_L))))/(cw**3*sw**2)',
                        order = {'QED' : 3})


c_UVmpmmZ2EW = Coupling(name = 'c_UVmpmmZ2EW',
                        value = '(ee*complex(0,0.5)*(2*SWCoup_UV_EW + cw**2*sw*complexconjugate(muWcft_UV_EW_R) + cw**2*(cw*AZWcft_UV_EW + sw*(2*eCoup_UV_EW + ZZWcft_UV_EW + muWcft_UV_EW_R))))/cw**3',
                        order = {'QED' : 3})


c_UVttpttmZ1EW = Coupling(name = 'c_UVttpttmZ1EW',
                          value = '(ee*complex(0,0.25)*(2*SWCoup_UV_EW*sw**2 + 2*cw**3*AZWcft_UV_EW*sw**2 + cw**2*sw*(-1 + 2*sw**2)*complexconjugate(tauWcft_UV_EW_L) + cw**2*(2*SWCoup_UV_EW + sw*(-1 + 2*sw**2)*(2*eCoup_UV_EW + ZZWcft_UV_EW + tauWcft_UV_EW_L))))/(cw**3*sw**2)',
                          order = {'QED' : 3})


c_UVttpttmZ2EW = Coupling(name = 'c_UVttpttmZ2EW',
                          value = '(ee*complex(0,0.5)*(2*SWCoup_UV_EW + cw**2*sw*complexconjugate(tauWcft_UV_EW_R) + cw**2*(cw*AZWcft_UV_EW + sw*(2*eCoup_UV_EW + ZZWcft_UV_EW + tauWcft_UV_EW_R))))/cw**3',
                          order = {'QED' : 3})


c_UVuxuZ1EW = Coupling(name = 'c_UVuxuZ1EW',
                       value = '(ee*complex(0,-0.08333333333333333)*(2*SWCoup_UV_EW*sw**2 + 4*cw**3*AZWcft_UV_EW*sw**2 + cw**2*sw*(-3 + 4*sw**2)*complexconjugate(uWcft_UV_EW_L) + cw**2*(6*SWCoup_UV_EW + sw*(-3 + 4*sw**2)*(2*eCoup_UV_EW + ZZWcft_UV_EW + uWcft_UV_EW_L))))/(cw**3*sw**2)',
                       order = {'QED' : 3})


c_UVuxuZ2EW = Coupling(name = 'c_UVuxuZ2EW',
                       value = '(ee*complex(0,-0.3333333333333333)*(2*SWCoup_UV_EW + cw**2*sw*complexconjugate(uWcft_UV_EW_R) + cw**2*(cw*AZWcft_UV_EW + sw*(2*eCoup_UV_EW + ZZWcft_UV_EW + uWcft_UV_EW_R))))/cw**3',
                       order = {'QED' : 3})


c_UVcxuZ1EW = Coupling(name = 'c_UVcxuZ1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVcxuZ2EW = Coupling(name = 'c_UVcxuZ2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVtxuZ1EW = Coupling(name = 'c_UVtxuZ1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVtxuZ2EW = Coupling(name = 'c_UVtxuZ2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVuxcZ1EW = Coupling(name = 'c_UVuxcZ1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVuxcZ2EW = Coupling(name = 'c_UVuxcZ2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVcxcZ1EW = Coupling(name = 'c_UVcxcZ1EW',
                       value = '(ee*complex(0,-0.08333333333333333)*(2*SWCoup_UV_EW*sw**2 + 4*cw**3*AZWcft_UV_EW*sw**2 + cw**2*sw*(-3 + 4*sw**2)*complexconjugate(cWcft_UV_EW_L) + cw**2*(6*SWCoup_UV_EW + sw*(-3 + 4*sw**2)*(2*eCoup_UV_EW + ZZWcft_UV_EW + cWcft_UV_EW_L))))/(cw**3*sw**2)',
                       order = {'QED' : 3})


c_UVcxcZ2EW = Coupling(name = 'c_UVcxcZ2EW',
                       value = '(ee*complex(0,-0.3333333333333333)*(2*SWCoup_UV_EW + cw**2*sw*complexconjugate(cWcft_UV_EW_R) + cw**2*(cw*AZWcft_UV_EW + sw*(2*eCoup_UV_EW + ZZWcft_UV_EW + cWcft_UV_EW_R))))/cw**3',
                       order = {'QED' : 3})


c_UVtxcZ1EW = Coupling(name = 'c_UVtxcZ1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVtxcZ2EW = Coupling(name = 'c_UVtxcZ2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVuxtZ1EW = Coupling(name = 'c_UVuxtZ1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVuxtZ2EW = Coupling(name = 'c_UVuxtZ2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVcxtZ1EW = Coupling(name = 'c_UVcxtZ1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVcxtZ2EW = Coupling(name = 'c_UVcxtZ2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVtxtZ1EW = Coupling(name = 'c_UVtxtZ1EW',
                       value = '(ee*complex(0,-0.08333333333333333)*(2*SWCoup_UV_EW*sw**2 + 4*cw**3*AZWcft_UV_EW*sw**2 + cw**2*sw*(-3 + 4*sw**2)*complexconjugate(tWcft_UV_EW_L) + cw**2*(6*SWCoup_UV_EW + sw*(-3 + 4*sw**2)*(2*eCoup_UV_EW + ZZWcft_UV_EW + tWcft_UV_EW_L))))/(cw**3*sw**2)',
                       order = {'QED' : 3})


c_UVtxtZ2EW = Coupling(name = 'c_UVtxtZ2EW',
                       value = '(ee*complex(0,-0.3333333333333333)*(2*SWCoup_UV_EW + cw**2*sw*complexconjugate(tWcft_UV_EW_R) + cw**2*(cw*AZWcft_UV_EW + sw*(2*eCoup_UV_EW + ZZWcft_UV_EW + tWcft_UV_EW_R))))/cw**3',
                       order = {'QED' : 3})


c_UVdxdZ1EW = Coupling(name = 'c_UVdxdZ1EW',
                       value = '(ee*complex(0,0.08333333333333333)*(-2*SWCoup_UV_EW*sw**2 + 2*cw**3*AZWcft_UV_EW*sw**2 + cw**2*sw*(-3 + 2*sw**2)*complexconjugate(dWcft_UV_EW_L) + cw**2*(6*SWCoup_UV_EW + sw*(-3 + 2*sw**2)*(2*eCoup_UV_EW + ZZWcft_UV_EW + dWcft_UV_EW_L))))/(cw**3*sw**2)',
                       order = {'QED' : 3})


c_UVdxdZ2EW = Coupling(name = 'c_UVdxdZ2EW',
                       value = '(ee*complex(0,0.16666666666666666)*(2*SWCoup_UV_EW + cw**2*sw*complexconjugate(dWcft_UV_EW_R) + cw**2*(cw*AZWcft_UV_EW + sw*(2*eCoup_UV_EW + ZZWcft_UV_EW + dWcft_UV_EW_R))))/cw**3',
                       order = {'QED' : 3})


c_UVsxdZ1EW = Coupling(name = 'c_UVsxdZ1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVsxdZ2EW = Coupling(name = 'c_UVsxdZ2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVbxdZ1EW = Coupling(name = 'c_UVbxdZ1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVbxdZ2EW = Coupling(name = 'c_UVbxdZ2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVdxsZ1EW = Coupling(name = 'c_UVdxsZ1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVdxsZ2EW = Coupling(name = 'c_UVdxsZ2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVsxsZ1EW = Coupling(name = 'c_UVsxsZ1EW',
                       value = '(ee*complex(0,0.08333333333333333)*(-2*SWCoup_UV_EW*sw**2 + 2*cw**3*AZWcft_UV_EW*sw**2 + cw**2*sw*(-3 + 2*sw**2)*complexconjugate(sWcft_UV_EW_L) + cw**2*(6*SWCoup_UV_EW + sw*(-3 + 2*sw**2)*(2*eCoup_UV_EW + ZZWcft_UV_EW + sWcft_UV_EW_L))))/(cw**3*sw**2)',
                       order = {'QED' : 3})


c_UVsxsZ2EW = Coupling(name = 'c_UVsxsZ2EW',
                       value = '(ee*complex(0,0.16666666666666666)*(2*SWCoup_UV_EW + cw**2*sw*complexconjugate(sWcft_UV_EW_R) + cw**2*(cw*AZWcft_UV_EW + sw*(2*eCoup_UV_EW + ZZWcft_UV_EW + sWcft_UV_EW_R))))/cw**3',
                       order = {'QED' : 3})


c_UVbxsZ1EW = Coupling(name = 'c_UVbxsZ1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVbxsZ2EW = Coupling(name = 'c_UVbxsZ2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVdxbZ1EW = Coupling(name = 'c_UVdxbZ1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVdxbZ2EW = Coupling(name = 'c_UVdxbZ2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVsxbZ1EW = Coupling(name = 'c_UVsxbZ1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVsxbZ2EW = Coupling(name = 'c_UVsxbZ2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVbxbZ1EW = Coupling(name = 'c_UVbxbZ1EW',
                       value = '(ee*complex(0,0.08333333333333333)*(-2*SWCoup_UV_EW*sw**2 + 2*cw**3*AZWcft_UV_EW*sw**2 + cw**2*sw*(-3 + 2*sw**2)*complexconjugate(bWcft_UV_EW_L) + cw**2*(6*SWCoup_UV_EW + sw*(-3 + 2*sw**2)*(2*eCoup_UV_EW + ZZWcft_UV_EW + bWcft_UV_EW_L))))/(cw**3*sw**2)',
                       order = {'QED' : 3})


c_UVbxbZ2EW = Coupling(name = 'c_UVbxbZ2EW',
                       value = '(ee*complex(0,0.16666666666666666)*(2*SWCoup_UV_EW + cw**2*sw*complexconjugate(bWcft_UV_EW_R) + cw**2*(cw*AZWcft_UV_EW + sw*(2*eCoup_UV_EW + ZZWcft_UV_EW + bWcft_UV_EW_R))))/cw**3',
                       order = {'QED' : 3})


c_UVepveWm1EW = Coupling(name = 'c_UVepveWm1EW',
                         value = '(ee*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(veWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + eWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                         order = {'QED' : 3})


c_UVmpvmWm1EW = Coupling(name = 'c_UVmpvmWm1EW',
                         value = '(ee*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(vmWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + muWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                         order = {'QED' : 3})


c_UVttpvtWm1EW = Coupling(name = 'c_UVttpvtWm1EW',
                          value = '(ee*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(vtWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + tauWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                          order = {'QED' : 3})


c_UVvexemWp1EW = Coupling(name = 'c_UVvexemWp1EW',
                          value = '(ee*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(eWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + veWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                          order = {'QED' : 3})


c_UVvmxmmWp1EW = Coupling(name = 'c_UVvmxmmWp1EW',
                          value = '(ee*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(muWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + vmWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                          order = {'QED' : 3})


c_UVvtxttmWp1EW = Coupling(name = 'c_UVvtxttmWp1EW',
                           value = '(ee*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(tauWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + vtWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                           order = {'QED' : 3})


c_UVdxuWm1EW = Coupling(name = 'c_UVdxuWm1EW',
                        value = '(ee*CKM11*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(uWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + dWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVsxuWm1EW = Coupling(name = 'c_UVsxuWm1EW',
                        value = '(ee*CKM12*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(uWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + sWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVbxuWm1EW = Coupling(name = 'c_UVbxuWm1EW',
                        value = '(ee*CKM13*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(uWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + bWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVdxcWm1EW = Coupling(name = 'c_UVdxcWm1EW',
                        value = '(ee*CKM21*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(cWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + dWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVsxcWm1EW = Coupling(name = 'c_UVsxcWm1EW',
                        value = '(ee*CKM22*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(cWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + sWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVbxcWm1EW = Coupling(name = 'c_UVbxcWm1EW',
                        value = '(ee*CKM23*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(cWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + bWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVdxtWm1EW = Coupling(name = 'c_UVdxtWm1EW',
                        value = '(ee*CKM31*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(tWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + dWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVsxtWm1EW = Coupling(name = 'c_UVsxtWm1EW',
                        value = '(ee*CKM32*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(tWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + sWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVbxtWm1EW = Coupling(name = 'c_UVbxtWm1EW',
                        value = '(ee*CKM33*complex(0,0.5)*(-2*SWCoup_UV_EW + sw*complexconjugate(tWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + bWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVuxdWp1EW = Coupling(name = 'c_UVuxdWp1EW',
                        value = '(ee*complex(0,0.5)*complexconjugate(CKM11)*(-2*SWCoup_UV_EW + sw*complexconjugate(dWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + uWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVcxdWp1EW = Coupling(name = 'c_UVcxdWp1EW',
                        value = '(ee*complex(0,0.5)*complexconjugate(CKM21)*(-2*SWCoup_UV_EW + sw*complexconjugate(dWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + cWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVtxdWp1EW = Coupling(name = 'c_UVtxdWp1EW',
                        value = '(ee*complex(0,0.5)*complexconjugate(CKM31)*(-2*SWCoup_UV_EW + sw*complexconjugate(dWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + tWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVuxsWp1EW = Coupling(name = 'c_UVuxsWp1EW',
                        value = '(ee*complex(0,0.5)*complexconjugate(CKM12)*(-2*SWCoup_UV_EW + sw*complexconjugate(sWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + uWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVcxsWp1EW = Coupling(name = 'c_UVcxsWp1EW',
                        value = '(ee*complex(0,0.5)*complexconjugate(CKM22)*(-2*SWCoup_UV_EW + sw*complexconjugate(sWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + cWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVtxsWp1EW = Coupling(name = 'c_UVtxsWp1EW',
                        value = '(ee*complex(0,0.5)*complexconjugate(CKM32)*(-2*SWCoup_UV_EW + sw*complexconjugate(sWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + tWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVuxbWp1EW = Coupling(name = 'c_UVuxbWp1EW',
                        value = '(ee*complex(0,0.5)*complexconjugate(CKM13)*(-2*SWCoup_UV_EW + sw*complexconjugate(bWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + uWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVcxbWp1EW = Coupling(name = 'c_UVcxbWp1EW',
                        value = '(ee*complex(0,0.5)*complexconjugate(CKM23)*(-2*SWCoup_UV_EW + sw*complexconjugate(bWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + cWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVtxbWp1EW = Coupling(name = 'c_UVtxbWp1EW',
                        value = '(ee*complex(0,0.5)*complexconjugate(CKM33)*(-2*SWCoup_UV_EW + sw*complexconjugate(bWcft_UV_EW_L) + sw*(2*eCoup_UV_EW + WWcft_UV_EW + tWcft_UV_EW_L)))/(cmath.sqrt(2)*sw**2)',
                        order = {'QED' : 3})


c_UVepemH1EW = Coupling(name = 'c_UVepemH1EW',
                        value = '(ee*Me*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(eWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + eWcft_UV_EW_L))))/(MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVepemH2EW = Coupling(name = 'c_UVepemH2EW',
                        value = '(ee*Me*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(eWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + eWcft_UV_EW_R))))/(MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVmpmmH1EW = Coupling(name = 'c_UVmpmmH1EW',
                        value = '(ee*MM*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(muWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + muWcft_UV_EW_L))))/(MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVmpmmH2EW = Coupling(name = 'c_UVmpmmH2EW',
                        value = '(ee*MM*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(muWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + muWcft_UV_EW_R))))/(MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVttpttmH1EW = Coupling(name = 'c_UVttpttmH1EW',
                          value = '(ee*MTA*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(tauWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + tauWcft_UV_EW_L))))/(MW**3*sw**2)',
                          order = {'QED' : 3})


c_UVttpttmH2EW = Coupling(name = 'c_UVttpttmH2EW',
                          value = '(ee*MTA*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(tauWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + tauWcft_UV_EW_R))))/(MW**3*sw**2)',
                          order = {'QED' : 3})


c_UVuxuH1EW = Coupling(name = 'c_UVuxuH1EW',
                       value = '(ee*MU*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(uWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + uWcft_UV_EW_L))))/(MW**3*sw**2)',
                       order = {'QED' : 3})


c_UVuxuH2EW = Coupling(name = 'c_UVuxuH2EW',
                       value = '(ee*MU*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(uWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + uWcft_UV_EW_R))))/(MW**3*sw**2)',
                       order = {'QED' : 3})


c_UVcxuH1EW = Coupling(name = 'c_UVcxuH1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVcxuH2EW = Coupling(name = 'c_UVcxuH2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVtxuH1EW = Coupling(name = 'c_UVtxuH1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVtxuH2EW = Coupling(name = 'c_UVtxuH2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVuxcH1EW = Coupling(name = 'c_UVuxcH1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVuxcH2EW = Coupling(name = 'c_UVuxcH2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVcxcH1EW = Coupling(name = 'c_UVcxcH1EW',
                       value = '(ee*MC*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(cWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + cWcft_UV_EW_L))))/(MW**3*sw**2)',
                       order = {'QED' : 3})


c_UVcxcH2EW = Coupling(name = 'c_UVcxcH2EW',
                       value = '(ee*MC*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(cWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + cWcft_UV_EW_R))))/(MW**3*sw**2)',
                       order = {'QED' : 3})


c_UVtxcH1EW = Coupling(name = 'c_UVtxcH1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVtxcH2EW = Coupling(name = 'c_UVtxcH2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVuxtH1EW = Coupling(name = 'c_UVuxtH1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVuxtH2EW = Coupling(name = 'c_UVuxtH2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVcxtH1EW = Coupling(name = 'c_UVcxtH1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVcxtH2EW = Coupling(name = 'c_UVcxtH2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVtxtH1EW = Coupling(name = 'c_UVtxtH1EW',
                       value = '(ee*complex(0,-0.25)*(-2*SWCoup_UV_EW*MT*MW**2 + MT*MW**2*sw*complexconjugate(tWcft_UV_EW_R) + sw*(-(WMass2_UV_EW*MT) + MW**2*(2*eCoup_UV_EW*MT + HWcft_UV_EW*MT + 2*tMass_UV_EW + MT*tWcft_UV_EW_L))))/(MW**3*sw**2)',
                       order = {'QED' : 3})


c_UVtxtH2EW = Coupling(name = 'c_UVtxtH2EW',
                       value = '(ee*complex(0,-0.25)*(-2*SWCoup_UV_EW*MT*MW**2 + MT*MW**2*sw*complexconjugate(tWcft_UV_EW_L) + sw*(-(WMass2_UV_EW*MT) + MW**2*(2*eCoup_UV_EW*MT + HWcft_UV_EW*MT + 2*tMass_UV_EW + MT*tWcft_UV_EW_R))))/(MW**3*sw**2)',
                       order = {'QED' : 3})


c_UVdxdH1EW = Coupling(name = 'c_UVdxdH1EW',
                       value = '(ee*MD*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(dWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + dWcft_UV_EW_L))))/(MW**3*sw**2)',
                       order = {'QED' : 3})


c_UVdxdH2EW = Coupling(name = 'c_UVdxdH2EW',
                       value = '(ee*MD*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(dWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + dWcft_UV_EW_R))))/(MW**3*sw**2)',
                       order = {'QED' : 3})


c_UVsxdH1EW = Coupling(name = 'c_UVsxdH1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVsxdH2EW = Coupling(name = 'c_UVsxdH2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVbxdH1EW = Coupling(name = 'c_UVbxdH1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVbxdH2EW = Coupling(name = 'c_UVbxdH2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVdxsH1EW = Coupling(name = 'c_UVdxsH1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVdxsH2EW = Coupling(name = 'c_UVdxsH2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVsxsH1EW = Coupling(name = 'c_UVsxsH1EW',
                       value = '(ee*MS*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(sWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + sWcft_UV_EW_L))))/(MW**3*sw**2)',
                       order = {'QED' : 3})


c_UVsxsH2EW = Coupling(name = 'c_UVsxsH2EW',
                       value = '(ee*MS*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(sWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + sWcft_UV_EW_R))))/(MW**3*sw**2)',
                       order = {'QED' : 3})


c_UVbxsH1EW = Coupling(name = 'c_UVbxsH1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVbxsH2EW = Coupling(name = 'c_UVbxsH2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVdxbH1EW = Coupling(name = 'c_UVdxbH1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVdxbH2EW = Coupling(name = 'c_UVdxbH2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVsxbH1EW = Coupling(name = 'c_UVsxbH1EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVsxbH2EW = Coupling(name = 'c_UVsxbH2EW',
                       value = '0',
                       order = {'QED' : 0})


c_UVbxbH1EW = Coupling(name = 'c_UVbxbH1EW',
                       value = '(ee*MB*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(bWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + bWcft_UV_EW_L))))/(MW**3*sw**2)',
                       order = {'QED' : 3})


c_UVbxbH2EW = Coupling(name = 'c_UVbxbH2EW',
                       value = '(ee*MB*complex(0,-0.25)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(bWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + HWcft_UV_EW + bWcft_UV_EW_R))))/(MW**3*sw**2)',
                       order = {'QED' : 3})


c_UVepemG01EW = Coupling(name = 'c_UVepemG01EW',
                         value = '-(ee*Me*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(eWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + eWcft_UV_EW_L))))/(4.*MW**3*sw**2)',
                         order = {'QED' : 3})


c_UVepemG02EW = Coupling(name = 'c_UVepemG02EW',
                         value = '(ee*Me*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(eWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + eWcft_UV_EW_R))))/(4.*MW**3*sw**2)',
                         order = {'QED' : 3})


c_UVmpmmG01EW = Coupling(name = 'c_UVmpmmG01EW',
                         value = '-(ee*MM*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(muWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + muWcft_UV_EW_L))))/(4.*MW**3*sw**2)',
                         order = {'QED' : 3})


c_UVmpmmG02EW = Coupling(name = 'c_UVmpmmG02EW',
                         value = '(ee*MM*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(muWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + muWcft_UV_EW_R))))/(4.*MW**3*sw**2)',
                         order = {'QED' : 3})


c_UVttpttmG01EW = Coupling(name = 'c_UVttpttmG01EW',
                           value = '-(ee*MTA*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(tauWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + tauWcft_UV_EW_L))))/(4.*MW**3*sw**2)',
                           order = {'QED' : 3})


c_UVttpttmG02EW = Coupling(name = 'c_UVttpttmG02EW',
                           value = '(ee*MTA*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(tauWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + tauWcft_UV_EW_R))))/(4.*MW**3*sw**2)',
                           order = {'QED' : 3})


c_UVuxuG01EW = Coupling(name = 'c_UVuxuG01EW',
                        value = '(ee*MU*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(uWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + uWcft_UV_EW_L))))/(4.*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVuxuG02EW = Coupling(name = 'c_UVuxuG02EW',
                        value = '-(ee*MU*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(uWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + uWcft_UV_EW_R))))/(4.*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVcxuG01EW = Coupling(name = 'c_UVcxuG01EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVcxuG02EW = Coupling(name = 'c_UVcxuG02EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVtxuG01EW = Coupling(name = 'c_UVtxuG01EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVtxuG02EW = Coupling(name = 'c_UVtxuG02EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVuxcG01EW = Coupling(name = 'c_UVuxcG01EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVuxcG02EW = Coupling(name = 'c_UVuxcG02EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVcxcG01EW = Coupling(name = 'c_UVcxcG01EW',
                        value = '(ee*MC*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(cWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + cWcft_UV_EW_L))))/(4.*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVcxcG02EW = Coupling(name = 'c_UVcxcG02EW',
                        value = '-(ee*MC*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(cWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + cWcft_UV_EW_R))))/(4.*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVtxcG01EW = Coupling(name = 'c_UVtxcG01EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVtxcG02EW = Coupling(name = 'c_UVtxcG02EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVuxtG01EW = Coupling(name = 'c_UVuxtG01EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVuxtG02EW = Coupling(name = 'c_UVuxtG02EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVcxtG01EW = Coupling(name = 'c_UVcxtG01EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVcxtG02EW = Coupling(name = 'c_UVcxtG02EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVtxtG01EW = Coupling(name = 'c_UVtxtG01EW',
                        value = '(ee*(-2*SWCoup_UV_EW*MT*MW**2 + MT*MW**2*sw*complexconjugate(tWcft_UV_EW_R) + sw*(-(WMass2_UV_EW*MT) + MW**2*(2*eCoup_UV_EW*MT + G0Wcft_UV_EW*MT + 2*tMass_UV_EW + MT*tWcft_UV_EW_L))))/(4.*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVtxtG02EW = Coupling(name = 'c_UVtxtG02EW',
                        value = '-(ee*(-2*SWCoup_UV_EW*MT*MW**2 + MT*MW**2*sw*complexconjugate(tWcft_UV_EW_L) + sw*(-(WMass2_UV_EW*MT) + MW**2*(2*eCoup_UV_EW*MT + G0Wcft_UV_EW*MT + 2*tMass_UV_EW + MT*tWcft_UV_EW_R))))/(4.*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVdxdG01EW = Coupling(name = 'c_UVdxdG01EW',
                        value = '-(ee*MD*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(dWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + dWcft_UV_EW_L))))/(4.*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVdxdG02EW = Coupling(name = 'c_UVdxdG02EW',
                        value = '(ee*MD*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(dWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + dWcft_UV_EW_R))))/(4.*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVsxdG01EW = Coupling(name = 'c_UVsxdG01EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVsxdG02EW = Coupling(name = 'c_UVsxdG02EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVbxdG01EW = Coupling(name = 'c_UVbxdG01EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVbxdG02EW = Coupling(name = 'c_UVbxdG02EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVdxsG01EW = Coupling(name = 'c_UVdxsG01EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVdxsG02EW = Coupling(name = 'c_UVdxsG02EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVsxsG01EW = Coupling(name = 'c_UVsxsG01EW',
                        value = '-(ee*MS*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(sWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + sWcft_UV_EW_L))))/(4.*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVsxsG02EW = Coupling(name = 'c_UVsxsG02EW',
                        value = '(ee*MS*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(sWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + sWcft_UV_EW_R))))/(4.*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVbxsG01EW = Coupling(name = 'c_UVbxsG01EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVbxsG02EW = Coupling(name = 'c_UVbxsG02EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVdxbG01EW = Coupling(name = 'c_UVdxbG01EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVdxbG02EW = Coupling(name = 'c_UVdxbG02EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVsxbG01EW = Coupling(name = 'c_UVsxbG01EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVsxbG02EW = Coupling(name = 'c_UVsxbG02EW',
                        value = '0',
                        order = {'QED' : 0})


c_UVbxbG01EW = Coupling(name = 'c_UVbxbG01EW',
                        value = '-(ee*MB*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(bWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + bWcft_UV_EW_L))))/(4.*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVbxbG02EW = Coupling(name = 'c_UVbxbG02EW',
                        value = '(ee*MB*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(bWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + G0Wcft_UV_EW + bWcft_UV_EW_R))))/(4.*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVdxuGm1EW = Coupling(name = 'c_UVdxuGm1EW',
                        value = '(ee*MU*CKM11*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(uWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + dWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVdxuGm2EW = Coupling(name = 'c_UVdxuGm2EW',
                        value = '-(ee*MD*CKM11*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(uWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + dWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVsxuGm1EW = Coupling(name = 'c_UVsxuGm1EW',
                        value = '(ee*MU*CKM12*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(uWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + sWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVsxuGm2EW = Coupling(name = 'c_UVsxuGm2EW',
                        value = '-(ee*MS*CKM12*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(uWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + sWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVbxuGm1EW = Coupling(name = 'c_UVbxuGm1EW',
                        value = '(ee*MU*CKM13*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(uWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + bWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVbxuGm2EW = Coupling(name = 'c_UVbxuGm2EW',
                        value = '-(ee*MB*CKM13*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(uWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + bWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVdxcGm1EW = Coupling(name = 'c_UVdxcGm1EW',
                        value = '(ee*MC*CKM21*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(cWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + dWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVdxcGm2EW = Coupling(name = 'c_UVdxcGm2EW',
                        value = '-(ee*MD*CKM21*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(cWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + dWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVsxcGm1EW = Coupling(name = 'c_UVsxcGm1EW',
                        value = '(ee*MC*CKM22*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(cWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + sWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVsxcGm2EW = Coupling(name = 'c_UVsxcGm2EW',
                        value = '-(ee*MS*CKM22*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(cWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + sWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVbxcGm1EW = Coupling(name = 'c_UVbxcGm1EW',
                        value = '(ee*MC*CKM23*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(cWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + bWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVbxcGm2EW = Coupling(name = 'c_UVbxcGm2EW',
                        value = '-(ee*MB*CKM23*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(cWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + bWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVdxtGm1EW = Coupling(name = 'c_UVdxtGm1EW',
                        value = '(ee*CKM31*(-2*SWCoup_UV_EW*MT*MW**2 + MT*MW**2*sw*complexconjugate(tWcft_UV_EW_R) + sw*(-(WMass2_UV_EW*MT) + MW**2*(2*eCoup_UV_EW*MT + GpWcft_UV_EW*MT + 2*tMass_UV_EW + MT*dWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVdxtGm2EW = Coupling(name = 'c_UVdxtGm2EW',
                        value = '-(ee*MD*CKM31*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(tWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + dWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVsxtGm1EW = Coupling(name = 'c_UVsxtGm1EW',
                        value = '(ee*CKM32*(-2*SWCoup_UV_EW*MT*MW**2 + MT*MW**2*sw*complexconjugate(tWcft_UV_EW_R) + sw*(-(WMass2_UV_EW*MT) + MW**2*(2*eCoup_UV_EW*MT + GpWcft_UV_EW*MT + 2*tMass_UV_EW + MT*sWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVsxtGm2EW = Coupling(name = 'c_UVsxtGm2EW',
                        value = '-(ee*MS*CKM32*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(tWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + sWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVbxtGm1EW = Coupling(name = 'c_UVbxtGm1EW',
                        value = '(ee*CKM33*(-2*SWCoup_UV_EW*MT*MW**2 + MT*MW**2*sw*complexconjugate(tWcft_UV_EW_R) + sw*(-(WMass2_UV_EW*MT) + MW**2*(2*eCoup_UV_EW*MT + GpWcft_UV_EW*MT + 2*tMass_UV_EW + MT*bWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVbxtGm2EW = Coupling(name = 'c_UVbxtGm2EW',
                        value = '-(ee*MB*CKM33*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(tWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + bWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVuxdGp1EW = Coupling(name = 'c_UVuxdGp1EW',
                        value = '(ee*MD*complexconjugate(CKM11)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(dWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + uWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVuxdGp2EW = Coupling(name = 'c_UVuxdGp2EW',
                        value = '-(ee*MU*complexconjugate(CKM11)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(dWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + uWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVcxdGp1EW = Coupling(name = 'c_UVcxdGp1EW',
                        value = '(ee*MD*complexconjugate(CKM21)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(dWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + cWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVcxdGp2EW = Coupling(name = 'c_UVcxdGp2EW',
                        value = '-(ee*MC*complexconjugate(CKM21)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(dWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + cWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVtxdGp1EW = Coupling(name = 'c_UVtxdGp1EW',
                        value = '(ee*MD*complexconjugate(CKM31)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(dWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + tWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVtxdGp2EW = Coupling(name = 'c_UVtxdGp2EW',
                        value = '-(ee*MT*complexconjugate(CKM31)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(dWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + tWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVuxsGp1EW = Coupling(name = 'c_UVuxsGp1EW',
                        value = '(ee*MS*complexconjugate(CKM12)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(sWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + uWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVuxsGp2EW = Coupling(name = 'c_UVuxsGp2EW',
                        value = '-(ee*MU*complexconjugate(CKM12)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(sWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + uWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVcxsGp1EW = Coupling(name = 'c_UVcxsGp1EW',
                        value = '(ee*MS*complexconjugate(CKM22)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(sWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + cWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVcxsGp2EW = Coupling(name = 'c_UVcxsGp2EW',
                        value = '-(ee*MC*complexconjugate(CKM22)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(sWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + cWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVtxsGp1EW = Coupling(name = 'c_UVtxsGp1EW',
                        value = '(ee*MS*complexconjugate(CKM32)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(sWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + tWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVtxsGp2EW = Coupling(name = 'c_UVtxsGp2EW',
                        value = '-(ee*MT*complexconjugate(CKM32)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(sWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + tWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVuxbGp1EW = Coupling(name = 'c_UVuxbGp1EW',
                        value = '(ee*MB*complexconjugate(CKM13)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(bWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + uWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVuxbGp2EW = Coupling(name = 'c_UVuxbGp2EW',
                        value = '-(ee*MU*complexconjugate(CKM13)*(-2*SWCoup_UV_EW*MT*MW**2 + MT*MW**2*sw*complexconjugate(bWcft_UV_EW_L) + sw*(-(WMass2_UV_EW*MT) + MW**2*(2*eCoup_UV_EW*MT + GpWcft_UV_EW*MT + 2*tMass_UV_EW + MT*uWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MT*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVcxbGp1EW = Coupling(name = 'c_UVcxbGp1EW',
                        value = '(ee*MB*complexconjugate(CKM23)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(bWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + cWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVcxbGp2EW = Coupling(name = 'c_UVcxbGp2EW',
                        value = '-(ee*MC*complexconjugate(CKM23)*(-2*SWCoup_UV_EW*MT*MW**2 + MT*MW**2*sw*complexconjugate(bWcft_UV_EW_L) + sw*(-(WMass2_UV_EW*MT) + MW**2*(2*eCoup_UV_EW*MT + GpWcft_UV_EW*MT + 2*tMass_UV_EW + MT*cWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MT*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVtxbGp1EW = Coupling(name = 'c_UVtxbGp1EW',
                        value = '(ee*MB*complexconjugate(CKM33)*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(bWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + tWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVtxbGp2EW = Coupling(name = 'c_UVtxbGp2EW',
                        value = '-(ee*complexconjugate(CKM33)*(-2*SWCoup_UV_EW*MT*MW**2 + MT*MW**2*sw*complexconjugate(bWcft_UV_EW_L) + sw*(-(WMass2_UV_EW*MT) + MW**2*(2*eCoup_UV_EW*MT + GpWcft_UV_EW*MT + 2*tMass_UV_EW + MT*tWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                        order = {'QED' : 3})


c_UVepveGm1EW = Coupling(name = 'c_UVepveGm1EW',
                         value = '-(ee*Me*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(veWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + eWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                         order = {'QED' : 3})


c_UVmpvmGm1EW = Coupling(name = 'c_UVmpvmGm1EW',
                         value = '-(ee*MM*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(vmWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + muWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                         order = {'QED' : 3})


c_UVttpvtGm1EW = Coupling(name = 'c_UVttpvtGm1EW',
                          value = '-(ee*MTA*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(vtWcft_UV_EW_L) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + tauWcft_UV_EW_R))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                          order = {'QED' : 3})


c_UVvexemGp1EW = Coupling(name = 'c_UVvexemGp1EW',
                          value = '(ee*Me*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(eWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + veWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                          order = {'QED' : 3})


c_UVvmxmmGp1EW = Coupling(name = 'c_UVvmxmmGp1EW',
                          value = '(ee*MM*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(muWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + vmWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                          order = {'QED' : 3})


c_UVvtxttmGp1EW = Coupling(name = 'c_UVvtxttmGp1EW',
                           value = '(ee*MTA*(-2*SWCoup_UV_EW*MW**2 + MW**2*sw*complexconjugate(tauWcft_UV_EW_R) + sw*(-WMass2_UV_EW + MW**2*(2*eCoup_UV_EW + GpWcft_UV_EW + vtWcft_UV_EW_L))))/(2.*cmath.sqrt(2)*MW**3*sw**2)',
                           order = {'QED' : 3})


# ============== #
# Mixed QCD-QED  #
# ============== #

UV_Htt = Coupling(name = 'UV_Htt',
                 value = '-((complex(0,1)*yt)/cmath.sqrt(2))*UV_yuk_t',
                 order = {'QED':1,'QCD':2})

UV_Hbb = Coupling(name = 'UV_Hbb',
                 value = '-((complex(0,1)*yb)/cmath.sqrt(2))*UV_yuk_b',
                 order = {'QED':1,'QCD':2})

# ============================== #
# Goldstone UV CT couplings      #
# ============================== #

GC_UV_1015 = Coupling(name = 'GC_UV_1015',
                      value = 'I133*UV_yuk_b',
                      order = {'QED':1,'QCD':2})

GC_UV_1021 = Coupling(name = 'GC_UV_1021',
                      value = '-I233*UV_yuk_t',
                      order = {'QED':1,'QCD':2})

GC_UV_1027 = Coupling(name = 'GC_UV_1027',
                      value = 'I333*UV_yuk_t',
                      order = {'QED':1,'QCD':2})

GC_UV_1030 = Coupling(name = 'GC_UV_1030',
                      value = '-I433*UV_yuk_b',
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
                      value = '(-I222)*(UV_yuk_c)',
                      order = {'QED':1,'QCD':2})

GC_UV_1019 = Coupling(name = 'GC_UV_1019',
                      value = '(-I223)*(UV_yuk_t)',
                      order = {'QED':1,'QCD':2})

GC_UV_1013 = Coupling(name = 'GC_UV_1013',
                      value = '(I131)*(UV_yuk_b)',
                      order = {'QED':1,'QCD':2})

GC_UV_1014 = Coupling(name = 'GC_UV_1014',
                      value = '(I132)*(UV_yuk_b)',
                      order = {'QED':1,'QCD':2})

GC_UV_1020 = Coupling(name = 'GC_UV_1020',
                      value = '(-I232)*(UV_yuk_c)',
                      order = {'QED':1,'QCD':2})

GC_UV_1028 = Coupling(name = 'GC_UV_1028',
                      value = '(-I413)*(UV_yuk_b)',
                      order = {'QED':1,'QCD':2})

GC_UV_1022 = Coupling(name = 'GC_UV_1022',
                      value = '(I321)*(UV_yuk_c)',
                      order = {'QED':1,'QCD':2})

GC_UV_1023 = Coupling(name = 'GC_UV_1023',
                      value = '(I322)*(UV_yuk_c)',
                      order = {'QED':1,'QCD':2})

GC_UV_1024 = Coupling(name = 'GC_UV_1024',
                      value = '(I323)*(UV_yuk_c)',
                      order = {'QED':1,'QCD':2})

GC_UV_1029 = Coupling(name = 'GC_UV_1029',
                      value = '(-I423)*(UV_yuk_b)',
                      order = {'QED':1,'QCD':2})

GC_UV_1025 = Coupling(name = 'GC_UV_1025',
                      value = '(I331)*(UV_yuk_t)',
                      order = {'QED':1,'QCD':2})

GC_UV_1026 = Coupling(name = 'GC_UV_1026',
                      value = '(I332)*(UV_yuk_t)',
                      order = {'QED':1,'QCD':2})

UV_GUULEW = Coupling(name = 'UV_GUULEW',
                     value = 'complex(0,1)/2.*G*(complexconjugate(uWcft_UV_EW_L)+uWcft_UV_EW_L)',
                     order = {'QED':2,'QCD':1})

UV_GUUREW = Coupling(name = 'UV_GUUREW',
                     value = 'complex(0,1)/2.*G*(complexconjugate(uWcft_UV_EW_R)+uWcft_UV_EW_R)',
                     order = {'QED':2,'QCD':1})

UV_GDDLEW = Coupling(name = 'UV_GDDLEW',
                     value = 'complex(0,1)/2.*G*(complexconjugate(dWcft_UV_EW_L)+dWcft_UV_EW_L)',
                     order = {'QED':2,'QCD':1})

UV_GDDREW = Coupling(name = 'UV_GDDREW',
                     value = 'complex(0,1)/2.*G*(complexconjugate(dWcft_UV_EW_R)+dWcft_UV_EW_R)',
                     order = {'QED':2,'QCD':1})

UV_GSSLEW = Coupling(name = 'UV_GSSLEW',
                     value = 'complex(0,1)/2.*G*(complexconjugate(sWcft_UV_EW_L)+sWcft_UV_EW_L)',
                     order = {'QED':2,'QCD':1})

UV_GSSREW = Coupling(name = 'UV_GSSREW',
                     value = 'complex(0,1)/2.*G*(complexconjugate(sWcft_UV_EW_R)+sWcft_UV_EW_R)',
                     order = {'QED':2,'QCD':1})

UV_GCCLEW = Coupling(name = 'UV_GCCLEW',
                     value = 'complex(0,1)/2.*G*(complexconjugate(cWcft_UV_EW_L)+cWcft_UV_EW_L)',
                     order = {'QED':2,'QCD':1})

UV_GCCREW = Coupling(name = 'UV_GCCREW',
                     value = 'complex(0,1)/2.*G*(complexconjugate(cWcft_UV_EW_R)+cWcft_UV_EW_R)',
                     order = {'QED':2,'QCD':1})

UV_GBBLEW = Coupling(name = 'UV_GBBLEW',
                     value = 'complex(0,1)/2.*G*(complexconjugate(bWcft_UV_EW_L)+bWcft_UV_EW_L)',
                     order = {'QED':2,'QCD':1})

UV_GBBREW = Coupling(name = 'UV_GBBREW',
                     value = 'complex(0,1)/2.*G*(complexconjugate(bWcft_UV_EW_R)+bWcft_UV_EW_R)',
                     order = {'QED':2,'QCD':1})

UV_GTTLEW = Coupling(name = 'UV_GTTLEW',
                     value = 'complex(0,1)/2.*G*(complexconjugate(tWcft_UV_EW_L)+tWcft_UV_EW_L)',
                     order = {'QED':2,'QCD':1})

UV_GTTREW = Coupling(name = 'UV_GTTREW',
                     value = 'complex(0,1)/2.*G*(complexconjugate(tWcft_UV_EW_R)+tWcft_UV_EW_R)',
                     order = {'QED':2,'QCD':1})




