# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51

from object_library import all_CTparameters, CTParameter

from function_library import complexconjugate, re, im, csc, sec, acsc, asec

################
# R2 vertices  #
################

# ========= #
# Pure QCD  #
# ========= #

RGR2 = CTParameter(name = 'RGR2',
              type = 'real',
              value = {0:'-(3.0/2.0)*G**4/(96.0*cmath.pi**2)'},
              texname = 'RGR2')

# ============== #
# Mixed QCD-QED  #
# ============== #

R2MixedFactor = CTParameter(name = 'R2MixedFactor',
              type = 'real',
              value = {0:'-(G**2*(1.0+lhv)*(Ncol**2-1.0))/(2.0*Ncol*16.0*cmath.pi**2)'},
              texname = 'R2MixedFactor')

################
# UV vertices  #
################

# ========= #
# Pure QCD  #
# ========= #

G_UVg = CTParameter(name = 'G_UVg',
                    type = 'real',
                    value = {-1:'-((G**2)/(2.0*48.0*cmath.pi**2))*11.0*CA'},
                    texname = '\delta Gg')

G_UVq = CTParameter(name = 'G_UVq',
                    type = 'real',
                    value = {-1:'((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF'},
                    texname = '\delta Gq')

G_UVb = CTParameter(name = 'G_UVb',
                    type = 'real',
                    value = {-1:'((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF',
                              0:'((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*cmath.log(MU_R**2/MB**2)'},
                    texname = '\delta Gb')

G_UVt = CTParameter(name = 'G_UVt',
                    type = 'real',
                    value = {-1:'((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF',
                              0:'((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*cmath.log(MU_R**2/MT**2)'},
                    texname = '\delta Gt')

GWcft_UV_b = CTParameter(name = 'GWcft_UV_b',
                         type = 'real',
                         value = {-1:'-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF',
                                   0:'-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*cmath.log(MU_R**2/MB**2)'
                                 },
                         texname = '\delta G_{wfct\_b}')

GWcft_UV_t = CTParameter(name = 'GWcft_UV_t',
                         type = 'real',
                         value = {-1:'-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF',
                                   0:'-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*cmath.log(MU_R**2/MT**2)'
                                 },
                         texname = '\delta G_{wfct\_t}')

bWcft_UV = CTParameter(name = 'bWcft_UV',
                       type = 'real',
                       value = {-1:'-((G**2)/(2.0*16.0*cmath.pi**2))*3.0*CF',
                                 0:'-((G**2)/(2.0*16.0*cmath.pi**2))*CF*(4.0+3.0*cmath.log(MU_R**2/MB**2))'
                               },
                       texname = '\delta Z_b')

tWcft_UV = CTParameter(name = 'tWcft_UV',
                       type = 'real',
                       value = {-1:'-((G**2)/(2.0*16.0*cmath.pi**2))*3.0*CF',
                                 0:'-((G**2)/(2.0*16.0*cmath.pi**2))*CF*(4.0+3.0*cmath.log(MU_R**2/MT**2))'
                               },
                       texname = '\delta Z_t')

bMass_UV = CTParameter(name = 'bMass_UV',
                       type = 'real',
                       value = {-1:'complex(0,1)*((G**2)/(16.0*cmath.pi**2))*(3.0*CF)*MB',
                                 0:'complex(0,1)*((G**2)/(16.0*cmath.pi**2))*CF*(4.0+3.0*cmath.log(MU_R**2/MB**2))*MB'
                               },
                       texname = '\delta m_b')

tMass_UV = CTParameter(name = 'tMass_UV',
                       type = 'real',
                       value = {-1:'complex(0,1)*((G**2)/(16.0*cmath.pi**2))*3.0*CF*MT',
                                 0:'complex(0,1)*((G**2)/(16.0*cmath.pi**2))*CF*(4.0+3.0*cmath.log(MU_R**2/MT**2))*MT'
                               },
                       texname = '\delta m_t')

# ============== #
# Mixed QCD-QED  #
# ============== #

UV_yuk_b = CTParameter(name = 'UV_yuk_b',
                       type = 'real',
                       value = {-1:'-((G**2)/(16.0*cmath.pi**2))*3.0*CF*2.0',
                                 0:'-((G**2)/(16.0*cmath.pi**2))*3.0*CF*(3.0*cmath.log(MU_R**2/MB**2)+4.0)*2.0'
                               },
                       texname = '\delta y_b')

UV_yuk_t = CTParameter(name = 'UV_yuk_t',
                       type = 'real',
                       value = {-1:'-((G**2)/(16.0*cmath.pi**2))*3.0*CF*2.0',
                                 0:'-((G**2)/(16.0*cmath.pi**2))*3.0*CF*(3.0*cmath.log(MU_R**2/MT**2)+4.0)*2.0'
                               },
                       texname = '\delta y_t')
