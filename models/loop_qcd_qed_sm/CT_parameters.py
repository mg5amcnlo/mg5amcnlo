# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51

from object_library import all_CTparameters, CTParameter

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, arg

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

# ============== #
# Pure QED       #
# ============== #

R2SS = CTParameter(name = 'R2SS',
	type = 'real',
	value = {0:'ee**2/(16.0*cmath.pi**2*sw**2)'},
	texname = 'R2SS')

R2VV = CTParameter(name = 'R2VV',
                   type = 'real',
                   value = {0:'ee**2/cmath.pi**2'},
                   texname = 'R2VV')

R2SFF = CTParameter(name = 'R2SFF',
                    type = 'real',
                    value = {0:'ee**3/cmath.pi**2'},
                    texname = 'R2SFF')

R24S = CTParameter(name = 'R24S',
                     type = 'real',
                     value = {0:'ee**4/cmath.pi**2'},
                     texname = 'R24S')

# ============== #
# Mixed QED-QCD  #
# ============== #

R2GQQ2 = CTParameter(name = 'R2GQQ2',
                     type = 'real',
                     value = {0:'-G*ee**2/cmath.pi**2'},
                     texname = 'R2GQQ2')

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
                       type = 'complex',
                       value = {-1:'complex(0,1)*((G**2)/(16.0*cmath.pi**2))*(3.0*CF)*MB',
                                 0:'complex(0,1)*((G**2)/(16.0*cmath.pi**2))*CF*(4.0+3.0*cmath.log(MU_R**2/MB**2))*MB'
                               },
                       texname = '\delta m_b')

tMass_UV = CTParameter(name = 'tMass_UV',
                       type = 'complex',
                       value = {-1:'complex(0,1)*((G**2)/(16.0*cmath.pi**2))*3.0*CF*MT',
                                 0:'complex(0,1)*((G**2)/(16.0*cmath.pi**2))*CF*(4.0+3.0*cmath.log(MU_R**2/MT**2))*MT'
                               },
                       texname = '\delta m_t')

# ============== #
# QED            #
# ============== #

tMass_UV_QED = CTParameter(name = "tMass_UV_QED",
			  type = "complex",
			  value = {-1:"-(aEW*MT*(9*cw**2*MT**2 + 3*MW**2 + 6*cw**2*MW**2 + 24*MW**2*sw**2 - \
                                    32*cw**2*MW**2*sw**2 -32*MW**2*sw**4))/(96.*cw**2*MW**2*cmath.pi*sw**2)",
                                   0:"-(aEW*(-18*cw**2*MH**2*MT**4 + 144*cw**2*MT**6 +\
                                 36*MT**4*MW**2 + 18*cw**2*MT**4*MW**2 - 36*cw**2*MT\
                                **2*MW**4 - 18*cw**2*MT**4*MZ**2 - 18*MT**2*MW**2*MZ\
                                **2 + 192*MT**4*MW**2*sw**2 - 256*cw**2*MT**4*MW\
                                **2*sw**2 + 48*MT**2*MW**2*MZ**2*sw**2 - 256*MT**\
                                4*MW**2*sw**4 - 64*MT**2*MW**2*MZ**2*sw**4 - 9*cw**\
                                2*MH*MT**2*(-MH**2 + 4*MT**2)**1.5*arg(-MH + cmath.sqrt\
                                (-MH**2 + 4*MT**2)*complex(0,1)) + 9*cw**2*MH*MT\
                                **2*(-MH**2 + 4*MT**2)**1.5*arg(MH + cmath.sqrt\
                                (-MH**2 + 4*MT**2)*complex(0,1)) - 9*cw**2*MH*MT**\
                                2*(-MH**2 + 4*MT**2)**1.5*arg(MH**2 - 2*MT**2 +\
                                 MH*cmath.sqrt(-MH**2 + 4*MT**2)*complex(0,1)) + 9*cw\
                                **2*MH*MT**2*(-MH**2 + 4*MT**2)**1.5*arg(-MH**2 +\
                                 2*MT**2 + MH*cmath.sqrt(-MH**2 + 4*MT**2)*complex(\
                                0,1)) - 18*MT**2*MW**2*MZ*cmath.sqrt(4*MT**2 - MZ**\
                                2)*arg(-MZ + cmath.sqrt(4*MT**2 - MZ**2)*complex(\
                                0,1)) + 9*cw**2*MT**2*MZ**3*cmath.sqrt(4*MT**2 - MZ\
                                **2)*arg(-MZ + cmath.sqrt(4*MT**2 - MZ**2)*complex\
                                (0,1)) + 9*MW**2*MZ**3*cmath.sqrt(4*MT**2 - MZ**2\
                                )*arg(-MZ + cmath.sqrt(4*MT**2 - MZ**2)*complex(0,1\
                                )) - 48*MT**2*MW**2*MZ*cmath.sqrt(4*MT**2 - MZ**2\
                                )*sw**2*arg(-MZ + cmath.sqrt(4*MT**2 - MZ**2)*complex\
                                (0,1)) - 24*MW**2*MZ**3*cmath.sqrt(4*MT**2 - MZ\
                                **2)*sw**2*arg(-MZ + cmath.sqrt(4*MT**2 - MZ**2)\
                                *complex(0,1)) + 64*MT**2*MW**2*MZ*cmath.sqrt(4*MT**\
                                2 - MZ**2)*sw**4*arg(-MZ + cmath.sqrt(4*MT**2 - MZ\
                                **2)*complex(0,1)) + 32*MW**2*MZ**3*cmath.sqrt(4*MT\
                                **2 - MZ**2)*sw**4*arg(-MZ + cmath.sqrt(4*MT**2 -\
                                 MZ**2)*complex(0,1)) + 18*MT**2*MW**2*MZ*cmath.sqrt\
                                (4*MT**2 - MZ**2)*arg(MZ + cmath.sqrt(4*MT**2 -\
                                 MZ**2)*complex(0,1)) - 9*cw**2*MT**2*MZ**3*cmath.sqrt\
                                (4*MT**2 - MZ**2)*arg(MZ + cmath.sqrt(4*MT**2 -\
                                 MZ**2)*complex(0,1)) - 9*MW**2*MZ**3*cmath.sqrt(\
                                4*MT**2 - MZ**2)*arg(MZ + cmath.sqrt(4*MT**2 - MZ**\
                                2)*complex(0,1)) + 48*MT**2*MW**2*MZ*cmath.sqrt(4*MT\
                                **2 - MZ**2)*sw**2*arg(MZ + cmath.sqrt(4*MT**2 -\
                                 MZ**2)*complex(0,1)) + 24*MW**2*MZ**3*cmath.sqrt(\
                                4*MT**2 - MZ**2)*sw**2*arg(MZ + cmath.sqrt(4*MT**2 \
                                - MZ**2)*complex(0,1)) - 64*MT**2*MW**2*MZ*cmath.sqrt\
                                (4*MT**2 - MZ**2)*sw**4*arg(MZ + cmath.sqrt(4*MT\
                                **2 - MZ**2)*complex(0,1)) - 32*MW**2*MZ**3*cmath.sqrt\
                                (4*MT**2 - MZ**2)*sw**4*arg(MZ + cmath.sqrt(\
                                4*MT**2 - MZ**2)*complex(0,1)) + 18*MT**2*MW**\
                                2*MZ*cmath.sqrt(4*MT**2 - MZ**2)*arg(2*MT**2 - MZ**2 +\
                                 MZ*cmath.sqrt(4*MT**2 - MZ**2)*complex(0,1)) - 9*cw\
                                **2*MT**2*MZ**3*cmath.sqrt(4*MT**2 - MZ**2)*arg(\
                                2*MT**2 - MZ**2 + MZ*cmath.sqrt(4*MT**2 - MZ**2)\
                                *complex(0,1)) - 9*MW**2*MZ**3*cmath.sqrt(4*MT**2 -\
                                 MZ**2)*arg(2*MT**2 - MZ**2 + MZ*cmath.sqrt(4*MT**2 \
                                - MZ**2)*complex(0,1)) + 48*MT**2*MW**2*MZ*cmath.sqrt\
                                (4*MT**2 - MZ**2)*sw**2*arg(2*MT**2 - MZ**2 +\
                                 MZ*cmath.sqrt(4*MT**2 - MZ**2)*complex(0,1)) + 24*MW\
                                **2*MZ**3*cmath.sqrt(4*MT**2 - MZ**2)*sw**2*arg(\
                                2*MT**2 - MZ**2 + MZ*cmath.sqrt(4*MT**2 - MZ**2)\
                                *complex(0,1)) - 64*MT**2*MW**2*MZ*cmath.sqrt(4*MT**\
                                2 - MZ**2)*sw**4*arg(2*MT**2 - MZ**2 + MZ*cmath.sqrt\
                                (4*MT**2 - MZ**2)*complex(0,1)) - 32*MW**2*MZ**\
                                3*cmath.sqrt(4*MT**2 - MZ**2)*sw**4*arg(2*MT**2 - MZ\
                                **2 + MZ*cmath.sqrt(4*MT**2 - MZ**2)*complex(0,1)\
                                ) - 18*MT**2*MW**2*MZ*cmath.sqrt(4*MT**2 - MZ**2)\
                                *arg(-2*MT**2 + MZ**2 + MZ*cmath.sqrt(4*MT**2 - MZ**\
                                2)*complex(0,1)) + 9*cw**2*MT**2*MZ**3*cmath.sqrt\
                                (4*MT**2 - MZ**2)*arg(-2*MT**2 + MZ**2 + MZ*cmath.sqrt\
                                (4*MT**2 - MZ**2)*complex(0,1)) + 9*MW**2*MZ**\
                                3*cmath.sqrt(4*MT**2 - MZ**2)*arg(-2*MT**2 + MZ**2 \
                                + MZ*cmath.sqrt(4*MT**2 - MZ**2)*complex(0,1)) -\
                                 48*MT**2*MW**2*MZ*cmath.sqrt(4*MT**2 - MZ**2)*sw**\
                                2*arg(-2*MT**2 + MZ**2 + MZ*cmath.sqrt(4*MT**2 - MZ\
                                **2)*complex(0,1)) - 24*MW**2*MZ**3*cmath.sqrt(4*MT\
                                **2 - MZ**2)*sw**2*arg(-2*MT**2 + MZ**2 + MZ*cmath.sqrt\
                                (4*MT**2 - MZ**2)*complex(0,1)) + 64*MT**2*MW\
                                **2*MZ*cmath.sqrt(4*MT**2 - MZ**2)*sw**4*arg(-2*MT\
                                **2 + MZ**2 + MZ*cmath.sqrt(4*MT**2 - MZ**2)*complex\
                                (0,1)) + 32*MW**2*MZ**3*cmath.sqrt(4*MT**2 - MZ\
                                **2)*sw**4*arg(-2*MT**2 + MZ**2 + MZ*cmath.sqrt(\
                                4*MT**2 - MZ**2)*complex(0,1)) + 252*cw**2*MT**6*reglog\
                                (2) - 18*cw**2*MH**2*MT**4*reglog(4) - 18*cw**\
                                2*MT**4*MZ**2*reglog(4) - 18*MT**2*MW**2*MZ**2*reglog\
                                (4) + 48*MT**2*MW**2*MZ**2*sw**2*reglog(4) - 64*MT\
                                **2*MW**2*MZ**2*sw**4*reglog(4) + 18*cw**2*MT**\
                                4*MW**2*reglog(16) - 64*cw**2*MT**4*MW**2*sw**2*reglog\
                                (64) + 48*MT**4*MW**2*sw**2*reglog(1024) - 64*MT\
                                **4*MW**2*sw**4*reglog(1024) - 36*cw**2*MH**2*MT\
                                **4*reglog(MH) - 36*cw**2*MH**2*MT**4*reglog(MT) +\
                                 216*cw**2*MT**6*reglog(MT) + 108*MT**4*MW**2*reglog\
                                (MT) - 36*cw**2*MT**4*MZ**2*reglog(MT) - 36*MT**\
                                2*MW**2*MZ**2*reglog(MT) + 96*MT**4*MW**2*sw**2*reglog\
                                (MT) + 384*cw**2*MT**4*MW**2*sw**2*reglog(MT) \
                                + 96*MT**2*MW**2*MZ**2*sw**2*reglog(MT) - 128*MT**\
                                4*MW**2*sw**4*reglog(MT) - 128*MT**2*MW**2*MZ**2*sw\
                                **4*reglog(MT) + 9*cw**2*MH**4*MT**2*reglog(4*MH**\
                                2*MT**2) - 36*cw**2*MH**2*MT**4*reglog(4*MH**2*MT\
                                **2) - 9*cw**2*MH**4*MT**2*reglog(4*MT**4) + 54*cw\
                                **2*MH**2*MT**4*reglog(4*MT**4) - 72*cw**2*MT**\
                                6*reglog(4*MT**4) - 36*MT**4*MW**2*reglog(4*MT**4) +\
                                 18*cw**2*MT**4*MZ**2*reglog(4*MT**4) + 36*MT**2*MW\
                                **2*MZ**2*reglog(4*MT**4) - 9*cw**2*MT**2*MZ**4*reglog\
                                (4*MT**4) - 9*MW**2*MZ**4*reglog(4*MT**4) - 96*MT\
                                **4*MW**2*sw**2*reglog(4*MT**4) + 24*MW**2*MZ**\
                                4*sw**2*reglog(4*MT**4) + 128*MT**4*MW**2*sw**4*reglog\
                                (4*MT**4) - 32*MW**2*MZ**4*sw**4*reglog(4*MT**\
                                4) - 18*cw**2*MT**6*reglog(MT - MW) + 54*cw**2*MT**\
                                2*MW**4*reglog(MT - MW) - 36*cw**2*MW**6*reglog(\
                                MT - MW) - 72*cw**2*MT**4*MW**2*reglog(MW) - 108*cw\
                                **2*MT**2*MW**4*reglog(MW) + 72*cw**2*MW**6*reglog\
                                (MW) - 18*cw**2*MT**6*reglog(MT + MW) + 54*cw**2*MT\
                                **2*MW**4*reglog(MT + MW) - 36*cw**2*MW**6*reglog\
                                (MT + MW) - 36*cw**2*MT**4*MZ**2*reglog(MZ) - 36*MT\
                                **2*MW**2*MZ**2*reglog(MZ) + 96*MT**2*MW**2*MZ**\
                                2*sw**2*reglog(MZ) - 128*MT**2*MW**2*MZ**2*sw**4*reglog\
                                (MZ) - 18*MT**2*MW**2*MZ**2*reglog(4*MT**2*MZ**\
                                2) + 9*cw**2*MT**2*MZ**4*reglog(4*MT**2*MZ**2) +\
                                 9*MW**2*MZ**4*reglog(4*MT**2*MZ**2) - 48*MT**2*MW**\
                                2*MZ**2*sw**2*reglog(4*MT**2*MZ**2) - 24*MW**2*MZ\
                                **4*sw**2*reglog(4*MT**2*MZ**2) + 64*MT**2*MW**2*MZ\
                                **2*sw**4*reglog(4*MT**2*MZ**2) + 32*MW**2*MZ**4*sw\
                                **4*reglog(4*MT**2*MZ**2) + 54*cw**2*MT**6*reglog\
                                (cmath.pi) + 36*cw**2*MT**4*MW**2*reglog(cmath.pi\
                                ) + 144*MT**4*MW**2*sw**2*reglog(cmath.pi) - 192*cw\
                                **2*MT**4*MW**2*sw**2*reglog(cmath.pi) - 192*MT**\
                                4*MW**2*sw**4*reglog(cmath.pi) - 54*cw**2*MT**6*reglog\
                                (4*cmath.pi) - 18*MT**4*MW**2*reglog(4*cmath.pi\
                                ) - 36*cw**2*MT**4*MW**2*reglog(4*cmath.pi) - 144*MT\
                                **4*MW**2*sw**2*reglog(4*cmath.pi) + 192*cw**2*MT\
                                **4*MW**2*sw**2*reglog(4*cmath.pi) + 192*MT**4*MW\
                                **2*sw**4*reglog(4*cmath.pi) + 18*MT**4*MW**2*reglog\
                                (64*cmath.pi) + 108*cw**2*MT**6*reglog(MU_R) +\
                                 36*MT**4*MW**2*reglog(MU_R) + 72*cw**2*MT**4*MW**\
                                2*reglog(MU_R) + 288*MT**4*MW**2*sw**2*reglog(MU_R)\
                                 - 384*cw**2*MT**4*MW**2*sw**2*reglog(MU_R) - 384*MT\
                                **4*MW**2*sw**4*reglog(MU_R)))/(576.*cw**2*MT**\
                                3*MW**2*cmath.pi*sw**2)"
                                   },
                          texname = '\delta m_t^{QED}')

# ============== #
# Mixed QCD-QED  #
# ============== #

UV_yuk_b = CTParameter(name = 'UV_yuk_b',
                       type = 'real',
                       value = {-1:'-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*3.0*CF*2.0',
                                 0:'-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*CF*(3.0*cmath.log(MU_R**2/MB**2)+4.0)*2.0'
                               },
                       texname = '\delta y_b')

UV_yuk_t = CTParameter(name = 'UV_yuk_t',
                       type = 'real',
                       value = {-1:'-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*3.0*CF*2.0',
                                 0:'-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*CF*(3.0*cmath.log(MU_R**2/MT**2)+4.0)*2.0'
                               },
                       texname = '\delta y_t')
