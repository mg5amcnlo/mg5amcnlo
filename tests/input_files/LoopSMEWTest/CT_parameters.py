# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51

from object_library import all_CTparameters, CTParameter

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, arg, reglog

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

G_UVc = CTParameter(name = 'G_UVc',
                    type = 'real',
                    value = {-1:'((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF',
                    0:'cond(MC,0.0,-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*reglog(MC**2/MU_R**2))'},
                    texname = '\delta Gc')

G_UVb = CTParameter(name = 'G_UVb',
                    type = 'real',
                    value = {-1:'((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF',
                    0:'cond(MB,0.0,-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*reglog(MB**2/MU_R**2))'},
                    texname = '\delta Gb')

G_UVt = CTParameter(name = 'G_UVt',
                    type = 'real',
                    value = {-1:'((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF',
                    0:'cond(MT,0.0,-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*reglog(MT**2/MU_R**2))'},
                    texname = '\delta Gt')

GWcft_UV_c = CTParameter(name = 'GWcft_UV_c',
                         type = 'real',
                         value = {-1:'cond(MC,0.0,-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF)',
                         0:'cond(MC,0.0,((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*reglog(MC**2/MU_R**2))'
                         },
                         texname = '\delta G_{wfct\_c}')

GWcft_UV_b = CTParameter(name = 'GWcft_UV_b',
                         type = 'real',
                         value = {-1:'cond(MB,0.0,-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF)',
                         0:'cond(MB,0.0,((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*reglog(MB**2/MU_R**2))'
                         },
                         texname = '\delta G_{wfct\_b}')

GWcft_UV_t = CTParameter(name = 'GWcft_UV_t',
                         type = 'real ',
                         value = {-1:'cond(MT,0.0,-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF)',
                         0:'cond(MT,0.0,((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*reglog(MT**2/MU_R**2))'
                         },
                         texname = '\delta G_{wfct\_t}')

cWcft_UV = CTParameter(name = 'cWcft_UV',
                       type = 'real',
                       value = {-1:'cond(MC,0.0,-((G**2)/(2.0*16.0*cmath.pi**2))*3.0*CF)',
                       0:'cond(MC,0.0,-((G**2)/(2.0*16.0*cmath.pi**2))*CF*(4.0-3.0*reglog(MC**2/MU_R**2)))'
                       },
                       texname = '\delta Z_c')

bWcft_UV = CTParameter(name = 'bWcft_UV',
                       type = 'real',
                       value = {-1:'cond(MB,0.0,-((G**2)/(2.0*16.0*cmath.pi**2))*3.0*CF)',
                       0:'cond(MB,0.0,-((G**2)/(2.0*16.0*cmath.pi**2))*CF*(4.0-3.0*reglog(MB**2/MU_R**2)))'
                       },
                       texname = '\delta Z_b')

tWcft_UV = CTParameter(name = 'tWcft_UV',
                       type = 'real',
                       value = {-1:'cond(MT,0.0,-((G**2)/(2.0*16.0*cmath.pi**2))*3.0*CF)',
                       0:'cond(MT,0.0,-((G**2)/(2.0*16.0*cmath.pi**2))*CF*(4.0-3.0*reglog(MT**2/MU_R**2)))'
                       },
                       texname = '\delta Z_t')

bMass_UV = CTParameter(name = 'bMass_UV',
                       type = 'complex',
                       value = {-1:'cond(MB,0.0,complex(0,1)*((G**2)/(16.0*cmath.pi**2))*(3.0*CF)*MB)',
                       0:'cond(MB,0.0,complex(0,1)*((G**2)/(16.0*cmath.pi**2))*CF*(4.0-3.0*reglog(MB**2/MU_R**2))*MB)'
                       },
                       texname = '\delta m_b')

cMass_UV = CTParameter(name = 'cMass_UV',
                       type = 'complex',
                       value = {-1:'cond(MC,0.0,complex(0,1)*((G**2)/(16.0*cmath.pi**2))*(3.0*CF)*MC)',
                       0:'cond(MC,0.0,complex(0,1)*((G**2)/(16.0*cmath.pi**2))*CF*(4.0-3.0*reglog(MC**2/MU_R**2))*MC)'
                       },
                       texname = '\delta m_c')

tMass_UV = CTParameter(name = 'tMass_UV',
                       type = 'complex',
                       value = {-1:'cond(MT,0.0,complex(0,1)*((G**2)/(16.0*cmath.pi**2))*3.0*CF*MT)',
                       0:'cond(MT,0.0,complex(0,1)*((G**2)/(16.0*cmath.pi**2))*CF*(4.0-3.0*reglog(MT**2/MU_R**2))*MT)'
                       },
                       texname = '\delta m_t')



# ================================== #
# QED                                #
# Generate automatically by WriteUFO #
# ================================== #
# in order to improve the load speed, we use fake values for the following CT parameters
HiggsTadpole_UV_EW = CTParameter(name = 'HiggsTadpole_UV_EW',
                                type = 'complex',
                                value = {-1:'1.23',
                                        0:'1.23'},
                                texname = '\delta ht^{EW}')


tMass_UV_EW = CTParameter(name = 'tMass_UV_EW',
                          type = 'complex',
                          value = {-1:'1.24',
                                   0:'1.24'},
                          texname = '\delta m_t^{EW}')

HMass2_UV_EW = CTParameter(name = 'HMass2_UV_EW',
                           type = 'complex',
                           value = {-1:'1.25',
                                    0:'1.25'},
                           texname = '\delta m2_H^{EW}')

WMass2_UV_EW = CTParameter(name = 'WMass2_UV_EW',
                           type = 'complex',
                           value = {-1:'1.26',
                                    0:'1.26'},
                           texname = '\delta m2_W^{EW}')

ZMass2_UV_EW = CTParameter(name = 'ZMass2_UV_EW',
                          type = 'complex',
                          value = {-1:'1.27',
                                   0:'1.27'},
                          texname = '\delta m2_Z^{EW}')


tWcft_UV_EW_R = CTParameter(name = 'tWcft_UV_EW_R',
                            type = 'complex',
                            value = {-1:'1.28',
                                     0:'1.28'},
                            texname = '\delta ZR_t^{EW}')




cWcft_UV_EW_R = CTParameter(name = 'cWcft_UV_EW_R',
                            type = 'complex',
                            value = {-1:'1.29',
                                     0:'1.29'},
                            texname = '\delta ZR_c^{EW}')


uWcft_UV_EW_R = CTParameter(name = 'uWcft_UV_EW_R',
                           type = 'complex',
                           value = {-1:'1.29',
                                    0:'1.29'},
                           texname = '\delta ZR_u^{EW}')



bWcft_UV_EW_R = CTParameter(name = 'bWcft_UV_EW_R',
                            type = 'complex',
                            value = {-1:'1.31',
                                     0:'1.31'},
                            texname = '\delta ZR_b^{EW}')


sWcft_UV_EW_R = CTParameter(name = 'sWcft_UV_EW_R',
                            type = 'complex',
                            value = {-1:'1.32',
                                     0:'1.32'},
                            texname = '\delta ZR_s^{EW}')


dWcft_UV_EW_R = CTParameter(name = 'dWcft_UV_EW_R',
                            type = 'complex',
                            value = {-1:'1.33',
                                     0:'1.33'},
                            texname = '\delta ZR_d^{EW}')


tauWcft_UV_EW_R = CTParameter(name = 'tauWcft_UV_EW_R',
                              type = 'complex',
                              value = {-1:'1.34',
                                       0:'1.34'},
                              texname = '\delta ZR_tau^{EW}')


muWcft_UV_EW_R = CTParameter(name = 'muWcft_UV_EW_R',
                             type = 'complex',
                             value = {-1:'1.34',
                                      0:'1.34'},
                             texname = '\delta ZR_mu^{EW}')


eWcft_UV_EW_R = CTParameter(name = 'eWcft_UV_EW_R',
                            type = 'complex',
                            value = {-1:'1.34',
                                     0:'1.34'},
                            texname = '\delta ZR_e^{EW}')

tWcft_UV_EW_L = CTParameter(name = 'tWcft_UV_EW_L',
                            type = 'complex',
                            value = {-1:'2.31',
                                     0:'2.32'},
                            texname = '\delta ZL_t^{EW}')



cWcft_UV_EW_L = CTParameter(name = 'cWcft_UV_EW_L',
                            type = 'complex',
                            value = {-1:'2.33',
                                     0:'2.34'},
                            texname = '\delta ZL_c^{EW}')


uWcft_UV_EW_L = CTParameter(name = 'uWcft_UV_EW_L',
                            type = 'complex',
                            value = {-1:'2.33',
                                     0:'2.34'},
                            texname = '\delta ZL_u^{EW}')



bWcft_UV_EW_L = CTParameter(name = 'bWcft_UV_EW_L',
                            type = 'complex',
                            value = {-1:'2.35',
                                     0:'2.36'},
                            texname = '\delta ZL_b^{EW}')


sWcft_UV_EW_L = CTParameter(name = 'sWcft_UV_EW_L',
                            type = 'complex',
                            value = {-1:'2.37',
                                     0:'2.38'},
                            texname = '\delta ZL_s^{EW}')


dWcft_UV_EW_L = CTParameter(name = 'dWcft_UV_EW_L',
                            type = 'complex',
                            value = {-1:'2.37',
                                     0:'2.38'},
                            texname = '\delta ZL_d^{EW}')


tauWcft_UV_EW_L = CTParameter(name = 'tauWcft_UV_EW_L',
                              type = 'complex',
                              value = {-1:'2.39',
                                       0:'2.41'},
                              texname = '\delta ZL_tau^{EW}')


muWcft_UV_EW_L = CTParameter(name = 'muWcft_UV_EW_L',
                             type = 'complex',
                             value = {-1:'2.39',
                                      0:'2.41'},
                             texname = '\delta ZL_mu^{EW}')


eWcft_UV_EW_L = CTParameter(name = 'eWcft_UV_EW_L',
                            type = 'complex',
                            value = {-1:'2.39',
                                     0:'2.41'},
                            texname = '\delta ZL_e^{EW}')


vtWcft_UV_EW_L = CTParameter(name = 'vtWcft_UV_EW_L',
                             type = 'complex',
                             value = {-1:'2.51',
                                      0:'2.52'},
                             texname = '\delta ZL_vt^{EW}')


vmWcft_UV_EW_L = CTParameter(name = 'vmWcft_UV_EW_L',
                             type = 'complex',
                             value = {-1:'2.51',
                                      0:'2.52'},
                             texname = '\delta ZL_vm^{EW}')


veWcft_UV_EW_L = CTParameter(name = 'veWcft_UV_EW_L',
                             type = 'complex',
                             value = {-1:'2.51',
                                      0:'2.52'},
                             texname = '\delta ZL_ve^{EW}')

HWcft_UV_EW = CTParameter(name = 'HWcft_UV_EW',
                          type = 'complex',
                          value = {-1:'2.53',
                                   0:'2.53'},
                          texname = '\delta Z_{H}^{EW}')

G0Wcft_UV_EW = CTParameter(name = 'G0Wcft_UV_EW',
                           type = 'complex',
                           value = {-1:'2.54',
                                    0:'2.55'},
                           texname = '\delta Z_{G0}^{EW}')

GpWcft_UV_EW = CTParameter(name = 'GpWcft_UV_EW',
                           type = 'complex',
                           value = {-1:'2.57',
                                    0:'2.57'},
                           texname = '\delta Z_{Gp}^{EW}')


WWcft_UV_EW = CTParameter(name = 'WWcft_UV_EW',
                          type = 'complex',
                          value = {-1:'3.37',
                                   0:'3.35'},
                          texname = '\delta Z_{W}^{EW}')


ZZWcft_UV_EW = CTParameter(name = 'ZZWcft_UV_EW',
                           type = 'complex',
                           value = {-1:'3.13',
                                    0:'3.153'},
                           texname = '\delta Z_{ZZ}^{EW}')


AZWcft_UV_EW = CTParameter(name = 'AZWcft_UV_EW',
                           type = 'complex',
                           value = {-1:'3.31',
                                    0:'3.321'},
                           texname = '\delta Z_{AZ}^{EW}')

ZAWcft_UV_EW = CTParameter(name = 'ZAWcft_UV_EW',
                           type = 'complex',
                           value = {-1:'3.73',
                                     0:'3.73'},
                           texname = '\delta Z_{ZA}^{EW}')

AAWcft_UV_EW = CTParameter(name = 'AAWcft_UV_EW',
                           type = 'complex',
                           value = {-1:'3.53',
                                     0:'3.532'},
                           texname = '\delta Z_{AA}^{EW}')

eCoup_UV_EW = CTParameter(name = 'eCoup_UV_EW',
                          type = 'complex',
                          value = {-1:'3.1212',
                                   0:'2.1313'},
                          texname = '\delta e')


SWCoup_UV_EW = CTParameter(name = 'SWCoup_UV_EW',
                           type = 'complex',
                           value = {-1:'-2.131',
                                    0:'2.341'},
                           texname = '\delta SW')


CWCoup_UV_EW = CTParameter(name = 'CWCoup_UV_EW',
                           type = 'complex',
                           value = {-1:'2.131',
                                    0:'-2.341'},
                           texname = '\delta CW')

# ============== #
# Mixed QCD-QED  #
# ============== #

UV_yuk_c = CTParameter(name = 'UV_yuk_c',
                       type = 'real',
                       value = {-1:'-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*3.0*CF*2.0',
                       0:'cond(MC,0.0,-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*CF*(-3.0*reglog(MC**2/MU_R**2)+4.0)*2.0)'
                       },
                       texname = '\delta y_c')

UV_yuk_b = CTParameter(name = 'UV_yuk_b',
                       type = 'real',
                       value = {-1:'-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*3.0*CF*2.0',
                       0:'cond(MB,0.0,-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*CF*(-3.0*reglog(MB**2/MU_R**2)+4.0)*2.0)'
                       },
                       texname = '\delta y_b')

UV_yuk_t = CTParameter(name = 'UV_yuk_t',
                       type = 'real',
                       value = {-1:'-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*3.0*CF*2.0',
                       0:'cond(MT,0.0,-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*CF*(-3.0*reglog(MT**2/MU_R**2)+4.0)*2.0)'
                       },
                       texname = '\delta y_t')

