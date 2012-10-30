# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51


from object_library import all_vertices, all_CTvertices, Vertex, CTVertex
import particles as P
import CT_couplings as C
import lorentz as L

################
# R2 vertices  #
################

# ========= #
# Pure QCD  #
# ========= #

# ggg R2
V_R23G = CTVertex(name = 'V_R23G',
              particles = [ P.G, P.G, P.G ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVV1 ],
              loop_particles = [ [[P.u], [P.d], [P.c], [P.s], [P.b], [P.t]],
                               [[P.G]] ],
              couplings = {(0,0,0):C.R2_3Gq, (0,0,1):C.R2_3Gg},
              type = 'R2' )

#=============================================================================================
#  4-gluon R2 vertex
#=============================================================================================

# Keep in mind that Delta8(a,b) is 1/2 Tr(a,b)
# EDIT HSS
# I am still puzzled by the normalization factors.
# END EDIT HSS

V_R24G = CTVertex(name = 'V_R24G',
              particles = [ P.G, P.G, P.G,  P.G ],
              color = [ 'Tr(1,2)*Tr(3,4)' , 'Tr(1,3)*Tr(2,4)' , 'Tr(1,4)*Tr(2,3)', \
                        'd(-1,1,2)*d(-1,3,4)' , 'd(-1,1,3)*d(-1,2,4)' , 'd(-1,1,4)*d(-1,2,3)'],
              lorentz = [  L.R2_4G_1234, L.R2_4G_1324, L.R2_4G_1423 ],
              loop_particles = [ [[P.G]], [[P.u],[P.d],[P.c],[P.s],[P.b],[P.t]] ],
              couplings = {(0,0,0):C.GC_4GR2_Gluon_delta5,(0,1,0):C.GC_4GR2_Gluon_delta7,(0,2,0):C.GC_4GR2_Gluon_delta7, \
                           (1,0,0):C.GC_4GR2_Gluon_delta7,(1,1,0):C.GC_4GR2_Gluon_delta5,(1,2,0):C.GC_4GR2_Gluon_delta7, \
                           (2,0,0):C.GC_4GR2_Gluon_delta7,(2,1,0):C.GC_4GR2_Gluon_delta7,(2,2,0):C.GC_4GR2_Gluon_delta5, \
                           (3,0,0):C.GC_4GR2_4Struct,(3,1,0):C.GC_4GR2_2Struct,(3,2,0):C.GC_4GR2_2Struct, \
                           (4,0,0):C.GC_4GR2_2Struct,(4,1,0):C.GC_4GR2_4Struct,(4,2,0):C.GC_4GR2_2Struct, \
                           (5,0,0):C.GC_4GR2_2Struct,(5,1,0):C.GC_4GR2_2Struct,(5,2,0):C.GC_4GR2_4Struct , \
                           (0,0,1):C.GC_4GR2_Fermion_delta11,(0,1,1):C.GC_4GR2_Fermion_delta5,(0,2,1):C.GC_4GR2_Fermion_delta5, \
                           (1,0,1):C.GC_4GR2_Fermion_delta5,(1,1,1):C.GC_4GR2_Fermion_delta11,(1,2,1):C.GC_4GR2_Fermion_delta5, \
                           (2,0,1):C.GC_4GR2_Fermion_delta5,(2,1,1):C.GC_4GR2_Fermion_delta5,(2,2,1):C.GC_4GR2_Fermion_delta11, \
                           (3,0,1):C.GC_4GR2_11Struct,(3,1,1):C.GC_4GR2_5Struct,(3,2,1):C.GC_4GR2_5Struct, \
                           (4,0,1):C.GC_4GR2_5Struct,(4,1,1):C.GC_4GR2_11Struct,(4,2,1):C.GC_4GR2_5Struct, \
                           (5,0,1):C.GC_4GR2_5Struct,(5,1,1):C.GC_4GR2_5Struct,(5,2,1):C.GC_4GR2_11Struct },
              type = 'R2')

#=============================================================================================

# gdd~
V_R2GDD = CTVertex(name = 'V_R2GDD',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles =[[[P.d,P.G]]],                 
              couplings = {(0,0,0):C.R2_GQQ},
              type = 'R2')

# guu~              
V_R2GUU = CTVertex(name = 'V_R2GUU',
               particles = [ P.u__tilde__, P.u, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               loop_particles =[[[P.u,P.G]]],
               couplings = {(0,0,0):C.R2_GQQ},
               type = 'R2')  

# gss~
V_R2GSS = CTVertex(name = 'V_R2GSS',
              particles = [ P.s__tilde__, P.s, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles =[[[P.s,P.G]]],
              couplings = {(0,0,0):C.R2_GQQ},
              type = 'R2')

# gcc~              
V_R2GCC = CTVertex(name = 'V_R2GCC',
               particles = [ P.c__tilde__, P.c, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               loop_particles =[[[P.c,P.G]]],               
               couplings = {(0,0,0):C.R2_GQQ},
               type = 'R2')  

# gbb~
V_R2GBB = CTVertex(name = 'V_R2GBB',
              particles = [ P.b__tilde__, P.b, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles =[[[P.b,P.G]]],
              couplings = {(0,0,0):C.R2_GQQ},
              type = 'R2')

# gtt~              
V_R2GTT = CTVertex(name = 'V_R2GTT',
               particles = [ P.t__tilde__, P.t, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               loop_particles =[[[P.t,P.G]]],               
               couplings = {(0,0,0):C.R2_GQQ},
               type = 'R2')

# gg             
V_R2GG = CTVertex(name = 'V_R2GG',
               particles = [ P.G, P.G ],
               color = [ 'Tr(1,2)' ],
               lorentz = [ L.R2_GG_1, L.R2_GG_2, L.R2_GG_3],
               loop_particles = [ [[P.u],[P.d],[P.c],[P.s]],
                                  [[P.b]],
                                  [[P.t]],
                                  [[P.G]] ],
               couplings = {(0,0,0):C.R2_GGq,
                            (0,0,1):C.R2_GGq,(0,2,1):C.R2_GGb,
                            (0,0,2):C.R2_GGq,(0,2,2):C.R2_GGt,
                            (0,0,3):C.R2_GGg_1, (0,1,3):C.R2_GGg_2},
               type = 'R2')

# d~d            
V_R2DD = CTVertex(name = 'V_R2DD',
               particles = [ P.d__tilde__, P.d ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_1 ],
               loop_particles = [[[P.d,P.G]]],
               couplings = {(0,0,0):C.R2_QQq},
               type = 'R2') 

# u~u            
V_R2UU = CTVertex(name = 'V_R2UU',
               particles = [ P.u__tilde__, P.u ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_1 ],
               loop_particles = [[[P.u,P.G]]],            
               couplings = {(0,0,0):C.R2_QQq},
               type = 'R2')

# s~s            
V_R2SS = CTVertex(name = 'V_R2SS',
               particles = [ P.s__tilde__, P.s ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_1 ],
               loop_particles = [[[P.s,P.G]]],                
               couplings = {(0,0,0):C.R2_QQq},
               type = 'R2')

# c~c            
V_R2CC = CTVertex(name = 'V_R2CC',
               particles = [ P.c__tilde__, P.c ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_1 ],
               loop_particles = [[[P.c,P.G]]],
               couplings = {(0,0,0):C.R2_QQq},                
               type = 'R2') 

# b~b            
V_R2BB = CTVertex(name = 'V_R2BB',
               particles = [ P.b__tilde__, P.b ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_1, L.R2_QQ_2 ],
               loop_particles = [[[P.b,P.G]]],
               couplings = {(0,0,0):C.R2_QQq,(0,1,0):C.R2_QQb},                
               type = 'R2')

# t~t            
V_R2TT = CTVertex(name = 'V_R2TT',
               particles = [ P.t__tilde__, P.t ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_1, L.R2_QQ_2 ],
               loop_particles = [[[P.t,P.G]]],
               couplings = {(0,0,0):C.R2_QQq,(0,1,0):C.R2_QQt},
               type = 'R2')

# ============== #
# Mixed QCD-QED  #
# ============== #

# R2 for the A and Z couplings to the quarks

V_R2ddA = CTVertex(name = 'V_R2ddA',
              particles = [ P.d__tilde__, P.d, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.d,P.G]]],
              couplings = {(0,0,0):C.R2_DDA},
              type = 'R2')

V_R2ssA = CTVertex(name = 'V_R2ssA',
              particles = [ P.s__tilde__, P.s, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.s,P.G]]],
              couplings = {(0,0,0):C.R2_DDA},
              type = 'R2')

V_R2bbA = CTVertex(name = 'V_R2bbA',
              particles = [ P.b__tilde__, P.b, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.b,P.G]]],
              couplings = {(0,0,0):C.R2_DDA},
              type = 'R2')

V_R2uuA = CTVertex(name = 'V_R2uuA',
              particles = [ P.u__tilde__, P.u, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u,P.G]]],
              couplings = {(0,0,0):C.R2_UUA},
              type = 'R2')

V_R2ccA = CTVertex(name = 'V_R2ccA',
              particles = [ P.c__tilde__, P.c, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.c,P.G]]],
              couplings = {(0,0,0):C.R2_UUA},
              type = 'R2')

V_R2ttA = CTVertex(name = 'V_R2ttA',
              particles = [ P.t__tilde__, P.t, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.t,P.G]]],
              couplings = {(0,0,0):C.R2_UUA},
              type = 'R2')

V_R2ddZ = CTVertex(name = 'V_R2ddZ',
              particles = [ P.d__tilde__, P.d, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              loop_particles = [[[P.d,P.G]]],
              couplings = {(0,0,0):C.R2_DDZ_V2,(0,1,0):C.R2_DDZ_V3},
              type = 'R2')

V_R2ssZ = CTVertex(name = 'V_R2ssZ',
              particles = [ P.s__tilde__, P.s, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              loop_particles = [[[P.s,P.G]]],
              couplings = {(0,0,0):C.R2_DDZ_V2,(0,1,0):C.R2_DDZ_V3},
              type = 'R2')

V_R2bbZ = CTVertex(name = 'V_R2bbZ',
              particles = [ P.b__tilde__, P.b, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              loop_particles = [[[P.b,P.G]]],
              couplings = {(0,0,0):C.R2_DDZ_V2,(0,1,0):C.R2_DDZ_V3},
              type = 'R2')

V_R2uuZ = CTVertex(name = 'V_R2uuZ',
              particles = [ P.u__tilde__, P.u, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV5 ],
              loop_particles = [[[P.u,P.G]]],
              couplings = {(0,0,0):C.R2_UUZ_V2,(0,1,0):C.R2_UUZ_V5},
              type = 'R2')

V_R2ccZ = CTVertex(name = 'V_R2ccZ',
              particles = [ P.c__tilde__, P.c, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV5 ],
              loop_particles = [[[P.c,P.G]]],
              couplings = {(0,0,0):C.R2_UUZ_V2,(0,1,0):C.R2_UUZ_V5},
              type = 'R2')

V_R2ttZ = CTVertex(name = 'V_R2ttZ',
              particles = [ P.t__tilde__, P.t, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV5 ],
              loop_particles = [[[P.t,P.G]]],
              couplings = {(0,0,0):C.R2_UUZ_V2,(0,1,0):C.R2_UUZ_V5},
              type = 'R2')

# R2 for the W couplings to the quarks with most general CKM

V_R2dxuW = CTVertex(name = 'V_R2dxuW',
              particles = [ P.d__tilde__, P.u, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.d,P.u,P.G]]],                   
              couplings = {(0,0,0):C.R2_dxuW},
              type = 'R2')

V_R2dxcW = CTVertex(name = 'V_R2dxcW',
              particles = [ P.d__tilde__, P.c, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.d,P.c,P.G]]],                   
              couplings = {(0,0,0):C.R2_dxcW},
              type = 'R2')

V_R2dxtW = CTVertex(name = 'V_R2dxtW',
              particles = [ P.d__tilde__, P.t, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.d,P.t,P.G]]],                   
              couplings = {(0,0,0):C.R2_dxtW},
              type = 'R2')

V_R2sxuW = CTVertex(name = 'V_R2sxuW',
              particles = [ P.s__tilde__, P.u, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.s,P.u,P.G]]],                   
              couplings = {(0,0,0):C.R2_sxuW},
              type = 'R2')

V_R2sxcW = CTVertex(name = 'V_R2sxcW',
              particles = [ P.s__tilde__, P.c, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.s,P.c,P.G]]],                   
              couplings = {(0,0,0):C.R2_sxcW},
              type = 'R2')

V_R2sxtW = CTVertex(name = 'V_R2sxtW',
              particles = [ P.s__tilde__, P.t, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.s,P.t,P.G]]],                   
              couplings = {(0,0,0):C.R2_sxtW},
              type = 'R2')

V_R2bxuW = CTVertex(name = 'V_R2bxuW',
              particles = [ P.b__tilde__, P.u, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.b,P.u,P.G]]],                   
              couplings = {(0,0,0):C.R2_bxuW},
              type = 'R2')

V_R2bxcW = CTVertex(name = 'V_R2bxcW',
              particles = [ P.b__tilde__, P.c, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.b,P.c,P.G]]],                   
              couplings = {(0,0,0):C.R2_bxcW},
              type = 'R2')

V_R2bxtW = CTVertex(name = 'V_R2bxtW',
              particles = [ P.b__tilde__, P.t, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.b,P.t,P.G]]],                   
              couplings = {(0,0,0):C.R2_bxtW},
              type = 'R2')

V_R2uxdW = CTVertex(name = 'V_R2uxdW',
              particles = [ P.u__tilde__, P.d, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.u,P.d,P.G]]],                   
              couplings = {(0,0,0):C.R2_uxdW},
              type = 'R2')

V_R2cxdW = CTVertex(name = 'V_R2cxdW',
              particles = [ P.c__tilde__, P.d, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.c,P.d,P.G]]],                   
              couplings = {(0,0,0):C.R2_cxdW},
              type = 'R2')

V_R2txdW = CTVertex(name = 'V_R2txdW',
              particles = [ P.t__tilde__, P.d, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.t,P.d,P.G]]],                   
              couplings = {(0,0,0):C.R2_txdW},
              type = 'R2')

V_R2uxsW = CTVertex(name = 'V_R2uxsW',
              particles = [ P.u__tilde__, P.s, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.u,P.s,P.G]]],                   
              couplings = {(0,0,0):C.R2_uxsW},
              type = 'R2')

V_R2cxsW = CTVertex(name = 'V_R2cxsW',
              particles = [ P.c__tilde__, P.s, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.c,P.s,P.G]]],                   
              couplings = {(0,0,0):C.R2_cxsW},
              type = 'R2')

V_R2txsW = CTVertex(name = 'V_R2txsW',
              particles = [ P.t__tilde__, P.s, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.t,P.s,P.G]]],                   
              couplings = {(0,0,0):C.R2_txsW},
              type = 'R2')

V_R2uxbW = CTVertex(name = 'V_R2uxbW',
              particles = [ P.u__tilde__, P.b, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.u,P.b,P.G]]],                   
              couplings = {(0,0,0):C.R2_uxbW},
              type = 'R2')

V_R2cxbW = CTVertex(name = 'V_R2cxbW',
              particles = [ P.c__tilde__, P.b, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.c,P.b,P.G]]],                   
              couplings = {(0,0,0):C.R2_cxbW},
              type = 'R2')

V_R2txbW = CTVertex(name = 'V_R2txbW',
              particles = [ P.t__tilde__, P.b, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.t,P.b,P.G]]],                   
              couplings = {(0,0,0):C.R2_txbW},
              type = 'R2')

# R2 for SQQ~ 

V_bbG0 = CTVertex(name = 'V_bbG0',
              particles = [ P.b__tilde__, P.b, P.G0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              loop_particles = [[[P.b,P.G]]],
              couplings = {(0,0,0):C.R2_bbG0},
              type = 'R2')

V_ttG0 = CTVertex(name = 'V_ttG0',
              particles = [ P.t__tilde__, P.t, P.G0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              loop_particles = [[[P.t,P.G]]],
              couplings = {(0,0,0):C.R2_ttG0},
              type = 'R2')

V_ccG0 = CTVertex(name = 'V_ccG0',
              particles = [ P.c__tilde__, P.c, P.G0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              loop_particles = [[[P.c,P.G]]],
              couplings = {(0,0,0):C.R2_ccG0},
              type = 'R2')

V_uuG0 = CTVertex(name = 'V_uuG0',
              particles = [ P.u__tilde__, P.u, P.G0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              loop_particles = [[[P.u,P.G]]],
              couplings = {(0,0,0):C.R2_uuG0},
              type = 'R2')

V_ddG0 = CTVertex(name = 'V_ddG0',
              particles = [ P.d__tilde__, P.d, P.G0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              loop_particles = [[[P.d,P.G]]],
              couplings = {(0,0,0):C.R2_ddG0},
              type = 'R2')

V_ssG0 = CTVertex(name = 'V_ssG0',
              particles = [ P.s__tilde__, P.s, P.G0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              loop_particles = [[[P.s,P.G]]],
              couplings = {(0,0,0):C.R2_ssG0},
              type = 'R2')

V_bbH = CTVertex(name = 'V_bbH',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4 ],
              loop_particles = [[[P.b,P.G]]],
              couplings = {(0,0,0):C.R2_bbH},
              type = 'R2')

V_ttH = CTVertex(name = 'V_ttH',
              particles = [ P.t__tilde__, P.t, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4 ],
              loop_particles = [[[P.t,P.G]]],
              couplings = {(0,0,0):C.R2_ttH},
              type = 'R2')

V_ccH = CTVertex(name = 'V_ccH',
              particles = [ P.c__tilde__, P.c, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4 ],
              loop_particles = [[[P.c,P.G]]],
              couplings = {(0,0,0):C.R2_ccH},
              type = 'R2')

V_uuH = CTVertex(name = 'V_uuH',
              particles = [ P.u__tilde__, P.u, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4 ],
              loop_particles = [[[P.u,P.G]]],
              couplings = {(0,0,0):C.R2_uuH},
              type = 'R2')

V_ddH = CTVertex(name = 'V_ddH',
              particles = [ P.d__tilde__, P.d, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4 ],
              loop_particles = [[[P.d,P.G]]],
              couplings = {(0,0,0):C.R2_ddH},
              type = 'R2')

V_ssH = CTVertex(name = 'V_ssH',
              particles = [ P.s__tilde__, P.s, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4 ],
              loop_particles = [[[P.s,P.G]]],
              couplings = {(0,0,0):C.R2_ssH},
              type = 'R2')

V_uxdGp = CTVertex(name = 'V_uxdGp',
              particles = [ P.u__tilde__, P.d, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.u, P.d, P.G]]],
              couplings = {(0,0,0):C.R2_uxdGp, (0,1,0):C.R2_uxdGpA},
              type = 'R2')

V_uxsGp = CTVertex(name = 'V_uxsGp',
              particles = [ P.u__tilde__, P.s, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.u, P.s, P.G]]],
              couplings = {(0,0,0):C.R2_uxsGp, (0,1,0):C.R2_uxsGpA},
              type = 'R2')

V_uxbGp = CTVertex(name = 'V_uxbGp',
              particles = [ P.u__tilde__, P.b, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.u, P.b, P.G]]],
              couplings = {(0,0,0):C.R2_uxbGp, (0,1,0):C.R2_uxbGpA},
              type = 'R2')

V_cxdGp = CTVertex(name = 'V_cxdGp',
              particles = [ P.c__tilde__, P.d, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.c, P.d, P.G]]],
              couplings = {(0,0,0):C.R2_cxdGp, (0,1,0):C.R2_cxdGpA},
              type = 'R2')

V_cxsGp = CTVertex(name = 'V_cxsGp',
              particles = [ P.c__tilde__, P.s, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.c, P.s, P.G]]],
              couplings = {(0,0,0):C.R2_cxsGp, (0,1,0):C.R2_cxsGpA},
              type = 'R2')

V_cxbGp = CTVertex(name = 'V_cxbGp',
              particles = [ P.c__tilde__, P.b, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.c, P.b, P.G]]],
              couplings = {(0,0,0):C.R2_cxbGp, (0,1,0):C.R2_cxbGpA},
              type = 'R2')

V_txdGp = CTVertex(name = 'V_txdGp',
              particles = [ P.t__tilde__, P.d, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.t, P.d, P.G]]],
              couplings = {(0,0,0):C.R2_txdGp, (0,1,0):C.R2_txdGpA},
              type = 'R2')

V_txsGp = CTVertex(name = 'V_txsGp',
              particles = [ P.t__tilde__, P.s, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.t, P.s, P.G]]],
              couplings = {(0,0,0):C.R2_txsGp, (0,1,0):C.R2_txsGpA},
              type = 'R2')

V_txbGp = CTVertex(name = 'V_txbGp',
              particles = [ P.t__tilde__, P.b, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.t, P.b, P.G]]],
              couplings = {(0,0,0):C.R2_txbGp, (0,1,0):C.R2_txbGpA},
              type = 'R2')

V_dxuGm = CTVertex(name = 'V_dxuGm',
              particles = [ P.d__tilde__, P.u, P.G__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.u, P.d, P.G]]],
              couplings = {(0,0,0):C.R2_dxuGm, (0,1,0):C.R2_dxuGmA},
              type = 'R2')

V_sxuGm = CTVertex(name = 'V_sxuGm',
              particles = [ P.s__tilde__, P.u, P.G__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.u, P.s, P.G]]],
              couplings = {(0,0,0):C.R2_sxuGm, (0,1,0):C.R2_sxuGmA},
              type = 'R2')

V_bxuGm = CTVertex(name = 'V_bxuGm',
              particles = [ P.b__tilde__, P.u, P.G__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.u, P.b, P.G]]],
              couplings = {(0,0,0):C.R2_bxuGm, (0,1,0):C.R2_bxuGmA},
              type = 'R2')

V_dxcGm = CTVertex(name = 'V_dxcGm',
              particles = [ P.d__tilde__, P.c, P.G__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.c, P.d, P.G]]],
              couplings = {(0,0,0):C.R2_dxcGm, (0,1,0):C.R2_dxcGmA},
              type = 'R2')

V_sxcGm = CTVertex(name = 'V_sxcGpm',
              particles = [ P.s__tilde__, P.c, P.G__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.c, P.s, P.G]]],
              couplings = {(0,0,0):C.R2_sxcGm, (0,1,0):C.R2_sxcGmA},
              type = 'R2')

V_bxcGm = CTVertex(name = 'V_bxcGm',
              particles = [ P.b__tilde__, P.c, P.G__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.c, P.b, P.G]]],
              couplings = {(0,0,0):C.R2_bxcGm, (0,1,0):C.R2_bxcGmA},
              type = 'R2')

V_dxtGm = CTVertex(name = 'V_dxtGm',
              particles = [ P.d__tilde__, P.t, P.G__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.t, P.d, P.G]]],
              couplings = {(0,0,0):C.R2_dxtGm, (0,1,0):C.R2_dxtGmA},
              type = 'R2')

V_sxtGm = CTVertex(name = 'V_sxtGm',
              particles = [ P.s__tilde__, P.t, P.G__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.t, P.s, P.G]]],
              couplings = {(0,0,0):C.R2_sxtGm, (0,1,0):C.R2_sxtGmA},
              type = 'R2')

V_bxtGm = CTVertex(name = 'V_bxtGm',
              particles = [ P.b__tilde__, P.t, P.G__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4, L.FFS2 ],
              loop_particles = [[[P.t, P.b, P.G]]],
              couplings = {(0,0,0):C.R2_bxtGm, (0,1,0):C.R2_bxtGmA},
              type = 'R2')

# R2 interactions non proportional to the SM

# R2 for SGG

V_GGH = CTVertex(name = 'V_GGH',
              particles = [ P.G, P.G, P.H ],
              color = [ 'Tr(1,2)' ],
              lorentz = [ L.VVS1 ],
              loop_particles = [[[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]]],
              couplings = {(0,0,0):C.R2_GGHu,(0,0,1):C.R2_GGHd,(0,0,2):C.R2_GGHs,
                           (0,0,3):C.R2_GGHc,(0,0,4):C.R2_GGHb,(0,0,5):C.R2_GGHt},
              type = 'R2')

# R2 for SSGG

V_GGHH = CTVertex(name = 'V_GGHH',
              particles = [ P.G, P.G, P.H, P.H ],
              color = [ 'Tr(1,2)' ],
              lorentz = [ L.R2_GGHH ],
              loop_particles = [[[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]]],
              couplings = {(0,0,0):C.R2_GGHHu,(0,0,1):C.R2_GGHHd,(0,0,2):C.R2_GGHHs,
                           (0,0,3):C.R2_GGHHc,(0,0,4):C.R2_GGHHb,(0,0,5):C.R2_GGHHt},
              type = 'R2')

V_GGG0G0 = CTVertex(name = 'V_GGG0G0',
              particles = [ P.G, P.G, P.G0, P.G0 ],
              color = [ 'Tr(1,2)' ],
              lorentz = [ L.R2_GGHH ],
              loop_particles = [[[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]]],
              couplings = {(0,0,0):C.R2_GGG0G0u,(0,0,1):C.R2_GGG0G0d,(0,0,2):C.R2_GGG0G0s,
                           (0,0,3):C.R2_GGG0G0c,(0,0,4):C.R2_GGG0G0b,(0,0,5):C.R2_GGG0G0t},
              type = 'R2')

V_GGGmGp = CTVertex(name = 'V_GGGmGp',
              particles = [ P.G, P.G, P.G__minus__, P.G__plus__ ],
              color = [ 'Tr(1,2)' ],
              lorentz = [ L.R2_GGHH ],
              loop_particles = [[[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
              couplings = {(0,0,0):C.R2_GGGmGpud,(0,0,1):C.R2_GGGmGpus,(0,0,2):C.R2_GGGmGpub,
                           (0,0,3):C.R2_GGGmGpcd,(0,0,4):C.R2_GGGmGpcs,(0,0,5):C.R2_GGGmGpcb,
                           (0,0,6):C.R2_GGGmGptd,(0,0,7):C.R2_GGGmGpts,(0,0,8):C.R2_GGGmGptb},
              type = 'R2')

# R2 for the weak vector bosons interaction with gluons

V_GGZ = CTVertex(name = 'V_GGZ',
              particles = [ P.G, P.G, P.Z ],
              color = [ 'Tr(1,2)' ],
              lorentz = [ L.R2_GGZ ],
              loop_particles = [[[P.u],[P.c],[P.t]],[[P.d],[P.s],[P.b]]],
              couplings = {(0,0,0):C.R2_GGZup,(0,0,1):C.R2_GGZdown},
              type = 'R2')



V_GGZZ = CTVertex(name = 'V_GGZZ',
              particles = [ P.G, P.G, P.Z, P.Z ],
              color = [ 'Tr(1,2)' ],
              lorentz = [ L.R2_GGVV ],
              loop_particles = [[[P.u],[P.c],[P.t]],[[P.d],[P.s],[P.b]]],
              couplings = {(0,0,0):C.R2_GGZZup,(0,0,1):C.R2_GGZZdown},
              type = 'R2')

V_GGAA = CTVertex(name = 'V_GGAA',
              particles = [ P.G, P.G, P.A, P.A ],
              color = [ 'Tr(1,2)' ],
              lorentz = [ L.R2_GGVV ],
              loop_particles = [[[P.u],[P.c],[P.t]],[[P.d],[P.s],[P.b]]],
              couplings = {(0,0,0):C.R2_GGAAup,(0,0,1):C.R2_GGAAdown},
              type = 'R2')

V_GGZA = CTVertex(name = 'V_GGZA',
              particles = [ P.G, P.G, P.Z, P.A ],
              color = [ 'Tr(1,2)' ],
              lorentz = [ L.R2_GGVV ],
              loop_particles = [[[P.u],[P.c],[P.t]],[[P.d],[P.s],[P.b]]],
              couplings = {(0,0,0):C.R2_GGZAup,(0,0,1):C.R2_GGZAdown},
              type = 'R2')

V_GGWW = CTVertex(name = 'V_GGWW',
              particles = [ P.G, P.G, P.W__minus__, P.W__plus__ ],
              color = [ 'Tr(1,2)' ],
              lorentz = [ L.R2_GGVV ],
              loop_particles = [[[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
              couplings = {(0,0,0):C.R2_GGWWud,(0,0,1):C.R2_GGWWus,(0,0,2):C.R2_GGWWub,
                           (0,0,3):C.R2_GGWWcd,(0,0,4):C.R2_GGWWcs,(0,0,5):C.R2_GGWWcb,
                           (0,0,6):C.R2_GGWWtd,(0,0,7):C.R2_GGWWts,(0,0,8):C.R2_GGWWtb},
              type = 'R2')

V_GGGZ = CTVertex(name = 'V_GGGZ',
              particles = [ P.G, P.G, P.G, P.Z ],
              color = [ 'd(1,2,3)' , 'f(1,2,3)'],
              lorentz = [ L.R2_GGVV, L.R2_GGGVa ],
              loop_particles = [[[P.u],[P.c],[P.t]],[[P.d],[P.s],[P.b]]],
              couplings = {(0,0,0):C.R2_GGGZvecUp,(0,0,1):C.R2_GGGZvecDown,
                           (1,1,0):C.R2_GGGZaxialUp,(1,1,1):C.R2_GGGZaxialDown},
              type = 'R2')

V_GGGA = CTVertex(name = 'V_GGGA',
              particles = [ P.G, P.G, P.G, P.A ],
              color = [ 'd(1,2,3)'],
              lorentz = [ L.R2_GGVV ],
              loop_particles = [[[P.u],[P.c],[P.t]],[[P.d],[P.s],[P.b]]],
              couplings = {(0,0,0):C.R2_GGGAvecUp,(0,0,1):C.R2_GGGAvecDown},
              type = 'R2')

# ========= #
# Pure QED  #
# ========= #

# R2 with 2 external legs

# R2 for SS

V_R2HH = CTVertex(name = 'V_R2HH',
                  particles = [ P.H, P.H ],
                  color = [ '1' ],
                  lorentz = [ L.R2_SS_1, L.R2_SS_2 ],
                  loop_particles = [[[P.H],[P.G0],[P.G__plus__],[P.Z],[P.W__plus__],[P.Z,P.G0],[P.W__plus__,P.G__plus__]],
                               [[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                               [[P.u]],[[P.d]],[[P.s]],
                               [[P.c]],[[P.b]],[[P.t]]],
                  couplings = {(0,0,0):C.R2_HHboson1, (0,1,0):C.R2_HHboson2,
                               (0,0,1):C.R2_HHe1,(0,1,1):C.R2_HHe2,(0,0,2):C.R2_HHm1,(0,1,2):C.R2_HHm2,(0,0,3):C.R2_HHtau1,(0,1,3):C.R2_HHtau2,
                               (0,0,4):C.R2_HHu1,(0,1,4):C.R2_HHu2,(0,0,5):C.R2_HHd1,(0,1,5):C.R2_HHd2,(0,0,6):C.R2_HHs1,(0,1,6):C.R2_HHs2,
                               (0,0,7):C.R2_HHc1,(0,1,7):C.R2_HHc2,(0,0,8):C.R2_HHb1,(0,1,8):C.R2_HHb2,(0,0,9):C.R2_HHt1,(0,1,9):C.R2_HHt2},
                  type = 'R2')

V_R2G0G0 = CTVertex(name = 'V_R2G0G0',
                  particles = [ P.G0, P.G0 ],
                  color = [ '1' ],
                  lorentz = [ L.R2_SS_1, L.R2_SS_2 ],
                  loop_particles = [[[P.H],[P.G0],[P.Z],[P.W__plus__],[P.G__plus__],[P.H,P.G0],[P.Z,P.H],[P.W__plus__,P.G__plus__]],
                               [[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                               [[P.u]],[[P.d]],[[P.s]],
                               [[P.c]],[[P.b]],[[P.t]]],
                  couplings = {(0,0,0):C.R2_G0G0boson1, (0,1,0):C.R2_G0G0boson2,
                               (0,0,1):C.R2_HHe1,(0,1,1):C.R2_HHe2,(0,0,2):C.R2_HHm1,(0,1,2):C.R2_HHm2,(0,0,3):C.R2_HHtau1,(0,1,3):C.R2_HHtau2,
                               (0,0,4):C.R2_HHu1,(0,1,4):C.R2_HHu2,(0,0,5):C.R2_HHd1,(0,1,5):C.R2_HHd2,(0,0,6):C.R2_HHs1,(0,1,6):C.R2_HHs2,
                               (0,0,7):C.R2_HHc1,(0,1,7):C.R2_HHc2,(0,0,8):C.R2_HHb1,(0,1,8):C.R2_HHb2,(0,0,9):C.R2_HHt1,(0,1,9):C.R2_HHt2},
                  type = 'R2')

V_R2GmGp = CTVertex(name = 'V_R2GmGp',
                  particles = [ P.G__minus__, P.G__plus__ ],
                  color = [ '1' ],
                  lorentz = [ L.R2_SS_1, L.R2_SS_2 ],
                  loop_particles = [[[P.H],[P.Z],[P.W__plus__],[P.A],[P.G0],[P.G__plus__],[P.G__plus__,P.H],[P.W__plus__,P.A],[P.W__plus__,P.Z],[P.G__plus__,P.A],[P.G__plus__,P.Z],[P.W__plus__,P.H],[P.W__plus__,P.G0]],
                               [[P.e__minus__, P.ve]],[[P.m__minus__, P.vm]],[[P.tt__minus__, P.vt]],
                               [[P.u, P.d]],[[P.u, P.s]],[[P.u, P.b]],
                               [[P.c, P.d]],[[P.c, P.s]],[[P.c, P.b]],
                               [[P.t, P.d]],[[P.t, P.s]],[[P.t, P.b]]],
                  couplings = {(0,0,0):C.R2_GmGpboson1, (0,1,0):C.R2_GmGpboson2,
                               (0,0,1):C.R2_GmGpe,(0,1,1):C.R2_HHe2,(0,0,2):C.R2_GmGpm,(0,1,2):C.R2_HHm2,(0,0,3):C.R2_GmGptau,(0,1,3):C.R2_HHtau2,
                               (0,0,4):C.R2_GmGpud1,(0,1,4):C.R2_GmGpud2,(0,0,5):C.R2_GmGpus1,(0,1,5):C.R2_GmGpus2,(0,0,6):C.R2_GmGpub1,(0,1,6):C.R2_GmGpub2,
                               (0,0,7):C.R2_GmGpcd1,(0,1,7):C.R2_GmGpcd2,(0,0,8):C.R2_GmGpcs1,(0,1,8):C.R2_GmGpcs2,(0,0,9):C.R2_GmGpcb1,(0,1,9):C.R2_GmGpcb2,
                               (0,0,10):C.R2_GmGptd1,(0,1,10):C.R2_GmGptd2,(0,0,11):C.R2_GmGpts1,(0,1,11):C.R2_GmGpts2,(0,0,12):C.R2_GmGptb1,(0,1,12):C.R2_GmGptb2},
                  type = 'R2')

 # R2 for VV

V_R2AA = CTVertex(name = 'V_R2AA',
                   particles = [ P.A, P.A ],
                   color = ['1' ],
                   lorentz = [L.R2_GG_1, L.R2_GG_2, L.R2_GG_3],
                   loop_particles = [[[P.W__plus__],[P.G__plus__],[P.W__plus__,P.G__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],[[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]]],
                   couplings = {(0,0,0):C.R2_AAboson1,(0,1,0):C.R2_AAboson2,(0,2,0):C.R2_AAboson3,
                                (0,0,1):C.R2_AAl,(0,2,1):C.R2_AAe3,(0,0,2):C.R2_AAl,(0,2,2):C.R2_AAm3,(0,0,3):C.R2_AAl,(0,2,3):C.R2_AAtau3,
                                (0,0,4):C.R2_AAU,(0,2,4):C.R2_AAu3,(0,0,5):C.R2_AAD,(0,2,5):C.R2_AAd3,(0,0,6):C.R2_AAD,(0,2,6):C.R2_AAs3,
                                (0,0,7):C.R2_AAU,(0,2,7):C.R2_AAc3,(0,0,8):C.R2_AAD,(0,2,8):C.R2_AAb3,(0,0,9):C.R2_AAU,(0,2,9):C.R2_AAt3},
                   type = 'R2')

V_R2AZ = CTVertex(name = 'V_R2AZ',
                   particles = [ P.A, P.Z ],
                   color = [ '1' ],
                   lorentz = [L.R2_GG_1, L.R2_GG_2, L.R2_GG_3],
                   loop_particles = [[[P.W__plus__],[P.G__plus__],[P.W__plus__,P.G__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],[[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]]],
                   couplings = {(0,0,0):C.R2_AZboson1,(0,1,0):C.R2_AZboson2,(0,2,0):C.R2_AZboson3,
                                (0,0,1):C.R2_AZl,(0,2,1):C.R2_AZe3,(0,0,2):C.R2_AZl,(0,2,2):C.R2_AZm3,(0,0,3):C.R2_AZl,(0,2,3):C.R2_AZtau3,
                                (0,0,4):C.R2_AZU,(0,2,4):C.R2_AZu3,(0,0,5):C.R2_AZD,(0,2,5):C.R2_AZd3,(0,0,6):C.R2_AZD,(0,2,6):C.R2_AZs3,
                                (0,0,7):C.R2_AZU,(0,2,7):C.R2_AZc3,(0,0,8):C.R2_AZD,(0,2,8):C.R2_AZb3,(0,0,9):C.R2_AZU,(0,2,9):C.R2_AZt3},
                   type = 'R2')

V_R2ZZ = CTVertex(name = 'V_R2ZZ',
                   particles = [ P.Z, P.Z ],
                   color = [ '1' ],
                   lorentz = [L.R2_GG_1, L.R2_GG_2, L.R2_GG_3],
                   loop_particles = [[[P.H],[P.G0],[P.G__plus__],[P.W__plus__],[P.H,P.G0],[P.Z,P.H],[P.G__plus__,P.W__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],[[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]],[[P.ve],[P.vm],[P.vt]]],
                   couplings = {(0,0,0):C.R2_ZZboson1,(0,1,0):C.R2_ZZboson2,(0,2,0):C.R2_ZZboson3,
                                (0,0,1):C.R2_ZZl,(0,2,1):C.R2_ZZe3,(0,0,2):C.R2_ZZl,(0,2,2):C.R2_ZZm3,(0,0,3):C.R2_ZZl,(0,2,3):C.R2_ZZtau3,
                                (0,0,4):C.R2_ZZU,(0,2,4):C.R2_ZZu3,(0,0,5):C.R2_ZZD,(0,2,5):C.R2_ZZd3,(0,0,6):C.R2_ZZD,(0,2,6):C.R2_ZZs3,
                                (0,0,7):C.R2_ZZU,(0,2,7):C.R2_ZZc3,(0,0,8):C.R2_ZZD,(0,2,8):C.R2_ZZb3,(0,0,9):C.R2_ZZU,(0,2,9):C.R2_ZZt3,
                                (0,0,10):C.R2_ZZv},
                   type = 'R2')

V_R2WW = CTVertex(name = 'V_R2WW',
                   particles = [ P.W__minus__, P.W__plus__ ],
                   color = [ '1' ],
                   lorentz = [L.R2_GG_1, L.R2_GG_2, L.R2_GG_3],
                   loop_particles = [[[P.H],[P.W__plus__],[P.G__plus__],[P.G0],[P.A],[P.Z],[P.H,P.G__plus__],[P.G0,P.G__plus__],[P.A,P.W__plus__],[P.W__plus__,P.Z],[P.A,P.G__plus__],[P.Z,P.G__plus__],[P.H,P.W__plus__]],
                                     [[P.e__minus__, P.ve]],[[P.m__minus__, P.vm]],[[P.tt__minus__, P.vt]],
                                     [[P.u, P.d]],[[P.u, P.s]],[[P.u, P.b]],
                                     [[P.c, P.d]],[[P.c, P.s]],[[P.c, P.b]],
                                     [[P.t, P.d]],[[P.t, P.s]],[[P.t, P.b]]],
                   couplings = {(0,0,0):C.R2_WWboson1,(0,1,0):C.R2_WWboson2,(0,2,0):C.R2_WWboson3,
                                (0,0,1):C.R2_WWl,(0,2,1):C.R2_WWe3,(0,0,2):C.R2_WWl,(0,2,2):C.R2_WWm3,(0,0,3):C.R2_WWl,(0,2,3):C.R2_WWtau3,
                                (0,0,4):C.R2_WWud1,(0,2,4):C.R2_WWud3,(0,0,5):C.R2_WWus1,(0,2,5):C.R2_WWus3,(0,0,6):C.R2_WWub1,(0,2,6):C.R2_WWub3,
                                (0,0,7):C.R2_WWcd1,(0,2,7):C.R2_WWcd3,(0,0,8):C.R2_WWcs1,(0,2,8):C.R2_WWcs3,(0,0,9):C.R2_WWcb1,(0,2,9):C.R2_WWcb3,
                                (0,0,10):C.R2_WWtd1,(0,2,10):C.R2_WWtd3,(0,0,11):C.R2_WWts1,(0,2,11):C.R2_WWts3,(0,0,12):C.R2_WWtb1,(0,2,12):C.R2_WWtb3},
                   type = 'R2')

   # R2 for FF~

V_R2UU2 = CTVertex(name = 'V_R2UU2',
               particles = [ P.u__tilde__, P.u ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
               loop_particles = [[[P.u,P.H],[P.u,P.G0],[P.A,P.u],[P.u, P.Z]],[[P.G__plus__,P.d],[P.W__plus__,P.d]],[[P.G__plus__,P.s],[P.W__plus__,P.s]],[[P.G__plus__,P.b],[P.W__plus__,P.b]]],            
               couplings = {(0,0,0):C.R2_UUC0, (0,1,0):C.R2_UUCm,(0,2,0):C.R2_QQCp0,
                            (0,2,1):C.R2_QQCpud,(0,2,2):C.R2_QQCpus,(0,2,3):C.R2_QQCpub},
               type = 'R2')

V_R2CC2 = CTVertex(name = 'V_R2CC2',
               particles = [ P.c__tilde__, P.c ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
               loop_particles = [[[P.c,P.H],[P.c,P.G0],[P.c,P.A],[P.c, P.Z]],[[P.G__plus__,P.d],[P.W__plus__,P.d]],[[P.G__plus__,P.s],[P.W__plus__,P.s]],[[P.G__plus__,P.b],[P.W__plus__,P.b]]],            
               couplings = {(0,0,0):C.R2_CCC0, (0,1,0):C.R2_UUCm,(0,2,0):C.R2_QQCp0,
                            (0,2,1):C.R2_QQCpcd,(0,2,2):C.R2_QQCpcs,(0,2,3):C.R2_QQCpcb},
               type = 'R2')

V_R2TT2 = CTVertex(name = 'V_R2TT2',
               particles = [ P.t__tilde__, P.t ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
               loop_particles = [[[P.t,P.H],[P.t,P.G0],[P.t,P.A],[P.t, P.Z]],[[P.G__plus__,P.d],[P.W__plus__,P.d]],[[P.G__plus__,P.s],[P.W__plus__,P.s]],[[P.G__plus__,P.b],[P.W__plus__,P.b]]],            
               couplings = {(0,0,0):C.R2_TTC0, (0,1,0):C.R2_UUCm,(0,2,0):C.R2_QQCp0,
                            (0,2,1):C.R2_QQCptd,(0,2,2):C.R2_QQCpts,(0,2,3):C.R2_QQCptb},
               type = 'R2')

V_R2DD2 = CTVertex(name = 'V_R2DD2',
               particles = [ P.d__tilde__, P.d ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
               loop_particles = [[[P.d,P.H],[P.d,P.G0],[P.d,P.A],[P.d, P.Z]],[[P.G__plus__,P.u],[P.W__plus__,P.u]],[[P.G__plus__,P.c],[P.W__plus__,P.c]],[[P.G__plus__,P.t],[P.W__plus__,P.t]]],            
               couplings = {(0,0,0):C.R2_DDC0, (0,1,0):C.R2_DDCm,(0,2,0):C.R2_QQCp0,
                            (0,2,1):C.R2_QQCpud,(0,2,2):C.R2_QQCpcd,(0,2,3):C.R2_QQCptd},
               type = 'R2')
    
V_R2SS2 = CTVertex(name = 'V_R2SS2',
               particles = [ P.s__tilde__, P.s ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
               loop_particles = [[[P.s,P.H],[P.s,P.G0],[P.s,P.A],[P.s, P.Z]],[[P.G__plus__,P.u],[P.W__plus__,P.u]],[[P.G__plus__,P.c],[P.W__plus__,P.c]],[[P.G__plus__,P.t],[P.W__plus__,P.t]]],            
               couplings = {(0,0,0):C.R2_SSC0, (0,1,0):C.R2_DDCm,(0,2,0):C.R2_QQCp0,
                            (0,2,1):C.R2_QQCpus,(0,2,2):C.R2_QQCpcs,(0,2,3):C.R2_QQCpts},
               type = 'R2')

V_R2BB2 = CTVertex(name = 'V_R2BB2',
               particles = [ P.b__tilde__, P.b ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
               loop_particles = [[[P.b,P.H],[P.b,P.G0],[P.b,P.A],[P.b, P.Z]],[[P.G__plus__,P.u],[P.W__plus__,P.u]],[[P.G__plus__,P.c],[P.W__plus__,P.c]],[[P.G__plus__,P.t],[P.W__plus__,P.t]]],            
               couplings = {(0,0,0):C.R2_BBC0, (0,1,0):C.R2_DDCm,(0,2,0):C.R2_QQCp0,
                            (0,2,1):C.R2_QQCpub,(0,2,2):C.R2_QQCpcb,(0,2,3):C.R2_QQCptb},
               type = 'R2')

V_R2ee = CTVertex(name = 'V_R2ee',
                      particles = [ P.e__plus__, P.e__minus__ ],
                      color = [ '1' ],
                      lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
                      loop_particles = [[[P.e__minus__,P.H],[P.e__minus__,P.G0],[P.e__minus__,P.A],[P.e__minus__, P.Z]],[[P.G__plus__,P.ve],[P.W__plus__,P.ve]]],
                      couplings = {(0,0,0):C.R2_EEC0,(0,1,0):C.R2_LLCm,(0,2,0):C.R2_LLCp0,(0,2,1):C.R2_LLCplv},
                      type = 'R2')

V_R2mm = CTVertex(name = 'V_R2mm',
                      particles = [ P.m__plus__, P.m__minus__ ],
                      color = [ '1' ],
                      lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
                      loop_particles = [[[P.m__minus__,P.H],[P.m__minus__,P.G0],[P.m__minus__,P.A],[P.m__minus__, P.Z]],[[P.G__plus__,P.vm],[P.W__plus__,P.vm]]],
                      couplings = {(0,0,0):C.R2_MMC0,(0,1,0):C.R2_LLCm,(0,2,0):C.R2_LLCp0,(0,2,1):C.R2_LLCplv},
                      type = 'R2')

V_R2tautau = CTVertex(name = 'V_R2tautau',
                      particles = [ P.tt__plus__, P.tt__minus__ ],
                      color = [ '1' ],
                      lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
                      loop_particles = [[[P.tt__minus__,P.H],[P.tt__minus__,P.G0],[P.tt__minus__,P.A],[P.tt__minus__, P.Z]],[[P.G__plus__,P.vt],[P.W__plus__,P.vt]]],
                      couplings = {(0,0,0):C.R2_TATAC0,(0,1,0):C.R2_LLCm,(0,2,0):C.R2_LLCp0,(0,2,1):C.R2_LLCplv},
                      type = 'R2')

V_R2veve = CTVertex(name = 'V_R2veve',
                      particles = [ P.ve__tilde__, P.ve ],
                      color = [ '1' ],
                      lorentz = [ L.R2_QQ_4 ],
                      loop_particles = [[[P.ve,P.Z]],[[P.G__plus__,P.e__minus__],[P.W__plus__,P.e__minus__]]],
                      couplings = {(0,0,0):C.R2_nunuCp0,(0,0,1):C.R2_LLCplv},
                      type = 'R2')

V_R2vmvm = CTVertex(name = 'V_R2vmvm',
                      particles = [ P.vm__tilde__, P.vm ],
                      color = [ '1' ],
                      lorentz = [ L.R2_QQ_4 ],
                      loop_particles = [[[P.vm, P.Z]],[[P.G__plus__,P.e__minus__],[P.W__plus__,P.m__minus__]]],
                      couplings = {(0,0,0):C.R2_nunuCp0,(0,0,1):C.R2_LLCplv},
                      type = 'R2')

V_R2vtvt = CTVertex(name = 'V_R2vtvt',
                      particles = [ P.vt__tilde__, P.vt ],
                      color = [ '1' ],
                      lorentz = [ L.R2_QQ_4 ],
                      loop_particles = [[[P.vt, P.Z]],[[P.G__plus__,P.tt__minus__],[P.W__plus__,P.tt__minus__]]],
                      couplings = {(0,0,0):C.R2_nunuCp0,(0,0,1):C.R2_LLCplv},
                      type = 'R2')

# R2 with 3 external legs

# R2 for SFF~

V_uuH2 = CTVertex(name = 'V_uuH2',
                   particles = [ P.u__tilde__, P.u, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS4 ],
                   loop_particles = [[[P.u,P.H],[P.u,P.G0],[P.u,P.A],[P.Z, P.u],[P.u,P.G0,P.Z]],[[P.G__plus__,P.W__plus__,P.d],[P.G__plus__,P.d],[P.W__plus__, P.d]],[[P.G__plus__,P.W__plus__,P.s],[P.G__plus__,P.s],[P.W__plus__, P.s]],[[P.G__plus__,P.W__plus__,P.b],[P.G__plus__,P.b],[P.W__plus__,P.b]]],
                   couplings = {(0,0,0):C.R2_Huu, (0,0,1):C.R2_Huu_d, (0,0,2):C.R2_Huu_s, (0,0,3):C.R2_Huu_b},
                   type = 'R2')

V_ccH2 = CTVertex(name = 'V_ccH2',
                   particles = [ P.c__tilde__, P.c, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS4 ],
                   loop_particles = [[[P.c,P.H],[P.c,P.G0],[P.c,P.A],[P.c,P.Z],[P.G0, P.Z, P.c]],[[P.G__plus__,P.W__plus__,P.d],[P.G__plus__,P.d],[P.W__plus__, P.d]],[[P.G__plus__,P.W__plus__,P.s],[P.G__plus__,P.s],[P.W__plus__, P.s]],[[P.G__plus__,P.W__plus__,P.b],[P.G__plus__,P.b],[P.W__plus__,P.b]]],
                   couplings = {(0,0,0):C.R2_Hcc, (0,0,1):C.R2_Hcc_d, (0,0,2):C.R2_Hcc_s, (0,0,3):C.R2_Hcc_b},
                   type = 'R2')

V_ttH2 = CTVertex(name = 'V_ttH2',
                   particles = [ P.t__tilde__, P.t, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS4 ],
                   loop_particles = [[[P.t,P.H],[P.t,P.G0],[P.t,P.A],[P.t,P.Z],[P.G0, P.Z, P.t]],[[P.G__plus__,P.W__plus__,P.d],[P.G__plus__,P.d],[P.W__plus__, P.d]],[[P.G__plus__,P.W__plus__,P.s],[P.G__plus__,P.s],[P.W__plus__, P.s]],[[P.G__plus__,P.W__plus__,P.b],[P.G__plus__,P.b],[P.W__plus__,P.b]]],
                   couplings = {(0,0,0):C.R2_Htt, (0,0,1):C.R2_Htt_d, (0,0,2):C.R2_Htt_s, (0,0,3):C.R2_Htt_b},
                   type = 'R2')

V_ddH2 = CTVertex(name = 'V_ddH2',
                   particles = [ P.d__tilde__, P.d, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS4 ],
                   loop_particles = [[[P.d,P.H],[P.d,P.G0],[P.d,P.A],[P.d,P.Z],[P.G0, P.Z, P.d]],[[P.G__plus__,P.W__plus__,P.u],[P.G__plus__,P.u],[P.W__plus__, P.u]],[[P.G__plus__,P.W__plus__,P.c],[P.G__plus__,P.c],[P.W__plus__, P.c]],[[P.G__plus__,P.W__plus__,P.t],[P.G__plus__,P.t],[P.W__plus__,P.t]]],
                   couplings = {(0,0,0):C.R2_Hdd, (0,0,1):C.R2_Hdd_u, (0,0,2):C.R2_Hdd_c, (0,0,3):C.R2_Hdd_t},
                   type = 'R2')

V_ssH2 = CTVertex(name = 'V_ssH2',
                   particles = [ P.s__tilde__, P.s, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS4 ],
                   loop_particles = [[[P.s,P.H],[P.s,P.G0],[P.s,P.A],[P.s,P.Z],[P.G0, P.Z, P.s]],[[P.G__plus__,P.W__plus__,P.u],[P.G__plus__,P.u],[P.W__plus__, P.u]],[[P.G__plus__,P.W__plus__,P.c],[P.G__plus__,P.c],[P.W__plus__, P.c]],[[P.G__plus__,P.W__plus__,P.t],[P.G__plus__,P.t],[P.W__plus__,P.t]]],
                   couplings = {(0,0,0):C.R2_Hss, (0,0,1):C.R2_Hss_u, (0,0,2):C.R2_Hss_c, (0,0,3):C.R2_Hss_t},
                   type = 'R2')

V_bbH2 = CTVertex(name = 'V_bbH2',
                   particles = [ P.b__tilde__, P.b, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS4 ],
                   loop_particles = [[[P.b,P.H],[P.b,P.G0],[P.b,P.A],[P.b,P.Z],[P.G0, P.Z, P.b]],[[P.G__plus__,P.W__plus__,P.u],[P.G__plus__,P.u],[P.W__plus__, P.u]],[[P.G__plus__,P.W__plus__,P.c],[P.G__plus__,P.c],[P.W__plus__, P.c]],[[P.G__plus__,P.W__plus__,P.t],[P.G__plus__,P.t],[P.W__plus__,P.t]]],
                   couplings = {(0,0,0):C.R2_Hbb, (0,0,1):C.R2_Hbb_u, (0,0,2):C.R2_Hbb_c, (0,0,3):C.R2_Hbb_t},
                   type = 'R2')

V_eeH = CTVertex(name = 'V_eeH',
                   particles = [ P.e__plus__, P.e__minus__, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.FFS4 ],
                   loop_particles = [[[P.e__minus__,P.H],[P.e__minus__,P.G0],[P.e__minus__,P.A],[P.e__minus__,P.Z],[P.G0, P.Z, P.e__minus__]],[[P.W__plus__,P.G__plus__,P.ve],[P.G__plus__,P.ve],[P.W__plus__, P.ve]]],
                   couplings = {(0,0,0):C.R2_Hee, (0,0,1):C.R2_Hee_v},
                   type = 'R2')

V_mmH = CTVertex(name = 'V_mmH',
                   particles = [ P.m__plus__, P.m__minus__, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.FFS4 ],
                   loop_particles = [[[P.m__minus__,P.H],[P.m__minus__,P.G0],[P.m__minus__,P.A],[P.m__minus__,P.Z],[P.G0, P.Z, P.m__minus__]],[[P.G__plus__,P.W__plus__,P.vm],[P.G__plus__,P.vm],[P.W__plus__, P.vm]]],
                   couplings = {(0,0,0):C.R2_Hmm, (0,0,1):C.R2_Hmm_v},
                   type = 'R2')

V_tautauH = CTVertex(name = 'V_tautauH',
                   particles = [ P.tt__plus__, P.tt__minus__, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.FFS4 ],
                   loop_particles = [[[P.tt__minus__,P.H],[P.tt__minus__,P.G0],[P.tt__minus__,P.A],[P.tt__minus__,P.Z],[P.G0, P.Z, P.tt__minus__]],[[P.G__plus__,P.W__plus__,P.vt],[P.G__plus__,P.vt],[P.W__plus__, P.vt]]],
                   couplings = {(0,0,0):C.R2_Htautau, (0,0,1):C.R2_Htautau_v},
                   type = 'R2')

V_uuG02 = CTVertex(name = 'V_uuG02',
                   particles = [ P.u__tilde__, P.u, P.G0 ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.H,P.u],[P.G0,P.u],[P.A,P.u],[P.Z,P.u],[P.H, P.G0, P.u],[P.H,P.Z,P.u]],[[P.W__plus__,P.G__plus__,P.d],[P.G__plus__,P.d],[P.W__plus__, P.d]],[[P.W__plus__,P.G__plus__,P.s],[P.G__plus__,P.s],[P.W__plus__, P.s]],[[P.G__plus__,P.W__plus__,P.b],[P.G__plus__,P.b],[P.W__plus__,P.b]]],
                   couplings = {(0,0,0):C.R2_G0uu, (0,0,1):C.R2_G0uu_d, (0,0,2):C.R2_G0uu_s, (0,0,3):C.R2_G0uu_b},
                   type = 'R2')

V_ccG02 = CTVertex(name = 'V_ccG02',
                   particles = [ P.c__tilde__, P.c, P.G0 ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.H,P.c],[P.G0,P.c],[P.A,P.c],[P.Z,P.c],[P.H, P.G0, P.c],[P.H,P.Z,P.c]],[[P.W__plus__,P.G__plus__,P.d],[P.G__plus__,P.d],[P.W__plus__, P.d]],[[P.G__plus__,P.W__plus__,P.s],[P.G__plus__,P.s],[P.W__plus__, P.s]],[[P.W__plus__,P.G__plus__,P.b],[P.G__plus__,P.b],[P.W__plus__,P.b]]],
                   couplings = {(0,0,0):C.R2_G0cc, (0,0,1):C.R2_G0cc_d, (0,0,2):C.R2_G0cc_s, (0,0,3):C.R2_G0cc_b},
                   type = 'R2')

V_ttG02 = CTVertex(name = 'V_ttG02',
                   particles = [ P.t__tilde__, P.t, P.G0 ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.H,P.t],[P.G0,P.t],[P.A,P.t],[P.Z,P.t],[P.H, P.G0, P.t],[P.H,P.Z,P.t]],[[P.G__plus__,P.W__plus__,P.d],[P.G__plus__,P.d],[P.W__plus__, P.d]],[[P.W__plus__,P.G__plus__,P.s],[P.G__plus__,P.s],[P.W__plus__, P.s]],[[P.G__plus__,P.W__plus__,P.b],[P.G__plus__,P.b],[P.W__plus__,P.b]]],
                   couplings = {(0,0,0):C.R2_G0tt, (0,0,1):C.R2_G0tt_d, (0,0,2):C.R2_G0tt_s, (0,0,3):C.R2_G0tt_b},
                   type = 'R2')

V_ddG02 = CTVertex(name = 'V_ddG02',
                   particles = [ P.d__tilde__, P.d, P.G0 ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.d,P.H],[P.d,P.G0],[P.d,P.A],[P.d,P.Z],[P.H, P.G0, P.d],[P.H,P.Z,P.d]],[[P.G__plus__,P.W__plus__,P.u],[P.G__plus__,P.u],[P.W__plus__, P.u]],[[P.W__plus__,P.G__plus__,P.c],[P.G__plus__,P.c],[P.W__plus__, P.c]],[[P.G__plus__,P.W__plus__,P.t],[P.G__plus__,P.t],[P.W__plus__,P.t]]],
                   couplings = {(0,0,0):C.R2_G0dd, (0,0,1):C.R2_G0dd_u, (0,0,2):C.R2_G0dd_c, (0,0,3):C.R2_G0dd_t},
                   type = 'R2')

V_ssG02 = CTVertex(name = 'V_ssG02',
                   particles = [ P.s__tilde__, P.s, P.G0 ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.H,P.s],[P.G0,P.s],[P.A,P.s],[P.Z,P.s],[P.H,P.G0,P.s],[P.H,P.Z,P.s]],[[P.W__plus__,P.G__plus__,P.u],[P.G__plus__,P.u],[P.W__plus__, P.u]],[[P.W__plus__,P.G__plus__,P.c],[P.G__plus__,P.c],[P.W__plus__, P.c]],[[P.W__plus__,P.G__plus__,P.t],[P.G__plus__,P.t],[P.W__plus__,P.t]]],
                   couplings = {(0,0,0):C.R2_G0ss, (0,0,1):C.R2_G0ss_u, (0,0,2):C.R2_G0ss_c, (0,0,3):C.R2_G0ss_t},
                   type = 'R2')

V_bbG02 = CTVertex(name = 'V_bbG02',
                   particles = [ P.b__tilde__, P.b, P.G0 ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.H,P.b],[P.G0,P.b],[P.A,P.b],[P.Z,P.b],[P.H,P.G0,P.b],[P.H,P.Z,P.b]],[[P.W__plus__,P.G__plus__,P.u],[P.G__plus__,P.u],[P.W__plus__, P.u]],[[P.G__plus__,P.W__plus__,P.c],[P.G__plus__,P.c],[P.W__plus__, P.c]],[[P.W__plus__,P.G__plus__,P.t],[P.G__plus__,P.t],[P.W__plus__,P.t]]],
                   couplings = {(0,0,0):C.R2_G0bb, (0,0,1):C.R2_G0bb_u, (0,0,2):C.R2_G0bb_c, (0,0,3):C.R2_G0bb_t},
                   type = 'R2')

V_eeG0 = CTVertex(name = 'V_eeG0',
                   particles = [ P.e__plus__, P.e__minus__, P.G0 ],
                   color = [ '1' ],
                   lorentz = [ L.FFS4 ],
                   loop_particles = [[[P.H,P.e__minus__],[P.A,P.e__minus__],[P.G0,P.e__minus__],[P.Z,P.e__minus__],[P.H,P.G0,P.e__minus__],[P.H, P.Z, P.e__minus__]],[[P.W__plus__, P.G__plus__, P.ve]]],
                   couplings = {(0,0,0):C.R2_G0ee, (0,0,1):C.R2_G0ee_v},
                   type = 'R2')

V_mmG0 = CTVertex(name = 'V_mmG0',
                   particles = [ P.m__plus__, P.m__minus__, P.G0 ],
                   color = [ '1' ],
                   lorentz = [ L.FFS4 ],
                   loop_particles = [[[P.H,P.m__minus__],[P.G0,P.m__minus__],[P.A,P.m__minus__],[P.Z,P.m__minus__],[P.H, P.Z, P.m__minus__],[P.H, P.G0, P.m__minus__]],[[P.W__plus__, P.G__plus__,P.vm]]],
                   couplings = {(0,0,0):C.R2_G0mm, (0,0,1):C.R2_G0mm_v},
                   type = 'R2')

V_tautauG0 = CTVertex(name = 'V_tautauG0',
                   particles = [ P.tt__plus__, P.tt__minus__, P.G0 ],
                   color = [ '1' ],
                   lorentz = [ L.FFS4 ],
                   loop_particles = [[[P.H,P.tt__minus__],[P.G0,P.tt__minus__],[P.A,P.tt__minus__],[P.Z,P.tt__minus__],[P.H,P.Z, P.tt__minus__],[P.H,P.G0, P.tt__minus__]],[[P.W__plus__,P.G__plus__,P.vt]]],
                   couplings = {(0,0,0):C.R2_G0tautau, (0,0,1):C.R2_G0tautau_v},
                   type = 'R2')

V_uxdGp2 = CTVertex(name = 'V_uxdGp2',
                    particles = [ P.u__tilde__, P.d, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.d,P.u,P.H],[P.d,P.u,P.G0],[P.u,P.H,P.G__plus__],[P.d,P.G__plus__,P.H],[P.A,P.u,P.d],[P.Z,P.d,P.u],[P.u,P.A,P.G__plus__],[P.d,P.A,P.G__plus__],[P.d,P.Z,P.G__plus__],[P.u,P.Z,P.G__plus__],[P.H,P.W__plus__,P.d],[P.H,P.W__plus__,P.u],[P.G0,P.W__plus__,P.u],[P.G0,P.W__plus__,P.d],[P.W__plus__,P.A,P.u],[P.W__plus__,P.A,P.d],[P.W__plus__,P.Z,P.u],[P.W__plus__,P.Z,P.d]]],
                    couplings = {(0,0,0):C.R2_uxdGp2Cm,(0,1,0):C.R2_uxdGp2Cp},
                    type = 'R2')

V_uxsGp2 = CTVertex(name = 'V_uxsGp2',
                    particles = [ P.u__tilde__, P.s, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.s,P.u,P.H],[P.s,P.u,P.G0],[P.u,P.H,P.G__plus__],[P.s,P.G__plus__,P.H],[P.A,P.u,P.s],[P.Z,P.u,P.s],[P.u,P.A,P.G__plus__],[P.s,P.A,P.G__plus__],[P.G0,P.W__plus__,P.u],[P.G0,P.W__plus__,P.s],[P.W__plus__,P.A,P.u],[P.W__plus__,P.A,P.s],[P.W__plus__,P.Z,P.u],[P.W__plus__,P.Z, P.s]]],
                    couplings = {(0,0,0):C.R2_uxsGp2Cm,(0,1,0):C.R2_uxsGp2Cp},
                    type = 'R2')

V_uxbGp2 = CTVertex(name = 'V_uxbGp2',
                    particles = [ P.u__tilde__, P.b, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.b,P.u,P.H],[P.b,P.u,P.G0],[P.u,P.H,P.G__plus__],[P.b,P.G__plus__,P.H],[P.A,P.u,P.b],[P.Z,P.u,P.b],[P.u,P.A,P.G__plus__],[P.b,P.A,P.G__plus__],[P.G0,P.W__plus__,P.u],[P.G0,P.W__plus__,P.b],[P.W__plus__,P.A,P.u],[P.W__plus__,P.A,P.b],[P.W__plus__,P.Z,P.u],[P.W__plus__,P.Z, P.b]]],
                    couplings = {(0,0,0):C.R2_uxbGp2Cm,(0,1,0):C.R2_uxbGp2Cp},
                    type = 'R2')

V_cxdGp2 = CTVertex(name = 'V_cxdGp2',
                    particles = [ P.c__tilde__, P.d, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.d,P.c,P.H],[P.d,P.c,P.G0],[P.c,P.H,P.G__plus__],[P.d,P.G__plus__,P.H],[P.A,P.c,P.d],[P.Z,P.c,P.d],[P.c,P.A,P.G__plus__],[P.d,P.A,P.G__plus__],[P.G0,P.W__plus__,P.c],[P.G0,P.W__plus__,P.d],[P.W__plus__,P.A,P.c],[P.W__plus__,P.A,P.d],[P.W__plus__,P.Z,P.c],[P.W__plus__,P.Z, P.d]]],
                    couplings = {(0,0,0):C.R2_cxdGp2Cm,(0,1,0):C.R2_cxdGp2Cp},
                    type = 'R2')

V_cxsGp2 = CTVertex(name = 'V_cxsGp2',
                    particles = [ P.c__tilde__, P.s, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.s,P.c,P.H],[P.s,P.c,P.G0],[P.c,P.H,P.G__plus__],[P.s,P.G__plus__,P.H],[P.A,P.c,P.s],[P.Z,P.c,P.s],[P.c,P.A,P.G__plus__],[P.s,P.A,P.G__plus__],[P.G0,P.W__plus__,P.c],[P.G0,P.W__plus__,P.s],[P.W__plus__,P.A,P.c],[P.W__plus__,P.A,P.s],[P.W__plus__,P.Z,P.c],[P.W__plus__,P.Z, P.s]]],
                    couplings = {(0,0,0):C.R2_cxsGp2Cm,(0,1,0):C.R2_cxsGp2Cp},
                    type = 'R2')

V_cxbGp2 = CTVertex(name = 'V_cxbGp2',
                    particles = [ P.c__tilde__, P.b, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.b,P.c,P.H],[P.b,P.c,P.G0],[P.c,P.H,P.G__plus__],[P.b,P.G__plus__,P.H],[P.A,P.c,P.b],[P.Z,P.c,P.b],[P.c,P.A,P.G__plus__],[P.b,P.A,P.G__plus__],[P.G0,P.W__plus__,P.c],[P.G0,P.W__plus__,P.b],[P.W__plus__,P.A,P.c],[P.W__plus__,P.A,P.b],[P.W__plus__,P.Z,P.c],[P.W__plus__,P.Z, P.b]]],
                    couplings = {(0,0,0):C.R2_cxbGp2Cm,(0,1,0):C.R2_cxbGp2Cp},
                    type = 'R2')

V_txdGp2 = CTVertex(name = 'V_txdGp2',
                    particles = [ P.t__tilde__, P.d, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.d,P.t,P.H],[P.d,P.t,P.G0],[P.t,P.H,P.G__plus__],[P.d,P.G__plus__,P.H],[P.A,P.t,P.d],[P.Z,P.t,P.d],[P.t,P.A,P.G__plus__],[P.d,P.A,P.G__plus__],[P.G0,P.W__plus__,P.t],[P.G0,P.W__plus__,P.d],[P.W__plus__,P.A,P.t],[P.W__plus__,P.A,P.d],[P.W__plus__,P.Z,P.t],[P.W__plus__,P.Z, P.d]]],
                    couplings = {(0,0,0):C.R2_txdGp2Cm,(0,1,0):C.R2_txdGp2Cp},
                    type = 'R2')

V_txsGp2 = CTVertex(name = 'V_txsGp2',
                    particles = [ P.t__tilde__, P.s, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.s,P.t,P.H],[P.s,P.t,P.G0],[P.t,P.H,P.G__plus__],[P.s,P.G__plus__,P.H],[P.A,P.t,P.s],[P.Z,P.t,P.s],[P.t,P.A,P.G__plus__],[P.s,P.A,P.G__plus__],[P.G0,P.W__plus__,P.t],[P.G0,P.W__plus__,P.s],[P.W__plus__,P.A,P.t],[P.W__plus__,P.A,P.s],[P.W__plus__,P.Z,P.t],[P.W__plus__,P.Z, P.s]]],
                    couplings = {(0,0,0):C.R2_txsGp2Cm,(0,1,0):C.R2_txsGp2Cp},
                    type = 'R2')

V_txbGp2 = CTVertex(name = 'V_txbGp2',
                    particles = [ P.t__tilde__, P.b, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.b,P.t,P.H],[P.b,P.t,P.G0],[P.t,P.H,P.G__plus__],[P.b,P.G__plus__,P.H],[P.A,P.t,P.b],[P.Z,P.t,P.b],[P.t,P.A,P.G__plus__],[P.b,P.A,P.G__plus__],[P.G0,P.W__plus__,P.t],[P.G0,P.W__plus__,P.b],[P.W__plus__,P.A,P.t],[P.W__plus__,P.A,P.b],[P.W__plus__,P.Z,P.t],[P.W__plus__,P.Z, P.b]]],
                    couplings = {(0,0,0):C.R2_txbGp2Cm,(0,1,0):C.R2_txbGp2Cp},
                    type = 'R2')

V_dxuGm2 = CTVertex(name = 'V_dxuGm2',
                    particles = [ P.d__tilde__, P.u, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.d,P.u,P.H],[P.d,P.u,P.G0],[P.u,P.H,P.G__plus__],[P.d,P.G__plus__,P.H],[P.A,P.u,P.d],[P.Z,P.u,P.d],[P.u,P.A,P.G__plus__],[P.d,P.A,P.G__plus__],[P.G0,P.W__plus__,P.u],[P.G0,P.W__plus__,P.d],[P.W__plus__,P.A,P.u],[P.W__plus__,P.A,P.d],[P.W__plus__,P.Z,P.u],[P.W__plus__,P.Z, P.d]]],
                    couplings = {(0,0,0):C.R2_dxuGm2Cm,(0,1,0):C.R2_dxuGm2Cp},
                    type = 'R2')

V_sxuGm2 = CTVertex(name = 'V_sxuGm2',
                    particles = [ P.s__tilde__, P.u, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.s,P.u,P.H],[P.s,P.u,P.G0],[P.u,P.H,P.G__plus__],[P.s,P.G__plus__,P.H],[P.A,P.u,P.s],[P.Z,P.u,P.s],[P.u,P.A,P.G__plus__],[P.s,P.A,P.G__plus__],[P.G0,P.W__plus__,P.u],[P.G0,P.W__plus__,P.s],[P.W__plus__,P.A,P.u],[P.W__plus__,P.A,P.s],[P.W__plus__,P.Z,P.u],[P.W__plus__,P.Z, P.s]]],
                    couplings = {(0,0,0):C.R2_sxuGm2Cm,(0,1,0):C.R2_sxuGm2Cp},
                    type = 'R2')

V_bxuGm2 = CTVertex(name = 'V_bxuGm2',
                    particles = [ P.b__tilde__, P.u, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.b,P.u,P.H],[P.b,P.u,P.G0],[P.u,P.H,P.G__plus__],[P.b,P.G__plus__,P.H],[P.A,P.u,P.b],[P.Z,P.u,P.b],[P.u,P.A,P.G__plus__],[P.b,P.A,P.G__plus__],[P.G0,P.W__plus__,P.u],[P.G0,P.W__plus__,P.b],[P.W__plus__,P.A,P.u],[P.W__plus__,P.A,P.b],[P.W__plus__,P.Z,P.u],[P.W__plus__,P.Z, P.b]]],
                    couplings = {(0,0,0):C.R2_bxuGm2Cm,(0,1,0):C.R2_bxuGm2Cp},
                    type = 'R2')

V_dxcGm2 = CTVertex(name = 'V_dxcGm2',
                    particles = [ P.d__tilde__, P.c, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.d,P.c,P.H],[P.d,P.c,P.G0],[P.c,P.H,P.G__plus__],[P.d,P.G__plus__,P.H],[P.A,P.c,P.d],[P.Z,P.c,P.d],[P.c,P.A,P.G__plus__],[P.d,P.A,P.G__plus__],[P.G0,P.W__plus__,P.c],[P.G0,P.W__plus__,P.d],[P.W__plus__,P.A,P.c],[P.W__plus__,P.A,P.d],[P.W__plus__,P.Z,P.c],[P.W__plus__,P.Z, P.d]]],
                    couplings = {(0,0,0):C.R2_dxcGm2Cm,(0,1,0):C.R2_dxcGm2Cp},
                    type = 'R2')

V_sxcGm2 = CTVertex(name = 'V_sxcGm2',
                    particles = [ P.s__tilde__, P.c, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.s,P.c,P.H],[P.s,P.c,P.G0],[P.c,P.H,P.G__plus__],[P.s,P.G__plus__,P.H],[P.A,P.c,P.s],[P.Z,P.c,P.s],[P.c,P.A,P.G__plus__],[P.s,P.A,P.G__plus__],[P.G0,P.W__plus__,P.c],[P.G0,P.W__plus__,P.s],[P.W__plus__,P.A,P.c],[P.W__plus__,P.A,P.s],[P.W__plus__,P.Z,P.c],[P.W__plus__,P.Z, P.s]]],
                    couplings = {(0,0,0):C.R2_sxcGm2Cm,(0,1,0):C.R2_sxcGm2Cp},
                    type = 'R2')

V_bxcGm2 = CTVertex(name = 'V_bxcGm2',
                    particles = [ P.b__tilde__, P.c, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.b,P.c,P.H],[P.b,P.c,P.G0],[P.c,P.H,P.G__plus__],[P.b,P.G__plus__,P.H],[P.A,P.c,P.b],[P.Z,P.c,P.b],[P.c,P.A,P.G__plus__],[P.b,P.A,P.G__plus__],[P.G0,P.W__plus__,P.c],[P.G0,P.W__plus__,P.b],[P.W__plus__,P.A,P.c],[P.W__plus__,P.A,P.b],[P.W__plus__,P.Z,P.c],[P.W__plus__,P.Z, P.b]]],
                    couplings = {(0,0,0):C.R2_bxcGm2Cm,(0,1,0):C.R2_bxcGm2Cp},
                    type = 'R2')

V_dxtGm2 = CTVertex(name = 'V_dxtGm2',
                    particles = [ P.d__tilde__, P.t, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.d,P.t,P.H],[P.d,P.t,P.G0],[P.t,P.H,P.G__plus__],[P.d,P.G__plus__,P.H],[P.A,P.t,P.d],[P.Z,P.t,P.d],[P.t,P.A,P.G__plus__],[P.d,P.A,P.G__plus__],[P.G0,P.W__plus__,P.t],[P.G0,P.W__plus__,P.d],[P.W__plus__,P.A,P.t],[P.W__plus__,P.A,P.d],[P.W__plus__,P.Z,P.t],[P.W__plus__,P.Z, P.d]]],
                    couplings = {(0,0,0):C.R2_dxtGm2Cm,(0,1,0):C.R2_dxtGm2Cp},
                    type = 'R2')

V_sxtGm2 = CTVertex(name = 'V_sxtGm2',
                    particles = [ P.s__tilde__, P.t, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.s,P.t,P.H],[P.s,P.t,P.G0],[P.t,P.H,P.G__plus__],[P.s,P.G__plus__,P.H],[P.A,P.t,P.s],[P.Z,P.t,P.s],[P.t,P.A,P.G__plus__],[P.s,P.A,P.G__plus__],[P.G0,P.W__plus__,P.t],[P.G0,P.W__plus__,P.s],[P.W__plus__,P.A,P.t],[P.W__plus__,P.A,P.s],[P.W__plus__,P.Z,P.t],[P.W__plus__,P.Z, P.s]]],
                    couplings = {(0,0,0):C.R2_sxtGm2Cm,(0,1,0):C.R2_sxtGm2Cp},
                    type = 'R2')

V_bxtGm2 = CTVertex(name = 'V_bxtGm2',
                    particles = [ P.b__tilde__, P.t, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.b,P.t,P.H],[P.b,P.t,P.G0],[P.t,P.H,P.G__plus__],[P.b,P.G__plus__,P.H],[P.A,P.t,P.b],[P.Z,P.t,P.b],[P.t,P.A,P.G__plus__],[P.b,P.A,P.G__plus__],[P.G0,P.W__plus__,P.t],[P.G0,P.W__plus__,P.b],[P.W__plus__,P.A,P.t],[P.W__plus__,P.A,P.b],[P.W__plus__,P.Z,P.t],[P.W__plus__,P.Z, P.b]]],
                    couplings = {(0,0,0):C.R2_bxtGm2Cm,(0,1,0):C.R2_bxtGm2Cp},
                    type = 'R2')

V_vexeGp = CTVertex(name = 'V_vexeGp',
                    particles = [ P.ve__tilde__, P.e__minus__, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.FFS1 ],
                    loop_particles = [[[P.H,P.G__plus__,P.e__minus__],[P.Z,P.e__minus__,P.ve],[P.G__plus__,P.Z,P.ve],[P.H,P.W__plus__,P.e__minus__],[P.G0,P.W__plus__,P.e__minus__],[P.A,P.G__plus__,P.e__minus__],[P.Z,P.G__plus__,P.e__minus__],[P.A,P.W__plus__,P.e__minus__],[P.Z,P.W__plus__,P.e__minus__],[P.Z,P.W__plus__,P.ve]]],
                    couplings = {(0,0,0):C.R2_vexeGpCm},
                    type = 'R2')

V_vmxmGp = CTVertex(name = 'V_vmxmGp',
                    particles = [ P.vm__tilde__, P.m__minus__, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.FFS1 ],
                    loop_particles = [[[P.H,P.G__plus__,P.m__minus__],[P.Z,P.m__minus__,P.vm],[P.G__plus__,P.Z,P.vm],[P.H,P.W__plus__,P.m__minus__],[P.G0,P.W__plus__,P.m__minus__],[P.A,P.G__plus__,P.m__minus__],[P.Z,P.G__plus__,P.m__minus__],[P.A,P.W__plus__,P.m__minus__],[P.Z,P.W__plus__,P.m__minus__],[P.Z,P.W__plus__,P.vm]]],
                    couplings = {(0,0,0):C.R2_vmxmGpCm},
                    type = 'R2')

V_vtxtauGp = CTVertex(name = 'V_vtxtauGp',
                    particles = [ P.vt__tilde__, P.tt__minus__, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.FFS1 ],
                    loop_particles = [[[P.H,P.G__plus__,P.tt__minus__],[P.Z,P.tt__minus__,P.vt],[P.G__plus__,P.Z,P.vt],[P.H,P.W__plus__,P.tt__minus__],[P.G0,P.W__plus__,P.tt__minus__],[P.A,P.G__plus__,P.tt__minus__],[P.Z,P.G__plus__,P.tt__minus__],[P.A,P.W__plus__,P.tt__minus__],[P.Z,P.W__plus__,P.tt__minus__],[P.Z,P.W__plus__,P.vt]]],
                    couplings = {(0,0,0):C.R2_vtxtauGpCm},
                    type = 'R2')

V_exveGm = CTVertex(name = 'V_exveGm',
                    particles = [ P.e__plus__, P.ve, P.G__minus__ ],
                    color = [ '1' ],
                    lorentz = [ L.FFS3 ],
                    loop_particles = [[[P.H,P.G__plus__,P.e__minus__],[P.Z,P.e__minus__,P.ve],[P.G__plus__,P.Z,P.ve],[P.H,P.W__plus__,P.e__minus__],[P.G0,P.W__plus__,P.e__minus__],[P.A,P.G__plus__,P.e__minus__],[P.Z,P.G__plus__,P.e__minus__],[P.A,P.W__plus__,P.e__minus__],[P.Z,P.W__plus__,P.e__minus__],[P.Z,P.W__plus__,P.ve]]],
                    couplings = {(0,0,0):C.R2_exveGmCp},
                    type = 'R2')

V_mxvmGm = CTVertex(name = 'V_mxvmGm',
                    particles = [ P.m__plus__, P.vm, P.G__minus__ ],
                    color = [ '1' ],
                    lorentz = [ L.FFS3 ],
                    loop_particles = [[[P.H,P.G__plus__,P.m__minus__],[P.Z,P.m__minus__,P.vm],[P.G__plus__,P.Z,P.vm],[P.H,P.W__plus__,P.m__minus__],[P.G0,P.W__plus__,P.m__minus__],[P.A,P.G__plus__,P.m__minus__],[P.Z,P.G__plus__,P.m__minus__],[P.A,P.W__plus__,P.m__minus__],[P.Z,P.W__plus__,P.m__minus__],[P.Z,P.W__plus__,P.vm]]],
                    couplings = {(0,0,0):C.R2_mxvmGmCp},
                    type = 'R2')

V_tauxvtGm = CTVertex(name = 'V_tauxvtGm',
                    particles = [ P.tt__plus__, P.vt, P.G__minus__ ],
                    color = [ '1' ],
                    lorentz = [ L.FFS3 ],
                    loop_particles = [[[P.H,P.G__plus__,P.tt__minus__],[P.Z,P.tt__minus__,P.vt],[P.G__plus__,P.Z,P.vt],[P.H,P.W__plus__,P.tt__minus__],[P.G0,P.W__plus__,P.tt__minus__],[P.A,P.G__plus__,P.tt__minus__],[P.Z,P.G__plus__,P.tt__minus__],[P.A,P.W__plus__,P.tt__minus__],[P.Z,P.W__plus__,P.tt__minus__],[P.Z,P.W__plus__,P.vt]]],
                    couplings = {(0,0,0):C.R2_tauxvtGmCp},
                    type = 'R2')

# R2 for VFF~

V_R2ddA2 = CTVertex(name = 'V_R2ddA2',
              particles = [ P.d__tilde__, P.d, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.d,P.H],[P.d,P.G0],[P.d,P.A],[P.d,P.Z]],[[P.G__plus__,P.W__plus__,P.u],[P.G__plus__,P.u],[P.W__plus__,P.u]],[[P.G__plus__,P.W__plus__,P.c],[P.G__plus__,P.c],[P.W__plus__,P.c]],[[P.G__plus__,P.W__plus__,P.t],[P.G__plus__,P.t],[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_ddA2Cp, (0,0,1):C.R2_ddA2Cp_u, (0,0,2):C.R2_ddA2Cp_c,(0,0,3):C.R2_ddA2Cp_t,
                           (0,1,0):C.R2_ddA2Cm, (0,1,1):C.R2_ddA2Cm_u, (0,1,2):C.R2_ddA2Cm_c,(0,1,3):C.R2_ddA2Cm_t},
              type = 'R2')

V_R2ssA2 = CTVertex(name = 'V_R2ssA2',
              particles = [ P.s__tilde__, P.s, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.s,P.H],[P.s,P.G0],[P.s,P.A],[P.s,P.Z]],[[P.G__plus__,P.W__plus__,P.u],[P.G__plus__,P.u],[P.W__plus__,P.u]],[[P.G__plus__,P.W__plus__,P.c],[P.G__plus__,P.c],[P.W__plus__,P.c]],[[P.G__plus__,P.W__plus__,P.t],[P.G__plus__,P.t],[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_ssA2Cp, (0,0,1):C.R2_ssA2Cp_u, (0,0,2):C.R2_ssA2Cp_c,(0,0,3):C.R2_ssA2Cp_t,
                           (0,1,0):C.R2_ssA2Cm, (0,1,1):C.R2_ssA2Cm_u, (0,1,2):C.R2_ssA2Cm_c,(0,1,3):C.R2_ssA2Cm_t},
              type = 'R2')

V_R2bbA2 = CTVertex(name = 'V_R2bbA2',
              particles = [ P.b__tilde__, P.b, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.b,P.H],[P.b,P.G0],[P.b,P.A],[P.b,P.Z]],[[P.G__plus__,P.W__plus__,P.u],[P.G__plus__,P.u],[P.W__plus__,P.u]],[[P.G__plus__,P.W__plus__,P.c],[P.G__plus__,P.c],[P.W__plus__,P.c]],[[P.G__plus__,P.W__plus__,P.t],[P.G__plus__,P.t],[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_bbA2Cp, (0,0,1):C.R2_bbA2Cp_u, (0,0,2):C.R2_bbA2Cp_c,(0,0,3):C.R2_bbA2Cp_t,
                           (0,1,0):C.R2_bbA2Cm, (0,1,1):C.R2_bbA2Cm_u, (0,1,2):C.R2_bbA2Cm_c,(0,1,3):C.R2_bbA2Cm_t},
              type = 'R2')

V_R2uuA2 = CTVertex(name = 'V_R2uuA2',
              particles = [ P.u__tilde__, P.u, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.u,P.H],[P.u,P.G0],[P.u,P.A],[P.u,P.Z]],[[P.G__plus__,P.W__plus__,P.d],[P.G__plus__,P.d],[P.W__plus__,P.d]],[[P.G__plus__,P.W__plus__,P.s],[P.G__plus__,P.s],[P.W__plus__,P.s]],[[P.G__plus__,P.W__plus__,P.b],[P.G__plus__,P.b],[P.W__plus__,P.b]]],
              couplings = {(0,0,0):C.R2_uuA2Cp, (0,0,1):C.R2_uuA2Cp_d, (0,0,2):C.R2_uuA2Cp_s,(0,0,3):C.R2_uuA2Cp_b,
                           (0,1,0):C.R2_uuA2Cm, (0,1,1):C.R2_uuA2Cm_d, (0,1,2):C.R2_uuA2Cm_s,(0,1,3):C.R2_uuA2Cm_b},
              type = 'R2')

V_R2ccA2 = CTVertex(name = 'V_R2ccA2',
              particles = [ P.c__tilde__, P.c, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.c,P.H],[P.c,P.G0],[P.c,P.A],[P.c,P.Z]],[[P.G__plus__,P.W__plus__,P.d],[P.G__plus__,P.d],[P.W__plus__,P.d]],[[P.G__plus__,P.W__plus__,P.s],[P.G__plus__,P.s],[P.W__plus__,P.s]],[[P.G__plus__,P.W__plus__,P.b],[P.G__plus__,P.b],[P.W__plus__,P.b]]],
              couplings = {(0,0,0):C.R2_ccA2Cp, (0,0,1):C.R2_ccA2Cp_d, (0,0,2):C.R2_ccA2Cp_s,(0,0,3):C.R2_ccA2Cp_b,
                           (0,1,0):C.R2_ccA2Cm, (0,1,1):C.R2_ccA2Cm_d, (0,1,2):C.R2_ccA2Cm_s,(0,1,3):C.R2_ccA2Cm_b},
              type = 'R2')

V_R2ttA2 = CTVertex(name = 'V_R2ttA2',
              particles = [ P.t__tilde__, P.t, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.c,P.H],[P.c,P.G0],[P.c,P.A],[P.c,P.Z]],[[P.G__plus__,P.W__plus__,P.d],[P.G__plus__,P.d],[P.W__plus__,P.d]],[[P.G__plus__,P.W__plus__,P.s],[P.G__plus__,P.s],[P.W__plus__,P.s]],[[P.G__plus__,P.W__plus__,P.b],[P.G__plus__,P.b],[P.W__plus__,P.b]]],
              couplings = {(0,0,0):C.R2_ttA2Cp, (0,0,1):C.R2_ttA2Cp_d, (0,0,2):C.R2_ttA2Cp_s,(0,0,3):C.R2_ttA2Cp_b,
                           (0,1,0):C.R2_ttA2Cm, (0,1,1):C.R2_ttA2Cm_d, (0,1,2):C.R2_ttA2Cm_s,(0,1,3):C.R2_ttA2Cm_b},
              type = 'R2')

V_R2eeA = CTVertex(name = 'V_R2eeA',
              particles = [ P.e__plus__, P.e__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.e__minus__,P.H],[P.e__minus__,P.G0],[P.e__minus__,P.A],[P.e__minus__,P.Z]],[[P.G__plus__,P.W__plus__,P.ve],[P.G__plus__,P.ve],[P.W__plus__,P.ve]]],
              couplings = {(0,0,0):C.R2_eeACp, (0,0,1):C.R2_llACp,(0,1,0):C.R2_eeACm},
              type = 'R2')

V_R2mmA = CTVertex(name = 'V_R2mmA',
              particles = [ P.m__plus__, P.m__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.m__minus__,P.H],[P.m__minus__,P.G0],[P.m__minus__,P.A],[P.m__minus__,P.Z]],[[P.G__plus__,P.W__plus__,P.vm],[P.G__plus__,P.vm],[P.W__plus__,P.vm]]],
              couplings = {(0,0,0):C.R2_mmACp, (0,0,1):C.R2_llACp,(0,1,0):C.R2_mmACm},
              type = 'R2')

V_R2tautauA = CTVertex(name = 'V_R2tautauA',
              particles = [ P.tt__plus__, P.tt__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.tt__minus__,P.H],[P.tt__minus__,P.G0],[P.tt__minus__,P.A],[P.tt__minus__,P.Z]],[[P.G__plus__,P.W__plus__,P.vt],[P.G__plus__,P.vt],[P.W__plus__,P.vt]]],
              couplings = {(0,0,0):C.R2_tautauACp, (0,0,1):C.R2_llACp,(0,1,0):C.R2_tautauACm},
              type = 'R2')

V_R2veveA = CTVertex(name = 'V_R2veveA',
              particles = [ P.ve__tilde__, P.ve, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.G__plus__,P.W__plus__,P.e__minus__],[P.G__plus__,P.e__minus__],[P.W__plus__,P.e__minus__]]],
              couplings = {(0,0,0):C.R2_veveACp},
              type = 'R2')

V_R2vmvmA = CTVertex(name = 'V_R2vmvmA',
              particles = [ P.vm__tilde__, P.vm, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.W__plus__,P.G__plus__,P.m__minus__],[P.G__plus__,P.m__minus__],[P.W__plus__,P.m__minus__]]],
              couplings = {(0,0,0):C.R2_vmvmACp},
              type = 'R2')

V_R2vtvtA = CTVertex(name = 'V_R2vtvtA',
              particles = [ P.vt__tilde__, P.vt, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.G__plus__,P.W__plus__,P.tt__minus__],[P.G__plus__,P.tt__minus__],[P.W__plus__,P.tt__minus__]]],
              couplings = {(0,0,0):C.R2_vtvtACp},
              type = 'R2')

V_R2ddZ2 = CTVertex(name = 'V_R2ddZ2',
              particles = [ P.d__tilde__, P.d, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.d,P.H],[P.d,P.G0],[P.H,P.G0,P.d],[P.d,P.A],[P.d,P.Z],[P.H,P.Z,P.d]],[[P.G__plus__,P.W__plus__,P.u],[P.G__plus__,P.u],[P.W__plus__,P.u]],[[P.W__plus__,P.G__plus__,P.c],[P.G__plus__,P.c],[P.W__plus__,P.c]],[[P.W__plus__,P.G__plus__,P.t],[P.G__plus__,P.t],[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_ddZ2Cp, (0,0,1):C.R2_ddZ2Cp_u, (0,0,2):C.R2_ddZ2Cp_c,(0,0,3):C.R2_ddZ2Cp_t,
                           (0,1,0):C.R2_ddZ2Cm, (0,1,1):C.R2_ddZ2Cm_u, (0,1,2):C.R2_ddZ2Cm_c,(0,1,3):C.R2_ddZ2Cm_t},
              type = 'R2')

V_R2ssZ2 = CTVertex(name = 'V_R2ssZ2',
              particles = [ P.s__tilde__, P.s, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.s,P.H],[P.s,P.G0],[P.H,P.G0,P.s],[P.s,P.A],[P.s,P.Z],[P.H,P.Z,P.s]],[[P.G__plus__,P.W__plus__,P.u],[P.G__plus__,P.u],[P.W__plus__,P.u]],[[P.W__plus__,P.G__plus__,P.c],[P.G__plus__,P.c],[P.W__plus__,P.c]],[[P.W__plus__,P.G__plus__,P.t],[P.G__plus__,P.t],[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_ssZ2Cp, (0,0,1):C.R2_ssZ2Cp_u, (0,0,2):C.R2_ssZ2Cp_c,(0,0,3):C.R2_ssZ2Cp_t,
                           (0,1,0):C.R2_ssZ2Cm, (0,1,1):C.R2_ssZ2Cm_u, (0,1,2):C.R2_ssZ2Cm_c,(0,1,3):C.R2_ssZ2Cm_t},
              type = 'R2')

V_R2bbZ2 = CTVertex(name = 'V_R2bbZ2',
              particles = [ P.b__tilde__, P.b, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.b,P.H],[P.b,P.G0],[P.H,P.G0,P.b],[P.b,P.A],[P.b,P.Z],[P.H,P.Z,P.b]],[[P.G__plus__,P.W__plus__,P.u],[P.G__plus__,P.u],[P.W__plus__,P.u]],[[P.W__plus__,P.G__plus__,P.c],[P.G__plus__,P.c],[P.W__plus__,P.c]],[[P.W__plus__,P.G__plus__,P.t],[P.G__plus__,P.t],[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_bbZ2Cp, (0,0,1):C.R2_bbZ2Cp_u, (0,0,2):C.R2_bbZ2Cp_c,(0,0,3):C.R2_bbZ2Cp_t,
                           (0,1,0):C.R2_bbZ2Cm, (0,1,1):C.R2_bbZ2Cm_u, (0,1,2):C.R2_bbZ2Cm_c,(0,1,3):C.R2_bbZ2Cm_t},
              type = 'R2')

V_R2uuZ2 = CTVertex(name = 'V_R2uuZ2',
              particles = [ P.u__tilde__, P.u, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.u,P.H],[P.u,P.G0],[P.H,P.G0,P.u],[P.u,P.A],[P.u,P.Z],[P.H,P.Z,P.u]],[[P.W__plus__,P.G__plus__,P.d],[P.G__plus__,P.d],[P.W__plus__,P.d]],[[P.W__plus__,P.G__plus__,P.s],[P.G__plus__,P.s],[P.W__plus__,P.s]],[[P.G__plus__,P.W__plus__,P.b],[P.G__plus__,P.b],[P.W__plus__,P.b]]],
              couplings = {(0,0,0):C.R2_uuZ2Cp, (0,0,1):C.R2_uuZ2Cp_d, (0,0,2):C.R2_uuZ2Cp_s,(0,0,3):C.R2_uuZ2Cp_b,
                           (0,1,0):C.R2_uuZ2Cm, (0,1,1):C.R2_uuZ2Cm_d, (0,1,2):C.R2_uuZ2Cm_s,(0,1,3):C.R2_uuZ2Cm_b},
              type = 'R2')

V_R2ccZ2 = CTVertex(name = 'V_R2ccZ2',
              particles = [ P.c__tilde__, P.c, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.c,P.H],[P.c,P.G0],[P.H,P.G0,P.c],[P.c,P.A],[P.c,P.Z],[P.H,P.Z,P.c]],[[P.W__plus__,P.G__plus__,P.d],[P.G__plus__,P.d],[P.W__plus__,P.d]],[[P.W__plus__,P.G__plus__,P.s],[P.G__plus__,P.s],[P.W__plus__,P.s]],[[P.G__plus__,P.W__plus__,P.b],[P.G__plus__,P.b],[P.W__plus__,P.b]]],
              couplings = {(0,0,0):C.R2_ccZ2Cp, (0,0,1):C.R2_ccZ2Cp_d, (0,0,2):C.R2_ccZ2Cp_s,(0,0,3):C.R2_ccZ2Cp_b,
                           (0,1,0):C.R2_ccZ2Cm, (0,1,1):C.R2_ccZ2Cm_d, (0,1,2):C.R2_ccZ2Cm_s,(0,1,3):C.R2_ccZ2Cm_b},
              type = 'R2')

V_R2ttZ2 = CTVertex(name = 'V_R2ttZ2',
              particles = [ P.t__tilde__, P.t, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.t,P.H],[P.t,P.G0],[P.H,P.G0,P.t],[P.t,P.A],[P.t,P.Z],[P.H,P.Z,P.t]],[[P.W__plus__,P.G__plus__,P.d],[P.G__plus__,P.d],[P.W__plus__,P.d]],[[P.W__plus__,P.G__plus__,P.s],[P.G__plus__,P.s],[P.W__plus__,P.s]],[[P.G__plus__,P.W__plus__,P.b],[P.G__plus__,P.b],[P.W__plus__,P.b]]],
              couplings = {(0,0,0):C.R2_ttZ2Cp, (0,0,1):C.R2_ttZ2Cp_d, (0,0,2):C.R2_ttZ2Cp_s,(0,0,3):C.R2_ttZ2Cp_b,
                           (0,1,0):C.R2_ttZ2Cm, (0,1,1):C.R2_ttZ2Cm_d, (0,1,2):C.R2_ttZ2Cm_s,(0,1,3):C.R2_ttZ2Cm_b},
              type = 'R2')

######################################################################################################
#
# HSS 21/09/2012
# Following, we just select one diagram instead of treating diagrams equally as above
#
######################################################################################################

V_R2eeZ = CTVertex(name = 'V_R2eeZ',
              particles = [ P.e__plus__, P.e__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.e__minus__,P.H]],[[P.W__plus__,P.ve]]],
              couplings = {(0,0,0):C.R2_eeZCp, (0,0,1):C.R2_llZCp,(0,1,0):C.R2_eeZCm, (0,1,1):C.R2_eeZCm_v},
              type = 'R2')

V_R2mmZ = CTVertex(name = 'V_R2mmZ',
              particles = [ P.m__plus__, P.m__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.m__minus__,P.H]],[[P.W__plus__,P.vm]]],
              couplings = {(0,0,0):C.R2_mmZCp, (0,0,1):C.R2_llZCp,(0,1,0):C.R2_mmZCm, (0,1,1):C.R2_mmZCm_v},
              type = 'R2')

V_R2tautauZ = CTVertex(name = 'V_R2tautauZ',
              particles = [ P.tt__plus__, P.tt__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.tt__minus__,P.H]],[[P.W__plus__,P.vt]]],
              couplings = {(0,0,0):C.R2_tautauZCp, (0,0,1):C.R2_llZCp,(0,1,0):C.R2_tautauZCm, (0,1,1):C.R2_tautauZCm_v},
              type = 'R2')

V_R2veveZ = CTVertex(name = 'V_R2veveZ',
              particles = [ P.ve__tilde__, P.ve, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.ve,P.Z]],[[P.W__plus__,P.e__minus__]]],
              couplings = {(0,0,0):C.R2_vvZCp, (0,0,1):C.R2_veveZCp_e},
              type = 'R2')

V_R2vmvmZ = CTVertex(name = 'V_R2vmvmZ',
              particles = [ P.vm__tilde__, P.vm, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.vm,P.Z]],[[P.W__plus__,P.m__minus__]]],
              couplings = {(0,0,0):C.R2_vvZCp, (0,0,1):C.R2_vmvmZCp_m},
              type = 'R2')

V_R2vtvtZ = CTVertex(name = 'V_R2vtvtZ',
              particles = [ P.vt__tilde__, P.vt, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.vt,P.Z]],[[P.W__plus__,P.tt__minus__]]],
              couplings = {(0,0,0):C.R2_vvZCp, (0,0,1):C.R2_vtvtZCp_tau},
              type = 'R2')

V_R2dxuW2 = CTVertex(name = 'V_R2dxuW2',
                     particles = [ P.d__tilde__, P.u, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.u,P.d]]],
                     couplings = {(0,0,0):C.R2_dxuW2Cp},
                     type = 'R2')

V_R2sxuW2 = CTVertex(name = 'V_R2sxuW2',
                     particles = [ P.s__tilde__, P.u, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.u,P.s]]],
                     couplings = {(0,0,0):C.R2_sxuW2Cp},
                     type = 'R2')

V_R2bxuW2 = CTVertex(name = 'V_R2bxuW2',
                     particles = [ P.b__tilde__, P.u, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.u,P.b]]],
                     couplings = {(0,0,0):C.R2_bxuW2Cp},
                     type = 'R2')

V_R2dxcW2 = CTVertex(name = 'V_R2dxcW2',
                     particles = [ P.d__tilde__, P.c, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.c,P.d]]],
                     couplings = {(0,0,0):C.R2_dxcW2Cp},
                     type = 'R2')

V_R2sxcW2 = CTVertex(name = 'V_R2sxcW2',
                     particles = [ P.s__tilde__, P.c, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.c,P.s]]],
                     couplings = {(0,0,0):C.R2_sxcW2Cp},
                     type = 'R2')

V_R2bxcW2 = CTVertex(name = 'V_R2bxcW2',
                     particles = [ P.b__tilde__, P.c, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.c,P.b]]],
                     couplings = {(0,0,0):C.R2_bxcW2Cp},
                     type = 'R2')

V_R2dxtW2 = CTVertex(name = 'V_R2dxtW2',
                     particles = [ P.d__tilde__, P.t, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.t,P.d]]],
                     couplings = {(0,0,0):C.R2_dxtW2Cp},
                     type = 'R2')

V_R2sxtW2 = CTVertex(name = 'V_R2sxtW2',
                     particles = [ P.s__tilde__, P.t, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.s,P.t]]],
                     couplings = {(0,0,0):C.R2_sxtW2Cp},
                     type = 'R2')

V_R2bxtW2 = CTVertex(name = 'V_R2bxtW2',
                     particles = [ P.b__tilde__, P.t, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.t,P.b]]],
                     couplings = {(0,0,0):C.R2_bxtW2Cp},
                     type = 'R2')

V_R2uxdW2 = CTVertex(name = 'V_R2uxdW2',
                     particles = [ P.u__tilde__, P.d, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.u,P.d]]],
                     couplings = {(0,0,0):C.R2_uxdW2Cp},
                     type = 'R2')

V_R2uxsW2 = CTVertex(name = 'V_R2uxsW2',
                     particles = [ P.u__tilde__, P.s, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.u,P.s]]],
                     couplings = {(0,0,0):C.R2_uxsW2Cp},
                     type = 'R2')

V_R2uxbW2 = CTVertex(name = 'V_R2uxbW2',
                     particles = [ P.u__tilde__, P.b, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.u,P.b]]],
                     couplings = {(0,0,0):C.R2_uxbW2Cp},
                     type = 'R2')

V_R2cxdW2 = CTVertex(name = 'V_R2cxdW2',
                     particles = [ P.c__tilde__, P.d, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.c,P.d]]],
                     couplings = {(0,0,0):C.R2_cxdW2Cp},
                     type = 'R2')

V_R2cxsW2 = CTVertex(name = 'V_R2cxsW2',
                     particles = [ P.c__tilde__, P.s, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.c,P.s]]],
                     couplings = {(0,0,0):C.R2_cxsW2Cp},
                     type = 'R2')

V_R2cxbW2 = CTVertex(name = 'V_R2cxbW2',
                     particles = [ P.c__tilde__, P.b, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.c,P.b]]],
                     couplings = {(0,0,0):C.R2_cxbW2Cp},
                     type = 'R2')

V_R2txdW2 = CTVertex(name = 'V_R2txdW2',
                     particles = [ P.t__tilde__, P.d, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.t,P.d]]],
                     couplings = {(0,0,0):C.R2_txdW2Cp},
                     type = 'R2')


V_R2txsW2 = CTVertex(name = 'V_R2txsW2',
                     particles = [ P.t__tilde__, P.s, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.t,P.s]]],
                     couplings = {(0,0,0):C.R2_txsW2Cp},
                     type = 'R2')

V_R2txbW2 = CTVertex(name = 'V_R2txbW2',
                     particles = [ P.t__tilde__, P.b, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.H,P.t,P.b]]],
                     couplings = {(0,0,0):C.R2_txbW2Cp},
                     type = 'R2')

V_R2vexeW = CTVertex(name = 'V_R2vexeW',
                     particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__],
                     color = [ '1' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.e__minus__,P.ve]]],
                     couplings = {(0,0,0):C.R2_vlW},
                     type = 'R2')

V_R2vmxmW = CTVertex(name = 'V_R2vmxmW',
                     particles = [ P.vm__tilde__, P.m__minus__, P.W__plus__],
                     color = [ '1' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.m__minus__,P.vm]]],
                     couplings = {(0,0,0):C.R2_vlW},
                     type = 'R2')

V_R2vtxtauW = CTVertex(name = 'V_R2vtxtauW',
                     particles = [ P.vt__tilde__, P.tt__minus__, P.W__plus__],
                     color = [ '1' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.tt__minus__,P.vt]]],
                     couplings = {(0,0,0):C.R2_vlW},
                     type = 'R2')

V_R2exveW = CTVertex(name = 'V_R2exveW',
                     particles = [ P.e__plus__, P.ve, P.W__minus__],
                     color = [ '1' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.e__minus__,P.ve]]],
                     couplings = {(0,0,0):C.R2_vlW},
                     type = 'R2')

V_R2mxvmW = CTVertex(name = 'V_R2mxvmW',
                     particles = [ P.m__plus__, P.vm, P.W__minus__],
                     color = [ '1' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.m__minus__,P.vm]]],
                     couplings = {(0,0,0):C.R2_vlW},
                     type = 'R2')

V_R2tauxvtW = CTVertex(name = 'V_R2tauxvtW',
                     particles = [ P.tt__plus__, P.vt, P.W__minus__],
                     color = [ '1' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.tt__minus__,P.vt]]],
                     couplings = {(0,0,0):C.R2_vlW},
                     type = 'R2')

# R2 for SSS

V_R2HHH = CTVertex(name = 'V_R2HHH',
                   particles = [ P.H, P.H, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.SSS1 ],
                   loop_particles = [[[P.H]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                     [[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]]],
                   couplings = {(0,0,0):C.R2_HHHboson, (0,0,1):C.R2_HHHe, (0,0,2):C.R2_HHHm, (0,0,3):C.R2_HHHtau,
                                (0,0,4):C.R2_HHHu,(0,0,5):C.R2_HHHd,(0,0,6):C.R2_HHHs,
                                (0,0,7):C.R2_HHHc,(0,0,8):C.R2_HHHb,(0,0,9):C.R2_HHHt},
                   type = 'R2')

V_R2G0G0H = CTVertex(name = 'V_R2G0G0H',
                   particles = [ P.G0, P.G0, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.SSS1 ],
                   loop_particles = [[[P.H,P.G0]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                     [[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]]],
                   couplings = {(0,0,0):C.R2_G0G0Hboson, (0,0,1):C.R2_G0G0He, (0,0,2):C.R2_G0G0Hm, (0,0,3):C.R2_G0G0Htau,
                                (0,0,4):C.R2_G0G0Hu,(0,0,5):C.R2_G0G0Hd,(0,0,6):C.R2_G0G0Hs,
                                (0,0,7):C.R2_G0G0Hc,(0,0,8):C.R2_G0G0Hb,(0,0,9):C.R2_G0G0Ht},
                   type = 'R2')

V_R2GmGpH = CTVertex(name = 'V_R2GmGpH',
                   particles = [ P.G__minus__, P.G__plus__, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.SSS1 ],
                   loop_particles = [[[P.H,P.G__plus__]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                     [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                     [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                     [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                   couplings = {(0,0,0):C.R2_GmGpHboson, (0,0,1):C.R2_GmGpHe, (0,0,2):C.R2_GmGpHm, (0,0,3):C.R2_GmGpHtau,
                                (0,0,4):C.R2_GmGpHud,(0,0,5):C.R2_GmGpHus,(0,0,6):C.R2_GmGpHub,
                                (0,0,7):C.R2_GmGpHcd,(0,0,8):C.R2_GmGpHcs,(0,0,9):C.R2_GmGpHcb,
                                (0,0,10):C.R2_GmGpHtd,(0,0,11):C.R2_GmGpHts,(0,0,12):C.R2_GmGpHtb},
                   type = 'R2')

# R2 for VSS

V_R2AG0H = CTVertex(name = 'V_R2AG0H',
                    particles = [ P.A, P.G0, P.H ],
                    color = [ '1' ],
                    lorentz = [ L.VSS1 ],
                    loop_particles = [[[P.W__plus__,P.G__plus__]]],
                    couplings = {(0,0,0):C.R2_AG0H},
                    type = 'R2')

V_R2ZG0H = CTVertex(name = 'V_R2ZG0H',
                    particles = [ P.Z, P.G0, P.H ],
                    color = [ '1' ],
                    lorentz = [ L.VSS1 ],
                    loop_particles = [[[P.W__plus__, P.G__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                      [[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]]],
                    couplings = {(0,0,0):C.R2_ZG0H,(0,0,1):C.R2_ZG0He,(0,0,2):C.R2_ZG0Hm,(0,0,3):C.R2_ZG0Htau,
                                 (0,0,4):C.R2_ZG0Hu,(0,0,5):C.R2_ZG0Hd,(0,0,6):C.R2_ZG0Hs,
                                 (0,0,7):C.R2_ZG0Hc,(0,0,8):C.R2_ZG0Hb,(0,0,9):C.R2_ZG0Ht},
                    type = 'R2')

V_R2AGmGp = CTVertex(name = 'V_R2AGmGp',
                    particles = [ P.A, P.G__minus__, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VSS1 ],
                    loop_particles = [[[P.G__plus__, P.A ]],
                                      [[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_AGmGp,(0,0,1):C.R2_AGmGpe,(0,0,2):C.R2_AGmGpm,(0,0,3):C.R2_AGmGptau,
                                 (0,0,4):C.R2_AGmGpud,(0,0,5):C.R2_AGmGpus,(0,0,6):C.R2_AGmGpub,
                                 (0,0,7):C.R2_AGmGpcd,(0,0,8):C.R2_AGmGpcs,(0,0,9):C.R2_AGmGpcb,
                                 (0,0,10):C.R2_AGmGptd,(0,0,11):C.R2_AGmGpts,(0,0,12):C.R2_AGmGptb},
                    type = 'R2')

V_R2ZGmGp = CTVertex(name = 'V_R2AGmGp',
                    particles = [ P.Z, P.G__minus__, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VSS1 ],
                    loop_particles = [[[P.G__plus__,P.Z]],
                                      [[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_ZGmGp,(0,0,1):C.R2_ZGmGpe,(0,0,2):C.R2_ZGmGpm,(0,0,3):C.R2_ZGmGptau,
                                 (0,0,4):C.R2_ZGmGpud,(0,0,5):C.R2_ZGmGpus,(0,0,6):C.R2_ZGmGpub,
                                 (0,0,7):C.R2_ZGmGpcd,(0,0,8):C.R2_ZGmGpcs,(0,0,9):C.R2_ZGmGpcb,
                                 (0,0,10):C.R2_ZGmGptd,(0,0,11):C.R2_ZGmGpts,(0,0,12):C.R2_ZGmGptb},
                    type = 'R2')

V_R2WGpH = CTVertex(name = 'V_R2WGpH',
                    particles = [ P.W__minus__,P.G__plus__,P.H ],
                    color = [ '1' ],
                    lorentz = [ L.VSS1 ],
                    loop_particles = [[[P.W__plus__, P.A]],
                                      [[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_WGpH,(0,0,1):C.R2_WGpHe,(0,0,2):C.R2_WGpHm,(0,0,3):C.R2_WGpHtau,
                                 (0,0,4):C.R2_WGpHud,(0,0,5):C.R2_WGpHus,(0,0,6):C.R2_WGpHub,
                                 (0,0,7):C.R2_WGpHcd,(0,0,8):C.R2_WGpHcs,(0,0,9):C.R2_WGpHcb,
                                 (0,0,10):C.R2_WGpHtd,(0,0,11):C.R2_WGpHts,(0,0,12):C.R2_WGpHtb},
                    type = 'R2')

V_R2WGmH = CTVertex(name = 'V_R2WGmH',
                    particles = [ P.W__plus__,P.G__minus__,P.H ],
                    color = [ '1' ],
                    lorentz = [ L.VSS1 ],
                    loop_particles = [[[P.W__plus__, P.A]],
                                      [[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_WGpH,(0,0,1):C.R2_WGpHe,(0,0,2):C.R2_WGpHm,(0,0,3):C.R2_WGpHtau,
                                 (0,0,4):C.R2_WGpHud,(0,0,5):C.R2_WGpHus,(0,0,6):C.R2_WGpHub,
                                 (0,0,7):C.R2_WGpHcd,(0,0,8):C.R2_WGpHcs,(0,0,9):C.R2_WGpHcb,
                                 (0,0,10):C.R2_WGpHtd,(0,0,11):C.R2_WGpHts,(0,0,12):C.R2_WGpHtb},
                    type = 'R2')

V_R2WGpG0 = CTVertex(name = 'V_R2WGpG0',
                    particles = [ P.W__minus__,P.G__plus__,P.G0 ],
                    color = [ '1' ],
                    lorentz = [ L.VSS1 ],
                    loop_particles = [[[P.W__plus__, P.G__plus__,P.H]],
                                      [[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_WGpG0,(0,0,1):C.R2_WGpG0e,(0,0,2):C.R2_WGpG0m,(0,0,3):C.R2_WGpG0tau,
                                 (0,0,4):C.R2_WGpG0ud,(0,0,5):C.R2_WGpG0us,(0,0,6):C.R2_WGpG0ub,
                                 (0,0,7):C.R2_WGpG0cd,(0,0,8):C.R2_WGpG0cs,(0,0,9):C.R2_WGpG0cb,
                                 (0,0,10):C.R2_WGpG0td,(0,0,11):C.R2_WGpG0ts,(0,0,12):C.R2_WGpG0tb},
                    type = 'R2')

V_R2WG0Gm = CTVertex(name = 'V_R2WG0Gm',
                    particles = [ P.W__plus__,P.G0,P.G__minus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VSS1 ],
                    loop_particles = [[[P.W__plus__, P.G__plus__,P.H]],
                                      [[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_WGpG0,(0,0,1):C.R2_WGpG0e,(0,0,2):C.R2_WGpG0m,(0,0,3):C.R2_WGpG0tau,
                                 (0,0,4):C.R2_WGpG0ud,(0,0,5):C.R2_WGpG0us,(0,0,6):C.R2_WGpG0ub,
                                 (0,0,7):C.R2_WGpG0cd,(0,0,8):C.R2_WGpG0cs,(0,0,9):C.R2_WGpG0cb,
                                 (0,0,10):C.R2_WGpG0td,(0,0,11):C.R2_WGpG0ts,(0,0,12):C.R2_WGpG0tb},
                    type = 'R2')

# R2 for VVS

V_R2AAH = CTVertex(name = 'V_R2AAH',
                   particles = [ P.A, P.A, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.VVS1 ],
                   loop_particles = [[[P.W__plus__,P.G__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                     [[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]]],
                   couplings = {(0,0,0):C.R2_AAH,(0,0,1):C.R2_AAHe,(0,0,2):C.R2_AAHm,(0,0,3):C.R2_AAHtau,
                                (0,0,4):C.R2_AAHu,(0,0,5):C.R2_AAHd,(0,0,6):C.R2_AAHs,(0,0,7):C.R2_AAHc,(0,0,8):C.R2_AAHb,(0,0,9):C.R2_AAHt},
                   type = 'R2')

V_R2AZH = CTVertex(name = 'V_R2AZH',
                   particles = [ P.A, P.Z, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.VVS1 ],
                   loop_particles = [[[P.W__plus__,P.G__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                     [[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]]],
                   couplings = {(0,0,0):C.R2_AZH,(0,0,1):C.R2_AZHe,(0,0,2):C.R2_AZHm,(0,0,3):C.R2_AZHtau,
                                (0,0,4):C.R2_AZHu,(0,0,5):C.R2_AZHd,(0,0,6):C.R2_AZHs,(0,0,7):C.R2_AZHc,(0,0,8):C.R2_AZHb,(0,0,9):C.R2_AZHt},
                   type = 'R2')

V_R2ZZH = CTVertex(name = 'V_R2ZZH',
                   particles = [ P.Z, P.Z, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.VVS1 ],
                   loop_particles = [[[P.W__plus__,P.G__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                     [[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]]],
                   couplings = {(0,0,0):C.R2_ZZH,(0,0,1):C.R2_ZZHe,(0,0,2):C.R2_ZZHm,(0,0,3):C.R2_ZZHtau,
                                (0,0,4):C.R2_ZZHu,(0,0,5):C.R2_ZZHd,(0,0,6):C.R2_ZZHs,(0,0,7):C.R2_ZZHc,(0,0,8):C.R2_ZZHb,(0,0,9):C.R2_ZZHt},
                   type = 'R2')

V_R2WWH = CTVertex(name = 'V_R2WWH',
                   particles = [ P.W__minus__, P.W__plus__, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.VVS1 ],
                   loop_particles = [[[P.G__plus__,P.H]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                     [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],[[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],[[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                   couplings = {(0,0,0):C.R2_WWH,(0,0,1):C.R2_WWHe,(0,0,2):C.R2_WWHm,(0,0,3):C.R2_WWHtau,
                                (0,0,4):C.R2_WWHud,(0,0,5):C.R2_WWHus,(0,0,6):C.R2_WWHub,(0,0,7):C.R2_WWHcd,(0,0,8):C.R2_WWHcs,(0,0,9):C.R2_WWHcb,
                                (0,0,10):C.R2_WWHtd,(0,0,11):C.R2_WWHts,(0,0,12):C.R2_WWHtb},
                   type = 'R2')

V_R2WAGp = CTVertex(name = 'V_R2WAGp',
                   particles = [ P.W__minus__, P.A, P.G__plus__ ],
                   color = [ '1' ],
                   lorentz = [ L.VVS1 ],
                   loop_particles = [[[P.W__plus__,P.Z]],
                                     [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                     [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                     [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                   couplings = {(0,0,0):C.R2_WAGp,
                                (0,0,1):C.R2_WAGpud,(0,0,2):C.R2_WAGpus,(0,0,3):C.R2_WAGpub,
                                (0,0,4):C.R2_WAGpcd,(0,0,5):C.R2_WAGpcs,(0,0,6):C.R2_WAGpcb,
                                (0,0,7):C.R2_WAGptd,(0,0,8):C.R2_WAGpts,(0,0,9):C.R2_WAGptb},
                   type = 'R2')

V_R2WAGm = CTVertex(name = 'V_R2WAGm',
                   particles = [ P.W__plus__, P.A, P.G__minus__ ],
                   color = [ '1' ],
                   lorentz = [ L.VVS1 ],
                   loop_particles = [[[P.W__plus__,P.Z]],
                                     [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                     [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                     [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                   couplings = {(0,0,0):C.R2_WAGm,
                                (0,0,1):C.R2_WAGmud,(0,0,2):C.R2_WAGmus,(0,0,3):C.R2_WAGmub,
                                (0,0,4):C.R2_WAGmcd,(0,0,5):C.R2_WAGmcs,(0,0,6):C.R2_WAGmcb,
                                (0,0,7):C.R2_WAGmtd,(0,0,8):C.R2_WAGmts,(0,0,9):C.R2_WAGmtb},
                   type = 'R2')

V_R2WZGp = CTVertex(name = 'V_R2WZGp',
                   particles = [ P.W__minus__, P.Z, P.G__plus__ ],
                   color = [ '1' ],
                   lorentz = [ L.VVS1 ],
                   loop_particles = [[[P.W__plus__,P.Z]],
                                     [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                     [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                     [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                   couplings = {(0,0,0):C.R2_WZGp,
                                (0,0,1):C.R2_WZGpud,(0,0,2):C.R2_WZGpus,(0,0,3):C.R2_WZGpub,
                                (0,0,4):C.R2_WZGpcd,(0,0,5):C.R2_WZGpcs,(0,0,6):C.R2_WZGpcb,
                                (0,0,7):C.R2_WZGptd,(0,0,8):C.R2_WZGpts,(0,0,9):C.R2_WZGptb},
                   type = 'R2')

V_R2WZGm = CTVertex(name = 'V_R2WZGm',
                   particles = [ P.W__plus__, P.Z, P.G__minus__ ],
                   color = [ '1' ],
                   lorentz = [ L.VVS1 ],
                   loop_particles = [[[P.W__plus__,P.Z]],
                                     [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                     [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                     [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                   couplings = {(0,0,0):C.R2_WZGm,
                                (0,0,1):C.R2_WZGmud,(0,0,2):C.R2_WZGmus,(0,0,3):C.R2_WZGmub,
                                (0,0,4):C.R2_WZGmcd,(0,0,5):C.R2_WZGmcs,(0,0,6):C.R2_WZGmcb,
                                (0,0,7):C.R2_WZGmtd,(0,0,8):C.R2_WZGmts,(0,0,9):C.R2_WZGmtb},
                   type = 'R2')

# R2 for VVV

V_R2AWW = CTVertex(name = 'V_R2AWW',
                   particles = [ P.A, P.W__minus__, P.W__plus__ ],
                   color = [ '1' ],
                   lorentz = [ L.VVV1 ],
                   loop_particles = [[[P.W__plus__,P.A]],
                                     [[P.e__minus__,P.ve],[P.m__minus__,P.vm],[P.tt__minus__,P.vt]],
                                     [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                     [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                     [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                   couplings = {(0,0,0):C.R2_AWW,(0,0,1):C.R2_AWWlv,
                                (0,0,2):C.R2_AWWud,(0,0,3):C.R2_AWWus,(0,0,4):C.R2_AWWub,
                                (0,0,5):C.R2_AWWcd,(0,0,6):C.R2_AWWcs,(0,0,7):C.R2_AWWcb,
                                (0,0,8):C.R2_AWWtd,(0,0,9):C.R2_AWWts,(0,0,10):C.R2_AWWtb},
                   type = 'R2')

V_R2ZWW = CTVertex(name = 'V_R2ZWW',
                   particles = [ P.Z, P.W__minus__, P.W__plus__ ],
                   color = [ '1' ],
                   lorentz = [ L.VVV1 ],
                   loop_particles = [[[P.W__plus__,P.Z]],
                                     [[P.e__minus__,P.ve],[P.m__minus__,P.vm],[P.tt__minus__,P.vt]],
                                     [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                     [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                     [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                   couplings = {(0,0,0):C.R2_ZWW,(0,0,1):C.R2_ZWWlv,
                                (0,0,2):C.R2_ZWWud,(0,0,3):C.R2_ZWWus,(0,0,4):C.R2_ZWWub,
                                (0,0,5):C.R2_ZWWcd,(0,0,6):C.R2_ZWWcs,(0,0,7):C.R2_ZWWcb,
                                (0,0,8):C.R2_ZWWtd,(0,0,9):C.R2_ZWWts,(0,0,10):C.R2_ZWWtb},
                   type = 'R2')

# R2 with 4 external legs

# R2 for SSSS

V_R2HHHH = CTVertex(name = 'V_R2HHHH',
                   particles = [ P.H, P.H, P.H, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.SSSS1 ],
                   loop_particles = [[[P.H]],
                                     [[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                     [[P.u]],[[P.d]],[[P.s]],
                                     [[P.c]],[[P.b]],[[P.t]]],
                   couplings = {(0,0,0):C.R2_HHHH,
                                (0,0,1):C.R2_HHHHe,(0,0,2):C.R2_HHHHm,(0,0,3):C.R2_HHHHtau,
                                (0,0,4):C.R2_HHHHu,(0,0,5):C.R2_HHHHd,(0,0,6):C.R2_HHHHs,
                                (0,0,7):C.R2_HHHHc,(0,0,8):C.R2_HHHHb,(0,0,9):C.R2_HHHHt},
                   type = 'R2')

V_R2G0G0G0G0 = CTVertex(name = 'V_R2G0G0G0G0',
                   particles = [ P.G0, P.G0, P.G0, P.G0 ],
                   color = [ '1' ],
                   lorentz = [ L.SSSS1 ],
                   loop_particles = [[[P.G0,P.H]],
                                     [[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                     [[P.u]],[[P.d]],[[P.s]],
                                     [[P.c]],[[P.b]],[[P.t]]],
                   couplings = {(0,0,0):C.R2_HHHH,
                                (0,0,1):C.R2_HHHHe,(0,0,2):C.R2_HHHHm,(0,0,3):C.R2_HHHHtau,
                                (0,0,4):C.R2_HHHHu,(0,0,5):C.R2_HHHHd,(0,0,6):C.R2_HHHHs,
                                (0,0,7):C.R2_HHHHc,(0,0,8):C.R2_HHHHb,(0,0,9):C.R2_HHHHt},
                   type = 'R2')

V_R2G0G0HH = CTVertex(name = 'V_R2G0G0HH',
                   particles = [ P.G0, P.G0, P.H, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.SSSS1 ],
                   loop_particles = [[[P.G0,P.H]],
                                     [[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                     [[P.u]],[[P.d]],[[P.s]],
                                     [[P.c]],[[P.b]],[[P.t]]],
                   couplings = {(0,0,0):C.R2_G0G0HH,
                                (0,0,1):C.R2_G0G0HHe,(0,0,2):C.R2_G0G0HHm,(0,0,3):C.R2_G0G0HHtau,
                                (0,0,4):C.R2_G0G0HHu,(0,0,5):C.R2_G0G0HHd,(0,0,6):C.R2_G0G0HHs,
                                (0,0,7):C.R2_G0G0HHc,(0,0,8):C.R2_G0G0HHb,(0,0,9):C.R2_G0G0HHt},
                   type = 'R2')

V_R2GmGpHH = CTVertex(name = 'V_R2GmGpHH',
                   particles = [ P.G__minus__, P.G__plus__, P.H, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.SSSS1 ],
                   loop_particles = [[[P.W__plus__,P.Z,P.H,P.A]],
                                     [[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                     [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                     [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                     [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                   couplings = {(0,0,0):C.R2_GmGpHH,
                                (0,0,1):C.R2_GmGpHHe,(0,0,2):C.R2_GmGpHHm,(0,0,3):C.R2_GmGpHHtau,
                                (0,0,4):C.R2_GmGpHHud,(0,0,5):C.R2_GmGpHHus,(0,0,6):C.R2_GmGpHHub,
                                (0,0,7):C.R2_GmGpHHcd,(0,0,8):C.R2_GmGpHHcs,(0,0,9):C.R2_GmGpHHcb,
                                (0,0,10):C.R2_GmGpHHtd,(0,0,11):C.R2_GmGpHHts,(0,0,12):C.R2_GmGpHHtb},
                   type = 'R2')

V_R2GmGpG0G0 = CTVertex(name = 'V_R2GmGpG0G0',
                   particles = [ P.G__minus__, P.G__plus__, P.G0, P.G0 ],
                   color = [ '1' ],
                   lorentz = [ L.SSSS1 ],
                   loop_particles = [[[P.G__plus__,P.A]],
                                     [[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                     [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                     [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                     [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                   couplings = {(0,0,0):C.R2_GmGpHH,
                                (0,0,1):C.R2_GmGpHHe,(0,0,2):C.R2_GmGpHHm,(0,0,3):C.R2_GmGpHHtau,
                                (0,0,4):C.R2_GmGpHHud,(0,0,5):C.R2_GmGpHHus,(0,0,6):C.R2_GmGpHHub,
                                (0,0,7):C.R2_GmGpHHcd,(0,0,8):C.R2_GmGpHHcs,(0,0,9):C.R2_GmGpHHcb,
                                (0,0,10):C.R2_GmGpHHtd,(0,0,11):C.R2_GmGpHHts,(0,0,12):C.R2_GmGpHHtb},
                   type = 'R2')

V_R2GmGmGpGp = CTVertex(name = 'V_R2GmGmGpGp',
                   particles = [ P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
                   color = [ '1' ],
                   lorentz = [ L.SSSS1 ],
                   loop_particles = [[[P.G__plus__,P.H]],
                                     [[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                     [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                     [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                     [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]],
                                     [[P.u,P.c,P.d]],[[P.u,P.t,P.d]],[[P.c,P.t,P.d]],
                                     [[P.u,P.c,P.s]],[[P.u,P.t,P.s]],[[P.c,P.t,P.s]],
                                     [[P.u,P.c,P.b]],[[P.u,P.t,P.b]],[[P.c,P.t,P.b]],
                                     [[P.u,P.d,P.s]],[[P.u,P.d,P.b]],[[P.u,P.s,P.b]],
                                     [[P.c,P.d,P.s]],[[P.c,P.d,P.b]],[[P.c,P.s,P.b]],
                                     [[P.t,P.d,P.s]],[[P.t,P.d,P.b]],[[P.t,P.s,P.b]],
                                     [[P.u,P.c,P.d,P.s]],[[P.u,P.c,P.d,P.b]],[[P.u,P.c,P.s,P.b]],
                                     [[P.u,P.t,P.d,P.s]],[[P.u,P.t,P.d,P.b]],[[P.u,P.t,P.s,P.b]],
                                     [[P.c,P.t,P.d,P.s]],[[P.c,P.t,P.d,P.b]],[[P.c,P.t,P.s,P.b]]
                                     ],
                   couplings = {(0,0,0):C.R2_GmGmGpGp,
                                (0,0,1):C.R2_GmGmGpGpe,(0,0,2):C.R2_GmGmGpGpm,(0,0,3):C.R2_GmGmGpGptau,
                                (0,0,4):C.R2_GmGmGpGpud,(0,0,5):C.R2_GmGmGpGpus,(0,0,6):C.R2_GmGmGpGpub,
                                (0,0,7):C.R2_GmGmGpGpcd,(0,0,8):C.R2_GmGmGpGpcs,(0,0,9):C.R2_GmGmGpGpcb,
                                (0,0,10):C.R2_GmGmGpGptd,(0,0,11):C.R2_GmGmGpGpts,(0,0,12):C.R2_GmGmGpGptb,
                                (0,0,13):C.R2_GmGmGpGpucd,(0,0,14):C.R2_GmGmGpGputd,(0,0,15):C.R2_GmGmGpGpctd,
                                (0,0,16):C.R2_GmGmGpGpucs,(0,0,17):C.R2_GmGmGpGputs,(0,0,18):C.R2_GmGmGpGpcts,
                                (0,0,19):C.R2_GmGmGpGpucb,(0,0,20):C.R2_GmGmGpGputb,(0,0,21):C.R2_GmGmGpGpctb,
                                (0,0,22):C.R2_GmGmGpGpuds,(0,0,23):C.R2_GmGmGpGpudb,(0,0,24):C.R2_GmGmGpGpusb,
                                (0,0,25):C.R2_GmGmGpGpcds,(0,0,26):C.R2_GmGmGpGpcdb,(0,0,27):C.R2_GmGmGpGpcsb,
                                (0,0,28):C.R2_GmGmGpGptds,(0,0,29):C.R2_GmGmGpGptdb,(0,0,30):C.R2_GmGmGpGptsb,
                                (0,0,31):C.R2_GmGmGpGpucds,(0,0,32):C.R2_GmGmGpGpucdb,(0,0,33):C.R2_GmGmGpGpucsb,
                                (0,0,34):C.R2_GmGmGpGputds,(0,0,35):C.R2_GmGmGpGputdb,(0,0,36):C.R2_GmGmGpGputsb,
                                (0,0,37):C.R2_GmGmGpGpctds,(0,0,38):C.R2_GmGmGpGpctdb,(0,0,39):C.R2_GmGmGpGpctsb},
                   type = 'R2')

# R2 for VVVV

V_R2AAAA = CTVertex(name = 'V_R2AAAA',
                    particles = [ P.A, P.A, P.A, P.A ],
                    color = [ '1' ],
                    lorentz = [ L.R2_VVVV1 ],
                    loop_particles = [[[P.G__plus__]],[[P.e__minus__],[P.m__minus__],[P.tt__minus__]],[[P.u],[P.c],[P.t]],[[P.d],[P.s],[P.b]]],
                    couplings = {(0,0,0):C.R2_AAAA,(0,0,1):C.R2_AAAAl,(0,0,2):C.R2_AAAAu,(0,0,3):C.R2_AAAAd},
                    type = 'R2')

V_R2AAAZ = CTVertex(name = 'V_R2AAAZ',
                    particles = [ P.A, P.A, P.A, P.Z ],
                    color = [ '1' ],
                    lorentz = [ L.R2_VVVV1 ],
                    loop_particles = [[[P.G__plus__]],[[P.e__minus__],[P.m__minus__],[P.tt__minus__]],[[P.u],[P.c],[P.t]],[[P.d],[P.s],[P.b]]],
                    couplings = {(0,0,0):C.R2_AAAZ,(0,0,1):C.R2_AAAZl,(0,0,2):C.R2_AAAZu,(0,0,3):C.R2_AAAZd},
                    type = 'R2')

V_R2AAZZ = CTVertex(name = 'V_R2AAZZ',
                    particles = [ P.A, P.A, P.Z, P.Z ],
                    color = [ '1' ],
                    lorentz = [ L.R2_VVVV1 ],
                    loop_particles = [[[P.G__plus__]],[[P.e__minus__],[P.m__minus__],[P.tt__minus__]],[[P.u],[P.c],[P.t]],[[P.d],[P.s],[P.b]]],
                    couplings = {(0,0,0):C.R2_AAZZ,(0,0,1):C.R2_AAZZl,(0,0,2):C.R2_AAZZu,(0,0,3):C.R2_AAZZd},
                    type = 'R2')

V_R2AZZZ = CTVertex(name = 'V_R2AZZZ',
                    particles = [ P.A, P.Z, P.Z, P.Z ],
                    color = [ '1' ],
                    lorentz = [ L.R2_VVVV1 ],
                    loop_particles = [[[P.G__plus__]],[[P.e__minus__],[P.m__minus__],[P.tt__minus__]],[[P.u],[P.c],[P.t]],[[P.d],[P.s],[P.b]]],
                    couplings = {(0,0,0):C.R2_AZZZ,(0,0,1):C.R2_AZZZl,(0,0,2):C.R2_AZZZu,(0,0,3):C.R2_AZZZd},
                    type = 'R2')

V_R2ZZZZ = CTVertex(name = 'V_R2ZZZZ',
                    particles = [ P.Z, P.Z, P.Z, P.Z ],
                    color = [ '1' ],
                    lorentz = [ L.R2_VVVV1 ],
                    loop_particles = [[[P.G__plus__]],[[P.ve],[P.vm],[P.vt]],[[P.e__minus__],[P.m__minus__],[P.tt__minus__]],[[P.u],[P.c],[P.t]],[[P.d],[P.s],[P.b]]],
                    couplings = {(0,0,0):C.R2_ZZZZ,(0,0,1):C.R2_ZZZZv,(0,0,2):C.R2_ZZZZl,(0,0,3):C.R2_ZZZZu,(0,0,4):C.R2_ZZZZd},
                    type = 'R2')

V_R2AAWW = CTVertex(name = 'V_R2AAWW',
                    particles = [ P.A, P.A, P.W__minus__, P.W__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.R2_VVVV2, L.R2_VVVV3 ],
                    loop_particles = [[[P.G__plus__]],[[P.ve,P.e__minus__],[P.vm,P.m__minus__],[P.vt,P.tt__minus__]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_AAWW1,(0,0,1):C.R2_AAWW1lv,
                                 (0,0,2):C.R2_AAWW1ud,(0,0,3):C.R2_AAWW1us,(0,0,4):C.R2_AAWW1ub,
                                 (0,0,5):C.R2_AAWW1cd,(0,0,6):C.R2_AAWW1cs,(0,0,7):C.R2_AAWW1cb,
                                 (0,0,8):C.R2_AAWW1td,(0,0,9):C.R2_AAWW1ts,(0,0,10):C.R2_AAWW1tb,
                                 (0,1,0):C.R2_AAWW2,(0,1,1):C.R2_AAWW2lv,
                                 (0,1,2):C.R2_AAWW2ud,(0,1,3):C.R2_AAWW2us,(0,1,4):C.R2_AAWW2ub,
                                 (0,1,5):C.R2_AAWW2cd,(0,1,6):C.R2_AAWW2cs,(0,1,7):C.R2_AAWW2cb,
                                 (0,1,8):C.R2_AAWW2td,(0,1,9):C.R2_AAWW2ts,(0,1,10):C.R2_AAWW2tb},
                    type = 'R2')

V_R2AZWW = CTVertex(name = 'V_R2AZWW',
                    particles = [ P.A, P.Z, P.W__minus__, P.W__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.R2_VVVV2, L.R2_VVVV3 ],
                    loop_particles = [[[P.G__plus__]],[[P.ve,P.e__minus__],[P.vm,P.m__minus__],[P.vt,P.tt__minus__]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_AZWW1,(0,0,1):C.R2_AZWW1lv,
                                 (0,0,2):C.R2_AZWW1ud,(0,0,3):C.R2_AZWW1us,(0,0,4):C.R2_AZWW1ub,
                                 (0,0,5):C.R2_AZWW1cd,(0,0,6):C.R2_AZWW1cs,(0,0,7):C.R2_AZWW1cb,
                                 (0,0,8):C.R2_AZWW1td,(0,0,9):C.R2_AZWW1ts,(0,0,10):C.R2_AZWW1tb,
                                 (0,1,0):C.R2_AZWW2,(0,1,1):C.R2_AZWW2lv,
                                 (0,1,2):C.R2_AZWW2ud,(0,1,3):C.R2_AZWW2us,(0,1,4):C.R2_AZWW2ub,
                                 (0,1,5):C.R2_AZWW2cd,(0,1,6):C.R2_AZWW2cs,(0,1,7):C.R2_AZWW2cb,
                                 (0,1,8):C.R2_AZWW2td,(0,1,9):C.R2_AZWW2ts,(0,1,10):C.R2_AZWW2tb},
                    type = 'R2')

V_R2ZZWW = CTVertex(name = 'V_R2ZZWW',
                    particles = [ P.Z, P.Z, P.W__minus__, P.W__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.R2_VVVV2, L.R2_VVVV3 ],
                    loop_particles = [[[P.W__plus__]],[[P.ve,P.e__minus__],[P.vm,P.m__minus__],[P.vt,P.tt__minus__]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_ZZWW1,(0,0,1):C.R2_ZZWW1lv,
                                 (0,0,2):C.R2_ZZWW1ud,(0,0,3):C.R2_ZZWW1us,(0,0,4):C.R2_ZZWW1ub,
                                 (0,0,5):C.R2_ZZWW1cd,(0,0,6):C.R2_ZZWW1cs,(0,0,7):C.R2_ZZWW1cb,
                                 (0,0,8):C.R2_ZZWW1td,(0,0,9):C.R2_ZZWW1ts,(0,0,10):C.R2_ZZWW1tb,
                                 (0,1,0):C.R2_ZZWW2,(0,1,1):C.R2_ZZWW2lv,
                                 (0,1,2):C.R2_ZZWW2ud,(0,1,3):C.R2_ZZWW2us,(0,1,4):C.R2_ZZWW2ub,
                                 (0,1,5):C.R2_ZZWW2cd,(0,1,6):C.R2_ZZWW2cs,(0,1,7):C.R2_ZZWW2cb,
                                 (0,1,8):C.R2_ZZWW2td,(0,1,9):C.R2_ZZWW2ts,(0,1,10):C.R2_ZZWW2tb},
                    type = 'R2')

V_R2WWWW = CTVertex(name = 'V_R2WWWW',
                    particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.R2_VVVV2, L.R2_VVVV3 ],
                    loop_particles = [[[P.G__plus__,P.H]],[[P.ve,P.e__minus__],[P.vm,P.m__minus__],[P.vt,P.tt__minus__]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]],
                                      [[P.u,P.c,P.d]],[[P.u,P.t,P.d]],[[P.c,P.t,P.d]],
                                     [[P.u,P.c,P.s]],[[P.u,P.t,P.s]],[[P.c,P.t,P.s]],
                                     [[P.u,P.c,P.b]],[[P.u,P.t,P.b]],[[P.c,P.t,P.b]],
                                     [[P.u,P.d,P.s]],[[P.u,P.d,P.b]],[[P.u,P.s,P.b]],
                                    [[P.c,P.d,P.s]],[[P.c,P.d,P.b]],[[P.c,P.s,P.b]],
                                     [[P.t,P.d,P.s]],[[P.t,P.d,P.b]],[[P.t,P.s,P.b]],
                                     [[P.u,P.c,P.d,P.s]],[[P.u,P.c,P.d,P.b]],[[P.u,P.c,P.s,P.b]],
                                     [[P.u,P.t,P.d,P.s]],[[P.u,P.t,P.d,P.b]],[[P.u,P.t,P.s,P.b]],
                                     [[P.c,P.t,P.d,P.s]],[[P.c,P.t,P.d,P.b]],[[P.c,P.t,P.s,P.b]]],
                    couplings = {(0,0,0):C.R2_WWWW1,(0,0,1):C.R2_WWWW1lv,
                                 (0,0,2):C.R2_WWWW1ud,(0,0,3):C.R2_WWWW1us,(0,0,4):C.R2_WWWW1ub,
                                 (0,0,5):C.R2_WWWW1cd,(0,0,6):C.R2_WWWW1cs,(0,0,7):C.R2_WWWW1cb,
                                 (0,0,8):C.R2_WWWW1td,(0,0,9):C.R2_WWWW1ts,(0,0,10):C.R2_WWWW1tb,
                                 (0,1,0):C.R2_WWWW2,(0,1,1):C.R2_WWWW2lv,
                                 (0,1,2):C.R2_WWWW2ud,(0,1,3):C.R2_WWWW2us,(0,1,4):C.R2_WWWW2ub,
                                 (0,1,5):C.R2_WWWW2cd,(0,1,6):C.R2_WWWW2cs,(0,1,7):C.R2_WWWW2cb,
                                 (0,1,8):C.R2_WWWW2td,(0,1,9):C.R2_WWWW2ts,(0,1,10):C.R2_WWWW2tb,
                                 (0,0,11):C.R2_WWWW1ucd,(0,0,12):C.R2_WWWW1utd,(0,0,13):C.R2_WWWW1ctd,
                                 (0,0,14):C.R2_WWWW1ucs,(0,0,15):C.R2_WWWW1uts,(0,0,16):C.R2_WWWW1cts,
                                 (0,0,17):C.R2_WWWW1ucb,(0,0,18):C.R2_WWWW1utb,(0,0,19):C.R2_WWWW1ctb,
                                 (0,0,20):C.R2_WWWW1uds,(0,0,21):C.R2_WWWW1udb,(0,0,22):C.R2_WWWW1usb,
                                 (0,0,23):C.R2_WWWW1cds,(0,0,24):C.R2_WWWW1cdb,(0,0,25):C.R2_WWWW1csb,
                                 (0,0,26):C.R2_WWWW1tds,(0,0,27):C.R2_WWWW1tdb,(0,0,28):C.R2_WWWW1tsb,
                                 (0,0,29):C.R2_WWWW1ucds,(0,0,30):C.R2_WWWW1ucdb,(0,0,31):C.R2_WWWW1ucsb,
                                 (0,0,32):C.R2_WWWW1utds,(0,0,33):C.R2_WWWW1utdb,(0,0,34):C.R2_WWWW1utsb,
                                 (0,0,35):C.R2_WWWW1ctds,(0,0,36):C.R2_WWWW1ctdb,(0,0,37):C.R2_WWWW1ctsb,
                                 (0,1,11):C.R2_WWWW2ucd,(0,1,12):C.R2_WWWW2utd,(0,1,13):C.R2_WWWW2ctd,
                                 (0,1,14):C.R2_WWWW2ucs,(0,1,15):C.R2_WWWW2uts,(0,1,16):C.R2_WWWW2cts,
                                 (0,1,17):C.R2_WWWW2ucb,(0,1,18):C.R2_WWWW2utb,(0,1,19):C.R2_WWWW2ctb,
                                 (0,1,20):C.R2_WWWW2uds,(0,1,21):C.R2_WWWW2udb,(0,1,22):C.R2_WWWW2usb,
                                 (0,1,23):C.R2_WWWW2cds,(0,1,24):C.R2_WWWW2cdb,(0,1,25):C.R2_WWWW2csb,
                                 (0,1,26):C.R2_WWWW2tds,(0,1,27):C.R2_WWWW2tdb,(0,1,28):C.R2_WWWW2tsb,
                                 (0,1,29):C.R2_WWWW2ucds,(0,1,30):C.R2_WWWW2ucdb,(0,1,31):C.R2_WWWW2ucsb,
                                 (0,1,32):C.R2_WWWW2utds,(0,1,33):C.R2_WWWW2utdb,(0,1,34):C.R2_WWWW2utsb,
                                 (0,1,35):C.R2_WWWW2ctds,(0,1,36):C.R2_WWWW2ctdb,(0,1,37):C.R2_WWWW2ctsb},
                    type = 'R2')

# R2 for VVSS

V_R2AAHH = CTVertex(name = 'V_R2AAHH',
                    particles = [ P.A, P.A, P.H, P.H ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                      [[P.u]],[[P.d]],[[P.s]],
                                      [[P.c]],[[P.b]],[[P.t]]],
                    couplings = {(0,0,0):C.R2_AAHH,(0,0,1):C.R2_AAHHe,
                                 (0,0,2):C.R2_AAHHm,(0,0,3):C.R2_AAHHtau,(0,0,4):C.R2_AAHHu,
                                 (0,0,5):C.R2_AAHHd,(0,0,6):C.R2_AAHHs,(0,0,7):C.R2_AAHHc,
                                 (0,0,8):C.R2_AAHHb,(0,0,9):C.R2_AAHHt},
                    type = 'R2')

V_R2AAG0G0 = CTVertex(name = 'V_R2AAG0G0',
                    particles = [ P.A, P.A, P.G0, P.G0 ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.W__plus__,P.G__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                      [[P.u]],[[P.d]],[[P.s]],
                                      [[P.c]],[[P.b]],[[P.t]]],
                    couplings = {(0,0,0):C.R2_AAHH,(0,0,1):C.R2_AAHHe,
                                 (0,0,2):C.R2_AAHHm,(0,0,3):C.R2_AAHHtau,(0,0,4):C.R2_AAHHu,
                                 (0,0,5):C.R2_AAHHd,(0,0,6):C.R2_AAHHs,(0,0,7):C.R2_AAHHc,
                                 (0,0,8):C.R2_AAHHb,(0,0,9):C.R2_AAHHt},
                    type = 'R2')

V_R2AZHH = CTVertex(name = 'V_R2AZHH',
                    particles = [ P.A, P.Z, P.H, P.H ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                      [[P.u]],[[P.d]],[[P.s]],
                                      [[P.c]],[[P.b]],[[P.t]]],
                    couplings = {(0,0,0):C.R2_AZHH,(0,0,1):C.R2_AZHHe,
                                 (0,0,2):C.R2_AZHHm,(0,0,3):C.R2_AZHHtau,(0,0,4):C.R2_AZHHu,
                                 (0,0,5):C.R2_AZHHd,(0,0,6):C.R2_AZHHs,(0,0,7):C.R2_AZHHc,
                                 (0,0,8):C.R2_AZHHb,(0,0,9):C.R2_AZHHt},
                    type = 'R2')

V_R2AZG0G0 = CTVertex(name = 'V_R2AZG0G0',
                    particles = [ P.A, P.Z, P.G0, P.G0 ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                      [[P.u]],[[P.d]],[[P.s]],
                                      [[P.c]],[[P.b]],[[P.t]]],
                    couplings = {(0,0,0):C.R2_AZHH,(0,0,1):C.R2_AZHHe,
                                 (0,0,2):C.R2_AZHHm,(0,0,3):C.R2_AZHHtau,(0,0,4):C.R2_AZHHu,
                                 (0,0,5):C.R2_AZHHd,(0,0,6):C.R2_AZHHs,(0,0,7):C.R2_AZHHc,
                                 (0,0,8):C.R2_AZHHb,(0,0,9):C.R2_AZHHt},
                    type = 'R2')

V_R2ZZHH = CTVertex(name = 'V_R2ZZHH',
                    particles = [ P.Z, P.Z, P.H, P.H ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                      [[P.u]],[[P.d]],[[P.s]],
                                      [[P.c]],[[P.b]],[[P.t]]],
                    couplings = {(0,0,0):C.R2_ZZHH,(0,0,1):C.R2_ZZHHe,
                                 (0,0,2):C.R2_ZZHHm,(0,0,3):C.R2_ZZHHtau,(0,0,4):C.R2_ZZHHu,
                                 (0,0,5):C.R2_ZZHHd,(0,0,6):C.R2_ZZHHs,(0,0,7):C.R2_ZZHHc,
                                 (0,0,8):C.R2_ZZHHb,(0,0,9):C.R2_ZZHHt},
                    type = 'R2')

V_R2ZZG0G0 = CTVertex(name = 'V_R2ZZG0G0',
                    particles = [ P.Z, P.Z, P.G0, P.G0 ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],
                                      [[P.u]],[[P.d]],[[P.s]],
                                      [[P.c]],[[P.b]],[[P.t]]],
                    couplings = {(0,0,0):C.R2_ZZHH,(0,0,1):C.R2_ZZHHe,
                                 (0,0,2):C.R2_ZZHHm,(0,0,3):C.R2_ZZHHtau,(0,0,4):C.R2_ZZHHu,
                                 (0,0,5):C.R2_ZZHHd,(0,0,6):C.R2_ZZHHs,(0,0,7):C.R2_ZZHHc,
                                 (0,0,8):C.R2_ZZHHb,(0,0,9):C.R2_ZZHHt},
                    type = 'R2')

V_R2WWHH = CTVertex(name = 'V_R2WWHH',
                    particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_WWHH,(0,0,1):C.R2_WWHHe,
                                 (0,0,2):C.R2_WWHHm,(0,0,3):C.R2_WWHHtau,(0,0,4):C.R2_WWHHud,
                                 (0,0,5):C.R2_WWHHus,(0,0,6):C.R2_WWHHub,(0,0,7):C.R2_WWHHcd,
                                 (0,0,8):C.R2_WWHHcs,(0,0,9):C.R2_WWHHcb,(0,0,10):C.R2_WWHHtd,
                                 (0,0,11):C.R2_WWHHts,(0,0,12):C.R2_WWHHtb},
                    type = 'R2')

V_R2WWG0G0 = CTVertex(name = 'V_R2WWG0G0',
                    particles = [ P.W__minus__, P.W__plus__, P.G0, P.G0 ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.W__plus__,P.Z]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_WWHH,(0,0,1):C.R2_WWHHe,
                                 (0,0,2):C.R2_WWHHm,(0,0,3):C.R2_WWHHtau,(0,0,4):C.R2_WWHHud,
                                 (0,0,5):C.R2_WWHHus,(0,0,6):C.R2_WWHHub,(0,0,7):C.R2_WWHHcd,
                                 (0,0,8):C.R2_WWHHcs,(0,0,9):C.R2_WWHHcb,(0,0,10):C.R2_WWHHtd,
                                 (0,0,11):C.R2_WWHHts,(0,0,12):C.R2_WWHHtb},
                    type = 'R2')

V_R2WAG0Gp = CTVertex(name = 'V_R2WAG0Gp',
                    particles = [ P.W__minus__, P.A, P.G0, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__,P.H,P.G0]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_WAG0Gp,(0,0,1):C.R2_WAG0Gpe,
                                 (0,0,2):C.R2_WAG0Gpm,(0,0,3):C.R2_WAG0Gptau,(0,0,4):C.R2_WAG0Gpud,
                                 (0,0,5):C.R2_WAG0Gpus,(0,0,6):C.R2_WAG0Gpub,(0,0,7):C.R2_WAG0Gpcd,
                                 (0,0,8):C.R2_WAG0Gpcs,(0,0,9):C.R2_WAG0Gpcb,(0,0,10):C.R2_WAG0Gptd,
                                 (0,0,11):C.R2_WAG0Gpts,(0,0,12):C.R2_WAG0Gptb},
                    type = 'R2')

V_R2WAG0Gm = CTVertex(name = 'V_R2WAG0Gm',
                    particles = [ P.W__plus__, P.A, P.G0, P.G__minus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__,P.H,P.G0]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_WAG0Gp,(0,0,1):C.R2_WAG0Gpe,
                                 (0,0,2):C.R2_WAG0Gpm,(0,0,3):C.R2_WAG0Gptau,(0,0,4):C.R2_WAG0Gpud,
                                 (0,0,5):C.R2_WAG0Gpus,(0,0,6):C.R2_WAG0Gpub,(0,0,7):C.R2_WAG0Gpcd,
                                 (0,0,8):C.R2_WAG0Gpcs,(0,0,9):C.R2_WAG0Gpcb,(0,0,10):C.R2_WAG0Gptd,
                                 (0,0,11):C.R2_WAG0Gpts,(0,0,12):C.R2_WAG0Gptb},
                    type = 'R2')

V_R2WAHGp = CTVertex(name = 'V_R2WAHGp',
                    particles = [ P.W__minus__, P.A, P.H, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__,P.H]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_WAHGp,(0,0,1):C.R2_WAHGpe,
                                 (0,0,2):C.R2_WAHGpm,(0,0,3):C.R2_WAHGptau,(0,0,4):C.R2_WAHGpud,
                                 (0,0,5):C.R2_WAHGpus,(0,0,6):C.R2_WAHGpub,(0,0,7):C.R2_WAHGpcd,
                                 (0,0,8):C.R2_WAHGpcs,(0,0,9):C.R2_WAHGpcb,(0,0,10):C.R2_WAHGptd,
                                 (0,0,11):C.R2_WAHGpts,(0,0,12):C.R2_WAHGptb},
                    type = 'R2')

V_R2WAHGm = CTVertex(name = 'V_R2WAHGm',
                    particles = [ P.W__plus__, P.A, P.H, P.G__minus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__,P.H]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_WAHGm,(0,0,1):C.R2_WAHGme,
                                 (0,0,2):C.R2_WAHGmm,(0,0,3):C.R2_WAHGmtau,(0,0,4):C.R2_WAHGmud,
                                 (0,0,5):C.R2_WAHGmus,(0,0,6):C.R2_WAHGmub,(0,0,7):C.R2_WAHGmcd,
                                 (0,0,8):C.R2_WAHGmcs,(0,0,9):C.R2_WAHGmcb,(0,0,10):C.R2_WAHGmtd,
                                 (0,0,11):C.R2_WAHGmts,(0,0,12):C.R2_WAHGmtb},
                    type = 'R2')

V_R2WZG0Gp = CTVertex(name = 'V_R2WZG0Gp',
                    particles = [ P.W__minus__, P.Z, P.G0, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__,P.H,P.G0]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_WZG0Gp,(0,0,1):C.R2_WZG0Gpe,
                                 (0,0,2):C.R2_WZG0Gpm,(0,0,3):C.R2_WZG0Gptau,(0,0,4):C.R2_WZG0Gpud,
                                 (0,0,5):C.R2_WZG0Gpus,(0,0,6):C.R2_WZG0Gpub,(0,0,7):C.R2_WZG0Gpcd,
                                 (0,0,8):C.R2_WZG0Gpcs,(0,0,9):C.R2_WZG0Gpcb,(0,0,10):C.R2_WZG0Gptd,
                                 (0,0,11):C.R2_WZG0Gpts,(0,0,12):C.R2_WZG0Gptb},
                    type = 'R2')

V_R2WZG0Gm = CTVertex(name = 'V_R2WZG0Gm',
                    particles = [ P.W__plus__, P.Z, P.G0, P.G__minus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__,P.H,P.G0]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_WZG0Gp,(0,0,1):C.R2_WZG0Gpe,
                                 (0,0,2):C.R2_WZG0Gpm,(0,0,3):C.R2_WZG0Gptau,(0,0,4):C.R2_WZG0Gpud,
                                 (0,0,5):C.R2_WZG0Gpus,(0,0,6):C.R2_WZG0Gpub,(0,0,7):C.R2_WZG0Gpcd,
                                 (0,0,8):C.R2_WZG0Gpcs,(0,0,9):C.R2_WZG0Gpcb,(0,0,10):C.R2_WZG0Gptd,
                                 (0,0,11):C.R2_WZG0Gpts,(0,0,12):C.R2_WZG0Gptb},
                    type = 'R2')

V_R2WZHGp = CTVertex(name = 'V_R2WZHGp',
                    particles = [ P.W__minus__, P.Z, P.H, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__,P.H]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_WZHGp,(0,0,1):C.R2_WZHGpe,
                                 (0,0,2):C.R2_WZHGpm,(0,0,3):C.R2_WZHGptau,(0,0,4):C.R2_WZHGpud,
                                 (0,0,5):C.R2_WZHGpus,(0,0,6):C.R2_WZHGpub,(0,0,7):C.R2_WZHGpcd,
                                 (0,0,8):C.R2_WZHGpcs,(0,0,9):C.R2_WZHGpcb,(0,0,10):C.R2_WZHGptd,
                                 (0,0,11):C.R2_WZHGpts,(0,0,12):C.R2_WZHGptb},
                    type = 'R2')

V_R2WZHGm = CTVertex(name = 'V_R2WZHGm',
                    particles = [ P.W__plus__, P.Z, P.H, P.G__minus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__,P.H]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_WZHGm,(0,0,1):C.R2_WZHGme,
                                 (0,0,2):C.R2_WZHGmm,(0,0,3):C.R2_WZHGmtau,(0,0,4):C.R2_WZHGmud,
                                 (0,0,5):C.R2_WZHGmus,(0,0,6):C.R2_WZHGmub,(0,0,7):C.R2_WZHGmcd,
                                 (0,0,8):C.R2_WZHGmcs,(0,0,9):C.R2_WZHGmcb,(0,0,10):C.R2_WZHGmtd,
                                 (0,0,11):C.R2_WZHGmts,(0,0,12):C.R2_WZHGmtb},
                    type = 'R2')

V_R2AAGmGp = CTVertex(name = 'V_R2AAGmGp',
                    particles = [ P.A, P.A, P.G__minus__, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__,P.H]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_AAGmGp,(0,0,1):C.R2_AAGmGpe,
                                 (0,0,2):C.R2_AAGmGpm,(0,0,3):C.R2_AAGmGptau,(0,0,4):C.R2_AAGmGpud,
                                 (0,0,5):C.R2_AAGmGpus,(0,0,6):C.R2_AAGmGpub,(0,0,7):C.R2_AAGmGpcd,
                                 (0,0,8):C.R2_AAGmGpcs,(0,0,9):C.R2_AAGmGpcb,(0,0,10):C.R2_AAGmGptd,
                                 (0,0,11):C.R2_AAGmGpts,(0,0,12):C.R2_AAGmGptb},
                    type = 'R2')

V_R2AZGmGp = CTVertex(name = 'V_R2AZGmGp',
                    particles = [ P.A, P.Z, P.G__minus__, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__,P.H]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_AZGmGp,(0,0,1):C.R2_AZGmGpe,
                                 (0,0,2):C.R2_AZGmGpm,(0,0,3):C.R2_AZGmGptau,(0,0,4):C.R2_AZGmGpud,
                                 (0,0,5):C.R2_AZGmGpus,(0,0,6):C.R2_AZGmGpub,(0,0,7):C.R2_AZGmGpcd,
                                 (0,0,8):C.R2_AZGmGpcs,(0,0,9):C.R2_AZGmGpcb,(0,0,10):C.R2_AZGmGptd,
                                 (0,0,11):C.R2_AZGmGpts,(0,0,12):C.R2_AZGmGptb},
                    type = 'R2')

V_R2ZZGmGp = CTVertex(name = 'V_R2ZZGmGp',
                    particles = [ P.Z, P.Z, P.G__minus__, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__,P.H]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_ZZGmGp,(0,0,1):C.R2_ZZGmGpe,
                                 (0,0,2):C.R2_ZZGmGpm,(0,0,3):C.R2_ZZGmGptau,(0,0,4):C.R2_ZZGmGpud,
                                 (0,0,5):C.R2_ZZGmGpus,(0,0,6):C.R2_ZZGmGpub,(0,0,7):C.R2_ZZGmGpcd,
                                 (0,0,8):C.R2_ZZGmGpcs,(0,0,9):C.R2_ZZGmGpcb,(0,0,10):C.R2_ZZGmGptd,
                                 (0,0,11):C.R2_ZZGmGpts,(0,0,12):C.R2_ZZGmGptb},
                    type = 'R2')


V_R2WWGmGp = CTVertex(name = 'V_R2WWGmGp',
                    particles = [ P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.VVSS1 ],
                    loop_particles = [[[P.G__plus__,P.H]],[[P.e__minus__,P.ve]],[[P.m__minus__,P.vm]],[[P.tt__minus__,P.vt]],
                                      [[P.u,P.d]],[[P.u,P.s]],[[P.u,P.b]],
                                      [[P.c,P.d]],[[P.c,P.s]],[[P.c,P.b]],
                                      [[P.t,P.d]],[[P.t,P.s]],[[P.t,P.b]],
                                      [[P.u,P.c,P.d]],[[P.u,P.t,P.d]],[[P.c,P.t,P.d]],
                                     [[P.u,P.c,P.s]],[[P.u,P.t,P.s]],[[P.c,P.t,P.s]],
                                     [[P.u,P.c,P.b]],[[P.u,P.t,P.b]],[[P.c,P.t,P.b]],
                                     [[P.u,P.d,P.s]],[[P.u,P.d,P.b]],[[P.u,P.s,P.b]],
                                     [[P.c,P.d,P.s]],[[P.c,P.d,P.b]],[[P.c,P.s,P.b]],
                                     [[P.t,P.d,P.s]],[[P.t,P.d,P.b]],[[P.t,P.s,P.b]],
                                     [[P.u,P.c,P.d,P.s]],[[P.u,P.c,P.d,P.b]],[[P.u,P.c,P.s,P.b]],
                                     [[P.u,P.t,P.d,P.s]],[[P.u,P.t,P.d,P.b]],[[P.u,P.t,P.s,P.b]],
                                     [[P.c,P.t,P.d,P.s]],[[P.c,P.t,P.d,P.b]],[[P.c,P.t,P.s,P.b]]],
                    couplings = {(0,0,0):C.R2_WWGmGp,(0,0,1):C.R2_WWGmGpe,
                                 (0,0,2):C.R2_WWGmGpm,(0,0,3):C.R2_WWGmGptau,(0,0,4):C.R2_WWGmGpud,
                                 (0,0,5):C.R2_WWGmGpus,(0,0,6):C.R2_WWGmGpub,(0,0,7):C.R2_WWGmGpcd,
                                 (0,0,8):C.R2_WWGmGpcs,(0,0,9):C.R2_WWGmGpcb,(0,0,10):C.R2_WWGmGptd,
                                 (0,0,11):C.R2_WWGmGpts,(0,0,12):C.R2_WWGmGptb,
                                 (0,0,13):C.R2_WWGmGpucd,(0,0,14):C.R2_WWGmGputd,(0,0,15):C.R2_WWGmGpctd,
                                 (0,0,16):C.R2_WWGmGpucs,(0,0,17):C.R2_WWGmGputs,(0,0,18):C.R2_WWGmGpcts,
                                 (0,0,19):C.R2_WWGmGpucb,(0,0,20):C.R2_WWGmGputb,(0,0,21):C.R2_WWGmGpctb,
                                 (0,0,22):C.R2_WWGmGpuds,(0,0,23):C.R2_WWGmGpudb,(0,0,24):C.R2_WWGmGpusb,
                                 (0,0,25):C.R2_WWGmGpcds,(0,0,26):C.R2_WWGmGpcdb,(0,0,27):C.R2_WWGmGpcsb,
                                 (0,0,28):C.R2_WWGmGptds,(0,0,29):C.R2_WWGmGptdb,(0,0,30):C.R2_WWGmGptsb,
                                 (0,0,31):C.R2_WWGmGpucds,(0,0,32):C.R2_WWGmGpucdb,(0,0,33):C.R2_WWGmGpucsb,
                                 (0,0,34):C.R2_WWGmGputds,(0,0,35):C.R2_WWGmGputdb,(0,0,36):C.R2_WWGmGputsb,
                                 (0,0,37):C.R2_WWGmGpctds,(0,0,38):C.R2_WWGmGpctdb,(0,0,39):C.R2_WWGmGpctsb},
                    type = 'R2')

# ============== #
# Mixed QED-QCD  #
# ============== #

V_R2GDD2 = CTVertex(name = 'V_R2GDD2',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles =[[[P.d,P.H]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],                 
              couplings = {(0,0,0):C.R2_GDD2Cp,(0,1,0):C.R2_GDD2Cm,
                           (0,0,1):C.R2_GDD2Cpu,(0,1,1):C.R2_GDD2Cmu,
                           (0,0,2):C.R2_GDD2Cpc,(0,1,2):C.R2_GDD2Cmc,
                           (0,0,3):C.R2_GDD2Cpt,(0,1,3):C.R2_GDD2Cmt},
              type = 'R2')

V_R2GSS2 = CTVertex(name = 'V_R2GSS2',
              particles = [ P.s__tilde__, P.s, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles =[[[P.s,P.H]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],                 
              couplings = {(0,0,0):C.R2_GSS2Cp,(0,1,0):C.R2_GSS2Cm,
                           (0,0,1):C.R2_GSS2Cpu,(0,1,1):C.R2_GSS2Cmu,
                           (0,0,2):C.R2_GSS2Cpc,(0,1,2):C.R2_GSS2Cmc,
                           (0,0,3):C.R2_GSS2Cpt,(0,1,3):C.R2_GSS2Cmt},
              type = 'R2')

V_R2GBB2 = CTVertex(name = 'V_R2GBB2',
              particles = [ P.b__tilde__, P.b, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles =[[[P.b,P.H]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],                 
              couplings = {(0,0,0):C.R2_GBB2Cp,(0,1,0):C.R2_GBB2Cm,
                           (0,0,1):C.R2_GBB2Cpu,(0,1,1):C.R2_GBB2Cmu,
                           (0,0,2):C.R2_GBB2Cpc,(0,1,2):C.R2_GBB2Cmc,
                           (0,0,3):C.R2_GBB2Cpt,(0,1,3):C.R2_GBB2Cmt},
              type = 'R2')

V_R2GUU2 = CTVertex(name = 'V_R2GUU2',
              particles = [ P.u__tilde__, P.u, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles =[[[P.u,P.H]],[[P.d,P.W__plus__]],[[P.s,P.W__plus__]],[[P.b,P.W__plus__]]],                 
              couplings = {(0,0,0):C.R2_GUU2Cp,(0,1,0):C.R2_GUU2Cm,
                           (0,0,1):C.R2_GUU2Cpd,(0,1,1):C.R2_GUU2Cmd,
                           (0,0,2):C.R2_GUU2Cps,(0,1,2):C.R2_GUU2Cms,
                           (0,0,3):C.R2_GUU2Cpb,(0,1,3):C.R2_GUU2Cmb},
              type = 'R2')

V_R2GCC2 = CTVertex(name = 'V_R2GCC2',
              particles = [ P.c__tilde__, P.c, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles =[[[P.c,P.H]],[[P.d,P.W__plus__]],[[P.s,P.W__plus__]],[[P.b,P.W__plus__]]],                 
              couplings = {(0,0,0):C.R2_GCC2Cp,(0,1,0):C.R2_GCC2Cm,
                           (0,0,1):C.R2_GCC2Cpd,(0,1,1):C.R2_GCC2Cmd,
                           (0,0,2):C.R2_GCC2Cps,(0,1,2):C.R2_GCC2Cms,
                           (0,0,3):C.R2_GCC2Cpb,(0,1,3):C.R2_GCC2Cmb},
              type = 'R2')

V_R2GTT2 = CTVertex(name = 'V_R2GTT2',
              particles = [ P.t__tilde__, P.t, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles =[[[P.t,P.H]],[[P.d,P.W__plus__]],[[P.s,P.W__plus__]],[[P.b,P.W__plus__]]],                 
              couplings = {(0,0,0):C.R2_GTT2Cp,(0,1,0):C.R2_GTT2Cm,
                           (0,0,1):C.R2_GTT2Cpd,(0,1,1):C.R2_GTT2Cmd,
                           (0,0,2):C.R2_GTT2Cps,(0,1,2):C.R2_GTT2Cms,
                           (0,0,3):C.R2_GTT2Cpb,(0,1,3):C.R2_GTT2Cmb},
              type = 'R2')

################
# UV vertices  #
################

# ========= #
# Pure QCD  #
# ========= #

# There are the alpha_s renormalization vertices

# ggg
V_UV1eps3G = CTVertex(name = 'V_UV1eps3G',
              particles = [ P.G, P.G, P.G ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_3Gq,(0,0,1):C.UV_3Gb,(0,0,2):C.UV_3Gt,(0,0,3):C.UV_3Gg},
              type = 'UV')

# gggg
V_UV4G = CTVertex(name = 'V_UV1eps4G',
              particles = [ P.G, P.G, P.G, P.G ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVV1, L.VVVV3, L.VVVV4 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_4Gq,(0,0,1):C.UV_4Gb,(0,0,2):C.UV_4Gt,(0,0,3):C.UV_4Gg,
                           (1,1,0):C.UV_4Gq,(1,1,1):C.UV_4Gb,(1,1,2):C.UV_4Gt,(1,1,3):C.UV_4Gg,
                           (2,2,0):C.UV_4Gq,(2,2,1):C.UV_4Gb,(2,2,2):C.UV_4Gt,(2,2,3):C.UV_4Gg},
              type = 'UV')

# gdd~
V_UVGDD = CTVertex(name = 'V_UVGDD',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UV')

# guu~
V_UVGUU = CTVertex(name = 'V_UVGUU',
              particles = [ P.u__tilde__, P.u, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UV')

# gcc~
V_UVGCC = CTVertex(name = 'V_UVGCC',
              particles = [ P.c__tilde__, P.c, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UV')

# gss~
V_UVGSS = CTVertex(name = 'V_UVGSS',
              particles = [ P.s__tilde__, P.s, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UV')

# gbb~
V_UVGBB = CTVertex(name = 'V_UVGBB',
              particles = [ P.b__tilde__, P.b, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UV')

# gtt~
V_UVGTT = CTVertex(name = 'V_UVGTT',
              particles = [ P.t__tilde__, P.t, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UV')

# These are the mass renormalization vertices.

# b~b         
V_UVbMass = CTVertex(name = 'V_UVbMass',
               particles = [ P.b__tilde__, P.b ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2 ],
               loop_particles = [[[P.G,P.b]]],                   
               couplings = {(0,0,0):C.UV_bMass},
               type = 'UVmass') 

# t~t         
V_UVtMass = CTVertex(name = 'V_UVtMass',
               particles = [ P.t__tilde__, P.t ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2 ],
               loop_particles = [[[P.G,P.t]]],                   
               couplings = {(0,0,0):C.UV_tMass},
               type = 'UVmass')

# ============== #
# QED            #
# ============== #

V_UVtMassQED = CTVertex(name = 'V_UVtMassQED',
                        particles = [ P.t__tilde__, P.t ],
                        color = [ 'Identity(1,2)' ],
                        lorentz = [L.R2_QQ_2],
                        loop_particles = [[[P.A,P.t]]],
                        couplings = {(0,0,0):C.UV_tMassQED},
                        type = 'UVmass')

# ============== #
# Mixed QCD-QED  #
# ============== #

V_UVHtt = CTVertex(name = 'V_UVHtt',
              particles = [ P.t__tilde__, P.t, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              loop_particles = [[[P.G,P.t]]],                   
              couplings = {(0,0,0):C.UV_Htt},
              type = 'UV')

V_UVHbb = CTVertex(name = 'V_UVHbb',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              loop_particles = [[[P.G,P.b]]],
              couplings = {(0,0,0):C.UV_Hbb},
              type = 'UV')

