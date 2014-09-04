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


# The CT Vertex below is the one written by hand from the expression in the QCD R2 paper
# This implementation seems to yield correct result for g g > g g but not for g g > g g g  and g g > g g g g
# or anytime one of the outter gluon of the vertex is offshell.

# Keep in mind that Delta8(a,b) is 1/2 Tr(a,b)
# I am still puzzled by the normalization factors.

#V_R24G = CTVertex(name = 'V_R24G',
#              particles = [ P.G, P.G, P.G,  P.G ],
#              color = [ 'Tr(1,2)*Tr(3,4)' , 'Tr(1,3)*Tr(2,4)' , 'Tr(1,4)*Tr(2,3)', \
#                        'd(-1,1,2)*d(-1,3,4)' , 'd(-1,1,3)*d(-1,2,4)' , 'd(-1,1,4)*d(-1,2,3)'],
#              lorentz = [  L.R2_4G_1234, L.R2_4G_1324, L.R2_4G_1423 ],
#              loop_particles = [ [[P.G]], [[P.u],[P.d],[P.c],[P.s],[P.b],[P.t]] ],
#              couplings = {(0,0,0):C.GC_4GR2_Gluon_delta5,(0,1,0):C.GC_4GR2_Gluon_delta7,(0,2,0):C.GC_4GR2_Gluon_delta7, \
#                           (1,0,0):C.GC_4GR2_Gluon_delta7,(1,1,0):C.GC_4GR2_Gluon_delta5,(1,2,0):C.GC_4GR2_Gluon_delta7, \
#                           (2,0,0):C.GC_4GR2_Gluon_delta7,(2,1,0):C.GC_4GR2_Gluon_delta7,(2,2,0):C.GC_4GR2_Gluon_delta5, \
#                           (3,0,0):C.GC_4GR2_4Struct,(3,1,0):C.GC_4GR2_2Struct,(3,2,0):C.GC_4GR2_2Struct, \
#                           (4,0,0):C.GC_4GR2_2Struct,(4,1,0):C.GC_4GR2_4Struct,(4,2,0):C.GC_4GR2_2Struct, \
#                           (5,0,0):C.GC_4GR2_2Struct,(5,1,0):C.GC_4GR2_2Struct,(5,2,0):C.GC_4GR2_4Struct , \
#                           (0,0,1):C.GC_4GR2_Fermion_delta11,(0,1,1):C.GC_4GR2_Fermion_delta5,(0,2,1):C.GC_4GR2_Fermion_delta5, \
#                           (1,0,1):C.GC_4GR2_Fermion_delta5,(1,1,1):C.GC_4GR2_Fermion_delta11,(1,2,1):C.GC_4GR2_Fermion_delta5, \
#                           (2,0,1):C.GC_4GR2_Fermion_delta5,(2,1,1):C.GC_4GR2_Fermion_delta5,(2,2,1):C.GC_4GR2_Fermion_delta11, \
#                           (3,0,1):C.GC_4GR2_11Struct,(3,1,1):C.GC_4GR2_5Struct,(3,2,1):C.GC_4GR2_5Struct, \
#                           (4,0,1):C.GC_4GR2_5Struct,(4,1,1):C.GC_4GR2_11Struct,(4,2,1):C.GC_4GR2_5Struct, \
#                           (5,0,1):C.GC_4GR2_5Struct,(5,1,1):C.GC_4GR2_5Struct,(5,2,1):C.GC_4GR2_11Struct },
#              type = 'R2')

# The CT Vertex below is the one written automatically by FR
# Gives the same result as above for g g > g g but not as soon as one of the outter gluon is offshell.

V_R2RGA = CTVertex(name = 'V_R2RGA',
                   type = 'R2',
                   particles = [ P.G, P.G, P.G, P.G ],
                   color = [ 'd(-1,1,3)*d(-1,2,4)', 'd(-1,1,3)*f(-1,2,4)', 'd(-1,1,4)*d(-1,2,3)', 'd(-1,1,4)*f(-1,2,3)', 'd(-1,2,3)*f(-1,1,4)', 'd(-1,2,4)*f(-1,1,3)', 'f(-1,1,2)*f(-1,3,4)', 'f(-1,1,3)*f(-1,2,4)', 'f(-1,1,4)*f(-1,2,3)', 'Identity(1,2)*Identity(3,4)', 'Identity(1,3)*Identity(2,4)', 'Identity(1,4)*Identity(2,3)' ],
                   lorentz = [ L.R2RGA_VVVV10, L.R2RGA_VVVV2, L.R2RGA_VVVV3, L.R2RGA_VVVV5 ],
                   loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.G] ] ],
                   couplings = {(2,1,0):C.R2GC_137_43,(2,1,1):C.R2GC_137_44,(0,1,0):C.R2GC_137_43,(0,1,1):C.R2GC_137_44,(4,1,0):C.R2GC_145_58,(4,1,1):C.R2GC_145_59,(3,1,0):C.R2GC_145_58,(3,1,1):C.R2GC_145_59,(8,1,0):C.R2GC_138_45,(8,1,1):C.R2GC_138_46,(7,1,0):C.R2GC_144_56,(7,1,1):C.R2GC_144_57,(6,1,0):C.R2GC_141_50,(6,1,1):C.R2GC_141_51,(5,1,0):C.R2GC_145_58,(5,1,1):C.R2GC_145_59,(1,1,0):C.R2GC_145_58,(1,1,1):C.R2GC_145_59,(11,0,0):C.R2GC_140_48,(11,0,1):C.R2GC_140_49,(10,0,0):C.R2GC_140_48,(10,0,1):C.R2GC_140_49,(9,0,1):C.R2GC_139_47,(2,2,0):C.R2GC_137_43,(2,2,1):C.R2GC_137_44,(0,2,0):C.R2GC_137_43,(0,2,1):C.R2GC_137_44,(6,2,0):C.R2GC_142_52,(6,2,1):C.R2GC_142_53,(4,2,0):C.R2GC_145_58,(4,2,1):C.R2GC_145_59,(3,2,0):C.R2GC_145_58,(3,2,1):C.R2GC_145_59,(8,2,0):C.R2GC_144_56,(8,2,1):C.R2GC_144_57,(5,2,0):C.R2GC_145_58,(5,2,1):C.R2GC_145_59,(1,2,0):C.R2GC_145_58,(1,2,1):C.R2GC_145_59,(7,2,0):C.R2GC_138_45,(7,2,1):C.R2GC_138_46,(2,3,0):C.R2GC_137_43,(2,3,1):C.R2GC_137_44,(0,3,0):C.R2GC_137_43,(0,3,1):C.R2GC_137_44,(4,3,0):C.R2GC_145_58,(4,3,1):C.R2GC_145_59,(3,3,0):C.R2GC_145_58,(3,3,1):C.R2GC_145_59,(8,3,0):C.R2GC_143_54,(8,3,1):C.R2GC_143_55,(7,3,0):C.R2GC_143_54,(7,3,1):C.R2GC_143_55,(5,3,0):C.R2GC_145_58,(5,3,1):C.R2GC_145_59,(1,3,0):C.R2GC_145_58,(1,3,1):C.R2GC_145_59})

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
                  loop_particles = [[[P.H]],
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
                  loop_particles = [[[P.H]],
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
                  loop_particles = [[[P.H]],
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
                   loop_particles = [[[P.W__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],[[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]]],
                   couplings = {(0,0,0):C.R2_AAboson1,(0,1,0):C.R2_AAboson2,(0,2,0):C.R2_AAboson3,
                                (0,0,1):C.R2_AAl,(0,2,1):C.R2_AAe3,(0,0,2):C.R2_AAl,(0,2,2):C.R2_AAm3,(0,0,3):C.R2_AAl,(0,2,3):C.R2_AAtau3,
                                (0,0,4):C.R2_AAU,(0,2,4):C.R2_AAu3,(0,0,5):C.R2_AAD,(0,2,5):C.R2_AAd3,(0,0,6):C.R2_AAD,(0,2,6):C.R2_AAs3,
                                (0,0,7):C.R2_AAU,(0,2,7):C.R2_AAc3,(0,0,8):C.R2_AAD,(0,2,8):C.R2_AAb3,(0,0,9):C.R2_AAU,(0,2,9):C.R2_AAt3},
                   type = 'R2')

V_R2AZ = CTVertex(name = 'V_R2AZ',
                   particles = [ P.A, P.Z ],
                   color = [ '1' ],
                   lorentz = [L.R2_GG_1, L.R2_GG_2, L.R2_GG_3],
                   loop_particles = [[[P.W__plus__]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],[[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]]],
                   couplings = {(0,0,0):C.R2_AZboson1,(0,1,0):C.R2_AZboson2,(0,2,0):C.R2_AZboson3,
                                (0,0,1):C.R2_AZl,(0,2,1):C.R2_AZe3,(0,0,2):C.R2_AZl,(0,2,2):C.R2_AZm3,(0,0,3):C.R2_AZl,(0,2,3):C.R2_AZtau3,
                                (0,0,4):C.R2_AZU,(0,2,4):C.R2_AZu3,(0,0,5):C.R2_AZD,(0,2,5):C.R2_AZd3,(0,0,6):C.R2_AZD,(0,2,6):C.R2_AZs3,
                                (0,0,7):C.R2_AZU,(0,2,7):C.R2_AZc3,(0,0,8):C.R2_AZD,(0,2,8):C.R2_AZb3,(0,0,9):C.R2_AZU,(0,2,9):C.R2_AZt3},
                   type = 'R2')

V_R2ZZ = CTVertex(name = 'V_R2ZZ',
                   particles = [ P.Z, P.Z ],
                   color = [ '1' ],
                   lorentz = [L.R2_GG_1, L.R2_GG_2, L.R2_GG_3],
                   loop_particles = [[[P.H]],[[P.e__minus__]],[[P.m__minus__]],[[P.tt__minus__]],[[P.u]],[[P.d]],[[P.s]],[[P.c]],[[P.b]],[[P.t]],[[P.ve],[P.vm],[P.vt]]],
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
                   loop_particles = [[[P.H]],
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
               loop_particles = [[[P.Z,P.u]],[[P.W__plus__,P.d]],[[P.W__plus__,P.s]],[[P.W__plus__,P.b]]],
               couplings = {(0,0,0):C.R2_UUC0, (0,1,0):C.R2_UUCm,(0,2,0):C.R2_QQCp0,
                            (0,2,1):C.R2_QQCpud,(0,2,2):C.R2_QQCpus,(0,2,3):C.R2_QQCpub},
               type = 'R2')

V_R2CC2 = CTVertex(name = 'V_R2CC2',
               particles = [ P.c__tilde__, P.c ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
               loop_particles = [[[P.c, P.Z]],[[P.W__plus__,P.d]],[[P.W__plus__,P.s]],[[P.W__plus__,P.b]]],
               couplings = {(0,0,0):C.R2_CCC0, (0,1,0):C.R2_UUCm,(0,2,0):C.R2_QQCp0,
                            (0,2,1):C.R2_QQCpcd,(0,2,2):C.R2_QQCpcs,(0,2,3):C.R2_QQCpcb},
               type = 'R2')

V_R2TT2 = CTVertex(name = 'V_R2TT2',
               particles = [ P.t__tilde__, P.t ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
               loop_particles = [[[P.t, P.Z]],[[P.W__plus__,P.d]],[[P.W__plus__,P.s]],[[P.W__plus__,P.b]]],
               couplings = {(0,0,0):C.R2_TTC0, (0,1,0):C.R2_UUCm,(0,2,0):C.R2_QQCp0,
                            (0,2,1):C.R2_QQCptd,(0,2,2):C.R2_QQCpts,(0,2,3):C.R2_QQCptb},
               type = 'R2')

V_R2DD2 = CTVertex(name = 'V_R2DD2',
               particles = [ P.d__tilde__, P.d ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
               loop_particles = [[[P.d, P.Z]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],
               couplings = {(0,0,0):C.R2_DDC0, (0,1,0):C.R2_DDCm,(0,2,0):C.R2_QQCp0,
                            (0,2,1):C.R2_QQCpud,(0,2,2):C.R2_QQCpcd,(0,2,3):C.R2_QQCptd},
               type = 'R2')
    
V_R2SS2 = CTVertex(name = 'V_R2SS2',
               particles = [ P.s__tilde__, P.s ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
               loop_particles = [[[P.s, P.Z]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],
               couplings = {(0,0,0):C.R2_SSC0, (0,1,0):C.R2_DDCm,(0,2,0):C.R2_QQCp0,
                            (0,2,1):C.R2_QQCpus,(0,2,2):C.R2_QQCpcs,(0,2,3):C.R2_QQCpts},
               type = 'R2')

V_R2BB2 = CTVertex(name = 'V_R2BB2',
               particles = [ P.b__tilde__, P.b ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
               loop_particles = [[[P.b, P.Z]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],
               couplings = {(0,0,0):C.R2_BBC0, (0,1,0):C.R2_DDCm,(0,2,0):C.R2_QQCp0,
                            (0,2,1):C.R2_QQCpub,(0,2,2):C.R2_QQCpcb,(0,2,3):C.R2_QQCptb},
               type = 'R2')

V_R2ee = CTVertex(name = 'V_R2ee',
                      particles = [ P.e__plus__, P.e__minus__ ],
                      color = [ '1' ],
                      lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
                      loop_particles = [[[P.e__minus__, P.Z]],[[P.W__plus__,P.ve]]],
                      couplings = {(0,0,0):C.R2_EEC0,(0,1,0):C.R2_LLCm,(0,2,0):C.R2_LLCp0,(0,2,1):C.R2_LLCplv},
                      type = 'R2')

V_R2mm = CTVertex(name = 'V_R2mm',
                      particles = [ P.m__plus__, P.m__minus__ ],
                      color = [ '1' ],
                      lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
                      loop_particles = [[[P.m__minus__, P.Z]],[[P.W__plus__,P.vm]]],
                      couplings = {(0,0,0):C.R2_MMC0,(0,1,0):C.R2_LLCm,(0,2,0):C.R2_LLCp0,(0,2,1):C.R2_LLCplv},
                      type = 'R2')

V_R2tautau = CTVertex(name = 'V_R2tautau',
                      particles = [ P.tt__plus__, P.tt__minus__ ],
                      color = [ '1' ],
                      lorentz = [ L.R2_QQ_2, L.R2_QQ_3, L.R2_QQ_4 ],
                      loop_particles = [[[P.tt__minus__, P.Z]],[[P.W__plus__,P.vt]]],
                      couplings = {(0,0,0):C.R2_TATAC0,(0,1,0):C.R2_LLCm,(0,2,0):C.R2_LLCp0,(0,2,1):C.R2_LLCplv},
                      type = 'R2')

V_R2veve = CTVertex(name = 'V_R2veve',
                      particles = [ P.ve__tilde__, P.ve ],
                      color = [ '1' ],
                      lorentz = [ L.R2_QQ_4 ],
                      loop_particles = [[[P.ve,P.Z]],[[P.W__plus__,P.e__minus__]]],
                      couplings = {(0,0,0):C.R2_nunuCp0,(0,0,1):C.R2_LLCplv},
                      type = 'R2')

V_R2vmvm = CTVertex(name = 'V_R2vmvm',
                      particles = [ P.vm__tilde__, P.vm ],
                      color = [ '1' ],
                      lorentz = [ L.R2_QQ_4 ],
                      loop_particles = [[[P.vm, P.Z]],[[P.W__plus__,P.m__minus__]]],
                      couplings = {(0,0,0):C.R2_nunuCp0,(0,0,1):C.R2_LLCplv},
                      type = 'R2')

V_R2vtvt = CTVertex(name = 'V_R2vtvt',
                      particles = [ P.vt__tilde__, P.vt ],
                      color = [ '1' ],
                      lorentz = [ L.R2_QQ_4 ],
                      loop_particles = [[[P.vt, P.Z]],[[P.W__plus__,P.tt__minus__]]],
                      couplings = {(0,0,0):C.R2_nunuCp0,(0,0,1):C.R2_LLCplv},
                      type = 'R2')

# R2 with 3 external legs

# R2 for SFF~

V_uuH2 = CTVertex(name = 'V_uuH2',
                   particles = [ P.u__tilde__, P.u, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS5 ],
                   loop_particles = [[[P.Z, P.u]],[[P.W__plus__, P.d]],[[P.W__plus__, P.s]],[[P.W__plus__,P.b]]],
                   couplings = {(0,0,0):C.R2_Huu, (0,0,1):C.R2_Huu_d, (0,0,2):C.R2_Huu_s, (0,0,3):C.R2_Huu_b},
                   type = 'R2')

V_ccH2 = CTVertex(name = 'V_ccH2',
                   particles = [ P.c__tilde__, P.c, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS5 ],
                   loop_particles = [[[P.c,P.Z]],[[P.W__plus__, P.d]],[[P.W__plus__, P.s]],[[P.W__plus__,P.b]]],
                   couplings = {(0,0,0):C.R2_Hcc, (0,0,1):C.R2_Hcc_d, (0,0,2):C.R2_Hcc_s, (0,0,3):C.R2_Hcc_b},
                   type = 'R2')

V_ttH2 = CTVertex(name = 'V_ttH2',
                   particles = [ P.t__tilde__, P.t, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS5 ],
                   loop_particles = [[[P.t,P.Z]],[[P.W__plus__, P.d]],[[P.W__plus__, P.s]],[[P.W__plus__,P.b]]],
                   couplings = {(0,0,0):C.R2_Htt, (0,0,1):C.R2_Htt_d, (0,0,2):C.R2_Htt_s, (0,0,3):C.R2_Htt_b},
                   type = 'R2')

V_ddH2 = CTVertex(name = 'V_ddH2',
                   particles = [ P.d__tilde__, P.d, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS5 ],
                   loop_particles = [[[P.d,P.Z]],[[P.W__plus__, P.u]],[[P.W__plus__, P.c]],[[P.W__plus__,P.t]]],
                   couplings = {(0,0,0):C.R2_Hdd, (0,0,1):C.R2_Hdd_u, (0,0,2):C.R2_Hdd_c, (0,0,3):C.R2_Hdd_t},
                   type = 'R2')

V_ssH2 = CTVertex(name = 'V_ssH2',
                   particles = [ P.s__tilde__, P.s, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS5 ],
                   loop_particles = [[[P.s,P.Z]],[[P.W__plus__, P.u]],[[P.W__plus__, P.c]],[[P.W__plus__,P.t]]],
                   couplings = {(0,0,0):C.R2_Hss, (0,0,1):C.R2_Hss_u, (0,0,2):C.R2_Hss_c, (0,0,3):C.R2_Hss_t},
                   type = 'R2')

V_bbH2 = CTVertex(name = 'V_bbH2',
                   particles = [ P.b__tilde__, P.b, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS5 ],
                   loop_particles = [[[P.b,P.Z]],[[P.W__plus__, P.u]],[[P.W__plus__, P.c]],[[P.W__plus__,P.t]]],
                   couplings = {(0,0,0):C.R2_Hbb, (0,0,1):C.R2_Hbb_u, (0,0,2):C.R2_Hbb_c, (0,0,3):C.R2_Hbb_t},
                   type = 'R2')

V_eeH = CTVertex(name = 'V_eeH',
                   particles = [ P.e__plus__, P.e__minus__, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.FFS5 ],
                   loop_particles = [[[P.e__minus__,P.Z]],[[P.W__plus__, P.ve]]],
                   couplings = {(0,0,0):C.R2_Hee, (0,0,1):C.R2_Hee_v},
                   type = 'R2')

V_mmH = CTVertex(name = 'V_mmH',
                   particles = [ P.m__plus__, P.m__minus__, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.FFS5 ],
                   loop_particles = [[[P.m__minus__,P.Z]],[[P.W__plus__, P.vm]]],
                   couplings = {(0,0,0):C.R2_Hmm, (0,0,1):C.R2_Hmm_v},
                   type = 'R2')

V_tautauH = CTVertex(name = 'V_tautauH',
                   particles = [ P.tt__plus__, P.tt__minus__, P.H ],
                   color = [ '1' ],
                   lorentz = [ L.FFS5 ],
                   loop_particles = [[[P.tt__minus__,P.Z]],[[P.W__plus__, P.vt]]],
                   couplings = {(0,0,0):C.R2_Htautau, (0,0,1):C.R2_Htautau_v},
                   type = 'R2')

V_uuG02 = CTVertex(name = 'V_uuG02',
                   particles = [ P.u__tilde__, P.u, P.G0 ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.Z,P.u]],[[P.W__plus__, P.d]],[[P.W__plus__, P.s]],[[P.W__plus__,P.b]]],
                   couplings = {(0,0,0):C.R2_G0uu, (0,0,1):C.R2_G0uu_d, (0,0,2):C.R2_G0uu_s, (0,0,3):C.R2_G0uu_b},
                   type = 'R2')

V_ccG02 = CTVertex(name = 'V_ccG02',
                   particles = [ P.c__tilde__, P.c, P.G0 ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.Z,P.c]],[[P.W__plus__, P.d]],[[P.W__plus__, P.s]],[[P.W__plus__,P.b]]],
                   couplings = {(0,0,0):C.R2_G0cc, (0,0,1):C.R2_G0cc_d, (0,0,2):C.R2_G0cc_s, (0,0,3):C.R2_G0cc_b},
                   type = 'R2')

V_ttG02 = CTVertex(name = 'V_ttG02',
                   particles = [ P.t__tilde__, P.t, P.G0 ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.Z,P.t]],[[P.W__plus__, P.d]],[[P.W__plus__, P.s]],[[P.W__plus__,P.b]]],
                   couplings = {(0,0,0):C.R2_G0tt, (0,0,1):C.R2_G0tt_d, (0,0,2):C.R2_G0tt_s, (0,0,3):C.R2_G0tt_b},
                   type = 'R2')

V_ddG02 = CTVertex(name = 'V_ddG02',
                   particles = [ P.d__tilde__, P.d, P.G0 ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.d,P.Z]],[[P.W__plus__, P.u]],[[P.W__plus__, P.c]],[[P.W__plus__,P.t]]],
                   couplings = {(0,0,0):C.R2_G0dd, (0,0,1):C.R2_G0dd_u, (0,0,2):C.R2_G0dd_c, (0,0,3):C.R2_G0dd_t},
                   type = 'R2')

V_ssG02 = CTVertex(name = 'V_ssG02',
                   particles = [ P.s__tilde__, P.s, P.G0 ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.Z,P.s]],[[P.W__plus__, P.u]],[[P.W__plus__, P.c]],[[P.W__plus__,P.t]]],
                   couplings = {(0,0,0):C.R2_G0ss, (0,0,1):C.R2_G0ss_u, (0,0,2):C.R2_G0ss_c, (0,0,3):C.R2_G0ss_t},
                   type = 'R2')

V_bbG02 = CTVertex(name = 'V_bbG02',
                   particles = [ P.b__tilde__, P.b, P.G0 ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.Z,P.b]],[[P.W__plus__, P.u]],[[P.W__plus__, P.c]],[[P.W__plus__,P.t]]],
                   couplings = {(0,0,0):C.R2_G0bb, (0,0,1):C.R2_G0bb_u, (0,0,2):C.R2_G0bb_c, (0,0,3):C.R2_G0bb_t},
                   type = 'R2')

V_eeG0 = CTVertex(name = 'V_eeG0',
                   particles = [ P.e__plus__, P.e__minus__, P.G0 ],
                   color = [ '1' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.Z,P.e__minus__]],[[P.W__plus__, P.G__plus__, P.ve]]],
                   couplings = {(0,0,0):C.R2_G0ee, (0,0,1):C.R2_G0ee_v},
                   type = 'R2')

V_mmG0 = CTVertex(name = 'V_mmG0',
                   particles = [ P.m__plus__, P.m__minus__, P.G0 ],
                   color = [ '1' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.Z,P.m__minus__]],[[P.W__plus__, P.G__plus__,P.vm]]],
                   couplings = {(0,0,0):C.R2_G0mm, (0,0,1):C.R2_G0mm_v},
                   type = 'R2')

V_tautauG0 = CTVertex(name = 'V_tautauG0',
                   particles = [ P.tt__plus__, P.tt__minus__, P.G0 ],
                   color = [ '1' ],
                   lorentz = [ L.FFS2 ],
                   loop_particles = [[[P.Z,P.tt__minus__]],[[P.W__plus__,P.G__plus__,P.vt]]],
                   couplings = {(0,0,0):C.R2_G0tautau, (0,0,1):C.R2_G0tautau_v},
                   type = 'R2')

V_uxdGp2 = CTVertex(name = 'V_uxdGp2',
                    particles = [ P.u__tilde__, P.d, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.d,P.u]]],
                    couplings = {(0,0,0):C.R2_uxdGp2Cm,(0,1,0):C.R2_uxdGp2Cp},
                    type = 'R2')

V_uxsGp2 = CTVertex(name = 'V_uxsGp2',
                    particles = [ P.u__tilde__, P.s, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.u,P.s]]],
                    couplings = {(0,0,0):C.R2_uxsGp2Cm,(0,1,0):C.R2_uxsGp2Cp},
                    type = 'R2')

V_uxbGp2 = CTVertex(name = 'V_uxbGp2',
                    particles = [ P.u__tilde__, P.b, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.u,P.b]]],
                    couplings = {(0,0,0):C.R2_uxbGp2Cm,(0,1,0):C.R2_uxbGp2Cp},
                    type = 'R2')

V_cxdGp2 = CTVertex(name = 'V_cxdGp2',
                    particles = [ P.c__tilde__, P.d, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.c,P.d]]],
                    couplings = {(0,0,0):C.R2_cxdGp2Cm,(0,1,0):C.R2_cxdGp2Cp},
                    type = 'R2')

V_cxsGp2 = CTVertex(name = 'V_cxsGp2',
                    particles = [ P.c__tilde__, P.s, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.c,P.s]]],
                    couplings = {(0,0,0):C.R2_cxsGp2Cm,(0,1,0):C.R2_cxsGp2Cp},
                    type = 'R2')

V_cxbGp2 = CTVertex(name = 'V_cxbGp2',
                    particles = [ P.c__tilde__, P.b, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.c,P.b]]],
                    couplings = {(0,0,0):C.R2_cxbGp2Cm,(0,1,0):C.R2_cxbGp2Cp},
                    type = 'R2')

V_txdGp2 = CTVertex(name = 'V_txdGp2',
                    particles = [ P.t__tilde__, P.d, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.t,P.d]]],
                    couplings = {(0,0,0):C.R2_txdGp2Cm,(0,1,0):C.R2_txdGp2Cp},
                    type = 'R2')

V_txsGp2 = CTVertex(name = 'V_txsGp2',
                    particles = [ P.t__tilde__, P.s, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.t,P.s]]],
                    couplings = {(0,0,0):C.R2_txsGp2Cm,(0,1,0):C.R2_txsGp2Cp},
                    type = 'R2')

V_txbGp2 = CTVertex(name = 'V_txbGp2',
                    particles = [ P.t__tilde__, P.b, P.G__plus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_txbGp2Cm,(0,1,0):C.R2_txbGp2Cp},
                    type = 'R2')

V_dxuGm2 = CTVertex(name = 'V_dxuGm2',
                    particles = [ P.d__tilde__, P.u, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.u,P.d]]],
                    couplings = {(0,0,0):C.R2_dxuGm2Cm,(0,1,0):C.R2_dxuGm2Cp},
                    type = 'R2')

V_sxuGm2 = CTVertex(name = 'V_sxuGm2',
                    particles = [ P.s__tilde__, P.u, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.u,P.s]]],
                    couplings = {(0,0,0):C.R2_sxuGm2Cm,(0,1,0):C.R2_sxuGm2Cp},
                    type = 'R2')

V_bxuGm2 = CTVertex(name = 'V_bxuGm2',
                    particles = [ P.b__tilde__, P.u, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.u,P.b]]],
                    couplings = {(0,0,0):C.R2_bxuGm2Cm,(0,1,0):C.R2_bxuGm2Cp},
                    type = 'R2')

V_dxcGm2 = CTVertex(name = 'V_dxcGm2',
                    particles = [ P.d__tilde__, P.c, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.c,P.d]]],
                    couplings = {(0,0,0):C.R2_dxcGm2Cm,(0,1,0):C.R2_dxcGm2Cp},
                    type = 'R2')

V_sxcGm2 = CTVertex(name = 'V_sxcGm2',
                    particles = [ P.s__tilde__, P.c, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.c,P.s]]],
                    couplings = {(0,0,0):C.R2_sxcGm2Cm,(0,1,0):C.R2_sxcGm2Cp},
                    type = 'R2')

V_bxcGm2 = CTVertex(name = 'V_bxcGm2',
                    particles = [ P.b__tilde__, P.c, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.c,P.b]]],
                    couplings = {(0,0,0):C.R2_bxcGm2Cm,(0,1,0):C.R2_bxcGm2Cp},
                    type = 'R2')

V_dxtGm2 = CTVertex(name = 'V_dxtGm2',
                    particles = [ P.d__tilde__, P.t, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.t,P.d]]],
                    couplings = {(0,0,0):C.R2_dxtGm2Cm,(0,1,0):C.R2_dxtGm2Cp},
                    type = 'R2')

V_sxtGm2 = CTVertex(name = 'V_sxtGm2',
                    particles = [ P.s__tilde__, P.t, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.t,P.s]]],
                    couplings = {(0,0,0):C.R2_sxtGm2Cm,(0,1,0):C.R2_sxtGm2Cp},
                    type = 'R2')

V_bxtGm2 = CTVertex(name = 'V_bxtGm2',
                    particles = [ P.b__tilde__, P.t, P.G__minus__ ],
                    color = [ 'Identity(1,2)' ],
                    lorentz = [ L.FFS1, L.FFS3 ],
                    loop_particles = [[[P.Z,P.t,P.b]]],
                    couplings = {(0,0,0):C.R2_bxtGm2Cm,(0,1,0):C.R2_bxtGm2Cp},
                    type = 'R2')

V_vexeGp = CTVertex(name = 'V_vexeGp',
                    particles = [ P.ve__tilde__, P.e__minus__, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.FFS1 ],
                    loop_particles = [[[P.Z,P.W__plus__,P.e__minus__]]],
                    couplings = {(0,0,0):C.R2_vexeGpCm},
                    type = 'R2')

V_vmxmGp = CTVertex(name = 'V_vmxmGp',
                    particles = [ P.vm__tilde__, P.m__minus__, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.FFS1 ],
                    loop_particles = [[[P.Z,P.W__plus__,P.m__minus__]]],
                    couplings = {(0,0,0):C.R2_vmxmGpCm},
                    type = 'R2')

V_vtxtauGp = CTVertex(name = 'V_vtxtauGp',
                    particles = [ P.vt__tilde__, P.tt__minus__, P.G__plus__ ],
                    color = [ '1' ],
                    lorentz = [ L.FFS1 ],
                    loop_particles = [[[P.Z,P.W__plus__,P.tt__minus__]]],
                    couplings = {(0,0,0):C.R2_vtxtauGpCm},
                    type = 'R2')

V_exveGm = CTVertex(name = 'V_exveGm',
                    particles = [ P.e__plus__, P.ve, P.G__minus__ ],
                    color = [ '1' ],
                    lorentz = [ L.FFS3 ],
                    loop_particles = [[[P.Z,P.W__plus__,P.e__minus__]]],
                    couplings = {(0,0,0):C.R2_exveGmCp},
                    type = 'R2')

V_mxvmGm = CTVertex(name = 'V_mxvmGm',
                    particles = [ P.m__plus__, P.vm, P.G__minus__ ],
                    color = [ '1' ],
                    lorentz = [ L.FFS3 ],
                    loop_particles = [[[P.Z,P.W__plus__,P.m__minus__]]],
                    couplings = {(0,0,0):C.R2_mxvmGmCp},
                    type = 'R2')

V_tauxvtGm = CTVertex(name = 'V_tauxvtGm',
                    particles = [ P.tt__plus__, P.vt, P.G__minus__ ],
                    color = [ '1' ],
                    lorentz = [ L.FFS3 ],
                    loop_particles = [[[P.Z,P.W__plus__,P.tt__minus__]]],
                    couplings = {(0,0,0):C.R2_tauxvtGmCp},
                    type = 'R2')

# R2 for VFF~

V_R2ddA2 = CTVertex(name = 'V_R2ddA2',
              particles = [ P.d__tilde__, P.d, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.d,P.Z]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_ddA2Cp, (0,0,1):C.R2_ddA2Cp_u, (0,0,2):C.R2_ddA2Cp_c,(0,0,3):C.R2_ddA2Cp_t,
                           (0,1,0):C.R2_ddA2Cm, (0,1,1):C.R2_ddA2Cm_u, (0,1,2):C.R2_ddA2Cm_c,(0,1,3):C.R2_ddA2Cm_t},
              type = 'R2')

V_R2ssA2 = CTVertex(name = 'V_R2ssA2',
              particles = [ P.s__tilde__, P.s, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.s,P.Z]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_ssA2Cp, (0,0,1):C.R2_ssA2Cp_u, (0,0,2):C.R2_ssA2Cp_c,(0,0,3):C.R2_ssA2Cp_t,
                           (0,1,0):C.R2_ssA2Cm, (0,1,1):C.R2_ssA2Cm_u, (0,1,2):C.R2_ssA2Cm_c,(0,1,3):C.R2_ssA2Cm_t},
              type = 'R2')

V_R2bbA2 = CTVertex(name = 'V_R2bbA2',
              particles = [ P.b__tilde__, P.b, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.b,P.Z]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_bbA2Cp, (0,0,1):C.R2_bbA2Cp_u, (0,0,2):C.R2_bbA2Cp_c,(0,0,3):C.R2_bbA2Cp_t,
                           (0,1,0):C.R2_bbA2Cm, (0,1,1):C.R2_bbA2Cm_u, (0,1,2):C.R2_bbA2Cm_c,(0,1,3):C.R2_bbA2Cm_t},
              type = 'R2')

V_R2uuA2 = CTVertex(name = 'V_R2uuA2',
              particles = [ P.u__tilde__, P.u, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.u,P.Z]],[[P.W__plus__,P.d]],[[P.W__plus__,P.s]],[[P.W__plus__,P.b]]],
              couplings = {(0,0,0):C.R2_uuA2Cp, (0,0,1):C.R2_uuA2Cp_d, (0,0,2):C.R2_uuA2Cp_s,(0,0,3):C.R2_uuA2Cp_b,
                           (0,1,0):C.R2_uuA2Cm, (0,1,1):C.R2_uuA2Cm_d, (0,1,2):C.R2_uuA2Cm_s,(0,1,3):C.R2_uuA2Cm_b},
              type = 'R2')

V_R2ccA2 = CTVertex(name = 'V_R2ccA2',
              particles = [ P.c__tilde__, P.c, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.c,P.Z]],[[P.W__plus__,P.d]],[[P.W__plus__,P.s]],[[P.W__plus__,P.b]]],
              couplings = {(0,0,0):C.R2_ccA2Cp, (0,0,1):C.R2_ccA2Cp_d, (0,0,2):C.R2_ccA2Cp_s,(0,0,3):C.R2_ccA2Cp_b,
                           (0,1,0):C.R2_ccA2Cm, (0,1,1):C.R2_ccA2Cm_d, (0,1,2):C.R2_ccA2Cm_s,(0,1,3):C.R2_ccA2Cm_b},
              type = 'R2')

V_R2ttA2 = CTVertex(name = 'V_R2ttA2',
              particles = [ P.t__tilde__, P.t, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.t,P.Z]],[[P.W__plus__,P.d]],[[P.W__plus__,P.s]],[[P.W__plus__,P.b]]],
              couplings = {(0,0,0):C.R2_ttA2Cp, (0,0,1):C.R2_ttA2Cp_d, (0,0,2):C.R2_ttA2Cp_s,(0,0,3):C.R2_ttA2Cp_b,
                           (0,1,0):C.R2_ttA2Cm, (0,1,1):C.R2_ttA2Cm_d, (0,1,2):C.R2_ttA2Cm_s,(0,1,3):C.R2_ttA2Cm_b},
              type = 'R2')

V_R2eeA = CTVertex(name = 'V_R2eeA',
              particles = [ P.e__plus__, P.e__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.e__minus__,P.Z]],[[P.W__plus__,P.ve]]],
              couplings = {(0,0,0):C.R2_eeACp, (0,0,1):C.R2_llACp,(0,1,0):C.R2_eeACm},
              type = 'R2')

V_R2mmA = CTVertex(name = 'V_R2mmA',
              particles = [ P.m__plus__, P.m__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.m__minus__,P.Z]],[[P.W__plus__,P.vm]]],
              couplings = {(0,0,0):C.R2_mmACp, (0,0,1):C.R2_llACp,(0,1,0):C.R2_mmACm},
              type = 'R2')

V_R2tautauA = CTVertex(name = 'V_R2tautauA',
              particles = [ P.tt__plus__, P.tt__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.tt__minus__,P.Z]],[[P.W__plus__,P.vt]]],
              couplings = {(0,0,0):C.R2_tautauACp, (0,0,1):C.R2_llACp,(0,1,0):C.R2_tautauACm},
              type = 'R2')

V_R2veveA = CTVertex(name = 'V_R2veveA',
              particles = [ P.ve__tilde__, P.ve, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.W__plus__,P.e__minus__]]],
              couplings = {(0,0,0):C.R2_veveACp},
              type = 'R2')

V_R2vmvmA = CTVertex(name = 'V_R2vmvmA',
              particles = [ P.vm__tilde__, P.vm, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.W__plus__,P.m__minus__]]],
              couplings = {(0,0,0):C.R2_vmvmACp},
              type = 'R2')

V_R2vtvtA = CTVertex(name = 'V_R2vtvtA',
              particles = [ P.vt__tilde__, P.vt, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              loop_particles = [[[P.W__plus__,P.tt__minus__]]],
              couplings = {(0,0,0):C.R2_vtvtACp},
              type = 'R2')

V_R2ddZ2 = CTVertex(name = 'V_R2ddZ2',
              particles = [ P.d__tilde__, P.d, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.d,P.Z]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_ddZ2Cp, (0,0,1):C.R2_ddZ2Cp_u, (0,0,2):C.R2_ddZ2Cp_c,(0,0,3):C.R2_ddZ2Cp_t,
                           (0,1,0):C.R2_ddZ2Cm, (0,1,1):C.R2_ddZ2Cm_u, (0,1,2):C.R2_ddZ2Cm_c,(0,1,3):C.R2_ddZ2Cm_t},
              type = 'R2')

V_R2ssZ2 = CTVertex(name = 'V_R2ssZ2',
              particles = [ P.s__tilde__, P.s, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.s,P.Z]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_ssZ2Cp, (0,0,1):C.R2_ssZ2Cp_u, (0,0,2):C.R2_ssZ2Cp_c,(0,0,3):C.R2_ssZ2Cp_t,
                           (0,1,0):C.R2_ssZ2Cm, (0,1,1):C.R2_ssZ2Cm_u, (0,1,2):C.R2_ssZ2Cm_c,(0,1,3):C.R2_ssZ2Cm_t},
              type = 'R2')

V_R2bbZ2 = CTVertex(name = 'V_R2bbZ2',
              particles = [ P.b__tilde__, P.b, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.b,P.Z]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_bbZ2Cp, (0,0,1):C.R2_bbZ2Cp_u, (0,0,2):C.R2_bbZ2Cp_c,(0,0,3):C.R2_bbZ2Cp_t,
                           (0,1,0):C.R2_bbZ2Cm, (0,1,1):C.R2_bbZ2Cm_u, (0,1,2):C.R2_bbZ2Cm_c,(0,1,3):C.R2_bbZ2Cm_t},
              type = 'R2')

V_R2uuZ2 = CTVertex(name = 'V_R2uuZ2',
              particles = [ P.u__tilde__, P.u, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.u,P.Z]],[[P.W__plus__,P.d]],[[P.W__plus__,P.s]],[[P.W__plus__,P.b]]],
              couplings = {(0,0,0):C.R2_uuZ2Cp, (0,0,1):C.R2_uuZ2Cp_d, (0,0,2):C.R2_uuZ2Cp_s,(0,0,3):C.R2_uuZ2Cp_b,
                           (0,1,0):C.R2_uuZ2Cm, (0,1,1):C.R2_uuZ2Cm_d, (0,1,2):C.R2_uuZ2Cm_s,(0,1,3):C.R2_uuZ2Cm_b},
              type = 'R2')

V_R2ccZ2 = CTVertex(name = 'V_R2ccZ2',
              particles = [ P.c__tilde__, P.c, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.c,P.Z]],[[P.W__plus__,P.d]],[[P.W__plus__,P.s]],[[P.W__plus__,P.b]]],
              couplings = {(0,0,0):C.R2_ccZ2Cp, (0,0,1):C.R2_ccZ2Cp_d, (0,0,2):C.R2_ccZ2Cp_s,(0,0,3):C.R2_ccZ2Cp_b,
                           (0,1,0):C.R2_ccZ2Cm, (0,1,1):C.R2_ccZ2Cm_d, (0,1,2):C.R2_ccZ2Cm_s,(0,1,3):C.R2_ccZ2Cm_b},
              type = 'R2')

V_R2ttZ2 = CTVertex(name = 'V_R2ttZ2',
              particles = [ P.t__tilde__, P.t, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.t,P.Z]],[[P.W__plus__,P.d]],[[P.W__plus__,P.s]],[[P.W__plus__,P.b]]],
              couplings = {(0,0,0):C.R2_ttZ2Cp, (0,0,1):C.R2_ttZ2Cp_d, (0,0,2):C.R2_ttZ2Cp_s,(0,0,3):C.R2_ttZ2Cp_b,
                           (0,1,0):C.R2_ttZ2Cm, (0,1,1):C.R2_ttZ2Cm_d, (0,1,2):C.R2_ttZ2Cm_s,(0,1,3):C.R2_ttZ2Cm_b},
              type = 'R2')


V_R2eeZ = CTVertex(name = 'V_R2eeZ',
              particles = [ P.e__plus__, P.e__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.e__minus__,P.Z]],[[P.W__plus__,P.ve]]],
              couplings = {(0,0,0):C.R2_eeZCp, (0,0,1):C.R2_llZCp,(0,1,0):C.R2_eeZCm, (0,1,1):C.R2_eeZCm_v},
              type = 'R2')

V_R2mmZ = CTVertex(name = 'V_R2mmZ',
              particles = [ P.m__plus__, P.m__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.m__minus__,P.Z]],[[P.W__plus__,P.vm]]],
              couplings = {(0,0,0):C.R2_mmZCp, (0,0,1):C.R2_llZCp,(0,1,0):C.R2_mmZCm, (0,1,1):C.R2_mmZCm_v},
              type = 'R2')

V_R2tautauZ = CTVertex(name = 'V_R2tautauZ',
              particles = [ P.tt__plus__, P.tt__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles = [[[P.tt__minus__,P.Z]],[[P.W__plus__,P.vt]]],
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
                     loop_particles = [[[P.Z,P.u,P.d]]],
                     couplings = {(0,0,0):C.R2_dxuW2Cp},
                     type = 'R2')

V_R2sxuW2 = CTVertex(name = 'V_R2sxuW2',
                     particles = [ P.s__tilde__, P.u, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.u,P.s]]],
                     couplings = {(0,0,0):C.R2_sxuW2Cp},
                     type = 'R2')

V_R2bxuW2 = CTVertex(name = 'V_R2bxuW2',
                     particles = [ P.b__tilde__, P.u, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.u,P.b]]],
                     couplings = {(0,0,0):C.R2_bxuW2Cp},
                     type = 'R2')

V_R2dxcW2 = CTVertex(name = 'V_R2dxcW2',
                     particles = [ P.d__tilde__, P.c, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.c,P.d]]],
                     couplings = {(0,0,0):C.R2_dxcW2Cp},
                     type = 'R2')

V_R2sxcW2 = CTVertex(name = 'V_R2sxcW2',
                     particles = [ P.s__tilde__, P.c, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.c,P.s]]],
                     couplings = {(0,0,0):C.R2_sxcW2Cp},
                     type = 'R2')

V_R2bxcW2 = CTVertex(name = 'V_R2bxcW2',
                     particles = [ P.b__tilde__, P.c, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.c,P.b]]],
                     couplings = {(0,0,0):C.R2_bxcW2Cp},
                     type = 'R2')

V_R2dxtW2 = CTVertex(name = 'V_R2dxtW2',
                     particles = [ P.d__tilde__, P.t, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.t,P.d]]],
                     couplings = {(0,0,0):C.R2_dxtW2Cp},
                     type = 'R2')

V_R2sxtW2 = CTVertex(name = 'V_R2sxtW2',
                     particles = [ P.s__tilde__, P.t, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.s,P.t]]],
                     couplings = {(0,0,0):C.R2_sxtW2Cp},
                     type = 'R2')

V_R2bxtW2 = CTVertex(name = 'V_R2bxtW2',
                     particles = [ P.b__tilde__, P.t, P.W__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.t,P.b]]],
                     couplings = {(0,0,0):C.R2_bxtW2Cp},
                     type = 'R2')

V_R2uxdW2 = CTVertex(name = 'V_R2uxdW2',
                     particles = [ P.u__tilde__, P.d, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.u,P.d]]],
                     couplings = {(0,0,0):C.R2_uxdW2Cp},
                     type = 'R2')

V_R2uxsW2 = CTVertex(name = 'V_R2uxsW2',
                     particles = [ P.u__tilde__, P.s, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.u,P.s]]],
                     couplings = {(0,0,0):C.R2_uxsW2Cp},
                     type = 'R2')

V_R2uxbW2 = CTVertex(name = 'V_R2uxbW2',
                     particles = [ P.u__tilde__, P.b, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.u,P.b]]],
                     couplings = {(0,0,0):C.R2_uxbW2Cp},
                     type = 'R2')

V_R2cxdW2 = CTVertex(name = 'V_R2cxdW2',
                     particles = [ P.c__tilde__, P.d, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.c,P.d]]],
                     couplings = {(0,0,0):C.R2_cxdW2Cp},
                     type = 'R2')

V_R2cxsW2 = CTVertex(name = 'V_R2cxsW2',
                     particles = [ P.c__tilde__, P.s, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.c,P.s]]],
                     couplings = {(0,0,0):C.R2_cxsW2Cp},
                     type = 'R2')

V_R2cxbW2 = CTVertex(name = 'V_R2cxbW2',
                     particles = [ P.c__tilde__, P.b, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.c,P.b]]],
                     couplings = {(0,0,0):C.R2_cxbW2Cp},
                     type = 'R2')

V_R2txdW2 = CTVertex(name = 'V_R2txdW2',
                     particles = [ P.t__tilde__, P.d, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.t,P.d]]],
                     couplings = {(0,0,0):C.R2_txdW2Cp},
                     type = 'R2')


V_R2txsW2 = CTVertex(name = 'V_R2txsW2',
                     particles = [ P.t__tilde__, P.s, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.t,P.s]]],
                     couplings = {(0,0,0):C.R2_txsW2Cp},
                     type = 'R2')

V_R2txbW2 = CTVertex(name = 'V_R2txbW2',
                     particles = [ P.t__tilde__, P.b, P.W__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFV2 ],
                     loop_particles = [[[P.Z,P.t,P.b]]],
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
                   loop_particles = [[[P.H]],
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
              loop_particles =[[[P.d,P.Z]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_GDD2Cp,(0,1,0):C.R2_GDD2Cm,
                           (0,0,1):C.R2_GDD2Cpu,(0,1,1):C.R2_GDD2Cmu,
                           (0,0,2):C.R2_GDD2Cpc,(0,1,2):C.R2_GDD2Cmc,
                           (0,0,3):C.R2_GDD2Cpt,(0,1,3):C.R2_GDD2Cmt},
              type = 'R2')

V_R2GSS2 = CTVertex(name = 'V_R2GSS2',
              particles = [ P.s__tilde__, P.s, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles =[[[P.s,P.Z]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_GSS2Cp,(0,1,0):C.R2_GSS2Cm,
                           (0,0,1):C.R2_GSS2Cpu,(0,1,1):C.R2_GSS2Cmu,
                           (0,0,2):C.R2_GSS2Cpc,(0,1,2):C.R2_GSS2Cmc,
                           (0,0,3):C.R2_GSS2Cpt,(0,1,3):C.R2_GSS2Cmt},
              type = 'R2')

V_R2GBB2 = CTVertex(name = 'V_R2GBB2',
              particles = [ P.b__tilde__, P.b, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles =[[[P.b,P.Z]],[[P.W__plus__,P.u]],[[P.W__plus__,P.c]],[[P.W__plus__,P.t]]],
              couplings = {(0,0,0):C.R2_GBB2Cp,(0,1,0):C.R2_GBB2Cm,
                           (0,0,1):C.R2_GBB2Cpu,(0,1,1):C.R2_GBB2Cmu,
                           (0,0,2):C.R2_GBB2Cpc,(0,1,2):C.R2_GBB2Cmc,
                           (0,0,3):C.R2_GBB2Cpt,(0,1,3):C.R2_GBB2Cmt},
              type = 'R2')

V_R2GUU2 = CTVertex(name = 'V_R2GUU2',
              particles = [ P.u__tilde__, P.u, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles =[[[P.u,P.Z]],[[P.d,P.W__plus__]],[[P.s,P.W__plus__]],[[P.b,P.W__plus__]]],
              couplings = {(0,0,0):C.R2_GUU2Cp,(0,1,0):C.R2_GUU2Cm,
                           (0,0,1):C.R2_GUU2Cpd,(0,1,1):C.R2_GUU2Cmd,
                           (0,0,2):C.R2_GUU2Cps,(0,1,2):C.R2_GUU2Cms,
                           (0,0,3):C.R2_GUU2Cpb,(0,1,3):C.R2_GUU2Cmb},
              type = 'R2')

V_R2GCC2 = CTVertex(name = 'V_R2GCC2',
              particles = [ P.c__tilde__, P.c, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles =[[[P.c,P.Z]],[[P.d,P.W__plus__]],[[P.s,P.W__plus__]],[[P.b,P.W__plus__]]],
              couplings = {(0,0,0):C.R2_GCC2Cp,(0,1,0):C.R2_GCC2Cm,
                           (0,0,1):C.R2_GCC2Cpd,(0,1,1):C.R2_GCC2Cmd,
                           (0,0,2):C.R2_GCC2Cps,(0,1,2):C.R2_GCC2Cms,
                           (0,0,3):C.R2_GCC2Cpb,(0,1,3):C.R2_GCC2Cmb},
              type = 'R2')

V_R2GTT2 = CTVertex(name = 'V_R2GTT2',
              particles = [ P.t__tilde__, P.t, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              loop_particles =[[[P.t,P.Z]],[[P.d,P.W__plus__]],[[P.s,P.W__plus__]],[[P.b,P.W__plus__]]],
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

# These are the alpha_s renormalization vertices

# ggg
V_UV1eps3G = CTVertex(name = 'V_UV1eps3G',
              particles = [ P.G, P.G, P.G ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_3Gq,(0,0,1):C.UV_3Gb,(0,0,2):C.UV_3Gt,(0,0,3):C.UV_3Gg},
              type = 'UVtree')

# gggg
V_UV4G = CTVertex(name = 'V_UV1eps4G',
              particles = [ P.G, P.G, P.G, P.G ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVV1, L.VVVV3, L.VVVV4 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_4Gq,(0,0,1):C.UV_4Gb,(0,0,2):C.UV_4Gt,(0,0,3):C.UV_4Gg,
                           (1,1,0):C.UV_4Gq,(1,1,1):C.UV_4Gb,(1,1,2):C.UV_4Gt,(1,1,3):C.UV_4Gg,
                           (2,2,0):C.UV_4Gq,(2,2,1):C.UV_4Gb,(2,2,2):C.UV_4Gt,(2,2,3):C.UV_4Gg},
              type = 'UVtree')

# gdd~
V_UVGDD = CTVertex(name = 'V_UVGDD',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UVtree')

# guu~
V_UVGUU = CTVertex(name = 'V_UVGUU',
              particles = [ P.u__tilde__, P.u, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UVtree')

# gcc~
V_UVGCC = CTVertex(name = 'V_UVGCC',
              particles = [ P.c__tilde__, P.c, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UVtree')

# gss~
V_UVGSS = CTVertex(name = 'V_UVGSS',
              particles = [ P.s__tilde__, P.s, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UVtree')

# gbb~
V_UVGBB = CTVertex(name = 'V_UVGBB',
              particles = [ P.b__tilde__, P.b, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UVtree')

# gtt~
V_UVGTT = CTVertex(name = 'V_UVGTT',
              particles = [ P.t__tilde__, P.t, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UVtree')

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

# ================================== #
# QED                                #
# Generate automatically by WriteUFO #
# ================================== #

V_UVWmWpMassEW = CTVertex(name = 'V_UVWmWpMassEW',
                          particles = [ P.W__plus__, P.W__minus__ ],
                          color = [ '1' ],
                          lorentz = [ L.l_WmWpMass1, L.l_WmWpMass2, L.l_WmWpMass3 ],
                          loop_particles = [[[P.A,P.W__plus__]]],
                          couplings = {(0, 0, 0) : C.c_UVWmWpMass1EW, (0, 1, 0) : C.c_UVWmWpMass2EW, (0, 2, 0) : C.c_UVWmWpMass3EW},
                          type = 'UVmass')


V_UVZMassEW = CTVertex(name = 'V_UVZMassEW',
                       particles = [ P.Z, P.Z ],
                       color = [ '1' ],
                       lorentz = [ L.l_WmWpMass1, L.l_WmWpMass2, L.l_WmWpMass3 ],
                       loop_particles = [[[P.H]]],
                       couplings = {(0, 0, 0) : C.c_UVZMass1EW, (0, 1, 0) : C.c_UVZMass2EW, (0, 2, 0) : C.c_UVZMass3EW},
                       type = 'UVmass')


V_UVAMassEW = CTVertex(name = 'V_UVAMassEW',
                       particles = [ P.A, P.A ],
                       color = [ '1' ],
                       lorentz = [ L.l_WmWpMass1, L.l_WmWpMass2, L.l_WmWpMass3 ],
                       loop_particles = [[[P.e__minus__]]],
                       couplings = {(0, 0, 0) : C.c_UVAMass1EW, (0, 2, 0) : C.c_UVAMass2EW},
                       type = 'UVmass')


V_UVAZMassEW = CTVertex(name = 'V_UVAZMassEW',
                        particles = [ P.A, P.Z ],
                        color = [ '1' ],
                        lorentz = [ L.l_WmWpMass1, L.l_WmWpMass2, L.l_WmWpMass3 ],
                        loop_particles = [[[P.W__plus__]]],
                        couplings = {(0, 0, 0) : C.c_UVAZMass1EW, (0, 1, 0) : C.c_UVAZMass2EW, (0, 2, 0) : C.c_UVAZMass3EW},
                        type = 'UVmass')


V_UVGpWmMassEW = CTVertex(name = 'V_UVGpWmMassEW',
                          particles = [ P.G__plus__, P.W__minus__ ],
                          color = [ '1' ],
                          lorentz = [ L.l_GpWmMass4, L.l_GpWmMass5 ],
                          loop_particles = [[[P.W__plus__,P.A]]],
                          couplings = {(0, 0, 0) : C.c_UVGpWmMass1EW, (0, 1, 0) : C.c_UVGpWmMass2EW},
                          type = 'UVmass')


V_UVGmWpMassEW = CTVertex(name = 'V_UVGmWpMassEW',
                          particles = [ P.G__minus__, P.W__plus__ ],
                          color = [ '1' ],
                          lorentz = [ L.l_GpWmMass4, L.l_GpWmMass5 ],
                          loop_particles = [[[P.W__plus__,P.A]]],
                          couplings = {(0, 0, 0) : C.c_UVGmWpMass1EW, (0, 1, 0) : C.c_UVGmWpMass2EW},
                          type = 'UVmass')


V_UVG0ZMassEW = CTVertex(name = 'V_UVG0ZMassEW',
                         particles = [ P.G0, P.Z ],
                         color = [ '1' ],
                         lorentz = [ L.l_GpWmMass4, L.l_GpWmMass5 ],
                         loop_particles = [[[P.Z,P.H]]],
                         couplings = {(0, 0, 0) : C.c_UVG0ZMass1EW, (0, 1, 0) : C.c_UVG0ZMass2EW},
                         type = 'UVmass')


V_UVAG0MassEW = CTVertex(name = 'V_UVAG0MassEW',
                         particles = [ P.G0, P.A ],
                         color = [ '1' ],
                         lorentz = [ L.l_GpWmMass4, L.l_GpWmMass5 ],
                         loop_particles = [[[P.W__plus__,P.G__plus__]]],
                         couplings = {(0, 0, 0) : C.c_UVAG0Mass1EW, (0, 1, 0) : C.c_UVAG0Mass2EW},
                         type = 'UVmass')


V_UVHMassEW = CTVertex(name = 'V_UVHMassEW',
                       particles = [ P.H, P.H ],
                       color = [ '1' ],
                       lorentz = [ L.l_HMass6, L.l_HMass7 ],
                       loop_particles = [[[P.H]]],
                       couplings = {(0, 0, 0) : C.c_UVHMass1EW, (0, 1, 0) : C.c_UVHMass2EW},
                       type = 'UVmass')


V_UVG0MassEW = CTVertex(name = 'V_UVG0MassEW',
                        particles = [ P.G0, P.G0 ],
                        color = [ '1' ],
                        lorentz = [ L.l_HMass6, L.l_HMass7 ],
                        loop_particles = [[[P.H]]],
                        couplings = {(0, 0, 0) : C.c_UVG0Mass1EW, (0, 1, 0) : C.c_UVG0Mass2EW},
                        type = 'UVmass')


V_UVGmGpMassEW = CTVertex(name = 'V_UVGmGpMassEW',
                          particles = [ P.G__minus__, P.G__plus__ ],
                          color = [ '1' ],
                          lorentz = [ L.l_HMass6, L.l_HMass7 ],
                          loop_particles = [[[P.H]]],
                          couplings = {(0, 0, 0) : C.c_UVGmGpMass1EW, (0, 1, 0) : C.c_UVG0Mass2EW},
                          type = 'UVmass')


V_UVvevexMassEW = CTVertex(name = 'V_UVvevexMassEW',
                           particles = [ P.ve__tilde__, P.ve ],
                           color = [ '1' ],
                           lorentz = [ L.l_vevexMass8, L.l_vevexMass9, L.l_vevexMass10, L.l_vevexMass11 ],
                           loop_particles = [[[P.ve,P.Z]]],
                           couplings = {(0, 0, 0) : C.c_UVvevexMass1EW, (0, 1, 0) : C.c_UVvevexMass2EW},
                           type = 'UVmass')


V_UVvmvmxMassEW = CTVertex(name = 'V_UVvmvmxMassEW',
                           particles = [ P.vm__tilde__, P.vm ],
                           color = [ '1' ],
                           lorentz = [ L.l_vevexMass8, L.l_vevexMass9, L.l_vevexMass10, L.l_vevexMass11 ],
                           loop_particles = [[[P.vm,P.Z]]],
                           couplings = {(0, 0, 0) : C.c_UVvmvmxMass1EW, (0, 1, 0) : C.c_UVvmvmxMass2EW},
                           type = 'UVmass')


V_UVvtvtxMassEW = CTVertex(name = 'V_UVvtvtxMassEW',
                           particles = [ P.vt__tilde__, P.vt ],
                           color = [ '1' ],
                           lorentz = [ L.l_vevexMass8, L.l_vevexMass9, L.l_vevexMass10, L.l_vevexMass11 ],
                           loop_particles = [[[P.vt,P.Z]]],
                           couplings = {(0, 0, 0) : C.c_UVvtvtxMass1EW, (0, 1, 0) : C.c_UVvtvtxMass2EW},
                           type = 'UVmass')


V_UVemepMassEW = CTVertex(name = 'V_UVemepMassEW',
                          particles = [ P.e__plus__, P.e__minus__ ],
                          color = [ '1' ],
                          lorentz = [ L.l_vevexMass8, L.l_vevexMass9, L.l_vevexMass10, L.l_vevexMass11 ],
                          loop_particles = [[[P.e__minus__,P.A]]],
                          couplings = {(0, 0, 0) : C.c_UVemepMass1EW, (0, 1, 0) : C.c_UVemepMass2EW, (0, 2, 0) : C.c_UVemepMass3EW, (0, 3, 0) : C.c_UVemepMass4EW},
                          type = 'UVmass')


V_UVmmmpMassEW = CTVertex(name = 'V_UVmmmpMassEW',
                          particles = [ P.m__plus__, P.m__minus__ ],
                          color = [ '1' ],
                          lorentz = [ L.l_vevexMass8, L.l_vevexMass9, L.l_vevexMass10, L.l_vevexMass11 ],
                          loop_particles = [[[P.m__minus__,P.A]]],
                          couplings = {(0, 0, 0) : C.c_UVmmmpMass1EW, (0, 1, 0) : C.c_UVmmmpMass2EW, (0, 2, 0) : C.c_UVmmmpMass3EW, (0, 3, 0) : C.c_UVmmmpMass4EW},
                          type = 'UVmass')


V_UVttmttpMassEW = CTVertex(name = 'V_UVttmttpMassEW',
                            particles = [ P.tt__plus__, P.tt__minus__ ],
                            color = [ '1' ],
                            lorentz = [ L.l_vevexMass8, L.l_vevexMass9, L.l_vevexMass10, L.l_vevexMass11 ],
                            loop_particles = [[[P.tt__minus__,P.A]]],
                            couplings = {(0, 0, 0) : C.c_UVttmttpMass1EW, (0, 1, 0) : C.c_UVttmttpMass2EW, (0, 2, 0) : C.c_UVttmttpMass3EW, (0, 3, 0) : C.c_UVttmttpMass4EW},
                            type = 'UVmass')


V_UVuuxMassEW = CTVertex(name = 'V_UVuuxMassEW',
                         particles = [ P.u__tilde__, P.u ],
                         color = [ 'Identity(1,2)' ],
                         lorentz = [ L.l_vevexMass8, L.l_vevexMass9, L.l_vevexMass10, L.l_vevexMass11 ],
                         loop_particles = [[[P.u,P.A]]],
                         couplings = {(0, 0, 0) : C.c_UVuuxMass1EW, (0, 1, 0) : C.c_UVuuxMass2EW, (0, 2, 0) : C.c_UVuuxMass3EW, (0, 3, 0) : C.c_UVuuxMass4EW},
                         type = 'UVmass')


V_UVccxMassEW = CTVertex(name = 'V_UVccxMassEW',
                         particles = [ P.c__tilde__, P.c ],
                         color = [ 'Identity(1,2)' ],
                         lorentz = [ L.l_vevexMass8, L.l_vevexMass9, L.l_vevexMass10, L.l_vevexMass11 ],
                         loop_particles = [[[P.c,P.A]]],
                         couplings = {(0, 0, 0) : C.c_UVccxMass1EW, (0, 1, 0) : C.c_UVccxMass2EW, (0, 2, 0) : C.c_UVccxMass3EW, (0, 3, 0) : C.c_UVccxMass4EW},
                         type = 'UVmass')


V_UVttxMassEW = CTVertex(name = 'V_UVttxMassEW',
                         particles = [ P.t__tilde__, P.t ],
                         color = [ 'Identity(1,2)' ],
                         lorentz = [ L.l_vevexMass8, L.l_vevexMass9, L.l_vevexMass10, L.l_vevexMass11 ],
                         loop_particles = [[[P.t,P.A]]],
                         couplings = {(0, 0, 0) : C.c_UVttxMass1EW, (0, 1, 0) : C.c_UVttxMass2EW, (0, 2, 0) : C.c_UVttxMass3EW, (0, 3, 0) : C.c_UVttxMass4EW},
                         type = 'UVmass')


V_UVddxMassEW = CTVertex(name = 'V_UVddxMassEW',
                         particles = [ P.d__tilde__, P.d ],
                         color = [ 'Identity(1,2)' ],
                         lorentz = [ L.l_vevexMass8, L.l_vevexMass9, L.l_vevexMass10, L.l_vevexMass11 ],
                         loop_particles = [[[P.d,P.A]]],
                         couplings = {(0, 0, 0) : C.c_UVddxMass1EW, (0, 1, 0) : C.c_UVddxMass2EW, (0, 2, 0) : C.c_UVddxMass3EW, (0, 3, 0) : C.c_UVddxMass4EW},
                         type = 'UVmass')


V_UVssxMassEW = CTVertex(name = 'V_UVssxMassEW',
                         particles = [ P.s__tilde__, P.s ],
                         color = [ 'Identity(1,2)' ],
                         lorentz = [ L.l_vevexMass8, L.l_vevexMass9, L.l_vevexMass10, L.l_vevexMass11 ],
                         loop_particles = [[[P.s,P.A]]],
                         couplings = {(0, 0, 0) : C.c_UVssxMass1EW, (0, 1, 0) : C.c_UVssxMass2EW, (0, 2, 0) : C.c_UVssxMass3EW, (0, 3, 0) : C.c_UVssxMass4EW},
                         type = 'UVmass')


V_UVbbxMassEW = CTVertex(name = 'V_UVbbxMassEW',
                         particles = [ P.b__tilde__, P.b ],
                         color = [ 'Identity(1,2)' ],
                         lorentz = [ L.l_vevexMass8, L.l_vevexMass9, L.l_vevexMass10, L.l_vevexMass11 ],
                         loop_particles = [[[P.b,P.A]]],
                         couplings = {(0, 0, 0) : C.c_UVbbxMass1EW, (0, 1, 0) : C.c_UVbbxMass2EW, (0, 2, 0) : C.c_UVbbxMass3EW, (0, 3, 0) : C.c_UVbbxMass4EW},
                         type = 'UVmass')


V_UVWpWpWmWmEW = CTVertex(name = 'V_UVWpWpWmWmEW',
                          particles = [ P.W__plus__, P.W__plus__, P.W__minus__, P.W__minus__ ],
                          color = [ '1' ],
                          lorentz = [ L.l_WpWpWmWm12, L.l_WpWpWmWm13, L.l_WpWpWmWm14 ],
                          loop_particles = [  ],
                          couplings = {(0, 0, 0) : C.c_UVWpWpWmWm1EW, (0, 1, 0) : C.c_UVWpWpWmWm2EW, (0, 2, 0) : C.c_UVWpWpWmWm2EW},
                          type = 'UVtree')


V_UVWpWmZZEW = CTVertex(name = 'V_UVWpWmZZEW',
                        particles = [ P.W__plus__, P.W__minus__, P.Z, P.Z ],
                        color = [ '1' ],
                        lorentz = [ L.l_WpWpWmWm12, L.l_WpWpWmWm13, L.l_WpWpWmWm14 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVWpWmZZ1EW, (0, 1, 0) : C.c_UVWpWmZZ2EW, (0, 2, 0) : C.c_UVWpWmZZ2EW},
                        type = 'UVtree')


V_UVWpWmAZEW = CTVertex(name = 'V_UVWpWmAZEW',
                        particles = [ P.W__plus__, P.W__minus__, P.A, P.Z ],
                        color = [ '1' ],
                        lorentz = [ L.l_WpWpWmWm12, L.l_WpWpWmWm13, L.l_WpWpWmWm14 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVWpWmAZ1EW, (0, 1, 0) : C.c_UVWpWmAZ2EW, (0, 2, 0) : C.c_UVWpWmAZ2EW},
                        type = 'UVtree')


V_UVWpWmAAEW = CTVertex(name = 'V_UVWpWmAAEW',
                        particles = [ P.W__plus__, P.W__minus__, P.A, P.A ],
                        color = [ '1' ],
                        lorentz = [ L.l_WpWpWmWm12, L.l_WpWpWmWm13, L.l_WpWpWmWm14 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVWpWmAA1EW, (0, 1, 0) : C.c_UVWpWmAA2EW, (0, 2, 0) : C.c_UVWpWmAA2EW},
                        type = 'UVtree')


V_UVAWpWmEW = CTVertex(name = 'V_UVAWpWmEW',
                       particles = [ P.A, P.W__plus__, P.W__minus__ ],
                       color = [ '1' ],
                       lorentz = [ L.l_AWpWm15 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVAWpWm1EW},
                       type = 'UVtree')


V_UVZWpWmEW = CTVertex(name = 'V_UVZWpWmEW',
                       particles = [ P.Z, P.W__plus__, P.W__minus__ ],
                       color = [ '1' ],
                       lorentz = [ L.l_AWpWm15 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVZWpWm1EW},
                       type = 'UVtree')


V_UVHHHHEW = CTVertex(name = 'V_UVHHHHEW',
                      particles = [ P.H, P.H, P.H, P.H ],
                      color = [ '1' ],
                      lorentz = [ L.l_HHHH16 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVHHHH1EW},
                      type = 'UVtree')


V_UVHHG0G0EW = CTVertex(name = 'V_UVHHG0G0EW',
                        particles = [ P.H, P.H, P.G0, P.G0 ],
                        color = [ '1' ],
                        lorentz = [ L.l_HHHH16 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVHHG0G01EW},
                        type = 'UVtree')


V_UVHHGmGpEW = CTVertex(name = 'V_UVHHGmGpEW',
                        particles = [ P.H, P.H, P.G__minus__, P.G__plus__ ],
                        color = [ '1' ],
                        lorentz = [ L.l_HHHH16 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVHHGmGp1EW},
                        type = 'UVtree')


V_UVG0G0G0G0EW = CTVertex(name = 'V_UVG0G0G0G0EW',
                          particles = [ P.G0, P.G0, P.G0, P.G0 ],
                          color = [ '1' ],
                          lorentz = [ L.l_HHHH16 ],
                          loop_particles = [  ],
                          couplings = {(0, 0, 0) : C.c_UVG0G0G0G01EW},
                          type = 'UVtree')


V_UVG0G0GmGpEW = CTVertex(name = 'V_UVG0G0GmGpEW',
                          particles = [ P.G0, P.G0, P.G__minus__, P.G__plus__ ],
                          color = [ '1' ],
                          lorentz = [ L.l_HHHH16 ],
                          loop_particles = [  ],
                          couplings = {(0, 0, 0) : C.c_UVG0G0GmGp1EW},
                          type = 'UVtree')


V_UVGmGmGpGpEW = CTVertex(name = 'V_UVGmGmGpGpEW',
                          particles = [ P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
                          color = [ '1' ],
                          lorentz = [ L.l_HHHH16 ],
                          loop_particles = [  ],
                          couplings = {(0, 0, 0) : C.c_UVGmGmGpGp1EW},
                          type = 'UVtree')


V_UVHHHEW = CTVertex(name = 'V_UVHHHEW',
                     particles = [ P.H, P.H, P.H ],
                     color = [ '1' ],
                     lorentz = [ L.l_HHH17 ],
                     loop_particles = [  ],
                     couplings = {(0, 0, 0) : C.c_UVHHH1EW},
                     type = 'UVtree')


V_UVHG0G0EW = CTVertex(name = 'V_UVHG0G0EW',
                       particles = [ P.H, P.G0, P.G0 ],
                       color = [ '1' ],
                       lorentz = [ L.l_HHH17 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVHG0G01EW},
                       type = 'UVtree')


V_UVGmHGpEW = CTVertex(name = 'V_UVGmHGpEW',
                       particles = [ P.G__minus__, P.H, P.G__plus__ ],
                       color = [ '1' ],
                       lorentz = [ L.l_HHH17 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVGmHGp1EW},
                       type = 'UVtree')


V_UVHHWmWpEW = CTVertex(name = 'V_UVHHWmWpEW',
                        particles = [ P.H, P.H, P.W__minus__, P.W__plus__ ],
                        color = [ '1' ],
                        lorentz = [ L.l_HHWmWp18 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVHHWmWp1EW},
                        type = 'UVtree')


V_UVG0G0WmWpEW = CTVertex(name = 'V_UVG0G0WmWpEW',
                          particles = [ P.G0, P.G0, P.W__minus__, P.W__plus__ ],
                          color = [ '1' ],
                          lorentz = [ L.l_HHWmWp18 ],
                          loop_particles = [  ],
                          couplings = {(0, 0, 0) : C.c_UVG0G0WmWp1EW},
                          type = 'UVtree')


V_UVGmGpWmWpEW = CTVertex(name = 'V_UVGmGpWmWpEW',
                          particles = [ P.G__minus__, P.G__plus__, P.W__minus__, P.W__plus__ ],
                          color = [ '1' ],
                          lorentz = [ L.l_HHWmWp18 ],
                          loop_particles = [  ],
                          couplings = {(0, 0, 0) : C.c_UVGmGpWmWp1EW},
                          type = 'UVtree')


V_UVGmGpZZEW = CTVertex(name = 'V_UVGmGpZZEW',
                        particles = [ P.G__minus__, P.G__plus__, P.Z, P.Z ],
                        color = [ '1' ],
                        lorentz = [ L.l_HHWmWp18 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVGmGpZZ1EW},
                        type = 'UVtree')


V_UVGmGpAZEW = CTVertex(name = 'V_UVGmGpAZEW',
                        particles = [ P.G__minus__, P.G__plus__, P.A, P.Z ],
                        color = [ '1' ],
                        lorentz = [ L.l_HHWmWp18 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVGmGpAZ1EW},
                        type = 'UVtree')


V_UVGmGpAAEW = CTVertex(name = 'V_UVGmGpAAEW',
                        particles = [ P.G__minus__, P.G__plus__, P.A, P.A ],
                        color = [ '1' ],
                        lorentz = [ L.l_HHWmWp18 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVGmGpAA1EW},
                        type = 'UVtree')


V_UVHHZZEW = CTVertex(name = 'V_UVHHZZEW',
                      particles = [ P.H, P.H, P.Z, P.Z ],
                      color = [ '1' ],
                      lorentz = [ L.l_HHWmWp18 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVHHZZ1EW},
                      type = 'UVtree')


V_UVG0G0ZZEW = CTVertex(name = 'V_UVG0G0ZZEW',
                        particles = [ P.G0, P.G0, P.Z, P.Z ],
                        color = [ '1' ],
                        lorentz = [ L.l_HHWmWp18 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVG0G0ZZ1EW},
                        type = 'UVtree')


V_UVHHAZEW = CTVertex(name = 'V_UVHHAZEW',
                      particles = [ P.H, P.H, P.A, P.Z ],
                      color = [ '1' ],
                      lorentz = [ L.l_HHWmWp18 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVHHAZ1EW},
                      type = 'UVtree')


V_UVG0G0AZEW = CTVertex(name = 'V_UVG0G0AZEW',
                        particles = [ P.G0, P.G0, P.A, P.Z ],
                        color = [ '1' ],
                        lorentz = [ L.l_HHWmWp18 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVHHAZ1EW},
                        type = 'UVtree')


V_UVHGmWpZEW = CTVertex(name = 'V_UVHGmWpZEW',
                        particles = [ P.H, P.G__minus__, P.W__plus__, P.Z ],
                        color = [ '1' ],
                        lorentz = [ L.l_HHWmWp18 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVHGmWpZ1EW},
                        type = 'UVtree')


V_UVHGpWmZEW = CTVertex(name = 'V_UVHGpWmZEW',
                        particles = [ P.H, P.G__plus__, P.W__minus__, P.Z ],
                        color = [ '1' ],
                        lorentz = [ L.l_HHWmWp18 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVHGpWmZ1EW},
                        type = 'UVtree')


V_UVHGpWmAEW = CTVertex(name = 'V_UVHGpWmAEW',
                        particles = [ P.H, P.G__plus__, P.W__minus__, P.A ],
                        color = [ '1' ],
                        lorentz = [ L.l_HHWmWp18 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVHGpWmA1EW},
                        type = 'UVtree')


V_UVHGmWpAEW = CTVertex(name = 'V_UVHGmWpAEW',
                        particles = [ P.H, P.G__minus__, P.W__plus__, P.A ],
                        color = [ '1' ],
                        lorentz = [ L.l_HHWmWp18 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVHGmWpA1EW},
                        type = 'UVtree')


V_UVGpG0ZWmEW = CTVertex(name = 'V_UVGpG0ZWmEW',
                         particles = [ P.G__plus__, P.G0, P.Z, P.W__minus__ ],
                         color = [ '1' ],
                         lorentz = [ L.l_HHWmWp18 ],
                         loop_particles = [  ],
                         couplings = {(0, 0, 0) : C.c_UVGpG0ZWm1EW},
                         type = 'UVtree')


V_UVGmG0ZWpEW = CTVertex(name = 'V_UVGmG0ZWpEW',
                         particles = [ P.G__minus__, P.G0, P.Z, P.W__plus__ ],
                         color = [ '1' ],
                         lorentz = [ L.l_HHWmWp18 ],
                         loop_particles = [  ],
                         couplings = {(0, 0, 0) : C.c_UVGpG0ZWm1EW},
                         type = 'UVtree')


V_UVGpG0AWmEW = CTVertex(name = 'V_UVGpG0AWmEW',
                         particles = [ P.G__plus__, P.G0, P.A, P.W__minus__ ],
                         color = [ '1' ],
                         lorentz = [ L.l_HHWmWp18 ],
                         loop_particles = [  ],
                         couplings = {(0, 0, 0) : C.c_UVGpG0AWm1EW},
                         type = 'UVtree')


V_UVGmG0AWpEW = CTVertex(name = 'V_UVGmG0AWpEW',
                         particles = [ P.G__minus__, P.G0, P.A, P.W__plus__ ],
                         color = [ '1' ],
                         lorentz = [ L.l_HHWmWp18 ],
                         loop_particles = [  ],
                         couplings = {(0, 0, 0) : C.c_UVGpG0AWm1EW},
                         type = 'UVtree')


V_UVG0HAEW = CTVertex(name = 'V_UVG0HAEW',
                      particles = [ P.G0, P.H, P.A ],
                      color = [ '1' ],
                      lorentz = [ L.l_G0HA19 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVG0HA1EW},
                      type = 'UVtree')


V_UVG0HZEW = CTVertex(name = 'V_UVG0HZEW',
                      particles = [ P.G0, P.H, P.Z ],
                      color = [ '1' ],
                      lorentz = [ L.l_G0HA19 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVG0HZ1EW},
                      type = 'UVtree')


V_UVGpGmAEW = CTVertex(name = 'V_UVGpGmAEW',
                       particles = [ P.G__plus__, P.G__minus__, P.A ],
                       color = [ '1' ],
                       lorentz = [ L.l_G0HA19 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVGpGmA1EW},
                       type = 'UVtree')


V_UVGpGmZEW = CTVertex(name = 'V_UVGpGmZEW',
                       particles = [ P.G__plus__, P.G__minus__, P.Z ],
                       color = [ '1' ],
                       lorentz = [ L.l_G0HA19 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVGpGmZ1EW},
                       type = 'UVtree')


V_UVGpHWmEW = CTVertex(name = 'V_UVGpHWmEW',
                       particles = [ P.G__plus__, P.H, P.W__minus__ ],
                       color = [ '1' ],
                       lorentz = [ L.l_G0HA19 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVGpHWm1EW},
                       type = 'UVtree')


V_UVGmHWpEW = CTVertex(name = 'V_UVGmHWpEW',
                       particles = [ P.G__minus__, P.H, P.W__plus__ ],
                       color = [ '1' ],
                       lorentz = [ L.l_G0HA19 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVGpHWm1EW},
                       type = 'UVtree')


V_UVGpG0WmEW = CTVertex(name = 'V_UVGpG0WmEW',
                        particles = [ P.G__plus__, P.G0, P.W__minus__ ],
                        color = [ '1' ],
                        lorentz = [ L.l_G0HA19 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVGpG0Wm1EW},
                        type = 'UVtree')


V_UVGmG0WpEW = CTVertex(name = 'V_UVGmG0WpEW',
                        particles = [ P.G__minus__, P.G0, P.W__plus__ ],
                        color = [ '1' ],
                        lorentz = [ L.l_G0HA19 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVGmG0Wp1EW},
                        type = 'UVtree')


V_UVHWpWmEW = CTVertex(name = 'V_UVHWpWmEW',
                       particles = [ P.H, P.W__plus__, P.W__minus__ ],
                       color = [ '1' ],
                       lorentz = [ L.l_HWpWm20 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVHWpWm1EW},
                       type = 'UVtree')


V_UVHZZEW = CTVertex(name = 'V_UVHZZEW',
                     particles = [ P.H, P.Z, P.Z ],
                     color = [ '1' ],
                     lorentz = [ L.l_HWpWm20 ],
                     loop_particles = [  ],
                     couplings = {(0, 0, 0) : C.c_UVHZZ1EW},
                     type = 'UVtree')


V_UVHZAEW = CTVertex(name = 'V_UVHZAEW',
                     particles = [ P.H, P.Z, P.A ],
                     color = [ '1' ],
                     lorentz = [ L.l_HWpWm20 ],
                     loop_particles = [  ],
                     couplings = {(0, 0, 0) : C.c_UVHZA1EW},
                     type = 'UVtree')


V_UVGmWpZEW = CTVertex(name = 'V_UVGmWpZEW',
                       particles = [ P.G__minus__, P.W__plus__, P.Z ],
                       color = [ '1' ],
                       lorentz = [ L.l_HWpWm20 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVGmWpZ1EW},
                       type = 'UVtree')


V_UVGpWmZEW = CTVertex(name = 'V_UVGpWmZEW',
                       particles = [ P.G__plus__, P.W__minus__, P.Z ],
                       color = [ '1' ],
                       lorentz = [ L.l_HWpWm20 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVGpWmZ1EW},
                       type = 'UVtree')


V_UVGmWpAEW = CTVertex(name = 'V_UVGmWpAEW',
                       particles = [ P.G__minus__, P.W__plus__, P.A ],
                       color = [ '1' ],
                       lorentz = [ L.l_HWpWm20 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVGmWpA1EW},
                       type = 'UVtree')


V_UVGpWmAEW = CTVertex(name = 'V_UVGpWmAEW',
                       particles = [ P.G__plus__, P.W__minus__, P.A ],
                       color = [ '1' ],
                       lorentz = [ L.l_HWpWm20 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVGpWmA1EW},
                       type = 'UVtree')


V_UVvexveAEW = CTVertex(name = 'V_UVvexveAEW',
                        particles = [ P.ve__tilde__, P.ve, P.A ],
                        color = [ '1' ],
                        lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVvexveA1EW},
                        type = 'UVtree')


V_UVvmxvmAEW = CTVertex(name = 'V_UVvmxvmAEW',
                        particles = [ P.vm__tilde__, P.vm, P.A ],
                        color = [ '1' ],
                        lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVvexveA1EW},
                        type = 'UVtree')


V_UVvtxvtAEW = CTVertex(name = 'V_UVvtxvtAEW',
                        particles = [ P.vt__tilde__, P.vt, P.A ],
                        color = [ '1' ],
                        lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVvexveA1EW},
                        type = 'UVtree')


V_UVepemAEW = CTVertex(name = 'V_UVepemAEW',
                       particles = [ P.e__plus__, P.e__minus__, P.A ],
                       color = [ '1' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVepemA1EW, (0, 1, 0) : C.c_UVepemA2EW},
                       type = 'UVtree')


V_UVmpmmAEW = CTVertex(name = 'V_UVmpmmAEW',
                       particles = [ P.m__plus__, P.m__minus__, P.A ],
                       color = [ '1' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVmpmmA1EW, (0, 1, 0) : C.c_UVmpmmA2EW},
                       type = 'UVtree')


V_UVttpttmAEW = CTVertex(name = 'V_UVttpttmAEW',
                         particles = [ P.tt__plus__, P.tt__minus__, P.A ],
                         color = [ '1' ],
                         lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                         loop_particles = [  ],
                         couplings = {(0, 0, 0) : C.c_UVttpttmA1EW, (0, 1, 0) : C.c_UVttpttmA2EW},
                         type = 'UVtree')


V_UVuxuAEW = CTVertex(name = 'V_UVuxuAEW',
                      particles = [ P.u__tilde__, P.u, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [[[P.u, P.A]]],
                      couplings = {(0, 0, 0) : C.c_UVuxuA1EW, (0, 1, 0) : C.c_UVuxuA2EW},
                      type = 'UVtree')


V_UVcxuAEW = CTVertex(name = 'V_UVcxuAEW',
                      particles = [ P.c__tilde__, P.u, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVcxuA1EW, (0, 1, 0) : C.c_UVcxuA2EW},
                      type = 'UVtree')


V_UVtxuAEW = CTVertex(name = 'V_UVtxuAEW',
                      particles = [ P.t__tilde__, P.u, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVtxuA1EW, (0, 1, 0) : C.c_UVtxuA2EW},
                      type = 'UVtree')


V_UVuxcAEW = CTVertex(name = 'V_UVuxcAEW',
                      particles = [ P.u__tilde__, P.c, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVuxcA1EW, (0, 1, 0) : C.c_UVuxcA2EW},
                      type = 'UVtree')


V_UVcxcAEW = CTVertex(name = 'V_UVcxcAEW',
                      particles = [ P.c__tilde__, P.c, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [[[P.c, P.A]]],
                      couplings = {(0, 0, 0) : C.c_UVcxcA1EW, (0, 1, 0) : C.c_UVcxcA2EW},
                      type = 'UVtree')


V_UVtxcAEW = CTVertex(name = 'V_UVtxcAEW',
                      particles = [ P.t__tilde__, P.c, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVtxcA1EW, (0, 1, 0) : C.c_UVtxcA2EW},
                      type = 'UVtree')


V_UVuxtAEW = CTVertex(name = 'V_UVuxtAEW',
                      particles = [ P.u__tilde__, P.t, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVuxtA1EW, (0, 1, 0) : C.c_UVuxtA2EW},
                      type = 'UVtree')


V_UVcxtAEW = CTVertex(name = 'V_UVcxtAEW',
                      particles = [ P.c__tilde__, P.t, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVcxtA1EW, (0, 1, 0) : C.c_UVcxtA2EW},
                      type = 'UVtree')


V_UVtxtAEW = CTVertex(name = 'V_UVtxtAEW',
                      particles = [ P.t__tilde__, P.t, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [[[P.t,P.A]]],
                      couplings = {(0, 0, 0) : C.c_UVtxtA1EW, (0, 1, 0) : C.c_UVtxtA2EW},
                      type = 'UVtree')


V_UVdxdAEW = CTVertex(name = 'V_UVdxdAEW',
                      particles = [ P.d__tilde__, P.d, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [[[P.d,P.A]]],
                      couplings = {(0, 0, 0) : C.c_UVdxdA1EW, (0, 1, 0) : C.c_UVdxdA2EW},
                      type = 'UVtree')


V_UVsxdAEW = CTVertex(name = 'V_UVsxdAEW',
                      particles = [ P.s__tilde__, P.d, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVsxdA1EW, (0, 1, 0) : C.c_UVsxdA2EW},
                      type = 'UVtree')


V_UVbxdAEW = CTVertex(name = 'V_UVbxdAEW',
                      particles = [ P.b__tilde__, P.d, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVbxdA1EW, (0, 1, 0) : C.c_UVbxdA2EW},
                      type = 'UVtree')


V_UVdxsAEW = CTVertex(name = 'V_UVdxsAEW',
                      particles = [ P.d__tilde__, P.s, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVdxsA1EW, (0, 1, 0) : C.c_UVdxsA2EW},
                      type = 'UVtree')


V_UVsxsAEW = CTVertex(name = 'V_UVsxsAEW',
                      particles = [ P.s__tilde__, P.s, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [[[P.s,P.A]]],
                      couplings = {(0, 0, 0) : C.c_UVsxsA1EW, (0, 1, 0) : C.c_UVsxsA2EW},
                      type = 'UVtree')


V_UVbxsAEW = CTVertex(name = 'V_UVbxsAEW',
                      particles = [ P.b__tilde__, P.s, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVbxsA1EW, (0, 1, 0) : C.c_UVbxsA2EW},
                      type = 'UVtree')


V_UVdxbAEW = CTVertex(name = 'V_UVdxbAEW',
                      particles = [ P.d__tilde__, P.b, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVdxbA1EW, (0, 1, 0) : C.c_UVdxbA2EW},
                      type = 'UVtree')


V_UVsxbAEW = CTVertex(name = 'V_UVsxbAEW',
                      particles = [ P.s__tilde__, P.b, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVsxbA1EW, (0, 1, 0) : C.c_UVsxbA2EW},
                      type = 'UVtree')


V_UVbxbAEW = CTVertex(name = 'V_UVbxbAEW',
                      particles = [ P.b__tilde__, P.b, P.A ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [[[P.b,P.A]]],
                      couplings = {(0, 0, 0) : C.c_UVbxbA1EW, (0, 1, 0) : C.c_UVbxbA2EW},
                      type = 'UVtree')


V_UVvexveZEW = CTVertex(name = 'V_UVvexveZEW',
                        particles = [ P.ve__tilde__, P.ve, P.Z ],
                        color = [ '1' ],
                        lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVvexveZ1EW},
                        type = 'UVtree')


V_UVvmxvmZEW = CTVertex(name = 'V_UVvmxvmZEW',
                        particles = [ P.vm__tilde__, P.vm, P.Z ],
                        color = [ '1' ],
                        lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVvmxvmZ1EW},
                        type = 'UVtree')


V_UVvtxvtZEW = CTVertex(name = 'V_UVvtxvtZEW',
                        particles = [ P.vt__tilde__, P.vt, P.Z ],
                        color = [ '1' ],
                        lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVvtxvtZ1EW},
                        type = 'UVtree')


V_UVepemZEW = CTVertex(name = 'V_UVepemZEW',
                       particles = [ P.e__plus__, P.e__minus__, P.Z ],
                       color = [ '1' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVepemZ1EW, (0, 1, 0) : C.c_UVepemZ2EW},
                       type = 'UVtree')


V_UVmpmmZEW = CTVertex(name = 'V_UVmpmmZEW',
                       particles = [ P.m__plus__, P.m__minus__, P.Z ],
                       color = [ '1' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVmpmmZ1EW, (0, 1, 0) : C.c_UVmpmmZ2EW},
                       type = 'UVtree')


V_UVttpttmZEW = CTVertex(name = 'V_UVttpttmZEW',
                         particles = [ P.tt__plus__, P.tt__minus__, P.Z ],
                         color = [ '1' ],
                         lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                         loop_particles = [  ],
                         couplings = {(0, 0, 0) : C.c_UVttpttmZ1EW, (0, 1, 0) : C.c_UVttpttmZ2EW},
                         type = 'UVtree')


V_UVuxuZEW = CTVertex(name = 'V_UVuxuZEW',
                      particles = [ P.u__tilde__, P.u, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVuxuZ1EW, (0, 1, 0) : C.c_UVuxuZ2EW},
                      type = 'UVtree')


V_UVcxuZEW = CTVertex(name = 'V_UVcxuZEW',
                      particles = [ P.c__tilde__, P.u, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVcxuZ1EW, (0, 1, 0) : C.c_UVcxuZ2EW},
                      type = 'UVtree')


V_UVtxuZEW = CTVertex(name = 'V_UVtxuZEW',
                      particles = [ P.t__tilde__, P.u, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVtxuZ1EW, (0, 1, 0) : C.c_UVtxuZ2EW},
                      type = 'UVtree')


V_UVuxcZEW = CTVertex(name = 'V_UVuxcZEW',
                      particles = [ P.u__tilde__, P.c, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVuxcZ1EW, (0, 1, 0) : C.c_UVuxcZ2EW},
                      type = 'UVtree')


V_UVcxcZEW = CTVertex(name = 'V_UVcxcZEW',
                      particles = [ P.c__tilde__, P.c, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVcxcZ1EW, (0, 1, 0) : C.c_UVcxcZ2EW},
                      type = 'UVtree')


V_UVtxcZEW = CTVertex(name = 'V_UVtxcZEW',
                      particles = [ P.t__tilde__, P.c, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVtxcZ1EW, (0, 1, 0) : C.c_UVtxcZ2EW},
                      type = 'UVtree')


V_UVuxtZEW = CTVertex(name = 'V_UVuxtZEW',
                      particles = [ P.u__tilde__, P.t, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVuxtZ1EW, (0, 1, 0) : C.c_UVuxtZ2EW},
                      type = 'UVtree')


V_UVcxtZEW = CTVertex(name = 'V_UVcxtZEW',
                      particles = [ P.c__tilde__, P.t, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVcxtZ1EW, (0, 1, 0) : C.c_UVcxtZ2EW},
                      type = 'UVtree')


V_UVtxtZEW = CTVertex(name = 'V_UVtxtZEW',
                      particles = [ P.t__tilde__, P.t, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVtxtZ1EW, (0, 1, 0) : C.c_UVtxtZ2EW},
                      type = 'UVtree')


V_UVdxdZEW = CTVertex(name = 'V_UVdxdZEW',
                      particles = [ P.d__tilde__, P.d, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVdxdZ1EW, (0, 1, 0) : C.c_UVdxdZ2EW},
                      type = 'UVtree')


V_UVsxdZEW = CTVertex(name = 'V_UVsxdZEW',
                      particles = [ P.s__tilde__, P.d, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVsxdZ1EW, (0, 1, 0) : C.c_UVsxdZ2EW},
                      type = 'UVtree')


V_UVbxdZEW = CTVertex(name = 'V_UVbxdZEW',
                      particles = [ P.b__tilde__, P.d, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVbxdZ1EW, (0, 1, 0) : C.c_UVbxdZ2EW},
                      type = 'UVtree')


V_UVdxsZEW = CTVertex(name = 'V_UVdxsZEW',
                      particles = [ P.d__tilde__, P.s, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVdxsZ1EW, (0, 1, 0) : C.c_UVdxsZ2EW},
                      type = 'UVtree')


V_UVsxsZEW = CTVertex(name = 'V_UVsxsZEW',
                      particles = [ P.s__tilde__, P.s, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVsxsZ1EW, (0, 1, 0) : C.c_UVsxsZ2EW},
                      type = 'UVtree')


V_UVbxsZEW = CTVertex(name = 'V_UVbxsZEW',
                      particles = [ P.b__tilde__, P.s, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVbxsZ1EW, (0, 1, 0) : C.c_UVbxsZ2EW},
                      type = 'UVtree')


V_UVdxbZEW = CTVertex(name = 'V_UVdxbZEW',
                      particles = [ P.d__tilde__, P.b, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVdxbZ1EW, (0, 1, 0) : C.c_UVdxbZ2EW},
                      type = 'UVtree')


V_UVsxbZEW = CTVertex(name = 'V_UVsxbZEW',
                      particles = [ P.s__tilde__, P.b, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVsxbZ1EW, (0, 1, 0) : C.c_UVsxbZ2EW},
                      type = 'UVtree')


V_UVbxbZEW = CTVertex(name = 'V_UVbxbZEW',
                      particles = [ P.b__tilde__, P.b, P.Z ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVbxbZ1EW, (0, 1, 0) : C.c_UVbxbZ2EW},
                      type = 'UVtree')


V_UVepveWmEW = CTVertex(name = 'V_UVepveWmEW',
                        particles = [ P.e__plus__, P.ve, P.W__minus__ ],
                        color = [ '1' ],
                        lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVepveWm1EW},
                        type = 'UVtree')


V_UVmpvmWmEW = CTVertex(name = 'V_UVmpvmWmEW',
                        particles = [ P.m__plus__, P.vm, P.W__minus__ ],
                        color = [ '1' ],
                        lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVmpvmWm1EW},
                        type = 'UVtree')


V_UVttpvtWmEW = CTVertex(name = 'V_UVttpvtWmEW',
                         particles = [ P.tt__plus__, P.vt, P.W__minus__ ],
                         color = [ '1' ],
                         lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                         loop_particles = [  ],
                         couplings = {(0, 0, 0) : C.c_UVttpvtWm1EW},
                         type = 'UVtree')


V_UVvexemWpEW = CTVertex(name = 'V_UVvexemWpEW',
                         particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
                         color = [ '1' ],
                         lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                         loop_particles = [  ],
                         couplings = {(0, 0, 0) : C.c_UVvexemWp1EW},
                         type = 'UVtree')


V_UVvmxmmWpEW = CTVertex(name = 'V_UVvmxmmWpEW',
                         particles = [ P.vm__tilde__, P.m__minus__, P.W__plus__ ],
                         color = [ '1' ],
                         lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                         loop_particles = [  ],
                         couplings = {(0, 0, 0) : C.c_UVvmxmmWp1EW},
                         type = 'UVtree')


V_UVvtxttmWpEW = CTVertex(name = 'V_UVvtxttmWpEW',
                          particles = [ P.vt__tilde__, P.tt__minus__, P.W__plus__ ],
                          color = [ '1' ],
                          lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                          loop_particles = [  ],
                          couplings = {(0, 0, 0) : C.c_UVvtxttmWp1EW},
                          type = 'UVtree')


V_UVdxuWmEW = CTVertex(name = 'V_UVdxuWmEW',
                       particles = [ P.d__tilde__, P.u, P.W__minus__ ],
                       color = [ '1', 'Identity(1,2)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVdxuWm1EW},
                       type = 'UVtree')


V_UVsxuWmEW = CTVertex(name = 'V_UVsxuWmEW',
                       particles = [ P.s__tilde__, P.u, P.W__minus__ ],
                       color = [ '1', 'Identity(1,2)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVsxuWm1EW},
                       type = 'UVtree')


V_UVbxuWmEW = CTVertex(name = 'V_UVbxuWmEW',
                       particles = [ P.b__tilde__, P.u, P.W__minus__ ],
                       color = [ '1', 'Identity(1,2)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVbxuWm1EW},
                       type = 'UVtree')


V_UVdxcWmEW = CTVertex(name = 'V_UVdxcWmEW',
                       particles = [ P.d__tilde__, P.c, P.W__minus__ ],
                       color = [ '1', 'Identity(1,2)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVdxcWm1EW},
                       type = 'UVtree')


V_UVsxcWmEW = CTVertex(name = 'V_UVsxcWmEW',
                       particles = [ P.s__tilde__, P.c, P.W__minus__ ],
                       color = [ '1', 'Identity(1,2)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVsxcWm1EW},
                       type = 'UVtree')


V_UVbxcWmEW = CTVertex(name = 'V_UVbxcWmEW',
                       particles = [ P.b__tilde__, P.c, P.W__minus__ ],
                       color = [ '1', 'Identity(1,2)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVbxcWm1EW},
                       type = 'UVtree')


V_UVdxtWmEW = CTVertex(name = 'V_UVdxtWmEW',
                       particles = [ P.d__tilde__, P.t, P.W__minus__ ],
                       color = [ '1', 'Identity(1,2)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVdxtWm1EW},
                       type = 'UVtree')


V_UVsxtWmEW = CTVertex(name = 'V_UVsxtWmEW',
                       particles = [ P.s__tilde__, P.t, P.W__minus__ ],
                       color = [ '1', 'Identity(1,2)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVsxtWm1EW},
                       type = 'UVtree')


V_UVbxtWmEW = CTVertex(name = 'V_UVbxtWmEW',
                       particles = [ P.b__tilde__, P.t, P.W__minus__ ],
                       color = [ '1', 'Identity(1,2)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVbxtWm1EW},
                       type = 'UVtree')


V_UVuxdWpEW = CTVertex(name = 'V_UVuxdWpEW',
                       particles = [ P.u__tilde__, P.d, P.W__plus__ ],
                       color = [ '1', 'Identity(2,1)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVuxdWp1EW},
                       type = 'UVtree')


V_UVcxdWpEW = CTVertex(name = 'V_UVcxdWpEW',
                       particles = [ P.c__tilde__, P.d, P.W__plus__ ],
                       color = [ '1', 'Identity(2,1)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVcxdWp1EW},
                       type = 'UVtree')


V_UVtxdWpEW = CTVertex(name = 'V_UVtxdWpEW',
                       particles = [ P.t__tilde__, P.d, P.W__plus__ ],
                       color = [ '1', 'Identity(2,1)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVtxdWp1EW},
                       type = 'UVtree')


V_UVuxsWpEW = CTVertex(name = 'V_UVuxsWpEW',
                       particles = [ P.u__tilde__, P.s, P.W__plus__ ],
                       color = [ '1', 'Identity(2,1)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVuxsWp1EW},
                       type = 'UVtree')


V_UVcxsWpEW = CTVertex(name = 'V_UVcxsWpEW',
                       particles = [ P.c__tilde__, P.s, P.W__plus__ ],
                       color = [ '1', 'Identity(2,1)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVcxsWp1EW},
                       type = 'UVtree')


V_UVtxsWpEW = CTVertex(name = 'V_UVtxsWpEW',
                       particles = [ P.t__tilde__, P.s, P.W__plus__ ],
                       color = [ '1', 'Identity(2,1)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVtxsWp1EW},
                       type = 'UVtree')


V_UVuxbWpEW = CTVertex(name = 'V_UVuxbWpEW',
                       particles = [ P.u__tilde__, P.b, P.W__plus__ ],
                       color = [ '1', 'Identity(2,1)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVuxbWp1EW},
                       type = 'UVtree')


V_UVcxbWpEW = CTVertex(name = 'V_UVcxbWpEW',
                       particles = [ P.c__tilde__, P.b, P.W__plus__ ],
                       color = [ '1', 'Identity(2,1)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVcxbWp1EW},
                       type = 'UVtree')


V_UVtxbWpEW = CTVertex(name = 'V_UVtxbWpEW',
                       particles = [ P.t__tilde__, P.b, P.W__plus__ ],
                       color = [ '1', 'Identity(2,1)' ],
                       lorentz = [ L.l_vexveA21, L.l_vexveA22 ],
                       loop_particles = [  ],
                       couplings = {(1, 0, 0) : C.c_UVtxbWp1EW},
                       type = 'UVtree')


V_UVepemHEW = CTVertex(name = 'V_UVepemHEW',
                       particles = [ P.e__plus__, P.e__minus__, P.H ],
                       color = [ '1' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVepemH1EW, (0, 1, 0) : C.c_UVepemH2EW},
                       type = 'UVtree')


V_UVmpmmHEW = CTVertex(name = 'V_UVmpmmHEW',
                       particles = [ P.m__plus__, P.m__minus__, P.H ],
                       color = [ '1' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVmpmmH1EW, (0, 1, 0) : C.c_UVmpmmH2EW},
                       type = 'UVtree')


V_UVttpttmHEW = CTVertex(name = 'V_UVttpttmHEW',
                         particles = [ P.tt__plus__, P.tt__minus__, P.H ],
                         color = [ '1' ],
                         lorentz = [ L.l_epemH23, L.l_epemH24 ],
                         loop_particles = [  ],
                         couplings = {(0, 0, 0) : C.c_UVttpttmH1EW, (0, 1, 0) : C.c_UVttpttmH2EW},
                         type = 'UVtree')


V_UVuxuHEW = CTVertex(name = 'V_UVuxuHEW',
                      particles = [ P.u__tilde__, P.u, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVuxuH1EW, (0, 1, 0) : C.c_UVuxuH2EW},
                      type = 'UVtree')


V_UVcxuHEW = CTVertex(name = 'V_UVcxuHEW',
                      particles = [ P.c__tilde__, P.u, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVcxuH1EW, (0, 1, 0) : C.c_UVcxuH2EW},
                      type = 'UVtree')


V_UVtxuHEW = CTVertex(name = 'V_UVtxuHEW',
                      particles = [ P.t__tilde__, P.u, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVtxuH1EW, (0, 1, 0) : C.c_UVtxuH2EW},
                      type = 'UVtree')


V_UVuxcHEW = CTVertex(name = 'V_UVuxcHEW',
                      particles = [ P.u__tilde__, P.c, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVuxcH1EW, (0, 1, 0) : C.c_UVuxcH2EW},
                      type = 'UVtree')


V_UVcxcHEW = CTVertex(name = 'V_UVcxcHEW',
                      particles = [ P.c__tilde__, P.c, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVcxcH1EW, (0, 1, 0) : C.c_UVcxcH2EW},
                      type = 'UVtree')


V_UVtxcHEW = CTVertex(name = 'V_UVtxcHEW',
                      particles = [ P.t__tilde__, P.c, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVtxcH1EW, (0, 1, 0) : C.c_UVtxcH2EW},
                      type = 'UVtree')


V_UVuxtHEW = CTVertex(name = 'V_UVuxtHEW',
                      particles = [ P.u__tilde__, P.t, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVuxtH1EW, (0, 1, 0) : C.c_UVuxtH2EW},
                      type = 'UVtree')


V_UVcxtHEW = CTVertex(name = 'V_UVcxtHEW',
                      particles = [ P.c__tilde__, P.t, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVcxtH1EW, (0, 1, 0) : C.c_UVcxtH2EW},
                      type = 'UVtree')


V_UVtxtHEW = CTVertex(name = 'V_UVtxtHEW',
                      particles = [ P.t__tilde__, P.t, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVtxtH1EW, (0, 1, 0) : C.c_UVtxtH2EW},
                      type = 'UVtree')


V_UVdxdHEW = CTVertex(name = 'V_UVdxdHEW',
                      particles = [ P.d__tilde__, P.d, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVdxdH1EW, (0, 1, 0) : C.c_UVdxdH2EW},
                      type = 'UVtree')


V_UVsxdHEW = CTVertex(name = 'V_UVsxdHEW',
                      particles = [ P.s__tilde__, P.d, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVsxdH1EW, (0, 1, 0) : C.c_UVsxdH2EW},
                      type = 'UVtree')


V_UVbxdHEW = CTVertex(name = 'V_UVbxdHEW',
                      particles = [ P.b__tilde__, P.d, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVbxdH1EW, (0, 1, 0) : C.c_UVbxdH2EW},
                      type = 'UVtree')


V_UVdxsHEW = CTVertex(name = 'V_UVdxsHEW',
                      particles = [ P.d__tilde__, P.s, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVdxsH1EW, (0, 1, 0) : C.c_UVdxsH2EW},
                      type = 'UVtree')


V_UVsxsHEW = CTVertex(name = 'V_UVsxsHEW',
                      particles = [ P.s__tilde__, P.s, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVsxsH1EW, (0, 1, 0) : C.c_UVsxsH2EW},
                      type = 'UVtree')


V_UVbxsHEW = CTVertex(name = 'V_UVbxsHEW',
                      particles = [ P.b__tilde__, P.s, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVbxsH1EW, (0, 1, 0) : C.c_UVbxsH2EW},
                      type = 'UVtree')


V_UVdxbHEW = CTVertex(name = 'V_UVdxbHEW',
                      particles = [ P.d__tilde__, P.b, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVdxbH1EW, (0, 1, 0) : C.c_UVdxbH2EW},
                      type = 'UVtree')


V_UVsxbHEW = CTVertex(name = 'V_UVsxbHEW',
                      particles = [ P.s__tilde__, P.b, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVsxbH1EW, (0, 1, 0) : C.c_UVsxbH2EW},
                      type = 'UVtree')


V_UVbxbHEW = CTVertex(name = 'V_UVbxbHEW',
                      particles = [ P.b__tilde__, P.b, P.H ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.l_epemH23, L.l_epemH24 ],
                      loop_particles = [  ],
                      couplings = {(0, 0, 0) : C.c_UVbxbH1EW, (0, 1, 0) : C.c_UVbxbH2EW},
                      type = 'UVtree')


V_UVepemG0EW = CTVertex(name = 'V_UVepemG0EW',
                        particles = [ P.e__plus__, P.e__minus__, P.G0 ],
                        color = [ '1' ],
                        lorentz = [ L.l_epemH23, L.l_epemH24 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVepemG01EW, (0, 1, 0) : C.c_UVepemG02EW},
                        type = 'UVtree')


V_UVmpmmG0EW = CTVertex(name = 'V_UVmpmmG0EW',
                        particles = [ P.m__plus__, P.m__minus__, P.G0 ],
                        color = [ '1' ],
                        lorentz = [ L.l_epemH23, L.l_epemH24 ],
                        loop_particles = [  ],
                        couplings = {(0, 0, 0) : C.c_UVmpmmG01EW, (0, 1, 0) : C.c_UVmpmmG02EW},
                        type = 'UVtree')


V_UVttpttmG0EW = CTVertex(name = 'V_UVttpttmG0EW',
                          particles = [ P.tt__plus__, P.tt__minus__, P.G0 ],
                          color = [ '1' ],
                          lorentz = [ L.l_epemH23, L.l_epemH24 ],
                          loop_particles = [  ],
                          couplings = {(0, 0, 0) : C.c_UVttpttmG01EW, (0, 1, 0) : C.c_UVttpttmG02EW},
                          type = 'UVtree')


V_UVuxuG0EW = CTVertex(name = 'V_UVuxuG0EW',
                       particles = [ P.u__tilde__, P.u, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVuxuG01EW, (0, 1, 0) : C.c_UVuxuG02EW},
                       type = 'UVtree')


V_UVcxuG0EW = CTVertex(name = 'V_UVcxuG0EW',
                       particles = [ P.c__tilde__, P.u, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVcxuG01EW, (0, 1, 0) : C.c_UVcxuG02EW},
                       type = 'UVtree')


V_UVtxuG0EW = CTVertex(name = 'V_UVtxuG0EW',
                       particles = [ P.t__tilde__, P.u, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVtxuG01EW, (0, 1, 0) : C.c_UVtxuG02EW},
                       type = 'UVtree')


V_UVuxcG0EW = CTVertex(name = 'V_UVuxcG0EW',
                       particles = [ P.u__tilde__, P.c, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVuxcG01EW, (0, 1, 0) : C.c_UVuxcG02EW},
                       type = 'UVtree')


V_UVcxcG0EW = CTVertex(name = 'V_UVcxcG0EW',
                       particles = [ P.c__tilde__, P.c, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVcxcG01EW, (0, 1, 0) : C.c_UVcxcG02EW},
                       type = 'UVtree')


V_UVtxcG0EW = CTVertex(name = 'V_UVtxcG0EW',
                       particles = [ P.t__tilde__, P.c, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVtxcG01EW, (0, 1, 0) : C.c_UVtxcG02EW},
                       type = 'UVtree')


V_UVuxtG0EW = CTVertex(name = 'V_UVuxtG0EW',
                       particles = [ P.u__tilde__, P.t, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVuxtG01EW, (0, 1, 0) : C.c_UVuxtG02EW},
                       type = 'UVtree')


V_UVcxtG0EW = CTVertex(name = 'V_UVcxtG0EW',
                       particles = [ P.c__tilde__, P.t, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVcxtG01EW, (0, 1, 0) : C.c_UVcxtG02EW},
                       type = 'UVtree')


V_UVtxtG0EW = CTVertex(name = 'V_UVtxtG0EW',
                       particles = [ P.t__tilde__, P.t, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVtxtG01EW, (0, 1, 0) : C.c_UVtxtG02EW},
                       type = 'UVtree')


V_UVdxdG0EW = CTVertex(name = 'V_UVdxdG0EW',
                       particles = [ P.d__tilde__, P.d, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVdxdG01EW, (0, 1, 0) : C.c_UVdxdG02EW},
                       type = 'UVtree')


V_UVsxdG0EW = CTVertex(name = 'V_UVsxdG0EW',
                       particles = [ P.s__tilde__, P.d, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVsxdG01EW, (0, 1, 0) : C.c_UVsxdG02EW},
                       type = 'UVtree')


V_UVbxdG0EW = CTVertex(name = 'V_UVbxdG0EW',
                       particles = [ P.b__tilde__, P.d, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVbxdG01EW, (0, 1, 0) : C.c_UVbxdG02EW},
                       type = 'UVtree')


V_UVdxsG0EW = CTVertex(name = 'V_UVdxsG0EW',
                       particles = [ P.d__tilde__, P.s, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVdxsG01EW, (0, 1, 0) : C.c_UVdxsG02EW},
                       type = 'UVtree')


V_UVsxsG0EW = CTVertex(name = 'V_UVsxsG0EW',
                       particles = [ P.s__tilde__, P.s, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVsxsG01EW, (0, 1, 0) : C.c_UVsxsG02EW},
                       type = 'UVtree')


V_UVbxsG0EW = CTVertex(name = 'V_UVbxsG0EW',
                       particles = [ P.b__tilde__, P.s, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVbxsG01EW, (0, 1, 0) : C.c_UVbxsG02EW},
                       type = 'UVtree')


V_UVdxbG0EW = CTVertex(name = 'V_UVdxbG0EW',
                       particles = [ P.d__tilde__, P.b, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVdxbG01EW, (0, 1, 0) : C.c_UVdxbG02EW},
                       type = 'UVtree')


V_UVsxbG0EW = CTVertex(name = 'V_UVsxbG0EW',
                       particles = [ P.s__tilde__, P.b, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVsxbG01EW, (0, 1, 0) : C.c_UVsxbG02EW},
                       type = 'UVtree')


V_UVbxbG0EW = CTVertex(name = 'V_UVbxbG0EW',
                       particles = [ P.b__tilde__, P.b, P.G0 ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVbxbG01EW, (0, 1, 0) : C.c_UVbxbG02EW},
                       type = 'UVtree')


V_UVdxuGmEW = CTVertex(name = 'V_UVdxuGmEW',
                       particles = [ P.d__tilde__, P.u, P.G__minus__ ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVdxuGm1EW, (0, 1, 0) : C.c_UVdxuGm2EW},
                       type = 'UVtree')


V_UVsxuGmEW = CTVertex(name = 'V_UVsxuGmEW',
                       particles = [ P.s__tilde__, P.u, P.G__minus__ ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVsxuGm1EW, (0, 1, 0) : C.c_UVsxuGm2EW},
                       type = 'UVtree')


V_UVbxuGmEW = CTVertex(name = 'V_UVbxuGmEW',
                       particles = [ P.b__tilde__, P.u, P.G__minus__ ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVbxuGm1EW, (0, 1, 0) : C.c_UVbxuGm2EW},
                       type = 'UVtree')


V_UVdxcGmEW = CTVertex(name = 'V_UVdxcGmEW',
                       particles = [ P.d__tilde__, P.c, P.G__minus__ ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVdxcGm1EW, (0, 1, 0) : C.c_UVdxcGm2EW},
                       type = 'UVtree')


V_UVsxcGmEW = CTVertex(name = 'V_UVsxcGmEW',
                       particles = [ P.s__tilde__, P.c, P.G__minus__ ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVsxcGm1EW, (0, 1, 0) : C.c_UVsxcGm2EW},
                       type = 'UVtree')


V_UVbxcGmEW = CTVertex(name = 'V_UVbxcGmEW',
                       particles = [ P.b__tilde__, P.c, P.G__minus__ ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVbxcGm1EW, (0, 1, 0) : C.c_UVbxcGm2EW},
                       type = 'UVtree')


V_UVdxtGmEW = CTVertex(name = 'V_UVdxtGmEW',
                       particles = [ P.d__tilde__, P.t, P.G__minus__ ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVdxtGm1EW, (0, 1, 0) : C.c_UVdxtGm2EW},
                       type = 'UVtree')


V_UVsxtGmEW = CTVertex(name = 'V_UVsxtGmEW',
                       particles = [ P.s__tilde__, P.t, P.G__minus__ ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVsxtGm1EW, (0, 1, 0) : C.c_UVsxtGm2EW},
                       type = 'UVtree')


V_UVbxtGmEW = CTVertex(name = 'V_UVbxtGmEW',
                       particles = [ P.b__tilde__, P.t, P.G__minus__ ],
                       color = [ 'Identity(1,2)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVbxtGm1EW, (0, 1, 0) : C.c_UVbxtGm2EW},
                       type = 'UVtree')


V_UVuxdGpEW = CTVertex(name = 'V_UVuxdGpEW',
                       particles = [ P.u__tilde__, P.d, P.G__plus__ ],
                       color = [ 'Identity(2,1)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVuxdGp1EW, (0, 1, 0) : C.c_UVuxdGp2EW},
                       type = 'UVtree')


V_UVcxdGpEW = CTVertex(name = 'V_UVcxdGpEW',
                       particles = [ P.c__tilde__, P.d, P.G__plus__ ],
                       color = [ 'Identity(2,1)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVcxdGp1EW, (0, 1, 0) : C.c_UVcxdGp2EW},
                       type = 'UVtree')


V_UVtxdGpEW = CTVertex(name = 'V_UVtxdGpEW',
                       particles = [ P.t__tilde__, P.d, P.G__plus__ ],
                       color = [ 'Identity(2,1)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVtxdGp1EW, (0, 1, 0) : C.c_UVtxdGp2EW},
                       type = 'UVtree')


V_UVuxsGpEW = CTVertex(name = 'V_UVuxsGpEW',
                       particles = [ P.u__tilde__, P.s, P.G__plus__ ],
                       color = [ 'Identity(2,1)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVuxsGp1EW, (0, 1, 0) : C.c_UVuxsGp2EW},
                       type = 'UVtree')


V_UVcxsGpEW = CTVertex(name = 'V_UVcxsGpEW',
                       particles = [ P.c__tilde__, P.s, P.G__plus__ ],
                       color = [ 'Identity(2,1)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVcxsGp1EW, (0, 1, 0) : C.c_UVcxsGp2EW},
                       type = 'UVtree')


V_UVtxsGpEW = CTVertex(name = 'V_UVtxsGpEW',
                       particles = [ P.t__tilde__, P.s, P.G__plus__ ],
                       color = [ 'Identity(2,1)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVtxsGp1EW, (0, 1, 0) : C.c_UVtxsGp2EW},
                       type = 'UVtree')


V_UVuxbGpEW = CTVertex(name = 'V_UVuxbGpEW',
                       particles = [ P.u__tilde__, P.b, P.G__plus__ ],
                       color = [ 'Identity(2,1)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVuxbGp1EW, (0, 1, 0) : C.c_UVuxbGp2EW},
                       type = 'UVtree')


V_UVcxbGpEW = CTVertex(name = 'V_UVcxbGpEW',
                       particles = [ P.c__tilde__, P.b, P.G__plus__ ],
                       color = [ 'Identity(2,1)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVcxbGp1EW, (0, 1, 0) : C.c_UVcxbGp2EW},
                       type = 'UVtree')


V_UVtxbGpEW = CTVertex(name = 'V_UVtxbGpEW',
                       particles = [ P.t__tilde__, P.b, P.G__plus__ ],
                       color = [ 'Identity(2,1)' ],
                       lorentz = [ L.l_epemH23, L.l_epemH24 ],
                       loop_particles = [  ],
                       couplings = {(0, 0, 0) : C.c_UVtxbGp1EW, (0, 1, 0) : C.c_UVtxbGp2EW},
                       type = 'UVtree')


V_UVepveGmEW = CTVertex(name = 'V_UVepveGmEW',
                        particles = [ P.e__plus__, P.ve, P.G__minus__ ],
                        color = [ '1' ],
                        lorentz = [ L.l_epemH23, L.l_epemH24 ],
                        loop_particles = [  ],
                        couplings = {(0, 1, 0) : C.c_UVepveGm1EW},
                        type = 'UVtree')


V_UVmpvmGmEW = CTVertex(name = 'V_UVmpvmGmEW',
                        particles = [ P.m__plus__, P.vm, P.G__minus__ ],
                        color = [ '1' ],
                        lorentz = [ L.l_epemH23, L.l_epemH24 ],
                        loop_particles = [  ],
                        couplings = {(0, 1, 0) : C.c_UVmpvmGm1EW},
                        type = 'UVtree')


V_UVttpvtGmEW = CTVertex(name = 'V_UVttpvtGmEW',
                         particles = [ P.tt__plus__, P.vt, P.G__minus__ ],
                         color = [ '1' ],
                         lorentz = [ L.l_epemH23, L.l_epemH24 ],
                         loop_particles = [  ],
                         couplings = {(0, 1, 0) : C.c_UVttpvtGm1EW},
                         type = 'UVtree')


V_UVvexemGpEW = CTVertex(name = 'V_UVvexemGpEW',
                         particles = [ P.ve__tilde__, P.e__minus__, P.G__plus__ ],
                         color = [ '1' ],
                         lorentz = [ L.l_epemH23, L.l_epemH24 ],
                         loop_particles = [  ],
                         couplings = {(0, 0, 0) : C.c_UVvexemGp1EW},
                         type = 'UVtree')


V_UVvmxmmGpEW = CTVertex(name = 'V_UVvmxmmGpEW',
                         particles = [ P.vm__tilde__, P.m__minus__, P.G__plus__ ],
                         color = [ '1' ],
                         lorentz = [ L.l_epemH23, L.l_epemH24 ],
                         loop_particles = [  ],
                         couplings = {(0, 0, 0) : C.c_UVvmxmmGp1EW},
                         type = 'UVtree')


V_UVvtxttmGpEW = CTVertex(name = 'V_UVvtxttmGpEW',
                          particles = [ P.vt__tilde__, P.tt__minus__, P.G__plus__ ],
                          color = [ '1' ],
                          lorentz = [ L.l_epemH23, L.l_epemH24 ],
                          loop_particles = [  ],
                          couplings = {(0, 0, 0) : C.c_UVvtxttmGp1EW},
                          type = 'UVtree')


# ============== #
# Mixed QCD-QED  #
# ============== #

V_UVHtt = CTVertex(name = 'V_UVHtt',
              particles = [ P.t__tilde__, P.t, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS5 ],
              loop_particles = [[[P.G,P.t]]],                   
              couplings = {(0,0,0):C.UV_Htt},
              type = 'UVtree')

V_UVHbb = CTVertex(name = 'V_UVHbb',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS5 ],
              loop_particles = [[[P.G,P.b]]],
              couplings = {(0,0,0):C.UV_Hbb},
              type = 'UVtree')

# ============================== #
# Goldstone UV counter vertices  #
# ============================== #

V_UV_txbGp = CTVertex(name = 'V_UV_txbGp',
                      particles = [ P.t__tilde__, P.b, P.G__plus__ ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.FFS1, L.FFS3 ],
                      loop_particles = [[[P.G,P.b,P.t]]],
                      couplings = {(0,0,0):C.GC_UV_1015,(0,1,0):C.GC_UV_1021},
                      type = 'UVtree')

V_UV_bxtGm = CTVertex(name = 'V_UV_bxtGm',
                      particles = [ P.b__tilde__, P.t, P.G__minus__ ],
                      color = [ 'Identity(1,2)' ],
                      lorentz = [ L.FFS1, L.FFS3 ],
                      loop_particles = [[[P.G,P.b,P.t]]],
                      couplings = {(0,0,0):C.GC_UV_1027,(0,1,0):C.GC_UV_1030},
                      type = 'UVtree')

V_UV_bxbG0 = CTVertex(name = 'V_UV_bxbG0',
                     particles = [ P.b__tilde__, P.b, P.G0 ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFS2 ],
                     loop_particles = [[[P.G,P.b]]],
                     couplings = {(0,0,0):C.GC_UV_1082},
                     type = 'UVtree')

V_UV_cxcG0 = CTVertex(name = 'V_UV_cxcG0',
                     particles = [ P.c__tilde__, P.c, P.G0 ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFS2 ],
                     loop_particles = [[[P.G,P.c]]],
                     couplings = {(0,0,0):C.GC_UV_1085},
                     type = 'UVtree')

V_UV_txtG0 = CTVertex(name = 'V_UV_txtG0',
                     particles = [ P.t__tilde__, P.t, P.G0 ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFS2 ],
                     loop_particles = [[[P.G,P.t]]],
                     couplings = {(0,0,0):C.GC_UV_1095},
                     type = 'UVtree')

V_UV_cxsGp = CTVertex(name = 'V_UV_cxsGp',
                     particles = [ P.c__tilde__, P.s, P.G__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFS3 ],
                     loop_particles = [[[P.G,P.c,P.s]]],
                     couplings = {(0,0,0):C.GC_UV_1018},
                     type = 'UVtree')

V_UV_txsGp = CTVertex(name = 'V_UV_txsGp',
                     particles = [ P.t__tilde__, P.s, P.G__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFS3 ],
                     loop_particles = [[[P.G,P.t,P.s]]],
                     couplings = {(0,0,0):C.GC_UV_1019},
                     type = 'UVtree')

V_UV_uxbGp = CTVertex(name = 'V_UV_uxbGp',
                     particles = [ P.u__tilde__, P.b, P.G__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFS1 ],
                     loop_particles = [[[P.G,P.u,P.b]]],
                     couplings = {(0,0,0):C.GC_UV_1013},
                     type = 'UVtree')

V_UV_cxbGp = CTVertex(name = 'V_UV_cxbGp',
                     particles = [ P.c__tilde__, P.b, P.G__plus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFS1, L.FFS3 ],
                     loop_particles = [[[P.G,P.c,P.b]]],
                     couplings = {(0,0,0):C.GC_UV_1014,(0,1,0):C.GC_UV_1020},
                     type = 'UVtree')

V_UV_bxuGm = CTVertex(name = 'V_UV_bxuGm',
                     particles = [ P.b__tilde__, P.u, P.G__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFS1 ],
                     loop_particles = [[[P.G,P.u,P.b]]],
                     couplings = {(0,0,0):C.GC_UV_1028},
                     type = 'UVtree')

V_UV_dxcGm = CTVertex(name = 'V_UV_dxcGm',
                     particles = [ P.d__tilde__, P.c, P.G__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFS1 ],
                     loop_particles = [[[P.G,P.c,P.d]]],
                     couplings = {(0,0,0):C.GC_UV_1022},
                     type = 'UVtree')

V_UV_sxcGm = CTVertex(name = 'V_UV_sxcGm',
                     particles = [ P.s__tilde__, P.c, P.G__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFS1 ],
                     loop_particles = [[[P.G,P.c,P.s]]],
                     couplings = {(0,0,0):C.GC_UV_1023},
                     type = 'UVtree')

V_UV_bxcGm = CTVertex(name = 'V_UV_bxcGm',
                     particles = [ P.b__tilde__, P.c, P.G__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFS1, L.FFS3 ],
                     loop_particles = [[[P.G,P.c,P.b]]],
                     couplings = {(0,0,0):C.GC_UV_1024,(0,1,0):C.GC_UV_1029},
                     type = 'UVtree')

V_UV_dxtGm = CTVertex(name = 'V_UV_dxtGm',
                     particles = [ P.d__tilde__, P.t, P.G__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFS1 ],
                     loop_particles = [[[P.G,P.t,P.d]]],
                     couplings = {(0,0,0):C.GC_UV_1025},
                     type = 'UVtree')

V_UV_sxtGm = CTVertex(name = 'V_UV_sxtGm',
                     particles = [ P.s__tilde__, P.t, P.G__minus__ ],
                     color = [ 'Identity(1,2)' ],
                     lorentz = [ L.FFS1 ],
                     loop_particles = [[[P.G,P.s,P.t]]],
                     couplings = {(0,0,0):C.GC_UV_1026},
                     type = 'UVtree')

# treat the diagonal of CKM
# guu~
V_UVGUUEW = CTVertex(name = 'V_UVGUUEW',
                   particles = [ P.u__tilde__, P.u, P.G ],
                   color = [ 'T(3,2,1)' ],
                   lorentz = [ L.FFV2, L.FFV6 ],
                   loop_particles = [[[P.u,P.Z]]],
                   couplings = {(0,0,0):C.UV_GUULEW,(0,1,0):C.UV_GUUREW},
                   type = 'UVtree')

# gdd~
V_UVGDDEW = CTVertex(name = 'V_UVGDDEW',
                     particles = [ P.d__tilde__, P.d, P.G ],
                     color = [ 'T(3,2,1)' ],
                     lorentz = [ L.FFV2, L.FFV6 ],
                     loop_particles = [[[P.d,P.Z]]],
                     couplings = {(0,0,0):C.UV_GDDLEW,(0,1,0):C.UV_GDDREW},
                     type = 'UVtree')

# gss~
V_UVGSSEW = CTVertex(name = 'V_UVGSSEW',
                     particles = [ P.s__tilde__, P.s, P.G ],
                     color = [ 'T(3,2,1)' ],
                     lorentz = [ L.FFV2, L.FFV6 ],
                     loop_particles = [[[P.s,P.Z]]],
                     couplings = {(0,0,0):C.UV_GSSLEW,(0,1,0):C.UV_GSSREW},
                     type = 'UVtree')

# gcc~
V_UVGCCEW = CTVertex(name = 'V_UVGCCEW',
                     particles = [ P.c__tilde__, P.c, P.G ],
                     color = [ 'T(3,2,1)' ],
                     lorentz = [ L.FFV2, L.FFV6 ],
                     loop_particles = [[[P.c,P.Z]]],
                     couplings = {(0,0,0):C.UV_GCCLEW,(0,1,0):C.UV_GCCREW},
                     type = 'UVtree')

# gbb~
V_UVGBBEW = CTVertex(name = 'V_UVGBBEW',
                     particles = [ P.b__tilde__, P.b, P.G ],
                     color = [ 'T(3,2,1)' ],
                     lorentz = [ L.FFV2, L.FFV6 ],
                     loop_particles = [[[P.b,P.Z]]],
                     couplings = {(0,0,0):C.UV_GBBLEW,(0,1,0):C.UV_GBBREW},
                     type = 'UVtree')

# gtt~
V_UVGTTEW = CTVertex(name = 'V_UVGTTEW',
                     particles = [ P.t__tilde__, P.t, P.G ],
                     color = [ 'T(3,2,1)' ],
                     lorentz = [ L.FFV2, L.FFV6 ],
                     loop_particles = [[[P.t,P.Z]]],
                     couplings = {(0,0,0):C.UV_GTTLEW,(0,1,0):C.UV_GTTREW},
                     type = 'UVtree')



