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

# ggg R2
V_R23G = CTVertex(name = 'V_R23G',
              particles = [ P.G, P.G, P.G ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVV1 ],
              loop_particles = [ [[P.u], [P.d], [P.c], [P.s]],
                               [[P.G]] ],
              couplings = {(0,0,0):C.R2_3Gq, (0,0,1):C.R2_3Gg},
              type = 'R2' )

#=============================================================================================
#  4-gluon R2 vertex
#=============================================================================================

# Keep in mind that Delta8(a,b) is 1/2 Tr(a,b)

V_R24G = Vertex(name = 'V_R24G',
              particles = [ P.G, P.G, P.G, P.G ],
              color = [ 'Tr(1,2)*Tr(3,4)' , 'Tr(1,3)*Tr(2,4)' , 'Tr(1,4)*Tr(2,3)', \
                        'd(-1,1,2)*d(-1,3,4)' , 'd(-1,1,3)*d(-1,2,4)' , 'd(-1,1,4)*d(-1,2,3)'],
              lorentz = [  L.R2_4G_1234, L.R2_4G_1324, L.R2_4G_1423 ],
              loop_particles = [ [[P.G]], [[P.u],[P.d],[P.c],[P.s]] ],
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
V_R2GDD = Vertex(name = 'V_R2GDD',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles =[[[P.d,P.G]]],                 
              couplings = {(0,0,0):C.R2_GQQ},
              type = 'R2')

# guu~              
V_R2GUU = Vertex(name = 'V_R2GUU',
               particles = [ P.u__tilde__, P.u, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               loop_particles =[[[P.u,P.G]]],
               couplings = {(0,0,0):C.R2_GQQ},
               type = 'R2')  

# gss~
V_R2GDD = Vertex(name = 'V_R2GSS',
              particles = [ P.s__tilde__, P.s, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles =[[[P.s,P.G]]],
              couplings = {(0,0,0):C.R2_GQQ},
              type = 'R2')

# gcc~              
V_R2GUU = Vertex(name = 'V_R2GCC',
               particles = [ P.c__tilde__, P.c, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               loop_particles =[[[P.c,P.G]]],               
               couplings = {(0,0,0):C.R2_GQQ},
               type = 'R2')  

# gg             
V_R2GG = Vertex(name = 'V_R2GG',
               particles = [ P.G, P.G ],
               color = [ 'Identity(1,2)' ],
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
V_R2DD = Vertex(name = 'V_R2DD',
               particles = [ P.d__tilde__, P.d ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_1 ],
               loop_particles = [[[P.d,P.G]]],
               couplings = {(0,0,0):C.R2_QQq},
               type = 'R2') 

# u~u            
V_R2UU = Vertex(name = 'V_R2UU',
               particles = [ P.u__tilde__, P.u ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_1 ],
               loop_particles = [[[P.u,P.G]]],            
               couplings = {(0,0,0):C.R2_QQq},
               type = 'R2')

# s~s            
V_R2SS = Vertex(name = 'V_R2SS',
               particles = [ P.s__tilde__, P.s ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_1 ],
               loop_particles = [[[P.s,P.G]]],                
               couplings = {(0,0,0):C.R2_QQq},
               type = 'R2')

# c~c            
V_R2CC = Vertex(name = 'V_R2CC',
               particles = [ P.c__tilde__, P.c ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_1 ],
               loop_particles = [[[P.c,P.G]]],
               couplings = {(0,0,0):C.R2_QQq},                
               type = 'R2') 

# b~b            
V_R2BB = Vertex(name = 'V_R2BB',
               particles = [ P.b__tilde__, P.b ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_1, L.R2_QQ_2 ],
               loop_particles = [[[P.b,P.G]]],
               couplings = {(0,0,0):C.R2_QQq,(0,1,0):C.R2_QQb},                
               type = 'R2')

# t~t            
V_R2TT = Vertex(name = 'V_R2TT',
               particles = [ P.t__tilde__, P.t ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_1, L.R2_QQ_2 ],
               loop_particles = [[[P.t,P.G]]],
               couplings = {(0,0,0):C.R2_QQq,(0,1,0):C.R2_QQt},
               type = 'R2')

################
# UV vertices  #
################

# There are the alpha_s renormalization vertices

# ggg
V_UV1eps3G = Vertex(name = 'V_UV1eps3G',
              particles = [ P.G, P.G, P.G ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_3Gq,(0,0,1):C.UV_3Gb,(0,0,2):C.UV_3Gt,(0,0,3):C.UV_3Gg},
              type = 'UV')

# gggg
V_UV4G = Vertex(name = 'V_UV1eps4G',
              particles = [ P.G, P.G, P.G, P.G ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVV1, L.VVVV3, L.VVVV4 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_4Gq,(0,0,1):C.UV_4Gb,(0,0,2):C.UV_4Gt,(0,0,3):C.UV_4Gg,
                           (1,1,0):C.UV_4Gq,(1,1,1):C.UV_4Gb,(1,1,2):C.UV_4Gt,(1,1,3):C.UV_4Gg,
                           (2,2,0):C.UV_4Gq,(2,2,1):C.UV_4Gb,(2,2,2):C.UV_4Gt,(2,2,3):C.UV_4Gg},
              type = 'UV')

# gdd~
V_UVGDD = Vertex(name = 'V_UVGDD',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UV')

# guu~
V_UVGUU = Vertex(name = 'V_UVGUU',
              particles = [ P.u__tilde__, P.u, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UV')

# gcc~
V_UVGCC = Vertex(name = 'V_UVGCC',
              particles = [ P.c__tilde__, P.c, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UV')

# gss~
V_UVGSS = Vertex(name = 'V_UVGSS',
              particles = [ P.s__tilde__, P.s, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UV')

# gbb~
V_UVGBB = Vertex(name = 'V_UVGBB',
              particles = [ P.b__tilde__, P.b, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UV')

# gtt~
V_UVGTT = Vertex(name = 'V_UVGTT',
              particles = [ P.t__tilde__, P.t, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.c],[P.s]],[[P.b]],[[P.t]],[[P.G]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQb,(0,0,2):C.UV_GQQt,(0,0,3):C.UV_GQQg},
              type = 'UV')

# These are the mass renormalization vertices.

# b~b         
V_UVbMass = Vertex(name = 'V_UVbMass',
               particles = [ P.b__tilde__, P.b ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2 ],
               loop_particles = [[[P.G,P.b]]],                   
               couplings = {(0,0,0):C.UV_bMass},
               type = 'UVmass') 

# t~t         
V_UVbMass = Vertex(name = 'V_UVtMass',
               particles = [ P.t__tilde__, P.t ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2 ],
               loop_particles = [[[P.G,P.t]]],                   
               couplings = {(0,0,0):C.UV_tMass},
               type = 'UVmass')
