# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51


from object_library import all_vertices, all_CTvertices, Vertex, CTVertex
import particles as P
import CT_couplings as C
import lorentz as L

#####################
# DUMMY R2 vertices #
#####################

# ggg R2
V_R23G = CTVertex(name = 'V_R23G',
              particles = [ P.g, P.g, P.g ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVV1 ],
              loop_particles = [ [[P.u], [P.d], [P.c], [P.s], [P.b], [P.t]],
                               [[P.g]] ],
              couplings = {(0,0,0):C.R2_3Gq, (0,0,1):C.R2_3Gg},
              type = 'R2' )

######################
# DUMMY UV vertices  #
######################

# t~t         
V_UVtMass = CTVertex(name = 'V_UVtMass',
               particles = [ P.t__tilde__, P.t ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_2 ],
               loop_particles = [[[P.g,P.t]]],                   
               couplings = {(0,0,0):C.UV_tMass},
               type = 'UVmass')

# gtt~                                                                          
V_UVGTT = CTVertex(name = 'V_UVGTT',
              particles = [ P.t__tilde__, P.t, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              loop_particles = [[[P.u],[P.d],[P.s]],[[P.c]],[[P.b]],[[P.t]],[[P.g]]],
              couplings = {(0,0,0):C.UV_GQQq,(0,0,1):C.UV_GQQc,(0,0,2):C.UV_GQQb,(0,0,3):C.UV_GQQt,(0,0,4):C.UV_GQQg},
              type = 'UV')
