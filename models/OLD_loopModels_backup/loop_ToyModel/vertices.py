# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L

# It was number 36 in sm
V_1 = Vertex(name = 'V_1',
              particles = [ P.G, P.G, P.G ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVV1 ],
              couplings = {(0,0):C.GC_9},
              type = ['base',()])
              
# It was number 37 in sm
V_2 = Vertex(name = 'V_2',
              particles = [ P.G, P.G, P.G, P.G ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVV1, L.VVVV3, L.VVVV4 ],
              couplings = {(1,1):C.GC_11,(0,0):C.GC_11,(2,2):C.GC_11},
              type = ['base',()])

# It was number 80 in sm
V_3 = Vertex(name = 'V_3',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_10},
              type = ['base',()])
              
# It was number 128 in sm
V_4 = Vertex(name = 'V_4',
               particles = [ P.u__tilde__, P.u, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_10},
               type = ['base',()])