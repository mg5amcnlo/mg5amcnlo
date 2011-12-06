# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51


from object_library import all_vertices, all_CTvertices, Vertex, CTVertex
import particles as P
import couplings as C
import lorentz as L

V_1 = Vertex(name = 'V_1',
              particles = [ P.G, P.G, P.G ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVV1 ],
              couplings = {(0,0):C.GC_9},
              type = ['base',()])
              
V_2 = Vertex(name = 'V_2',
              particles = [ P.G, P.G, P.G, P.G ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVV1, L.VVVV3, L.VVVV4 ],
              couplings = {(1,1):C.GC_11,(0,0):C.GC_11,(2,2):C.GC_11},
              type = ['base',()])

V_3 = Vertex(name = 'V_3',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_10},
              type = ['base',()])

V_4 = Vertex(name = 'V_4',
               particles = [ P.u__tilde__, P.u, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_10},
               type = ['base',()])

V_5 = Vertex(name = 'V_5',
              particles = [ P.s__tilde__, P.s, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_10},
              type = ['base',()])

V_6 = Vertex(name = 'V_6',
               particles = [ P.c__tilde__, P.c, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_10},
               type = ['base',()])

V_7 = Vertex(name = 'V_7',
              particles = [ P.b__tilde__, P.b, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_10},
              type = ['base',()])

V_8 = Vertex(name = 'V_8',
               particles = [ P.t__tilde__, P.t, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_10},
               type = ['base',()])

# QCD ghost
V_9 = Vertex(name = 'V_9',
               particles = [ P.gh__tilde__, P.gh, P.G ],
               color = [ 'f(1,3,2)' ],
               lorentz = [ L.GHGHG ],
               couplings = {(0,0):C.GC_9},
               type = ['base',()])
