# This file was automatically created by FeynRules 1.7.55
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (October 6, 2011)
# Date: Wed 8 Aug 2012 14:16:24


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L

V_12 = Vertex(name = 'V_12',
              particles = [ P.A, P.A, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS3 ],
              couplings = {(0,0):C.GC_1})

V_13 = Vertex(name = 'V_13',
              particles = [ P.G, P.G, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVS3 ],
              couplings = {(0,0):C.GC_13})

V_37 = Vertex(name = 'V_37',
              particles = [ P.G, P.G, P.h1 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_16})

V_41 = Vertex(name = 'V_41',
              particles = [ P.G, P.G, P.G, P.H ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVVS2 ],
              couplings = {(0,0):C.GC_14})

V_42 = Vertex(name = 'V_42',
              particles = [ P.G, P.G, P.G, P.G, P.H ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVVS1, L.VVVVS2, L.VVVVS3 ],
              couplings = {(1,1):C.GC_15,(0,0):C.GC_15,(2,2):C.GC_15})

V_43 = Vertex(name = 'V_43',
              particles = [ P.G, P.G, P.G, P.h1 ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVVS1 ],
              couplings = {(0,0):C.GC_17})

