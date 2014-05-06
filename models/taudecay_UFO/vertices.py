# This file was automatically created by FeynRules 2.0.6
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (February 23, 2011)
# Date: Wed 18 Dec 2013 14:14:59


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.vt__tilde__, P.ta__minus__, P.pi__plus__ ],
             color = [ '1' ],
             lorentz = [ L.FFS1 ],
             couplings = {(0,0):C.GC_1})

V_2 = Vertex(name = 'V_2',
             particles = [ P.ta__plus__, P.vt, P.pi__minus__ ],
             color = [ '1' ],
             lorentz = [ L.FFS1 ],
             couplings = {(0,0):C.GC_1})

