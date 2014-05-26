# This file was automatically created by FeynRules 2.0.25
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (February 23, 2011)
# Date: Thu 8 May 2014 12:30:33


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.vt__tilde__, P.ta__minus__, P.pi0, P.pi__plus__ ],
             color = [ '1' ],
             lorentz = [ L.FFSS1 ],
             couplings = {(0,0):C.GC_3})

V_2 = Vertex(name = 'V_2',
             particles = [ P.vt__tilde__, P.ta__minus__, P.pi__plus__ ],
             color = [ '1' ],
             lorentz = [ L.FFS1 ],
             couplings = {(0,0):C.GC_2})

V_3 = Vertex(name = 'V_3',
             particles = [ P.vt__tilde__, P.ta__minus__, P.e__plus__, P.ve ],
             color = [ '1' ],
             lorentz = [ L.FFFF1 ],
             couplings = {(0,0):C.GC_1})

V_4 = Vertex(name = 'V_4',
             particles = [ P.vt__tilde__, P.ta__minus__, P.mu__plus__, P.vm ],
             color = [ '1' ],
             lorentz = [ L.FFFF1 ],
             couplings = {(0,0):C.GC_1})

V_5 = Vertex(name = 'V_5',
             particles = [ P.ta__plus__, P.vt, P.pi0, P.pi__minus__ ],
             color = [ '1' ],
             lorentz = [ L.FFSS1 ],
             couplings = {(0,0):C.GC_3})

V_6 = Vertex(name = 'V_6',
             particles = [ P.ta__plus__, P.vt, P.pi__minus__ ],
             color = [ '1' ],
             lorentz = [ L.FFS1 ],
             couplings = {(0,0):C.GC_2})

V_7 = Vertex(name = 'V_7',
             particles = [ P.ve__tilde__, P.e__minus__, P.ta__plus__, P.vt ],
             color = [ '1' ],
             lorentz = [ L.FFFF1 ],
             couplings = {(0,0):C.GC_1})

V_8 = Vertex(name = 'V_8',
             particles = [ P.vm__tilde__, P.mu__minus__, P.ta__plus__, P.vt ],
             color = [ '1' ],
             lorentz = [ L.FFFF1 ],
             couplings = {(0,0):C.GC_1})

#V_9 = Vertex(name = 'V_9',
#             particles = [ P.vt__tilde__, P.ta__minus__, P.pi__minus__, P.pi__plus__, P.pi__plus__ ],
#             color = [ '1' ],
#             lorentz = [ L.FFSSS1, L.FFSSS2, L.FFSSS4 ],
#             couplings = {(0,2):C.GC_8,(0,1):C.GC_5,(0,0):C.GC_7})
#
#V_10 = Vertex(name = 'V_10',
#              particles = [ P.ta__plus__, P.vt, P.pi__minus__, P.pi__minus__, P.pi__plus__ ],
#              color = [ '1' ],
#              lorentz = [ L.FFSSS2, L.FFSSS3, L.FFSSS4 ],
#              couplings = {(0,0):C.GC_6,(0,2):C.GC_8,(0,1):C.GC_4})

V_9 = Vertex(name = 'V_9',
             particles = [ P.vt__tilde__, P.ta__minus__, P.pi__plus__, P.pi__plus__, P.pi__minus__ ],
             color = [ '1' ],
             lorentz = [ L.FFSSS5, L.FFSSS6, L.FFSSS7 ],
              couplings = {(0,0):C.GC_6,(0,2):C.GC_8,(0,1):C.GC_4})

V_10 = Vertex(name = 'V_10',
              particles = [ P.ta__plus__, P.vt, P.pi__minus__, P.pi__minus__, P.pi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.FFSSS5, L.FFSSS6, L.FFSSS7 ],
              couplings = {(0,0):C.GC_6,(0,2):C.GC_8,(0,1):C.GC_4})

V_11 = Vertex(name = 'V_11',
              particles = [ P.vt__tilde__, P.ta__minus__, P.pi0, P.pi0, P.pi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.FFSSS2, L.FFSSS3, L.FFSSS4 ],
              couplings = {(0,0):C.GC_6,(0,2):C.GC_8,(0,1):C.GC_4})

V_12 = Vertex(name = 'V_12',
              particles = [ P.ta__plus__, P.vt, P.pi0, P.pi0, P.pi__minus__ ],
              color = [ '1' ],
              lorentz = [ L.FFSSS2, L.FFSSS3, L.FFSSS4 ],
              couplings = {(0,0):C.GC_6,(0,2):C.GC_8,(0,1):C.GC_4})
