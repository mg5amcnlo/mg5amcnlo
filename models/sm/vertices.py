# This file was automatically created by FeynRules $Revision: 161 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (February 19, 2009)
# Date: Thu 20 May 2010 23:00:45


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(particles = [ P.G, P.G, P.G ],
             color = [ 'f(1,2,3)' ],
             lorentz = [ L.L_9 ],
             couplings = {(0,0):C.GC_5})

V_2 = Vertex(particles = [ P.G, P.G, P.G, P.G ],
             color = [ 'f(2,3,a1)*f(a1,1,4)', 'f(2,4,a1)*f(a1,1,3)', 'f(3,4,a1)*f(a1,1,2)' ],
             lorentz = [ L.L_12, L.L_14, L.L_15 ],
             couplings = {(1,1):C.GC_6,(2,0):C.GC_6,(0,2):C.GC_6})

V_3 = Vertex(particles = [ P.A, P.W__plus__, P.W__minus__ ],
             color = [ '1' ],
             lorentz = [ L.L_9 ],
             couplings = {(0,0):C.GC_21})

V_4 = Vertex(particles = [ P.A, P.A, P.W__plus__, P.W__minus__ ],
             color = [ '1' ],
             lorentz = [ L.L_13 ],
             couplings = {(0,0):C.GC_23})

V_5 = Vertex(particles = [ P.W__plus__, P.W__minus__, P.Z ],
             color = [ '1' ],
             lorentz = [ L.L_9 ],
             couplings = {(0,0):C.GC_7})

V_6 = Vertex(particles = [ P.W__plus__, P.W__plus__, P.W__minus__, P.W__minus__ ],
             color = [ '1' ],
             lorentz = [ L.L_13 ],
             couplings = {(0,0):C.GC_8})

V_7 = Vertex(particles = [ P.A, P.W__plus__, P.W__minus__, P.Z ],
             color = [ '1' ],
             lorentz = [ L.L_16 ],
             couplings = {(0,0):C.GC_22})

V_8 = Vertex(particles = [ P.W__plus__, P.W__minus__, P.Z, P.Z ],
             color = [ '1' ],
             lorentz = [ L.L_13 ],
             couplings = {(0,0):C.GC_9})

V_9 = Vertex(particles = [ P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.L_10 ],
             couplings = {(0,0):C.GC_10})

V_10 = Vertex(particles = [ P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.L_1 ],
              couplings = {(0,0):C.GC_26})

V_11 = Vertex(particles = [ P.H, P.H, P.W__plus__, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.L_11 ],
              couplings = {(0,0):C.GC_11})

V_12 = Vertex(particles = [ P.H, P.W__plus__, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.L_3 ],
              couplings = {(0,0):C.GC_27})

V_13 = Vertex(particles = [ P.H, P.H, P.Z, P.Z ],
              color = [ '1' ],
              lorentz = [ L.L_11 ],
              couplings = {(0,0):C.GC_25})

V_14 = Vertex(particles = [ P.H, P.Z, P.Z ],
              color = [ '1' ],
              lorentz = [ L.L_3 ],
              couplings = {(0,0):C.GC_28})

V_15 = Vertex(particles = [ P.A, P.d__tilde__, P.d ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_1})

V_16 = Vertex(particles = [ P.A, P.s__tilde__, P.s ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_1})

V_17 = Vertex(particles = [ P.A, P.b__tilde__, P.b ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_1})

V_18 = Vertex(particles = [ P.A, P.e__plus__, P.e__minus__ ],
              color = [ '1' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_3})

V_19 = Vertex(particles = [ P.A, P.m__plus__, P.m__minus__ ],
              color = [ '1' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_3})

V_20 = Vertex(particles = [ P.A, P.tt__plus__, P.tt__minus__ ],
              color = [ '1' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_3})

V_21 = Vertex(particles = [ P.A, P.u__tilde__, P.u ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_2})

V_22 = Vertex(particles = [ P.A, P.c__tilde__, P.c ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_2})

V_23 = Vertex(particles = [ P.A, P.t__tilde__, P.t ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_2})

V_24 = Vertex(particles = [ P.G, P.d__tilde__, P.d ],
              color = [ 'T(1,2,3)' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_4})

V_25 = Vertex(particles = [ P.G, P.s__tilde__, P.s ],
              color = [ 'T(1,2,3)' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_4})

V_26 = Vertex(particles = [ P.G, P.b__tilde__, P.b ],
              color = [ 'T(1,2,3)' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_4})

V_27 = Vertex(particles = [ P.H, P.b__tilde__, P.b ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_2 ],
              couplings = {(0,0):C.GC_29})

V_28 = Vertex(particles = [ P.Z, P.d__tilde__, P.d ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5, L.L_6 ],
              couplings = {(0,0):C.GC_17,(0,1):C.GC_19})

V_29 = Vertex(particles = [ P.Z, P.s__tilde__, P.s ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5, L.L_6 ],
              couplings = {(0,0):C.GC_17,(0,1):C.GC_19})

V_30 = Vertex(particles = [ P.Z, P.b__tilde__, P.b ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5, L.L_6 ],
              couplings = {(0,0):C.GC_17,(0,1):C.GC_19})

V_31 = Vertex(particles = [ P.W__plus__, P.u__tilde__, P.d ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_13})

V_32 = Vertex(particles = [ P.W__plus__, P.c__tilde__, P.d ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_15})

V_33 = Vertex(particles = [ P.W__plus__, P.u__tilde__, P.s ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_14})

V_34 = Vertex(particles = [ P.W__plus__, P.c__tilde__, P.s ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_16})

V_35 = Vertex(particles = [ P.W__plus__, P.t__tilde__, P.b ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_12})

V_36 = Vertex(particles = [ P.W__minus__, P.d__tilde__, P.u ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_33})

V_37 = Vertex(particles = [ P.W__minus__, P.d__tilde__, P.c ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_35})

V_38 = Vertex(particles = [ P.W__minus__, P.s__tilde__, P.u ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_34})

V_39 = Vertex(particles = [ P.W__minus__, P.s__tilde__, P.c ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_36})

V_40 = Vertex(particles = [ P.W__minus__, P.b__tilde__, P.t ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_12})

V_41 = Vertex(particles = [ P.G, P.u__tilde__, P.u ],
              color = [ 'T(1,2,3)' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_4})

V_42 = Vertex(particles = [ P.G, P.c__tilde__, P.c ],
              color = [ 'T(1,2,3)' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_4})

V_43 = Vertex(particles = [ P.G, P.t__tilde__, P.t ],
              color = [ 'T(1,2,3)' ],
              lorentz = [ L.L_4 ],
              couplings = {(0,0):C.GC_4})

V_44 = Vertex(particles = [ P.H, P.tt__plus__, P.tt__minus__ ],
              color = [ '1' ],
              lorentz = [ L.L_2 ],
              couplings = {(0,0):C.GC_32})

V_45 = Vertex(particles = [ P.H, P.c__tilde__, P.c ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_2 ],
              couplings = {(0,0):C.GC_30})

V_46 = Vertex(particles = [ P.H, P.t__tilde__, P.t ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_2 ],
              couplings = {(0,0):C.GC_31})

V_47 = Vertex(particles = [ P.Z, P.e__plus__, P.e__minus__ ],
              color = [ '1' ],
              lorentz = [ L.L_5, L.L_7 ],
              couplings = {(0,0):C.GC_17,(0,1):C.GC_20})

V_48 = Vertex(particles = [ P.Z, P.m__plus__, P.m__minus__ ],
              color = [ '1' ],
              lorentz = [ L.L_5, L.L_7 ],
              couplings = {(0,0):C.GC_17,(0,1):C.GC_20})

V_49 = Vertex(particles = [ P.Z, P.tt__plus__, P.tt__minus__ ],
              color = [ '1' ],
              lorentz = [ L.L_5, L.L_7 ],
              couplings = {(0,0):C.GC_17,(0,1):C.GC_20})

V_50 = Vertex(particles = [ P.W__plus__, P.ve__tilde__, P.e__minus__ ],
              color = [ '1' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_12})

V_51 = Vertex(particles = [ P.W__plus__, P.vm__tilde__, P.m__minus__ ],
              color = [ '1' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_12})

V_52 = Vertex(particles = [ P.W__plus__, P.vt__tilde__, P.tt__minus__ ],
              color = [ '1' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_12})

V_53 = Vertex(particles = [ P.W__minus__, P.e__plus__, P.ve ],
              color = [ '1' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_12})

V_54 = Vertex(particles = [ P.W__minus__, P.m__plus__, P.vm ],
              color = [ '1' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_12})

V_55 = Vertex(particles = [ P.W__minus__, P.tt__plus__, P.vt ],
              color = [ '1' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_12})

V_56 = Vertex(particles = [ P.Z, P.u__tilde__, P.u ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5, L.L_8 ],
              couplings = {(0,0):C.GC_18,(0,1):C.GC_19})

V_57 = Vertex(particles = [ P.Z, P.c__tilde__, P.c ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5, L.L_8 ],
              couplings = {(0,0):C.GC_18,(0,1):C.GC_19})

V_58 = Vertex(particles = [ P.Z, P.t__tilde__, P.t ],
              color = [ 'Identity(2,3)' ],
              lorentz = [ L.L_5, L.L_8 ],
              couplings = {(0,0):C.GC_18,(0,1):C.GC_19})

V_59 = Vertex(particles = [ P.Z, P.ve__tilde__, P.ve ],
              color = [ '1' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_24})

V_60 = Vertex(particles = [ P.Z, P.vm__tilde__, P.vm ],
              color = [ '1' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_24})

V_61 = Vertex(particles = [ P.Z, P.vt__tilde__, P.vt ],
              color = [ '1' ],
              lorentz = [ L.L_5 ],
              couplings = {(0,0):C.GC_24})

