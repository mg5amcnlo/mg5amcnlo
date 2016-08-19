# This file was automatically created by FeynRules 2.1.46
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (November 6, 2010)
# Date: Wed 19 Mar 2014 10:24:00


from object_library import all_vertices, all_CTvertices, Vertex, CTVertex
import particles as P
import CT_couplings as C
import lorentz as L


V_1 = CTVertex(name = 'V_1',
               type = 'R2',
               particles = [ P.g, P.g, P.g ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVV8 ],
               loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.g] ], [ [P.go] ] ],
               couplings = {(0,0,0):C.R2GC_1598_725,(0,0,1):C.R2GC_1598_726,(0,0,2):C.R2GC_1598_727})

V_2 = CTVertex(name = 'V_2',
               type = 'R2',
               particles = [ P.g, P.g, P.g, P.g ],
               color = [ 'd(-1,1,3)*d(-1,2,4)', 'd(-1,1,3)*f(-1,2,4)', 'd(-1,1,4)*d(-1,2,3)', 'd(-1,1,4)*f(-1,2,3)', 'd(-1,2,3)*f(-1,1,4)', 'd(-1,2,4)*f(-1,1,3)', 'f(-1,1,2)*f(-1,3,4)', 'f(-1,1,3)*f(-1,2,4)', 'f(-1,1,4)*f(-1,2,3)', 'Identity(1,2)*Identity(3,4)', 'Identity(1,3)*Identity(2,4)', 'Identity(1,4)*Identity(2,3)' ],
               lorentz = [ L.VVVV2, L.VVVV3, L.VVVV5, L.VVVV8 ],
               loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.g] ], [ [P.go] ] ],
               couplings = {(2,0,0):C.R2GC_1189_15,(2,0,1):C.R2GC_1189_16,(2,0,2):C.R2GC_1189_17,(0,0,0):C.R2GC_1189_15,(0,0,1):C.R2GC_1189_16,(0,0,2):C.R2GC_1189_17,(4,0,0):C.R2GC_1191_19,(4,0,1):C.R2GC_1191_20,(4,0,2):C.R2GC_1191_21,(3,0,0):C.R2GC_1191_19,(3,0,1):C.R2GC_1191_20,(3,0,2):C.R2GC_1191_21,(8,0,0):C.R2GC_1190_18,(8,0,1):C.R2GC_1189_17,(8,0,2):C.R2GC_1189_16,(6,0,0):C.R2GC_1196_28,(6,0,1):C.R2GC_1602_734,(6,0,2):C.R2GC_1196_30,(7,0,0):C.R2GC_1197_31,(7,0,1):C.R2GC_1603_735,(7,0,2):C.R2GC_1197_33,(5,0,0):C.R2GC_1191_19,(5,0,1):C.R2GC_1191_20,(5,0,2):C.R2GC_1191_21,(1,0,0):C.R2GC_1191_19,(1,0,1):C.R2GC_1191_20,(1,0,2):C.R2GC_1191_21,(11,3,0):C.R2GC_1188_12,(11,3,1):C.R2GC_1188_13,(11,3,2):C.R2GC_1188_14,(10,3,0):C.R2GC_1188_12,(10,3,1):C.R2GC_1188_13,(10,3,2):C.R2GC_1188_14,(9,3,1):C.R2GC_1176_3,(9,3,2):C.R2GC_1176_4,(0,1,0):C.R2GC_1189_15,(0,1,1):C.R2GC_1189_16,(0,1,2):C.R2GC_1189_17,(2,1,0):C.R2GC_1189_15,(2,1,1):C.R2GC_1189_16,(2,1,2):C.R2GC_1189_17,(5,1,0):C.R2GC_1191_19,(5,1,1):C.R2GC_1191_20,(5,1,2):C.R2GC_1191_21,(1,1,0):C.R2GC_1191_19,(1,1,1):C.R2GC_1191_20,(1,1,2):C.R2GC_1191_21,(7,1,0):C.R2GC_1197_31,(7,1,1):C.R2GC_1197_32,(7,1,2):C.R2GC_1197_33,(4,1,0):C.R2GC_1191_19,(4,1,1):C.R2GC_1191_20,(4,1,2):C.R2GC_1191_21,(3,1,0):C.R2GC_1191_19,(3,1,1):C.R2GC_1191_20,(3,1,2):C.R2GC_1191_21,(8,1,0):C.R2GC_1190_18,(8,1,1):C.R2GC_1604_736,(8,1,2):C.R2GC_1189_16,(6,1,0):C.R2GC_1599_728,(6,1,1):C.R2GC_1599_729,(6,1,2):C.R2GC_1599_730,(0,2,0):C.R2GC_1189_15,(0,2,1):C.R2GC_1189_16,(0,2,2):C.R2GC_1189_17,(2,2,0):C.R2GC_1189_15,(2,2,1):C.R2GC_1189_16,(2,2,2):C.R2GC_1189_17,(5,2,0):C.R2GC_1191_19,(5,2,1):C.R2GC_1191_20,(5,2,2):C.R2GC_1191_21,(1,2,0):C.R2GC_1191_19,(1,2,1):C.R2GC_1191_20,(1,2,2):C.R2GC_1191_21,(7,2,0):C.R2GC_1600_731,(7,2,1):C.R2GC_1189_16,(7,2,2):C.R2GC_1600_732,(4,2,0):C.R2GC_1191_19,(4,2,1):C.R2GC_1191_20,(4,2,2):C.R2GC_1191_21,(3,2,0):C.R2GC_1191_19,(3,2,1):C.R2GC_1191_20,(3,2,2):C.R2GC_1191_21,(8,2,0):C.R2GC_1190_18,(8,2,1):C.R2GC_1601_733,(8,2,2):C.R2GC_1189_16,(6,2,0):C.R2GC_1196_28,(6,2,1):C.R2GC_1196_29,(6,2,2):C.R2GC_1196_30})

V_3 = CTVertex(name = 'V_3',
               type = 'R2',
               particles = [ P.b__tilde__, P.b, P.h02 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS2 ],
               loop_particles = [ [ [P.b, P.g] ] ],
               couplings = {(0,0,0):C.R2GC_1449_493})

V_4 = CTVertex(name = 'V_4',
               type = 'R2',
               particles = [ P.b__tilde__, P.b, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1 ],
               loop_particles = [ [ [P.b, P.g] ] ],
               couplings = {(0,0,0):C.R2GC_1450_494})

V_5 = CTVertex(name = 'V_5',
               type = 'R2',
               particles = [ P.t__tilde__, P.b, P.H__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS4 ],
               loop_particles = [ [ [P.b, P.g, P.t] ] ],
               couplings = {(0,1,0):C.R2GC_1643_797,(0,0,0):C.R2GC_2205_1076})

V_6 = CTVertex(name = 'V_6',
               type = 'R2',
               particles = [ P.b__tilde__, P.t, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS4 ],
               loop_particles = [ [ [P.b, P.g, P.t] ] ],
               couplings = {(0,1,0):C.R2GC_1639_793,(0,0,0):C.R2GC_2204_1075})

V_7 = CTVertex(name = 'V_7',
               type = 'R2',
               particles = [ P.t__tilde__, P.t, P.h01 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS2 ],
               loop_particles = [ [ [P.g, P.t] ] ],
               couplings = {(0,0,0):C.R2GC_1638_792})

V_8 = CTVertex(name = 'V_8',
               type = 'R2',
               particles = [ P.t__tilde__, P.t, P.A0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1 ],
               loop_particles = [ [ [P.g, P.t] ] ],
               couplings = {(0,0,0):C.R2GC_1640_794})

V_9 = CTVertex(name = 'V_9',
               type = 'R2',
               particles = [ P.n1, P.d, P.sd1__tilde__ ],
               color = [ 'Identity(2,3)' ],
               lorentz = [ L.FFS4 ],
               loop_particles = [ [ [P.d, P.g, P.sd1] ] ],
               couplings = {(0,0,0):C.R2GC_1262_259})

V_10 = CTVertex(name = 'V_10',
                type = 'R2',
                particles = [ P.n2, P.d, P.sd1__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.d, P.g, P.sd1] ] ],
                couplings = {(0,0,0):C.R2GC_1263_260})

V_11 = CTVertex(name = 'V_11',
                type = 'R2',
                particles = [ P.n3, P.d, P.sd1__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.d, P.g, P.sd1] ] ],
                couplings = {(0,0,0):C.R2GC_1264_261})

V_12 = CTVertex(name = 'V_12',
                type = 'R2',
                particles = [ P.n4, P.d, P.sd1__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.d, P.g, P.sd1] ] ],
                couplings = {(0,0,0):C.R2GC_1265_262})

V_13 = CTVertex(name = 'V_13',
                type = 'R2',
                particles = [ P.a, P.sd1__tilde__, P.sd1 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.VSS2 ],
                loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ] ],
                couplings = {(0,0,0):C.R2GC_1255_243,(0,0,1):C.R2GC_1255_244})

V_14 = CTVertex(name = 'V_14',
                type = 'R2',
                particles = [ P.u__tilde__, P.x1__plus__, P.sd1 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.g, P.sd1, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_1459_501})

V_15 = CTVertex(name = 'V_15',
                type = 'R2',
                particles = [ P.u__tilde__, P.x2__plus__, P.sd1 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.g, P.sd1, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_1460_502})

V_16 = CTVertex(name = 'V_16',
                type = 'R2',
                particles = [ P.d__tilde__, P.n1, P.sd1 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.d, P.g, P.sd1] ] ],
                couplings = {(0,0,0):C.R2GC_1251_239})

V_17 = CTVertex(name = 'V_17',
                type = 'R2',
                particles = [ P.d__tilde__, P.n2, P.sd1 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.d, P.g, P.sd1] ] ],
                couplings = {(0,0,0):C.R2GC_1252_240})

V_18 = CTVertex(name = 'V_18',
                type = 'R2',
                particles = [ P.d__tilde__, P.n3, P.sd1 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.d, P.g, P.sd1] ] ],
                couplings = {(0,0,0):C.R2GC_1253_241})

V_19 = CTVertex(name = 'V_19',
                type = 'R2',
                particles = [ P.d__tilde__, P.n4, P.sd1 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.d, P.g, P.sd1] ] ],
                couplings = {(0,0,0):C.R2GC_1254_242})

V_20 = CTVertex(name = 'V_20',
                type = 'R2',
                particles = [ P.a, P.a, P.sd1__tilde__, P.sd1 ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ] ],
                couplings = {(0,0,0):C.R2GC_1256_245,(0,0,1):C.R2GC_1256_246})

V_21 = CTVertex(name = 'V_21',
                type = 'R2',
                particles = [ P.h02, P.sd1__tilde__, P.sd1 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.g, P.sd1] ] ],
                couplings = {(0,0,0):C.R2GC_1916_1001})

V_22 = CTVertex(name = 'V_22',
                type = 'R2',
                particles = [ P.h01, P.sd1__tilde__, P.sd1 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.g, P.sd1] ] ],
                couplings = {(0,0,0):C.R2GC_1915_1000})

V_23 = CTVertex(name = 'V_23',
                type = 'R2',
                particles = [ P.n1, P.s, P.sd2__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.g, P.s, P.sd2] ] ],
                couplings = {(0,0,0):C.R2GC_1288_298})

V_24 = CTVertex(name = 'V_24',
                type = 'R2',
                particles = [ P.n2, P.s, P.sd2__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.g, P.s, P.sd2] ] ],
                couplings = {(0,0,0):C.R2GC_1289_299})

V_25 = CTVertex(name = 'V_25',
                type = 'R2',
                particles = [ P.n3, P.s, P.sd2__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.g, P.s, P.sd2] ] ],
                couplings = {(0,0,0):C.R2GC_1290_300})

V_26 = CTVertex(name = 'V_26',
                type = 'R2',
                particles = [ P.n4, P.s, P.sd2__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.g, P.s, P.sd2] ] ],
                couplings = {(0,0,0):C.R2GC_1291_301})

V_27 = CTVertex(name = 'V_27',
                type = 'R2',
                particles = [ P.a, P.sd2__tilde__, P.sd2 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.VSS2 ],
                loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ] ],
                couplings = {(0,0,1):C.R2GC_1281_282,(0,0,0):C.R2GC_1281_283})

V_28 = CTVertex(name = 'V_28',
                type = 'R2',
                particles = [ P.c__tilde__, P.x1__plus__, P.sd2 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.c, P.g, P.sd2] ] ],
                couplings = {(0,0,0):C.R2GC_1505_574})

V_29 = CTVertex(name = 'V_29',
                type = 'R2',
                particles = [ P.c__tilde__, P.x2__plus__, P.sd2 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.c, P.g, P.sd2] ] ],
                couplings = {(0,0,0):C.R2GC_1506_575})

V_30 = CTVertex(name = 'V_30',
                type = 'R2',
                particles = [ P.s__tilde__, P.n1, P.sd2 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.g, P.s, P.sd2] ] ],
                couplings = {(0,0,0):C.R2GC_1277_278})

V_31 = CTVertex(name = 'V_31',
                type = 'R2',
                particles = [ P.s__tilde__, P.n2, P.sd2 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.g, P.s, P.sd2] ] ],
                couplings = {(0,0,0):C.R2GC_1278_279})

V_32 = CTVertex(name = 'V_32',
                type = 'R2',
                particles = [ P.s__tilde__, P.n3, P.sd2 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.g, P.s, P.sd2] ] ],
                couplings = {(0,0,0):C.R2GC_1279_280})

V_33 = CTVertex(name = 'V_33',
                type = 'R2',
                particles = [ P.s__tilde__, P.n4, P.sd2 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.g, P.s, P.sd2] ] ],
                couplings = {(0,0,0):C.R2GC_1280_281})

V_34 = CTVertex(name = 'V_34',
                type = 'R2',
                particles = [ P.a, P.a, P.sd2__tilde__, P.sd2 ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ] ],
                couplings = {(0,0,1):C.R2GC_1282_284,(0,0,0):C.R2GC_1282_285})

V_35 = CTVertex(name = 'V_35',
                type = 'R2',
                particles = [ P.h02, P.sd2__tilde__, P.sd2 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.g, P.sd2] ] ],
                couplings = {(0,0,0):C.R2GC_1918_1003})

V_36 = CTVertex(name = 'V_36',
                type = 'R2',
                particles = [ P.h01, P.sd2__tilde__, P.sd2 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.g, P.sd2] ] ],
                couplings = {(0,0,0):C.R2GC_1917_1002})

V_37 = CTVertex(name = 'V_37',
                type = 'R2',
                particles = [ P.n1, P.b, P.sd3__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ] ],
                couplings = {(0,0,1):C.R2GC_1437_473,(0,0,0):C.R2GC_1437_474,(0,1,1):C.R2GC_1433_469})

V_38 = CTVertex(name = 'V_38',
                type = 'R2',
                particles = [ P.n2, P.b, P.sd3__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ] ],
                couplings = {(0,0,1):C.R2GC_1438_475,(0,0,0):C.R2GC_1438_476,(0,1,1):C.R2GC_1434_470})

V_39 = CTVertex(name = 'V_39',
                type = 'R2',
                particles = [ P.n3, P.b, P.sd3__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ] ],
                couplings = {(0,0,1):C.R2GC_1439_477,(0,0,0):C.R2GC_1439_478,(0,1,1):C.R2GC_1435_471})

V_40 = CTVertex(name = 'V_40',
                type = 'R2',
                particles = [ P.n4, P.b, P.sd3__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ] ],
                couplings = {(0,0,1):C.R2GC_1440_479,(0,0,0):C.R2GC_1440_480,(0,1,1):C.R2GC_1436_472})

V_41 = CTVertex(name = 'V_41',
                type = 'R2',
                particles = [ P.a, P.sd3__tilde__, P.sd3 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.VSS2 ],
                loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ] ],
                couplings = {(0,0,0):C.R2GC_1299_316,(0,0,1):C.R2GC_1299_317})

V_42 = CTVertex(name = 'V_42',
                type = 'R2',
                particles = [ P.t__tilde__, P.x1__plus__, P.sd3 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.su6] ], [ [P.g, P.sd3, P.t] ] ],
                couplings = {(0,0,1):C.R2GC_1632_782,(0,1,0):C.R2GC_1626_772,(0,1,1):C.R2GC_1626_773})

V_43 = CTVertex(name = 'V_43',
                type = 'R2',
                particles = [ P.t__tilde__, P.x2__plus__, P.sd3 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.su6] ], [ [P.g, P.sd3, P.t] ] ],
                couplings = {(0,0,1):C.R2GC_1634_785,(0,1,0):C.R2GC_1627_774,(0,1,1):C.R2GC_1627_775})

V_44 = CTVertex(name = 'V_44',
                type = 'R2',
                particles = [ P.b__tilde__, P.n1, P.sd3 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ] ],
                couplings = {(0,1,1):C.R2GC_1421_449,(0,1,0):C.R2GC_1421_450,(0,0,1):C.R2GC_1425_457})

V_45 = CTVertex(name = 'V_45',
                type = 'R2',
                particles = [ P.b__tilde__, P.n2, P.sd3 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ] ],
                couplings = {(0,1,1):C.R2GC_1422_451,(0,1,0):C.R2GC_1422_452,(0,0,1):C.R2GC_1427_460})

V_46 = CTVertex(name = 'V_46',
                type = 'R2',
                particles = [ P.b__tilde__, P.n3, P.sd3 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ] ],
                couplings = {(0,1,1):C.R2GC_1423_453,(0,1,0):C.R2GC_1423_454,(0,0,1):C.R2GC_1429_463})

V_47 = CTVertex(name = 'V_47',
                type = 'R2',
                particles = [ P.b__tilde__, P.n4, P.sd3 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ] ],
                couplings = {(0,1,1):C.R2GC_1424_455,(0,1,0):C.R2GC_1424_456,(0,0,1):C.R2GC_1431_466})

V_48 = CTVertex(name = 'V_48',
                type = 'R2',
                particles = [ P.a, P.a, P.sd3__tilde__, P.sd3 ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ] ],
                couplings = {(0,0,0):C.R2GC_1300_318,(0,0,1):C.R2GC_1300_319})

V_49 = CTVertex(name = 'V_49',
                type = 'R2',
                particles = [ P.h02, P.sd3__tilde__, P.sd3 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ] ],
                couplings = {(0,0,0):C.R2GC_1919_1004,(0,0,1):C.R2GC_1919_1005})

V_50 = CTVertex(name = 'V_50',
                type = 'R2',
                particles = [ P.h01, P.sd3__tilde__, P.sd3 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ] ],
                couplings = {(0,0,0):C.R2GC_1928_1017,(0,0,1):C.R2GC_1928_1018})

V_51 = CTVertex(name = 'V_51',
                type = 'R2',
                particles = [ P.n1, P.d, P.sd4__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.d, P.g, P.sd4] ] ],
                couplings = {(0,0,0):C.R2GC_1334_375})

V_52 = CTVertex(name = 'V_52',
                type = 'R2',
                particles = [ P.n2, P.d, P.sd4__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.d, P.g, P.sd4] ] ],
                couplings = {(0,0,0):C.R2GC_1335_376})

V_53 = CTVertex(name = 'V_53',
                type = 'R2',
                particles = [ P.n3, P.d, P.sd4__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.d, P.g, P.sd4] ] ],
                couplings = {(0,0,0):C.R2GC_1336_377})

V_54 = CTVertex(name = 'V_54',
                type = 'R2',
                particles = [ P.n4, P.d, P.sd4__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.d, P.g, P.sd4] ] ],
                couplings = {(0,0,0):C.R2GC_1337_378})

V_55 = CTVertex(name = 'V_55',
                type = 'R2',
                particles = [ P.a, P.sd4__tilde__, P.sd4 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.VSS2 ],
                loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ] ],
                couplings = {(0,0,0):C.R2GC_1323_352,(0,0,1):C.R2GC_1323_353})

V_56 = CTVertex(name = 'V_56',
                type = 'R2',
                particles = [ P.d__tilde__, P.n1, P.sd4 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.d, P.g, P.sd4] ] ],
                couplings = {(0,0,0):C.R2GC_1319_348})

V_57 = CTVertex(name = 'V_57',
                type = 'R2',
                particles = [ P.d__tilde__, P.n2, P.sd4 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.d, P.g, P.sd4] ] ],
                couplings = {(0,0,0):C.R2GC_1320_349})

V_58 = CTVertex(name = 'V_58',
                type = 'R2',
                particles = [ P.d__tilde__, P.n3, P.sd4 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.d, P.g, P.sd4] ] ],
                couplings = {(0,0,0):C.R2GC_1321_350})

V_59 = CTVertex(name = 'V_59',
                type = 'R2',
                particles = [ P.d__tilde__, P.n4, P.sd4 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.d, P.g, P.sd4] ] ],
                couplings = {(0,0,0):C.R2GC_1322_351})

V_60 = CTVertex(name = 'V_60',
                type = 'R2',
                particles = [ P.a, P.a, P.sd4__tilde__, P.sd4 ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ] ],
                couplings = {(0,0,0):C.R2GC_1324_354,(0,0,1):C.R2GC_1324_355})

V_61 = CTVertex(name = 'V_61',
                type = 'R2',
                particles = [ P.h02, P.sd4__tilde__, P.sd4 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.g, P.sd4] ] ],
                couplings = {(0,0,0):C.R2GC_1921_1007})

V_62 = CTVertex(name = 'V_62',
                type = 'R2',
                particles = [ P.h01, P.sd4__tilde__, P.sd4 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.g, P.sd4] ] ],
                couplings = {(0,0,0):C.R2GC_1920_1006})

V_63 = CTVertex(name = 'V_63',
                type = 'R2',
                particles = [ P.n1, P.s, P.sd5__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.g, P.s, P.sd5] ] ],
                couplings = {(0,0,0):C.R2GC_1359_411})

V_64 = CTVertex(name = 'V_64',
                type = 'R2',
                particles = [ P.n2, P.s, P.sd5__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.g, P.s, P.sd5] ] ],
                couplings = {(0,0,0):C.R2GC_1360_412})

V_65 = CTVertex(name = 'V_65',
                type = 'R2',
                particles = [ P.n3, P.s, P.sd5__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.g, P.s, P.sd5] ] ],
                couplings = {(0,0,0):C.R2GC_1361_413})

V_66 = CTVertex(name = 'V_66',
                type = 'R2',
                particles = [ P.n4, P.s, P.sd5__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3 ],
                loop_particles = [ [ [P.g, P.s, P.sd5] ] ],
                couplings = {(0,0,0):C.R2GC_1362_414})

V_67 = CTVertex(name = 'V_67',
                type = 'R2',
                particles = [ P.a, P.sd5__tilde__, P.sd5 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.VSS2 ],
                loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ] ],
                couplings = {(0,0,1):C.R2GC_1348_388,(0,0,0):C.R2GC_1348_389})

V_68 = CTVertex(name = 'V_68',
                type = 'R2',
                particles = [ P.s__tilde__, P.n1, P.sd5 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.g, P.s, P.sd5] ] ],
                couplings = {(0,0,0):C.R2GC_1344_384})

V_69 = CTVertex(name = 'V_69',
                type = 'R2',
                particles = [ P.s__tilde__, P.n2, P.sd5 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.g, P.s, P.sd5] ] ],
                couplings = {(0,0,0):C.R2GC_1345_385})

V_70 = CTVertex(name = 'V_70',
                type = 'R2',
                particles = [ P.s__tilde__, P.n3, P.sd5 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.g, P.s, P.sd5] ] ],
                couplings = {(0,0,0):C.R2GC_1346_386})

V_71 = CTVertex(name = 'V_71',
                type = 'R2',
                particles = [ P.s__tilde__, P.n4, P.sd5 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.g, P.s, P.sd5] ] ],
                couplings = {(0,0,0):C.R2GC_1347_387})

V_72 = CTVertex(name = 'V_72',
                type = 'R2',
                particles = [ P.a, P.a, P.sd5__tilde__, P.sd5 ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ] ],
                couplings = {(0,0,1):C.R2GC_1349_390,(0,0,0):C.R2GC_1349_391})

V_73 = CTVertex(name = 'V_73',
                type = 'R2',
                particles = [ P.h02, P.sd5__tilde__, P.sd5 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.g, P.sd5] ] ],
                couplings = {(0,0,0):C.R2GC_1923_1009})

V_74 = CTVertex(name = 'V_74',
                type = 'R2',
                particles = [ P.h01, P.sd5__tilde__, P.sd5 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.g, P.sd5] ] ],
                couplings = {(0,0,0):C.R2GC_1922_1008})

V_75 = CTVertex(name = 'V_75',
                type = 'R2',
                particles = [ P.n1, P.b, P.sd6__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ] ],
                couplings = {(0,1,1):C.R2GC_1441_481,(0,1,0):C.R2GC_1441_482,(0,0,1):C.R2GC_1445_489})

V_76 = CTVertex(name = 'V_76',
                type = 'R2',
                particles = [ P.n2, P.b, P.sd6__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ] ],
                couplings = {(0,1,1):C.R2GC_1442_483,(0,1,0):C.R2GC_1442_484,(0,0,1):C.R2GC_1446_490})

V_77 = CTVertex(name = 'V_77',
                type = 'R2',
                particles = [ P.n3, P.b, P.sd6__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ] ],
                couplings = {(0,1,1):C.R2GC_1443_485,(0,1,0):C.R2GC_1443_486,(0,0,1):C.R2GC_1447_491})

V_78 = CTVertex(name = 'V_78',
                type = 'R2',
                particles = [ P.n4, P.b, P.sd6__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ] ],
                couplings = {(0,1,1):C.R2GC_1444_487,(0,1,0):C.R2GC_1444_488,(0,0,1):C.R2GC_1448_492})

V_79 = CTVertex(name = 'V_79',
                type = 'R2',
                particles = [ P.a, P.sd6__tilde__, P.sd6 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.VSS2 ],
                loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ] ],
                couplings = {(0,0,0):C.R2GC_1390_419,(0,0,1):C.R2GC_1390_420})

V_80 = CTVertex(name = 'V_80',
                type = 'R2',
                particles = [ P.h02, P.sd3, P.sd6__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3, P.sd6] ] ],
                couplings = {(0,0,0):C.R2GC_1930_1021,(0,0,1):C.R2GC_1930_1022})

V_81 = CTVertex(name = 'V_81',
                type = 'R2',
                particles = [ P.h01, P.sd3, P.sd6__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3, P.sd6] ] ],
                couplings = {(0,0,0):C.R2GC_1926_1014,(0,0,1):C.R2GC_1926_1015})

V_82 = CTVertex(name = 'V_82',
                type = 'R2',
                particles = [ P.G0, P.sd3, P.sd6__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3, P.sd6] ] ],
                couplings = {(0,0,0):C.R2GC_2193_1063,(0,0,1):C.R2GC_2193_1064})

V_83 = CTVertex(name = 'V_83',
                type = 'R2',
                particles = [ P.A0, P.sd3, P.sd6__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3, P.sd6] ] ],
                couplings = {(0,0,0):C.R2GC_2190_1058,(0,0,1):C.R2GC_2190_1059})

V_84 = CTVertex(name = 'V_84',
                type = 'R2',
                particles = [ P.b__tilde__, P.n1, P.sd6 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ] ],
                couplings = {(0,0,1):C.R2GC_1426_458,(0,0,0):C.R2GC_1426_459,(0,1,1):C.R2GC_1415_445})

V_85 = CTVertex(name = 'V_85',
                type = 'R2',
                particles = [ P.b__tilde__, P.n2, P.sd6 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ] ],
                couplings = {(0,0,1):C.R2GC_1428_461,(0,0,0):C.R2GC_1428_462,(0,1,1):C.R2GC_1416_446})

V_86 = CTVertex(name = 'V_86',
                type = 'R2',
                particles = [ P.b__tilde__, P.n3, P.sd6 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ] ],
                couplings = {(0,0,1):C.R2GC_1430_464,(0,0,0):C.R2GC_1430_465,(0,1,1):C.R2GC_1417_447})

V_87 = CTVertex(name = 'V_87',
                type = 'R2',
                particles = [ P.b__tilde__, P.n4, P.sd6 ],
                color = [ 'Identity(1,3)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ] ],
                couplings = {(0,0,1):C.R2GC_1432_467,(0,0,0):C.R2GC_1432_468,(0,1,1):C.R2GC_1418_448})

V_88 = CTVertex(name = 'V_88',
                type = 'R2',
                particles = [ P.a, P.a, P.sd6__tilde__, P.sd6 ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ] ],
                couplings = {(0,0,0):C.R2GC_1391_421,(0,0,1):C.R2GC_1391_422})

V_89 = CTVertex(name = 'V_89',
                type = 'R2',
                particles = [ P.h02, P.sd6__tilde__, P.sd6 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ] ],
                couplings = {(0,0,0):C.R2GC_1925_1012,(0,0,1):C.R2GC_1925_1013})

V_90 = CTVertex(name = 'V_90',
                type = 'R2',
                particles = [ P.h01, P.sd6__tilde__, P.sd6 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.SSS1 ],
                loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ] ],
                couplings = {(0,0,0):C.R2GC_1931_1023,(0,0,1):C.R2GC_1931_1024})

V_91 = CTVertex(name = 'V_91',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.h01 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS2 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_1927_1016})

V_92 = CTVertex(name = 'V_92',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.h02 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS2 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_1940_1033})

V_93 = CTVertex(name = 'V_93',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.A0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS1 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_2191_1060})

V_94 = CTVertex(name = 'V_94',
                type = 'R2',
                particles = [ P.t__tilde__, P.b, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_1642_796,(0,1,0):C.R2GC_2206_1077})

V_95 = CTVertex(name = 'V_95',
                type = 'R2',
                particles = [ P.b__tilde__, P.t, P.H__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_1641_795,(0,1,0):C.R2GC_2202_1073})

V_96 = CTVertex(name = 'V_96',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS1 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2203_1074})

V_97 = CTVertex(name = 'V_97',
                type = 'R2',
                particles = [ P.n1, P.u, P.su1__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.g, P.su1, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_1479_539})

V_98 = CTVertex(name = 'V_98',
                type = 'R2',
                particles = [ P.n2, P.u, P.su1__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.g, P.su1, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_1480_540})

V_99 = CTVertex(name = 'V_99',
                type = 'R2',
                particles = [ P.n3, P.u, P.su1__tilde__ ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.FFS4 ],
                loop_particles = [ [ [P.g, P.su1, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_1481_541})

V_100 = CTVertex(name = 'V_100',
                 type = 'R2',
                 particles = [ P.n4, P.u, P.su1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.su1, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1482_542})

V_101 = CTVertex(name = 'V_101',
                 type = 'R2',
                 particles = [ P.a, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS2 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,1):C.R2GC_1472_523,(0,0,0):C.R2GC_1472_524})

V_102 = CTVertex(name = 'V_102',
                 type = 'R2',
                 particles = [ P.G__plus__, P.sd1, P.su1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_2197_1068})

V_103 = CTVertex(name = 'V_103',
                 type = 'R2',
                 particles = [ P.H__plus__, P.sd1, P.su1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_2196_1067})

V_104 = CTVertex(name = 'V_104',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.n1, P.su1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.su1, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1463_507})

V_105 = CTVertex(name = 'V_105',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.n2, P.su1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.su1, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1464_508})

V_106 = CTVertex(name = 'V_106',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.n3, P.su1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.su1, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1465_509})

V_107 = CTVertex(name = 'V_107',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.n4, P.su1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.su1, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1466_510})

V_108 = CTVertex(name = 'V_108',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.x1__minus__, P.su1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.g, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_1495_567})

V_109 = CTVertex(name = 'V_109',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.x2__minus__, P.su1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.g, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_1496_568})

V_110 = CTVertex(name = 'V_110',
                 type = 'R2',
                 particles = [ P.G__minus__, P.sd1__tilde__, P.su1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_2195_1066})

V_111 = CTVertex(name = 'V_111',
                 type = 'R2',
                 particles = [ P.H__minus__, P.sd1__tilde__, P.su1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_2194_1065})

V_112 = CTVertex(name = 'V_112',
                 type = 'R2',
                 particles = [ P.a, P.a, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,1):C.R2GC_1473_525,(0,0,0):C.R2GC_1473_526})

V_113 = CTVertex(name = 'V_113',
                 type = 'R2',
                 particles = [ P.h02, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_1933_1026})

V_114 = CTVertex(name = 'V_114',
                 type = 'R2',
                 particles = [ P.h01, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_1932_1025})

V_115 = CTVertex(name = 'V_115',
                 type = 'R2',
                 particles = [ P.n1, P.c, P.su2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1526_614})

V_116 = CTVertex(name = 'V_116',
                 type = 'R2',
                 particles = [ P.n2, P.c, P.su2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1527_615})

V_117 = CTVertex(name = 'V_117',
                 type = 'R2',
                 particles = [ P.n3, P.c, P.su2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1528_616})

V_118 = CTVertex(name = 'V_118',
                 type = 'R2',
                 particles = [ P.n4, P.c, P.su2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1529_617})

V_119 = CTVertex(name = 'V_119',
                 type = 'R2',
                 particles = [ P.a, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1519_598,(0,0,1):C.R2GC_1519_599,(0,1,0):C.R2GC_1518_596,(0,1,1):C.R2GC_1518_597})

V_120 = CTVertex(name = 'V_120',
                 type = 'R2',
                 particles = [ P.G__plus__, P.sd2, P.su2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_2201_1072})

V_121 = CTVertex(name = 'V_121',
                 type = 'R2',
                 particles = [ P.H__plus__, P.sd2, P.su2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_2200_1071})

V_122 = CTVertex(name = 'V_122',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.n1, P.su2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1509_580})

V_123 = CTVertex(name = 'V_123',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.n2, P.su2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1510_581})

V_124 = CTVertex(name = 'V_124',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.n3, P.su2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1511_582})

V_125 = CTVertex(name = 'V_125',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.n4, P.su2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1512_583})

V_126 = CTVertex(name = 'V_126',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.x1__minus__, P.su2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.s, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1542_642})

V_127 = CTVertex(name = 'V_127',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.x2__minus__, P.su2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.s, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1543_643})

V_128 = CTVertex(name = 'V_128',
                 type = 'R2',
                 particles = [ P.G__minus__, P.sd2__tilde__, P.su2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_2199_1070})

V_129 = CTVertex(name = 'V_129',
                 type = 'R2',
                 particles = [ P.H__minus__, P.sd2__tilde__, P.su2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_2198_1069})

V_130 = CTVertex(name = 'V_130',
                 type = 'R2',
                 particles = [ P.a, P.a, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1520_600,(0,0,1):C.R2GC_1520_601})

V_131 = CTVertex(name = 'V_131',
                 type = 'R2',
                 particles = [ P.h02, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1935_1028})

V_132 = CTVertex(name = 'V_132',
                 type = 'R2',
                 particles = [ P.h01, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1934_1027})

V_133 = CTVertex(name = 'V_133',
                 type = 'R2',
                 particles = [ P.n1, P.t, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su6, P.t] ], [ [P.g, P.su3, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1688_879,(0,0,0):C.R2GC_1688_880,(0,1,1):C.R2GC_1682_873})

V_134 = CTVertex(name = 'V_134',
                 type = 'R2',
                 particles = [ P.n2, P.t, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su6, P.t] ], [ [P.g, P.su3, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1689_881,(0,0,0):C.R2GC_1689_882,(0,1,1):C.R2GC_1683_874})

V_135 = CTVertex(name = 'V_135',
                 type = 'R2',
                 particles = [ P.n3, P.t, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su6, P.t] ], [ [P.g, P.su3, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1690_883,(0,0,0):C.R2GC_1690_884,(0,1,1):C.R2GC_1684_875})

V_136 = CTVertex(name = 'V_136',
                 type = 'R2',
                 particles = [ P.n4, P.t, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su6, P.t] ], [ [P.g, P.su3, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1691_885,(0,0,0):C.R2GC_1691_886,(0,1,1):C.R2GC_1685_876})

V_137 = CTVertex(name = 'V_137',
                 type = 'R2',
                 particles = [ P.a, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.R2GC_1655_823,(0,0,0):C.R2GC_1655_824,(0,1,1):C.R2GC_1654_821,(0,1,0):C.R2GC_1654_822})

V_138 = CTVertex(name = 'V_138',
                 type = 'R2',
                 particles = [ P.G__plus__, P.sd3, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.R2GC_2216_1092,(0,0,1):C.R2GC_2216_1093})

V_139 = CTVertex(name = 'V_139',
                 type = 'R2',
                 particles = [ P.H__plus__, P.sd3, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.R2GC_2212_1086,(0,0,1):C.R2GC_2212_1087})

V_140 = CTVertex(name = 'V_140',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.n1, P.su3 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su6, P.t] ], [ [P.g, P.su3, P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_1673_861,(0,1,0):C.R2GC_1673_862,(0,0,1):C.R2GC_1677_869})

V_141 = CTVertex(name = 'V_141',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.n2, P.su3 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su6, P.t] ], [ [P.g, P.su3, P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_1674_863,(0,1,0):C.R2GC_1674_864,(0,0,1):C.R2GC_1678_870})

V_142 = CTVertex(name = 'V_142',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.n3, P.su3 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su6, P.t] ], [ [P.g, P.su3, P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_1675_865,(0,1,0):C.R2GC_1675_866,(0,0,1):C.R2GC_1679_871})

V_143 = CTVertex(name = 'V_143',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.n4, P.su3 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su6, P.t] ], [ [P.g, P.su3, P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_1676_867,(0,1,0):C.R2GC_1676_868,(0,0,1):C.R2GC_1680_872})

V_144 = CTVertex(name = 'V_144',
                 type = 'R2',
                 particles = [ P.x1__minus__, P.b__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g, P.su3] ], [ [P.go, P.sd6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1694_891,(0,1,0):C.R2GC_1647_805,(0,1,1):C.R2GC_1647_806})

V_145 = CTVertex(name = 'V_145',
                 type = 'R2',
                 particles = [ P.x2__minus__, P.b__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g, P.su3] ], [ [P.go, P.sd6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1695_892,(0,1,0):C.R2GC_1648_807,(0,1,1):C.R2GC_1648_808})

V_146 = CTVertex(name = 'V_146',
                 type = 'R2',
                 particles = [ P.G__minus__, P.sd3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.R2GC_2207_1078,(0,0,1):C.R2GC_2207_1079})

V_147 = CTVertex(name = 'V_147',
                 type = 'R2',
                 particles = [ P.H__minus__, P.sd3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.R2GC_2208_1080,(0,0,1):C.R2GC_2208_1081})

V_148 = CTVertex(name = 'V_148',
                 type = 'R2',
                 particles = [ P.G__minus__, P.sd6__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd6, P.su3] ] ],
                 couplings = {(0,0,0):C.R2GC_2209_1082,(0,0,1):C.R2GC_2209_1083})

V_149 = CTVertex(name = 'V_149',
                 type = 'R2',
                 particles = [ P.H__minus__, P.sd6__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd6, P.su3] ] ],
                 couplings = {(0,0,0):C.R2GC_2214_1090,(0,0,1):C.R2GC_2214_1091})

V_150 = CTVertex(name = 'V_150',
                 type = 'R2',
                 particles = [ P.a, P.a, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.R2GC_1656_825,(0,0,0):C.R2GC_1656_826})

V_151 = CTVertex(name = 'V_151',
                 type = 'R2',
                 particles = [ P.h02, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.R2GC_1942_1036,(0,0,0):C.R2GC_1942_1037})

V_152 = CTVertex(name = 'V_152',
                 type = 'R2',
                 particles = [ P.h01, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.R2GC_1941_1034,(0,0,0):C.R2GC_1941_1035})

V_153 = CTVertex(name = 'V_153',
                 type = 'R2',
                 particles = [ P.n1, P.u, P.su4__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.su4, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1566_679})

V_154 = CTVertex(name = 'V_154',
                 type = 'R2',
                 particles = [ P.n2, P.u, P.su4__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.su4, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1567_680})

V_155 = CTVertex(name = 'V_155',
                 type = 'R2',
                 particles = [ P.n3, P.u, P.su4__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.su4, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1568_681})

V_156 = CTVertex(name = 'V_156',
                 type = 'R2',
                 particles = [ P.n4, P.u, P.su4__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.su4, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1569_682})

V_157 = CTVertex(name = 'V_157',
                 type = 'R2',
                 particles = [ P.a, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ] ],
                 couplings = {(0,0,1):C.R2GC_1555_656,(0,0,0):C.R2GC_1555_657,(0,1,1):C.R2GC_1554_654,(0,1,0):C.R2GC_1554_655})

V_158 = CTVertex(name = 'V_158',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.n1, P.su4 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.su4, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1550_650})

V_159 = CTVertex(name = 'V_159',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.n2, P.su4 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.su4, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1551_651})

V_160 = CTVertex(name = 'V_160',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.n3, P.su4 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.su4, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1552_652})

V_161 = CTVertex(name = 'V_161',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.n4, P.su4 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.su4, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1553_653})

V_162 = CTVertex(name = 'V_162',
                 type = 'R2',
                 particles = [ P.a, P.a, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ] ],
                 couplings = {(0,0,1):C.R2GC_1556_658,(0,0,0):C.R2GC_1556_659})

V_163 = CTVertex(name = 'V_163',
                 type = 'R2',
                 particles = [ P.h02, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.su4] ] ],
                 couplings = {(0,0,0):C.R2GC_1937_1030})

V_164 = CTVertex(name = 'V_164',
                 type = 'R2',
                 particles = [ P.h01, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.su4] ] ],
                 couplings = {(0,0,0):C.R2GC_1936_1029})

V_165 = CTVertex(name = 'V_165',
                 type = 'R2',
                 particles = [ P.n1, P.c, P.su5__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1592_717})

V_166 = CTVertex(name = 'V_166',
                 type = 'R2',
                 particles = [ P.n2, P.c, P.su5__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1593_718})

V_167 = CTVertex(name = 'V_167',
                 type = 'R2',
                 particles = [ P.n3, P.c, P.su5__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1594_719})

V_168 = CTVertex(name = 'V_168',
                 type = 'R2',
                 particles = [ P.n4, P.c, P.su5__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1595_720})

V_169 = CTVertex(name = 'V_169',
                 type = 'R2',
                 particles = [ P.a, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1581_694,(0,0,1):C.R2GC_1581_695,(0,1,0):C.R2GC_1580_692,(0,1,1):C.R2GC_1580_693})

V_170 = CTVertex(name = 'V_170',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.n1, P.su5 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1576_688})

V_171 = CTVertex(name = 'V_171',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.n2, P.su5 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1577_689})

V_172 = CTVertex(name = 'V_172',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.n3, P.su5 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1578_690})

V_173 = CTVertex(name = 'V_173',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.n4, P.su5 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1579_691})

V_174 = CTVertex(name = 'V_174',
                 type = 'R2',
                 particles = [ P.a, P.a, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1582_696,(0,0,1):C.R2GC_1582_697})

V_175 = CTVertex(name = 'V_175',
                 type = 'R2',
                 particles = [ P.h02, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1939_1032})

V_176 = CTVertex(name = 'V_176',
                 type = 'R2',
                 particles = [ P.h01, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1938_1031})

V_177 = CTVertex(name = 'V_177',
                 type = 'R2',
                 particles = [ P.n1, P.t, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3, P.t] ], [ [P.g, P.su6, P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_1749_937,(0,1,0):C.R2GC_1749_938,(0,0,1):C.R2GC_1755_949})

V_178 = CTVertex(name = 'V_178',
                 type = 'R2',
                 particles = [ P.n2, P.t, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3, P.t] ], [ [P.g, P.su6, P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_1750_939,(0,1,0):C.R2GC_1750_940,(0,0,1):C.R2GC_1756_950})

V_179 = CTVertex(name = 'V_179',
                 type = 'R2',
                 particles = [ P.n3, P.t, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3, P.t] ], [ [P.g, P.su6, P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_1751_941,(0,1,0):C.R2GC_1751_942,(0,0,1):C.R2GC_1757_951})

V_180 = CTVertex(name = 'V_180',
                 type = 'R2',
                 particles = [ P.n4, P.t, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3, P.t] ], [ [P.g, P.su6, P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_1752_943,(0,1,0):C.R2GC_1752_944,(0,0,1):C.R2GC_1758_952})

V_181 = CTVertex(name = 'V_181',
                 type = 'R2',
                 particles = [ P.a, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(0,0,1):C.R2GC_1738_914,(0,0,0):C.R2GC_1738_915,(0,1,1):C.R2GC_1737_912,(0,1,0):C.R2GC_1737_913})

V_182 = CTVertex(name = 'V_182',
                 type = 'R2',
                 particles = [ P.H__plus__, P.sd3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_2225_1108,(0,0,1):C.R2GC_2225_1109})

V_183 = CTVertex(name = 'V_183',
                 type = 'R2',
                 particles = [ P.G__plus__, P.sd3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_2230_1116,(0,0,1):C.R2GC_2230_1117})

V_184 = CTVertex(name = 'V_184',
                 type = 'R2',
                 particles = [ P.h01, P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_1946_1044,(0,0,1):C.R2GC_1946_1045})

V_185 = CTVertex(name = 'V_185',
                 type = 'R2',
                 particles = [ P.h02, P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_1948_1048,(0,0,1):C.R2GC_1948_1049})

V_186 = CTVertex(name = 'V_186',
                 type = 'R2',
                 particles = [ P.A0, P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_2223_1104,(0,0,1):C.R2GC_2223_1105})

V_187 = CTVertex(name = 'V_187',
                 type = 'R2',
                 particles = [ P.G0, P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_2229_1114,(0,0,1):C.R2GC_2229_1115})

V_188 = CTVertex(name = 'V_188',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.n1, P.su6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3, P.t] ], [ [P.g, P.su6, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1709_904,(0,0,0):C.R2GC_1709_905,(0,1,1):C.R2GC_1705_900})

V_189 = CTVertex(name = 'V_189',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.n2, P.su6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3, P.t] ], [ [P.g, P.su6, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1710_906,(0,0,0):C.R2GC_1710_907,(0,1,1):C.R2GC_1706_901})

V_190 = CTVertex(name = 'V_190',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.n3, P.su6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3, P.t] ], [ [P.g, P.su6, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1711_908,(0,0,0):C.R2GC_1711_909,(0,1,1):C.R2GC_1707_902})

V_191 = CTVertex(name = 'V_191',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.n4, P.su6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3, P.t] ], [ [P.g, P.su6, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1712_910,(0,0,0):C.R2GC_1712_911,(0,1,1):C.R2GC_1708_903})

V_192 = CTVertex(name = 'V_192',
                 type = 'R2',
                 particles = [ P.a, P.a, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(0,0,1):C.R2GC_1739_916,(0,0,0):C.R2GC_1739_917})

V_193 = CTVertex(name = 'V_193',
                 type = 'R2',
                 particles = [ P.h02, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(0,0,1):C.R2GC_1947_1046,(0,0,0):C.R2GC_1947_1047})

V_194 = CTVertex(name = 'V_194',
                 type = 'R2',
                 particles = [ P.h01, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(0,0,1):C.R2GC_1945_1042,(0,0,0):C.R2GC_1945_1043})

V_195 = CTVertex(name = 'V_195',
                 type = 'R2',
                 particles = [ P.go, P.d, P.sd1__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.g, P.go] ], [ [P.d, P.g, P.sd1] ], [ [P.g, P.go, P.sd1] ] ],
                 couplings = {(0,0,0):C.R2GC_1766_960,(0,0,1):C.R2GC_1766_961,(0,0,2):C.R2GC_1766_962})

V_196 = CTVertex(name = 'V_196',
                 type = 'R2',
                 particles = [ P.g, P.sd1__tilde__, P.sd1 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.R2GC_1258_249,(0,0,1):C.R2GC_1258_250,(0,1,0):C.R2GC_1257_247,(0,1,1):C.R2GC_1257_248})

V_197 = CTVertex(name = 'V_197',
                 type = 'R2',
                 particles = [ P.go, P.s, P.sd2__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.go, P.s] ], [ [P.g, P.go, P.sd2] ], [ [P.g, P.s, P.sd2] ] ],
                 couplings = {(0,0,0):C.R2GC_1767_963,(0,0,1):C.R2GC_1767_964,(0,0,2):C.R2GC_1767_965})

V_198 = CTVertex(name = 'V_198',
                 type = 'R2',
                 particles = [ P.g, P.sd2__tilde__, P.sd2 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ] ],
                 couplings = {(0,0,1):C.R2GC_1284_288,(0,0,0):C.R2GC_1284_289,(0,1,1):C.R2GC_1283_286,(0,1,0):C.R2GC_1283_287})

V_199 = CTVertex(name = 'V_199',
                 type = 'R2',
                 particles = [ P.go, P.b, P.sd3__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g, P.go] ], [ [P.b, P.g, P.sd3] ], [ [P.g, P.go, P.sd3] ] ],
                 couplings = {(0,0,0):C.R2GC_1768_966,(0,0,1):C.R2GC_1768_967,(0,0,2):C.R2GC_1768_968})

V_200 = CTVertex(name = 'V_200',
                 type = 'R2',
                 particles = [ P.g, P.sd3__tilde__, P.sd3 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,0,0):C.R2GC_1302_322,(0,0,1):C.R2GC_1302_323,(0,1,0):C.R2GC_1301_320,(0,1,1):C.R2GC_1301_321})

V_201 = CTVertex(name = 'V_201',
                 type = 'R2',
                 particles = [ P.go, P.d, P.sd4__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.g, P.go] ], [ [P.d, P.g, P.sd4] ], [ [P.g, P.go, P.sd4] ] ],
                 couplings = {(0,0,0):C.R2GC_1769_969,(0,0,1):C.R2GC_1769_970,(0,0,2):C.R2GC_1769_971})

V_202 = CTVertex(name = 'V_202',
                 type = 'R2',
                 particles = [ P.g, P.sd4__tilde__, P.sd4 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.R2GC_1326_358,(0,0,1):C.R2GC_1326_359,(0,1,0):C.R2GC_1325_356,(0,1,1):C.R2GC_1325_357})

V_203 = CTVertex(name = 'V_203',
                 type = 'R2',
                 particles = [ P.go, P.s, P.sd5__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.go, P.s] ], [ [P.g, P.go, P.sd5] ], [ [P.g, P.s, P.sd5] ] ],
                 couplings = {(0,0,0):C.R2GC_1770_972,(0,0,1):C.R2GC_1770_973,(0,0,2):C.R2GC_1770_974})

V_204 = CTVertex(name = 'V_204',
                 type = 'R2',
                 particles = [ P.g, P.sd5__tilde__, P.sd5 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ] ],
                 couplings = {(0,0,1):C.R2GC_1351_394,(0,0,0):C.R2GC_1351_395,(0,1,1):C.R2GC_1350_392,(0,1,0):C.R2GC_1350_393})

V_205 = CTVertex(name = 'V_205',
                 type = 'R2',
                 particles = [ P.go, P.b, P.sd6__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.b, P.g, P.go] ], [ [P.b, P.g, P.sd6] ], [ [P.g, P.go, P.sd6] ] ],
                 couplings = {(0,0,0):C.R2GC_1771_975,(0,0,1):C.R2GC_1771_976,(0,0,2):C.R2GC_1771_977})

V_206 = CTVertex(name = 'V_206',
                 type = 'R2',
                 particles = [ P.g, P.sd6__tilde__, P.sd6 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,0,0):C.R2GC_1393_425,(0,0,1):C.R2GC_1393_426,(0,1,0):C.R2GC_1392_423,(0,1,1):C.R2GC_1392_424})

V_207 = CTVertex(name = 'V_207',
                 type = 'R2',
                 particles = [ P.go, P.u, P.su1__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.go, P.su1] ], [ [P.g, P.go, P.u] ], [ [P.g, P.su1, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1772_978,(0,0,1):C.R2GC_1772_979,(0,0,2):C.R2GC_1772_980})

V_208 = CTVertex(name = 'V_208',
                 type = 'R2',
                 particles = [ P.g, P.su1__tilde__, P.su1 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,1):C.R2GC_1475_529,(0,0,0):C.R2GC_1475_530,(0,1,1):C.R2GC_1474_527,(0,1,0):C.R2GC_1474_528})

V_209 = CTVertex(name = 'V_209',
                 type = 'R2',
                 particles = [ P.go, P.c, P.su2__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g, P.go] ], [ [P.c, P.g, P.su2] ], [ [P.g, P.go, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1773_981,(0,0,1):C.R2GC_1773_982,(0,0,2):C.R2GC_1773_983})

V_210 = CTVertex(name = 'V_210',
                 type = 'R2',
                 particles = [ P.g, P.su2__tilde__, P.su2 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1522_604,(0,0,1):C.R2GC_1522_605,(0,1,0):C.R2GC_1521_602,(0,1,1):C.R2GC_1521_603})

V_211 = CTVertex(name = 'V_211',
                 type = 'R2',
                 particles = [ P.go, P.t, P.su3__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.go, P.su3] ], [ [P.g, P.go, P.t] ], [ [P.g, P.su3, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1774_984,(0,0,1):C.R2GC_1774_985,(0,0,2):C.R2GC_1774_986})

V_212 = CTVertex(name = 'V_212',
                 type = 'R2',
                 particles = [ P.g, P.su3__tilde__, P.su3 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.R2GC_1658_829,(0,0,0):C.R2GC_1658_830,(0,1,1):C.R2GC_1657_827,(0,1,0):C.R2GC_1657_828})

V_213 = CTVertex(name = 'V_213',
                 type = 'R2',
                 particles = [ P.go, P.u, P.su4__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.go, P.su4] ], [ [P.g, P.go, P.u] ], [ [P.g, P.su4, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1775_987,(0,0,1):C.R2GC_1775_988,(0,0,2):C.R2GC_1775_989})

V_214 = CTVertex(name = 'V_214',
                 type = 'R2',
                 particles = [ P.g, P.su4__tilde__, P.su4 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ] ],
                 couplings = {(0,0,1):C.R2GC_1558_662,(0,0,0):C.R2GC_1558_663,(0,1,1):C.R2GC_1557_660,(0,1,0):C.R2GC_1557_661})

V_215 = CTVertex(name = 'V_215',
                 type = 'R2',
                 particles = [ P.go, P.c, P.su5__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g, P.go] ], [ [P.c, P.g, P.su5] ], [ [P.g, P.go, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1776_990,(0,0,1):C.R2GC_1776_991,(0,0,2):C.R2GC_1776_992})

V_216 = CTVertex(name = 'V_216',
                 type = 'R2',
                 particles = [ P.g, P.su5__tilde__, P.su5 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1584_700,(0,0,1):C.R2GC_1584_701,(0,1,0):C.R2GC_1583_698,(0,1,1):C.R2GC_1583_699})

V_217 = CTVertex(name = 'V_217',
                 type = 'R2',
                 particles = [ P.go, P.t, P.su6__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.go, P.su6] ], [ [P.g, P.go, P.t] ], [ [P.g, P.su6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1777_993,(0,0,1):C.R2GC_1777_994,(0,0,2):C.R2GC_1777_995})

V_218 = CTVertex(name = 'V_218',
                 type = 'R2',
                 particles = [ P.g, P.su6__tilde__, P.su6 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(0,0,1):C.R2GC_1741_920,(0,0,0):C.R2GC_1741_921,(0,1,1):C.R2GC_1740_918,(0,1,0):C.R2GC_1740_919})

V_219 = CTVertex(name = 'V_219',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.go, P.sd1 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.g, P.go] ], [ [P.d, P.g, P.sd1] ], [ [P.g, P.go, P.sd1] ] ],
                 couplings = {(0,0,0):C.R2GC_1605_737,(0,0,1):C.R2GC_1605_738,(0,0,2):C.R2GC_1605_739})

V_220 = CTVertex(name = 'V_220',
                 type = 'R2',
                 particles = [ P.a, P.g, P.sd1__tilde__, P.sd1 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.R2GC_1259_251,(0,0,1):C.R2GC_1259_252})

V_221 = CTVertex(name = 'V_221',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.go, P.sd2 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.go, P.s] ], [ [P.g, P.go, P.sd2] ], [ [P.g, P.s, P.sd2] ] ],
                 couplings = {(0,0,0):C.R2GC_1606_740,(0,0,1):C.R2GC_1606_741,(0,0,2):C.R2GC_1606_742})

V_222 = CTVertex(name = 'V_222',
                 type = 'R2',
                 particles = [ P.a, P.g, P.sd2__tilde__, P.sd2 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ] ],
                 couplings = {(0,0,1):C.R2GC_1285_290,(0,0,0):C.R2GC_1285_291})

V_223 = CTVertex(name = 'V_223',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.go, P.sd3 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.b, P.g, P.go] ], [ [P.b, P.g, P.sd3] ], [ [P.g, P.go, P.sd3] ] ],
                 couplings = {(0,0,0):C.R2GC_1607_743,(0,0,1):C.R2GC_1607_744,(0,0,2):C.R2GC_1607_745})

V_224 = CTVertex(name = 'V_224',
                 type = 'R2',
                 particles = [ P.a, P.g, P.sd3__tilde__, P.sd3 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,0,0):C.R2GC_1303_324,(0,0,1):C.R2GC_1303_325})

V_225 = CTVertex(name = 'V_225',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.go, P.sd4 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.g, P.go] ], [ [P.d, P.g, P.sd4] ], [ [P.g, P.go, P.sd4] ] ],
                 couplings = {(0,0,0):C.R2GC_1608_746,(0,0,1):C.R2GC_1608_747,(0,0,2):C.R2GC_1608_748})

V_226 = CTVertex(name = 'V_226',
                 type = 'R2',
                 particles = [ P.a, P.g, P.sd4__tilde__, P.sd4 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.R2GC_1327_360,(0,0,1):C.R2GC_1327_361})

V_227 = CTVertex(name = 'V_227',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.go, P.sd5 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.go, P.s] ], [ [P.g, P.go, P.sd5] ], [ [P.g, P.s, P.sd5] ] ],
                 couplings = {(0,0,0):C.R2GC_1609_749,(0,0,1):C.R2GC_1609_750,(0,0,2):C.R2GC_1609_751})

V_228 = CTVertex(name = 'V_228',
                 type = 'R2',
                 particles = [ P.a, P.g, P.sd5__tilde__, P.sd5 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ] ],
                 couplings = {(0,0,1):C.R2GC_1352_396,(0,0,0):C.R2GC_1352_397})

V_229 = CTVertex(name = 'V_229',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.go, P.sd6 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g, P.go] ], [ [P.b, P.g, P.sd6] ], [ [P.g, P.go, P.sd6] ] ],
                 couplings = {(0,0,0):C.R2GC_1610_752,(0,0,1):C.R2GC_1610_753,(0,0,2):C.R2GC_1610_754})

V_230 = CTVertex(name = 'V_230',
                 type = 'R2',
                 particles = [ P.a, P.g, P.sd6__tilde__, P.sd6 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,0,0):C.R2GC_1394_427,(0,0,1):C.R2GC_1394_428})

V_231 = CTVertex(name = 'V_231',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.go, P.su1 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.go, P.su1] ], [ [P.g, P.go, P.u] ], [ [P.g, P.su1, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1611_755,(0,0,1):C.R2GC_1611_756,(0,0,2):C.R2GC_1611_757})

V_232 = CTVertex(name = 'V_232',
                 type = 'R2',
                 particles = [ P.a, P.g, P.su1__tilde__, P.su1 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,1):C.R2GC_1476_531,(0,0,0):C.R2GC_1476_532})

V_233 = CTVertex(name = 'V_233',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.go, P.su2 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g, P.go] ], [ [P.c, P.g, P.su2] ], [ [P.g, P.go, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1612_758,(0,0,1):C.R2GC_1612_759,(0,0,2):C.R2GC_1612_760})

V_234 = CTVertex(name = 'V_234',
                 type = 'R2',
                 particles = [ P.a, P.g, P.su2__tilde__, P.su2 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1523_606,(0,0,1):C.R2GC_1523_607})

V_235 = CTVertex(name = 'V_235',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.go, P.su3 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.g, P.go, P.su3] ], [ [P.g, P.go, P.t] ], [ [P.g, P.su3, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1646_802,(0,0,1):C.R2GC_1646_803,(0,0,2):C.R2GC_1646_804})

V_236 = CTVertex(name = 'V_236',
                 type = 'R2',
                 particles = [ P.a, P.g, P.su3__tilde__, P.su3 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.R2GC_1659_831,(0,0,0):C.R2GC_1659_832})

V_237 = CTVertex(name = 'V_237',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.go, P.su4 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.go, P.su4] ], [ [P.g, P.go, P.u] ], [ [P.g, P.su4, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1613_761,(0,0,1):C.R2GC_1613_762,(0,0,2):C.R2GC_1613_763})

V_238 = CTVertex(name = 'V_238',
                 type = 'R2',
                 particles = [ P.a, P.g, P.su4__tilde__, P.su4 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ] ],
                 couplings = {(0,0,1):C.R2GC_1559_664,(0,0,0):C.R2GC_1559_665})

V_239 = CTVertex(name = 'V_239',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.go, P.su5 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g, P.go] ], [ [P.c, P.g, P.su5] ], [ [P.g, P.go, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1614_764,(0,0,1):C.R2GC_1614_765,(0,0,2):C.R2GC_1614_766})

V_240 = CTVertex(name = 'V_240',
                 type = 'R2',
                 particles = [ P.a, P.g, P.su5__tilde__, P.su5 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1585_702,(0,0,1):C.R2GC_1585_703})

V_241 = CTVertex(name = 'V_241',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.go, P.su6 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.go, P.su6] ], [ [P.g, P.go, P.t] ], [ [P.g, P.su6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1702_897,(0,0,1):C.R2GC_1702_898,(0,0,2):C.R2GC_1702_899})

V_242 = CTVertex(name = 'V_242',
                 type = 'R2',
                 particles = [ P.a, P.g, P.su6__tilde__, P.su6 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(0,0,1):C.R2GC_1742_922,(0,0,0):C.R2GC_1742_923})

V_243 = CTVertex(name = 'V_243',
                 type = 'R2',
                 particles = [ P.g, P.g, P.sd4__tilde__, P.sd4 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g] ], [ [P.g, P.sd4] ] ],
                 couplings = {(2,0,1):C.R2GC_1328_362,(2,0,0):C.R2GC_1328_363,(2,0,2):C.R2GC_1328_364,(1,0,1):C.R2GC_1328_362,(1,0,0):C.R2GC_1328_363,(1,0,2):C.R2GC_1328_364,(0,0,1):C.R2GC_1209_47,(0,0,0):C.R2GC_1209_48,(0,0,2):C.R2GC_1209_49})

V_244 = CTVertex(name = 'V_244',
                 type = 'R2',
                 particles = [ P.g, P.g, P.sd5__tilde__, P.sd5 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.g] ], [ [P.go, P.s] ], [ [P.g, P.sd5] ] ],
                 couplings = {(2,0,0):C.R2GC_1353_398,(2,0,2):C.R2GC_1353_399,(2,0,1):C.R2GC_1353_400,(1,0,0):C.R2GC_1353_398,(1,0,2):C.R2GC_1353_399,(1,0,1):C.R2GC_1353_400,(0,0,0):C.R2GC_1210_50,(0,0,2):C.R2GC_1210_51,(0,0,1):C.R2GC_1210_52})

V_245 = CTVertex(name = 'V_245',
                 type = 'R2',
                 particles = [ P.g, P.g, P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g] ], [ [P.g, P.sd6] ] ],
                 couplings = {(2,0,1):C.R2GC_1395_429,(2,0,0):C.R2GC_1395_430,(2,0,2):C.R2GC_1395_431,(1,0,1):C.R2GC_1395_429,(1,0,0):C.R2GC_1395_430,(1,0,2):C.R2GC_1395_431,(0,0,1):C.R2GC_1211_53,(0,0,0):C.R2GC_1211_54,(0,0,2):C.R2GC_1211_55})

V_246 = CTVertex(name = 'V_246',
                 type = 'R2',
                 particles = [ P.g, P.g, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.g] ], [ [P.go, P.u] ], [ [P.g, P.su4] ] ],
                 couplings = {(2,0,0):C.R2GC_1560_666,(2,0,2):C.R2GC_1560_667,(2,0,1):C.R2GC_1560_668,(1,0,0):C.R2GC_1560_666,(1,0,2):C.R2GC_1560_667,(1,0,1):C.R2GC_1560_668,(0,0,0):C.R2GC_1215_65,(0,0,2):C.R2GC_1215_66,(0,0,1):C.R2GC_1215_67})

V_247 = CTVertex(name = 'V_247',
                 type = 'R2',
                 particles = [ P.g, P.g, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g] ], [ [P.g, P.su5] ] ],
                 couplings = {(2,0,1):C.R2GC_1586_704,(2,0,0):C.R2GC_1586_705,(2,0,2):C.R2GC_1586_706,(1,0,1):C.R2GC_1586_704,(1,0,0):C.R2GC_1586_705,(1,0,2):C.R2GC_1586_706,(0,0,1):C.R2GC_1216_68,(0,0,0):C.R2GC_1216_69,(0,0,2):C.R2GC_1216_70})

V_248 = CTVertex(name = 'V_248',
                 type = 'R2',
                 particles = [ P.g, P.g, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.g] ], [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(2,0,0):C.R2GC_1743_924,(2,0,2):C.R2GC_1743_925,(2,0,1):C.R2GC_1743_926,(1,0,0):C.R2GC_1743_924,(1,0,2):C.R2GC_1743_925,(1,0,1):C.R2GC_1743_926,(0,0,0):C.R2GC_1217_71,(0,0,2):C.R2GC_1217_72,(0,0,1):C.R2GC_1217_73})

V_249 = CTVertex(name = 'V_249',
                 type = 'R2',
                 particles = [ P.g, P.g, P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g] ], [ [P.g, P.sd1] ] ],
                 couplings = {(2,0,1):C.R2GC_1260_253,(2,0,0):C.R2GC_1260_254,(2,0,2):C.R2GC_1260_255,(1,0,1):C.R2GC_1260_253,(1,0,0):C.R2GC_1260_254,(1,0,2):C.R2GC_1260_255,(0,0,1):C.R2GC_1206_38,(0,0,0):C.R2GC_1206_39,(0,0,2):C.R2GC_1206_40})

V_250 = CTVertex(name = 'V_250',
                 type = 'R2',
                 particles = [ P.g, P.g, P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.g] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ] ],
                 couplings = {(2,0,0):C.R2GC_1286_292,(2,0,2):C.R2GC_1286_293,(2,0,1):C.R2GC_1286_294,(1,0,0):C.R2GC_1286_292,(1,0,2):C.R2GC_1286_293,(1,0,1):C.R2GC_1286_294,(0,0,0):C.R2GC_1207_41,(0,0,2):C.R2GC_1207_42,(0,0,1):C.R2GC_1207_43})

V_251 = CTVertex(name = 'V_251',
                 type = 'R2',
                 particles = [ P.g, P.g, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g] ], [ [P.g, P.sd3] ] ],
                 couplings = {(2,0,1):C.R2GC_1304_326,(2,0,0):C.R2GC_1304_327,(2,0,2):C.R2GC_1304_328,(1,0,1):C.R2GC_1304_326,(1,0,0):C.R2GC_1304_327,(1,0,2):C.R2GC_1304_328,(0,0,1):C.R2GC_1208_44,(0,0,0):C.R2GC_1208_45,(0,0,2):C.R2GC_1208_46})

V_252 = CTVertex(name = 'V_252',
                 type = 'R2',
                 particles = [ P.g, P.g, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.g] ], [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(2,0,0):C.R2GC_1477_533,(2,0,2):C.R2GC_1477_534,(2,0,1):C.R2GC_1477_535,(1,0,0):C.R2GC_1477_533,(1,0,2):C.R2GC_1477_534,(1,0,1):C.R2GC_1477_535,(0,0,0):C.R2GC_1212_56,(0,0,2):C.R2GC_1212_57,(0,0,1):C.R2GC_1212_58})

V_253 = CTVertex(name = 'V_253',
                 type = 'R2',
                 particles = [ P.g, P.g, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g] ], [ [P.g, P.su2] ] ],
                 couplings = {(2,0,1):C.R2GC_1524_608,(2,0,0):C.R2GC_1524_609,(2,0,2):C.R2GC_1524_610,(1,0,1):C.R2GC_1524_608,(1,0,0):C.R2GC_1524_609,(1,0,2):C.R2GC_1524_610,(0,0,1):C.R2GC_1213_59,(0,0,0):C.R2GC_1213_60,(0,0,2):C.R2GC_1213_61})

V_254 = CTVertex(name = 'V_254',
                 type = 'R2',
                 particles = [ P.g, P.g, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.g] ], [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(2,0,0):C.R2GC_1660_833,(2,0,2):C.R2GC_1660_834,(2,0,1):C.R2GC_1660_835,(1,0,0):C.R2GC_1660_833,(1,0,2):C.R2GC_1660_834,(1,0,1):C.R2GC_1660_835,(0,0,0):C.R2GC_1214_62,(0,0,2):C.R2GC_1214_63,(0,0,1):C.R2GC_1214_64})

V_255 = CTVertex(name = 'V_255',
                 type = 'R2',
                 particles = [ P.h02, P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3, P.sd6] ] ],
                 couplings = {(0,0,0):C.R2GC_1929_1019,(0,0,1):C.R2GC_1929_1020})

V_256 = CTVertex(name = 'V_256',
                 type = 'R2',
                 particles = [ P.G0, P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3, P.sd6] ] ],
                 couplings = {(0,0,0):C.R2GC_2192_1061,(0,0,1):C.R2GC_2192_1062})

V_257 = CTVertex(name = 'V_257',
                 type = 'R2',
                 particles = [ P.h01, P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3, P.sd6] ] ],
                 couplings = {(0,0,0):C.R2GC_1924_1010,(0,0,1):C.R2GC_1924_1011})

V_258 = CTVertex(name = 'V_258',
                 type = 'R2',
                 particles = [ P.A0, P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3, P.sd6] ] ],
                 couplings = {(0,0,0):C.R2GC_2187_1056,(0,0,1):C.R2GC_2187_1057})

V_259 = CTVertex(name = 'V_259',
                 type = 'R2',
                 particles = [ P.G__plus__, P.sd6, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd6, P.su3] ] ],
                 couplings = {(0,0,0):C.R2GC_2213_1088,(0,0,1):C.R2GC_2213_1089})

V_260 = CTVertex(name = 'V_260',
                 type = 'R2',
                 particles = [ P.H__plus__, P.sd6, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd6, P.su3] ] ],
                 couplings = {(0,0,0):C.R2GC_2211_1084,(0,0,1):C.R2GC_2211_1085})

V_261 = CTVertex(name = 'V_261',
                 type = 'R2',
                 particles = [ P.H__minus__, P.sd3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_2218_1096,(0,0,1):C.R2GC_2218_1097})

V_262 = CTVertex(name = 'V_262',
                 type = 'R2',
                 particles = [ P.G__minus__, P.sd3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_2217_1094,(0,0,1):C.R2GC_2217_1095})

V_263 = CTVertex(name = 'V_263',
                 type = 'R2',
                 particles = [ P.h01, P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_1944_1040,(0,0,1):C.R2GC_1944_1041})

V_264 = CTVertex(name = 'V_264',
                 type = 'R2',
                 particles = [ P.A0, P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_2221_1102,(0,0,1):C.R2GC_2221_1103})

V_265 = CTVertex(name = 'V_265',
                 type = 'R2',
                 particles = [ P.h02, P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_1943_1038,(0,0,1):C.R2GC_1943_1039})

V_266 = CTVertex(name = 'V_266',
                 type = 'R2',
                 particles = [ P.G0, P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_2220_1100,(0,0,1):C.R2GC_2220_1101})

V_267 = CTVertex(name = 'V_267',
                 type = 'R2',
                 particles = [ P.x1__minus__, P.u, P.sd1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.sd1, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1457_499})

V_268 = CTVertex(name = 'V_268',
                 type = 'R2',
                 particles = [ P.x1__minus__, P.c, P.sd2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g, P.sd2] ] ],
                 couplings = {(0,0,0):C.R2GC_1503_572})

V_269 = CTVertex(name = 'V_269',
                 type = 'R2',
                 particles = [ P.x1__minus__, P.t, P.sd3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.go, P.su6] ], [ [P.g, P.sd3, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1636_788,(0,0,1):C.R2GC_1636_789,(0,1,1):C.R2GC_1628_776})

V_270 = CTVertex(name = 'V_270',
                 type = 'R2',
                 particles = [ P.x1__minus__, P.t, P.sd6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.b, P.go, P.su3] ], [ [P.g, P.sd6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1630_778,(0,0,1):C.R2GC_1630_779})

V_271 = CTVertex(name = 'V_271',
                 type = 'R2',
                 particles = [ P.x2__minus__, P.u, P.sd1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.sd1, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1458_500})

V_272 = CTVertex(name = 'V_272',
                 type = 'R2',
                 particles = [ P.x2__minus__, P.c, P.sd2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g, P.sd2] ] ],
                 couplings = {(0,0,0):C.R2GC_1504_573})

V_273 = CTVertex(name = 'V_273',
                 type = 'R2',
                 particles = [ P.x2__minus__, P.t, P.sd3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.go, P.su6] ], [ [P.g, P.sd3, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1637_790,(0,0,1):C.R2GC_1637_791,(0,1,1):C.R2GC_1629_777})

V_274 = CTVertex(name = 'V_274',
                 type = 'R2',
                 particles = [ P.x2__minus__, P.t, P.sd6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.b, P.go, P.su3] ], [ [P.g, P.sd6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1631_780,(0,0,1):C.R2GC_1631_781})

V_275 = CTVertex(name = 'V_275',
                 type = 'R2',
                 particles = [ P.d, P.x1__plus__, P.su1__tilde__ ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.g, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_1493_565})

V_276 = CTVertex(name = 'V_276',
                 type = 'R2',
                 particles = [ P.s, P.x1__plus__, P.su2__tilde__ ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.s, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1540_640})

V_277 = CTVertex(name = 'V_277',
                 type = 'R2',
                 particles = [ P.x1__plus__, P.b, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g, P.su3] ], [ [P.go, P.sd6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1692_887,(0,0,1):C.R2GC_1692_888,(0,1,0):C.R2GC_1686_877})

V_278 = CTVertex(name = 'V_278',
                 type = 'R2',
                 particles = [ P.x1__plus__, P.b, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g, P.su6] ], [ [P.go, P.sd3, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1753_945,(0,0,1):C.R2GC_1753_946})

V_279 = CTVertex(name = 'V_279',
                 type = 'R2',
                 particles = [ P.d, P.x2__plus__, P.su1__tilde__ ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.g, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_1494_566})

V_280 = CTVertex(name = 'V_280',
                 type = 'R2',
                 particles = [ P.s, P.x2__plus__, P.su2__tilde__ ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.g, P.s, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1541_641})

V_281 = CTVertex(name = 'V_281',
                 type = 'R2',
                 particles = [ P.x2__plus__, P.b, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g, P.su3] ], [ [P.go, P.sd6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1693_889,(0,0,1):C.R2GC_1693_890,(0,1,0):C.R2GC_1687_878})

V_282 = CTVertex(name = 'V_282',
                 type = 'R2',
                 particles = [ P.x2__plus__, P.b, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g, P.su6] ], [ [P.go, P.sd3, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1754_947,(0,0,1):C.R2GC_1754_948})

V_283 = CTVertex(name = 'V_283',
                 type = 'R2',
                 particles = [ P.W__minus__, P.sd1__tilde__, P.su1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.d, P.go, P.u] ], [ [P.g, P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_1468_513,(0,0,1):C.R2GC_1468_514,(0,1,0):C.R2GC_1467_511,(0,1,1):C.R2GC_1467_512})

V_284 = CTVertex(name = 'V_284',
                 type = 'R2',
                 particles = [ P.W__minus__, P.sd2__tilde__, P.su2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.c, P.go, P.s] ], [ [P.g, P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1514_586,(0,0,1):C.R2GC_1514_587,(0,1,0):C.R2GC_1513_584,(0,1,1):C.R2GC_1513_585})

V_285 = CTVertex(name = 'V_285',
                 type = 'R2',
                 particles = [ P.W__minus__, P.sd3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.R2GC_1650_811,(0,0,1):C.R2GC_1650_812,(0,1,0):C.R2GC_1649_809,(0,1,1):C.R2GC_1649_810})

V_286 = CTVertex(name = 'V_286',
                 type = 'R2',
                 particles = [ P.a, P.W__minus__, P.sd1__tilde__, P.su1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go, P.u] ], [ [P.g, P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_1469_515,(0,0,1):C.R2GC_1469_516})

V_287 = CTVertex(name = 'V_287',
                 type = 'R2',
                 particles = [ P.a, P.W__minus__, P.sd2__tilde__, P.su2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go, P.s] ], [ [P.g, P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1515_588,(0,0,1):C.R2GC_1515_589})

V_288 = CTVertex(name = 'V_288',
                 type = 'R2',
                 particles = [ P.a, P.W__minus__, P.sd3__tilde__, P.su3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.R2GC_1651_813,(0,0,1):C.R2GC_1651_814})

V_289 = CTVertex(name = 'V_289',
                 type = 'R2',
                 particles = [ P.g, P.W__minus__, P.sd1__tilde__, P.su1 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,1):C.R2GC_1470_517,(0,0,3):C.R2GC_1470_518,(0,0,0):C.R2GC_1470_519,(0,0,2):C.R2GC_1470_520})

V_290 = CTVertex(name = 'V_290',
                 type = 'R2',
                 particles = [ P.g, P.W__minus__, P.sd2__tilde__, P.su2 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,1):C.R2GC_1516_590,(0,0,3):C.R2GC_1516_591,(0,0,0):C.R2GC_1516_592,(0,0,2):C.R2GC_1516_593})

V_291 = CTVertex(name = 'V_291',
                 type = 'R2',
                 particles = [ P.g, P.W__minus__, P.sd3__tilde__, P.su3 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.R2GC_1652_815,(0,0,3):C.R2GC_1652_816,(0,0,0):C.R2GC_1652_817,(0,0,2):C.R2GC_1652_818})

V_292 = CTVertex(name = 'V_292',
                 type = 'R2',
                 particles = [ P.W__plus__, P.sd1, P.su1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.d, P.go, P.u] ], [ [P.g, P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_1488_553,(0,0,1):C.R2GC_1488_554,(0,1,0):C.R2GC_1489_555,(0,1,1):C.R2GC_1489_556})

V_293 = CTVertex(name = 'V_293',
                 type = 'R2',
                 particles = [ P.W__plus__, P.sd2, P.su2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.c, P.go, P.s] ], [ [P.g, P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1535_628,(0,0,1):C.R2GC_1535_629,(0,1,0):C.R2GC_1536_630,(0,1,1):C.R2GC_1536_631})

V_294 = CTVertex(name = 'V_294',
                 type = 'R2',
                 particles = [ P.W__plus__, P.sd3, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.R2GC_1667_849,(0,0,1):C.R2GC_1667_850,(0,1,0):C.R2GC_1668_851,(0,1,1):C.R2GC_1668_852})

V_295 = CTVertex(name = 'V_295',
                 type = 'R2',
                 particles = [ P.a, P.W__plus__, P.sd1, P.su1__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go, P.u] ], [ [P.g, P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_1490_557,(0,0,1):C.R2GC_1490_558})

V_296 = CTVertex(name = 'V_296',
                 type = 'R2',
                 particles = [ P.a, P.W__plus__, P.sd2, P.su2__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go, P.s] ], [ [P.g, P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1537_632,(0,0,1):C.R2GC_1537_633})

V_297 = CTVertex(name = 'V_297',
                 type = 'R2',
                 particles = [ P.a, P.W__plus__, P.sd3, P.su3__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.R2GC_1669_853,(0,0,1):C.R2GC_1669_854})

V_298 = CTVertex(name = 'V_298',
                 type = 'R2',
                 particles = [ P.g, P.W__plus__, P.sd1, P.su1__tilde__ ],
                 color = [ 'T(1,3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,1):C.R2GC_1491_559,(0,0,3):C.R2GC_1491_560,(0,0,0):C.R2GC_1491_561,(0,0,2):C.R2GC_1491_562})

V_299 = CTVertex(name = 'V_299',
                 type = 'R2',
                 particles = [ P.g, P.W__plus__, P.sd2, P.su2__tilde__ ],
                 color = [ 'T(1,3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,1):C.R2GC_1538_634,(0,0,3):C.R2GC_1538_635,(0,0,0):C.R2GC_1538_636,(0,0,2):C.R2GC_1538_637})

V_300 = CTVertex(name = 'V_300',
                 type = 'R2',
                 particles = [ P.g, P.W__plus__, P.sd3, P.su3__tilde__ ],
                 color = [ 'T(1,3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.R2GC_1670_855,(0,0,3):C.R2GC_1670_856,(0,0,0):C.R2GC_1670_857,(0,0,2):C.R2GC_1670_858})

V_301 = CTVertex(name = 'V_301',
                 type = 'R2',
                 particles = [ P.W__minus__, P.W__plus__, P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ] ],
                 couplings = {(0,0,1):C.R2GC_1261_256,(0,0,0):C.R2GC_1261_257,(0,0,2):C.R2GC_1261_258})

V_302 = CTVertex(name = 'V_302',
                 type = 'R2',
                 particles = [ P.W__minus__, P.W__plus__, P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ] ],
                 couplings = {(0,0,1):C.R2GC_1287_295,(0,0,0):C.R2GC_1287_296,(0,0,2):C.R2GC_1287_297})

V_303 = CTVertex(name = 'V_303',
                 type = 'R2',
                 particles = [ P.W__minus__, P.W__plus__, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ] ],
                 couplings = {(0,0,1):C.R2GC_1305_329,(0,0,0):C.R2GC_1305_330,(0,0,2):C.R2GC_1305_331})

V_304 = CTVertex(name = 'V_304',
                 type = 'R2',
                 particles = [ P.W__minus__, P.W__plus__, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go, P.u] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,2):C.R2GC_1478_536,(0,0,0):C.R2GC_1478_537,(0,0,1):C.R2GC_1478_538})

V_305 = CTVertex(name = 'V_305',
                 type = 'R2',
                 particles = [ P.W__minus__, P.W__plus__, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go, P.s] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,2):C.R2GC_1525_611,(0,0,0):C.R2GC_1525_612,(0,0,1):C.R2GC_1525_613})

V_306 = CTVertex(name = 'V_306',
                 type = 'R2',
                 particles = [ P.W__minus__, P.W__plus__, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,2):C.R2GC_1661_836,(0,0,0):C.R2GC_1661_837,(0,0,1):C.R2GC_1661_838})

V_307 = CTVertex(name = 'V_307',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.x1__plus__, P.sd6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.b, P.go, P.su3] ], [ [P.g, P.sd6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1633_783,(0,0,1):C.R2GC_1633_784})

V_308 = CTVertex(name = 'V_308',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.x2__plus__, P.sd6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.b, P.go, P.su3] ], [ [P.g, P.sd6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1635_786,(0,0,1):C.R2GC_1635_787})

V_309 = CTVertex(name = 'V_309',
                 type = 'R2',
                 particles = [ P.H__plus__, P.sd6, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd6, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_2224_1106,(0,0,1):C.R2GC_2224_1107})

V_310 = CTVertex(name = 'V_310',
                 type = 'R2',
                 particles = [ P.G__plus__, P.sd6, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd6, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_2226_1110,(0,0,1):C.R2GC_2226_1111})

V_311 = CTVertex(name = 'V_311',
                 type = 'R2',
                 particles = [ P.x1__minus__, P.b__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.b, P.g, P.su6] ], [ [P.go, P.sd3, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1761_953,(0,0,1):C.R2GC_1761_954})

V_312 = CTVertex(name = 'V_312',
                 type = 'R2',
                 particles = [ P.x2__minus__, P.b__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.b, P.g, P.su6] ], [ [P.go, P.sd3, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1762_955,(0,0,1):C.R2GC_1762_956})

V_313 = CTVertex(name = 'V_313',
                 type = 'R2',
                 particles = [ P.H__minus__, P.sd6__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd6, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_2227_1112,(0,0,1):C.R2GC_2227_1113})

V_314 = CTVertex(name = 'V_314',
                 type = 'R2',
                 particles = [ P.G__minus__, P.sd6__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd6, P.su6] ] ],
                 couplings = {(0,0,0):C.R2GC_2219_1098,(0,0,1):C.R2GC_2219_1099})

V_315 = CTVertex(name = 'V_315',
                 type = 'R2',
                 particles = [ P.Z, P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.R2GC_1266_263,(0,0,1):C.R2GC_1266_264,(0,1,0):C.R2GC_1267_265,(0,1,1):C.R2GC_1267_266})

V_316 = CTVertex(name = 'V_316',
                 type = 'R2',
                 particles = [ P.a, P.Z, P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.R2GC_1268_267,(0,0,1):C.R2GC_1268_268})

V_317 = CTVertex(name = 'V_317',
                 type = 'R2',
                 particles = [ P.Z, P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ] ],
                 couplings = {(0,0,1):C.R2GC_1292_302,(0,0,0):C.R2GC_1292_303,(0,1,1):C.R2GC_1293_304,(0,1,0):C.R2GC_1293_305})

V_318 = CTVertex(name = 'V_318',
                 type = 'R2',
                 particles = [ P.a, P.Z, P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ] ],
                 couplings = {(0,0,1):C.R2GC_1294_306,(0,0,0):C.R2GC_1294_307})

V_319 = CTVertex(name = 'V_319',
                 type = 'R2',
                 particles = [ P.Z, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,0,0):C.R2GC_1309_338,(0,0,1):C.R2GC_1309_339,(0,1,0):C.R2GC_1308_336,(0,1,1):C.R2GC_1308_337})

V_320 = CTVertex(name = 'V_320',
                 type = 'R2',
                 particles = [ P.a, P.Z, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,0,0):C.R2GC_1310_340,(0,0,1):C.R2GC_1310_341})

V_321 = CTVertex(name = 'V_321',
                 type = 'R2',
                 particles = [ P.Z, P.sd4__tilde__, P.sd4 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.R2GC_1330_367,(0,0,1):C.R2GC_1330_368,(0,1,0):C.R2GC_1331_369,(0,1,1):C.R2GC_1331_370})

V_322 = CTVertex(name = 'V_322',
                 type = 'R2',
                 particles = [ P.a, P.Z, P.sd4__tilde__, P.sd4 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.R2GC_1332_371,(0,0,1):C.R2GC_1332_372})

V_323 = CTVertex(name = 'V_323',
                 type = 'R2',
                 particles = [ P.Z, P.sd5__tilde__, P.sd5 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ] ],
                 couplings = {(0,0,1):C.R2GC_1355_403,(0,0,0):C.R2GC_1355_404,(0,1,1):C.R2GC_1356_405,(0,1,0):C.R2GC_1356_406})

V_324 = CTVertex(name = 'V_324',
                 type = 'R2',
                 particles = [ P.a, P.Z, P.sd5__tilde__, P.sd5 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ] ],
                 couplings = {(0,0,1):C.R2GC_1357_407,(0,0,0):C.R2GC_1357_408})

V_325 = CTVertex(name = 'V_325',
                 type = 'R2',
                 particles = [ P.Z, P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,0,0):C.R2GC_1397_434,(0,0,1):C.R2GC_1397_435,(0,1,0):C.R2GC_1398_436,(0,1,1):C.R2GC_1398_437})

V_326 = CTVertex(name = 'V_326',
                 type = 'R2',
                 particles = [ P.a, P.Z, P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,0,0):C.R2GC_1399_438,(0,0,1):C.R2GC_1399_439})

V_327 = CTVertex(name = 'V_327',
                 type = 'R2',
                 particles = [ P.Z, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,1):C.R2GC_1484_545,(0,0,0):C.R2GC_1484_546,(0,1,1):C.R2GC_1483_543,(0,1,0):C.R2GC_1483_544})

V_328 = CTVertex(name = 'V_328',
                 type = 'R2',
                 particles = [ P.a, P.Z, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,1):C.R2GC_1485_547,(0,0,0):C.R2GC_1485_548})

V_329 = CTVertex(name = 'V_329',
                 type = 'R2',
                 particles = [ P.Z, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1531_620,(0,0,1):C.R2GC_1531_621,(0,1,0):C.R2GC_1530_618,(0,1,1):C.R2GC_1530_619})

V_330 = CTVertex(name = 'V_330',
                 type = 'R2',
                 particles = [ P.a, P.Z, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1532_622,(0,0,1):C.R2GC_1532_623})

V_331 = CTVertex(name = 'V_331',
                 type = 'R2',
                 particles = [ P.Z, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.R2GC_1666_847,(0,0,0):C.R2GC_1666_848,(0,1,1):C.R2GC_1665_845,(0,1,0):C.R2GC_1665_846})

V_332 = CTVertex(name = 'V_332',
                 type = 'R2',
                 particles = [ P.a, P.Z, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.R2GC_1662_839,(0,0,0):C.R2GC_1662_840})

V_333 = CTVertex(name = 'V_333',
                 type = 'R2',
                 particles = [ P.Z, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ] ],
                 couplings = {(0,0,1):C.R2GC_1563_673,(0,0,0):C.R2GC_1563_674,(0,1,1):C.R2GC_1562_671,(0,1,0):C.R2GC_1562_672})

V_334 = CTVertex(name = 'V_334',
                 type = 'R2',
                 particles = [ P.a, P.Z, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ] ],
                 couplings = {(0,0,1):C.R2GC_1564_675,(0,0,0):C.R2GC_1564_676})

V_335 = CTVertex(name = 'V_335',
                 type = 'R2',
                 particles = [ P.Z, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1589_711,(0,0,1):C.R2GC_1589_712,(0,1,0):C.R2GC_1588_709,(0,1,1):C.R2GC_1588_710})

V_336 = CTVertex(name = 'V_336',
                 type = 'R2',
                 particles = [ P.a, P.Z, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1590_713,(0,0,1):C.R2GC_1590_714})

V_337 = CTVertex(name = 'V_337',
                 type = 'R2',
                 particles = [ P.Z, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(0,0,1):C.R2GC_1746_931,(0,0,0):C.R2GC_1746_932,(0,1,1):C.R2GC_1745_929,(0,1,0):C.R2GC_1745_930})

V_338 = CTVertex(name = 'V_338',
                 type = 'R2',
                 particles = [ P.a, P.Z, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(0,0,1):C.R2GC_1747_933,(0,0,0):C.R2GC_1747_934})

V_339 = CTVertex(name = 'V_339',
                 type = 'R2',
                 particles = [ P.g, P.Z, P.sd1__tilde__, P.sd1 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.R2GC_1269_269,(0,0,1):C.R2GC_1269_270})

V_340 = CTVertex(name = 'V_340',
                 type = 'R2',
                 particles = [ P.g, P.Z, P.sd2__tilde__, P.sd2 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ] ],
                 couplings = {(0,0,1):C.R2GC_1295_308,(0,0,0):C.R2GC_1295_309})

V_341 = CTVertex(name = 'V_341',
                 type = 'R2',
                 particles = [ P.g, P.Z, P.sd3__tilde__, P.sd3 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,0,0):C.R2GC_1306_332,(0,0,1):C.R2GC_1306_333})

V_342 = CTVertex(name = 'V_342',
                 type = 'R2',
                 particles = [ P.g, P.Z, P.sd4__tilde__, P.sd4 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.R2GC_1333_373,(0,0,1):C.R2GC_1333_374})

V_343 = CTVertex(name = 'V_343',
                 type = 'R2',
                 particles = [ P.g, P.Z, P.sd5__tilde__, P.sd5 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ] ],
                 couplings = {(0,0,1):C.R2GC_1358_409,(0,0,0):C.R2GC_1358_410})

V_344 = CTVertex(name = 'V_344',
                 type = 'R2',
                 particles = [ P.g, P.Z, P.sd6__tilde__, P.sd6 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,0,0):C.R2GC_1400_440,(0,0,1):C.R2GC_1400_441})

V_345 = CTVertex(name = 'V_345',
                 type = 'R2',
                 particles = [ P.g, P.Z, P.su1__tilde__, P.su1 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,1):C.R2GC_1486_549,(0,0,0):C.R2GC_1486_550})

V_346 = CTVertex(name = 'V_346',
                 type = 'R2',
                 particles = [ P.g, P.Z, P.su2__tilde__, P.su2 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1533_624,(0,0,1):C.R2GC_1533_625})

V_347 = CTVertex(name = 'V_347',
                 type = 'R2',
                 particles = [ P.g, P.Z, P.su3__tilde__, P.su3 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.R2GC_1663_841,(0,0,0):C.R2GC_1663_842})

V_348 = CTVertex(name = 'V_348',
                 type = 'R2',
                 particles = [ P.g, P.Z, P.su4__tilde__, P.su4 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ] ],
                 couplings = {(0,0,1):C.R2GC_1565_677,(0,0,0):C.R2GC_1565_678})

V_349 = CTVertex(name = 'V_349',
                 type = 'R2',
                 particles = [ P.g, P.Z, P.su5__tilde__, P.su5 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1591_715,(0,0,1):C.R2GC_1591_716})

V_350 = CTVertex(name = 'V_350',
                 type = 'R2',
                 particles = [ P.g, P.Z, P.su6__tilde__, P.su6 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(0,0,1):C.R2GC_1748_935,(0,0,0):C.R2GC_1748_936})

V_351 = CTVertex(name = 'V_351',
                 type = 'R2',
                 particles = [ P.W__minus__, P.Z, P.sd1__tilde__, P.su1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go, P.u] ], [ [P.g, P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_1471_521,(0,0,1):C.R2GC_1471_522})

V_352 = CTVertex(name = 'V_352',
                 type = 'R2',
                 particles = [ P.W__minus__, P.Z, P.sd2__tilde__, P.su2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go, P.s] ], [ [P.g, P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1517_594,(0,0,1):C.R2GC_1517_595})

V_353 = CTVertex(name = 'V_353',
                 type = 'R2',
                 particles = [ P.W__minus__, P.Z, P.sd3__tilde__, P.su3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.R2GC_1653_819,(0,0,1):C.R2GC_1653_820})

V_354 = CTVertex(name = 'V_354',
                 type = 'R2',
                 particles = [ P.W__plus__, P.Z, P.sd1, P.su1__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go, P.u] ], [ [P.g, P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.R2GC_1492_563,(0,0,1):C.R2GC_1492_564})

V_355 = CTVertex(name = 'V_355',
                 type = 'R2',
                 particles = [ P.W__plus__, P.Z, P.sd2, P.su2__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go, P.s] ], [ [P.g, P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1539_638,(0,0,1):C.R2GC_1539_639})

V_356 = CTVertex(name = 'V_356',
                 type = 'R2',
                 particles = [ P.W__plus__, P.Z, P.sd3, P.su3__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.g, P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.R2GC_1671_859,(0,0,1):C.R2GC_1671_860})

V_357 = CTVertex(name = 'V_357',
                 type = 'R2',
                 particles = [ P.Z, P.Z, P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.R2GC_1270_271,(0,0,1):C.R2GC_1270_272})

V_358 = CTVertex(name = 'V_358',
                 type = 'R2',
                 particles = [ P.Z, P.Z, P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ] ],
                 couplings = {(0,0,1):C.R2GC_1296_310,(0,0,0):C.R2GC_1296_311})

V_359 = CTVertex(name = 'V_359',
                 type = 'R2',
                 particles = [ P.Z, P.Z, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,0,0):C.R2GC_1307_334,(0,0,1):C.R2GC_1307_335})

V_360 = CTVertex(name = 'V_360',
                 type = 'R2',
                 particles = [ P.Z, P.Z, P.sd4__tilde__, P.sd4 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.R2GC_1329_365,(0,0,1):C.R2GC_1329_366})

V_361 = CTVertex(name = 'V_361',
                 type = 'R2',
                 particles = [ P.Z, P.Z, P.sd5__tilde__, P.sd5 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ] ],
                 couplings = {(0,0,1):C.R2GC_1354_401,(0,0,0):C.R2GC_1354_402})

V_362 = CTVertex(name = 'V_362',
                 type = 'R2',
                 particles = [ P.Z, P.Z, P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,0,0):C.R2GC_1396_432,(0,0,1):C.R2GC_1396_433})

V_363 = CTVertex(name = 'V_363',
                 type = 'R2',
                 particles = [ P.Z, P.Z, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,1):C.R2GC_1487_551,(0,0,0):C.R2GC_1487_552})

V_364 = CTVertex(name = 'V_364',
                 type = 'R2',
                 particles = [ P.Z, P.Z, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1534_626,(0,0,1):C.R2GC_1534_627})

V_365 = CTVertex(name = 'V_365',
                 type = 'R2',
                 particles = [ P.Z, P.Z, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.R2GC_1664_843,(0,0,0):C.R2GC_1664_844})

V_366 = CTVertex(name = 'V_366',
                 type = 'R2',
                 particles = [ P.Z, P.Z, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ] ],
                 couplings = {(0,0,1):C.R2GC_1561_669,(0,0,0):C.R2GC_1561_670})

V_367 = CTVertex(name = 'V_367',
                 type = 'R2',
                 particles = [ P.Z, P.Z, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1587_707,(0,0,1):C.R2GC_1587_708})

V_368 = CTVertex(name = 'V_368',
                 type = 'R2',
                 particles = [ P.Z, P.Z, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(0,0,1):C.R2GC_1744_927,(0,0,0):C.R2GC_1744_928})

V_369 = CTVertex(name = 'V_369',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1198_34})

V_370 = CTVertex(name = 'V_370',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1201_37})

V_371 = CTVertex(name = 'V_371',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1198_34})

V_372 = CTVertex(name = 'V_372',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_1198_34})

V_373 = CTVertex(name = 'V_373',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1201_37})

V_374 = CTVertex(name = 'V_374',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1201_37})

V_375 = CTVertex(name = 'V_375',
                 type = 'R2',
                 particles = [ P.go, P.go, P.g ],
                 color = [ 'd(2,1,3)', 'f(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV2, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.g, P.go] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(1,0,6):C.R2GC_1764_958,(1,2,0):C.R2GC_1232_91,(1,2,1):C.R2GC_1232_92,(1,2,2):C.R2GC_1232_93,(1,2,3):C.R2GC_1232_94,(1,2,4):C.R2GC_1232_95,(1,2,5):C.R2GC_1232_96,(1,2,7):C.R2GC_1232_97,(1,2,8):C.R2GC_1232_98,(1,2,9):C.R2GC_1232_99,(1,2,10):C.R2GC_1232_100,(1,2,11):C.R2GC_1233_102,(1,2,12):C.R2GC_1232_101,(1,3,0):C.R2GC_1232_91,(1,3,1):C.R2GC_1232_92,(1,3,2):C.R2GC_1232_93,(1,3,3):C.R2GC_1232_94,(1,3,4):C.R2GC_1232_95,(1,3,5):C.R2GC_1232_96,(1,3,7):C.R2GC_1232_97,(1,3,8):C.R2GC_1232_98,(1,3,9):C.R2GC_1232_99,(1,3,10):C.R2GC_1232_100,(1,3,12):C.R2GC_1232_101,(0,1,11):C.R2GC_1229_78,(1,1,11):C.R2GC_1228_77,(0,2,0):C.R2GC_1235_114,(0,2,1):C.R2GC_1235_115,(0,2,2):C.R2GC_1235_116,(0,2,3):C.R2GC_1235_117,(0,2,4):C.R2GC_1235_118,(0,2,5):C.R2GC_1235_119,(0,2,7):C.R2GC_1235_120,(0,2,8):C.R2GC_1235_121,(0,2,9):C.R2GC_1235_122,(0,2,10):C.R2GC_1235_123,(0,2,12):C.R2GC_1235_124,(0,3,0):C.R2GC_1234_103,(0,3,1):C.R2GC_1234_104,(0,3,2):C.R2GC_1234_105,(0,3,3):C.R2GC_1234_106,(0,3,4):C.R2GC_1234_107,(0,3,5):C.R2GC_1234_108,(0,3,7):C.R2GC_1234_109,(0,3,8):C.R2GC_1234_110,(0,3,9):C.R2GC_1234_111,(0,3,10):C.R2GC_1234_112,(0,3,12):C.R2GC_1234_113})

V_376 = CTVertex(name = 'V_376',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ] ],
                 couplings = {(0,0,0):C.R2GC_1199_35,(0,1,1):C.R2GC_1413_443,(0,2,2):C.R2GC_1410_442})

V_377 = CTVertex(name = 'V_377',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.go, P.su2] ], [ [P.go, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1199_35,(0,1,1):C.R2GC_1499_569,(0,2,2):C.R2GC_1572_683})

V_378 = CTVertex(name = 'V_378',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.go, P.sd1] ], [ [P.go, P.sd4] ] ],
                 couplings = {(0,0,0):C.R2GC_1199_35,(0,1,1):C.R2GC_1247_233,(0,2,2):C.R2GC_1315_342})

V_379 = CTVertex(name = 'V_379',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.go, P.sd2] ], [ [P.go, P.sd5] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,2):C.R2GC_1199_35,(0,1,0):C.R2GC_1273_273,(0,2,1):C.R2GC_1340_379})

V_380 = CTVertex(name = 'V_380',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.R2GC_1199_35,(0,1,0):C.R2GC_1617_767,(0,2,1):C.R2GC_1621_769})

V_381 = CTVertex(name = 'V_381',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.go, P.su1] ], [ [P.go, P.su4] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.R2GC_1199_35,(0,1,0):C.R2GC_1453_495,(0,2,1):C.R2GC_1546_644})

V_382 = CTVertex(name = 'V_382',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_1500_570})

V_383 = CTVertex(name = 'V_383',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1622_770})

V_384 = CTVertex(name = 'V_384',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1454_496})

V_385 = CTVertex(name = 'V_385',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1625_771})

V_386 = CTVertex(name = 'V_386',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1456_498})

V_387 = CTVertex(name = 'V_387',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_1502_571})

V_388 = CTVertex(name = 'V_388',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1248_234,(0,1,0):C.R2GC_1316_343})

V_389 = CTVertex(name = 'V_389',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1455_497,(0,1,0):C.R2GC_1547_645})

V_390 = CTVertex(name = 'V_390',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1248_234,(0,1,0):C.R2GC_1316_343})

V_391 = CTVertex(name = 'V_391',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_1248_234,(0,1,0):C.R2GC_1316_343})

V_392 = CTVertex(name = 'V_392',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1455_497,(0,1,0):C.R2GC_1547_645})

V_393 = CTVertex(name = 'V_393',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1455_497,(0,1,0):C.R2GC_1547_645})

V_394 = CTVertex(name = 'V_394',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1414_444,(0,2,0):C.R2GC_1414_444,(0,1,0):C.R2GC_1200_36,(0,3,0):C.R2GC_1200_36})

V_395 = CTVertex(name = 'V_395',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1200_36})

V_396 = CTVertex(name = 'V_396',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1200_36})

V_397 = CTVertex(name = 'V_397',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_1200_36})

V_398 = CTVertex(name = 'V_398',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1618_768,(0,2,0):C.R2GC_1618_768,(0,1,0):C.R2GC_1200_36,(0,3,0):C.R2GC_1200_36})

V_399 = CTVertex(name = 'V_399',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1200_36})

V_400 = CTVertex(name = 'V_400',
                 type = 'R2',
                 particles = [ P.go, P.go ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5 ],
                 loop_particles = [ [ [P.g, P.go] ] ],
                 couplings = {(0,0,0):C.R2GC_1765_959,(0,2,0):C.R2GC_1765_959,(0,1,0):C.R2GC_1763_957,(0,3,0):C.R2GC_1763_957})

V_401 = CTVertex(name = 'V_401',
                 type = 'R2',
                 particles = [ P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,1):C.R2GC_1462_505,(0,0,0):C.R2GC_1462_506,(0,1,1):C.R2GC_1461_503,(0,1,0):C.R2GC_1461_504})

V_402 = CTVertex(name = 'V_402',
                 type = 'R2',
                 particles = [ P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.R2GC_1508_578,(0,0,1):C.R2GC_1508_579,(0,1,0):C.R2GC_1507_576,(0,1,1):C.R2GC_1507_577})

V_403 = CTVertex(name = 'V_403',
                 type = 'R2',
                 particles = [ P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.R2GC_1645_800,(0,0,0):C.R2GC_1645_801,(0,1,1):C.R2GC_1644_798,(0,1,0):C.R2GC_1644_799})

V_404 = CTVertex(name = 'V_404',
                 type = 'R2',
                 particles = [ P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ] ],
                 couplings = {(0,0,1):C.R2GC_1549_648,(0,0,0):C.R2GC_1549_649,(0,1,1):C.R2GC_1548_646,(0,1,0):C.R2GC_1548_647})

V_405 = CTVertex(name = 'V_405',
                 type = 'R2',
                 particles = [ P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.R2GC_1575_686,(0,0,1):C.R2GC_1575_687,(0,1,0):C.R2GC_1574_684,(0,1,1):C.R2GC_1574_685})

V_406 = CTVertex(name = 'V_406',
                 type = 'R2',
                 particles = [ P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(0,0,1):C.R2GC_1701_895,(0,0,0):C.R2GC_1701_896,(0,1,1):C.R2GC_1700_893,(0,1,0):C.R2GC_1700_894})

V_407 = CTVertex(name = 'V_407',
                 type = 'R2',
                 particles = [ P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.R2GC_1250_237,(0,0,1):C.R2GC_1250_238,(0,1,0):C.R2GC_1249_235,(0,1,1):C.R2GC_1249_236})

V_408 = CTVertex(name = 'V_408',
                 type = 'R2',
                 particles = [ P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ] ],
                 couplings = {(0,0,1):C.R2GC_1276_276,(0,0,0):C.R2GC_1276_277,(0,1,1):C.R2GC_1275_274,(0,1,0):C.R2GC_1275_275})

V_409 = CTVertex(name = 'V_409',
                 type = 'R2',
                 particles = [ P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,0,0):C.R2GC_1298_314,(0,0,1):C.R2GC_1298_315,(0,1,0):C.R2GC_1297_312,(0,1,1):C.R2GC_1297_313})

V_410 = CTVertex(name = 'V_410',
                 type = 'R2',
                 particles = [ P.sd4__tilde__, P.sd4 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.R2GC_1318_346,(0,0,1):C.R2GC_1318_347,(0,1,0):C.R2GC_1317_344,(0,1,1):C.R2GC_1317_345})

V_411 = CTVertex(name = 'V_411',
                 type = 'R2',
                 particles = [ P.sd5__tilde__, P.sd5 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ] ],
                 couplings = {(0,0,1):C.R2GC_1343_382,(0,0,0):C.R2GC_1343_383,(0,1,1):C.R2GC_1342_380,(0,1,0):C.R2GC_1342_381})

V_412 = CTVertex(name = 'V_412',
                 type = 'R2',
                 particles = [ P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,0,0):C.R2GC_1368_417,(0,0,1):C.R2GC_1368_418,(0,1,0):C.R2GC_1367_415,(0,1,1):C.R2GC_1367_416})

V_413 = CTVertex(name = 'V_413',
                 type = 'R2',
                 particles = [ P.g, P.g ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VV1, L.VV2, L.VV3 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.g] ], [ [P.go] ], [ [P.t] ] ],
                 couplings = {(0,0,2):C.R2GC_1597_724,(0,1,0):C.R2GC_1177_5,(0,1,3):C.R2GC_1177_6,(0,1,4):C.R2GC_1177_7,(0,2,1):C.R2GC_1596_721,(0,2,2):C.R2GC_1596_722,(0,2,3):C.R2GC_1596_723})

V_414 = CTVertex(name = 'V_414',
                 type = 'R2',
                 particles = [ P.go, P.go, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV2 ],
                 loop_particles = [ [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1231_79,(0,0,1):C.R2GC_1231_80,(0,0,2):C.R2GC_1231_81,(0,0,3):C.R2GC_1231_82,(0,0,4):C.R2GC_1231_83,(0,0,5):C.R2GC_1231_84,(0,0,6):C.R2GC_1231_85,(0,0,7):C.R2GC_1231_86,(0,0,8):C.R2GC_1231_87,(0,0,9):C.R2GC_1231_88,(0,0,10):C.R2GC_1231_89,(0,0,11):C.R2GC_1231_90})

V_415 = CTVertex(name = 'V_415',
                 type = 'R2',
                 particles = [ P.go, P.go, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV2 ],
                 loop_particles = [ [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1240_173,(0,0,1):C.R2GC_1240_174,(0,0,2):C.R2GC_1240_175,(0,0,3):C.R2GC_1240_176,(0,0,4):C.R2GC_1240_177,(0,0,5):C.R2GC_1240_178,(0,0,6):C.R2GC_1240_179,(0,0,7):C.R2GC_1240_180,(0,0,8):C.R2GC_1240_181,(0,0,9):C.R2GC_1240_182,(0,0,10):C.R2GC_1240_183,(0,0,11):C.R2GC_1240_184})

V_416 = CTVertex(name = 'V_416',
                 type = 'R2',
                 particles = [ P.go, P.n1, P.g ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1236_125,(0,0,1):C.R2GC_1236_126,(0,0,2):C.R2GC_1236_127,(0,0,3):C.R2GC_1236_128,(0,0,4):C.R2GC_1236_129,(0,0,5):C.R2GC_1236_130,(0,0,6):C.R2GC_1236_131,(0,0,7):C.R2GC_1236_132,(0,0,8):C.R2GC_1236_133,(0,0,9):C.R2GC_1236_134,(0,0,10):C.R2GC_1236_135,(0,0,11):C.R2GC_1236_136,(0,1,0):C.R2GC_1241_185,(0,1,1):C.R2GC_1241_186,(0,1,2):C.R2GC_1241_187,(0,1,3):C.R2GC_1241_188,(0,1,4):C.R2GC_1241_189,(0,1,5):C.R2GC_1241_190,(0,1,6):C.R2GC_1241_191,(0,1,7):C.R2GC_1241_192,(0,1,8):C.R2GC_1241_193,(0,1,9):C.R2GC_1241_194,(0,1,10):C.R2GC_1241_195,(0,1,11):C.R2GC_1241_196})

V_417 = CTVertex(name = 'V_417',
                 type = 'R2',
                 particles = [ P.go, P.n2, P.g ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1237_137,(0,0,1):C.R2GC_1237_138,(0,0,2):C.R2GC_1237_139,(0,0,3):C.R2GC_1237_140,(0,0,4):C.R2GC_1237_141,(0,0,5):C.R2GC_1237_142,(0,0,6):C.R2GC_1237_143,(0,0,7):C.R2GC_1237_144,(0,0,8):C.R2GC_1237_145,(0,0,9):C.R2GC_1237_146,(0,0,10):C.R2GC_1237_147,(0,0,11):C.R2GC_1237_148,(0,1,0):C.R2GC_1242_197,(0,1,1):C.R2GC_1242_198,(0,1,2):C.R2GC_1242_199,(0,1,3):C.R2GC_1242_200,(0,1,4):C.R2GC_1242_201,(0,1,5):C.R2GC_1242_202,(0,1,6):C.R2GC_1242_203,(0,1,7):C.R2GC_1242_204,(0,1,8):C.R2GC_1242_205,(0,1,9):C.R2GC_1242_206,(0,1,10):C.R2GC_1242_207,(0,1,11):C.R2GC_1242_208})

V_418 = CTVertex(name = 'V_418',
                 type = 'R2',
                 particles = [ P.go, P.n3, P.g ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1238_149,(0,0,1):C.R2GC_1238_150,(0,0,2):C.R2GC_1238_151,(0,0,3):C.R2GC_1238_152,(0,0,4):C.R2GC_1238_153,(0,0,5):C.R2GC_1238_154,(0,0,6):C.R2GC_1238_155,(0,0,7):C.R2GC_1238_156,(0,0,8):C.R2GC_1238_157,(0,0,9):C.R2GC_1238_158,(0,0,10):C.R2GC_1238_159,(0,0,11):C.R2GC_1238_160,(0,1,0):C.R2GC_1243_209,(0,1,1):C.R2GC_1243_210,(0,1,2):C.R2GC_1243_211,(0,1,3):C.R2GC_1243_212,(0,1,4):C.R2GC_1243_213,(0,1,5):C.R2GC_1243_214,(0,1,6):C.R2GC_1243_215,(0,1,7):C.R2GC_1243_216,(0,1,8):C.R2GC_1243_217,(0,1,9):C.R2GC_1243_218,(0,1,10):C.R2GC_1243_219,(0,1,11):C.R2GC_1243_220})

V_419 = CTVertex(name = 'V_419',
                 type = 'R2',
                 particles = [ P.go, P.n4, P.g ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1239_161,(0,0,1):C.R2GC_1239_162,(0,0,2):C.R2GC_1239_163,(0,0,3):C.R2GC_1239_164,(0,0,4):C.R2GC_1239_165,(0,0,5):C.R2GC_1239_166,(0,0,6):C.R2GC_1239_167,(0,0,7):C.R2GC_1239_168,(0,0,8):C.R2GC_1239_169,(0,0,9):C.R2GC_1239_170,(0,0,10):C.R2GC_1239_171,(0,0,11):C.R2GC_1239_172,(0,1,0):C.R2GC_1244_221,(0,1,1):C.R2GC_1244_222,(0,1,2):C.R2GC_1244_223,(0,1,3):C.R2GC_1244_224,(0,1,4):C.R2GC_1244_225,(0,1,5):C.R2GC_1244_226,(0,1,6):C.R2GC_1244_227,(0,1,7):C.R2GC_1244_228,(0,1,8):C.R2GC_1244_229,(0,1,9):C.R2GC_1244_230,(0,1,10):C.R2GC_1244_231,(0,1,11):C.R2GC_1244_232})

V_420 = CTVertex(name = 'V_420',
                 type = 'R2',
                 particles = [ P.g, P.g, P.h01 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1913_996,(0,0,1):C.R2GC_1913_997})

V_421 = CTVertex(name = 'V_421',
                 type = 'R2',
                 particles = [ P.g, P.g, P.h02 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1914_998,(0,0,1):C.R2GC_1914_999})

V_422 = CTVertex(name = 'V_422',
                 type = 'R2',
                 particles = [ P.a, P.a, P.g, P.g ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVVV2, L.VVVV3, L.VVVV5 ],
                 loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1186_8,(0,0,1):C.R2GC_1186_9,(0,1,0):C.R2GC_1186_8,(0,1,1):C.R2GC_1186_9,(0,2,0):C.R2GC_1186_8,(0,2,1):C.R2GC_1186_9})

V_423 = CTVertex(name = 'V_423',
                 type = 'R2',
                 particles = [ P.a, P.g, P.g, P.Z ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VVVV2, L.VVVV3, L.VVVV5 ],
                 loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1194_24,(0,0,1):C.R2GC_1194_25,(0,1,0):C.R2GC_1194_24,(0,1,1):C.R2GC_1194_25,(0,2,0):C.R2GC_1194_24,(0,2,1):C.R2GC_1194_25})

V_424 = CTVertex(name = 'V_424',
                 type = 'R2',
                 particles = [ P.g, P.g, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVV2, L.VVVV3, L.VVVV5 ],
                 loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1192_22,(0,0,1):C.R2GC_1192_23,(0,1,0):C.R2GC_1192_22,(0,1,1):C.R2GC_1192_23,(0,2,0):C.R2GC_1192_22,(0,2,1):C.R2GC_1192_23})

V_425 = CTVertex(name = 'V_425',
                 type = 'R2',
                 particles = [ P.g, P.g, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVV2, L.VVVV3, L.VVVV5 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s] ], [ [P.d, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1221_74,(0,0,1):C.R2GC_1221_75,(0,0,2):C.R2GC_1221_76,(0,1,0):C.R2GC_1221_74,(0,1,1):C.R2GC_1221_75,(0,1,2):C.R2GC_1221_76,(0,2,0):C.R2GC_1221_74,(0,2,1):C.R2GC_1221_75,(0,2,2):C.R2GC_1221_76})

V_426 = CTVertex(name = 'V_426',
                 type = 'R2',
                 particles = [ P.a, P.g, P.g, P.g ],
                 color = [ 'd(2,3,4)' ],
                 lorentz = [ L.VVVV2, L.VVVV3, L.VVVV5 ],
                 loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1187_10,(0,0,1):C.R2GC_1187_11,(0,1,0):C.R2GC_1187_10,(0,1,1):C.R2GC_1187_11,(0,2,0):C.R2GC_1187_10,(0,2,1):C.R2GC_1187_11})

V_427 = CTVertex(name = 'V_427',
                 type = 'R2',
                 particles = [ P.g, P.g, P.g, P.Z ],
                 color = [ 'd(1,2,3)', 'f(1,2,3)' ],
                 lorentz = [ L.VVVV1, L.VVVV2, L.VVVV3, L.VVVV5 ],
                 loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                 couplings = {(1,0,0):C.R2GC_1171_1,(1,0,1):C.R2GC_1171_2,(0,1,0):C.R2GC_1195_26,(0,1,1):C.R2GC_1195_27,(0,2,0):C.R2GC_1195_26,(0,2,1):C.R2GC_1195_27,(0,3,0):C.R2GC_1195_26,(0,3,1):C.R2GC_1195_27})

V_428 = CTVertex(name = 'V_428',
                 type = 'R2',
                 particles = [ P.g, P.g, P.h01, P.h01 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1955_1050,(0,0,1):C.R2GC_1955_1051})

V_429 = CTVertex(name = 'V_429',
                 type = 'R2',
                 particles = [ P.g, P.g, P.h01, P.h02 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1969_1054,(0,0,1):C.R2GC_1969_1055})

V_430 = CTVertex(name = 'V_430',
                 type = 'R2',
                 particles = [ P.g, P.g, P.h02, P.h02 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1956_1052,(0,0,1):C.R2GC_1956_1053})

V_431 = CTVertex(name = 'V_431',
                 type = 'R2',
                 particles = [ P.g, P.g, P.A0, P.A0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2254_1118,(0,0,1):C.R2GC_2254_1119})

V_432 = CTVertex(name = 'V_432',
                 type = 'R2',
                 particles = [ P.g, P.g, P.H__minus__, P.H__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2255_1120})

V_433 = CTVertex(name = 'V_433',
                 type = 'R2',
                 particles = [ P.g, P.g, P.A0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2272_1125,(0,0,1):C.R2GC_2272_1126})

V_434 = CTVertex(name = 'V_434',
                 type = 'R2',
                 particles = [ P.g, P.g, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2257_1122,(0,0,1):C.R2GC_2257_1123})

V_435 = CTVertex(name = 'V_435',
                 type = 'R2',
                 particles = [ P.g, P.g, P.G__plus__, P.H__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2271_1124})

V_436 = CTVertex(name = 'V_436',
                 type = 'R2',
                 particles = [ P.g, P.g, P.G__minus__, P.H__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2271_1124})

V_437 = CTVertex(name = 'V_437',
                 type = 'R2',
                 particles = [ P.g, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2256_1121})

V_438 = CTVertex(name = 'V_438',
                 type = 'UV',
                 particles = [ P.g, P.g, P.g ],
                 color = [ 'f(1,2,3)' ],
                 lorentz = [ L.VVV10, L.VVV7, L.VVV8, L.VVV9 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_1598_2404,(0,2,1):C.UVGC_1598_2405,(0,2,4):C.UVGC_1598_2406,(0,2,5):C.UVGC_1598_2407,(0,2,6):C.UVGC_1598_2408,(0,2,7):C.UVGC_1598_2409,(0,2,8):C.UVGC_1598_2410,(0,2,9):C.UVGC_1598_2411,(0,2,10):C.UVGC_1598_2412,(0,2,11):C.UVGC_1598_2413,(0,2,12):C.UVGC_1598_2414,(0,2,13):C.UVGC_1598_2415,(0,2,14):C.UVGC_1598_2416,(0,2,15):C.UVGC_1598_2417,(0,2,16):C.UVGC_1598_2418,(0,2,17):C.UVGC_1598_2419,(0,0,2):C.UVGC_1172_1,(0,1,3):C.UVGC_1173_2,(0,3,5):C.UVGC_1178_18,(0,3,6):C.UVGC_1178_19,(0,3,7):C.UVGC_1178_20,(0,3,8):C.UVGC_1178_21,(0,3,9):C.UVGC_1178_22,(0,3,10):C.UVGC_1178_23,(0,3,11):C.UVGC_1178_24,(0,3,12):C.UVGC_1178_25,(0,3,13):C.UVGC_1178_26,(0,3,14):C.UVGC_1178_27,(0,3,15):C.UVGC_1178_28,(0,3,16):C.UVGC_1178_29})

V_439 = CTVertex(name = 'V_439',
                 type = 'UV',
                 particles = [ P.g, P.g, P.g, P.g ],
                 color = [ 'd(-1,1,3)*d(-1,2,4)', 'd(-1,1,3)*f(-1,2,4)', 'd(-1,1,4)*d(-1,2,3)', 'd(-1,1,4)*f(-1,2,3)', 'd(-1,2,3)*f(-1,1,4)', 'd(-1,2,4)*f(-1,1,3)', 'f(-1,1,2)*f(-1,3,4)', 'f(-1,1,3)*f(-1,2,4)', 'f(-1,1,4)*f(-1,2,3)', 'Identity(1,2)*Identity(3,4)', 'Identity(1,3)*Identity(2,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.VVVV2, L.VVVV3, L.VVVV5, L.VVVV8 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(2,0,3):C.UVGC_1189_152,(2,0,4):C.UVGC_1189_153,(2,0,6):C.UVGC_1189_154,(2,0,7):C.UVGC_1189_155,(2,0,8):C.UVGC_1189_156,(2,0,9):C.UVGC_1189_157,(2,0,10):C.UVGC_1189_158,(2,0,11):C.UVGC_1189_159,(2,0,12):C.UVGC_1189_160,(2,0,13):C.UVGC_1189_161,(2,0,14):C.UVGC_1189_162,(2,0,15):C.UVGC_1189_163,(2,0,16):C.UVGC_1189_164,(2,0,17):C.UVGC_1189_165,(0,0,3):C.UVGC_1189_152,(0,0,4):C.UVGC_1189_153,(0,0,6):C.UVGC_1189_154,(0,0,7):C.UVGC_1189_155,(0,0,8):C.UVGC_1189_156,(0,0,9):C.UVGC_1189_157,(0,0,10):C.UVGC_1189_158,(0,0,11):C.UVGC_1189_159,(0,0,12):C.UVGC_1189_160,(0,0,13):C.UVGC_1189_161,(0,0,14):C.UVGC_1189_162,(0,0,15):C.UVGC_1189_163,(0,0,16):C.UVGC_1189_164,(0,0,17):C.UVGC_1189_165,(4,0,3):C.UVGC_1191_178,(4,0,4):C.UVGC_1191_179,(4,0,6):C.UVGC_1191_180,(4,0,7):C.UVGC_1191_181,(4,0,8):C.UVGC_1191_182,(4,0,9):C.UVGC_1191_183,(4,0,10):C.UVGC_1191_184,(4,0,11):C.UVGC_1191_185,(4,0,12):C.UVGC_1191_186,(4,0,13):C.UVGC_1191_187,(4,0,14):C.UVGC_1191_188,(4,0,15):C.UVGC_1191_189,(4,0,16):C.UVGC_1191_190,(4,0,17):C.UVGC_1191_191,(3,0,3):C.UVGC_1191_178,(3,0,4):C.UVGC_1191_179,(3,0,6):C.UVGC_1191_180,(3,0,7):C.UVGC_1191_181,(3,0,8):C.UVGC_1191_182,(3,0,9):C.UVGC_1191_183,(3,0,10):C.UVGC_1191_184,(3,0,11):C.UVGC_1191_185,(3,0,12):C.UVGC_1191_186,(3,0,13):C.UVGC_1191_187,(3,0,14):C.UVGC_1191_188,(3,0,15):C.UVGC_1191_189,(3,0,16):C.UVGC_1191_190,(3,0,17):C.UVGC_1191_191,(8,0,3):C.UVGC_1189_153,(8,0,4):C.UVGC_1189_152,(8,0,6):C.UVGC_1190_166,(8,0,7):C.UVGC_1190_167,(8,0,8):C.UVGC_1190_168,(8,0,9):C.UVGC_1190_169,(8,0,10):C.UVGC_1190_170,(8,0,11):C.UVGC_1190_171,(8,0,12):C.UVGC_1190_172,(8,0,13):C.UVGC_1190_173,(8,0,14):C.UVGC_1190_174,(8,0,15):C.UVGC_1190_175,(8,0,16):C.UVGC_1190_176,(8,0,17):C.UVGC_1190_177,(6,0,0):C.UVGC_1602_2469,(6,0,2):C.UVGC_1602_2470,(6,0,3):C.UVGC_1602_2471,(6,0,4):C.UVGC_1602_2472,(6,0,5):C.UVGC_1602_2473,(6,0,6):C.UVGC_1602_2474,(6,0,7):C.UVGC_1602_2475,(6,0,8):C.UVGC_1602_2476,(6,0,9):C.UVGC_1602_2477,(6,0,10):C.UVGC_1602_2478,(6,0,11):C.UVGC_1602_2479,(6,0,12):C.UVGC_1602_2480,(6,0,13):C.UVGC_1602_2481,(6,0,14):C.UVGC_1602_2482,(6,0,15):C.UVGC_1602_2483,(6,0,16):C.UVGC_1602_2484,(6,0,17):C.UVGC_1602_2485,(6,0,18):C.UVGC_1602_2486,(7,0,0):C.UVGC_1602_2469,(7,0,2):C.UVGC_1602_2470,(7,0,3):C.UVGC_1603_2487,(7,0,4):C.UVGC_1603_2488,(7,0,5):C.UVGC_1602_2473,(7,0,6):C.UVGC_1603_2489,(7,0,7):C.UVGC_1603_2490,(7,0,8):C.UVGC_1603_2491,(7,0,9):C.UVGC_1603_2492,(7,0,10):C.UVGC_1603_2493,(7,0,11):C.UVGC_1603_2494,(7,0,12):C.UVGC_1603_2495,(7,0,13):C.UVGC_1603_2496,(7,0,14):C.UVGC_1603_2497,(7,0,15):C.UVGC_1603_2498,(7,0,16):C.UVGC_1603_2499,(7,0,17):C.UVGC_1603_2500,(7,0,18):C.UVGC_1602_2486,(5,0,3):C.UVGC_1191_178,(5,0,4):C.UVGC_1191_179,(5,0,6):C.UVGC_1191_180,(5,0,7):C.UVGC_1191_181,(5,0,8):C.UVGC_1191_182,(5,0,9):C.UVGC_1191_183,(5,0,10):C.UVGC_1191_184,(5,0,11):C.UVGC_1191_185,(5,0,12):C.UVGC_1191_186,(5,0,13):C.UVGC_1191_187,(5,0,14):C.UVGC_1191_188,(5,0,15):C.UVGC_1191_189,(5,0,16):C.UVGC_1191_190,(5,0,17):C.UVGC_1191_191,(1,0,3):C.UVGC_1191_178,(1,0,4):C.UVGC_1191_179,(1,0,6):C.UVGC_1191_180,(1,0,7):C.UVGC_1191_181,(1,0,8):C.UVGC_1191_182,(1,0,9):C.UVGC_1191_183,(1,0,10):C.UVGC_1191_184,(1,0,11):C.UVGC_1191_185,(1,0,12):C.UVGC_1191_186,(1,0,13):C.UVGC_1191_187,(1,0,14):C.UVGC_1191_188,(1,0,15):C.UVGC_1191_189,(1,0,16):C.UVGC_1191_190,(1,0,17):C.UVGC_1191_191,(11,3,3):C.UVGC_1188_138,(11,3,4):C.UVGC_1188_139,(11,3,6):C.UVGC_1188_140,(11,3,7):C.UVGC_1188_141,(11,3,8):C.UVGC_1188_142,(11,3,9):C.UVGC_1188_143,(11,3,10):C.UVGC_1188_144,(11,3,11):C.UVGC_1188_145,(11,3,12):C.UVGC_1188_146,(11,3,13):C.UVGC_1188_147,(11,3,14):C.UVGC_1188_148,(11,3,15):C.UVGC_1188_149,(11,3,16):C.UVGC_1188_150,(11,3,17):C.UVGC_1188_151,(10,3,3):C.UVGC_1188_138,(10,3,4):C.UVGC_1188_139,(10,3,6):C.UVGC_1188_140,(10,3,7):C.UVGC_1188_141,(10,3,8):C.UVGC_1188_142,(10,3,9):C.UVGC_1188_143,(10,3,10):C.UVGC_1188_144,(10,3,11):C.UVGC_1188_145,(10,3,12):C.UVGC_1188_146,(10,3,13):C.UVGC_1188_147,(10,3,14):C.UVGC_1188_148,(10,3,15):C.UVGC_1188_149,(10,3,16):C.UVGC_1188_150,(10,3,17):C.UVGC_1188_151,(9,3,3):C.UVGC_1176_4,(9,3,4):C.UVGC_1176_5,(0,1,3):C.UVGC_1189_152,(0,1,4):C.UVGC_1189_153,(0,1,6):C.UVGC_1189_154,(0,1,7):C.UVGC_1189_155,(0,1,8):C.UVGC_1189_156,(0,1,9):C.UVGC_1189_157,(0,1,10):C.UVGC_1189_158,(0,1,11):C.UVGC_1189_159,(0,1,12):C.UVGC_1189_160,(0,1,13):C.UVGC_1189_161,(0,1,14):C.UVGC_1189_162,(0,1,15):C.UVGC_1189_163,(0,1,16):C.UVGC_1189_164,(0,1,17):C.UVGC_1189_165,(2,1,3):C.UVGC_1189_152,(2,1,4):C.UVGC_1189_153,(2,1,6):C.UVGC_1189_154,(2,1,7):C.UVGC_1189_155,(2,1,8):C.UVGC_1189_156,(2,1,9):C.UVGC_1189_157,(2,1,10):C.UVGC_1189_158,(2,1,11):C.UVGC_1189_159,(2,1,12):C.UVGC_1189_160,(2,1,13):C.UVGC_1189_161,(2,1,14):C.UVGC_1189_162,(2,1,15):C.UVGC_1189_163,(2,1,16):C.UVGC_1189_164,(2,1,17):C.UVGC_1189_165,(5,1,3):C.UVGC_1191_178,(5,1,4):C.UVGC_1191_179,(5,1,6):C.UVGC_1191_180,(5,1,7):C.UVGC_1191_181,(5,1,8):C.UVGC_1191_182,(5,1,9):C.UVGC_1191_183,(5,1,10):C.UVGC_1191_184,(5,1,11):C.UVGC_1191_185,(5,1,12):C.UVGC_1191_186,(5,1,13):C.UVGC_1191_187,(5,1,14):C.UVGC_1191_188,(5,1,15):C.UVGC_1191_189,(5,1,16):C.UVGC_1191_190,(5,1,17):C.UVGC_1191_191,(1,1,3):C.UVGC_1191_178,(1,1,4):C.UVGC_1191_179,(1,1,6):C.UVGC_1191_180,(1,1,7):C.UVGC_1191_181,(1,1,8):C.UVGC_1191_182,(1,1,9):C.UVGC_1191_183,(1,1,10):C.UVGC_1191_184,(1,1,11):C.UVGC_1191_185,(1,1,12):C.UVGC_1191_186,(1,1,13):C.UVGC_1191_187,(1,1,14):C.UVGC_1191_188,(1,1,15):C.UVGC_1191_189,(1,1,16):C.UVGC_1191_190,(1,1,17):C.UVGC_1191_191,(7,1,1):C.UVGC_1196_240,(7,1,3):C.UVGC_1197_255,(7,1,4):C.UVGC_1197_256,(7,1,5):C.UVGC_1196_242,(7,1,6):C.UVGC_1197_257,(7,1,7):C.UVGC_1197_258,(7,1,8):C.UVGC_1197_259,(7,1,9):C.UVGC_1197_260,(7,1,10):C.UVGC_1197_261,(7,1,11):C.UVGC_1197_262,(7,1,12):C.UVGC_1197_263,(7,1,13):C.UVGC_1197_264,(7,1,14):C.UVGC_1197_265,(7,1,15):C.UVGC_1197_266,(7,1,16):C.UVGC_1197_267,(7,1,17):C.UVGC_1197_268,(4,1,3):C.UVGC_1191_178,(4,1,4):C.UVGC_1191_179,(4,1,6):C.UVGC_1191_180,(4,1,7):C.UVGC_1191_181,(4,1,8):C.UVGC_1191_182,(4,1,9):C.UVGC_1191_183,(4,1,10):C.UVGC_1191_184,(4,1,11):C.UVGC_1191_185,(4,1,12):C.UVGC_1191_186,(4,1,13):C.UVGC_1191_187,(4,1,14):C.UVGC_1191_188,(4,1,15):C.UVGC_1191_189,(4,1,16):C.UVGC_1191_190,(4,1,17):C.UVGC_1191_191,(3,1,3):C.UVGC_1191_178,(3,1,4):C.UVGC_1191_179,(3,1,6):C.UVGC_1191_180,(3,1,7):C.UVGC_1191_181,(3,1,8):C.UVGC_1191_182,(3,1,9):C.UVGC_1191_183,(3,1,10):C.UVGC_1191_184,(3,1,11):C.UVGC_1191_185,(3,1,12):C.UVGC_1191_186,(3,1,13):C.UVGC_1191_187,(3,1,14):C.UVGC_1191_188,(3,1,15):C.UVGC_1191_189,(3,1,16):C.UVGC_1191_190,(3,1,17):C.UVGC_1191_191,(8,1,0):C.UVGC_1604_2501,(8,1,2):C.UVGC_1604_2502,(8,1,3):C.UVGC_1604_2503,(8,1,4):C.UVGC_1604_2504,(8,1,5):C.UVGC_1604_2505,(8,1,6):C.UVGC_1604_2506,(8,1,7):C.UVGC_1604_2507,(8,1,8):C.UVGC_1604_2508,(8,1,9):C.UVGC_1604_2509,(8,1,10):C.UVGC_1604_2510,(8,1,11):C.UVGC_1604_2511,(8,1,12):C.UVGC_1604_2512,(8,1,13):C.UVGC_1604_2513,(8,1,14):C.UVGC_1604_2514,(8,1,15):C.UVGC_1604_2515,(8,1,16):C.UVGC_1604_2516,(8,1,17):C.UVGC_1604_2517,(8,1,18):C.UVGC_1604_2518,(6,1,0):C.UVGC_1599_2420,(6,1,3):C.UVGC_1599_2421,(6,1,4):C.UVGC_1599_2422,(6,1,5):C.UVGC_1599_2423,(6,1,6):C.UVGC_1599_2424,(6,1,7):C.UVGC_1599_2425,(6,1,8):C.UVGC_1599_2426,(6,1,9):C.UVGC_1599_2427,(6,1,10):C.UVGC_1599_2428,(6,1,11):C.UVGC_1599_2429,(6,1,12):C.UVGC_1599_2430,(6,1,13):C.UVGC_1599_2431,(6,1,14):C.UVGC_1599_2432,(6,1,15):C.UVGC_1599_2433,(6,1,16):C.UVGC_1599_2434,(6,1,17):C.UVGC_1599_2435,(6,1,18):C.UVGC_1599_2436,(0,2,3):C.UVGC_1189_152,(0,2,4):C.UVGC_1189_153,(0,2,6):C.UVGC_1189_154,(0,2,7):C.UVGC_1189_155,(0,2,8):C.UVGC_1189_156,(0,2,9):C.UVGC_1189_157,(0,2,10):C.UVGC_1189_158,(0,2,11):C.UVGC_1189_159,(0,2,12):C.UVGC_1189_160,(0,2,13):C.UVGC_1189_161,(0,2,14):C.UVGC_1189_162,(0,2,15):C.UVGC_1189_163,(0,2,16):C.UVGC_1189_164,(0,2,17):C.UVGC_1189_165,(2,2,3):C.UVGC_1189_152,(2,2,4):C.UVGC_1189_153,(2,2,6):C.UVGC_1189_154,(2,2,7):C.UVGC_1189_155,(2,2,8):C.UVGC_1189_156,(2,2,9):C.UVGC_1189_157,(2,2,10):C.UVGC_1189_158,(2,2,11):C.UVGC_1189_159,(2,2,12):C.UVGC_1189_160,(2,2,13):C.UVGC_1189_161,(2,2,14):C.UVGC_1189_162,(2,2,15):C.UVGC_1189_163,(2,2,16):C.UVGC_1189_164,(2,2,17):C.UVGC_1189_165,(5,2,3):C.UVGC_1191_178,(5,2,4):C.UVGC_1191_179,(5,2,6):C.UVGC_1191_180,(5,2,7):C.UVGC_1191_181,(5,2,8):C.UVGC_1191_182,(5,2,9):C.UVGC_1191_183,(5,2,10):C.UVGC_1191_184,(5,2,11):C.UVGC_1191_185,(5,2,12):C.UVGC_1191_186,(5,2,13):C.UVGC_1191_187,(5,2,14):C.UVGC_1191_188,(5,2,15):C.UVGC_1191_189,(5,2,16):C.UVGC_1191_190,(5,2,17):C.UVGC_1191_191,(1,2,3):C.UVGC_1191_178,(1,2,4):C.UVGC_1191_179,(1,2,6):C.UVGC_1191_180,(1,2,7):C.UVGC_1191_181,(1,2,8):C.UVGC_1191_182,(1,2,9):C.UVGC_1191_183,(1,2,10):C.UVGC_1191_184,(1,2,11):C.UVGC_1191_185,(1,2,12):C.UVGC_1191_186,(1,2,13):C.UVGC_1191_187,(1,2,14):C.UVGC_1191_188,(1,2,15):C.UVGC_1191_189,(1,2,16):C.UVGC_1191_190,(1,2,17):C.UVGC_1191_191,(7,2,0):C.UVGC_1599_2420,(7,2,3):C.UVGC_1600_2437,(7,2,4):C.UVGC_1600_2438,(7,2,5):C.UVGC_1599_2423,(7,2,6):C.UVGC_1600_2439,(7,2,7):C.UVGC_1600_2440,(7,2,8):C.UVGC_1600_2441,(7,2,9):C.UVGC_1600_2442,(7,2,10):C.UVGC_1600_2443,(7,2,11):C.UVGC_1600_2444,(7,2,12):C.UVGC_1600_2445,(7,2,13):C.UVGC_1600_2446,(7,2,14):C.UVGC_1600_2447,(7,2,15):C.UVGC_1600_2448,(7,2,16):C.UVGC_1600_2449,(7,2,17):C.UVGC_1600_2450,(7,2,18):C.UVGC_1599_2436,(4,2,3):C.UVGC_1191_178,(4,2,4):C.UVGC_1191_179,(4,2,6):C.UVGC_1191_180,(4,2,7):C.UVGC_1191_181,(4,2,8):C.UVGC_1191_182,(4,2,9):C.UVGC_1191_183,(4,2,10):C.UVGC_1191_184,(4,2,11):C.UVGC_1191_185,(4,2,12):C.UVGC_1191_186,(4,2,13):C.UVGC_1191_187,(4,2,14):C.UVGC_1191_188,(4,2,15):C.UVGC_1191_189,(4,2,16):C.UVGC_1191_190,(4,2,17):C.UVGC_1191_191,(3,2,3):C.UVGC_1191_178,(3,2,4):C.UVGC_1191_179,(3,2,6):C.UVGC_1191_180,(3,2,7):C.UVGC_1191_181,(3,2,8):C.UVGC_1191_182,(3,2,9):C.UVGC_1191_183,(3,2,10):C.UVGC_1191_184,(3,2,11):C.UVGC_1191_185,(3,2,12):C.UVGC_1191_186,(3,2,13):C.UVGC_1191_187,(3,2,14):C.UVGC_1191_188,(3,2,15):C.UVGC_1191_189,(3,2,16):C.UVGC_1191_190,(3,2,17):C.UVGC_1191_191,(8,2,0):C.UVGC_1601_2451,(8,2,2):C.UVGC_1601_2452,(8,2,3):C.UVGC_1601_2453,(8,2,4):C.UVGC_1601_2454,(8,2,5):C.UVGC_1601_2455,(8,2,6):C.UVGC_1601_2456,(8,2,7):C.UVGC_1601_2457,(8,2,8):C.UVGC_1601_2458,(8,2,9):C.UVGC_1601_2459,(8,2,10):C.UVGC_1601_2460,(8,2,11):C.UVGC_1601_2461,(8,2,12):C.UVGC_1601_2462,(8,2,13):C.UVGC_1601_2463,(8,2,14):C.UVGC_1601_2464,(8,2,15):C.UVGC_1601_2465,(8,2,16):C.UVGC_1601_2466,(8,2,17):C.UVGC_1601_2467,(8,2,18):C.UVGC_1601_2468,(6,2,1):C.UVGC_1196_240,(6,2,3):C.UVGC_1196_241,(6,2,4):C.UVGC_1176_4,(6,2,5):C.UVGC_1196_242,(6,2,6):C.UVGC_1196_243,(6,2,7):C.UVGC_1196_244,(6,2,8):C.UVGC_1196_245,(6,2,9):C.UVGC_1196_246,(6,2,10):C.UVGC_1196_247,(6,2,11):C.UVGC_1196_248,(6,2,12):C.UVGC_1196_249,(6,2,13):C.UVGC_1196_250,(6,2,14):C.UVGC_1196_251,(6,2,15):C.UVGC_1196_252,(6,2,16):C.UVGC_1196_253,(6,2,17):C.UVGC_1196_254})

V_440 = CTVertex(name = 'V_440',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.h02 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS2 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1449_1595,(0,0,1):C.UVGC_1449_1596,(0,0,2):C.UVGC_1449_1597})

V_441 = CTVertex(name = 'V_441',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1450_1598,(0,0,1):C.UVGC_1450_1599,(0,0,2):C.UVGC_1450_1600})

V_442 = CTVertex(name = 'V_442',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.H__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1643_2900,(0,1,6):C.UVGC_1643_2901,(0,1,2):C.UVGC_1643_2902,(0,1,3):C.UVGC_1643_2903,(0,1,4):C.UVGC_1643_2904,(0,1,5):C.UVGC_1643_2905,(0,1,1):C.UVGC_1643_2906,(0,0,0):C.UVGC_2205_4312,(0,0,6):C.UVGC_2205_4313,(0,0,2):C.UVGC_2205_4314,(0,0,3):C.UVGC_2205_4315,(0,0,4):C.UVGC_2205_4316,(0,0,5):C.UVGC_2205_4317,(0,0,1):C.UVGC_2205_4318})

V_443 = CTVertex(name = 'V_443',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1639_2876,(0,1,6):C.UVGC_1639_2877,(0,1,2):C.UVGC_1639_2878,(0,1,3):C.UVGC_1639_2879,(0,1,4):C.UVGC_1639_2880,(0,1,5):C.UVGC_1639_2881,(0,1,1):C.UVGC_1639_2882,(0,0,0):C.UVGC_2204_4305,(0,0,6):C.UVGC_2204_4306,(0,0,2):C.UVGC_2204_4307,(0,0,3):C.UVGC_2204_4308,(0,0,4):C.UVGC_2204_4309,(0,0,5):C.UVGC_2204_4310,(0,0,1):C.UVGC_2204_4311})

V_444 = CTVertex(name = 'V_444',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.h01 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS2 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_1638_2873,(0,0,0):C.UVGC_1638_2874,(0,0,1):C.UVGC_1638_2875})

V_445 = CTVertex(name = 'V_445',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.A0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS1 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_1640_2883,(0,0,0):C.UVGC_1640_2884,(0,0,1):C.UVGC_1640_2885})

V_446 = CTVertex(name = 'V_446',
                 type = 'UV',
                 particles = [ P.n1, P.d, P.sd1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd1] ], [ [P.go, P.sd1] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.UVGC_1262_715,(0,0,1):C.UVGC_1262_716,(0,0,4):C.UVGC_1262_717,(0,0,3):C.UVGC_1262_718,(0,0,2):C.UVGC_1262_719})

V_447 = CTVertex(name = 'V_447',
                 type = 'UV',
                 particles = [ P.n2, P.d, P.sd1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd1] ], [ [P.go, P.sd1] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.UVGC_1263_720,(0,0,1):C.UVGC_1263_721,(0,0,4):C.UVGC_1263_722,(0,0,3):C.UVGC_1263_723,(0,0,2):C.UVGC_1263_724})

V_448 = CTVertex(name = 'V_448',
                 type = 'UV',
                 particles = [ P.n3, P.d, P.sd1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd1] ], [ [P.go, P.sd1] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.UVGC_1264_725,(0,0,1):C.UVGC_1264_726,(0,0,4):C.UVGC_1264_727,(0,0,3):C.UVGC_1264_728,(0,0,2):C.UVGC_1264_729})

V_449 = CTVertex(name = 'V_449',
                 type = 'UV',
                 particles = [ P.n4, P.d, P.sd1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd1] ], [ [P.go, P.sd1] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.UVGC_1265_730,(0,0,1):C.UVGC_1265_731,(0,0,4):C.UVGC_1265_732,(0,0,3):C.UVGC_1265_733,(0,0,2):C.UVGC_1265_734})

V_450 = CTVertex(name = 'V_450',
                 type = 'UV',
                 particles = [ P.a, P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS2 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.UVGC_1255_656,(0,0,1):C.UVGC_1255_657})

V_451 = CTVertex(name = 'V_451',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.x1__plus__, P.sd1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.go, P.su1] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1459_1626,(0,0,2):C.UVGC_1459_1627,(0,0,4):C.UVGC_1459_1628,(0,0,1):C.UVGC_1459_1629,(0,0,3):C.UVGC_1459_1630})

V_452 = CTVertex(name = 'V_452',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.x2__plus__, P.sd1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.go, P.su1] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1460_1631,(0,0,2):C.UVGC_1460_1632,(0,0,4):C.UVGC_1460_1633,(0,0,1):C.UVGC_1460_1634,(0,0,3):C.UVGC_1460_1635})

V_453 = CTVertex(name = 'V_453',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.n1, P.sd1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd1] ], [ [P.go, P.sd1] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.UVGC_1251_636,(0,0,1):C.UVGC_1251_637,(0,0,4):C.UVGC_1251_638,(0,0,3):C.UVGC_1251_639,(0,0,2):C.UVGC_1251_640})

V_454 = CTVertex(name = 'V_454',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.n2, P.sd1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd1] ], [ [P.go, P.sd1] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.UVGC_1252_641,(0,0,1):C.UVGC_1252_642,(0,0,4):C.UVGC_1252_643,(0,0,3):C.UVGC_1252_644,(0,0,2):C.UVGC_1252_645})

V_455 = CTVertex(name = 'V_455',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.n3, P.sd1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd1] ], [ [P.go, P.sd1] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.UVGC_1253_646,(0,0,1):C.UVGC_1253_647,(0,0,4):C.UVGC_1253_648,(0,0,3):C.UVGC_1253_649,(0,0,2):C.UVGC_1253_650})

V_456 = CTVertex(name = 'V_456',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.n4, P.sd1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd1] ], [ [P.go, P.sd1] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.UVGC_1254_651,(0,0,1):C.UVGC_1254_652,(0,0,4):C.UVGC_1254_653,(0,0,3):C.UVGC_1254_654,(0,0,2):C.UVGC_1254_655})

V_457 = CTVertex(name = 'V_457',
                 type = 'UV',
                 particles = [ P.a, P.a, P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ], [ [P.sd1] ] ],
                 couplings = {(0,0,2):C.UVGC_1256_658,(0,0,0):C.UVGC_1256_659,(0,0,1):C.UVGC_1256_660})

V_458 = CTVertex(name = 'V_458',
                 type = 'UV',
                 particles = [ P.h02, P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ], [ [P.sd1] ] ],
                 couplings = {(0,0,2):C.UVGC_1916_4021,(0,0,0):C.UVGC_1916_4022,(0,0,1):C.UVGC_1916_4023})

V_459 = CTVertex(name = 'V_459',
                 type = 'UV',
                 particles = [ P.h01, P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ], [ [P.sd1] ] ],
                 couplings = {(0,0,2):C.UVGC_1915_4018,(0,0,0):C.UVGC_1915_4019,(0,0,1):C.UVGC_1915_4020})

V_460 = CTVertex(name = 'V_460',
                 type = 'UV',
                 particles = [ P.n1, P.s, P.sd2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd2] ], [ [P.g, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.s, P.sd2] ] ],
                 couplings = {(0,0,2):C.UVGC_1288_850,(0,0,3):C.UVGC_1288_851,(0,0,0):C.UVGC_1288_852,(0,0,1):C.UVGC_1288_853,(0,0,4):C.UVGC_1288_854})

V_461 = CTVertex(name = 'V_461',
                 type = 'UV',
                 particles = [ P.n2, P.s, P.sd2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd2] ], [ [P.g, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.s, P.sd2] ] ],
                 couplings = {(0,0,2):C.UVGC_1289_855,(0,0,3):C.UVGC_1289_856,(0,0,0):C.UVGC_1289_857,(0,0,1):C.UVGC_1289_858,(0,0,4):C.UVGC_1289_859})

V_462 = CTVertex(name = 'V_462',
                 type = 'UV',
                 particles = [ P.n3, P.s, P.sd2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd2] ], [ [P.g, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.s, P.sd2] ] ],
                 couplings = {(0,0,2):C.UVGC_1290_860,(0,0,3):C.UVGC_1290_861,(0,0,0):C.UVGC_1290_862,(0,0,1):C.UVGC_1290_863,(0,0,4):C.UVGC_1290_864})

V_463 = CTVertex(name = 'V_463',
                 type = 'UV',
                 particles = [ P.n4, P.s, P.sd2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd2] ], [ [P.g, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.s, P.sd2] ] ],
                 couplings = {(0,0,2):C.UVGC_1291_865,(0,0,3):C.UVGC_1291_866,(0,0,0):C.UVGC_1291_867,(0,0,1):C.UVGC_1291_868,(0,0,4):C.UVGC_1291_869})

V_464 = CTVertex(name = 'V_464',
                 type = 'UV',
                 particles = [ P.a, P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS2 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ] ],
                 couplings = {(0,0,1):C.UVGC_1281_791,(0,0,0):C.UVGC_1281_792})

V_465 = CTVertex(name = 'V_465',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.x1__plus__, P.sd2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.sd2] ], [ [P.go, P.s] ], [ [P.go, P.su2] ], [ [P.g, P.sd2] ] ],
                 couplings = {(0,0,0):C.UVGC_1505_1880,(0,0,4):C.UVGC_1505_1881,(0,0,2):C.UVGC_1505_1882,(0,0,3):C.UVGC_1505_1883,(0,0,1):C.UVGC_1505_1884})

V_466 = CTVertex(name = 'V_466',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.x2__plus__, P.sd2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.sd2] ], [ [P.go, P.s] ], [ [P.go, P.su2] ], [ [P.g, P.sd2] ] ],
                 couplings = {(0,0,0):C.UVGC_1506_1885,(0,0,4):C.UVGC_1506_1886,(0,0,2):C.UVGC_1506_1887,(0,0,3):C.UVGC_1506_1888,(0,0,1):C.UVGC_1506_1889})

V_467 = CTVertex(name = 'V_467',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.n1, P.sd2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd2] ], [ [P.g, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.s, P.sd2] ] ],
                 couplings = {(0,0,2):C.UVGC_1277_771,(0,0,3):C.UVGC_1277_772,(0,0,0):C.UVGC_1277_773,(0,0,1):C.UVGC_1277_774,(0,0,4):C.UVGC_1277_775})

V_468 = CTVertex(name = 'V_468',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.n2, P.sd2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd2] ], [ [P.g, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.s, P.sd2] ] ],
                 couplings = {(0,0,2):C.UVGC_1278_776,(0,0,3):C.UVGC_1278_777,(0,0,0):C.UVGC_1278_778,(0,0,1):C.UVGC_1278_779,(0,0,4):C.UVGC_1278_780})

V_469 = CTVertex(name = 'V_469',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.n3, P.sd2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd2] ], [ [P.g, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.s, P.sd2] ] ],
                 couplings = {(0,0,2):C.UVGC_1279_781,(0,0,3):C.UVGC_1279_782,(0,0,0):C.UVGC_1279_783,(0,0,1):C.UVGC_1279_784,(0,0,4):C.UVGC_1279_785})

V_470 = CTVertex(name = 'V_470',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.n4, P.sd2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd2] ], [ [P.g, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.s, P.sd2] ] ],
                 couplings = {(0,0,2):C.UVGC_1280_786,(0,0,3):C.UVGC_1280_787,(0,0,0):C.UVGC_1280_788,(0,0,1):C.UVGC_1280_789,(0,0,4):C.UVGC_1280_790})

V_471 = CTVertex(name = 'V_471',
                 type = 'UV',
                 particles = [ P.a, P.a, P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.sd2] ] ],
                 couplings = {(0,0,2):C.UVGC_1282_793,(0,0,1):C.UVGC_1282_794,(0,0,0):C.UVGC_1282_795})

V_472 = CTVertex(name = 'V_472',
                 type = 'UV',
                 particles = [ P.h02, P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.sd2] ] ],
                 couplings = {(0,0,2):C.UVGC_1918_4027,(0,0,1):C.UVGC_1918_4028,(0,0,0):C.UVGC_1918_4029})

V_473 = CTVertex(name = 'V_473',
                 type = 'UV',
                 particles = [ P.h01, P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.sd2] ] ],
                 couplings = {(0,0,2):C.UVGC_1917_4024,(0,0,1):C.UVGC_1917_4025,(0,0,0):C.UVGC_1917_4026})

V_474 = CTVertex(name = 'V_474',
                 type = 'UV',
                 particles = [ P.n1, P.b, P.sd3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,0,0):C.UVGC_1437_1515,(0,0,1):C.UVGC_1437_1516,(0,0,6):C.UVGC_1437_1517,(0,0,4):C.UVGC_1437_1518,(0,0,5):C.UVGC_1437_1519,(0,0,3):C.UVGC_1437_1520,(0,0,2):C.UVGC_1437_1521,(0,1,0):C.UVGC_1433_1491,(0,1,1):C.UVGC_1433_1492,(0,1,6):C.UVGC_1433_1493,(0,1,4):C.UVGC_1433_1494,(0,1,5):C.UVGC_1433_1495,(0,1,3):C.UVGC_1433_1496})

V_475 = CTVertex(name = 'V_475',
                 type = 'UV',
                 particles = [ P.n2, P.b, P.sd3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,0,0):C.UVGC_1438_1522,(0,0,1):C.UVGC_1438_1523,(0,0,6):C.UVGC_1438_1524,(0,0,4):C.UVGC_1438_1525,(0,0,5):C.UVGC_1438_1526,(0,0,3):C.UVGC_1438_1527,(0,0,2):C.UVGC_1438_1528,(0,1,0):C.UVGC_1434_1497,(0,1,1):C.UVGC_1434_1498,(0,1,6):C.UVGC_1434_1499,(0,1,4):C.UVGC_1434_1500,(0,1,5):C.UVGC_1434_1501,(0,1,3):C.UVGC_1434_1502})

V_476 = CTVertex(name = 'V_476',
                 type = 'UV',
                 particles = [ P.n3, P.b, P.sd3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,0,0):C.UVGC_1439_1529,(0,0,1):C.UVGC_1439_1530,(0,0,6):C.UVGC_1439_1531,(0,0,4):C.UVGC_1439_1532,(0,0,5):C.UVGC_1439_1533,(0,0,3):C.UVGC_1439_1534,(0,0,2):C.UVGC_1439_1535,(0,1,0):C.UVGC_1435_1503,(0,1,1):C.UVGC_1435_1504,(0,1,6):C.UVGC_1435_1505,(0,1,4):C.UVGC_1435_1506,(0,1,5):C.UVGC_1435_1507,(0,1,3):C.UVGC_1435_1508})

V_477 = CTVertex(name = 'V_477',
                 type = 'UV',
                 particles = [ P.n4, P.b, P.sd3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,0,0):C.UVGC_1440_1536,(0,0,1):C.UVGC_1440_1537,(0,0,6):C.UVGC_1440_1538,(0,0,4):C.UVGC_1440_1539,(0,0,5):C.UVGC_1440_1540,(0,0,3):C.UVGC_1440_1541,(0,0,2):C.UVGC_1440_1542,(0,1,0):C.UVGC_1436_1509,(0,1,1):C.UVGC_1436_1510,(0,1,6):C.UVGC_1436_1511,(0,1,4):C.UVGC_1436_1512,(0,1,5):C.UVGC_1436_1513,(0,1,3):C.UVGC_1436_1514})

V_478 = CTVertex(name = 'V_478',
                 type = 'UV',
                 particles = [ P.a, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS2 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,0,0):C.UVGC_1299_902,(0,0,1):C.UVGC_1299_903})

V_479 = CTVertex(name = 'V_479',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.x1__plus__, P.sd3 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.su6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1632_2827,(0,0,4):C.UVGC_1632_2828,(0,0,6):C.UVGC_1632_2829,(0,0,2):C.UVGC_1632_2830,(0,0,3):C.UVGC_1632_2831,(0,0,5):C.UVGC_1632_2832,(0,1,0):C.UVGC_1626_2781,(0,1,4):C.UVGC_1626_2782,(0,1,6):C.UVGC_1626_2783,(0,1,2):C.UVGC_1626_2784,(0,1,3):C.UVGC_1626_2785,(0,1,1):C.UVGC_1626_2786,(0,1,5):C.UVGC_1626_2787})

V_480 = CTVertex(name = 'V_480',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.x2__plus__, P.sd3 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.su6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1634_2843,(0,0,4):C.UVGC_1634_2844,(0,0,6):C.UVGC_1634_2845,(0,0,2):C.UVGC_1634_2846,(0,0,3):C.UVGC_1634_2847,(0,0,5):C.UVGC_1634_2848,(0,1,0):C.UVGC_1627_2788,(0,1,4):C.UVGC_1627_2789,(0,1,6):C.UVGC_1627_2790,(0,1,2):C.UVGC_1627_2791,(0,1,3):C.UVGC_1627_2792,(0,1,1):C.UVGC_1627_2793,(0,1,5):C.UVGC_1627_2794})

V_481 = CTVertex(name = 'V_481',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.n1, P.sd3 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,1,0):C.UVGC_1421_1411,(0,1,1):C.UVGC_1421_1412,(0,1,6):C.UVGC_1421_1413,(0,1,4):C.UVGC_1421_1414,(0,1,5):C.UVGC_1421_1415,(0,1,3):C.UVGC_1421_1416,(0,1,2):C.UVGC_1421_1417,(0,0,0):C.UVGC_1425_1439,(0,0,1):C.UVGC_1425_1440,(0,0,6):C.UVGC_1425_1441,(0,0,4):C.UVGC_1425_1442,(0,0,5):C.UVGC_1425_1443,(0,0,3):C.UVGC_1425_1444})

V_482 = CTVertex(name = 'V_482',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.n2, P.sd3 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,1,0):C.UVGC_1422_1418,(0,1,1):C.UVGC_1422_1419,(0,1,6):C.UVGC_1422_1420,(0,1,4):C.UVGC_1422_1421,(0,1,5):C.UVGC_1422_1422,(0,1,3):C.UVGC_1422_1423,(0,1,2):C.UVGC_1422_1424,(0,0,0):C.UVGC_1427_1452,(0,0,1):C.UVGC_1427_1453,(0,0,6):C.UVGC_1427_1454,(0,0,4):C.UVGC_1427_1455,(0,0,5):C.UVGC_1427_1456,(0,0,3):C.UVGC_1427_1457})

V_483 = CTVertex(name = 'V_483',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.n3, P.sd3 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,1,0):C.UVGC_1423_1425,(0,1,1):C.UVGC_1423_1426,(0,1,6):C.UVGC_1423_1427,(0,1,4):C.UVGC_1423_1428,(0,1,5):C.UVGC_1423_1429,(0,1,3):C.UVGC_1423_1430,(0,1,2):C.UVGC_1423_1431,(0,0,0):C.UVGC_1429_1465,(0,0,1):C.UVGC_1429_1466,(0,0,6):C.UVGC_1429_1467,(0,0,4):C.UVGC_1429_1468,(0,0,5):C.UVGC_1429_1469,(0,0,3):C.UVGC_1429_1470})

V_484 = CTVertex(name = 'V_484',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.n4, P.sd3 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd6] ], [ [P.b, P.g, P.sd3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,1,0):C.UVGC_1424_1432,(0,1,1):C.UVGC_1424_1433,(0,1,6):C.UVGC_1424_1434,(0,1,4):C.UVGC_1424_1435,(0,1,5):C.UVGC_1424_1436,(0,1,3):C.UVGC_1424_1437,(0,1,2):C.UVGC_1424_1438,(0,0,0):C.UVGC_1431_1478,(0,0,1):C.UVGC_1431_1479,(0,0,6):C.UVGC_1431_1480,(0,0,4):C.UVGC_1431_1481,(0,0,5):C.UVGC_1431_1482,(0,0,3):C.UVGC_1431_1483})

V_485 = CTVertex(name = 'V_485',
                 type = 'UV',
                 particles = [ P.a, P.a, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ], [ [P.sd3] ] ],
                 couplings = {(0,0,2):C.UVGC_1300_904,(0,0,0):C.UVGC_1300_905,(0,0,1):C.UVGC_1300_906})

V_486 = CTVertex(name = 'V_486',
                 type = 'UV',
                 particles = [ P.h02, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ], [ [P.sd3] ] ],
                 couplings = {(0,0,5):C.UVGC_1919_4030,(0,0,0):C.UVGC_1919_4031,(0,0,1):C.UVGC_1919_4032,(0,0,4):C.UVGC_1919_4033,(0,0,2):C.UVGC_1919_4034,(0,0,3):C.UVGC_1919_4035})

V_487 = CTVertex(name = 'V_487',
                 type = 'UV',
                 particles = [ P.h01, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ], [ [P.sd3] ] ],
                 couplings = {(0,0,5):C.UVGC_1928_4071,(0,0,0):C.UVGC_1928_4072,(0,0,1):C.UVGC_1928_4073,(0,0,4):C.UVGC_1928_4074,(0,0,2):C.UVGC_1928_4075,(0,0,3):C.UVGC_1928_4076})

V_488 = CTVertex(name = 'V_488',
                 type = 'UV',
                 particles = [ P.n1, P.d, P.sd4__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd4] ], [ [P.go, P.sd4] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.UVGC_1334_1099,(0,0,1):C.UVGC_1334_1100,(0,0,4):C.UVGC_1334_1101,(0,0,3):C.UVGC_1334_1102,(0,0,2):C.UVGC_1334_1103})

V_489 = CTVertex(name = 'V_489',
                 type = 'UV',
                 particles = [ P.n2, P.d, P.sd4__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd4] ], [ [P.go, P.sd4] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.UVGC_1335_1104,(0,0,1):C.UVGC_1335_1105,(0,0,4):C.UVGC_1335_1106,(0,0,3):C.UVGC_1335_1107,(0,0,2):C.UVGC_1335_1108})

V_490 = CTVertex(name = 'V_490',
                 type = 'UV',
                 particles = [ P.n3, P.d, P.sd4__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd4] ], [ [P.go, P.sd4] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.UVGC_1336_1109,(0,0,1):C.UVGC_1336_1110,(0,0,4):C.UVGC_1336_1111,(0,0,3):C.UVGC_1336_1112,(0,0,2):C.UVGC_1336_1113})

V_491 = CTVertex(name = 'V_491',
                 type = 'UV',
                 particles = [ P.n4, P.d, P.sd4__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd4] ], [ [P.go, P.sd4] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.UVGC_1337_1114,(0,0,1):C.UVGC_1337_1115,(0,0,4):C.UVGC_1337_1116,(0,0,3):C.UVGC_1337_1117,(0,0,2):C.UVGC_1337_1118})

V_492 = CTVertex(name = 'V_492',
                 type = 'UV',
                 particles = [ P.a, P.sd4__tilde__, P.sd4 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS2 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.UVGC_1323_1019,(0,0,1):C.UVGC_1323_1020})

V_493 = CTVertex(name = 'V_493',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.n1, P.sd4 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd4] ], [ [P.go, P.sd4] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.UVGC_1319_999,(0,0,1):C.UVGC_1319_1000,(0,0,4):C.UVGC_1319_1001,(0,0,3):C.UVGC_1319_1002,(0,0,2):C.UVGC_1319_1003})

V_494 = CTVertex(name = 'V_494',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.n2, P.sd4 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd4] ], [ [P.go, P.sd4] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.UVGC_1320_1004,(0,0,1):C.UVGC_1320_1005,(0,0,4):C.UVGC_1320_1006,(0,0,3):C.UVGC_1320_1007,(0,0,2):C.UVGC_1320_1008})

V_495 = CTVertex(name = 'V_495',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.n3, P.sd4 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd4] ], [ [P.go, P.sd4] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.UVGC_1321_1009,(0,0,1):C.UVGC_1321_1010,(0,0,4):C.UVGC_1321_1011,(0,0,3):C.UVGC_1321_1012,(0,0,2):C.UVGC_1321_1013})

V_496 = CTVertex(name = 'V_496',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.n4, P.sd4 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.sd4] ], [ [P.go, P.sd4] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.UVGC_1322_1014,(0,0,1):C.UVGC_1322_1015,(0,0,4):C.UVGC_1322_1016,(0,0,3):C.UVGC_1322_1017,(0,0,2):C.UVGC_1322_1018})

V_497 = CTVertex(name = 'V_497',
                 type = 'UV',
                 particles = [ P.a, P.a, P.sd4__tilde__, P.sd4 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ], [ [P.sd4] ] ],
                 couplings = {(0,0,2):C.UVGC_1324_1021,(0,0,0):C.UVGC_1324_1022,(0,0,1):C.UVGC_1324_1023})

V_498 = CTVertex(name = 'V_498',
                 type = 'UV',
                 particles = [ P.h02, P.sd4__tilde__, P.sd4 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ], [ [P.sd4] ] ],
                 couplings = {(0,0,2):C.UVGC_1921_4039,(0,0,0):C.UVGC_1921_4040,(0,0,1):C.UVGC_1921_4041})

V_499 = CTVertex(name = 'V_499',
                 type = 'UV',
                 particles = [ P.h01, P.sd4__tilde__, P.sd4 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ], [ [P.sd4] ] ],
                 couplings = {(0,0,2):C.UVGC_1920_4036,(0,0,0):C.UVGC_1920_4037,(0,0,1):C.UVGC_1920_4038})

V_500 = CTVertex(name = 'V_500',
                 type = 'UV',
                 particles = [ P.n1, P.s, P.sd5__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd5] ], [ [P.g, P.s] ], [ [P.g, P.sd5] ], [ [P.g, P.s, P.sd5] ] ],
                 couplings = {(0,0,2):C.UVGC_1359_1227,(0,0,3):C.UVGC_1359_1228,(0,0,0):C.UVGC_1359_1229,(0,0,1):C.UVGC_1359_1230,(0,0,4):C.UVGC_1359_1231})

V_501 = CTVertex(name = 'V_501',
                 type = 'UV',
                 particles = [ P.n2, P.s, P.sd5__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd5] ], [ [P.g, P.s] ], [ [P.g, P.sd5] ], [ [P.g, P.s, P.sd5] ] ],
                 couplings = {(0,0,2):C.UVGC_1360_1232,(0,0,3):C.UVGC_1360_1233,(0,0,0):C.UVGC_1360_1234,(0,0,1):C.UVGC_1360_1235,(0,0,4):C.UVGC_1360_1236})

V_502 = CTVertex(name = 'V_502',
                 type = 'UV',
                 particles = [ P.n3, P.s, P.sd5__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd5] ], [ [P.g, P.s] ], [ [P.g, P.sd5] ], [ [P.g, P.s, P.sd5] ] ],
                 couplings = {(0,0,2):C.UVGC_1361_1237,(0,0,3):C.UVGC_1361_1238,(0,0,0):C.UVGC_1361_1239,(0,0,1):C.UVGC_1361_1240,(0,0,4):C.UVGC_1361_1241})

V_503 = CTVertex(name = 'V_503',
                 type = 'UV',
                 particles = [ P.n4, P.s, P.sd5__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd5] ], [ [P.g, P.s] ], [ [P.g, P.sd5] ], [ [P.g, P.s, P.sd5] ] ],
                 couplings = {(0,0,2):C.UVGC_1362_1242,(0,0,3):C.UVGC_1362_1243,(0,0,0):C.UVGC_1362_1244,(0,0,1):C.UVGC_1362_1245,(0,0,4):C.UVGC_1362_1246})

V_504 = CTVertex(name = 'V_504',
                 type = 'UV',
                 particles = [ P.a, P.sd5__tilde__, P.sd5 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS2 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ] ],
                 couplings = {(0,0,1):C.UVGC_1348_1147,(0,0,0):C.UVGC_1348_1148})

V_505 = CTVertex(name = 'V_505',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.n1, P.sd5 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd5] ], [ [P.g, P.s] ], [ [P.g, P.sd5] ], [ [P.g, P.s, P.sd5] ] ],
                 couplings = {(0,0,2):C.UVGC_1344_1127,(0,0,3):C.UVGC_1344_1128,(0,0,0):C.UVGC_1344_1129,(0,0,1):C.UVGC_1344_1130,(0,0,4):C.UVGC_1344_1131})

V_506 = CTVertex(name = 'V_506',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.n2, P.sd5 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd5] ], [ [P.g, P.s] ], [ [P.g, P.sd5] ], [ [P.g, P.s, P.sd5] ] ],
                 couplings = {(0,0,2):C.UVGC_1345_1132,(0,0,3):C.UVGC_1345_1133,(0,0,0):C.UVGC_1345_1134,(0,0,1):C.UVGC_1345_1135,(0,0,4):C.UVGC_1345_1136})

V_507 = CTVertex(name = 'V_507',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.n3, P.sd5 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd5] ], [ [P.g, P.s] ], [ [P.g, P.sd5] ], [ [P.g, P.s, P.sd5] ] ],
                 couplings = {(0,0,2):C.UVGC_1346_1137,(0,0,3):C.UVGC_1346_1138,(0,0,0):C.UVGC_1346_1139,(0,0,1):C.UVGC_1346_1140,(0,0,4):C.UVGC_1346_1141})

V_508 = CTVertex(name = 'V_508',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.n4, P.sd5 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.go, P.sd5] ], [ [P.g, P.s] ], [ [P.g, P.sd5] ], [ [P.g, P.s, P.sd5] ] ],
                 couplings = {(0,0,2):C.UVGC_1347_1142,(0,0,3):C.UVGC_1347_1143,(0,0,0):C.UVGC_1347_1144,(0,0,1):C.UVGC_1347_1145,(0,0,4):C.UVGC_1347_1146})

V_509 = CTVertex(name = 'V_509',
                 type = 'UV',
                 particles = [ P.a, P.a, P.sd5__tilde__, P.sd5 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ], [ [P.sd5] ] ],
                 couplings = {(0,0,2):C.UVGC_1349_1149,(0,0,1):C.UVGC_1349_1150,(0,0,0):C.UVGC_1349_1151})

V_510 = CTVertex(name = 'V_510',
                 type = 'UV',
                 particles = [ P.h02, P.sd5__tilde__, P.sd5 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ], [ [P.sd5] ] ],
                 couplings = {(0,0,2):C.UVGC_1923_4045,(0,0,1):C.UVGC_1923_4046,(0,0,0):C.UVGC_1923_4047})

V_511 = CTVertex(name = 'V_511',
                 type = 'UV',
                 particles = [ P.h01, P.sd5__tilde__, P.sd5 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ], [ [P.sd5] ] ],
                 couplings = {(0,0,2):C.UVGC_1922_4042,(0,0,1):C.UVGC_1922_4043,(0,0,0):C.UVGC_1922_4044})

V_512 = CTVertex(name = 'V_512',
                 type = 'UV',
                 particles = [ P.n1, P.b, P.sd6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,1,0):C.UVGC_1441_1543,(0,1,1):C.UVGC_1441_1544,(0,1,6):C.UVGC_1441_1545,(0,1,4):C.UVGC_1441_1546,(0,1,5):C.UVGC_1441_1547,(0,1,3):C.UVGC_1441_1548,(0,1,2):C.UVGC_1441_1549,(0,0,0):C.UVGC_1445_1571,(0,0,1):C.UVGC_1445_1572,(0,0,6):C.UVGC_1445_1573,(0,0,4):C.UVGC_1445_1574,(0,0,5):C.UVGC_1445_1575,(0,0,3):C.UVGC_1445_1576})

V_513 = CTVertex(name = 'V_513',
                 type = 'UV',
                 particles = [ P.n2, P.b, P.sd6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,1,0):C.UVGC_1442_1550,(0,1,1):C.UVGC_1442_1551,(0,1,6):C.UVGC_1442_1552,(0,1,4):C.UVGC_1442_1553,(0,1,5):C.UVGC_1442_1554,(0,1,3):C.UVGC_1442_1555,(0,1,2):C.UVGC_1442_1556,(0,0,0):C.UVGC_1446_1577,(0,0,1):C.UVGC_1446_1578,(0,0,6):C.UVGC_1446_1579,(0,0,4):C.UVGC_1446_1580,(0,0,5):C.UVGC_1446_1581,(0,0,3):C.UVGC_1446_1582})

V_514 = CTVertex(name = 'V_514',
                 type = 'UV',
                 particles = [ P.n3, P.b, P.sd6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,1,0):C.UVGC_1443_1557,(0,1,1):C.UVGC_1443_1558,(0,1,6):C.UVGC_1443_1559,(0,1,4):C.UVGC_1443_1560,(0,1,5):C.UVGC_1443_1561,(0,1,3):C.UVGC_1443_1562,(0,1,2):C.UVGC_1443_1563,(0,0,0):C.UVGC_1447_1583,(0,0,1):C.UVGC_1447_1584,(0,0,6):C.UVGC_1447_1585,(0,0,4):C.UVGC_1447_1586,(0,0,5):C.UVGC_1447_1587,(0,0,3):C.UVGC_1447_1588})

V_515 = CTVertex(name = 'V_515',
                 type = 'UV',
                 particles = [ P.n4, P.b, P.sd6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,1,0):C.UVGC_1444_1564,(0,1,1):C.UVGC_1444_1565,(0,1,6):C.UVGC_1444_1566,(0,1,4):C.UVGC_1444_1567,(0,1,5):C.UVGC_1444_1568,(0,1,3):C.UVGC_1444_1569,(0,1,2):C.UVGC_1444_1570,(0,0,0):C.UVGC_1448_1589,(0,0,1):C.UVGC_1448_1590,(0,0,6):C.UVGC_1448_1591,(0,0,4):C.UVGC_1448_1592,(0,0,5):C.UVGC_1448_1593,(0,0,3):C.UVGC_1448_1594})

V_516 = CTVertex(name = 'V_516',
                 type = 'UV',
                 particles = [ P.a, P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS2 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1390_1276,(0,0,1):C.UVGC_1390_1277})

V_517 = CTVertex(name = 'V_517',
                 type = 'UV',
                 particles = [ P.h02, P.sd3, P.sd6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ], [ [P.g, P.sd6] ], [ [P.sd3, P.sd6], [P.g, P.sd3, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1930_4084,(0,0,1):C.UVGC_1930_4085,(0,0,4):C.UVGC_1930_4086,(0,0,5):C.UVGC_1930_4087,(0,0,2):C.UVGC_1930_4088,(0,0,3):C.UVGC_1930_4089,(0,0,6):C.UVGC_1930_4090})

V_518 = CTVertex(name = 'V_518',
                 type = 'UV',
                 particles = [ P.h01, P.sd3, P.sd6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ], [ [P.g, P.sd6] ], [ [P.sd3, P.sd6], [P.g, P.sd3, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1926_4061,(0,0,1):C.UVGC_1926_4062,(0,0,4):C.UVGC_1926_4063,(0,0,5):C.UVGC_1926_4064,(0,0,2):C.UVGC_1926_4065,(0,0,3):C.UVGC_1926_4066,(0,0,6):C.UVGC_1926_4067})

V_519 = CTVertex(name = 'V_519',
                 type = 'UV',
                 particles = [ P.G0, P.sd3, P.sd6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ], [ [P.g, P.sd6] ], [ [P.sd3, P.sd6], [P.g, P.sd3, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_2193_4240,(0,0,1):C.UVGC_2193_4241,(0,0,4):C.UVGC_2193_4242,(0,0,5):C.UVGC_2193_4243,(0,0,2):C.UVGC_2193_4244,(0,0,3):C.UVGC_2193_4245,(0,0,6):C.UVGC_2193_4246})

V_520 = CTVertex(name = 'V_520',
                 type = 'UV',
                 particles = [ P.A0, P.sd3, P.sd6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ], [ [P.g, P.sd6] ], [ [P.sd3, P.sd6], [P.g, P.sd3, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_2190_4223,(0,0,1):C.UVGC_2190_4224,(0,0,4):C.UVGC_2190_4225,(0,0,5):C.UVGC_2190_4226,(0,0,2):C.UVGC_2190_4227,(0,0,3):C.UVGC_2190_4228,(0,0,6):C.UVGC_2190_4229})

V_521 = CTVertex(name = 'V_521',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.n1, P.sd6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1426_1445,(0,0,1):C.UVGC_1426_1446,(0,0,6):C.UVGC_1426_1447,(0,0,4):C.UVGC_1426_1448,(0,0,5):C.UVGC_1426_1449,(0,0,3):C.UVGC_1426_1450,(0,0,2):C.UVGC_1426_1451,(0,1,0):C.UVGC_1415_1381,(0,1,1):C.UVGC_1415_1382,(0,1,6):C.UVGC_1415_1383,(0,1,4):C.UVGC_1415_1384,(0,1,5):C.UVGC_1415_1385,(0,1,3):C.UVGC_1415_1386})

V_522 = CTVertex(name = 'V_522',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.n2, P.sd6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1428_1458,(0,0,1):C.UVGC_1428_1459,(0,0,6):C.UVGC_1428_1460,(0,0,4):C.UVGC_1428_1461,(0,0,5):C.UVGC_1428_1462,(0,0,3):C.UVGC_1428_1463,(0,0,2):C.UVGC_1428_1464,(0,1,0):C.UVGC_1416_1387,(0,1,1):C.UVGC_1416_1388,(0,1,6):C.UVGC_1416_1389,(0,1,4):C.UVGC_1416_1390,(0,1,5):C.UVGC_1416_1391,(0,1,3):C.UVGC_1416_1392})

V_523 = CTVertex(name = 'V_523',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.n3, P.sd6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1430_1471,(0,0,1):C.UVGC_1430_1472,(0,0,6):C.UVGC_1430_1473,(0,0,4):C.UVGC_1430_1474,(0,0,5):C.UVGC_1430_1475,(0,0,3):C.UVGC_1430_1476,(0,0,2):C.UVGC_1430_1477,(0,1,0):C.UVGC_1417_1393,(0,1,1):C.UVGC_1417_1394,(0,1,6):C.UVGC_1417_1395,(0,1,4):C.UVGC_1417_1396,(0,1,5):C.UVGC_1417_1397,(0,1,3):C.UVGC_1417_1398})

V_524 = CTVertex(name = 'V_524',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.n4, P.sd6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.sd3] ], [ [P.b, P.g, P.sd6] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1432_1484,(0,0,1):C.UVGC_1432_1485,(0,0,6):C.UVGC_1432_1486,(0,0,4):C.UVGC_1432_1487,(0,0,5):C.UVGC_1432_1488,(0,0,3):C.UVGC_1432_1489,(0,0,2):C.UVGC_1432_1490,(0,1,0):C.UVGC_1418_1399,(0,1,1):C.UVGC_1418_1400,(0,1,6):C.UVGC_1418_1401,(0,1,4):C.UVGC_1418_1402,(0,1,5):C.UVGC_1418_1403,(0,1,3):C.UVGC_1418_1404})

V_525 = CTVertex(name = 'V_525',
                 type = 'UV',
                 particles = [ P.a, P.a, P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ], [ [P.sd6] ] ],
                 couplings = {(0,0,2):C.UVGC_1391_1278,(0,0,0):C.UVGC_1391_1279,(0,0,1):C.UVGC_1391_1280})

V_526 = CTVertex(name = 'V_526',
                 type = 'UV',
                 particles = [ P.h02, P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd6] ], [ [P.sd6] ] ],
                 couplings = {(0,0,5):C.UVGC_1925_4055,(0,0,0):C.UVGC_1925_4056,(0,0,1):C.UVGC_1925_4057,(0,0,4):C.UVGC_1925_4058,(0,0,2):C.UVGC_1925_4059,(0,0,3):C.UVGC_1925_4060})

V_527 = CTVertex(name = 'V_527',
                 type = 'UV',
                 particles = [ P.h01, P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd6] ], [ [P.sd6] ] ],
                 couplings = {(0,0,5):C.UVGC_1931_4091,(0,0,0):C.UVGC_1931_4092,(0,0,1):C.UVGC_1931_4093,(0,0,4):C.UVGC_1931_4094,(0,0,2):C.UVGC_1931_4095,(0,0,3):C.UVGC_1931_4096})

V_528 = CTVertex(name = 'V_528',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.h01 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS2 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1927_4068,(0,0,1):C.UVGC_1927_4069,(0,0,2):C.UVGC_1927_4070})

V_529 = CTVertex(name = 'V_529',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.h02 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS2 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_1940_4121,(0,0,0):C.UVGC_1940_4122,(0,0,1):C.UVGC_1940_4123})

V_530 = CTVertex(name = 'V_530',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.A0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_2191_4230,(0,0,1):C.UVGC_2191_4231,(0,0,2):C.UVGC_2191_4232})

V_531 = CTVertex(name = 'V_531',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1642_2893,(0,0,6):C.UVGC_1642_2894,(0,0,2):C.UVGC_1642_2895,(0,0,3):C.UVGC_1642_2896,(0,0,4):C.UVGC_1642_2897,(0,0,5):C.UVGC_1642_2898,(0,0,1):C.UVGC_1642_2899,(0,1,0):C.UVGC_2206_4319,(0,1,6):C.UVGC_2206_4320,(0,1,2):C.UVGC_2206_4321,(0,1,3):C.UVGC_2206_4322,(0,1,4):C.UVGC_2206_4323,(0,1,5):C.UVGC_2206_4324,(0,1,1):C.UVGC_2206_4325})

V_532 = CTVertex(name = 'V_532',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.H__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1641_2886,(0,0,6):C.UVGC_1641_2887,(0,0,2):C.UVGC_1641_2888,(0,0,3):C.UVGC_1641_2889,(0,0,4):C.UVGC_1641_2890,(0,0,5):C.UVGC_1641_2891,(0,0,1):C.UVGC_1641_2892,(0,1,0):C.UVGC_2202_4295,(0,1,6):C.UVGC_2202_4296,(0,1,2):C.UVGC_2202_4297,(0,1,3):C.UVGC_2202_4298,(0,1,4):C.UVGC_2202_4299,(0,1,5):C.UVGC_2202_4300,(0,1,1):C.UVGC_2202_4301})

V_533 = CTVertex(name = 'V_533',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS1 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2203_4302,(0,0,0):C.UVGC_2203_4303,(0,0,1):C.UVGC_2203_4304})

V_534 = CTVertex(name = 'V_534',
                 type = 'UV',
                 particles = [ P.n1, P.u, P.su1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su1] ], [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.g, P.su1, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1479_1755,(0,0,4):C.UVGC_1479_1756,(0,0,0):C.UVGC_1479_1757,(0,0,1):C.UVGC_1479_1758,(0,0,3):C.UVGC_1479_1759})

V_535 = CTVertex(name = 'V_535',
                 type = 'UV',
                 particles = [ P.n2, P.u, P.su1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su1] ], [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.g, P.su1, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1480_1760,(0,0,4):C.UVGC_1480_1761,(0,0,0):C.UVGC_1480_1762,(0,0,1):C.UVGC_1480_1763,(0,0,3):C.UVGC_1480_1764})

V_536 = CTVertex(name = 'V_536',
                 type = 'UV',
                 particles = [ P.n3, P.u, P.su1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su1] ], [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.g, P.su1, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1481_1765,(0,0,4):C.UVGC_1481_1766,(0,0,0):C.UVGC_1481_1767,(0,0,1):C.UVGC_1481_1768,(0,0,3):C.UVGC_1481_1769})

V_537 = CTVertex(name = 'V_537',
                 type = 'UV',
                 particles = [ P.n4, P.u, P.su1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su1] ], [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.g, P.su1, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1482_1770,(0,0,4):C.UVGC_1482_1771,(0,0,0):C.UVGC_1482_1772,(0,0,1):C.UVGC_1482_1773,(0,0,3):C.UVGC_1482_1774})

V_538 = CTVertex(name = 'V_538',
                 type = 'UV',
                 particles = [ P.a, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS2 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,1):C.UVGC_1472_1697,(0,0,0):C.UVGC_1472_1698})

V_539 = CTVertex(name = 'V_539',
                 type = 'UV',
                 particles = [ P.G__plus__, P.sd1, P.su1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ], [ [P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_2197_4265,(0,0,2):C.UVGC_2197_4266,(0,0,4):C.UVGC_2197_4267,(0,0,1):C.UVGC_2197_4268,(0,0,5):C.UVGC_2197_4269,(0,0,3):C.UVGC_2197_4270})

V_540 = CTVertex(name = 'V_540',
                 type = 'UV',
                 particles = [ P.H__plus__, P.sd1, P.su1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ], [ [P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_2196_4259,(0,0,2):C.UVGC_2196_4260,(0,0,4):C.UVGC_2196_4261,(0,0,1):C.UVGC_2196_4262,(0,0,5):C.UVGC_2196_4263,(0,0,3):C.UVGC_2196_4264})

V_541 = CTVertex(name = 'V_541',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.n1, P.su1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.su1] ], [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.g, P.su1, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1463_1640,(0,0,4):C.UVGC_1463_1641,(0,0,0):C.UVGC_1463_1642,(0,0,1):C.UVGC_1463_1643,(0,0,3):C.UVGC_1463_1644})

V_542 = CTVertex(name = 'V_542',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.n2, P.su1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.su1] ], [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.g, P.su1, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1464_1645,(0,0,4):C.UVGC_1464_1646,(0,0,0):C.UVGC_1464_1647,(0,0,1):C.UVGC_1464_1648,(0,0,3):C.UVGC_1464_1649})

V_543 = CTVertex(name = 'V_543',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.n3, P.su1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.su1] ], [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.g, P.su1, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1465_1650,(0,0,4):C.UVGC_1465_1651,(0,0,0):C.UVGC_1465_1652,(0,0,1):C.UVGC_1465_1653,(0,0,3):C.UVGC_1465_1654})

V_544 = CTVertex(name = 'V_544',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.n4, P.su1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.su1] ], [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.g, P.su1, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1466_1655,(0,0,4):C.UVGC_1466_1656,(0,0,0):C.UVGC_1466_1657,(0,0,1):C.UVGC_1466_1658,(0,0,3):C.UVGC_1466_1659})

V_545 = CTVertex(name = 'V_545',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.x1__minus__, P.su1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.su1] ], [ [P.go, P.sd1] ], [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_1495_1846,(0,0,4):C.UVGC_1495_1847,(0,0,2):C.UVGC_1495_1848,(0,0,3):C.UVGC_1495_1849,(0,0,1):C.UVGC_1495_1850})

V_546 = CTVertex(name = 'V_546',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.x2__minus__, P.su1 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.su1] ], [ [P.go, P.sd1] ], [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_1496_1851,(0,0,4):C.UVGC_1496_1852,(0,0,2):C.UVGC_1496_1853,(0,0,3):C.UVGC_1496_1854,(0,0,1):C.UVGC_1496_1855})

V_547 = CTVertex(name = 'V_547',
                 type = 'UV',
                 particles = [ P.G__minus__, P.sd1__tilde__, P.su1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ], [ [P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_2195_4253,(0,0,2):C.UVGC_2195_4254,(0,0,4):C.UVGC_2195_4255,(0,0,1):C.UVGC_2195_4256,(0,0,5):C.UVGC_2195_4257,(0,0,3):C.UVGC_2195_4258})

V_548 = CTVertex(name = 'V_548',
                 type = 'UV',
                 particles = [ P.H__minus__, P.sd1__tilde__, P.su1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ], [ [P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_2194_4247,(0,0,2):C.UVGC_2194_4248,(0,0,4):C.UVGC_2194_4249,(0,0,1):C.UVGC_2194_4250,(0,0,5):C.UVGC_2194_4251,(0,0,3):C.UVGC_2194_4252})

V_549 = CTVertex(name = 'V_549',
                 type = 'UV',
                 particles = [ P.a, P.a, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.su1] ] ],
                 couplings = {(0,0,2):C.UVGC_1473_1699,(0,0,1):C.UVGC_1473_1700,(0,0,0):C.UVGC_1473_1701})

V_550 = CTVertex(name = 'V_550',
                 type = 'UV',
                 particles = [ P.h02, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.su1] ] ],
                 couplings = {(0,0,2):C.UVGC_1933_4100,(0,0,1):C.UVGC_1933_4101,(0,0,0):C.UVGC_1933_4102})

V_551 = CTVertex(name = 'V_551',
                 type = 'UV',
                 particles = [ P.h01, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.su1] ] ],
                 couplings = {(0,0,2):C.UVGC_1932_4097,(0,0,1):C.UVGC_1932_4098,(0,0,0):C.UVGC_1932_4099})

V_552 = CTVertex(name = 'V_552',
                 type = 'UV',
                 particles = [ P.n1, P.c, P.su2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su2] ], [ [P.go, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1526_2011,(0,0,1):C.UVGC_1526_2012,(0,0,4):C.UVGC_1526_2013,(0,0,3):C.UVGC_1526_2014,(0,0,2):C.UVGC_1526_2015})

V_553 = CTVertex(name = 'V_553',
                 type = 'UV',
                 particles = [ P.n2, P.c, P.su2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su2] ], [ [P.go, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1527_2016,(0,0,1):C.UVGC_1527_2017,(0,0,4):C.UVGC_1527_2018,(0,0,3):C.UVGC_1527_2019,(0,0,2):C.UVGC_1527_2020})

V_554 = CTVertex(name = 'V_554',
                 type = 'UV',
                 particles = [ P.n3, P.c, P.su2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su2] ], [ [P.go, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1528_2021,(0,0,1):C.UVGC_1528_2022,(0,0,4):C.UVGC_1528_2023,(0,0,3):C.UVGC_1528_2024,(0,0,2):C.UVGC_1528_2025})

V_555 = CTVertex(name = 'V_555',
                 type = 'UV',
                 particles = [ P.n4, P.c, P.su2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su2] ], [ [P.go, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1529_2026,(0,0,1):C.UVGC_1529_2027,(0,0,4):C.UVGC_1529_2028,(0,0,3):C.UVGC_1529_2029,(0,0,2):C.UVGC_1529_2030})

V_556 = CTVertex(name = 'V_556',
                 type = 'UV',
                 particles = [ P.a, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1519_1953,(0,0,1):C.UVGC_1519_1954,(0,1,0):C.UVGC_1518_1951,(0,1,1):C.UVGC_1518_1952})

V_557 = CTVertex(name = 'V_557',
                 type = 'UV',
                 particles = [ P.G__plus__, P.sd2, P.su2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ], [ [P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_2201_4289,(0,0,2):C.UVGC_2201_4290,(0,0,4):C.UVGC_2201_4291,(0,0,1):C.UVGC_2201_4292,(0,0,5):C.UVGC_2201_4293,(0,0,3):C.UVGC_2201_4294})

V_558 = CTVertex(name = 'V_558',
                 type = 'UV',
                 particles = [ P.H__plus__, P.sd2, P.su2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ], [ [P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_2200_4283,(0,0,2):C.UVGC_2200_4284,(0,0,4):C.UVGC_2200_4285,(0,0,1):C.UVGC_2200_4286,(0,0,5):C.UVGC_2200_4287,(0,0,3):C.UVGC_2200_4288})

V_559 = CTVertex(name = 'V_559',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.n1, P.su2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su2] ], [ [P.go, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1509_1894,(0,0,1):C.UVGC_1509_1895,(0,0,4):C.UVGC_1509_1896,(0,0,3):C.UVGC_1509_1897,(0,0,2):C.UVGC_1509_1898})

V_560 = CTVertex(name = 'V_560',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.n2, P.su2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su2] ], [ [P.go, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1510_1899,(0,0,1):C.UVGC_1510_1900,(0,0,4):C.UVGC_1510_1901,(0,0,3):C.UVGC_1510_1902,(0,0,2):C.UVGC_1510_1903})

V_561 = CTVertex(name = 'V_561',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.n3, P.su2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su2] ], [ [P.go, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1511_1904,(0,0,1):C.UVGC_1511_1905,(0,0,4):C.UVGC_1511_1906,(0,0,3):C.UVGC_1511_1907,(0,0,2):C.UVGC_1511_1908})

V_562 = CTVertex(name = 'V_562',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.n4, P.su2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su2] ], [ [P.go, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1512_1909,(0,0,1):C.UVGC_1512_1910,(0,0,4):C.UVGC_1512_1911,(0,0,3):C.UVGC_1512_1912,(0,0,2):C.UVGC_1512_1913})

V_563 = CTVertex(name = 'V_563',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.x1__minus__, P.su2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.go, P.sd2] ], [ [P.g, P.s] ], [ [P.g, P.s, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1542_2102,(0,0,2):C.UVGC_1542_2103,(0,0,4):C.UVGC_1542_2104,(0,0,1):C.UVGC_1542_2105,(0,0,3):C.UVGC_1542_2106})

V_564 = CTVertex(name = 'V_564',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.x2__minus__, P.su2 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.go, P.sd2] ], [ [P.g, P.s] ], [ [P.g, P.s, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1543_2107,(0,0,2):C.UVGC_1543_2108,(0,0,4):C.UVGC_1543_2109,(0,0,1):C.UVGC_1543_2110,(0,0,3):C.UVGC_1543_2111})

V_565 = CTVertex(name = 'V_565',
                 type = 'UV',
                 particles = [ P.G__minus__, P.sd2__tilde__, P.su2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ], [ [P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_2199_4277,(0,0,2):C.UVGC_2199_4278,(0,0,4):C.UVGC_2199_4279,(0,0,1):C.UVGC_2199_4280,(0,0,5):C.UVGC_2199_4281,(0,0,3):C.UVGC_2199_4282})

V_566 = CTVertex(name = 'V_566',
                 type = 'UV',
                 particles = [ P.H__minus__, P.sd2__tilde__, P.su2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ], [ [P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_2198_4271,(0,0,2):C.UVGC_2198_4272,(0,0,4):C.UVGC_2198_4273,(0,0,1):C.UVGC_2198_4274,(0,0,5):C.UVGC_2198_4275,(0,0,3):C.UVGC_2198_4276})

V_567 = CTVertex(name = 'V_567',
                 type = 'UV',
                 particles = [ P.a, P.a, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ], [ [P.su2] ] ],
                 couplings = {(0,0,2):C.UVGC_1520_1955,(0,0,0):C.UVGC_1520_1956,(0,0,1):C.UVGC_1520_1957})

V_568 = CTVertex(name = 'V_568',
                 type = 'UV',
                 particles = [ P.h02, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ], [ [P.su2] ] ],
                 couplings = {(0,0,2):C.UVGC_1935_4106,(0,0,0):C.UVGC_1935_4107,(0,0,1):C.UVGC_1935_4108})

V_569 = CTVertex(name = 'V_569',
                 type = 'UV',
                 particles = [ P.h01, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ], [ [P.su2] ] ],
                 couplings = {(0,0,2):C.UVGC_1934_4103,(0,0,0):C.UVGC_1934_4104,(0,0,1):C.UVGC_1934_4105})

V_570 = CTVertex(name = 'V_570',
                 type = 'UV',
                 particles = [ P.n1, P.t, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.su6, P.t] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su3, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,4):C.UVGC_1688_3200,(0,0,6):C.UVGC_1688_3201,(0,0,0):C.UVGC_1688_3202,(0,0,1):C.UVGC_1688_3203,(0,0,3):C.UVGC_1688_3204,(0,0,5):C.UVGC_1688_3205,(0,0,2):C.UVGC_1688_3206,(0,1,4):C.UVGC_1682_3164,(0,1,6):C.UVGC_1682_3165,(0,1,0):C.UVGC_1682_3166,(0,1,1):C.UVGC_1682_3167,(0,1,3):C.UVGC_1682_3168,(0,1,5):C.UVGC_1682_3169})

V_571 = CTVertex(name = 'V_571',
                 type = 'UV',
                 particles = [ P.n2, P.t, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.su6, P.t] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su3, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,4):C.UVGC_1689_3207,(0,0,6):C.UVGC_1689_3208,(0,0,0):C.UVGC_1689_3209,(0,0,1):C.UVGC_1689_3210,(0,0,3):C.UVGC_1689_3211,(0,0,5):C.UVGC_1689_3212,(0,0,2):C.UVGC_1689_3213,(0,1,4):C.UVGC_1683_3170,(0,1,6):C.UVGC_1683_3171,(0,1,0):C.UVGC_1683_3172,(0,1,1):C.UVGC_1683_3173,(0,1,3):C.UVGC_1683_3174,(0,1,5):C.UVGC_1683_3175})

V_572 = CTVertex(name = 'V_572',
                 type = 'UV',
                 particles = [ P.n3, P.t, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.su6, P.t] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su3, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,4):C.UVGC_1690_3214,(0,0,6):C.UVGC_1690_3215,(0,0,0):C.UVGC_1690_3216,(0,0,1):C.UVGC_1690_3217,(0,0,3):C.UVGC_1690_3218,(0,0,5):C.UVGC_1690_3219,(0,0,2):C.UVGC_1690_3220,(0,1,4):C.UVGC_1684_3176,(0,1,6):C.UVGC_1684_3177,(0,1,0):C.UVGC_1684_3178,(0,1,1):C.UVGC_1684_3179,(0,1,3):C.UVGC_1684_3180,(0,1,5):C.UVGC_1684_3181})

V_573 = CTVertex(name = 'V_573',
                 type = 'UV',
                 particles = [ P.n4, P.t, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.su6, P.t] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su3, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,4):C.UVGC_1691_3221,(0,0,6):C.UVGC_1691_3222,(0,0,0):C.UVGC_1691_3223,(0,0,1):C.UVGC_1691_3224,(0,0,3):C.UVGC_1691_3225,(0,0,5):C.UVGC_1691_3226,(0,0,2):C.UVGC_1691_3227,(0,1,4):C.UVGC_1685_3182,(0,1,6):C.UVGC_1685_3183,(0,1,0):C.UVGC_1685_3184,(0,1,1):C.UVGC_1685_3185,(0,1,3):C.UVGC_1685_3186,(0,1,5):C.UVGC_1685_3187})

V_574 = CTVertex(name = 'V_574',
                 type = 'UV',
                 particles = [ P.a, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.UVGC_1655_2987,(0,0,0):C.UVGC_1655_2988,(0,1,1):C.UVGC_1654_2985,(0,1,0):C.UVGC_1654_2986})

V_575 = CTVertex(name = 'V_575',
                 type = 'UV',
                 particles = [ P.G__plus__, P.sd3, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ], [ [P.g, P.t] ], [ [P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_2216_4403,(0,0,1):C.UVGC_2216_4404,(0,0,8):C.UVGC_2216_4405,(0,0,10):C.UVGC_2216_4406,(0,0,11):C.UVGC_2216_4407,(0,0,3):C.UVGC_2216_4408,(0,0,4):C.UVGC_2216_4409,(0,0,5):C.UVGC_2216_4410,(0,0,6):C.UVGC_2216_4411,(0,0,7):C.UVGC_2216_4412,(0,0,12):C.UVGC_2216_4413,(0,0,2):C.UVGC_2216_4414,(0,0,9):C.UVGC_2216_4415})

V_576 = CTVertex(name = 'V_576',
                 type = 'UV',
                 particles = [ P.H__plus__, P.sd3, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ], [ [P.g, P.t] ], [ [P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_2212_4371,(0,0,1):C.UVGC_2212_4372,(0,0,8):C.UVGC_2212_4373,(0,0,10):C.UVGC_2212_4374,(0,0,11):C.UVGC_2212_4375,(0,0,3):C.UVGC_2212_4376,(0,0,4):C.UVGC_2212_4377,(0,0,5):C.UVGC_2212_4378,(0,0,6):C.UVGC_2212_4379,(0,0,7):C.UVGC_2212_4380,(0,0,12):C.UVGC_2212_4381,(0,0,2):C.UVGC_2212_4382,(0,0,9):C.UVGC_2212_4383})

V_577 = CTVertex(name = 'V_577',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.n1, P.su3 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.su6, P.t] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su3, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,4):C.UVGC_1673_3111,(0,1,6):C.UVGC_1673_3112,(0,1,0):C.UVGC_1673_3113,(0,1,1):C.UVGC_1673_3114,(0,1,3):C.UVGC_1673_3115,(0,1,5):C.UVGC_1673_3116,(0,1,2):C.UVGC_1673_3117,(0,0,4):C.UVGC_1677_3139,(0,0,6):C.UVGC_1677_3140,(0,0,0):C.UVGC_1677_3141,(0,0,1):C.UVGC_1677_3142,(0,0,3):C.UVGC_1677_3143,(0,0,5):C.UVGC_1677_3144})

V_578 = CTVertex(name = 'V_578',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.n2, P.su3 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.su6, P.t] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su3, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,4):C.UVGC_1674_3118,(0,1,6):C.UVGC_1674_3119,(0,1,0):C.UVGC_1674_3120,(0,1,1):C.UVGC_1674_3121,(0,1,3):C.UVGC_1674_3122,(0,1,5):C.UVGC_1674_3123,(0,1,2):C.UVGC_1674_3124,(0,0,4):C.UVGC_1678_3145,(0,0,6):C.UVGC_1678_3146,(0,0,0):C.UVGC_1678_3147,(0,0,1):C.UVGC_1678_3148,(0,0,3):C.UVGC_1678_3149,(0,0,5):C.UVGC_1678_3150})

V_579 = CTVertex(name = 'V_579',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.n3, P.su3 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.su6, P.t] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su3, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,4):C.UVGC_1675_3125,(0,1,6):C.UVGC_1675_3126,(0,1,0):C.UVGC_1675_3127,(0,1,1):C.UVGC_1675_3128,(0,1,3):C.UVGC_1675_3129,(0,1,5):C.UVGC_1675_3130,(0,1,2):C.UVGC_1675_3131,(0,0,4):C.UVGC_1679_3151,(0,0,6):C.UVGC_1679_3152,(0,0,0):C.UVGC_1679_3153,(0,0,1):C.UVGC_1679_3154,(0,0,3):C.UVGC_1679_3155,(0,0,5):C.UVGC_1679_3156})

V_580 = CTVertex(name = 'V_580',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.n4, P.su3 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.su6, P.t] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su3, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,4):C.UVGC_1676_3132,(0,1,6):C.UVGC_1676_3133,(0,1,0):C.UVGC_1676_3134,(0,1,1):C.UVGC_1676_3135,(0,1,3):C.UVGC_1676_3136,(0,1,5):C.UVGC_1676_3137,(0,1,2):C.UVGC_1676_3138,(0,0,4):C.UVGC_1680_3157,(0,0,6):C.UVGC_1680_3158,(0,0,0):C.UVGC_1680_3159,(0,0,1):C.UVGC_1680_3160,(0,0,3):C.UVGC_1680_3161,(0,0,5):C.UVGC_1680_3162})

V_581 = CTVertex(name = 'V_581',
                 type = 'UV',
                 particles = [ P.x1__minus__, P.b__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.su3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.sd6, P.t] ], [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,6):C.UVGC_1694_3242,(0,0,5):C.UVGC_1694_3243,(0,0,1):C.UVGC_1694_3244,(0,1,0):C.UVGC_1647_2934,(0,1,6):C.UVGC_1647_2935,(0,1,2):C.UVGC_1647_2936,(0,1,3):C.UVGC_1647_2937,(0,1,5):C.UVGC_1647_2938,(0,1,1):C.UVGC_1647_2939,(0,1,4):C.UVGC_1647_2940})

V_582 = CTVertex(name = 'V_582',
                 type = 'UV',
                 particles = [ P.x2__minus__, P.b__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.su3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.sd6, P.t] ], [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,6):C.UVGC_1695_3245,(0,0,5):C.UVGC_1695_3246,(0,0,1):C.UVGC_1695_3247,(0,1,0):C.UVGC_1648_2941,(0,1,6):C.UVGC_1648_2942,(0,1,2):C.UVGC_1648_2943,(0,1,3):C.UVGC_1648_2944,(0,1,5):C.UVGC_1648_2945,(0,1,1):C.UVGC_1648_2946,(0,1,4):C.UVGC_1648_2947})

V_583 = CTVertex(name = 'V_583',
                 type = 'UV',
                 particles = [ P.G__minus__, P.sd3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ], [ [P.g, P.t] ], [ [P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_2207_4326,(0,0,1):C.UVGC_2207_4327,(0,0,8):C.UVGC_2207_4328,(0,0,10):C.UVGC_2207_4329,(0,0,11):C.UVGC_2207_4330,(0,0,3):C.UVGC_2207_4331,(0,0,4):C.UVGC_2207_4332,(0,0,5):C.UVGC_2207_4333,(0,0,6):C.UVGC_2207_4334,(0,0,7):C.UVGC_2207_4335,(0,0,12):C.UVGC_2207_4336,(0,0,2):C.UVGC_2207_4337,(0,0,9):C.UVGC_2207_4338})

V_584 = CTVertex(name = 'V_584',
                 type = 'UV',
                 particles = [ P.H__minus__, P.sd3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ], [ [P.g, P.t] ], [ [P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_2208_4339,(0,0,1):C.UVGC_2208_4340,(0,0,8):C.UVGC_2208_4341,(0,0,10):C.UVGC_2208_4342,(0,0,11):C.UVGC_2208_4343,(0,0,3):C.UVGC_2208_4344,(0,0,4):C.UVGC_2208_4345,(0,0,5):C.UVGC_2208_4346,(0,0,6):C.UVGC_2208_4347,(0,0,7):C.UVGC_2208_4348,(0,0,12):C.UVGC_2208_4349,(0,0,2):C.UVGC_2208_4350,(0,0,9):C.UVGC_2208_4351})

V_585 = CTVertex(name = 'V_585',
                 type = 'UV',
                 particles = [ P.G__minus__, P.sd6__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.t] ], [ [P.g, P.sd6] ], [ [P.g, P.su3] ], [ [P.sd6, P.su3], [P.g, P.sd6, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_2209_4352,(0,0,1):C.UVGC_2209_4353,(0,0,6):C.UVGC_2209_4354,(0,0,7):C.UVGC_2209_4355,(0,0,3):C.UVGC_2209_4356,(0,0,4):C.UVGC_2209_4357,(0,0,5):C.UVGC_2209_4358,(0,0,8):C.UVGC_2209_4359,(0,0,2):C.UVGC_2209_4360})

V_586 = CTVertex(name = 'V_586',
                 type = 'UV',
                 particles = [ P.H__minus__, P.sd6__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.t] ], [ [P.g, P.sd6] ], [ [P.g, P.su3] ], [ [P.sd6, P.su3], [P.g, P.sd6, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_2214_4393,(0,0,1):C.UVGC_2214_4394,(0,0,6):C.UVGC_2214_4395,(0,0,7):C.UVGC_2214_4396,(0,0,3):C.UVGC_2214_4397,(0,0,4):C.UVGC_2214_4398,(0,0,5):C.UVGC_2214_4399,(0,0,8):C.UVGC_2214_4400,(0,0,2):C.UVGC_2214_4401})

V_587 = CTVertex(name = 'V_587',
                 type = 'UV',
                 particles = [ P.a, P.a, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.su3] ] ],
                 couplings = {(0,0,2):C.UVGC_1656_2989,(0,0,1):C.UVGC_1656_2990,(0,0,0):C.UVGC_1656_2991})

V_588 = CTVertex(name = 'V_588',
                 type = 'UV',
                 particles = [ P.h02, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.t] ], [ [P.su3] ] ],
                 couplings = {(0,0,5):C.UVGC_1942_4130,(0,0,3):C.UVGC_1942_4131,(0,0,4):C.UVGC_1942_4132,(0,0,0):C.UVGC_1942_4133,(0,0,1):C.UVGC_1942_4134,(0,0,2):C.UVGC_1942_4135})

V_589 = CTVertex(name = 'V_589',
                 type = 'UV',
                 particles = [ P.h01, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.t] ], [ [P.su3] ] ],
                 couplings = {(0,0,5):C.UVGC_1941_4124,(0,0,3):C.UVGC_1941_4125,(0,0,4):C.UVGC_1941_4126,(0,0,0):C.UVGC_1941_4127,(0,0,1):C.UVGC_1941_4128,(0,0,2):C.UVGC_1941_4129})

V_590 = CTVertex(name = 'V_590',
                 type = 'UV',
                 particles = [ P.n1, P.u, P.su4__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.su4] ], [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.g, P.su4, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1566_2222,(0,0,4):C.UVGC_1566_2223,(0,0,0):C.UVGC_1566_2224,(0,0,1):C.UVGC_1566_2225,(0,0,3):C.UVGC_1566_2226})

V_591 = CTVertex(name = 'V_591',
                 type = 'UV',
                 particles = [ P.n2, P.u, P.su4__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.su4] ], [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.g, P.su4, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1567_2227,(0,0,4):C.UVGC_1567_2228,(0,0,0):C.UVGC_1567_2229,(0,0,1):C.UVGC_1567_2230,(0,0,3):C.UVGC_1567_2231})

V_592 = CTVertex(name = 'V_592',
                 type = 'UV',
                 particles = [ P.n3, P.u, P.su4__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.su4] ], [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.g, P.su4, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1568_2232,(0,0,4):C.UVGC_1568_2233,(0,0,0):C.UVGC_1568_2234,(0,0,1):C.UVGC_1568_2235,(0,0,3):C.UVGC_1568_2236})

V_593 = CTVertex(name = 'V_593',
                 type = 'UV',
                 particles = [ P.n4, P.u, P.su4__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.go, P.su4] ], [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.g, P.su4, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1569_2237,(0,0,4):C.UVGC_1569_2238,(0,0,0):C.UVGC_1569_2239,(0,0,1):C.UVGC_1569_2240,(0,0,3):C.UVGC_1569_2241})

V_594 = CTVertex(name = 'V_594',
                 type = 'UV',
                 particles = [ P.a, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ] ],
                 couplings = {(0,0,1):C.UVGC_1555_2142,(0,0,0):C.UVGC_1555_2143,(0,1,1):C.UVGC_1554_2140,(0,1,0):C.UVGC_1554_2141})

V_595 = CTVertex(name = 'V_595',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.n1, P.su4 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su4] ], [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.g, P.su4, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1550_2120,(0,0,4):C.UVGC_1550_2121,(0,0,0):C.UVGC_1550_2122,(0,0,1):C.UVGC_1550_2123,(0,0,3):C.UVGC_1550_2124})

V_596 = CTVertex(name = 'V_596',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.n2, P.su4 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su4] ], [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.g, P.su4, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1551_2125,(0,0,4):C.UVGC_1551_2126,(0,0,0):C.UVGC_1551_2127,(0,0,1):C.UVGC_1551_2128,(0,0,3):C.UVGC_1551_2129})

V_597 = CTVertex(name = 'V_597',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.n3, P.su4 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su4] ], [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.g, P.su4, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1552_2130,(0,0,4):C.UVGC_1552_2131,(0,0,0):C.UVGC_1552_2132,(0,0,1):C.UVGC_1552_2133,(0,0,3):C.UVGC_1552_2134})

V_598 = CTVertex(name = 'V_598',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.n4, P.su4 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su4] ], [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.g, P.su4, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1553_2135,(0,0,4):C.UVGC_1553_2136,(0,0,0):C.UVGC_1553_2137,(0,0,1):C.UVGC_1553_2138,(0,0,3):C.UVGC_1553_2139})

V_599 = CTVertex(name = 'V_599',
                 type = 'UV',
                 particles = [ P.a, P.a, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.su4] ] ],
                 couplings = {(0,0,2):C.UVGC_1556_2144,(0,0,1):C.UVGC_1556_2145,(0,0,0):C.UVGC_1556_2146})

V_600 = CTVertex(name = 'V_600',
                 type = 'UV',
                 particles = [ P.h02, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.su4] ] ],
                 couplings = {(0,0,2):C.UVGC_1937_4112,(0,0,1):C.UVGC_1937_4113,(0,0,0):C.UVGC_1937_4114})

V_601 = CTVertex(name = 'V_601',
                 type = 'UV',
                 particles = [ P.h01, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.su4] ] ],
                 couplings = {(0,0,2):C.UVGC_1936_4109,(0,0,1):C.UVGC_1936_4110,(0,0,0):C.UVGC_1936_4111})

V_602 = CTVertex(name = 'V_602',
                 type = 'UV',
                 particles = [ P.n1, P.c, P.su5__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su5] ], [ [P.go, P.su5] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.UVGC_1592_2352,(0,0,1):C.UVGC_1592_2353,(0,0,4):C.UVGC_1592_2354,(0,0,3):C.UVGC_1592_2355,(0,0,2):C.UVGC_1592_2356})

V_603 = CTVertex(name = 'V_603',
                 type = 'UV',
                 particles = [ P.n2, P.c, P.su5__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su5] ], [ [P.go, P.su5] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.UVGC_1593_2357,(0,0,1):C.UVGC_1593_2358,(0,0,4):C.UVGC_1593_2359,(0,0,3):C.UVGC_1593_2360,(0,0,2):C.UVGC_1593_2361})

V_604 = CTVertex(name = 'V_604',
                 type = 'UV',
                 particles = [ P.n3, P.c, P.su5__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su5] ], [ [P.go, P.su5] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.UVGC_1594_2362,(0,0,1):C.UVGC_1594_2363,(0,0,4):C.UVGC_1594_2364,(0,0,3):C.UVGC_1594_2365,(0,0,2):C.UVGC_1594_2366})

V_605 = CTVertex(name = 'V_605',
                 type = 'UV',
                 particles = [ P.n4, P.c, P.su5__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su5] ], [ [P.go, P.su5] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.UVGC_1595_2367,(0,0,1):C.UVGC_1595_2368,(0,0,4):C.UVGC_1595_2369,(0,0,3):C.UVGC_1595_2370,(0,0,2):C.UVGC_1595_2371})

V_606 = CTVertex(name = 'V_606',
                 type = 'UV',
                 particles = [ P.a, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.UVGC_1581_2272,(0,0,1):C.UVGC_1581_2273,(0,1,0):C.UVGC_1580_2270,(0,1,1):C.UVGC_1580_2271})

V_607 = CTVertex(name = 'V_607',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.n1, P.su5 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su5] ], [ [P.go, P.su5] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.UVGC_1576_2250,(0,0,1):C.UVGC_1576_2251,(0,0,4):C.UVGC_1576_2252,(0,0,3):C.UVGC_1576_2253,(0,0,2):C.UVGC_1576_2254})

V_608 = CTVertex(name = 'V_608',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.n2, P.su5 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su5] ], [ [P.go, P.su5] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.UVGC_1577_2255,(0,0,1):C.UVGC_1577_2256,(0,0,4):C.UVGC_1577_2257,(0,0,3):C.UVGC_1577_2258,(0,0,2):C.UVGC_1577_2259})

V_609 = CTVertex(name = 'V_609',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.n3, P.su5 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su5] ], [ [P.go, P.su5] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.UVGC_1578_2260,(0,0,1):C.UVGC_1578_2261,(0,0,4):C.UVGC_1578_2262,(0,0,3):C.UVGC_1578_2263,(0,0,2):C.UVGC_1578_2264})

V_610 = CTVertex(name = 'V_610',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.n4, P.su5 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.su5] ], [ [P.go, P.su5] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.UVGC_1579_2265,(0,0,1):C.UVGC_1579_2266,(0,0,4):C.UVGC_1579_2267,(0,0,3):C.UVGC_1579_2268,(0,0,2):C.UVGC_1579_2269})

V_611 = CTVertex(name = 'V_611',
                 type = 'UV',
                 particles = [ P.a, P.a, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ], [ [P.su5] ] ],
                 couplings = {(0,0,2):C.UVGC_1582_2274,(0,0,0):C.UVGC_1582_2275,(0,0,1):C.UVGC_1582_2276})

V_612 = CTVertex(name = 'V_612',
                 type = 'UV',
                 particles = [ P.h02, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ], [ [P.su5] ] ],
                 couplings = {(0,0,2):C.UVGC_1939_4118,(0,0,0):C.UVGC_1939_4119,(0,0,1):C.UVGC_1939_4120})

V_613 = CTVertex(name = 'V_613',
                 type = 'UV',
                 particles = [ P.h01, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ], [ [P.su5] ] ],
                 couplings = {(0,0,2):C.UVGC_1938_4115,(0,0,0):C.UVGC_1938_4116,(0,0,1):C.UVGC_1938_4117})

V_614 = CTVertex(name = 'V_614',
                 type = 'UV',
                 particles = [ P.n1, P.t, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su3, P.t] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.su6, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,4):C.UVGC_1749_3439,(0,1,6):C.UVGC_1749_3440,(0,1,0):C.UVGC_1749_3441,(0,1,2):C.UVGC_1749_3442,(0,1,3):C.UVGC_1749_3443,(0,1,5):C.UVGC_1749_3444,(0,1,1):C.UVGC_1749_3445,(0,0,4):C.UVGC_1755_3487,(0,0,6):C.UVGC_1755_3488,(0,0,0):C.UVGC_1755_3489,(0,0,2):C.UVGC_1755_3490,(0,0,3):C.UVGC_1755_3491,(0,0,5):C.UVGC_1755_3492})

V_615 = CTVertex(name = 'V_615',
                 type = 'UV',
                 particles = [ P.n2, P.t, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su3, P.t] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.su6, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,4):C.UVGC_1750_3446,(0,1,6):C.UVGC_1750_3447,(0,1,0):C.UVGC_1750_3448,(0,1,2):C.UVGC_1750_3449,(0,1,3):C.UVGC_1750_3450,(0,1,5):C.UVGC_1750_3451,(0,1,1):C.UVGC_1750_3452,(0,0,4):C.UVGC_1756_3493,(0,0,6):C.UVGC_1756_3494,(0,0,0):C.UVGC_1756_3495,(0,0,2):C.UVGC_1756_3496,(0,0,3):C.UVGC_1756_3497,(0,0,5):C.UVGC_1756_3498})

V_616 = CTVertex(name = 'V_616',
                 type = 'UV',
                 particles = [ P.n3, P.t, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su3, P.t] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.su6, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,4):C.UVGC_1751_3453,(0,1,6):C.UVGC_1751_3454,(0,1,0):C.UVGC_1751_3455,(0,1,2):C.UVGC_1751_3456,(0,1,3):C.UVGC_1751_3457,(0,1,5):C.UVGC_1751_3458,(0,1,1):C.UVGC_1751_3459,(0,0,4):C.UVGC_1757_3499,(0,0,6):C.UVGC_1757_3500,(0,0,0):C.UVGC_1757_3501,(0,0,2):C.UVGC_1757_3502,(0,0,3):C.UVGC_1757_3503,(0,0,5):C.UVGC_1757_3504})

V_617 = CTVertex(name = 'V_617',
                 type = 'UV',
                 particles = [ P.n4, P.t, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su3, P.t] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.su6, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,4):C.UVGC_1752_3460,(0,1,6):C.UVGC_1752_3461,(0,1,0):C.UVGC_1752_3462,(0,1,2):C.UVGC_1752_3463,(0,1,3):C.UVGC_1752_3464,(0,1,5):C.UVGC_1752_3465,(0,1,1):C.UVGC_1752_3466,(0,0,4):C.UVGC_1758_3505,(0,0,6):C.UVGC_1758_3506,(0,0,0):C.UVGC_1758_3507,(0,0,2):C.UVGC_1758_3508,(0,0,3):C.UVGC_1758_3509,(0,0,5):C.UVGC_1758_3510})

V_618 = CTVertex(name = 'V_618',
                 type = 'UV',
                 particles = [ P.a, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(0,0,1):C.UVGC_1738_3359,(0,0,0):C.UVGC_1738_3360,(0,1,1):C.UVGC_1737_3357,(0,1,0):C.UVGC_1737_3358})

V_619 = CTVertex(name = 'V_619',
                 type = 'UV',
                 particles = [ P.H__plus__, P.sd3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.sd3, P.su6], [P.g, P.sd3, P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2225_4482,(0,0,5):C.UVGC_2225_4483,(0,0,6):C.UVGC_2225_4484,(0,0,7):C.UVGC_2225_4485,(0,0,2):C.UVGC_2225_4486,(0,0,3):C.UVGC_2225_4487,(0,0,4):C.UVGC_2225_4488,(0,0,8):C.UVGC_2225_4489,(0,0,1):C.UVGC_2225_4490})

V_620 = CTVertex(name = 'V_620',
                 type = 'UV',
                 particles = [ P.G__plus__, P.sd3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.sd3, P.su6], [P.g, P.sd3, P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2230_4525,(0,0,5):C.UVGC_2230_4526,(0,0,6):C.UVGC_2230_4527,(0,0,7):C.UVGC_2230_4528,(0,0,2):C.UVGC_2230_4529,(0,0,3):C.UVGC_2230_4530,(0,0,4):C.UVGC_2230_4531,(0,0,8):C.UVGC_2230_4532,(0,0,1):C.UVGC_2230_4533})

V_621 = CTVertex(name = 'V_621',
                 type = 'UV',
                 particles = [ P.h01, P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.su3, P.su6], [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,3):C.UVGC_1946_4156,(0,0,4):C.UVGC_1946_4157,(0,0,5):C.UVGC_1946_4158,(0,0,0):C.UVGC_1946_4159,(0,0,1):C.UVGC_1946_4160,(0,0,2):C.UVGC_1946_4161,(0,0,6):C.UVGC_1946_4162})

V_622 = CTVertex(name = 'V_622',
                 type = 'UV',
                 particles = [ P.h02, P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.su3, P.su6], [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,3):C.UVGC_1948_4169,(0,0,4):C.UVGC_1948_4170,(0,0,5):C.UVGC_1948_4171,(0,0,0):C.UVGC_1948_4172,(0,0,1):C.UVGC_1948_4173,(0,0,2):C.UVGC_1948_4174,(0,0,6):C.UVGC_1948_4175})

V_623 = CTVertex(name = 'V_623',
                 type = 'UV',
                 particles = [ P.A0, P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.su3, P.su6], [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,3):C.UVGC_2223_4462,(0,0,4):C.UVGC_2223_4463,(0,0,5):C.UVGC_2223_4464,(0,0,0):C.UVGC_2223_4465,(0,0,1):C.UVGC_2223_4466,(0,0,2):C.UVGC_2223_4467,(0,0,6):C.UVGC_2223_4468})

V_624 = CTVertex(name = 'V_624',
                 type = 'UV',
                 particles = [ P.G0, P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.su3, P.su6], [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,3):C.UVGC_2229_4518,(0,0,4):C.UVGC_2229_4519,(0,0,5):C.UVGC_2229_4520,(0,0,0):C.UVGC_2229_4521,(0,0,1):C.UVGC_2229_4522,(0,0,2):C.UVGC_2229_4523,(0,0,6):C.UVGC_2229_4524})

V_625 = CTVertex(name = 'V_625',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.n1, P.su6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su3, P.t] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.su6, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,4):C.UVGC_1709_3305,(0,0,6):C.UVGC_1709_3306,(0,0,0):C.UVGC_1709_3307,(0,0,2):C.UVGC_1709_3308,(0,0,3):C.UVGC_1709_3309,(0,0,5):C.UVGC_1709_3310,(0,0,1):C.UVGC_1709_3311,(0,1,4):C.UVGC_1705_3281,(0,1,6):C.UVGC_1705_3282,(0,1,0):C.UVGC_1705_3283,(0,1,2):C.UVGC_1705_3284,(0,1,3):C.UVGC_1705_3285,(0,1,5):C.UVGC_1705_3286})

V_626 = CTVertex(name = 'V_626',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.n2, P.su6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su3, P.t] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.su6, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,4):C.UVGC_1710_3312,(0,0,6):C.UVGC_1710_3313,(0,0,0):C.UVGC_1710_3314,(0,0,2):C.UVGC_1710_3315,(0,0,3):C.UVGC_1710_3316,(0,0,5):C.UVGC_1710_3317,(0,0,1):C.UVGC_1710_3318,(0,1,4):C.UVGC_1706_3287,(0,1,6):C.UVGC_1706_3288,(0,1,0):C.UVGC_1706_3289,(0,1,2):C.UVGC_1706_3290,(0,1,3):C.UVGC_1706_3291,(0,1,5):C.UVGC_1706_3292})

V_627 = CTVertex(name = 'V_627',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.n3, P.su6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su3, P.t] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.su6, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,4):C.UVGC_1711_3319,(0,0,6):C.UVGC_1711_3320,(0,0,0):C.UVGC_1711_3321,(0,0,2):C.UVGC_1711_3322,(0,0,3):C.UVGC_1711_3323,(0,0,5):C.UVGC_1711_3324,(0,0,1):C.UVGC_1711_3325,(0,1,4):C.UVGC_1707_3293,(0,1,6):C.UVGC_1707_3294,(0,1,0):C.UVGC_1707_3295,(0,1,2):C.UVGC_1707_3296,(0,1,3):C.UVGC_1707_3297,(0,1,5):C.UVGC_1707_3298})

V_628 = CTVertex(name = 'V_628',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.n4, P.su6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su3, P.t] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.su6, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,4):C.UVGC_1712_3326,(0,0,6):C.UVGC_1712_3327,(0,0,0):C.UVGC_1712_3328,(0,0,2):C.UVGC_1712_3329,(0,0,3):C.UVGC_1712_3330,(0,0,5):C.UVGC_1712_3331,(0,0,1):C.UVGC_1712_3332,(0,1,4):C.UVGC_1708_3299,(0,1,6):C.UVGC_1708_3300,(0,1,0):C.UVGC_1708_3301,(0,1,2):C.UVGC_1708_3302,(0,1,3):C.UVGC_1708_3303,(0,1,5):C.UVGC_1708_3304})

V_629 = CTVertex(name = 'V_629',
                 type = 'UV',
                 particles = [ P.a, P.a, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.su6] ] ],
                 couplings = {(0,0,2):C.UVGC_1739_3361,(0,0,1):C.UVGC_1739_3362,(0,0,0):C.UVGC_1739_3363})

V_630 = CTVertex(name = 'V_630',
                 type = 'UV',
                 particles = [ P.h02, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.su6] ] ],
                 couplings = {(0,0,5):C.UVGC_1947_4163,(0,0,3):C.UVGC_1947_4164,(0,0,4):C.UVGC_1947_4165,(0,0,0):C.UVGC_1947_4166,(0,0,1):C.UVGC_1947_4167,(0,0,2):C.UVGC_1947_4168})

V_631 = CTVertex(name = 'V_631',
                 type = 'UV',
                 particles = [ P.h01, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.su6] ] ],
                 couplings = {(0,0,5):C.UVGC_1945_4150,(0,0,3):C.UVGC_1945_4151,(0,0,4):C.UVGC_1945_4152,(0,0,0):C.UVGC_1945_4153,(0,0,1):C.UVGC_1945_4154,(0,0,2):C.UVGC_1945_4155})

V_632 = CTVertex(name = 'V_632',
                 type = 'UV',
                 particles = [ P.go, P.d, P.sd1__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.go] ], [ [P.d, P.g, P.sd1] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.go] ], [ [P.go, P.sd1] ], [ [P.g, P.go] ], [ [P.g, P.go, P.sd1] ], [ [P.g, P.sd1] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1] ], [ [P.su1, P.u] ], [ [P.su2] ], [ [P.su3] ], [ [P.su3, P.t] ], [ [P.su4] ], [ [P.su4, P.u] ], [ [P.su5] ], [ [P.su6] ], [ [P.su6, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1766_3570,(0,0,11):C.UVGC_1766_3571,(0,0,16):C.UVGC_1766_3572,(0,0,17):C.UVGC_1766_3573,(0,0,18):C.UVGC_1766_3574,(0,0,19):C.UVGC_1766_3575,(0,0,20):C.UVGC_1766_3576,(0,0,21):C.UVGC_1766_3577,(0,0,24):C.UVGC_1766_3578,(0,0,26):C.UVGC_1766_3579,(0,0,27):C.UVGC_1766_3580,(0,0,29):C.UVGC_1766_3581,(0,0,31):C.UVGC_1766_3582,(0,0,32):C.UVGC_1766_3583,(0,0,34):C.UVGC_1766_3584,(0,0,1):C.UVGC_1766_3585,(0,0,2):C.UVGC_1766_3586,(0,0,3):C.UVGC_1766_3587,(0,0,4):C.UVGC_1766_3588,(0,0,5):C.UVGC_1766_3589,(0,0,6):C.UVGC_1766_3590,(0,0,9):C.UVGC_1766_3591,(0,0,10):C.UVGC_1766_3592,(0,0,13):C.UVGC_1766_3593,(0,0,15):C.UVGC_1766_3594,(0,0,12):C.UVGC_1766_3595,(0,0,22):C.UVGC_1766_3596,(0,0,23):C.UVGC_1766_3597,(0,0,25):C.UVGC_1766_3598,(0,0,28):C.UVGC_1766_3599,(0,0,30):C.UVGC_1766_3600,(0,0,33):C.UVGC_1766_3601,(0,0,7):C.UVGC_1766_3602,(0,0,8):C.UVGC_1766_3603,(0,0,14):C.UVGC_1766_3604})

V_633 = CTVertex(name = 'V_633',
                 type = 'UV',
                 particles = [ P.g, P.sd1__tilde__, P.sd1 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.d, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.UVGC_1258_667,(0,0,1):C.UVGC_1258_668,(0,0,3):C.UVGC_1258_669,(0,0,4):C.UVGC_1258_670,(0,0,2):C.UVGC_1258_671,(0,0,5):C.UVGC_1258_672,(0,1,0):C.UVGC_1257_661,(0,1,1):C.UVGC_1257_662,(0,1,3):C.UVGC_1257_663,(0,1,4):C.UVGC_1257_664,(0,1,2):C.UVGC_1257_665,(0,1,5):C.UVGC_1257_666})

V_634 = CTVertex(name = 'V_634',
                 type = 'UV',
                 particles = [ P.go, P.s, P.sd2__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.go] ], [ [P.go, P.s] ], [ [P.go, P.sd2] ], [ [P.g, P.go] ], [ [P.g, P.go, P.s] ], [ [P.g, P.go, P.sd2] ], [ [P.g, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.s, P.sd2] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1] ], [ [P.su1, P.u] ], [ [P.su2] ], [ [P.su3] ], [ [P.su3, P.t] ], [ [P.su4] ], [ [P.su4, P.u] ], [ [P.su5] ], [ [P.su6] ], [ [P.su6, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1767_3605,(0,0,7):C.UVGC_1767_3606,(0,0,16):C.UVGC_1767_3607,(0,0,17):C.UVGC_1767_3608,(0,0,18):C.UVGC_1767_3609,(0,0,19):C.UVGC_1767_3610,(0,0,20):C.UVGC_1767_3611,(0,0,21):C.UVGC_1767_3612,(0,0,24):C.UVGC_1767_3613,(0,0,26):C.UVGC_1767_3614,(0,0,27):C.UVGC_1767_3615,(0,0,29):C.UVGC_1767_3616,(0,0,31):C.UVGC_1767_3617,(0,0,32):C.UVGC_1767_3618,(0,0,34):C.UVGC_1767_3619,(0,0,1):C.UVGC_1767_3620,(0,0,2):C.UVGC_1767_3621,(0,0,3):C.UVGC_1767_3622,(0,0,4):C.UVGC_1767_3623,(0,0,5):C.UVGC_1767_3624,(0,0,6):C.UVGC_1767_3625,(0,0,10):C.UVGC_1767_3626,(0,0,13):C.UVGC_1767_3627,(0,0,14):C.UVGC_1767_3628,(0,0,8):C.UVGC_1767_3629,(0,0,9):C.UVGC_1767_3630,(0,0,22):C.UVGC_1767_3631,(0,0,23):C.UVGC_1767_3632,(0,0,25):C.UVGC_1767_3633,(0,0,28):C.UVGC_1767_3634,(0,0,30):C.UVGC_1767_3635,(0,0,33):C.UVGC_1767_3636,(0,0,11):C.UVGC_1767_3637,(0,0,12):C.UVGC_1767_3638,(0,0,15):C.UVGC_1767_3639})

V_635 = CTVertex(name = 'V_635',
                 type = 'UV',
                 particles = [ P.g, P.sd2__tilde__, P.sd2 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ] ],
                 couplings = {(0,0,0):C.UVGC_1284_802,(0,0,1):C.UVGC_1284_803,(0,0,2):C.UVGC_1284_804,(0,0,3):C.UVGC_1284_805,(0,0,5):C.UVGC_1284_806,(0,0,4):C.UVGC_1284_807,(0,1,0):C.UVGC_1283_796,(0,1,1):C.UVGC_1283_797,(0,1,2):C.UVGC_1283_798,(0,1,3):C.UVGC_1283_799,(0,1,5):C.UVGC_1283_800,(0,1,4):C.UVGC_1283_801})

V_636 = CTVertex(name = 'V_636',
                 type = 'UV',
                 particles = [ P.go, P.b, P.sd3__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.g, P.go] ], [ [P.b, P.g, P.sd3] ], [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.go] ], [ [P.g, P.go, P.sd3] ], [ [P.g, P.sd3] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1] ], [ [P.su1, P.u] ], [ [P.su2] ], [ [P.su3] ], [ [P.su3, P.t] ], [ [P.su4] ], [ [P.su4, P.u] ], [ [P.su5] ], [ [P.su6] ], [ [P.su6, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1768_3640,(0,1,11):C.UVGC_1768_3641,(0,1,17):C.UVGC_1768_3642,(0,1,18):C.UVGC_1768_3643,(0,1,19):C.UVGC_1768_3644,(0,1,20):C.UVGC_1768_3645,(0,1,21):C.UVGC_1768_3646,(0,1,22):C.UVGC_1768_3647,(0,1,25):C.UVGC_1768_3648,(0,1,27):C.UVGC_1768_3649,(0,1,28):C.UVGC_1768_3650,(0,1,30):C.UVGC_1768_3651,(0,1,32):C.UVGC_1768_3652,(0,1,33):C.UVGC_1768_3653,(0,1,35):C.UVGC_1768_3654,(0,1,1):C.UVGC_1768_3655,(0,1,2):C.UVGC_1768_3656,(0,1,5):C.UVGC_1768_3657,(0,1,6):C.UVGC_1768_3658,(0,1,7):C.UVGC_1768_3659,(0,1,8):C.UVGC_1768_3660,(0,1,9):C.UVGC_1768_3661,(0,1,10):C.UVGC_1768_3662,(0,1,14):C.UVGC_1768_3663,(0,1,16):C.UVGC_1768_3664,(0,1,12):C.UVGC_1768_3665,(0,1,13):C.UVGC_1768_3666,(0,1,23):C.UVGC_1768_3667,(0,1,24):C.UVGC_1768_3668,(0,1,26):C.UVGC_1768_3669,(0,1,29):C.UVGC_1768_3670,(0,1,31):C.UVGC_1768_3671,(0,1,34):C.UVGC_1768_3672,(0,1,3):C.UVGC_1768_3673,(0,1,4):C.UVGC_1768_3674,(0,1,15):C.UVGC_1768_3675,(0,0,2):C.UVGC_1312_990})

V_637 = CTVertex(name = 'V_637',
                 type = 'UV',
                 particles = [ P.g, P.sd3__tilde__, P.sd3 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.go] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,0,0):C.UVGC_1302_913,(0,0,2):C.UVGC_1302_914,(0,0,3):C.UVGC_1302_915,(0,0,4):C.UVGC_1302_916,(0,0,1):C.UVGC_1302_917,(0,0,5):C.UVGC_1302_918,(0,1,0):C.UVGC_1301_907,(0,1,2):C.UVGC_1301_908,(0,1,3):C.UVGC_1301_909,(0,1,4):C.UVGC_1301_910,(0,1,1):C.UVGC_1301_911,(0,1,5):C.UVGC_1301_912})

V_638 = CTVertex(name = 'V_638',
                 type = 'UV',
                 particles = [ P.go, P.d, P.sd4__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.go] ], [ [P.d, P.g, P.sd4] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.go] ], [ [P.go, P.sd4] ], [ [P.g, P.go] ], [ [P.g, P.go, P.sd4] ], [ [P.g, P.sd4] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1] ], [ [P.su1, P.u] ], [ [P.su2] ], [ [P.su3] ], [ [P.su3, P.t] ], [ [P.su4] ], [ [P.su4, P.u] ], [ [P.su5] ], [ [P.su6] ], [ [P.su6, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1769_3676,(0,0,11):C.UVGC_1769_3677,(0,0,16):C.UVGC_1769_3678,(0,0,17):C.UVGC_1769_3679,(0,0,18):C.UVGC_1769_3680,(0,0,19):C.UVGC_1769_3681,(0,0,20):C.UVGC_1769_3682,(0,0,21):C.UVGC_1769_3683,(0,0,24):C.UVGC_1769_3684,(0,0,26):C.UVGC_1769_3685,(0,0,27):C.UVGC_1769_3686,(0,0,29):C.UVGC_1769_3687,(0,0,31):C.UVGC_1769_3688,(0,0,32):C.UVGC_1769_3689,(0,0,34):C.UVGC_1769_3690,(0,0,1):C.UVGC_1769_3691,(0,0,2):C.UVGC_1769_3692,(0,0,3):C.UVGC_1769_3693,(0,0,4):C.UVGC_1769_3694,(0,0,5):C.UVGC_1769_3695,(0,0,6):C.UVGC_1769_3696,(0,0,9):C.UVGC_1769_3697,(0,0,10):C.UVGC_1769_3698,(0,0,13):C.UVGC_1769_3699,(0,0,15):C.UVGC_1769_3700,(0,0,12):C.UVGC_1769_3701,(0,0,22):C.UVGC_1769_3702,(0,0,23):C.UVGC_1769_3703,(0,0,25):C.UVGC_1769_3704,(0,0,28):C.UVGC_1769_3705,(0,0,30):C.UVGC_1769_3706,(0,0,33):C.UVGC_1769_3707,(0,0,7):C.UVGC_1769_3708,(0,0,8):C.UVGC_1769_3709,(0,0,14):C.UVGC_1769_3710})

V_639 = CTVertex(name = 'V_639',
                 type = 'UV',
                 particles = [ P.g, P.sd4__tilde__, P.sd4 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.d, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.UVGC_1326_1030,(0,0,1):C.UVGC_1326_1031,(0,0,3):C.UVGC_1326_1032,(0,0,4):C.UVGC_1326_1033,(0,0,2):C.UVGC_1326_1034,(0,0,5):C.UVGC_1326_1035,(0,1,0):C.UVGC_1325_1024,(0,1,1):C.UVGC_1325_1025,(0,1,3):C.UVGC_1325_1026,(0,1,4):C.UVGC_1325_1027,(0,1,2):C.UVGC_1325_1028,(0,1,5):C.UVGC_1325_1029})

V_640 = CTVertex(name = 'V_640',
                 type = 'UV',
                 particles = [ P.go, P.s, P.sd5__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.go] ], [ [P.go, P.s] ], [ [P.go, P.sd5] ], [ [P.g, P.go] ], [ [P.g, P.go, P.s] ], [ [P.g, P.go, P.sd5] ], [ [P.g, P.s] ], [ [P.g, P.sd5] ], [ [P.g, P.s, P.sd5] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1] ], [ [P.su1, P.u] ], [ [P.su2] ], [ [P.su3] ], [ [P.su3, P.t] ], [ [P.su4] ], [ [P.su4, P.u] ], [ [P.su5] ], [ [P.su6] ], [ [P.su6, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1770_3711,(0,0,7):C.UVGC_1770_3712,(0,0,16):C.UVGC_1770_3713,(0,0,17):C.UVGC_1770_3714,(0,0,18):C.UVGC_1770_3715,(0,0,19):C.UVGC_1770_3716,(0,0,20):C.UVGC_1770_3717,(0,0,21):C.UVGC_1770_3718,(0,0,24):C.UVGC_1770_3719,(0,0,26):C.UVGC_1770_3720,(0,0,27):C.UVGC_1770_3721,(0,0,29):C.UVGC_1770_3722,(0,0,31):C.UVGC_1770_3723,(0,0,32):C.UVGC_1770_3724,(0,0,34):C.UVGC_1770_3725,(0,0,1):C.UVGC_1770_3726,(0,0,2):C.UVGC_1770_3727,(0,0,3):C.UVGC_1770_3728,(0,0,4):C.UVGC_1770_3729,(0,0,5):C.UVGC_1770_3730,(0,0,6):C.UVGC_1770_3731,(0,0,10):C.UVGC_1770_3732,(0,0,13):C.UVGC_1770_3733,(0,0,14):C.UVGC_1770_3734,(0,0,8):C.UVGC_1770_3735,(0,0,9):C.UVGC_1770_3736,(0,0,22):C.UVGC_1770_3737,(0,0,23):C.UVGC_1770_3738,(0,0,25):C.UVGC_1770_3739,(0,0,28):C.UVGC_1770_3740,(0,0,30):C.UVGC_1770_3741,(0,0,33):C.UVGC_1770_3742,(0,0,11):C.UVGC_1770_3743,(0,0,12):C.UVGC_1770_3744,(0,0,15):C.UVGC_1770_3745})

V_641 = CTVertex(name = 'V_641',
                 type = 'UV',
                 particles = [ P.g, P.sd5__tilde__, P.sd5 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.s] ], [ [P.g, P.sd5] ] ],
                 couplings = {(0,0,0):C.UVGC_1351_1158,(0,0,1):C.UVGC_1351_1159,(0,0,2):C.UVGC_1351_1160,(0,0,3):C.UVGC_1351_1161,(0,0,5):C.UVGC_1351_1162,(0,0,4):C.UVGC_1351_1163,(0,1,0):C.UVGC_1350_1152,(0,1,1):C.UVGC_1350_1153,(0,1,2):C.UVGC_1350_1154,(0,1,3):C.UVGC_1350_1155,(0,1,5):C.UVGC_1350_1156,(0,1,4):C.UVGC_1350_1157})

V_642 = CTVertex(name = 'V_642',
                 type = 'UV',
                 particles = [ P.go, P.b, P.sd6__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.g, P.go] ], [ [P.b, P.g, P.sd6] ], [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.go] ], [ [P.g, P.go, P.sd6] ], [ [P.g, P.sd6] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1] ], [ [P.su1, P.u] ], [ [P.su2] ], [ [P.su3] ], [ [P.su3, P.t] ], [ [P.su4] ], [ [P.su4, P.u] ], [ [P.su5] ], [ [P.su6] ], [ [P.su6, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1771_3746,(0,0,11):C.UVGC_1771_3747,(0,0,17):C.UVGC_1771_3748,(0,0,18):C.UVGC_1771_3749,(0,0,19):C.UVGC_1771_3750,(0,0,20):C.UVGC_1771_3751,(0,0,21):C.UVGC_1771_3752,(0,0,22):C.UVGC_1771_3753,(0,0,25):C.UVGC_1771_3754,(0,0,27):C.UVGC_1771_3755,(0,0,28):C.UVGC_1771_3756,(0,0,30):C.UVGC_1771_3757,(0,0,32):C.UVGC_1771_3758,(0,0,33):C.UVGC_1771_3759,(0,0,35):C.UVGC_1771_3760,(0,0,1):C.UVGC_1771_3761,(0,0,2):C.UVGC_1771_3762,(0,0,5):C.UVGC_1771_3763,(0,0,6):C.UVGC_1771_3764,(0,0,7):C.UVGC_1771_3765,(0,0,8):C.UVGC_1771_3766,(0,0,9):C.UVGC_1771_3767,(0,0,10):C.UVGC_1771_3768,(0,0,14):C.UVGC_1771_3769,(0,0,16):C.UVGC_1771_3770,(0,0,12):C.UVGC_1771_3771,(0,0,13):C.UVGC_1771_3772,(0,0,23):C.UVGC_1771_3773,(0,0,24):C.UVGC_1771_3774,(0,0,26):C.UVGC_1771_3775,(0,0,29):C.UVGC_1771_3776,(0,0,31):C.UVGC_1771_3777,(0,0,34):C.UVGC_1771_3778,(0,0,3):C.UVGC_1771_3779,(0,0,4):C.UVGC_1771_3780,(0,0,15):C.UVGC_1771_3781,(0,1,2):C.UVGC_1371_1257})

V_643 = CTVertex(name = 'V_643',
                 type = 'UV',
                 particles = [ P.g, P.sd6__tilde__, P.sd6 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.go] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1393_1287,(0,0,2):C.UVGC_1393_1288,(0,0,3):C.UVGC_1393_1289,(0,0,4):C.UVGC_1393_1290,(0,0,1):C.UVGC_1393_1291,(0,0,5):C.UVGC_1393_1292,(0,1,0):C.UVGC_1392_1281,(0,1,2):C.UVGC_1392_1282,(0,1,3):C.UVGC_1392_1283,(0,1,4):C.UVGC_1392_1284,(0,1,1):C.UVGC_1392_1285,(0,1,5):C.UVGC_1392_1286})

V_644 = CTVertex(name = 'V_644',
                 type = 'UV',
                 particles = [ P.go, P.u, P.su1__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.go] ], [ [P.go, P.su1] ], [ [P.go, P.u] ], [ [P.g, P.go] ], [ [P.g, P.go, P.su1] ], [ [P.g, P.go, P.u] ], [ [P.g, P.su1] ], [ [P.g, P.su1, P.u] ], [ [P.g, P.u] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1] ], [ [P.su1, P.u] ], [ [P.su2] ], [ [P.su3] ], [ [P.su3, P.t] ], [ [P.su4] ], [ [P.su4, P.u] ], [ [P.su5] ], [ [P.su6] ], [ [P.su6, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1772_3782,(0,0,7):C.UVGC_1772_3783,(0,0,16):C.UVGC_1772_3784,(0,0,17):C.UVGC_1772_3785,(0,0,18):C.UVGC_1772_3786,(0,0,19):C.UVGC_1772_3787,(0,0,20):C.UVGC_1772_3788,(0,0,21):C.UVGC_1772_3789,(0,0,24):C.UVGC_1772_3790,(0,0,26):C.UVGC_1772_3791,(0,0,27):C.UVGC_1772_3792,(0,0,29):C.UVGC_1772_3793,(0,0,31):C.UVGC_1772_3794,(0,0,32):C.UVGC_1772_3795,(0,0,34):C.UVGC_1772_3796,(0,0,1):C.UVGC_1772_3797,(0,0,2):C.UVGC_1772_3798,(0,0,3):C.UVGC_1772_3799,(0,0,4):C.UVGC_1772_3800,(0,0,5):C.UVGC_1772_3801,(0,0,6):C.UVGC_1772_3802,(0,0,10):C.UVGC_1772_3803,(0,0,13):C.UVGC_1772_3804,(0,0,15):C.UVGC_1772_3805,(0,0,8):C.UVGC_1772_3806,(0,0,9):C.UVGC_1772_3807,(0,0,22):C.UVGC_1772_3808,(0,0,23):C.UVGC_1772_3809,(0,0,25):C.UVGC_1772_3810,(0,0,28):C.UVGC_1772_3811,(0,0,30):C.UVGC_1772_3812,(0,0,33):C.UVGC_1772_3813,(0,0,11):C.UVGC_1772_3814,(0,0,12):C.UVGC_1772_3815,(0,0,14):C.UVGC_1772_3816})

V_645 = CTVertex(name = 'V_645',
                 type = 'UV',
                 particles = [ P.g, P.su1__tilde__, P.su1 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_1475_1708,(0,0,1):C.UVGC_1475_1709,(0,0,2):C.UVGC_1475_1710,(0,0,3):C.UVGC_1475_1711,(0,0,5):C.UVGC_1475_1712,(0,0,4):C.UVGC_1475_1713,(0,1,0):C.UVGC_1474_1702,(0,1,1):C.UVGC_1474_1703,(0,1,2):C.UVGC_1474_1704,(0,1,3):C.UVGC_1474_1705,(0,1,5):C.UVGC_1474_1706,(0,1,4):C.UVGC_1474_1707})

V_646 = CTVertex(name = 'V_646',
                 type = 'UV',
                 particles = [ P.go, P.c, P.su2__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.go] ], [ [P.c, P.g, P.su2] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.go] ], [ [P.go, P.su2] ], [ [P.g, P.go] ], [ [P.g, P.go, P.su2] ], [ [P.g, P.su2] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1] ], [ [P.su1, P.u] ], [ [P.su2] ], [ [P.su3] ], [ [P.su3, P.t] ], [ [P.su4] ], [ [P.su4, P.u] ], [ [P.su5] ], [ [P.su6] ], [ [P.su6, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1773_3817,(0,0,11):C.UVGC_1773_3818,(0,0,16):C.UVGC_1773_3819,(0,0,17):C.UVGC_1773_3820,(0,0,18):C.UVGC_1773_3821,(0,0,19):C.UVGC_1773_3822,(0,0,20):C.UVGC_1773_3823,(0,0,21):C.UVGC_1773_3824,(0,0,24):C.UVGC_1773_3825,(0,0,26):C.UVGC_1773_3826,(0,0,27):C.UVGC_1773_3827,(0,0,29):C.UVGC_1773_3828,(0,0,31):C.UVGC_1773_3829,(0,0,32):C.UVGC_1773_3830,(0,0,34):C.UVGC_1773_3831,(0,0,1):C.UVGC_1773_3832,(0,0,2):C.UVGC_1773_3833,(0,0,3):C.UVGC_1773_3834,(0,0,4):C.UVGC_1773_3835,(0,0,7):C.UVGC_1773_3836,(0,0,8):C.UVGC_1773_3837,(0,0,9):C.UVGC_1773_3838,(0,0,10):C.UVGC_1773_3839,(0,0,13):C.UVGC_1773_3840,(0,0,15):C.UVGC_1773_3841,(0,0,12):C.UVGC_1773_3842,(0,0,22):C.UVGC_1773_3843,(0,0,23):C.UVGC_1773_3844,(0,0,25):C.UVGC_1773_3845,(0,0,28):C.UVGC_1773_3846,(0,0,30):C.UVGC_1773_3847,(0,0,33):C.UVGC_1773_3848,(0,0,5):C.UVGC_1773_3849,(0,0,6):C.UVGC_1773_3850,(0,0,14):C.UVGC_1773_3851})

V_647 = CTVertex(name = 'V_647',
                 type = 'UV',
                 particles = [ P.g, P.su2__tilde__, P.su2 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1522_1964,(0,0,1):C.UVGC_1522_1965,(0,0,3):C.UVGC_1522_1966,(0,0,4):C.UVGC_1522_1967,(0,0,2):C.UVGC_1522_1968,(0,0,5):C.UVGC_1522_1969,(0,1,0):C.UVGC_1521_1958,(0,1,1):C.UVGC_1521_1959,(0,1,3):C.UVGC_1521_1960,(0,1,4):C.UVGC_1521_1961,(0,1,2):C.UVGC_1521_1962,(0,1,5):C.UVGC_1521_1963})

V_648 = CTVertex(name = 'V_648',
                 type = 'UV',
                 particles = [ P.go, P.t, P.su3__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.go] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.go] ], [ [P.g, P.go, P.su3] ], [ [P.g, P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su3, P.t] ], [ [P.g, P.t] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1] ], [ [P.su1, P.u] ], [ [P.su2] ], [ [P.su3] ], [ [P.su3, P.t] ], [ [P.su4] ], [ [P.su4, P.u] ], [ [P.su5] ], [ [P.su6] ], [ [P.su6, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1774_3852,(0,1,7):C.UVGC_1774_3853,(0,1,17):C.UVGC_1774_3854,(0,1,18):C.UVGC_1774_3855,(0,1,19):C.UVGC_1774_3856,(0,1,20):C.UVGC_1774_3857,(0,1,21):C.UVGC_1774_3858,(0,1,22):C.UVGC_1774_3859,(0,1,25):C.UVGC_1774_3860,(0,1,27):C.UVGC_1774_3861,(0,1,28):C.UVGC_1774_3862,(0,1,30):C.UVGC_1774_3863,(0,1,32):C.UVGC_1774_3864,(0,1,33):C.UVGC_1774_3865,(0,1,35):C.UVGC_1774_3866,(0,1,1):C.UVGC_1774_3867,(0,1,2):C.UVGC_1774_3868,(0,1,3):C.UVGC_1774_3869,(0,1,4):C.UVGC_1774_3870,(0,1,5):C.UVGC_1774_3871,(0,1,6):C.UVGC_1774_3872,(0,1,11):C.UVGC_1774_3873,(0,1,14):C.UVGC_1774_3874,(0,1,16):C.UVGC_1774_3875,(0,1,8):C.UVGC_1774_3876,(0,1,9):C.UVGC_1774_3877,(0,1,10):C.UVGC_1774_3878,(0,1,23):C.UVGC_1774_3879,(0,1,24):C.UVGC_1774_3880,(0,1,26):C.UVGC_1774_3881,(0,1,29):C.UVGC_1774_3882,(0,1,31):C.UVGC_1774_3883,(0,1,34):C.UVGC_1774_3884,(0,1,12):C.UVGC_1774_3885,(0,1,13):C.UVGC_1774_3886,(0,1,15):C.UVGC_1774_3887,(0,0,10):C.UVGC_1681_3163})

V_649 = CTVertex(name = 'V_649',
                 type = 'UV',
                 particles = [ P.g, P.su3__tilde__, P.su3 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_1658_2998,(0,0,1):C.UVGC_1658_2999,(0,0,2):C.UVGC_1658_3000,(0,0,3):C.UVGC_1658_3001,(0,0,5):C.UVGC_1658_3002,(0,0,4):C.UVGC_1658_3003,(0,1,0):C.UVGC_1657_2992,(0,1,1):C.UVGC_1657_2993,(0,1,2):C.UVGC_1657_2994,(0,1,3):C.UVGC_1657_2995,(0,1,5):C.UVGC_1657_2996,(0,1,4):C.UVGC_1657_2997})

V_650 = CTVertex(name = 'V_650',
                 type = 'UV',
                 particles = [ P.go, P.u, P.su4__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.go] ], [ [P.go, P.su4] ], [ [P.go, P.u] ], [ [P.g, P.go] ], [ [P.g, P.go, P.su4] ], [ [P.g, P.go, P.u] ], [ [P.g, P.su4] ], [ [P.g, P.su4, P.u] ], [ [P.g, P.u] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1] ], [ [P.su1, P.u] ], [ [P.su2] ], [ [P.su3] ], [ [P.su3, P.t] ], [ [P.su4] ], [ [P.su4, P.u] ], [ [P.su5] ], [ [P.su6] ], [ [P.su6, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1775_3888,(0,0,7):C.UVGC_1775_3889,(0,0,16):C.UVGC_1775_3890,(0,0,17):C.UVGC_1775_3891,(0,0,18):C.UVGC_1775_3892,(0,0,19):C.UVGC_1775_3893,(0,0,20):C.UVGC_1775_3894,(0,0,21):C.UVGC_1775_3895,(0,0,24):C.UVGC_1775_3896,(0,0,26):C.UVGC_1775_3897,(0,0,27):C.UVGC_1775_3898,(0,0,29):C.UVGC_1775_3899,(0,0,31):C.UVGC_1775_3900,(0,0,32):C.UVGC_1775_3901,(0,0,34):C.UVGC_1775_3902,(0,0,1):C.UVGC_1775_3903,(0,0,2):C.UVGC_1775_3904,(0,0,3):C.UVGC_1775_3905,(0,0,4):C.UVGC_1775_3906,(0,0,5):C.UVGC_1775_3907,(0,0,6):C.UVGC_1775_3908,(0,0,10):C.UVGC_1775_3909,(0,0,13):C.UVGC_1775_3910,(0,0,15):C.UVGC_1775_3911,(0,0,8):C.UVGC_1775_3912,(0,0,9):C.UVGC_1775_3913,(0,0,22):C.UVGC_1775_3914,(0,0,23):C.UVGC_1775_3915,(0,0,25):C.UVGC_1775_3916,(0,0,28):C.UVGC_1775_3917,(0,0,30):C.UVGC_1775_3918,(0,0,33):C.UVGC_1775_3919,(0,0,11):C.UVGC_1775_3920,(0,0,12):C.UVGC_1775_3921,(0,0,14):C.UVGC_1775_3922})

V_651 = CTVertex(name = 'V_651',
                 type = 'UV',
                 particles = [ P.g, P.su4__tilde__, P.su4 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.u] ], [ [P.g, P.su4] ] ],
                 couplings = {(0,0,0):C.UVGC_1558_2153,(0,0,1):C.UVGC_1558_2154,(0,0,2):C.UVGC_1558_2155,(0,0,3):C.UVGC_1558_2156,(0,0,5):C.UVGC_1558_2157,(0,0,4):C.UVGC_1558_2158,(0,1,0):C.UVGC_1557_2147,(0,1,1):C.UVGC_1557_2148,(0,1,2):C.UVGC_1557_2149,(0,1,3):C.UVGC_1557_2150,(0,1,5):C.UVGC_1557_2151,(0,1,4):C.UVGC_1557_2152})

V_652 = CTVertex(name = 'V_652',
                 type = 'UV',
                 particles = [ P.go, P.c, P.su5__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.go] ], [ [P.c, P.g, P.su5] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.go] ], [ [P.go, P.su5] ], [ [P.g, P.go] ], [ [P.g, P.go, P.su5] ], [ [P.g, P.su5] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1] ], [ [P.su1, P.u] ], [ [P.su2] ], [ [P.su3] ], [ [P.su3, P.t] ], [ [P.su4] ], [ [P.su4, P.u] ], [ [P.su5] ], [ [P.su6] ], [ [P.su6, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1776_3923,(0,0,11):C.UVGC_1776_3924,(0,0,16):C.UVGC_1776_3925,(0,0,17):C.UVGC_1776_3926,(0,0,18):C.UVGC_1776_3927,(0,0,19):C.UVGC_1776_3928,(0,0,20):C.UVGC_1776_3929,(0,0,21):C.UVGC_1776_3930,(0,0,24):C.UVGC_1776_3931,(0,0,26):C.UVGC_1776_3932,(0,0,27):C.UVGC_1776_3933,(0,0,29):C.UVGC_1776_3934,(0,0,31):C.UVGC_1776_3935,(0,0,32):C.UVGC_1776_3936,(0,0,34):C.UVGC_1776_3937,(0,0,1):C.UVGC_1776_3938,(0,0,2):C.UVGC_1776_3939,(0,0,3):C.UVGC_1776_3940,(0,0,4):C.UVGC_1776_3941,(0,0,7):C.UVGC_1776_3942,(0,0,8):C.UVGC_1776_3943,(0,0,9):C.UVGC_1776_3944,(0,0,10):C.UVGC_1776_3945,(0,0,13):C.UVGC_1776_3946,(0,0,15):C.UVGC_1776_3947,(0,0,12):C.UVGC_1776_3948,(0,0,22):C.UVGC_1776_3949,(0,0,23):C.UVGC_1776_3950,(0,0,25):C.UVGC_1776_3951,(0,0,28):C.UVGC_1776_3952,(0,0,30):C.UVGC_1776_3953,(0,0,33):C.UVGC_1776_3954,(0,0,5):C.UVGC_1776_3955,(0,0,6):C.UVGC_1776_3956,(0,0,14):C.UVGC_1776_3957})

V_653 = CTVertex(name = 'V_653',
                 type = 'UV',
                 particles = [ P.g, P.su5__tilde__, P.su5 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.UVGC_1584_2283,(0,0,1):C.UVGC_1584_2284,(0,0,3):C.UVGC_1584_2285,(0,0,4):C.UVGC_1584_2286,(0,0,2):C.UVGC_1584_2287,(0,0,5):C.UVGC_1584_2288,(0,1,0):C.UVGC_1583_2277,(0,1,1):C.UVGC_1583_2278,(0,1,3):C.UVGC_1583_2279,(0,1,4):C.UVGC_1583_2280,(0,1,2):C.UVGC_1583_2281,(0,1,5):C.UVGC_1583_2282})

V_654 = CTVertex(name = 'V_654',
                 type = 'UV',
                 particles = [ P.go, P.t, P.su6__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.go] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.go] ], [ [P.g, P.go, P.su6] ], [ [P.g, P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.su6, P.t] ], [ [P.g, P.t] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1] ], [ [P.su1, P.u] ], [ [P.su2] ], [ [P.su3] ], [ [P.su3, P.t] ], [ [P.su4] ], [ [P.su4, P.u] ], [ [P.su5] ], [ [P.su6] ], [ [P.su6, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1777_3958,(0,0,7):C.UVGC_1777_3959,(0,0,17):C.UVGC_1777_3960,(0,0,18):C.UVGC_1777_3961,(0,0,19):C.UVGC_1777_3962,(0,0,20):C.UVGC_1777_3963,(0,0,21):C.UVGC_1777_3964,(0,0,22):C.UVGC_1777_3965,(0,0,25):C.UVGC_1777_3966,(0,0,27):C.UVGC_1777_3967,(0,0,28):C.UVGC_1777_3968,(0,0,30):C.UVGC_1777_3969,(0,0,32):C.UVGC_1777_3970,(0,0,33):C.UVGC_1777_3971,(0,0,35):C.UVGC_1777_3972,(0,0,1):C.UVGC_1777_3973,(0,0,2):C.UVGC_1777_3974,(0,0,3):C.UVGC_1777_3975,(0,0,4):C.UVGC_1777_3976,(0,0,5):C.UVGC_1777_3977,(0,0,6):C.UVGC_1777_3978,(0,0,11):C.UVGC_1777_3979,(0,0,14):C.UVGC_1777_3980,(0,0,16):C.UVGC_1777_3981,(0,0,8):C.UVGC_1777_3982,(0,0,9):C.UVGC_1777_3983,(0,0,10):C.UVGC_1777_3984,(0,0,23):C.UVGC_1777_3985,(0,0,24):C.UVGC_1777_3986,(0,0,26):C.UVGC_1777_3987,(0,0,29):C.UVGC_1777_3988,(0,0,31):C.UVGC_1777_3989,(0,0,34):C.UVGC_1777_3990,(0,0,12):C.UVGC_1777_3991,(0,0,13):C.UVGC_1777_3992,(0,0,15):C.UVGC_1777_3993,(0,1,10):C.UVGC_1718_3338})

V_655 = CTVertex(name = 'V_655',
                 type = 'UV',
                 particles = [ P.g, P.su6__tilde__, P.su6 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1741_3370,(0,0,1):C.UVGC_1741_3371,(0,0,2):C.UVGC_1741_3372,(0,0,3):C.UVGC_1741_3373,(0,0,5):C.UVGC_1741_3374,(0,0,4):C.UVGC_1741_3375,(0,1,0):C.UVGC_1740_3364,(0,1,1):C.UVGC_1740_3365,(0,1,2):C.UVGC_1740_3366,(0,1,3):C.UVGC_1740_3367,(0,1,5):C.UVGC_1740_3368,(0,1,4):C.UVGC_1740_3369})

V_656 = CTVertex(name = 'V_656',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.go, P.sd1 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.go] ], [ [P.d, P.g, P.sd1] ], [ [P.go] ], [ [P.go, P.sd1] ], [ [P.g, P.go, P.sd1] ], [ [P.g, P.sd1] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1605_2519,(0,0,5):C.UVGC_1605_2520,(0,0,9):C.UVGC_1605_2521,(0,0,10):C.UVGC_1605_2522,(0,0,11):C.UVGC_1605_2523,(0,0,12):C.UVGC_1605_2524,(0,0,13):C.UVGC_1605_2525,(0,0,14):C.UVGC_1605_2526,(0,0,15):C.UVGC_1605_2527,(0,0,16):C.UVGC_1605_2528,(0,0,17):C.UVGC_1605_2529,(0,0,18):C.UVGC_1605_2530,(0,0,19):C.UVGC_1605_2531,(0,0,20):C.UVGC_1605_2532,(0,0,21):C.UVGC_1605_2533,(0,0,1):C.UVGC_1605_2534,(0,0,2):C.UVGC_1605_2535,(0,0,8):C.UVGC_1605_2536,(0,0,6):C.UVGC_1605_2537,(0,0,3):C.UVGC_1605_2538,(0,0,4):C.UVGC_1605_2539,(0,0,7):C.UVGC_1605_2540})

V_657 = CTVertex(name = 'V_657',
                 type = 'UV',
                 particles = [ P.a, P.g, P.sd1__tilde__, P.sd1 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.d, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd1] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1259_673,(0,0,1):C.UVGC_1259_674,(0,0,3):C.UVGC_1259_675,(0,0,4):C.UVGC_1259_676,(0,0,6):C.UVGC_1259_677,(0,0,7):C.UVGC_1259_678,(0,0,8):C.UVGC_1259_679,(0,0,9):C.UVGC_1259_680,(0,0,10):C.UVGC_1259_681,(0,0,11):C.UVGC_1259_682,(0,0,12):C.UVGC_1259_683,(0,0,13):C.UVGC_1259_684,(0,0,14):C.UVGC_1259_685,(0,0,15):C.UVGC_1259_686,(0,0,16):C.UVGC_1259_687,(0,0,17):C.UVGC_1259_688,(0,0,2):C.UVGC_1259_689,(0,0,5):C.UVGC_1259_690})

V_658 = CTVertex(name = 'V_658',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.go, P.sd2 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.go] ], [ [P.go, P.s] ], [ [P.go, P.sd2] ], [ [P.g, P.go, P.s] ], [ [P.g, P.go, P.sd2] ], [ [P.g, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.s, P.sd2] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1606_2541,(0,0,1):C.UVGC_1606_2542,(0,0,9):C.UVGC_1606_2543,(0,0,10):C.UVGC_1606_2544,(0,0,11):C.UVGC_1606_2545,(0,0,12):C.UVGC_1606_2546,(0,0,13):C.UVGC_1606_2547,(0,0,14):C.UVGC_1606_2548,(0,0,15):C.UVGC_1606_2549,(0,0,16):C.UVGC_1606_2550,(0,0,17):C.UVGC_1606_2551,(0,0,18):C.UVGC_1606_2552,(0,0,19):C.UVGC_1606_2553,(0,0,20):C.UVGC_1606_2554,(0,0,21):C.UVGC_1606_2555,(0,0,6):C.UVGC_1606_2556,(0,0,7):C.UVGC_1606_2557,(0,0,2):C.UVGC_1606_2558,(0,0,3):C.UVGC_1606_2559,(0,0,4):C.UVGC_1606_2560,(0,0,5):C.UVGC_1606_2561,(0,0,8):C.UVGC_1606_2562})

V_659 = CTVertex(name = 'V_659',
                 type = 'UV',
                 particles = [ P.a, P.g, P.sd2__tilde__, P.sd2 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1285_808,(0,0,1):C.UVGC_1285_809,(0,0,2):C.UVGC_1285_810,(0,0,3):C.UVGC_1285_811,(0,0,6):C.UVGC_1285_812,(0,0,7):C.UVGC_1285_813,(0,0,8):C.UVGC_1285_814,(0,0,9):C.UVGC_1285_815,(0,0,10):C.UVGC_1285_816,(0,0,11):C.UVGC_1285_817,(0,0,12):C.UVGC_1285_818,(0,0,13):C.UVGC_1285_819,(0,0,14):C.UVGC_1285_820,(0,0,15):C.UVGC_1285_821,(0,0,16):C.UVGC_1285_822,(0,0,17):C.UVGC_1285_823,(0,0,5):C.UVGC_1285_824,(0,0,4):C.UVGC_1285_825})

V_660 = CTVertex(name = 'V_660',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.go, P.sd3 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.g, P.go] ], [ [P.b, P.g, P.sd3] ], [ [P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.go, P.sd3] ], [ [P.g, P.sd3] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1607_2563,(0,0,5):C.UVGC_1607_2564,(0,0,10):C.UVGC_1607_2565,(0,0,11):C.UVGC_1607_2566,(0,0,12):C.UVGC_1607_2567,(0,0,13):C.UVGC_1607_2568,(0,0,14):C.UVGC_1607_2569,(0,0,15):C.UVGC_1607_2570,(0,0,16):C.UVGC_1607_2571,(0,0,17):C.UVGC_1607_2572,(0,0,18):C.UVGC_1607_2573,(0,0,19):C.UVGC_1607_2574,(0,0,20):C.UVGC_1607_2575,(0,0,21):C.UVGC_1607_2576,(0,0,22):C.UVGC_1607_2577,(0,0,1):C.UVGC_1607_2578,(0,0,2):C.UVGC_1607_2579,(0,0,9):C.UVGC_1607_2580,(0,0,6):C.UVGC_1607_2581,(0,0,7):C.UVGC_1607_2582,(0,0,3):C.UVGC_1607_2583,(0,0,4):C.UVGC_1607_2584,(0,0,8):C.UVGC_1607_2585,(0,1,2):C.UVGC_1311_989})

V_661 = CTVertex(name = 'V_661',
                 type = 'UV',
                 particles = [ P.a, P.g, P.sd3__tilde__, P.sd3 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.go] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd3] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1303_919,(0,0,2):C.UVGC_1303_920,(0,0,3):C.UVGC_1303_921,(0,0,4):C.UVGC_1303_922,(0,0,6):C.UVGC_1303_923,(0,0,7):C.UVGC_1303_924,(0,0,8):C.UVGC_1303_925,(0,0,9):C.UVGC_1303_926,(0,0,10):C.UVGC_1303_927,(0,0,11):C.UVGC_1303_928,(0,0,12):C.UVGC_1303_929,(0,0,13):C.UVGC_1303_930,(0,0,14):C.UVGC_1303_931,(0,0,15):C.UVGC_1303_932,(0,0,16):C.UVGC_1303_933,(0,0,17):C.UVGC_1303_934,(0,0,1):C.UVGC_1303_935,(0,0,5):C.UVGC_1303_936})

V_662 = CTVertex(name = 'V_662',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.go, P.sd4 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.d, P.g] ], [ [P.d, P.go] ], [ [P.d, P.g, P.go] ], [ [P.d, P.g, P.sd4] ], [ [P.go] ], [ [P.go, P.sd4] ], [ [P.g, P.go, P.sd4] ], [ [P.g, P.sd4] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1608_2586,(0,0,5):C.UVGC_1608_2587,(0,0,9):C.UVGC_1608_2588,(0,0,10):C.UVGC_1608_2589,(0,0,11):C.UVGC_1608_2590,(0,0,12):C.UVGC_1608_2591,(0,0,13):C.UVGC_1608_2592,(0,0,14):C.UVGC_1608_2593,(0,0,15):C.UVGC_1608_2594,(0,0,16):C.UVGC_1608_2595,(0,0,17):C.UVGC_1608_2596,(0,0,18):C.UVGC_1608_2597,(0,0,19):C.UVGC_1608_2598,(0,0,20):C.UVGC_1608_2599,(0,0,21):C.UVGC_1608_2600,(0,0,1):C.UVGC_1608_2601,(0,0,2):C.UVGC_1608_2602,(0,0,8):C.UVGC_1608_2603,(0,0,6):C.UVGC_1608_2604,(0,0,3):C.UVGC_1608_2605,(0,0,4):C.UVGC_1608_2606,(0,0,7):C.UVGC_1608_2607})

V_663 = CTVertex(name = 'V_663',
                 type = 'UV',
                 particles = [ P.a, P.g, P.sd4__tilde__, P.sd4 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.d, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd4] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1327_1036,(0,0,1):C.UVGC_1327_1037,(0,0,3):C.UVGC_1327_1038,(0,0,4):C.UVGC_1327_1039,(0,0,6):C.UVGC_1327_1040,(0,0,7):C.UVGC_1327_1041,(0,0,8):C.UVGC_1327_1042,(0,0,9):C.UVGC_1327_1043,(0,0,10):C.UVGC_1327_1044,(0,0,11):C.UVGC_1327_1045,(0,0,12):C.UVGC_1327_1046,(0,0,13):C.UVGC_1327_1047,(0,0,14):C.UVGC_1327_1048,(0,0,15):C.UVGC_1327_1049,(0,0,16):C.UVGC_1327_1050,(0,0,17):C.UVGC_1327_1051,(0,0,2):C.UVGC_1327_1052,(0,0,5):C.UVGC_1327_1053})

V_664 = CTVertex(name = 'V_664',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.go, P.sd5 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.go] ], [ [P.go, P.s] ], [ [P.go, P.sd5] ], [ [P.g, P.go, P.s] ], [ [P.g, P.go, P.sd5] ], [ [P.g, P.s] ], [ [P.g, P.sd5] ], [ [P.g, P.s, P.sd5] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1609_2608,(0,0,1):C.UVGC_1609_2609,(0,0,9):C.UVGC_1609_2610,(0,0,10):C.UVGC_1609_2611,(0,0,11):C.UVGC_1609_2612,(0,0,12):C.UVGC_1609_2613,(0,0,13):C.UVGC_1609_2614,(0,0,14):C.UVGC_1609_2615,(0,0,15):C.UVGC_1609_2616,(0,0,16):C.UVGC_1609_2617,(0,0,17):C.UVGC_1609_2618,(0,0,18):C.UVGC_1609_2619,(0,0,19):C.UVGC_1609_2620,(0,0,20):C.UVGC_1609_2621,(0,0,21):C.UVGC_1609_2622,(0,0,6):C.UVGC_1609_2623,(0,0,7):C.UVGC_1609_2624,(0,0,2):C.UVGC_1609_2625,(0,0,3):C.UVGC_1609_2626,(0,0,4):C.UVGC_1609_2627,(0,0,5):C.UVGC_1609_2628,(0,0,8):C.UVGC_1609_2629})

V_665 = CTVertex(name = 'V_665',
                 type = 'UV',
                 particles = [ P.a, P.g, P.sd5__tilde__, P.sd5 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.s] ], [ [P.g, P.sd5] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1352_1164,(0,0,1):C.UVGC_1352_1165,(0,0,2):C.UVGC_1352_1166,(0,0,3):C.UVGC_1352_1167,(0,0,6):C.UVGC_1352_1168,(0,0,7):C.UVGC_1352_1169,(0,0,8):C.UVGC_1352_1170,(0,0,9):C.UVGC_1352_1171,(0,0,10):C.UVGC_1352_1172,(0,0,11):C.UVGC_1352_1173,(0,0,12):C.UVGC_1352_1174,(0,0,13):C.UVGC_1352_1175,(0,0,14):C.UVGC_1352_1176,(0,0,15):C.UVGC_1352_1177,(0,0,16):C.UVGC_1352_1178,(0,0,17):C.UVGC_1352_1179,(0,0,5):C.UVGC_1352_1180,(0,0,4):C.UVGC_1352_1181})

V_666 = CTVertex(name = 'V_666',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.go, P.sd6 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.g, P.go] ], [ [P.b, P.g, P.sd6] ], [ [P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.go, P.sd6] ], [ [P.g, P.sd6] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1610_2630,(0,1,5):C.UVGC_1610_2631,(0,1,10):C.UVGC_1610_2632,(0,1,11):C.UVGC_1610_2633,(0,1,12):C.UVGC_1610_2634,(0,1,13):C.UVGC_1610_2635,(0,1,14):C.UVGC_1610_2636,(0,1,15):C.UVGC_1610_2637,(0,1,16):C.UVGC_1610_2638,(0,1,17):C.UVGC_1610_2639,(0,1,18):C.UVGC_1610_2640,(0,1,19):C.UVGC_1610_2641,(0,1,20):C.UVGC_1610_2642,(0,1,21):C.UVGC_1610_2643,(0,1,22):C.UVGC_1610_2644,(0,1,1):C.UVGC_1610_2645,(0,1,2):C.UVGC_1610_2646,(0,1,9):C.UVGC_1610_2647,(0,1,6):C.UVGC_1610_2648,(0,1,7):C.UVGC_1610_2649,(0,1,3):C.UVGC_1610_2650,(0,1,4):C.UVGC_1610_2651,(0,1,8):C.UVGC_1610_2652,(0,0,2):C.UVGC_1366_1250})

V_667 = CTVertex(name = 'V_667',
                 type = 'UV',
                 particles = [ P.a, P.g, P.sd6__tilde__, P.sd6 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.go] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd6] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1394_1293,(0,0,2):C.UVGC_1394_1294,(0,0,3):C.UVGC_1394_1295,(0,0,4):C.UVGC_1394_1296,(0,0,6):C.UVGC_1394_1297,(0,0,7):C.UVGC_1394_1298,(0,0,8):C.UVGC_1394_1299,(0,0,9):C.UVGC_1394_1300,(0,0,10):C.UVGC_1394_1301,(0,0,11):C.UVGC_1394_1302,(0,0,12):C.UVGC_1394_1303,(0,0,13):C.UVGC_1394_1304,(0,0,14):C.UVGC_1394_1305,(0,0,15):C.UVGC_1394_1306,(0,0,16):C.UVGC_1394_1307,(0,0,17):C.UVGC_1394_1308,(0,0,1):C.UVGC_1394_1309,(0,0,5):C.UVGC_1394_1310})

V_668 = CTVertex(name = 'V_668',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.go, P.su1 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.go] ], [ [P.go, P.su1] ], [ [P.go, P.u] ], [ [P.g, P.go, P.su1] ], [ [P.g, P.go, P.u] ], [ [P.g, P.su1] ], [ [P.g, P.su1, P.u] ], [ [P.g, P.u] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1611_2653,(0,0,1):C.UVGC_1611_2654,(0,0,9):C.UVGC_1611_2655,(0,0,10):C.UVGC_1611_2656,(0,0,11):C.UVGC_1611_2657,(0,0,12):C.UVGC_1611_2658,(0,0,13):C.UVGC_1611_2659,(0,0,14):C.UVGC_1611_2660,(0,0,15):C.UVGC_1611_2661,(0,0,16):C.UVGC_1611_2662,(0,0,17):C.UVGC_1611_2663,(0,0,18):C.UVGC_1611_2664,(0,0,19):C.UVGC_1611_2665,(0,0,20):C.UVGC_1611_2666,(0,0,21):C.UVGC_1611_2667,(0,0,6):C.UVGC_1611_2668,(0,0,8):C.UVGC_1611_2669,(0,0,2):C.UVGC_1611_2670,(0,0,3):C.UVGC_1611_2671,(0,0,4):C.UVGC_1611_2672,(0,0,5):C.UVGC_1611_2673,(0,0,7):C.UVGC_1611_2674})

V_669 = CTVertex(name = 'V_669',
                 type = 'UV',
                 particles = [ P.a, P.g, P.su1__tilde__, P.su1 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1476_1714,(0,0,1):C.UVGC_1476_1715,(0,0,2):C.UVGC_1476_1716,(0,0,3):C.UVGC_1476_1717,(0,0,6):C.UVGC_1476_1718,(0,0,7):C.UVGC_1476_1719,(0,0,8):C.UVGC_1476_1720,(0,0,9):C.UVGC_1476_1721,(0,0,10):C.UVGC_1476_1722,(0,0,11):C.UVGC_1476_1723,(0,0,12):C.UVGC_1476_1724,(0,0,13):C.UVGC_1476_1725,(0,0,14):C.UVGC_1476_1726,(0,0,15):C.UVGC_1476_1727,(0,0,16):C.UVGC_1476_1728,(0,0,17):C.UVGC_1476_1729,(0,0,5):C.UVGC_1476_1730,(0,0,4):C.UVGC_1476_1731})

V_670 = CTVertex(name = 'V_670',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.go, P.su2 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS3 ],
                 loop_particles = [ [ [P.b] ], [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.go] ], [ [P.c, P.g, P.su2] ], [ [P.go] ], [ [P.go, P.su2] ], [ [P.g, P.go, P.su2] ], [ [P.g, P.su2] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1612_2675,(0,0,5):C.UVGC_1612_2676,(0,0,9):C.UVGC_1612_2677,(0,0,10):C.UVGC_1612_2678,(0,0,11):C.UVGC_1612_2679,(0,0,12):C.UVGC_1612_2680,(0,0,13):C.UVGC_1612_2681,(0,0,14):C.UVGC_1612_2682,(0,0,15):C.UVGC_1612_2683,(0,0,16):C.UVGC_1612_2684,(0,0,17):C.UVGC_1612_2685,(0,0,18):C.UVGC_1612_2686,(0,0,19):C.UVGC_1612_2687,(0,0,20):C.UVGC_1612_2688,(0,0,21):C.UVGC_1612_2689,(0,0,1):C.UVGC_1612_2690,(0,0,2):C.UVGC_1612_2691,(0,0,8):C.UVGC_1612_2692,(0,0,6):C.UVGC_1612_2693,(0,0,3):C.UVGC_1612_2694,(0,0,4):C.UVGC_1612_2695,(0,0,7):C.UVGC_1612_2696})

V_671 = CTVertex(name = 'V_671',
                 type = 'UV',
                 particles = [ P.a, P.g, P.su2__tilde__, P.su2 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.su2] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1523_1970,(0,0,1):C.UVGC_1523_1971,(0,0,3):C.UVGC_1523_1972,(0,0,4):C.UVGC_1523_1973,(0,0,6):C.UVGC_1523_1974,(0,0,7):C.UVGC_1523_1975,(0,0,8):C.UVGC_1523_1976,(0,0,9):C.UVGC_1523_1977,(0,0,10):C.UVGC_1523_1978,(0,0,11):C.UVGC_1523_1979,(0,0,12):C.UVGC_1523_1980,(0,0,13):C.UVGC_1523_1981,(0,0,14):C.UVGC_1523_1982,(0,0,15):C.UVGC_1523_1983,(0,0,16):C.UVGC_1523_1984,(0,0,17):C.UVGC_1523_1985,(0,0,2):C.UVGC_1523_1986,(0,0,5):C.UVGC_1523_1987})

V_672 = CTVertex(name = 'V_672',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.go, P.su3 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.go] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.go, P.su3] ], [ [P.g, P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su3, P.t] ], [ [P.g, P.t] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1646_2911,(0,0,1):C.UVGC_1646_2912,(0,0,10):C.UVGC_1646_2913,(0,0,11):C.UVGC_1646_2914,(0,0,12):C.UVGC_1646_2915,(0,0,13):C.UVGC_1646_2916,(0,0,14):C.UVGC_1646_2917,(0,0,15):C.UVGC_1646_2918,(0,0,16):C.UVGC_1646_2919,(0,0,17):C.UVGC_1646_2920,(0,0,18):C.UVGC_1646_2921,(0,0,19):C.UVGC_1646_2922,(0,0,20):C.UVGC_1646_2923,(0,0,21):C.UVGC_1646_2924,(0,0,22):C.UVGC_1646_2925,(0,0,7):C.UVGC_1646_2926,(0,0,9):C.UVGC_1646_2927,(0,0,2):C.UVGC_1646_2928,(0,0,3):C.UVGC_1646_2929,(0,0,4):C.UVGC_1646_2930,(0,0,5):C.UVGC_1646_2931,(0,0,6):C.UVGC_1646_2932,(0,0,8):C.UVGC_1646_2933,(0,1,4):C.UVGC_1672_3110})

V_673 = CTVertex(name = 'V_673',
                 type = 'UV',
                 particles = [ P.a, P.g, P.su3__tilde__, P.su3 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1659_3004,(0,0,1):C.UVGC_1659_3005,(0,0,2):C.UVGC_1659_3006,(0,0,3):C.UVGC_1659_3007,(0,0,6):C.UVGC_1659_3008,(0,0,7):C.UVGC_1659_3009,(0,0,8):C.UVGC_1659_3010,(0,0,9):C.UVGC_1659_3011,(0,0,10):C.UVGC_1659_3012,(0,0,11):C.UVGC_1659_3013,(0,0,12):C.UVGC_1659_3014,(0,0,13):C.UVGC_1659_3015,(0,0,14):C.UVGC_1659_3016,(0,0,15):C.UVGC_1659_3017,(0,0,16):C.UVGC_1659_3018,(0,0,17):C.UVGC_1659_3019,(0,0,5):C.UVGC_1659_3020,(0,0,4):C.UVGC_1659_3021})

V_674 = CTVertex(name = 'V_674',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.go, P.su4 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.go] ], [ [P.go, P.su4] ], [ [P.go, P.u] ], [ [P.g, P.go, P.su4] ], [ [P.g, P.go, P.u] ], [ [P.g, P.su4] ], [ [P.g, P.su4, P.u] ], [ [P.g, P.u] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1613_2697,(0,0,1):C.UVGC_1613_2698,(0,0,9):C.UVGC_1613_2699,(0,0,10):C.UVGC_1613_2700,(0,0,11):C.UVGC_1613_2701,(0,0,12):C.UVGC_1613_2702,(0,0,13):C.UVGC_1613_2703,(0,0,14):C.UVGC_1613_2704,(0,0,15):C.UVGC_1613_2705,(0,0,16):C.UVGC_1613_2706,(0,0,17):C.UVGC_1613_2707,(0,0,18):C.UVGC_1613_2708,(0,0,19):C.UVGC_1613_2709,(0,0,20):C.UVGC_1613_2710,(0,0,21):C.UVGC_1613_2711,(0,0,6):C.UVGC_1613_2712,(0,0,8):C.UVGC_1613_2713,(0,0,2):C.UVGC_1613_2714,(0,0,3):C.UVGC_1613_2715,(0,0,4):C.UVGC_1613_2716,(0,0,5):C.UVGC_1613_2717,(0,0,7):C.UVGC_1613_2718})

V_675 = CTVertex(name = 'V_675',
                 type = 'UV',
                 particles = [ P.a, P.g, P.su4__tilde__, P.su4 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1559_2159,(0,0,1):C.UVGC_1559_2160,(0,0,2):C.UVGC_1559_2161,(0,0,3):C.UVGC_1559_2162,(0,0,6):C.UVGC_1559_2163,(0,0,7):C.UVGC_1559_2164,(0,0,8):C.UVGC_1559_2165,(0,0,9):C.UVGC_1559_2166,(0,0,10):C.UVGC_1559_2167,(0,0,11):C.UVGC_1559_2168,(0,0,12):C.UVGC_1559_2169,(0,0,13):C.UVGC_1559_2170,(0,0,14):C.UVGC_1559_2171,(0,0,15):C.UVGC_1559_2172,(0,0,16):C.UVGC_1559_2173,(0,0,17):C.UVGC_1559_2174,(0,0,5):C.UVGC_1559_2175,(0,0,4):C.UVGC_1559_2176})

V_676 = CTVertex(name = 'V_676',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.go, P.su5 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.c, P.g] ], [ [P.c, P.go] ], [ [P.c, P.g, P.go] ], [ [P.c, P.g, P.su5] ], [ [P.go] ], [ [P.go, P.su5] ], [ [P.g, P.go, P.su5] ], [ [P.g, P.su5] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1614_2719,(0,0,5):C.UVGC_1614_2720,(0,0,9):C.UVGC_1614_2721,(0,0,10):C.UVGC_1614_2722,(0,0,11):C.UVGC_1614_2723,(0,0,12):C.UVGC_1614_2724,(0,0,13):C.UVGC_1614_2725,(0,0,14):C.UVGC_1614_2726,(0,0,15):C.UVGC_1614_2727,(0,0,16):C.UVGC_1614_2728,(0,0,17):C.UVGC_1614_2729,(0,0,18):C.UVGC_1614_2730,(0,0,19):C.UVGC_1614_2731,(0,0,20):C.UVGC_1614_2732,(0,0,21):C.UVGC_1614_2733,(0,0,1):C.UVGC_1614_2734,(0,0,2):C.UVGC_1614_2735,(0,0,8):C.UVGC_1614_2736,(0,0,6):C.UVGC_1614_2737,(0,0,3):C.UVGC_1614_2738,(0,0,4):C.UVGC_1614_2739,(0,0,7):C.UVGC_1614_2740})

V_677 = CTVertex(name = 'V_677',
                 type = 'UV',
                 particles = [ P.a, P.g, P.su5__tilde__, P.su5 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.su5] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1585_2289,(0,0,1):C.UVGC_1585_2290,(0,0,3):C.UVGC_1585_2291,(0,0,4):C.UVGC_1585_2292,(0,0,6):C.UVGC_1585_2293,(0,0,7):C.UVGC_1585_2294,(0,0,8):C.UVGC_1585_2295,(0,0,9):C.UVGC_1585_2296,(0,0,10):C.UVGC_1585_2297,(0,0,11):C.UVGC_1585_2298,(0,0,12):C.UVGC_1585_2299,(0,0,13):C.UVGC_1585_2300,(0,0,14):C.UVGC_1585_2301,(0,0,15):C.UVGC_1585_2302,(0,0,16):C.UVGC_1585_2303,(0,0,17):C.UVGC_1585_2304,(0,0,2):C.UVGC_1585_2305,(0,0,5):C.UVGC_1585_2306})

V_678 = CTVertex(name = 'V_678',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.go, P.su6 ],
                 color = [ 'T(2,3,1)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b] ], [ [P.go] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.go, P.su6] ], [ [P.g, P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.su6, P.t] ], [ [P.g, P.t] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1702_3256,(0,1,1):C.UVGC_1702_3257,(0,1,10):C.UVGC_1702_3258,(0,1,11):C.UVGC_1702_3259,(0,1,12):C.UVGC_1702_3260,(0,1,13):C.UVGC_1702_3261,(0,1,14):C.UVGC_1702_3262,(0,1,15):C.UVGC_1702_3263,(0,1,16):C.UVGC_1702_3264,(0,1,17):C.UVGC_1702_3265,(0,1,18):C.UVGC_1702_3266,(0,1,19):C.UVGC_1702_3267,(0,1,20):C.UVGC_1702_3268,(0,1,21):C.UVGC_1702_3269,(0,1,22):C.UVGC_1702_3270,(0,1,7):C.UVGC_1702_3271,(0,1,9):C.UVGC_1702_3272,(0,1,2):C.UVGC_1702_3273,(0,1,3):C.UVGC_1702_3274,(0,1,4):C.UVGC_1702_3275,(0,1,5):C.UVGC_1702_3276,(0,1,6):C.UVGC_1702_3277,(0,1,8):C.UVGC_1702_3278,(0,0,4):C.UVGC_1699_3251})

V_679 = CTVertex(name = 'V_679',
                 type = 'UV',
                 particles = [ P.a, P.g, P.su6__tilde__, P.su6 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1742_3376,(0,0,1):C.UVGC_1742_3377,(0,0,2):C.UVGC_1742_3378,(0,0,3):C.UVGC_1742_3379,(0,0,6):C.UVGC_1742_3380,(0,0,7):C.UVGC_1742_3381,(0,0,8):C.UVGC_1742_3382,(0,0,9):C.UVGC_1742_3383,(0,0,10):C.UVGC_1742_3384,(0,0,11):C.UVGC_1742_3385,(0,0,12):C.UVGC_1742_3386,(0,0,13):C.UVGC_1742_3387,(0,0,14):C.UVGC_1742_3388,(0,0,15):C.UVGC_1742_3389,(0,0,16):C.UVGC_1742_3390,(0,0,17):C.UVGC_1742_3391,(0,0,5):C.UVGC_1742_3392,(0,0,4):C.UVGC_1742_3393})

V_680 = CTVertex(name = 'V_680',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sd4__tilde__, P.sd4 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.d, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd4] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(2,0,0):C.UVGC_1328_1054,(2,0,1):C.UVGC_1328_1055,(2,0,3):C.UVGC_1328_1056,(2,0,4):C.UVGC_1328_1057,(2,0,6):C.UVGC_1328_1058,(2,0,7):C.UVGC_1328_1059,(2,0,8):C.UVGC_1328_1060,(2,0,9):C.UVGC_1209_318,(2,0,10):C.UVGC_1328_1061,(2,0,11):C.UVGC_1328_1062,(2,0,12):C.UVGC_1328_1063,(2,0,13):C.UVGC_1328_1064,(2,0,14):C.UVGC_1328_1065,(2,0,15):C.UVGC_1328_1066,(2,0,16):C.UVGC_1328_1067,(2,0,17):C.UVGC_1328_1068,(2,0,2):C.UVGC_1328_1069,(2,0,5):C.UVGC_1328_1070,(1,0,0):C.UVGC_1328_1054,(1,0,1):C.UVGC_1328_1055,(1,0,3):C.UVGC_1328_1056,(1,0,4):C.UVGC_1328_1057,(1,0,6):C.UVGC_1328_1058,(1,0,7):C.UVGC_1328_1059,(1,0,8):C.UVGC_1328_1060,(1,0,9):C.UVGC_1209_318,(1,0,10):C.UVGC_1328_1061,(1,0,11):C.UVGC_1328_1062,(1,0,12):C.UVGC_1328_1063,(1,0,13):C.UVGC_1328_1064,(1,0,14):C.UVGC_1328_1065,(1,0,15):C.UVGC_1328_1066,(1,0,16):C.UVGC_1328_1067,(1,0,17):C.UVGC_1328_1068,(1,0,2):C.UVGC_1328_1069,(1,0,5):C.UVGC_1328_1070,(0,0,3):C.UVGC_1209_314,(0,0,6):C.UVGC_1209_315,(0,0,7):C.UVGC_1209_316,(0,0,8):C.UVGC_1209_317,(0,0,9):C.UVGC_1209_318,(0,0,10):C.UVGC_1209_319,(0,0,11):C.UVGC_1209_320,(0,0,12):C.UVGC_1209_321,(0,0,13):C.UVGC_1209_322,(0,0,14):C.UVGC_1209_323,(0,0,15):C.UVGC_1209_324,(0,0,16):C.UVGC_1209_325,(0,0,17):C.UVGC_1209_326,(0,0,5):C.UVGC_1209_327})

V_681 = CTVertex(name = 'V_681',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sd5__tilde__, P.sd5 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.s] ], [ [P.g, P.sd5] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(2,0,0):C.UVGC_1353_1182,(2,0,1):C.UVGC_1353_1183,(2,0,2):C.UVGC_1353_1184,(2,0,3):C.UVGC_1353_1185,(2,0,6):C.UVGC_1353_1186,(2,0,7):C.UVGC_1353_1187,(2,0,8):C.UVGC_1353_1188,(2,0,9):C.UVGC_1353_1189,(2,0,10):C.UVGC_1210_333,(2,0,11):C.UVGC_1353_1190,(2,0,12):C.UVGC_1353_1191,(2,0,13):C.UVGC_1353_1192,(2,0,14):C.UVGC_1353_1193,(2,0,15):C.UVGC_1353_1194,(2,0,16):C.UVGC_1353_1195,(2,0,17):C.UVGC_1353_1196,(2,0,5):C.UVGC_1353_1197,(2,0,4):C.UVGC_1353_1198,(1,0,0):C.UVGC_1353_1182,(1,0,1):C.UVGC_1353_1183,(1,0,2):C.UVGC_1353_1184,(1,0,3):C.UVGC_1353_1185,(1,0,6):C.UVGC_1353_1186,(1,0,7):C.UVGC_1353_1187,(1,0,8):C.UVGC_1353_1188,(1,0,9):C.UVGC_1353_1189,(1,0,10):C.UVGC_1210_333,(1,0,11):C.UVGC_1353_1190,(1,0,12):C.UVGC_1353_1191,(1,0,13):C.UVGC_1353_1192,(1,0,14):C.UVGC_1353_1193,(1,0,15):C.UVGC_1353_1194,(1,0,16):C.UVGC_1353_1195,(1,0,17):C.UVGC_1353_1196,(1,0,5):C.UVGC_1353_1197,(1,0,4):C.UVGC_1353_1198,(0,0,2):C.UVGC_1210_328,(0,0,6):C.UVGC_1210_329,(0,0,7):C.UVGC_1210_330,(0,0,8):C.UVGC_1210_331,(0,0,9):C.UVGC_1210_332,(0,0,10):C.UVGC_1210_333,(0,0,11):C.UVGC_1210_334,(0,0,12):C.UVGC_1210_335,(0,0,13):C.UVGC_1210_336,(0,0,14):C.UVGC_1210_337,(0,0,15):C.UVGC_1210_338,(0,0,16):C.UVGC_1210_339,(0,0,17):C.UVGC_1210_340,(0,0,5):C.UVGC_1210_341})

V_682 = CTVertex(name = 'V_682',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.go] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd6] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(2,0,0):C.UVGC_1395_1311,(2,0,2):C.UVGC_1395_1312,(2,0,3):C.UVGC_1395_1313,(2,0,4):C.UVGC_1395_1314,(2,0,6):C.UVGC_1395_1315,(2,0,7):C.UVGC_1395_1316,(2,0,8):C.UVGC_1395_1317,(2,0,9):C.UVGC_1395_1318,(2,0,10):C.UVGC_1395_1319,(2,0,11):C.UVGC_1211_348,(2,0,12):C.UVGC_1395_1320,(2,0,13):C.UVGC_1395_1321,(2,0,14):C.UVGC_1395_1322,(2,0,15):C.UVGC_1395_1323,(2,0,16):C.UVGC_1395_1324,(2,0,17):C.UVGC_1395_1325,(2,0,1):C.UVGC_1395_1326,(2,0,5):C.UVGC_1395_1327,(1,0,0):C.UVGC_1395_1311,(1,0,2):C.UVGC_1395_1312,(1,0,3):C.UVGC_1395_1313,(1,0,4):C.UVGC_1395_1314,(1,0,6):C.UVGC_1395_1315,(1,0,7):C.UVGC_1395_1316,(1,0,8):C.UVGC_1395_1317,(1,0,9):C.UVGC_1395_1318,(1,0,10):C.UVGC_1395_1319,(1,0,11):C.UVGC_1211_348,(1,0,12):C.UVGC_1395_1320,(1,0,13):C.UVGC_1395_1321,(1,0,14):C.UVGC_1395_1322,(1,0,15):C.UVGC_1395_1323,(1,0,16):C.UVGC_1395_1324,(1,0,17):C.UVGC_1395_1325,(1,0,1):C.UVGC_1395_1326,(1,0,5):C.UVGC_1395_1327,(0,0,3):C.UVGC_1211_342,(0,0,6):C.UVGC_1211_343,(0,0,7):C.UVGC_1211_344,(0,0,8):C.UVGC_1211_345,(0,0,9):C.UVGC_1211_346,(0,0,10):C.UVGC_1211_347,(0,0,11):C.UVGC_1211_348,(0,0,12):C.UVGC_1211_349,(0,0,13):C.UVGC_1211_350,(0,0,14):C.UVGC_1211_351,(0,0,15):C.UVGC_1211_352,(0,0,16):C.UVGC_1211_353,(0,0,17):C.UVGC_1211_354,(0,0,5):C.UVGC_1211_355})

V_683 = CTVertex(name = 'V_683',
                 type = 'UV',
                 particles = [ P.g, P.g, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(2,0,0):C.UVGC_1560_2177,(2,0,1):C.UVGC_1560_2178,(2,0,2):C.UVGC_1560_2179,(2,0,3):C.UVGC_1560_2180,(2,0,6):C.UVGC_1560_2181,(2,0,7):C.UVGC_1560_2182,(2,0,8):C.UVGC_1560_2183,(2,0,9):C.UVGC_1560_2184,(2,0,10):C.UVGC_1560_2185,(2,0,11):C.UVGC_1560_2186,(2,0,12):C.UVGC_1560_2187,(2,0,13):C.UVGC_1560_2188,(2,0,14):C.UVGC_1560_2189,(2,0,15):C.UVGC_1215_408,(2,0,16):C.UVGC_1560_2190,(2,0,17):C.UVGC_1560_2191,(2,0,5):C.UVGC_1560_2192,(2,0,4):C.UVGC_1560_2193,(1,0,0):C.UVGC_1560_2177,(1,0,1):C.UVGC_1560_2178,(1,0,2):C.UVGC_1560_2179,(1,0,3):C.UVGC_1560_2180,(1,0,6):C.UVGC_1560_2181,(1,0,7):C.UVGC_1560_2182,(1,0,8):C.UVGC_1560_2183,(1,0,9):C.UVGC_1560_2184,(1,0,10):C.UVGC_1560_2185,(1,0,11):C.UVGC_1560_2186,(1,0,12):C.UVGC_1560_2187,(1,0,13):C.UVGC_1560_2188,(1,0,14):C.UVGC_1560_2189,(1,0,15):C.UVGC_1215_408,(1,0,16):C.UVGC_1560_2190,(1,0,17):C.UVGC_1560_2191,(1,0,5):C.UVGC_1560_2192,(1,0,4):C.UVGC_1560_2193,(0,0,2):C.UVGC_1215_398,(0,0,6):C.UVGC_1215_399,(0,0,7):C.UVGC_1215_400,(0,0,8):C.UVGC_1215_401,(0,0,9):C.UVGC_1215_402,(0,0,10):C.UVGC_1215_403,(0,0,11):C.UVGC_1215_404,(0,0,12):C.UVGC_1215_405,(0,0,13):C.UVGC_1215_406,(0,0,14):C.UVGC_1215_407,(0,0,15):C.UVGC_1215_408,(0,0,16):C.UVGC_1215_409,(0,0,17):C.UVGC_1215_410,(0,0,5):C.UVGC_1215_411})

V_684 = CTVertex(name = 'V_684',
                 type = 'UV',
                 particles = [ P.g, P.g, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.su5] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(2,0,0):C.UVGC_1586_2307,(2,0,1):C.UVGC_1586_2308,(2,0,3):C.UVGC_1586_2309,(2,0,4):C.UVGC_1586_2310,(2,0,6):C.UVGC_1586_2311,(2,0,7):C.UVGC_1586_2312,(2,0,8):C.UVGC_1586_2313,(2,0,9):C.UVGC_1586_2314,(2,0,10):C.UVGC_1586_2315,(2,0,11):C.UVGC_1586_2316,(2,0,12):C.UVGC_1586_2317,(2,0,13):C.UVGC_1586_2318,(2,0,14):C.UVGC_1586_2319,(2,0,15):C.UVGC_1586_2320,(2,0,16):C.UVGC_1216_423,(2,0,17):C.UVGC_1586_2321,(2,0,2):C.UVGC_1586_2322,(2,0,5):C.UVGC_1586_2323,(1,0,0):C.UVGC_1586_2307,(1,0,1):C.UVGC_1586_2308,(1,0,3):C.UVGC_1586_2309,(1,0,4):C.UVGC_1586_2310,(1,0,6):C.UVGC_1586_2311,(1,0,7):C.UVGC_1586_2312,(1,0,8):C.UVGC_1586_2313,(1,0,9):C.UVGC_1586_2314,(1,0,10):C.UVGC_1586_2315,(1,0,11):C.UVGC_1586_2316,(1,0,12):C.UVGC_1586_2317,(1,0,13):C.UVGC_1586_2318,(1,0,14):C.UVGC_1586_2319,(1,0,15):C.UVGC_1586_2320,(1,0,16):C.UVGC_1216_423,(1,0,17):C.UVGC_1586_2321,(1,0,2):C.UVGC_1586_2322,(1,0,5):C.UVGC_1586_2323,(0,0,3):C.UVGC_1216_412,(0,0,6):C.UVGC_1216_413,(0,0,7):C.UVGC_1216_414,(0,0,8):C.UVGC_1216_415,(0,0,9):C.UVGC_1216_416,(0,0,10):C.UVGC_1216_417,(0,0,11):C.UVGC_1216_418,(0,0,12):C.UVGC_1216_419,(0,0,13):C.UVGC_1216_420,(0,0,14):C.UVGC_1216_421,(0,0,15):C.UVGC_1216_422,(0,0,16):C.UVGC_1216_423,(0,0,17):C.UVGC_1216_424,(0,0,5):C.UVGC_1216_425})

V_685 = CTVertex(name = 'V_685',
                 type = 'UV',
                 particles = [ P.g, P.g, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(2,0,0):C.UVGC_1743_3394,(2,0,1):C.UVGC_1743_3395,(2,0,2):C.UVGC_1743_3396,(2,0,3):C.UVGC_1743_3397,(2,0,6):C.UVGC_1743_3398,(2,0,7):C.UVGC_1743_3399,(2,0,8):C.UVGC_1743_3400,(2,0,9):C.UVGC_1743_3401,(2,0,10):C.UVGC_1743_3402,(2,0,11):C.UVGC_1743_3403,(2,0,12):C.UVGC_1743_3404,(2,0,13):C.UVGC_1743_3405,(2,0,14):C.UVGC_1743_3406,(2,0,15):C.UVGC_1743_3407,(2,0,16):C.UVGC_1743_3408,(2,0,17):C.UVGC_1217_438,(2,0,5):C.UVGC_1743_3409,(2,0,4):C.UVGC_1743_3410,(1,0,0):C.UVGC_1743_3394,(1,0,1):C.UVGC_1743_3395,(1,0,2):C.UVGC_1743_3396,(1,0,3):C.UVGC_1743_3397,(1,0,6):C.UVGC_1743_3398,(1,0,7):C.UVGC_1743_3399,(1,0,8):C.UVGC_1743_3400,(1,0,9):C.UVGC_1743_3401,(1,0,10):C.UVGC_1743_3402,(1,0,11):C.UVGC_1743_3403,(1,0,12):C.UVGC_1743_3404,(1,0,13):C.UVGC_1743_3405,(1,0,14):C.UVGC_1743_3406,(1,0,15):C.UVGC_1743_3407,(1,0,16):C.UVGC_1743_3408,(1,0,17):C.UVGC_1217_438,(1,0,5):C.UVGC_1743_3409,(1,0,4):C.UVGC_1743_3410,(0,0,2):C.UVGC_1217_426,(0,0,6):C.UVGC_1217_427,(0,0,7):C.UVGC_1217_428,(0,0,8):C.UVGC_1217_429,(0,0,9):C.UVGC_1217_430,(0,0,10):C.UVGC_1217_431,(0,0,11):C.UVGC_1217_432,(0,0,12):C.UVGC_1217_433,(0,0,13):C.UVGC_1217_434,(0,0,14):C.UVGC_1217_435,(0,0,15):C.UVGC_1217_436,(0,0,16):C.UVGC_1217_437,(0,0,17):C.UVGC_1217_438,(0,0,5):C.UVGC_1217_439})

V_686 = CTVertex(name = 'V_686',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.d, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd1] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(2,0,0):C.UVGC_1260_691,(2,0,1):C.UVGC_1260_692,(2,0,3):C.UVGC_1260_693,(2,0,4):C.UVGC_1260_694,(2,0,6):C.UVGC_1206_273,(2,0,7):C.UVGC_1260_695,(2,0,8):C.UVGC_1260_696,(2,0,9):C.UVGC_1260_697,(2,0,10):C.UVGC_1260_698,(2,0,11):C.UVGC_1260_699,(2,0,12):C.UVGC_1260_700,(2,0,13):C.UVGC_1260_701,(2,0,14):C.UVGC_1260_702,(2,0,15):C.UVGC_1260_703,(2,0,16):C.UVGC_1260_704,(2,0,17):C.UVGC_1260_705,(2,0,2):C.UVGC_1260_706,(2,0,5):C.UVGC_1260_707,(1,0,0):C.UVGC_1260_691,(1,0,1):C.UVGC_1260_692,(1,0,3):C.UVGC_1260_693,(1,0,4):C.UVGC_1260_694,(1,0,6):C.UVGC_1206_273,(1,0,7):C.UVGC_1260_695,(1,0,8):C.UVGC_1260_696,(1,0,9):C.UVGC_1260_697,(1,0,10):C.UVGC_1260_698,(1,0,11):C.UVGC_1260_699,(1,0,12):C.UVGC_1260_700,(1,0,13):C.UVGC_1260_701,(1,0,14):C.UVGC_1260_702,(1,0,15):C.UVGC_1260_703,(1,0,16):C.UVGC_1260_704,(1,0,17):C.UVGC_1260_705,(1,0,2):C.UVGC_1260_706,(1,0,5):C.UVGC_1260_707,(0,0,3):C.UVGC_1206_272,(0,0,6):C.UVGC_1206_273,(0,0,7):C.UVGC_1206_274,(0,0,8):C.UVGC_1206_275,(0,0,9):C.UVGC_1206_276,(0,0,10):C.UVGC_1206_277,(0,0,11):C.UVGC_1206_278,(0,0,12):C.UVGC_1206_279,(0,0,13):C.UVGC_1206_280,(0,0,14):C.UVGC_1206_281,(0,0,15):C.UVGC_1206_282,(0,0,16):C.UVGC_1206_283,(0,0,17):C.UVGC_1206_284,(0,0,5):C.UVGC_1206_285})

V_687 = CTVertex(name = 'V_687',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(2,0,0):C.UVGC_1286_826,(2,0,1):C.UVGC_1286_827,(2,0,2):C.UVGC_1286_828,(2,0,3):C.UVGC_1286_829,(2,0,6):C.UVGC_1286_830,(2,0,7):C.UVGC_1207_288,(2,0,8):C.UVGC_1286_831,(2,0,9):C.UVGC_1286_832,(2,0,10):C.UVGC_1286_833,(2,0,11):C.UVGC_1286_834,(2,0,12):C.UVGC_1286_835,(2,0,13):C.UVGC_1286_836,(2,0,14):C.UVGC_1286_837,(2,0,15):C.UVGC_1286_838,(2,0,16):C.UVGC_1286_839,(2,0,17):C.UVGC_1286_840,(2,0,5):C.UVGC_1286_841,(2,0,4):C.UVGC_1286_842,(1,0,0):C.UVGC_1286_826,(1,0,1):C.UVGC_1286_827,(1,0,2):C.UVGC_1286_828,(1,0,3):C.UVGC_1286_829,(1,0,6):C.UVGC_1286_830,(1,0,7):C.UVGC_1207_288,(1,0,8):C.UVGC_1286_831,(1,0,9):C.UVGC_1286_832,(1,0,10):C.UVGC_1286_833,(1,0,11):C.UVGC_1286_834,(1,0,12):C.UVGC_1286_835,(1,0,13):C.UVGC_1286_836,(1,0,14):C.UVGC_1286_837,(1,0,15):C.UVGC_1286_838,(1,0,16):C.UVGC_1286_839,(1,0,17):C.UVGC_1286_840,(1,0,5):C.UVGC_1286_841,(1,0,4):C.UVGC_1286_842,(0,0,2):C.UVGC_1207_286,(0,0,6):C.UVGC_1207_287,(0,0,7):C.UVGC_1207_288,(0,0,8):C.UVGC_1207_289,(0,0,9):C.UVGC_1207_290,(0,0,10):C.UVGC_1207_291,(0,0,11):C.UVGC_1207_292,(0,0,12):C.UVGC_1207_293,(0,0,13):C.UVGC_1207_294,(0,0,14):C.UVGC_1207_295,(0,0,15):C.UVGC_1207_296,(0,0,16):C.UVGC_1207_297,(0,0,17):C.UVGC_1207_298,(0,0,5):C.UVGC_1207_299})

V_688 = CTVertex(name = 'V_688',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.go] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd3] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(2,0,0):C.UVGC_1304_937,(2,0,2):C.UVGC_1304_938,(2,0,3):C.UVGC_1304_939,(2,0,4):C.UVGC_1304_940,(2,0,6):C.UVGC_1304_941,(2,0,7):C.UVGC_1304_942,(2,0,8):C.UVGC_1208_303,(2,0,9):C.UVGC_1304_943,(2,0,10):C.UVGC_1304_944,(2,0,11):C.UVGC_1304_945,(2,0,12):C.UVGC_1304_946,(2,0,13):C.UVGC_1304_947,(2,0,14):C.UVGC_1304_948,(2,0,15):C.UVGC_1304_949,(2,0,16):C.UVGC_1304_950,(2,0,17):C.UVGC_1304_951,(2,0,1):C.UVGC_1304_952,(2,0,5):C.UVGC_1304_953,(1,0,0):C.UVGC_1304_937,(1,0,2):C.UVGC_1304_938,(1,0,3):C.UVGC_1304_939,(1,0,4):C.UVGC_1304_940,(1,0,6):C.UVGC_1304_941,(1,0,7):C.UVGC_1304_942,(1,0,8):C.UVGC_1208_303,(1,0,9):C.UVGC_1304_943,(1,0,10):C.UVGC_1304_944,(1,0,11):C.UVGC_1304_945,(1,0,12):C.UVGC_1304_946,(1,0,13):C.UVGC_1304_947,(1,0,14):C.UVGC_1304_948,(1,0,15):C.UVGC_1304_949,(1,0,16):C.UVGC_1304_950,(1,0,17):C.UVGC_1304_951,(1,0,1):C.UVGC_1304_952,(1,0,5):C.UVGC_1304_953,(0,0,3):C.UVGC_1208_300,(0,0,6):C.UVGC_1208_301,(0,0,7):C.UVGC_1208_302,(0,0,8):C.UVGC_1208_303,(0,0,9):C.UVGC_1208_304,(0,0,10):C.UVGC_1208_305,(0,0,11):C.UVGC_1208_306,(0,0,12):C.UVGC_1208_307,(0,0,13):C.UVGC_1208_308,(0,0,14):C.UVGC_1208_309,(0,0,15):C.UVGC_1208_310,(0,0,16):C.UVGC_1208_311,(0,0,17):C.UVGC_1208_312,(0,0,5):C.UVGC_1208_313})

V_689 = CTVertex(name = 'V_689',
                 type = 'UV',
                 particles = [ P.g, P.g, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(2,0,0):C.UVGC_1477_1732,(2,0,1):C.UVGC_1477_1733,(2,0,2):C.UVGC_1477_1734,(2,0,3):C.UVGC_1477_1735,(2,0,6):C.UVGC_1477_1736,(2,0,7):C.UVGC_1477_1737,(2,0,8):C.UVGC_1477_1738,(2,0,9):C.UVGC_1477_1739,(2,0,10):C.UVGC_1477_1740,(2,0,11):C.UVGC_1477_1741,(2,0,12):C.UVGC_1212_363,(2,0,13):C.UVGC_1477_1742,(2,0,14):C.UVGC_1477_1743,(2,0,15):C.UVGC_1477_1744,(2,0,16):C.UVGC_1477_1745,(2,0,17):C.UVGC_1477_1746,(2,0,5):C.UVGC_1477_1747,(2,0,4):C.UVGC_1477_1748,(1,0,0):C.UVGC_1477_1732,(1,0,1):C.UVGC_1477_1733,(1,0,2):C.UVGC_1477_1734,(1,0,3):C.UVGC_1477_1735,(1,0,6):C.UVGC_1477_1736,(1,0,7):C.UVGC_1477_1737,(1,0,8):C.UVGC_1477_1738,(1,0,9):C.UVGC_1477_1739,(1,0,10):C.UVGC_1477_1740,(1,0,11):C.UVGC_1477_1741,(1,0,12):C.UVGC_1212_363,(1,0,13):C.UVGC_1477_1742,(1,0,14):C.UVGC_1477_1743,(1,0,15):C.UVGC_1477_1744,(1,0,16):C.UVGC_1477_1745,(1,0,17):C.UVGC_1477_1746,(1,0,5):C.UVGC_1477_1747,(1,0,4):C.UVGC_1477_1748,(0,0,2):C.UVGC_1212_356,(0,0,6):C.UVGC_1212_357,(0,0,7):C.UVGC_1212_358,(0,0,8):C.UVGC_1212_359,(0,0,9):C.UVGC_1212_360,(0,0,10):C.UVGC_1212_361,(0,0,11):C.UVGC_1212_362,(0,0,12):C.UVGC_1212_363,(0,0,13):C.UVGC_1212_364,(0,0,14):C.UVGC_1212_365,(0,0,15):C.UVGC_1212_366,(0,0,16):C.UVGC_1212_367,(0,0,17):C.UVGC_1212_368,(0,0,5):C.UVGC_1212_369})

V_690 = CTVertex(name = 'V_690',
                 type = 'UV',
                 particles = [ P.g, P.g, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.su2] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(2,0,0):C.UVGC_1524_1988,(2,0,1):C.UVGC_1524_1989,(2,0,3):C.UVGC_1524_1990,(2,0,4):C.UVGC_1524_1991,(2,0,6):C.UVGC_1524_1992,(2,0,7):C.UVGC_1524_1993,(2,0,8):C.UVGC_1524_1994,(2,0,9):C.UVGC_1524_1995,(2,0,10):C.UVGC_1524_1996,(2,0,11):C.UVGC_1524_1997,(2,0,12):C.UVGC_1524_1998,(2,0,13):C.UVGC_1213_378,(2,0,14):C.UVGC_1524_1999,(2,0,15):C.UVGC_1524_2000,(2,0,16):C.UVGC_1524_2001,(2,0,17):C.UVGC_1524_2002,(2,0,2):C.UVGC_1524_2003,(2,0,5):C.UVGC_1524_2004,(1,0,0):C.UVGC_1524_1988,(1,0,1):C.UVGC_1524_1989,(1,0,3):C.UVGC_1524_1990,(1,0,4):C.UVGC_1524_1991,(1,0,6):C.UVGC_1524_1992,(1,0,7):C.UVGC_1524_1993,(1,0,8):C.UVGC_1524_1994,(1,0,9):C.UVGC_1524_1995,(1,0,10):C.UVGC_1524_1996,(1,0,11):C.UVGC_1524_1997,(1,0,12):C.UVGC_1524_1998,(1,0,13):C.UVGC_1213_378,(1,0,14):C.UVGC_1524_1999,(1,0,15):C.UVGC_1524_2000,(1,0,16):C.UVGC_1524_2001,(1,0,17):C.UVGC_1524_2002,(1,0,2):C.UVGC_1524_2003,(1,0,5):C.UVGC_1524_2004,(0,0,3):C.UVGC_1213_370,(0,0,6):C.UVGC_1213_371,(0,0,7):C.UVGC_1213_372,(0,0,8):C.UVGC_1213_373,(0,0,9):C.UVGC_1213_374,(0,0,10):C.UVGC_1213_375,(0,0,11):C.UVGC_1213_376,(0,0,12):C.UVGC_1213_377,(0,0,13):C.UVGC_1213_378,(0,0,14):C.UVGC_1213_379,(0,0,15):C.UVGC_1213_380,(0,0,16):C.UVGC_1213_381,(0,0,17):C.UVGC_1213_382,(0,0,5):C.UVGC_1213_383})

V_691 = CTVertex(name = 'V_691',
                 type = 'UV',
                 particles = [ P.g, P.g, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(2,0,0):C.UVGC_1660_3022,(2,0,1):C.UVGC_1660_3023,(2,0,2):C.UVGC_1660_3024,(2,0,3):C.UVGC_1660_3025,(2,0,6):C.UVGC_1660_3026,(2,0,7):C.UVGC_1660_3027,(2,0,8):C.UVGC_1660_3028,(2,0,9):C.UVGC_1660_3029,(2,0,10):C.UVGC_1660_3030,(2,0,11):C.UVGC_1660_3031,(2,0,12):C.UVGC_1660_3032,(2,0,13):C.UVGC_1660_3033,(2,0,14):C.UVGC_1214_393,(2,0,15):C.UVGC_1660_3034,(2,0,16):C.UVGC_1660_3035,(2,0,17):C.UVGC_1660_3036,(2,0,5):C.UVGC_1660_3037,(2,0,4):C.UVGC_1660_3038,(1,0,0):C.UVGC_1660_3022,(1,0,1):C.UVGC_1660_3023,(1,0,2):C.UVGC_1660_3024,(1,0,3):C.UVGC_1660_3025,(1,0,6):C.UVGC_1660_3026,(1,0,7):C.UVGC_1660_3027,(1,0,8):C.UVGC_1660_3028,(1,0,9):C.UVGC_1660_3029,(1,0,10):C.UVGC_1660_3030,(1,0,11):C.UVGC_1660_3031,(1,0,12):C.UVGC_1660_3032,(1,0,13):C.UVGC_1660_3033,(1,0,14):C.UVGC_1214_393,(1,0,15):C.UVGC_1660_3034,(1,0,16):C.UVGC_1660_3035,(1,0,17):C.UVGC_1660_3036,(1,0,5):C.UVGC_1660_3037,(1,0,4):C.UVGC_1660_3038,(0,0,2):C.UVGC_1214_384,(0,0,6):C.UVGC_1214_385,(0,0,7):C.UVGC_1214_386,(0,0,8):C.UVGC_1214_387,(0,0,9):C.UVGC_1214_388,(0,0,10):C.UVGC_1214_389,(0,0,11):C.UVGC_1214_390,(0,0,12):C.UVGC_1214_391,(0,0,13):C.UVGC_1214_392,(0,0,14):C.UVGC_1214_393,(0,0,15):C.UVGC_1214_394,(0,0,16):C.UVGC_1214_395,(0,0,17):C.UVGC_1214_396,(0,0,5):C.UVGC_1214_397})

V_692 = CTVertex(name = 'V_692',
                 type = 'UV',
                 particles = [ P.h02, P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ], [ [P.g, P.sd6] ], [ [P.sd3, P.sd6], [P.g, P.sd3, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1929_4077,(0,0,1):C.UVGC_1929_4078,(0,0,4):C.UVGC_1929_4079,(0,0,5):C.UVGC_1929_4080,(0,0,2):C.UVGC_1929_4081,(0,0,3):C.UVGC_1929_4082,(0,0,6):C.UVGC_1929_4083})

V_693 = CTVertex(name = 'V_693',
                 type = 'UV',
                 particles = [ P.G0, P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ], [ [P.g, P.sd6] ], [ [P.sd3, P.sd6], [P.g, P.sd3, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_2192_4233,(0,0,1):C.UVGC_2192_4234,(0,0,4):C.UVGC_2192_4235,(0,0,5):C.UVGC_2192_4236,(0,0,2):C.UVGC_2192_4237,(0,0,3):C.UVGC_2192_4238,(0,0,6):C.UVGC_2192_4239})

V_694 = CTVertex(name = 'V_694',
                 type = 'UV',
                 particles = [ P.h01, P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ], [ [P.g, P.sd6] ], [ [P.sd3, P.sd6], [P.g, P.sd3, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1924_4048,(0,0,1):C.UVGC_1924_4049,(0,0,4):C.UVGC_1924_4050,(0,0,5):C.UVGC_1924_4051,(0,0,2):C.UVGC_1924_4052,(0,0,3):C.UVGC_1924_4053,(0,0,6):C.UVGC_1924_4054})

V_695 = CTVertex(name = 'V_695',
                 type = 'UV',
                 particles = [ P.A0, P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.g, P.sd3] ], [ [P.g, P.sd6] ], [ [P.sd3, P.sd6], [P.g, P.sd3, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_2187_4214,(0,0,1):C.UVGC_2187_4215,(0,0,4):C.UVGC_2187_4216,(0,0,5):C.UVGC_2187_4217,(0,0,2):C.UVGC_2187_4218,(0,0,3):C.UVGC_2187_4219,(0,0,6):C.UVGC_2187_4220})

V_696 = CTVertex(name = 'V_696',
                 type = 'UV',
                 particles = [ P.G__plus__, P.sd6, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.t] ], [ [P.g, P.sd6] ], [ [P.g, P.su3] ], [ [P.sd6, P.su3], [P.g, P.sd6, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_2213_4384,(0,0,1):C.UVGC_2213_4385,(0,0,6):C.UVGC_2213_4386,(0,0,7):C.UVGC_2213_4387,(0,0,3):C.UVGC_2213_4388,(0,0,4):C.UVGC_2213_4389,(0,0,5):C.UVGC_2213_4390,(0,0,8):C.UVGC_2213_4391,(0,0,2):C.UVGC_2213_4392})

V_697 = CTVertex(name = 'V_697',
                 type = 'UV',
                 particles = [ P.H__plus__, P.sd6, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.t] ], [ [P.g, P.sd6] ], [ [P.g, P.su3] ], [ [P.sd6, P.su3], [P.g, P.sd6, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_2211_4362,(0,0,1):C.UVGC_2211_4363,(0,0,6):C.UVGC_2211_4364,(0,0,7):C.UVGC_2211_4365,(0,0,3):C.UVGC_2211_4366,(0,0,4):C.UVGC_2211_4367,(0,0,5):C.UVGC_2211_4368,(0,0,8):C.UVGC_2211_4369,(0,0,2):C.UVGC_2211_4370})

V_698 = CTVertex(name = 'V_698',
                 type = 'UV',
                 particles = [ P.H__minus__, P.sd3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.sd3, P.su6], [P.g, P.sd3, P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2218_4425,(0,0,5):C.UVGC_2218_4426,(0,0,6):C.UVGC_2218_4427,(0,0,7):C.UVGC_2218_4428,(0,0,2):C.UVGC_2218_4429,(0,0,3):C.UVGC_2218_4430,(0,0,4):C.UVGC_2218_4431,(0,0,8):C.UVGC_2218_4432,(0,0,1):C.UVGC_2218_4433})

V_699 = CTVertex(name = 'V_699',
                 type = 'UV',
                 particles = [ P.G__minus__, P.sd3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.sd3, P.su6], [P.g, P.sd3, P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2217_4416,(0,0,5):C.UVGC_2217_4417,(0,0,6):C.UVGC_2217_4418,(0,0,7):C.UVGC_2217_4419,(0,0,2):C.UVGC_2217_4420,(0,0,3):C.UVGC_2217_4421,(0,0,4):C.UVGC_2217_4422,(0,0,8):C.UVGC_2217_4423,(0,0,1):C.UVGC_2217_4424})

V_700 = CTVertex(name = 'V_700',
                 type = 'UV',
                 particles = [ P.h01, P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.su3, P.su6], [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,3):C.UVGC_1944_4143,(0,0,4):C.UVGC_1944_4144,(0,0,5):C.UVGC_1944_4145,(0,0,0):C.UVGC_1944_4146,(0,0,1):C.UVGC_1944_4147,(0,0,2):C.UVGC_1944_4148,(0,0,6):C.UVGC_1944_4149})

V_701 = CTVertex(name = 'V_701',
                 type = 'UV',
                 particles = [ P.A0, P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.su3, P.su6], [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,3):C.UVGC_2221_4454,(0,0,4):C.UVGC_2221_4455,(0,0,5):C.UVGC_2221_4456,(0,0,0):C.UVGC_2221_4457,(0,0,1):C.UVGC_2221_4458,(0,0,2):C.UVGC_2221_4459,(0,0,6):C.UVGC_2221_4460})

V_702 = CTVertex(name = 'V_702',
                 type = 'UV',
                 particles = [ P.h02, P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.su3, P.su6], [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,3):C.UVGC_1943_4136,(0,0,4):C.UVGC_1943_4137,(0,0,5):C.UVGC_1943_4138,(0,0,0):C.UVGC_1943_4139,(0,0,1):C.UVGC_1943_4140,(0,0,2):C.UVGC_1943_4141,(0,0,6):C.UVGC_1943_4142})

V_703 = CTVertex(name = 'V_703',
                 type = 'UV',
                 particles = [ P.G0, P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.su3, P.su6], [P.g, P.su3, P.su6] ] ],
                 couplings = {(0,0,3):C.UVGC_2220_4447,(0,0,4):C.UVGC_2220_4448,(0,0,5):C.UVGC_2220_4449,(0,0,0):C.UVGC_2220_4450,(0,0,1):C.UVGC_2220_4451,(0,0,2):C.UVGC_2220_4452,(0,0,6):C.UVGC_2220_4453})

V_704 = CTVertex(name = 'V_704',
                 type = 'UV',
                 particles = [ P.x1__minus__, P.u, P.sd1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.go, P.su1] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1457_1616,(0,0,2):C.UVGC_1457_1617,(0,0,4):C.UVGC_1457_1618,(0,0,1):C.UVGC_1457_1619,(0,0,3):C.UVGC_1457_1620})

V_705 = CTVertex(name = 'V_705',
                 type = 'UV',
                 particles = [ P.x1__minus__, P.c, P.sd2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.sd2] ], [ [P.go, P.s] ], [ [P.go, P.su2] ], [ [P.g, P.sd2] ] ],
                 couplings = {(0,0,0):C.UVGC_1503_1870,(0,0,4):C.UVGC_1503_1871,(0,0,2):C.UVGC_1503_1872,(0,0,3):C.UVGC_1503_1873,(0,0,1):C.UVGC_1503_1874})

V_706 = CTVertex(name = 'V_706',
                 type = 'UV',
                 particles = [ P.x1__minus__, P.t, P.sd3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.su6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1636_2859,(0,0,4):C.UVGC_1636_2860,(0,0,6):C.UVGC_1636_2861,(0,0,2):C.UVGC_1636_2862,(0,0,3):C.UVGC_1636_2863,(0,0,1):C.UVGC_1636_2864,(0,0,5):C.UVGC_1636_2865,(0,1,0):C.UVGC_1628_2795,(0,1,4):C.UVGC_1628_2796,(0,1,6):C.UVGC_1628_2797,(0,1,2):C.UVGC_1628_2798,(0,1,3):C.UVGC_1628_2799,(0,1,5):C.UVGC_1628_2800})

V_707 = CTVertex(name = 'V_707',
                 type = 'UV',
                 particles = [ P.x1__minus__, P.t, P.sd6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.su3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.sd6] ], [ [P.g, P.sd6, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1630_2807,(0,1,1):C.UVGC_1630_2808,(0,1,7):C.UVGC_1630_2809,(0,1,9):C.UVGC_1630_2810,(0,1,3):C.UVGC_1630_2811,(0,1,4):C.UVGC_1630_2812,(0,1,5):C.UVGC_1630_2813,(0,1,6):C.UVGC_1630_2814,(0,1,2):C.UVGC_1630_2815,(0,1,8):C.UVGC_1630_2816,(0,0,1):C.UVGC_1406_1361})

V_708 = CTVertex(name = 'V_708',
                 type = 'UV',
                 particles = [ P.x2__minus__, P.u, P.sd1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.go, P.su1] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1458_1621,(0,0,2):C.UVGC_1458_1622,(0,0,4):C.UVGC_1458_1623,(0,0,1):C.UVGC_1458_1624,(0,0,3):C.UVGC_1458_1625})

V_709 = CTVertex(name = 'V_709',
                 type = 'UV',
                 particles = [ P.x2__minus__, P.c, P.sd2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.sd2] ], [ [P.go, P.s] ], [ [P.go, P.su2] ], [ [P.g, P.sd2] ] ],
                 couplings = {(0,0,0):C.UVGC_1504_1875,(0,0,4):C.UVGC_1504_1876,(0,0,2):C.UVGC_1504_1877,(0,0,3):C.UVGC_1504_1878,(0,0,1):C.UVGC_1504_1879})

V_710 = CTVertex(name = 'V_710',
                 type = 'UV',
                 particles = [ P.x2__minus__, P.t, P.sd3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.su6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1637_2866,(0,0,4):C.UVGC_1637_2867,(0,0,6):C.UVGC_1637_2868,(0,0,2):C.UVGC_1637_2869,(0,0,3):C.UVGC_1637_2870,(0,0,1):C.UVGC_1637_2871,(0,0,5):C.UVGC_1637_2872,(0,1,0):C.UVGC_1629_2801,(0,1,4):C.UVGC_1629_2802,(0,1,6):C.UVGC_1629_2803,(0,1,2):C.UVGC_1629_2804,(0,1,3):C.UVGC_1629_2805,(0,1,5):C.UVGC_1629_2806})

V_711 = CTVertex(name = 'V_711',
                 type = 'UV',
                 particles = [ P.x2__minus__, P.t, P.sd6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.su3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.sd6] ], [ [P.g, P.sd6, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1631_2817,(0,1,1):C.UVGC_1631_2818,(0,1,7):C.UVGC_1631_2819,(0,1,9):C.UVGC_1631_2820,(0,1,3):C.UVGC_1631_2821,(0,1,4):C.UVGC_1631_2822,(0,1,5):C.UVGC_1631_2823,(0,1,6):C.UVGC_1631_2824,(0,1,2):C.UVGC_1631_2825,(0,1,8):C.UVGC_1631_2826,(0,0,1):C.UVGC_1407_1362})

V_712 = CTVertex(name = 'V_712',
                 type = 'UV',
                 particles = [ P.d, P.x1__plus__, P.su1__tilde__ ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.g, P.su1] ], [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,2):C.UVGC_1493_1840,(0,0,1):C.UVGC_1493_1841,(0,0,0):C.UVGC_1493_1842})

V_713 = CTVertex(name = 'V_713',
                 type = 'UV',
                 particles = [ P.s, P.x1__plus__, P.su2__tilde__ ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.s, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1540_2096,(0,0,2):C.UVGC_1540_2097,(0,0,1):C.UVGC_1540_2098})

V_714 = CTVertex(name = 'V_714',
                 type = 'UV',
                 particles = [ P.x1__plus__, P.b, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.su3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.sd6, P.t] ], [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_1692_3228,(0,0,6):C.UVGC_1692_3229,(0,0,2):C.UVGC_1692_3230,(0,0,3):C.UVGC_1692_3231,(0,0,5):C.UVGC_1692_3232,(0,0,1):C.UVGC_1692_3233,(0,0,4):C.UVGC_1692_3234,(0,1,0):C.UVGC_1686_3188,(0,1,6):C.UVGC_1686_3189,(0,1,2):C.UVGC_1686_3190,(0,1,3):C.UVGC_1686_3191,(0,1,5):C.UVGC_1686_3192,(0,1,1):C.UVGC_1686_3193})

V_715 = CTVertex(name = 'V_715',
                 type = 'UV',
                 particles = [ P.x1__plus__, P.b, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.su6] ], [ [P.go, P.sd3] ], [ [P.go, P.sd3, P.t] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1753_3467,(0,1,8):C.UVGC_1753_3468,(0,1,9):C.UVGC_1753_3469,(0,1,2):C.UVGC_1753_3470,(0,1,4):C.UVGC_1753_3471,(0,1,5):C.UVGC_1753_3472,(0,1,6):C.UVGC_1753_3473,(0,1,7):C.UVGC_1753_3474,(0,1,1):C.UVGC_1753_3475,(0,1,3):C.UVGC_1753_3476,(0,0,7):C.UVGC_1759_3511})

V_716 = CTVertex(name = 'V_716',
                 type = 'UV',
                 particles = [ P.d, P.x2__plus__, P.su1__tilde__ ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.d, P.g, P.su1] ], [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,2):C.UVGC_1494_1843,(0,0,1):C.UVGC_1494_1844,(0,0,0):C.UVGC_1494_1845})

V_717 = CTVertex(name = 'V_717',
                 type = 'UV',
                 particles = [ P.s, P.x2__plus__, P.su2__tilde__ ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS4 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.s, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1541_2099,(0,0,2):C.UVGC_1541_2100,(0,0,1):C.UVGC_1541_2101})

V_718 = CTVertex(name = 'V_718',
                 type = 'UV',
                 particles = [ P.x2__plus__, P.b, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.su3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.sd6, P.t] ], [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_1693_3235,(0,0,6):C.UVGC_1693_3236,(0,0,2):C.UVGC_1693_3237,(0,0,3):C.UVGC_1693_3238,(0,0,5):C.UVGC_1693_3239,(0,0,1):C.UVGC_1693_3240,(0,0,4):C.UVGC_1693_3241,(0,1,0):C.UVGC_1687_3194,(0,1,6):C.UVGC_1687_3195,(0,1,2):C.UVGC_1687_3196,(0,1,3):C.UVGC_1687_3197,(0,1,5):C.UVGC_1687_3198,(0,1,1):C.UVGC_1687_3199})

V_719 = CTVertex(name = 'V_719',
                 type = 'UV',
                 particles = [ P.x2__plus__, P.b, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.su6] ], [ [P.go, P.sd3] ], [ [P.go, P.sd3, P.t] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1754_3477,(0,1,8):C.UVGC_1754_3478,(0,1,9):C.UVGC_1754_3479,(0,1,2):C.UVGC_1754_3480,(0,1,4):C.UVGC_1754_3481,(0,1,5):C.UVGC_1754_3482,(0,1,6):C.UVGC_1754_3483,(0,1,7):C.UVGC_1754_3484,(0,1,1):C.UVGC_1754_3485,(0,1,3):C.UVGC_1754_3486,(0,0,7):C.UVGC_1760_3512})

V_720 = CTVertex(name = 'V_720',
                 type = 'UV',
                 particles = [ P.W__minus__, P.sd1__tilde__, P.su1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.d, P.go, P.u] ], [ [P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_1468_1666,(0,0,3):C.UVGC_1468_1667,(0,0,5):C.UVGC_1468_1668,(0,0,2):C.UVGC_1468_1669,(0,0,1):C.UVGC_1468_1670,(0,0,4):C.UVGC_1468_1671,(0,1,0):C.UVGC_1467_1660,(0,1,3):C.UVGC_1467_1661,(0,1,5):C.UVGC_1467_1662,(0,1,2):C.UVGC_1467_1663,(0,1,1):C.UVGC_1467_1664,(0,1,4):C.UVGC_1467_1665})

V_721 = CTVertex(name = 'V_721',
                 type = 'UV',
                 particles = [ P.W__minus__, P.sd2__tilde__, P.su2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.c, P.go, P.s] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1514_1920,(0,0,3):C.UVGC_1514_1921,(0,0,5):C.UVGC_1514_1922,(0,0,2):C.UVGC_1514_1923,(0,0,1):C.UVGC_1514_1924,(0,0,4):C.UVGC_1514_1925,(0,1,0):C.UVGC_1513_1914,(0,1,3):C.UVGC_1513_1915,(0,1,5):C.UVGC_1513_1916,(0,1,2):C.UVGC_1513_1917,(0,1,1):C.UVGC_1513_1918,(0,1,4):C.UVGC_1513_1919})

V_722 = CTVertex(name = 'V_722',
                 type = 'UV',
                 particles = [ P.W__minus__, P.sd3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_1650_2954,(0,0,3):C.UVGC_1650_2955,(0,0,5):C.UVGC_1650_2956,(0,0,2):C.UVGC_1650_2957,(0,0,1):C.UVGC_1650_2958,(0,0,4):C.UVGC_1650_2959,(0,1,0):C.UVGC_1649_2948,(0,1,3):C.UVGC_1649_2949,(0,1,5):C.UVGC_1649_2950,(0,1,2):C.UVGC_1649_2951,(0,1,1):C.UVGC_1649_2952,(0,1,4):C.UVGC_1649_2953})

V_723 = CTVertex(name = 'V_723',
                 type = 'UV',
                 particles = [ P.a, P.W__minus__, P.sd1__tilde__, P.su1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.d, P.go, P.u] ], [ [P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ], [ [P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_1469_1672,(0,0,3):C.UVGC_1469_1673,(0,0,5):C.UVGC_1469_1674,(0,0,2):C.UVGC_1469_1675,(0,0,6):C.UVGC_1469_1676,(0,0,1):C.UVGC_1469_1677,(0,0,4):C.UVGC_1469_1678})

V_724 = CTVertex(name = 'V_724',
                 type = 'UV',
                 particles = [ P.a, P.W__minus__, P.sd2__tilde__, P.su2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.c, P.go, P.s] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ], [ [P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1515_1926,(0,0,3):C.UVGC_1515_1927,(0,0,5):C.UVGC_1515_1928,(0,0,2):C.UVGC_1515_1929,(0,0,6):C.UVGC_1515_1930,(0,0,1):C.UVGC_1515_1931,(0,0,4):C.UVGC_1515_1932})

V_725 = CTVertex(name = 'V_725',
                 type = 'UV',
                 particles = [ P.a, P.W__minus__, P.sd3__tilde__, P.su3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ], [ [P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_1651_2960,(0,0,3):C.UVGC_1651_2961,(0,0,5):C.UVGC_1651_2962,(0,0,2):C.UVGC_1651_2963,(0,0,6):C.UVGC_1651_2964,(0,0,1):C.UVGC_1651_2965,(0,0,4):C.UVGC_1651_2966})

V_726 = CTVertex(name = 'V_726',
                 type = 'UV',
                 particles = [ P.g, P.W__minus__, P.sd1__tilde__, P.su1 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.d, P.go] ], [ [P.d, P.go, P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ], [ [P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_1470_1679,(0,0,1):C.UVGC_1470_1680,(0,0,4):C.UVGC_1470_1681,(0,0,5):C.UVGC_1470_1682,(0,0,2):C.UVGC_1470_1683,(0,0,7):C.UVGC_1470_1684,(0,0,9):C.UVGC_1470_1685,(0,0,6):C.UVGC_1470_1686,(0,0,10):C.UVGC_1470_1687,(0,0,3):C.UVGC_1470_1688,(0,0,8):C.UVGC_1470_1689})

V_727 = CTVertex(name = 'V_727',
                 type = 'UV',
                 particles = [ P.g, P.W__minus__, P.sd2__tilde__, P.su2 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.go] ], [ [P.c, P.go, P.s] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ], [ [P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1516_1933,(0,0,1):C.UVGC_1516_1934,(0,0,4):C.UVGC_1516_1935,(0,0,5):C.UVGC_1516_1936,(0,0,2):C.UVGC_1516_1937,(0,0,7):C.UVGC_1516_1938,(0,0,9):C.UVGC_1516_1939,(0,0,6):C.UVGC_1516_1940,(0,0,10):C.UVGC_1516_1941,(0,0,3):C.UVGC_1516_1942,(0,0,8):C.UVGC_1516_1943})

V_728 = CTVertex(name = 'V_728',
                 type = 'UV',
                 particles = [ P.g, P.W__minus__, P.sd3__tilde__, P.su3 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ], [ [P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_1652_2967,(0,0,3):C.UVGC_1652_2968,(0,0,4):C.UVGC_1652_2969,(0,0,5):C.UVGC_1652_2970,(0,0,1):C.UVGC_1652_2971,(0,0,7):C.UVGC_1652_2972,(0,0,9):C.UVGC_1652_2973,(0,0,6):C.UVGC_1652_2974,(0,0,10):C.UVGC_1652_2975,(0,0,2):C.UVGC_1652_2976,(0,0,8):C.UVGC_1652_2977})

V_729 = CTVertex(name = 'V_729',
                 type = 'UV',
                 particles = [ P.W__plus__, P.sd1, P.su1__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.d, P.go, P.u] ], [ [P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_1488_1803,(0,0,3):C.UVGC_1488_1804,(0,0,5):C.UVGC_1488_1805,(0,0,2):C.UVGC_1488_1806,(0,0,1):C.UVGC_1488_1807,(0,0,4):C.UVGC_1488_1808,(0,1,0):C.UVGC_1489_1809,(0,1,3):C.UVGC_1489_1810,(0,1,5):C.UVGC_1489_1811,(0,1,2):C.UVGC_1489_1812,(0,1,1):C.UVGC_1489_1813,(0,1,4):C.UVGC_1489_1814})

V_730 = CTVertex(name = 'V_730',
                 type = 'UV',
                 particles = [ P.W__plus__, P.sd2, P.su2__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.c, P.go, P.s] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1535_2059,(0,0,3):C.UVGC_1535_2060,(0,0,5):C.UVGC_1535_2061,(0,0,2):C.UVGC_1535_2062,(0,0,1):C.UVGC_1535_2063,(0,0,4):C.UVGC_1535_2064,(0,1,0):C.UVGC_1536_2065,(0,1,3):C.UVGC_1536_2066,(0,1,5):C.UVGC_1536_2067,(0,1,2):C.UVGC_1536_2068,(0,1,1):C.UVGC_1536_2069,(0,1,4):C.UVGC_1536_2070})

V_731 = CTVertex(name = 'V_731',
                 type = 'UV',
                 particles = [ P.W__plus__, P.sd3, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_1667_3073,(0,0,3):C.UVGC_1667_3074,(0,0,5):C.UVGC_1667_3075,(0,0,2):C.UVGC_1667_3076,(0,0,1):C.UVGC_1667_3077,(0,0,4):C.UVGC_1667_3078,(0,1,0):C.UVGC_1668_3079,(0,1,3):C.UVGC_1668_3080,(0,1,5):C.UVGC_1668_3081,(0,1,2):C.UVGC_1668_3082,(0,1,1):C.UVGC_1668_3083,(0,1,4):C.UVGC_1668_3084})

V_732 = CTVertex(name = 'V_732',
                 type = 'UV',
                 particles = [ P.a, P.W__plus__, P.sd1, P.su1__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.d, P.go, P.u] ], [ [P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ], [ [P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_1490_1815,(0,0,3):C.UVGC_1490_1816,(0,0,5):C.UVGC_1490_1817,(0,0,2):C.UVGC_1490_1818,(0,0,6):C.UVGC_1490_1819,(0,0,1):C.UVGC_1490_1820,(0,0,4):C.UVGC_1490_1821})

V_733 = CTVertex(name = 'V_733',
                 type = 'UV',
                 particles = [ P.a, P.W__plus__, P.sd2, P.su2__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.c, P.go, P.s] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ], [ [P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1537_2071,(0,0,3):C.UVGC_1537_2072,(0,0,5):C.UVGC_1537_2073,(0,0,2):C.UVGC_1537_2074,(0,0,6):C.UVGC_1537_2075,(0,0,1):C.UVGC_1537_2076,(0,0,4):C.UVGC_1537_2077})

V_734 = CTVertex(name = 'V_734',
                 type = 'UV',
                 particles = [ P.a, P.W__plus__, P.sd3, P.su3__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ], [ [P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_1669_3085,(0,0,3):C.UVGC_1669_3086,(0,0,5):C.UVGC_1669_3087,(0,0,2):C.UVGC_1669_3088,(0,0,6):C.UVGC_1669_3089,(0,0,1):C.UVGC_1669_3090,(0,0,4):C.UVGC_1669_3091})

V_735 = CTVertex(name = 'V_735',
                 type = 'UV',
                 particles = [ P.g, P.W__plus__, P.sd1, P.su1__tilde__ ],
                 color = [ 'T(1,3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.d, P.go] ], [ [P.d, P.go, P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ], [ [P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_1491_1822,(0,0,1):C.UVGC_1491_1823,(0,0,4):C.UVGC_1491_1824,(0,0,5):C.UVGC_1491_1825,(0,0,2):C.UVGC_1491_1826,(0,0,7):C.UVGC_1491_1827,(0,0,9):C.UVGC_1491_1828,(0,0,6):C.UVGC_1491_1829,(0,0,10):C.UVGC_1491_1830,(0,0,3):C.UVGC_1491_1831,(0,0,8):C.UVGC_1491_1832})

V_736 = CTVertex(name = 'V_736',
                 type = 'UV',
                 particles = [ P.g, P.W__plus__, P.sd2, P.su2__tilde__ ],
                 color = [ 'T(1,3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.go] ], [ [P.c, P.go, P.s] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ], [ [P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1538_2078,(0,0,1):C.UVGC_1538_2079,(0,0,4):C.UVGC_1538_2080,(0,0,5):C.UVGC_1538_2081,(0,0,2):C.UVGC_1538_2082,(0,0,7):C.UVGC_1538_2083,(0,0,9):C.UVGC_1538_2084,(0,0,6):C.UVGC_1538_2085,(0,0,10):C.UVGC_1538_2086,(0,0,3):C.UVGC_1538_2087,(0,0,8):C.UVGC_1538_2088})

V_737 = CTVertex(name = 'V_737',
                 type = 'UV',
                 particles = [ P.g, P.W__plus__, P.sd3, P.su3__tilde__ ],
                 color = [ 'T(1,3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ], [ [P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_1670_3092,(0,0,3):C.UVGC_1670_3093,(0,0,4):C.UVGC_1670_3094,(0,0,5):C.UVGC_1670_3095,(0,0,1):C.UVGC_1670_3096,(0,0,7):C.UVGC_1670_3097,(0,0,9):C.UVGC_1670_3098,(0,0,6):C.UVGC_1670_3099,(0,0,10):C.UVGC_1670_3100,(0,0,2):C.UVGC_1670_3101,(0,0,8):C.UVGC_1670_3102})

V_738 = CTVertex(name = 'V_738',
                 type = 'UV',
                 particles = [ P.W__minus__, P.W__plus__, P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.d, P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ], [ [P.sd1] ], [ [P.sd1, P.su1] ] ],
                 couplings = {(0,0,5):C.UVGC_1261_708,(0,0,0):C.UVGC_1261_709,(0,0,2):C.UVGC_1261_710,(0,0,4):C.UVGC_1261_711,(0,0,6):C.UVGC_1261_712,(0,0,1):C.UVGC_1261_713,(0,0,3):C.UVGC_1261_714})

V_739 = CTVertex(name = 'V_739',
                 type = 'UV',
                 particles = [ P.W__minus__, P.W__plus__, P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go, P.s] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ], [ [P.sd2] ], [ [P.sd2, P.su2] ] ],
                 couplings = {(0,0,5):C.UVGC_1287_843,(0,0,2):C.UVGC_1287_844,(0,0,4):C.UVGC_1287_845,(0,0,1):C.UVGC_1287_846,(0,0,6):C.UVGC_1287_847,(0,0,0):C.UVGC_1287_848,(0,0,3):C.UVGC_1287_849})

V_740 = CTVertex(name = 'V_740',
                 type = 'UV',
                 particles = [ P.W__minus__, P.W__plus__, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ], [ [P.sd3] ], [ [P.sd3, P.su3] ] ],
                 couplings = {(0,0,5):C.UVGC_1305_954,(0,0,0):C.UVGC_1305_955,(0,0,2):C.UVGC_1305_956,(0,0,4):C.UVGC_1305_957,(0,0,6):C.UVGC_1305_958,(0,0,1):C.UVGC_1305_959,(0,0,3):C.UVGC_1305_960})

V_741 = CTVertex(name = 'V_741',
                 type = 'UV',
                 particles = [ P.W__minus__, P.W__plus__, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go, P.u] ], [ [P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ], [ [P.sd1, P.su1] ], [ [P.su1] ] ],
                 couplings = {(0,0,6):C.UVGC_1478_1749,(0,0,2):C.UVGC_1261_711,(0,0,4):C.UVGC_1478_1750,(0,0,1):C.UVGC_1478_1751,(0,0,5):C.UVGC_1478_1752,(0,0,0):C.UVGC_1478_1753,(0,0,3):C.UVGC_1478_1754})

V_742 = CTVertex(name = 'V_742',
                 type = 'UV',
                 particles = [ P.W__minus__, P.W__plus__, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.c, P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ], [ [P.sd2, P.su2] ], [ [P.su2] ] ],
                 couplings = {(0,0,6):C.UVGC_1525_2005,(0,0,0):C.UVGC_1525_2006,(0,0,2):C.UVGC_1287_845,(0,0,4):C.UVGC_1525_2007,(0,0,5):C.UVGC_1525_2008,(0,0,1):C.UVGC_1525_2009,(0,0,3):C.UVGC_1525_2010})

V_743 = CTVertex(name = 'V_743',
                 type = 'UV',
                 particles = [ P.W__minus__, P.W__plus__, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go, P.t] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ], [ [P.sd3, P.su3] ], [ [P.su3] ] ],
                 couplings = {(0,0,6):C.UVGC_1661_3039,(0,0,2):C.UVGC_1305_957,(0,0,4):C.UVGC_1661_3040,(0,0,1):C.UVGC_1661_3041,(0,0,5):C.UVGC_1661_3042,(0,0,0):C.UVGC_1661_3043,(0,0,3):C.UVGC_1661_3044})

V_744 = CTVertex(name = 'V_744',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.x1__plus__, P.sd6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.su3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.sd6] ], [ [P.g, P.sd6, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1633_2833,(0,0,1):C.UVGC_1633_2834,(0,0,7):C.UVGC_1633_2835,(0,0,9):C.UVGC_1633_2836,(0,0,3):C.UVGC_1633_2837,(0,0,4):C.UVGC_1633_2838,(0,0,5):C.UVGC_1633_2839,(0,0,6):C.UVGC_1633_2840,(0,0,2):C.UVGC_1633_2841,(0,0,8):C.UVGC_1633_2842,(0,1,1):C.UVGC_1369_1255})

V_745 = CTVertex(name = 'V_745',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.x2__plus__, P.sd6 ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.su3] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.sd6] ], [ [P.g, P.sd6, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1635_2849,(0,0,1):C.UVGC_1635_2850,(0,0,7):C.UVGC_1635_2851,(0,0,9):C.UVGC_1635_2852,(0,0,3):C.UVGC_1635_2853,(0,0,4):C.UVGC_1635_2854,(0,0,5):C.UVGC_1635_2855,(0,0,6):C.UVGC_1635_2856,(0,0,2):C.UVGC_1635_2857,(0,0,8):C.UVGC_1635_2858,(0,1,1):C.UVGC_1370_1256})

V_746 = CTVertex(name = 'V_746',
                 type = 'UV',
                 particles = [ P.H__plus__, P.sd6, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.sd6] ], [ [P.g, P.sd6, P.su6] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.sd6, P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2224_4469,(0,0,1):C.UVGC_2224_4470,(0,0,8):C.UVGC_2224_4471,(0,0,10):C.UVGC_2224_4472,(0,0,11):C.UVGC_2224_4473,(0,0,3):C.UVGC_2224_4474,(0,0,4):C.UVGC_2224_4475,(0,0,5):C.UVGC_2224_4476,(0,0,6):C.UVGC_2224_4477,(0,0,7):C.UVGC_2224_4478,(0,0,12):C.UVGC_2224_4479,(0,0,2):C.UVGC_2224_4480,(0,0,9):C.UVGC_2224_4481})

V_747 = CTVertex(name = 'V_747',
                 type = 'UV',
                 particles = [ P.G__plus__, P.sd6, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.sd6] ], [ [P.g, P.sd6, P.su6] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.sd6, P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2226_4491,(0,0,1):C.UVGC_2226_4492,(0,0,8):C.UVGC_2226_4493,(0,0,10):C.UVGC_2226_4494,(0,0,11):C.UVGC_2226_4495,(0,0,3):C.UVGC_2226_4496,(0,0,4):C.UVGC_2226_4497,(0,0,5):C.UVGC_2226_4498,(0,0,6):C.UVGC_2226_4499,(0,0,7):C.UVGC_2226_4500,(0,0,12):C.UVGC_2226_4501,(0,0,2):C.UVGC_2226_4502,(0,0,9):C.UVGC_2226_4503})

V_748 = CTVertex(name = 'V_748',
                 type = 'UV',
                 particles = [ P.x1__minus__, P.b__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g, P.su6] ], [ [P.go, P.sd3, P.t] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,5):C.UVGC_1761_3513,(0,0,6):C.UVGC_1761_3514,(0,0,2):C.UVGC_1761_3515,(0,0,3):C.UVGC_1761_3516,(0,0,4):C.UVGC_1761_3517,(0,0,0):C.UVGC_1761_3518,(0,0,1):C.UVGC_1761_3519,(0,1,4):C.UVGC_1703_3279})

V_749 = CTVertex(name = 'V_749',
                 type = 'UV',
                 particles = [ P.x2__minus__, P.b__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.FFS3, L.FFS4 ],
                 loop_particles = [ [ [P.b, P.g, P.su6] ], [ [P.go, P.sd3, P.t] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,5):C.UVGC_1762_3520,(0,0,6):C.UVGC_1762_3521,(0,0,2):C.UVGC_1762_3522,(0,0,3):C.UVGC_1762_3523,(0,0,4):C.UVGC_1762_3524,(0,0,0):C.UVGC_1762_3525,(0,0,1):C.UVGC_1762_3526,(0,1,4):C.UVGC_1704_3280})

V_750 = CTVertex(name = 'V_750',
                 type = 'UV',
                 particles = [ P.H__minus__, P.sd6__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.sd6] ], [ [P.g, P.sd6, P.su6] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.sd6, P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2227_4504,(0,0,1):C.UVGC_2227_4505,(0,0,8):C.UVGC_2227_4506,(0,0,10):C.UVGC_2227_4507,(0,0,11):C.UVGC_2227_4508,(0,0,3):C.UVGC_2227_4509,(0,0,4):C.UVGC_2227_4510,(0,0,5):C.UVGC_2227_4511,(0,0,6):C.UVGC_2227_4512,(0,0,7):C.UVGC_2227_4513,(0,0,12):C.UVGC_2227_4514,(0,0,2):C.UVGC_2227_4515,(0,0,9):C.UVGC_2227_4516})

V_751 = CTVertex(name = 'V_751',
                 type = 'UV',
                 particles = [ P.G__minus__, P.sd6__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.go, P.t] ], [ [P.g, P.sd6] ], [ [P.g, P.sd6, P.su6] ], [ [P.g, P.su6] ], [ [P.g, P.t] ], [ [P.sd6, P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2219_4434,(0,0,1):C.UVGC_2219_4435,(0,0,8):C.UVGC_2219_4436,(0,0,10):C.UVGC_2219_4437,(0,0,11):C.UVGC_2219_4438,(0,0,3):C.UVGC_2219_4439,(0,0,4):C.UVGC_2219_4440,(0,0,5):C.UVGC_2219_4441,(0,0,6):C.UVGC_2219_4442,(0,0,7):C.UVGC_2219_4443,(0,0,12):C.UVGC_2219_4444,(0,0,2):C.UVGC_2219_4445,(0,0,9):C.UVGC_2219_4446})

V_752 = CTVertex(name = 'V_752',
                 type = 'UV',
                 particles = [ P.Z, P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ] ],
                 couplings = {(0,0,0):C.UVGC_1266_735,(0,0,1):C.UVGC_1266_736,(0,1,0):C.UVGC_1267_737,(0,1,1):C.UVGC_1267_738})

V_753 = CTVertex(name = 'V_753',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ], [ [P.sd1] ] ],
                 couplings = {(0,0,2):C.UVGC_1268_739,(0,0,0):C.UVGC_1268_740,(0,0,1):C.UVGC_1268_741})

V_754 = CTVertex(name = 'V_754',
                 type = 'UV',
                 particles = [ P.Z, P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ] ],
                 couplings = {(0,0,1):C.UVGC_1292_870,(0,0,0):C.UVGC_1292_871,(0,1,1):C.UVGC_1293_872,(0,1,0):C.UVGC_1293_873})

V_755 = CTVertex(name = 'V_755',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.sd2] ] ],
                 couplings = {(0,0,2):C.UVGC_1294_874,(0,0,1):C.UVGC_1294_875,(0,0,0):C.UVGC_1294_876})

V_756 = CTVertex(name = 'V_756',
                 type = 'UV',
                 particles = [ P.Z, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ] ],
                 couplings = {(0,0,0):C.UVGC_1309_984,(0,0,1):C.UVGC_1309_985,(0,1,0):C.UVGC_1308_982,(0,1,1):C.UVGC_1308_983})

V_757 = CTVertex(name = 'V_757',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ], [ [P.sd3] ] ],
                 couplings = {(0,0,2):C.UVGC_1310_986,(0,0,0):C.UVGC_1310_987,(0,0,1):C.UVGC_1310_988})

V_758 = CTVertex(name = 'V_758',
                 type = 'UV',
                 particles = [ P.Z, P.sd4__tilde__, P.sd4 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ] ],
                 couplings = {(0,0,0):C.UVGC_1330_1074,(0,0,1):C.UVGC_1330_1075,(0,1,0):C.UVGC_1331_1076,(0,1,1):C.UVGC_1331_1077})

V_759 = CTVertex(name = 'V_759',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.sd4__tilde__, P.sd4 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ], [ [P.sd4] ] ],
                 couplings = {(0,0,2):C.UVGC_1332_1078,(0,0,0):C.UVGC_1332_1079,(0,0,1):C.UVGC_1332_1080})

V_760 = CTVertex(name = 'V_760',
                 type = 'UV',
                 particles = [ P.Z, P.sd5__tilde__, P.sd5 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ] ],
                 couplings = {(0,0,1):C.UVGC_1355_1202,(0,0,0):C.UVGC_1355_1203,(0,1,1):C.UVGC_1356_1204,(0,1,0):C.UVGC_1356_1205})

V_761 = CTVertex(name = 'V_761',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.sd5__tilde__, P.sd5 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ], [ [P.sd5] ] ],
                 couplings = {(0,0,2):C.UVGC_1357_1206,(0,0,1):C.UVGC_1357_1207,(0,0,0):C.UVGC_1357_1208})

V_762 = CTVertex(name = 'V_762',
                 type = 'UV',
                 particles = [ P.Z, P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1397_1331,(0,0,1):C.UVGC_1397_1332,(0,1,0):C.UVGC_1398_1333,(0,1,1):C.UVGC_1398_1334})

V_763 = CTVertex(name = 'V_763',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ], [ [P.sd6] ] ],
                 couplings = {(0,0,2):C.UVGC_1399_1335,(0,0,0):C.UVGC_1399_1336,(0,0,1):C.UVGC_1399_1337})

V_764 = CTVertex(name = 'V_764',
                 type = 'UV',
                 particles = [ P.Z, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ] ],
                 couplings = {(0,0,1):C.UVGC_1484_1777,(0,0,0):C.UVGC_1484_1778,(0,1,1):C.UVGC_1483_1775,(0,1,0):C.UVGC_1483_1776})

V_765 = CTVertex(name = 'V_765',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.su1] ] ],
                 couplings = {(0,0,2):C.UVGC_1485_1779,(0,0,1):C.UVGC_1485_1780,(0,0,0):C.UVGC_1485_1781})

V_766 = CTVertex(name = 'V_766',
                 type = 'UV',
                 particles = [ P.Z, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1531_2033,(0,0,1):C.UVGC_1531_2034,(0,1,0):C.UVGC_1530_2031,(0,1,1):C.UVGC_1530_2032})

V_767 = CTVertex(name = 'V_767',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ], [ [P.su2] ] ],
                 couplings = {(0,0,2):C.UVGC_1532_2035,(0,0,0):C.UVGC_1532_2036,(0,0,1):C.UVGC_1532_2037})

V_768 = CTVertex(name = 'V_768',
                 type = 'UV',
                 particles = [ P.Z, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ] ],
                 couplings = {(0,0,1):C.UVGC_1666_3071,(0,0,0):C.UVGC_1666_3072,(0,1,1):C.UVGC_1665_3069,(0,1,0):C.UVGC_1665_3070})

V_769 = CTVertex(name = 'V_769',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.su3] ] ],
                 couplings = {(0,0,2):C.UVGC_1662_3045,(0,0,1):C.UVGC_1662_3046,(0,0,0):C.UVGC_1662_3047})

V_770 = CTVertex(name = 'V_770',
                 type = 'UV',
                 particles = [ P.Z, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ] ],
                 couplings = {(0,0,1):C.UVGC_1563_2199,(0,0,0):C.UVGC_1563_2200,(0,1,1):C.UVGC_1562_2197,(0,1,0):C.UVGC_1562_2198})

V_771 = CTVertex(name = 'V_771',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.su4] ] ],
                 couplings = {(0,0,2):C.UVGC_1564_2201,(0,0,1):C.UVGC_1564_2202,(0,0,0):C.UVGC_1564_2203})

V_772 = CTVertex(name = 'V_772',
                 type = 'UV',
                 particles = [ P.Z, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ] ],
                 couplings = {(0,0,0):C.UVGC_1589_2329,(0,0,1):C.UVGC_1589_2330,(0,1,0):C.UVGC_1588_2327,(0,1,1):C.UVGC_1588_2328})

V_773 = CTVertex(name = 'V_773',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ], [ [P.su5] ] ],
                 couplings = {(0,0,2):C.UVGC_1590_2331,(0,0,0):C.UVGC_1590_2332,(0,0,1):C.UVGC_1590_2333})

V_774 = CTVertex(name = 'V_774',
                 type = 'UV',
                 particles = [ P.Z, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ] ],
                 couplings = {(0,0,1):C.UVGC_1746_3416,(0,0,0):C.UVGC_1746_3417,(0,1,1):C.UVGC_1745_3414,(0,1,0):C.UVGC_1745_3415})

V_775 = CTVertex(name = 'V_775',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.su6] ] ],
                 couplings = {(0,0,2):C.UVGC_1747_3418,(0,0,1):C.UVGC_1747_3419,(0,0,0):C.UVGC_1747_3420})

V_776 = CTVertex(name = 'V_776',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.sd1__tilde__, P.sd1 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.d, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd1] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1269_742,(0,0,1):C.UVGC_1269_743,(0,0,3):C.UVGC_1269_744,(0,0,4):C.UVGC_1269_745,(0,0,6):C.UVGC_1269_746,(0,0,7):C.UVGC_1269_747,(0,0,8):C.UVGC_1269_748,(0,0,9):C.UVGC_1269_749,(0,0,10):C.UVGC_1269_750,(0,0,11):C.UVGC_1269_751,(0,0,12):C.UVGC_1269_752,(0,0,13):C.UVGC_1269_753,(0,0,14):C.UVGC_1269_754,(0,0,15):C.UVGC_1269_755,(0,0,16):C.UVGC_1269_756,(0,0,17):C.UVGC_1269_757,(0,0,2):C.UVGC_1269_758,(0,0,5):C.UVGC_1269_759})

V_777 = CTVertex(name = 'V_777',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.sd2__tilde__, P.sd2 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1295_877,(0,0,1):C.UVGC_1295_878,(0,0,2):C.UVGC_1295_879,(0,0,3):C.UVGC_1295_880,(0,0,6):C.UVGC_1295_881,(0,0,7):C.UVGC_1295_882,(0,0,8):C.UVGC_1295_883,(0,0,9):C.UVGC_1295_884,(0,0,10):C.UVGC_1295_885,(0,0,11):C.UVGC_1295_886,(0,0,12):C.UVGC_1295_887,(0,0,13):C.UVGC_1295_888,(0,0,14):C.UVGC_1295_889,(0,0,15):C.UVGC_1295_890,(0,0,16):C.UVGC_1295_891,(0,0,17):C.UVGC_1295_892,(0,0,5):C.UVGC_1295_893,(0,0,4):C.UVGC_1295_894})

V_778 = CTVertex(name = 'V_778',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.sd3__tilde__, P.sd3 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.go] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd3] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1306_961,(0,0,2):C.UVGC_1306_962,(0,0,3):C.UVGC_1306_963,(0,0,4):C.UVGC_1306_964,(0,0,6):C.UVGC_1306_965,(0,0,7):C.UVGC_1306_966,(0,0,8):C.UVGC_1306_967,(0,0,9):C.UVGC_1306_968,(0,0,10):C.UVGC_1306_969,(0,0,11):C.UVGC_1306_970,(0,0,12):C.UVGC_1306_971,(0,0,13):C.UVGC_1306_972,(0,0,14):C.UVGC_1306_973,(0,0,15):C.UVGC_1306_974,(0,0,16):C.UVGC_1306_975,(0,0,17):C.UVGC_1306_976,(0,0,1):C.UVGC_1306_977,(0,0,5):C.UVGC_1306_978})

V_779 = CTVertex(name = 'V_779',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.sd4__tilde__, P.sd4 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.d, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd4] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1333_1081,(0,0,1):C.UVGC_1333_1082,(0,0,3):C.UVGC_1333_1083,(0,0,4):C.UVGC_1333_1084,(0,0,6):C.UVGC_1333_1085,(0,0,7):C.UVGC_1333_1086,(0,0,8):C.UVGC_1333_1087,(0,0,9):C.UVGC_1333_1088,(0,0,10):C.UVGC_1333_1089,(0,0,11):C.UVGC_1333_1090,(0,0,12):C.UVGC_1333_1091,(0,0,13):C.UVGC_1333_1092,(0,0,14):C.UVGC_1333_1093,(0,0,15):C.UVGC_1333_1094,(0,0,16):C.UVGC_1333_1095,(0,0,17):C.UVGC_1333_1096,(0,0,2):C.UVGC_1333_1097,(0,0,5):C.UVGC_1333_1098})

V_780 = CTVertex(name = 'V_780',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.sd5__tilde__, P.sd5 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.s] ], [ [P.g, P.sd5] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1358_1209,(0,0,1):C.UVGC_1358_1210,(0,0,2):C.UVGC_1358_1211,(0,0,3):C.UVGC_1358_1212,(0,0,6):C.UVGC_1358_1213,(0,0,7):C.UVGC_1358_1214,(0,0,8):C.UVGC_1358_1215,(0,0,9):C.UVGC_1358_1216,(0,0,10):C.UVGC_1358_1217,(0,0,11):C.UVGC_1358_1218,(0,0,12):C.UVGC_1358_1219,(0,0,13):C.UVGC_1358_1220,(0,0,14):C.UVGC_1358_1221,(0,0,15):C.UVGC_1358_1222,(0,0,16):C.UVGC_1358_1223,(0,0,17):C.UVGC_1358_1224,(0,0,5):C.UVGC_1358_1225,(0,0,4):C.UVGC_1358_1226})

V_781 = CTVertex(name = 'V_781',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.sd6__tilde__, P.sd6 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.go] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.sd6] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1400_1338,(0,0,2):C.UVGC_1400_1339,(0,0,3):C.UVGC_1400_1340,(0,0,4):C.UVGC_1400_1341,(0,0,6):C.UVGC_1400_1342,(0,0,7):C.UVGC_1400_1343,(0,0,8):C.UVGC_1400_1344,(0,0,9):C.UVGC_1400_1345,(0,0,10):C.UVGC_1400_1346,(0,0,11):C.UVGC_1400_1347,(0,0,12):C.UVGC_1400_1348,(0,0,13):C.UVGC_1400_1349,(0,0,14):C.UVGC_1400_1350,(0,0,15):C.UVGC_1400_1351,(0,0,16):C.UVGC_1400_1352,(0,0,17):C.UVGC_1400_1353,(0,0,1):C.UVGC_1400_1354,(0,0,5):C.UVGC_1400_1355})

V_782 = CTVertex(name = 'V_782',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.su1__tilde__, P.su1 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1486_1782,(0,0,1):C.UVGC_1486_1783,(0,0,2):C.UVGC_1486_1784,(0,0,3):C.UVGC_1486_1785,(0,0,6):C.UVGC_1486_1786,(0,0,7):C.UVGC_1486_1787,(0,0,8):C.UVGC_1486_1788,(0,0,9):C.UVGC_1486_1789,(0,0,10):C.UVGC_1486_1790,(0,0,11):C.UVGC_1486_1791,(0,0,12):C.UVGC_1486_1792,(0,0,13):C.UVGC_1486_1793,(0,0,14):C.UVGC_1486_1794,(0,0,15):C.UVGC_1486_1795,(0,0,16):C.UVGC_1486_1796,(0,0,17):C.UVGC_1486_1797,(0,0,5):C.UVGC_1486_1798,(0,0,4):C.UVGC_1486_1799})

V_783 = CTVertex(name = 'V_783',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.su2__tilde__, P.su2 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.su2] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1533_2038,(0,0,1):C.UVGC_1533_2039,(0,0,3):C.UVGC_1533_2040,(0,0,4):C.UVGC_1533_2041,(0,0,6):C.UVGC_1533_2042,(0,0,7):C.UVGC_1533_2043,(0,0,8):C.UVGC_1533_2044,(0,0,9):C.UVGC_1533_2045,(0,0,10):C.UVGC_1533_2046,(0,0,11):C.UVGC_1533_2047,(0,0,12):C.UVGC_1533_2048,(0,0,13):C.UVGC_1533_2049,(0,0,14):C.UVGC_1533_2050,(0,0,15):C.UVGC_1533_2051,(0,0,16):C.UVGC_1533_2052,(0,0,17):C.UVGC_1533_2053,(0,0,2):C.UVGC_1533_2054,(0,0,5):C.UVGC_1533_2055})

V_784 = CTVertex(name = 'V_784',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.su3__tilde__, P.su3 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1663_3048,(0,0,1):C.UVGC_1663_3049,(0,0,2):C.UVGC_1663_3050,(0,0,3):C.UVGC_1663_3051,(0,0,6):C.UVGC_1663_3052,(0,0,7):C.UVGC_1663_3053,(0,0,8):C.UVGC_1663_3054,(0,0,9):C.UVGC_1663_3055,(0,0,10):C.UVGC_1663_3056,(0,0,11):C.UVGC_1663_3057,(0,0,12):C.UVGC_1663_3058,(0,0,13):C.UVGC_1663_3059,(0,0,14):C.UVGC_1663_3060,(0,0,15):C.UVGC_1663_3061,(0,0,16):C.UVGC_1663_3062,(0,0,17):C.UVGC_1663_3063,(0,0,5):C.UVGC_1663_3064,(0,0,4):C.UVGC_1663_3065})

V_785 = CTVertex(name = 'V_785',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.su4__tilde__, P.su4 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1565_2204,(0,0,1):C.UVGC_1565_2205,(0,0,2):C.UVGC_1565_2206,(0,0,3):C.UVGC_1565_2207,(0,0,6):C.UVGC_1565_2208,(0,0,7):C.UVGC_1565_2209,(0,0,8):C.UVGC_1565_2210,(0,0,9):C.UVGC_1565_2211,(0,0,10):C.UVGC_1565_2212,(0,0,11):C.UVGC_1565_2213,(0,0,12):C.UVGC_1565_2214,(0,0,13):C.UVGC_1565_2215,(0,0,14):C.UVGC_1565_2216,(0,0,15):C.UVGC_1565_2217,(0,0,16):C.UVGC_1565_2218,(0,0,17):C.UVGC_1565_2219,(0,0,5):C.UVGC_1565_2220,(0,0,4):C.UVGC_1565_2221})

V_786 = CTVertex(name = 'V_786',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.su5__tilde__, P.su5 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.go] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.su5] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1591_2334,(0,0,1):C.UVGC_1591_2335,(0,0,3):C.UVGC_1591_2336,(0,0,4):C.UVGC_1591_2337,(0,0,6):C.UVGC_1591_2338,(0,0,7):C.UVGC_1591_2339,(0,0,8):C.UVGC_1591_2340,(0,0,9):C.UVGC_1591_2341,(0,0,10):C.UVGC_1591_2342,(0,0,11):C.UVGC_1591_2343,(0,0,12):C.UVGC_1591_2344,(0,0,13):C.UVGC_1591_2345,(0,0,14):C.UVGC_1591_2346,(0,0,15):C.UVGC_1591_2347,(0,0,16):C.UVGC_1591_2348,(0,0,17):C.UVGC_1591_2349,(0,0,2):C.UVGC_1591_2350,(0,0,5):C.UVGC_1591_2351})

V_787 = CTVertex(name = 'V_787',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.su6__tilde__, P.su6 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1748_3421,(0,0,1):C.UVGC_1748_3422,(0,0,2):C.UVGC_1748_3423,(0,0,3):C.UVGC_1748_3424,(0,0,6):C.UVGC_1748_3425,(0,0,7):C.UVGC_1748_3426,(0,0,8):C.UVGC_1748_3427,(0,0,9):C.UVGC_1748_3428,(0,0,10):C.UVGC_1748_3429,(0,0,11):C.UVGC_1748_3430,(0,0,12):C.UVGC_1748_3431,(0,0,13):C.UVGC_1748_3432,(0,0,14):C.UVGC_1748_3433,(0,0,15):C.UVGC_1748_3434,(0,0,16):C.UVGC_1748_3435,(0,0,17):C.UVGC_1748_3436,(0,0,5):C.UVGC_1748_3437,(0,0,4):C.UVGC_1748_3438})

V_788 = CTVertex(name = 'V_788',
                 type = 'UV',
                 particles = [ P.W__minus__, P.Z, P.sd1__tilde__, P.su1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.d, P.go, P.u] ], [ [P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ], [ [P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_1471_1690,(0,0,3):C.UVGC_1471_1691,(0,0,5):C.UVGC_1471_1692,(0,0,2):C.UVGC_1471_1693,(0,0,6):C.UVGC_1471_1694,(0,0,1):C.UVGC_1471_1695,(0,0,4):C.UVGC_1471_1696})

V_789 = CTVertex(name = 'V_789',
                 type = 'UV',
                 particles = [ P.W__minus__, P.Z, P.sd2__tilde__, P.su2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.c, P.go, P.s] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ], [ [P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1517_1944,(0,0,3):C.UVGC_1517_1945,(0,0,5):C.UVGC_1517_1946,(0,0,2):C.UVGC_1517_1947,(0,0,6):C.UVGC_1517_1948,(0,0,1):C.UVGC_1517_1949,(0,0,4):C.UVGC_1517_1950})

V_790 = CTVertex(name = 'V_790',
                 type = 'UV',
                 particles = [ P.W__minus__, P.Z, P.sd3__tilde__, P.su3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ], [ [P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_1653_2978,(0,0,3):C.UVGC_1653_2979,(0,0,5):C.UVGC_1653_2980,(0,0,2):C.UVGC_1653_2981,(0,0,6):C.UVGC_1653_2982,(0,0,1):C.UVGC_1653_2983,(0,0,4):C.UVGC_1653_2984})

V_791 = CTVertex(name = 'V_791',
                 type = 'UV',
                 particles = [ P.W__plus__, P.Z, P.sd1, P.su1__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.d, P.go, P.u] ], [ [P.go, P.u] ], [ [P.g, P.sd1] ], [ [P.g, P.sd1, P.su1] ], [ [P.g, P.su1] ], [ [P.sd1, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_1492_1833,(0,0,3):C.UVGC_1492_1834,(0,0,5):C.UVGC_1492_1835,(0,0,2):C.UVGC_1492_1836,(0,0,6):C.UVGC_1492_1837,(0,0,1):C.UVGC_1492_1838,(0,0,4):C.UVGC_1492_1839})

V_792 = CTVertex(name = 'V_792',
                 type = 'UV',
                 particles = [ P.W__plus__, P.Z, P.sd2, P.su2__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.c, P.go, P.s] ], [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.g, P.sd2, P.su2] ], [ [P.g, P.su2] ], [ [P.sd2, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1539_2089,(0,0,3):C.UVGC_1539_2090,(0,0,5):C.UVGC_1539_2091,(0,0,2):C.UVGC_1539_2092,(0,0,6):C.UVGC_1539_2093,(0,0,1):C.UVGC_1539_2094,(0,0,4):C.UVGC_1539_2095})

V_793 = CTVertex(name = 'V_793',
                 type = 'UV',
                 particles = [ P.W__plus__, P.Z, P.sd3, P.su3__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.b, P.go, P.t] ], [ [P.go, P.t] ], [ [P.g, P.sd3] ], [ [P.g, P.sd3, P.su3] ], [ [P.g, P.su3] ], [ [P.sd3, P.su3] ] ],
                 couplings = {(0,0,0):C.UVGC_1671_3103,(0,0,3):C.UVGC_1671_3104,(0,0,5):C.UVGC_1671_3105,(0,0,2):C.UVGC_1671_3106,(0,0,6):C.UVGC_1671_3107,(0,0,1):C.UVGC_1671_3108,(0,0,4):C.UVGC_1671_3109})

V_794 = CTVertex(name = 'V_794',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ], [ [P.sd1] ] ],
                 couplings = {(0,0,2):C.UVGC_1270_760,(0,0,0):C.UVGC_1270_761,(0,0,1):C.UVGC_1270_762})

V_795 = CTVertex(name = 'V_795',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.sd2] ] ],
                 couplings = {(0,0,2):C.UVGC_1296_895,(0,0,1):C.UVGC_1296_896,(0,0,0):C.UVGC_1296_897})

V_796 = CTVertex(name = 'V_796',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ], [ [P.sd3] ] ],
                 couplings = {(0,0,2):C.UVGC_1307_979,(0,0,0):C.UVGC_1307_980,(0,0,1):C.UVGC_1307_981})

V_797 = CTVertex(name = 'V_797',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.sd4__tilde__, P.sd4 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ], [ [P.sd4] ] ],
                 couplings = {(0,0,2):C.UVGC_1329_1071,(0,0,0):C.UVGC_1329_1072,(0,0,1):C.UVGC_1329_1073})

V_798 = CTVertex(name = 'V_798',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.sd5__tilde__, P.sd5 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ], [ [P.sd5] ] ],
                 couplings = {(0,0,2):C.UVGC_1354_1199,(0,0,1):C.UVGC_1354_1200,(0,0,0):C.UVGC_1354_1201})

V_799 = CTVertex(name = 'V_799',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ], [ [P.sd6] ] ],
                 couplings = {(0,0,2):C.UVGC_1396_1328,(0,0,0):C.UVGC_1396_1329,(0,0,1):C.UVGC_1396_1330})

V_800 = CTVertex(name = 'V_800',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.su1] ] ],
                 couplings = {(0,0,2):C.UVGC_1487_1800,(0,0,1):C.UVGC_1487_1801,(0,0,0):C.UVGC_1487_1802})

V_801 = CTVertex(name = 'V_801',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ], [ [P.su2] ] ],
                 couplings = {(0,0,2):C.UVGC_1534_2056,(0,0,0):C.UVGC_1534_2057,(0,0,1):C.UVGC_1534_2058})

V_802 = CTVertex(name = 'V_802',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.su3] ] ],
                 couplings = {(0,0,2):C.UVGC_1664_3066,(0,0,1):C.UVGC_1664_3067,(0,0,0):C.UVGC_1664_3068})

V_803 = CTVertex(name = 'V_803',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.su4] ] ],
                 couplings = {(0,0,2):C.UVGC_1561_2194,(0,0,1):C.UVGC_1561_2195,(0,0,0):C.UVGC_1561_2196})

V_804 = CTVertex(name = 'V_804',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ], [ [P.su5] ] ],
                 couplings = {(0,0,2):C.UVGC_1587_2324,(0,0,0):C.UVGC_1587_2325,(0,0,1):C.UVGC_1587_2326})

V_805 = CTVertex(name = 'V_805',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.su6] ] ],
                 couplings = {(0,0,2):C.UVGC_1744_3411,(0,0,1):C.UVGC_1744_3412,(0,0,0):C.UVGC_1744_3413})

V_806 = CTVertex(name = 'V_806',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1198_269,(0,1,0):C.UVGC_1409_1366,(0,1,1):C.UVGC_1412_1374,(0,1,2):C.UVGC_1412_1375,(0,2,0):C.UVGC_1409_1366,(0,2,1):C.UVGC_1409_1367,(0,2,2):C.UVGC_1409_1368})

V_807 = CTVertex(name = 'V_807',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.go, P.su2] ], [ [P.go, P.su5] ] ],
                 couplings = {(0,0,0):C.UVGC_1201_271,(0,1,0):C.UVGC_1452_1602,(0,1,1):C.UVGC_1498_1857,(0,2,0):C.UVGC_1452_1602,(0,2,2):C.UVGC_1571_2243})

V_808 = CTVertex(name = 'V_808',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.go, P.sd1] ], [ [P.go, P.sd4] ] ],
                 couplings = {(0,0,0):C.UVGC_1198_269,(0,1,0):C.UVGC_1246_623,(0,1,1):C.UVGC_1246_624,(0,2,0):C.UVGC_1246_623,(0,2,2):C.UVGC_1314_992})

V_809 = CTVertex(name = 'V_809',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.go, P.sd2] ], [ [P.go, P.sd5] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,2):C.UVGC_1198_269,(0,1,2):C.UVGC_1246_623,(0,1,0):C.UVGC_1272_764,(0,2,2):C.UVGC_1246_623,(0,2,1):C.UVGC_1339_1120})

V_810 = CTVertex(name = 'V_810',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_1201_271,(0,1,2):C.UVGC_1616_2744,(0,1,0):C.UVGC_1616_2745,(0,1,1):C.UVGC_1616_2746,(0,2,2):C.UVGC_1616_2744,(0,2,0):C.UVGC_1620_2755,(0,2,1):C.UVGC_1620_2756})

V_811 = CTVertex(name = 'V_811',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.go, P.su1] ], [ [P.go, P.su4] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1201_271,(0,1,2):C.UVGC_1452_1602,(0,1,0):C.UVGC_1452_1603,(0,2,2):C.UVGC_1452_1602,(0,2,1):C.UVGC_1545_2113})

V_812 = CTVertex(name = 'V_812',
                 type = 'UV',
                 particles = [ P.go, P.go, P.g ],
                 color = [ 'd(2,1,3)', 'f(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV2, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.go] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(1,0,0):C.UVGC_1764_3540,(1,0,3):C.UVGC_1764_3541,(1,0,8):C.UVGC_1764_3542,(1,0,9):C.UVGC_1764_3543,(1,0,1):C.UVGC_1764_3544,(1,0,2):C.UVGC_1764_3545,(1,0,4):C.UVGC_1764_3546,(1,0,5):C.UVGC_1764_3547,(1,0,6):C.UVGC_1764_3548,(1,0,7):C.UVGC_1764_3549,(1,0,10):C.UVGC_1764_3550,(1,0,11):C.UVGC_1764_3551,(1,0,12):C.UVGC_1764_3552,(1,0,13):C.UVGC_1764_3553,(1,0,14):C.UVGC_1764_3554,(1,0,15):C.UVGC_1764_3555,(1,0,16):C.UVGC_1764_3556,(1,2,1):C.UVGC_1232_478,(1,2,2):C.UVGC_1232_479,(1,2,4):C.UVGC_1232_480,(1,2,5):C.UVGC_1232_481,(1,2,6):C.UVGC_1232_482,(1,2,7):C.UVGC_1232_483,(1,2,11):C.UVGC_1232_484,(1,2,12):C.UVGC_1232_485,(1,2,13):C.UVGC_1233_490,(1,2,14):C.UVGC_1232_487,(1,2,15):C.UVGC_1232_488,(1,2,16):C.UVGC_1232_489,(1,3,1):C.UVGC_1232_478,(1,3,2):C.UVGC_1232_479,(1,3,4):C.UVGC_1232_480,(1,3,5):C.UVGC_1232_481,(1,3,6):C.UVGC_1232_482,(1,3,7):C.UVGC_1232_483,(1,3,11):C.UVGC_1232_484,(1,3,12):C.UVGC_1232_485,(1,3,13):C.UVGC_1232_486,(1,3,14):C.UVGC_1232_487,(1,3,15):C.UVGC_1232_488,(1,3,16):C.UVGC_1232_489,(0,1,13):C.UVGC_1229_453,(1,1,13):C.UVGC_1228_452,(0,2,1):C.UVGC_1235_502,(0,2,2):C.UVGC_1235_503,(0,2,4):C.UVGC_1235_504,(0,2,5):C.UVGC_1235_505,(0,2,6):C.UVGC_1235_506,(0,2,7):C.UVGC_1235_507,(0,2,11):C.UVGC_1235_508,(0,2,12):C.UVGC_1235_509,(0,2,14):C.UVGC_1235_510,(0,2,15):C.UVGC_1235_511,(0,2,16):C.UVGC_1235_512,(0,3,1):C.UVGC_1234_491,(0,3,2):C.UVGC_1234_492,(0,3,4):C.UVGC_1234_493,(0,3,5):C.UVGC_1234_494,(0,3,6):C.UVGC_1234_495,(0,3,7):C.UVGC_1234_496,(0,3,11):C.UVGC_1234_497,(0,3,12):C.UVGC_1234_498,(0,3,14):C.UVGC_1234_499,(0,3,15):C.UVGC_1234_500,(0,3,16):C.UVGC_1234_501})

V_813 = CTVertex(name = 'V_813',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.g] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ] ],
                 couplings = {(0,0,1):C.UVGC_1199_270,(0,1,0):C.UVGC_1247_625,(0,1,2):C.UVGC_1247_626,(0,1,3):C.UVGC_1247_627,(0,1,4):C.UVGC_1247_628,(0,1,1):C.UVGC_1410_1369,(0,1,5):C.UVGC_1413_1376,(0,1,6):C.UVGC_1413_1377,(0,2,0):C.UVGC_1247_625,(0,2,2):C.UVGC_1247_626,(0,2,3):C.UVGC_1247_627,(0,2,4):C.UVGC_1247_628,(0,2,1):C.UVGC_1410_1369,(0,2,5):C.UVGC_1410_1370,(0,2,6):C.UVGC_1410_1371})

V_814 = CTVertex(name = 'V_814',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.g] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.su2] ], [ [P.go, P.su5] ] ],
                 couplings = {(0,0,2):C.UVGC_1199_270,(0,1,0):C.UVGC_1247_625,(0,1,1):C.UVGC_1247_626,(0,1,3):C.UVGC_1247_627,(0,1,4):C.UVGC_1247_628,(0,1,2):C.UVGC_1247_629,(0,1,5):C.UVGC_1499_1858,(0,2,0):C.UVGC_1247_625,(0,2,1):C.UVGC_1247_626,(0,2,3):C.UVGC_1247_627,(0,2,4):C.UVGC_1247_628,(0,2,2):C.UVGC_1247_629,(0,2,6):C.UVGC_1572_2244})

V_815 = CTVertex(name = 'V_815',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.d, P.g] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.sd1] ], [ [P.go, P.sd4] ] ],
                 couplings = {(0,0,2):C.UVGC_1199_270,(0,1,0):C.UVGC_1247_625,(0,1,1):C.UVGC_1247_626,(0,1,3):C.UVGC_1247_627,(0,1,4):C.UVGC_1247_628,(0,1,2):C.UVGC_1247_629,(0,1,5):C.UVGC_1247_630,(0,2,0):C.UVGC_1247_625,(0,2,1):C.UVGC_1247_626,(0,2,3):C.UVGC_1247_627,(0,2,4):C.UVGC_1247_628,(0,2,2):C.UVGC_1247_629,(0,2,6):C.UVGC_1315_993})

V_816 = CTVertex(name = 'V_816',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.sd2] ], [ [P.go, P.sd5] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,6):C.UVGC_1199_270,(0,1,0):C.UVGC_1247_625,(0,1,1):C.UVGC_1247_626,(0,1,2):C.UVGC_1247_627,(0,1,3):C.UVGC_1247_628,(0,1,6):C.UVGC_1247_629,(0,1,4):C.UVGC_1273_765,(0,2,0):C.UVGC_1247_625,(0,2,1):C.UVGC_1247_626,(0,2,2):C.UVGC_1247_627,(0,2,3):C.UVGC_1247_628,(0,2,6):C.UVGC_1247_629,(0,2,5):C.UVGC_1340_1121})

V_817 = CTVertex(name = 'V_817',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,6):C.UVGC_1199_270,(0,1,0):C.UVGC_1247_625,(0,1,1):C.UVGC_1247_626,(0,1,2):C.UVGC_1247_627,(0,1,3):C.UVGC_1247_628,(0,1,6):C.UVGC_1617_2747,(0,1,4):C.UVGC_1617_2748,(0,1,5):C.UVGC_1617_2749,(0,2,0):C.UVGC_1247_625,(0,2,1):C.UVGC_1247_626,(0,2,2):C.UVGC_1247_627,(0,2,3):C.UVGC_1247_628,(0,2,6):C.UVGC_1617_2747,(0,2,4):C.UVGC_1621_2757,(0,2,5):C.UVGC_1621_2758})

V_818 = CTVertex(name = 'V_818',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.go, P.su1] ], [ [P.go, P.su4] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,6):C.UVGC_1199_270,(0,1,0):C.UVGC_1247_625,(0,1,1):C.UVGC_1247_626,(0,1,2):C.UVGC_1247_627,(0,1,3):C.UVGC_1247_628,(0,1,6):C.UVGC_1247_629,(0,1,4):C.UVGC_1453_1604,(0,2,0):C.UVGC_1247_625,(0,2,1):C.UVGC_1247_626,(0,2,2):C.UVGC_1247_627,(0,2,3):C.UVGC_1247_628,(0,2,6):C.UVGC_1247_629,(0,2,5):C.UVGC_1546_2114})

V_819 = CTVertex(name = 'V_819',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ], [ [P.go, P.sd2] ], [ [P.go, P.sd2, P.su2] ], [ [P.go, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1500_1859,(0,0,2):C.UVGC_1500_1860,(0,0,4):C.UVGC_1500_1861,(0,0,1):C.UVGC_1500_1862,(0,0,3):C.UVGC_1500_1863})

V_820 = CTVertex(name = 'V_820',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd3, P.su3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1622_2759,(0,0,7):C.UVGC_1622_2760,(0,0,2):C.UVGC_1622_2761,(0,0,4):C.UVGC_1622_2762,(0,0,5):C.UVGC_1622_2763,(0,0,6):C.UVGC_1622_2764,(0,0,1):C.UVGC_1622_2765,(0,0,3):C.UVGC_1622_2766})

V_821 = CTVertex(name = 'V_821',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ], [ [P.go, P.sd1] ], [ [P.go, P.sd1, P.su1] ], [ [P.go, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_1454_1605,(0,0,2):C.UVGC_1454_1606,(0,0,4):C.UVGC_1454_1607,(0,0,1):C.UVGC_1454_1608,(0,0,3):C.UVGC_1454_1609})

V_822 = CTVertex(name = 'V_822',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.go, P.sd3] ], [ [P.go, P.sd3, P.su3] ], [ [P.go, P.sd6] ], [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1625_2773,(0,0,7):C.UVGC_1625_2774,(0,0,2):C.UVGC_1625_2775,(0,0,4):C.UVGC_1625_2776,(0,0,5):C.UVGC_1625_2777,(0,0,6):C.UVGC_1625_2778,(0,0,1):C.UVGC_1625_2779,(0,0,3):C.UVGC_1625_2780})

V_823 = CTVertex(name = 'V_823',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ], [ [P.go, P.sd1] ], [ [P.go, P.sd1, P.su1] ], [ [P.go, P.su1] ] ],
                 couplings = {(0,0,0):C.UVGC_1456_1611,(0,0,2):C.UVGC_1456_1612,(0,0,4):C.UVGC_1456_1613,(0,0,1):C.UVGC_1456_1614,(0,0,3):C.UVGC_1456_1615})

V_824 = CTVertex(name = 'V_824',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ], [ [P.go, P.sd2] ], [ [P.go, P.sd2, P.su2] ], [ [P.go, P.su2] ] ],
                 couplings = {(0,0,0):C.UVGC_1502_1865,(0,0,2):C.UVGC_1502_1866,(0,0,4):C.UVGC_1502_1867,(0,0,1):C.UVGC_1502_1868,(0,0,3):C.UVGC_1502_1869})

V_825 = CTVertex(name = 'V_825',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ] ],
                 couplings = {(0,0,0):C.UVGC_1420_1408,(0,0,1):C.UVGC_1420_1409,(0,0,2):C.UVGC_1420_1410,(0,1,0):C.UVGC_1419_1405,(0,1,1):C.UVGC_1419_1406,(0,1,2):C.UVGC_1419_1407})

V_826 = CTVertex(name = 'V_826',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.go, P.su2] ], [ [P.go, P.su5] ] ],
                 couplings = {(0,0,0):C.UVGC_1501_1864,(0,1,1):C.UVGC_1573_2245})

V_827 = CTVertex(name = 'V_827',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.go, P.sd1] ], [ [P.go, P.sd4] ] ],
                 couplings = {(0,0,0):C.UVGC_1248_631,(0,1,1):C.UVGC_1316_994})

V_828 = CTVertex(name = 'V_828',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.go, P.sd2] ], [ [P.go, P.sd5] ] ],
                 couplings = {(0,0,0):C.UVGC_1274_766,(0,1,1):C.UVGC_1341_1122})

V_829 = CTVertex(name = 'V_829',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_1623_2767,(0,0,0):C.UVGC_1623_2768,(0,0,1):C.UVGC_1623_2769,(0,1,2):C.UVGC_1624_2770,(0,1,0):C.UVGC_1624_2771,(0,1,1):C.UVGC_1624_2772})

V_830 = CTVertex(name = 'V_830',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.go, P.su1] ], [ [P.go, P.su4] ] ],
                 couplings = {(0,0,0):C.UVGC_1455_1610,(0,1,1):C.UVGC_1547_2115})

V_831 = CTVertex(name = 'V_831',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF2, L.FF3, L.FF4, L.FF5 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.go, P.sd3] ], [ [P.go, P.sd6] ] ],
                 couplings = {(0,1,0):C.UVGC_1414_1378,(0,1,1):C.UVGC_1414_1379,(0,1,2):C.UVGC_1414_1380,(0,3,0):C.UVGC_1414_1378,(0,3,1):C.UVGC_1414_1379,(0,3,2):C.UVGC_1414_1380,(0,2,0):C.UVGC_1408_1363,(0,2,1):C.UVGC_1411_1372,(0,2,2):C.UVGC_1411_1373,(0,4,0):C.UVGC_1408_1363,(0,4,1):C.UVGC_1408_1364,(0,4,2):C.UVGC_1408_1365,(0,0,0):C.UVGC_1174_3})

V_832 = CTVertex(name = 'V_832',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF3, L.FF5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.go, P.su2] ], [ [P.go, P.su5] ] ],
                 couplings = {(0,0,0):C.UVGC_1174_3,(0,1,0):C.UVGC_1245_621,(0,1,1):C.UVGC_1497_1856,(0,2,0):C.UVGC_1245_621,(0,2,2):C.UVGC_1570_2242})

V_833 = CTVertex(name = 'V_833',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF3, L.FF5 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.go, P.sd1] ], [ [P.go, P.sd4] ] ],
                 couplings = {(0,0,0):C.UVGC_1174_3,(0,1,0):C.UVGC_1245_621,(0,1,1):C.UVGC_1245_622,(0,2,0):C.UVGC_1245_621,(0,2,2):C.UVGC_1313_991})

V_834 = CTVertex(name = 'V_834',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF3, L.FF5 ],
                 loop_particles = [ [ [P.go, P.sd2] ], [ [P.go, P.sd5] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,2):C.UVGC_1174_3,(0,1,2):C.UVGC_1245_621,(0,1,0):C.UVGC_1271_763,(0,2,2):C.UVGC_1245_621,(0,2,1):C.UVGC_1338_1119})

V_835 = CTVertex(name = 'V_835',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF2, L.FF3, L.FF4, L.FF5 ],
                 loop_particles = [ [ [P.go, P.su3] ], [ [P.go, P.su6] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,2):C.UVGC_1618_2750,(0,1,0):C.UVGC_1618_2751,(0,1,1):C.UVGC_1618_2752,(0,3,2):C.UVGC_1618_2750,(0,3,0):C.UVGC_1618_2751,(0,3,1):C.UVGC_1618_2752,(0,2,2):C.UVGC_1615_2741,(0,2,0):C.UVGC_1615_2742,(0,2,1):C.UVGC_1615_2743,(0,4,2):C.UVGC_1615_2741,(0,4,0):C.UVGC_1619_2753,(0,4,1):C.UVGC_1619_2754,(0,0,2):C.UVGC_1174_3})

V_836 = CTVertex(name = 'V_836',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF3, L.FF5 ],
                 loop_particles = [ [ [P.go, P.su1] ], [ [P.go, P.su4] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_1174_3,(0,1,2):C.UVGC_1245_621,(0,1,0):C.UVGC_1451_1601,(0,2,2):C.UVGC_1245_621,(0,2,1):C.UVGC_1544_2112})

V_837 = CTVertex(name = 'V_837',
                 type = 'UV',
                 particles = [ P.go, P.go ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF2, L.FF3, L.FF4, L.FF5 ],
                 loop_particles = [ [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.g, P.go] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1765_3557,(0,1,1):C.UVGC_1765_3558,(0,1,2):C.UVGC_1765_3559,(0,1,3):C.UVGC_1765_3560,(0,1,4):C.UVGC_1765_3561,(0,1,5):C.UVGC_1765_3562,(0,1,6):C.UVGC_1765_3563,(0,1,7):C.UVGC_1765_3564,(0,1,8):C.UVGC_1765_3565,(0,1,9):C.UVGC_1765_3566,(0,1,10):C.UVGC_1765_3567,(0,1,11):C.UVGC_1765_3568,(0,1,12):C.UVGC_1765_3569,(0,3,0):C.UVGC_1765_3557,(0,3,1):C.UVGC_1765_3558,(0,3,2):C.UVGC_1765_3559,(0,3,3):C.UVGC_1765_3560,(0,3,4):C.UVGC_1765_3561,(0,3,5):C.UVGC_1765_3562,(0,3,6):C.UVGC_1765_3563,(0,3,7):C.UVGC_1765_3564,(0,3,8):C.UVGC_1765_3565,(0,3,9):C.UVGC_1765_3566,(0,3,10):C.UVGC_1765_3567,(0,3,11):C.UVGC_1765_3568,(0,3,12):C.UVGC_1765_3569,(0,2,0):C.UVGC_1763_3527,(0,2,1):C.UVGC_1763_3528,(0,2,2):C.UVGC_1763_3529,(0,2,3):C.UVGC_1763_3530,(0,2,4):C.UVGC_1763_3531,(0,2,5):C.UVGC_1763_3532,(0,2,6):C.UVGC_1763_3533,(0,2,7):C.UVGC_1763_3534,(0,2,8):C.UVGC_1763_3535,(0,2,9):C.UVGC_1763_3536,(0,2,10):C.UVGC_1763_3537,(0,2,11):C.UVGC_1763_3538,(0,2,12):C.UVGC_1763_3539,(0,4,0):C.UVGC_1763_3527,(0,4,1):C.UVGC_1763_3528,(0,4,2):C.UVGC_1763_3529,(0,4,3):C.UVGC_1763_3530,(0,4,4):C.UVGC_1763_3531,(0,4,5):C.UVGC_1763_3532,(0,4,6):C.UVGC_1763_3533,(0,4,7):C.UVGC_1763_3534,(0,4,8):C.UVGC_1763_3535,(0,4,9):C.UVGC_1763_3536,(0,4,10):C.UVGC_1763_3537,(0,4,11):C.UVGC_1763_3538,(0,4,12):C.UVGC_1763_3539,(0,0,0):C.UVGC_1230_454,(0,0,1):C.UVGC_1230_455,(0,0,2):C.UVGC_1230_456,(0,0,3):C.UVGC_1230_457,(0,0,4):C.UVGC_1230_458,(0,0,5):C.UVGC_1230_459,(0,0,7):C.UVGC_1230_460,(0,0,8):C.UVGC_1230_461,(0,0,9):C.UVGC_1230_462,(0,0,10):C.UVGC_1230_463,(0,0,11):C.UVGC_1230_464,(0,0,12):C.UVGC_1230_465})

V_838 = CTVertex(name = 'V_838',
                 type = 'UV',
                 particles = [ P.su1__tilde__, P.su1 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su1] ], [ [P.su1] ] ],
                 couplings = {(0,0,2):C.UVGC_1462_1637,(0,0,1):C.UVGC_1462_1638,(0,0,0):C.UVGC_1462_1639,(0,1,0):C.UVGC_1461_1636})

V_839 = CTVertex(name = 'V_839',
                 type = 'UV',
                 particles = [ P.su2__tilde__, P.su2 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su2] ], [ [P.su2] ] ],
                 couplings = {(0,0,2):C.UVGC_1508_1891,(0,0,0):C.UVGC_1508_1892,(0,0,1):C.UVGC_1508_1893,(0,1,0):C.UVGC_1507_1890})

V_840 = CTVertex(name = 'V_840',
                 type = 'UV',
                 particles = [ P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su3] ], [ [P.su3] ] ],
                 couplings = {(0,0,2):C.UVGC_1645_2908,(0,0,1):C.UVGC_1645_2909,(0,0,0):C.UVGC_1645_2910,(0,1,0):C.UVGC_1644_2907})

V_841 = CTVertex(name = 'V_841',
                 type = 'UV',
                 particles = [ P.su4__tilde__, P.su4 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.go, P.u] ], [ [P.g, P.su4] ], [ [P.su4] ] ],
                 couplings = {(0,0,2):C.UVGC_1549_2117,(0,0,1):C.UVGC_1549_2118,(0,0,0):C.UVGC_1549_2119,(0,1,0):C.UVGC_1548_2116})

V_842 = CTVertex(name = 'V_842',
                 type = 'UV',
                 particles = [ P.su5__tilde__, P.su5 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.c, P.go] ], [ [P.g, P.su5] ], [ [P.su5] ] ],
                 couplings = {(0,0,2):C.UVGC_1575_2247,(0,0,0):C.UVGC_1575_2248,(0,0,1):C.UVGC_1575_2249,(0,1,0):C.UVGC_1574_2246})

V_843 = CTVertex(name = 'V_843',
                 type = 'UV',
                 particles = [ P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.go, P.t] ], [ [P.g, P.su6] ], [ [P.su6] ] ],
                 couplings = {(0,0,2):C.UVGC_1701_3253,(0,0,1):C.UVGC_1701_3254,(0,0,0):C.UVGC_1701_3255,(0,1,0):C.UVGC_1700_3252})

V_844 = CTVertex(name = 'V_844',
                 type = 'UV',
                 particles = [ P.sd1__tilde__, P.sd1 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd1] ], [ [P.sd1] ] ],
                 couplings = {(0,0,2):C.UVGC_1250_633,(0,0,0):C.UVGC_1250_634,(0,0,1):C.UVGC_1250_635,(0,1,0):C.UVGC_1249_632})

V_845 = CTVertex(name = 'V_845',
                 type = 'UV',
                 particles = [ P.sd2__tilde__, P.sd2 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd2] ], [ [P.sd2] ] ],
                 couplings = {(0,0,2):C.UVGC_1276_768,(0,0,1):C.UVGC_1276_769,(0,0,0):C.UVGC_1276_770,(0,1,0):C.UVGC_1275_767})

V_846 = CTVertex(name = 'V_846',
                 type = 'UV',
                 particles = [ P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd3] ], [ [P.sd3] ] ],
                 couplings = {(0,0,2):C.UVGC_1298_899,(0,0,0):C.UVGC_1298_900,(0,0,1):C.UVGC_1298_901,(0,1,0):C.UVGC_1297_898})

V_847 = CTVertex(name = 'V_847',
                 type = 'UV',
                 particles = [ P.sd4__tilde__, P.sd4 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.d, P.go] ], [ [P.g, P.sd4] ], [ [P.sd4] ] ],
                 couplings = {(0,0,2):C.UVGC_1318_996,(0,0,0):C.UVGC_1318_997,(0,0,1):C.UVGC_1318_998,(0,1,0):C.UVGC_1317_995})

V_848 = CTVertex(name = 'V_848',
                 type = 'UV',
                 particles = [ P.sd5__tilde__, P.sd5 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.go, P.s] ], [ [P.g, P.sd5] ], [ [P.sd5] ] ],
                 couplings = {(0,0,2):C.UVGC_1343_1124,(0,0,1):C.UVGC_1343_1125,(0,0,0):C.UVGC_1343_1126,(0,1,0):C.UVGC_1342_1123})

V_849 = CTVertex(name = 'V_849',
                 type = 'UV',
                 particles = [ P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.b, P.go] ], [ [P.g, P.sd6] ], [ [P.sd6] ] ],
                 couplings = {(0,0,2):C.UVGC_1368_1252,(0,0,0):C.UVGC_1368_1253,(0,0,1):C.UVGC_1368_1254,(0,1,0):C.UVGC_1367_1251})

V_850 = CTVertex(name = 'V_850',
                 type = 'UV',
                 particles = [ P.g, P.g ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VV1, L.VV2, L.VV3 ],
                 loop_particles = [ [ [P.b] ], [ [P.g] ], [ [P.ghG] ], [ [P.go] ], [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1597_2387,(0,0,1):C.UVGC_1597_2388,(0,0,2):C.UVGC_1597_2389,(0,0,3):C.UVGC_1597_2390,(0,0,4):C.UVGC_1597_2391,(0,0,5):C.UVGC_1597_2392,(0,0,6):C.UVGC_1597_2393,(0,0,7):C.UVGC_1597_2394,(0,0,8):C.UVGC_1597_2395,(0,0,9):C.UVGC_1597_2396,(0,0,10):C.UVGC_1597_2397,(0,0,11):C.UVGC_1597_2398,(0,0,12):C.UVGC_1597_2399,(0,0,13):C.UVGC_1597_2400,(0,0,14):C.UVGC_1597_2401,(0,0,15):C.UVGC_1597_2402,(0,0,16):C.UVGC_1597_2403,(0,1,4):C.UVGC_1177_6,(0,1,5):C.UVGC_1177_7,(0,1,6):C.UVGC_1177_8,(0,1,7):C.UVGC_1177_9,(0,1,8):C.UVGC_1177_10,(0,1,9):C.UVGC_1177_11,(0,1,10):C.UVGC_1177_12,(0,1,11):C.UVGC_1177_13,(0,1,12):C.UVGC_1177_14,(0,1,13):C.UVGC_1177_15,(0,1,14):C.UVGC_1177_16,(0,1,15):C.UVGC_1177_17,(0,2,0):C.UVGC_1596_2372,(0,2,3):C.UVGC_1596_2373,(0,2,4):C.UVGC_1596_2374,(0,2,5):C.UVGC_1596_2375,(0,2,6):C.UVGC_1596_2376,(0,2,7):C.UVGC_1596_2377,(0,2,8):C.UVGC_1596_2378,(0,2,9):C.UVGC_1596_2379,(0,2,10):C.UVGC_1596_2380,(0,2,11):C.UVGC_1596_2381,(0,2,12):C.UVGC_1596_2382,(0,2,13):C.UVGC_1596_2383,(0,2,14):C.UVGC_1596_2384,(0,2,15):C.UVGC_1596_2385,(0,2,16):C.UVGC_1596_2386})

V_851 = CTVertex(name = 'V_851',
                 type = 'UV',
                 particles = [ P.go, P.go, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV2 ],
                 loop_particles = [ [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1231_466,(0,0,1):C.UVGC_1231_467,(0,0,2):C.UVGC_1231_468,(0,0,3):C.UVGC_1231_469,(0,0,4):C.UVGC_1231_470,(0,0,5):C.UVGC_1231_471,(0,0,6):C.UVGC_1231_472,(0,0,7):C.UVGC_1231_473,(0,0,8):C.UVGC_1231_474,(0,0,9):C.UVGC_1231_475,(0,0,10):C.UVGC_1231_476,(0,0,11):C.UVGC_1231_477})

V_852 = CTVertex(name = 'V_852',
                 type = 'UV',
                 particles = [ P.go, P.go, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV2 ],
                 loop_particles = [ [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1240_561,(0,0,1):C.UVGC_1240_562,(0,0,2):C.UVGC_1240_563,(0,0,3):C.UVGC_1240_564,(0,0,4):C.UVGC_1240_565,(0,0,5):C.UVGC_1240_566,(0,0,6):C.UVGC_1240_567,(0,0,7):C.UVGC_1240_568,(0,0,8):C.UVGC_1240_569,(0,0,9):C.UVGC_1240_570,(0,0,10):C.UVGC_1240_571,(0,0,11):C.UVGC_1240_572})

V_853 = CTVertex(name = 'V_853',
                 type = 'UV',
                 particles = [ P.go, P.n1, P.g ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1236_513,(0,0,1):C.UVGC_1236_514,(0,0,2):C.UVGC_1236_515,(0,0,3):C.UVGC_1236_516,(0,0,4):C.UVGC_1236_517,(0,0,5):C.UVGC_1236_518,(0,0,6):C.UVGC_1236_519,(0,0,7):C.UVGC_1236_520,(0,0,8):C.UVGC_1236_521,(0,0,9):C.UVGC_1236_522,(0,0,10):C.UVGC_1236_523,(0,0,11):C.UVGC_1236_524,(0,1,0):C.UVGC_1241_573,(0,1,1):C.UVGC_1241_574,(0,1,2):C.UVGC_1241_575,(0,1,3):C.UVGC_1241_576,(0,1,4):C.UVGC_1241_577,(0,1,5):C.UVGC_1241_578,(0,1,6):C.UVGC_1241_579,(0,1,7):C.UVGC_1241_580,(0,1,8):C.UVGC_1241_581,(0,1,9):C.UVGC_1241_582,(0,1,10):C.UVGC_1241_583,(0,1,11):C.UVGC_1241_584})

V_854 = CTVertex(name = 'V_854',
                 type = 'UV',
                 particles = [ P.go, P.n2, P.g ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1237_525,(0,0,1):C.UVGC_1237_526,(0,0,2):C.UVGC_1237_527,(0,0,3):C.UVGC_1237_528,(0,0,4):C.UVGC_1237_529,(0,0,5):C.UVGC_1237_530,(0,0,6):C.UVGC_1237_531,(0,0,7):C.UVGC_1237_532,(0,0,8):C.UVGC_1237_533,(0,0,9):C.UVGC_1237_534,(0,0,10):C.UVGC_1237_535,(0,0,11):C.UVGC_1237_536,(0,1,0):C.UVGC_1242_585,(0,1,1):C.UVGC_1242_586,(0,1,2):C.UVGC_1242_587,(0,1,3):C.UVGC_1242_588,(0,1,4):C.UVGC_1242_589,(0,1,5):C.UVGC_1242_590,(0,1,6):C.UVGC_1242_591,(0,1,7):C.UVGC_1242_592,(0,1,8):C.UVGC_1242_593,(0,1,9):C.UVGC_1242_594,(0,1,10):C.UVGC_1242_595,(0,1,11):C.UVGC_1242_596})

V_855 = CTVertex(name = 'V_855',
                 type = 'UV',
                 particles = [ P.go, P.n3, P.g ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1238_537,(0,0,1):C.UVGC_1238_538,(0,0,2):C.UVGC_1238_539,(0,0,3):C.UVGC_1238_540,(0,0,4):C.UVGC_1238_541,(0,0,5):C.UVGC_1238_542,(0,0,6):C.UVGC_1238_543,(0,0,7):C.UVGC_1238_544,(0,0,8):C.UVGC_1238_545,(0,0,9):C.UVGC_1238_546,(0,0,10):C.UVGC_1238_547,(0,0,11):C.UVGC_1238_548,(0,1,0):C.UVGC_1243_597,(0,1,1):C.UVGC_1243_598,(0,1,2):C.UVGC_1243_599,(0,1,3):C.UVGC_1243_600,(0,1,4):C.UVGC_1243_601,(0,1,5):C.UVGC_1243_602,(0,1,6):C.UVGC_1243_603,(0,1,7):C.UVGC_1243_604,(0,1,8):C.UVGC_1243_605,(0,1,9):C.UVGC_1243_606,(0,1,10):C.UVGC_1243_607,(0,1,11):C.UVGC_1243_608})

V_856 = CTVertex(name = 'V_856',
                 type = 'UV',
                 particles = [ P.go, P.n4, P.g ],
                 color = [ 'Identity(1,3)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b, P.sd3] ], [ [P.b, P.sd6] ], [ [P.c, P.su2] ], [ [P.c, P.su5] ], [ [P.d, P.sd1] ], [ [P.d, P.sd4] ], [ [P.s, P.sd2] ], [ [P.s, P.sd5] ], [ [P.su1, P.u] ], [ [P.su3, P.t] ], [ [P.su4, P.u] ], [ [P.su6, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1239_549,(0,0,1):C.UVGC_1239_550,(0,0,2):C.UVGC_1239_551,(0,0,3):C.UVGC_1239_552,(0,0,4):C.UVGC_1239_553,(0,0,5):C.UVGC_1239_554,(0,0,6):C.UVGC_1239_555,(0,0,7):C.UVGC_1239_556,(0,0,8):C.UVGC_1239_557,(0,0,9):C.UVGC_1239_558,(0,0,10):C.UVGC_1239_559,(0,0,11):C.UVGC_1239_560,(0,1,0):C.UVGC_1244_609,(0,1,1):C.UVGC_1244_610,(0,1,2):C.UVGC_1244_611,(0,1,3):C.UVGC_1244_612,(0,1,4):C.UVGC_1244_613,(0,1,5):C.UVGC_1244_614,(0,1,6):C.UVGC_1244_615,(0,1,7):C.UVGC_1244_616,(0,1,8):C.UVGC_1244_617,(0,1,9):C.UVGC_1244_618,(0,1,10):C.UVGC_1244_619,(0,1,11):C.UVGC_1244_620})

V_857 = CTVertex(name = 'V_857',
                 type = 'UV',
                 particles = [ P.g, P.g, P.h01 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1913_3994,(0,0,1):C.UVGC_1913_3995,(0,0,2):C.UVGC_1913_3996,(0,0,3):C.UVGC_1913_3997,(0,0,4):C.UVGC_1913_3998,(0,0,5):C.UVGC_1913_3999,(0,0,6):C.UVGC_1913_4000,(0,0,7):C.UVGC_1913_4001,(0,0,8):C.UVGC_1913_4002,(0,0,9):C.UVGC_1913_4003,(0,0,10):C.UVGC_1913_4004,(0,0,11):C.UVGC_1913_4005})

V_858 = CTVertex(name = 'V_858',
                 type = 'UV',
                 particles = [ P.g, P.g, P.h02 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1914_4006,(0,0,1):C.UVGC_1914_4007,(0,0,2):C.UVGC_1914_4008,(0,0,3):C.UVGC_1914_4009,(0,0,4):C.UVGC_1914_4010,(0,0,5):C.UVGC_1914_4011,(0,0,6):C.UVGC_1914_4012,(0,0,7):C.UVGC_1914_4013,(0,0,8):C.UVGC_1914_4014,(0,0,9):C.UVGC_1914_4015,(0,0,10):C.UVGC_1914_4016,(0,0,11):C.UVGC_1914_4017})

V_859 = CTVertex(name = 'V_859',
                 type = 'UV',
                 particles = [ P.a, P.a, P.g, P.g ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVVV2, L.VVVV3, L.VVVV5 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1186_114,(0,0,1):C.UVGC_1186_115,(0,0,2):C.UVGC_1186_116,(0,0,3):C.UVGC_1186_117,(0,0,4):C.UVGC_1186_118,(0,0,5):C.UVGC_1186_119,(0,0,6):C.UVGC_1186_120,(0,0,7):C.UVGC_1186_121,(0,0,8):C.UVGC_1186_122,(0,0,9):C.UVGC_1186_123,(0,0,10):C.UVGC_1186_124,(0,0,11):C.UVGC_1186_125,(0,1,0):C.UVGC_1186_114,(0,1,1):C.UVGC_1186_115,(0,1,2):C.UVGC_1186_116,(0,1,3):C.UVGC_1186_117,(0,1,4):C.UVGC_1186_118,(0,1,5):C.UVGC_1186_119,(0,1,6):C.UVGC_1186_120,(0,1,7):C.UVGC_1186_121,(0,1,8):C.UVGC_1186_122,(0,1,9):C.UVGC_1186_123,(0,1,10):C.UVGC_1186_124,(0,1,11):C.UVGC_1186_125,(0,2,0):C.UVGC_1186_114,(0,2,1):C.UVGC_1186_115,(0,2,2):C.UVGC_1186_116,(0,2,3):C.UVGC_1186_117,(0,2,4):C.UVGC_1186_118,(0,2,5):C.UVGC_1186_119,(0,2,6):C.UVGC_1186_120,(0,2,7):C.UVGC_1186_121,(0,2,8):C.UVGC_1186_122,(0,2,9):C.UVGC_1186_123,(0,2,10):C.UVGC_1186_124,(0,2,11):C.UVGC_1186_125})

V_860 = CTVertex(name = 'V_860',
                 type = 'UV',
                 particles = [ P.a, P.g, P.g, P.Z ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VVVV2, L.VVVV3, L.VVVV5 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1194_216,(0,0,1):C.UVGC_1194_217,(0,0,2):C.UVGC_1194_218,(0,0,3):C.UVGC_1194_219,(0,0,4):C.UVGC_1194_220,(0,0,5):C.UVGC_1194_221,(0,0,6):C.UVGC_1194_222,(0,0,7):C.UVGC_1194_223,(0,0,8):C.UVGC_1194_224,(0,0,9):C.UVGC_1194_225,(0,0,10):C.UVGC_1194_226,(0,0,11):C.UVGC_1194_227,(0,1,0):C.UVGC_1194_216,(0,1,1):C.UVGC_1194_217,(0,1,2):C.UVGC_1194_218,(0,1,3):C.UVGC_1194_219,(0,1,4):C.UVGC_1194_220,(0,1,5):C.UVGC_1194_221,(0,1,6):C.UVGC_1194_222,(0,1,7):C.UVGC_1194_223,(0,1,8):C.UVGC_1194_224,(0,1,9):C.UVGC_1194_225,(0,1,10):C.UVGC_1194_226,(0,1,11):C.UVGC_1194_227,(0,2,0):C.UVGC_1194_216,(0,2,1):C.UVGC_1194_217,(0,2,2):C.UVGC_1194_218,(0,2,3):C.UVGC_1194_219,(0,2,4):C.UVGC_1194_220,(0,2,5):C.UVGC_1194_221,(0,2,6):C.UVGC_1194_222,(0,2,7):C.UVGC_1194_223,(0,2,8):C.UVGC_1194_224,(0,2,9):C.UVGC_1194_225,(0,2,10):C.UVGC_1194_226,(0,2,11):C.UVGC_1194_227})

V_861 = CTVertex(name = 'V_861',
                 type = 'UV',
                 particles = [ P.g, P.g, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVV2, L.VVVV3, L.VVVV5 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1192_192,(0,0,1):C.UVGC_1192_193,(0,0,2):C.UVGC_1192_194,(0,0,3):C.UVGC_1192_195,(0,0,4):C.UVGC_1192_196,(0,0,5):C.UVGC_1192_197,(0,0,6):C.UVGC_1192_198,(0,0,7):C.UVGC_1192_199,(0,0,8):C.UVGC_1192_200,(0,0,9):C.UVGC_1192_201,(0,0,10):C.UVGC_1192_202,(0,0,11):C.UVGC_1192_203,(0,1,0):C.UVGC_1192_192,(0,1,1):C.UVGC_1192_193,(0,1,2):C.UVGC_1192_194,(0,1,3):C.UVGC_1192_195,(0,1,4):C.UVGC_1192_196,(0,1,5):C.UVGC_1192_197,(0,1,6):C.UVGC_1192_198,(0,1,7):C.UVGC_1192_199,(0,1,8):C.UVGC_1192_200,(0,1,9):C.UVGC_1192_201,(0,1,10):C.UVGC_1192_202,(0,1,11):C.UVGC_1192_203,(0,2,0):C.UVGC_1193_204,(0,2,1):C.UVGC_1193_205,(0,2,2):C.UVGC_1193_206,(0,2,3):C.UVGC_1193_207,(0,2,4):C.UVGC_1193_208,(0,2,5):C.UVGC_1193_209,(0,2,6):C.UVGC_1193_210,(0,2,7):C.UVGC_1193_211,(0,2,8):C.UVGC_1193_212,(0,2,9):C.UVGC_1193_213,(0,2,10):C.UVGC_1193_214,(0,2,11):C.UVGC_1193_215})

V_862 = CTVertex(name = 'V_862',
                 type = 'UV',
                 particles = [ P.g, P.g, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVV2, L.VVVV3, L.VVVV5 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd1, P.su1] ], [ [P.sd2] ], [ [P.sd2, P.su2] ], [ [P.sd3] ], [ [P.sd3, P.su3] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ] ],
                 couplings = {(0,0,1):C.UVGC_1222_449,(0,0,3):C.UVGC_1222_450,(0,0,5):C.UVGC_1222_451,(0,1,1):C.UVGC_1222_449,(0,1,3):C.UVGC_1222_450,(0,1,5):C.UVGC_1222_451,(0,2,0):C.UVGC_1221_440,(0,2,2):C.UVGC_1221_441,(0,2,4):C.UVGC_1221_442,(0,2,6):C.UVGC_1221_443,(0,2,7):C.UVGC_1221_444,(0,2,8):C.UVGC_1221_445,(0,2,1):C.UVGC_1221_446,(0,2,3):C.UVGC_1221_447,(0,2,5):C.UVGC_1221_448})

V_863 = CTVertex(name = 'V_863',
                 type = 'UV',
                 particles = [ P.a, P.g, P.g, P.g ],
                 color = [ 'd(2,3,4)' ],
                 lorentz = [ L.VVVV2, L.VVVV3, L.VVVV5 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1187_126,(0,0,1):C.UVGC_1187_127,(0,0,2):C.UVGC_1187_128,(0,0,3):C.UVGC_1187_129,(0,0,4):C.UVGC_1187_130,(0,0,5):C.UVGC_1187_131,(0,0,6):C.UVGC_1187_132,(0,0,7):C.UVGC_1187_133,(0,0,8):C.UVGC_1187_134,(0,0,9):C.UVGC_1187_135,(0,0,10):C.UVGC_1187_136,(0,0,11):C.UVGC_1187_137,(0,1,0):C.UVGC_1187_126,(0,1,1):C.UVGC_1187_127,(0,1,2):C.UVGC_1187_128,(0,1,3):C.UVGC_1187_129,(0,1,4):C.UVGC_1187_130,(0,1,5):C.UVGC_1187_131,(0,1,6):C.UVGC_1187_132,(0,1,7):C.UVGC_1187_133,(0,1,8):C.UVGC_1187_134,(0,1,9):C.UVGC_1187_135,(0,1,10):C.UVGC_1187_136,(0,1,11):C.UVGC_1187_137,(0,2,0):C.UVGC_1187_126,(0,2,1):C.UVGC_1187_127,(0,2,2):C.UVGC_1187_128,(0,2,3):C.UVGC_1187_129,(0,2,4):C.UVGC_1187_130,(0,2,5):C.UVGC_1187_131,(0,2,6):C.UVGC_1187_132,(0,2,7):C.UVGC_1187_133,(0,2,8):C.UVGC_1187_134,(0,2,9):C.UVGC_1187_135,(0,2,10):C.UVGC_1187_136,(0,2,11):C.UVGC_1187_137})

V_864 = CTVertex(name = 'V_864',
                 type = 'UV',
                 particles = [ P.g, P.g, P.g, P.Z ],
                 color = [ 'd(1,2,3)' ],
                 lorentz = [ L.VVVV2, L.VVVV3, L.VVVV5 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1195_228,(0,0,1):C.UVGC_1195_229,(0,0,2):C.UVGC_1195_230,(0,0,3):C.UVGC_1195_231,(0,0,4):C.UVGC_1195_232,(0,0,5):C.UVGC_1195_233,(0,0,6):C.UVGC_1195_234,(0,0,7):C.UVGC_1195_235,(0,0,8):C.UVGC_1195_236,(0,0,9):C.UVGC_1195_237,(0,0,10):C.UVGC_1195_238,(0,0,11):C.UVGC_1195_239,(0,1,0):C.UVGC_1195_228,(0,1,1):C.UVGC_1195_229,(0,1,2):C.UVGC_1195_230,(0,1,3):C.UVGC_1195_231,(0,1,4):C.UVGC_1195_232,(0,1,5):C.UVGC_1195_233,(0,1,6):C.UVGC_1195_234,(0,1,7):C.UVGC_1195_235,(0,1,8):C.UVGC_1195_236,(0,1,9):C.UVGC_1195_237,(0,1,10):C.UVGC_1195_238,(0,1,11):C.UVGC_1195_239,(0,2,0):C.UVGC_1195_228,(0,2,1):C.UVGC_1195_229,(0,2,2):C.UVGC_1195_230,(0,2,3):C.UVGC_1195_231,(0,2,4):C.UVGC_1195_232,(0,2,5):C.UVGC_1195_233,(0,2,6):C.UVGC_1195_234,(0,2,7):C.UVGC_1195_235,(0,2,8):C.UVGC_1195_236,(0,2,9):C.UVGC_1195_237,(0,2,10):C.UVGC_1195_238,(0,2,11):C.UVGC_1195_239})

V_865 = CTVertex(name = 'V_865',
                 type = 'UV',
                 particles = [ P.g, P.g, P.h01, P.h01 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1955_4176,(0,0,1):C.UVGC_1955_4177,(0,0,2):C.UVGC_1955_4178,(0,0,3):C.UVGC_1955_4179,(0,0,4):C.UVGC_1955_4180,(0,0,5):C.UVGC_1955_4181,(0,0,6):C.UVGC_1955_4182,(0,0,7):C.UVGC_1955_4183,(0,0,8):C.UVGC_1955_4184,(0,0,9):C.UVGC_1955_4185,(0,0,10):C.UVGC_1955_4186,(0,0,11):C.UVGC_1955_4187})

V_866 = CTVertex(name = 'V_866',
                 type = 'UV',
                 particles = [ P.g, P.g, P.h01, P.h02 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1969_4200,(0,0,1):C.UVGC_1969_4201,(0,0,2):C.UVGC_1969_4202,(0,0,3):C.UVGC_1969_4203,(0,0,4):C.UVGC_1969_4204,(0,0,5):C.UVGC_1969_4205,(0,0,6):C.UVGC_1969_4206,(0,0,7):C.UVGC_1969_4207,(0,0,8):C.UVGC_1969_4208,(0,0,9):C.UVGC_1969_4209,(0,0,10):C.UVGC_1969_4210,(0,0,11):C.UVGC_1969_4211})

V_867 = CTVertex(name = 'V_867',
                 type = 'UV',
                 particles = [ P.g, P.g, P.h02, P.h02 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1956_4188,(0,0,1):C.UVGC_1956_4189,(0,0,2):C.UVGC_1956_4190,(0,0,3):C.UVGC_1956_4191,(0,0,4):C.UVGC_1956_4192,(0,0,5):C.UVGC_1956_4193,(0,0,6):C.UVGC_1956_4194,(0,0,7):C.UVGC_1956_4195,(0,0,8):C.UVGC_1956_4196,(0,0,9):C.UVGC_1956_4197,(0,0,10):C.UVGC_1956_4198,(0,0,11):C.UVGC_1956_4199})

V_868 = CTVertex(name = 'V_868',
                 type = 'UV',
                 particles = [ P.g, P.g, P.A0, P.A0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2254_4534,(0,0,1):C.UVGC_2254_4535,(0,0,2):C.UVGC_2254_4536,(0,0,3):C.UVGC_2254_4537,(0,0,4):C.UVGC_2254_4538,(0,0,5):C.UVGC_2254_4539,(0,0,6):C.UVGC_2254_4540,(0,0,7):C.UVGC_2254_4541,(0,0,8):C.UVGC_2254_4542,(0,0,9):C.UVGC_2254_4543,(0,0,10):C.UVGC_2254_4544,(0,0,11):C.UVGC_2254_4545})

V_869 = CTVertex(name = 'V_869',
                 type = 'UV',
                 particles = [ P.g, P.g, P.H__minus__, P.H__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2255_4546,(0,0,1):C.UVGC_2255_4547,(0,0,2):C.UVGC_2255_4548,(0,0,3):C.UVGC_2254_4537,(0,0,4):C.UVGC_2254_4538,(0,0,5):C.UVGC_2254_4539,(0,0,6):C.UVGC_2255_4549,(0,0,7):C.UVGC_2255_4550,(0,0,8):C.UVGC_2255_4551,(0,0,9):C.UVGC_2254_4543,(0,0,10):C.UVGC_2254_4544,(0,0,11):C.UVGC_2254_4545})

V_870 = CTVertex(name = 'V_870',
                 type = 'UV',
                 particles = [ P.g, P.g, P.A0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2272_4582,(0,0,1):C.UVGC_2272_4583,(0,0,2):C.UVGC_2272_4584,(0,0,3):C.UVGC_2271_4573,(0,0,4):C.UVGC_2271_4574,(0,0,5):C.UVGC_2271_4575,(0,0,6):C.UVGC_2272_4585,(0,0,7):C.UVGC_2272_4586,(0,0,8):C.UVGC_2272_4587,(0,0,9):C.UVGC_2271_4579,(0,0,10):C.UVGC_2271_4580,(0,0,11):C.UVGC_2271_4581})

V_871 = CTVertex(name = 'V_871',
                 type = 'UV',
                 particles = [ P.g, P.g, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2257_4564,(0,0,1):C.UVGC_2257_4565,(0,0,2):C.UVGC_2257_4566,(0,0,3):C.UVGC_2256_4555,(0,0,4):C.UVGC_2256_4556,(0,0,5):C.UVGC_2256_4557,(0,0,6):C.UVGC_2257_4567,(0,0,7):C.UVGC_2257_4568,(0,0,8):C.UVGC_2257_4569,(0,0,9):C.UVGC_2256_4561,(0,0,10):C.UVGC_2256_4562,(0,0,11):C.UVGC_2256_4563})

V_872 = CTVertex(name = 'V_872',
                 type = 'UV',
                 particles = [ P.g, P.g, P.G__plus__, P.H__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2271_4570,(0,0,1):C.UVGC_2271_4571,(0,0,2):C.UVGC_2271_4572,(0,0,3):C.UVGC_2271_4573,(0,0,4):C.UVGC_2271_4574,(0,0,5):C.UVGC_2271_4575,(0,0,6):C.UVGC_2271_4576,(0,0,7):C.UVGC_2271_4577,(0,0,8):C.UVGC_2271_4578,(0,0,9):C.UVGC_2271_4579,(0,0,10):C.UVGC_2271_4580,(0,0,11):C.UVGC_2271_4581})

V_873 = CTVertex(name = 'V_873',
                 type = 'UV',
                 particles = [ P.g, P.g, P.G__minus__, P.H__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2271_4570,(0,0,1):C.UVGC_2271_4571,(0,0,2):C.UVGC_2271_4572,(0,0,3):C.UVGC_2271_4573,(0,0,4):C.UVGC_2271_4574,(0,0,5):C.UVGC_2271_4575,(0,0,6):C.UVGC_2271_4576,(0,0,7):C.UVGC_2271_4577,(0,0,8):C.UVGC_2271_4578,(0,0,9):C.UVGC_2271_4579,(0,0,10):C.UVGC_2271_4580,(0,0,11):C.UVGC_2271_4581})

V_874 = CTVertex(name = 'V_874',
                 type = 'UV',
                 particles = [ P.g, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_2256_4552,(0,0,1):C.UVGC_2256_4553,(0,0,2):C.UVGC_2256_4554,(0,0,3):C.UVGC_2256_4555,(0,0,4):C.UVGC_2256_4556,(0,0,5):C.UVGC_2256_4557,(0,0,6):C.UVGC_2256_4558,(0,0,7):C.UVGC_2256_4559,(0,0,8):C.UVGC_2256_4560,(0,0,9):C.UVGC_2256_4561,(0,0,10):C.UVGC_2256_4562,(0,0,11):C.UVGC_2256_4563})

V_875 = CTVertex(name = 'V_875',
                 type = 'UV',
                 particles = [ P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1697_3249,(0,1,0):C.UVGC_1696_3248})

V_876 = CTVertex(name = 'V_876',
                 type = 'UV',
                 particles = [ P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS3 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1698_3250,(0,1,0):C.UVGC_1696_3248})

V_877 = CTVertex(name = 'V_877',
                 type = 'UV',
                 particles = [ P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS2 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1364_1248,(0,1,0):C.UVGC_1363_1247})

V_878 = CTVertex(name = 'V_878',
                 type = 'UV',
                 particles = [ P.sd3, P.sd6__tilde__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.SS1, L.SS3 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1365_1249,(0,1,0):C.UVGC_1363_1247})

V_879 = CTVertex(name = 'V_879',
                 type = 'UV',
                 particles = [ P.a, P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1720_3340,(0,1,0):C.UVGC_1719_3339})

V_880 = CTVertex(name = 'V_880',
                 type = 'UV',
                 particles = [ P.Z, P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1728_3348,(0,1,0):C.UVGC_1727_3347})

V_881 = CTVertex(name = 'V_881',
                 type = 'UV',
                 particles = [ P.g, P.su3__tilde__, P.su6 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1723_3343,(0,1,0):C.UVGC_1722_3342})

V_882 = CTVertex(name = 'V_882',
                 type = 'UV',
                 particles = [ P.a, P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1719_3339,(0,1,0):C.UVGC_1720_3340})

V_883 = CTVertex(name = 'V_883',
                 type = 'UV',
                 particles = [ P.Z, P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1727_3347,(0,1,0):C.UVGC_1728_3348})

V_884 = CTVertex(name = 'V_884',
                 type = 'UV',
                 particles = [ P.g, P.su3, P.su6__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1722_3342,(0,1,0):C.UVGC_1723_3343})

V_885 = CTVertex(name = 'V_885',
                 type = 'UV',
                 particles = [ P.W__plus__, P.sd3, P.su6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1732_3352,(0,1,0):C.UVGC_1733_3353})

V_886 = CTVertex(name = 'V_886',
                 type = 'UV',
                 particles = [ P.W__minus__, P.sd3__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1714_3334,(0,1,0):C.UVGC_1713_3333})

V_887 = CTVertex(name = 'V_887',
                 type = 'UV',
                 particles = [ P.W__plus__, P.sd6, P.su3__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1401_1356,(0,1,0):C.UVGC_1402_1357})

V_888 = CTVertex(name = 'V_888',
                 type = 'UV',
                 particles = [ P.a, P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1372_1258,(0,1,0):C.UVGC_1373_1259})

V_889 = CTVertex(name = 'V_889',
                 type = 'UV',
                 particles = [ P.Z, P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1385_1271,(0,1,0):C.UVGC_1386_1272})

V_890 = CTVertex(name = 'V_890',
                 type = 'UV',
                 particles = [ P.g, P.sd3__tilde__, P.sd6 ],
                 color = [ 'T(1,3,2)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1376_1262,(0,1,0):C.UVGC_1375_1261})

V_891 = CTVertex(name = 'V_891',
                 type = 'UV',
                 particles = [ P.W__minus__, P.sd6__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1381_1267,(0,1,0):C.UVGC_1380_1266})

V_892 = CTVertex(name = 'V_892',
                 type = 'UV',
                 particles = [ P.a, P.sd3, P.sd6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1373_1259,(0,1,0):C.UVGC_1372_1258})

V_893 = CTVertex(name = 'V_893',
                 type = 'UV',
                 particles = [ P.Z, P.sd3, P.sd6__tilde__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1386_1272,(0,1,0):C.UVGC_1385_1271})

V_894 = CTVertex(name = 'V_894',
                 type = 'UV',
                 particles = [ P.g, P.sd3, P.sd6__tilde__ ],
                 color = [ 'T(1,2,3)' ],
                 lorentz = [ L.VSS1, L.VSS3 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1375_1261,(0,1,0):C.UVGC_1376_1262})

V_895 = CTVertex(name = 'V_895',
                 type = 'UV',
                 particles = [ P.A0, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2210_4361})

V_896 = CTVertex(name = 'V_896',
                 type = 'UV',
                 particles = [ P.G0, P.su3__tilde__, P.su3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2215_4402})

V_897 = CTVertex(name = 'V_897',
                 type = 'UV',
                 particles = [ P.A0, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2222_4461})

V_898 = CTVertex(name = 'V_898',
                 type = 'UV',
                 particles = [ P.G0, P.su6__tilde__, P.su6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2228_4517})

V_899 = CTVertex(name = 'V_899',
                 type = 'UV',
                 particles = [ P.A0, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_2186_4213})

V_900 = CTVertex(name = 'V_900',
                 type = 'UV',
                 particles = [ P.G0, P.sd3__tilde__, P.sd3 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_2185_4212})

V_901 = CTVertex(name = 'V_901',
                 type = 'UV',
                 particles = [ P.A0, P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_2189_4222})

V_902 = CTVertex(name = 'V_902',
                 type = 'UV',
                 particles = [ P.G0, P.sd6__tilde__, P.sd6 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.SSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_2188_4221})

V_903 = CTVertex(name = 'V_903',
                 type = 'UV',
                 particles = [ P.a, P.a, P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1721_3341})

V_904 = CTVertex(name = 'V_904',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1729_3349})

V_905 = CTVertex(name = 'V_905',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1731_3351})

V_906 = CTVertex(name = 'V_906',
                 type = 'UV',
                 particles = [ P.W__minus__, P.W__plus__, P.su3__tilde__, P.su6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1726_3346})

V_907 = CTVertex(name = 'V_907',
                 type = 'UV',
                 particles = [ P.a, P.g, P.su3__tilde__, P.su6 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1724_3344})

V_908 = CTVertex(name = 'V_908',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.su3__tilde__, P.su6 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1730_3350})

V_909 = CTVertex(name = 'V_909',
                 type = 'UV',
                 particles = [ P.g, P.g, P.su3__tilde__, P.su6 ],
                 color = [ 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(1,0,0):C.UVGC_1725_3345,(0,0,0):C.UVGC_1725_3345})

V_910 = CTVertex(name = 'V_910',
                 type = 'UV',
                 particles = [ P.a, P.a, P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1721_3341})

V_911 = CTVertex(name = 'V_911',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1729_3349})

V_912 = CTVertex(name = 'V_912',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1731_3351})

V_913 = CTVertex(name = 'V_913',
                 type = 'UV',
                 particles = [ P.W__minus__, P.W__plus__, P.su3, P.su6__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1726_3346})

V_914 = CTVertex(name = 'V_914',
                 type = 'UV',
                 particles = [ P.a, P.g, P.su3, P.su6__tilde__ ],
                 color = [ 'T(2,3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1724_3344})

V_915 = CTVertex(name = 'V_915',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.su3, P.su6__tilde__ ],
                 color = [ 'T(1,3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1730_3350})

V_916 = CTVertex(name = 'V_916',
                 type = 'UV',
                 particles = [ P.g, P.g, P.su3, P.su6__tilde__ ],
                 color = [ 'T(1,-1,4)*T(2,3,-1)', 'T(1,3,-1)*T(2,-1,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(1,0,0):C.UVGC_1725_3345,(0,0,0):C.UVGC_1725_3345})

V_917 = CTVertex(name = 'V_917',
                 type = 'UV',
                 particles = [ P.a, P.W__plus__, P.sd3, P.su6__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1734_3354})

V_918 = CTVertex(name = 'V_918',
                 type = 'UV',
                 particles = [ P.W__plus__, P.Z, P.sd3, P.su6__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1736_3356})

V_919 = CTVertex(name = 'V_919',
                 type = 'UV',
                 particles = [ P.g, P.W__plus__, P.sd3, P.su6__tilde__ ],
                 color = [ 'T(1,3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1735_3355})

V_920 = CTVertex(name = 'V_920',
                 type = 'UV',
                 particles = [ P.a, P.W__minus__, P.sd3__tilde__, P.su6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1715_3335})

V_921 = CTVertex(name = 'V_921',
                 type = 'UV',
                 particles = [ P.W__minus__, P.Z, P.sd3__tilde__, P.su6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1717_3337})

V_922 = CTVertex(name = 'V_922',
                 type = 'UV',
                 particles = [ P.g, P.W__minus__, P.sd3__tilde__, P.su6 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.go, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1716_3336})

V_923 = CTVertex(name = 'V_923',
                 type = 'UV',
                 particles = [ P.a, P.W__plus__, P.sd6, P.su3__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1403_1358})

V_924 = CTVertex(name = 'V_924',
                 type = 'UV',
                 particles = [ P.W__plus__, P.Z, P.sd6, P.su3__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1405_1360})

V_925 = CTVertex(name = 'V_925',
                 type = 'UV',
                 particles = [ P.g, P.W__plus__, P.sd6, P.su3__tilde__ ],
                 color = [ 'T(1,3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1404_1359})

V_926 = CTVertex(name = 'V_926',
                 type = 'UV',
                 particles = [ P.a, P.a, P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1374_1260})

V_927 = CTVertex(name = 'V_927',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1387_1273})

V_928 = CTVertex(name = 'V_928',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1389_1275})

V_929 = CTVertex(name = 'V_929',
                 type = 'UV',
                 particles = [ P.W__minus__, P.W__plus__, P.sd3__tilde__, P.sd6 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1379_1265})

V_930 = CTVertex(name = 'V_930',
                 type = 'UV',
                 particles = [ P.a, P.g, P.sd3__tilde__, P.sd6 ],
                 color = [ 'T(2,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1377_1263})

V_931 = CTVertex(name = 'V_931',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.sd3__tilde__, P.sd6 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1388_1274})

V_932 = CTVertex(name = 'V_932',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sd3__tilde__, P.sd6 ],
                 color = [ 'T(1,-1,3)*T(2,4,-1)', 'T(1,4,-1)*T(2,-1,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(1,0,0):C.UVGC_1378_1264,(0,0,0):C.UVGC_1378_1264})

V_933 = CTVertex(name = 'V_933',
                 type = 'UV',
                 particles = [ P.a, P.W__minus__, P.sd6__tilde__, P.su3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1382_1268})

V_934 = CTVertex(name = 'V_934',
                 type = 'UV',
                 particles = [ P.W__minus__, P.Z, P.sd6__tilde__, P.su3 ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1384_1270})

V_935 = CTVertex(name = 'V_935',
                 type = 'UV',
                 particles = [ P.g, P.W__minus__, P.sd6__tilde__, P.su3 ],
                 color = [ 'T(1,4,3)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1383_1269})

V_936 = CTVertex(name = 'V_936',
                 type = 'UV',
                 particles = [ P.a, P.a, P.sd3, P.sd6__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1374_1260})

V_937 = CTVertex(name = 'V_937',
                 type = 'UV',
                 particles = [ P.a, P.Z, P.sd3, P.sd6__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1387_1273})

V_938 = CTVertex(name = 'V_938',
                 type = 'UV',
                 particles = [ P.Z, P.Z, P.sd3, P.sd6__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1389_1275})

V_939 = CTVertex(name = 'V_939',
                 type = 'UV',
                 particles = [ P.W__minus__, P.W__plus__, P.sd3, P.sd6__tilde__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1379_1265})

V_940 = CTVertex(name = 'V_940',
                 type = 'UV',
                 particles = [ P.a, P.g, P.sd3, P.sd6__tilde__ ],
                 color = [ 'T(2,3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1377_1263})

V_941 = CTVertex(name = 'V_941',
                 type = 'UV',
                 particles = [ P.g, P.Z, P.sd3, P.sd6__tilde__ ],
                 color = [ 'T(1,3,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(0,0,0):C.UVGC_1388_1274})

V_942 = CTVertex(name = 'V_942',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sd3, P.sd6__tilde__ ],
                 color = [ 'T(1,-1,4)*T(2,3,-1)', 'T(1,3,-1)*T(2,-1,4)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.b, P.go] ] ],
                 couplings = {(1,0,0):C.UVGC_1378_1264,(0,0,0):C.UVGC_1378_1264})

V_943 = CTVertex(name = 'V_943',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sv1__tilde__, P.sv1 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1179_30,(0,0,1):C.UVGC_1179_31,(0,0,2):C.UVGC_1179_32,(0,0,3):C.UVGC_1179_33,(0,0,4):C.UVGC_1179_34,(0,0,5):C.UVGC_1179_35,(0,0,6):C.UVGC_1179_36,(0,0,7):C.UVGC_1179_37,(0,0,8):C.UVGC_1179_38,(0,0,9):C.UVGC_1179_39,(0,0,10):C.UVGC_1179_40,(0,0,11):C.UVGC_1179_41})

V_944 = CTVertex(name = 'V_944',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sv2__tilde__, P.sv2 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1179_30,(0,0,1):C.UVGC_1179_31,(0,0,2):C.UVGC_1179_32,(0,0,3):C.UVGC_1179_33,(0,0,4):C.UVGC_1179_34,(0,0,5):C.UVGC_1179_35,(0,0,6):C.UVGC_1179_36,(0,0,7):C.UVGC_1179_37,(0,0,8):C.UVGC_1179_38,(0,0,9):C.UVGC_1179_39,(0,0,10):C.UVGC_1179_40,(0,0,11):C.UVGC_1179_41})

V_945 = CTVertex(name = 'V_945',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sv3__tilde__, P.sv3 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1179_30,(0,0,1):C.UVGC_1179_31,(0,0,2):C.UVGC_1179_32,(0,0,3):C.UVGC_1179_33,(0,0,4):C.UVGC_1179_34,(0,0,5):C.UVGC_1179_35,(0,0,6):C.UVGC_1179_36,(0,0,7):C.UVGC_1179_37,(0,0,8):C.UVGC_1179_38,(0,0,9):C.UVGC_1179_39,(0,0,10):C.UVGC_1179_40,(0,0,11):C.UVGC_1179_41})

V_946 = CTVertex(name = 'V_946',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sl1__plus__, P.sl1__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1180_42,(0,0,1):C.UVGC_1180_43,(0,0,2):C.UVGC_1180_44,(0,0,3):C.UVGC_1180_45,(0,0,4):C.UVGC_1180_46,(0,0,5):C.UVGC_1180_47,(0,0,6):C.UVGC_1180_48,(0,0,7):C.UVGC_1180_49,(0,0,8):C.UVGC_1180_50,(0,0,9):C.UVGC_1180_51,(0,0,10):C.UVGC_1180_52,(0,0,11):C.UVGC_1180_53})

V_947 = CTVertex(name = 'V_947',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sl2__plus__, P.sl2__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1181_54,(0,0,1):C.UVGC_1181_55,(0,0,2):C.UVGC_1181_56,(0,0,3):C.UVGC_1181_57,(0,0,4):C.UVGC_1181_58,(0,0,5):C.UVGC_1181_59,(0,0,6):C.UVGC_1181_60,(0,0,7):C.UVGC_1181_61,(0,0,8):C.UVGC_1181_62,(0,0,9):C.UVGC_1181_63,(0,0,10):C.UVGC_1181_64,(0,0,11):C.UVGC_1181_65})

V_948 = CTVertex(name = 'V_948',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sl3__plus__, P.sl3__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1182_66,(0,0,1):C.UVGC_1182_67,(0,0,2):C.UVGC_1182_68,(0,0,3):C.UVGC_1182_69,(0,0,4):C.UVGC_1182_70,(0,0,5):C.UVGC_1182_71,(0,0,6):C.UVGC_1182_72,(0,0,7):C.UVGC_1182_73,(0,0,8):C.UVGC_1182_74,(0,0,9):C.UVGC_1182_75,(0,0,10):C.UVGC_1182_76,(0,0,11):C.UVGC_1182_77})

V_949 = CTVertex(name = 'V_949',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sl4__plus__, P.sl4__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1183_78,(0,0,1):C.UVGC_1183_79,(0,0,2):C.UVGC_1183_80,(0,0,3):C.UVGC_1183_81,(0,0,4):C.UVGC_1183_82,(0,0,5):C.UVGC_1183_83,(0,0,6):C.UVGC_1183_84,(0,0,7):C.UVGC_1183_85,(0,0,8):C.UVGC_1183_86,(0,0,9):C.UVGC_1183_87,(0,0,10):C.UVGC_1183_88,(0,0,11):C.UVGC_1183_89})

V_950 = CTVertex(name = 'V_950',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sl5__plus__, P.sl5__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1184_90,(0,0,1):C.UVGC_1184_91,(0,0,2):C.UVGC_1184_92,(0,0,3):C.UVGC_1184_93,(0,0,4):C.UVGC_1184_94,(0,0,5):C.UVGC_1184_95,(0,0,6):C.UVGC_1184_96,(0,0,7):C.UVGC_1184_97,(0,0,8):C.UVGC_1184_98,(0,0,9):C.UVGC_1184_99,(0,0,10):C.UVGC_1184_100,(0,0,11):C.UVGC_1184_101})

V_951 = CTVertex(name = 'V_951',
                 type = 'UV',
                 particles = [ P.g, P.g, P.sl6__plus__, P.sl6__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS1 ],
                 loop_particles = [ [ [P.sd1] ], [ [P.sd2] ], [ [P.sd3] ], [ [P.sd4] ], [ [P.sd5] ], [ [P.sd6] ], [ [P.su1] ], [ [P.su2] ], [ [P.su3] ], [ [P.su4] ], [ [P.su5] ], [ [P.su6] ] ],
                 couplings = {(0,0,0):C.UVGC_1185_102,(0,0,1):C.UVGC_1185_103,(0,0,2):C.UVGC_1185_104,(0,0,3):C.UVGC_1185_105,(0,0,4):C.UVGC_1185_106,(0,0,5):C.UVGC_1185_107,(0,0,6):C.UVGC_1185_108,(0,0,7):C.UVGC_1185_109,(0,0,8):C.UVGC_1185_110,(0,0,9):C.UVGC_1185_111,(0,0,10):C.UVGC_1185_112,(0,0,11):C.UVGC_1185_113})

