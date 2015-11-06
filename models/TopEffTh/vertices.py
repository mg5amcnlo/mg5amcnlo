# This file was automatically created by FeynRules 2.4.1
# Mathematica version: 10.1.0  for Mac OS X x86 (64-bit) (March 24, 2015)
# Date: Thu 18 Jun 2015 16:26:34


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_9})

V_2 = Vertex(name = 'V_2',
             particles = [ P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_76})

V_3 = Vertex(name = 'V_3',
             particles = [ P.g, P.g, P.G0, P.G0 ],
             color = [ 'Identity(1,2)' ],
             lorentz = [ L.VVSS2 ],
             couplings = {(0,0):C.GC_29})

V_4 = Vertex(name = 'V_4',
             particles = [ P.g, P.g, P.G__minus__, P.G__plus__ ],
             color = [ 'Identity(1,2)' ],
             lorentz = [ L.VVSS2 ],
             couplings = {(0,0):C.GC_29})

V_5 = Vertex(name = 'V_5',
             particles = [ P.g, P.g, P.H, P.H ],
             color = [ 'Identity(1,2)' ],
             lorentz = [ L.VVSS2 ],
             couplings = {(0,0):C.GC_29})

V_6 = Vertex(name = 'V_6',
             particles = [ P.g, P.g, P.H ],
             color = [ 'Identity(1,2)' ],
             lorentz = [ L.VVS2 ],
             couplings = {(0,0):C.GC_78})

V_7 = Vertex(name = 'V_7',
             particles = [ P.g, P.g, P.g ],
             color = [ 'f(1,2,3)' ],
             lorentz = [ L.VVV1, L.VVV2 ],
             couplings = {(0,1):C.GC_28,(0,0):C.GC_6})

V_8 = Vertex(name = 'V_8',
             particles = [ P.ghG, P.ghG__tilde__, P.g ],
             color = [ 'f(1,2,3)' ],
             lorentz = [ L.UUV1 ],
             couplings = {(0,0):C.GC_6})

V_9 = Vertex(name = 'V_9',
             particles = [ P.g, P.g, P.g, P.g ],
             color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
             lorentz = [ L.VVVV1, L.VVVV2, L.VVVV4, L.VVVV5, L.VVVV7, L.VVVV8 ],
             couplings = {(0,1):C.GC_44,(1,5):C.GC_44,(2,4):C.GC_44,(1,2):C.GC_8,(0,0):C.GC_8,(2,3):C.GC_8})

V_10 = Vertex(name = 'V_10',
              particles = [ P.g, P.g, P.g, P.G0, P.G0 ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVVSS1 ],
              couplings = {(0,0):C.GC_45})

V_11 = Vertex(name = 'V_11',
              particles = [ P.g, P.g, P.g, P.G__minus__, P.G__plus__ ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVVSS1 ],
              couplings = {(0,0):C.GC_45})

V_12 = Vertex(name = 'V_12',
              particles = [ P.g, P.g, P.g, P.H, P.H ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVVSS1 ],
              couplings = {(0,0):C.GC_45})

V_13 = Vertex(name = 'V_13',
              particles = [ P.g, P.g, P.g, P.H ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVVS1 ],
              couplings = {(0,0):C.GC_84})

V_14 = Vertex(name = 'V_14',
              particles = [ P.g, P.g, P.g, P.g, P.G0, P.G0 ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVVSS1, L.VVVVSS2, L.VVVVSS3 ],
              couplings = {(1,1):C.GC_51,(0,0):C.GC_51,(2,2):C.GC_51})

V_15 = Vertex(name = 'V_15',
              particles = [ P.g, P.g, P.g, P.g, P.G__minus__, P.G__plus__ ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVVSS1, L.VVVVSS2, L.VVVVSS3 ],
              couplings = {(1,1):C.GC_51,(0,0):C.GC_51,(2,2):C.GC_51})

V_16 = Vertex(name = 'V_16',
              particles = [ P.g, P.g, P.g, P.g, P.H, P.H ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVVSS1, L.VVVVSS2, L.VVVVSS3 ],
              couplings = {(1,1):C.GC_51,(0,0):C.GC_51,(2,2):C.GC_51})

V_17 = Vertex(name = 'V_17',
              particles = [ P.g, P.g, P.g, P.g, P.H ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVVS1, L.VVVVS2, L.VVVVS3 ],
              couplings = {(1,1):C.GC_86,(0,0):C.GC_86,(2,2):C.GC_86})

V_18 = Vertex(name = 'V_18',
              particles = [ P.g, P.g, P.g, P.g, P.g ],
              color = [ 'f(-2,1,2)*f(-1,-2,3)*f(4,5,-1)', 'f(-2,1,2)*f(-1,-2,4)*f(3,5,-1)', 'f(-2,1,2)*f(-1,-2,5)*f(3,4,-1)', 'f(-2,1,3)*f(-1,-2,2)*f(4,5,-1)', 'f(-2,1,3)*f(-1,-2,4)*f(2,5,-1)', 'f(-2,1,3)*f(-1,-2,5)*f(2,4,-1)', 'f(-2,1,4)*f(-1,-2,2)*f(3,5,-1)', 'f(-2,1,4)*f(-1,-2,3)*f(2,5,-1)', 'f(-2,1,4)*f(-1,-2,5)*f(2,3,-1)', 'f(-2,1,5)*f(-1,-2,2)*f(3,4,-1)', 'f(-2,1,5)*f(-1,-2,3)*f(2,4,-1)', 'f(-2,1,5)*f(-1,-2,4)*f(2,3,-1)', 'f(-2,2,3)*f(-1,-2,1)*f(4,5,-1)', 'f(-2,2,3)*f(-1,-2,4)*f(1,5,-1)', 'f(-2,2,3)*f(-1,-2,5)*f(1,4,-1)', 'f(-2,2,4)*f(-1,-2,1)*f(3,5,-1)', 'f(-2,2,4)*f(-1,-2,3)*f(1,5,-1)', 'f(-2,2,4)*f(-1,-2,5)*f(1,3,-1)', 'f(-2,2,5)*f(-1,-2,1)*f(3,4,-1)', 'f(-2,2,5)*f(-1,-2,3)*f(1,4,-1)', 'f(-2,2,5)*f(-1,-2,4)*f(1,3,-1)', 'f(-2,3,4)*f(-1,-2,1)*f(2,5,-1)', 'f(-2,3,4)*f(-1,-2,2)*f(1,5,-1)', 'f(-2,3,4)*f(-1,-2,5)*f(1,2,-1)', 'f(-2,3,5)*f(-1,-2,1)*f(2,4,-1)', 'f(-2,3,5)*f(-1,-2,2)*f(1,4,-1)', 'f(-2,3,5)*f(-1,-2,4)*f(1,2,-1)', 'f(-2,4,5)*f(-1,-2,1)*f(2,3,-1)', 'f(-2,4,5)*f(-1,-2,2)*f(1,3,-1)', 'f(-2,4,5)*f(-1,-2,3)*f(1,2,-1)' ],
              lorentz = [ L.VVVVV1, L.VVVVV10, L.VVVVV11, L.VVVVV12, L.VVVVV13, L.VVVVV14, L.VVVVV15, L.VVVVV2, L.VVVVV3, L.VVVVV4, L.VVVVV5, L.VVVVV6, L.VVVVV7, L.VVVVV8, L.VVVVV9 ],
              couplings = {(24,9):C.GC_50,(21,10):C.GC_49,(18,10):C.GC_50,(15,9):C.GC_49,(28,7):C.GC_50,(22,14):C.GC_50,(9,14):C.GC_49,(3,7):C.GC_49,(29,8):C.GC_50,(16,1):C.GC_50,(10,1):C.GC_49,(0,8):C.GC_49,(26,4):C.GC_49,(20,3):C.GC_49,(4,3):C.GC_50,(1,4):C.GC_50,(25,13):C.GC_50,(6,13):C.GC_49,(19,2):C.GC_50,(7,2):C.GC_49,(23,6):C.GC_49,(17,5):C.GC_49,(5,5):C.GC_50,(2,6):C.GC_50,(27,0):C.GC_50,(12,0):C.GC_49,(13,11):C.GC_50,(11,11):C.GC_49,(14,12):C.GC_49,(8,12):C.GC_50})

V_19 = Vertex(name = 'V_19',
              particles = [ P.g, P.g, P.g, P.g, P.g, P.g ],
              color = [ 'f(-3,1,2)*f(-2,3,4)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,1,2)*f(-2,3,5)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,1,2)*f(-2,3,6)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,1,2)*f(-2,4,5)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,1,2)*f(-2,4,6)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,1,2)*f(-2,5,6)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,1,3)*f(-2,2,4)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,1,3)*f(-2,2,5)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,1,3)*f(-2,2,6)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,1,3)*f(-2,4,5)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,1,3)*f(-2,4,6)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,1,3)*f(-2,5,6)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,1,4)*f(-2,2,3)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,1,4)*f(-2,2,5)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,1,4)*f(-2,2,6)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,1,4)*f(-2,3,5)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,1,4)*f(-2,3,6)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,1,4)*f(-2,5,6)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,1,5)*f(-2,2,3)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,1,5)*f(-2,2,4)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,1,5)*f(-2,2,6)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,1,5)*f(-2,3,4)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,1,5)*f(-2,3,6)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,1,5)*f(-2,4,6)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,1,6)*f(-2,2,3)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,1,6)*f(-2,2,4)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,1,6)*f(-2,2,5)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,1,6)*f(-2,3,4)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,1,6)*f(-2,3,5)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,1,6)*f(-2,4,5)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,2,3)*f(-2,1,4)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,2,3)*f(-2,1,5)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,2,3)*f(-2,1,6)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,2,3)*f(-2,4,5)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,2,3)*f(-2,4,6)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,2,3)*f(-2,5,6)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,2,4)*f(-2,1,3)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,2,4)*f(-2,1,5)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,2,4)*f(-2,1,6)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,2,4)*f(-2,3,5)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,2,4)*f(-2,3,6)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,2,4)*f(-2,5,6)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,2,5)*f(-2,1,3)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,2,5)*f(-2,1,4)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,2,5)*f(-2,1,6)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,2,5)*f(-2,3,4)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,2,5)*f(-2,3,6)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,2,5)*f(-2,4,6)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,2,6)*f(-2,1,3)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,2,6)*f(-2,1,4)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,2,6)*f(-2,1,5)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,2,6)*f(-2,3,4)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,2,6)*f(-2,3,5)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,2,6)*f(-2,4,5)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,3,4)*f(-2,1,2)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,3,4)*f(-2,1,5)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,3,4)*f(-2,1,6)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,3,4)*f(-2,2,5)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,3,4)*f(-2,2,6)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,3,4)*f(-2,5,6)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,3,5)*f(-2,1,2)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,3,5)*f(-2,1,4)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,3,5)*f(-2,1,6)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,3,5)*f(-2,2,4)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,3,5)*f(-2,2,6)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,3,5)*f(-2,4,6)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,3,6)*f(-2,1,2)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,3,6)*f(-2,1,4)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,3,6)*f(-2,1,5)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,3,6)*f(-2,2,4)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,3,6)*f(-2,2,5)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,3,6)*f(-2,4,5)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,4,5)*f(-2,1,2)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,4,5)*f(-2,1,3)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,4,5)*f(-2,1,6)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,4,5)*f(-2,2,3)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,4,5)*f(-2,2,6)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,4,5)*f(-2,3,6)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,4,6)*f(-2,1,2)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,4,6)*f(-2,1,3)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,4,6)*f(-2,1,5)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,4,6)*f(-2,2,3)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,4,6)*f(-2,2,5)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,4,6)*f(-2,3,5)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,5,6)*f(-2,1,2)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,5,6)*f(-2,1,3)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,5,6)*f(-2,1,4)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,5,6)*f(-2,2,3)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,5,6)*f(-2,2,4)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,5,6)*f(-2,3,4)*f(-1,-2,-3)*f(1,2,-1)' ],
              lorentz = [ L.VVVVVV1, L.VVVVVV10, L.VVVVVV11, L.VVVVVV12, L.VVVVVV13, L.VVVVVV14, L.VVVVVV15, L.VVVVVV2, L.VVVVVV3, L.VVVVVV4, L.VVVVVV5, L.VVVVVV6, L.VVVVVV7, L.VVVVVV8, L.VVVVVV9 ],
              couplings = {(65,9):C.GC_53,(71,11):C.GC_52,(77,11):C.GC_53,(83,9):C.GC_52,(41,7):C.GC_53,(53,1):C.GC_53,(76,1):C.GC_52,(88,7):C.GC_52,(35,8):C.GC_53,(52,4):C.GC_53,(64,4):C.GC_52,(87,8):C.GC_52,(34,3):C.GC_52,(40,2):C.GC_52,(69,2):C.GC_53,(81,3):C.GC_53,(17,8):C.GC_52,(23,3):C.GC_53,(80,3):C.GC_52,(86,8):C.GC_53,(11,7):C.GC_52,(22,2):C.GC_53,(68,2):C.GC_52,(85,7):C.GC_53,(9,1):C.GC_52,(15,4):C.GC_52,(61,4):C.GC_53,(73,1):C.GC_53,(4,9):C.GC_52,(14,4):C.GC_53,(49,4):C.GC_52,(78,9):C.GC_53,(3,11):C.GC_53,(19,2):C.GC_52,(37,2):C.GC_53,(72,11):C.GC_52,(2,11):C.GC_52,(8,1):C.GC_53,(48,1):C.GC_52,(66,11):C.GC_53,(1,9):C.GC_53,(18,3):C.GC_52,(31,3):C.GC_53,(60,9):C.GC_52,(6,7):C.GC_53,(12,8):C.GC_53,(30,8):C.GC_52,(36,7):C.GC_52,(47,13):C.GC_53,(82,13):C.GC_52,(46,5):C.GC_53,(70,5):C.GC_52,(33,6):C.GC_52,(39,14):C.GC_52,(63,14):C.GC_53,(75,6):C.GC_53,(29,6):C.GC_53,(74,6):C.GC_52,(28,14):C.GC_53,(62,14):C.GC_52,(10,13):C.GC_52,(16,5):C.GC_52,(67,5):C.GC_53,(79,13):C.GC_53,(25,14):C.GC_52,(38,14):C.GC_53,(13,5):C.GC_53,(43,5):C.GC_52,(24,6):C.GC_52,(32,6):C.GC_53,(7,13):C.GC_53,(42,13):C.GC_52,(59,0):C.GC_53,(89,0):C.GC_52,(51,10):C.GC_53,(58,10):C.GC_52,(21,10):C.GC_52,(55,10):C.GC_53,(5,0):C.GC_52,(20,10):C.GC_53,(50,10):C.GC_52,(84,0):C.GC_53,(0,0):C.GC_53,(54,0):C.GC_52,(45,12):C.GC_52,(57,12):C.GC_53,(27,12):C.GC_53,(56,12):C.GC_52,(26,12):C.GC_52,(44,12):C.GC_53})

V_20 = Vertex(name = 'V_20',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_129,(0,1):C.GC_94})

V_21 = Vertex(name = 'V_21',
              particles = [ P.ta__plus__, P.ta__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_96})

V_22 = Vertex(name = 'V_22',
              particles = [ P.t__tilde__, P.t, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_128,(0,1):C.GC_95})

V_23 = Vertex(name = 'V_23',
              particles = [ P.d__tilde__, P.t, P.t__tilde__, P.d ],
              color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)', 'T(-1,2,3)*T(-1,4,1)' ],
              lorentz = [ L.FFFF1, L.FFFF2, L.FFFF4, L.FFFF5, L.FFFF6 ],
              couplings = {(2,0):C.GC_24,(1,0):C.GC_12,(2,1):C.GC_23,(2,4):C.GC_26,(0,2):C.GC_15,(0,3):C.GC_16})

V_24 = Vertex(name = 'V_24',
              particles = [ P.s__tilde__, P.t, P.t__tilde__, P.s ],
              color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)', 'T(-1,2,3)*T(-1,4,1)' ],
              lorentz = [ L.FFFF1, L.FFFF4, L.FFFF5, L.FFFF6 ],
              couplings = {(1,0):C.GC_12,(2,0):C.GC_10,(2,3):C.GC_26,(0,1):C.GC_15,(0,2):C.GC_16})

V_25 = Vertex(name = 'V_25',
              particles = [ P.t__tilde__, P.u, P.u__tilde__, P.t ],
              color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)', 'T(-1,2,3)*T(-1,4,1)' ],
              lorentz = [ L.FFFF1, L.FFFF4, L.FFFF5, L.FFFF6 ],
              couplings = {(1,0):C.GC_13,(2,0):C.GC_11,(2,3):C.GC_27,(0,1):C.GC_16,(0,2):C.GC_17})

V_26 = Vertex(name = 'V_26',
              particles = [ P.c__tilde__, P.t, P.t__tilde__, P.c ],
              color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)', 'T(-1,2,3)*T(-1,4,1)' ],
              lorentz = [ L.FFFF3, L.FFFF4, L.FFFF5, L.FFFF7 ],
              couplings = {(1,0):C.GC_13,(2,0):C.GC_11,(2,3):C.GC_27,(0,1):C.GC_17,(0,2):C.GC_16})

V_27 = Vertex(name = 'V_27',
              particles = [ P.b__tilde__, P.d, P.d__tilde__, P.b ],
              color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)', 'T(-1,2,3)*T(-1,4,1)' ],
              lorentz = [ L.FFFF1, L.FFFF5 ],
              couplings = {(1,0):C.GC_13,(2,0):C.GC_11,(0,1):C.GC_15})

V_28 = Vertex(name = 'V_28',
              particles = [ P.b__tilde__, P.s, P.s__tilde__, P.b ],
              color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)', 'T(-1,2,3)*T(-1,4,1)' ],
              lorentz = [ L.FFFF1, L.FFFF5 ],
              couplings = {(1,0):C.GC_13,(2,0):C.GC_11,(0,1):C.GC_15})

V_29 = Vertex(name = 'V_29',
              particles = [ P.b__tilde__, P.u, P.u__tilde__, P.b ],
              color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)', 'T(-1,2,3)*T(-1,4,1)' ],
              lorentz = [ L.FFFF1, L.FFFF5 ],
              couplings = {(1,0):C.GC_12,(2,0):C.GC_10,(0,1):C.GC_17})

V_30 = Vertex(name = 'V_30',
              particles = [ P.b__tilde__, P.c, P.c__tilde__, P.b ],
              color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)', 'T(-1,2,3)*T(-1,4,1)' ],
              lorentz = [ L.FFFF1, L.FFFF5 ],
              couplings = {(1,0):C.GC_12,(2,0):C.GC_10,(0,1):C.GC_17})

V_31 = Vertex(name = 'V_31',
              particles = [ P.a, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVV1 ],
              couplings = {(0,0):C.GC_4})

V_32 = Vertex(name = 'V_32',
              particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_54})

V_33 = Vertex(name = 'V_33',
              particles = [ P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_87})

V_34 = Vertex(name = 'V_34',
              particles = [ P.a, P.a, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV3 ],
              couplings = {(0,0):C.GC_5})

V_35 = Vertex(name = 'V_35',
              particles = [ P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVV1 ],
              couplings = {(0,0):C.GC_60})

V_36 = Vertex(name = 'V_36',
              particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV3 ],
              couplings = {(0,0):C.GC_55})

V_37 = Vertex(name = 'V_37',
              particles = [ P.a, P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV6 ],
              couplings = {(0,0):C.GC_61})

V_38 = Vertex(name = 'V_38',
              particles = [ P.Z, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_75})

V_39 = Vertex(name = 'V_39',
              particles = [ P.Z, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_93})

V_40 = Vertex(name = 'V_40',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV3 ],
              couplings = {(0,0):C.GC_56})

V_41 = Vertex(name = 'V_41',
              particles = [ P.e__plus__, P.ve, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_57})

V_42 = Vertex(name = 'V_42',
              particles = [ P.mu__plus__, P.vm, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_57})

V_43 = Vertex(name = 'V_43',
              particles = [ P.ta__plus__, P.vt, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_57})

V_44 = Vertex(name = 'V_44',
              particles = [ P.ve__tilde__, P.ve, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_74})

V_45 = Vertex(name = 'V_45',
              particles = [ P.vm__tilde__, P.vm, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_74})

V_46 = Vertex(name = 'V_46',
              particles = [ P.vt__tilde__, P.vt, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_74})

V_47 = Vertex(name = 'V_47',
              particles = [ P.e__plus__, P.e__minus__, P.a ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_48 = Vertex(name = 'V_48',
              particles = [ P.mu__plus__, P.mu__minus__, P.a ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_49 = Vertex(name = 'V_49',
              particles = [ P.ta__plus__, P.ta__minus__, P.a ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_50 = Vertex(name = 'V_50',
              particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_57})

V_51 = Vertex(name = 'V_51',
              particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_57})

V_52 = Vertex(name = 'V_52',
              particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_57})

V_53 = Vertex(name = 'V_53',
              particles = [ P.e__plus__, P.e__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              couplings = {(0,0):C.GC_58,(0,1):C.GC_70})

V_54 = Vertex(name = 'V_54',
              particles = [ P.mu__plus__, P.mu__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              couplings = {(0,0):C.GC_58,(0,1):C.GC_70})

V_55 = Vertex(name = 'V_55',
              particles = [ P.ta__plus__, P.ta__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV6 ],
              couplings = {(0,0):C.GC_58,(0,1):C.GC_70})

V_56 = Vertex(name = 'V_56',
              particles = [ P.c__tilde__, P.c, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_57 = Vertex(name = 'V_57',
              particles = [ P.t__tilde__, P.t, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3, L.FFV5, L.FFV8 ],
              couplings = {(0,1):C.GC_2,(0,0):C.GC_170,(0,2):C.GC_91})

V_58 = Vertex(name = 'V_58',
              particles = [ P.u__tilde__, P.u, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_59 = Vertex(name = 'V_59',
              particles = [ P.c__tilde__, P.c, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_7})

V_60 = Vertex(name = 'V_60',
              particles = [ P.t__tilde__, P.t, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV3, L.FFV5, L.FFV8 ],
              couplings = {(0,1):C.GC_7,(0,0):C.GC_145,(0,2):C.GC_79})

V_61 = Vertex(name = 'V_61',
              particles = [ P.u__tilde__, P.u, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_7})

V_62 = Vertex(name = 'V_62',
              particles = [ P.s__tilde__, P.c, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_57})

V_63 = Vertex(name = 'V_63',
              particles = [ P.b__tilde__, P.t, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              couplings = {(0,1):C.GC_165,(0,0):C.GC_57})

V_64 = Vertex(name = 'V_64',
              particles = [ P.b__tilde__, P.t, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_92})

V_65 = Vertex(name = 'V_65',
              particles = [ P.d__tilde__, P.u, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_57})

V_66 = Vertex(name = 'V_66',
              particles = [ P.c__tilde__, P.c, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV7 ],
              couplings = {(0,0):C.GC_59,(0,1):C.GC_69})

V_67 = Vertex(name = 'V_67',
              particles = [ P.t__tilde__, P.t, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3, L.FFV7, L.FFV8 ],
              couplings = {(0,0):C.GC_59,(0,2):C.GC_69,(0,1):C.GC_166,(0,3):C.GC_81})

V_68 = Vertex(name = 'V_68',
              particles = [ P.t__tilde__, P.t, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_138})

V_69 = Vertex(name = 'V_69',
              particles = [ P.u__tilde__, P.u, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV7 ],
              couplings = {(0,0):C.GC_59,(0,1):C.GC_69})

V_70 = Vertex(name = 'V_70',
              particles = [ P.b__tilde__, P.b, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_71 = Vertex(name = 'V_71',
              particles = [ P.d__tilde__, P.d, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_72 = Vertex(name = 'V_72',
              particles = [ P.s__tilde__, P.s, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_73 = Vertex(name = 'V_73',
              particles = [ P.b__tilde__, P.b, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_7})

V_74 = Vertex(name = 'V_74',
              particles = [ P.d__tilde__, P.d, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_7})

V_75 = Vertex(name = 'V_75',
              particles = [ P.s__tilde__, P.s, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_7})

V_76 = Vertex(name = 'V_76',
              particles = [ P.t__tilde__, P.b, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV8 ],
              couplings = {(0,1):C.GC_80,(0,0):C.GC_57})

V_77 = Vertex(name = 'V_77',
              particles = [ P.t__tilde__, P.b, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_108})

V_78 = Vertex(name = 'V_78',
              particles = [ P.u__tilde__, P.d, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_57})

V_79 = Vertex(name = 'V_79',
              particles = [ P.c__tilde__, P.s, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_57})

V_80 = Vertex(name = 'V_80',
              particles = [ P.b__tilde__, P.b, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV4 ],
              couplings = {(0,0):C.GC_58,(0,1):C.GC_69})

V_81 = Vertex(name = 'V_81',
              particles = [ P.b__tilde__, P.b, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_137})

V_82 = Vertex(name = 'V_82',
              particles = [ P.d__tilde__, P.d, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV4 ],
              couplings = {(0,0):C.GC_58,(0,1):C.GC_69})

V_83 = Vertex(name = 'V_83',
              particles = [ P.s__tilde__, P.s, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV4 ],
              couplings = {(0,0):C.GC_58,(0,1):C.GC_69})

V_84 = Vertex(name = 'V_84',
              particles = [ P.t__tilde__, P.b, P.d__tilde__, P.u ],
              color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
              lorentz = [ L.FFFF3 ],
              couplings = {(0,0):C.GC_14,(1,0):C.GC_25})

V_85 = Vertex(name = 'V_85',
              particles = [ P.u__tilde__, P.d, P.b__tilde__, P.t ],
              color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
              lorentz = [ L.FFFF3 ],
              couplings = {(0,0):C.GC_14,(1,0):C.GC_25})

V_86 = Vertex(name = 'V_86',
              particles = [ P.t__tilde__, P.b, P.s__tilde__, P.c ],
              color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
              lorentz = [ L.FFFF3 ],
              couplings = {(0,0):C.GC_14,(1,0):C.GC_25})

V_87 = Vertex(name = 'V_87',
              particles = [ P.c__tilde__, P.s, P.b__tilde__, P.t ],
              color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
              lorentz = [ L.FFFF3 ],
              couplings = {(0,0):C.GC_14,(1,0):C.GC_25})

V_88 = Vertex(name = 'V_88',
              particles = [ P.t__tilde__, P.t, P.G__minus__, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS1, L.FFSS2 ],
              couplings = {(0,0):C.GC_18,(0,1):C.GC_98})

V_89 = Vertex(name = 'V_89',
              particles = [ P.t__tilde__, P.t, P.a, P.G__minus__, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVSS1 ],
              couplings = {(0,0):C.GC_114})

V_90 = Vertex(name = 'V_90',
              particles = [ P.t__tilde__, P.t, P.G0, P.G0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS4 ],
              couplings = {(0,0):C.GC_110})

V_91 = Vertex(name = 'V_91',
              particles = [ P.t__tilde__, P.t, P.G0, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS3 ],
              couplings = {(0,0):C.GC_109})

V_92 = Vertex(name = 'V_92',
              particles = [ P.t__tilde__, P.t, P.G0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_127})

V_93 = Vertex(name = 'V_93',
              particles = [ P.t__tilde__, P.t, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS4 ],
              couplings = {(0,0):C.GC_110})

V_94 = Vertex(name = 'V_94',
              particles = [ P.b__tilde__, P.t, P.G0, P.G__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS1, L.FFSS2 ],
              couplings = {(0,0):C.GC_100,(0,1):C.GC_21})

V_95 = Vertex(name = 'V_95',
              particles = [ P.b__tilde__, P.t, P.G__minus__, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS1, L.FFSS2 ],
              couplings = {(0,0):C.GC_20,(0,1):C.GC_101})

V_96 = Vertex(name = 'V_96',
              particles = [ P.b__tilde__, P.t, P.G__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_77})

V_97 = Vertex(name = 'V_97',
              particles = [ P.b__tilde__, P.t, P.a, P.G0, P.G__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVSS1 ],
              couplings = {(0,0):C.GC_39})

V_98 = Vertex(name = 'V_98',
              particles = [ P.b__tilde__, P.t, P.a, P.G__minus__, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVSS1 ],
              couplings = {(0,0):C.GC_40})

V_99 = Vertex(name = 'V_99',
              particles = [ P.b__tilde__, P.t, P.a, P.G__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVS1, L.FFVS2 ],
              couplings = {(0,1):C.GC_162,(0,0):C.GC_82})

V_100 = Vertex(name = 'V_100',
               particles = [ P.t__tilde__, P.t, P.a, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS2, L.FFVS3 ],
               couplings = {(0,0):C.GC_164,(0,1):C.GC_72})

V_101 = Vertex(name = 'V_101',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_117})

V_102 = Vertex(name = 'V_102',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_118})

V_103 = Vertex(name = 'V_103',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS3 ],
               couplings = {(0,0):C.GC_132,(0,1):C.GC_35})

V_104 = Vertex(name = 'V_104',
               particles = [ P.t__tilde__, P.t, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_116})

V_105 = Vertex(name = 'V_105',
               particles = [ P.t__tilde__, P.t, P.W__plus__, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_118})

V_106 = Vertex(name = 'V_106',
               particles = [ P.t__tilde__, P.t, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2 ],
               couplings = {(0,0):C.GC_132,(0,1):C.GC_149})

V_107 = Vertex(name = 'V_107',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_62})

V_108 = Vertex(name = 'V_108',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_62})

V_109 = Vertex(name = 'V_109',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2 ],
               couplings = {(0,1):C.GC_148,(0,0):C.GC_88})

V_110 = Vertex(name = 'V_110',
               particles = [ P.t__tilde__, P.t, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_120})

V_111 = Vertex(name = 'V_111',
               particles = [ P.b__tilde__, P.t, P.Z, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_125})

V_112 = Vertex(name = 'V_112',
               particles = [ P.b__tilde__, P.t, P.Z, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_123})

V_113 = Vertex(name = 'V_113',
               particles = [ P.b__tilde__, P.t, P.Z, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2 ],
               couplings = {(0,1):C.GC_150,(0,0):C.GC_135})

V_114 = Vertex(name = 'V_114',
               particles = [ P.t__tilde__, P.t, P.Z, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_122})

V_115 = Vertex(name = 'V_115',
               particles = [ P.t__tilde__, P.t, P.Z, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_122})

V_116 = Vertex(name = 'V_116',
               particles = [ P.t__tilde__, P.t, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3 ],
               couplings = {(0,0):C.GC_134,(0,1):C.GC_152,(0,2):C.GC_37})

V_117 = Vertex(name = 'V_117',
               particles = [ P.t__tilde__, P.b, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_21,(0,1):C.GC_100})

V_118 = Vertex(name = 'V_118',
               particles = [ P.t__tilde__, P.b, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_99,(0,1):C.GC_22})

V_119 = Vertex(name = 'V_119',
               particles = [ P.b__tilde__, P.b, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_19,(0,1):C.GC_97})

V_120 = Vertex(name = 'V_120',
               particles = [ P.b__tilde__, P.b, P.a, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_113})

V_121 = Vertex(name = 'V_121',
               particles = [ P.t__tilde__, P.b, P.a, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_103})

V_122 = Vertex(name = 'V_122',
               particles = [ P.t__tilde__, P.b, P.a, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_102})

V_123 = Vertex(name = 'V_123',
               particles = [ P.t__tilde__, P.b, P.a, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS3 ],
               couplings = {(0,1):C.GC_71,(0,0):C.GC_106})

V_124 = Vertex(name = 'V_124',
               particles = [ P.b__tilde__, P.b, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_111})

V_125 = Vertex(name = 'V_125',
               particles = [ P.b__tilde__, P.b, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS3 ],
               couplings = {(0,0):C.GC_112})

V_126 = Vertex(name = 'V_126',
               particles = [ P.b__tilde__, P.b, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_130})

V_127 = Vertex(name = 'V_127',
               particles = [ P.b__tilde__, P.b, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_111})

V_128 = Vertex(name = 'V_128',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_62})

V_129 = Vertex(name = 'V_129',
               particles = [ P.b__tilde__, P.b, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_116})

V_130 = Vertex(name = 'V_130',
               particles = [ P.b__tilde__, P.b, P.W__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_115})

V_131 = Vertex(name = 'V_131',
               particles = [ P.b__tilde__, P.b, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1 ],
               couplings = {(0,0):C.GC_131})

V_132 = Vertex(name = 'V_132',
               particles = [ P.b__tilde__, P.b, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_117})

V_133 = Vertex(name = 'V_133',
               particles = [ P.b__tilde__, P.b, P.W__plus__, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_115})

V_134 = Vertex(name = 'V_134',
               particles = [ P.b__tilde__, P.b, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1 ],
               couplings = {(0,0):C.GC_131})

V_135 = Vertex(name = 'V_135',
               particles = [ P.b__tilde__, P.b, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_121})

V_136 = Vertex(name = 'V_136',
               particles = [ P.t__tilde__, P.b, P.Z, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_124})

V_137 = Vertex(name = 'V_137',
               particles = [ P.t__tilde__, P.b, P.Z, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_126})

V_138 = Vertex(name = 'V_138',
               particles = [ P.t__tilde__, P.b, P.Z, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS3 ],
               couplings = {(0,1):C.GC_36,(0,0):C.GC_136})

V_139 = Vertex(name = 'V_139',
               particles = [ P.b__tilde__, P.b, P.Z, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_119})

V_140 = Vertex(name = 'V_140',
               particles = [ P.b__tilde__, P.b, P.Z, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_119})

V_141 = Vertex(name = 'V_141',
               particles = [ P.b__tilde__, P.b, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1 ],
               couplings = {(0,0):C.GC_133})

V_142 = Vertex(name = 'V_142',
               particles = [ P.t__tilde__, P.b, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_105})

V_143 = Vertex(name = 'V_143',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_104})

V_144 = Vertex(name = 'V_144',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_104})

V_145 = Vertex(name = 'V_145',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS3 ],
               couplings = {(0,1):C.GC_33,(0,0):C.GC_107})

V_146 = Vertex(name = 'V_146',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_104})

V_147 = Vertex(name = 'V_147',
               particles = [ P.t__tilde__, P.t, P.g, P.G0 ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFVS2, L.FFVS3 ],
               couplings = {(0,0):C.GC_140,(0,1):C.GC_32})

V_148 = Vertex(name = 'V_148',
               particles = [ P.t__tilde__, P.t, P.g, P.H ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFVS2, L.FFVS3 ],
               couplings = {(0,0):C.GC_141,(0,1):C.GC_31})

V_149 = Vertex(name = 'V_149',
               particles = [ P.b__tilde__, P.t, P.g, P.G__minus__ ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFVS2 ],
               couplings = {(0,0):C.GC_139})

V_150 = Vertex(name = 'V_150',
               particles = [ P.t__tilde__, P.t, P.g, P.g, P.G0 ],
               color = [ 'f(-1,3,4)*T(-1,2,1)' ],
               lorentz = [ L.FFVVS1, L.FFVVS2 ],
               couplings = {(0,0):C.GC_144,(0,1):C.GC_48})

V_151 = Vertex(name = 'V_151',
               particles = [ P.t__tilde__, P.t, P.g, P.g, P.H ],
               color = [ 'f(-1,3,4)*T(-1,2,1)' ],
               lorentz = [ L.FFVVS1, L.FFVVS2 ],
               couplings = {(0,0):C.GC_143,(0,1):C.GC_47})

V_152 = Vertex(name = 'V_152',
               particles = [ P.t__tilde__, P.t, P.g, P.g ],
               color = [ 'f(-1,3,4)*T(-1,2,1)' ],
               lorentz = [ L.FFVV1, L.FFVV2 ],
               couplings = {(0,0):C.GC_146,(0,1):C.GC_85})

V_153 = Vertex(name = 'V_153',
               particles = [ P.b__tilde__, P.t, P.g, P.g, P.G__minus__ ],
               color = [ 'f(-1,3,4)*T(-1,2,1)' ],
               lorentz = [ L.FFVVS1 ],
               couplings = {(0,0):C.GC_142})

V_154 = Vertex(name = 'V_154',
               particles = [ P.t__tilde__, P.t, P.a, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS2, L.FFVS3 ],
               couplings = {(0,0):C.GC_163,(0,1):C.GC_73})

V_155 = Vertex(name = 'V_155',
               particles = [ P.t__tilde__, P.t, P.Z, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS2, L.FFVS3 ],
               couplings = {(0,0):C.GC_151,(0,1):C.GC_38})

V_156 = Vertex(name = 'V_156',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS2 ],
               couplings = {(0,0):C.GC_147})

V_157 = Vertex(name = 'V_157',
               particles = [ P.b__tilde__, P.t, P.a, P.W__minus__, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS1 ],
               couplings = {(0,0):C.GC_153})

V_158 = Vertex(name = 'V_158',
               particles = [ P.b__tilde__, P.t, P.a, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS1 ],
               couplings = {(0,0):C.GC_154})

V_159 = Vertex(name = 'V_159',
               particles = [ P.b__tilde__, P.t, P.a, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVV1 ],
               couplings = {(0,0):C.GC_167})

V_160 = Vertex(name = 'V_160',
               particles = [ P.t__tilde__, P.t, P.a, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS1 ],
               couplings = {(0,0):C.GC_155})

V_161 = Vertex(name = 'V_161',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS1, L.FFVVS2 ],
               couplings = {(0,0):C.GC_157,(0,1):C.GC_65})

V_162 = Vertex(name = 'V_162',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS1, L.FFVVS2 ],
               couplings = {(0,0):C.GC_158,(0,1):C.GC_64})

V_163 = Vertex(name = 'V_163',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVV1, L.FFVV2 ],
               couplings = {(0,0):C.GC_168,(0,1):C.GC_89})

V_164 = Vertex(name = 'V_164',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS1 ],
               couplings = {(0,0):C.GC_156})

V_165 = Vertex(name = 'V_165',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS1 ],
               couplings = {(0,0):C.GC_160})

V_166 = Vertex(name = 'V_166',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS1 ],
               couplings = {(0,0):C.GC_159})

V_167 = Vertex(name = 'V_167',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVV1 ],
               couplings = {(0,0):C.GC_169})

V_168 = Vertex(name = 'V_168',
               particles = [ P.t__tilde__, P.t, P.W__plus__, P.Z, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS1 ],
               couplings = {(0,0):C.GC_161})

V_169 = Vertex(name = 'V_169',
               particles = [ P.t__tilde__, P.t, P.a, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS2 ],
               couplings = {(0,0):C.GC_43})

V_170 = Vertex(name = 'V_170',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.Z, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS2 ],
               couplings = {(0,0):C.GC_68})

V_171 = Vertex(name = 'V_171',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS3 ],
               couplings = {(0,0):C.GC_34})

V_172 = Vertex(name = 'V_172',
               particles = [ P.t__tilde__, P.b, P.a, P.W__plus__, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS2 ],
               couplings = {(0,0):C.GC_41})

V_173 = Vertex(name = 'V_173',
               particles = [ P.t__tilde__, P.b, P.a, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS2 ],
               couplings = {(0,0):C.GC_42})

V_174 = Vertex(name = 'V_174',
               particles = [ P.t__tilde__, P.b, P.a, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVV2 ],
               couplings = {(0,0):C.GC_83})

V_175 = Vertex(name = 'V_175',
               particles = [ P.t__tilde__, P.b, P.W__minus__, P.W__plus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS2 ],
               couplings = {(0,0):C.GC_63})

V_176 = Vertex(name = 'V_176',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS2 ],
               couplings = {(0,0):C.GC_67})

V_177 = Vertex(name = 'V_177',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS2 ],
               couplings = {(0,0):C.GC_66})

V_178 = Vertex(name = 'V_178',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVV2 ],
               couplings = {(0,0):C.GC_90})

V_179 = Vertex(name = 'V_179',
               particles = [ P.t__tilde__, P.b, P.g, P.G__plus__ ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFVS3 ],
               couplings = {(0,0):C.GC_30})

V_180 = Vertex(name = 'V_180',
               particles = [ P.t__tilde__, P.b, P.g, P.g, P.G__plus__ ],
               color = [ 'f(-1,3,4)*T(-1,2,1)' ],
               lorentz = [ L.FFVVS2 ],
               couplings = {(0,0):C.GC_46})

