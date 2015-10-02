# This file was automatically created by FeynRules $Revision: 821 $
# Mathematica version: 7.0 for Microsoft Windows (32-bit) (February 18, 2009)
# Date: Mon 3 Oct 2011 13:27:06


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_30})

V_2 = Vertex(name = 'V_2',
             particles = [ P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_69})

V_3 = Vertex(name = 'V_3',
             particles = [ P.G, P.G, P.H, P.H ],
             color = [ 'Identity(1,2)' ],
             lorentz = [ L.VVSS2 ],
             couplings = {(0,0):C.GC_32})

V_4 = Vertex(name = 'V_4',
             particles = [ P.G, P.G, P.H ],
             color = [ 'Identity(1,2)' ],
             lorentz = [ L.VVS2 ],
             couplings = {(0,0):C.GC_70})

V_5 = Vertex(name = 'V_5',
             particles = [ P.ghG, P.ghG__tilde__, P.G ],
             color = [ 'f(3,1,2)' ],
             lorentz = [ L.UUV1 ],
             couplings = {(0,0):C.GC_4})

V_6 = Vertex(name = 'V_6',
             particles = [ P.G, P.G, P.G ],
             color = [ 'f(1,2,3)' ],
             lorentz = [ L.VVV1, L.VVV2 ],
             couplings = {(0,1):C.GC_31,(0,0):C.GC_4})

V_7 = Vertex(name = 'V_7',
             particles = [ P.G, P.G, P.G, P.G ],
             color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
             lorentz = [ L.VVVV1, L.VVVV2, L.VVVV4, L.VVVV5, L.VVVV7, L.VVVV8 ],
             couplings = {(0,1):C.GC_36,(1,5):C.GC_36,(2,4):C.GC_36,(1,2):C.GC_6,(0,0):C.GC_6,(2,3):C.GC_6})

V_8 = Vertex(name = 'V_8',
             particles = [ P.G, P.G, P.G, P.H, P.H ],
             color = [ 'f(1,2,3)' ],
             lorentz = [ L.VVVSS1 ],
             couplings = {(0,0):C.GC_37})

V_9 = Vertex(name = 'V_9',
             particles = [ P.G, P.G, P.G, P.H ],
             color = [ 'f(1,2,3)' ],
             lorentz = [ L.VVVS1 ],
             couplings = {(0,0):C.GC_74})

V_10 = Vertex(name = 'V_10',
              particles = [ P.G, P.G, P.G, P.G, P.H, P.H ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVVSS1, L.VVVVSS2, L.VVVVSS3 ],
              couplings = {(1,1):C.GC_41,(0,0):C.GC_41,(2,2):C.GC_41})

V_11 = Vertex(name = 'V_11',
              particles = [ P.G, P.G, P.G, P.G, P.H ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVVS1, L.VVVVS2, L.VVVVS3 ],
              couplings = {(1,1):C.GC_76,(0,0):C.GC_76,(2,2):C.GC_76})

V_12 = Vertex(name = 'V_12',
              particles = [ P.G, P.G, P.G, P.G, P.G ],
              color = [ 'f(-1,1,-2)*f(2,3,-1)*f(4,5,-2)', 'f(-1,1,-2)*f(2,3,-2)*f(4,5,-1)', 'f(-1,1,-2)*f(2,4,-1)*f(3,5,-2)', 'f(-1,1,-2)*f(2,4,-2)*f(3,5,-1)', 'f(-1,1,-2)*f(2,5,-1)*f(3,4,-2)', 'f(-1,1,-2)*f(2,5,-2)*f(3,4,-1)', 'f(-1,1,2)*f(3,-2,-1)*f(4,5,-2)', 'f(-1,1,2)*f(3,4,-2)*f(5,-2,-1)', 'f(-1,1,2)*f(3,5,-2)*f(4,-2,-1)', 'f(-1,1,3)*f(2,-2,-1)*f(4,5,-2)', 'f(-1,1,3)*f(2,4,-2)*f(5,-2,-1)', 'f(-1,1,3)*f(2,5,-2)*f(4,-2,-1)', 'f(-1,1,4)*f(2,-2,-1)*f(3,5,-2)', 'f(-1,1,4)*f(2,3,-2)*f(5,-2,-1)', 'f(-1,1,4)*f(2,5,-2)*f(3,-2,-1)', 'f(-1,1,5)*f(2,-2,-1)*f(3,4,-2)', 'f(-1,1,5)*f(2,3,-2)*f(4,-2,-1)', 'f(-1,1,5)*f(2,4,-2)*f(3,-2,-1)', 'f(-1,2,-2)*f(1,3,-2)*f(4,5,-1)', 'f(-1,2,-2)*f(1,4,-2)*f(3,5,-1)', 'f(-1,2,-2)*f(1,5,-2)*f(3,4,-1)', 'f(-1,2,3)*f(1,4,-2)*f(5,-2,-1)', 'f(-1,2,3)*f(1,5,-2)*f(4,-2,-1)', 'f(-1,2,4)*f(1,3,-2)*f(5,-2,-1)', 'f(-1,2,4)*f(1,5,-2)*f(3,-2,-1)', 'f(-1,2,5)*f(1,3,-2)*f(4,-2,-1)', 'f(-1,2,5)*f(1,4,-2)*f(3,-2,-1)', 'f(-1,3,-2)*f(1,2,-2)*f(4,5,-1)', 'f(-1,3,4)*f(1,2,-2)*f(5,-2,-1)', 'f(-1,3,5)*f(1,2,-2)*f(4,-2,-1)' ],
              lorentz = [ L.VVVVV1, L.VVVVV10, L.VVVVV11, L.VVVVV12, L.VVVVV13, L.VVVVV14, L.VVVVV15, L.VVVVV16, L.VVVVV17, L.VVVVV18, L.VVVVV19, L.VVVVV2, L.VVVVV20, L.VVVVV21, L.VVVVV22, L.VVVVV23, L.VVVVV24, L.VVVVV25, L.VVVVV26, L.VVVVV27, L.VVVVV28, L.VVVVV29, L.VVVVV3, L.VVVVV30, L.VVVVV4, L.VVVVV5, L.VVVVV6, L.VVVVV7, L.VVVVV8, L.VVVVV9 ],
              couplings = {(2,4):C.GC_39,(4,9):C.GC_40,(5,9):C.GC_39,(3,4):C.GC_40,(15,16):C.GC_39,(9,25):C.GC_39,(20,16):C.GC_40,(18,25):C.GC_40,(17,17):C.GC_39,(6,0):C.GC_39,(24,17):C.GC_40,(27,0):C.GC_40,(25,27):C.GC_39,(29,11):C.GC_39,(8,11):C.GC_40,(11,27):C.GC_40,(2,8):C.GC_40,(4,5):C.GC_39,(5,5):C.GC_40,(3,8):C.GC_39,(12,15):C.GC_39,(19,15):C.GC_40,(14,18):C.GC_39,(26,18):C.GC_40,(23,28):C.GC_39,(28,22):C.GC_39,(7,22):C.GC_40,(10,28):C.GC_40,(0,24):C.GC_39,(1,24):C.GC_40,(12,6):C.GC_39,(19,6):C.GC_40,(26,10):C.GC_39,(14,10):C.GC_40,(16,2):C.GC_39,(22,2):C.GC_40,(0,26):C.GC_40,(1,26):C.GC_39,(9,29):C.GC_39,(18,29):C.GC_40,(11,19):C.GC_39,(25,19):C.GC_40,(21,14):C.GC_39,(13,14):C.GC_40,(15,7):C.GC_39,(20,7):C.GC_40,(24,12):C.GC_39,(17,12):C.GC_40,(13,3):C.GC_39,(21,3):C.GC_40,(22,13):C.GC_39,(16,13):C.GC_40,(10,21):C.GC_39,(23,21):C.GC_40,(6,1):C.GC_39,(27,1):C.GC_40,(8,20):C.GC_39,(29,20):C.GC_40,(7,23):C.GC_39,(28,23):C.GC_40})

V_13 = Vertex(name = 'V_13',
              particles = [ P.G, P.G, P.G, P.G, P.G, P.G ],
              color = [ 'f(-2,-3,-1)*f(-1,1,2)*f(3,4,-2)*f(5,6,-3)', 'f(-2,-3,-1)*f(-1,1,2)*f(3,4,-3)*f(5,6,-2)', 'f(-2,-3,-1)*f(-1,1,2)*f(3,5,-2)*f(4,6,-3)', 'f(-2,-3,-1)*f(-1,1,2)*f(3,5,-3)*f(4,6,-2)', 'f(-2,-3,-1)*f(-1,1,2)*f(3,6,-2)*f(4,5,-3)', 'f(-2,-3,-1)*f(-1,1,2)*f(3,6,-3)*f(4,5,-2)', 'f(-2,-3,-1)*f(-1,1,3)*f(2,4,-2)*f(5,6,-3)', 'f(-2,-3,-1)*f(-1,1,3)*f(2,4,-3)*f(5,6,-2)', 'f(-2,-3,-1)*f(-1,1,3)*f(2,5,-2)*f(4,6,-3)', 'f(-2,-3,-1)*f(-1,1,3)*f(2,5,-3)*f(4,6,-2)', 'f(-2,-3,-1)*f(-1,1,3)*f(2,6,-2)*f(4,5,-3)', 'f(-2,-3,-1)*f(-1,1,3)*f(2,6,-3)*f(4,5,-2)', 'f(-2,-3,-1)*f(-1,1,4)*f(2,3,-2)*f(5,6,-3)', 'f(-2,-3,-1)*f(-1,1,4)*f(2,3,-3)*f(5,6,-2)', 'f(-2,-3,-1)*f(-1,1,4)*f(2,5,-2)*f(3,6,-3)', 'f(-2,-3,-1)*f(-1,1,4)*f(2,5,-3)*f(3,6,-2)', 'f(-2,-3,-1)*f(-1,1,4)*f(2,6,-2)*f(3,5,-3)', 'f(-2,-3,-1)*f(-1,1,4)*f(2,6,-3)*f(3,5,-2)', 'f(-2,-3,-1)*f(-1,1,5)*f(2,3,-2)*f(4,6,-3)', 'f(-2,-3,-1)*f(-1,1,5)*f(2,3,-3)*f(4,6,-2)', 'f(-2,-3,-1)*f(-1,1,5)*f(2,4,-2)*f(3,6,-3)', 'f(-2,-3,-1)*f(-1,1,5)*f(2,4,-3)*f(3,6,-2)', 'f(-2,-3,-1)*f(-1,1,5)*f(2,6,-2)*f(3,4,-3)', 'f(-2,-3,-1)*f(-1,1,5)*f(2,6,-3)*f(3,4,-2)', 'f(-2,-3,-1)*f(-1,1,6)*f(2,3,-2)*f(4,5,-3)', 'f(-2,-3,-1)*f(-1,1,6)*f(2,3,-3)*f(4,5,-2)', 'f(-2,-3,-1)*f(-1,1,6)*f(2,4,-2)*f(3,5,-3)', 'f(-2,-3,-1)*f(-1,1,6)*f(2,4,-3)*f(3,5,-2)', 'f(-2,-3,-1)*f(-1,1,6)*f(2,5,-2)*f(3,4,-3)', 'f(-2,-3,-1)*f(-1,1,6)*f(2,5,-3)*f(3,4,-2)', 'f(-2,-3,-1)*f(-1,2,3)*f(1,4,-2)*f(5,6,-3)', 'f(-2,-3,-1)*f(-1,2,3)*f(1,4,-3)*f(5,6,-2)', 'f(-2,-3,-1)*f(-1,2,3)*f(1,5,-2)*f(4,6,-3)', 'f(-2,-3,-1)*f(-1,2,3)*f(1,5,-3)*f(4,6,-2)', 'f(-2,-3,-1)*f(-1,2,3)*f(1,6,-2)*f(4,5,-3)', 'f(-2,-3,-1)*f(-1,2,3)*f(1,6,-3)*f(4,5,-2)', 'f(-2,-3,-1)*f(-1,2,4)*f(1,3,-2)*f(5,6,-3)', 'f(-2,-3,-1)*f(-1,2,4)*f(1,3,-3)*f(5,6,-2)', 'f(-2,-3,-1)*f(-1,2,4)*f(1,5,-2)*f(3,6,-3)', 'f(-2,-3,-1)*f(-1,2,4)*f(1,5,-3)*f(3,6,-2)', 'f(-2,-3,-1)*f(-1,2,4)*f(1,6,-2)*f(3,5,-3)', 'f(-2,-3,-1)*f(-1,2,4)*f(1,6,-3)*f(3,5,-2)', 'f(-2,-3,-1)*f(-1,2,5)*f(1,3,-2)*f(4,6,-3)', 'f(-2,-3,-1)*f(-1,2,5)*f(1,3,-3)*f(4,6,-2)', 'f(-2,-3,-1)*f(-1,2,5)*f(1,4,-2)*f(3,6,-3)', 'f(-2,-3,-1)*f(-1,2,5)*f(1,4,-3)*f(3,6,-2)', 'f(-2,-3,-1)*f(-1,2,5)*f(1,6,-2)*f(3,4,-3)', 'f(-2,-3,-1)*f(-1,2,5)*f(1,6,-3)*f(3,4,-2)', 'f(-2,-3,-1)*f(-1,2,6)*f(1,3,-2)*f(4,5,-3)', 'f(-2,-3,-1)*f(-1,2,6)*f(1,3,-3)*f(4,5,-2)', 'f(-2,-3,-1)*f(-1,2,6)*f(1,4,-2)*f(3,5,-3)', 'f(-2,-3,-1)*f(-1,2,6)*f(1,4,-3)*f(3,5,-2)', 'f(-2,-3,-1)*f(-1,2,6)*f(1,5,-2)*f(3,4,-3)', 'f(-2,-3,-1)*f(-1,2,6)*f(1,5,-3)*f(3,4,-2)', 'f(-2,-3,-1)*f(-1,3,4)*f(1,2,-2)*f(5,6,-3)', 'f(-2,-3,-1)*f(-1,3,4)*f(1,2,-3)*f(5,6,-2)', 'f(-2,-3,-1)*f(-1,3,4)*f(1,5,-2)*f(2,6,-3)', 'f(-2,-3,-1)*f(-1,3,4)*f(1,5,-3)*f(2,6,-2)', 'f(-2,-3,-1)*f(-1,3,4)*f(1,6,-2)*f(2,5,-3)', 'f(-2,-3,-1)*f(-1,3,4)*f(1,6,-3)*f(2,5,-2)', 'f(-2,-3,-1)*f(-1,3,5)*f(1,2,-2)*f(4,6,-3)', 'f(-2,-3,-1)*f(-1,3,5)*f(1,2,-3)*f(4,6,-2)', 'f(-2,-3,-1)*f(-1,3,5)*f(1,4,-2)*f(2,6,-3)', 'f(-2,-3,-1)*f(-1,3,5)*f(1,4,-3)*f(2,6,-2)', 'f(-2,-3,-1)*f(-1,3,5)*f(1,6,-2)*f(2,4,-3)', 'f(-2,-3,-1)*f(-1,3,5)*f(1,6,-3)*f(2,4,-2)', 'f(-2,-3,-1)*f(-1,3,6)*f(1,2,-2)*f(4,5,-3)', 'f(-2,-3,-1)*f(-1,3,6)*f(1,2,-3)*f(4,5,-2)', 'f(-2,-3,-1)*f(-1,3,6)*f(1,4,-2)*f(2,5,-3)', 'f(-2,-3,-1)*f(-1,3,6)*f(1,4,-3)*f(2,5,-2)', 'f(-2,-3,-1)*f(-1,3,6)*f(1,5,-2)*f(2,4,-3)', 'f(-2,-3,-1)*f(-1,3,6)*f(1,5,-3)*f(2,4,-2)', 'f(-2,-3,-1)*f(-1,4,5)*f(1,2,-2)*f(3,6,-3)', 'f(-2,-3,-1)*f(-1,4,5)*f(1,2,-3)*f(3,6,-2)', 'f(-2,-3,-1)*f(-1,4,5)*f(1,3,-2)*f(2,6,-3)', 'f(-2,-3,-1)*f(-1,4,5)*f(1,3,-3)*f(2,6,-2)', 'f(-2,-3,-1)*f(-1,4,5)*f(1,6,-2)*f(2,3,-3)', 'f(-2,-3,-1)*f(-1,4,5)*f(1,6,-3)*f(2,3,-2)', 'f(-2,-3,-1)*f(-1,4,6)*f(1,2,-2)*f(3,5,-3)', 'f(-2,-3,-1)*f(-1,4,6)*f(1,2,-3)*f(3,5,-2)', 'f(-2,-3,-1)*f(-1,4,6)*f(1,3,-2)*f(2,5,-3)', 'f(-2,-3,-1)*f(-1,4,6)*f(1,3,-3)*f(2,5,-2)', 'f(-2,-3,-1)*f(-1,4,6)*f(1,5,-2)*f(2,3,-3)', 'f(-2,-3,-1)*f(-1,4,6)*f(1,5,-3)*f(2,3,-2)', 'f(-2,-3,-1)*f(-1,5,6)*f(1,2,-2)*f(3,4,-3)', 'f(-2,-3,-1)*f(-1,5,6)*f(1,2,-3)*f(3,4,-2)', 'f(-2,-3,-1)*f(-1,5,6)*f(1,3,-2)*f(2,4,-3)', 'f(-2,-3,-1)*f(-1,5,6)*f(1,3,-3)*f(2,4,-2)', 'f(-2,-3,-1)*f(-1,5,6)*f(1,4,-2)*f(2,3,-3)', 'f(-2,-3,-1)*f(-1,5,6)*f(1,4,-3)*f(2,3,-2)' ],
              lorentz = [ L.VVVVVV1, L.VVVVVV10, L.VVVVVV11, L.VVVVVV12, L.VVVVVV13, L.VVVVVV14, L.VVVVVV15, L.VVVVVV2, L.VVVVVV3, L.VVVVVV4, L.VVVVVV5, L.VVVVVV6, L.VVVVVV7, L.VVVVVV8, L.VVVVVV9 ],
              couplings = {(5,11):C.GC_42,(4,11):C.GC_43,(3,9):C.GC_43,(2,9):C.GC_42,(11,1):C.GC_43,(10,1):C.GC_42,(7,7):C.GC_43,(6,7):C.GC_42,(17,4):C.GC_43,(16,4):C.GC_42,(13,8):C.GC_43,(12,8):C.GC_42,(21,2):C.GC_42,(20,2):C.GC_43,(19,3):C.GC_42,(18,3):C.GC_43,(33,3):C.GC_43,(32,3):C.GC_42,(31,8):C.GC_42,(30,8):C.GC_43,(39,2):C.GC_43,(38,2):C.GC_42,(37,7):C.GC_42,(36,7):C.GC_43,(51,4):C.GC_42,(50,4):C.GC_43,(49,1):C.GC_42,(48,1):C.GC_43,(63,4):C.GC_43,(62,4):C.GC_42,(61,9):C.GC_42,(60,9):C.GC_43,(71,2):C.GC_42,(70,2):C.GC_43,(67,11):C.GC_43,(66,11):C.GC_42,(75,1):C.GC_43,(74,1):C.GC_42,(73,11):C.GC_42,(72,11):C.GC_43,(83,3):C.GC_42,(82,3):C.GC_43,(79,9):C.GC_43,(78,9):C.GC_42,(89,8):C.GC_43,(88,8):C.GC_42,(87,7):C.GC_43,(86,7):C.GC_42,(9,13):C.GC_43,(8,13):C.GC_42,(15,5):C.GC_43,(14,5):C.GC_42,(27,14):C.GC_42,(26,14):C.GC_43,(25,6):C.GC_42,(24,6):C.GC_43,(35,6):C.GC_43,(34,6):C.GC_42,(41,14):C.GC_43,(40,14):C.GC_42,(45,5):C.GC_42,(44,5):C.GC_43,(43,13):C.GC_42,(42,13):C.GC_43,(65,14):C.GC_42,(64,14):C.GC_43,(69,5):C.GC_43,(68,5):C.GC_42,(77,6):C.GC_42,(76,6):C.GC_43,(81,13):C.GC_43,(80,13):C.GC_42,(1,0):C.GC_43,(0,0):C.GC_42,(23,10):C.GC_43,(22,10):C.GC_42,(53,10):C.GC_42,(52,10):C.GC_43,(57,10):C.GC_43,(56,10):C.GC_42,(55,0):C.GC_42,(54,0):C.GC_43,(85,0):C.GC_43,(84,0):C.GC_42,(29,12):C.GC_42,(28,12):C.GC_43,(47,12):C.GC_43,(46,12):C.GC_42,(59,12):C.GC_42,(58,12):C.GC_43})

V_14 = Vertex(name = 'V_14',
              particles = [ P.A, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVV1 ],
              couplings = {(0,0):C.GC_62})

V_15 = Vertex(name = 'V_15',
              particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_46})

V_16 = Vertex(name = 'V_16',
              particles = [ P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_79})

V_17 = Vertex(name = 'V_17',
              particles = [ P.A, P.A, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV3 ],
              couplings = {(0,0):C.GC_66})

V_18 = Vertex(name = 'V_18',
              particles = [ P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVV1 ],
              couplings = {(0,0):C.GC_27})

V_19 = Vertex(name = 'V_19',
              particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV3 ],
              couplings = {(0,0):C.GC_28})

V_20 = Vertex(name = 'V_20',
              particles = [ P.A, P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV6 ],
              couplings = {(0,0):C.GC_63})

V_21 = Vertex(name = 'V_21',
              particles = [ P.Z, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_68})

V_22 = Vertex(name = 'V_22',
              particles = [ P.Z, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_84})

V_23 = Vertex(name = 'V_23',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV3 ],
              couplings = {(0,0):C.GC_29})

V_24 = Vertex(name = 'V_24',
              particles = [ P.t__tilde__, P.b, P.W__plus__, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVSS1 ],
              couplings = {(0,0):C.GC_94})

V_25 = Vertex(name = 'V_25',
              particles = [ P.t__tilde__, P.b, P.W__plus__, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVS1, L.FFVS3 ],
              couplings = {(0,0):C.GC_95,(0,1):C.GC_34})

V_26 = Vertex(name = 'V_26',
              particles = [ P.t__tilde__, P.b, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV8 ],
              couplings = {(0,0):[ C.GC_96, C.GC_115 ],(0,1):C.GC_72})

V_27 = Vertex(name = 'V_27',
              particles = [ P.b__tilde__, P.b, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS1 ],
              couplings = {(0,0):C.GC_98})

V_28 = Vertex(name = 'V_28',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_85,(0,1):C.GC_102})

V_29 = Vertex(name = 'V_29',
              particles = [ P.b__tilde__, P.b, P.Z, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVSS1 ],
              couplings = {(0,0):C.GC_99})

V_30 = Vertex(name = 'V_30',
              particles = [ P.b__tilde__, P.b, P.Z, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVS1 ],
              couplings = {(0,0):C.GC_103})

V_31 = Vertex(name = 'V_31',
              particles = [ P.b__tilde__, P.b, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV5 ],
              couplings = {(0,0):[ C.GC_57, C.GC_105 ],(0,1):C.GC_60})

V_32 = Vertex(name = 'V_32',
              particles = [ P.t__tilde__, P.t, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS1 ],
              couplings = {(0,0):C.GC_97})

V_33 = Vertex(name = 'V_33',
              particles = [ P.t__tilde__, P.t, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_91,(0,1):C.GC_101})

V_34 = Vertex(name = 'V_34',
              particles = [ P.t__tilde__, P.t, P.Z, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVSS1 ],
              couplings = {(0,0):C.GC_100})

V_35 = Vertex(name = 'V_35',
              particles = [ P.t__tilde__, P.t, P.Z, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3 ],
              couplings = {(0,0):C.GC_104,(0,1):C.GC_121,(0,2):C.GC_35})

V_36 = Vertex(name = 'V_36',
              particles = [ P.t__tilde__, P.t, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3, L.FFV7, L.FFV8 ],
              couplings = {(0,0):[ C.GC_58, C.GC_106 ],(0,2):C.GC_60,(0,1):C.GC_127,(0,3):C.GC_73})

V_37 = Vertex(name = 'V_37',
              particles = [ P.b__tilde__, P.t, P.W__minus__, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVSS1 ],
              couplings = {(0,0):C.GC_59})

V_38 = Vertex(name = 'V_38',
              particles = [ P.b__tilde__, P.t, P.W__minus__, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVS1, L.FFVS2 ],
              couplings = {(0,0):C.GC_80,(0,1):C.GC_120})

V_39 = Vertex(name = 'V_39',
              particles = [ P.b__tilde__, P.t, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              couplings = {(0,0):[ C.GC_56, C.GC_83 ],(0,1):C.GC_126})

V_40 = Vertex(name = 'V_40',
              particles = [ P.s__tilde__, P.c, P.Tri__tilde__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_26})

V_41 = Vertex(name = 'V_41',
              particles = [ P.d__tilde__, P.u, P.Tri__tilde__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_26})

V_42 = Vertex(name = 'V_42',
              particles = [ P.c__tilde__, P.c, P.Tri0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_25})

V_43 = Vertex(name = 'V_43',
              particles = [ P.d__tilde__, P.d, P.Tri0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_24})

V_44 = Vertex(name = 'V_44',
              particles = [ P.s__tilde__, P.s, P.Tri0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_24})

V_45 = Vertex(name = 'V_45',
              particles = [ P.u__tilde__, P.u, P.Tri0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_25})

V_46 = Vertex(name = 'V_46',
              particles = [ P.u__tilde__, P.d, P.Tri ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_26})

V_47 = Vertex(name = 'V_47',
              particles = [ P.c__tilde__, P.s, P.Tri ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_26})

V_48 = Vertex(name = 'V_48',
              particles = [ P.b__tilde__, P.t, P.Tri__tilde__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_17})

V_49 = Vertex(name = 'V_49',
              particles = [ P.b__tilde__, P.b, P.Tri0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_15})

V_50 = Vertex(name = 'V_50',
              particles = [ P.t__tilde__, P.t, P.Tri0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_16})

V_51 = Vertex(name = 'V_51',
              particles = [ P.t__tilde__, P.b, P.Tri ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_17})

V_52 = Vertex(name = 'V_52',
              particles = [ P.c__tilde__, P.c, P.V8t ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV4 ],
              couplings = {(0,0):C.GC_13,(0,1):C.GC_14})

V_53 = Vertex(name = 'V_53',
              particles = [ P.d__tilde__, P.d, P.V8t ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV4 ],
              couplings = {(0,0):C.GC_13,(0,1):C.GC_12})

V_54 = Vertex(name = 'V_54',
              particles = [ P.s__tilde__, P.s, P.V8t ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV4 ],
              couplings = {(0,0):C.GC_13,(0,1):C.GC_12})

V_55 = Vertex(name = 'V_55',
              particles = [ P.u__tilde__, P.u, P.V8t ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV4 ],
              couplings = {(0,0):C.GC_13,(0,1):C.GC_14})

V_56 = Vertex(name = 'V_56',
              particles = [ P.t__tilde__, P.t, P.V8t ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV4 ],
              couplings = {(0,0):C.GC_11})

V_57 = Vertex(name = 'V_57',
              particles = [ P.b__tilde__, P.b, P.V8Q ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_7})

V_58 = Vertex(name = 'V_58',
              particles = [ P.c__tilde__, P.c, P.V8Q ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV4 ],
              couplings = {(0,0):C.GC_9,(0,1):C.GC_10})

V_59 = Vertex(name = 'V_59',
              particles = [ P.d__tilde__, P.d, P.V8Q ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV4 ],
              couplings = {(0,0):C.GC_9,(0,1):C.GC_8})

V_60 = Vertex(name = 'V_60',
              particles = [ P.s__tilde__, P.s, P.V8Q ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV4 ],
              couplings = {(0,0):C.GC_9,(0,1):C.GC_8})

V_61 = Vertex(name = 'V_61',
              particles = [ P.t__tilde__, P.t, P.V8Q ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_7})

V_62 = Vertex(name = 'V_62',
              particles = [ P.u__tilde__, P.u, P.V8Q ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2, L.FFV4 ],
              couplings = {(0,0):C.GC_9,(0,1):C.GC_10})

V_63 = Vertex(name = 'V_63',
              particles = [ P.s__tilde__, P.c, P.Tri8__tilde__ ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_23})

V_64 = Vertex(name = 'V_64',
              particles = [ P.d__tilde__, P.u, P.Tri8__tilde__ ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_23})

V_65 = Vertex(name = 'V_65',
              particles = [ P.c__tilde__, P.c, P.Tri80 ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_22})

V_66 = Vertex(name = 'V_66',
              particles = [ P.d__tilde__, P.d, P.Tri80 ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_21})

V_67 = Vertex(name = 'V_67',
              particles = [ P.s__tilde__, P.s, P.Tri80 ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_21})

V_68 = Vertex(name = 'V_68',
              particles = [ P.u__tilde__, P.u, P.Tri80 ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_22})

V_69 = Vertex(name = 'V_69',
              particles = [ P.u__tilde__, P.d, P.Tri8 ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_23})

V_70 = Vertex(name = 'V_70',
              particles = [ P.c__tilde__, P.s, P.Tri8 ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_23})

V_71 = Vertex(name = 'V_71',
              particles = [ P.b__tilde__, P.t, P.Tri8__tilde__ ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_20})

V_72 = Vertex(name = 'V_72',
              particles = [ P.b__tilde__, P.b, P.Tri80 ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_18})

V_73 = Vertex(name = 'V_73',
              particles = [ P.t__tilde__, P.t, P.Tri80 ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_19})

V_74 = Vertex(name = 'V_74',
              particles = [ P.t__tilde__, P.b, P.Tri8 ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_20})

V_75 = Vertex(name = 'V_75',
              particles = [ P.t__tilde__, P.t, P.A, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVS2, L.FFVS3 ],
              couplings = {(0,0):C.GC_124,(0,1):C.GC_64})

V_76 = Vertex(name = 'V_76',
              particles = [ P.t__tilde__, P.t, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1, L.FFV3, L.FFV8 ],
              couplings = {(0,0):C.GC_2,(0,1):C.GC_130,(0,2):C.GC_81})

V_77 = Vertex(name = 'V_77',
              particles = [ P.t__tilde__, P.b, P.A, P.W__plus__, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVVS2 ],
              couplings = {(0,0):C.GC_65})

V_78 = Vertex(name = 'V_78',
              particles = [ P.t__tilde__, P.b, P.A, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVV2 ],
              couplings = {(0,0):C.GC_82})

V_79 = Vertex(name = 'V_79',
              particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVVS1, L.FFVVS2 ],
              couplings = {(0,0):C.GC_122,(0,1):C.GC_44})

V_80 = Vertex(name = 'V_80',
              particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVV1, L.FFVV2 ],
              couplings = {(0,0):C.GC_128,(0,1):C.GC_77})

V_81 = Vertex(name = 'V_81',
              particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVVS2 ],
              couplings = {(0,0):C.GC_45})

V_82 = Vertex(name = 'V_82',
              particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVV2 ],
              couplings = {(0,0):C.GC_78})

V_83 = Vertex(name = 'V_83',
              particles = [ P.t__tilde__, P.t, P.G, P.H ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFVS2, L.FFVS3 ],
              couplings = {(0,0):C.GC_116,(0,1):C.GC_33})

V_84 = Vertex(name = 'V_84',
              particles = [ P.t__tilde__, P.t, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1, L.FFV3, L.FFV8 ],
              couplings = {(0,0):C.GC_5,(0,1):C.GC_118,(0,2):C.GC_71})

V_85 = Vertex(name = 'V_85',
              particles = [ P.t__tilde__, P.t, P.G, P.G, P.H ],
              color = [ 'f(3,4,-1)*T(-1,2,1)' ],
              lorentz = [ L.FFVVS1, L.FFVVS2 ],
              couplings = {(0,0):C.GC_117,(0,1):C.GC_38})

V_86 = Vertex(name = 'V_86',
              particles = [ P.t__tilde__, P.t, P.G, P.G ],
              color = [ 'f(3,4,-1)*T(-1,2,1)' ],
              lorentz = [ L.FFVV1, L.FFVV2 ],
              couplings = {(0,0):C.GC_119,(0,1):C.GC_75})

V_87 = Vertex(name = 'V_87',
              particles = [ P.b__tilde__, P.t, P.A, P.W__minus__, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVVS1 ],
              couplings = {(0,0):C.GC_125})

V_88 = Vertex(name = 'V_88',
              particles = [ P.b__tilde__, P.t, P.A, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVV1 ],
              couplings = {(0,0):C.GC_131})

V_89 = Vertex(name = 'V_89',
              particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVVS1 ],
              couplings = {(0,0):C.GC_123})

V_90 = Vertex(name = 'V_90',
              particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFVV1 ],
              couplings = {(0,0):C.GC_129})

V_91 = Vertex(name = 'V_91',
              particles = [ P.d__tilde__, P.d, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_92 = Vertex(name = 'V_92',
              particles = [ P.s__tilde__, P.s, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_93 = Vertex(name = 'V_93',
              particles = [ P.b__tilde__, P.b, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_94 = Vertex(name = 'V_94',
              particles = [ P.e__plus__, P.e__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_95 = Vertex(name = 'V_95',
              particles = [ P.m__plus__, P.m__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_96 = Vertex(name = 'V_96',
              particles = [ P.tt__plus__, P.tt__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_97 = Vertex(name = 'V_97',
              particles = [ P.u__tilde__, P.u, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_98 = Vertex(name = 'V_98',
              particles = [ P.c__tilde__, P.c, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_99 = Vertex(name = 'V_99',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_5})

V_100 = Vertex(name = 'V_100',
               particles = [ P.s__tilde__, P.s, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_5})

V_101 = Vertex(name = 'V_101',
               particles = [ P.b__tilde__, P.b, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_5})

V_102 = Vertex(name = 'V_102',
               particles = [ P.d__tilde__, P.d, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_87})

V_103 = Vertex(name = 'V_103',
               particles = [ P.s__tilde__, P.s, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_90})

V_104 = Vertex(name = 'V_104',
               particles = [ P.d__tilde__, P.d, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV5 ],
               couplings = {(0,0):C.GC_57,(0,1):C.GC_60})

V_105 = Vertex(name = 'V_105',
               particles = [ P.s__tilde__, P.s, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV5 ],
               couplings = {(0,0):C.GC_57,(0,1):C.GC_60})

V_106 = Vertex(name = 'V_106',
               particles = [ P.d__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_48})

V_107 = Vertex(name = 'V_107',
               particles = [ P.d__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_51})

V_108 = Vertex(name = 'V_108',
               particles = [ P.d__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_54})

V_109 = Vertex(name = 'V_109',
               particles = [ P.s__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_49})

V_110 = Vertex(name = 'V_110',
               particles = [ P.s__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_52})

V_111 = Vertex(name = 'V_111',
               particles = [ P.s__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_55})

V_112 = Vertex(name = 'V_112',
               particles = [ P.b__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_50})

V_113 = Vertex(name = 'V_113',
               particles = [ P.b__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_53})

V_114 = Vertex(name = 'V_114',
               particles = [ P.u__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_107})

V_115 = Vertex(name = 'V_115',
               particles = [ P.c__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_110})

V_116 = Vertex(name = 'V_116',
               particles = [ P.t__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_113})

V_117 = Vertex(name = 'V_117',
               particles = [ P.u__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_108})

V_118 = Vertex(name = 'V_118',
               particles = [ P.c__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_111})

V_119 = Vertex(name = 'V_119',
               particles = [ P.t__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_114})

V_120 = Vertex(name = 'V_120',
               particles = [ P.u__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_109})

V_121 = Vertex(name = 'V_121',
               particles = [ P.c__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_112})

V_122 = Vertex(name = 'V_122',
               particles = [ P.u__tilde__, P.u, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_5})

V_123 = Vertex(name = 'V_123',
               particles = [ P.c__tilde__, P.c, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_5})

V_124 = Vertex(name = 'V_124',
               particles = [ P.e__plus__, P.e__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_88})

V_125 = Vertex(name = 'V_125',
               particles = [ P.m__plus__, P.m__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_89})

V_126 = Vertex(name = 'V_126',
               particles = [ P.tt__plus__, P.tt__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_92})

V_127 = Vertex(name = 'V_127',
               particles = [ P.u__tilde__, P.u, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_93})

V_128 = Vertex(name = 'V_128',
               particles = [ P.c__tilde__, P.c, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_86})

V_129 = Vertex(name = 'V_129',
               particles = [ P.e__plus__, P.e__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV6 ],
               couplings = {(0,0):C.GC_57,(0,1):C.GC_61})

V_130 = Vertex(name = 'V_130',
               particles = [ P.m__plus__, P.m__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV6 ],
               couplings = {(0,0):C.GC_57,(0,1):C.GC_61})

V_131 = Vertex(name = 'V_131',
               particles = [ P.tt__plus__, P.tt__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV6 ],
               couplings = {(0,0):C.GC_57,(0,1):C.GC_61})

V_132 = Vertex(name = 'V_132',
               particles = [ P.e__plus__, P.ve, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_47})

V_133 = Vertex(name = 'V_133',
               particles = [ P.m__plus__, P.vm, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_47})

V_134 = Vertex(name = 'V_134',
               particles = [ P.tt__plus__, P.vt, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_47})

V_135 = Vertex(name = 'V_135',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_47})

V_136 = Vertex(name = 'V_136',
               particles = [ P.vm__tilde__, P.m__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_47})

V_137 = Vertex(name = 'V_137',
               particles = [ P.vt__tilde__, P.tt__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_47})

V_138 = Vertex(name = 'V_138',
               particles = [ P.u__tilde__, P.u, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV7 ],
               couplings = {(0,0):C.GC_58,(0,1):C.GC_60})

V_139 = Vertex(name = 'V_139',
               particles = [ P.c__tilde__, P.c, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV7 ],
               couplings = {(0,0):C.GC_58,(0,1):C.GC_60})

V_140 = Vertex(name = 'V_140',
               particles = [ P.ve__tilde__, P.ve, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_67})

V_141 = Vertex(name = 'V_141',
               particles = [ P.vm__tilde__, P.vm, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_67})

V_142 = Vertex(name = 'V_142',
               particles = [ P.vt__tilde__, P.vt, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_67})

