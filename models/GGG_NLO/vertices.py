# This file was automatically created by FeynRules 2.4.54
# Mathematica version: 11.0.0 for Linux x86 (64-bit) (July 28, 2016)
# Date: Tue 25 Oct 2016 14:05:33


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.G0, P.G0, P.G0, P.G0 ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_19})

V_2 = Vertex(name = 'V_2',
             particles = [ P.G0, P.G0, P.G__minus__, P.G__plus__ ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_17})

V_3 = Vertex(name = 'V_3',
             particles = [ P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_18})

V_4 = Vertex(name = 'V_4',
             particles = [ P.G0, P.G0, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_17})

V_5 = Vertex(name = 'V_5',
             particles = [ P.G__minus__, P.G__plus__, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_17})

V_6 = Vertex(name = 'V_6',
             particles = [ P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_19})

V_7 = Vertex(name = 'V_7',
             particles = [ P.G0, P.G0, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_62})

V_8 = Vertex(name = 'V_8',
             particles = [ P.G__minus__, P.G__plus__, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_62})

V_9 = Vertex(name = 'V_9',
             particles = [ P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_63})

V_10 = Vertex(name = 'V_10',
              particles = [ P.a, P.a, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_6})

V_11 = Vertex(name = 'V_11',
              particles = [ P.a, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_3})

V_12 = Vertex(name = 'V_12',
              particles = [ P.ghA, P.ghWm__tilde__, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_3})

V_13 = Vertex(name = 'V_13',
              particles = [ P.ghA, P.ghWp__tilde__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_4})

V_14 = Vertex(name = 'V_14',
              particles = [ P.ghWm, P.ghA__tilde__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_69})

V_15 = Vertex(name = 'V_15',
              particles = [ P.ghWm, P.ghA__tilde__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_3})

V_16 = Vertex(name = 'V_16',
              particles = [ P.ghWm, P.ghWm__tilde__, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_64})

V_17 = Vertex(name = 'V_17',
              particles = [ P.ghWm, P.ghWm__tilde__, P.H ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_65})

V_18 = Vertex(name = 'V_18',
              particles = [ P.ghWm, P.ghWm__tilde__, P.a ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_4})

V_19 = Vertex(name = 'V_19',
              particles = [ P.ghWm, P.ghWm__tilde__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_43})

V_20 = Vertex(name = 'V_20',
              particles = [ P.ghWm, P.ghZ__tilde__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_72})

V_21 = Vertex(name = 'V_21',
              particles = [ P.ghWm, P.ghZ__tilde__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_42})

V_22 = Vertex(name = 'V_22',
              particles = [ P.ghWp, P.ghA__tilde__, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_68})

V_23 = Vertex(name = 'V_23',
              particles = [ P.ghWp, P.ghA__tilde__, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_4})

V_24 = Vertex(name = 'V_24',
              particles = [ P.ghWp, P.ghWp__tilde__, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_67})

V_25 = Vertex(name = 'V_25',
              particles = [ P.ghWp, P.ghWp__tilde__, P.H ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_65})

V_26 = Vertex(name = 'V_26',
              particles = [ P.ghWp, P.ghWp__tilde__, P.a ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_3})

V_27 = Vertex(name = 'V_27',
              particles = [ P.ghWp, P.ghWp__tilde__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_42})

V_28 = Vertex(name = 'V_28',
              particles = [ P.ghWp, P.ghZ__tilde__, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_71})

V_29 = Vertex(name = 'V_29',
              particles = [ P.ghWp, P.ghZ__tilde__, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_43})

V_30 = Vertex(name = 'V_30',
              particles = [ P.ghZ, P.ghWm__tilde__, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_73})

V_31 = Vertex(name = 'V_31',
              particles = [ P.ghZ, P.ghWm__tilde__, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_42})

V_32 = Vertex(name = 'V_32',
              particles = [ P.ghZ, P.ghWp__tilde__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_70})

V_33 = Vertex(name = 'V_33',
              particles = [ P.ghZ, P.ghWp__tilde__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_43})

V_34 = Vertex(name = 'V_34',
              particles = [ P.ghZ, P.ghZ__tilde__, P.H ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_74})

V_35 = Vertex(name = 'V_35',
              particles = [ P.g, P.g, P.g ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVV2, L.VVV3, L.VVV5 ],
              couplings = {(0,2):C.GC_20,(0,1):C.GC_28,(0,0):C.GC_10})

V_36 = Vertex(name = 'V_36',
              particles = [ P.ghG, P.ghG__tilde__, P.g ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_10})

V_37 = Vertex(name = 'V_37',
              particles = [ P.g, P.g, P.g, P.g ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVV101, L.VVVV102, L.VVVV69, L.VVVV90, L.VVVV92, L.VVVV93, L.VVVV96, L.VVVV97, L.VVVV98 ],
              couplings = {(2,0):C.GC_22,(0,3):C.GC_29,(1,1):C.GC_22,(0,8):C.GC_21,(1,7):C.GC_29,(2,6):C.GC_29,(1,4):C.GC_12,(0,2):C.GC_12,(2,5):C.GC_12})

V_38 = Vertex(name = 'V_38',
              particles = [ P.g, P.g, P.g, P.g, P.g ],
              color = [ 'f(-2,1,2)*f(-1,-2,3)*f(4,5,-1)', 'f(-2,1,2)*f(-1,-2,4)*f(3,5,-1)', 'f(-2,1,2)*f(-1,-2,5)*f(3,4,-1)', 'f(-2,1,3)*f(-1,-2,2)*f(4,5,-1)', 'f(-2,1,3)*f(-1,-2,4)*f(2,5,-1)', 'f(-2,1,3)*f(-1,-2,5)*f(2,4,-1)', 'f(-2,1,4)*f(-1,-2,2)*f(3,5,-1)', 'f(-2,1,4)*f(-1,-2,3)*f(2,5,-1)', 'f(-2,1,4)*f(-1,-2,5)*f(2,3,-1)', 'f(-2,1,5)*f(-1,-2,2)*f(3,4,-1)', 'f(-2,1,5)*f(-1,-2,3)*f(2,4,-1)', 'f(-2,1,5)*f(-1,-2,4)*f(2,3,-1)', 'f(-2,2,3)*f(-1,-2,1)*f(4,5,-1)', 'f(-2,2,3)*f(-1,-2,4)*f(1,5,-1)', 'f(-2,2,3)*f(-1,-2,5)*f(1,4,-1)', 'f(-2,2,4)*f(-1,-2,1)*f(3,5,-1)', 'f(-2,2,4)*f(-1,-2,3)*f(1,5,-1)', 'f(-2,2,4)*f(-1,-2,5)*f(1,3,-1)', 'f(-2,2,5)*f(-1,-2,1)*f(3,4,-1)', 'f(-2,2,5)*f(-1,-2,3)*f(1,4,-1)', 'f(-2,2,5)*f(-1,-2,4)*f(1,3,-1)', 'f(-2,3,4)*f(-1,-2,1)*f(2,5,-1)', 'f(-2,3,4)*f(-1,-2,2)*f(1,5,-1)', 'f(-2,3,4)*f(-1,-2,5)*f(1,2,-1)', 'f(-2,3,5)*f(-1,-2,1)*f(2,4,-1)', 'f(-2,3,5)*f(-1,-2,2)*f(1,4,-1)', 'f(-2,3,5)*f(-1,-2,4)*f(1,2,-1)', 'f(-2,4,5)*f(-1,-2,1)*f(2,3,-1)', 'f(-2,4,5)*f(-1,-2,2)*f(1,3,-1)', 'f(-2,4,5)*f(-1,-2,3)*f(1,2,-1)' ],
              lorentz = [ L.VVVVV1, L.VVVVV10, L.VVVVV11, L.VVVVV12, L.VVVVV13, L.VVVVV14, L.VVVVV15, L.VVVVV16, L.VVVVV17, L.VVVVV18, L.VVVVV19, L.VVVVV2, L.VVVVV20, L.VVVVV21, L.VVVVV22, L.VVVVV23, L.VVVVV24, L.VVVVV25, L.VVVVV26, L.VVVVV27, L.VVVVV28, L.VVVVV29, L.VVVVV3, L.VVVVV30, L.VVVVV31, L.VVVVV32, L.VVVVV33, L.VVVVV34, L.VVVVV35, L.VVVVV36, L.VVVVV37, L.VVVVV38, L.VVVVV39, L.VVVVV4, L.VVVVV40, L.VVVVV41, L.VVVVV42, L.VVVVV43, L.VVVVV44, L.VVVVV45, L.VVVVV5, L.VVVVV6, L.VVVVV7, L.VVVVV8, L.VVVVV9 ],
              couplings = {(24,8):C.GC_31,(10,36):C.GC_25,(21,9):C.GC_30,(7,32):C.GC_24,(18,9):C.GC_31,(9,37):C.GC_25,(15,8):C.GC_30,(6,31):C.GC_24,(28,41):C.GC_31,(20,33):C.GC_25,(22,21):C.GC_31,(13,28):C.GC_24,(18,17):C.GC_23,(9,21):C.GC_30,(12,22):C.GC_24,(3,41):C.GC_30,(29,42):C.GC_31,(26,3):C.GC_25,(16,25):C.GC_31,(24,6):C.GC_23,(10,25):C.GC_30,(0,42):C.GC_30,(29,39):C.GC_23,(26,30):C.GC_30,(28,38):C.GC_23,(20,29):C.GC_30,(4,29):C.GC_31,(1,30):C.GC_31,(17,44):C.GC_25,(25,20):C.GC_31,(14,27):C.GC_24,(15,12):C.GC_23,(6,20):C.GC_30,(23,0):C.GC_25,(19,26):C.GC_31,(21,4):C.GC_23,(7,26):C.GC_30,(23,35):C.GC_30,(17,34):C.GC_30,(5,34):C.GC_31,(2,35):C.GC_31,(27,11):C.GC_31,(11,2):C.GC_25,(4,24):C.GC_24,(12,11):C.GC_30,(3,43):C.GC_24,(19,14):C.GC_25,(16,13):C.GC_24,(25,7):C.GC_23,(13,10):C.GC_31,(27,15):C.GC_23,(11,10):C.GC_30,(14,16):C.GC_30,(8,16):C.GC_31,(8,1):C.GC_25,(5,23):C.GC_24,(22,5):C.GC_23,(1,18):C.GC_24,(0,40):C.GC_24,(2,19):C.GC_24})

V_39 = Vertex(name = 'V_39',
              particles = [ P.g, P.g, P.g, P.g, P.g, P.g ],
              color = [ 'f(-2,-3,3)*f(-2,2,4)*f(-1,-3,5)*f(1,6,-1)', 'f(-2,-3,3)*f(-2,2,4)*f(-1,-3,6)*f(1,5,-1)', 'f(-2,-3,3)*f(-2,2,5)*f(-1,-3,4)*f(1,6,-1)', 'f(-2,-3,3)*f(-2,2,5)*f(-1,-3,6)*f(1,4,-1)', 'f(-2,-3,3)*f(-2,2,6)*f(-1,-3,4)*f(1,5,-1)', 'f(-2,-3,3)*f(-2,2,6)*f(-1,-3,5)*f(1,4,-1)', 'f(-2,-3,4)*f(-2,2,3)*f(-1,-3,5)*f(1,6,-1)', 'f(-2,-3,4)*f(-2,2,3)*f(-1,-3,6)*f(1,5,-1)', 'f(-2,-3,4)*f(-2,2,5)*f(-1,-3,3)*f(1,6,-1)', 'f(-2,-3,4)*f(-2,2,5)*f(-1,-3,6)*f(1,3,-1)', 'f(-2,-3,4)*f(-2,2,6)*f(-1,-3,3)*f(1,5,-1)', 'f(-2,-3,4)*f(-2,2,6)*f(-1,-3,5)*f(1,3,-1)', 'f(-2,-3,4)*f(-2,3,5)*f(-1,-3,6)*f(1,2,-1)', 'f(-2,-3,4)*f(-2,3,6)*f(-1,-3,5)*f(1,2,-1)', 'f(-2,-3,5)*f(-2,2,3)*f(-1,-3,4)*f(1,6,-1)', 'f(-2,-3,5)*f(-2,2,3)*f(-1,-3,6)*f(1,4,-1)', 'f(-2,-3,5)*f(-2,2,4)*f(-1,-3,3)*f(1,6,-1)', 'f(-2,-3,5)*f(-2,2,4)*f(-1,-3,6)*f(1,3,-1)', 'f(-2,-3,5)*f(-2,2,6)*f(-1,-3,3)*f(1,4,-1)', 'f(-2,-3,5)*f(-2,2,6)*f(-1,-3,4)*f(1,3,-1)', 'f(-2,-3,5)*f(-2,3,4)*f(-1,-3,6)*f(1,2,-1)', 'f(-2,-3,5)*f(-2,3,6)*f(-1,-3,4)*f(1,2,-1)', 'f(-2,-3,6)*f(-2,2,3)*f(-1,-3,4)*f(1,5,-1)', 'f(-2,-3,6)*f(-2,2,3)*f(-1,-3,5)*f(1,4,-1)', 'f(-2,-3,6)*f(-2,2,4)*f(-1,-3,3)*f(1,5,-1)', 'f(-2,-3,6)*f(-2,2,4)*f(-1,-3,5)*f(1,3,-1)', 'f(-2,-3,6)*f(-2,2,5)*f(-1,-3,3)*f(1,4,-1)', 'f(-2,-3,6)*f(-2,2,5)*f(-1,-3,4)*f(1,3,-1)', 'f(-2,-3,6)*f(-2,3,4)*f(-1,-3,5)*f(1,2,-1)', 'f(-2,-3,6)*f(-2,3,5)*f(-1,-3,4)*f(1,2,-1)', 'f(-3,1,2)*f(-2,3,4)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,1,2)*f(-2,3,5)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,1,2)*f(-2,3,6)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,1,2)*f(-2,4,5)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,1,2)*f(-2,4,6)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,1,2)*f(-2,5,6)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,1,3)*f(-2,2,4)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,1,3)*f(-2,2,5)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,1,3)*f(-2,2,6)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,1,3)*f(-2,4,5)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,1,3)*f(-2,4,6)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,1,3)*f(-2,5,6)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,1,4)*f(-2,2,3)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,1,4)*f(-2,2,5)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,1,4)*f(-2,2,6)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,1,4)*f(-2,3,5)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,1,4)*f(-2,3,6)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,1,4)*f(-2,5,6)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,1,5)*f(-2,2,3)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,1,5)*f(-2,2,4)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,1,5)*f(-2,2,6)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,1,5)*f(-2,3,4)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,1,5)*f(-2,3,6)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,1,5)*f(-2,4,6)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,1,6)*f(-2,2,3)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,1,6)*f(-2,2,4)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,1,6)*f(-2,2,5)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,1,6)*f(-2,3,4)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,1,6)*f(-2,3,5)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,1,6)*f(-2,4,5)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,2,3)*f(-2,1,4)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,2,3)*f(-2,1,5)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,2,3)*f(-2,1,6)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,2,3)*f(-2,-3,4)*f(-1,-2,1)*f(5,6,-1)', 'f(-3,2,3)*f(-2,-3,5)*f(-1,-2,1)*f(4,6,-1)', 'f(-3,2,3)*f(-2,-3,6)*f(-1,-2,1)*f(4,5,-1)', 'f(-3,2,3)*f(-2,4,5)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,2,3)*f(-2,4,6)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,2,3)*f(-2,5,6)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,2,4)*f(-2,1,3)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,2,4)*f(-2,1,5)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,2,4)*f(-2,1,6)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,2,4)*f(-2,-3,3)*f(-1,-2,1)*f(5,6,-1)', 'f(-3,2,4)*f(-2,-3,5)*f(-1,-2,1)*f(3,6,-1)', 'f(-3,2,4)*f(-2,3,5)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,2,4)*f(-2,-3,6)*f(-1,-2,1)*f(3,5,-1)', 'f(-3,2,4)*f(-2,3,6)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,2,4)*f(-2,5,6)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,2,5)*f(-2,1,3)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,2,5)*f(-2,1,4)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,2,5)*f(-2,1,6)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,2,5)*f(-2,-3,3)*f(-1,-2,1)*f(4,6,-1)', 'f(-3,2,5)*f(-2,-3,4)*f(-1,-2,1)*f(3,6,-1)', 'f(-3,2,5)*f(-2,3,4)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,2,5)*f(-2,-3,6)*f(-1,-2,1)*f(3,4,-1)', 'f(-3,2,5)*f(-2,3,6)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,2,5)*f(-2,4,6)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,2,6)*f(-2,1,3)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,2,6)*f(-2,1,4)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,2,6)*f(-2,1,5)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,2,6)*f(-2,-3,3)*f(-1,-2,1)*f(4,5,-1)', 'f(-3,2,6)*f(-2,-3,4)*f(-1,-2,1)*f(3,5,-1)', 'f(-3,2,6)*f(-2,3,4)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,2,6)*f(-2,-3,5)*f(-1,-2,1)*f(3,4,-1)', 'f(-3,2,6)*f(-2,3,5)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,2,6)*f(-2,4,5)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,3,4)*f(-2,1,2)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,3,4)*f(-2,1,5)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,3,4)*f(-2,1,6)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,3,4)*f(-2,2,5)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,3,4)*f(-2,2,6)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,3,4)*f(-2,-3,2)*f(-1,-2,1)*f(5,6,-1)', 'f(-3,3,4)*f(-2,-3,2)*f(-1,-2,5)*f(1,6,-1)', 'f(-3,3,4)*f(-2,-3,2)*f(-1,-2,6)*f(1,5,-1)', 'f(-3,3,4)*f(-2,-3,5)*f(-1,-2,1)*f(2,6,-1)', 'f(-3,3,4)*f(-2,-3,5)*f(-1,-2,2)*f(1,6,-1)', 'f(-3,3,4)*f(-2,-3,6)*f(-1,-2,1)*f(2,5,-1)', 'f(-3,3,4)*f(-2,-3,6)*f(-1,-2,2)*f(1,5,-1)', 'f(-3,3,4)*f(-2,5,6)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,3,5)*f(-2,1,2)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,3,5)*f(-2,1,4)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,3,5)*f(-2,1,6)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,3,5)*f(-2,2,4)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,3,5)*f(-2,2,6)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,3,5)*f(-2,-3,2)*f(-1,-2,1)*f(4,6,-1)', 'f(-3,3,5)*f(-2,-3,2)*f(-1,-2,4)*f(1,6,-1)', 'f(-3,3,5)*f(-2,-3,2)*f(-1,-2,6)*f(1,4,-1)', 'f(-3,3,5)*f(-2,-3,4)*f(-1,-2,1)*f(2,6,-1)', 'f(-3,3,5)*f(-2,-3,4)*f(-1,-2,2)*f(1,6,-1)', 'f(-3,3,5)*f(-2,-3,6)*f(-1,-2,1)*f(2,4,-1)', 'f(-3,3,5)*f(-2,-3,6)*f(-1,-2,2)*f(1,4,-1)', 'f(-3,3,5)*f(-2,4,6)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,3,6)*f(-2,1,2)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,3,6)*f(-2,1,4)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,3,6)*f(-2,1,5)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,3,6)*f(-2,2,4)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,3,6)*f(-2,2,5)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,3,6)*f(-2,-3,2)*f(-1,-2,1)*f(4,5,-1)', 'f(-3,3,6)*f(-2,-3,2)*f(-1,-2,4)*f(1,5,-1)', 'f(-3,3,6)*f(-2,-3,2)*f(-1,-2,5)*f(1,4,-1)', 'f(-3,3,6)*f(-2,-3,4)*f(-1,-2,1)*f(2,5,-1)', 'f(-3,3,6)*f(-2,-3,4)*f(-1,-2,2)*f(1,5,-1)', 'f(-3,3,6)*f(-2,-3,5)*f(-1,-2,1)*f(2,4,-1)', 'f(-3,3,6)*f(-2,-3,5)*f(-1,-2,2)*f(1,4,-1)', 'f(-3,3,6)*f(-2,4,5)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,4,5)*f(-2,1,2)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,4,5)*f(-2,1,3)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,4,5)*f(-2,1,6)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,4,5)*f(-2,2,3)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,4,5)*f(-2,2,6)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,4,5)*f(-2,-3,2)*f(-1,-2,1)*f(3,6,-1)', 'f(-3,4,5)*f(-2,-3,2)*f(-1,-2,3)*f(1,6,-1)', 'f(-3,4,5)*f(-2,-3,2)*f(-1,-2,6)*f(1,3,-1)', 'f(-3,4,5)*f(-2,-3,3)*f(-1,-2,1)*f(2,6,-1)', 'f(-3,4,5)*f(-2,-3,3)*f(-1,-2,2)*f(1,6,-1)', 'f(-3,4,5)*f(-2,-3,3)*f(-1,-2,6)*f(1,2,-1)', 'f(-3,4,5)*f(-2,-3,6)*f(-1,-2,1)*f(2,3,-1)', 'f(-3,4,5)*f(-2,-3,6)*f(-1,-2,2)*f(1,3,-1)', 'f(-3,4,5)*f(-2,-3,6)*f(-1,-2,3)*f(1,2,-1)', 'f(-3,4,5)*f(-2,3,6)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,4,6)*f(-2,1,2)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,4,6)*f(-2,1,3)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,4,6)*f(-2,1,5)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,4,6)*f(-2,2,3)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,4,6)*f(-2,2,5)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,4,6)*f(-2,-3,2)*f(-1,-2,1)*f(3,5,-1)', 'f(-3,4,6)*f(-2,-3,2)*f(-1,-2,3)*f(1,5,-1)', 'f(-3,4,6)*f(-2,-3,2)*f(-1,-2,5)*f(1,3,-1)', 'f(-3,4,6)*f(-2,-3,3)*f(-1,-2,1)*f(2,5,-1)', 'f(-3,4,6)*f(-2,-3,3)*f(-1,-2,2)*f(1,5,-1)', 'f(-3,4,6)*f(-2,-3,3)*f(-1,-2,5)*f(1,2,-1)', 'f(-3,4,6)*f(-2,-3,5)*f(-1,-2,1)*f(2,3,-1)', 'f(-3,4,6)*f(-2,-3,5)*f(-1,-2,2)*f(1,3,-1)', 'f(-3,4,6)*f(-2,-3,5)*f(-1,-2,3)*f(1,2,-1)', 'f(-3,4,6)*f(-2,3,5)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,5,6)*f(-2,1,2)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,5,6)*f(-2,1,3)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,5,6)*f(-2,1,4)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,5,6)*f(-2,2,3)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,5,6)*f(-2,2,4)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,5,6)*f(-2,-3,2)*f(-1,-2,1)*f(3,4,-1)', 'f(-3,5,6)*f(-2,-3,2)*f(-1,-2,3)*f(1,4,-1)', 'f(-3,5,6)*f(-2,-3,2)*f(-1,-2,4)*f(1,3,-1)', 'f(-3,5,6)*f(-2,-3,3)*f(-1,-2,1)*f(2,4,-1)', 'f(-3,5,6)*f(-2,-3,3)*f(-1,-2,2)*f(1,4,-1)', 'f(-3,5,6)*f(-2,-3,3)*f(-1,-2,4)*f(1,2,-1)', 'f(-3,5,6)*f(-2,-3,4)*f(-1,-2,1)*f(2,3,-1)', 'f(-3,5,6)*f(-2,-3,4)*f(-1,-2,2)*f(1,3,-1)', 'f(-3,5,6)*f(-2,-3,4)*f(-1,-2,3)*f(1,2,-1)', 'f(-3,5,6)*f(-2,3,4)*f(-1,-2,-3)*f(1,2,-1)' ],
              lorentz = [ L.VVVVVV1, L.VVVVVV10, L.VVVVVV100, L.VVVVVV101, L.VVVVVV102, L.VVVVVV103, L.VVVVVV104, L.VVVVVV105, L.VVVVVV11, L.VVVVVV12, L.VVVVVV13, L.VVVVVV14, L.VVVVVV15, L.VVVVVV16, L.VVVVVV17, L.VVVVVV18, L.VVVVVV19, L.VVVVVV2, L.VVVVVV20, L.VVVVVV21, L.VVVVVV22, L.VVVVVV23, L.VVVVVV24, L.VVVVVV25, L.VVVVVV26, L.VVVVVV27, L.VVVVVV28, L.VVVVVV29, L.VVVVVV3, L.VVVVVV30, L.VVVVVV31, L.VVVVVV32, L.VVVVVV33, L.VVVVVV34, L.VVVVVV35, L.VVVVVV36, L.VVVVVV37, L.VVVVVV38, L.VVVVVV39, L.VVVVVV4, L.VVVVVV40, L.VVVVVV41, L.VVVVVV42, L.VVVVVV43, L.VVVVVV44, L.VVVVVV45, L.VVVVVV46, L.VVVVVV47, L.VVVVVV48, L.VVVVVV49, L.VVVVVV5, L.VVVVVV50, L.VVVVVV51, L.VVVVVV52, L.VVVVVV53, L.VVVVVV54, L.VVVVVV55, L.VVVVVV56, L.VVVVVV57, L.VVVVVV58, L.VVVVVV59, L.VVVVVV6, L.VVVVVV60, L.VVVVVV61, L.VVVVVV62, L.VVVVVV63, L.VVVVVV64, L.VVVVVV65, L.VVVVVV66, L.VVVVVV67, L.VVVVVV68, L.VVVVVV69, L.VVVVVV7, L.VVVVVV70, L.VVVVVV71, L.VVVVVV72, L.VVVVVV73, L.VVVVVV74, L.VVVVVV75, L.VVVVVV76, L.VVVVVV77, L.VVVVVV78, L.VVVVVV79, L.VVVVVV8, L.VVVVVV80, L.VVVVVV81, L.VVVVVV82, L.VVVVVV83, L.VVVVVV84, L.VVVVVV85, L.VVVVVV86, L.VVVVVV87, L.VVVVVV88, L.VVVVVV89, L.VVVVVV9, L.VVVVVV90, L.VVVVVV91, L.VVVVVV92, L.VVVVVV93, L.VVVVVV94, L.VVVVVV95, L.VVVVVV96, L.VVVVVV97, L.VVVVVV98, L.VVVVVV99 ],
              couplings = {(121,65):C.GC_33,(134,75):C.GC_32,(149,75):C.GC_33,(164,65):C.GC_32,(160,61):C.GC_27,(145,0):C.GC_26,(13,21):C.GC_26,(12,1):C.GC_27,(77,35):C.GC_33,(95,92):C.GC_33,(139,92):C.GC_32,(169,35):C.GC_32,(172,39):C.GC_27,(142,72):C.GC_27,(19,87):C.GC_26,(17,10):C.GC_27,(68,45):C.GC_33,(94,100):C.GC_33,(113,100):C.GC_32,(168,45):C.GC_32,(171,18):C.GC_27,(116,22):C.GC_27,(18,99):C.GC_26,(15,42):C.GC_27,(67,98):C.GC_32,(76,93):C.GC_32,(125,93):C.GC_33,(153,98):C.GC_33,(131,85):C.GC_27,(159,78):C.GC_27,(7,95):C.GC_26,(1,88):C.GC_26,(47,45):C.GC_32,(53,98):C.GC_33,(152,98):C.GC_32,(167,45):C.GC_33,(41,35):C.GC_32,(52,93):C.GC_33,(124,93):C.GC_32,(166,35):C.GC_33,(39,92):C.GC_32,(45,100):C.GC_32,(110,100):C.GC_33,(136,92):C.GC_33,(117,79):C.GC_27,(143,81):C.GC_27,(34,65):C.GC_32,(44,100):C.GC_33,(88,100):C.GC_32,(150,65):C.GC_33,(33,75):C.GC_33,(49,93):C.GC_32,(70,93):C.GC_33,(135,75):C.GC_32,(73,19):C.GC_27,(140,69):C.GC_26,(32,75):C.GC_32,(38,92):C.GC_33,(87,92):C.GC_32,(122,75):C.GC_33,(31,65):C.GC_33,(48,98):C.GC_32,(61,98):C.GC_33,(109,65):C.GC_32,(64,50):C.GC_27,(114,52):C.GC_26,(36,35):C.GC_33,(42,45):C.GC_33,(60,45):C.GC_32,(69,35):C.GC_32,(63,28):C.GC_26,(72,15):C.GC_26,(86,89):C.GC_33,(154,89):C.GC_32,(157,17):C.GC_27,(27,86):C.GC_26,(25,38):C.GC_27,(85,101):C.GC_33,(126,101):C.GC_32,(129,13):C.GC_27,(26,97):C.GC_26,(23,11):C.GC_27,(66,102):C.GC_32,(74,91):C.GC_32,(112,91):C.GC_33,(138,102):C.GC_33,(118,80):C.GC_27,(144,82):C.GC_27,(6,96):C.GC_26,(0,90):C.GC_26,(59,102):C.GC_33,(137,102):C.GC_32,(58,91):C.GC_33,(111,91):C.GC_32,(40,89):C.GC_32,(46,101):C.GC_32,(123,101):C.GC_33,(151,89):C.GC_33,(130,84):C.GC_27,(158,77):C.GC_27,(55,91):C.GC_32,(71,91):C.GC_33,(75,26):C.GC_27,(155,68):C.GC_26,(43,101):C.GC_33,(79,101):C.GC_32,(54,102):C.GC_32,(62,102):C.GC_33,(65,83):C.GC_27,(127,53):C.GC_26,(37,89):C.GC_33,(78,89):C.GC_32,(108,23):C.GC_33,(179,23):C.GC_32,(175,94):C.GC_27,(21,62):C.GC_26,(20,8):C.GC_27,(11,33):C.GC_26,(9,9):C.GC_27,(133,6):C.GC_27,(174,29):C.GC_27,(3,34):C.GC_26,(92,67):C.GC_33,(100,67):C.GC_32,(156,55):C.GC_27,(103,64):C.GC_27,(10,60):C.GC_26,(51,67):C.GC_32,(97,67):C.GC_33,(104,31):C.GC_27,(35,23):C.GC_32,(50,67):C.GC_33,(89,67):C.GC_32,(165,23):C.GC_33,(82,56):C.GC_27,(81,51):C.GC_26,(30,23):C.GC_33,(96,23):C.GC_32,(101,16):C.GC_26,(128,46):C.GC_27,(24,37):C.GC_26,(22,12):C.GC_27,(83,76):C.GC_32,(99,76):C.GC_33,(105,32):C.GC_27,(14,36):C.GC_26,(2,49):C.GC_26,(132,2):C.GC_27,(173,30):C.GC_27,(57,76):C.GC_33,(98,76):C.GC_32,(56,76):C.GC_32,(80,76):C.GC_33,(84,70):C.GC_27,(170,24):C.GC_26,(29,63):C.GC_26,(28,25):C.GC_27,(120,4):C.GC_27,(5,14):C.GC_26,(141,58):C.GC_27,(102,66):C.GC_27,(8,57):C.GC_26,(106,40):C.GC_27,(91,59):C.GC_27,(90,54):C.GC_26,(107,43):C.GC_27,(4,47):C.GC_26,(115,48):C.GC_27,(16,20):C.GC_26,(119,103):C.GC_27,(93,73):C.GC_27,(163,71):C.GC_27,(178,27):C.GC_27,(162,104):C.GC_27,(177,41):C.GC_27,(161,5):C.GC_27,(176,44):C.GC_27,(148,74):C.GC_27,(147,3):C.GC_27,(146,7):C.GC_27})

V_40 = Vertex(name = 'V_40',
              particles = [ P.t__tilde__, P.b, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS3, L.FFS5 ],
              couplings = {(0,0):C.GC_13,(0,1):C.GC_14})

V_41 = Vertex(name = 'V_41',
              particles = [ P.b__tilde__, P.b, P.G0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4 ],
              couplings = {(0,0):C.GC_76})

V_42 = Vertex(name = 'V_42',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS6 ],
              couplings = {(0,0):C.GC_77})

V_43 = Vertex(name = 'V_43',
              particles = [ P.vt__tilde__, P.ta__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.FFS3 ],
              couplings = {(0,0):C.GC_81})

V_44 = Vertex(name = 'V_44',
              particles = [ P.ta__plus__, P.ta__minus__, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.FFS4 ],
              couplings = {(0,0):C.GC_82})

V_45 = Vertex(name = 'V_45',
              particles = [ P.ta__plus__, P.ta__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS6 ],
              couplings = {(0,0):C.GC_83})

V_46 = Vertex(name = 'V_46',
              particles = [ P.b__tilde__, P.t, P.G__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS3, L.FFS5 ],
              couplings = {(0,0):C.GC_15,(0,1):C.GC_16})

V_47 = Vertex(name = 'V_47',
              particles = [ P.t__tilde__, P.t, P.G0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4 ],
              couplings = {(0,0):C.GC_79})

V_48 = Vertex(name = 'V_48',
              particles = [ P.t__tilde__, P.t, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS6 ],
              couplings = {(0,0):C.GC_78})

V_49 = Vertex(name = 'V_49',
              particles = [ P.a, P.W__minus__, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_45})

V_50 = Vertex(name = 'V_50',
              particles = [ P.a, P.W__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_44})

V_51 = Vertex(name = 'V_51',
              particles = [ P.a, P.W__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_68})

V_52 = Vertex(name = 'V_52',
              particles = [ P.W__minus__, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_39})

V_53 = Vertex(name = 'V_53',
              particles = [ P.W__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_37})

V_54 = Vertex(name = 'V_54',
              particles = [ P.a, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVV2 ],
              couplings = {(0,0):C.GC_4})

V_55 = Vertex(name = 'V_55',
              particles = [ P.a, P.W__plus__, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_45})

V_56 = Vertex(name = 'V_56',
              particles = [ P.a, P.W__plus__, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_46})

V_57 = Vertex(name = 'V_57',
              particles = [ P.a, P.W__plus__, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_69})

V_58 = Vertex(name = 'V_58',
              particles = [ P.W__plus__, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_38})

V_59 = Vertex(name = 'V_59',
              particles = [ P.W__plus__, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_37})

V_60 = Vertex(name = 'V_60',
              particles = [ P.W__minus__, P.W__plus__, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_34})

V_61 = Vertex(name = 'V_61',
              particles = [ P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_34})

V_62 = Vertex(name = 'V_62',
              particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_34})

V_63 = Vertex(name = 'V_63',
              particles = [ P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_66})

V_64 = Vertex(name = 'V_64',
              particles = [ P.a, P.a, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV91 ],
              couplings = {(0,0):C.GC_5})

V_65 = Vertex(name = 'V_65',
              particles = [ P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVV2 ],
              couplings = {(0,0):C.GC_43})

V_66 = Vertex(name = 'V_66',
              particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV91 ],
              couplings = {(0,0):C.GC_35})

V_67 = Vertex(name = 'V_67',
              particles = [ P.ta__plus__, P.vt, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.FFS5 ],
              couplings = {(0,0):C.GC_80})

V_68 = Vertex(name = 'V_68',
              particles = [ P.a, P.Z, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_57})

V_69 = Vertex(name = 'V_69',
              particles = [ P.Z, P.G0, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_52})

V_70 = Vertex(name = 'V_70',
              particles = [ P.Z, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_55})

V_71 = Vertex(name = 'V_71',
              particles = [ P.W__minus__, P.Z, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_8})

V_72 = Vertex(name = 'V_72',
              particles = [ P.W__minus__, P.Z, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_9})

V_73 = Vertex(name = 'V_73',
              particles = [ P.W__minus__, P.Z, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_61})

V_74 = Vertex(name = 'V_74',
              particles = [ P.W__plus__, P.Z, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_8})

V_75 = Vertex(name = 'V_75',
              particles = [ P.W__plus__, P.Z, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_7})

V_76 = Vertex(name = 'V_76',
              particles = [ P.W__plus__, P.Z, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_60})

V_77 = Vertex(name = 'V_77',
              particles = [ P.a, P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV94 ],
              couplings = {(0,0):C.GC_47})

V_78 = Vertex(name = 'V_78',
              particles = [ P.Z, P.Z, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_59})

V_79 = Vertex(name = 'V_79',
              particles = [ P.Z, P.Z, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_58})

V_80 = Vertex(name = 'V_80',
              particles = [ P.Z, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_59})

V_81 = Vertex(name = 'V_81',
              particles = [ P.Z, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_75})

V_82 = Vertex(name = 'V_82',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV91 ],
              couplings = {(0,0):C.GC_36})

V_83 = Vertex(name = 'V_83',
              particles = [ P.e__plus__, P.e__minus__, P.a ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_84 = Vertex(name = 'V_84',
              particles = [ P.mu__plus__, P.mu__minus__, P.a ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_85 = Vertex(name = 'V_85',
              particles = [ P.ta__plus__, P.ta__minus__, P.a ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_86 = Vertex(name = 'V_86',
              particles = [ P.u__tilde__, P.u, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_87 = Vertex(name = 'V_87',
              particles = [ P.c__tilde__, P.c, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_88 = Vertex(name = 'V_88',
              particles = [ P.t__tilde__, P.t, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_89 = Vertex(name = 'V_89',
              particles = [ P.d__tilde__, P.d, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_90 = Vertex(name = 'V_90',
              particles = [ P.s__tilde__, P.s, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_91 = Vertex(name = 'V_91',
              particles = [ P.b__tilde__, P.b, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_92 = Vertex(name = 'V_92',
              particles = [ P.u__tilde__, P.u, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_11})

V_93 = Vertex(name = 'V_93',
              particles = [ P.c__tilde__, P.c, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_11})

V_94 = Vertex(name = 'V_94',
              particles = [ P.t__tilde__, P.t, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_11})

V_95 = Vertex(name = 'V_95',
              particles = [ P.d__tilde__, P.d, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_11})

V_96 = Vertex(name = 'V_96',
              particles = [ P.s__tilde__, P.s, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_11})

V_97 = Vertex(name = 'V_97',
              particles = [ P.b__tilde__, P.b, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_11})

V_98 = Vertex(name = 'V_98',
              particles = [ P.d__tilde__, P.u, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_40})

V_99 = Vertex(name = 'V_99',
              particles = [ P.s__tilde__, P.c, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_40})

V_100 = Vertex(name = 'V_100',
               particles = [ P.b__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_40})

V_101 = Vertex(name = 'V_101',
               particles = [ P.u__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_40})

V_102 = Vertex(name = 'V_102',
               particles = [ P.c__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_40})

V_103 = Vertex(name = 'V_103',
               particles = [ P.t__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_40})

V_104 = Vertex(name = 'V_104',
               particles = [ P.e__plus__, P.ve, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_40})

V_105 = Vertex(name = 'V_105',
               particles = [ P.mu__plus__, P.vm, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_40})

V_106 = Vertex(name = 'V_106',
               particles = [ P.ta__plus__, P.vt, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_40})

V_107 = Vertex(name = 'V_107',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_40})

V_108 = Vertex(name = 'V_108',
               particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_40})

V_109 = Vertex(name = 'V_109',
               particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_40})

V_110 = Vertex(name = 'V_110',
               particles = [ P.u__tilde__, P.u, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV7 ],
               couplings = {(0,0):C.GC_41,(0,1):C.GC_48})

V_111 = Vertex(name = 'V_111',
               particles = [ P.c__tilde__, P.c, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_54,(0,1):C.GC_50})

V_112 = Vertex(name = 'V_112',
               particles = [ P.t__tilde__, P.t, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_54,(0,1):C.GC_50})

V_113 = Vertex(name = 'V_113',
               particles = [ P.d__tilde__, P.d, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_53,(0,1):C.GC_49})

V_114 = Vertex(name = 'V_114',
               particles = [ P.s__tilde__, P.s, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_53,(0,1):C.GC_49})

V_115 = Vertex(name = 'V_115',
               particles = [ P.b__tilde__, P.b, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_53,(0,1):C.GC_49})

V_116 = Vertex(name = 'V_116',
               particles = [ P.ve__tilde__, P.ve, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_56})

V_117 = Vertex(name = 'V_117',
               particles = [ P.vm__tilde__, P.vm, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_56})

V_118 = Vertex(name = 'V_118',
               particles = [ P.vt__tilde__, P.vt, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_56})

V_119 = Vertex(name = 'V_119',
               particles = [ P.e__plus__, P.e__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_55,(0,1):C.GC_51})

V_120 = Vertex(name = 'V_120',
               particles = [ P.mu__plus__, P.mu__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_55,(0,1):C.GC_51})

V_121 = Vertex(name = 'V_121',
               particles = [ P.ta__plus__, P.ta__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_55,(0,1):C.GC_51})

