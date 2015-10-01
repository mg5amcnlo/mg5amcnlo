# This file was automatically created by FeynRules $Revision: 634 $
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (February 23, 2011)
# Date: Thu 28 Jul 2011 16:28:57


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_240})

V_2 = Vertex(name = 'V_2',
             particles = [ P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_261})

V_3 = Vertex(name = 'V_3',
             particles = [ P.G, P.G, P.G ],
             color = [ 'f(1,2,3)' ],
             lorentz = [ L.VVV2 ],
             couplings = {(0,0):C.GC_4})

V_4 = Vertex(name = 'V_4',
             particles = [ P.G, P.G, P.G, P.G ],
             color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
             lorentz = [ L.VVVV2, L.VVVV4, L.VVVV5 ],
             couplings = {(1,1):C.GC_6,(0,0):C.GC_6,(2,2):C.GC_6})

V_5 = Vertex(name = 'V_5',
             particles = [ P.A, P.W__minus__, P.W__plus__ ],
             color = [ '1' ],
             lorentz = [ L.VVV2 ],
             couplings = {(0,0):C.GC_256})

V_6 = Vertex(name = 'V_6',
             particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.VVSS1 ],
             couplings = {(0,0):C.GC_241})

V_7 = Vertex(name = 'V_7',
             particles = [ P.W__minus__, P.W__plus__, P.H ],
             color = [ '1' ],
             lorentz = [ L.VVS1 ],
             couplings = {(0,0):C.GC_262})

V_8 = Vertex(name = 'V_8',
             particles = [ P.A, P.A, P.W__minus__, P.W__plus__ ],
             color = [ '1' ],
             lorentz = [ L.VVVV3 ],
             couplings = {(0,0):C.GC_258})

V_9 = Vertex(name = 'V_9',
             particles = [ P.W__minus__, P.W__plus__, P.Z ],
             color = [ '1' ],
             lorentz = [ L.VVV2 ],
             couplings = {(0,0):C.GC_109})

V_10 = Vertex(name = 'V_10',
              particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV3 ],
              couplings = {(0,0):C.GC_110})

V_11 = Vertex(name = 'V_11',
              particles = [ P.A, P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV6 ],
              couplings = {(0,0):C.GC_257})

V_12 = Vertex(name = 'V_12',
              particles = [ P.Z, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_260})

V_13 = Vertex(name = 'V_13',
              particles = [ P.Z, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_263})

V_14 = Vertex(name = 'V_14',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV3 ],
              couplings = {(0,0):C.GC_111})

V_15 = Vertex(name = 'V_15',
              particles = [ P.G, P.G, P.SV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVS2 ],
              couplings = {(0,0):C.GC_35})

V_16 = Vertex(name = 'V_16',
              particles = [ P.G, P.G, P.VV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVV1 ],
              couplings = {(0,0):C.GC_99})

V_17 = Vertex(name = 'V_17',
              particles = [ P.G, P.G, P.G, P.SV ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVVS1 ],
              couplings = {(0,0):C.GC_36})

V_18 = Vertex(name = 'V_18',
              particles = [ P.G, P.G, P.G, P.VV ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVVV1 ],
              couplings = {(0,0):C.GC_100})

V_19 = Vertex(name = 'V_19',
              particles = [ P.G, P.G, P.TV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVT1 ],
              couplings = {(0,0):C.GC_53})

V_20 = Vertex(name = 'V_20',
              particles = [ P.G, P.G, P.G, P.TV ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVVT1 ],
              couplings = {(0,0):C.GC_54})

V_21 = Vertex(name = 'V_21',
              particles = [ P.d__tilde__, P.d, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_22 = Vertex(name = 'V_22',
              particles = [ P.s__tilde__, P.s, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_23 = Vertex(name = 'V_23',
              particles = [ P.b__tilde__, P.b, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_24 = Vertex(name = 'V_24',
              particles = [ P.e__plus__, P.e__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_25 = Vertex(name = 'V_25',
              particles = [ P.m__plus__, P.m__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_26 = Vertex(name = 'V_26',
              particles = [ P.tt__plus__, P.tt__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_27 = Vertex(name = 'V_27',
              particles = [ P.u__tilde__, P.u, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_28 = Vertex(name = 'V_28',
              particles = [ P.c__tilde__, P.c, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_29 = Vertex(name = 'V_29',
              particles = [ P.t__tilde__, P.t, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_30 = Vertex(name = 'V_30',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_5})

V_31 = Vertex(name = 'V_31',
              particles = [ P.s__tilde__, P.s, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_5})

V_32 = Vertex(name = 'V_32',
              particles = [ P.b__tilde__, P.b, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_5})

V_33 = Vertex(name = 'V_33',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_264})

V_34 = Vertex(name = 'V_34',
              particles = [ P.d__tilde__, P.d, P.SV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_19,(0,1):C.GC_31})

V_35 = Vertex(name = 'V_35',
              particles = [ P.d__tilde__, P.s, P.SV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_21,(0,1):C.GC_33})

V_36 = Vertex(name = 'V_36',
              particles = [ P.s__tilde__, P.d, P.SV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_20,(0,1):C.GC_32})

V_37 = Vertex(name = 'V_37',
              particles = [ P.s__tilde__, P.s, P.SV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_22,(0,1):C.GC_34})

V_38 = Vertex(name = 'V_38',
              particles = [ P.d__tilde__, P.d, P.TV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFT2, L.FFT4, L.FFT6, L.FFT8 ],
              couplings = {(0,3):C.GC_71,(0,2):C.GC_73,(0,1):C.GC_45,(0,0):C.GC_47})

V_39 = Vertex(name = 'V_39',
              particles = [ P.d__tilde__, P.s, P.TV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
              couplings = {(0,2):C.GC_76,(0,3):C.GC_75,(0,0):C.GC_50,(0,1):C.GC_49})

V_40 = Vertex(name = 'V_40',
              particles = [ P.s__tilde__, P.d, P.TV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
              couplings = {(0,2):C.GC_74,(0,3):C.GC_77,(0,0):C.GC_48,(0,1):C.GC_51})

V_41 = Vertex(name = 'V_41',
              particles = [ P.s__tilde__, P.s, P.TV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFT2, L.FFT4, L.FFT6, L.FFT8 ],
              couplings = {(0,3):C.GC_72,(0,2):C.GC_78,(0,1):C.GC_46,(0,0):C.GC_52})

V_42 = Vertex(name = 'V_42',
              particles = [ P.d__tilde__, P.d, P.VV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,1):C.GC_7,(0,0):C.GC_95})

V_43 = Vertex(name = 'V_43',
              particles = [ P.d__tilde__, P.s, P.VV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,1):C.GC_9,(0,0):C.GC_97})

V_44 = Vertex(name = 'V_44',
              particles = [ P.s__tilde__, P.d, P.VV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,1):C.GC_8,(0,0):C.GC_96})

V_45 = Vertex(name = 'V_45',
              particles = [ P.s__tilde__, P.s, P.VV ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,1):C.GC_10,(0,0):C.GC_98})

V_46 = Vertex(name = 'V_46',
              particles = [ P.d__tilde__, P.d, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3, L.FFV4 ],
              couplings = {(0,0):C.GC_252,(0,1):C.GC_254})

V_47 = Vertex(name = 'V_47',
              particles = [ P.s__tilde__, P.s, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3, L.FFV4 ],
              couplings = {(0,0):C.GC_252,(0,1):C.GC_254})

V_48 = Vertex(name = 'V_48',
              particles = [ P.b__tilde__, P.b, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3, L.FFV4 ],
              couplings = {(0,0):C.GC_252,(0,1):C.GC_254})

V_49 = Vertex(name = 'V_49',
              particles = [ P.d__tilde__, P.u, P.SVP__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_136,(0,1):C.GC_152})

V_50 = Vertex(name = 'V_50',
              particles = [ P.d__tilde__, P.c, P.SVP__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_140,(0,1):C.GC_156})

V_51 = Vertex(name = 'V_51',
              particles = [ P.s__tilde__, P.u, P.SVP__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_138,(0,1):C.GC_154})

V_52 = Vertex(name = 'V_52',
              particles = [ P.s__tilde__, P.c, P.SVP__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_142,(0,1):C.GC_158})

V_53 = Vertex(name = 'V_53',
              particles = [ P.d__tilde__, P.u, P.TVP__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
              couplings = {(0,2):C.GC_184,(0,3):C.GC_233,(0,0):C.GC_168,(0,1):C.GC_217})

V_54 = Vertex(name = 'V_54',
              particles = [ P.d__tilde__, P.c, P.TVP__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
              couplings = {(0,2):C.GC_188,(0,3):C.GC_237,(0,0):C.GC_172,(0,1):C.GC_221})

V_55 = Vertex(name = 'V_55',
              particles = [ P.s__tilde__, P.u, P.TVP__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
              couplings = {(0,2):C.GC_186,(0,3):C.GC_235,(0,0):C.GC_170,(0,1):C.GC_219})

V_56 = Vertex(name = 'V_56',
              particles = [ P.s__tilde__, P.c, P.TVP__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
              couplings = {(0,2):C.GC_190,(0,3):C.GC_239,(0,0):C.GC_174,(0,1):C.GC_223})

V_57 = Vertex(name = 'V_57',
              particles = [ P.d__tilde__, P.u, P.VVP__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,1):C.GC_120,(0,0):C.GC_200})

V_58 = Vertex(name = 'V_58',
              particles = [ P.d__tilde__, P.c, P.VVP__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,1):C.GC_124,(0,0):C.GC_204})

V_59 = Vertex(name = 'V_59',
              particles = [ P.s__tilde__, P.u, P.VVP__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,1):C.GC_122,(0,0):C.GC_202})

V_60 = Vertex(name = 'V_60',
              particles = [ P.s__tilde__, P.c, P.VVP__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,1):C.GC_126,(0,0):C.GC_206})

V_61 = Vertex(name = 'V_61',
              particles = [ P.d__tilde__, P.u, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_243})

V_62 = Vertex(name = 'V_62',
              particles = [ P.d__tilde__, P.c, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_246})

V_63 = Vertex(name = 'V_63',
              particles = [ P.d__tilde__, P.t, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_249})

V_64 = Vertex(name = 'V_64',
              particles = [ P.s__tilde__, P.u, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_244})

V_65 = Vertex(name = 'V_65',
              particles = [ P.s__tilde__, P.c, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_247})

V_66 = Vertex(name = 'V_66',
              particles = [ P.s__tilde__, P.t, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_250})

V_67 = Vertex(name = 'V_67',
              particles = [ P.b__tilde__, P.u, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_245})

V_68 = Vertex(name = 'V_68',
              particles = [ P.b__tilde__, P.c, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_248})

V_69 = Vertex(name = 'V_69',
              particles = [ P.b__tilde__, P.t, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_251})

V_70 = Vertex(name = 'V_70',
              particles = [ P.u__tilde__, P.d, P.SVP__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_137,(0,1):C.GC_153})

V_71 = Vertex(name = 'V_71',
              particles = [ P.c__tilde__, P.d, P.SVP__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_141,(0,1):C.GC_157})

V_72 = Vertex(name = 'V_72',
              particles = [ P.u__tilde__, P.s, P.SVP__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_139,(0,1):C.GC_155})

V_73 = Vertex(name = 'V_73',
              particles = [ P.c__tilde__, P.s, P.SVP__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_143,(0,1):C.GC_159})

V_74 = Vertex(name = 'V_74',
              particles = [ P.u__tilde__, P.d, P.TVP__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
              couplings = {(0,2):C.GC_232,(0,3):C.GC_185,(0,0):C.GC_216,(0,1):C.GC_169})

V_75 = Vertex(name = 'V_75',
              particles = [ P.c__tilde__, P.d, P.TVP__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
              couplings = {(0,2):C.GC_236,(0,3):C.GC_189,(0,0):C.GC_220,(0,1):C.GC_173})

V_76 = Vertex(name = 'V_76',
              particles = [ P.u__tilde__, P.s, P.TVP__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
              couplings = {(0,2):C.GC_234,(0,3):C.GC_187,(0,0):C.GC_218,(0,1):C.GC_171})

V_77 = Vertex(name = 'V_77',
              particles = [ P.c__tilde__, P.s, P.TVP__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
              couplings = {(0,2):C.GC_238,(0,3):C.GC_191,(0,0):C.GC_222,(0,1):C.GC_175})

V_78 = Vertex(name = 'V_78',
              particles = [ P.u__tilde__, P.d, P.VVP__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,1):C.GC_121,(0,0):C.GC_201})

V_79 = Vertex(name = 'V_79',
              particles = [ P.c__tilde__, P.d, P.VVP__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,1):C.GC_125,(0,0):C.GC_205})

V_80 = Vertex(name = 'V_80',
              particles = [ P.u__tilde__, P.s, P.VVP__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,1):C.GC_123,(0,0):C.GC_203})

V_81 = Vertex(name = 'V_81',
              particles = [ P.c__tilde__, P.s, P.VVP__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,1):C.GC_127,(0,0):C.GC_207})

V_82 = Vertex(name = 'V_82',
              particles = [ P.u__tilde__, P.d, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_267})

V_83 = Vertex(name = 'V_83',
              particles = [ P.c__tilde__, P.d, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_270})

V_84 = Vertex(name = 'V_84',
              particles = [ P.t__tilde__, P.d, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_273})

V_85 = Vertex(name = 'V_85',
              particles = [ P.u__tilde__, P.s, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_268})

V_86 = Vertex(name = 'V_86',
              particles = [ P.c__tilde__, P.s, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_271})

V_87 = Vertex(name = 'V_87',
              particles = [ P.t__tilde__, P.s, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_274})

V_88 = Vertex(name = 'V_88',
              particles = [ P.u__tilde__, P.b, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_269})

V_89 = Vertex(name = 'V_89',
              particles = [ P.c__tilde__, P.b, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_272})

V_90 = Vertex(name = 'V_90',
              particles = [ P.t__tilde__, P.b, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV3 ],
              couplings = {(0,0):C.GC_275})

V_91 = Vertex(name = 'V_91',
              particles = [ P.u__tilde__, P.u, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_5})

V_92 = Vertex(name = 'V_92',
              particles = [ P.c__tilde__, P.c, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_5})

V_93 = Vertex(name = 'V_93',
              particles = [ P.t__tilde__, P.t, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_5})

V_94 = Vertex(name = 'V_94',
              particles = [ P.tt__plus__, P.tt__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_266})

V_95 = Vertex(name = 'V_95',
              particles = [ P.t__tilde__, P.t, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_265})

V_96 = Vertex(name = 'V_96',
              particles = [ P.e__plus__, P.e__minus__, P.SV ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_23,(0,1):C.GC_37})

V_97 = Vertex(name = 'V_97',
              particles = [ P.e__plus__, P.m__minus__, P.SV ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_25,(0,1):C.GC_39})

V_98 = Vertex(name = 'V_98',
              particles = [ P.m__plus__, P.e__minus__, P.SV ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_24,(0,1):C.GC_38})

V_99 = Vertex(name = 'V_99',
              particles = [ P.m__plus__, P.m__minus__, P.SV ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_26,(0,1):C.GC_40})

V_100 = Vertex(name = 'V_100',
               particles = [ P.e__plus__, P.e__minus__, P.TV ],
               color = [ '1' ],
               lorentz = [ L.FFT2, L.FFT4, L.FFT6, L.FFT8 ],
               couplings = {(0,3):C.GC_79,(0,2):C.GC_81,(0,1):C.GC_55,(0,0):C.GC_57})

V_101 = Vertex(name = 'V_101',
               particles = [ P.e__plus__, P.m__minus__, P.TV ],
               color = [ '1' ],
               lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
               couplings = {(0,2):C.GC_84,(0,3):C.GC_83,(0,0):C.GC_60,(0,1):C.GC_59})

V_102 = Vertex(name = 'V_102',
               particles = [ P.m__plus__, P.e__minus__, P.TV ],
               color = [ '1' ],
               lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
               couplings = {(0,2):C.GC_82,(0,3):C.GC_85,(0,0):C.GC_58,(0,1):C.GC_61})

V_103 = Vertex(name = 'V_103',
               particles = [ P.m__plus__, P.m__minus__, P.TV ],
               color = [ '1' ],
               lorentz = [ L.FFT2, L.FFT4, L.FFT6, L.FFT8 ],
               couplings = {(0,3):C.GC_80,(0,2):C.GC_86,(0,1):C.GC_56,(0,0):C.GC_62})

V_104 = Vertex(name = 'V_104',
               particles = [ P.e__plus__, P.e__minus__, P.VV ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_11,(0,0):C.GC_101})

V_105 = Vertex(name = 'V_105',
               particles = [ P.e__plus__, P.m__minus__, P.VV ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_13,(0,0):C.GC_103})

V_106 = Vertex(name = 'V_106',
               particles = [ P.m__plus__, P.e__minus__, P.VV ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_12,(0,0):C.GC_102})

V_107 = Vertex(name = 'V_107',
               particles = [ P.m__plus__, P.m__minus__, P.VV ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_14,(0,0):C.GC_104})

V_108 = Vertex(name = 'V_108',
               particles = [ P.e__plus__, P.e__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV3, L.FFV5 ],
               couplings = {(0,0):C.GC_252,(0,1):C.GC_255})

V_109 = Vertex(name = 'V_109',
               particles = [ P.m__plus__, P.m__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV3, L.FFV5 ],
               couplings = {(0,0):C.GC_252,(0,1):C.GC_255})

V_110 = Vertex(name = 'V_110',
               particles = [ P.tt__plus__, P.tt__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV3, L.FFV5 ],
               couplings = {(0,0):C.GC_252,(0,1):C.GC_255})

V_111 = Vertex(name = 'V_111',
               particles = [ P.e__plus__, P.ve, P.SVP__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_128,(0,1):C.GC_144})

V_112 = Vertex(name = 'V_112',
               particles = [ P.e__plus__, P.vm, P.SVP__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_132,(0,1):C.GC_148})

V_113 = Vertex(name = 'V_113',
               particles = [ P.m__plus__, P.ve, P.SVP__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_130,(0,1):C.GC_146})

V_114 = Vertex(name = 'V_114',
               particles = [ P.m__plus__, P.vm, P.SVP__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_134,(0,1):C.GC_150})

V_115 = Vertex(name = 'V_115',
               particles = [ P.e__plus__, P.ve, P.TVP__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
               couplings = {(0,2):C.GC_176,(0,3):C.GC_225,(0,0):C.GC_160,(0,1):C.GC_209})

V_116 = Vertex(name = 'V_116',
               particles = [ P.e__plus__, P.vm, P.TVP__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
               couplings = {(0,2):C.GC_180,(0,3):C.GC_229,(0,0):C.GC_164,(0,1):C.GC_213})

V_117 = Vertex(name = 'V_117',
               particles = [ P.m__plus__, P.ve, P.TVP__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
               couplings = {(0,2):C.GC_178,(0,3):C.GC_227,(0,0):C.GC_162,(0,1):C.GC_211})

V_118 = Vertex(name = 'V_118',
               particles = [ P.m__plus__, P.vm, P.TVP__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
               couplings = {(0,2):C.GC_182,(0,3):C.GC_231,(0,0):C.GC_166,(0,1):C.GC_215})

V_119 = Vertex(name = 'V_119',
               particles = [ P.e__plus__, P.ve, P.VVP__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_112,(0,0):C.GC_192})

V_120 = Vertex(name = 'V_120',
               particles = [ P.e__plus__, P.vm, P.VVP__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_116,(0,0):C.GC_196})

V_121 = Vertex(name = 'V_121',
               particles = [ P.m__plus__, P.ve, P.VVP__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_114,(0,0):C.GC_194})

V_122 = Vertex(name = 'V_122',
               particles = [ P.m__plus__, P.vm, P.VVP__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_118,(0,0):C.GC_198})

V_123 = Vertex(name = 'V_123',
               particles = [ P.e__plus__, P.ve, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV3 ],
               couplings = {(0,0):C.GC_242})

V_124 = Vertex(name = 'V_124',
               particles = [ P.m__plus__, P.vm, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV3 ],
               couplings = {(0,0):C.GC_242})

V_125 = Vertex(name = 'V_125',
               particles = [ P.tt__plus__, P.vt, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV3 ],
               couplings = {(0,0):C.GC_242})

V_126 = Vertex(name = 'V_126',
               particles = [ P.ve__tilde__, P.e__minus__, P.SVP__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_129,(0,1):C.GC_145})

V_127 = Vertex(name = 'V_127',
               particles = [ P.vm__tilde__, P.e__minus__, P.SVP__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_133,(0,1):C.GC_149})

V_128 = Vertex(name = 'V_128',
               particles = [ P.ve__tilde__, P.m__minus__, P.SVP__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_131,(0,1):C.GC_147})

V_129 = Vertex(name = 'V_129',
               particles = [ P.vm__tilde__, P.m__minus__, P.SVP__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_135,(0,1):C.GC_151})

V_130 = Vertex(name = 'V_130',
               particles = [ P.ve__tilde__, P.e__minus__, P.TVP__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
               couplings = {(0,2):C.GC_224,(0,3):C.GC_177,(0,0):C.GC_208,(0,1):C.GC_161})

V_131 = Vertex(name = 'V_131',
               particles = [ P.vm__tilde__, P.e__minus__, P.TVP__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
               couplings = {(0,2):C.GC_228,(0,3):C.GC_181,(0,0):C.GC_212,(0,1):C.GC_165})

V_132 = Vertex(name = 'V_132',
               particles = [ P.ve__tilde__, P.m__minus__, P.TVP__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
               couplings = {(0,2):C.GC_226,(0,3):C.GC_179,(0,0):C.GC_210,(0,1):C.GC_163})

V_133 = Vertex(name = 'V_133',
               particles = [ P.vm__tilde__, P.m__minus__, P.TVP__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
               couplings = {(0,2):C.GC_230,(0,3):C.GC_183,(0,0):C.GC_214,(0,1):C.GC_167})

V_134 = Vertex(name = 'V_134',
               particles = [ P.ve__tilde__, P.e__minus__, P.VVP__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_113,(0,0):C.GC_193})

V_135 = Vertex(name = 'V_135',
               particles = [ P.vm__tilde__, P.e__minus__, P.VVP__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_117,(0,0):C.GC_197})

V_136 = Vertex(name = 'V_136',
               particles = [ P.ve__tilde__, P.m__minus__, P.VVP__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_115,(0,0):C.GC_195})

V_137 = Vertex(name = 'V_137',
               particles = [ P.vm__tilde__, P.m__minus__, P.VVP__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_119,(0,0):C.GC_199})

V_138 = Vertex(name = 'V_138',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV3 ],
               couplings = {(0,0):C.GC_242})

V_139 = Vertex(name = 'V_139',
               particles = [ P.vm__tilde__, P.m__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV3 ],
               couplings = {(0,0):C.GC_242})

V_140 = Vertex(name = 'V_140',
               particles = [ P.vt__tilde__, P.tt__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV3 ],
               couplings = {(0,0):C.GC_242})

V_141 = Vertex(name = 'V_141',
               particles = [ P.u__tilde__, P.u, P.SV ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_27,(0,1):C.GC_41})

V_142 = Vertex(name = 'V_142',
               particles = [ P.u__tilde__, P.c, P.SV ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_29,(0,1):C.GC_43})

V_143 = Vertex(name = 'V_143',
               particles = [ P.c__tilde__, P.u, P.SV ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_28,(0,1):C.GC_42})

V_144 = Vertex(name = 'V_144',
               particles = [ P.c__tilde__, P.c, P.SV ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_30,(0,1):C.GC_44})

V_145 = Vertex(name = 'V_145',
               particles = [ P.u__tilde__, P.u, P.TV ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFT2, L.FFT4, L.FFT6, L.FFT8 ],
               couplings = {(0,3):C.GC_87,(0,2):C.GC_89,(0,1):C.GC_63,(0,0):C.GC_65})

V_146 = Vertex(name = 'V_146',
               particles = [ P.u__tilde__, P.c, P.TV ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
               couplings = {(0,2):C.GC_92,(0,3):C.GC_91,(0,0):C.GC_68,(0,1):C.GC_67})

V_147 = Vertex(name = 'V_147',
               particles = [ P.c__tilde__, P.u, P.TV ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFT1, L.FFT3, L.FFT5, L.FFT7 ],
               couplings = {(0,2):C.GC_90,(0,3):C.GC_93,(0,0):C.GC_66,(0,1):C.GC_69})

V_148 = Vertex(name = 'V_148',
               particles = [ P.c__tilde__, P.c, P.TV ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFT2, L.FFT4, L.FFT6, L.FFT8 ],
               couplings = {(0,3):C.GC_88,(0,2):C.GC_94,(0,1):C.GC_64,(0,0):C.GC_70})

V_149 = Vertex(name = 'V_149',
               particles = [ P.u__tilde__, P.u, P.VV ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_15,(0,0):C.GC_105})

V_150 = Vertex(name = 'V_150',
               particles = [ P.u__tilde__, P.c, P.VV ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_17,(0,0):C.GC_107})

V_151 = Vertex(name = 'V_151',
               particles = [ P.c__tilde__, P.u, P.VV ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_16,(0,0):C.GC_106})

V_152 = Vertex(name = 'V_152',
               particles = [ P.c__tilde__, P.c, P.VV ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV2 ],
               couplings = {(0,1):C.GC_18,(0,0):C.GC_108})

V_153 = Vertex(name = 'V_153',
               particles = [ P.u__tilde__, P.u, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV3, L.FFV6 ],
               couplings = {(0,0):C.GC_253,(0,1):C.GC_254})

V_154 = Vertex(name = 'V_154',
               particles = [ P.c__tilde__, P.c, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV3, L.FFV6 ],
               couplings = {(0,0):C.GC_253,(0,1):C.GC_254})

V_155 = Vertex(name = 'V_155',
               particles = [ P.t__tilde__, P.t, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV3, L.FFV6 ],
               couplings = {(0,0):C.GC_253,(0,1):C.GC_254})

V_156 = Vertex(name = 'V_156',
               particles = [ P.ve__tilde__, P.ve, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV3 ],
               couplings = {(0,0):C.GC_259})

V_157 = Vertex(name = 'V_157',
               particles = [ P.vm__tilde__, P.vm, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV3 ],
               couplings = {(0,0):C.GC_259})

V_158 = Vertex(name = 'V_158',
               particles = [ P.vt__tilde__, P.vt, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV3 ],
               couplings = {(0,0):C.GC_259})

