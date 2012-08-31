# This file was automatically created by FeynRules 1.7.9
# Mathematica version: 8.0 for Linux x86 (64-bit) (February 23, 2011)
# Date: Fri 18 May 2012 14:43:24


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS2 ],
             couplings = {(0,0):C.GC_31})

V_2 = Vertex(name = 'V_2',
             particles = [ P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS2 ],
             couplings = {(0,0):C.GC_95})

V_3 = Vertex(name = 'V_3',
             particles = [ P.A, P.A, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.VVSS12, L.VVSS6 ],
             couplings = {(0,0):C.GC_83,(0,1):C.GC_82})

V_4 = Vertex(name = 'V_4',
             particles = [ P.A, P.A, P.H ],
             color = [ '1' ],
             lorentz = [ L.VVS11, L.VVS7 ],
             couplings = {(0,0):C.GC_123,(0,1):C.GC_122})

V_5 = Vertex(name = 'V_5',
             particles = [ P.A, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.VSS1 ],
             couplings = {(0,0):C.GC_48})

V_6 = Vertex(name = 'V_6',
             particles = [ P.A, P.W__minus__, P.W__plus__ ],
             color = [ '1' ],
             lorentz = [ L.VVV10, L.VVV11, L.VVV12, L.VVV13, L.VVV5, L.VVV6, L.VVV7 ],
             couplings = {(0,5):C.GC_113,(0,4):C.GC_105,(0,2):C.GC_57,(0,3):C.GC_55,(0,0):[ C.GC_47, C.GC_114 ],(0,6):C.GC_104,(0,1):C.GC_106})

V_7 = Vertex(name = 'V_7',
             particles = [ P.W__minus__, P.W__plus__, P.Z ],
             color = [ '1' ],
             lorentz = [ L.VVV10, L.VVV12, L.VVV13, L.VVV4, L.VVV6, L.VVV8, L.VVV9 ],
             couplings = {(0,4):C.GC_107,(0,3):C.GC_111,(0,1):C.GC_20,(0,2):C.GC_18,(0,0):[ C.GC_7, C.GC_108 ],(0,5):C.GC_112,(0,6):C.GC_110})

V_8 = Vertex(name = 'V_8',
             particles = [ P.A, P.Z, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.VVSS10, L.VVSS11, L.VVSS12, L.VVSS5, L.VVSS6, L.VVSS7 ],
             couplings = {(0,0):C.GC_81,(0,2):C.GC_76,(0,1):C.GC_80,(0,4):C.GC_50,(0,3):C.GC_65,(0,5):C.GC_9})

V_9 = Vertex(name = 'V_9',
             particles = [ P.A, P.Z, P.H ],
             color = [ '1' ],
             lorentz = [ L.VVS10, L.VVS11, L.VVS5, L.VVS6, L.VVS7, L.VVS9 ],
             couplings = {(0,5):C.GC_121,(0,1):C.GC_116,(0,0):C.GC_120,(0,4):C.GC_97,(0,2):C.GC_103,(0,3):C.GC_86})

V_10 = Vertex(name = 'V_10',
              particles = [ P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_8})

V_11 = Vertex(name = 'V_11',
              particles = [ P.ghG, P.ghG__tilde__, P.G ],
              color = [ 'f(3,1,2)' ],
              lorentz = [ L.UUV2 ],
              couplings = {(0,0):C.GC_4})

V_12 = Vertex(name = 'V_12',
              particles = [ P.G, P.G, P.G ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVV10 ],
              couplings = {(0,0):C.GC_4})

V_13 = Vertex(name = 'V_13',
              particles = [ P.G, P.G, P.G, P.G ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVV10, L.VVVV11, L.VVVV6 ],
              couplings = {(1,0):C.GC_6,(0,2):C.GC_6,(2,1):C.GC_6})

V_14 = Vertex(name = 'V_14',
              particles = [ P.A, P.A, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV14, L.VVVV7, L.VVVV9 ],
              couplings = {(0,1):C.GC_67,(0,0):C.GC_68,(0,2):[ C.GC_66, C.GC_128 ]})

V_15 = Vertex(name = 'V_15',
              particles = [ P.A, P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV12, L.VVVV13, L.VVVV8 ],
              couplings = {(0,2):C.GC_58,(0,1):C.GC_59,(0,0):[ C.GC_49, C.GC_127 ]})

V_16 = Vertex(name = 'V_16',
              particles = [ P.A, P.W__minus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVSS2, L.VVVSS3, L.VVVSS4, L.VVVSS6, L.VVVSS7 ],
              couplings = {(0,1):C.GC_54,(0,0):C.GC_15,(0,2):C.GC_14,(0,3):C.GC_56,(0,4):C.GC_16})

V_17 = Vertex(name = 'V_17',
              particles = [ P.A, P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVS10, L.VVVS5, L.VVVS6, L.VVVS7, L.VVVS9 ],
              couplings = {(0,2):C.GC_101,(0,1):C.GC_90,(0,3):C.GC_89,(0,4):C.GC_102,(0,0):C.GC_91})

V_18 = Vertex(name = 'V_18',
              particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS12, L.VVSS8, L.VVSS9 ],
              couplings = {(0,0):C.GC_13,(0,1):C.GC_11,(0,2):C.GC_32})

V_19 = Vertex(name = 'V_19',
              particles = [ P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS11, L.VVS7, L.VVS8 ],
              couplings = {(0,0):C.GC_88,(0,1):C.GC_87,(0,2):C.GC_96})

V_20 = Vertex(name = 'V_20',
              particles = [ P.A, P.A, P.A, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVVV1, L.VVVVV2 ],
              couplings = {(0,0):C.GC_72,(0,1):C.GC_73})

V_21 = Vertex(name = 'V_21',
              particles = [ P.A, P.A, P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVVV10, L.VVVVV4 ],
              couplings = {(0,1):C.GC_69,(0,0):C.GC_70})

V_22 = Vertex(name = 'V_22',
              particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV14, L.VVVV7, L.VVVV9 ],
              couplings = {(0,1):C.GC_21,(0,0):C.GC_24,(0,2):[ C.GC_10, C.GC_109 ]})

V_23 = Vertex(name = 'V_23',
              particles = [ P.A, P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVVV12, L.VVVVV5 ],
              couplings = {(0,1):C.GC_60,(0,0):C.GC_62})

V_24 = Vertex(name = 'V_24',
              particles = [ P.A, P.A, P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVVVV1 ],
              couplings = {(0,0):C.GC_71})

V_25 = Vertex(name = 'V_25',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVSS1, L.VVVSS3, L.VVVSS5, L.VVVSS6, L.VVVSS8 ],
              couplings = {(0,1):C.GC_17,(0,0):C.GC_52,(0,3):C.GC_19,(0,4):C.GC_53,(0,2):C.GC_51})

V_26 = Vertex(name = 'V_26',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVS11, L.VVVS4, L.VVVS6, L.VVVS8, L.VVVS9 ],
              couplings = {(0,2):C.GC_92,(0,1):C.GC_99,(0,4):C.GC_93,(0,0):C.GC_100,(0,3):C.GC_98})

V_27 = Vertex(name = 'V_27',
              particles = [ P.A, P.A, P.W__minus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVSS1 ],
              couplings = {(0,0):C.GC_85})

V_28 = Vertex(name = 'V_28',
              particles = [ P.A, P.A, P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVS1 ],
              couplings = {(0,0):C.GC_125})

V_29 = Vertex(name = 'V_29',
              particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVSS1 ],
              couplings = {(0,0):C.GC_23})

V_30 = Vertex(name = 'V_30',
              particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVS1 ],
              couplings = {(0,0):C.GC_94})

V_31 = Vertex(name = 'V_31',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV14, L.VVVV7, L.VVVV9 ],
              couplings = {(0,1):C.GC_22,(0,0):C.GC_25,(0,2):[ C.GC_12, C.GC_126 ]})

V_32 = Vertex(name = 'V_32',
              particles = [ P.A, P.W__minus__, P.W__plus__, P.Z, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVVV6, L.VVVVV8 ],
              couplings = {(0,0):C.GC_61,(0,1):C.GC_63})

V_33 = Vertex(name = 'V_33',
              particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVVV3, L.VVVVV9 ],
              couplings = {(0,0):C.GC_26,(0,1):C.GC_28})

V_34 = Vertex(name = 'V_34',
              particles = [ P.A, P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVVVV2 ],
              couplings = {(0,0):C.GC_64})

V_35 = Vertex(name = 'V_35',
              particles = [ P.Z, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS12, L.VVSS6, L.VVSS9 ],
              couplings = {(0,0):C.GC_79,(0,1):C.GC_75,(0,2):C.GC_78})

V_36 = Vertex(name = 'V_36',
              particles = [ P.Z, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS11, L.VVS7, L.VVS8 ],
              couplings = {(0,0):C.GC_119,(0,1):C.GC_115,(0,2):C.GC_118})

V_37 = Vertex(name = 'V_37',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVVV11, L.VVVVV7 ],
              couplings = {(0,1):C.GC_27,(0,0):C.GC_29})

V_38 = Vertex(name = 'V_38',
              particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.Z, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVVVV1 ],
              couplings = {(0,0):C.GC_30})

V_39 = Vertex(name = 'V_39',
              particles = [ P.A, P.W__minus__, P.W__plus__, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVSS2 ],
              couplings = {(0,0):C.GC_84})

V_40 = Vertex(name = 'V_40',
              particles = [ P.A, P.W__minus__, P.W__plus__, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVS2 ],
              couplings = {(0,0):C.GC_124})

V_41 = Vertex(name = 'V_41',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVSS1 ],
              couplings = {(0,0):C.GC_77})

V_42 = Vertex(name = 'V_42',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVS1 ],
              couplings = {(0,0):C.GC_117})

V_43 = Vertex(name = 'V_43',
              particles = [ P.d__tilde__, P.d, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_1})

V_44 = Vertex(name = 'V_44',
              particles = [ P.s__tilde__, P.s, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_1})

V_45 = Vertex(name = 'V_45',
              particles = [ P.b__tilde__, P.b, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_1})

V_46 = Vertex(name = 'V_46',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_5})

V_47 = Vertex(name = 'V_47',
              particles = [ P.s__tilde__, P.s, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_5})

V_48 = Vertex(name = 'V_48',
              particles = [ P.b__tilde__, P.b, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_5})

V_49 = Vertex(name = 'V_49',
              particles = [ P.d__tilde__, P.d, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_131})

V_50 = Vertex(name = 'V_50',
              particles = [ P.s__tilde__, P.s, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_134})

V_51 = Vertex(name = 'V_51',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_129})

V_52 = Vertex(name = 'V_52',
              particles = [ P.d__tilde__, P.d, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7, L.FFV8 ],
              couplings = {(0,0):C.GC_43,(0,1):C.GC_45})

V_53 = Vertex(name = 'V_53',
              particles = [ P.s__tilde__, P.s, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7, L.FFV8 ],
              couplings = {(0,0):C.GC_43,(0,1):C.GC_45})

V_54 = Vertex(name = 'V_54',
              particles = [ P.b__tilde__, P.b, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7, L.FFV8 ],
              couplings = {(0,1):C.GC_45,(0,0):C.GC_43})

V_55 = Vertex(name = 'V_55',
              particles = [ P.u__tilde__, P.d, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_138})

V_56 = Vertex(name = 'V_56',
              particles = [ P.c__tilde__, P.d, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_141})

V_57 = Vertex(name = 'V_57',
              particles = [ P.t__tilde__, P.d, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_144})

V_58 = Vertex(name = 'V_58',
              particles = [ P.u__tilde__, P.s, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_139})

V_59 = Vertex(name = 'V_59',
              particles = [ P.c__tilde__, P.s, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_142})

V_60 = Vertex(name = 'V_60',
              particles = [ P.t__tilde__, P.s, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_145})

V_61 = Vertex(name = 'V_61',
              particles = [ P.u__tilde__, P.b, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_140})

V_62 = Vertex(name = 'V_62',
              particles = [ P.c__tilde__, P.b, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_143})

V_63 = Vertex(name = 'V_63',
              particles = [ P.t__tilde__, P.b, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_146})

V_64 = Vertex(name = 'V_64',
              particles = [ P.e__plus__, P.e__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_3})

V_65 = Vertex(name = 'V_65',
              particles = [ P.m__plus__, P.m__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_3})

V_66 = Vertex(name = 'V_66',
              particles = [ P.tt__plus__, P.tt__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_3})

V_67 = Vertex(name = 'V_67',
              particles = [ P.e__plus__, P.e__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_132})

V_68 = Vertex(name = 'V_68',
              particles = [ P.m__plus__, P.m__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_133})

V_69 = Vertex(name = 'V_69',
              particles = [ P.tt__plus__, P.tt__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_136})

V_70 = Vertex(name = 'V_70',
              particles = [ P.e__plus__, P.e__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV7, L.FFV9 ],
              couplings = {(0,0):C.GC_43,(0,1):C.GC_46})

V_71 = Vertex(name = 'V_71',
              particles = [ P.m__plus__, P.m__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV7, L.FFV9 ],
              couplings = {(0,0):C.GC_43,(0,1):C.GC_46})

V_72 = Vertex(name = 'V_72',
              particles = [ P.tt__plus__, P.tt__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV7, L.FFV9 ],
              couplings = {(0,0):C.GC_43,(0,1):C.GC_46})

V_73 = Vertex(name = 'V_73',
              particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_33})

V_74 = Vertex(name = 'V_74',
              particles = [ P.vm__tilde__, P.m__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_33})

V_75 = Vertex(name = 'V_75',
              particles = [ P.vt__tilde__, P.tt__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_33})

V_76 = Vertex(name = 'V_76',
              particles = [ P.d__tilde__, P.u, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_34})

V_77 = Vertex(name = 'V_77',
              particles = [ P.s__tilde__, P.u, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_35})

V_78 = Vertex(name = 'V_78',
              particles = [ P.b__tilde__, P.u, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_36})

V_79 = Vertex(name = 'V_79',
              particles = [ P.d__tilde__, P.c, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_37})

V_80 = Vertex(name = 'V_80',
              particles = [ P.s__tilde__, P.c, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_38})

V_81 = Vertex(name = 'V_81',
              particles = [ P.b__tilde__, P.c, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_39})

V_82 = Vertex(name = 'V_82',
              particles = [ P.d__tilde__, P.t, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_40})

V_83 = Vertex(name = 'V_83',
              particles = [ P.s__tilde__, P.t, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_41})

V_84 = Vertex(name = 'V_84',
              particles = [ P.b__tilde__, P.t, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_42})

V_85 = Vertex(name = 'V_85',
              particles = [ P.u__tilde__, P.u, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_2})

V_86 = Vertex(name = 'V_86',
              particles = [ P.c__tilde__, P.c, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_2})

V_87 = Vertex(name = 'V_87',
              particles = [ P.t__tilde__, P.t, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_2})

V_88 = Vertex(name = 'V_88',
              particles = [ P.u__tilde__, P.u, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_5})

V_89 = Vertex(name = 'V_89',
              particles = [ P.c__tilde__, P.c, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_5})

V_90 = Vertex(name = 'V_90',
              particles = [ P.t__tilde__, P.t, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV6 ],
              couplings = {(0,0):C.GC_5})

V_91 = Vertex(name = 'V_91',
              particles = [ P.u__tilde__, P.u, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_137})

V_92 = Vertex(name = 'V_92',
              particles = [ P.c__tilde__, P.c, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_130})

V_93 = Vertex(name = 'V_93',
              particles = [ P.t__tilde__, P.t, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_135})

V_94 = Vertex(name = 'V_94',
              particles = [ P.u__tilde__, P.u, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV10, L.FFV7 ],
              couplings = {(0,1):C.GC_44,(0,0):C.GC_45})

V_95 = Vertex(name = 'V_95',
              particles = [ P.c__tilde__, P.c, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV10, L.FFV7 ],
              couplings = {(0,1):C.GC_44,(0,0):C.GC_45})

V_96 = Vertex(name = 'V_96',
              particles = [ P.t__tilde__, P.t, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV10, L.FFV7 ],
              couplings = {(0,1):C.GC_44,(0,0):C.GC_45})

V_97 = Vertex(name = 'V_97',
              particles = [ P.e__plus__, P.ve, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_33})

V_98 = Vertex(name = 'V_98',
              particles = [ P.m__plus__, P.vm, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_33})

V_99 = Vertex(name = 'V_99',
              particles = [ P.tt__plus__, P.vt, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV7 ],
              couplings = {(0,0):C.GC_33})

V_100 = Vertex(name = 'V_100',
               particles = [ P.ve__tilde__, P.ve, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV7 ],
               couplings = {(0,0):C.GC_74})

V_101 = Vertex(name = 'V_101',
               particles = [ P.vm__tilde__, P.vm, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV7 ],
               couplings = {(0,0):C.GC_74})

V_102 = Vertex(name = 'V_102',
               particles = [ P.vt__tilde__, P.vt, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV7 ],
               couplings = {(0,0):C.GC_74})

