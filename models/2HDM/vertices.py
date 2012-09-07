# This file was automatically created by FeynRules 1.7.55
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (October 6, 2011)
# Date: Wed 8 Aug 2012 14:37:00


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.a, P.a, P.h__minus__, P.h__plus__ ],
             color = [ '1' ],
             lorentz = [ L.VVSS1 ],
             couplings = {(0,0):C.GC_4})

V_2 = Vertex(name = 'V_2',
             particles = [ P.a, P.h__minus__, P.h__plus__ ],
             color = [ '1' ],
             lorentz = [ L.VSS1 ],
             couplings = {(0,0):C.GC_3})

V_3 = Vertex(name = 'V_3',
             particles = [ P.G, P.G, P.G ],
             color = [ 'f(1,2,3)' ],
             lorentz = [ L.VVV1 ],
             couplings = {(0,0):C.GC_5})

V_4 = Vertex(name = 'V_4',
             particles = [ P.G, P.G, P.G, P.G ],
             color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
             lorentz = [ L.VVVV1, L.VVVV3, L.VVVV4 ],
             couplings = {(1,1):C.GC_7,(0,0):C.GC_7,(2,2):C.GC_7})

V_5 = Vertex(name = 'V_5',
             particles = [ P.h1, P.h__minus__, P.h__plus__ ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_305})

V_6 = Vertex(name = 'V_6',
             particles = [ P.h1, P.h1, P.h1 ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_306})

V_7 = Vertex(name = 'V_7',
             particles = [ P.h2, P.h__minus__, P.h__plus__ ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_307})

V_8 = Vertex(name = 'V_8',
             particles = [ P.h1, P.h1, P.h2 ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_308})

V_9 = Vertex(name = 'V_9',
             particles = [ P.h1, P.h2, P.h2 ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_309})

V_10 = Vertex(name = 'V_10',
              particles = [ P.h2, P.h2, P.h2 ],
              color = [ '1' ],
              lorentz = [ L.SSS1 ],
              couplings = {(0,0):C.GC_310})

V_11 = Vertex(name = 'V_11',
              particles = [ P.h3, P.h__minus__, P.h__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSS1 ],
              couplings = {(0,0):C.GC_311})

V_12 = Vertex(name = 'V_12',
              particles = [ P.h1, P.h1, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.SSS1 ],
              couplings = {(0,0):C.GC_312})

V_13 = Vertex(name = 'V_13',
              particles = [ P.h1, P.h2, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.SSS1 ],
              couplings = {(0,0):C.GC_313})

V_14 = Vertex(name = 'V_14',
              particles = [ P.h2, P.h2, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.SSS1 ],
              couplings = {(0,0):C.GC_314})

V_15 = Vertex(name = 'V_15',
              particles = [ P.h1, P.h3, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.SSS1 ],
              couplings = {(0,0):C.GC_315})

V_16 = Vertex(name = 'V_16',
              particles = [ P.h2, P.h3, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.SSS1 ],
              couplings = {(0,0):C.GC_316})

V_17 = Vertex(name = 'V_17',
              particles = [ P.h3, P.h3, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.SSS1 ],
              couplings = {(0,0):C.GC_317})

V_18 = Vertex(name = 'V_18',
              particles = [ P.a, P.w__minus__, P.w__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVV1 ],
              couplings = {(0,0):C.GC_21})

V_19 = Vertex(name = 'V_19',
              particles = [ P.a, P.w__minus__, P.h1, P.h__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_32})

V_20 = Vertex(name = 'V_20',
              particles = [ P.w__minus__, P.h1, P.h__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_30})

V_21 = Vertex(name = 'V_21',
              particles = [ P.a, P.w__minus__, P.h2, P.h__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_40})

V_22 = Vertex(name = 'V_22',
              particles = [ P.w__minus__, P.h2, P.h__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_38})

V_23 = Vertex(name = 'V_23',
              particles = [ P.a, P.w__minus__, P.h3, P.h__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_51})

V_24 = Vertex(name = 'V_24',
              particles = [ P.w__minus__, P.h3, P.h__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_49})

V_25 = Vertex(name = 'V_25',
              particles = [ P.a, P.w__plus__, P.h1, P.h__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_33})

V_26 = Vertex(name = 'V_26',
              particles = [ P.w__plus__, P.h1, P.h__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_31})

V_27 = Vertex(name = 'V_27',
              particles = [ P.a, P.w__plus__, P.h2, P.h__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_41})

V_28 = Vertex(name = 'V_28',
              particles = [ P.w__plus__, P.h2, P.h__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_39})

V_29 = Vertex(name = 'V_29',
              particles = [ P.a, P.w__plus__, P.h3, P.h__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_52})

V_30 = Vertex(name = 'V_30',
              particles = [ P.w__plus__, P.h3, P.h__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_50})

V_31 = Vertex(name = 'V_31',
              particles = [ P.w__minus__, P.w__plus__, P.h__minus__, P.h__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_11})

V_32 = Vertex(name = 'V_32',
              particles = [ P.a, P.a, P.w__minus__, P.w__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV2 ],
              couplings = {(0,0):C.GC_23})

V_33 = Vertex(name = 'V_33',
              particles = [ P.w__minus__, P.w__plus__, P.h1 ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_169})

V_34 = Vertex(name = 'V_34',
              particles = [ P.w__minus__, P.w__plus__, P.h1, P.h1 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_34})

V_35 = Vertex(name = 'V_35',
              particles = [ P.w__minus__, P.w__plus__, P.h2 ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_170})

V_36 = Vertex(name = 'V_36',
              particles = [ P.w__minus__, P.w__plus__, P.h1, P.h2 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_43})

V_37 = Vertex(name = 'V_37',
              particles = [ P.w__minus__, P.w__plus__, P.h2, P.h2 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_45})

V_38 = Vertex(name = 'V_38',
              particles = [ P.w__minus__, P.w__plus__, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_171})

V_39 = Vertex(name = 'V_39',
              particles = [ P.w__minus__, P.w__plus__, P.h1, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_55})

V_40 = Vertex(name = 'V_40',
              particles = [ P.w__minus__, P.w__plus__, P.h2, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_57})

V_41 = Vertex(name = 'V_41',
              particles = [ P.w__minus__, P.w__plus__, P.h3, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_59})

V_42 = Vertex(name = 'V_42',
              particles = [ P.w__minus__, P.w__plus__, P.z ],
              color = [ '1' ],
              lorentz = [ L.VVV1 ],
              couplings = {(0,0):C.GC_8})

V_43 = Vertex(name = 'V_43',
              particles = [ P.w__minus__, P.w__minus__, P.w__plus__, P.w__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV2 ],
              couplings = {(0,0):C.GC_9})

V_44 = Vertex(name = 'V_44',
              particles = [ P.a, P.z, P.h__minus__, P.h__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_26})

V_45 = Vertex(name = 'V_45',
              particles = [ P.z, P.h__minus__, P.h__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_24})

V_46 = Vertex(name = 'V_46',
              particles = [ P.z, P.h1, P.h2 ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_42})

V_47 = Vertex(name = 'V_47',
              particles = [ P.z, P.h1, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_53})

V_48 = Vertex(name = 'V_48',
              particles = [ P.z, P.h2, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_54})

V_49 = Vertex(name = 'V_49',
              particles = [ P.w__minus__, P.z, P.h1, P.h__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_29})

V_50 = Vertex(name = 'V_50',
              particles = [ P.w__minus__, P.z, P.h2, P.h__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_37})

V_51 = Vertex(name = 'V_51',
              particles = [ P.w__minus__, P.z, P.h3, P.h__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_48})

V_52 = Vertex(name = 'V_52',
              particles = [ P.w__plus__, P.z, P.h1, P.h__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_28})

V_53 = Vertex(name = 'V_53',
              particles = [ P.w__plus__, P.z, P.h2, P.h__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_36})

V_54 = Vertex(name = 'V_54',
              particles = [ P.w__plus__, P.z, P.h3, P.h__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_47})

V_55 = Vertex(name = 'V_55',
              particles = [ P.a, P.w__minus__, P.w__plus__, P.z ],
              color = [ '1' ],
              lorentz = [ L.VVVV5 ],
              couplings = {(0,0):C.GC_22})

V_56 = Vertex(name = 'V_56',
              particles = [ P.z, P.z, P.h__minus__, P.h__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_27})

V_57 = Vertex(name = 'V_57',
              particles = [ P.z, P.z, P.h1 ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_172})

V_58 = Vertex(name = 'V_58',
              particles = [ P.z, P.z, P.h1, P.h1 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_35})

V_59 = Vertex(name = 'V_59',
              particles = [ P.z, P.z, P.h2 ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_173})

V_60 = Vertex(name = 'V_60',
              particles = [ P.z, P.z, P.h1, P.h2 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_44})

V_61 = Vertex(name = 'V_61',
              particles = [ P.z, P.z, P.h2, P.h2 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_46})

V_62 = Vertex(name = 'V_62',
              particles = [ P.z, P.z, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_174})

V_63 = Vertex(name = 'V_63',
              particles = [ P.z, P.z, P.h1, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_56})

V_64 = Vertex(name = 'V_64',
              particles = [ P.z, P.z, P.h2, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_58})

V_65 = Vertex(name = 'V_65',
              particles = [ P.z, P.z, P.h3, P.h3 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_60})

V_66 = Vertex(name = 'V_66',
              particles = [ P.w__minus__, P.w__plus__, P.z, P.z ],
              color = [ '1' ],
              lorentz = [ L.VVVV2 ],
              couplings = {(0,0):C.GC_10})

V_67 = Vertex(name = 'V_67',
              particles = [ P.d__tilde__, P.d, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_68 = Vertex(name = 'V_68',
              particles = [ P.s__tilde__, P.s, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_69 = Vertex(name = 'V_69',
              particles = [ P.b__tilde__, P.b, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_70 = Vertex(name = 'V_70',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_6})

V_71 = Vertex(name = 'V_71',
              particles = [ P.s__tilde__, P.s, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_6})

V_72 = Vertex(name = 'V_72',
              particles = [ P.b__tilde__, P.b, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_6})

V_73 = Vertex(name = 'V_73',
              particles = [ P.d__tilde__, P.d, P.h1 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_198,(0,1):C.GC_61})

V_74 = Vertex(name = 'V_74',
              particles = [ P.s__tilde__, P.d, P.h1 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_210,(0,1):C.GC_62})

V_75 = Vertex(name = 'V_75',
              particles = [ P.b__tilde__, P.d, P.h1 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_222,(0,1):C.GC_63})

V_76 = Vertex(name = 'V_76',
              particles = [ P.d__tilde__, P.s, P.h1 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_202,(0,1):C.GC_64})

V_77 = Vertex(name = 'V_77',
              particles = [ P.s__tilde__, P.s, P.h1 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_214,(0,1):C.GC_65})

V_78 = Vertex(name = 'V_78',
              particles = [ P.b__tilde__, P.s, P.h1 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_226,(0,1):C.GC_66})

V_79 = Vertex(name = 'V_79',
              particles = [ P.d__tilde__, P.b, P.h1 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_206,(0,1):C.GC_67})

V_80 = Vertex(name = 'V_80',
              particles = [ P.s__tilde__, P.b, P.h1 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_218,(0,1):C.GC_68})

V_81 = Vertex(name = 'V_81',
              particles = [ P.b__tilde__, P.b, P.h1 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_230,(0,1):C.GC_69})

V_82 = Vertex(name = 'V_82',
              particles = [ P.d__tilde__, P.d, P.h2 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_199,(0,1):C.GC_88})

V_83 = Vertex(name = 'V_83',
              particles = [ P.s__tilde__, P.d, P.h2 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_211,(0,1):C.GC_89})

V_84 = Vertex(name = 'V_84',
              particles = [ P.b__tilde__, P.d, P.h2 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_223,(0,1):C.GC_90})

V_85 = Vertex(name = 'V_85',
              particles = [ P.d__tilde__, P.s, P.h2 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_203,(0,1):C.GC_91})

V_86 = Vertex(name = 'V_86',
              particles = [ P.s__tilde__, P.s, P.h2 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_215,(0,1):C.GC_92})

V_87 = Vertex(name = 'V_87',
              particles = [ P.b__tilde__, P.s, P.h2 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_227,(0,1):C.GC_93})

V_88 = Vertex(name = 'V_88',
              particles = [ P.d__tilde__, P.b, P.h2 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_207,(0,1):C.GC_94})

V_89 = Vertex(name = 'V_89',
              particles = [ P.s__tilde__, P.b, P.h2 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_219,(0,1):C.GC_95})

V_90 = Vertex(name = 'V_90',
              particles = [ P.b__tilde__, P.b, P.h2 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_231,(0,1):C.GC_96})

V_91 = Vertex(name = 'V_91',
              particles = [ P.d__tilde__, P.d, P.h3 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_200,(0,1):C.GC_115})

V_92 = Vertex(name = 'V_92',
              particles = [ P.s__tilde__, P.d, P.h3 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_212,(0,1):C.GC_116})

V_93 = Vertex(name = 'V_93',
              particles = [ P.b__tilde__, P.d, P.h3 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_224,(0,1):C.GC_117})

V_94 = Vertex(name = 'V_94',
              particles = [ P.d__tilde__, P.s, P.h3 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_204,(0,1):C.GC_118})

V_95 = Vertex(name = 'V_95',
              particles = [ P.s__tilde__, P.s, P.h3 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_216,(0,1):C.GC_119})

V_96 = Vertex(name = 'V_96',
              particles = [ P.b__tilde__, P.s, P.h3 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_228,(0,1):C.GC_120})

V_97 = Vertex(name = 'V_97',
              particles = [ P.d__tilde__, P.b, P.h3 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_208,(0,1):C.GC_121})

V_98 = Vertex(name = 'V_98',
              particles = [ P.s__tilde__, P.b, P.h3 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_220,(0,1):C.GC_122})

V_99 = Vertex(name = 'V_99',
              particles = [ P.b__tilde__, P.b, P.h3 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_232,(0,1):C.GC_123})

V_100 = Vertex(name = 'V_100',
               particles = [ P.d__tilde__, P.d, P.z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_17,(0,1):C.GC_19})

V_101 = Vertex(name = 'V_101',
               particles = [ P.s__tilde__, P.s, P.z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_17,(0,1):C.GC_19})

V_102 = Vertex(name = 'V_102',
               particles = [ P.b__tilde__, P.b, P.z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,1):C.GC_19,(0,0):C.GC_17})

V_103 = Vertex(name = 'V_103',
               particles = [ P.u__tilde__, P.d, P.h__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_197,(0,1):C.GC_160})

V_104 = Vertex(name = 'V_104',
               particles = [ P.c__tilde__, P.d, P.h__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_209,(0,1):C.GC_161})

V_105 = Vertex(name = 'V_105',
               particles = [ P.t__tilde__, P.d, P.h__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_221,(0,1):C.GC_162})

V_106 = Vertex(name = 'V_106',
               particles = [ P.u__tilde__, P.s, P.h__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_201,(0,1):C.GC_163})

V_107 = Vertex(name = 'V_107',
               particles = [ P.c__tilde__, P.s, P.h__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_213,(0,1):C.GC_164})

V_108 = Vertex(name = 'V_108',
               particles = [ P.t__tilde__, P.s, P.h__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_225,(0,1):C.GC_165})

V_109 = Vertex(name = 'V_109',
               particles = [ P.u__tilde__, P.b, P.h__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_205,(0,1):C.GC_166})

V_110 = Vertex(name = 'V_110',
               particles = [ P.c__tilde__, P.b, P.h__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_217,(0,1):C.GC_167})

V_111 = Vertex(name = 'V_111',
               particles = [ P.t__tilde__, P.b, P.h__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_229,(0,1):C.GC_168})

V_112 = Vertex(name = 'V_112',
               particles = [ P.u__tilde__, P.d, P.w__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_193})

V_113 = Vertex(name = 'V_113',
               particles = [ P.c__tilde__, P.d, P.w__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_195})

V_114 = Vertex(name = 'V_114',
               particles = [ P.u__tilde__, P.s, P.w__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_194})

V_115 = Vertex(name = 'V_115',
               particles = [ P.c__tilde__, P.s, P.w__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_196})

V_116 = Vertex(name = 'V_116',
               particles = [ P.t__tilde__, P.b, P.w__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_12})

V_117 = Vertex(name = 'V_117',
               particles = [ P.e__plus__, P.e__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_3})

V_118 = Vertex(name = 'V_118',
               particles = [ P.m__plus__, P.m__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_3})

V_119 = Vertex(name = 'V_119',
               particles = [ P.tt__plus__, P.tt__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_3})

V_120 = Vertex(name = 'V_120',
               particles = [ P.e__plus__, P.e__minus__, P.h1 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_175,(0,0):C.GC_234,(0,1):C.GC_70})

V_121 = Vertex(name = 'V_121',
               particles = [ P.m__plus__, P.e__minus__, P.h1 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_246,(0,1):C.GC_71})

V_122 = Vertex(name = 'V_122',
               particles = [ P.tt__plus__, P.e__minus__, P.h1 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_258,(0,1):C.GC_72})

V_123 = Vertex(name = 'V_123',
               particles = [ P.e__plus__, P.m__minus__, P.h1 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_238,(0,1):C.GC_73})

V_124 = Vertex(name = 'V_124',
               particles = [ P.m__plus__, P.m__minus__, P.h1 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_178,(0,0):C.GC_250,(0,1):C.GC_74})

V_125 = Vertex(name = 'V_125',
               particles = [ P.tt__plus__, P.m__minus__, P.h1 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_262,(0,1):C.GC_75})

V_126 = Vertex(name = 'V_126',
               particles = [ P.e__plus__, P.tt__minus__, P.h1 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_242,(0,1):C.GC_76})

V_127 = Vertex(name = 'V_127',
               particles = [ P.m__plus__, P.tt__minus__, P.h1 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_254,(0,1):C.GC_77})

V_128 = Vertex(name = 'V_128',
               particles = [ P.tt__plus__, P.tt__minus__, P.h1 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_181,(0,0):C.GC_266,(0,1):C.GC_78})

V_129 = Vertex(name = 'V_129',
               particles = [ P.e__plus__, P.e__minus__, P.h2 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_176,(0,0):C.GC_235,(0,1):C.GC_97})

V_130 = Vertex(name = 'V_130',
               particles = [ P.m__plus__, P.e__minus__, P.h2 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_247,(0,1):C.GC_98})

V_131 = Vertex(name = 'V_131',
               particles = [ P.tt__plus__, P.e__minus__, P.h2 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_259,(0,1):C.GC_99})

V_132 = Vertex(name = 'V_132',
               particles = [ P.e__plus__, P.m__minus__, P.h2 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_239,(0,1):C.GC_100})

V_133 = Vertex(name = 'V_133',
               particles = [ P.m__plus__, P.m__minus__, P.h2 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_179,(0,0):C.GC_251,(0,1):C.GC_101})

V_134 = Vertex(name = 'V_134',
               particles = [ P.tt__plus__, P.m__minus__, P.h2 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_263,(0,1):C.GC_102})

V_135 = Vertex(name = 'V_135',
               particles = [ P.e__plus__, P.tt__minus__, P.h2 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_243,(0,1):C.GC_103})

V_136 = Vertex(name = 'V_136',
               particles = [ P.m__plus__, P.tt__minus__, P.h2 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_255,(0,1):C.GC_104})

V_137 = Vertex(name = 'V_137',
               particles = [ P.tt__plus__, P.tt__minus__, P.h2 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_182,(0,0):C.GC_267,(0,1):C.GC_105})

V_138 = Vertex(name = 'V_138',
               particles = [ P.e__plus__, P.e__minus__, P.h3 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_177,(0,0):C.GC_236,(0,1):C.GC_124})

V_139 = Vertex(name = 'V_139',
               particles = [ P.m__plus__, P.e__minus__, P.h3 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_248,(0,1):C.GC_125})

V_140 = Vertex(name = 'V_140',
               particles = [ P.tt__plus__, P.e__minus__, P.h3 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_260,(0,1):C.GC_126})

V_141 = Vertex(name = 'V_141',
               particles = [ P.e__plus__, P.m__minus__, P.h3 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_240,(0,1):C.GC_127})

V_142 = Vertex(name = 'V_142',
               particles = [ P.m__plus__, P.m__minus__, P.h3 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_180,(0,0):C.GC_252,(0,1):C.GC_128})

V_143 = Vertex(name = 'V_143',
               particles = [ P.tt__plus__, P.m__minus__, P.h3 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_264,(0,1):C.GC_129})

V_144 = Vertex(name = 'V_144',
               particles = [ P.e__plus__, P.tt__minus__, P.h3 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_244,(0,1):C.GC_130})

V_145 = Vertex(name = 'V_145',
               particles = [ P.m__plus__, P.tt__minus__, P.h3 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_256,(0,1):C.GC_131})

V_146 = Vertex(name = 'V_146',
               particles = [ P.tt__plus__, P.tt__minus__, P.h3 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_183,(0,0):C.GC_268,(0,1):C.GC_132})

V_147 = Vertex(name = 'V_147',
               particles = [ P.e__plus__, P.e__minus__, P.z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV4 ],
               couplings = {(0,0):C.GC_17,(0,1):C.GC_20})

V_148 = Vertex(name = 'V_148',
               particles = [ P.m__plus__, P.m__minus__, P.z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV4 ],
               couplings = {(0,0):C.GC_17,(0,1):C.GC_20})

V_149 = Vertex(name = 'V_149',
               particles = [ P.tt__plus__, P.tt__minus__, P.z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV4 ],
               couplings = {(0,0):C.GC_17,(0,1):C.GC_20})

V_150 = Vertex(name = 'V_150',
               particles = [ P.ve__tilde__, P.e__minus__, P.h__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_233})

V_151 = Vertex(name = 'V_151',
               particles = [ P.vm__tilde__, P.e__minus__, P.h__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_245})

V_152 = Vertex(name = 'V_152',
               particles = [ P.vt__tilde__, P.e__minus__, P.h__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_257})

V_153 = Vertex(name = 'V_153',
               particles = [ P.ve__tilde__, P.m__minus__, P.h__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_237})

V_154 = Vertex(name = 'V_154',
               particles = [ P.vm__tilde__, P.m__minus__, P.h__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_249})

V_155 = Vertex(name = 'V_155',
               particles = [ P.vt__tilde__, P.m__minus__, P.h__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_261})

V_156 = Vertex(name = 'V_156',
               particles = [ P.ve__tilde__, P.tt__minus__, P.h__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_241})

V_157 = Vertex(name = 'V_157',
               particles = [ P.vm__tilde__, P.tt__minus__, P.h__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_253})

V_158 = Vertex(name = 'V_158',
               particles = [ P.vt__tilde__, P.tt__minus__, P.h__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_265})

V_159 = Vertex(name = 'V_159',
               particles = [ P.ve__tilde__, P.e__minus__, P.w__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_12})

V_160 = Vertex(name = 'V_160',
               particles = [ P.vm__tilde__, P.m__minus__, P.w__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_12})

V_161 = Vertex(name = 'V_161',
               particles = [ P.vt__tilde__, P.tt__minus__, P.w__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_12})

V_162 = Vertex(name = 'V_162',
               particles = [ P.d__tilde__, P.u, P.h__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_269,(0,1):C.GC_142})

V_163 = Vertex(name = 'V_163',
               particles = [ P.s__tilde__, P.u, P.h__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_281,(0,1):C.GC_143})

V_164 = Vertex(name = 'V_164',
               particles = [ P.b__tilde__, P.u, P.h__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_293,(0,1):C.GC_144})

V_165 = Vertex(name = 'V_165',
               particles = [ P.d__tilde__, P.c, P.h__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_273,(0,1):C.GC_145})

V_166 = Vertex(name = 'V_166',
               particles = [ P.s__tilde__, P.c, P.h__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_285,(0,1):C.GC_146})

V_167 = Vertex(name = 'V_167',
               particles = [ P.b__tilde__, P.c, P.h__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_297,(0,1):C.GC_147})

V_168 = Vertex(name = 'V_168',
               particles = [ P.d__tilde__, P.t, P.h__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_277,(0,1):C.GC_148})

V_169 = Vertex(name = 'V_169',
               particles = [ P.s__tilde__, P.t, P.h__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_289,(0,1):C.GC_149})

V_170 = Vertex(name = 'V_170',
               particles = [ P.b__tilde__, P.t, P.h__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_301,(0,1):C.GC_150})

V_171 = Vertex(name = 'V_171',
               particles = [ P.d__tilde__, P.u, P.w__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_13})

V_172 = Vertex(name = 'V_172',
               particles = [ P.s__tilde__, P.u, P.w__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_14})

V_173 = Vertex(name = 'V_173',
               particles = [ P.d__tilde__, P.c, P.w__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_15})

V_174 = Vertex(name = 'V_174',
               particles = [ P.s__tilde__, P.c, P.w__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_16})

V_175 = Vertex(name = 'V_175',
               particles = [ P.b__tilde__, P.t, P.w__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_12})

V_176 = Vertex(name = 'V_176',
               particles = [ P.u__tilde__, P.u, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2})

V_177 = Vertex(name = 'V_177',
               particles = [ P.c__tilde__, P.c, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2})

V_178 = Vertex(name = 'V_178',
               particles = [ P.t__tilde__, P.t, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2})

V_179 = Vertex(name = 'V_179',
               particles = [ P.u__tilde__, P.u, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_6})

V_180 = Vertex(name = 'V_180',
               particles = [ P.c__tilde__, P.c, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_6})

V_181 = Vertex(name = 'V_181',
               particles = [ P.t__tilde__, P.t, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_6})

V_182 = Vertex(name = 'V_182',
               particles = [ P.u__tilde__, P.u, P.h1 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_184,(0,0):C.GC_270,(0,1):C.GC_79})

V_183 = Vertex(name = 'V_183',
               particles = [ P.c__tilde__, P.u, P.h1 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_282,(0,1):C.GC_80})

V_184 = Vertex(name = 'V_184',
               particles = [ P.t__tilde__, P.u, P.h1 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_294,(0,1):C.GC_81})

V_185 = Vertex(name = 'V_185',
               particles = [ P.u__tilde__, P.c, P.h1 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_274,(0,1):C.GC_82})

V_186 = Vertex(name = 'V_186',
               particles = [ P.c__tilde__, P.c, P.h1 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_187,(0,0):C.GC_286,(0,1):C.GC_83})

V_187 = Vertex(name = 'V_187',
               particles = [ P.t__tilde__, P.c, P.h1 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_298,(0,1):C.GC_84})

V_188 = Vertex(name = 'V_188',
               particles = [ P.u__tilde__, P.t, P.h1 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_278,(0,1):C.GC_85})

V_189 = Vertex(name = 'V_189',
               particles = [ P.c__tilde__, P.t, P.h1 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_290,(0,1):C.GC_86})

V_190 = Vertex(name = 'V_190',
               particles = [ P.t__tilde__, P.t, P.h1 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_190,(0,0):C.GC_302,(0,1):C.GC_87})

V_191 = Vertex(name = 'V_191',
               particles = [ P.u__tilde__, P.u, P.h2 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_185,(0,0):C.GC_271,(0,1):C.GC_106})

V_192 = Vertex(name = 'V_192',
               particles = [ P.c__tilde__, P.u, P.h2 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_283,(0,1):C.GC_107})

V_193 = Vertex(name = 'V_193',
               particles = [ P.t__tilde__, P.u, P.h2 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_295,(0,1):C.GC_108})

V_194 = Vertex(name = 'V_194',
               particles = [ P.u__tilde__, P.c, P.h2 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_275,(0,1):C.GC_109})

V_195 = Vertex(name = 'V_195',
               particles = [ P.c__tilde__, P.c, P.h2 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_188,(0,0):C.GC_287,(0,1):C.GC_110})

V_196 = Vertex(name = 'V_196',
               particles = [ P.t__tilde__, P.c, P.h2 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_299,(0,1):C.GC_111})

V_197 = Vertex(name = 'V_197',
               particles = [ P.u__tilde__, P.t, P.h2 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_279,(0,1):C.GC_112})

V_198 = Vertex(name = 'V_198',
               particles = [ P.c__tilde__, P.t, P.h2 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_291,(0,1):C.GC_113})

V_199 = Vertex(name = 'V_199',
               particles = [ P.t__tilde__, P.t, P.h2 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_191,(0,0):C.GC_303,(0,1):C.GC_114})

V_200 = Vertex(name = 'V_200',
               particles = [ P.u__tilde__, P.u, P.h3 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_186,(0,0):C.GC_272,(0,1):C.GC_133})

V_201 = Vertex(name = 'V_201',
               particles = [ P.c__tilde__, P.u, P.h3 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_284,(0,1):C.GC_134})

V_202 = Vertex(name = 'V_202',
               particles = [ P.t__tilde__, P.u, P.h3 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_296,(0,1):C.GC_135})

V_203 = Vertex(name = 'V_203',
               particles = [ P.u__tilde__, P.c, P.h3 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_276,(0,1):C.GC_136})

V_204 = Vertex(name = 'V_204',
               particles = [ P.c__tilde__, P.c, P.h3 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_189,(0,0):C.GC_288,(0,1):C.GC_137})

V_205 = Vertex(name = 'V_205',
               particles = [ P.t__tilde__, P.c, P.h3 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_300,(0,1):C.GC_138})

V_206 = Vertex(name = 'V_206',
               particles = [ P.u__tilde__, P.t, P.h3 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_280,(0,1):C.GC_139})

V_207 = Vertex(name = 'V_207',
               particles = [ P.c__tilde__, P.t, P.h3 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_292,(0,1):C.GC_140})

V_208 = Vertex(name = 'V_208',
               particles = [ P.t__tilde__, P.t, P.h3 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
               couplings = {(0,2):C.GC_192,(0,0):C.GC_304,(0,1):C.GC_141})

V_209 = Vertex(name = 'V_209',
               particles = [ P.u__tilde__, P.u, P.z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV5 ],
               couplings = {(0,0):C.GC_18,(0,1):C.GC_19})

V_210 = Vertex(name = 'V_210',
               particles = [ P.c__tilde__, P.c, P.z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV5 ],
               couplings = {(0,0):C.GC_18,(0,1):C.GC_19})

V_211 = Vertex(name = 'V_211',
               particles = [ P.t__tilde__, P.t, P.z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV5 ],
               couplings = {(0,0):C.GC_18,(0,1):C.GC_19})

V_212 = Vertex(name = 'V_212',
               particles = [ P.e__plus__, P.ve, P.h__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_151})

V_213 = Vertex(name = 'V_213',
               particles = [ P.m__plus__, P.ve, P.h__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_152})

V_214 = Vertex(name = 'V_214',
               particles = [ P.tt__plus__, P.ve, P.h__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_153})

V_215 = Vertex(name = 'V_215',
               particles = [ P.e__plus__, P.vm, P.h__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_154})

V_216 = Vertex(name = 'V_216',
               particles = [ P.m__plus__, P.vm, P.h__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_155})

V_217 = Vertex(name = 'V_217',
               particles = [ P.tt__plus__, P.vm, P.h__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_156})

V_218 = Vertex(name = 'V_218',
               particles = [ P.e__plus__, P.vt, P.h__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_157})

V_219 = Vertex(name = 'V_219',
               particles = [ P.m__plus__, P.vt, P.h__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_158})

V_220 = Vertex(name = 'V_220',
               particles = [ P.tt__plus__, P.vt, P.h__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_159})

V_221 = Vertex(name = 'V_221',
               particles = [ P.e__plus__, P.ve, P.w__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_12})

V_222 = Vertex(name = 'V_222',
               particles = [ P.m__plus__, P.vm, P.w__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_12})

V_223 = Vertex(name = 'V_223',
               particles = [ P.tt__plus__, P.vt, P.w__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_12})

V_224 = Vertex(name = 'V_224',
               particles = [ P.ve__tilde__, P.ve, P.z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_25})

V_225 = Vertex(name = 'V_225',
               particles = [ P.vm__tilde__, P.vm, P.z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_25})

V_226 = Vertex(name = 'V_226',
               particles = [ P.vt__tilde__, P.vt, P.z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_25})

