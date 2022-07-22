# This file was automatically created by FeynRules 2.4.78
# Mathematica version: 12.0.0 for Mac OS X x86 (64-bit) (April 7, 2019)
# Date: Wed 1 Apr 2020 19:35:44

import configuration
from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L

if configuration.bottomYukawa:
    
    V_171_bot = Vertex(name = 'V_171_bot',
                   particles = [ P.b__tilde__, P.b, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS37 ],
                   couplings = {(0,0):C.GC_1842_bot})
    V_172_bot = Vertex(name = 'V_172_bot',
                   particles = [ P.b__tilde__, P.b, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFS37 ],
                   couplings = {(0,0):C.GC_546_bot})
    V_182_bot = Vertex(name = 'V_182_bot',
                   particles = [ P.b__tilde__, P.b, P.H, P.H, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFSSS6 ],
                   couplings = {(0,0):C.GC_106_bot})
    V_186_bot = Vertex(name = 'V_186_bot',
                   particles = [ P.b__tilde__, P.b, P.H, P.H ],
                   color = [ 'Identity(1,2)' ],
                   lorentz = [ L.FFSS26 ],
                   couplings = {(0,0):C.GC_1822_bot})

V_1 = Vertex(name = 'V_1',
             particles = [ P.G0, P.G0, P.G0, P.G0, P.G0, P.G0 ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS2 ],
             couplings = {(0,0):C.GC_60})

V_2 = Vertex(name = 'V_2',
             particles = [ P.G0, P.G0, P.G0, P.G0, P.G__minus__, P.G__plus__ ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS2 ],
             couplings = {(0,0):C.GC_58})

V_3 = Vertex(name = 'V_3',
             particles = [ P.G0, P.G0, P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS2 ],
             couplings = {(0,0):C.GC_57})

V_4 = Vertex(name = 'V_4',
             particles = [ P.G__minus__, P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__, P.G__plus__ ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS2 ],
             couplings = {(0,0):C.GC_59})

V_5 = Vertex(name = 'V_5',
             particles = [ P.G0, P.G0, P.G0, P.G0, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS2 ],
             couplings = {(0,0):C.GC_58})

V_6 = Vertex(name = 'V_6',
             particles = [ P.G0, P.G0, P.G__minus__, P.G__plus__, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS2 ],
             couplings = {(0,0):C.GC_56})

V_7 = Vertex(name = 'V_7',
             particles = [ P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS2 ],
             couplings = {(0,0):C.GC_57})

V_8 = Vertex(name = 'V_8',
             particles = [ P.G0, P.G0, P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS2 ],
             couplings = {(0,0):C.GC_58})

V_9 = Vertex(name = 'V_9',
             particles = [ P.G__minus__, P.G__plus__, P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS2 ],
             couplings = {(0,0):C.GC_58})

V_10 = Vertex(name = 'V_10',
              particles = [ P.H, P.H, P.H, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSSS2 ],
              couplings = {(0,0):C.GC_60})

V_11 = Vertex(name = 'V_11',
              particles = [ P.G0, P.G0, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.SSSS12, L.SSSS7 ],
              couplings = {(0,1):C.GC_168,(0,0):C.GC_14})

V_12 = Vertex(name = 'V_12',
              particles = [ P.G0, P.G0, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.SSSS7 ],
              couplings = {(0,0):C.GC_320})

V_13 = Vertex(name = 'V_13',
              particles = [ P.G0, P.G0, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSSS7, L.SSSS8 ],
              couplings = {(0,0):C.GC_164,(0,1):C.GC_13})

V_14 = Vertex(name = 'V_14',
              particles = [ P.G0, P.G0, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSSS7 ],
              couplings = {(0,0):C.GC_318})

V_15 = Vertex(name = 'V_15',
              particles = [ P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSSS11, L.SSSS7, L.SSSS8 ],
              couplings = {(0,1):C.GC_166,(0,0):C.GC_55,(0,2):C.GC_62})

V_16 = Vertex(name = 'V_16',
              particles = [ P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSSS7 ],
              couplings = {(0,0):C.GC_319})

V_17 = Vertex(name = 'V_17',
              particles = [ P.G0, P.G0, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS7, L.SSSS8, L.SSSS9 ],
              couplings = {(0,0):C.GC_165,(0,2):C.GC_62,(0,1):C.GC_54})

V_18 = Vertex(name = 'V_18',
              particles = [ P.G0, P.G0, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS7 ],
              couplings = {(0,0):C.GC_318})

V_19 = Vertex(name = 'V_19',
              particles = [ P.G0, P.G0, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS7 ],
              couplings = {(0,0):C.GC_410})

V_20 = Vertex(name = 'V_20',
              particles = [ P.G__minus__, P.G__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS7, L.SSSS8 ],
              couplings = {(0,0):C.GC_163,(0,1):C.GC_13})

V_21 = Vertex(name = 'V_21',
              particles = [ P.G__minus__, P.G__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS7 ],
              couplings = {(0,0):C.GC_318})

V_22 = Vertex(name = 'V_22',
              particles = [ P.G__minus__, P.G__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS7 ],
              couplings = {(0,0):C.GC_410})

V_23 = Vertex(name = 'V_23',
              particles = [ P.H, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS12, L.SSSS7 ],
              couplings = {(0,1):C.GC_167,(0,0):C.GC_14})

V_24 = Vertex(name = 'V_24',
              particles = [ P.H, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS7 ],
              couplings = {(0,0):C.GC_320})

V_25 = Vertex(name = 'V_25',
              particles = [ P.H, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS7 ],
              couplings = {(0,0):C.GC_411})

V_26 = Vertex(name = 'V_26',
              particles = [ P.G0, P.G0, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS6, L.SSS7, L.SSS9 ],
              couplings = {(0,0):C.GC_321,(0,1):C.GC_334,(0,2):C.GC_328})

V_27 = Vertex(name = 'V_27',
              particles = [ P.G0, P.G0, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS6 ],
              couplings = {(0,0):C.GC_430})

V_28 = Vertex(name = 'V_28',
              particles = [ P.G__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS6, L.SSS9 ],
              couplings = {(0,0):C.GC_321,(0,1):C.GC_421})

V_29 = Vertex(name = 'V_29',
              particles = [ P.G__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS6 ],
              couplings = {(0,0):C.GC_429})

V_30 = Vertex(name = 'V_30',
              particles = [ P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS10, L.SSS6 ],
              couplings = {(0,1):C.GC_322,(0,0):C.GC_422})

V_31 = Vertex(name = 'V_31',
              particles = [ P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS6 ],
              couplings = {(0,0):C.GC_431})

V_32 = Vertex(name = 'V_32',
              particles = [ P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS6 ],
              couplings = {(0,0):C.GC_420})

V_33 = Vertex(name = 'V_33',
              particles = [ P.G0, P.G0, P.G0, P.G0, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSS2 ],
              couplings = {(0,0):C.GC_331})

V_34 = Vertex(name = 'V_34',
              particles = [ P.G0, P.G0, P.G__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSS2 ],
              couplings = {(0,0):C.GC_329})

V_35 = Vertex(name = 'V_35',
              particles = [ P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSS2 ],
              couplings = {(0,0):C.GC_330})

V_36 = Vertex(name = 'V_36',
              particles = [ P.G0, P.G0, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSS2 ],
              couplings = {(0,0):C.GC_331})

V_37 = Vertex(name = 'V_37',
              particles = [ P.G__minus__, P.G__plus__, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSS2 ],
              couplings = {(0,0):C.GC_331})

V_38 = Vertex(name = 'V_38',
              particles = [ P.H, P.H, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSS2 ],
              couplings = {(0,0):C.GC_332})

V_39 = Vertex(name = 'V_39',
              particles = [ P.a, P.a, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_311,(0,0):C.GC_6})

V_40 = Vertex(name = 'V_40',
              particles = [ P.a, P.a, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS14 ],
              couplings = {(0,0):C.GC_504})

V_41 = Vertex(name = 'V_41',
              particles = [ P.a, P.a, P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSSSS2 ],
              couplings = {(0,0):C.GC_150})

V_42 = Vertex(name = 'V_42',
              particles = [ P.a, P.G0, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS5 ],
              couplings = {(0,0):C.GC_471})

V_43 = Vertex(name = 'V_43',
              particles = [ P.a, P.G0, P.G__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSSSS19 ],
              couplings = {(0,0):C.GC_135})

V_44 = Vertex(name = 'V_44',
              particles = [ P.a, P.G0, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSSS9 ],
              couplings = {(0,0):C.GC_350})

V_45 = Vertex(name = 'V_45',
              particles = [ P.a, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS5 ],
              couplings = {(0,0):C.GC_3})

V_46 = Vertex(name = 'V_46',
              particles = [ P.a, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS5 ],
              couplings = {(0,0):C.GC_493})

V_47 = Vertex(name = 'V_47',
              particles = [ P.a, P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSSSS20 ],
              couplings = {(0,0):C.GC_134})

V_48 = Vertex(name = 'V_48',
              particles = [ P.G0, P.G__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS10 ],
              couplings = {(0,0):C.GC_61})

V_49 = Vertex(name = 'V_49',
              particles = [ P.G0, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSS8 ],
              couplings = {(0,0):C.GC_333})

V_50 = Vertex(name = 'V_50',
              particles = [ P.a, P.a, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS15 ],
              couplings = {(0,0):C.GC_310})

V_51 = Vertex(name = 'V_51',
              particles = [ P.a, P.a, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS15 ],
              couplings = {(0,0):C.GC_310})

V_52 = Vertex(name = 'V_52',
              particles = [ P.a, P.a, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS12 ],
              couplings = {(0,0):C.GC_461})

V_53 = Vertex(name = 'V_53',
              particles = [ P.g, P.g, P.G0, P.G0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVSS15 ],
              couplings = {(0,0):C.GC_65})

V_54 = Vertex(name = 'V_54',
              particles = [ P.g, P.g, P.G__minus__, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVSS15 ],
              couplings = {(0,0):C.GC_65})

V_55 = Vertex(name = 'V_55',
              particles = [ P.g, P.g, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVSS15 ],
              couplings = {(0,0):C.GC_65})

V_56 = Vertex(name = 'V_56',
              particles = [ P.g, P.g, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVS12 ],
              couplings = {(0,0):C.GC_336})

V_57 = Vertex(name = 'V_57',
              particles = [ P.a, P.W__minus__, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_122,(0,0):C.GC_201})

V_58 = Vertex(name = 'V_58',
              particles = [ P.a, P.W__minus__, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS14 ],
              couplings = {(0,0):C.GC_496})

V_59 = Vertex(name = 'V_59',
              particles = [ P.a, P.W__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_121,(0,0):C.GC_200})

V_60 = Vertex(name = 'V_60',
              particles = [ P.a, P.W__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS14 ],
              couplings = {(0,0):C.GC_495})

V_61 = Vertex(name = 'V_61',
              particles = [ P.a, P.W__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11, L.VVS12 ],
              couplings = {(0,1):C.GC_342,(0,0):C.GC_371})

V_62 = Vertex(name = 'V_62',
              particles = [ P.a, P.W__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11 ],
              couplings = {(0,0):C.GC_530})

V_63 = Vertex(name = 'V_63',
              particles = [ P.a, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVV10, L.VVV8, L.VVV9 ],
              couplings = {(0,0):C.GC_258,(0,1):C.GC_4,(0,2):C.GC_417})

V_64 = Vertex(name = 'V_64',
              particles = [ P.a, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVV8 ],
              couplings = {(0,0):C.GC_469})

V_65 = Vertex(name = 'V_65',
              particles = [ P.a, P.W__plus__, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_122,(0,0):C.GC_201})

V_66 = Vertex(name = 'V_66',
              particles = [ P.a, P.W__plus__, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS14 ],
              couplings = {(0,0):C.GC_496})

V_67 = Vertex(name = 'V_67',
              particles = [ P.a, P.W__plus__, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_123,(0,0):C.GC_202})

V_68 = Vertex(name = 'V_68',
              particles = [ P.a, P.W__plus__, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS14 ],
              couplings = {(0,0):C.GC_497})

V_69 = Vertex(name = 'V_69',
              particles = [ P.a, P.W__plus__, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11, L.VVS12 ],
              couplings = {(0,1):C.GC_343,(0,0):C.GC_372})

V_70 = Vertex(name = 'V_70',
              particles = [ P.a, P.W__plus__, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11 ],
              couplings = {(0,0):C.GC_535})

V_71 = Vertex(name = 'V_71',
              particles = [ P.W__minus__, P.W__plus__, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_70,(0,0):C.GC_179})

V_72 = Vertex(name = 'V_72',
              particles = [ P.W__minus__, P.W__plus__, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS14 ],
              couplings = {(0,0):C.GC_473})

V_73 = Vertex(name = 'V_73',
              particles = [ P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_70,(0,0):C.GC_179})

V_74 = Vertex(name = 'V_74',
              particles = [ P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS14 ],
              couplings = {(0,0):C.GC_475})

V_75 = Vertex(name = 'V_75',
              particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_70,(0,0):C.GC_179})

V_76 = Vertex(name = 'V_76',
              particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS14 ],
              couplings = {(0,0):C.GC_474})

V_77 = Vertex(name = 'V_77',
              particles = [ P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS11, L.VVS12 ],
              couplings = {(0,1):C.GC_339,(0,0):C.GC_363})

V_78 = Vertex(name = 'V_78',
              particles = [ P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS11 ],
              couplings = {(0,0):C.GC_528})

V_79 = Vertex(name = 'V_79',
              particles = [ P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVV10, L.VVV7, L.VVV8 ],
              couplings = {(0,0):C.GC_124,(0,2):C.GC_198,(0,1):C.GC_412})

V_80 = Vertex(name = 'V_80',
              particles = [ P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVV8 ],
              couplings = {(0,0):C.GC_484})

V_81 = Vertex(name = 'V_81',
              particles = [ P.a, P.Z, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_312,(0,0):C.GC_521})

V_82 = Vertex(name = 'V_82',
              particles = [ P.a, P.Z, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_313,(0,0):C.GC_276})

V_83 = Vertex(name = 'V_83',
              particles = [ P.a, P.Z, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS14 ],
              couplings = {(0,0):C.GC_522})

V_84 = Vertex(name = 'V_84',
              particles = [ P.a, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_312,(0,0):C.GC_521})

V_85 = Vertex(name = 'V_85',
              particles = [ P.a, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS11, L.VVS12 ],
              couplings = {(0,1):C.GC_462,(0,0):C.GC_543})

V_86 = Vertex(name = 'V_86',
              particles = [ P.W__minus__, P.Z, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_256,(0,0):C.GC_8})

V_87 = Vertex(name = 'V_87',
              particles = [ P.W__minus__, P.Z, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS14 ],
              couplings = {(0,0):C.GC_499})

V_88 = Vertex(name = 'V_88',
              particles = [ P.W__minus__, P.Z, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_255,(0,0):C.GC_7})

V_89 = Vertex(name = 'V_89',
              particles = [ P.W__minus__, P.Z, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS14 ],
              couplings = {(0,0):C.GC_500})

V_90 = Vertex(name = 'V_90',
              particles = [ P.W__minus__, P.Z, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11, L.VVS12 ],
              couplings = {(0,1):C.GC_401,(0,0):C.GC_323})

V_91 = Vertex(name = 'V_91',
              particles = [ P.W__minus__, P.Z, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11 ],
              couplings = {(0,0):C.GC_541})

V_92 = Vertex(name = 'V_92',
              particles = [ P.W__plus__, P.Z, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_256,(0,0):C.GC_8})

V_93 = Vertex(name = 'V_93',
              particles = [ P.W__plus__, P.Z, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS14 ],
              couplings = {(0,0):C.GC_499})

V_94 = Vertex(name = 'V_94',
              particles = [ P.W__plus__, P.Z, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_257,(0,0):C.GC_9})

V_95 = Vertex(name = 'V_95',
              particles = [ P.W__plus__, P.Z, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS14 ],
              couplings = {(0,0):C.GC_498})

V_96 = Vertex(name = 'V_96',
              particles = [ P.W__plus__, P.Z, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11, L.VVS12 ],
              couplings = {(0,1):C.GC_402,(0,0):C.GC_324})

V_97 = Vertex(name = 'V_97',
              particles = [ P.W__plus__, P.Z, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11 ],
              couplings = {(0,0):C.GC_536})

V_98 = Vertex(name = 'V_98',
              particles = [ P.Z, P.Z, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS14, L.VVSS15 ],
              couplings = {(0,1):C.GC_309,(0,0):C.GC_307})

V_99 = Vertex(name = 'V_99',
              particles = [ P.Z, P.Z, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS14 ],
              couplings = {(0,0):C.GC_523})

V_100 = Vertex(name = 'V_100',
               particles = [ P.Z, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSS14, L.VVSS15 ],
               couplings = {(0,1):C.GC_308,(0,0):C.GC_306})

V_101 = Vertex(name = 'V_101',
               particles = [ P.Z, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSS14 ],
               couplings = {(0,0):C.GC_524})

V_102 = Vertex(name = 'V_102',
               particles = [ P.Z, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSS14, L.VVSS15 ],
               couplings = {(0,1):C.GC_309,(0,0):C.GC_307})

V_103 = Vertex(name = 'V_103',
               particles = [ P.Z, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSS14 ],
               couplings = {(0,0):C.GC_525})

V_104 = Vertex(name = 'V_104',
               particles = [ P.Z, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVS11, L.VVS12 ],
               couplings = {(0,1):C.GC_460,(0,0):C.GC_459})

V_105 = Vertex(name = 'V_105',
               particles = [ P.Z, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVS11 ],
               couplings = {(0,0):C.GC_545})

V_106 = Vertex(name = 'V_106',
               particles = [ P.ghA, P.ghWm__tilde__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_531})

V_107 = Vertex(name = 'V_107',
               particles = [ P.ghA, P.ghWm__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_3})

V_108 = Vertex(name = 'V_108',
               particles = [ P.ghA, P.ghWm__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_470})

V_109 = Vertex(name = 'V_109',
               particles = [ P.ghA, P.ghWp__tilde__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_533})

V_110 = Vertex(name = 'V_110',
               particles = [ P.ghA, P.ghWp__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_4})

V_111 = Vertex(name = 'V_111',
               particles = [ P.ghA, P.ghWp__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_469})

V_112 = Vertex(name = 'V_112',
               particles = [ P.ghA, P.ghZ__tilde__, P.H ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_542})

V_113 = Vertex(name = 'V_113',
               particles = [ P.ghWm, P.ghA__tilde__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_372})

V_114 = Vertex(name = 'V_114',
               particles = [ P.ghWm, P.ghA__tilde__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_534})

V_115 = Vertex(name = 'V_115',
               particles = [ P.ghWm, P.ghA__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_3})

V_116 = Vertex(name = 'V_116',
               particles = [ P.ghWm, P.ghA__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_470})

V_117 = Vertex(name = 'V_117',
               particles = [ P.ghWm, P.ghWm__tilde__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_361})

V_118 = Vertex(name = 'V_118',
               particles = [ P.ghWm, P.ghWm__tilde__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_529})

V_119 = Vertex(name = 'V_119',
               particles = [ P.ghWm, P.ghWm__tilde__, P.H ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_362})

V_120 = Vertex(name = 'V_120',
               particles = [ P.ghWm, P.ghWm__tilde__, P.H ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_527})

V_121 = Vertex(name = 'V_121',
               particles = [ P.ghWm, P.ghWm__tilde__, P.a ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_4})

V_122 = Vertex(name = 'V_122',
               particles = [ P.ghWm, P.ghWm__tilde__, P.a ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_491})

V_123 = Vertex(name = 'V_123',
               particles = [ P.ghWm, P.ghWm__tilde__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_198})

V_124 = Vertex(name = 'V_124',
               particles = [ P.ghWm, P.ghWm__tilde__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_485})

V_125 = Vertex(name = 'V_125',
               particles = [ P.ghWm, P.ghZ__tilde__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_434})

V_126 = Vertex(name = 'V_126',
               particles = [ P.ghWm, P.ghZ__tilde__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_540})

V_127 = Vertex(name = 'V_127',
               particles = [ P.ghWm, P.ghZ__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_198})

V_128 = Vertex(name = 'V_128',
               particles = [ P.ghWm, P.ghZ__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_484})

V_129 = Vertex(name = 'V_129',
               particles = [ P.ghWp, P.ghA__tilde__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_371})

V_130 = Vertex(name = 'V_130',
               particles = [ P.ghWp, P.ghA__tilde__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_532})

V_131 = Vertex(name = 'V_131',
               particles = [ P.ghWp, P.ghA__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_4})

V_132 = Vertex(name = 'V_132',
               particles = [ P.ghWp, P.ghA__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_469})

V_133 = Vertex(name = 'V_133',
               particles = [ P.ghWp, P.ghWp__tilde__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_364})

V_134 = Vertex(name = 'V_134',
               particles = [ P.ghWp, P.ghWp__tilde__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_526})

V_135 = Vertex(name = 'V_135',
               particles = [ P.ghWp, P.ghWp__tilde__, P.H ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_362})

V_136 = Vertex(name = 'V_136',
               particles = [ P.ghWp, P.ghWp__tilde__, P.H ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_527})

V_137 = Vertex(name = 'V_137',
               particles = [ P.ghWp, P.ghWp__tilde__, P.a ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_3})

V_138 = Vertex(name = 'V_138',
               particles = [ P.ghWp, P.ghWp__tilde__, P.a ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_494})

V_139 = Vertex(name = 'V_139',
               particles = [ P.ghWp, P.ghWp__tilde__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_199})

V_140 = Vertex(name = 'V_140',
               particles = [ P.ghWp, P.ghWp__tilde__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_488})

V_141 = Vertex(name = 'V_141',
               particles = [ P.ghWp, P.ghZ__tilde__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_433})

V_142 = Vertex(name = 'V_142',
               particles = [ P.ghWp, P.ghZ__tilde__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_538})

V_143 = Vertex(name = 'V_143',
               particles = [ P.ghWp, P.ghZ__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_199})

V_144 = Vertex(name = 'V_144',
               particles = [ P.ghWp, P.ghZ__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_487})

V_145 = Vertex(name = 'V_145',
               particles = [ P.ghZ, P.ghA__tilde__, P.H ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_542})

V_146 = Vertex(name = 'V_146',
               particles = [ P.ghZ, P.ghWm__tilde__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_435})

V_147 = Vertex(name = 'V_147',
               particles = [ P.ghZ, P.ghWm__tilde__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_537})

V_148 = Vertex(name = 'V_148',
               particles = [ P.ghZ, P.ghWm__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_198})

V_149 = Vertex(name = 'V_149',
               particles = [ P.ghZ, P.ghWm__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_484})

V_150 = Vertex(name = 'V_150',
               particles = [ P.ghZ, P.ghWp__tilde__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_432})

V_151 = Vertex(name = 'V_151',
               particles = [ P.ghZ, P.ghWp__tilde__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_539})

V_152 = Vertex(name = 'V_152',
               particles = [ P.ghZ, P.ghWp__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_199})

V_153 = Vertex(name = 'V_153',
               particles = [ P.ghZ, P.ghWp__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_487})

V_154 = Vertex(name = 'V_154',
               particles = [ P.ghZ, P.ghZ__tilde__, P.H ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_458})

V_155 = Vertex(name = 'V_155',
               particles = [ P.ghZ, P.ghZ__tilde__, P.H ],
               color = [ '1' ],
               lorentz = [ L.UUS2 ],
               couplings = {(0,0):C.GC_544})

V_156 = Vertex(name = 'V_156',
               particles = [ P.ghG, P.ghG__tilde__, P.g ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.UUV2 ],
               couplings = {(0,0):C.GC_10})

V_157 = Vertex(name = 'V_157',
               particles = [ P.g, P.g, P.g ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVV8 ],
               couplings = {(0,0):C.GC_10})

V_158 = Vertex(name = 'V_158',
               particles = [ P.g, P.g, P.g, P.g ],
               color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
               lorentz = [ L.VVVV16, L.VVVV19, L.VVVV20 ],
               couplings = {(1,1):C.GC_12,(0,0):C.GC_12,(2,2):C.GC_12})

V_159 = Vertex(name = 'V_159',
               particles = [ P.g, P.g, P.g, P.G0, P.G0 ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVVSS25 ],
               couplings = {(0,0):C.GC_153})

V_160 = Vertex(name = 'V_160',
               particles = [ P.g, P.g, P.g, P.G__minus__, P.G__plus__ ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVVSS25 ],
               couplings = {(0,0):C.GC_153})

V_161 = Vertex(name = 'V_161',
               particles = [ P.g, P.g, P.g, P.H, P.H ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVVSS25 ],
               couplings = {(0,0):C.GC_153})

V_162 = Vertex(name = 'V_162',
               particles = [ P.g, P.g, P.g, P.H ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVVS17 ],
               couplings = {(0,0):C.GC_357})

V_163 = Vertex(name = 'V_163',
               particles = [ P.g, P.g, P.g, P.g, P.G0, P.G0 ],
               color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
               lorentz = [ L.VVVVSS6, L.VVVVSS8, L.VVVVSS9 ],
               couplings = {(1,1):C.GC_158,(0,0):C.GC_158,(2,2):C.GC_158})

V_164 = Vertex(name = 'V_164',
               particles = [ P.g, P.g, P.g, P.g, P.G__minus__, P.G__plus__ ],
               color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
               lorentz = [ L.VVVVSS6, L.VVVVSS8, L.VVVVSS9 ],
               couplings = {(1,1):C.GC_158,(0,0):C.GC_158,(2,2):C.GC_158})

V_165 = Vertex(name = 'V_165',
               particles = [ P.g, P.g, P.g, P.g, P.H, P.H ],
               color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
               lorentz = [ L.VVVVSS6, L.VVVVSS8, L.VVVVSS9 ],
               couplings = {(1,1):C.GC_158,(0,0):C.GC_158,(2,2):C.GC_158})

V_166 = Vertex(name = 'V_166',
               particles = [ P.g, P.g, P.g, P.g, P.H ],
               color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
               lorentz = [ L.VVVVS14, L.VVVVS17, L.VVVVS18 ],
               couplings = {(1,1):C.GC_359,(0,0):C.GC_359,(2,2):C.GC_359})

V_167 = Vertex(name = 'V_167',
               particles = [ P.b__tilde__, P.t, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS41, L.FFS45 ],
               couplings = {(0,0):C.GC_549,(0,1):C.GC_2079})

V_168 = Vertex(name = 'V_168',
               particles = [ P.b__tilde__, P.t, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS41 ],
               couplings = {(0,0):C.GC_2085})

V_169 = Vertex(name = 'V_169',
               particles = [ P.t__tilde__, P.t, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS45, L.FFS51, L.FFS55 ],
               couplings = {(0,1):C.GC_547,(0,0):C.GC_1834,(0,2):C.GC_1840})

V_170 = Vertex(name = 'V_170',
               particles = [ P.t__tilde__, P.t, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS51 ],
               couplings = {(0,0):C.GC_550})

V_171 = Vertex(name = 'V_171',
               particles = [ P.t__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS37 ],
               couplings = {(0,0):C.GC_1842})

V_172 = Vertex(name = 'V_172',
               particles = [ P.t__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS37 ],
               couplings = {(0,0):C.GC_546})

V_173 = Vertex(name = 'V_173',
               particles = [ P.b__tilde__, P.t, P.G0, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS7 ],
               couplings = {(0,0):C.GC_102})

V_174 = Vertex(name = 'V_174',
               particles = [ P.b__tilde__, P.t, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS7 ],
               couplings = {(0,0):C.GC_101})

V_175 = Vertex(name = 'V_175',
               particles = [ P.b__tilde__, P.t, P.G__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS7 ],
               couplings = {(0,0):C.GC_102})

V_176 = Vertex(name = 'V_176',
               particles = [ P.b__tilde__, P.t, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS27, L.FFSS30 ],
               couplings = {(0,0):C.GC_2024,(0,1):C.GC_1997})

V_177 = Vertex(name = 'V_177',
               particles = [ P.t__tilde__, P.t, P.G0, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS5 ],
               couplings = {(0,0):C.GC_108})

V_178 = Vertex(name = 'V_178',
               particles = [ P.t__tilde__, P.t, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS5 ],
               couplings = {(0,0):C.GC_107})

V_179 = Vertex(name = 'V_179',
               particles = [ P.t__tilde__, P.t, P.G0, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS6 ],
               couplings = {(0,0):C.GC_105})

V_180 = Vertex(name = 'V_180',
               particles = [ P.t__tilde__, P.t, P.G__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS6 ],
               couplings = {(0,0):C.GC_105})

V_181 = Vertex(name = 'V_181',
               particles = [ P.t__tilde__, P.t, P.G0, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS5 ],
               couplings = {(0,0):C.GC_107})

V_182 = Vertex(name = 'V_182',
               particles = [ P.t__tilde__, P.t, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS6 ],
               couplings = {(0,0):C.GC_106})

V_183 = Vertex(name = 'V_183',
               particles = [ P.t__tilde__, P.t, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS26 ],
               couplings = {(0,0):C.GC_1821})

V_184 = Vertex(name = 'V_184',
               particles = [ P.t__tilde__, P.t, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30, L.FFSS37, L.FFSS42 ],
               couplings = {(0,0):C.GC_1993,(0,2):C.GC_2005,(0,1):C.GC_2021})

V_185 = Vertex(name = 'V_185',
               particles = [ P.t__tilde__, P.t, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30, L.FFSS35, L.FFSS42 ],
               couplings = {(0,0):C.GC_1811,(0,2):C.GC_1817,(0,1):C.GC_1819})

V_186 = Vertex(name = 'V_186',
               particles = [ P.t__tilde__, P.t, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS26 ],
               couplings = {(0,0):C.GC_1822})

V_187 = Vertex(name = 'V_187',
               particles = [ P.t__tilde__, P.b, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS45, L.FFS52 ],
               couplings = {(0,1):C.GC_548,(0,0):C.GC_2079})

V_188 = Vertex(name = 'V_188',
               particles = [ P.t__tilde__, P.b, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS52 ],
               couplings = {(0,0):C.GC_2084})

V_189 = Vertex(name = 'V_189',
               particles = [ P.t__tilde__, P.b, P.G0, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS8 ],
               couplings = {(0,0):C.GC_103})

V_190 = Vertex(name = 'V_190',
               particles = [ P.t__tilde__, P.b, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS8 ],
               couplings = {(0,0):C.GC_104})

V_191 = Vertex(name = 'V_191',
               particles = [ P.t__tilde__, P.b, P.G__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS8 ],
               couplings = {(0,0):C.GC_103})

V_192 = Vertex(name = 'V_192',
               particles = [ P.t__tilde__, P.b, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30, L.FFSS36 ],
               couplings = {(0,1):C.GC_2023,(0,0):C.GC_1997})

V_193 = Vertex(name = 'V_193',
               particles = [ P.vt__tilde__, P.ta__minus__, P.b__tilde__, P.t ],
               color = [ 'Identity(3,4)' ],
               lorentz = [ L.FFFF18, L.FFFF19, L.FFFF20, L.FFFF21, L.FFFF23 ],
               couplings = {(0,0):C.GC_78,(0,3):C.GC_99,(0,1):C.GC_98,(0,2):C.GC_98,(0,4):C.GC_53})

V_194 = Vertex(name = 'V_194',
               particles = [ P.ta__plus__, P.ta__minus__, P.b__tilde__, P.b ],
               color = [ 'Identity(3,4)' ],
               lorentz = [ L.FFFF18, L.FFFF22, L.FFFF23, L.FFFF24 ],
               couplings = {(0,0):C.GC_29,(0,3):C.GC_75,(0,1):C.GC_53,(0,2):C.GC_53})

V_195 = Vertex(name = 'V_195',
               particles = [ P.t__tilde__, P.b, P.ta__plus__, P.vt ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFFF18, L.FFFF23, L.FFFF30, L.FFFF31, L.FFFF32 ],
               couplings = {(0,0):C.GC_78,(0,4):C.GC_99,(0,2):C.GC_98,(0,3):C.GC_98,(0,1):C.GC_53})

V_196 = Vertex(name = 'V_196',
               particles = [ P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSS5, L.VSS6 ],
               couplings = {(0,0):C.GC_194,(0,1):C.GC_416})

V_197 = Vertex(name = 'V_197',
               particles = [ P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSS5 ],
               couplings = {(0,0):C.GC_477})

V_198 = Vertex(name = 'V_198',
               particles = [ P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSS5 ],
               couplings = {(0,0):C.GC_192})

V_199 = Vertex(name = 'V_199',
               particles = [ P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSS5 ],
               couplings = {(0,0):C.GC_481})

V_200 = Vertex(name = 'V_200',
               particles = [ P.a, P.a, P.W__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVV18, L.VVVV24 ],
               couplings = {(0,1):C.GC_271,(0,0):C.GC_5})

V_201 = Vertex(name = 'V_201',
               particles = [ P.a, P.a, P.W__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVV18 ],
               couplings = {(0,0):C.GC_502})

V_202 = Vertex(name = 'V_202',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVV21, L.VVVV23 ],
               couplings = {(0,1):C.GC_149,(0,0):C.GC_203})

V_203 = Vertex(name = 'V_203',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVV21 ],
               couplings = {(0,0):C.GC_501})

V_204 = Vertex(name = 'V_204',
               particles = [ P.a, P.W__minus__, P.G0, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_247})

V_205 = Vertex(name = 'V_205',
               particles = [ P.a, P.W__minus__, P.G__minus__, P.G__plus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_246})

V_206 = Vertex(name = 'V_206',
               particles = [ P.a, P.W__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_398})

V_207 = Vertex(name = 'V_207',
               particles = [ P.W__minus__, P.G0, P.G0, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS17 ],
               couplings = {(0,0):C.GC_207})

V_208 = Vertex(name = 'V_208',
               particles = [ P.W__minus__, P.G0, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS22 ],
               couplings = {(0,0):C.GC_210})

V_209 = Vertex(name = 'V_209',
               particles = [ P.W__minus__, P.G0, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSS11 ],
               couplings = {(0,0):C.GC_376})

V_210 = Vertex(name = 'V_210',
               particles = [ P.W__minus__, P.G0, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSS13 ],
               couplings = {(0,0):C.GC_378})

V_211 = Vertex(name = 'V_211',
               particles = [ P.W__minus__, P.G0, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSSS23 ],
               couplings = {(0,0):C.GC_210})

V_212 = Vertex(name = 'V_212',
               particles = [ P.W__minus__, P.G__minus__, P.G__plus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS15 ],
               couplings = {(0,0):C.GC_211})

V_213 = Vertex(name = 'V_213',
               particles = [ P.W__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSS14 ],
               couplings = {(0,0):C.GC_379})

V_214 = Vertex(name = 'V_214',
               particles = [ P.W__minus__, P.W__minus__, P.G0, P.G0, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_185})

V_215 = Vertex(name = 'V_215',
               particles = [ P.W__minus__, P.W__minus__, P.G0, P.G__plus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_186})

V_216 = Vertex(name = 'V_216',
               particles = [ P.W__minus__, P.W__minus__, P.G__plus__, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_184})

V_217 = Vertex(name = 'V_217',
               particles = [ P.W__minus__, P.W__minus__, P.G0, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_368})

V_218 = Vertex(name = 'V_218',
               particles = [ P.W__minus__, P.W__minus__, P.G__plus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_367})

V_219 = Vertex(name = 'V_219',
               particles = [ P.W__minus__, P.W__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSS14 ],
               couplings = {(0,0):C.GC_414})

V_220 = Vertex(name = 'V_220',
               particles = [ P.a, P.a, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS22 ],
               couplings = {(0,0):C.GC_147})

V_221 = Vertex(name = 'V_221',
               particles = [ P.a, P.a, P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS22 ],
               couplings = {(0,0):C.GC_148})

V_222 = Vertex(name = 'V_222',
               particles = [ P.a, P.a, P.W__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVS14 ],
               couplings = {(0,0):C.GC_355})

V_223 = Vertex(name = 'V_223',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVVSS21, L.VVVSS25 ],
               couplings = {(0,1):C.GC_139,(0,0):C.GC_236})

V_224 = Vertex(name = 'V_224',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS21, L.VVVSS25 ],
               couplings = {(0,1):C.GC_139,(0,0):C.GC_235})

V_225 = Vertex(name = 'V_225',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS21, L.VVVSS25 ],
               couplings = {(0,1):C.GC_139,(0,0):C.GC_236})

V_226 = Vertex(name = 'V_226',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVS13, L.VVVS17 ],
               couplings = {(0,1):C.GC_351,(0,0):C.GC_394})

V_227 = Vertex(name = 'V_227',
               particles = [ P.a, P.W__minus__, P.Z, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS21, L.VVVSS23 ],
               couplings = {(0,0):C.GC_242,(0,1):C.GC_260})

V_228 = Vertex(name = 'V_228',
               particles = [ P.a, P.W__minus__, P.Z, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS21, L.VVVSS23 ],
               couplings = {(0,0):C.GC_243,(0,1):C.GC_261})

V_229 = Vertex(name = 'V_229',
               particles = [ P.a, P.W__minus__, P.Z, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVS13, L.VVVS15 ],
               couplings = {(0,0):C.GC_397,(0,1):C.GC_403})

V_230 = Vertex(name = 'V_230',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVV18, L.VVVV24 ],
               couplings = {(0,1):C.GC_244,(0,0):C.GC_180})

V_231 = Vertex(name = 'V_231',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVV18 ],
               couplings = {(0,0):C.GC_472})

V_232 = Vertex(name = 'V_232',
               particles = [ P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VSS5, L.VSS6 ],
               couplings = {(0,0):C.GC_193,(0,1):C.GC_415})

V_233 = Vertex(name = 'V_233',
               particles = [ P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VSS5 ],
               couplings = {(0,0):C.GC_476})

V_234 = Vertex(name = 'V_234',
               particles = [ P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSS5 ],
               couplings = {(0,0):C.GC_192})

V_235 = Vertex(name = 'V_235',
               particles = [ P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSS5 ],
               couplings = {(0,0):C.GC_481})

V_236 = Vertex(name = 'V_236',
               particles = [ P.a, P.a, P.a, P.W__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVV7 ],
               couplings = {(0,0):C.GC_272})

V_237 = Vertex(name = 'V_237',
               particles = [ P.a, P.a, P.W__minus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVV10 ],
               couplings = {(0,0):C.GC_152})

V_238 = Vertex(name = 'V_238',
               particles = [ P.a, P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVV12 ],
               couplings = {(0,0):C.GC_250})

V_239 = Vertex(name = 'V_239',
               particles = [ P.a, P.a, P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV3 ],
               couplings = {(0,0):C.GC_252})

V_240 = Vertex(name = 'V_240',
               particles = [ P.a, P.W__plus__, P.G0, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_247})

V_241 = Vertex(name = 'V_241',
               particles = [ P.a, P.W__plus__, P.G__minus__, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_248})

V_242 = Vertex(name = 'V_242',
               particles = [ P.a, P.W__plus__, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_399})

V_243 = Vertex(name = 'V_243',
               particles = [ P.W__plus__, P.G0, P.G0, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS17 ],
               couplings = {(0,0):C.GC_207})

V_244 = Vertex(name = 'V_244',
               particles = [ P.W__plus__, P.G0, P.G__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS22 ],
               couplings = {(0,0):C.GC_209})

V_245 = Vertex(name = 'V_245',
               particles = [ P.W__plus__, P.G0, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSS11 ],
               couplings = {(0,0):C.GC_376})

V_246 = Vertex(name = 'V_246',
               particles = [ P.W__plus__, P.G0, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSS13 ],
               couplings = {(0,0):C.GC_377})

V_247 = Vertex(name = 'V_247',
               particles = [ P.W__plus__, P.G0, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSSS18 ],
               couplings = {(0,0):C.GC_208})

V_248 = Vertex(name = 'V_248',
               particles = [ P.W__plus__, P.G__minus__, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS14 ],
               couplings = {(0,0):C.GC_207})

V_249 = Vertex(name = 'V_249',
               particles = [ P.W__plus__, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSS12 ],
               couplings = {(0,0):C.GC_376})

V_250 = Vertex(name = 'V_250',
               particles = [ P.W__minus__, P.W__plus__, P.G0, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_183})

V_251 = Vertex(name = 'V_251',
               particles = [ P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_183})

V_252 = Vertex(name = 'V_252',
               particles = [ P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_366})

V_253 = Vertex(name = 'V_253',
               particles = [ P.W__plus__, P.W__plus__, P.G0, P.G0, P.G__minus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_185})

V_254 = Vertex(name = 'V_254',
               particles = [ P.W__plus__, P.W__plus__, P.G0, P.G__minus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_182})

V_255 = Vertex(name = 'V_255',
               particles = [ P.W__plus__, P.W__plus__, P.G__minus__, P.G__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_184})

V_256 = Vertex(name = 'V_256',
               particles = [ P.W__plus__, P.W__plus__, P.G0, P.G__minus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_365})

V_257 = Vertex(name = 'V_257',
               particles = [ P.W__plus__, P.W__plus__, P.G__minus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_367})

V_258 = Vertex(name = 'V_258',
               particles = [ P.W__plus__, P.W__plus__, P.G__minus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSS14 ],
               couplings = {(0,0):C.GC_414})

V_259 = Vertex(name = 'V_259',
               particles = [ P.a, P.a, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS22 ],
               couplings = {(0,0):C.GC_146})

V_260 = Vertex(name = 'V_260',
               particles = [ P.a, P.a, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS22 ],
               couplings = {(0,0):C.GC_148})

V_261 = Vertex(name = 'V_261',
               particles = [ P.a, P.a, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVS14 ],
               couplings = {(0,0):C.GC_355})

V_262 = Vertex(name = 'V_262',
               particles = [ P.a, P.W__plus__, P.Z, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS21, L.VVVSS23 ],
               couplings = {(0,0):C.GC_241,(0,1):C.GC_259})

V_263 = Vertex(name = 'V_263',
               particles = [ P.a, P.W__plus__, P.Z, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS21, L.VVVSS23 ],
               couplings = {(0,0):C.GC_243,(0,1):C.GC_261})

V_264 = Vertex(name = 'V_264',
               particles = [ P.a, P.W__plus__, P.Z, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVS13, L.VVVS15 ],
               couplings = {(0,0):C.GC_397,(0,1):C.GC_403})

V_265 = Vertex(name = 'V_265',
               particles = [ P.a, P.a, P.W__minus__, P.W__plus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS7 ],
               couplings = {(0,0):C.GC_151})

V_266 = Vertex(name = 'V_266',
               particles = [ P.a, P.a, P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS7 ],
               couplings = {(0,0):C.GC_151})

V_267 = Vertex(name = 'V_267',
               particles = [ P.a, P.a, P.W__minus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS7 ],
               couplings = {(0,0):C.GC_151})

V_268 = Vertex(name = 'V_268',
               particles = [ P.a, P.a, P.W__minus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVS16 ],
               couplings = {(0,0):C.GC_356})

V_269 = Vertex(name = 'V_269',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVVSS23, L.VVVSS25 ],
               couplings = {(0,1):C.GC_234,(0,0):C.GC_141})

V_270 = Vertex(name = 'V_270',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS23, L.VVVSS25 ],
               couplings = {(0,1):C.GC_234,(0,0):C.GC_140})

V_271 = Vertex(name = 'V_271',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS23, L.VVVSS25 ],
               couplings = {(0,1):C.GC_234,(0,0):C.GC_141})

V_272 = Vertex(name = 'V_272',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVS15, L.VVVS17 ],
               couplings = {(0,1):C.GC_393,(0,0):C.GC_352})

V_273 = Vertex(name = 'V_273',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS7 ],
               couplings = {(0,0):C.GC_187})

V_274 = Vertex(name = 'V_274',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS7 ],
               couplings = {(0,0):C.GC_187})

V_275 = Vertex(name = 'V_275',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS7 ],
               couplings = {(0,0):C.GC_187})

V_276 = Vertex(name = 'V_276',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVS16 ],
               couplings = {(0,0):C.GC_369})

V_277 = Vertex(name = 'V_277',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVV18, L.VVVV24 ],
               couplings = {(0,1):C.GC_245,(0,0):C.GC_181})

V_278 = Vertex(name = 'V_278',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVV18 ],
               couplings = {(0,0):C.GC_503})

V_279 = Vertex(name = 'V_279',
               particles = [ P.Z, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSS4, L.VSS5 ],
               couplings = {(0,1):C.GC_275,(0,0):C.GC_511})

V_280 = Vertex(name = 'V_280',
               particles = [ P.Z, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSS5 ],
               couplings = {(0,0):C.GC_509})

V_281 = Vertex(name = 'V_281',
               particles = [ P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSS5 ],
               couplings = {(0,0):C.GC_274})

V_282 = Vertex(name = 'V_282',
               particles = [ P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSS5 ],
               couplings = {(0,0):C.GC_510})

V_283 = Vertex(name = 'V_283',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVV8 ],
               couplings = {(0,0):C.GC_251})

V_284 = Vertex(name = 'V_284',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVV9 ],
               couplings = {(0,0):C.GC_189})

V_285 = Vertex(name = 'V_285',
               particles = [ P.a, P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV4 ],
               couplings = {(0,0):C.GC_191})

V_286 = Vertex(name = 'V_286',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVV11 ],
               couplings = {(0,0):C.GC_190})

V_287 = Vertex(name = 'V_287',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV3 ],
               couplings = {(0,0):C.GC_178})

V_288 = Vertex(name = 'V_288',
               particles = [ P.a, P.Z, P.G0, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_304})

V_289 = Vertex(name = 'V_289',
               particles = [ P.a, P.Z, P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_305})

V_290 = Vertex(name = 'V_290',
               particles = [ P.a, P.Z, P.G__minus__, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_304})

V_291 = Vertex(name = 'V_291',
               particles = [ P.a, P.Z, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_457})

V_292 = Vertex(name = 'V_292',
               particles = [ P.Z, P.G0, P.G0, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS16 ],
               couplings = {(0,0):C.GC_284})

V_293 = Vertex(name = 'V_293',
               particles = [ P.Z, P.G0, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS19 ],
               couplings = {(0,0):C.GC_283})

V_294 = Vertex(name = 'V_294',
               particles = [ P.Z, P.G0, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS24 ],
               couplings = {(0,0):C.GC_285})

V_295 = Vertex(name = 'V_295',
               particles = [ P.Z, P.G0, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VSSS16 ],
               couplings = {(0,0):C.GC_446})

V_296 = Vertex(name = 'V_296',
               particles = [ P.Z, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSS9 ],
               couplings = {(0,0):C.GC_445})

V_297 = Vertex(name = 'V_297',
               particles = [ P.Z, P.G0, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSS15 ],
               couplings = {(0,0):C.GC_447})

V_298 = Vertex(name = 'V_298',
               particles = [ P.Z, P.G0, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSSS21 ],
               couplings = {(0,0):C.GC_281})

V_299 = Vertex(name = 'V_299',
               particles = [ P.Z, P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSSS20 ],
               couplings = {(0,0):C.GC_282})

V_300 = Vertex(name = 'V_300',
               particles = [ P.Z, P.G__minus__, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS13 ],
               couplings = {(0,0):C.GC_281})

V_301 = Vertex(name = 'V_301',
               particles = [ P.Z, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSS10 ],
               couplings = {(0,0):C.GC_444})

V_302 = Vertex(name = 'V_302',
               particles = [ P.W__minus__, P.Z, P.G0, P.G0, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_174})

V_303 = Vertex(name = 'V_303',
               particles = [ P.W__minus__, P.Z, P.G0, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_173})

V_304 = Vertex(name = 'V_304',
               particles = [ P.W__minus__, P.Z, P.G0, P.G0, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_171})

V_305 = Vertex(name = 'V_305',
               particles = [ P.W__minus__, P.Z, P.G__minus__, P.G__plus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_176})

V_306 = Vertex(name = 'V_306',
               particles = [ P.W__minus__, P.Z, P.G0, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_172})

V_307 = Vertex(name = 'V_307',
               particles = [ P.W__minus__, P.Z, P.G__plus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_169})

V_308 = Vertex(name = 'V_308',
               particles = [ P.W__minus__, P.Z, P.G0, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_438})

V_309 = Vertex(name = 'V_309',
               particles = [ P.W__minus__, P.Z, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_441})

V_310 = Vertex(name = 'V_310',
               particles = [ P.W__minus__, P.Z, P.G0, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_439})

V_311 = Vertex(name = 'V_311',
               particles = [ P.W__minus__, P.Z, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_436})

V_312 = Vertex(name = 'V_312',
               particles = [ P.W__plus__, P.Z, P.G0, P.G0, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_174})

V_313 = Vertex(name = 'V_313',
               particles = [ P.W__plus__, P.Z, P.G0, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_173})

V_314 = Vertex(name = 'V_314',
               particles = [ P.W__plus__, P.Z, P.G0, P.G0, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_175})

V_315 = Vertex(name = 'V_315',
               particles = [ P.W__plus__, P.Z, P.G__minus__, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_170})

V_316 = Vertex(name = 'V_316',
               particles = [ P.W__plus__, P.Z, P.G0, P.G__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_172})

V_317 = Vertex(name = 'V_317',
               particles = [ P.W__plus__, P.Z, P.G__minus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_177})

V_318 = Vertex(name = 'V_318',
               particles = [ P.W__plus__, P.Z, P.G0, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_440})

V_319 = Vertex(name = 'V_319',
               particles = [ P.W__plus__, P.Z, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_437})

V_320 = Vertex(name = 'V_320',
               particles = [ P.W__plus__, P.Z, P.G0, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_439})

V_321 = Vertex(name = 'V_321',
               particles = [ P.W__plus__, P.Z, P.G__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_442})

V_322 = Vertex(name = 'V_322',
               particles = [ P.Z, P.Z, P.G0, P.G0, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_317})

V_323 = Vertex(name = 'V_323',
               particles = [ P.Z, P.Z, P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_316})

V_324 = Vertex(name = 'V_324',
               particles = [ P.Z, P.Z, P.G0, P.G0, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_315})

V_325 = Vertex(name = 'V_325',
               particles = [ P.Z, P.Z, P.H, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_317})

V_326 = Vertex(name = 'V_326',
               particles = [ P.Z, P.Z, P.G0, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_314})

V_327 = Vertex(name = 'V_327',
               particles = [ P.Z, P.Z, P.G__minus__, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS2 ],
               couplings = {(0,0):C.GC_314})

V_328 = Vertex(name = 'V_328',
               particles = [ P.Z, P.Z, P.G0, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_464})

V_329 = Vertex(name = 'V_329',
               particles = [ P.Z, P.Z, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_465})

V_330 = Vertex(name = 'V_330',
               particles = [ P.Z, P.Z, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS2 ],
               couplings = {(0,0):C.GC_463})

V_331 = Vertex(name = 'V_331',
               particles = [ P.W__minus__, P.Z, P.Z, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS24 ],
               couplings = {(0,0):C.GC_147})

V_332 = Vertex(name = 'V_332',
               particles = [ P.W__minus__, P.Z, P.Z, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS24 ],
               couplings = {(0,0):C.GC_148})

V_333 = Vertex(name = 'V_333',
               particles = [ P.W__minus__, P.Z, P.Z, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVS16 ],
               couplings = {(0,0):C.GC_355})

V_334 = Vertex(name = 'V_334',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS10 ],
               couplings = {(0,0):C.GC_249})

V_335 = Vertex(name = 'V_335',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS10 ],
               couplings = {(0,0):C.GC_249})

V_336 = Vertex(name = 'V_336',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS10 ],
               couplings = {(0,0):C.GC_249})

V_337 = Vertex(name = 'V_337',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVS19 ],
               couplings = {(0,0):C.GC_400})

V_338 = Vertex(name = 'V_338',
               particles = [ P.W__plus__, P.Z, P.Z, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS24 ],
               couplings = {(0,0):C.GC_146})

V_339 = Vertex(name = 'V_339',
               particles = [ P.W__plus__, P.Z, P.Z, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS24 ],
               couplings = {(0,0):C.GC_148})

V_340 = Vertex(name = 'V_340',
               particles = [ P.W__plus__, P.Z, P.Z, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVS16 ],
               couplings = {(0,0):C.GC_355})

V_341 = Vertex(name = 'V_341',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS7 ],
               couplings = {(0,0):C.GC_188})

V_342 = Vertex(name = 'V_342',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS7 ],
               couplings = {(0,0):C.GC_188})

V_343 = Vertex(name = 'V_343',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS7 ],
               couplings = {(0,0):C.GC_188})

V_344 = Vertex(name = 'V_344',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVS16 ],
               couplings = {(0,0):C.GC_370})

V_345 = Vertex(name = 'V_345',
               particles = [ P.t__tilde__, P.t, P.u__tilde__, P.u ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
               lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
               couplings = {(0,0):C.GC_31,(1,0):C.GC_35,(0,2):C.GC_87,(1,2):C.GC_88,(0,3):C.GC_112,(1,3):C.GC_113,(0,1):C.GC_109,(1,1):C.GC_110})

V_346 = Vertex(name = 'V_346',
               particles = [ P.b__tilde__, P.b, P.u__tilde__, P.u ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
               lorentz = [ L.FFFF18, L.FFFF28 ],
               couplings = {(0,0):C.GC_30,(1,0):C.GC_34,(0,1):C.GC_87,(1,1):C.GC_88})

V_347 = Vertex(name = 'V_347',
               particles = [ P.t__tilde__, P.b, P.d__tilde__, P.u ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
               lorentz = [ L.FFFF18 ],
               couplings = {(0,0):C.GC_82,(1,0):C.GC_84})

V_348 = Vertex(name = 'V_348',
               particles = [ P.c__tilde__, P.c, P.t__tilde__, P.t ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
               lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
               couplings = {(0,0):C.GC_31,(1,0):C.GC_35,(0,1):C.GC_87,(1,1):C.GC_88,(0,2):C.GC_109,(1,2):C.GC_110,(0,3):C.GC_112,(1,3):C.GC_113})

V_349 = Vertex(name = 'V_349',
               particles = [ P.b__tilde__, P.b, P.c__tilde__, P.c ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
               lorentz = [ L.FFFF18, L.FFFF28 ],
               couplings = {(0,0):C.GC_30,(1,0):C.GC_34,(0,1):C.GC_87,(1,1):C.GC_88})

V_350 = Vertex(name = 'V_350',
               particles = [ P.t__tilde__, P.b, P.s__tilde__, P.c ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
               lorentz = [ L.FFFF18 ],
               couplings = {(0,0):C.GC_82,(1,0):C.GC_84})

V_351 = Vertex(name = 'V_351',
               particles = [ P.u__tilde__, P.d, P.b__tilde__, P.t ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
               lorentz = [ L.FFFF18 ],
               couplings = {(0,0):C.GC_82,(1,0):C.GC_84})

V_352 = Vertex(name = 'V_352',
               particles = [ P.d__tilde__, P.d, P.t__tilde__, P.t ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
               lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
               couplings = {(0,0):C.GC_30,(1,0):C.GC_34,(0,1):C.GC_71,(1,1):C.GC_72,(0,2):C.GC_109,(1,2):C.GC_110,(0,3):C.GC_89,(1,3):C.GC_90})

V_353 = Vertex(name = 'V_353',
               particles = [ P.b__tilde__, P.b, P.d__tilde__, P.d ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
               lorentz = [ L.FFFF18, L.FFFF28 ],
               couplings = {(0,0):C.GC_31,(1,0):C.GC_35,(0,1):C.GC_71,(1,1):C.GC_72})

V_354 = Vertex(name = 'V_354',
               particles = [ P.c__tilde__, P.s, P.b__tilde__, P.t ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
               lorentz = [ L.FFFF18 ],
               couplings = {(0,0):C.GC_82,(1,0):C.GC_84})

V_355 = Vertex(name = 'V_355',
               particles = [ P.s__tilde__, P.s, P.t__tilde__, P.t ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
               lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
               couplings = {(0,0):C.GC_30,(1,0):C.GC_34,(0,1):C.GC_71,(1,1):C.GC_72,(0,2):C.GC_109,(1,2):C.GC_110,(0,3):C.GC_89,(1,3):C.GC_90})

V_356 = Vertex(name = 'V_356',
               particles = [ P.b__tilde__, P.b, P.s__tilde__, P.s ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'T(-1,2,1)*T(-1,4,3)' ],
               lorentz = [ L.FFFF18, L.FFFF28 ],
               couplings = {(0,0):C.GC_31,(1,0):C.GC_35,(0,1):C.GC_71,(1,1):C.GC_72})

V_357 = Vertex(name = 'V_357',
               particles = [ P.t__tilde__, P.t, P.t__tilde__, P.t ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)', 'T(-1,2,1)*T(-1,4,3)', 'T(-1,2,3)*T(-1,4,1)' ],
               lorentz = [ L.FFFF17, L.FFFF18, L.FFFF24, L.FFFF25, L.FFFF26, L.FFFF27, L.FFFF28, L.FFFF29 ],
               couplings = {(1,0):C.GC_33,(0,1):C.GC_33,(0,2):C.GC_85,(2,2):C.GC_86,(1,5):C.GC_85,(3,5):C.GC_86,(1,3):C.GC_85,(3,3):C.GC_86,(1,4):C.GC_111,(0,6):C.GC_85,(2,6):C.GC_86,(0,7):C.GC_111})

V_358 = Vertex(name = 'V_358',
               particles = [ P.t__tilde__, P.b, P.b__tilde__, P.t ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)', 'T(-1,2,3)*T(-1,4,1)' ],
               lorentz = [ L.FFFF17, L.FFFF18, L.FFFF25, L.FFFF26, L.FFFF27 ],
               couplings = {(1,0):C.GC_32,(0,1):C.GC_83,(1,4):C.GC_71,(2,4):C.GC_72,(1,2):C.GC_85,(2,2):C.GC_86,(1,3):C.GC_89,(2,3):C.GC_90})

V_359 = Vertex(name = 'V_359',
               particles = [ P.b__tilde__, P.b, P.b__tilde__, P.b ],
               color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)', 'T(-1,2,1)*T(-1,4,3)', 'T(-1,2,3)*T(-1,4,1)' ],
               lorentz = [ L.FFFF17, L.FFFF18, L.FFFF24, L.FFFF25, L.FFFF27, L.FFFF28 ],
               couplings = {(1,0):C.GC_33,(0,1):C.GC_33,(0,2):C.GC_71,(2,2):C.GC_72,(1,4):C.GC_71,(3,4):C.GC_72,(1,3):C.GC_71,(3,3):C.GC_72,(0,5):C.GC_71,(2,5):C.GC_72})

V_360 = Vertex(name = 'V_360',
               particles = [ P.t__tilde__, P.t, P.ve__tilde__, P.ve ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFFF18, L.FFFF24 ],
               couplings = {(0,0):C.GC_27,(0,1):C.GC_94})

V_361 = Vertex(name = 'V_361',
               particles = [ P.ve__tilde__, P.e__minus__, P.b__tilde__, P.t ],
               color = [ 'Identity(3,4)' ],
               lorentz = [ L.FFFF18 ],
               couplings = {(0,0):C.GC_76})

V_362 = Vertex(name = 'V_362',
               particles = [ P.t__tilde__, P.b, P.e__plus__, P.ve ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFFF18 ],
               couplings = {(0,0):C.GC_76})

V_363 = Vertex(name = 'V_363',
               particles = [ P.b__tilde__, P.b, P.e__plus__, P.e__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFFF18, L.FFFF28 ],
               couplings = {(0,0):C.GC_27,(0,1):C.GC_73})

V_364 = Vertex(name = 'V_364',
               particles = [ P.t__tilde__, P.t, P.vm__tilde__, P.vm ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFFF18, L.FFFF24 ],
               couplings = {(0,0):C.GC_28,(0,1):C.GC_95})

V_365 = Vertex(name = 'V_365',
               particles = [ P.vm__tilde__, P.mu__minus__, P.b__tilde__, P.t ],
               color = [ 'Identity(3,4)' ],
               lorentz = [ L.FFFF18 ],
               couplings = {(0,0):C.GC_77})

V_366 = Vertex(name = 'V_366',
               particles = [ P.t__tilde__, P.b, P.mu__plus__, P.vm ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFFF18 ],
               couplings = {(0,0):C.GC_77})

V_367 = Vertex(name = 'V_367',
               particles = [ P.b__tilde__, P.b, P.mu__plus__, P.mu__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFFF18, L.FFFF28 ],
               couplings = {(0,0):C.GC_28,(0,1):C.GC_74})

V_368 = Vertex(name = 'V_368',
               particles = [ P.t__tilde__, P.t, P.vt__tilde__, P.vt ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFFF18, L.FFFF24 ],
               couplings = {(0,0):C.GC_29,(0,1):C.GC_96})

V_369 = Vertex(name = 'V_369',
               particles = [ P.e__plus__, P.e__minus__, P.t__tilde__, P.t ],
               color = [ 'Identity(3,4)' ],
               lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
               couplings = {(0,0):C.GC_79,(0,1):C.GC_73,(0,2):C.GC_94,(0,3):C.GC_91})

V_370 = Vertex(name = 'V_370',
               particles = [ P.b__tilde__, P.b, P.ve__tilde__, P.ve ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFFF18 ],
               couplings = {(0,0):C.GC_79})

V_371 = Vertex(name = 'V_371',
               particles = [ P.mu__plus__, P.mu__minus__, P.t__tilde__, P.t ],
               color = [ 'Identity(3,4)' ],
               lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
               couplings = {(0,0):C.GC_80,(0,1):C.GC_74,(0,2):C.GC_95,(0,3):C.GC_92})

V_372 = Vertex(name = 'V_372',
               particles = [ P.b__tilde__, P.b, P.vm__tilde__, P.vm ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFFF18 ],
               couplings = {(0,0):C.GC_80})

V_373 = Vertex(name = 'V_373',
               particles = [ P.ta__plus__, P.ta__minus__, P.t__tilde__, P.t ],
               color = [ 'Identity(3,4)' ],
               lorentz = [ L.FFFF18, L.FFFF19, L.FFFF20, L.FFFF21, L.FFFF24, L.FFFF28, L.FFFF29, L.FFFF30, L.FFFF31, L.FFFF32 ],
               couplings = {(0,0):C.GC_81,(0,3):C.GC_100,(0,1):C.GC_97,(0,2):C.GC_97,(0,4):C.GC_75,(0,5):C.GC_96,(0,6):C.GC_93,(0,9):C.GC_100,(0,7):C.GC_97,(0,8):C.GC_97})

V_374 = Vertex(name = 'V_374',
               particles = [ P.b__tilde__, P.b, P.vt__tilde__, P.vt ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFFF18 ],
               couplings = {(0,0):C.GC_81})

V_375 = Vertex(name = 'V_375',
               particles = [ P.u__tilde__, P.u, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV87, L.FFV91 ],
               couplings = {(0,1):C.GC_2,(0,2):C.GC_466,(0,0):C.GC_1154})

V_376 = Vertex(name = 'V_376',
               particles = [ P.u__tilde__, P.u, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_490})

V_377 = Vertex(name = 'V_377',
               particles = [ P.c__tilde__, P.c, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV87, L.FFV91 ],
               couplings = {(0,1):C.GC_2,(0,2):C.GC_466,(0,0):C.GC_1154})

V_378 = Vertex(name = 'V_378',
               particles = [ P.c__tilde__, P.c, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_490})

V_379 = Vertex(name = 'V_379',
               particles = [ P.t__tilde__, P.t, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV110, L.FFV70, L.FFV87, L.FFV91 ],
               couplings = {(0,2):C.GC_2,(0,3):C.GC_466,(0,1):C.GC_1150,(0,0):C.GC_443})

V_380 = Vertex(name = 'V_380',
               particles = [ P.t__tilde__, P.t, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_1151})

V_381 = Vertex(name = 'V_381',
               particles = [ P.d__tilde__, P.d, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV83, L.FFV87 ],
               couplings = {(0,2):C.GC_1,(0,1):C.GC_466,(0,0):C.GC_1155})

V_382 = Vertex(name = 'V_382',
               particles = [ P.d__tilde__, P.d, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_489})

V_383 = Vertex(name = 'V_383',
               particles = [ P.s__tilde__, P.s, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV83, L.FFV87 ],
               couplings = {(0,2):C.GC_1,(0,1):C.GC_466,(0,0):C.GC_1155})

V_384 = Vertex(name = 'V_384',
               particles = [ P.s__tilde__, P.s, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_489})

V_385 = Vertex(name = 'V_385',
               particles = [ P.b__tilde__, P.b, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV83, L.FFV87 ],
               couplings = {(0,2):C.GC_1,(0,1):C.GC_466,(0,0):C.GC_1152})

V_386 = Vertex(name = 'V_386',
               particles = [ P.b__tilde__, P.b, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_489})

V_387 = Vertex(name = 'V_387',
               particles = [ P.u__tilde__, P.u, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV57 ],
               couplings = {(0,0):C.GC_11})

V_388 = Vertex(name = 'V_388',
               particles = [ P.c__tilde__, P.c, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV57 ],
               couplings = {(0,0):C.GC_11})

V_389 = Vertex(name = 'V_389',
               particles = [ P.t__tilde__, P.t, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV110, L.FFV87 ],
               couplings = {(0,1):C.GC_11,(0,0):C.GC_358})

V_390 = Vertex(name = 'V_390',
               particles = [ P.d__tilde__, P.d, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV57 ],
               couplings = {(0,0):C.GC_11})

V_391 = Vertex(name = 'V_391',
               particles = [ P.s__tilde__, P.s, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV57 ],
               couplings = {(0,0):C.GC_11})

V_392 = Vertex(name = 'V_392',
               particles = [ P.b__tilde__, P.b, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV57 ],
               couplings = {(0,0):C.GC_11})

V_393 = Vertex(name = 'V_393',
               particles = [ P.d__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_195})

V_394 = Vertex(name = 'V_394',
               particles = [ P.d__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_1991})

V_395 = Vertex(name = 'V_395',
               particles = [ P.s__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_195})

V_396 = Vertex(name = 'V_396',
               particles = [ P.s__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_1991})

V_397 = Vertex(name = 'V_397',
               particles = [ P.b__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV80 ],
               couplings = {(0,1):C.GC_340,(0,0):C.GC_195})

V_398 = Vertex(name = 'V_398',
               particles = [ P.b__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_1992})

V_399 = Vertex(name = 'V_399',
               particles = [ P.u__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_195})

V_400 = Vertex(name = 'V_400',
               particles = [ P.u__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_1991})

V_401 = Vertex(name = 'V_401',
               particles = [ P.c__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_195})

V_402 = Vertex(name = 'V_402',
               particles = [ P.c__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_1991})

V_403 = Vertex(name = 'V_403',
               particles = [ P.t__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV109, L.FFV70 ],
               couplings = {(0,0):C.GC_340,(0,1):C.GC_195})

V_404 = Vertex(name = 'V_404',
               particles = [ P.t__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_1992})

V_405 = Vertex(name = 'V_405',
               particles = [ P.u__tilde__, P.u, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV82, L.FFV87, L.FFV91 ],
               couplings = {(0,0):C.GC_196,(0,3):C.GC_253,(0,2):C.GC_483,(0,1):C.GC_1804})

V_406 = Vertex(name = 'V_406',
               particles = [ P.u__tilde__, P.u, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV91 ],
               couplings = {(0,0):C.GC_1801,(0,1):C.GC_505})

V_407 = Vertex(name = 'V_407',
               particles = [ P.c__tilde__, P.c, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV82, L.FFV87, L.FFV91 ],
               couplings = {(0,0):C.GC_196,(0,3):C.GC_253,(0,2):C.GC_483,(0,1):C.GC_1804})

V_408 = Vertex(name = 'V_408',
               particles = [ P.c__tilde__, P.c, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV91 ],
               couplings = {(0,0):C.GC_1801,(0,1):C.GC_505})

V_409 = Vertex(name = 'V_409',
               particles = [ P.t__tilde__, P.t, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV110, L.FFV70, L.FFV82, L.FFV87, L.FFV91 ],
               couplings = {(0,1):C.GC_196,(0,4):C.GC_253,(0,3):C.GC_1227,(0,2):C.GC_1803,(0,0):C.GC_341})

V_410 = Vertex(name = 'V_410',
               particles = [ P.t__tilde__, P.t, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV91 ],
               couplings = {(0,0):C.GC_1797,(0,1):C.GC_505})

V_411 = Vertex(name = 'V_411',
               particles = [ P.d__tilde__, P.d, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV82, L.FFV83, L.FFV87 ],
               couplings = {(0,0):C.GC_197,(0,2):C.GC_253,(0,3):C.GC_482,(0,1):C.GC_1800})

V_412 = Vertex(name = 'V_412',
               particles = [ P.d__tilde__, P.d, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV83 ],
               couplings = {(0,0):C.GC_1802,(0,1):C.GC_505})

V_413 = Vertex(name = 'V_413',
               particles = [ P.s__tilde__, P.s, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV82, L.FFV83, L.FFV87 ],
               couplings = {(0,0):C.GC_197,(0,2):C.GC_253,(0,3):C.GC_482,(0,1):C.GC_1800})

V_414 = Vertex(name = 'V_414',
               particles = [ P.s__tilde__, P.s, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV83 ],
               couplings = {(0,0):C.GC_1802,(0,1):C.GC_505})

V_415 = Vertex(name = 'V_415',
               particles = [ P.b__tilde__, P.b, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV82, L.FFV83, L.FFV87 ],
               couplings = {(0,0):C.GC_197,(0,2):C.GC_253,(0,3):C.GC_482,(0,1):C.GC_1800})

V_416 = Vertex(name = 'V_416',
               particles = [ P.b__tilde__, P.b, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV83 ],
               couplings = {(0,0):C.GC_1799,(0,1):C.GC_505})

V_417 = Vertex(name = 'V_417',
               particles = [ P.ve__tilde__, P.ve, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_468})

V_418 = Vertex(name = 'V_418',
               particles = [ P.vm__tilde__, P.vm, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_468})

V_419 = Vertex(name = 'V_419',
               particles = [ P.vt__tilde__, P.vt, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_468})

V_420 = Vertex(name = 'V_420',
               particles = [ P.e__plus__, P.e__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV70, L.FFV87, L.FFV89 ],
               couplings = {(0,1):C.GC_3,(0,2):C.GC_467,(0,0):C.GC_413})

V_421 = Vertex(name = 'V_421',
               particles = [ P.e__plus__, P.e__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_492})

V_422 = Vertex(name = 'V_422',
               particles = [ P.mu__plus__, P.mu__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV70, L.FFV87, L.FFV89 ],
               couplings = {(0,1):C.GC_3,(0,2):C.GC_467,(0,0):C.GC_413})

V_423 = Vertex(name = 'V_423',
               particles = [ P.mu__plus__, P.mu__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_492})

V_424 = Vertex(name = 'V_424',
               particles = [ P.ta__plus__, P.ta__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV70, L.FFV87, L.FFV89 ],
               couplings = {(0,1):C.GC_3,(0,2):C.GC_467,(0,0):C.GC_413})

V_425 = Vertex(name = 'V_425',
               particles = [ P.ta__plus__, P.ta__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_492})

V_426 = Vertex(name = 'V_426',
               particles = [ P.e__plus__, P.ve, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_195})

V_427 = Vertex(name = 'V_427',
               particles = [ P.e__plus__, P.ve, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_478})

V_428 = Vertex(name = 'V_428',
               particles = [ P.mu__plus__, P.vm, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_195})

V_429 = Vertex(name = 'V_429',
               particles = [ P.mu__plus__, P.vm, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_479})

V_430 = Vertex(name = 'V_430',
               particles = [ P.ta__plus__, P.vt, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_195})

V_431 = Vertex(name = 'V_431',
               particles = [ P.ta__plus__, P.vt, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_480})

V_432 = Vertex(name = 'V_432',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_195})

V_433 = Vertex(name = 'V_433',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_478})

V_434 = Vertex(name = 'V_434',
               particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_195})

V_435 = Vertex(name = 'V_435',
               particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_479})

V_436 = Vertex(name = 'V_436',
               particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_195})

V_437 = Vertex(name = 'V_437',
               particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_480})

V_438 = Vertex(name = 'V_438',
               particles = [ P.ve__tilde__, P.ve, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_273})

V_439 = Vertex(name = 'V_439',
               particles = [ P.ve__tilde__, P.ve, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_514})

V_440 = Vertex(name = 'V_440',
               particles = [ P.vm__tilde__, P.vm, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_273})

V_441 = Vertex(name = 'V_441',
               particles = [ P.vm__tilde__, P.vm, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_516})

V_442 = Vertex(name = 'V_442',
               particles = [ P.vt__tilde__, P.vt, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_273})

V_443 = Vertex(name = 'V_443',
               particles = [ P.vt__tilde__, P.vt, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_518})

V_444 = Vertex(name = 'V_444',
               particles = [ P.e__plus__, P.e__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV70, L.FFV82, L.FFV86, L.FFV87, L.FFV89 ],
               couplings = {(0,0):C.GC_197,(0,4):C.GC_254,(0,3):C.GC_486,(0,2):C.GC_418,(0,1):C.GC_512})

V_445 = Vertex(name = 'V_445',
               particles = [ P.e__plus__, P.e__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV70, L.FFV89 ],
               couplings = {(0,0):C.GC_513,(0,1):C.GC_507})

V_446 = Vertex(name = 'V_446',
               particles = [ P.mu__plus__, P.mu__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV70, L.FFV82, L.FFV86, L.FFV87, L.FFV89 ],
               couplings = {(0,0):C.GC_197,(0,4):C.GC_254,(0,3):C.GC_486,(0,2):C.GC_419,(0,1):C.GC_519})

V_447 = Vertex(name = 'V_447',
               particles = [ P.mu__plus__, P.mu__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV70, L.FFV89 ],
               couplings = {(0,0):C.GC_515,(0,1):C.GC_506})

V_448 = Vertex(name = 'V_448',
               particles = [ P.ta__plus__, P.ta__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV70, L.FFV82, L.FFV87, L.FFV89 ],
               couplings = {(0,0):C.GC_197,(0,3):C.GC_254,(0,2):C.GC_486,(0,1):C.GC_520})

V_449 = Vertex(name = 'V_449',
               particles = [ P.ta__plus__, P.ta__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV70, L.FFV89 ],
               couplings = {(0,0):C.GC_517,(0,1):C.GC_508})

V_450 = Vertex(name = 'V_450',
               particles = [ P.c__tilde__, P.c, P.a, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2018,(0,1):C.GC_2020})

V_451 = Vertex(name = 'V_451',
               particles = [ P.c__tilde__, P.c, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_1815,(0,1):C.GC_1818})

V_452 = Vertex(name = 'V_452',
               particles = [ P.c__tilde__, P.c, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS45, L.FFS55 ],
               couplings = {(0,0):C.GC_1838,(0,1):C.GC_1841})

V_453 = Vertex(name = 'V_453',
               particles = [ P.c__tilde__, P.c, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_2003,(0,1):C.GC_2006})

V_454 = Vertex(name = 'V_454',
               particles = [ P.t__tilde__, P.t, P.a, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2007,(0,1):C.GC_2019})

V_455 = Vertex(name = 'V_455',
               particles = [ P.u__tilde__, P.u, P.a, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2018,(0,1):C.GC_2020})

V_456 = Vertex(name = 'V_456',
               particles = [ P.u__tilde__, P.u, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_1815,(0,1):C.GC_1818})

V_457 = Vertex(name = 'V_457',
               particles = [ P.u__tilde__, P.u, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS45, L.FFS55 ],
               couplings = {(0,0):C.GC_1838,(0,1):C.GC_1841})

V_458 = Vertex(name = 'V_458',
               particles = [ P.u__tilde__, P.u, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_2003,(0,1):C.GC_2006})

V_459 = Vertex(name = 'V_459',
               particles = [ P.c__tilde__, P.c, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2040,(0,1):C.GC_2046})

V_460 = Vertex(name = 'V_460',
               particles = [ P.c__tilde__, P.c, P.W__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2039,(0,1):C.GC_2045})

V_461 = Vertex(name = 'V_461',
               particles = [ P.c__tilde__, P.c, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_2109,(0,0):C.GC_2113})

V_462 = Vertex(name = 'V_462',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2027,(0,1):C.GC_2043})

V_463 = Vertex(name = 'V_463',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2028,(0,1):C.GC_2042})

V_464 = Vertex(name = 'V_464',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS165, L.FFVS94 ],
               couplings = {(0,2):C.GC_2101,(0,0):C.GC_2111,(0,1):C.GC_118})

V_465 = Vertex(name = 'V_465',
               particles = [ P.u__tilde__, P.u, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2040,(0,1):C.GC_2046})

V_466 = Vertex(name = 'V_466',
               particles = [ P.u__tilde__, P.u, P.W__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2039,(0,1):C.GC_2045})

V_467 = Vertex(name = 'V_467',
               particles = [ P.u__tilde__, P.u, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_2109,(0,0):C.GC_2113})

V_468 = Vertex(name = 'V_468',
               particles = [ P.c__tilde__, P.c, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2040,(0,1):C.GC_2046})

V_469 = Vertex(name = 'V_469',
               particles = [ P.c__tilde__, P.c, P.W__plus__, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2041,(0,1):C.GC_2047})

V_470 = Vertex(name = 'V_470',
               particles = [ P.c__tilde__, P.c, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_2110,(0,0):C.GC_2114})

V_471 = Vertex(name = 'V_471',
               particles = [ P.t__tilde__, P.t, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2027,(0,1):C.GC_2043})

V_472 = Vertex(name = 'V_472',
               particles = [ P.t__tilde__, P.t, P.W__plus__, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2026,(0,1):C.GC_2044})

V_473 = Vertex(name = 'V_473',
               particles = [ P.t__tilde__, P.t, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS108, L.FFVS120, L.FFVS94 ],
               couplings = {(0,2):C.GC_2100,(0,1):C.GC_2112,(0,0):C.GC_117})

V_474 = Vertex(name = 'V_474',
               particles = [ P.u__tilde__, P.u, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2040,(0,1):C.GC_2046})

V_475 = Vertex(name = 'V_475',
               particles = [ P.u__tilde__, P.u, P.W__plus__, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2041,(0,1):C.GC_2047})

V_476 = Vertex(name = 'V_476',
               particles = [ P.u__tilde__, P.u, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_2110,(0,0):C.GC_2114})

V_477 = Vertex(name = 'V_477',
               particles = [ P.c__tilde__, P.c, P.Z, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_1827,(0,1):C.GC_1830})

V_478 = Vertex(name = 'V_478',
               particles = [ P.c__tilde__, P.c, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2060,(0,1):C.GC_2062})

V_479 = Vertex(name = 'V_479',
               particles = [ P.c__tilde__, P.c, P.Z, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_1827,(0,1):C.GC_1830})

V_480 = Vertex(name = 'V_480',
               particles = [ P.c__tilde__, P.c, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_1849,(0,0):C.GC_1852})

V_481 = Vertex(name = 'V_481',
               particles = [ P.t__tilde__, P.t, P.Z, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_1823,(0,1):C.GC_1829})

V_482 = Vertex(name = 'V_482',
               particles = [ P.t__tilde__, P.t, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2049,(0,1):C.GC_2061})

V_483 = Vertex(name = 'V_483',
               particles = [ P.t__tilde__, P.t, P.Z, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_1823,(0,1):C.GC_1829})

V_484 = Vertex(name = 'V_484',
               particles = [ P.t__tilde__, P.t, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS166, L.FFVS94 ],
               couplings = {(0,2):C.GC_1845,(0,0):C.GC_1851,(0,1):C.GC_120})

V_485 = Vertex(name = 'V_485',
               particles = [ P.u__tilde__, P.u, P.Z, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_1827,(0,1):C.GC_1830})

V_486 = Vertex(name = 'V_486',
               particles = [ P.u__tilde__, P.u, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2060,(0,1):C.GC_2062})

V_487 = Vertex(name = 'V_487',
               particles = [ P.u__tilde__, P.u, P.Z, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_1827,(0,1):C.GC_1830})

V_488 = Vertex(name = 'V_488',
               particles = [ P.u__tilde__, P.u, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_1849,(0,0):C.GC_1852})

V_489 = Vertex(name = 'V_489',
               particles = [ P.b__tilde__, P.b, P.a, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2015,(0,1):C.GC_2016})

V_490 = Vertex(name = 'V_490',
               particles = [ P.b__tilde__, P.b, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_1813,(0,1):C.GC_1814})

V_491 = Vertex(name = 'V_491',
               particles = [ P.b__tilde__, P.b, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS45, L.FFS55 ],
               couplings = {(0,0):C.GC_1836,(0,1):C.GC_1837})

V_492 = Vertex(name = 'V_492',
               particles = [ P.b__tilde__, P.b, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_2000,(0,1):C.GC_2002})

V_493 = Vertex(name = 'V_493',
               particles = [ P.d__tilde__, P.d, P.a, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2017,(0,1):C.GC_2016})

V_494 = Vertex(name = 'V_494',
               particles = [ P.d__tilde__, P.d, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_1816,(0,1):C.GC_1814})

V_495 = Vertex(name = 'V_495',
               particles = [ P.d__tilde__, P.d, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS45, L.FFS55 ],
               couplings = {(0,0):C.GC_1839,(0,1):C.GC_1837})

V_496 = Vertex(name = 'V_496',
               particles = [ P.d__tilde__, P.d, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_2004,(0,1):C.GC_2002})

V_497 = Vertex(name = 'V_497',
               particles = [ P.s__tilde__, P.s, P.a, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2017,(0,1):C.GC_2016})

V_498 = Vertex(name = 'V_498',
               particles = [ P.s__tilde__, P.s, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_1816,(0,1):C.GC_1814})

V_499 = Vertex(name = 'V_499',
               particles = [ P.s__tilde__, P.s, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS45, L.FFS55 ],
               couplings = {(0,0):C.GC_1839,(0,1):C.GC_1837})

V_500 = Vertex(name = 'V_500',
               particles = [ P.s__tilde__, P.s, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_2004,(0,1):C.GC_2002})

V_501 = Vertex(name = 'V_501',
               particles = [ P.b__tilde__, P.b, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2032,(0,1):C.GC_2037})

V_502 = Vertex(name = 'V_502',
               particles = [ P.b__tilde__, P.b, P.W__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2031,(0,1):C.GC_2036})

V_503 = Vertex(name = 'V_503',
               particles = [ P.b__tilde__, P.b, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_2104,(0,0):C.GC_2107})

V_504 = Vertex(name = 'V_504',
               particles = [ P.d__tilde__, P.d, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2040,(0,1):C.GC_2037})

V_505 = Vertex(name = 'V_505',
               particles = [ P.d__tilde__, P.d, P.W__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2039,(0,1):C.GC_2036})

V_506 = Vertex(name = 'V_506',
               particles = [ P.d__tilde__, P.d, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_2109,(0,0):C.GC_2107})

V_507 = Vertex(name = 'V_507',
               particles = [ P.s__tilde__, P.s, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2040,(0,1):C.GC_2037})

V_508 = Vertex(name = 'V_508',
               particles = [ P.s__tilde__, P.s, P.W__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2039,(0,1):C.GC_2036})

V_509 = Vertex(name = 'V_509',
               particles = [ P.s__tilde__, P.s, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_2109,(0,0):C.GC_2107})

V_510 = Vertex(name = 'V_510',
               particles = [ P.b__tilde__, P.b, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2032,(0,1):C.GC_2037})

V_511 = Vertex(name = 'V_511',
               particles = [ P.b__tilde__, P.b, P.W__plus__, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2035,(0,1):C.GC_2038})

V_512 = Vertex(name = 'V_512',
               particles = [ P.b__tilde__, P.b, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_2106,(0,0):C.GC_2108})

V_513 = Vertex(name = 'V_513',
               particles = [ P.d__tilde__, P.d, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2040,(0,1):C.GC_2037})

V_514 = Vertex(name = 'V_514',
               particles = [ P.d__tilde__, P.d, P.W__plus__, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2041,(0,1):C.GC_2038})

V_515 = Vertex(name = 'V_515',
               particles = [ P.d__tilde__, P.d, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_2110,(0,0):C.GC_2108})

V_516 = Vertex(name = 'V_516',
               particles = [ P.s__tilde__, P.s, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2040,(0,1):C.GC_2037})

V_517 = Vertex(name = 'V_517',
               particles = [ P.s__tilde__, P.s, P.W__plus__, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2041,(0,1):C.GC_2038})

V_518 = Vertex(name = 'V_518',
               particles = [ P.s__tilde__, P.s, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_2110,(0,0):C.GC_2108})

V_519 = Vertex(name = 'V_519',
               particles = [ P.b__tilde__, P.b, P.Z, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_1825,(0,1):C.GC_1826})

V_520 = Vertex(name = 'V_520',
               particles = [ P.b__tilde__, P.b, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2057,(0,1):C.GC_2058})

V_521 = Vertex(name = 'V_521',
               particles = [ P.b__tilde__, P.b, P.Z, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_1825,(0,1):C.GC_1826})

V_522 = Vertex(name = 'V_522',
               particles = [ P.b__tilde__, P.b, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_1847,(0,0):C.GC_1848})

V_523 = Vertex(name = 'V_523',
               particles = [ P.d__tilde__, P.d, P.Z, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_1828,(0,1):C.GC_1826})

V_524 = Vertex(name = 'V_524',
               particles = [ P.d__tilde__, P.d, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2059,(0,1):C.GC_2058})

V_525 = Vertex(name = 'V_525',
               particles = [ P.d__tilde__, P.d, P.Z, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_1828,(0,1):C.GC_1826})

V_526 = Vertex(name = 'V_526',
               particles = [ P.d__tilde__, P.d, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_1850,(0,0):C.GC_1848})

V_527 = Vertex(name = 'V_527',
               particles = [ P.s__tilde__, P.s, P.Z, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_1828,(0,1):C.GC_1826})

V_528 = Vertex(name = 'V_528',
               particles = [ P.s__tilde__, P.s, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_2059,(0,1):C.GC_2058})

V_529 = Vertex(name = 'V_529',
               particles = [ P.s__tilde__, P.s, P.Z, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_1828,(0,1):C.GC_1826})

V_530 = Vertex(name = 'V_530',
               particles = [ P.s__tilde__, P.s, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_1850,(0,0):C.GC_1848})

V_531 = Vertex(name = 'V_531',
               particles = [ P.ve__tilde__, P.ve, P.a, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_39})

V_532 = Vertex(name = 'V_532',
               particles = [ P.ve__tilde__, P.ve, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_17})

V_533 = Vertex(name = 'V_533',
               particles = [ P.ve__tilde__, P.ve, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS45 ],
               couplings = {(0,0):C.GC_423})

V_534 = Vertex(name = 'V_534',
               particles = [ P.ve__tilde__, P.ve, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_15})

V_535 = Vertex(name = 'V_535',
               particles = [ P.vm__tilde__, P.vm, P.a, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_41})

V_536 = Vertex(name = 'V_536',
               particles = [ P.vm__tilde__, P.vm, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_21})

V_537 = Vertex(name = 'V_537',
               particles = [ P.vm__tilde__, P.vm, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS45 ],
               couplings = {(0,0):C.GC_425})

V_538 = Vertex(name = 'V_538',
               particles = [ P.vm__tilde__, P.vm, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_19})

V_539 = Vertex(name = 'V_539',
               particles = [ P.vt__tilde__, P.vt, P.a, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_43})

V_540 = Vertex(name = 'V_540',
               particles = [ P.vt__tilde__, P.vt, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_25})

V_541 = Vertex(name = 'V_541',
               particles = [ P.vt__tilde__, P.vt, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS45 ],
               couplings = {(0,0):C.GC_427})

V_542 = Vertex(name = 'V_542',
               particles = [ P.vt__tilde__, P.vt, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_23})

V_543 = Vertex(name = 'V_543',
               particles = [ P.ve__tilde__, P.ve, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_216})

V_544 = Vertex(name = 'V_544',
               particles = [ P.ve__tilde__, P.ve, P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_215})

V_545 = Vertex(name = 'V_545',
               particles = [ P.ve__tilde__, P.ve, P.W__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_382})

V_546 = Vertex(name = 'V_546',
               particles = [ P.vm__tilde__, P.vm, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_219})

V_547 = Vertex(name = 'V_547',
               particles = [ P.vm__tilde__, P.vm, P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_218})

V_548 = Vertex(name = 'V_548',
               particles = [ P.vm__tilde__, P.vm, P.W__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_384})

V_549 = Vertex(name = 'V_549',
               particles = [ P.vt__tilde__, P.vt, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_222})

V_550 = Vertex(name = 'V_550',
               particles = [ P.vt__tilde__, P.vt, P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_221})

V_551 = Vertex(name = 'V_551',
               particles = [ P.vt__tilde__, P.vt, P.W__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_386})

V_552 = Vertex(name = 'V_552',
               particles = [ P.ve__tilde__, P.ve, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_216})

V_553 = Vertex(name = 'V_553',
               particles = [ P.ve__tilde__, P.ve, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_217})

V_554 = Vertex(name = 'V_554',
               particles = [ P.ve__tilde__, P.ve, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_383})

V_555 = Vertex(name = 'V_555',
               particles = [ P.vm__tilde__, P.vm, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_219})

V_556 = Vertex(name = 'V_556',
               particles = [ P.vm__tilde__, P.vm, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_220})

V_557 = Vertex(name = 'V_557',
               particles = [ P.vm__tilde__, P.vm, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_385})

V_558 = Vertex(name = 'V_558',
               particles = [ P.vt__tilde__, P.vt, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_222})

V_559 = Vertex(name = 'V_559',
               particles = [ P.vt__tilde__, P.vt, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_223})

V_560 = Vertex(name = 'V_560',
               particles = [ P.vt__tilde__, P.vt, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_387})

V_561 = Vertex(name = 'V_561',
               particles = [ P.ve__tilde__, P.ve, P.Z, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_289})

V_562 = Vertex(name = 'V_562',
               particles = [ P.ve__tilde__, P.ve, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_290})

V_563 = Vertex(name = 'V_563',
               particles = [ P.ve__tilde__, P.ve, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_289})

V_564 = Vertex(name = 'V_564',
               particles = [ P.ve__tilde__, P.ve, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_449})

V_565 = Vertex(name = 'V_565',
               particles = [ P.vm__tilde__, P.vm, P.Z, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_293})

V_566 = Vertex(name = 'V_566',
               particles = [ P.vm__tilde__, P.vm, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_294})

V_567 = Vertex(name = 'V_567',
               particles = [ P.vm__tilde__, P.vm, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_293})

V_568 = Vertex(name = 'V_568',
               particles = [ P.vm__tilde__, P.vm, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_451})

V_569 = Vertex(name = 'V_569',
               particles = [ P.vt__tilde__, P.vt, P.Z, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_297})

V_570 = Vertex(name = 'V_570',
               particles = [ P.vt__tilde__, P.vt, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_298})

V_571 = Vertex(name = 'V_571',
               particles = [ P.vt__tilde__, P.vt, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_297})

V_572 = Vertex(name = 'V_572',
               particles = [ P.vt__tilde__, P.vt, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_453})

V_573 = Vertex(name = 'V_573',
               particles = [ P.e__plus__, P.e__minus__, P.a, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_38,(0,1):C.GC_136})

V_574 = Vertex(name = 'V_574',
               particles = [ P.e__plus__, P.e__minus__, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_18,(0,1):C.GC_64})

V_575 = Vertex(name = 'V_575',
               particles = [ P.e__plus__, P.e__minus__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS45, L.FFS55 ],
               couplings = {(0,0):C.GC_424,(0,1):C.GC_335})

V_576 = Vertex(name = 'V_576',
               particles = [ P.e__plus__, P.e__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_16,(0,1):C.GC_63})

V_577 = Vertex(name = 'V_577',
               particles = [ P.mu__plus__, P.mu__minus__, P.a, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_40,(0,1):C.GC_137})

V_578 = Vertex(name = 'V_578',
               particles = [ P.mu__plus__, P.mu__minus__, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_22,(0,1):C.GC_67})

V_579 = Vertex(name = 'V_579',
               particles = [ P.mu__plus__, P.mu__minus__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS45, L.FFS55 ],
               couplings = {(0,0):C.GC_426,(0,1):C.GC_337})

V_580 = Vertex(name = 'V_580',
               particles = [ P.mu__plus__, P.mu__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_20,(0,1):C.GC_66})

V_581 = Vertex(name = 'V_581',
               particles = [ P.ta__plus__, P.ta__minus__, P.a, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_42,(0,1):C.GC_138})

V_582 = Vertex(name = 'V_582',
               particles = [ P.ta__plus__, P.ta__minus__, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_26,(0,1):C.GC_69})

V_583 = Vertex(name = 'V_583',
               particles = [ P.ta__plus__, P.ta__minus__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS45, L.FFS55 ],
               couplings = {(0,0):C.GC_428,(0,1):C.GC_338})

V_584 = Vertex(name = 'V_584',
               particles = [ P.ta__plus__, P.ta__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS30, L.FFSS42 ],
               couplings = {(0,0):C.GC_24,(0,1):C.GC_68})

V_585 = Vertex(name = 'V_585',
               particles = [ P.e__plus__, P.e__minus__, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_216,(0,1):C.GC_213})

V_586 = Vertex(name = 'V_586',
               particles = [ P.e__plus__, P.e__minus__, P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_215,(0,1):C.GC_212})

V_587 = Vertex(name = 'V_587',
               particles = [ P.e__plus__, P.e__minus__, P.W__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_382,(0,0):C.GC_380})

V_588 = Vertex(name = 'V_588',
               particles = [ P.mu__plus__, P.mu__minus__, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_219,(0,1):C.GC_225})

V_589 = Vertex(name = 'V_589',
               particles = [ P.mu__plus__, P.mu__minus__, P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_218,(0,1):C.GC_224})

V_590 = Vertex(name = 'V_590',
               particles = [ P.mu__plus__, P.mu__minus__, P.W__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_384,(0,0):C.GC_388})

V_591 = Vertex(name = 'V_591',
               particles = [ P.ta__plus__, P.ta__minus__, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_222,(0,1):C.GC_228})

V_592 = Vertex(name = 'V_592',
               particles = [ P.ta__plus__, P.ta__minus__, P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_221,(0,1):C.GC_227})

V_593 = Vertex(name = 'V_593',
               particles = [ P.ta__plus__, P.ta__minus__, P.W__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_386,(0,0):C.GC_390})

V_594 = Vertex(name = 'V_594',
               particles = [ P.e__plus__, P.e__minus__, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_216,(0,1):C.GC_213})

V_595 = Vertex(name = 'V_595',
               particles = [ P.e__plus__, P.e__minus__, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_217,(0,1):C.GC_214})

V_596 = Vertex(name = 'V_596',
               particles = [ P.e__plus__, P.e__minus__, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_383,(0,0):C.GC_381})

V_597 = Vertex(name = 'V_597',
               particles = [ P.mu__plus__, P.mu__minus__, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_219,(0,1):C.GC_225})

V_598 = Vertex(name = 'V_598',
               particles = [ P.mu__plus__, P.mu__minus__, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_220,(0,1):C.GC_226})

V_599 = Vertex(name = 'V_599',
               particles = [ P.mu__plus__, P.mu__minus__, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_385,(0,0):C.GC_389})

V_600 = Vertex(name = 'V_600',
               particles = [ P.ta__plus__, P.ta__minus__, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_222,(0,1):C.GC_228})

V_601 = Vertex(name = 'V_601',
               particles = [ P.ta__plus__, P.ta__minus__, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_223,(0,1):C.GC_229})

V_602 = Vertex(name = 'V_602',
               particles = [ P.ta__plus__, P.ta__minus__, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_387,(0,0):C.GC_391})

V_603 = Vertex(name = 'V_603',
               particles = [ P.e__plus__, P.e__minus__, P.Z, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_291,(0,1):C.GC_287})

V_604 = Vertex(name = 'V_604',
               particles = [ P.e__plus__, P.e__minus__, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_288,(0,1):C.GC_286})

V_605 = Vertex(name = 'V_605',
               particles = [ P.e__plus__, P.e__minus__, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_291,(0,1):C.GC_287})

V_606 = Vertex(name = 'V_606',
               particles = [ P.e__plus__, P.e__minus__, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_450,(0,0):C.GC_448})

V_607 = Vertex(name = 'V_607',
               particles = [ P.mu__plus__, P.mu__minus__, P.Z, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_295,(0,1):C.GC_301})

V_608 = Vertex(name = 'V_608',
               particles = [ P.mu__plus__, P.mu__minus__, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_292,(0,1):C.GC_300})

V_609 = Vertex(name = 'V_609',
               particles = [ P.mu__plus__, P.mu__minus__, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_295,(0,1):C.GC_301})

V_610 = Vertex(name = 'V_610',
               particles = [ P.mu__plus__, P.mu__minus__, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_452,(0,0):C.GC_455})

V_611 = Vertex(name = 'V_611',
               particles = [ P.ta__plus__, P.ta__minus__, P.Z, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_299,(0,1):C.GC_303})

V_612 = Vertex(name = 'V_612',
               particles = [ P.ta__plus__, P.ta__minus__, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_296,(0,1):C.GC_302})

V_613 = Vertex(name = 'V_613',
               particles = [ P.ta__plus__, P.ta__minus__, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13, L.FFVSS14 ],
               couplings = {(0,0):C.GC_299,(0,1):C.GC_303})

V_614 = Vertex(name = 'V_614',
               particles = [ P.ta__plus__, P.ta__minus__, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS120, L.FFVS94 ],
               couplings = {(0,1):C.GC_454,(0,0):C.GC_456})

V_615 = Vertex(name = 'V_615',
               particles = [ P.s__tilde__, P.c, P.a, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2009})

V_616 = Vertex(name = 'V_616',
               particles = [ P.s__tilde__, P.c, P.a, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2010})

V_617 = Vertex(name = 'V_617',
               particles = [ P.s__tilde__, P.c, P.a, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_2081})

V_618 = Vertex(name = 'V_618',
               particles = [ P.s__tilde__, P.c, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_1995})

V_619 = Vertex(name = 'V_619',
               particles = [ P.s__tilde__, P.c, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_1994})

V_620 = Vertex(name = 'V_620',
               particles = [ P.s__tilde__, P.c, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS45 ],
               couplings = {(0,0):C.GC_2078})

V_621 = Vertex(name = 'V_621',
               particles = [ P.b__tilde__, P.t, P.a, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2012})

V_622 = Vertex(name = 'V_622',
               particles = [ P.b__tilde__, P.t, P.a, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2013})

V_623 = Vertex(name = 'V_623',
               particles = [ P.b__tilde__, P.t, P.a, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS108, L.FFVS94 ],
               couplings = {(0,0):C.GC_278,(0,1):C.GC_2083})

V_624 = Vertex(name = 'V_624',
               particles = [ P.b__tilde__, P.t, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_1998})

V_625 = Vertex(name = 'V_625',
               particles = [ P.d__tilde__, P.u, P.a, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2009})

V_626 = Vertex(name = 'V_626',
               particles = [ P.d__tilde__, P.u, P.a, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2010})

V_627 = Vertex(name = 'V_627',
               particles = [ P.d__tilde__, P.u, P.a, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_2081})

V_628 = Vertex(name = 'V_628',
               particles = [ P.d__tilde__, P.u, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_1995})

V_629 = Vertex(name = 'V_629',
               particles = [ P.d__tilde__, P.u, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_1994})

V_630 = Vertex(name = 'V_630',
               particles = [ P.d__tilde__, P.u, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS45 ],
               couplings = {(0,0):C.GC_2078})

V_631 = Vertex(name = 'V_631',
               particles = [ P.s__tilde__, P.c, P.W__minus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2025})

V_632 = Vertex(name = 'V_632',
               particles = [ P.s__tilde__, P.c, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2025})

V_633 = Vertex(name = 'V_633',
               particles = [ P.s__tilde__, P.c, P.W__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2025})

V_634 = Vertex(name = 'V_634',
               particles = [ P.s__tilde__, P.c, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_2099})

V_635 = Vertex(name = 'V_635',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2029})

V_636 = Vertex(name = 'V_636',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2029})

V_637 = Vertex(name = 'V_637',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2029})

V_638 = Vertex(name = 'V_638',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS108, L.FFVS94 ],
               couplings = {(0,0):C.GC_115,(0,1):C.GC_2102})

V_639 = Vertex(name = 'V_639',
               particles = [ P.d__tilde__, P.u, P.W__minus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2025})

V_640 = Vertex(name = 'V_640',
               particles = [ P.d__tilde__, P.u, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2025})

V_641 = Vertex(name = 'V_641',
               particles = [ P.d__tilde__, P.u, P.W__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2025})

V_642 = Vertex(name = 'V_642',
               particles = [ P.d__tilde__, P.u, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_2099})

V_643 = Vertex(name = 'V_643',
               particles = [ P.s__tilde__, P.c, P.Z, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2051})

V_644 = Vertex(name = 'V_644',
               particles = [ P.s__tilde__, P.c, P.Z, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2052})

V_645 = Vertex(name = 'V_645',
               particles = [ P.s__tilde__, P.c, P.Z, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_2122})

V_646 = Vertex(name = 'V_646',
               particles = [ P.b__tilde__, P.t, P.Z, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2054})

V_647 = Vertex(name = 'V_647',
               particles = [ P.b__tilde__, P.t, P.Z, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2055})

V_648 = Vertex(name = 'V_648',
               particles = [ P.b__tilde__, P.t, P.Z, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS108, L.FFVS94 ],
               couplings = {(0,0):C.GC_36,(0,1):C.GC_2124})

V_649 = Vertex(name = 'V_649',
               particles = [ P.d__tilde__, P.u, P.Z, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2051})

V_650 = Vertex(name = 'V_650',
               particles = [ P.d__tilde__, P.u, P.Z, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2052})

V_651 = Vertex(name = 'V_651',
               particles = [ P.d__tilde__, P.u, P.Z, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_2122})

V_652 = Vertex(name = 'V_652',
               particles = [ P.t__tilde__, P.b, P.a, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2012})

V_653 = Vertex(name = 'V_653',
               particles = [ P.t__tilde__, P.b, P.a, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2011})

V_654 = Vertex(name = 'V_654',
               particles = [ P.t__tilde__, P.b, P.a, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS165, L.FFVS94 ],
               couplings = {(0,0):C.GC_277,(0,1):C.GC_2082})

V_655 = Vertex(name = 'V_655',
               particles = [ P.t__tilde__, P.b, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_1999})

V_656 = Vertex(name = 'V_656',
               particles = [ P.u__tilde__, P.d, P.a, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2009})

V_657 = Vertex(name = 'V_657',
               particles = [ P.u__tilde__, P.d, P.a, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2008})

V_658 = Vertex(name = 'V_658',
               particles = [ P.u__tilde__, P.d, P.a, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_2080})

V_659 = Vertex(name = 'V_659',
               particles = [ P.u__tilde__, P.d, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_1996})

V_660 = Vertex(name = 'V_660',
               particles = [ P.u__tilde__, P.d, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_1994})

V_661 = Vertex(name = 'V_661',
               particles = [ P.u__tilde__, P.d, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS45 ],
               couplings = {(0,0):C.GC_2078})

V_662 = Vertex(name = 'V_662',
               particles = [ P.c__tilde__, P.s, P.a, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2009})

V_663 = Vertex(name = 'V_663',
               particles = [ P.c__tilde__, P.s, P.a, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2008})

V_664 = Vertex(name = 'V_664',
               particles = [ P.c__tilde__, P.s, P.a, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_2080})

V_665 = Vertex(name = 'V_665',
               particles = [ P.c__tilde__, P.s, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_1996})

V_666 = Vertex(name = 'V_666',
               particles = [ P.c__tilde__, P.s, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_1994})

V_667 = Vertex(name = 'V_667',
               particles = [ P.c__tilde__, P.s, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS45 ],
               couplings = {(0,0):C.GC_2078})

V_668 = Vertex(name = 'V_668',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2029})

V_669 = Vertex(name = 'V_669',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2029})

V_670 = Vertex(name = 'V_670',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2029})

V_671 = Vertex(name = 'V_671',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS165, L.FFVS94 ],
               couplings = {(0,0):C.GC_115,(0,1):C.GC_2102})

V_672 = Vertex(name = 'V_672',
               particles = [ P.u__tilde__, P.d, P.W__plus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2025})

V_673 = Vertex(name = 'V_673',
               particles = [ P.u__tilde__, P.d, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2025})

V_674 = Vertex(name = 'V_674',
               particles = [ P.u__tilde__, P.d, P.W__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2025})

V_675 = Vertex(name = 'V_675',
               particles = [ P.u__tilde__, P.d, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_2099})

V_676 = Vertex(name = 'V_676',
               particles = [ P.c__tilde__, P.s, P.W__plus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2025})

V_677 = Vertex(name = 'V_677',
               particles = [ P.c__tilde__, P.s, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2025})

V_678 = Vertex(name = 'V_678',
               particles = [ P.c__tilde__, P.s, P.W__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2025})

V_679 = Vertex(name = 'V_679',
               particles = [ P.c__tilde__, P.s, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_2099})

V_680 = Vertex(name = 'V_680',
               particles = [ P.t__tilde__, P.b, P.Z, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2054})

V_681 = Vertex(name = 'V_681',
               particles = [ P.t__tilde__, P.b, P.Z, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2053})

V_682 = Vertex(name = 'V_682',
               particles = [ P.t__tilde__, P.b, P.Z, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS165, L.FFVS94 ],
               couplings = {(0,0):C.GC_37,(0,1):C.GC_2123})

V_683 = Vertex(name = 'V_683',
               particles = [ P.u__tilde__, P.d, P.Z, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2051})

V_684 = Vertex(name = 'V_684',
               particles = [ P.u__tilde__, P.d, P.Z, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2050})

V_685 = Vertex(name = 'V_685',
               particles = [ P.u__tilde__, P.d, P.Z, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_2121})

V_686 = Vertex(name = 'V_686',
               particles = [ P.c__tilde__, P.s, P.Z, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2051})

V_687 = Vertex(name = 'V_687',
               particles = [ P.c__tilde__, P.s, P.Z, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_2050})

V_688 = Vertex(name = 'V_688',
               particles = [ P.c__tilde__, P.s, P.Z, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_2121})

V_689 = Vertex(name = 'V_689',
               particles = [ P.e__plus__, P.ve, P.a, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_126})

V_690 = Vertex(name = 'V_690',
               particles = [ P.e__plus__, P.ve, P.a, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_127})

V_691 = Vertex(name = 'V_691',
               particles = [ P.e__plus__, P.ve, P.a, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_345})

V_692 = Vertex(name = 'V_692',
               particles = [ P.e__plus__, P.ve, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_45})

V_693 = Vertex(name = 'V_693',
               particles = [ P.e__plus__, P.ve, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_44})

V_694 = Vertex(name = 'V_694',
               particles = [ P.e__plus__, P.ve, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS45 ],
               couplings = {(0,0):C.GC_325})

V_695 = Vertex(name = 'V_695',
               particles = [ P.mu__plus__, P.vm, P.a, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_129})

V_696 = Vertex(name = 'V_696',
               particles = [ P.mu__plus__, P.vm, P.a, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_130})

V_697 = Vertex(name = 'V_697',
               particles = [ P.mu__plus__, P.vm, P.a, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_347})

V_698 = Vertex(name = 'V_698',
               particles = [ P.mu__plus__, P.vm, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_48})

V_699 = Vertex(name = 'V_699',
               particles = [ P.mu__plus__, P.vm, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_47})

V_700 = Vertex(name = 'V_700',
               particles = [ P.mu__plus__, P.vm, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS45 ],
               couplings = {(0,0):C.GC_326})

V_701 = Vertex(name = 'V_701',
               particles = [ P.ta__plus__, P.vt, P.a, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_132})

V_702 = Vertex(name = 'V_702',
               particles = [ P.ta__plus__, P.vt, P.a, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_133})

V_703 = Vertex(name = 'V_703',
               particles = [ P.ta__plus__, P.vt, P.a, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_349})

V_704 = Vertex(name = 'V_704',
               particles = [ P.ta__plus__, P.vt, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_51})

V_705 = Vertex(name = 'V_705',
               particles = [ P.ta__plus__, P.vt, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_50})

V_706 = Vertex(name = 'V_706',
               particles = [ P.ta__plus__, P.vt, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS45 ],
               couplings = {(0,0):C.GC_327})

V_707 = Vertex(name = 'V_707',
               particles = [ P.e__plus__, P.ve, P.W__minus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_204})

V_708 = Vertex(name = 'V_708',
               particles = [ P.e__plus__, P.ve, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_204})

V_709 = Vertex(name = 'V_709',
               particles = [ P.e__plus__, P.ve, P.W__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_204})

V_710 = Vertex(name = 'V_710',
               particles = [ P.e__plus__, P.ve, P.W__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_373})

V_711 = Vertex(name = 'V_711',
               particles = [ P.mu__plus__, P.vm, P.W__minus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_205})

V_712 = Vertex(name = 'V_712',
               particles = [ P.mu__plus__, P.vm, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_205})

V_713 = Vertex(name = 'V_713',
               particles = [ P.mu__plus__, P.vm, P.W__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_205})

V_714 = Vertex(name = 'V_714',
               particles = [ P.mu__plus__, P.vm, P.W__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_374})

V_715 = Vertex(name = 'V_715',
               particles = [ P.ta__plus__, P.vt, P.W__minus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_206})

V_716 = Vertex(name = 'V_716',
               particles = [ P.ta__plus__, P.vt, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_206})

V_717 = Vertex(name = 'V_717',
               particles = [ P.ta__plus__, P.vt, P.W__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_206})

V_718 = Vertex(name = 'V_718',
               particles = [ P.ta__plus__, P.vt, P.W__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_375})

V_719 = Vertex(name = 'V_719',
               particles = [ P.e__plus__, P.ve, P.Z, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_263})

V_720 = Vertex(name = 'V_720',
               particles = [ P.e__plus__, P.ve, P.Z, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_264})

V_721 = Vertex(name = 'V_721',
               particles = [ P.e__plus__, P.ve, P.Z, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_405})

V_722 = Vertex(name = 'V_722',
               particles = [ P.mu__plus__, P.vm, P.Z, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_266})

V_723 = Vertex(name = 'V_723',
               particles = [ P.mu__plus__, P.vm, P.Z, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_267})

V_724 = Vertex(name = 'V_724',
               particles = [ P.mu__plus__, P.vm, P.Z, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_407})

V_725 = Vertex(name = 'V_725',
               particles = [ P.ta__plus__, P.vt, P.Z, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_269})

V_726 = Vertex(name = 'V_726',
               particles = [ P.ta__plus__, P.vt, P.Z, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_270})

V_727 = Vertex(name = 'V_727',
               particles = [ P.ta__plus__, P.vt, P.Z, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_409})

V_728 = Vertex(name = 'V_728',
               particles = [ P.ve__tilde__, P.e__minus__, P.a, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_126})

V_729 = Vertex(name = 'V_729',
               particles = [ P.ve__tilde__, P.e__minus__, P.a, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_125})

V_730 = Vertex(name = 'V_730',
               particles = [ P.ve__tilde__, P.e__minus__, P.a, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_344})

V_731 = Vertex(name = 'V_731',
               particles = [ P.ve__tilde__, P.e__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_46})

V_732 = Vertex(name = 'V_732',
               particles = [ P.ve__tilde__, P.e__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_44})

V_733 = Vertex(name = 'V_733',
               particles = [ P.ve__tilde__, P.e__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS45 ],
               couplings = {(0,0):C.GC_325})

V_734 = Vertex(name = 'V_734',
               particles = [ P.vm__tilde__, P.mu__minus__, P.a, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_129})

V_735 = Vertex(name = 'V_735',
               particles = [ P.vm__tilde__, P.mu__minus__, P.a, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_128})

V_736 = Vertex(name = 'V_736',
               particles = [ P.vm__tilde__, P.mu__minus__, P.a, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_346})

V_737 = Vertex(name = 'V_737',
               particles = [ P.vm__tilde__, P.mu__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_49})

V_738 = Vertex(name = 'V_738',
               particles = [ P.vm__tilde__, P.mu__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_47})

V_739 = Vertex(name = 'V_739',
               particles = [ P.vm__tilde__, P.mu__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS45 ],
               couplings = {(0,0):C.GC_326})

V_740 = Vertex(name = 'V_740',
               particles = [ P.vt__tilde__, P.ta__minus__, P.a, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_132})

V_741 = Vertex(name = 'V_741',
               particles = [ P.vt__tilde__, P.ta__minus__, P.a, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_131})

V_742 = Vertex(name = 'V_742',
               particles = [ P.vt__tilde__, P.ta__minus__, P.a, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_348})

V_743 = Vertex(name = 'V_743',
               particles = [ P.vt__tilde__, P.ta__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_52})

V_744 = Vertex(name = 'V_744',
               particles = [ P.vt__tilde__, P.ta__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS30 ],
               couplings = {(0,0):C.GC_50})

V_745 = Vertex(name = 'V_745',
               particles = [ P.vt__tilde__, P.ta__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS45 ],
               couplings = {(0,0):C.GC_327})

V_746 = Vertex(name = 'V_746',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_204})

V_747 = Vertex(name = 'V_747',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_204})

V_748 = Vertex(name = 'V_748',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_204})

V_749 = Vertex(name = 'V_749',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_373})

V_750 = Vertex(name = 'V_750',
               particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_205})

V_751 = Vertex(name = 'V_751',
               particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_205})

V_752 = Vertex(name = 'V_752',
               particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_205})

V_753 = Vertex(name = 'V_753',
               particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_374})

V_754 = Vertex(name = 'V_754',
               particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_206})

V_755 = Vertex(name = 'V_755',
               particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_206})

V_756 = Vertex(name = 'V_756',
               particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_206})

V_757 = Vertex(name = 'V_757',
               particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_375})

V_758 = Vertex(name = 'V_758',
               particles = [ P.ve__tilde__, P.e__minus__, P.Z, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_263})

V_759 = Vertex(name = 'V_759',
               particles = [ P.ve__tilde__, P.e__minus__, P.Z, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_262})

V_760 = Vertex(name = 'V_760',
               particles = [ P.ve__tilde__, P.e__minus__, P.Z, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_404})

V_761 = Vertex(name = 'V_761',
               particles = [ P.vm__tilde__, P.mu__minus__, P.Z, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_266})

V_762 = Vertex(name = 'V_762',
               particles = [ P.vm__tilde__, P.mu__minus__, P.Z, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_265})

V_763 = Vertex(name = 'V_763',
               particles = [ P.vm__tilde__, P.mu__minus__, P.Z, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_406})

V_764 = Vertex(name = 'V_764',
               particles = [ P.vt__tilde__, P.ta__minus__, P.Z, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_269})

V_765 = Vertex(name = 'V_765',
               particles = [ P.vt__tilde__, P.ta__minus__, P.Z, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS13 ],
               couplings = {(0,0):C.GC_268})

V_766 = Vertex(name = 'V_766',
               particles = [ P.vt__tilde__, P.ta__minus__, P.Z, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS94 ],
               couplings = {(0,0):C.GC_408})

V_767 = Vertex(name = 'V_767',
               particles = [ P.b__tilde__, P.t, P.g, P.G__minus__ ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFVS108 ],
               couplings = {(0,0):C.GC_154})

V_768 = Vertex(name = 'V_768',
               particles = [ P.t__tilde__, P.t, P.g, P.G0 ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFVS88 ],
               couplings = {(0,0):C.GC_157})

V_769 = Vertex(name = 'V_769',
               particles = [ P.t__tilde__, P.t, P.g, P.H ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFVS127 ],
               couplings = {(0,0):C.GC_156})

V_770 = Vertex(name = 'V_770',
               particles = [ P.b__tilde__, P.t, P.g, P.g, P.G__minus__ ],
               color = [ 'f(-1,3,4)*T(-1,2,1)' ],
               lorentz = [ L.FFVVS109 ],
               couplings = {(0,0):C.GC_159})

V_771 = Vertex(name = 'V_771',
               particles = [ P.t__tilde__, P.t, P.g, P.g, P.G0 ],
               color = [ 'f(-1,3,4)*T(-1,2,1)' ],
               lorentz = [ L.FFVVS141 ],
               couplings = {(0,0):C.GC_162})

V_772 = Vertex(name = 'V_772',
               particles = [ P.t__tilde__, P.t, P.g, P.g, P.H ],
               color = [ 'f(-1,3,4)*T(-1,2,1)' ],
               lorentz = [ L.FFVVS151 ],
               couplings = {(0,0):C.GC_161})

V_773 = Vertex(name = 'V_773',
               particles = [ P.t__tilde__, P.t, P.g, P.g ],
               color = [ 'f(-1,3,4)*T(-1,2,1)' ],
               lorentz = [ L.FFVV186 ],
               couplings = {(0,0):C.GC_360})

V_774 = Vertex(name = 'V_774',
               particles = [ P.t__tilde__, P.t, P.a, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS88 ],
               couplings = {(0,0):C.GC_280})

V_775 = Vertex(name = 'V_775',
               particles = [ P.t__tilde__, P.t, P.Z, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS88 ],
               couplings = {(0,0):C.GC_119})

V_776 = Vertex(name = 'V_776',
               particles = [ P.t__tilde__, P.t, P.a, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS127 ],
               couplings = {(0,0):C.GC_279})

V_777 = Vertex(name = 'V_777',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS108 ],
               couplings = {(0,0):C.GC_114})

V_778 = Vertex(name = 'V_778',
               particles = [ P.t__tilde__, P.t, P.a, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS109 ],
               couplings = {(0,0):C.GC_145})

V_779 = Vertex(name = 'V_779',
               particles = [ P.b__tilde__, P.t, P.a, P.W__minus__, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS109 ],
               couplings = {(0,0):C.GC_142})

V_780 = Vertex(name = 'V_780',
               particles = [ P.b__tilde__, P.t, P.a, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS109 ],
               couplings = {(0,0):C.GC_144})

V_781 = Vertex(name = 'V_781',
               particles = [ P.b__tilde__, P.t, P.a, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVV128 ],
               couplings = {(0,0):C.GC_354})

V_782 = Vertex(name = 'V_782',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS79 ],
               couplings = {(0,0):C.GC_233})

V_783 = Vertex(name = 'V_783',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS120 ],
               couplings = {(0,0):C.GC_232})

V_784 = Vertex(name = 'V_784',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVV168 ],
               couplings = {(0,0):C.GC_392})

V_785 = Vertex(name = 'V_785',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS109 ],
               couplings = {(0,0):C.GC_231})

V_786 = Vertex(name = 'V_786',
               particles = [ P.t__tilde__, P.t, P.W__plus__, P.Z, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS109 ],
               couplings = {(0,0):C.GC_240})

V_787 = Vertex(name = 'V_787',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS109 ],
               couplings = {(0,0):C.GC_237})

V_788 = Vertex(name = 'V_788',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS109 ],
               couplings = {(0,0):C.GC_239})

V_789 = Vertex(name = 'V_789',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVV128 ],
               couplings = {(0,0):C.GC_396})

V_790 = Vertex(name = 'V_790',
               particles = [ P.t__tilde__, P.b, P.g, P.G__plus__ ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFVS165 ],
               couplings = {(0,0):C.GC_155})

V_791 = Vertex(name = 'V_791',
               particles = [ P.t__tilde__, P.b, P.g, P.g, P.G__plus__ ],
               color = [ 'f(-1,3,4)*T(-1,2,1)' ],
               lorentz = [ L.FFVVS150 ],
               couplings = {(0,0):C.GC_160})

V_792 = Vertex(name = 'V_792',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS165 ],
               couplings = {(0,0):C.GC_116})

V_793 = Vertex(name = 'V_793',
               particles = [ P.t__tilde__, P.t, P.a, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS150 ],
               couplings = {(0,0):C.GC_145})

V_794 = Vertex(name = 'V_794',
               particles = [ P.t__tilde__, P.b, P.a, P.W__plus__, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS150 ],
               couplings = {(0,0):C.GC_142})

V_795 = Vertex(name = 'V_795',
               particles = [ P.t__tilde__, P.b, P.a, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS150 ],
               couplings = {(0,0):C.GC_143})

V_796 = Vertex(name = 'V_796',
               particles = [ P.t__tilde__, P.b, P.a, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVV185 ],
               couplings = {(0,0):C.GC_353})

V_797 = Vertex(name = 'V_797',
               particles = [ P.t__tilde__, P.b, P.W__minus__, P.W__plus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS150 ],
               couplings = {(0,0):C.GC_230})

V_798 = Vertex(name = 'V_798',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.Z, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS150 ],
               couplings = {(0,0):C.GC_240})

V_799 = Vertex(name = 'V_799',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS150 ],
               couplings = {(0,0):C.GC_237})

V_800 = Vertex(name = 'V_800',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVVS150 ],
               couplings = {(0,0):C.GC_238})

V_801 = Vertex(name = 'V_801',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVV185 ],
               couplings = {(0,0):C.GC_395})

