# This file was automatically created by FeynRules $Revision: 216 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Thu 22 Jul 2010 16:38:04


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.G, P.G, P.G ],
             color = [ 'f(1,2,3)' ],
             lorentz = [ L.VVV1 ],
             couplings = {(0,0):C.GC_9})

V_2 = Vertex(name = 'V_2',
             particles = [ P.G, P.G, P.G, P.G ],
             color = [ 'f(2,3,\'a1\')*f(\'a1\',1,4)', 'f(2,4,\'a1\')*f(\'a1\',1,3)', 'f(3,4,\'a1\')*f(\'a1\',1,2)' ],
             lorentz = [ L.VVVV1, L.VVVV3, L.VVVV4 ],
             couplings = {(1,1):C.GC_11,(2,0):C.GC_11,(0,2):C.GC_11})

V_3 = Vertex(name = 'V_3',
             particles = [ P.A, P.W__minus__, P.W__plus__ ],
             color = [ '1' ],
             lorentz = [ L.VVV1 ],
             couplings = {(0,0):C.GC_49})

V_4 = Vertex(name = 'V_4',
             particles = [ P.A, P.A, P.W__minus__, P.W__plus__ ],
             color = [ '1' ],
             lorentz = [ L.VVVV2 ],
             couplings = {(0,0):C.GC_51})

V_5 = Vertex(name = 'V_5',
             particles = [ P.W__minus__, P.W__plus__, P.Z ],
             color = [ '1' ],
             lorentz = [ L.VVV1 ],
             couplings = {(0,0):C.GC_12})

V_6 = Vertex(name = 'V_6',
             particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
             color = [ '1' ],
             lorentz = [ L.VVVV2 ],
             couplings = {(0,0):C.GC_13})

V_7 = Vertex(name = 'V_7',
             particles = [ P.A, P.W__minus__, P.W__plus__, P.Z ],
             color = [ '1' ],
             lorentz = [ L.VVVV5 ],
             couplings = {(0,0):C.GC_50})

V_8 = Vertex(name = 'V_8',
             particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
             color = [ '1' ],
             lorentz = [ L.VVVV2 ],
             couplings = {(0,0):C.GC_14})

V_9 = Vertex(name = 'V_9',
             particles = [ P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_17})

V_10 = Vertex(name = 'V_10',
              particles = [ P.H, P.H, P.phi0, P.phi0 ],
              color = [ '1' ],
              lorentz = [ L.SSSS1 ],
              couplings = {(0,0):C.GC_15})

V_11 = Vertex(name = 'V_11',
              particles = [ P.phi0, P.phi0, P.phi0, P.phi0 ],
              color = [ '1' ],
              lorentz = [ L.SSSS1 ],
              couplings = {(0,0):C.GC_17})

V_12 = Vertex(name = 'V_12',
              particles = [ P.H, P.H, P.phi__minus__, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSSS1 ],
              couplings = {(0,0):C.GC_15})

V_13 = Vertex(name = 'V_13',
              particles = [ P.phi0, P.phi0, P.phi__minus__, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSSS1 ],
              couplings = {(0,0):C.GC_15})

V_14 = Vertex(name = 'V_14',
              particles = [ P.phi__minus__, P.phi__minus__, P.phi__plus__, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSSS1 ],
              couplings = {(0,0):C.GC_16})

V_15 = Vertex(name = 'V_15',
              particles = [ P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS1 ],
              couplings = {(0,0):C.GC_63})

V_16 = Vertex(name = 'V_16',
              particles = [ P.H, P.phi0, P.phi0 ],
              color = [ '1' ],
              lorentz = [ L.SSS1 ],
              couplings = {(0,0):C.GC_62})

V_17 = Vertex(name = 'V_17',
              particles = [ P.H, P.phi__minus__, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSS1 ],
              couplings = {(0,0):C.GC_62})

V_18 = Vertex(name = 'V_18',
              particles = [ P.A, P.A, P.phi__minus__, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_5})

V_19 = Vertex(name = 'V_19',
              particles = [ P.A, P.phi__minus__, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_3})

V_20 = Vertex(name = 'V_20',
              particles = [ P.A, P.W__minus__, P.H, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_38})

V_21 = Vertex(name = 'V_21',
              particles = [ P.A, P.W__minus__, P.phi0, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_39})

V_22 = Vertex(name = 'V_22',
              particles = [ P.A, P.W__minus__, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_65})

V_23 = Vertex(name = 'V_23',
              particles = [ P.W__minus__, P.H, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_23})

V_24 = Vertex(name = 'V_24',
              particles = [ P.W__minus__, P.phi0, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_22})

V_25 = Vertex(name = 'V_25',
              particles = [ P.A, P.W__plus__, P.H, P.phi__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_40})

V_26 = Vertex(name = 'V_26',
              particles = [ P.A, P.W__plus__, P.phi0, P.phi__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_39})

V_27 = Vertex(name = 'V_27',
              particles = [ P.A, P.W__plus__, P.phi__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_66})

V_28 = Vertex(name = 'V_28',
              particles = [ P.W__plus__, P.H, P.phi__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_23})

V_29 = Vertex(name = 'V_29',
              particles = [ P.W__plus__, P.phi0, P.phi__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_21})

V_30 = Vertex(name = 'V_30',
              particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_20})

V_31 = Vertex(name = 'V_31',
              particles = [ P.W__minus__, P.W__plus__, P.phi0, P.phi0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_20})

V_32 = Vertex(name = 'V_32',
              particles = [ P.W__minus__, P.W__plus__, P.phi__minus__, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_20})

V_33 = Vertex(name = 'V_33',
              particles = [ P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_64})

V_34 = Vertex(name = 'V_34',
              particles = [ P.A, P.Z, P.phi__minus__, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_55})

V_35 = Vertex(name = 'V_35',
              particles = [ P.Z, P.H, P.phi0 ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_54})

V_36 = Vertex(name = 'V_36',
              particles = [ P.Z, P.phi__minus__, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_52})

V_37 = Vertex(name = 'V_37',
              particles = [ P.W__minus__, P.Z, P.H, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_8})

V_38 = Vertex(name = 'V_38',
              particles = [ P.W__minus__, P.Z, P.phi0, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_7})

V_39 = Vertex(name = 'V_39',
              particles = [ P.W__minus__, P.Z, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_61})

V_40 = Vertex(name = 'V_40',
              particles = [ P.W__plus__, P.Z, P.H, P.phi__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_6})

V_41 = Vertex(name = 'V_41',
              particles = [ P.W__plus__, P.Z, P.phi0, P.phi__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_7})

V_42 = Vertex(name = 'V_42',
              particles = [ P.W__plus__, P.Z, P.phi__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_60})

V_43 = Vertex(name = 'V_43',
              particles = [ P.Z, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_59})

V_44 = Vertex(name = 'V_44',
              particles = [ P.Z, P.Z, P.phi0, P.phi0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_59})

V_45 = Vertex(name = 'V_45',
              particles = [ P.Z, P.Z, P.phi__minus__, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_58})

V_46 = Vertex(name = 'V_46',
              particles = [ P.Z, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_67})

V_47 = Vertex(name = 'V_47',
              particles = [ P.ghA, P.ghWm__tilde__, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV2 ],
              couplings = {(0,0):C.GC_3})

V_48 = Vertex(name = 'V_48',
              particles = [ P.ghA, P.ghWp__tilde__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV2 ],
              couplings = {(0,0):C.GC_4})

V_49 = Vertex(name = 'V_49',
              particles = [ P.ghA__tilde__, P.ghWm, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_19})

V_50 = Vertex(name = 'V_50',
              particles = [ P.ghA__tilde__, P.ghWm, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_3})

V_51 = Vertex(name = 'V_51',
              particles = [ P.ghWm__tilde__, P.ghWm, P.H ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_42})

V_52 = Vertex(name = 'V_52',
              particles = [ P.ghWm__tilde__, P.ghWm, P.phi0 ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_41})

V_53 = Vertex(name = 'V_53',
              particles = [ P.ghWm__tilde__, P.ghWm, P.A ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_4})

V_54 = Vertex(name = 'V_54',
              particles = [ P.ghWm__tilde__, P.ghWm, P.Z ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_37})

V_55 = Vertex(name = 'V_55',
              particles = [ P.ghWm, P.ghZ__tilde__, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_56})

V_56 = Vertex(name = 'V_56',
              particles = [ P.ghWm, P.ghZ__tilde__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV2 ],
              couplings = {(0,0):C.GC_36})

V_57 = Vertex(name = 'V_57',
              particles = [ P.ghA__tilde__, P.ghWp, P.phi__minus__ ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_18})

V_58 = Vertex(name = 'V_58',
              particles = [ P.ghA__tilde__, P.ghWp, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_4})

V_59 = Vertex(name = 'V_59',
              particles = [ P.ghWp__tilde__, P.ghWp, P.H ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_42})

V_60 = Vertex(name = 'V_60',
              particles = [ P.ghWp__tilde__, P.ghWp, P.phi0 ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_43})

V_61 = Vertex(name = 'V_61',
              particles = [ P.ghWp__tilde__, P.ghWp, P.A ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_3})

V_62 = Vertex(name = 'V_62',
              particles = [ P.ghWp__tilde__, P.ghWp, P.Z ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_36})

V_63 = Vertex(name = 'V_63',
              particles = [ P.ghWp, P.ghZ__tilde__, P.phi__minus__ ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_57})

V_64 = Vertex(name = 'V_64',
              particles = [ P.ghWp, P.ghZ__tilde__, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV2 ],
              couplings = {(0,0):C.GC_37})

V_65 = Vertex(name = 'V_65',
              particles = [ P.ghWm__tilde__, P.ghZ, P.phi__minus__ ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_45})

V_66 = Vertex(name = 'V_66',
              particles = [ P.ghWm__tilde__, P.ghZ, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_36})

V_67 = Vertex(name = 'V_67',
              particles = [ P.ghWp__tilde__, P.ghZ, P.phi__plus__ ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_44})

V_68 = Vertex(name = 'V_68',
              particles = [ P.ghWp__tilde__, P.ghZ, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_37})

V_69 = Vertex(name = 'V_69',
              particles = [ P.ghZ__tilde__, P.ghZ, P.H ],
              color = [ '1' ],
              lorentz = [ L.UUS1 ],
              couplings = {(0,0):C.GC_46})

V_70 = Vertex(name = 'V_70',
              particles = [ P.ghG__tilde__, P.ghG, P.G ],
              color = [ 'f(3,2,1)' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_9})

V_71 = Vertex(name = 'V_71',
              particles = [ P.d__tilde__, P.d, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_72 = Vertex(name = 'V_72',
              particles = [ P.s__tilde__, P.s, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_73 = Vertex(name = 'V_73',
              particles = [ P.b__tilde__, P.b, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_1})

V_74 = Vertex(name = 'V_74',
              particles = [ P.e__plus__, P.e__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_75 = Vertex(name = 'V_75',
              particles = [ P.m__plus__, P.m__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_76 = Vertex(name = 'V_76',
              particles = [ P.tt__plus__, P.tt__minus__, P.A ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_77 = Vertex(name = 'V_77',
              particles = [ P.u__tilde__, P.u, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_78 = Vertex(name = 'V_78',
              particles = [ P.c__tilde__, P.c, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_79 = Vertex(name = 'V_79',
              particles = [ P.t__tilde__, P.t, P.A ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_80 = Vertex(name = 'V_80',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_10})

V_81 = Vertex(name = 'V_81',
              particles = [ P.s__tilde__, P.s, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_10})

V_82 = Vertex(name = 'V_82',
              particles = [ P.b__tilde__, P.b, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_10})

V_83 = Vertex(name = 'V_83',
              particles = [ P.d__tilde__, P.d, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_78})

V_84 = Vertex(name = 'V_84',
              particles = [ P.s__tilde__, P.s, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_91})

V_85 = Vertex(name = 'V_85',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
              couplings = {(0,0):C.GC_68})

V_86 = Vertex(name = 'V_86',
              particles = [ P.d__tilde__, P.d, P.phi0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_79})

V_87 = Vertex(name = 'V_87',
              particles = [ P.s__tilde__, P.s, P.phi0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_92})

V_88 = Vertex(name = 'V_88',
              particles = [ P.b__tilde__, P.b, P.phi0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_69})

V_89 = Vertex(name = 'V_89',
              particles = [ P.d__tilde__, P.d, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              couplings = {(0,0):C.GC_34,(0,1):C.GC_47})

V_90 = Vertex(name = 'V_90',
              particles = [ P.s__tilde__, P.s, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              couplings = {(0,0):C.GC_34,(0,1):C.GC_47})

V_91 = Vertex(name = 'V_91',
              particles = [ P.b__tilde__, P.b, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              couplings = {(0,0):C.GC_34,(0,1):C.GC_47})

V_92 = Vertex(name = 'V_92',
              particles = [ P.d__tilde__, P.u, P.phi__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS3, L.FFS4 ],
              couplings = {(0,0):C.GC_107,(0,1):C.GC_80})

V_93 = Vertex(name = 'V_93',
              particles = [ P.d__tilde__, P.c, P.phi__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS3, L.FFS4 ],
              couplings = {(0,0):C.GC_75,(0,1):C.GC_81})

V_94 = Vertex(name = 'V_94',
              particles = [ P.d__tilde__, P.t, P.phi__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS3, L.FFS4 ],
              couplings = {(0,0):C.GC_98,(0,1):C.GC_82})

V_95 = Vertex(name = 'V_95',
              particles = [ P.s__tilde__, P.u, P.phi__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS3, L.FFS4 ],
              couplings = {(0,0):C.GC_108,(0,1):C.GC_93})

V_96 = Vertex(name = 'V_96',
              particles = [ P.s__tilde__, P.c, P.phi__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS3, L.FFS4 ],
              couplings = {(0,0):C.GC_76,(0,1):C.GC_94})

V_97 = Vertex(name = 'V_97',
              particles = [ P.s__tilde__, P.t, P.phi__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS3, L.FFS4 ],
              couplings = {(0,0):C.GC_99,(0,1):C.GC_95})

V_98 = Vertex(name = 'V_98',
              particles = [ P.b__tilde__, P.u, P.phi__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS3, L.FFS4 ],
              couplings = {(0,0):C.GC_109,(0,1):C.GC_70})

V_99 = Vertex(name = 'V_99',
              particles = [ P.b__tilde__, P.c, P.phi__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS3, L.FFS4 ],
              couplings = {(0,0):C.GC_77,(0,1):C.GC_71})

V_100 = Vertex(name = 'V_100',
               particles = [ P.b__tilde__, P.t, P.phi__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS4 ],
               couplings = {(0,0):C.GC_100,(0,1):C.GC_72})

V_101 = Vertex(name = 'V_101',
               particles = [ P.d__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_25})

V_102 = Vertex(name = 'V_102',
               particles = [ P.d__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_28})

V_103 = Vertex(name = 'V_103',
               particles = [ P.d__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_31})

V_104 = Vertex(name = 'V_104',
               particles = [ P.s__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_26})

V_105 = Vertex(name = 'V_105',
               particles = [ P.s__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_29})

V_106 = Vertex(name = 'V_106',
               particles = [ P.s__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_32})

V_107 = Vertex(name = 'V_107',
               particles = [ P.b__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_27})

V_108 = Vertex(name = 'V_108',
               particles = [ P.b__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_30})

V_109 = Vertex(name = 'V_109',
               particles = [ P.b__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_33})

V_110 = Vertex(name = 'V_110',
               particles = [ P.u__tilde__, P.d, P.phi__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS4 ],
               couplings = {(0,0):C.GC_111,(0,1):C.GC_112})

V_111 = Vertex(name = 'V_111',
               particles = [ P.c__tilde__, P.d, P.phi__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS4 ],
               couplings = {(0,0):C.GC_121,(0,1):C.GC_120})

V_112 = Vertex(name = 'V_112',
               particles = [ P.t__tilde__, P.d, P.phi__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS4 ],
               couplings = {(0,0):C.GC_129,(0,1):C.GC_130})

V_113 = Vertex(name = 'V_113',
               particles = [ P.u__tilde__, P.s, P.phi__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS4 ],
               couplings = {(0,0):C.GC_114,(0,1):C.GC_115})

V_114 = Vertex(name = 'V_114',
               particles = [ P.c__tilde__, P.s, P.phi__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS4 ],
               couplings = {(0,0):C.GC_124,(0,1):C.GC_123})

V_115 = Vertex(name = 'V_115',
               particles = [ P.t__tilde__, P.s, P.phi__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS4 ],
               couplings = {(0,0):C.GC_132,(0,1):C.GC_133})

V_116 = Vertex(name = 'V_116',
               particles = [ P.u__tilde__, P.b, P.phi__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS4 ],
               couplings = {(0,0):C.GC_117,(0,1):C.GC_118})

V_117 = Vertex(name = 'V_117',
               particles = [ P.c__tilde__, P.b, P.phi__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS4 ],
               couplings = {(0,0):C.GC_126,(0,1):C.GC_127})

V_118 = Vertex(name = 'V_118',
               particles = [ P.t__tilde__, P.b, P.phi__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS4 ],
               couplings = {(0,0):C.GC_135,(0,1):C.GC_136})

V_119 = Vertex(name = 'V_119',
               particles = [ P.u__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_110})

V_120 = Vertex(name = 'V_120',
               particles = [ P.c__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_119})

V_121 = Vertex(name = 'V_121',
               particles = [ P.t__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_128})

V_122 = Vertex(name = 'V_122',
               particles = [ P.u__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_113})

V_123 = Vertex(name = 'V_123',
               particles = [ P.c__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_122})

V_124 = Vertex(name = 'V_124',
               particles = [ P.t__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_131})

V_125 = Vertex(name = 'V_125',
               particles = [ P.u__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_116})

V_126 = Vertex(name = 'V_126',
               particles = [ P.c__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_125})

V_127 = Vertex(name = 'V_127',
               particles = [ P.t__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_134})

V_128 = Vertex(name = 'V_128',
               particles = [ P.u__tilde__, P.u, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_10})

V_129 = Vertex(name = 'V_129',
               particles = [ P.c__tilde__, P.c, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_10})

V_130 = Vertex(name = 'V_130',
               particles = [ P.t__tilde__, P.t, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_10})

V_131 = Vertex(name = 'V_131',
               particles = [ P.e__plus__, P.e__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_85})

V_132 = Vertex(name = 'V_132',
               particles = [ P.m__plus__, P.m__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_89})

V_133 = Vertex(name = 'V_133',
               particles = [ P.tt__plus__, P.tt__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_103})

V_134 = Vertex(name = 'V_134',
               particles = [ P.u__tilde__, P.u, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_106})

V_135 = Vertex(name = 'V_135',
               particles = [ P.c__tilde__, P.c, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_74})

V_136 = Vertex(name = 'V_136',
               particles = [ P.t__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_97})

V_137 = Vertex(name = 'V_137',
               particles = [ P.e__plus__, P.e__minus__, P.phi0 ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_86})

V_138 = Vertex(name = 'V_138',
               particles = [ P.m__plus__, P.m__minus__, P.phi0 ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_90})

V_139 = Vertex(name = 'V_139',
               particles = [ P.tt__plus__, P.tt__minus__, P.phi0 ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_104})

V_140 = Vertex(name = 'V_140',
               particles = [ P.e__plus__, P.e__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV4 ],
               couplings = {(0,0):C.GC_34,(0,1):C.GC_48})

V_141 = Vertex(name = 'V_141',
               particles = [ P.m__plus__, P.m__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV4 ],
               couplings = {(0,0):C.GC_34,(0,1):C.GC_48})

V_142 = Vertex(name = 'V_142',
               particles = [ P.tt__plus__, P.tt__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV4 ],
               couplings = {(0,0):C.GC_34,(0,1):C.GC_48})

V_143 = Vertex(name = 'V_143',
               particles = [ P.e__plus__, P.ve, P.phi__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS4 ],
               couplings = {(0,0):C.GC_83})

V_144 = Vertex(name = 'V_144',
               particles = [ P.m__plus__, P.vm, P.phi__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS4 ],
               couplings = {(0,0):C.GC_87})

V_145 = Vertex(name = 'V_145',
               particles = [ P.tt__plus__, P.vt, P.phi__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS4 ],
               couplings = {(0,0):C.GC_101})

V_146 = Vertex(name = 'V_146',
               particles = [ P.e__plus__, P.ve, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_24})

V_147 = Vertex(name = 'V_147',
               particles = [ P.m__plus__, P.vm, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_24})

V_148 = Vertex(name = 'V_148',
               particles = [ P.tt__plus__, P.vt, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_24})

V_149 = Vertex(name = 'V_149',
               particles = [ P.ve__tilde__, P.e__minus__, P.phi__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS3 ],
               couplings = {(0,0):C.GC_84})

V_150 = Vertex(name = 'V_150',
               particles = [ P.vm__tilde__, P.m__minus__, P.phi__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS3 ],
               couplings = {(0,0):C.GC_88})

V_151 = Vertex(name = 'V_151',
               particles = [ P.vt__tilde__, P.tt__minus__, P.phi__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS3 ],
               couplings = {(0,0):C.GC_102})

V_152 = Vertex(name = 'V_152',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_24})

V_153 = Vertex(name = 'V_153',
               particles = [ P.vm__tilde__, P.m__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_24})

V_154 = Vertex(name = 'V_154',
               particles = [ P.vt__tilde__, P.tt__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_24})

V_155 = Vertex(name = 'V_155',
               particles = [ P.u__tilde__, P.u, P.phi0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_105})

V_156 = Vertex(name = 'V_156',
               particles = [ P.c__tilde__, P.c, P.phi0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_73})

V_157 = Vertex(name = 'V_157',
               particles = [ P.t__tilde__, P.t, P.phi0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_96})

V_158 = Vertex(name = 'V_158',
               particles = [ P.u__tilde__, P.u, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV5 ],
               couplings = {(0,0):C.GC_35,(0,1):C.GC_47})

V_159 = Vertex(name = 'V_159',
               particles = [ P.c__tilde__, P.c, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV5 ],
               couplings = {(0,0):C.GC_35,(0,1):C.GC_47})

V_160 = Vertex(name = 'V_160',
               particles = [ P.t__tilde__, P.t, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV5 ],
               couplings = {(0,0):C.GC_35,(0,1):C.GC_47})

V_161 = Vertex(name = 'V_161',
               particles = [ P.ve__tilde__, P.ve, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_53})

V_162 = Vertex(name = 'V_162',
               particles = [ P.vm__tilde__, P.vm, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_53})

V_163 = Vertex(name = 'V_163',
               particles = [ P.vt__tilde__, P.vt, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_53})

