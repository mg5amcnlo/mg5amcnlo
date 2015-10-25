# This file was automatically created by FeynRules 1.7.51
# Mathematica version: 8.0 for Linux x86 (64-bit) (February 23, 2011)
# Date: Thu 2 Aug 2012 10:15:23


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.FB__tilde__, P.FB, P.FA__tilde__, P.FB ],
             color = [ '1' ],
             lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF3, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
             couplings = {(0,11):C.GC_118,(0,12):C.GC_148,(0,10):C.GC_150,(0,0):C.GC_27,(0,9):C.GC_27,(0,3):C.GC_128,(0,6):C.GC_80,(0,4):C.GC_128,(0,5):C.GC_91,(0,7):C.GC_80,(0,8):C.GC_91,(0,13):C.GC_27,(0,14):C.GC_59,(0,15):C.GC_27,(0,16):C.GC_59,(0,1):C.GC_59,(0,2):C.GC_59})

V_2 = Vertex(name = 'V_2',
             particles = [ P.FD__tilde__, P.FC, P.phi ],
             color = [ '1' ],
             lorentz = [ L.FFS2, L.FFS3 ],
             couplings = {(0,0):C.GC_3,(0,1):C.GC_9})

V_3 = Vertex(name = 'V_3',
             particles = [ P.FA__tilde__, P.FA, P.FD__tilde__, P.FC ],
             color = [ '1' ],
             lorentz = [ L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF5, L.FFFF6 ],
             couplings = {(0,6):C.GC_109,(0,5):C.GC_44,(0,2):C.GC_70,(0,3):C.GC_113,(0,4):C.GC_92,(0,7):C.GC_44,(0,0):C.GC_49,(0,1):C.GC_49})

V_4 = Vertex(name = 'V_4',
             particles = [ P.FB__tilde__, P.FA, P.FD__tilde__, P.FC ],
             color = [ '1' ],
             lorentz = [ L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF5, L.FFFF6 ],
             couplings = {(0,6):C.GC_119,(0,5):C.GC_54,(0,2):C.GC_81,(0,3):C.GC_129,(0,4):C.GC_100,(0,7):C.GC_21,(0,0):C.GC_64,(0,1):C.GC_37})

V_5 = Vertex(name = 'V_5',
             particles = [ P.FA__tilde__, P.FB, P.FD__tilde__, P.FC ],
             color = [ '1' ],
             lorentz = [ L.FFFF10, L.FFFF11, L.FFFF12, L.FFFF13, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF5, L.FFFF6 ],
             couplings = {(0,7):C.GC_122,(0,6):C.GC_18,(0,2):C.GC_149,(0,4):C.GC_132,(0,5):C.GC_102,(0,3):C.GC_84,(0,8):C.GC_51,(0,0):C.GC_35,(0,1):C.GC_61})

V_6 = Vertex(name = 'V_6',
             particles = [ P.FB__tilde__, P.FB, P.FD__tilde__, P.FC ],
             color = [ '1' ],
             lorentz = [ L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF5, L.FFFF6 ],
             couplings = {(0,6):C.GC_72,(0,5):C.GC_26,(0,2):C.GC_88,(0,3):C.GC_94,(0,4):C.GC_104,(0,7):C.GC_26,(0,0):C.GC_39,(0,1):C.GC_39})

V_7 = Vertex(name = 'V_7',
             particles = [ P.FC__tilde__, P.FD, P.phi ],
             color = [ '1' ],
             lorentz = [ L.FFS2, L.FFS3 ],
             couplings = {(0,0):C.GC_10,(0,1):C.GC_4})

V_8 = Vertex(name = 'V_8',
             particles = [ P.FA__tilde__, P.FA, P.FC__tilde__, P.FD ],
             color = [ '1' ],
             lorentz = [ L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF5, L.FFFF6 ],
             couplings = {(0,6):C.GC_110,(0,5):C.GC_48,(0,2):C.GC_71,(0,3):C.GC_114,(0,4):C.GC_93,(0,7):C.GC_48,(0,0):C.GC_43,(0,1):C.GC_43})

V_9 = Vertex(name = 'V_9',
             particles = [ P.FB__tilde__, P.FA, P.FC__tilde__, P.FD ],
             color = [ '1' ],
             lorentz = [ L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF5, L.FFFF6 ],
             couplings = {(0,6):C.GC_121,(0,5):C.GC_62,(0,2):C.GC_83,(0,3):C.GC_131,(0,4):C.GC_103,(0,7):C.GC_34,(0,0):C.GC_52,(0,1):C.GC_19})

V_10 = Vertex(name = 'V_10',
              particles = [ P.FA__tilde__, P.FB, P.FC__tilde__, P.FD ],
              color = [ '1' ],
              lorentz = [ L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF5, L.FFFF6 ],
              couplings = {(0,6):C.GC_120,(0,5):C.GC_36,(0,2):C.GC_82,(0,3):C.GC_130,(0,4):C.GC_101,(0,7):C.GC_63,(0,0):C.GC_20,(0,1):C.GC_53})

V_11 = Vertex(name = 'V_11',
              particles = [ P.FB__tilde__, P.FB, P.FC__tilde__, P.FD ],
              color = [ '1' ],
              lorentz = [ L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF5, L.FFFF6 ],
              couplings = {(0,6):C.GC_73,(0,5):C.GC_38,(0,2):C.GC_89,(0,3):C.GC_95,(0,4):C.GC_105,(0,7):C.GC_38,(0,0):C.GC_25,(0,1):C.GC_25})

V_12 = Vertex(name = 'V_12',
              particles = [ P.FD__tilde__, P.FC, P.FC__tilde__, P.FD ],
              color = [ '1' ],
              lorentz = [ L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF5, L.FFFF6 ],
              couplings = {(0,6):C.GC_76,(0,5):C.GC_30,(0,2):C.GC_98,(0,3):C.GC_99,(0,4):C.GC_108,(0,7):C.GC_40,(0,0):C.GC_15,(0,1):C.GC_31})

V_13 = Vertex(name = 'V_13',
              particles = [ P.FA__tilde__, P.FA, P.FA__tilde__, P.FA ],
              color = [ '1' ],
              lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
              couplings = {(0,10):C.GC_147,(0,11):C.GC_147,(0,0):C.GC_50,(0,9):C.GC_50,(0,3):C.GC_144,(0,6):C.GC_144,(0,4):C.GC_144,(0,5):C.GC_138,(0,7):C.GC_144,(0,8):C.GC_138,(0,12):C.GC_50,(0,13):C.GC_50,(0,14):C.GC_50,(0,15):C.GC_50,(0,1):C.GC_50,(0,2):C.GC_50})

V_14 = Vertex(name = 'V_14',
              particles = [ P.FB__tilde__, P.FA, P.FB__tilde__, P.FA ],
              color = [ '1' ],
              lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
              couplings = {(0,10):C.GC_135,(0,11):C.GC_135,(0,0):C.GC_69,(0,9):C.GC_69,(0,3):C.GC_123,(0,6):C.GC_123,(0,4):C.GC_123,(0,5):C.GC_85,(0,7):C.GC_123,(0,8):C.GC_85,(0,12):C.GC_58,(0,13):C.GC_58,(0,14):C.GC_58,(0,15):C.GC_24,(0,1):C.GC_58,(0,2):C.GC_24})

V_15 = Vertex(name = 'V_15',
              particles = [ P.FA__tilde__, P.FB, P.FA__tilde__, P.FB ],
              color = [ '1' ],
              lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
              couplings = {(0,10):C.GC_136,(0,11):C.GC_136,(0,0):C.GC_23,(0,9):C.GC_23,(0,3):C.GC_124,(0,6):C.GC_124,(0,4):C.GC_124,(0,5):C.GC_86,(0,7):C.GC_124,(0,8):C.GC_86,(0,12):C.GC_57,(0,13):C.GC_57,(0,14):C.GC_57,(0,15):C.GC_68,(0,1):C.GC_57,(0,2):C.GC_68})

V_16 = Vertex(name = 'V_16',
              particles = [ P.FB__tilde__, P.FB, P.FB__tilde__, P.FB ],
              color = [ '1' ],
              lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
              couplings = {(0,10):C.GC_140,(0,11):C.GC_140,(0,0):C.GC_29,(0,9):C.GC_29,(0,3):C.GC_142,(0,6):C.GC_142,(0,4):C.GC_142,(0,5):C.GC_143,(0,7):C.GC_142,(0,8):C.GC_143,(0,12):C.GC_29,(0,13):C.GC_29,(0,14):C.GC_29,(0,15):C.GC_29,(0,1):C.GC_29,(0,2):C.GC_29})

V_17 = Vertex(name = 'V_17',
              particles = [ P.FD__tilde__, P.FC, P.FD__tilde__, P.FC ],
              color = [ '1' ],
              lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
              couplings = {(0,10):C.GC_74,(0,11):C.GC_74,(0,0):C.GC_17,(0,9):C.GC_17,(0,3):C.GC_96,(0,6):C.GC_96,(0,4):C.GC_96,(0,5):C.GC_106,(0,7):C.GC_96,(0,8):C.GC_106,(0,12):C.GC_33,(0,13):C.GC_33,(0,14):C.GC_33,(0,15):C.GC_42,(0,1):C.GC_33,(0,2):C.GC_42})

V_18 = Vertex(name = 'V_18',
              particles = [ P.FC__tilde__, P.FD, P.FC__tilde__, P.FD ],
              color = [ '1' ],
              lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
              couplings = {(0,10):C.GC_75,(0,11):C.GC_75,(0,0):C.GC_41,(0,9):C.GC_41,(0,3):C.GC_97,(0,6):C.GC_97,(0,4):C.GC_97,(0,5):C.GC_107,(0,7):C.GC_97,(0,8):C.GC_107,(0,12):C.GC_32,(0,13):C.GC_32,(0,14):C.GC_32,(0,15):C.GC_16,(0,1):C.GC_32,(0,2):C.GC_16})

V_19 = Vertex(name = 'V_19',
              particles = [ P.FA__tilde__, P.FA, P.phi ],
              color = [ '1' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_11})

V_20 = Vertex(name = 'V_20',
              particles = [ P.FB__tilde__, P.FA, P.phi ],
              color = [ '1' ],
              lorentz = [ L.FFS2, L.FFS3 ],
              couplings = {(0,0):C.GC_13,(0,1):C.GC_5})

V_21 = Vertex(name = 'V_21',
              particles = [ P.FB__tilde__, P.FA, P.FA__tilde__, P.FA ],
              color = [ '1' ],
              lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
              couplings = {(0,10):C.GC_133,(0,11):C.GC_133,(0,0):C.GC_66,(0,9):C.GC_66,(0,3):C.GC_111,(0,6):C.GC_115,(0,4):C.GC_111,(0,5):C.GC_77,(0,7):C.GC_115,(0,8):C.GC_77,(0,12):C.GC_46,(0,13):C.GC_66,(0,14):C.GC_46,(0,15):C.GC_46,(0,1):C.GC_66,(0,2):C.GC_46})

V_22 = Vertex(name = 'V_22',
              particles = [ P.FA__tilde__, P.FB, P.phi ],
              color = [ '1' ],
              lorentz = [ L.FFS2, L.FFS3 ],
              couplings = {(0,0):C.GC_6,(0,1):C.GC_14})

V_23 = Vertex(name = 'V_23',
              particles = [ P.FA__tilde__, P.FA, P.FA__tilde__, P.FB ],
              color = [ '1' ],
              lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
              couplings = {(0,10):C.GC_134,(0,11):C.GC_134,(0,0):C.GC_45,(0,9):C.GC_45,(0,3):C.GC_116,(0,6):C.GC_116,(0,4):C.GC_112,(0,5):C.GC_78,(0,7):C.GC_112,(0,8):C.GC_78,(0,12):C.GC_45,(0,13):C.GC_45,(0,14):C.GC_65,(0,15):C.GC_65,(0,1):C.GC_65,(0,2):C.GC_65})

V_24 = Vertex(name = 'V_24',
              particles = [ P.FB__tilde__, P.FA, P.FA__tilde__, P.FB ],
              color = [ '1' ],
              lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
              couplings = {(0,10):C.GC_145,(0,11):C.GC_137,(0,0):C.GC_47,(0,9):C.GC_56,(0,3):C.GC_126,(0,6):C.GC_139,(0,4):C.GC_146,(0,5):C.GC_141,(0,7):C.GC_125,(0,8):C.GC_87,(0,12):C.GC_22,(0,13):C.GC_47,(0,14):C.GC_47,(0,15):C.GC_47,(0,1):C.GC_67,(0,2):C.GC_55})

V_25 = Vertex(name = 'V_25',
              particles = [ P.FB__tilde__, P.FB, P.phi ],
              color = [ '1' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_7})

V_26 = Vertex(name = 'V_26',
              particles = [ P.FB__tilde__, P.FA, P.FB__tilde__, P.FB ],
              color = [ '1' ],
              lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF17, L.FFFF18, L.FFFF2, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
              couplings = {(0,10):C.GC_117,(0,11):C.GC_117,(0,0):C.GC_60,(0,9):C.GC_60,(0,3):C.GC_79,(0,6):C.GC_79,(0,4):C.GC_127,(0,5):C.GC_90,(0,7):C.GC_127,(0,8):C.GC_90,(0,12):C.GC_28,(0,13):C.GC_28,(0,14):C.GC_60,(0,15):C.GC_28,(0,1):C.GC_60,(0,2):C.GC_28})

V_27 = Vertex(name = 'V_27',
              particles = [ P.FA__tilde__, P.FA, P.V ],
              color = [ '1' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,0):C.GC_12,(0,1):C.GC_1})

V_28 = Vertex(name = 'V_28',
              particles = [ P.FB__tilde__, P.FA, P.V ],
              color = [ '1' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,0):C.GC_13,(0,1):C.GC_5})

V_29 = Vertex(name = 'V_29',
              particles = [ P.FA__tilde__, P.FB, P.V ],
              color = [ '1' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,0):C.GC_14,(0,1):C.GC_6})

V_30 = Vertex(name = 'V_30',
              particles = [ P.FB__tilde__, P.FB, P.V ],
              color = [ '1' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,0):C.GC_2,(0,1):C.GC_8})

V_31 = Vertex(name = 'V_31',
              particles = [ P.FD__tilde__, P.FC, P.V ],
              color = [ '1' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,0):C.GC_3,(0,1):C.GC_9})

V_32 = Vertex(name = 'V_32',
              particles = [ P.FC__tilde__, P.FD, P.V ],
              color = [ '1' ],
              lorentz = [ L.FFV1, L.FFV2 ],
              couplings = {(0,0):C.GC_4,(0,1):C.GC_10})

