# This file was automatically created by FeynRules $Revision: 999 $
# Mathematica version: 7.0 for Linux x86 (64-bit) (February 18, 2009)
# Date: Mon 30 Jan 2012 19:57:04


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L
# ===================================================
# QCD vertices
# ===================================================


V_36 = Vertex(name = 'V_36',
              particles = [ P.G, P.G, P.G ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVV1 ],
              couplings = {(0,0):C.GC_10})

V_37 = Vertex(name = 'V_37',
              particles = [ P.G, P.G, P.G, P.G ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVV1, L.VVVV3, L.VVVV4 ],
              couplings = {(1,1):C.GC_12,(0,0):C.GC_12,(2,2):C.GC_12})

V_74 = Vertex(name = 'V_74',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_11})

V_75 = Vertex(name = 'V_75',
              particles = [ P.s__tilde__, P.s, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_11})

V_76 = Vertex(name = 'V_76',
              particles = [ P.b__tilde__, P.b, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_11})

V_143 = Vertex(name = 'V_143',
               particles = [ P.u__tilde__, P.u, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_11})

V_144 = Vertex(name = 'V_144',
               particles = [ P.c__tilde__, P.c, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_11})

# ===================================================
# QED vertices
# ===================================================

# BEGIN SSSS

V_1 = Vertex(name = 'V_1',
             particles = [ P.G0, P.G0, P.G0, P.G0 ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_51})

V_2 = Vertex(name = 'V_2',
             particles = [ P.G0, P.G0, P.G__minus__, P.G__plus__ ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_49})

V_3 = Vertex(name = 'V_3',
             particles = [ P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_50})

V_4 = Vertex(name = 'V_4',
             particles = [ P.G0, P.G0, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_49})

V_5 = Vertex(name = 'V_5',
             particles = [ P.G__minus__, P.G__plus__, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_49})

V_6 = Vertex(name = 'V_6',
             particles = [ P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_51})
# END SSSS

# BEGIN SSS
"""
V_7 = Vertex(name = 'V_7',
             particles = [ P.G0, P.G0, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_86})


"""
V_VV_26 = Vertex(name = 'V_VV_26',
              particles = [ P.G0, P.G0, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS6, L.SSS7, L.SSS9 ],
              couplings = {(0,0):C.GC_SM_321,(0,1):C.GC_VV_334,(0,2):C.GC_VV_328})

#V_VV_27 = Vertex(name = 'V_VV_27',
#              particles = [ P.G0, P.G0, P.H ],
#              color = [ '1' ],
#              lorentz = [ L.SSS6 ],
#              couplings = {(0,0):C.GC_VV_430})

"""
V_8 = Vertex(name = 'V_8',
             particles = [ P.G__minus__, P.G__plus__, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_86})


"""
V_VV_28 = Vertex(name = 'V_VV_28',
              particles = [ P.G__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS6, L.SSS9 ],
              couplings = {(0,0):C.GC_SM_321,(0,1):C.GC_VV_421})


#V_VV_29 = Vertex(name = 'V_VV_29',
#              particles = [ P.G__minus__, P.G__plus__, P.H ],
#              color = [ '1' ],
#              lorentz = [ L.SSS6 ],
#              couplings = {(0,0):C.GC_VV_429})



V_9 = Vertex(name = 'V_9',
             particles = [ P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_87})

# END SSS

# BEGIN VVSS

V_10 = Vertex(name = 'V_10',
              particles = [ P.A, P.A, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_6})

V_39 = Vertex(name = 'V_39',
              particles = [ P.A, P.W__minus__, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_73})

V_40 = Vertex(name = 'V_40',
              particles = [ P.A, P.W__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_72})

V_47 = Vertex(name = 'V_47',
              particles = [ P.A, P.W__plus__, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_73})

V_48 = Vertex(name = 'V_48',
              particles = [ P.A, P.W__plus__, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_74})

V_52 = Vertex(name = 'V_52',
              particles = [ P.W__minus__, P.W__plus__, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_52})

V_53 = Vertex(name = 'V_53',
              particles = [ P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_52})

V_54 = Vertex(name = 'V_54',
              particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_52})

V_58 = Vertex(name = 'V_58',
              particles = [ P.A, P.Z, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_81})

V_61 = Vertex(name = 'V_61',
              particles = [ P.W__minus__, P.Z, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_8})

V_62 = Vertex(name = 'V_62',
              particles = [ P.W__minus__, P.Z, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_9})

V_64 = Vertex(name = 'V_64',
              particles = [ P.W__plus__, P.Z, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_8})

V_65 = Vertex(name = 'V_65',
              particles = [ P.W__plus__, P.Z, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_7})

V_67 = Vertex(name = 'V_67',
              particles = [ P.Z, P.Z, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_83})

V_68 = Vertex(name = 'V_68',
              particles = [ P.Z, P.Z, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_82})

V_69 = Vertex(name = 'V_69',
              particles = [ P.Z, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1 ],
              couplings = {(0,0):C.GC_83})

# END VVSS

# BEGIN VSS

"""

V_11 = Vertex(name = 'V_11',
              particles = [ P.A, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_3})


"""
V_VV_45 = Vertex(name = 'V_VV_45',
              particles = [ P.A, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS5 ],
              couplings = {(0,0):C.GC_SM_3})

V_VV_46 = Vertex(name = 'V_VV_46',
              particles = [ P.A, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS5 ],
              couplings = {(0,0):C.GC_VV_493})


"""
V_42 = Vertex(name = 'V_42',
              particles = [ P.W__minus__, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_57})

"""
V_VV_196 = Vertex(name = 'V_VV_196',
               particles = [ P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSS5, L.VSS6 ],
               couplings = {(0,0):C.GC_SM_194,(0,1):C.GC_VV_416})


#V_VV_197 = Vertex(name = 'V_VV_197',
#               particles = [ P.W__minus__, P.G0, P.G__plus__ ],
#              color = [ '1' ],
#               lorentz = [ L.VSS5 ],
#               couplings = {(0,0):C.GC_VV_477})


"""
V_43 = Vertex(name = 'V_43',
              particles = [ P.W__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_55})

"""

V_VV_198 = Vertex(name = 'V_VV_198',
               particles = [ P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSS5 ],
               couplings = {(0,0):C.GC_SM_192})

#V_VV_199 = Vertex(name = 'V_VV_199',
#               particles = [ P.W__minus__, P.G__plus__, P.H ],
#               color = [ '1' ],
#               lorentz = [ L.VSS5 ],
#               couplings = {(0,0):C.GC_VV_481})


"""
V_50 = Vertex(name = 'V_50',
              particles = [ P.W__plus__, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_56})

"""
V_VV_232 = Vertex(name = 'V_VV_232',
               particles = [ P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VSS5, L.VSS6 ],
               couplings = {(0,0):C.GC_SM_193,(0,1):C.GC_VV_415})

#V_VV_233 = Vertex(name = 'V_VV_233',
#               particles = [ P.W__plus__, P.G0, P.G__minus__ ],
#               color = [ '1' ],
#               lorentz = [ L.VSS5 ],
#               couplings = {(0,0):C.GC_VV_476})



"""

V_51 = Vertex(name = 'V_51',
              particles = [ P.W__plus__, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_55})


"""
V_VV_234 = Vertex(name = 'V_VV_234',
               particles = [ P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSS5 ],
               couplings = {(0,0):C.GC_SM_192})


#V_VV_235 = Vertex(name = 'V_VV_235',
#               particles = [ P.W__plus__, P.G__minus__, P.H ],
#               color = [ '1' ],
#               lorentz = [ L.VSS5 ],
#               couplings = {(0,0):C.GC_VV_481})

"""
V_59 = Vertex(name = 'V_59',
              particles = [ P.Z, P.G0, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_78})


"""
V_VV_279 = Vertex(name = 'V_VV_279',
               particles = [ P.Z, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSS4, L.VSS5 ],
               couplings = {(0,1):C.GC_SM_275_m,(0,0):C.GC_VV_511_m})

#V_VV_280 = Vertex(name = 'V_VV_280',
#               particles = [ P.Z, P.G0, P.H ],
#               color = [ '1' ],
#               lorentz = [ L.VSS5 ],
#               couplings = {(0,0):C.GC_VV_509_m})



"""
V_60 = Vertex(name = 'V_60',
              particles = [ P.Z, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1 ],
              couplings = {(0,0):C.GC_79})


"""
V_VV_281 = Vertex(name = 'V_VV_281',
               particles = [ P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSS5 ],
               couplings = {(0,0):C.GC_SM_274_m})

V_VV_282 = Vertex(name = 'V_VV_282',
               particles = [ P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSS5 ],
               couplings = {(0,0):C.GC_VV_510_m})



# END VSS


# BEGIN VVV


"""
V_38 = Vertex(name = 'V_38',
              particles = [ P.A, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVV1 ],
              couplings = {(0,0):C.GC_4})



"""
V_VV_63 = Vertex(name = 'V_VV_63',
              particles = [ P.A, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVV10, L.VVV8, L.VVV9 ],
              couplings = {(0,0):C.GC_VV_258,(0,1):C.GC_SM_4,(0,2):C.GC_VV_417})

#V_VV_64 = Vertex(name = 'V_VV_64',
#              particles = [ P.A, P.W__minus__, P.W__plus__ ],
#              color = [ '1' ],
#              lorentz = [ L.VVV8 ],
#              couplings = {(0,0):C.GC_VV_469})



"""
V_45 = Vertex(name = 'V_45',
              particles = [ P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVV1 ],
              couplings = {(0,0):C.GC_71})



"""
V_VV_79 = Vertex(name = 'V_VV_79',
              particles = [ P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVV10, L.VVV7, L.VVV8 ],
              couplings = {(0,0):C.GC_VV_124_m,(0,2):C.GC_SM_198_m,(0,1):C.GC_VV_412_m})



#V_VV_80 = Vertex(name = 'V_VV_80',
#              particles = [ P.W__minus__, P.W__plus__, P.Z ],
#              color = [ '1' ],
#              lorentz = [ L.VVV8 ],
#              couplings = {(0,0):C.GC_VV_484_m})


# END VVV

# BEGIN VVS
"""
V_41 = Vertex(name = 'V_41',
              particles = [ P.A, P.W__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_92})

"""
V_VV_61 = Vertex(name = 'V_VV_61',
              particles = [ P.A, P.W__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11, L.VVS12 ],
              couplings = {(0,1):C.GC_VV_342,(0,0):C.GC_SM_371})

V_VV_62 = Vertex(name = 'V_VV_62',
              particles = [ P.A, P.W__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11 ],
              couplings = {(0,0):C.GC_VV_530})

"""
V_49 = Vertex(name = 'V_49',
              particles = [ P.A, P.W__plus__, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_93})

"""
V_VV_69 = Vertex(name = 'V_VV_69',
              particles = [ P.A, P.W__plus__, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11, L.VVS12 ],
              couplings = {(0,1):C.GC_VV_343,(0,0):C.GC_SM_372})

V_VV_70 = Vertex(name = 'V_VV_70',
              particles = [ P.A, P.W__plus__, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11 ],
              couplings = {(0,0):C.GC_VV_535})




"""
V_55 = Vertex(name = 'V_55',
              particles = [ P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_90})


"""
V_VV_77 = Vertex(name = 'V_VV_77',
              particles = [ P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS11, L.VVS12 ],
              couplings = {(0,1):C.GC_VV_339,(0,0):C.GC_SM_363})

#V_VV_78 = Vertex(name = 'V_VV_78',
#              particles = [ P.W__minus__, P.W__plus__, P.H ],
#              color = [ '1' ],
#              lorentz = [ L.VVS11 ],
#              couplings = {(0,0):C.GC_VV_528})



"""
V_63 = Vertex(name = 'V_63',
              particles = [ P.W__minus__, P.Z, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_85})


"""
V_VV_90 = Vertex(name = 'V_VV_90',
              particles = [ P.W__minus__, P.Z, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11, L.VVS12 ],
              couplings = {(0,1):C.GC_VV_401_m,(0,0):C.GC_SM_323_m})

V_VV_91 = Vertex(name = 'V_VV_91',
              particles = [ P.W__minus__, P.Z, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11 ],
              couplings = {(0,0):C.GC_VV_541_m})



"""

V_66 = Vertex(name = 'V_66',
              particles = [ P.W__plus__, P.Z, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_84})

"""
V_VV_96 = Vertex(name = 'V_VV_96',
              particles = [ P.W__plus__, P.Z, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11, L.VVS12 ],
              couplings = {(0,1):C.GC_VV_402_m,(0,0):C.GC_SM_324_m})

V_VV_97 = Vertex(name = 'V_VV_97',
              particles = [ P.W__plus__, P.Z, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS11 ],
              couplings = {(0,0):C.GC_VV_536_m})

"""
V_70 = Vertex(name = 'V_70',
              particles = [ P.Z, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_99})

"""
V_VV_104 = Vertex(name = 'V_VV_104',
               particles = [ P.Z, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVS11, L.VVS12 ],
               couplings = {(0,1):C.GC_VV_460,(0,0):C.GC_SM_459})

#V_VV_105 = Vertex(name = 'V_VV_105',
#               particles = [ P.Z, P.Z, P.H ],
#               color = [ '1' ],
#               lorentz = [ L.VVS11 ],
#               couplings = {(0,0):C.GC_VV_545})


# END VVS

# BEGIN VVVV

V_44 = Vertex(name = 'V_44',
              particles = [ P.A, P.A, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV2 ],
              couplings = {(0,0):C.GC_5})

V_46 = Vertex(name = 'V_46',
              particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV2 ],
              couplings = {(0,0):C.GC_53})

V_56 = Vertex(name = 'V_56',
              particles = [ P.A, P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV5 ],
              couplings = {(0,0):C.GC_75})

V_57 = Vertex(name = 'V_57',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV2 ],
              couplings = {(0,0):C.GC_54})

# END VVVV

# BEGIN FFV
"""
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
"""



V_VV_381 = Vertex(name = 'V_VV_381',
               particles = [ P.d__tilde__, P.d, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV83, L.FFV87 ],
               couplings = {(0,2):C.GC_SM_1,(0,1):C.GC_VV_466,(0,0):C.GC_VV_1155})

V_VV_382 = Vertex(name = 'V_VV_382',
               particles = [ P.d__tilde__, P.d, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_VV_489})


V_VV_383 = Vertex(name = 'V_VV_383',
               particles = [ P.s__tilde__, P.s, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV83, L.FFV87 ],
               couplings = {(0,2):C.GC_SM_1,(0,1):C.GC_VV_466,(0,0):C.GC_VV_1155})

V_VV_384 = Vertex(name = 'V_VV_384',
               particles = [ P.s__tilde__, P.s, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_VV_489})


V_VV_385 = Vertex(name = 'V_VV_385',
               particles = [ P.b__tilde__, P.b, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV83, L.FFV87 ],
               couplings = {(0,2):C.GC_SM_1,(0,1):C.GC_VV_466,(0,0):C.GC_VV_1152})

V_VV_386 = Vertex(name = 'V_VV_386',
               particles = [ P.b__tilde__, P.b, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_VV_489})


"""

V_83 = Vertex(name = 'V_83',
              particles = [ P.d__tilde__, P.d, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              couplings = {(0,0):C.GC_68,(0,1):C.GC_76})

V_84 = Vertex(name = 'V_84',
              particles = [ P.s__tilde__, P.s, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              couplings = {(0,0):C.GC_68,(0,1):C.GC_76})
              
V_85 = Vertex(name = 'V_85',
              particles = [ P.b__tilde__, P.b, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              couplings = {(0,0):C.GC_68,(0,1):C.GC_76})

"""



V_VV_411 = Vertex(name = 'V_VV_411',
               particles = [ P.d__tilde__, P.d, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV82, L.FFV83, L.FFV87 ],
               couplings = {(0,0):C.GC_SM_197_m,(0,2):C.GC_SM_253_m,(0,3):C.GC_VV_482_m,(0,1):C.GC_VV_1800_m})

V_VV_412 = Vertex(name = 'V_VV_412',
               particles = [ P.d__tilde__, P.d, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV83 ],
               couplings = {(0,0):C.GC_VV_1802_m,(0,1):C.GC_VV_505_m})



V_VV_413 = Vertex(name = 'V_VV_413',
               particles = [ P.s__tilde__, P.s, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV82, L.FFV83, L.FFV87 ],
               couplings = {(0,0):C.GC_SM_197_m,(0,2):C.GC_SM_253_m,(0,3):C.GC_VV_482_m,(0,1):C.GC_VV_1800_m})

V_VV_414 = Vertex(name = 'V_VV_414',
               particles = [ P.s__tilde__, P.s, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV83 ],
               couplings = {(0,0):C.GC_VV_1802_m,(0,1):C.GC_VV_505_m})


V_VV_415 = Vertex(name = 'V_VV_415',
               particles = [ P.b__tilde__, P.b, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV82, L.FFV83, L.FFV87 ],
               couplings = {(0,0):C.GC_SM_197_m,(0,2):C.GC_SM_253_m,(0,3):C.GC_VV_482_m,(0,1):C.GC_VV_1800_m})

V_VV_416 = Vertex(name = 'V_VV_416',
               particles = [ P.b__tilde__, P.b, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV83 ],
               couplings = {(0,0):C.GC_VV_1799_m,(0,1):C.GC_VV_505_m})


"""

V_95 = Vertex(name = 'V_95',
              particles = [ P.u__tilde__, P.d, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_58})

V_99 = Vertex(name = 'V_99',
              particles = [ P.c__tilde__, P.s, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_58})

V_103 = Vertex(name = 'V_103',
               particles = [ P.t__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_58})

"""



V_VV_399 = Vertex(name = 'V_VV_399',
               particles = [ P.u__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_SM_195})

V_VV_400 = Vertex(name = 'V_VV_400',
               particles = [ P.u__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_VV_1991})


V_VV_401 = Vertex(name = 'V_VV_401',
               particles = [ P.c__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_SM_195})

V_VV_402 = Vertex(name = 'V_VV_402',
               particles = [ P.c__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_VV_1991})


V_VV_403 = Vertex(name = 'V_VV_403',
               particles = [ P.t__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV109, L.FFV70 ],
               couplings = {(0,0):C.GC_VV_340,(0,1):C.GC_SM_195})

V_VV_404 = Vertex(name = 'V_VV_404',
               particles = [ P.t__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_VV_1992})





V_104 = Vertex(name = 'V_104',
               particles = [ P.e__plus__, P.e__minus__, P.A ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_3})

V_105 = Vertex(name = 'V_105',
               particles = [ P.m__plus__, P.m__minus__, P.A ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_3})

V_106 = Vertex(name = 'V_106',
               particles = [ P.tt__plus__, P.tt__minus__, P.A ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_3})

V_113 = Vertex(name = 'V_113',
               particles = [ P.e__plus__, P.e__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV4 ],
               couplings = {(0,0):C.GC_68,(0,1):C.GC_77})

V_114 = Vertex(name = 'V_114',
               particles = [ P.m__plus__, P.m__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV4 ],
               couplings = {(0,0):C.GC_68,(0,1):C.GC_77})

V_115 = Vertex(name = 'V_115',
               particles = [ P.tt__plus__, P.tt__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV4 ],
               couplings = {(0,0):C.GC_68,(0,1):C.GC_77})

V_119 = Vertex(name = 'V_119',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_58})

V_120 = Vertex(name = 'V_120',
               particles = [ P.vm__tilde__, P.m__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_58})


V_121 = Vertex(name = 'V_121',
               particles = [ P.vt__tilde__, P.tt__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_58})





"""

V_131 = Vertex(name = 'V_131',
               particles = [ P.d__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_58})

V_135 = Vertex(name = 'V_135',
               particles = [ P.s__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_58})

V_139 = Vertex(name = 'V_139',
               particles = [ P.b__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_58})
"""


V_VV_393 = Vertex(name = 'V_VV_393',
               particles = [ P.d__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_SM_195})

V_VV_394 = Vertex(name = 'V_VV_394',
               particles = [ P.d__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_VV_1991})


V_VV_395 = Vertex(name = 'V_VV_395',
               particles = [ P.s__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_SM_195})

V_VV_396 = Vertex(name = 'V_VV_396',
               particles = [ P.s__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_VV_1991})



V_VV_397 = Vertex(name = 'V_VV_397',
               particles = [ P.b__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV80 ],
               couplings = {(0,1):C.GC_VV_340,(0,0):C.GC_SM_195})

V_VV_398 = Vertex(name = 'V_VV_398',
               particles = [ P.b__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70 ],
               couplings = {(0,0):C.GC_VV_1992})





"""

V_140 = Vertex(name = 'V_140',
               particles = [ P.u__tilde__, P.u, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2})

V_141 = Vertex(name = 'V_141',
               particles = [ P.c__tilde__, P.c, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2})

"""


V_VV_375 = Vertex(name = 'V_VV_375',
               particles = [ P.u__tilde__, P.u, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV87, L.FFV91 ],
               couplings = {(0,1):C.GC_SM_2,(0,2):C.GC_VV_466,(0,0):C.GC_VV_1154})

V_VV_376 = Vertex(name = 'V_VV_376',
               particles = [ P.u__tilde__, P.u, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_VV_490})


V_VV_377 = Vertex(name = 'V_VV_377',
               particles = [ P.c__tilde__, P.c, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV87, L.FFV91 ],
               couplings = {(0,1):C.GC_SM_2,(0,2):C.GC_VV_466,(0,0):C.GC_VV_1154})

V_VV_378 = Vertex(name = 'V_VV_378',
               particles = [ P.c__tilde__, P.c, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_VV_490})


"""
V_152 = Vertex(name = 'V_152',
               particles = [ P.u__tilde__, P.u, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV5 ],
               couplings = {(0,0):C.GC_69,(0,1):C.GC_76})

V_153 = Vertex(name = 'V_153',
               particles = [ P.c__tilde__, P.c, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV5 ],
               couplings = {(0,0):C.GC_69,(0,1):C.GC_76})
"""

V_VV_405 = Vertex(name = 'V_VV_405',
               particles = [ P.u__tilde__, P.u, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV82, L.FFV87, L.FFV91 ],
               couplings = {(0,0):C.GC_SM_196_m,(0,3):C.GC_SM_253_m,(0,2):C.GC_VV_483_m,(0,1):C.GC_VV_1804_m})

V_VV_406 = Vertex(name = 'V_VV_406',
               particles = [ P.u__tilde__, P.u, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV91 ],
               couplings = {(0,0):C.GC_VV_1801_m,(0,1):C.GC_VV_505_m})


V_VV_407 = Vertex(name = 'V_VV_407',
               particles = [ P.c__tilde__, P.c, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV82, L.FFV87, L.FFV91 ],
               couplings = {(0,0):C.GC_SM_196_m,(0,3):C.GC_SM_253_m,(0,2):C.GC_VV_483_m,(0,1):C.GC_VV_1804_m})

V_VV_408 = Vertex(name = 'V_VV_408',
               particles = [ P.c__tilde__, P.c, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV91 ],
               couplings = {(0,0):C.GC_VV_1801_m,(0,1):C.GC_VV_505_m})












V_158 = Vertex(name = 'V_158',
               particles = [ P.e__plus__, P.ve, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_58})

V_159 = Vertex(name = 'V_159',
               particles = [ P.m__plus__, P.vm, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_58})

V_160 = Vertex(name = 'V_160',
               particles = [ P.tt__plus__, P.vt, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_58})

V_161 = Vertex(name = 'V_161',
               particles = [ P.ve__tilde__, P.ve, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_80})
               
V_162 = Vertex(name = 'V_162',
               particles = [ P.vm__tilde__, P.vm, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_80})

V_163 = Vertex(name = 'V_163',
               particles = [ P.vt__tilde__, P.vt, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_80})

# END FFV

# BEGIN FFS

"""
V_94 = Vertex(name = 'V_94',
              particles = [ P.t__tilde__, P.b, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS3 ],
              couplings = {(0,1):C.GC_30})

V_130 = Vertex(name = 'V_130',
               particles = [ P.b__tilde__, P.t, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS3 ],
               couplings = {(0,0):C.GC_39})
"""

V_VV_187 = Vertex(name = 'V_VV_187',
               particles = [ P.t__tilde__, P.b, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS45, L.FFS52 ],
               couplings = {(0,1):C.GC_SM_548,(0,0):C.GC_VV_2079})


#V_VV_188 = Vertex(name = 'V_VV_188',
#               particles = [ P.t__tilde__, P.b, P.G__plus__ ],
#               color = [ 'Identity(1,2)' ],
#               lorentz = [ L.FFS52 ],
#               couplings = {(0,0):C.GC_VV_2084})


V_VV_167 = Vertex(name = 'V_VV_167',
               particles = [ P.b__tilde__, P.t, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS41, L.FFS45 ],
               couplings = {(0,0):C.GC_SM_549,(0,1):C.GC_VV_2079})

#V_VV_168 = Vertex(name = 'V_VV_168',
#               particles = [ P.b__tilde__, P.t, P.G__minus__ ],
#               color = [ 'Identity(1,2)' ],
#               lorentz = [ L.FFS41 ],
#               couplings = {(0,0):C.GC_VV_2085})



"""

V_148 = Vertex(name = 'V_148',
               particles = [ P.t__tilde__, P.t, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_117})

V_151 = Vertex(name = 'V_151',
               particles = [ P.t__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS4 ],
               couplings = {(0,0):C.GC_116})
"""

V_VV_169 = Vertex(name = 'V_VV_169',
               particles = [ P.t__tilde__, P.t, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS45, L.FFS51, L.FFS55 ],
               couplings = {(0,1):C.GC_SM_547,(0,0):C.GC_VV_1834,(0,2):C.GC_VV_1840})

#V_VV_170 = Vertex(name = 'V_VV_170',
#               particles = [ P.t__tilde__, P.t, P.G0 ],
#               color = [ 'Identity(1,2)' ],
#               lorentz = [ L.FFS51 ],
#               couplings = {(0,0):C.GC_VV_550})



V_VV_171 = Vertex(name = 'V_VV_171',
               particles = [ P.t__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS37 ],
               couplings = {(0,0):C.GC_VV_1842})

V_VV_172 = Vertex(name = 'V_VV_172',
               particles = [ P.t__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS37 ],
               couplings = {(0,0):C.GC_SM_546})



# END FFS

# ttbar


V_145 = Vertex(name = 'V_145',
               particles = [ P.t__tilde__, P.t, P.G ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_11})

"""
V_142 = Vertex(name = 'V_142',
               particles = [ P.t__tilde__, P.t, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2})
"""



V_VV_379 = Vertex(name = 'V_VV_379',
               particles = [ P.t__tilde__, P.t, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV110, L.FFV70, L.FFV87, L.FFV91 ],
               couplings = {(0,2):C.GC_SM_2,(0,3):C.GC_VV_466,(0,1):C.GC_VV_1150,(0,0):C.GC_VV_443})

V_VV_380 = Vertex(name = 'V_VV_380',
               particles = [ P.t__tilde__, P.t, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV87 ],
               couplings = {(0,0):C.GC_VV_1151})


"""
V_154 = Vertex(name = 'V_154',
               particles = [ P.t__tilde__, P.t, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV5 ],
               couplings = {(0,0):C.GC_69,(0,1):C.GC_76})

"""



V_VV_409 = Vertex(name = 'V_VV_409',
               particles = [ P.t__tilde__, P.t, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV110, L.FFV70, L.FFV82, L.FFV87, L.FFV91 ],
               couplings = {(0,1):C.GC_SM_196_m,(0,4):C.GC_SM_253_m,(0,3):C.GC_VV_1227_m,(0,2):C.GC_VV_1803_m,(0,0):C.GC_VV_341_m})

V_VV_410 = Vertex(name = 'V_VV_410',
               particles = [ P.t__tilde__, P.t, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV70, L.FFV91 ],
               couplings = {(0,0):C.GC_VV_1797_m,(0,1):C.GC_VV_505_m})







"SMEFT di-boson"

#V_VV_42 = Vertex(name = 'V_VV_42',
#              particles = [ P.A, P.G0, P.H ],
#              color = [ '1' ],
#              lorentz = [ L.VSS5 ],
#              couplings = {(0,0):C.GC_VV_471})

"""
V_VV_56 = Vertex(name = 'V_VV_56',
              particles = [ P.G, P.G, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVS12 ],
              couplings = {(0,0):C.GC_VV_336})
"""

# here we introduce some dummy couplings which eventually will be set to 0
V_VV_52 = Vertex(name = 'V_VV_52',
              particles = [ P.A, P.A, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS12, L.VVS11 ],
              couplings = {(0,0):C.GC_VV_461, (0,1):C.GC_qed_set0})

V_VV_85 = Vertex(name = 'V_VV_85',
              particles = [ P.A, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS11, L.VVS12 ],
              couplings = {(0,1):C.GC_VV_462_m,(0,0):C.GC_qed_set0})
              #couplings = {(0,1):C.GC_VV_462_m,(0,0):C.GC_VV_543_m}) # GC_VV_543_m is 0 in couplings.py 
