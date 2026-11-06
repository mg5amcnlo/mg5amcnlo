from importlib import reload
import object_library
reload(object_library)
from object_library import all_boundstates, Boundstate
import parameters as Param

##############################
# Bounded electrons in atoms #
##############################

# PDG ID uses 10ZZZAAA11

# e in C12
e_in_C12   = Boundstate(pdg_code = 1000601211,
                        name = 'e_in_C12',
                        particles = ['e-'],
                        shells = [-101, -201, 201, -202],
                        mass = Param.Me,
                        principal = 1,
                        spin = 2,
                        orbital = 0,
                        J = 0,
                        color = 1,
                        charge = -1,
                        ldme = 1.0,
                        texname = 'e_{C^{12}}')

# e in W186
e_in_W186  = Boundstate(pdg_code = 1007418611,
                        name = 'e_in_W186',
                        particles = ['e-'],
                        shells = [-101, -201, 201, -202, -301, 301, -302, 302, -303, -401, 401, 
                                  -402, 402, -403, 403, -404, -501, 501, -502, 502, -503, -601],
                        mass = Param.Me,
                        principal = 1,
                        spin = 2,
                        orbital = 0,
                        J = 0,
                        color = 1,
                        charge = -1,
                        ldme = 1.0,
                        texname = 'e_{W^{186}}')

# e in Pb208
e_in_Pb208 = Boundstate(pdg_code = 1008220811,
                        name = 'e_in_Pb208',
                        particles = ['e-'],
                        shells = [-101, -201, 201, -202, -301, 301, -302, 302, -303, -401, 401,
                                  -402, 402, -403, 403, -404, -501, 501, -502, 502, -503, -601, 
                                  601, -602],
                        mass = Param.Me,
                        principal = 1,
                        spin = 2,
                        orbital = 0,
                        J = 0,
                        color = 1,
                        charge = -1,
                        ldme = 1.0,
                        texname = 'e_{Pb^{208}}')

# e in U238
e_in_U238  = Boundstate(pdg_code = 1009223811,
                        name = 'e_in_U238',
                        particles = ['e-'],
                        shells = [-101, -201, 201, -202, -301, 301, -302, 302, -303, -401, 401,
                                  -402, 402, -403, 403, -404, -501, 501, -502, 502, -503, 503,
                                  -504, -601, 601, -602, 602, -603, -701],
                        mass = Param.Me,
                        principal = 1,
                        spin = 2,
                        orbital = 0,
                        J = 0,
                        color = 1,
                        charge = -1,
                        ldme = 1.0,
                        texname = 'e_{U^{238}}')
