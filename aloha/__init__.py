complex_mass = False  # Tag for activating the complex mass scheme
unitary_gauge = True  # Tag choosing between Feynman Gauge or unitary gauge
                      # 0/False: Feynman
                      # 1/True: unitary
                      # 2: axial
                      # 3: Feynman Diagram gauge (5D aloha)
loop_mode = False     # Tag for encoding momenta with complex number.
dual_mode = 0         # Number of P-wave onia in the current process (0/False
                      # when none): selects the dual-number HELAS routines.
npwave = [0]          # P-wave multiplicities the exporter has to write HELAS
                      # libraries for. 0 is the plain (non-dual) library.
mp_precision = False  # Tag for passing parameter in quadruple precision
aloha_prefix = 'mdl_'


class ALOHAERROR(Exception): pass
