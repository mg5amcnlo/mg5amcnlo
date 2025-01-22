complex_mass = False  # Tag for activating the complex mass scheme
unitary_gauge = True  # Tag choosing between Feynman Gauge or unitary gauge
                      # 0/False: Feynman
                      # 1/True: unitary
                      # 2: axial
                      # 3: Feynman Diagram gauge (5D aloha)
loop_mode = False     # Tag for encoding momenta with complex number.
mp_precision = False  # Tag for passing parameter in quadruple precision
aloha_prefix = 'mdl_'


class ALOHAERROR(Exception): pass
