# This file was automatically created by FeynRules 1.7.69
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (November 6, 2010)
# Date: Mon 1 Oct 2012 14:58:26

from object_library import all_propagators, Propagator


# define only once the denominator since this is always the same
denominator = "P('mu', id) * P('mu', id) - Mass(id) * Mass(id) + complex(0,1) * Mass(id) * Width(id)"

# propagator for the scalar
S = Propagator(name = "S",
               numerator = "1",
               denominator = denominator
               )

# propagator for the incoming fermion # the one for the outcomming is computed on the flight
F = Propagator(name = "F",
                numerator = "(Gamma('mu', 1, 2) * P('mu', id) + Mass(id) * Identity(1, 2))",
                denominator = denominator
               )

# massive vector in the unitary gauge, can't be use for massless particles
V1 = Propagator(name = "V1",
                numerator = "(-1 * Metric(1, 2) + Metric(1,'mu')* P('mu', id) * P(2, id) / Mass(id)**2 ",
                denominator = denominator
               )

# massless vector and massive vector in unitary gauge
V2 = Propagator(name = "V2",
                numerator = "-1 * Metric(1, 2)",
                denominator =  "P('mu', id) * P('mu', id)"
               )


