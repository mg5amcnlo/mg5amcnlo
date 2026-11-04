# type: ignore
from __future__ import absolute_import
from . import particles
from . import couplings
from . import lorentz
from . import parameters
from . import vertices
from . import coupling_orders
from . import function_library
from . import object_library

gauge = [0]

all_particles = particles.all_particles
all_vertices = vertices.all_vertices
all_couplings = couplings.all_couplings
all_lorentz = lorentz.all_lorentz
all_parameters = parameters.all_parameters
all_orders = coupling_orders.all_orders
all_functions = function_library.all_functions


__author__ = "Valentin Hirschi"
__version__ = "1.0"
__email__ = "valentin.hirschi@gmail.com"
