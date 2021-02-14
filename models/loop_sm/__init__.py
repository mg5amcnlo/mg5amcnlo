#from __future__ import absolute_import
from __future__ import absolute_import
import particles
import couplings
import CT_couplings
import lorentz
import parameters
import CT_parameters
import vertices
import CT_vertices
import write_param_card
import coupling_orders
import object_library
import function_library

# model options
gauge = [0, 1]

all_particles = particles.all_particles
all_vertices = vertices.all_vertices
all_CTvertices = vertices.all_CTvertices
all_couplings = couplings.all_couplings
all_lorentz = lorentz.all_lorentz
all_parameters = parameters.all_parameters
all_CTparameters = CT_parameters.all_CTparameters
all_functions = function_library.all_functions
all_orders = coupling_orders.all_orders

__author__ = "V. Hirschi"
__version__ = "1.2"
__email__ = "valentin.hirschi@gmail.com"
