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
import function_library

gauge = [1]

all_particles = particles.all_particles
all_vertices = vertices.all_vertices
all_CTvertices = CT_vertices.all_CTvertices
all_couplings = couplings.all_couplings
all_lorentz = lorentz.all_lorentz
all_parameters = parameters.all_parameters
all_CTparameters = CT_parameters.all_CTparameters
all_functions = function_library.all_functions
all_orders = coupling_orders.all_orders

__author__ = "Huasheng Shao"
__version__ = "1.0"
__email__ = "huasheng.shao@cern.ch"
