
import particles
import couplings
import lorentz
import parameters
import vertices
import write_param_card
import coupling_orders

# model options
gauge = [0,1]

all_particles = particles.all_particles
all_vertices = vertices.all_vertices
all_CTvertices = [] 
all_couplings = couplings.all_couplings
all_lorentz = lorentz.all_lorentz
all_parameters = parameters.all_parameters
all_CTparameters = []
all_functions = function_library.all_functions
all_orders = coupling_orders.all_orders

__author__ = "Priscila de Aquino"
__version__ = "2.1"
__email__ = "priscila@itf.kuleuven.be"
