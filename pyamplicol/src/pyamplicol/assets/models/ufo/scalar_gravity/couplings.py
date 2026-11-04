# type: ignore
from __future__ import absolute_import
from .object_library import all_couplings, Coupling
from . import parameters as Param

for i_scalar in list(range(Param.N_SCALARS)):
    globals()['SSTmpart%d' % i_scalar] = Coupling(name='SSTmpart%d' % i_scalar,
                                                  value='%s**2*kappa*complex(0,1)/2' % Param.scalar_mass_parameters[
                                                      i_scalar].name if i_scalar > 0 else "ZERO",
                                                  order={'GRAV': 1})

    globals()['SSTTTmpart%d' % i_scalar] = Coupling(name='SSTTTmpart%d' % i_scalar,
                                                    value='%s**2*kappa**3*complex(0,1)/32' % Param.scalar_mass_parameters[
                                                        i_scalar].name if i_scalar > 0 else "ZERO",
                                                    order={'GRAV': 3})

    globals()['SSTTmpart%d' % i_scalar] = Coupling(name='SSTTmpart%d' % i_scalar,
                                                   value='%s**2*kappa**2*complex(0,1)/8' % Param.scalar_mass_parameters[
                                                       i_scalar].name if i_scalar > 0 else "ZERO",
                                                   order={'GRAV': 2})

SST = Coupling(name='SST',
               value='kappa*complex(0,1)/2',
               order={'GRAV': 1})

SSTT = Coupling(name='SSTT',
                value='kappa**2*complex(0,1)/8',
                order={'GRAV': 2})

SSTTT = Coupling(name='SSTTT',
                 value='kappa**3*complex(0,1)/32',
                 order={'GRAV': 3})

TTT = Coupling(name='TTT',
               value='kappa*complex(0,1)/8',
               order={'GRAV': 1})

TTTT = Coupling(name='TTTT',
                value='kappa**2*complex(0,1)/32',
                order={'GRAV': 2})
