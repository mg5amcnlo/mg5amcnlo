// The LHEH5 code below has been modified from the original
// https://bitbucket.org/iamholger/lheh5/src/master/ authored by
// Holger Schulz, and developed under the Scientific Discovery
// through Advanced Computing (SciDAC) program funded by
// the U.S. Department of Energy, Office of Science, Advanced Scientific
// Computing Research.
//
// Note, this header can be used in conjuction with LHAHF5.h.
//
// Fermilab Software Legal Information (BSD License)
// Copyright (c) 2009, FERMI NATIONAL ACCELERATOR LABORATORY
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the
// distribution.
//
// Neither the name of the FERMI NATIONAL ACCELERATOR LABORATORY, nor
// the names of its contributors may be used to endorse or promote
// products derived from this software without specific prior written
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef Pythia8_LHEH5_H
#define Pythia8_LHEH5_H

// Standard includes.
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

// MPI includes.
#include <mpi.h>

// HighFive includes.
#include "highfive/H5File.hpp"
#include "highfive/H5DataSet.hpp"

using namespace HighFive;

namespace LHEH5 {

//==========================================================================

// Particle struct.

struct Particle {
  int id, status, mother1, mother2, color1, color2;
  double px, py, pz, e, m, lifetime, spin;
  // id        .. IDUP
  // color1/2  .. ICOLUP firt/second
  // mother1/2 .. MOTHUP first/second
  // status    .. ISTUP
  // px ... m  .. PUP[..]
  // lifetime  .. VTIMUP
  // spin      .. SPINUP
  // (UP ... user process)
};

//--------------------------------------------------------------------------

// Print a particle.

std::ostream& operator << (std::ostream& os, Particle const & p) {
  os << "\tpx: " << p.px  << " py: " << p.py  << " pz: " << p.pz
     << " e: " << p.e <<  "\n";
  return os;
}

//==========================================================================

// Event header struct.

struct EventHeader {
  // Event info
  int    nparticles; // corr to NUP
  int    pid;        // this is all LHAu-::setProcess
  std::vector<double> weights;
  //double weight;
  size_t trials;
  double scale;
  double rscale;
  double fscale;
  double aqed;
  double aqcd;
  int    npLO;
  int    npNLO;
};

//--------------------------------------------------------------------------

// Print an event header.

std::ostream& operator << (std::ostream& os, EventHeader const & eh) {
  os << "\tnparticles: " << eh.nparticles  << " procid: " << eh.pid
     << " weights: { ";
  for (const auto& w : eh.weights) os << w << " ";
  os << "} trials: " << eh.trials <<  "\n";
  os << "\tscale: " << eh.scale  << " rscale: " << eh.rscale << " fscale: "
     << eh.fscale  << " aqed: " << eh.aqed << " aqcd: " << eh.aqcd <<  "\n";
  os << "\tnpLO: " << eh.npLO  << " npNLO: " << eh.npNLO <<  "\n";
  return os;
}

//==========================================================================

// Old event struct.

struct Events {
  // Lookup.
  std::vector<size_t> _vstart;
  std::vector<size_t> _vend;
  // Particles.
  std::vector<int>    _vid;
  std::vector<int>    _vstatus;
  std::vector<int>    _vmother1;
  std::vector<int>    _vmother2;
  std::vector<int>    _vcolor1;
  std::vector<int>    _vcolor2;
  std::vector<double> _vpx;
  std::vector<double> _vpy;
  std::vector<double> _vpz;
  std::vector<double> _ve;
  std::vector<double> _vm;
  std::vector<double> _vlifetime;
  std::vector<double> _vspin;
  // Event info.
  std::vector<int>    _vnparticles;
  std::vector<int>    _vpid;
  std::vector<double> _vweight;
  std::vector<size_t> _vtrials;
  std::vector<double> _vscale;
  std::vector<double> _vrscale;
  std::vector<double> _vfscale;
  std::vector<double> _vaqed;
  std::vector<double> _vaqcd;
  std::vector<int>    _vnpLO;
  std::vector<int>    _vnpNLO;
  size_t _particle_offset;
  Particle mkParticle(size_t idx) const;
  std::vector<Particle> mkEvent(size_t ievent) const;
  EventHeader mkEventHeader(int ievent) const;
};

//--------------------------------------------------------------------------

// Make a particle, given an index.

Particle Events::mkParticle(size_t idx) const {
  return {std::move(_vid[idx]), std::move(_vstatus[idx]),
      std::move(_vmother1[idx]), std::move(_vmother2[idx]),
      std::move(_vcolor1[idx]), std::move(_vcolor2[idx]),
      std::move(_vpx[idx]), std::move(_vpy[idx]), std::move(_vpz[idx]),
      std::move(_ve[idx]), std::move(_vm[idx]),
      std::move(_vlifetime[idx]),std::move(_vspin[idx])};
}

//--------------------------------------------------------------------------

// Make an event, given an index.

std::vector<Particle> Events::mkEvent(size_t ievent) const {
  std::vector<Particle> _E;
  // NOTE we need to subtract the particle offset here as the
  // particle properties run from 0 and not the global index
  // when using batches.
  size_t id = _vstart[ievent] - _particle_offset;
  for ( ; id <(_vend[ievent] - _particle_offset); ++id)
    _E.push_back(mkParticle( id));

  // Make sure beam particles are ordered according to convention
  // i.e. first particle has positive z-momentum
  if (_E[0].pz <0) std::swap<Particle>(_E[0], _E[1]);

  return _E;
}

//--------------------------------------------------------------------------

// Make an event header, given an index.

EventHeader Events::mkEventHeader(int iev) const {
  return {std::move(_vnparticles[iev]), std::move(_vpid[iev]),
      std::vector<double>(1,_vweight[iev]), std::move(_vtrials[iev]),
      std::move(_vscale[iev]), std::move(_vrscale[iev]),
      std::move(_vfscale[iev]),
      std::move(_vaqed[iev]), std::move(_vaqcd[iev]),
      std::move(_vnpLO[iev]), std::move(_vnpNLO[iev])};
}

//--------------------------------------------------------------------------

// Read function, returns an Events struct.

Events readEvents(Group& g_index, Group& g_particle, Group& g_event,
  size_t first_event, size_t n_events) {
  // Lookup
  std::vector<size_t> _vstart, _vend;
  // Particles
  std::vector<int>    _vid, _vstatus, _vmother1, _vmother2, _vcolor1, _vcolor2;
  std::vector<double> _vpx, _vpy, _vpz, _ve, _vm, _vlifetime, _vspin;
  // Event info
  std::vector<int>    _vnparticles, _vpid, _vnpLO, _vnpNLO;
  std::vector<size_t> _vtrials      ;
  std::vector<double> _vweight, _vscale, _vrscale, _vfscale, _vaqed, _vaqcd;

  // Lookup.
  DataSet _start = g_index.getDataSet("start");
  DataSet _end   = g_index.getDataSet("end");
  // Particles.
  DataSet _id       =  g_particle.getDataSet("id");
  DataSet _status   =  g_particle.getDataSet("status");
  DataSet _mother1  =  g_particle.getDataSet("mother1");
  DataSet _mother2  =  g_particle.getDataSet("mother2");
  DataSet _color1   =  g_particle.getDataSet("color1");
  DataSet _color2   =  g_particle.getDataSet("color2");
  DataSet _px       =  g_particle.getDataSet("px");
  DataSet _py       =  g_particle.getDataSet("py");
  DataSet _pz       =  g_particle.getDataSet("pz");
  DataSet _e        =  g_particle.getDataSet("e");
  DataSet _m        =  g_particle.getDataSet("m");
  DataSet _lifetime =  g_particle.getDataSet("lifetime");
  DataSet _spin     =  g_particle.getDataSet("spin");
  // Event info.
  DataSet _nparticles =  g_event.getDataSet("nparticles");
  DataSet _pid        =  g_event.getDataSet("pid");
  DataSet _weight     =  g_event.getDataSet("weight");
  DataSet _trials     =  g_event.getDataSet("trials");
  DataSet _scale      =  g_event.getDataSet("scale");
  DataSet _rscale     =  g_event.getDataSet("rscale");
  DataSet _fscale     =  g_event.getDataSet("fscale");
  DataSet _aqed       =  g_event.getDataSet("aqed");
  DataSet _aqcd       =  g_event.getDataSet("aqcd");
  DataSet _npLO       =  g_event.getDataSet("npLO");
  DataSet _npNLO       =  g_event.getDataSet("npNLO");

  std::vector<size_t> offset_e = {first_event};
  std::vector<size_t> readsize_e = {n_events};

  _start.select(offset_e, readsize_e).read(_vstart);
  _end.select(  offset_e, readsize_e).read(_vend  );
  std::vector<size_t> offset_p   = {_vstart.front()};
  std::vector<size_t> readsize_p = {_vend.back()-_vstart.front()};

  int RESP = _vend.back()-_vstart.front();
  _vid.reserve(RESP);
  _vstatus  .reserve(RESP);
  _vmother1 .reserve(RESP);
  _vmother2 .reserve(RESP);
  _vcolor1  .reserve(RESP);
  _vcolor2  .reserve(RESP);
  _vpx      .reserve(RESP);
  _vpy      .reserve(RESP);
  _vpz      .reserve(RESP);
  _ve       .reserve(RESP);
  _vm       .reserve(RESP);
  _vlifetime.reserve(RESP);
  _vspin    .reserve(RESP);

  _vnparticles.reserve(n_events);
  _vpid       .reserve(n_events);
  _vweight    .reserve(n_events);
  _vtrials    .reserve(n_events);
  _vscale     .reserve(n_events);
  _vrscale    .reserve(n_events);
  _vfscale    .reserve(n_events);
  _vaqed      .reserve(n_events);
  _vaqcd      .reserve(n_events);
  _vnpLO      .reserve(n_events);
  _vnpNLO     .reserve(n_events);

  // This is using HighFive's read.
  _id      .select(offset_p, readsize_p).read(_vid      );
  _status  .select(offset_p, readsize_p).read(_vstatus  );
  _mother1 .select(offset_p, readsize_p).read(_vmother1 );
  _mother2 .select(offset_p, readsize_p).read(_vmother2 );
  _color1  .select(offset_p, readsize_p).read(_vcolor1  );
  _color2  .select(offset_p, readsize_p).read(_vcolor2  );
  _px      .select(offset_p, readsize_p).read(_vpx      );
  _py      .select(offset_p, readsize_p).read(_vpy      );
  _pz      .select(offset_p, readsize_p).read(_vpz      );
  _e       .select(offset_p, readsize_p).read(_ve       );
  _m       .select(offset_p, readsize_p).read(_vm       );
  _lifetime.select(offset_p, readsize_p).read(_vlifetime);
  _spin    .select(offset_p, readsize_p).read(_vspin    );

  _nparticles.select(offset_e, readsize_e).read(_vnparticles);
  _pid       .select(offset_e, readsize_e).read(_vpid       );
  _weight    .select(offset_e, readsize_e).read(_vweight    );
  _trials    .select(offset_e, readsize_e).read(_vtrials    );
  _scale     .select(offset_e, readsize_e).read(_vscale     );
  _rscale    .select(offset_e, readsize_e).read(_vrscale    );
  _fscale    .select(offset_e, readsize_e).read(_vfscale    );
  _aqed      .select(offset_e, readsize_e).read(_vaqed      );
  _aqcd      .select(offset_e, readsize_e).read(_vaqcd      );
  _npLO      .select(offset_e, readsize_e).read(_vnpLO      );
  _npNLO     .select(offset_e, readsize_e).read(_vnpNLO     );

  return {
    std::move(_vstart),
      std::move(_vend),
      std::move(_vid),
      std::move(_vstatus),
      std::move(_vmother1),
      std::move(_vmother2),
      std::move(_vcolor1),
      std::move(_vcolor2),
      std::move(_vpx),
      std::move(_vpy),
      std::move(_vpz),
      std::move(_ve),
      std::move(_vm),
      std::move(_vlifetime),
      std::move(_vspin),
      std::move(_vnparticles),
      std::move(_vpid),
      std::move(_vweight),
      std::move(_vtrials),
      std::move(_vscale),
      std::move(_vrscale),
      std::move(_vfscale),
      std::move(_vaqed),
      std::move(_vaqcd),
      std::move(_vnpLO),
      std::move(_vnpNLO),
      offset_p[0]};
}

//==========================================================================

// New events struct.

struct Events2 {
  // Lookup.
  std::vector<size_t> _vstart;
  // Particles.
  std::vector<int> _vid;
  std::vector<int> _vstatus;
  std::vector<int> _vmother1;
  std::vector<int> _vmother2;
  std::vector<int> _vcolor1;
  std::vector<int> _vcolor2;
  std::vector<double> _vpx;
  std::vector<double> _vpy;
  std::vector<double> _vpz;
  std::vector<double> _ve;
  std::vector<double> _vm;
  std::vector<double> _vlifetime;
  std::vector<double> _vspin;
  // Event info.
  std::vector<int>                 _vnparticles;
  std::vector<int>                 _vpid;
  std::vector<std::vector<double>> _vweightvec;
  std::vector<size_t>              _vtrials;
  std::vector<double>              _vscale;
  std::vector<double>              _vrscale;
  std::vector<double>              _vfscale;
  std::vector<double>              _vaqed;
  std::vector<double>              _vaqcd;
  int npLO;
  int npNLO;
  size_t _particle_offset;

  Particle mkParticle(size_t idx) const;
  std::vector<Particle> mkEvent(size_t ievent) const;
  EventHeader mkEventHeader(int ievent) const;
};

//--------------------------------------------------------------------------

// Make a particle, given an index.

Particle Events2::mkParticle(size_t idx) const {
  return {std::move(_vid[idx]), std::move(_vstatus[idx]),
      std::move(_vmother1[idx]), std::move(_vmother2[idx]),
      std::move(_vcolor1[idx]), std::move(_vcolor2[idx]),
      std::move(_vpx[idx]), std::move(_vpy[idx]), std::move(_vpz[idx]),
      std::move(_ve[idx]), std::move(_vm[idx]),
      std::move(_vlifetime[idx]),std::move(_vspin[idx])};
}

//--------------------------------------------------------------------------

// Make an event, given an index.

std::vector<Particle> Events2::mkEvent(size_t ievent) const {
  std::vector<Particle> _E;
  // NOTE we need to subtract the particle offset here as the particle
  // properties run from 0 and not the global index when using batches.
  size_t partno = _vstart[ievent] - _particle_offset;
  for (int id=0; id <_vnparticles[ievent];++id){
    _E.push_back(mkParticle(partno));
    partno++;
  }
  // Make sure beam particles are ordered according to convention
  // i.e. first particle has positive z-momentum
  if (_E[0].pz < 0.) std::swap<Particle>(_E[0], _E[1]);
  return _E;
}

//--------------------------------------------------------------------------

// Make an event header, given an index.

EventHeader Events2::mkEventHeader(int iev) const {
  return {std::move(_vnparticles[iev]), std::move(_vpid[iev]),
      std::move(_vweightvec[iev]), std::move(_vtrials[iev]),
      std::move(_vscale[iev]), std::move(_vrscale[iev]),
      std::move(_vfscale[iev]),
      std::move(_vaqed[iev]), std::move(_vaqcd[iev]),
      npLO, npNLO,
      };
}

//--------------------------------------------------------------------------

// Read function, returns an Events struct --- this is for the new structure.

Events2 readEvents2(Group& g_particle, Group& g_event, size_t first_event,
  size_t n_events, int npLO, int npNLO, bool hasMultiWts) {
  // Lookup.
  std::vector<size_t> _vstart;
  // Particles.
  std::vector<int>    _vid, _vstatus, _vmother1, _vmother2, _vcolor1, _vcolor2;
  std::vector<double> _vpx, _vpy, _vpz, _ve, _vm, _vlifetime, _vspin;
  // Event info.
  std::vector<int>    _vnparticles, _vpid;
  std::vector<size_t> _vtrials      ;
  std::vector<double> _vscale, _vrscale, _vfscale, _vaqed, _vaqcd;

  // Lookup.
  DataSet _start = g_event.getDataSet("start");
  // Particles.
  DataSet _id       =  g_particle.getDataSet("id");
  DataSet _status   =  g_particle.getDataSet("status");
  DataSet _mother1  =  g_particle.getDataSet("mother1");
  DataSet _mother2  =  g_particle.getDataSet("mother2");
  DataSet _color1   =  g_particle.getDataSet("color1");
  DataSet _color2   =  g_particle.getDataSet("color2");
  DataSet _px       =  g_particle.getDataSet("px");
  DataSet _py       =  g_particle.getDataSet("py");
  DataSet _pz       =  g_particle.getDataSet("pz");
  DataSet _e        =  g_particle.getDataSet("e");
  DataSet _m        =  g_particle.getDataSet("m");
  DataSet _lifetime =  g_particle.getDataSet("lifetime");
  DataSet _spin     =  g_particle.getDataSet("spin");
  // Event info.
  DataSet _nparticles =  g_event.getDataSet("nparticles");
  DataSet _pid        =  g_event.getDataSet("pid");
  DataSet _weight     =  g_event.getDataSet("weight");
  DataSet _trials     =  g_event.getDataSet("trials");
  DataSet _scale      =  g_event.getDataSet("scale");
  DataSet _rscale     =  g_event.getDataSet("rscale");
  DataSet _fscale     =  g_event.getDataSet("fscale");
  DataSet _aqed       =  g_event.getDataSet("aqed");
  DataSet _aqcd       =  g_event.getDataSet("aqcd");

  std::vector<size_t> offset_e = {first_event};
  std::vector<size_t> readsize_e = {n_events};

  // We now know the first event to read.
  _start.select(offset_e, readsize_e).read(_vstart);

  // That's the first particle.
  std::vector<size_t> offset_p   = {_vstart.front()};
  // The last particle is last entry in start + nparticles of that event.
  _vnparticles.reserve(n_events);
  _nparticles.select(offset_e, readsize_e).read(_vnparticles);

  size_t RESP = _vstart.back() -   _vstart.front() + _vnparticles.back();
  std::vector<size_t> readsize_p = {RESP};
  _vid.reserve(RESP);
  _vstatus  .reserve(RESP);
  _vmother1 .reserve(RESP);
  _vmother2 .reserve(RESP);
  _vcolor1  .reserve(RESP);
  _vcolor2  .reserve(RESP);
  _vpx      .reserve(RESP);
  _vpy      .reserve(RESP);
  _vpz      .reserve(RESP);
  _ve       .reserve(RESP);
  _vm       .reserve(RESP);
  _vlifetime.reserve(RESP);
  _vspin    .reserve(RESP);
  _vpid     .reserve(n_events);
  _vtrials  .reserve(n_events);
  _vscale   .reserve(n_events);
  _vrscale  .reserve(n_events);
  _vfscale  .reserve(n_events);
  _vaqed    .reserve(n_events);
  _vaqcd    .reserve(n_events);

  // This is using HighFive's read.
  _id      .select(offset_p, readsize_p).read(_vid      );
  _status  .select(offset_p, readsize_p).read(_vstatus  );
  _mother1 .select(offset_p, readsize_p).read(_vmother1 );
  _mother2 .select(offset_p, readsize_p).read(_vmother2 );
  _color1  .select(offset_p, readsize_p).read(_vcolor1  );
  _color2  .select(offset_p, readsize_p).read(_vcolor2  );
  _px      .select(offset_p, readsize_p).read(_vpx      );
  _py      .select(offset_p, readsize_p).read(_vpy      );
  _pz      .select(offset_p, readsize_p).read(_vpz      );
  _e       .select(offset_p, readsize_p).read(_ve       );
  _m       .select(offset_p, readsize_p).read(_vm       );
  _lifetime.select(offset_p, readsize_p).read(_vlifetime);
  _spin    .select(offset_p, readsize_p).read(_vspin    );
  _pid       .select(offset_e, readsize_e).read(_vpid   );
  // Read event weights depending on format.
  std::vector<size_t> wtdim = _weight.getSpace().getDimensions();
  size_t n_weights = wtdim.size() > 1 ? wtdim[1] : 1;
  std::vector< std::vector<double> > _vweightvec(n_events,
    std::vector<double>(n_weights));
  if (hasMultiWts) {
    // The weights are stored in a (possibly one-dimensional) vector.
    std::vector<size_t> offsets = {first_event,first_event};
    std::vector<size_t> counts  = {n_events, n_weights};
    _weight  .select(offsets, counts).read(_vweightvec  );
  } else {
    // The weights are stored as a single floating point value.
    std::vector<double> _vweight;
    _weight  .select(offset_e, readsize_e).read(_vweight);
    _vweightvec.resize(1);
    _vweightvec[0] = _vweight;
  }
  _trials    .select(offset_e, readsize_e).read(_vtrials);
  _scale     .select(offset_e, readsize_e).read(_vscale );
  _rscale    .select(offset_e, readsize_e).read(_vrscale);
  _fscale    .select(offset_e, readsize_e).read(_vfscale);
  _aqed      .select(offset_e, readsize_e).read(_vaqed  );
  _aqcd      .select(offset_e, readsize_e).read(_vaqcd  );

  return {
      std::move(_vstart     ),
      std::move(_vid        ),
      std::move(_vstatus    ),
      std::move(_vmother1   ),
      std::move(_vmother2   ),
      std::move(_vcolor1    ),
      std::move(_vcolor2    ),
      std::move(_vpx        ),
      std::move(_vpy        ),
      std::move(_vpz        ),
      std::move(_ve         ),
      std::move(_vm         ),
      std::move(_vlifetime  ),
      std::move(_vspin      ),
      std::move(_vnparticles),
      std::move(_vpid       ),
      std::move(_vweightvec ),
      std::move(_vtrials    ),
      std::move(_vscale     ),
      std::move(_vrscale    ),
      std::move(_vfscale    ),
      std::move(_vaqed      ),
      std::move(_vaqcd      ),
      npLO,
      npNLO,
      offset_p[0],
      };
}

//==========================================================================

} // end namespace LHEH5

#endif // Pythia8_LHEH5_H
