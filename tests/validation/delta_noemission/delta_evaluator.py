"""Call this branch's MC@NLO-Delta routines and shipped Sudakov table.

No kernels are reimplemented: the Fortran probability routine and its helpers
are extracted verbatim at build time. Native start/stop scales are compiled
from the production source. The PDF provider must be supplied by the caller.

Common-cut evaluation uses the soft limit of the production dipole masses and
sets every live stopping scale to q. Actual-real evaluation instead passes the
full real kinematics to delta_scale_matrices. These are different observables.
Momenta are positive-energy (E, px, py, pz); incoming legs are first.
"""
from pathlib import Path
import ctypes as C
import gzip
import hashlib
import math
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.unit_tests.fks.test_momentum_maps import fortran_routine
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'
SOURCE = TEMPLATE / 'montecarlocounter.f'
PDF_CALLBACK = C.CFUNCTYPE(C.c_double, C.c_int, C.c_int, C.c_int, C.c_double, C.c_double)

WRAPPER = '''module delta_pdf_interface
  use iso_c_binding
  implicit none
  abstract interface
    function pdf_interface(beam,id,lp,x,q) bind(C) result(f)
      import c_int,c_double
      integer(c_int), value :: beam,id,lp
      real(c_double), value :: x,q
      real(c_double) :: f
    end function
  end interface
  procedure(pdf_interface), pointer :: pdf_ptr => null()
contains
  subroutine delta_set_pdf(f) bind(C)
    type(c_funptr), value :: f
    call c_f_procpointer(f,pdf_ptr)
  end subroutine
end module
function pdg2pdf(beam,id,lp,x,q) result(f)
  use delta_pdf_interface
  implicit none
  integer :: beam,id,lp
  double precision :: x,q,f
  if (.not.associated(pdf_ptr)) error stop 'PDF callback required'
  f=pdf_ptr(beam,id,lp,x,q)
end function
module delta_c_api
  use iso_c_binding
  use delta_production
  use mcatnlo_delta_scales, only: pythia8_starting_scales, delta_scale_matrices
  implicit none
contains
  subroutine delta_select(n,ids,links,special,starting,stopping,masses,dead, &
                          counts,partners,types,starts,stops,dmasses) bind(C)
    use process_module, only: valid_dipole_n
    integer(c_int), value :: n,special
    integer(c_int), intent(in) :: ids(n),links(n,n),dead(n,n)
    real(c_double), intent(in) :: starting(n,n),stopping(n,n),masses(n,n)
    integer(c_int), intent(out) :: counts(n),partners(2,n),types(2,n)
    real(c_double), intent(out) :: starts(2,n),stops(2,n),dmasses(2,n)
    logical*1 :: dead_native(n,n)
    type(delta_connections) :: connections(n)
    integer :: i
    nexternal=n+1
    if (allocated(valid_dipole_n)) deallocate(valid_dipole_n)
    allocate(valid_dipole_n(n,n,1))
    valid_dipole_n(:,:,1)=links/=0
    dead_native=dead/=0
    call get_delta_connections(starting,stopping,dead_native,masses,ids, &
                               1,special/=0,1.01d0,7000d0,connections)
    do i=1,n
      counts(i)=connections(i)%count
      partners(:,i)=connections(i)%partner
      types(:,i)=connections(i)%sudakov_type
      starts(:,i)=connections(i)%start
      stops(:,i)=connections(i)%stop
      dmasses(:,i)=connections(i)%mass
    enddo
  end subroutine
  subroutine delta_leg(ipart,pdg,nflav,flav,beam,x,nline,partner,typ, &
                       starting,stopping,masses,value,raw) bind(C)
    integer(c_int), value :: ipart,pdg,nflav,beam,nline
    integer(c_int), intent(in) :: flav(nflav),partner(2),typ(2)
    real(c_double), value :: x
    real(c_double), intent(in) :: starting(2),stopping(2),masses(2)
    real(c_double), intent(out) :: value,raw
    double precision :: mcmass(21)
    type(delta_connections) :: conn
    conn%count=nline
    conn%partner=partner
    conn%sudakov_type=typ
    conn%start=starting
    conn%stop=stopping
    conn%mass=masses
    mcmass=0d0
    include 'MCmasses_PYTHIA8.inc'
    value=delta_leg_probability(ipart,pdg,flav,beam,x,conn,mcmass)
    raw=delta_leg_probability_raw(ipart,pdg,flav,beam,x,conn,mcmass)
  end subroutine
  subroutine delta_starts(n,p,mass,links,q,scales,status) bind(C)
    integer(c_int), value :: n
    real(c_double), intent(in) :: p(0:3,n),mass(n)
    integer(c_int), intent(in) :: links(n,n)
    real(c_double), value :: q
    real(c_double), intent(out) :: scales(n,n)
    integer(c_int), intent(out) :: status
    logical :: connected(n,n)
    connected=links/=0
    call pythia8_starting_scales(n,p,mass,connected,q,scales,status)
  end subroutine
  subroutine delta_stops(n,emitted,p,ids,colors,mc,mb,scales,masses,dead,status) bind(C)
    integer(c_int), value :: n,emitted
    real(c_double), intent(in) :: p(0:3,n)
    integer(c_int), intent(in) :: ids(n),colors(2,n-1)
    real(c_double), value :: mc,mb
    real(c_double), intent(out) :: scales(0:n,0:n),masses(0:n,0:n)
    integer(c_int), intent(out) :: dead(0:n,0:n),status
    logical :: native_dead(0:n,0:n)
    call delta_scale_matrices(n,emitted,p,ids,colors,mc,mb,scales,masses,native_dead,status)
    dead=merge(1,0,native_dead)
  end subroutine
end module
'''


def _array(values, dtype=float):
    return np.asfortranarray(values, dtype=np.float64 if dtype is float else np.int32)


def _ptr(a):
    return a.ctypes.data_as(C.POINTER(C.c_double if a.dtype==np.float64 else C.c_int))


class DeltaEvaluator:
    """Production Delta with a caller-owned PDF function pdf(pdg,x,q_GeV).

    An instance is not thread-safe: its Fortran PDF callback is mutable. Use
    separate processes for parallel runs. Temporary build products are retained
    until close() or context-manager exit; supply work to retain provenance.
    """
    def __init__(self, work=None, table=None):
        self._tmp = tempfile.TemporaryDirectory(prefix='delta_evaluator_') if work is None else None
        self.work = (Path(self._tmp.name) if self._tmp else Path(work)).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        table = Path(table) if table else ROOT/'vendor/SudGen/sudakov.f'
        source = SOURCE.read_text()
        dtype = re.search(r'^      type :: delta_connections\n.*?^      end type delta_connections',
                          source, re.M|re.S).group()
        names = ['delta_leg_probability','gl_safe','pysudakov_safe',
                 'get_parton_id','setSudType','get_to_zero','emscafun','get_delta_connections']
        routines = [fortran_routine(SOURCE,n) for n in names]
        # Diagnostic only: identical routine before the final [0,1] clipping.
        raw = routines[0].replace('delta_leg_probability','delta_leg_probability_raw')
        assert 'min(max(PIk,0d0),1d0)' in raw
        raw = raw.replace('min(max(PIk,0d0),1d0)','PIk')
        selected = ('      module process_module\n'
                    '      logical, allocatable :: valid_dipole_n(:,:,:)\n'
                    '      end module process_module\n'
                    '      module delta_production\n      implicit none\n'
                    '      integer, parameter :: nincoming=2\n'
                    '      integer :: nexternal=0\n' + dtype +
                    '\n      contains\n' + '\n'.join(routines+[raw])+
                    '\n      end module delta_production\n')
        (self.work/'delta_production.f').write_text(selected)
        data = gzip.decompress(table.read_bytes()) if table.suffix=='.gz' else table.read_bytes()
        (self.work/'sudakov.f').write_bytes(data)
        (self.work/'delta_wrapper.f90').write_text(WRAPPER)
        cern = ROOT/'Template/NLO/Source/CERNLIB'
        lib = self.work/'delta.so'
        cmd = ['gfortran','-O2','-std=legacy','-shared','-fPIC',
               '-ffixed-line-length-none','-I'+str(TEMPLATE),'-J'+str(self.work),
               str(TEMPLATE/'mcatnlo_delta_scales.f90'),str(self.work/'delta_production.f'),
               str(self.work/'delta_wrapper.f90'),str(self.work/'sudakov.f'),
               *(str(cern/f) for f in ['dfint.f','kerset.f','abend.f']),'-o',str(lib)]
        result = subprocess.run(cmd,cwd=self.work,capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
        self.lib = C.CDLL(str(lib))
        self.lib.delta_set_pdf.argtypes = [PDF_CALLBACK]
        self.lib.delta_starts.argtypes = [C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_double),
            C.POINTER(C.c_int),C.c_double,C.POINTER(C.c_double),C.POINTER(C.c_int)]
        self.lib.delta_stops.argtypes = [C.c_int,C.c_int,C.POINTER(C.c_double),
            C.POINTER(C.c_int),C.POINTER(C.c_int),C.c_double,C.c_double,
            C.POINTER(C.c_double),C.POINTER(C.c_double),C.POINTER(C.c_int),C.POINTER(C.c_int)]
        self.lib.delta_select.argtypes = [C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_int),C.c_int,
            *([C.POINTER(C.c_double)]*3),*([C.POINTER(C.c_int)]*4),
            *([C.POINTER(C.c_double)]*3)]
        self.lib.delta_leg.argtypes = [C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),
            C.c_int,C.c_double,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_int),
            C.POINTER(C.c_double),C.POINTER(C.c_double),C.POINTER(C.c_double),
            C.POINTER(C.c_double),C.POINTER(C.c_double)]
        self.provenance = dict(table=str(table.resolve()),table_sha256=hashlib.sha256(data).hexdigest(),
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [SOURCE,TEMPLATE/'mcatnlo_delta_scales.f90',TEMPLATE/'MCmasses_PYTHIA8.inc']},
            extracted_routines=names,compile_command=cmd)

    def close(self):
        if self._tmp:
            self._tmp.cleanup()
            self._tmp=None

    def __enter__(self): return self
    def __exit__(self,*exc): self.close()

    @staticmethod
    def connections(colors):
        """Color endpoints, preserving duplicate gluon-to-gluon connections."""
        n=len(colors)
        partners=[[] for _ in range(n)]
        for i in range(n):
            for c,tag in enumerate(colors[i]):
                if tag==0: continue
                found=[j for j in range(n) if j!=i and
                       colors[j][1-c if (i<2)==(j<2) else c]==tag]
                if len(found)!=1:
                    raise ValueError(('Invalid color connection',i,c,tag,found))
                partners[i]+=found
        if any(len(p)>2 for p in partners): raise ValueError('Too many color lines')
        return partners

    def starting_scales(self,born):
        if 'starts' in born:
            n=len(born['ids'])
            scales=_array(born['starts'])
            if scales.shape!=(n,n) or not np.isfinite(scales).all():
                raise ValueError('Invalid supplied directed starting-scale matrix')
            for i,partners in enumerate(self.connections(born['colors'])):
                if any(scales[i,j]<=0 for j in partners):
                    raise ValueError(('Nonpositive supplied color-dipole scale',i,partners))
            return scales.copy(order='F')
        p=_array(born['p']).T.copy(order='F')
        mass=_array(born['masses'])
        n=len(mass)
        partners=self.connections(born['colors'])
        links=_array([[int(j in partners[i]) for j in range(n)] for i in range(n)],int)
        scales=np.zeros((n,n),order='F')
        status=C.c_int()
        self.lib.delta_starts(n,_ptr(p),_ptr(mass),_ptr(links),float(born['scalup']),
                              _ptr(scales),C.byref(status))
        if status.value: raise ValueError(('Native starting scales failed',status.value,born))
        return scales

    def real_scales(self,born,real,mc=0.,mb=0.):
        """Native scales in Born order; masses above the 7 TeV table are capped."""
        n=len(real['ids']); emitted=real.get('emitted',n)
        p=_array(real['p']).T.copy(order='F'); ids=_array(real['ids'],int)
        colors=_array(born['colors'],int).T.copy(order='F')
        scales=np.zeros((n+1,n+1),order='F'); masses=np.zeros_like(scales,order='F')
        dead=np.zeros((n+1,n+1),dtype=np.int32,order='F'); status=C.c_int()
        self.lib.delta_stops(n,emitted,_ptr(p),_ptr(ids),_ptr(colors),mc,mb,
                            _ptr(scales),_ptr(masses),_ptr(dead),C.byref(status))
        if status.value: raise ValueError(('Native stopping scales failed',status.value,real))
        idx=[i for i in range(1,n+1) if i!=emitted]
        return scales[np.ix_(idx,idx)], np.minimum(masses[np.ix_(idx,idx)],7000.), dead[np.ix_(idx,idx)].astype(bool)

    @staticmethod
    def born_dipole_masses(born):
        """Soft-emission limit of delta_scale_matrices, including ISR->II."""
        p=np.array(born['p']); n=len(p)
        masses=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                k=1-i if i<2 else j
                q=(-p[i] if i<2 else p[i])+(-p[k] if k<2 else p[k])
                masses[i,j]=min(math.sqrt(abs(q[0]**2-np.dot(q[1:],q[1:]))),7000.)
        return masses

    def evaluate(self,born,q,pdf):
        """Uniform shower evolution cutoff q on the given Born state."""
        start=self.starting_scales(born)
        stop=np.full_like(start,float(q))
        return self.evaluate_matrices(born,start,stop,self.born_dipole_masses(born),
                                      stop>=start,pdf)

    def evaluate_real(self,born,real,pdf,mc=0.,mb=0.):
        """Production real-dependent Delta, with charm/bottom model masses."""
        start=self.starting_scales(born)
        stop,mass,dead=self.real_scales(born,real,mc,mb)
        return self.evaluate_matrices(born,start,stop,mass,dead|(stop>start),pdf)

    def select_connections(self,born,start,stop,mass,dead):
        """Call the unmodified production connection selector with fixture state."""
        ids=_array(born.get('table_ids',born['ids']),int)
        n=len(ids)
        partners=self.connections(born['colors'])
        links=_array([[int(j in partners[i]) for j in range(n)] for i in range(n)],int)
        special=any(len(p)==2 and p[0]==p[1] for p in partners)
        counts=np.zeros(n,dtype=np.int32)
        ptn=np.zeros((2,n),dtype=np.int32,order='F'); types=np.zeros_like(ptn,order='F')
        starts=np.zeros((2,n),order='F'); stops=np.zeros_like(starts,order='F')
        masses=np.zeros_like(starts,order='F')
        inputs=[_array(a) for a in (start,stop,mass)]+[_array(dead,int)]
        self.lib.delta_select(n,_ptr(ids),_ptr(links),int(special),*[_ptr(a) for a in inputs],
            _ptr(counts),_ptr(ptn),_ptr(types),_ptr(starts),_ptr(stops),_ptr(masses))
        return [[dict(partner=int(ptn[k,i]),type=int(types[k,i]),start=starts[k,i],
                      stop=stops[k,i],mass=masses[k,i]) for k in range(counts[i])]
                for i in range(n)]

    def evaluate_matrices(self,born,start,stop,mass,dead,pdf):
        exceptions=[]
        def callback(beam,id,lp,x,q):
            try:
                f=float(pdf(21 if id==0 else id,x,q))
                if not math.isfinite(f): raise ValueError(('Nonfinite PDF',id,x,q,f))
                return f
            except Exception as exc:
                exceptions.append(exc)
                return float('nan')
        self._callback=PDF_CALLBACK(callback)
        self.lib.delta_set_pdf(self._callback)
        selected=self.select_connections(born,start,stop,mass,dead)
        vals=[]; raw_vals=[]; details=[]; nonfinite=[]
        for i,pdg in enumerate(born.get('table_ids',born['ids'])):
            lines=selected[i]
            live=[d['partner']-1 for d in lines]
            if not live:
                vals.append(1.); raw_vals.append(1.); details.append([]); nonfinite.append(False)
                continue
            starts=[d['start'] for d in lines]
            stops=[d['stop'] for d in lines]
            masses=[d['mass'] for d in lines]
            types=[d['type'] for d in lines]
            pad=lambda a,dtype=float:_array(a+[0]*(2-len(a)),dtype)
            arrays=[pad([j+1 for j in live],int),pad(types,int),pad(starts),pad(stops),pad(masses)]
            flavours=_array(born.get('flavours',[[v] for v in born['ids']])[i],int)
            val=C.c_double(); raw=C.c_double()
            self.lib.delta_leg(i+1,pdg,len(flavours),_ptr(flavours),1 if i<2 else 0,
                born['x'][i] if i<2 else 0.,len(live),*[_ptr(a) for a in arrays],C.byref(val),C.byref(raw))
            if exceptions: raise exceptions[0]
            if not math.isfinite(val.value):
                raise ValueError(('Nonfinite production Delta',i,starts,stops,masses,val.value,raw.value))
            vals.append(val.value)
            raw_vals.append(raw.value if math.isfinite(raw.value) else None)
            nonfinite.append(not math.isfinite(raw.value))
            details.append([dict(partner=j+1,type=t,start=a,stop=b,mass=m)
                            for j,t,a,b,m in zip(live,types,starts,stops,masses)])
        return dict(delta=math.prod(vals),isr=math.prod(vals[:2]),fsr=math.prod(vals[2:]),
                    legs=vals,raw_legs=raw_vals,
                    clipped=[b is not None and a!=b for a,b in zip(vals,raw_vals)],
                    nonfinite_raw=nonfinite,connections=details)
