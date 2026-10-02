#!/usr/bin/env python3
"""Source-level PYTHIA 8.318 comparison; this is not a shower event run.

Compile production radiation inverses, z/t/J and prefactors, with
small include fixtures. The reference forward maps are transcribed from
SimpleTimeShower::branch and SimpleSpaceShower::branch in tag pythia8318.
Only the Python standard library and gfortran are needed.
"""
import math
import random
import subprocess
import sys
import atexit
import shutil
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.unit_tests.fks.test_momentum_maps import (
    fortran_routine as routine, fks_test_module, mc_counterterm_test_module)

WORK = Path(tempfile.mkdtemp(prefix='pythia8318-matching-audit-'))
atexit.register(shutil.rmtree, WORK)
SRC = ROOT / 'Template/NLO/SubProcesses'

for name, data in {
    'nexternal.inc':'      integer nexternal,nincoming\n      parameter(nexternal=5,nincoming=2)\n',
    'genps.inc':'      integer max_branch,max_particles\n      parameter(max_branch=8,max_particles=8)\n',
    'born_nhel.inc':'      integer max_bcol\n      parameter(max_bcol=1)\n',
    'native.f90':'module mc_native_context\nlogical :: native_mapping=.true.\nend module\n',
    'scale.f90':'module scale_module\ndouble precision :: shower_scale_nbody_max(4,4)=1000d0\nend module\n',
}.items():
    (WORK/name).write_text(data)
for name in ('fks_powers.inc',):
    shutil.copyfile(SRC/name, WORK/name)

(WORK/'phase_space.f').write_text(fks_test_module(('invert_fks_radiation',), SRC))
routines = [routine(SRC/'fks_singular.f', n) for n in ('rotate_invar', 'trp_rotate_invar')]
routines += [mc_counterterm_test_module(
             ('zPY8','xiPY8','xjacPY8','dinvariants_dFKS','xfact_ileg12','xfact_ileg3','xfact_ileg4','get_dead_zone','get_angle'), SRC)]
routines += [routine(ROOT/'Template/NLO/Source/kin_functions.f',n) for n in ('dot','rho','threedot')]
(WORK/'routines.f').write_text('\n'.join(routines))
(WORK/'driver.f90').write_text('''program audit
  use mc_counterterms, only: zPY8,xiPY8,xjacPY8,xfact_ileg12, &
       xfact_ileg3,xfact_ileg4,get_dead_zone, &
       ileg,fksfather,xm12,xm22,w1,w2,yij,x,xij,betad,betas,kn,knbar,kn0,shat_n1
  use fks_phase_space_data,only: bound_born => tau_Born_lower_bound, &
       bound_res => tau_lower_bound_resonance,bound_tau => tau_lower_bound,vkn => veckn_ev, &
       vknbar => veckbarn_ev,ve => xp0jfks,p_i_fks_cnt
  use process_module, only: next_n1,nincoming_mod,mass_n,shower_mc_mod
  use fks_phase_space_helpers, only: dot,rho,boost_n1_to_its_cms,get_xi_from_p,get_yij_from_p
  use fks_phase_space, only: invert_fks_radiation
  implicit none
  integer i,jf,ios,ifks,jfks
  common/fks_indices/ifks,jfks
  double precision pmass(5),omx(2)
  common/to_mass/pmass
  common/to_ee_omx1/omx
  logical soft,coll,zone
  common/sctests/soft,coll
  double precision p(0:3,5),pc(0:3,5),pb(0:3,-8:4),m,mb2,mk2,rt(3),jac,ps,tau,yb,xb(2),rap
  double precision z,t,jz,f,rf,yy,phi,s,rs,stw,pyweight
  next_n1=5
  nincoming_mod=2
  shower_mc_mod='PYTHIA8'
  allocate(mass_n(4))
  bound_born=1d-12
  bound_res=1d-12
  bound_tau=1d-12
  omx=0d0
  soft=.false.
  coll=.false.
  stw=4d6
  ifks=5
  do
    read(*,*,iostat=ios)jf,m
    if(ios.ne.0)exit
    do i=1,5
      read(*,*)p(:,i)
    enddo
    jfks=jf
    pmass=0d0
    pmass(jf)=m
    jac=1d0
    ps=1d0
    call invert_fks_radiation(rt,jac,ps,stw,tau,yb,xb,p,pb)
    if(jac.le.0d0)stop 2
    call boost_n1_to_its_cms(p,pc,rap)
    s=2d0*dot(pc(:,1),pc(:,2))
    rs=sqrt(s)
    rf=get_xi_from_p(5,jf,pc)
    yy=get_yij_from_p(5,jf,pc,p_i_fks_cnt(:,0))
    shat_n1=s
    x=1d0-rf
    yij=yy
    kn=rho(pc(:,jf))
    knbar=rho(pb(:,jf))
    kn0=pc(0,jf)
    vkn=kn
    vknbar=knbar
    ve=kn0
    if(jf.le.2)then
      ileg=jf
      xm12=0d0
      xm22=0d0
    else
      mk2=dot(pc(:,4),pc(:,4))
      if(m.gt.0d0)then
        ileg=3
        xm12=m*m
        xm22=mk2
        w1=2d0*dot(pc(:,3),pc(:,5))
        w2=s*rf-w1
      else
        ileg=4
        xm12=mk2
        xm22=0d0
        w2=2d0*dot(pc(:,3),pc(:,5))
        w1=s*rf-w2
      endif
    endif
    betas=1d0+(xm12-xm22)/s
    betad=sqrt((1d0-(xm12-xm22)/s)**2-4d0*xm22/s)
    xij=2d0*(1d0-xm12/s-rf)/(2d0-rf*(1d0-yy))
    z=zPY8()
    t=xiPY8(z)
    jz=xjacPY8(z)
    if(ileg.le.2)f=xfact_ileg12(1)
    if(ileg.eq.3)f=xfact_ileg3(1)
    if(ileg.eq.4)f=xfact_ileg4(1)
    fksfather=jf
    mass_n=0d0
    mass_n(3)=m
    mass_n(4)=sqrt(max(0d0,dot(pb(:,4),pb(:,4))))
    call get_dead_zone(z,t,pb(:,1:4),sqrt(t),4,zone,pyweight)
    if(.not.zone)stop 3
    write(*,'(24ES25.16)')z,t,jz,f,rf,yy,xb,pb(:,1:4)
  enddo
end program
''')
cmd=['gfortran','-O2','-std=legacy','-ffixed-line-length-none','-ffree-line-length-none',
     '-fno-automatic','-ffunction-sections','-fdata-sections',
     '-Wl,-dead_strip' if sys.platform=='darwin' else '-Wl,--gc-sections',
     '-I',str(WORK),str(SRC/'process_module.f90'),str(SRC/'fks_phase_space_data.f'),
     str(SRC/'genps_fks_helpers.f'),
     'native.f90','scale.f90',
     str(SRC/'genps_fks_radiation.f'),'phase_space.f','routines.f',str(SRC/'boostwdir2.f'),
     str(SRC/'resonance_recoil.f'),str(SRC/'initial_recoil.f'),
     str(ROOT/'HELAS/boostx.F'),'driver.f90','-o','audit']
subprocess.run(cmd,cwd=WORK,check=True,capture_output=True,text=True)

def plus(a,b):return [x+y for x,y in zip(a,b)]
def scale(a,x):return [x*y for y in a]
def dot(a,b):return a[0]*b[0]-sum(x*y for x,y in zip(a[1:],b[1:]))
def norm(a):return math.sqrt(sum(x*x for x in a))
def boost(p,b):
    b2=sum(x*x for x in b)
    if b2==0:return p[:]
    g=1/math.sqrt(1-b2);bp=sum(x*y for x,y in zip(b,p[1:]))
    return [g*(p[0]+bp)]+[p[i+1]+((g-1)*bp/b2+g*p[0])*b[i] for i in range(3)]
def rot(p,theta,phi):
    x,y,z=p[1:]
    a=math.cos(theta)*x+math.sin(theta)*z
    return [p[0],math.cos(phi)*a-math.sin(phi)*y,
            math.sin(phi)*a+math.cos(phi)*y,-math.sin(theta)*x+math.cos(theta)*z]

def fsr(m,M,z,t):
    S=1e6;R=1000.;q=m*m+t/(z*(1-z));E=(S+q-M*M)/(2*R);k=math.sqrt(E*E-q)
    pz1=(E*E*z-q/2)/k;pz2=(E*E*(1-z)-q/2)/k
    pt=math.sqrt(q*(E*E*z*(1-z)-q/4)/(k*k))
    f=m*m/q
    pt*=1-f;pz1+=f*pz2;pz2*=1-f
    r=rot([math.sqrt(m*m+pt*pt+pz1*pz1),pt,0.,pz1],.6,.4)
    e=rot([math.sqrt(pt*pt+pz2*pz2),-pt,0.,pz2],.6,.4)
    rec=rot([R-E,0.,0.,-k],.6,.4)
    EB=(S+m*m-M*M)/(2*R);pB=math.sqrt(EB*EB-m*m)
    B=[rot([EB,0.,0.,pB],.6,.4),rot([R-EB,0.,0.,-pB],.6,.4)]
    beams=[[500.,0.,0.,500.],[500.,0.,0.,-500.]]
    J=(1-f)*(S+q-M*M)/(32*math.pi**3*(2*R*pB)*z*(1-z))
    return beams+[r,rec,e],beams+B,J

def isr():
    # PYTHIA II branch with Born x=(0.12,0.20), z=0.6, Q2=25600.
    z=.6;Q=25600.;t=(1-z)*Q;S=96000.;root=math.sqrt(S);phi=.7
    a=[120.,0.,0.,120.];b=[200.,0.,0.,-200.]
    p=[root/2,root/2*math.sqrt(.91),0.,root/2*.3]
    B=[a,b,boost(p,[0,0,-.25]),boost([p[0],-p[1],0.,-p[3]],[0,0,-.25])]
    er=(S+Q)/(2*root)
    pt=math.sqrt(Q-z*(S+Q)*Q/S)*S/(z*(S+Q))
    pz=root/2*((S-Q)/(z*(S+Q))+Q/S)
    mother=[math.hypot(pt,pz),pt,0.,pz]
    sister=[math.hypot(pt,pz-er),pt,0.,pz-er]
    rec=[er,0.,0.,-er]
    P=plus(mother,rec);v=[-P[i]/P[0] for i in (1,2,3)]
    tmp=boost(mother,v);theta=math.atan2(tmp[1],tmp[3])
    def forward(q):return rot(boost(q,v),-theta,phi)
    R=[forward(mother),forward(rec)]
    for pB in B[2:]:R.append(forward(rot(boost(pB,[0,0,.25]),0,-phi)))
    R.append(forward(sister))
    return R,B,z,t

random.seed(8318)
cases=[]
for m in (0.,50.,173.):
    for M in (40.,180.):
        for i in range(30):
            z=random.uniform(.6,.88);t=random.uniform(500.,3500.)
            p,b,J=fsr(m,M,z,t)
            cases.append((3,m,p,b,z,t,J))
for m in (0.,50.,173.):
    for M in (40.,173.,180.):
        accepted=0
        while accepted<180:
            z=random.uniform(.015,.985)
            D=(1000-M)**2-m*m
            t=random.uniform(.001,.98)*D*z*(1-z)
            try:p,b,J=fsr(m,M,z,t)
            except ValueError:continue
            cases.append((3,m,p,b,z,t,J))
            accepted+=1
p,b,z,t=isr();cases.append((1,0.,p,b,z,t,1/(32*math.pi**3*(1-z))))
data=''.join(f'{jf} {m}\n'+''.join(' '.join(f'{x:.17g}' for x in row)+'\n' for row in p)
             for jf,m,p,*_ in cases)
(WORK/'inputs.txt').write_text(data)
run=subprocess.run([str(WORK/'audit')],input=data,text=True,capture_output=True,check=True)
lines=run.stdout.splitlines()
if len(lines)!=len(cases):raise RuntimeError(run.stdout)
max_z=max_t=max_J=max_born=0.
negative=[]
for case,line in zip(cases,lines):
    jf,m,p,b,z,t,J=case;out=list(map(float,line.split()))
    if len(out)!=24:raise RuntimeError(line)
    zo,to,jac,f,xi,y,xb1,xb2=out[:8]
    max_z=max(max_z,abs(zo-z));max_t=max(max_t,abs(to/t-1))
    coeff=f*jac/(xi*xi*(1-y))
    expected=1/(16*math.pi**3*J)/(z if jf<=2 else 1)
    max_J=max(max_J,abs(coeff/expected-1))
    if coeff<0:
        negative.append((m,math.sqrt(max(0,dot(p[3],p[3]))),z,t,xi,y,f,coeff,expected))
    if jf>2:
        max_born=max(max_born,max(abs(a-c)/1000 for a,c in zip(out[8:],sum(b,[]))))
    else:
        born3=out[16:20];born1=out[8:12]
        print('ISR PYTHIA Born fractions:',b[0][0]/1000,b[1][0]/1000)
        print('ISR FKS Born fractions:',xb1,xb2)
        print('ISR MC PDF fractions:',xb1/zo,xb2)
        print('ISR real beam fractions:',p[0][0]/1000,p[1][0]/1000)
        print('ISR PYTHIA versus FKS (2 a.p3 / sB):',2*dot(b[0],b[2])/96000,2*dot(born1,born3)/96000)
print('FSR samples:',len(cases)-1)
print('Max abs z error:',max_z)
print('Max relative t error:',max_t)
print('Max relative full scalar Jacobian coefficient error:',max_J)
print('Max FSR Born momentum error / 1000 GeV:',max_born)
print('Negative production Jacobian coefficient count:',len(negative))
for entry in sorted(negative,key=lambda a:abs(a[1]-173.))[:3]:
    print('Negative coefficient fixture (m,M,z,t,xi,y,xfact,coefficient,positive reference):',entry)
assert max_z<1e-10 and max_t<1e-10 and max_J<1e-9 and max_born<1e-9
assert not negative

# A physical g->gg point with a massive global recoil.
S=1e6;M=400.;z=.8;t=15000.;q=t/(z*(1-z));r=M*M/S
x1=(1-r+q/S)*z;x2=1+r-q/S
D=1-r/(x1+x2-1-r)*(1+r-x2)/(1-r-x1)
print('Gluon recoil dead-cone weight at S=1e6,M=400,z=.8,t=15000:',D)

# Change the initialization of otherwise unset locals. The corrected
# xjacPY8 threshold is a PARAMETER and must be independent of these flags.
poisoncmd=cmd[:1]+['-finit-real=inf']+cmd[1:]
poisoncmd[-1]='audit_poison'
subprocess.run(poisoncmd,cwd=WORK,check=True,capture_output=True,text=True)
poison=subprocess.run([str(WORK/'audit_poison')],input='\n'.join(data.splitlines()[:6])+'\n',text=True,capture_output=True,check=True)
old=float(lines[0].split()[2]);new=float(poison.stdout.split()[2])
print('First FSR xjacPY8 with default versus poisoned unset locals:',old,new)
assert abs(old/new-1)<1e-12
