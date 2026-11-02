#!/usr/bin/env python3
"""Read-only native EE2/DY component limits at exact three-body points.

Build an audit executable outside a *completed* generated process, reusing its
compiled objects and cards. No generation, shower, or process-file edits occur.
The points are specified independently by s(q,g)/Q^2=a and s(qbar,g)/Q^2=b.
The reported logarithmic-shell coefficients use dPhi_3 proportional to da db:
lambda^2 H in the soft limit and lambda H in either collinear limit. Finite
scans are numerical diagnostics, not a proof of integrability.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess


METADATA = r"""
      subroutine native_tc_limit_metadata(request,p,pborn,ids)
      use fks_phase_space_data, only: xi_i_fks_ev,y_ij_fks_ev
      use mc_counterterms, only: get_qMC,macnlops_delta_profile
      use weight_lines, only: macnlops_sudakov_max
      implicit none
      include 'nexternal.inc'
      integer request,ids(nexternal),n,status
      double precision p(0:3,nexternal),pborn(0:3,nexternal-1),
     $ t,a,cdf,q(macnlops_sudakov_max),w(macnlops_sudakov_max),
     $ fsr(macnlops_sudakov_max),fks_Sij,fks_Hij
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      external fks_Sij,fks_Hij
      t=get_qMC(xi_i_fks_ev,y_ij_fks_ev)
      a=0d0
      cdf=0d0
      status=-1
      if(t.gt.0d0)call macnlops_delta_profile(p,pborn,ids,t,a,n,
     $ q,w,fsr,cdf,status)
      write(*,'(a,2(1x,i10),7(1x,es26.17e3))')
     $ 'NATIVE_TC_LIMIT_V1',request,status,t,a,cdf,xi_i_fks_ev,
     $ y_ij_fks_ev,fks_Sij(p,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev),
     $ fks_Hij(p,i_fks,j_fks)
      end

      subroutine native_tc_source_compare(p,pcms,ids,flow,fraction,norm,
     $ measure,factors_saved,mc_saved,pdf_saved)
c Evaluate the actual source routine without the evaluator's unit replay.
c This probes whether Delta multiplies its real AND G-replacement pieces.
      use weight_lines
      use process_module
      use scale_module
      use mc_counterterms, only: fksfather,mc_shower_scale_mass
      use fks_phase_space_data, only: p1_cnt
      implicit none
      include 'nexternal.inc'
      integer ids(nexternal),flow,MCcntcalled,i_fks,j_fks
      common/c_MCcntcalled/MCcntcalled
      common/fks_indices/i_fks,j_fks
      double precision p(0:3,nexternal),pcms(0:3,nexternal),fraction,norm,
     $ measure,factors_saved(9),mc_saved(8),pdf_saved(3),
     $ factors(9),mc_factors(8),pdf_factors(3),probne,h,
     $ macnlops_flavour_sum,rwgt,reference
      common/factor_n1body/factors
      common/factor_n1body_NLOPS/mc_factors
      common/factor_pdfsch/pdf_factors
      logical cuts_born,cuts_real,passcuts,passcuts_native_born
      external macnlops_flavour_sum,passcuts,passcuts_native_born
      call reset_delta_contexts()
      born_flow_picked=flow
      call init_process_module_n1body_flow(flow,.true.)
      call compute_shower_scale_n1body(p,i_fks,j_fks,
     $ mc_shower_scale_mass())
      factors=factors_saved
      mc_factors=mc_saved
      pdf_factors=pdf_saved
      call include_born_flow_weight(fraction,1d0)
      call set_cms_stuff(0)
      cuts_born=passcuts_native_born(p1_cnt(0,1,0),rwgt,reference)
      call set_cms_stuff(-100)
      cuts_real=passcuts(p,rwgt)
      icontr=0
      MCcntcalled=0
      mc_H_only=.true.
      mc_S_only=.false.
      call compute_native_NLOPS_weights(p,pcms,cuts_born,cuts_real,probne)
      call include_PDF_and_alphas
      h=macnlops_flavour_sum(ids)/measure*norm
      write(*,'(a,2(1x,es26.17e3))')'NATIVE_TC_SOURCE_V1',probne,h
      icontr=0
      call reset_delta_contexts()
      end
"""


def run(command, cwd, **kwargs):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                            **kwargs)
    if result.returncode:
        raise RuntimeError('Command failed: %s\n%s\n%s' %
                           (shlex.join(map(str, command)), result.stdout,
                            result.stderr))
    return result


def rotate(p):
    angle, phi = .73, .31
    x = math.cos(angle)*p[1] + math.sin(angle)*p[3]
    z = -math.sin(angle)*p[1] + math.cos(angle)*p[3]
    return [p[0], math.cos(phi)*x-math.sin(phi)*p[2],
            math.sin(phi)*x+math.cos(phi)*p[2], z]


def ee2_point(a, b, energy):
    if min(a, b, 1-a-b) <= 0:
        raise ValueError('Unphysical three-body invariants')
    eq, eg = energy*(1-b)/2, energy*(a+b)/2
    cosine = 1-2*a/((1-b)*(a+b))
    gluon = [eg, eg*math.sqrt(max(0., 1-cosine*cosine)), 0., eg*cosine]
    quark = [eq, 0., 0., eq]
    antiquark = [energy*(1-a)/2] + [-quark[i]-gluon[i] for i in range(1, 4)]
    finals = list(map(rotate, [quark, antiquark, gluon]))
    residual = max(abs(sum(p[i] for p in finals) -
                       (energy if i == 0 else 0)) for i in range(4))
    shell = max(abs(p[0]**2-sum(x*x for x in p[1:])) for p in finals)
    if residual > 1e-11*energy or shell > 1e-11*energy**2:
        raise AssertionError('Invalid independent three-body construction')
    y23 = min(2*min(p[0]**2, q[0]**2) *
              (1-sum(p[k]*q[k] for k in range(1, 4))/(p[0]*q[0])) /
              energy**2 for i, p in enumerate(finals) for q in finals[i+1:])
    return [[energy/2, 0., 0., energy/2],
            [energy/2, 0., 0., -energy/2]]+finals, y23


def boost(p, velocity):
    b2 = sum(x*x for x in velocity)
    if b2 == 0:
        return list(p)
    gamma = 1/math.sqrt(1-b2)
    bp = sum(p[k+1]*velocity[k] for k in range(3))
    return [gamma*(p[0]+bp)] + [p[k+1] +
        ((gamma-1)*bp/b2+gamma*p[0])*velocity[k] for k in range(3)]


def dy_point(xi, cosine, mass):
    """Exact recoil against an on-shell dilepton pair, then a fixed lab boost."""
    energy = mass/math.sqrt(1-xi)
    eg = energy*xi/2
    gluon = [eg, eg*math.sqrt(1-cosine*cosine)*math.cos(.29),
             eg*math.sqrt(1-cosine*cosine)*math.sin(.29), eg*cosine]
    pair = [energy-eg]+[-x for x in gluon[1:]]
    c, phi = .21, .43
    lep = [mass/2, mass/2*math.sqrt(1-c*c)*math.cos(phi),
           mass/2*math.sqrt(1-c*c)*math.sin(phi), mass/2*c]
    leptons = [boost(p, [x/pair[0] for x in pair[1:]])
               for p in (lep,[lep[0]]+[-x for x in lep[1:]])]
    lab = [boost(p,[0.,0.,math.tanh(.2)]) for p in
           [[energy/2,0.,0.,energy/2],[energy/2,0.,0.,-energy/2]]+
           leptons+[gluon]]
    return lab, math.hypot(gluon[1],gluon[2])


def build(directory, output):
    evaluator = (directory/'macnlops_evaluator.f').read_text()
    driver = (directory/'driver_macnlops.f').read_text()
    # The extra call reads the already established native point. It is placed
    # after the complete scalar densities and before native-context cleanup.
    anchor = "      status=0\n 900  continue\n"
    if driver.count(anchor) != 1:
        raise RuntimeError('Density cleanup changed; review audit insertion')
    driver = driver.replace(anchor,
        "      call native_tc_limit_metadata(0,p,p_born,ids)\n"
        "      call native_tc_source_compare(p,pcms,ids,component_flow,\n"
        "     $ flows(component_flow)/total,component_normalization,measure,\n"
        "     $ factors_save,mc_save,pdf_save)\n"+anchor)
    source = output/'native_limits.f'
    obj, binary = output/'native_limits.o', output/'native_limits'
    source.write_text(evaluator+'\n'+driver+'\n'+METADATA)
    log = (directory/'macnlops_onestep.build.log').read_text().replace('\\\n', ' ')
    links = [shlex.split(line) for line in log.splitlines()
             if ' -o macnlops_onestep ' in line and ' -c ' not in line]
    if len(links) != 1:
        raise RuntimeError('Expected one completed onestep link command')
    link = links[0]
    link[link.index('-o')+1] = str(binary)
    link[link.index('driver_macnlops.o')] = str(obj)
    link = [s for s in link if not s.endswith('/MacNLOPSMatch.o')]
    link += ['-Wl,-rpath,'+str(directory.parent.parent/'lib')]
    compile_command = ['gfortran', '-O', '-fPIC', '-ffixed-line-length-none',
                       '-fno-common', '-I.', '-I../../lib', '-c',
                       str(source), '-o', str(obj)]
    run(compile_command, directory)
    run(link, directory)
    (output/'build_commands.json').write_text(json.dumps(
        {'cwd':str(directory), 'compile':compile_command, 'link':link}, indent=2)+'\n')
    return binary


def audit(directory, output, energy=91.188, minimum_power=6, process='ee2'):
    directory, output = directory.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary = build(directory, output)
    records, metadata = [], []
    sequences = (('soft', 'qg_collinear', 'qbarg_collinear') if process=='ee2'
                 else ('soft','beam1_collinear','beam2_collinear'))
    for sequence in sequences:
        for power in range(2, minimum_power+1):
            lam = 10.**-power
            a, b = ((.37*lam, .63*lam) if sequence == 'soft' else
                    (lam, .2) if sequence == 'qg_collinear' else (.2, lam))
            if process == 'ee2':
                momenta, observable = ee2_point(a, b, energy)
                kinematics = {'a':a, 'b':b}
            else:
                xi, c = ((lam,.31) if sequence=='soft' else
                         (.2,1-2*lam) if sequence=='beam1_collinear' else
                         (.2,-1+2*lam))
                momenta, observable = dy_point(xi,c,energy)
                kinematics = {'xi_requested':xi, 'cosine_requested':c}
            for config in (1, 2):
                for owner in (1, 2):
                    request = len(records)+1
                    ids = ([-11, 11, 1, -1, 21] if process=='ee2' else
                           [2,-2,11,-11,21])
                    records.append('MACNLOPS_POINT_V3 %d %d 5\n' % (request, owner)+
                        ''.join('%d %s\n' % (pdg, ' '.join('%.17g'%x for x in p))
                                for pdg, p in zip(ids, momenta))+
                        'COMPONENT_LOCAL_V1 %d %d 1 4 1\nEND_POINT\n' % (config, config))
                    metadata.append(dict(request=request, sequence=sequence,
                        lambda_=lam, **kinematics, observable=observable,
                        config=config, flow=1,
                        owner=owner))
    point_input = ''.join(records)
    (output/'points.txt').write_text(point_input)
    # The native PDF wrapper searches relative to argv[0]. A relative argv[0]
    # keeps its lookup identical to the provider, while executable= selects
    # our external audit binary without creating anything in the process.
    result = run(['./native_tc_limits_audit'], directory, executable=str(binary),
                 input=point_input, timeout=300)
    (output/'native_stdout.log').write_text(result.stdout)
    (output/'native_stderr.log').write_text(result.stderr)
    densities, profiles, sources = [], [], []
    for line in result.stdout.splitlines():
        f = line.split()
        if f and f[0] == 'MACNLOPS_DENSITY_V1':
            densities.append((int(f[1]), int(f[2]), *map(float, f[3:])))
        if f and f[0] == 'NATIVE_TC_LIMIT_V1':
            profiles.append((int(f[2]), *map(float, f[3:])))
        if f and f[0] == 'NATIVE_TC_SOURCE_V1':
            sources.append(tuple(map(float, f[1:])))
    if any(len(x) != len(metadata) for x in (densities, profiles, sources)):
        raise RuntimeError('Missing native points; inspect native_stdout.log')
    for row, density, profile, source_pair in zip(metadata, densities, profiles, sources):
        request, status, r, d, k = density
        if request != row['request'] or status:
            raise RuntimeError('Invalid native point %s status %d' % (row, status))
        ps, tc, source, cdf, xi, y, sij, hij = profile
        if ps != 0 or not all(map(math.isfinite,(r,d,k,tc,source,cdf,xi,y,sij,hij))):
            raise RuntimeError('Invalid profile at point %s' % row)
        h = r-d
        lam = row['lambda_']
        factor = lam**(2 if row['sequence']=='soft' else 1)
        probne, source_h = source_pair
        row.update(status=status, r=r, d=d, k=k, h0=h,
                   tc=tc, source=source, cdf=cdf, profile_status=ps,
                   xi=xi, y=y, fks_sij=sij, fks_hij=hij,
                   h0_over_r=h/r if r else None,
                   h0_log_shell=factor*h,
                   damped_h_log_shell=factor*h*source,
                   irsafe_h_log_shell=factor*h*row['observable'],
                   source_sample_probability=probne, source_sample_h=source_h,
                   source_factorization_error=abs(source_h-probne*h)/max(abs(r),abs(d),1.))
    result = {'process':str(directory), 'process_type':process,'energy':energy,
              'energy_definition':'centre-of-mass energy' if process=='ee2' else 'dilepton invariant mass',
              'construction':('independent exact massless three-body invariants' if process=='ee2'
                              else 'exact dilepton decay with recoil and fixed longitudinal boost'),
              'measure':'soft lambda d(lambda), collinear d(lambda), times smooth finite factors',
              'observable':'Durham y23' if process=='ee2' else 'dilepton transverse momentum [GeV]',
              'points':metadata,
              'fingerprints':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [directory/'driver_macnlops.f',
                                        directory/'montecarlocounter.f', source_path()]}}
    summary = []
    for sequence in sequences:
        for config in (1, 2):
            for owner in (1, 2):
                rows = [r for r in metadata if (r['sequence'],r['config'],r['owner']) ==
                        (sequence,config,owner)]
                last = {k:v for k,v in rows[-1].items()
                                if k in ('sequence','config','owner','lambda_', 'h0_over_r',
                                         'h0_log_shell','tc','source','fks_sij')}
                # Deep soft R-D cancellation loses precision sooner than the
                # individual real/MC terms. Use 1e-4 -> 1e-5 for its slope.
                fit_rows = (rows[-3:-1] if sequence=='soft' and minimum_power>=6
                            else rows[-2:])
                low, high = fit_rows
                last.update(fit_lambdas=[low['lambda_'],high['lambda_']],
                    h0_power=math.log(abs(high['h0']/low['h0'])) /
                             math.log(high['lambda_']/low['lambda_']),
                    log_shell_ratio=abs(high['h0_log_shell']/low['h0_log_shell']))
                summary.append(last)
    result['summary'] = summary
    result['max_source_factorization_error'] = max(r['source_factorization_error'] for r in metadata)
    result['finite_scan_checks_pass'] = (
        result['max_source_factorization_error'] < 1e-10 and
        all(r['log_shell_ratio'] < .5 for r in summary))
    result['precision_note'] = ('Soft slope uses lambda=1e-4 to 1e-5; at 1e-6 '
        'the difference of individually O(lambda^-2) real/MC terms begins losing '
        'relative accuracy. The full native source is already exactly zero there.')
    (output/'limits.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def source_path():
    return Path(__file__).resolve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subprocess_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--energy', type=float, default=91.188)
    parser.add_argument('--minimum-power', type=int, default=6)
    parser.add_argument('--process', choices=('ee2','dy'), default='ee2')
    args = parser.parse_args()
    print(json.dumps(audit(args.subprocess_dir, args.output, args.energy,
                           args.minimum_power,args.process), indent=2))
