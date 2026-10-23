#!/usr/bin/env python3
"""Compare the new evaluator to ordinary repartition_MC_H numerically.

Usage: python3 docs/audits/check_macnlops_source_h.py PROCESS/SubProcesses/P0_uux_emep
       python3 docs/audits/check_macnlops_source_h.py --process ttbar PROCESS/SubProcesses/P0_gg_ttx

The supplied subprocess must be a compiled u u~ > e+ e- [real=QCD]
export, with its macnlops_evaluator already built. The audit visits 100
flavour-resolved real configurations, including both qqbar owners and qg/gq.
It tests the original generation routine with its outer FKS, diagram and
phase-space factors, independently of the evaluator's density assembly.
The ttbar option instead checks four gg -> ttbar g owners at 25 massive
points, with the standard-model top mass 173 GeV. Its Born basis has two
colour flows, which are summed exactly. Neither option tests integration
or a full shower.
"""
import argparse
import math
from pathlib import Path
import random
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def boost(p, velocity):
    b2 = sum(v*v for v in velocity)
    if b2 < 1.e-30:
        return p[:]
    gamma = 1 / math.sqrt(1-b2)
    bp = sum(p[i+1]*velocity[i] for i in range(3))
    return [gamma*(p[0]+bp)] + [p[i+1] +
        ((gamma-1)*bp/b2+gamma*p[0])*velocity[i] for i in range(3)]


def points():
    rng = random.Random(20261020)
    records = []
    for _ in range(25):
        root = rng.uniform(200.,1800.)
        energy = rng.uniform(.001,.3)*root
        c,phi,y = rng.uniform(-.97,.97),rng.uniform(-math.pi,math.pi),rng.uniform(-1.,1.)
        emitted = [energy,energy*math.sqrt(1-c*c)*math.cos(phi),
                   energy*math.sqrt(1-c*c)*math.sin(phi),energy*c]
        q = [root-energy] + [-v for v in emitted[1:]]
        mass = math.sqrt(root*root-2*root*energy)
        c1,phi1 = rng.uniform(-.8,.8),rng.uniform(-math.pi,math.pi)
        lepton = [mass/2,mass/2*math.sqrt(1-c1*c1)*math.cos(phi1),
                  mass/2*math.sqrt(1-c1*c1)*math.sin(phi1),mass/2*c1]
        velocity = [v/q[0] for v in q[1:]]
        leptons = [boost(lepton,velocity),
                   boost([lepton[0]]+[-v for v in lepton[1:]],velocity)]
        event = [[root/2,0.,0.,root/2],[root/2,0.,0.,-root/2]] + leptons + [emitted]
        event = [boost(p,[0.,0.,math.tanh(y)]) for p in event]
        for owner,ids in ((1,[2,-2,11,-11,21]),(2,[2,-2,11,-11,21]),
                          (3,[21,-2,11,-11,-2]),(4,[2,21,11,-11,2])):
            request = len(records)+1
            records.append('MACNLOPS_POINT_V1 %d %d 5\n' % (request,owner) +
                ''.join('%d %s\n' % (pdg,' '.join('%.17g' % x for x in p))
                        for pdg,p in zip(ids,event)) + 'END_POINT\n')
    return records


def points_ttbar():
    rng = random.Random(20261020)
    records = []
    for _ in range(25):
        root = rng.uniform(700., 1800.)
        energy = rng.uniform(.001, .15) * root
        c, phi, y = (rng.uniform(-.8, .8), rng.uniform(-math.pi, math.pi),
                     rng.uniform(-1., 1.))
        emitted = [energy, energy * math.sqrt(1 - c*c) * math.cos(phi),
                   energy * math.sqrt(1 - c*c) * math.sin(phi), energy*c]
        pair = [root - energy] + [-v for v in emitted[1:]]
        mass = math.sqrt(root*root - 2*root*energy)
        c1, phi1 = rng.uniform(-.8, .8), rng.uniform(-math.pi, math.pi)
        momentum = math.sqrt(mass*mass/4 - 173.**2)
        top = [mass/2, momentum * math.sqrt(1 - c1*c1) * math.cos(phi1),
               momentum * math.sqrt(1 - c1*c1) * math.sin(phi1), momentum*c1]
        velocity = [v / pair[0] for v in pair[1:]]
        tops = [boost(top, velocity),
                boost([top[0]] + [-v for v in top[1:]], velocity)]
        event = [[root/2, 0., 0., root/2], [root/2, 0., 0., -root/2]] + tops + [emitted]
        event = [boost(p, [0., 0., math.tanh(y)]) for p in event]
        for owner in (1, 2, 3, 4):
            request = len(records) + 1
            records.append('MACNLOPS_POINT_V1 %d %d 5\n' % (request, owner) +
                ''.join('%d %s\n' % (pdg, ' '.join('%.17g' % x for x in p))
                        for pdg, p in zip((21, 21, 6, -6, 21), event)) + 'END_POINT\n')
    return records


def check_source_h(subprocess_dir, point_records=None):
    directory = Path(subprocess_dir).resolve()
    original = (directory/'driver_macnlops.f').read_text()
    # Reuse the evaluator's initialization and communication protocol;
    # evaluate the source generation path separately at every same real point.
    source = original.replace('program macnlops_evaluator',
                              'program macnlops_source_h_audit',1)
    declaration = '      double precision lab(0:3,nexternal),real_density,'
    if source.count(declaration) != 1:
        raise RuntimeError('Evaluator initialization changed: update the audit adapter')
    source = source.replace(declaration,
        '      double precision source_h,source_measure,lab(0:3,nexternal),real_density,',1)
    output = "         write(*,'(a,2(1x,i12),3(1x,es26.17e3))')"
    if source.count(output) != 1:
        raise RuntimeError('Evaluator output changed: update the audit adapter')
    source = source.replace(output,
        "         call source_h_audit(owner,ids,lab,source_h,source_measure)\n"
        "         write(*,'(a,1x,i5,3(1x,es26.17e3))')\n"
        "     $     'SOURCE_H_AUDIT',request,real_density-subtraction_density,\n"
        "     $     source_h,source_measure\n" + output,1)
    source += '\n' + (ROOT/'tests/input_files/macnlops_source_h_audit.f').read_text()
    dry_run = subprocess.run(['make','-n','-W','driver_macnlops.o','macnlops_evaluator'],
        cwd=directory,check=True,capture_output=True,text=True).stdout
    links = [line for line in dry_run.splitlines()
             if ' -o macnlops_evaluator ' in line]
    if len(links) != 1:
        raise RuntimeError('Build the evaluator before running its source-H audit')
    command = shlex.split(links[0])
    # The generated LHAPDF wrapper locates lib/PDFsets from argv[0].
    with tempfile.TemporaryDirectory(prefix='.macnlops-source-h-',
                                     dir=directory) as temporary:
        temporary = Path(temporary)
        src,obj,executable = (temporary/name for name in ('audit.f','audit.o','audit'))
        src.write_text(source)
        subprocess.run(['gfortran','-O','-fPIC','-ffixed-line-length-none','-fno-common',
            '-I.','-I../../lib','-c',str(src),'-o',str(obj)],cwd=directory,
            check=True,capture_output=True,text=True)
        command[command.index('-o')+1] = str(executable)
        command[command.index('driver_macnlops.o')] = str(obj)
        command += ['-Wl,-rpath,' + str(directory.parent.parent/'lib')]
        subprocess.run(command,cwd=directory,check=True,capture_output=True,text=True)
        result = subprocess.run([str(executable)],cwd=directory,
            input=''.join(points() if point_records is None else point_records),
            capture_output=True,text=True,timeout=120,check=True)
    records,densities = [],{}
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and fields[0] == 'SOURCE_H_AUDIT':
            records.append((int(fields[1]),*[float(x) for x in fields[2:]]))
        if fields and fields[0] == 'MACNLOPS_DENSITY_V1':
            densities[int(fields[1])] = (int(fields[2]),*[float(x) for x in fields[3:]])
    if len(records) != 100 or len(densities) != 100:
        raise AssertionError('Source audit did not evaluate every point:\n'+result.stdout[-12000:])
    maximum = h_maximum = 0.
    for request,prediction,actual,measure in records:
        status,real,subtraction,shower = densities[request]
        if status != 0 or measure <= 0 or not all(map(math.isfinite,
                (prediction,actual,measure,real,subtraction,shower))):
            raise AssertionError('Invalid source/evaluator point %d:\n%s' %
                                 (request,result.stdout[-12000:]))
        error = abs(prediction-actual)/max(abs(real),abs(subtraction),1.)
        maximum = max(maximum,error)
        h_maximum = max(h_maximum,abs(prediction-actual)/max(abs(prediction),1.))
        if error > 1.e-10:
            raise AssertionError('Source H mismatch at point %d: %.17g != %.17g' %
                                 (request,prediction,actual))
    return {'points':len(records),'max_error_relative_real_subtraction':maximum,
            'max_error_relative_hard':h_maximum}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--process', choices=('dy', 'ttbar'), default='dy')
    parser.add_argument('subprocess_dir',type=Path)
    args = parser.parse_args()
    print(check_source_h(args.subprocess_dir,
                         points_ttbar() if args.process == 'ttbar' else None))
