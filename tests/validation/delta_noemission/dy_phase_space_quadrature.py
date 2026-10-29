#!/usr/bin/env python3
"""Illustrative massless light-quark forward/backward ISR kernel integrals.

Uses the identical Pythia PDF13 provider as the matched shower. This is a
calculation of pre-branching continuous rates, not another shower simulation.
The total quark PDF is used, without explicit Pythia valence/sea sampling.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    a = argparse.ArgumentParser()
    a.add_argument('study', type=Path)
    a.add_argument('output', type=Path)
    a.add_argument('--hard-scale', type=float)
    args = a.parse_args()
    study, out = args.study.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bridge = study/'pdf13/analysis_pdf13/pythia_pdf.so'
    pythia = Path('/export/tmp/rikkert/Pythia/pythia8313')
    pdf = C.CDLL(str(bridge))
    pdf.pythia_pdf_init.argtypes = [C.c_char_p, C.c_char_p]
    pdf.pythia_pdf_init.restype = C.c_int
    pdf.pythia_pdf_xf.argtypes = [C.c_int, C.c_double, C.c_double]
    pdf.pythia_pdf_xf.restype = C.c_double
    assert pdf.pythia_pdf_init(str(pythia/'share/Pythia8/xmldoc').encode(), b'13') == 0
    lib = C.CDLL(str(study/'pdf13/analysis_pdf13/PDFmode1/delta.so'))
    table = lib.pysudakov_
    table.argtypes = [C.POINTER(C.c_double), C.POINTER(C.c_double),
        C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_double)]
    table.restype = C.c_double
    masses = (C.c_double*21)(.33,.33,.5,1.5,4.8,*([0.]*16))
    mass = 91.1876
    hard = args.hard_scale if args.hard_scale is not None else mass
    x = mass/13000.
    cf = 4./3.
    beta = 23./(12.*np.pi)
    coupling_mz = 91.188  # AlphaStrong's reference value, independent of Born M.
    lam2 = coupling_mz**2*np.exp(-1./(beta*.130))

    def coupling(t):
        return 1./(beta*np.log(t/lam2))

    def zmax(t):
        r = t/mass**2
        return 1.-.5*r*(np.sqrt(1.+4./r)-1.)

    def virtual(t):
        z = zmax(t)
        # Integral_0^zmax dz z[Pqq(z)+Pgq(z)].
        return cf*(-2.*np.log(1.-z)-1.5*z*z)

    def raw_table(q,pid=2):
        aa,bb,ii,tt=C.c_double(q),C.c_double(mass),C.c_int(abs(pid)),C.c_int(1)
        return table(C.byref(aa),C.byref(bb),C.byref(ii),C.byref(tt),masses)

    rows = []
    for cut in [q for q in [5.,10.,20.,30.,50.,75.] if q < hard]:
        lo,hi=np.log(cut*cut),np.log(hard*hard)
        iv,ev=quad(lambda lt: coupling(np.exp(lt))/(2.*np.pi)*virtual(np.exp(lt)),
            lo,hi,epsabs=1.e-8)
        forward=np.exp(-iv)
        stored=raw_table(cut)/raw_table(hard)
        legs = {}
        for pid in [1,2,-1,-2,3,-3]:
            def rate(t):
                pt=np.sqrt(t)
                den=pdf.pythia_pdf_xf(pid,x,pt)
                def integrand(z):
                    fq=pdf.pythia_pdf_xf(pid,x/z,pt)
                    fg=pdf.pythia_pdf_xf(21,x/z,pt)
                    return (cf*(1.+z*z)/(1.-z)*fq
                        +.5*(z*z+(1.-z)**2)*fg)/den
                return coupling(t)/(2.*np.pi)*quad(integrand,x,zmax(t),epsabs=1.e-7,limit=100)[0]
            ir,er=quad(lambda lt:rate(np.exp(lt)),lo,hi,epsabs=1.e-5,limit=100)
            f=pdf.pythia_pdf_xf(pid,x,cut)/pdf.pythia_pdf_xf(pid,x,hard)
            stored_leg=raw_table(cut,pid)/raw_table(hard,pid)
            def boundary_terms(t):
                pt=np.sqrt(t);z0=zmax(t)
                den=pdf.pythia_pdf_xf(pid,x,pt)
                fac=coupling(t)/(2.*np.pi)
                virtual_difference=cf*z0*(1.-z0)
                diagonal=quad(lambda z:cf*(1.+z*z)/(1.-z)*
                    (pdf.pythia_pdf_xf(pid,x/z,pt)/den-1.),z0,1.,epsabs=1.e-8)[0]
                gluon=quad(lambda z:.5*(z*z+(1.-z)**2)*
                    pdf.pythia_pdf_xf(21,x/z,pt)/den,z0,1.,epsabs=1.e-8)[0]
                return fac*np.array([virtual_difference,diagonal,gluon])
            boundary=np.array([quad(lambda lt:boundary_terms(np.exp(lt))[i],
                lo,hi,epsabs=1.e-6)[0] for i in range(3)])
            legs[str(pid)]=dict(pdf_ratio=f,backward_kernel_noemission=float(np.exp(-ir)),
                forward_times_pdf=float(f*forward),stored_times_pdf=float(f*stored_leg),
                log_difference=float(-np.log(f)+iv-ir),
                boundary_correction_terms=dict(zip(
                    ['virtual_prescription','omitted_diagonal_real_minus_virtual','omitted_gluon_to_quark_real'],
                    boundary.tolist())),
                boundary_correction_sum=float(boundary.sum()),
                boundary_vs_pdf_ratio_residual=float(-np.log(f)+iv-ir-boundary.sum()),
                backward_rate_integral=ir,outer_quad_estimated_absolute_error=er)
        events={}
        for flavour in [1,2,3]:
            l,r=legs[str(flavour)],legs[str(-flavour)]
            backward=l['backward_kernel_noemission']*r['backward_kernel_noemission']
            fresh=l['forward_times_pdf']*r['forward_times_pdf']
            archive=l['stored_times_pdf']*r['stored_times_pdf']
            events[str(flavour)]=dict(backward_kernel_probability=backward,
                forward_times_pdf_probability=fresh,archive_times_pdf_probability=archive,
                fresh_ratio_to_backward=fresh/backward,archive_ratio_to_backward=archive/backward,
                boundary_correction_terms={key:l['boundary_correction_terms'][key]+r['boundary_correction_terms'][key]
                    for key in l['boundary_correction_terms']},
                boundary_correction_sum=l['boundary_correction_sum']+r['boundary_correction_sum'])
        rows.append(dict(cut=cut,forward_kernel_probability=float(forward),
            stored_up_probability=stored,forward_rate_integral=iv,
            forward_quad_estimated_absolute_error=ev,zmax_at_cut=float(zmax(cut*cut)),
            legs=legs,events=events))
    data=dict(mass_GeV=mass,hard_scale_GeV=hard,x=x,pdf='Pythia13',alphaS_MZ=.130,
        alphaS_reference_MZ_GeV=coupling_mz,
        alphaS_order=1,CMW=False,pT0_GeV=0.,zmax_at_hard=float(zmax(hard*hard)),
        formulae=dict(forward_kernel='V1(zmax)=CF*(-2*log(1-zmax)-1.5*zmax^2)',
            forward_probability='exp(-integral dlog(t)*alphaS(t)/(2*pi)*V1(t))',
            backward_kernel='integral_x^zmax dz [Pqq(z)*xfq(x/z,t)+Pqg(z)*xfg(x/z,t)]/xfq(x,t)',
            quark_kernel='Pqq=CF*(1+z^2)/(1-z)',gluon_to_quark_kernel='Pqg=TR*(z^2+(1-z)^2)',
            phase_space='zmax=1-r/2*(sqrt(1+4/r)-1), r=t/mdip^2'),
        boundary_identity='Dlog(f)/Dlog(t)+alphaS*V1/(2*pi)-R_backward = alphaS/(2*pi)*'
            '{CF*Z*(1-Z) + integral_Z^1 dz Pqq(z)[xfq(x/z)/xfq(x)-1] '
            '+ integral_Z^1 dz Pqg(z)*xfg(x/z)/xfq(x)}, Z=zmax; assuming LO DGLAP evolution',
        rows=rows,
        interpretation='Illustrative rate-level calculation with the same finite-z kinematic bound '
            'and LO PDF provider. It numerically tests whether multiplying a PDF-free forward '
            'Sudakov by ordinary DGLAP PDF ratios reconstructs the finite-phase-space backward rate.',
        limitations=['Not a new full-shower sample',
            'Total light-quark PDF used; explicit Pythia valence/sea selection is not modeled',
            'Massless light-quark kinematics; randomized sister flavour/mass in forward code is omitted',
            'Single central Born point, not a replacement for the full Born ensemble',
            'Quadrature error is not the full shower/modeling uncertainty'],
        source_sha256={str(p):sha(p) for p in [Path(__file__),bridge,
            pythia/'src/SimpleSpaceShower.cc',pythia/'share/Pythia8/pdfdata/NNPDF23_lo_as_0130_qed_0000.dat']})
    filename='phase_space_quadrature.json' if args.hard_scale is None else f'phase_space_quadrature_Q{hard:g}.json'
    (out/filename).write_text(json.dumps(data,indent=2)+'\n')
    for r in rows:
        print(r['cut'],'forward',r['forward_kernel_probability'],'archive',r['stored_up_probability'],
            'uubar',r['events']['2'],flush=True)


if __name__=='__main__':main()
