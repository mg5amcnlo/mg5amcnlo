"""Shared binning and generated histogram-only native shower analysis."""
import math

FIELDS = ('primary','system_mass','system_y','object1_pt','object2_pt',
          'jet1_pt','jet2_pt','fiducial_primary','fiducial_pass','y23','y34',
          'y45','nDurhamJets','Cparameter','pair_mass','pair_pt',
          'pair_delta_phi','pair_delta_y','radiation_jet1_pt')
# name, bins, minimum, maximum, value index (None=total), log10, fiducial
HISTOGRAMS = [
    ('total',1,0.,1.,None,False,False),
    ('primary_0_100',100,0.,100.,0,False,False),
    ('primary_log10',140,-4.,3.,0,True,False),
    ('primary_0_500',100,0.,500.,0,False,False),
    ('system_mass',100,0.,2000.,1,False,False),
    ('system_rapidity',80,-8.,8.,2,False,False),
    ('object1_pt',100,0.,500.,3,False,False),
    ('object2_pt',100,0.,500.,4,False,False),
    ('jet1_pt',100,0.,500.,5,False,False),
    ('jet2_pt',100,0.,500.,6,False,False),
    ('y23_log10',100,-8.,0.,9,True,False),
    ('y34_log10',100,-8.,0.,10,True,False),
    ('y45_log10',100,-8.,0.,11,True,False),
    ('Durham_njets_ycut001',12,-.5,11.5,12,False,False),
    ('Cparameter',100,0.,1.,13,False,False),
    ('pair_mass',100,0.,1000.,14,False,False),
    ('pair_pt',100,0.,500.,15,False,False),
    ('pair_delta_phi',64,0.,math.pi,16,False,False),
    ('pair_delta_y',100,-10.,10.,17,False,False),
    ('radiation_jet1_pt',100,0.,500.,18,False,False),
    ('fiducial40_total',1,0.,1.,None,False,True),
    ('fiducial40_primary_0_100',100,0.,100.,7,False,True),
    ('fiducial40_primary_log10',140,-4.,3.,7,True,True),
    ('fiducial40_primary_0_500',100,0.,500.,7,False,True),
]
WEIGHTS = ('source','matched','veto','removal','source_S','source_H')
ADDED_IDS = ('macnlops_veto_only','macnlops_removal_only',
             'native_tc_source_S','native_tc_source_H')


def fortran_hook(code):
    lines=['c Generated histogram-only native-t_c analysis; no event streams.',
           '      subroutine RCLOS()', '      end', '',
           '      subroutine PYABEG(nnn,wwwi)',
           '      use HwU_wgts_info_len', "      include 'reweight0.inc'",
           '      integer nnn,i,nwgt_analysis,idxs,idxh,idxm',
           '      integer*8 counts(9)',
           '      common/c_analysis/nwgt_analysis,idxs,idxh,idxm',
           '      common/c_counts/counts',
           '      character*(wgts_info_len) wwwi(*),names(max_weight_shower)',
           '      idxs=0', '      idxh=0', '      idxm=0', '      counts=0',
           '      do i=1,nnn','         names(i)=wwwi(i)',
           "         if(index(wwwi(i),'native_tc_source_S').gt.0)idxs=i",
           "         if(index(wwwi(i),'native_tc_source_H').gt.0)idxh=i",
           "         if(trim(adjustl(wwwi(i))).eq.'macnlops')idxm=i",
           '      enddo',
           "      if(idxs.eq.0.or.idxh.eq.0)stop 'Missing source class weights'",
           '      nwgt_analysis=nnn','      call HwU_inithist(nnn,names)',
           '      call set_error_estimation(0)']
    for i,(name,n,lo,hi,_,__,___) in enumerate(HISTOGRAMS,1):
        lines.append(f"      call HwU_book({i},'{name} |T@NATIVE_TC',")
        lines.append(f'     $     {n},{lo:.17e}d0,{hi:.17e}d0)'.replace('e+00d0','d+00').replace('e+01d0','d+01').replace('e+02d0','d+02').replace('e+03d0','d+03').replace('e-01d0','d-01'))
    lines += ['      end','','      subroutine PYAEND(ievttot)',
              "      include 'HEPMC.INC'", '      double precision ievttot',
              '      integer*8 counts(9)', '      common/c_counts/counts',
              '      call finalize_histograms(nevhep)',
              "      open(unit=99,file='MADatNLO.HwU',status='replace')",
              '      call HwU_output(99,ievttot)', '      close(99)',
              "      open(unit=96,file='native_tc_counts.dat',status='replace')",
              "      write(96,'(9(i20,1x))') counts", '      close(96)', '      end','',
              '      subroutine PYANAL(nnn,xww)',
              '      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite',
              "      include 'HEPMC.INC'", "      include 'reweight0.inc'",
              '      integer nnn,i,nfinal,nwgt_analysis,status,idxs,idxh,idxm',
              '      integer ids(nmxhep),processcode',
              '      integer*8 counts(9)',
              '      common/c_analysis/nwgt_analysis,idxs,idxh,idxm',
              '      common/c_counts/counts',
              '      double precision xww(*),www(max_weight_shower)',
              '      double precision momenta(4,nmxhep),v(19)',
              f'      parameter(processcode={code})',
              "      if(xww(1).eq.0d0)stop 'Zero nominal diagnostic weight'",
              '      do i=1,nwgt_analysis', '         www(i)=evwgt*xww(i)/xww(1)',
              '      enddo', '      counts(1)=counts(1)+1',
              '      if(xww(idxs).ne.0d0)then',
              '         counts(2)=counts(2)+1',
              '         if(xww(idxs).lt.0d0)counts(4)=counts(4)+1',
              '      else if(xww(idxh).ne.0d0)then',
              '         counts(3)=counts(3)+1',
              '         if(xww(idxh).lt.0d0)counts(5)=counts(5)+1',
              '      else', "         stop 'Missing event source class'", '      endif',
              '      if(idxm.gt.0)then',
              '         if(xww(idxm).eq.0d0)counts(6)=counts(6)+1',
              '         if(xww(idxm).lt.0d0)then',
              '            if(xww(idxs).ne.0d0)counts(7)=counts(7)+1',
              '            if(xww(idxh).ne.0d0)counts(8)=counts(8)+1',
              '         endif', '      endif',
              '      nfinal=0', '      do i=1,nhep',
              '         if(isthep(i).ne.1)cycle', '         nfinal=nfinal+1',
              '         ids(nfinal)=idhep(i)', '         momenta(:,nfinal)=phep(1:4,i)',
              '      enddo',
              '      call native_tc_observables(nfinal,ids,momenta,',
              '     $     processcode,v,status)',
              "      if(status.ne.0)stop 'Native-tc observable failure'",
              '      if(v(9).gt.0.5d0)counts(9)=counts(9)+1']
    for i,(_,__,___,____,field,log,fid) in enumerate(HISTOGRAMS,1):
        if fid:lines.append('      if(v(9).gt.0.5d0)then')
        if field is None:lines.append(f'      call HwU_fill({i},0.5d0,www)')
        else:
            f=field+1
            lines.append(f'      if(ieee_is_finite(v({f})))then')
            if log:lines.append(f'         if(v({f}).gt.0d0)call HwU_fill({i},log10(v({f})),www)')
            else:lines.append(f'         call HwU_fill({i},v({f}),www)')
            lines.append('      endif')
        if fid:lines.append('      endif')
    lines += ['      call HwU_add_points','      end','']
    return '\n'.join(lines)


def durham_cut(njets):
    return '''      logical function dummy_cuts(p,istatus,ipdg)
      implicit none
      include 'nexternal.inc'
      double precision p(0:4,nexternal),q(0:3,nexternal)
      double precision qtot(0:3),s,a,b,c,y,ymin
      integer istatus(nexternal),ipdg(nexternal),i,j,n,ii,jj
      dummy_cuts=.false.
      n=0
      do i=nincoming+1,nexternal
         if(istatus(i).ne.1.or.p(0,i).le.0d0)cycle
         if(abs(ipdg(i)).gt.5.and.ipdg(i).ne.21)cycle
         n=n+1
         q(:,n)=p(0:3,i)
      enddo
      qtot=p(0:3,1)+p(0:3,2)
      s=qtot(0)**2-sum(qtot(1:3)**2)
      if(s.le.0d0)return
 10   if(n.lt.NJETS)return
      ymin=1d99
      ii=0
      jj=0
      do i=1,n
         do j=i+1,n
            a=sqrt(sum(q(1:3,i)**2))
            b=sqrt(sum(q(1:3,j)**2))
            if(a.gt.0d0.and.b.gt.0d0)then
               c=sum(q(1:3,i)*q(1:3,j))/(a*b)
            else if(a.gt.0d0)then
               c=q(3,i)/a
            else if(b.gt.0d0)then
               c=q(3,j)/b
            else
               c=1d0
            endif
            c=max(-1d0,min(1d0,c))
            y=2d0*min(q(0,i)**2,q(0,j)**2)*(1d0-c)/s
            if(y.lt.ymin)then
               ymin=y
               ii=i
               jj=j
            endif
         enddo
      enddo
      if(ymin.ge.0.01d0)then
         dummy_cuts=.true.
         return
      endif
      if(ii.eq.0.or.jj.eq.0)return
      q(:,ii)=q(:,ii)+q(:,jj)
      do i=jj,n-1
         q(:,i)=q(:,i+1)
      enddo
      n=n-1
      goto 10
      end
'''.replace('NJETS',str(njets))
