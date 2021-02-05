      subroutine PAR_REN_%(mename)s(P,NHEL,hell,invariants)
C     compute the contribution due to the parameter renormalisation
      implicit none
      include 'nexternal.inc'
      double precision p(0:3,nexternal-1), ans_summed
      integer nhel(nexternal-1),hell
      double precision invariants(nexternal-1, nexternal-1)
      include 'coupl.inc'
      include '../../Source/MODEL/input.inc'

C     ipara = 1->AEWm1; 2->MZ; 3->MW; 4->MT/YMT; 5->MH
      integer imaxpara, ipara
      parameter (imaxpara=5)

      include 'orders.inc'
      double complex amp_split_ewsud_der(amp_split_size,imaxpara)
      common /to_amp_split_ewsud_der/ amp_split_ewsud_der

      integer i, j

      double precision xdelta, xsave, xup, xdn, matxup, matxdn, matxcen
      double complex amp_split_up(amp_split_size), amp_split_dn(amp_split_size), amp_split_cen(amp_split_size)

      DOUBLE COMPLEX AMP_SPLIT_EWSUD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_SPLIT_EWSUD/ AMP_SPLIT_EWSUD

      xdelta = 1d-5

C     check that mdl_mt and mdl_ymt are equal
      if (mdl_mt.ne.mdl_ymt) then
         write(*,*) 'ERROR mdl_mt and mdl_ymt should be equal!', mdl_mt, mdl_ymt
         stop 1
      endif

      do i = 1, imaxpara
        if (i.eq.1) then
          xsave = aewm1
        else if (i.eq.2) then
          xsave = mdl_mz
        else if (i.eq.3) then
          xsave = mdl_mw
        else if (i.eq.4) then
          xsave = mdl_mt
        else if (i.eq.5) then
          xsave = mdl_mh
        endif

        xup = xsave * (1d0 + xdelta)
        xdn = xsave * (1d0 - xdelta)

C        call %(mename)s(P,NHEL,%(hell)s matxcen)
C        amp_split_cen(:) = amp_split_ewsud(:)
        
C       vary the parameter up
        if (i.eq.1) then
          aewm1 = xup
        else if (i.eq.2) then
          mdl_mz = xup
        else if (i.eq.3) then
          mdl_mw = xup
        else if (i.eq.4) then
          mdl_mt = xup
        else if (i.eq.5) then
          mdl_mh = xup
        endif

        call coup()
        call %(mename)s(P,NHEL,%(hell)s matxup)
        amp_split_up(:) = amp_split_ewsud(:)

C       vary the parameter down
        if (i.eq.1) then
          aewm1 = xdn
        else if (i.eq.2) then
          mdl_mz = xdn
        else if (i.eq.3) then
          mdl_mw = xdn
        else if (i.eq.4) then
          mdl_mt = xdn
        else if (i.eq.5) then
          mdl_mh = xdn
        endif

        call coup()
        call %(mename)s(P,NHEL,%(hell)s matxdn)
        amp_split_dn(:) = amp_split_ewsud(:)

        amp_split_ewsud_der(:,i) = (amp_split_up(:)-amp_split_dn(:))/(2*xsave*xdelta)

C       reset the parameter 
        if (i.eq.1) then
          aewm1 = xsave
        else if (i.eq.2) then
          mdl_mz = xsave
        else if (i.eq.3) then
          mdl_mw = xsave
        else if (i.eq.4) then
          mdl_mt = xsave
        else if (i.eq.5) then
          mdl_mh = xsave
        endif

        call coup()

      enddo

      call get_par_ren_alphamz(invariants)

      return
      end




