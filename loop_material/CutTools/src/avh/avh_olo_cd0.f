************************************************************************
* This is the file  avh_olo_cd0.f  of the package                      *
*                                                                      *
*                               OneLOop                                *
*                                                                      *
* for the evaluation of 1-loop scalar 1-, 2-, 3- and 4-point functions *
*                                                                      *
* author: Andreas van Hameren <hamerenREMOVETHIS@ifj.edu.pl>           *
*   date: 09-09-2010                                                   *
************************************************************************
*                                                                      *
* Have a look at the file  avh_olo_hello.f  for more information.      *
*                                                                      *
************************************************************************

      subroutine avh_olo_cd0( rslt ,pp_in,mm_in ,ap_in )
*  ********************************************************************
*  * Finite 1-loop scalar 4-point function for complex internal masses
*  * Based on the formulas from
*  *   Dao Thi Nhung and Le Duc Ninh, arXiv:0902.0325 [hep-ph]
*  *   G. 't Hooft and M.J.G. Veltman, Nucl.Phys.B153:365-401,1979 
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,pp_in(6),mm_in(4) ,pp(6),mm(4)
      double precision ap_in(6) ,ap(6) ,aptmp(6)
     &,rem,imm,hh,small,avh_olo_prec
      double complex
     & a,b,c,d,e,f,g,h,j,k,x1,x2,sdnt,o1,j1,e1,one,zero
     &,avh_olo_tfun,avh_olo_t13fun,trm1,trm2,trm3
      parameter( zero=(0d0,0d0) ,one=(1d0,0d0) )
      integer
     & icase,ii,jj,ll(6),lp(6,3),lm(4,3),base(4)
     &,avh_olo_un_get
      data base/8,4,2,1/
      data lp/1,2,3,4,5,6 ,5,2,6,4,1,3 ,1,6,3,5,4,2/
      data lm/1,2,3,4     ,1,3,2,4     ,1,2,4,3    /
*
c      write(6,*) 'MESSAGE from avh_olo_cd0: you are calling me' !DEBUG
*
      rslt(0) = zero
      rslt(1) = zero
      rslt(2) = zero
*
      small = avh_olo_prec()
*
      hh = 0d0
      do ii=1,6
        aptmp(ii) = ap_in(ii)
        if (aptmp(ii).gt.hh) hh = aptmp(ii)
      enddo
      hh = 1d2*small*hh
      do ii=1,6
        if (aptmp(ii).lt.hh) aptmp(ii) = 0d0
      enddo
*
      if (aptmp(5).eq.0d0.or.aptmp(6).eq.0d0) then
        if (aptmp(1).eq.0d0.or.aptmp(3).eq.0d0) then
          if (aptmp(2).eq.0d0.or.aptmp(4).eq.0d0) then
            if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &        'ERROR in avh_olo_cd0: no choice with non-zero s and t'
          else
            jj = 3
          endif
        else
          jj = 2
        endif
      else
        jj = 1
      endif
      do ii=1,6
        ap(ii) = aptmp(lp(ii,jj))
        if (ap(ii).gt.0d0) then
          pp(ii) = pp_in(lp(ii,jj))
        else
          pp(ii) = zero
        endif
      enddo
      do ii=1,4
        rem = dreal(mm_in(lm(ii,jj)))
        imm = dimag(mm_in(lm(ii,jj)))
        hh = small*dabs(rem)
        if (dabs(imm).lt.hh) imm = -hh
        mm(ii) = dcmplx(rem,imm)
      enddo
*
      icase = 0
      do ii=1,4
        if (ap(ii).gt.0d0) icase = icase + base(ii)
      enddo
*
      if (icase.lt.15) then
* at least one exernal mass equal zero
        call avh_olo_d0per(icase,ll)
        if (icase.eq.0.or.icase.eq.1.or.icase.eq.5) then
* * two opposite masses equal zero
          a = pp(ll(5)) - pp(ll(1))
          c = pp(ll(4)) - pp(ll(5)) - pp(ll(3))
          g = pp(ll(2))
          h = pp(ll(6)) - pp(ll(2)) - pp(ll(3))
          d = mm(ll(3)) - mm(ll(4)) - pp(ll(3))
          e = mm(ll(1)) - mm(ll(3)) + pp(ll(3)) - pp(ll(4))
          f = mm(ll(4))
          j = mm(ll(2)) - mm(ll(3)) - pp(ll(6)) + pp(ll(3))
          trm1 = avh_olo_t13fun( a,c,g,h ,d,e,f,j )
          rslt(0) = trm1
        else
          a = pp(ll(3))
          b = pp(ll(2))
          c = pp(ll(6)) - pp(ll(2)) - pp(ll(3))
          h = pp(ll(4)) - pp(ll(5)) - pp(ll(6)) + pp(ll(2))
          j = pp(ll(5)) - pp(ll(1)) - pp(ll(2))
          d = mm(ll(3)) - mm(ll(4)) - pp(ll(3))
          e = mm(ll(2)) - mm(ll(3)) - pp(ll(6)) + pp(ll(3))
          k = mm(ll(1)) - mm(ll(2)) + pp(ll(6)) - pp(ll(4))
          f = mm(ll(4))
          trm1 = avh_olo_tfun( a,b,c,h,j ,d,e,f,k )
          trm2 = avh_olo_tfun( a,b+j,c+h,h,j ,d,e+k,f,k )
          rslt(0) = trm1 - trm2
        endif
c        write(6,*) 'At least one external mass zero' !DEBUG
c        write(6,*) "a,b,c=", dreal(a),dreal(b),dreal(c) !DEBUG
c        write(6,*) "g,h,j=", dreal(g),dreal(h),dreal(j) !DEBUG
c        write(6,*) "d=", d !DEBUG
c        write(6,*) "e=", e !DEBUG
c        write(6,*) "k=", k !DEBUG
c        write(6,*) "f=", f !DEBUG
c        write(6,*) 'cd0',trm1 !DEBUG
c        write(6,*) 'cd0',trm2 !DEBUG
      else
* no extenal mass equal zero
        if     ( dreal((pp(5)-pp(1)-pp(2))**2-4*pp(1)*pp(2)) .gt. 0d0 )
     &  then !12
          icase = 0 ! no permutation
        elseif ( dreal((pp(6)-pp(2)-pp(3))**2-4*pp(2)*pp(3)) .gt. 0d0 )
     &  then !23
          icase = 8 ! 1 cyclic permutation
        elseif ( dreal((pp(4)-pp(5)-pp(3))**2-4*pp(5)*pp(3)) .gt. 0d0 )
     &  then !34
          icase = 4 ! 2 cyclic permutations
        elseif ( dreal((pp(4)-pp(1)-pp(6))**2-4*pp(1)*pp(6)) .gt. 0d0 )
     &  then !41
          icase = 2 ! 3 cyclic permutations
        else
          if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &      'ERROR in avh_olo_cd0: no positive lambda, returning 0'
          return
        endif
        call avh_olo_d0per(icase,ll)
        a = pp(ll(3))
        b = pp(ll(2))
        g = pp(ll(1))
        c = pp(ll(6)) - pp(ll(2)) - pp(ll(3))
        h = pp(ll(4)) - pp(ll(5)) - pp(ll(6)) + pp(ll(2))
        j = pp(ll(5)) - pp(ll(1)) - pp(ll(2))
        d = mm(ll(3)) - mm(ll(4)) - pp(ll(3))
        e = mm(ll(2)) - mm(ll(3)) - pp(ll(6)) + pp(ll(3))
        k = mm(ll(1)) - mm(ll(2)) + pp(ll(6)) - pp(ll(4))
        f = mm(ll(4))
c        write(6,*) "a,b,c=", dreal(a),dreal(b),dreal(c) !DEBUG
c        write(6,*) "g,h,j=", dreal(g),dreal(h),dreal(j) !DEBUG
c        write(6,*) "d=", d !DEBUG
c        write(6,*) "e=", e !DEBUG
c        write(6,*) "k=", k !DEBUG
c        write(6,*) "f=", f !DEBUG
        call avh_olo_abc( x1,x2 ,sdnt ,g,j,b ,0 )
        if (dimag(sdnt).ne.0d0) then
          if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &      'ERROR in avh_olo_cd0: no real solution for alpha'
     &     ,', returning 0'
          return
        endif
!BAD        if (dabs(dreal(x1)).gt.dabs(dreal(x2))) then
        if (dabs(dreal(x1)).lt.dabs(dreal(x2))) then !BETTER
          sdnt = x1
          x1 = x2
          x2 = sdnt
        endif
        o1 = one-x1
        j1 = j+2*g*x1
        e1 = e+k*x1
        trm1 = avh_olo_tfun( a+b+c,g,j+h,c+2*b+(h+j)*x1,j1 ,d+e,k,f,e1 )
        trm2 = avh_olo_tfun( a,b+g+j,c+h,c+h*x1,o1*j1 ,d,e+k,f,e1 )
        trm3 = avh_olo_tfun( a,b,c,c+h*x1,-j1*x1 ,d,e,f,e1 )
        rslt(0) = -trm1 + o1*trm2 + x1*trm3
!        write(6,*) '  -trm1',  -trm1 !DEBUG
!        write(6,*) 'o1*trm2',o1*trm2 !DEBUG
!        write(6,*) 'x1*trm3',x1*trm3 !DEBUG
      endif
      end


      double complex function avh_olo_t13fun( aa,cc,gg,hh ,dd,ee,ff,jj )
*  ********************************************************************
*  * /1   /x                             y
*  * | dx |  dy -----------------------------------------------------
*  * /0   /0    (gy^2 + hxy + dx + jy + f)*(ax^2 + cxy + dx + ey + f)
*  *
*  * jj should have negative imaginary part
*  ********************************************************************
      implicit none
      double complex aa,cc,dd,ee,ff,gg,hh,jj
      double complex
     & kk,ll,nn,y1,y2,sdnt,rslt,ieps,one,zero
     &,avh_olo_s3fun
      parameter( zero=(0d0,0d0) ,one=(1d0,0d0) )
      double precision
     & small
     &,avh_olo_prec
*
c      write(6,*) 'MESSAGE from avh_olo_t13fun: you are calling me' !DEBUG
*
      small = avh_olo_prec()**2
      ieps = dcmplx(0d0,small*dabs(dreal(ff)))
*
      kk = hh*aa - cc*gg
      ll = aa*dd + hh*ee - dd*gg - cc*jj
      nn = dd*(ee - jj) + (hh - cc)*(ff-ieps)
      call avh_olo_abc( y1,y2 ,sdnt ,kk,ll,nn ,0 )
*
      rslt =      - avh_olo_s3fun( y1,y2 ,zero,one ,aa   ,ee+cc,dd+ff )
      rslt = rslt + avh_olo_s3fun( y1,y2 ,zero,one ,gg   ,jj+hh,dd+ff )
      rslt = rslt - avh_olo_s3fun( y1,y2 ,zero,one ,gg+hh,dd+jj,ff )
      rslt = rslt + avh_olo_s3fun( y1,y2 ,zero,one ,aa+cc,ee+dd,ff )
*
      avh_olo_t13fun = rslt/kk
      end


      double complex function avh_olo_t1fun( aa,cc,gg,hh ,dd,ee,ff,jj )
*  ********************************************************************
*  * /1   /x                         1
*  * | dx |  dy ----------------------------------------------
*  * /0   /0    (g*x + h*x + j)*(a*x^2 + c*xy + d*x + e*y + f)
*  *
*  * jj should have negative imaginary part
*  ********************************************************************
      implicit none
      double complex aa,cc,gg,hh ,dd,ee,ff,jj
      double complex
     & kk,ll,nn,y1,y2,sdnt,rslt,ieps,one,zero
     &,avh_olo_s3fun
      parameter( zero=(0d0,0d0) ,one=(1d0,0d0) )
      double precision
     & small
     &,avh_olo_prec
*
c      write(6,*) 'MESSAGE from avh_olo_t1fun: you are calling me' !DEBUG
*
      small = avh_olo_prec()**2
      ieps = dcmplx(0d0,small*dabs(dreal(ff)))
*
      kk = hh*aa - cc*gg
      ll = hh*dd - cc*jj - ee*gg
      nn = hh*(ff-ieps) - ee*jj
      call avh_olo_abc( y1,y2 ,sdnt ,kk,ll,nn ,0 )
*
      rslt =      - avh_olo_s3fun( y1,y2 ,zero,one ,aa+cc,dd+ee,ff )
      rslt = rslt + avh_olo_s3fun( y1,y2 ,zero,one ,zero ,gg+hh,jj )
      rslt = rslt - avh_olo_s3fun( y1,y2 ,zero,one ,zero ,gg   ,jj )
      rslt = rslt + avh_olo_s3fun( y1,y2 ,zero,one ,aa   ,dd   ,ff )
*
      avh_olo_t1fun = rslt/kk
      end


      double complex function avh_olo_tfun( aa,bb,cc ,gin,hin
     &                                     ,dd,ee,ff ,jin )
*  ********************************************************************
*  * /1   /x                             1
*  * | dx |  dy ------------------------------------------------------
*  * /0   /0    (g*x + h*x + j)*(a*x^2 + b*y^2 + c*xy + d*x + e*y + f)
*  ********************************************************************
      implicit none
      double complex aa,bb,cc ,gin,hin ,dd,ee,ff,jin
      double complex
     & gg,hh,jj,zz(2),beta,tmpa(2),tmpb(2),tmpc(2),kiz(2),ll,nn,kk,y1,y2
     &,yy(2,2),sdnt,rslt,ieps,one,zero
     &,avh_olo_t1fun,avh_olo_s3fun,avh_olo_plnr
      parameter( zero=(0d0,0d0) ,one=(1d0,0d0) )
      double precision
     & sj,ab1,ab2,ac1,ac2,abab,acac,abac,det,ap1,ap2,apab,apac
     &,x1(2,2),x2(2,2),xmin,small
     &,avh_olo_prec
      integer
     & iz,iy,izmin
      logical
     & pp(2,2),p1,p2
*
c      write(6,*) 'MESSAGE from avh_olo_tfun: you are calling me' !DEBUG
*
      sj = dimag(jin)
      if (sj.eq.0d0) then
        sj = -1d0
      else
        sj = dsign(1d0,dimag(jin))
      endif
      gg = -sj*gin
      hh = -sj*hin
      jj = -sj*jin
*
      if     (bb.eq.zero) then
        avh_olo_tfun = -sj*avh_olo_t1fun( aa,cc,gg,hh ,dd,ee,ff,jj )
        return
      elseif (aa.eq.zero) then
        avh_olo_tfun = -sj*avh_olo_t1fun( bb+cc,-cc,-gg-hh,gg
     &               ,-dd-ee-2*(bb+cc),dd+cc,dd+ee+bb+cc+ff,gg+hh+jj )
        return
      endif
*
      small = avh_olo_prec()**2
      ieps = dcmplx(0d0,small*dabs(dreal(ff)))
*
      call avh_olo_abc( zz(1),zz(2) ,sdnt ,bb,cc,aa ,0 )
      if (cdabs(zz(1)).gt.cdabs(zz(2))) then
        beta = zz(1)
        zz(1) = zz(2)
        zz(2) = beta
      endif
*
      do iz=1,2
        beta = zz(iz)
        tmpa(iz) = gg + beta*hh
        tmpb(iz) = cc + 2*beta*bb
        tmpc(iz) = dd + beta*ee
        kiz(iz) =        bb*tmpa(iz)               - hh*tmpb(iz)
        ll      =        ee*tmpa(iz) - hh*tmpc(iz) - jj*tmpb(iz)
        nn      = (ff-ieps)*tmpa(iz) - jj*tmpc(iz)
        call avh_olo_abc( yy(iz,1),yy(iz,2) ,sdnt ,kiz(iz),ll,nn ,0 )
        if (dabs(dimag(beta)).ne.0d0) then
          ab1 = dreal(-beta)
          ab2 = dimag(-beta)
          ac1 = ab1+1d0 !dreal(one-beta)
          ac2 = ab2     !dimag(one-beta)
          abab = ab1*ab1 + ab2*ab2
          acac = ac1*ac1 + ac2*ac2
          abac = ab1*ac1 + ab2*ac2
          det = abab*acac - abac*abac
          do iy=1,2
            ap1 = dreal(yy(iz,iy))
            ap2 = dimag(yy(iz,iy))
            apab = ap1*ab1 + ap2*ab2
            apac = ap1*ac1 + ap2*ac2
            x1(iz,iy) = ( acac*apab - abac*apac )/det
            x2(iz,iy) = (-abac*apab + abab*apac )/det
          enddo
        else
          do iy=1,2
            x1(iz,iy) = -1d0
            x2(iz,iy) = -1d0
          enddo
        endif
      enddo
      xmin = 1d0
      izmin = 2
      do iz=1,2
      do iy=1,2
        if ( x1(iz,iy).ge.0d0.and.x2(iz,iy).ge.0d0
     &      .and.x1(iz,iy)+x2(iz,iy).le.1d0 ) then
          pp(iz,iy) = .true.
          if (x1(iz,iy).lt.xmin) then
            xmin = x1(iz,iy)
            izmin = iz
          endif
          if (x2(iz,iy).lt.xmin) then
            xmin = x2(iz,iy)
            izmin = iz
          endif
        else
          pp(iz,iy) = .false.
        endif
      enddo
      enddo
      iz = izmin+1
      if (iz.eq.3) iz = 1
*
      beta = zz(iz)
!      write(6,*) '-----> beta  ',beta !DEBUG
      kk = kiz(iz)
      y1 = yy(iz,1)
      y2 = yy(iz,2)
      p1 = pp(iz,1)
      p2 = pp(iz,2)
*
      rslt =
     & + avh_olo_s3fun( y1,y2 ,beta ,one      ,zero    ,hh   ,gg+jj )
     & - avh_olo_s3fun( y1,y2 ,zero ,one-beta ,zero    ,gg+hh,   jj )
     & + avh_olo_s3fun( y1,y2 ,zero ,   -beta ,zero    ,gg   ,   jj )
     & - avh_olo_s3fun( y1,y2 ,beta ,one      ,bb      ,cc+ee,aa+dd+ff )
     & + avh_olo_s3fun( y1,y2 ,zero ,one-beta ,aa+bb+cc,dd+ee,ff       )
     & - avh_olo_s3fun( y1,y2 ,zero ,   -beta ,aa      ,dd   ,ff       )
!      sdnt= avh_olo_s3fun(y1,y2,beta ,one      ,zero    ,hh   ,gg+jj )   !DEBUG
!      write(6,*) 'tfun 1',sdnt                                           !DEBUG
!      sdnt=-avh_olo_s3fun(y1,y2,zero ,one-beta ,zero    ,gg+hh,   jj )   !DEBUG
!      write(6,*) 'tfun 2',sdnt                                           !DEBUG
!      sdnt= avh_olo_s3fun(y1,y2,zero ,   -beta ,zero    ,gg   ,   jj )   !DEBUG
!      write(6,*) 'tfun 3',sdnt                                           !DEBUG
!      sdnt=-avh_olo_s3fun(y1,y2,beta ,one      ,bb      ,cc+ee,aa+dd+ff) !DEBUG
!      write(6,*) 'tfun 4',sdnt                                           !DEBUG
!      sdnt= avh_olo_s3fun(y1,y2,zero ,one-beta ,aa+bb+cc,dd+ee,ff      ) !DEBUG
!      write(6,*) 'tfun 5',sdnt                                           !DEBUG
!      sdnt=-avh_olo_s3fun(y1,y2,zero ,   -beta ,aa      ,dd   ,ff      ) !DEBUG
!      write(6,*) 'tfun 6',sdnt                                           !DEBUG
!      write(6,*) 'tfun rslt',rslt                                        !DEBUG
*
      sdnt = avh_olo_plnr( y1,y2 ,p1,p2, tmpa(iz),tmpb(iz),tmpc(iz) )
!      write(6,*) '-----> plnr',p1,p2,sdnt !DEBUG
!WRONG      rslt = rslt + sdnt
      if (dimag(beta).le.0d0) then !RIGHT
        rslt = rslt + sdnt         !RIGHT
      else                         !RIGHT
        rslt = rslt - sdnt         !RIGHT
      endif                        !RIGHT
*
      avh_olo_tfun = -sj*rslt/kk
      end


      double complex function avh_olo_s3fun( y1i,y2i ,dd,ee ,aa,bb,cin )
*  ********************************************************************
*  * Calculate
*  *            ( S3(y1i) - S3(y2i) )/( y1i - y2i )
*  * where
*  *               /1    ee * ln( aa*x^2 + bb*x + cc )
*  *       S3(y) = |  dx -----------------------------
*  *               /0           ee*x - y - dd
*  *
*  * y1i,y2i should have a non-zero imaginary part
*  ********************************************************************
      implicit none
      double complex y1i,y2i ,dd,ee ,aa,bb,cc,cin
      double complex
     & y1,y2,fy1y2,qq,z1,z2,rslt,zero,one
     &,avh_olo_logc,avh_olo_r1fun,avh_olo_r0fun
!     &,tmp !DEBUG
      parameter( zero=(0d0,0d0) ,one=(1d0,0d0) )
      integer
     & iq,ieta,avh_olo_un_get,avh_olo_eta3
      double precision
     & rea,reb,rez1,rez2,imz1,imz2,small,thrs,simc,twopi
     &,avh_olo_prec,avh_olo_pi
      logical init
      data init/.true./
      save init,small,thrs,twopi
      if (init) then
        init = .false.
        small = avh_olo_prec()**2
        if (avh_olo_prec().gt.1d-24) then
          thrs = 1d3*avh_olo_prec()
        else
          thrs = 1d6*avh_olo_prec()
        endif
        twopi = 2*avh_olo_pi()
      endif
*
!      write(6,*) 'tfun y1i,y2i',y1i,y2i!DEBUG
!      write(6,*) 'tfun dd     ',dd     !DEBUG
!      write(6,*) 'tfun ee     ',ee     !DEBUG
!      write(6,*) 'tfun aa     ',aa     !DEBUG
!      write(6,*) 'tfun bb     ',bb     !DEBUG
!      write(6,*) 'tfun cin    ',cin    !DEBUG
      if (ee.eq.zero) then
        avh_olo_s3fun = zero
        return
      endif
*
      cc = cin
      rea = cdabs(aa)
      reb = cdabs(bb)
      simc = cdabs(cc)
      if (simc.lt.thrs*min(rea,reb)) cc = zero
*
      simc = dimag(cc)
      if (simc.eq.0d0) then
        simc = dimag(bb)
        if (simc.eq.0d0) simc = -1d0
      endif
      simc = dsign(1d0,simc)
*
      y1 = (dd+y1i)/ee
      y2 = (dd+y2i)/ee
      if (dimag(y1).eq.0d0) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_s3fun: y1 has zero imaginary part'
      endif
      if (dimag(y2).eq.0d0) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_s3fun: y2 has zero imaginary part'
      endif
      fy1y2 = avh_olo_r0fun( y1,y2 )
*
      if     (aa.ne.zero) then
*
        call avh_olo_abc( z1,z2 ,qq ,aa,bb,cc ,0 )
        rea  = dsign(1d0,dreal(aa))
        rez1 = dreal(z1)
        rez2 = dreal(z2) 
        imz1 = dimag(z1) ! sign(Im(a*z1*z2)) = simc
        imz2 = dimag(z2)
        if(imz1.eq.0d0) imz1 = simc*rea*dsign(1d0,rez2)*dabs(small*rez1)
        if(imz2.eq.0d0) imz2 = simc*rea*dsign(1d0,rez1)*dabs(small*rez2)
        z1 = dcmplx( rez1,imz1 )
        z2 = dcmplx( rez2,imz2 )
        ieta = avh_olo_eta3( -z1,-imz1 ,-z2,-imz2 ,zero,simc*rea )
        call avh_olo_conv( qq,iq ,aa,simc )
        rslt = fy1y2 * ( avh_olo_logc( qq,iq )
     &                   + dcmplx(0d0,twopi*dble(ieta)) )
     &       + avh_olo_r1fun( z1,y1,y2,fy1y2 )
     &       + avh_olo_r1fun( z2,y1,y2,fy1y2 )
!         tmp = fy1y2 * ( avh_olo_logc( qq,iq )           !DEBUG
!     &                  + dcmplx(0d0,twopi*dble(ieta)) ) !DEBUG
!         write(6,*) 'tfun s3fun 1',tmp                   !DEBUG
!         tmp = avh_olo_r1fun( z1,y1,y2,fy1y2 )           !DEBUG
!         write(6,*) 'tfun s3fun 2',tmp                   !DEBUG
!         tmp = avh_olo_r1fun( z2,y1,y2,fy1y2 )           !DEBUG
!         write(6,*) 'tfun s3fun 3',tmp                   !DEBUG
!         write(6,*) 'tfun s3fun rslt',rslt               !DEBUG
c        write(6,*) 's3fun             a,b,c',aa,bb,cc !DEBUG
c        write(6,*) 's3fun ',fy1y2 *  avh_olo_logc( qq,iq ) !DEBUG
c        write(6,*) 's3fun with aa=/=0      ',rslt,z1,z2,ieta !DEBUG
*
      elseif (bb.ne.zero) then
*
        z1 = -cc/bb ! - i|eps|Re(b)
        reb  = dreal(bb)
        rez1 = dreal(z1)
        imz1 = dimag(z1)
        if (dabs(imz1).eq.0d0) then
          imz1 = -simc*reb*dabs(small*rez1/reb)
          z1 = dcmplx( rez1,imz1 )
        endif
        call avh_olo_conv( qq,iq ,bb,simc )
        ieta = avh_olo_eta3( bb,simc ,-z1,-imz1 ,cc,simc )
        rslt = fy1y2 * ( avh_olo_logc( qq,iq )
     &                   + dcmplx(0d0,twopi*dble(ieta)) )
     &       + avh_olo_r1fun( z1,y1,y2,fy1y2 )
c        write(6,*) 's3fun               b,c',bb,cc !DEBUG
c        write(6,*) 's3fun ',fy1y2 *  avh_olo_logc( qq,iq ) !DEBUG
c        write(6,*) 's3fun with aa=0, bb=/=0',rslt,z1,ieta !DEBUG
*
      elseif (cc.ne.zero) then
*
        call avh_olo_conv( qq,iq ,cc,simc )
        rslt = avh_olo_logc( qq,iq )*fy1y2
*
      else!if (aa=bb=cc=0)
*
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_s3fun: cc equal zero, returning 0'
        rslt = zero
*
      endif
*
      avh_olo_s3fun = rslt/ee
      end


      double complex function avh_olo_r1fun( zz,y1,y2,fy1y2 )
*  ********************************************************************
*  * calculates  ( R1(y1,z) - R1(y2,z) )/( y1 - y2 )
*  * where
*  *                          /     / 1-y \       / 1-z \ \
*  *      R1(y,z) = ln(y-z) * | log |-----| - log |-----| |
*  *                          \     \ -y  /       \ -z  / / 
*  *
*  *                      /    y-z \       /    y-z \
*  *                - Li2 |1 - ----| + Li2 |1 - ----|
*  *                      \    -z  /       \    1-z /
*  *
*  *                                     / 1-y1 \       / 1-y2 \
*  *                                 log |------| - log |------| 
*  * input fy1y2 should be equal to      \  -y1 /       \  -y2 /
*  *                                 ---------------------------
*  *                                           y1 - y2
*  ********************************************************************
      implicit none
      double complex y1,y2,zz,fy1y2
      double complex
     & q1z,q2z,qq,rslt,q1,q2,q3,q4,oz,one,zero,trm
     &,avh_olo_logc,avh_olo_r0fun,avh_olo_logc2,avh_olo_li2c2
     &,avh_olo_li2c
!     &,hlp1,hlp2,hlp3,hlp4,hlp5 !DEBUG
      parameter( zero=(0d0,0d0) ,one=(1d0,0d0) )
      double precision
     & h12,hz1,hz2,hzz,hoz,pi
     &,avh_olo_pi
      integer
     & i1z,i2z,iq,i1,i2,i3,i4
      logical
     & zzsmall,ozsmall
*
      pi = avh_olo_pi()
*
      oz = one-zz
      h12 = cdabs(y1-y2)
      hz1 = cdabs(y1-zz)
      hz2 = cdabs(y2-zz)
      hzz = cdabs(zz)
      hoz = cdabs(oz)
      call avh_olo_conv( q1z,i1z ,y1-zz,0d0 )
      call avh_olo_conv( q2z,i2z ,y2-zz,0d0 )
*
!      write(6,*) 'y1-z',y1-zz !DEBUG
!      write(6,*) 'y2-z',y2-zz !DEBUG
!      write(6,*) '   z',zz    !DEBUG
!      write(6,*) ' 1-z',oz    !DEBUG
      zzsmall = .false.
      ozsmall = .false.
      if     (hzz.lt.hz1.and.hzz.lt.hz2.and.hzz.lt.hoz) then ! |z| < |y1-z|,|y2-z|
        zzsmall = .true.
        call avh_olo_rat( q1,i1 ,q1z,i1z ,q2z,i2z ) ! q1 = (y1-z)/(y2-z)
        call avh_olo_prd( q2,i2 ,q1z,i1z ,q2z,i2z ) ! q2 = (y1-z)*(y2-z)
        call avh_olo_conv( q3,i3 ,(y2-one)/y2,0d0 ) ! q3 = (y2-1)/y2
        call avh_olo_conv( q4,i4 ,oz,0d0 )          ! q4 = 1-z
        rslt = fy1y2*avh_olo_logc( q1z,i1z )
     &       - ( avh_olo_logc( q2,i2 )/2
     &          +avh_olo_logc( q3,i3 )
     &          -avh_olo_logc( q4,i4 ) )*avh_olo_logc2( q1,i1 )/(y2-zz)
c        write(6,*) 'r1fun 1',rslt !DEBUG
!        hlp1 = fy1y2*avh_olo_logc( q1z,i1z ) !DEBUG
!        hlp2 = avh_olo_logc( q2,i2 )/2 !DEBUG
!        hlp3 = avh_olo_logc( q3,i3 ) !DEBUG
!        hlp4 =-avh_olo_logc( q4,i4 )  !DEBUG
!        hlp5 = -(hlp2+hlp3+hlp4)*avh_olo_logc2( q1,i1 )/(y2-zz) !DEBUG
!        write(6,*) 'r1fun 1' !DEBUG
!     &    ,(hlp2+hlp3+hlp4)/(cdabs(hlp2)+cdabs(hlp3)+cdabs(hlp4)) !DEBUG
!        write(6,*) 'r1fun 1',(hlp1+hlp5)/(cdabs(hlp1)+cdabs(hlp5)) !DEBUG
      elseif (hoz.lt.hz1.and.hoz.lt.hz2) then ! |1-z| < |y1-z|,|y2-z|
        ozsmall = .true.
        call avh_olo_rat( q1,i1 ,q1z,i1z ,q2z,i2z ) ! q1 = (y1-z)/(y2-z)
        call avh_olo_prd( q2,i2 ,q1z,i1z ,q2z,i2z ) ! q2 = (y1-z)*(y2-z)
        call avh_olo_conv( q3,i3 ,(y2-one)/y2,0d0 ) ! q3 = (y2-1)/y2
        call avh_olo_conv( q4,i4 ,-zz,0d0 )         ! q4 = -z
        rslt = fy1y2*avh_olo_logc( q1z,i1z )
     &       - (-avh_olo_logc( q2,i2 )/2
     &          +avh_olo_logc( q3,i3 )
     &          +avh_olo_logc( q4,i4 ) )*avh_olo_logc2( q1,i1 )/(y2-zz)
c        write(6,*) 'r1fun 2',rslt !DEBUG
!        hlp1 = fy1y2*avh_olo_logc( q1z,i1z ) !DEBUG
!        hlp2 = avh_olo_logc( q2,i2 )/2 !DEBUG
!        hlp3 = avh_olo_logc( q3,i3 ) !DEBUG
!        hlp4 =-avh_olo_logc( q4,i4 )  !DEBUG
!        hlp5 = -(hlp2+hlp3+hlp4)*avh_olo_logc2( q1,i1 )/(y2-zz) !DEBUG
!        write(6,*) 'r1fun 2' !DEBUG
!     &    ,(hlp2+hlp3+hlp4)/(cdabs(hlp2)+cdabs(hlp3)+cdabs(hlp4)) !DEBUG
!        write(6,*) 'r1fun 2',(hlp1+hlp5)/(cdabs(hlp1)+cdabs(hlp5)) !DEBUG
      elseif (h12.le.hz2.and.hz2.le.hz1) then ! |y1-y2| < |y2-z| < |y1-z|
        call avh_olo_rat( qq,iq, q1z,i1z ,q2z,i2z ) ! qq = (y1-z)/(y2-z)
        rslt = fy1y2*avh_olo_logc( q1z,i1z )
     &       - avh_olo_r0fun( y2,zz )*avh_olo_logc2( qq,iq )        
!        hlp1 = fy1y2*avh_olo_logc( q1z,i1z ) !DEBUG
!        hlp2 = -avh_olo_r0fun( y2,zz )*avh_olo_logc2( qq,iq ) !DEBUG
!        write(6,*) 'r1fun3',(hlp1+hlp2)/(cdabs(hlp1)+cdabs(hlp2)) !DEBUG
c        write(6,*) 'r1fun 3',rslt !DEBUG
      elseif (h12.le.hz1.and.hz1.le.hz2) then ! |y1-y2| < |y2-z| < |y1-z|
        call avh_olo_rat( qq,iq, q2z,i2z ,q1z,i1z )! qq = (y2-z)/(y1-z)
        rslt = fy1y2*avh_olo_logc( q2z,i2z )
     &       - avh_olo_r0fun( y1,zz )*avh_olo_logc2( qq,iq )        
!        hlp1 = fy1y2*avh_olo_logc( q2z,i2z ) !DEBUG
!        hlp2 = -avh_olo_r0fun( y1,zz )*avh_olo_logc2( qq,iq ) !DEBUG
!        write(6,*) 'r1fun4',(hlp1+hlp2)/(cdabs(hlp1)+cdabs(hlp2)) !DEBUG
c        write(6,*) 'r1fun 4',rslt !DEBUG
      else!if(hz1.lt.h12.or.hz2.lt.h12) then ! |y2-z|,|y1-z| < |y1-y2|
        rslt = zero
        if (hz1.ne.0d0) rslt = rslt + (y1-zz)*avh_olo_logc( q1z,i1z )
     &                                       *avh_olo_r0fun( y1,zz )
        if (hz2.ne.0d0) rslt = rslt - (y2-zz)*avh_olo_logc( q2z,i2z )
     &                                       *avh_olo_r0fun( y2,zz )
        rslt = rslt/(y1-y2)
!        hlp1 = (y1-zz)*avh_olo_logc( q1z,i1z )*avh_olo_r0fun( y1,zz ) !DEBUG
!        hlp2 =-(y2-zz)*avh_olo_logc( q2z,i2z )*avh_olo_r0fun( y2,zz ) !DEBUG
!        write(6,*) 'r1fun5',(hlp1+hlp2)/(cdabs(hlp1)+cdabs(hlp2)) !DEBUG
c        write(6,*) 'r1fun 5',rslt !DEBUG
      endif
*
      if (zzsmall) then ! |z| < |y1-z|,|y2-z|
        call avh_olo_conv( qq ,iq ,-zz,0d0 )
        call avh_olo_rat( q1,i1 ,qq,iq ,q1z,i1z ) ! (-z)/(y1-z)
        call avh_olo_rat( q2,i2 ,qq,iq ,q2z,i2z ) ! (-z)/(y2-z)
        trm = avh_olo_li2c( q1,i1 ) - avh_olo_li2c( q2,i2 )
        rslt = rslt + trm/(y1-y2)
!        hlp1 = rslt !DEBUG
!        hlp2 = trm/(y1-y2) !DEBUG
!        hlp3 = avh_olo_li2c( q1,i1 ) !DEBUG
!        hlp4 =- avh_olo_li2c( q2,i2 )  !DEBUG
!        write(6,*) 'r1fun 01',(hlp1+hlp2)/(cdabs(hlp1)+cdabs(hlp2)) !DEBUG
!        write(6,*) 'r1fun 01',(hlp3+hlp4)/(cdabs(hlp3)+cdabs(hlp4)) !DEBUG
!        write(6,*) 'r1fun 01 ', rslt-trm/(y1-y2) !DEBUG
!        write(6,*) 'r1fun 01', trm/(y1-y2) !DEBUG
      else
        call avh_olo_conv( qq ,iq ,-zz,0d0 )
        call avh_olo_rat( q1,i1 ,q1z,i1z ,qq,iq ) ! (y1-z)/(-z)
        call avh_olo_rat( q2,i2 ,q2z,i2z ,qq,iq ) ! (y2-z)/(-z)
        rslt = rslt + avh_olo_li2c2( q1,i1 ,q2,i2 )/zz
!        hlp1 = rslt !DEBUG
!        hlp2 = avh_olo_li2c2( q1,i1 ,q2,i2 )/zz !DEBUG
!        write(6,*) 'r1fun 02',(hlp1+hlp2)/(cdabs(hlp1)+cdabs(hlp2)) !DEBUG
c        write(6,*) 'r1fun 02', avh_olo_li2c2( q1,i1 ,q2,i2 )/zz !DEBUG
c        write(6,*) 'r1fun 02 ', rslt !DEBUG
      endif
*
      if (ozsmall) then ! |1-z| < |y1-z|,|y2-z|
        call avh_olo_conv( qq ,iq ,oz,0d0 )
        call avh_olo_rat( q1,i1 ,qq,iq ,q1z,i1z ) ! (1-z)/(y1-z)
        call avh_olo_rat( q2,i2 ,qq,iq ,q2z,i2z ) ! (1-z)/(y2-z)
        trm = avh_olo_li2c( q1,i1 ) - avh_olo_li2c( q2,i2 )
        rslt = rslt - trm/(y1-y2)
!        hlp1 = rslt !DEBUG
!        hlp2 = -trm/(y1-y2) !DEBUG
!        hlp3 = avh_olo_li2c( q1,i1 ) !DEBUG
!        hlp4 =- avh_olo_li2c( q2,i2 )  !DEBUG
!        write(6,*) 'r1fun 03',(hlp1+hlp2)/(cdabs(hlp1)+cdabs(hlp2)) !DEBUG
!        write(6,*) 'r1fun 03',(hlp3+hlp4)/(cdabs(hlp3)+cdabs(hlp4)) !DEBUG
c        write(6,*) 'r1fun 03', - trm/(y1-y2)  !DEBUG
c        write(6,*) 'r1fun 03 ', rslt !DEBUG
      else
        call avh_olo_conv( qq,iq ,oz,0d0 )
        call avh_olo_rat( q1,i1 ,q1z,i1z ,qq,iq ) ! (y1-z)/(1-z)
        call avh_olo_rat( q2,i2 ,q2z,i2z ,qq,iq ) ! (y2-z)/(1-z)
        rslt = rslt + avh_olo_li2c2( q1,i1 ,q2,i2 )/oz
!        hlp1 = rslt !DEBUG
!        hlp2 = avh_olo_li2c2( q1,i1 ,q2,i2 )/oz !DEBUG
!        write(6,*) 'r1fun 04',(hlp1+hlp2)/(cdabs(hlp1)+cdabs(hlp2)) !DEBUG
c        write(6,*) 'r1fun 04', avh_olo_li2c2( q1,i1 ,q2,i2 )/oz !DEBUG
c        write(6,*) 'r1fun 04 ', rslt !DEBUG
      endif
      avh_olo_r1fun = rslt
c      write(6,*) 'r1fun',rslt !DEBUG
      end


      double complex function avh_olo_r0fun( y1,y2 )
*  ********************************************************************
*  *      / 1-y1 \       / 1-y2 \
*  *  log |------| - log |------| 
*  *      \  -y1 /       \  -y2 /
*  *  ---------------------------
*  *            y1 - y2
*  *
*  * y1,y2 should have non-zero imaginary parts
*  ********************************************************************
      implicit none
      double complex y1,y2
      double complex
     & q1,q2,qq,oy1,oy2,log1,log2,one
     &,avh_olo_logc2,avh_olo_logc
      parameter( one=(1d0,0d0) )
      integer
     & i1,i2,ii
      call avh_olo_conv( q1,i1 ,-y1,0d0 )
      call avh_olo_conv( q2,i2 ,-y2,0d0 )
      call avh_olo_rat( qq,ii ,q2,i2 ,q1,i1 )
      log1 = avh_olo_logc2( qq,ii )/y1 ! log((-y2)/(-y1))/(y1-y2)
      oy1 = one-y1
      oy2 = one-y2
      call avh_olo_conv( q1,i1 ,oy1,0d0 )
      call avh_olo_conv( q2,i2 ,oy2,0d0 )
      call avh_olo_rat( qq,ii ,q2,i2 ,q1,i1 )
      log2 = avh_olo_logc2( qq,ii )/oy1 ! -log((1-y2)/(1-y1))/(y1-y2)
      avh_olo_r0fun = log1 + log2
      end


      double complex function avh_olo_plnr( y1,y2 ,p1,p2 ,aa,bb,cc )
*  ********************************************************************
*  *                   /   a    \          /   a    \
*  *            p1*log |--------| - p2*log |--------| 
*  *                   \ b*y1+c /          \ b*y2+c /
*  * 2*pi*imag* -------------------------------------
*  *                           y1 - y2
*  * 
*  * p1,p2 are logical, to be interpreted as 0,1 in the formula above 
*  ********************************************************************
      implicit none
      double complex y1,y2 ,aa,bb,cc
      logical p1,p2
      double complex
     & x1,q1,x2,q2,xx,rslt,twopii
     &,avh_olo_logc,avh_olo_logc2
      double precision
     & avh_olo_pi
      integer
     & i1,i2,ii
     &,avh_olo_un_get
*
      if (p1) then
        x1 = bb*y1 + cc
        xx = aa/x1
        if (dimag(xx).eq.0d0) then
          if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &      'ERROR in avh_olo_plnr: aa/x1 has zero imaginary part'
        endif
        call avh_olo_conv( q1,i1 ,xx,0d0 )
      endif
      if (p2) then
        x2 = bb*y2 + cc
        xx = aa/x2
        if (dimag(xx).eq.0d0) then
          if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &      'ERROR in avh_olo_plnr: aa/x2 has zero imaginary part'
        endif
        call avh_olo_conv( q2,i2 ,xx,0d0 )
      endif
      if (p1) then
        if (p2) then
          call avh_olo_rat( xx,ii ,q2,i2 ,q1,i1 )
          twopii = dcmplx(0d0,2*avh_olo_pi())
!WRONG          rslt = avh_olo_logc2( xx,ii ) * twopii*bb/x1
          rslt = avh_olo_logc2( xx,ii ) * twopii*bb/x2 !RIGHT
        else
          twopii = dcmplx(0d0,2*avh_olo_pi())
          rslt = avh_olo_logc( q1,i1 ) * twopii/(y1-y2)
        endif
      elseif (p2) then
        twopii = dcmplx(0d0,2*avh_olo_pi())
        rslt = avh_olo_logc( q2,i2 ) * twopii/(y2-y1) ! minus sign
      else
        rslt = dcmplx(0d0)
      endif
      avh_olo_plnr = rslt
      end


      integer function avh_olo_eta3( aa,sa ,bb,sb ,cc,sc )
*  ********************************************************************
*  *     theta(-Im(a))*theta(-Im(b))*theta( Im(c))
*  *   - theta( Im(a))*theta( Im(b))*theta(-Im(c))
*  * where a,b,c are interpreted as a+i|eps|sa, b+i|eps|sb, c+i|eps|sc
*  ********************************************************************
      implicit none
      double complex aa,bb,cc
      double precision sa,sb,sc
     &,ima,imb,imc
      ima = dimag(aa)
      imb = dimag(bb)
      imc = dimag(cc)
      if (ima.eq.0d0) ima = sa
      if (imb.eq.0d0) imb = sb
      if (imc.eq.0d0) imc = sc
      ima = dsign(1d0,ima)
      imb = dsign(1d0,imb)
      imc = dsign(1d0,imc)
      if (ima.eq.imb.and.ima.ne.imc) then
        avh_olo_eta3 = idnint(imc)
      else
        avh_olo_eta3 = 0
      endif
      end
 
      integer function avh_olo_eta2( aa,sa ,bb,sb )
*  ********************************************************************
*  * The same as  avh_olo_eta3, but with  c=a*b, so that
*  *   2*pi*imag*eta(a,b) = log(a*b) - log(a) - log(b)
*  ********************************************************************
      implicit none
      double complex aa,bb
      double precision sa,sb
     &,rea,reb,ima,imb,imab
      rea = dreal(aa)
      reb = dreal(bb)
      ima = dimag(aa)
      imb = dimag(bb)
      imab = rea*imb + reb*ima
      if (ima.eq.0d0) ima = sa
      if (imb.eq.0d0) imb = sb
      if (imab.eq.0d0) imab = dsign(rea,sb) + dsign(reb,sa)
      ima  = dsign(1d0,ima)
      imb  = dsign(1d0,imb)
      imab = dsign(1d0,imab)
      if (ima.eq.imb.and.ima.ne.imab) then
        avh_olo_eta2 = idnint(imab)
      else
        avh_olo_eta2 = 0
      endif
      end
