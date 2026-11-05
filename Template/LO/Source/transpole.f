      Subroutine transpole(pole1,width1,x1,y,jac)
c**********************************************************************
c     This routine transfers evenly spaced x values between 0 and 1
c     to y values with a pole at y=pole with width width and returns
c     the appropriate jacobian for this.  If x<del or x>1-del, uses
c     a linear transformation.  This ensures ability to cover entire 
c     region, even away from B.W.
c
c     If pole<0 then assumes have sqrt(1d0/(x^2+a^2)) type pole
c     If pole<0 then assumes have x/(x^2+a^2) type pole
c
c**********************************************************************
      implicit none
c
c     Constants
c
      double precision del
      parameter       (del=1d-22)          !Must agree with del in untranspole
c
c     Arguments
c
      double precision pole,width,y,jac
      double precision x1

c
c     Local
c
      double precision z,zmin,zmax,xmin,xmax,ez
      double precision pole1,width1,x,xc
      double precision a,b
c
c     small width treatment
c
      double precision small_width_treatment
      common/narrow_width/small_width_treatment
c
c     s-hat map with a knee (pole=-25), parameters set in set_peaks:
c     lower end of the log region, fraction of the points in it and
c     log(width/shat_floor)
c
      double precision shat_floor, knee_frac, knee_log
      common/to_shat_knee/shat_floor, knee_frac, knee_log
      double precision xb, x2
c-----
c  Begin Code
c-----
      pole=pole1
      width=width1
      if (pole.eq.-25d0.and.shat_floor.le.0d0) pole=-2d0 !knee not set

      x = x1
      if (pole .gt. 0d0) then
         if (width.lt.pole*small_width_treatment)then
            width = pole * small_width_treatment
            jac = jac * width/width1
         endif

         zmin = atan((-pole)/width)/width
         zmax = atan((1d0-pole)/width)/width
         if (x .gt. del .and. x .lt. 1d0-del) then
            z = zmin+(zmax-zmin)*x
            y = pole+width*tan(width*z)
            jac = jac *(width/cos(width*z))**2*(zmax-zmin)
         elseif (x .lt. del) then
            xmin = 0d0
            z    = zmin+(zmax-zmin)*del
            xmax = pole+width*tan(width*z)
            y = xmin+x*(xmax-xmin)/del
            jac = jac*(xmax-xmin)/del
         else
            xmax = 1d0
            z    = zmin+(zmax-zmin)*(1d0-del)
            xmin = pole+width*tan(width*z)
            y = xmin+(x+del-1d0)*(xmax-xmin)/del
            jac = jac*(xmax-xmin)/del
         endif
      elseif(pole .gt. -1d0) then       !1/sqrt(x^2+width^2) t-channel
         if (x .gt. .5d0) then          !Don't do anything here t>0
            y=x
         else
            zmin = log(2d0*width)       !2*width is because x->1-2*x
            zmax = log(1d0+sqrt(1d0+4d0*width*width))
            x=1d0-x*2d0
            z = zmin+(zmax-zmin)*x
            ez = exp(z)
            y = (1d0-.5d0*(ez-4d0*width*width/ez))/2d0
            jac = jac *(zmax-zmin)*.5d0*(ez+4d0*width*width/ez)
c            x = .5d0*(1d0-x)
         endif
c-------
c    tjs 3/5/2011  Perform 1/x transformation  using y=xo^(1-x)
c-------
      elseif(pole .eq. -15d0 .and. width .gt. 0d0) then !1/x   limit of width         
c         if (x .lt. width) then      !No transformation below cutoff
         xc = width
         xc = 1d0/(1d0-log(width))
         if (x .le. xc) then      !No transformation below cutoff
            y=x*width/xc
            jac = jac * width / xc
         else
            z = (x-xc)/(1d0-xc)
            y=width**(1d0-z)
            jac = jac * y * (-log(width))/(1d0-xc)
c            write(*,*) "trans",x,y,z
         endif
c         write(*,*) 'Transpole called',x,y
         return
      elseif(pole .eq. -25d0 .and. width .gt. 0d0) then !1/x^2 above knee
c-------
c     s-hat with a knee at width coming from non-forced BW: 1/x^2 above
c     the knee (as for pole=-2), log between shat_floor and the knee
c     with a fraction knee_frac of the points, identity below shat_floor
c-------
         xc = shat_floor
         xb = width
         x2 = xc + knee_frac
         if (x .lt. xc) then
            y = x
         elseif (x .lt. x2) then
            y = xc*exp(knee_log*(x-xc)/knee_frac)
            jac = jac * y * knee_log/knee_frac
         else
            z = 1d0 - (x-x2)*(1d0-xb)/(1d0-x2)
            y = xb/z
            jac = jac * xb/(z*z) * (1d0-xb)/(1d0-x2)
         endif
         return
      elseif(pole .ge. -2d0 .and. width .gt. 0d0) then !1/x^2   limit of width
         if (x .lt. width) then      !No transformation below cutoff
            y=x
         else
c---------
c   tjs 5/1/2008  modified for any y=x^-n transformation       
c-----------
            z = 1d0 - x + width
            b = ( 1d0-width) / (width**(pole+1d0) - 1d0)
            a = width - b
            y = a + b * z**(pole+1)
            jac = jac * abs((pole+1d0) * b * z**(pole))
c            write(*,*) "pre-trans",x,y
c            call untranspole(pole,width,x,y,jac)
c            write(*,*) "post-trans",x,y
c-----uncomment for 1/x^2 tjs -------
c            x = 1d0-x+width
c            y=width/x
c            jac = jac*width/(x*x)
c------------------------------------
            

c            write(*,*) 'trans',x,width/(x*x)
         endif

      elseif(pole .gt. -1d99) then       !1/sqrt(x^2+width^2) s-channel
         zmin = log(width)
         zmax = log(1d0+sqrt(1d0+width*width))
         if (x .gt. del .and. x .lt. 1d0-del) then
            z = zmin+(zmax-zmin)*x
            ez = exp(z)
            y = .5d0*(ez-width*width/ez)
            jac = jac *(zmax-zmin)*.5d0*(ez+width*width/ez)
         elseif (x .le. del) then
            xmin = 0d0
            z    = zmin+(zmax-zmin)*del
            ez   = exp(z)
            xmax = .5d0*(ez-width*width/ez)
            y = xmin+x*(xmax-xmin)/del
            jac = jac*(xmax-xmin)/del
         else
            xmax = 1d0
            z    = zmin+(zmax-zmin)*(1d0-del)
            ez   = exp(z)
            xmin = .5d0*(ez-width*width/ez)
            y = xmin+(x+del-1d0)*(xmax-xmin)/del
            jac = jac*(xmax-xmin)/del
         endif
      elseif(pole .gt. -8d99) then
         zmin = .5d0*log(width*width)
         zmax = .5d0*log(1d0+width*width)
         if (x .gt. del .and. x .lt. 1d0-del) then
            z = zmin+(zmax-zmin)*x
            ez = exp(2d0*z)
            y = sqrt(ez-width*width)
            jac = jac *(zmax-zmin)*ez/sqrt(ez-width*width)
         elseif (x .lt. del) then
            xmin = 0d0
            z    = zmin+(zmax-zmin)*del
            xmax = sqrt(exp(2d0*z)-width*width)
            y = xmin+x*(xmax-xmin)/del
            jac = jac*(xmax-xmin)/del
         else
            xmax = 1d0
            z    = zmin+(zmax-zmin)*(1d0-del)
            xmin = sqrt(exp(2d0*z)-width*width)
            y = xmin+(x+del-1d0)*(xmax-xmin)/del
            jac = jac*(xmax-xmin)/del
         endif
      endif
      end

      Subroutine untranspole(pole1,width1,x,y1,jac)
c**********************************************************************
c     This routine transfers takes values of y for a given pole and
c     width, and returns the value of x (which an evenly placed
c     random number) would have been used to get that value of y.
c     it also returns the jacobian associated with this choice.
c**********************************************************************
      implicit none
c
c     Constants
c
      double precision del
      parameter       (del=1d-22)          !Must agree with del in untranspole
c
c     Arguments
c
      double precision pole1,width1,y1,jac
      real*8 x
c
c     small width treatment
c
      double precision small_width_treatment
      common/narrow_width/small_width_treatment
c
c     Local
c
      double precision z,zmin,zmax,xmin,xmax,ez
      double precision pole,width,y,xc
      double precision a,b
      double precision xgmin,xgmax       ! these should be identical 
      parameter (xgmin=0d0, xgmax=1d0)   ! to the ones in genps.inc
c
c     s-hat map with a knee (pole=-25), parameters set in set_peaks:
c     lower end of the log region, fraction of the points in it and
c     log(width/shat_floor)
c
      double precision shat_floor, knee_frac, knee_log
      common/to_shat_knee/shat_floor, knee_frac, knee_log
      double precision xb, x2
c-----
c  Begin Code
c-----
      pole=pole1
      width=width1
      if (pole.eq.-25d0.and.shat_floor.le.0d0) pole=-2d0 !knee not set
      y = y1
      if (pole .gt. 0d0) then                   !BW 
         if (width.lt.pole*small_width_treatment)then
            width = pole * small_width_treatment
            jac = jac * width/width1
         endif
         zmin = atan((-pole)/width)/width
         zmax = atan((1d0-pole)/width)/width
         z = atan((y-pole)/width)/width
         x = (z-zmin)/(zmax-zmin)
         if (x .le. del) then
            xmin = 0d0
            z    = zmin+(zmax-zmin)*del
            xmax = pole+width*tan(width*z)
            if(xmin.lt.xmax) then
               x = (y-xmin)*del/(xmax-xmin)
            else
               x=xmin
            endif
            jac = jac*(xmax-xmin)/del
         elseif (x .ge. 1d0-del) then
            xmax = 1d0
            z    = zmin+(zmax-zmin)*(1d0-del)
            xmin = pole+width*tan(width*z)
            if(xmin.lt.xmax) then
               x = (y-xmin)*del/(xmax-xmin)-del+1d0
            else
               x=xmin
            endif
            jac = jac*(xmax-xmin)/del
c RF (2014/07/07): code is not protected against this special case. In this case,
c simply set x to 1 and the jac to zero so that this PS point will not
c contribute (but you do get the correct xbin_min and xbin_max in
c sample_get_x)
            if (y.eq.xgmax .and. xmin.ge.xgmax) then
               x=1d0
               jac=0d0
            endif
         else
            jac = jac *(width/cos(width*z))**2*(zmax-zmin)
         endif
c-------
c    tjs 3/5/2011  Perform 1/x transformation  using y=xo^(1-x)
c-------
      elseif(pole .eq. -15d0 .and. width .gt. 0d0) then !1/x   limit of width
         xc = 1d0/(1d0-log(width))
c         xc = width
         if (y .le. width) then      !No transformation below cutoff
            x = y*xc/width
         else
            z = 1d0-log(y)/log(width)
            x = z*(1d0-xc) + xc
c            write(*,*) "untrans",x,y,z
         endif
         return
      elseif(pole .eq. -25d0 .and. width .gt. 0d0) then !inverse of transpole
         xc = shat_floor
         xb = width
         x2 = xc + knee_frac
         if (y .lt. xc) then
            x = y
         elseif (y .lt. xb) then
            x = xc + knee_frac*log(y/xc)/knee_log
            jac = jac * y * knee_log/knee_frac
         else
            z = xb/y
            x = x2 + (1d0-z)*(1d0-x2)/(1d0-xb)
            jac = jac * xb/(z*z) * (1d0-xb)/(1d0-x2)
         endif
         return
      elseif(pole .gt. -1d0) then !1/sqrt((.5-x)^2+width^2)  t-channel
         if (y .gt. .5d0) then
            x=y
         else
            zmin = log(width*2d0)
            zmax = log(1d0+sqrt(1d0+4d0*width*width))
            y = (1d0-2d0*y)
            z = log(y+sqrt(y*y+4d0*width*width))
            x = (z - zmin)/(zmax-zmin)
            x = .5d0*(1d0-x)
            ez = exp(z)
            jac = jac *(zmax-zmin)*.5d0*(ez+4d0*width*width/ez)
            y = (1d0-y)/2d0
         endif

      elseif(pole .gt. -5d0 .and. width .gt. 0d0) then !1/x^2   limit of width
         if (y .lt. width) then      !No transformation below cutoff
            x=y
         else
c---------
c   tjs 5/1/2008  modified for any y=x^-n transformation       
c-----------
            b = ( 1d0-width) / (width**(pole+1d0) - 1d0)
            a = width - b
            z = ((y-a)/b)**(1d0/(pole+1)) 
            x = 1d0 - z + width
            jac = jac * abs((pole+1d0) * b * z**(pole))

c-------------------
c Uncomment below for y=1/x^2
c-------------------
c            x=width/y
c            write(*,*) 'untr',x,width/(x*x)
c            jac = jac*width/(x*x)
c            x = 1d0-x+width
         endif

      elseif(pole .gt. -5d99) then !1/sqrt(x^2+width^2)  s-channel
         zmin = log(width)
         zmax = log(1d0+sqrt(1d0+width*width))
         if (pole .gt. -1d0 .and. y .lt. -pole) y=-pole-y
         z = log(y+sqrt(y*y+width*width))
         x = (z - zmin)/(zmax-zmin)
         if (x .gt. del .and. x .lt. 1d0-del) then
            ez = exp(z)
            jac = jac *(zmax-zmin)*.5d0*(ez+width*width/ez)
         elseif (x .lt. del) then
            xmin = 0d0
            z    = zmin+(zmax-zmin)*del
            ez   = exp(z)
            xmax = .5d0*(ez-width*width/ez)
c            y = xmin+x*(xmax-xmin)/del
            if(xmin.lt.xmax) then
               x = (y-xmin)*del/(xmax-xmin)
            else
               x=xmin
            endif
            jac = jac*(xmax-xmin)/del
         else
            xmax = 1d0
            z    = zmin+(zmax-zmin)*(1d0-del)
            ez   = exp(z)
            xmin = .5d0*(ez-width*width/ez)
c            y = xmin+(x+del-1d0)*(xmax-xmin)/del
            x = (y-xmin)*del/(xmax-xmin)-del+1d0
            jac = jac*(xmax-xmin)/del
         endif
      endif
      end

      subroutine transpole_tail(pole1,width1,x,y,jac)
c**********************************************************************
c     As transpole for a B.W. (pole>0), for a resonance free to go off
c     shell: the arctan map inside |y-pole| < k*width and 1/|y-pole|
c     (logarithmic) tails outside, k = bw_tail_k(), the density continuous
c     at the boundaries (see bwtail_setup). The B.W. density falls like
c     1/y^2 while a vector resonance decaying to massless fermions falls
c     like 1/y, so the plain arctan map leaves its off-shell tail with
c     weights growing like y.
c**********************************************************************
      implicit none
      double precision pole1,width1,x,y,jac
      double precision pole,width,z
      double precision bwi_l,bwi_w,bwi_u,bwc,bwza,bwzb,bwu
      double precision small_width_treatment
      common/narrow_width/small_width_treatment
      double precision bw_tail_k

      if (pole1 .le. 0d0) then
         call transpole(pole1,width1,x,y,jac)
         return
      endif
      pole = pole1
      width = width1
      if (width.lt.pole*small_width_treatment)then
         width = pole * small_width_treatment
         jac = jac * width/width1
      endif
      call bwtail_setup(pole,width,bwi_l,bwi_w,bwi_u,bwc,bwza,bwzb)
      bwu = x*(bwi_l+bwi_w+bwi_u)
      if (bwu .lt. bwi_l) then
         y = pole - pole*exp(-bwu/bwc)
         jac = jac*(bwi_l+bwi_w+bwi_u)*(pole-y)/bwc
      elseif (bwu .lt. bwi_l+bwi_w) then
         z = bwza + (bwu-bwi_l)*width
         y = pole + width*tan(z)
         jac = jac*(bwi_l+bwi_w+bwi_u)*((y-pole)**2+width**2)
      else
         y = pole + bw_tail_k()*width*exp((bwu-bwi_l-bwi_w)/bwc)
         jac = jac*(bwi_l+bwi_w+bwi_u)*(y-pole)/bwc
      endif
      end


      subroutine untranspole_tail(pole1,width1,x,y1,jac)
c**********************************************************************
c     Inverse of transpole_tail: returns the x giving y1, and multiplies
c     jac by dy/dx.
c**********************************************************************
      implicit none
      double precision pole1,width1,x,y1,jac
      double precision pole,width,y,z,bwk
      double precision bwi_l,bwi_w,bwi_u,bwc,bwza,bwzb,bwu
      double precision small_width_treatment
      common/narrow_width/small_width_treatment
      double precision bw_tail_k

      if (pole1 .le. 0d0) then
         call untranspole(pole1,width1,x,y1,jac)
         return
      endif
      pole = pole1
      width = width1
      y = y1
      if (width.lt.pole*small_width_treatment)then
         width = pole * small_width_treatment
         jac = jac * width/width1
      endif
      bwk = bw_tail_k()
      call bwtail_setup(pole,width,bwi_l,bwi_w,bwi_u,bwc,bwza,bwzb)
      if (y .lt. pole-bwk*width) then
         bwu = bwc*log(pole/(pole-y))
         jac = jac*(bwi_l+bwi_w+bwi_u)*(pole-y)/bwc
      elseif (y .le. pole+bwk*width) then
         z = atan((y-pole)/width)
         bwu = bwi_l + (z-bwza)/width
         jac = jac*(bwi_l+bwi_w+bwi_u)*((y-pole)**2+width**2)
      else
         bwu = bwi_l + bwi_w + bwc*log((y-pole)/(bwk*width))
         jac = jac*(bwi_l+bwi_w+bwi_u)*(y-pole)/bwc
      endif
      x = bwu/(bwi_l+bwi_w+bwi_u)
      end


      double precision function bw_tail_k()
c**********************************************************************
c     Half-width, in units of the (normalised) M*Gamma, of the arctan core
c     of the Breit-Wigner map with 1/|y-pole| tails; the single definition
c     used by every routine of that map and of its $-window variant.
c**********************************************************************
      implicit none
      bw_tail_k = 15d0
      end


      double precision function bw_tail_c(width)
c**********************************************************************
c     Normalisation of the 1/|y-pole| tails, c/|y-pole|, that makes the
c     density continuous with 1/((y-pole)^2+width^2) at |y-pole| = k*width
c**********************************************************************
      implicit none
      double precision width
      double precision bw_tail_k, bwk
      bwk = bw_tail_k()
      bw_tail_c = bwk/(width*(bwk*bwk+1d0))
      end


      subroutine bwtail_setup(pole,width,bwi_l,bwi_w,bwi_u,bwc,bwza,bwzb)
c**********************************************************************
c     masses of the three pieces of the Breit-Wigner map on 0<y<1:
c     lower 1/(pole-y) tail, arctan window |y-pole|<k*width, upper
c     1/(y-pole) tail; bwc sets the tails so that the density
c     1/((y-pole)^2+width^2) is continuous at pole +- k*width.
c     They only depend on (pole,width), fixed per integration dimension:
c     the last ncache settings are kept instead of being recomputed for
c     every phase-space point.
c**********************************************************************
      implicit none
      double precision pole,width,bwi_l,bwi_w,bwi_u,bwc,bwza,bwzb
      integer ncache
      parameter (ncache=16)
      double precision cpole(ncache), cwidth(ncache), cval(6,ncache)
      integer nfill, inext, i
      save cpole, cwidth, cval, nfill, inext
      data nfill, inext /0, 1/
      double precision ylo, yhi, bwk
      double precision bw_tail_k, bw_tail_c

      do i=1,nfill
         if (cpole(i).eq.pole .and. cwidth(i).eq.width) then
            bwi_l = cval(1,i)
            bwi_w = cval(2,i)
            bwi_u = cval(3,i)
            bwc = cval(4,i)
            bwza = cval(5,i)
            bwzb = cval(6,i)
            return
         endif
      enddo
      bwk = bw_tail_k()
      bwc = bw_tail_c(width)
      ylo = pole-bwk*width
      yhi = pole+bwk*width
c     lower tail on [0, min(ylo,1)], window on [max(0,ylo), min(1,yhi)],
c     upper tail on [yhi, 1]
      bwi_l = 0d0
      if (ylo .gt. 0d0) bwi_l = bwc*log(pole/(pole-min(ylo,1d0)))
      bwi_u = 0d0
      if (yhi .lt. 1d0) bwi_u = bwc*log((1d0-pole)/(bwk*width))
      bwza = atan((max(0d0,ylo)-pole)/width)
      bwzb = atan((min(1d0,yhi)-pole)/width)
      bwi_w = max(0d0, (bwzb-bwza)/width)
      cpole(inext) = pole
      cwidth(inext) = width
      cval(1,inext) = bwi_l
      cval(2,inext) = bwi_w
      cval(3,inext) = bwi_u
      cval(4,inext) = bwc
      cval(5,inext) = bwza
      cval(6,inext) = bwzb
      nfill = max(nfill, inext)
      inext = mod(inext, ncache) + 1
      end


      subroutine bw_window_segments(pole,width,wlo,whi,wfac,nseg,sa,sb,
     &     st,sm,c,tot)
c**********************************************************************
c     Segments of the $-excluded Breit-Wigner density on [0,1]: outside
c     the window [wlo,whi] it is the resonance density with 1/|y-pole|
c     tails of bwtail_setup (arctan core |y-pole| < k*width, continuous
c     1/|y-pole| outside), inside the window the constant c = wfac * (mean
c     of that density at the two window edges). Segment k is [sa(k),sb(k)]
c     of type st(k) (1 lower tail, 2 core, 3 upper tail, 4 window) and
c     mass sm(k); tot is the total mass. Cached like bwtail_setup.
c**********************************************************************
      implicit none
      double precision pole,width,wlo,whi,wfac,c,tot
      integer nseg, st(6)
      double precision sa(6), sb(6), sm(6)
      integer ncache
      parameter (ncache=16)
      double precision ckey(5,ncache), csa(6,ncache), csb(6,ncache)
      double precision csm(6,ncache), cc_(ncache), ctot(ncache)
      integer cnseg(ncache), cst(6,ncache), nfill, inext, k
      save ckey, csa, csb, csm, cc_, ctot, cnseg, cst, nfill, inext
      data nfill, inext /0, 1/

      do k=1,nfill
         if (ckey(1,k).eq.pole .and. ckey(2,k).eq.width .and.
     &        ckey(3,k).eq.wlo .and. ckey(4,k).eq.whi .and.
     &        ckey(5,k).eq.wfac) then
            nseg = cnseg(k)
            sa(:) = csa(:,k)
            sb(:) = csb(:,k)
            sm(:) = csm(:,k)
            st(:) = cst(:,k)
            c = cc_(k)
            tot = ctot(k)
            return
         endif
      enddo
      call bw_window_compute(pole,width,wlo,whi,wfac,nseg,sa,sb,st,sm,
     &     c,tot)
      ckey(1,inext) = pole
      ckey(2,inext) = width
      ckey(3,inext) = wlo
      ckey(4,inext) = whi
      ckey(5,inext) = wfac
      cnseg(inext) = nseg
      csa(:,inext) = sa(:)
      csb(:,inext) = sb(:)
      csm(:,inext) = sm(:)
      cst(:,inext) = st(:)
      cc_(inext) = c
      ctot(inext) = tot
      nfill = max(nfill, inext)
      inext = mod(inext, ncache) + 1
      end


      subroutine bw_window_compute(pole,width,wlo,whi,wfac,nseg,sa,sb,
     &     st,sm,c,tot)
c**********************************************************************
c     Computes the segments of bw_window_segments (see there).
c**********************************************************************
      implicit none
      double precision pole,width,wlo,whi,wfac,c,tot
      integer nseg, st(6)
      double precision sa(6), sb(6), sm(6)
      double precision cc, ylo, yhi, bp(6), tmp, a, b, mid, bwk
      double precision bw_tail_density, bw_tail_k, bw_tail_c
      integer i, j, nb, t
      bwk = bw_tail_k()
      cc = bw_tail_c(width)
      ylo = pole-bwk*width
      yhi = pole+bwk*width
      c = wfac*0.5d0*(bw_tail_density(pole,width,wlo)
     &     + bw_tail_density(pole,width,whi))
c     sorted break points inside [0,1]
      nb = 2
      bp(1) = 0d0
      bp(2) = 1d0
      do i=1,4
         if (i.eq.1) tmp = ylo
         if (i.eq.2) tmp = yhi
         if (i.eq.3) tmp = wlo
         if (i.eq.4) tmp = whi
         if (tmp.gt.0d0 .and. tmp.lt.1d0) then
            nb = nb+1
            bp(nb) = tmp
         endif
      enddo
      do i=2,nb
         tmp = bp(i)
         j = i-1
         do while (j.ge.1)
            if (bp(j).le.tmp) exit
            bp(j+1) = bp(j)
            j = j-1
         enddo
         bp(j+1) = tmp
      enddo
      nseg = 0
      tot = 0d0
      do i=1,6
         sa(i) = 0d0
         sb(i) = 0d0
         sm(i) = 0d0
         st(i) = 0
      enddo
      do i=1,nb-1
         a = bp(i)
         b = bp(i+1)
         if (b.le.a) cycle
         mid = 0.5d0*(a+b)
         if (mid.gt.wlo .and. mid.lt.whi) then
            t = 4
         elseif (mid.lt.ylo) then
            t = 1
         elseif (mid.gt.yhi) then
            t = 3
         else
            t = 2
         endif
         nseg = nseg+1
         sa(nseg) = a
         sb(nseg) = b
         st(nseg) = t
         if (t.eq.1) then
            sm(nseg) = cc*log((pole-a)/(pole-b))
         elseif (t.eq.3) then
            sm(nseg) = cc*log((b-pole)/(a-pole))
         elseif (t.eq.2) then
            sm(nseg) = (atan((b-pole)/width)-atan((a-pole)/width))/width
         else
            sm(nseg) = c*(b-a)
         endif
         tot = tot + sm(nseg)
      enddo
      end


      double precision function bw_tail_density(pole,width,y)
c**********************************************************************
c     unnormalised density of the Breit-Wigner map with 1/|y-pole| tails
c     (see bwtail_setup): 1/((y-pole)^2+width^2) for |y-pole| < k*width
c**********************************************************************
      implicit none
      double precision pole,width,y
      double precision bw_tail_k, bw_tail_c
      if (abs(y-pole).le.bw_tail_k()*width) then
         bw_tail_density = 1d0/((y-pole)**2+width**2)
      else
         bw_tail_density = bw_tail_c(width)/abs(y-pole)
      endif
      end


      subroutine transpole_win(pole1,width1,wlo1,whi1,wfac,x,y,jac)
c**********************************************************************
c     As transpole for a B.W. (pole>0) but for a $-excluded propagator:
c     outside the window [wlo,whi] (in the same s/stot units as pole) the
c     density is that of the Breit-Wigner map with 1/|y-pole| tails
c     (the off-shell tail of a resonance falls like 1/s, see transpole_tail),
c     inside it is flat, at the mean of the two edge values times wfac,
c     so that the excluded pole region is not oversampled (wfac=0: not
c     sampled at all).
c**********************************************************************
      implicit none
      double precision pole1,width1,wlo1,whi1,wfac,x,y,jac
      double precision pole,width,wlo,whi,c,tot,u,g,bwk
      integer nseg, st(6), k
      double precision sa(6), sb(6), sm(6)
      double precision small_width_treatment
      common/narrow_width/small_width_treatment
      double precision bw_tail_density, bw_tail_k

      pole = pole1
      width = width1
      if (width.lt.pole*small_width_treatment) then
         width = pole*small_width_treatment
         jac = jac*width/width1
      endif
      wlo = max(wlo1,0d0)
      whi = min(whi1,1d0)
      if (wlo.ge.whi) then
         call transpole_tail(pole1,width1,x,y,jac)
         return
      endif
      call bw_window_segments(pole,width,wlo,whi,wfac,nseg,sa,sb,st,sm,
     &     c,tot)
      bwk = bw_tail_k()
      u = x*tot
      k = 1
      do while (k.lt.nseg .and. u.ge.sm(k))
         u = u - sm(k)
         k = k+1
      enddo
      if (st(k).eq.4 .and. sm(k).le.0d0) then
c        empty window: only reached at the boundary, stay on its edge
         y = sa(k)
         jac = 0d0
         return
      endif
      if (st(k).eq.1) then
         y = pole - (pole-sa(k))*exp(-u*width*(bwk*bwk+1d0)/bwk)
      elseif (st(k).eq.3) then
         y = pole + (sa(k)-pole)*exp(u*width*(bwk*bwk+1d0)/bwk)
      elseif (st(k).eq.2) then
         y = pole + width*tan(atan((sa(k)-pole)/width) + u*width)
      else
         y = sa(k) + u/c
      endif
      y = min(max(y,sa(k)),sb(k))
      if (st(k).eq.4) then
         g = c
      else
         g = bw_tail_density(pole,width,y)
      endif
      jac = jac*tot/g
      end


      subroutine untranspole_win(pole1,width1,wlo1,whi1,wfac,x,y,jac)
c**********************************************************************
c     Inverse of transpole_win: returns the x giving y, and multiplies
c     jac by dy/dx.
c**********************************************************************
      implicit none
      double precision pole1,width1,wlo1,whi1,wfac,x,y,jac
      double precision pole,width,wlo,whi,c,tot,u,g,yy,cc
      integer nseg, st(6), k
      double precision sa(6), sb(6), sm(6)
      double precision small_width_treatment
      common/narrow_width/small_width_treatment
      double precision bw_tail_density, bw_tail_c

      pole = pole1
      width = width1
      if (width.lt.pole*small_width_treatment) then
         width = pole*small_width_treatment
         jac = jac*width/width1
      endif
      wlo = max(wlo1,0d0)
      whi = min(whi1,1d0)
      if (wlo.ge.whi) then
         call untranspole_tail(pole1,width1,x,y,jac)
         return
      endif
      call bw_window_segments(pole,width,wlo,whi,wfac,nseg,sa,sb,st,sm,
     &     c,tot)
      cc = bw_tail_c(width)
      yy = min(max(y,0d0),1d0)
      u = 0d0
      k = 1
      do while (k.lt.nseg .and. yy.gt.sb(k))
         u = u + sm(k)
         k = k+1
      enddo
      if (st(k).eq.1) then
         u = u + cc*log((pole-sa(k))/(pole-yy))
         g = bw_tail_density(pole,width,yy)
      elseif (st(k).eq.3) then
         u = u + cc*log((yy-pole)/(sa(k)-pole))
         g = bw_tail_density(pole,width,yy)
      elseif (st(k).eq.2) then
         u = u + (atan((yy-pole)/width)-atan((sa(k)-pole)/width))/width
         g = bw_tail_density(pole,width,yy)
      else
         u = u + c*(yy-sa(k))
         g = c
         if (c.le.0d0) then
            x = u/tot
            jac = 0d0
            return
         endif
      endif
      x = u/tot
      jac = jac*tot/g
      end
