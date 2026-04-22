c     This function calculate the reduced structure function, with energy
c     fraction given by "x", at scale "Qsquare"
      real*8 function eepdf_tilde(x,Qsquare,n,partonid,beamid)
      implicit none
      real*8 x,Qsquare
      real*8 me
      data me /0.511d-3/
      real*8 PI
      real*8 alphaem
c     In Gmu scheme
      data alphaem/0.007562397d0/
      real*8 beta
      integer n,partonid,beamid
      real*8 isr_tilde_racoon
      real*8 res
      data res/0d0/

      PI=4.D0*DATAN(1.D0)
      
      beta = alphaem/PI * (dlog(Qsquare/me/me)-1)

c     electron beam
      if (beamid .eq. 11) then
c     other partons are zero
          if (partonid .ne. 11) then
             res = 0d0
          else
             if (n .eq. 1) then
                res = isr_tilde_racoon(x,beta)
             else
                res = 0d0
             endif
          endif
      else if (beamid .eq. -11) then
          if (partonid .ne. -11) then
              res = 0d0
          else
             if (n .eq. 1) then
                res = isr_tilde_racoon(x,beta)
             else
                res = 0d0
             endif
          endif
      endif
      eepdf_tilde = res
      end
      
c     https://arxiv.org/pdf/hep-ph/0302198.pdf, eq.(2.44)
c     note that beta_e in eq.(2.45) is twice our beta
c     so eq.(2.44) needs to be corrected by some factor of 2
      real*8 function isr_tilde_racoon(x,beta)
      implicit none
      real*8 x,beta
      real*8 res
      real*8 PI
      real*8 gE
      real*8 logx, logomx
      real*8 dlgam,DDILOG
      external dlgam,DDILOG
      PI=4.D0*DATAN(1.D0)
      gE=0.5772156649d0
      res=0d0
      if (x .lt. 0.9999999d0) then
         logx=dlog(x)
         logomx=dlog(1d0-x)
c     ----------------------------
c     order alpha
         res=-beta*(1d0+x)/2d0
c     order alpha^2
         res=res-(beta**2)/8d0*(
     c        (1d0+3d0*x*x)/(1d0-x)*logx
     c        +4d0*(1d0+x)*logomx+5d0+x)
c     order alpha^3
         res=res-(beta**3)/128d0*(
     c        (1d0+x)*(6d0*DDILOG(x)+12d0*(logomx**2)-3d0*PI**2)
     c        +(1d0/(1d0-x)) * (
     c        (3d0/2d0)*(1d0+8d0*x+3d0*x**2)*logx
     c        +6d0*(x+5d0)*(1d0-x)*logomx
     c        +12d0*(1d0+x**2)*logx*logomx
     c        -1d0/2d0*(1d0+7d0*x**2)*logx**2
     c        +1d0/4d0*(39d0-24d0*x-15d0*x**2)))
c     ----------------------------
         res=res*(1d0-x)**(1-beta)
      endif
      res=res+exp(beta*(-gE+3d0/4d0))/exp(dlgam(1d0+beta))*beta
      isr_tilde_racoon=res
      end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc      
      function dlgam(xx)
c real logarithm of gamma function
      implicit real * 8 (a-h,o-z)
      real * 8 cof(6),stp,gt,g,cst
      data cof,stp/76.18009173d0,-86.50532033d0,24.01409822d0,
     # -1.231739516d0,.120858003d-2,-.536382d-5,2.50662827465d0/
      data gt,g/5.5d0,5.0d0/
      data cst/4.081061466d0/
      x = xx - 1
      xpgt = x + gt
      xmp5  = x + .5d0
      s0 = 1
      do 1 j=1,6
        x = x + 1
        tmp = cof(j)/x
        s0  = s0 + tmp
  1     continue
      r10 = log(s0)
      dlgam = xmp5*(log(xpgt)-1) + r10 - cst
      return
      end      

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      
