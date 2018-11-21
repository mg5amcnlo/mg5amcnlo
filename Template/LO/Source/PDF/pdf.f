      subroutine pftopdg(ih,x,q,pdf)
c***************************************************************************
c     Wrapper for calling the pdf of MCFM
c***************************************************************************
      implicit none
c
c     Arguments
c
      DOUBLE  PRECISION x,q,pdf(-7:7)
      INTEGER IH
C
C     Include
C
      include 'pdf.inc'
C      
      call fdist(ih,x, q, pdf)

      return	
      end


      subroutine fdist(ih,x,xmu,fx)
C***********************************************************************
C     MCFM PDF CALLING ROUTINE
C***********************************************************************
      implicit none
      integer ih,i
      double precision fx(-7:7),x,xmu,nnfx(-6:7)
      double precision u_val,d_val,u_sea,d_sea,s_sea,c_sea,b_sea,gluon
      double precision Ctq3df,Ctq4Fn,Ctq5Pdf,Ctq6Pdf,Ctq5L
      double precision q2max
      double precision epa_electron,epa_proton
      include 'pdf.inc'

      integer mode,Iprtn,Irt

      do Iprtn=-7,7
         fx(Iprtn)=0d0
      enddo
C---  set to zero if x out of range
      if (x .ge. 1d0) then
         return
      endif
      if (pdlabel(1:4) .eq. 'nn23') then
         call NNevolvePDF(x,xmu,nnfx)
         do i=-5,5
            fx(i)=nnfx(i)/x
         enddo
         fx(7)=nnfx(7)/x
c      elseif     ((pdlabel(1:3) .eq. 'mrs')
c     .   .or. (pdlabel(2:4) .eq. 'mrs')) then
c
c             if     (pdlabel .eq. 'mrs02nl') then
c             mode=1
c             call mrst2002(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif     (pdlabel .eq. 'mrs02nn') then
c             mode=2
c             call mrst2002(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif     (pdlabel .eq. 'mrs0119') then
c             mode=1
c             call mrst2001(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs0117') then
c             mode=2
c             call mrst2001(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs0121') then
c             mode=3
c             call mrst2001(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs01_j') then
c             mode=4
c             call mrst2001(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif     (pdlabel .eq. 'mrs99_1') then
c             mode=1
c             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs99_2') then
c             mode=2
c             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs99_3') then
c             mode=3
c             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs99_4') then
c             mode=4
c             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs99_5') then
c             mode=5
c             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs99_6') then
c             mode=6
c             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs99_7') then
c             mode=7
c             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs99_8') then
c             mode=8
c             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs99_9') then
c             mode=9
c             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs9910') then
c             mode=10
c             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs9911') then
c             mode=11
c             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs9912') then
c             mode=12
c             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs98z1') then
c             mode=1
c             call mrs98(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs98z2') then
c             mode=2 
c             call mrs98(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs98z3') then
c             mode=3
c             call mrs98(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs98z4') then
c             mode=4
c             call mrs98(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs98z5') then
c             mode=5
c             call mrs98(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs98l1') then
c             mode=1
c             call mrs98lo(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs98l2') then
c             mode=2 
c             call mrs98lo(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs98l3') then
c             mode=3
c             call mrs98lo(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs98l4') then
c             mode=4
c             call mrs98lo(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs98l5') then
c             mode=5
c             call mrs98lo(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             elseif (pdlabel .eq. 'mrs98ht') then
c             mode=1
c             call mrs98ht(x,xmu,mode,u_val,d_val,u_sea,d_sea,
c     &                          s_sea,c_sea,b_sea,gluon)
c             endif
c-----assign mrs to standard grid
c            fx(-5)=b_sea/x
c            fx(-4)=c_sea/x
c            fx(-3)=s_sea/x
c            fx( 0)=gluon/x
c            fx(+3)=fx(-3)
c            fx(+4)=fx(-4)
c            fx(+5)=fx(-5)
c               fx(1)=(d_val+d_sea)/x
c               fx(2)=(u_val+u_sea)/x
c               fx(-1)=d_sea/x
c               fx(-2)=u_sea/x
C
c      elseif (pdlabel(1:5) .eq. 'cteq3') then
C     
c         if (pdlabel .eq. 'cteq3_m') then
c            mode=1
c         elseif (pdlabel .eq. 'cteq3_l') then
c            mode=2
c         elseif (pdlabel .eq. 'cteq3_d') then
c            mode=3
c         endif
c         fx(-5)=Ctq3df(mode,-5,x,xmu,Irt)/x
c         fx(-4)=Ctq3df(mode,-4,x,xmu,Irt)/x
c         fx(-3)=Ctq3df(mode,-3,x,xmu,Irt)/x
c         
c         fx(0)=Ctq3df(mode,0,x,xmu,Irt)/x
c         
c         fx(+3)=Ctq3df(mode,+3,x,xmu,Irt)/x
c         fx(+4)=Ctq3df(mode,+4,x,xmu,Irt)/x
c         fx(+5)=Ctq3df(mode,+5,x,xmu,Irt)/x
c            fx(-1)=Ctq3df(mode,-2,x,xmu,Irt)/x
c            fx(-2)=Ctq3df(mode,-1,x,xmu,Irt)/x
c            fx(1)=Ctq3df(mode,+2,x,xmu,Irt)/x+fx(-1)
c            fx(2)=Ctq3df(mode,+1,x,xmu,Irt)/x+fx(-2)
C     
c      elseif (pdlabel(1:5) .eq. 'cteq4') then
C     
c         if (pdlabel .eq. 'cteq4_m') then
c            mode=1
c         elseif (pdlabel .eq. 'cteq4_d') then
c            mode=2
c         elseif (pdlabel .eq. 'cteq4_l') then
c            mode=3
c         elseif (pdlabel .eq. 'cteq4a1') then
c            mode=4
c         elseif (pdlabel .eq. 'cteq4a2') then
c            mode=5
c         elseif (pdlabel .eq. 'cteq4a3') then
c            mode=6
c         elseif (pdlabel .eq. 'cteq4a4') then
c            mode=7
c         elseif (pdlabel .eq. 'cteq4a5') then
c            mode=8
c         elseif (pdlabel .eq. 'cteq4hj') then
c            mode=9
c         elseif (pdlabel .eq. 'cteq4lq') then
c            mode=10
c         endif
c         
c         fx(-5)=Ctq4Fn(mode,-5,x,xmu)
c         fx(-4)=Ctq4Fn(mode,-4,x,xmu)
c         fx(-3)=Ctq4Fn(mode,-3,x,xmu)
c         
c         fx(0)=Ctq4Fn(mode,0,x,xmu)
c         
c         fx(+3)=Ctq4Fn(mode,+3,x,xmu)
c         fx(+4)=Ctq4Fn(mode,+4,x,xmu)
c         fx(+5)=Ctq4Fn(mode,+5,x,xmu)
c            fx(1)=Ctq4Fn(mode,+2,x,xmu)
c            fx(2)=Ctq4Fn(mode,+1,x,xmu)
c            fx(-1)=Ctq4Fn(mode,-2,x,xmu)
c            fx(-2)=Ctq4Fn(mode,-1,x,xmu)
C
c      elseif (pdlabel .eq. 'cteq5l1') then
C
c         fx(-5)=Ctq5L(-5,x,xmu)
c         fx(-4)=Ctq5L(-4,x,xmu)
c         fx(-3)=Ctq5L(-3,x,xmu)
c         
c         fx(0)=Ctq5L(0,x,xmu)
c         
c         fx(+3)=Ctq5L(+3,x,xmu)
c         fx(+4)=Ctq5L(+4,x,xmu)
c         fx(+5)=Ctq5L(+5,x,xmu)
c         
c            fx(1)=Ctq5L(+2,x,xmu)
c            fx(2)=Ctq5L(+1,x,xmu)
c            fx(-1)=Ctq5L(-2,x,xmu)
c            fx(-2)=Ctq5L(-1,x,xmu)
C         
c      elseif ((pdlabel(1:5) .eq. 'cteq5') .or. 
c     .        (pdlabel(1:4) .eq. 'ctq5')) then
C         
c         fx(-5)=Ctq5Pdf(-5,x,xmu)
c         fx(-4)=Ctq5Pdf(-4,x,xmu)
c         fx(-3)=Ctq5Pdf(-3,x,xmu)
c         
c         fx(0)=Ctq5Pdf(0,x,xmu)
c         
c         fx(+3)=Ctq5Pdf(+3,x,xmu)
c         fx(+4)=Ctq5Pdf(+4,x,xmu)
c         fx(+5)=Ctq5Pdf(+5,x,xmu)
c         
c            fx(1)=Ctq5Pdf(+2,x,xmu)
c            fx(2)=Ctq5Pdf(+1,x,xmu)
c            fx(-1)=Ctq5Pdf(-2,x,xmu)
c            fx(-2)=Ctq5Pdf(-1,x,xmu)
C                  
      elseif (pdlabel(1:5) .eq. 'cteq6') then
C         
         fx(-5)=Ctq6Pdf(-5,x,xmu)
         fx(-4)=Ctq6Pdf(-4,x,xmu)
         fx(-3)=Ctq6Pdf(-3,x,xmu)
         
         fx(0)=Ctq6Pdf(0,x,xmu)
         
         fx(+3)=Ctq6Pdf(+3,x,xmu)
         fx(+4)=Ctq6Pdf(+4,x,xmu)
         fx(+5)=Ctq6Pdf(+5,x,xmu)
         
            fx(1)=Ctq6Pdf(+2,x,xmu)
            fx(2)=Ctq6Pdf(+1,x,xmu)
            fx(-1)=Ctq6Pdf(-2,x,xmu)
            fx(-2)=Ctq6Pdf(-1,x,xmu)
      endif      
c
c  a "diffractive" photon
c      
      q2max=xmu*xmu
      if(ih .eq. 3) then  !from the electron
          fx(7)=epa_electron(x,q2max)
      elseif(ih .eq. 2) then  !from a proton without breaking
          fx(7)=epa_proton(x,q2max)
      endif      
      
      return
      end
      
  

