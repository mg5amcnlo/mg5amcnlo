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
      elseif     ((pdlabel(1:3) .eq. 'mrs')
     .   .or. (pdlabel(2:4) .eq. 'mrs')) then

             if     (pdlabel .eq. 'mrs02nl') then
             mode=1
             call mrst2002(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif     (pdlabel .eq. 'mrs02nn') then
             mode=2
             call mrst2002(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif     (pdlabel .eq. 'mrs0119') then
             mode=1
             call mrst2001(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs0117') then
             mode=2
             call mrst2001(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs0121') then
             mode=3
             call mrst2001(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs01_j') then
             mode=4
             call mrst2001(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif     (pdlabel .eq. 'mrs99_1') then
             mode=1
             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs99_2') then
             mode=2
             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs99_3') then
             mode=3
             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs99_4') then
             mode=4
             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs99_5') then
             mode=5
             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs99_6') then
             mode=6
             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs99_7') then
             mode=7
             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs99_8') then
             mode=8
             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs99_9') then
             mode=9
             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs9910') then
             mode=10
             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs9911') then
             mode=11
             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs9912') then
             mode=12
             call mrs99(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs98z1') then
             mode=1
             call mrs98(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs98z2') then
             mode=2 
             call mrs98(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs98z3') then
             mode=3
             call mrs98(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs98z4') then
             mode=4
             call mrs98(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs98z5') then
             mode=5
             call mrs98(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs98l1') then
             mode=1
             call mrs98lo(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs98l2') then
             mode=2 
             call mrs98lo(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs98l3') then
             mode=3
             call mrs98lo(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs98l4') then
             mode=4
             call mrs98lo(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs98l5') then
             mode=5
             call mrs98lo(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             elseif (pdlabel .eq. 'mrs98ht') then
             mode=1
             call mrs98ht(x,xmu,mode,u_val,d_val,u_sea,d_sea,
     &                          s_sea,c_sea,b_sea,gluon)
             endif
c-----assign mrs to standard grid
            fx(-5)=b_sea/x
            fx(-4)=c_sea/x
            fx(-3)=s_sea/x
            fx( 0)=gluon/x
            fx(+3)=fx(-3)
            fx(+4)=fx(-4)
            fx(+5)=fx(-5)
               fx(1)=(d_val+d_sea)/x
               fx(2)=(u_val+u_sea)/x
               fx(-1)=d_sea/x
               fx(-2)=u_sea/x
C
      elseif (pdlabel(1:5) .eq. 'cteq3') then
C     
         if (pdlabel .eq. 'cteq3_m') then
            mode=1
         elseif (pdlabel .eq. 'cteq3_l') then
            mode=2
         elseif (pdlabel .eq. 'cteq3_d') then
            mode=3
         endif
         fx(-5)=Ctq3df(mode,-5,x,xmu,Irt)/x
         fx(-4)=Ctq3df(mode,-4,x,xmu,Irt)/x
         fx(-3)=Ctq3df(mode,-3,x,xmu,Irt)/x
         
         fx(0)=Ctq3df(mode,0,x,xmu,Irt)/x
         
         fx(+3)=Ctq3df(mode,+3,x,xmu,Irt)/x
         fx(+4)=Ctq3df(mode,+4,x,xmu,Irt)/x
         fx(+5)=Ctq3df(mode,+5,x,xmu,Irt)/x
            fx(-1)=Ctq3df(mode,-2,x,xmu,Irt)/x
            fx(-2)=Ctq3df(mode,-1,x,xmu,Irt)/x
            fx(1)=Ctq3df(mode,+2,x,xmu,Irt)/x+fx(-1)
            fx(2)=Ctq3df(mode,+1,x,xmu,Irt)/x+fx(-2)
C     
      elseif (pdlabel(1:5) .eq. 'cteq4') then
C     
         if (pdlabel .eq. 'cteq4_m') then
            mode=1
         elseif (pdlabel .eq. 'cteq4_d') then
            mode=2
         elseif (pdlabel .eq. 'cteq4_l') then
            mode=3
         elseif (pdlabel .eq. 'cteq4a1') then
            mode=4
         elseif (pdlabel .eq. 'cteq4a2') then
            mode=5
         elseif (pdlabel .eq. 'cteq4a3') then
            mode=6
         elseif (pdlabel .eq. 'cteq4a4') then
            mode=7
         elseif (pdlabel .eq. 'cteq4a5') then
            mode=8
         elseif (pdlabel .eq. 'cteq4hj') then
            mode=9
         elseif (pdlabel .eq. 'cteq4lq') then
            mode=10
         endif
         
         fx(-5)=Ctq4Fn(mode,-5,x,xmu)
         fx(-4)=Ctq4Fn(mode,-4,x,xmu)
         fx(-3)=Ctq4Fn(mode,-3,x,xmu)
         
         fx(0)=Ctq4Fn(mode,0,x,xmu)
         
         fx(+3)=Ctq4Fn(mode,+3,x,xmu)
         fx(+4)=Ctq4Fn(mode,+4,x,xmu)
         fx(+5)=Ctq4Fn(mode,+5,x,xmu)
            fx(1)=Ctq4Fn(mode,+2,x,xmu)
            fx(2)=Ctq4Fn(mode,+1,x,xmu)
            fx(-1)=Ctq4Fn(mode,-2,x,xmu)
            fx(-2)=Ctq4Fn(mode,-1,x,xmu)
C
      elseif (pdlabel .eq. 'cteq5l1') then
C
         fx(-5)=Ctq5L(-5,x,xmu)
         fx(-4)=Ctq5L(-4,x,xmu)
         fx(-3)=Ctq5L(-3,x,xmu)
         
         fx(0)=Ctq5L(0,x,xmu)
         
         fx(+3)=Ctq5L(+3,x,xmu)
         fx(+4)=Ctq5L(+4,x,xmu)
         fx(+5)=Ctq5L(+5,x,xmu)
         
            fx(1)=Ctq5L(+2,x,xmu)
            fx(2)=Ctq5L(+1,x,xmu)
            fx(-1)=Ctq5L(-2,x,xmu)
            fx(-2)=Ctq5L(-1,x,xmu)
C         
      elseif ((pdlabel(1:5) .eq. 'cteq5') .or. 
     .        (pdlabel(1:4) .eq. 'ctq5')) then
C         
         fx(-5)=Ctq5Pdf(-5,x,xmu)
         fx(-4)=Ctq5Pdf(-4,x,xmu)
         fx(-3)=Ctq5Pdf(-3,x,xmu)
         
         fx(0)=Ctq5Pdf(0,x,xmu)
         
         fx(+3)=Ctq5Pdf(+3,x,xmu)
         fx(+4)=Ctq5Pdf(+4,x,xmu)
         fx(+5)=Ctq5Pdf(+5,x,xmu)
         
            fx(1)=Ctq5Pdf(+2,x,xmu)
            fx(2)=Ctq5Pdf(+1,x,xmu)
            fx(-1)=Ctq5Pdf(-2,x,xmu)
            fx(-2)=Ctq5Pdf(-1,x,xmu)
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
      
  

