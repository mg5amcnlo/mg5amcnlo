      subroutine run_printout
      implicit none
c
c     local
c
      integer i,iformat
      character*2 ab(2)
      real*8 ene
      double precision  Zero, One, Two, Three, Four, Half, Rt2
      parameter( Zero = 0.0d0, One = 1.0d0, Two = 2.0d0 )
c
c     include
c
      include 'PDF/pdf.inc'
      include 'maxparticles.inc'
      include 'run.inc'
      include 'alfas.inc'
c
c output all info
c
      write(6,*)            
      write(6,*)  'Collider parameters:'
      write(6,*)  '--------------------'

      do i=1,2
         IF(LPP(i).EQ. 0) ab(i)='e'
         IF(LPP(i).EQ. 1) ab(i)='P'
         IF(LPP(i).EQ.-1) ab(i)='Pb'
         IF(LPP(i).EQ.2) ab(i)='a2'
         IF(LPP(i).EQ.3) ab(i)='a3'
      enddo

      ene=2d0*dsqrt(ebeam(1)*ebeam(2))

      write(6,*)  
      write(6,*) 'Running at ',ab(1),ab(2),'  machine @ ', ene, ' GeV'
      write(6,*) 'PDF set = ',pdlabel
      write(6,'(1x,a12,1x,f6.4,a12,i1,a7)') 
     &     'alpha_s(Mz)=', asmz ,' running at ', nloop , ' loops.'
      if(lpp(1).ne.0.or.lpp(2).ne.0) then    
      write(6,'(1x,a12,1x,f6.4,a12,i1,a7)') 
     &     'alpha_s(Mz)=', asmz ,' running at ', nloop , ' loops. Value tuned to the PDF set.'
      else 
      write(6,'(1x,a12,1x,f6.4,a12,i1,a7)') 
     &     'alpha_s(Mz)=', asmz ,' running at ', nloop , ' loops. Value set in param_card.dat'
      endif      
           
      if(fixed_ren_scale) then
         write(6,*) 'Renormalization scale fixed @ ',scale 
      else
         write(6,*) 'Renormalization scale set on event-by-event basis'
      endif
      if(fixed_fac_scale) then
         write(6,*) 'Factorization scales  fixed @ ',
     &   dsqrt(q2fact(1)),dsqrt(q2fact(2)) 
      else
         write(6,*) 'Factorization   scale set on event-by-event basis'
      endif
   
      write(6,*)  
      write(6,*)  
      
      return
      end

