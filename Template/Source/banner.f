      subroutine create_banner(lunw, swgt, nw)
      implicit none
c      include 'nexternal.inc'
c
c     parameters
c
      integer    MaxParticles
      parameter (MaxParticles=15)
c
c     Arguments
c
      integer lunw,  nw
      double precision swgt(99999)
c
c     Local
c
      integer ic(7,MaxParticles),next, luni
      double precision P(0:3,MaxParticles),wgt
      real*8 sum,mxwgt
      logical done
      integer i,imax,j, nevent, nfound
      character*35 infile,outfile
      integer iseed
      data iseed/9999/
      character*30 process,QED,QCD
c--cuts
c      double precision etmin(3:nexternal),etamax(3:nexternal)
c      double precision                    r2min(3:nexternal,3:nexternal)
c      double precision s_min(nexternal,nexternal)
c      common/to_cuts/  etmin     ,etamax     , r2min, s_min
c
c     open the Process/input.dat
c
      luni = 67
      infile='Process/input.dat'
      i=0
      done=.false.
      do while (.not. done .and. i .lt. 5)
c         write(*,*) 'Attempting to open file ',infile
         open(unit=luni,file=infile,status='old',err=101)
         done = .true.
 101     i = i+1
         infile='../' // infile         
      enddo
      if (done) then
         read(luni,'(a30)') process
         read(luni,'(a30)') QCD
         read(luni,'(a30)') QED
         close(luni)
      else
         process = '???? Process'
         QCD     = '???? QCD'
         QED     = '???? QED'
      endif

c      call get_seed(iseed)
c
c     All the info is gathered. Now start writing it out.
c

      call write_para(lunw)
      write(lunw,'(a70)') '##                                                                    '
      write(lunw,'(a70)') '##===================                                                 '
      write(lunw,'(a70)') '## Run-time options                                                   '
      write(lunw,'(a70)') '##===================                                                 '
      write(lunw,'(a70)') '##                                                                    '
      write(lunw,'(a70)') '##********************************************************************'     
      write(lunw,'(a70)') '## Random Number seed                                                *'
      write(lunw,'(a70)') '##********************************************************************'    
      write(lunw,'(a3,i8,a)') '## ',iseed,'   = iseed'    
c      write(lunw,'(a70)') '##********************************************************************'     
c      write(lunw,'(a70)') '## Standard Cuts                                                     *'
c      write(lunw,'(a70)') '##********************************************************************'    
c      write(lunw,'(a13,8i8)')   '## Particle  ',(i,i=3,nexternal)
c      write(lunw,'(a13,8f8.1)') '## Et       >',(etmin(i),i=3,nexternal)
c      write(lunw,'(a13,8f8.1)') '## Eta      <',(etamax(i),i=3,nexternal)
c      do j=3,nexternal-1
c         write(lunw,'(a,i2,a,8f8.1)') '## d R #',j,'  >',(-0.0,i=3,j),
c     &        (r2min(i,j),i=j+1,nexternal)
c         do i=j+1,nexternal
c            r2min(i,j)=r2min(i,j)**2 !Since r2 returns distance squared
c         enddo
c      enddo
c      do j=1,nexternal-1
c         write(lunw,'(a,i2,a,8f8.1)') '## s min #',j,'>',
c     &        (s_min(i,j),i=3,nexternal)
c      enddo
c      write(lunw,'(a70)') '##********************************************************************'    
c
c     Now write out specific information on the event set
c
      done=.false.
      nevent=0
      nfound=0
      sum=0d0
      mxwgt=-1d0
      do i=1,nw
         sum=sum+swgt(i)
         mxwgt = max(swgt(i),mxwgt)
      enddo
      nevent = nw
      write(lunw,'(a70)') '##                                                                    '
      write(lunw,'(a70)') '##===================                                                 '
      write(lunw,'(a70)') '## Event information                                                  '
      write(lunw,'(a70)') '##===================                                                 '
      write(lunw,'(a70)') '##                                                                    '
      write(lunw,'(a70)') '##********************************************************************'    
      write(lunw,'(a12,a30)') '## Process: ',process                
      write(lunw,'(a12,a30)') '## Max QCD: ',QCD
      write(lunw,'(a12,a30)') '## Max QED: ',QED              
      write(lunw,'(a70)') '##********************************************************************'    
      write(lunw,'(a30,i10)')   '##  Number of Events       :  ',nevent
      write(lunw,'(a30,e10.5)') '##  Integrated weight (pb) :  ',sum
      write(lunw,'(a30,e10.5)') '##  Max wgt                :  ',mxwgt
      write(lunw,'(a30,e10.5)') '##  Average wgt            :  ',sum/nevent
      write(lunw,'(a70)') '##********************************************************************'    

c      close(lunw)

      end
