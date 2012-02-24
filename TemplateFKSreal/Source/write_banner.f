      program write_banner
      implicit none
c
c     parameters
c
      integer    MaxParticles
      parameter (MaxParticles=15)
c
c     Local
c
      integer lunr, lunw,luni
      data luni,lunr,lunw/14,15,16/ ! unit numbers for reading and writing
      integer ic(7,MaxParticles),next
      double precision P(0:4,MaxParticles),wgt
      real*8 sum,mxwgt
      logical done
      integer i,imax,j
      integer nevent,nfound
      character*25 infile,outfile
      integer iseed
      data iseed/9999/
      character*30 process,QED,QCD
      double precision scale,aqcd,aqed
      integer ievent
c--cuts
c      double precision etmin(3:nexternal),etamax(3:nexternal)
c      double precision                    r2min(3:nexternal,3:nexternal)
c      double precision s_min(nexternal,nexternal)
c      common/to_cuts/  etmin     ,etamax     , r2min, s_min
c
c     open the event file
c
      write(*,*) 'input the event file (e.g. Events/events.dat)'
      read(*,'(a)')  infile
      open(unit=lunr,file=infile,status='old')
c
c     open banner file
c
      write(*,*) 'input the banner file name (e.g. banner-events.dat)'
      read(*,'(a)')  outfile
      open(lunw,file=outfile,status='unknown')
c
c     gather the info
c
      call setpara('param_card.dat')
c      call setcuts
c      call get_seed(iseed)
c
c     All the info is gathered. Now start writing it out.
c
      call write_para(lunw)
      write(lunw,'(a70)') '##                                                                    '
      write(lunw,'(a70)') '##-------------------                                                 '
      write(lunw,'(a70)') '## Run-time options                                                   '
      write(lunw,'(a70)') '##-------------------                                                 '
      write(lunw,'(a70)') '##                                                                    '
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
c      do j=3,nexternal-1
c         write(lunw,'(a,i2,a,8f8.1)') '## s min #',j,'>',
c     &        (s_min(i,j),i=3,nexternal)
c      enddo
      write(lunw,'(a70)') '##********************************************************************'    
c
c     Now write out specific information on the event set
c
      done=.false.
      nevent=0
      nfound=0
      sum=0d0
      mxwgt=-1d0
      do while (.not. done)
         call read_event(lunr,P,wgt,next,ic,ievent,scale,aqcd,aqed,done)
         sum=sum+wgt
         mxwgt = max(wgt,mxwgt)
         nevent=nevent+1
      enddo

      write(lunw,'(a70)') '##                                                                    '
      write(lunw,'(a70)') '##-------------------                                                 '
      write(lunw,'(a70)') '## Event information                                                  '
      write(lunw,'(a70)') '##-------------------                                                 '
      write(lunw,'(a70)') '##                                                                    '
      write(lunw,'(a70)') '##********************************************************************'    
      write(lunw,'(a30,i10)')   '##  Number of Events       :  ',nevent
      write(lunw,'(a30,e10.5)') '##  Integrated weight (pb) :  ',sum
      write(lunw,'(a30,e10.5)') '##  Max wgt                :  ',mxwgt
      write(lunw,'(a30,e10.5)') '##  Average wgt            :  ',sum/nevent
      write(lunw,'(a70)') '##********************************************************************'    

      rewind(lunr)
      done = .false.
      nevent=0
      do while (.not. done)
         call read_event(lunr,P,wgt,next,ic,ievent,scale,aqcd,aqed,done)
         nevent=nevent+1
         if (.not. done) then
            call write_event(lunw,p,wgt,next,ic,nevent,scale,aqcd,aqed)
         endif
      enddo

      close(lunw)
      close(lunr)

      end
