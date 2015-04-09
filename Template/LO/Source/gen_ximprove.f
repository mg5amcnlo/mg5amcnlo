      program gen_ximprove
c*****************************************************************************
c     Program to combine results from all of the different sub amplitudes 
c     and given total  cross section and error.
c*****************************************************************************
      implicit none
c
c     Constants
c
      character*(*) rfile
      parameter (rfile='results.dat')
      character*(*) symfile
      parameter (symfile='symfact.dat')

      include 'maxparticles.inc'
      include 'run_config.inc'
      include 'maxconfigs.inc'
c
c     global
c
      integer max_np,min_iter
      common/max_np/max_np,min_iter

c
c     local
c
      double precision xsec(lmaxconfigs), xerr(lmaxconfigs)
      double precision xerru(lmaxconfigs),xerrc(lmaxconfigs)
      double precision xmax(lmaxconfigs), eff(lmaxconfigs)
      double precision xlum(lmaxconfigs)
      double precision ysec, yerr, yeff, ymax
      double precision tsec, terr, teff, tmax, xi
      integer nw(lmaxconfigs), nevents(lmaxconfigs), maxit
      integer nunwgt(lmaxconfigs)
      character*80 fname, gname(lmaxconfigs)
      integer i,j,k,l,n,ipp
      double precision xtot,errtot,err_goal
      double precision errtotu,errtotc
      integer mfact(lmaxconfigs)
      logical parallel, gen_events
      character*20 param(maxpara),value(maxpara)
      integer npara, nreq, ngran, nhel_refine
      integer ij, kl, ioffset
      integer*8 iseed     !tjs 20/6/2012 to avoid integer overflow
      logical Gridpack,gridrun
      logical split_channels
      common /to_split/split_channels
      integer ncode,npos
      character*20 formstr
      logical file_exists
      character*30 filename

c-----
c  Begin Code
c-----
      call load_para(npara,param,value)
      call get_logical(npara,param,value," gridpack ",gridpack,.false.)
      call get_integer(npara,param,value," nhel ",nhel_refine,0)
c     If different card options set for nhel_refine and nhel_survey:
      call get_integer(npara,param,value," nhel_refine ",nhel_refine,
     $     1*nhel_refine)
      if (.not. Gridpack) then
         write(*,'(a,a)')'Enter fractional accuracy (<1)',
     &        ', or number events (>1), max processes per job',
     &        ', and whether to split channels (T/F)'
         read(5,*) err_goal, max_np, split_channels
         min_iter=3
         parallel = .false.
         if (err_goal .lt. 1) then
            write(*,'(a,f8.2,a)') 'Running for accuracy of ',
     $           err_goal*100,'%'
            gen_events=.false.         
         elseif (err_goal .gt. 1) then
            write(*,'(a,f9.0,a)') 'Generating ',err_goal,
     &           ' unweighted events.'
            gen_events=.true.         
            err_goal = err_goal * 1.2 !Extra factor to ensure works
         else
            write(*,*) 'Error, need non_zero goal'
            stop
         endif
      else
         gen_events=.true.
         split_channels=.false.
c        Allow all the way down to a single iteration for gridruns
         min_iter=1
         call get_integer(npara,param,value," gevents "  ,nreq  ,2000   )
         err_goal = 1.2*nreq ! extra factor to ensure works
         call get_int8(npara,param,value," gseed "  ,iseed  ,4321   )
         call get_integer(npara,param,value," ngran "  ,ngran  , -1)
         if (ngran.eq.-1) ngran = 1
         write(*,*) "Running on Grid to generate ",nreq," events"
         write(*,*) "   with granularity equal to ",ngran
c
c     TJS 3/13/2008
c     Modified to allow for more sequences
c     iseed can be between 0 and 30081*30081
c     before patern repeats
c     JA 11/2/2011: Check for ioffset, as in ntuple (ranmar.f)
c     TJS  20/6/2012 changed mod value to 30081 to avoid duplicate sequences
c
         call get_offset(ioffset)
         iseed = iseed * 31300       
         ij=1802 + mod(iseed,30081)      
         kl=9373 + (iseed/30081) + ioffset
         write(*,'($a,i6,a3,i6)') 'Using random seed offset: ',ioffset
         write(*,*) ' with seed', iseed
         do while (ij .gt. 31328)
            ij = ij - 31328
         enddo
         do while (kl .gt. 30081)
            kl = kl - 30081
         enddo
         write(*,*) "Using random seeds",ij,kl
        call rmarin(ij,kl)
      endif
      open(unit=15,file=symfile,status='old',err=999)
      errtot=0d0
      errtotu=0d0
      errtotc=0d0
      xtot = 0d0
      i = 0
c     ncode is number of digits needed for the bw coding
      ncode=int(dlog10(3d0)*(max_particles-3))+1
      do while (.true.)
         read(15,*,err=99,end=99) xi,j
         if (j .gt. 0) then
            i = i+1
            k = int(xi*(1+10**(-ncode)))
            npos=int(dlog10(dble(k)))+1
            if ( (xi-k) .eq. 0) then
c              Write with correct number of digits
               write(formstr,'(a,i1,a)') '(a,i',npos,',a,a)'
               write(fname, formstr) 'G',k,'/',rfile
            else               !Handle B.W.
c              Write with correct number of digits
               write(formstr,'(a,i1,a,i1,a)') '(a,f',npos+ncode+1,
     $                 '.',ncode,',a,a)'
               write(fname, formstr) 'G',xi,'/',rfile
            endif
c              write(*,*) 'log name ',fname
         endif
         if (j .gt. 0) then
            gname(i)=fname
            nevents(i)=0d0
            xsec(i)=0d0
            xerr(i)=0d0
            nw(i)  =0d0
            mfact(i)=j

c
c     Read in integration data from run
c
            open(unit=25,file=fname,status='old',err=95)
            read(25,*,err=94,end=94) xsec(i),xerru(i),xerrc(i),nevents(i),nw(i),maxit,
     &           nunwgt(i),xlum(i)
            if (xsec(i) .eq. 0d0) xlum(i)=1d99     !zero cross section
            xlum(i) = xlum(i)/1000   !convert to fb^-1 
            xerr(i)=sqrt(xerru(i)**2+xerrc(i)**2)
            if (.false.) then
c            maxit = 2
               tmax = -1d0
               terr = 0d0
               teff = 0d0
               tsec = 0d0
               do k=1,maxit
                  read(25,*,err=92) l,ysec,yerr,yeff,ymax
                  if (k .gt. 1) tmax = max(tmax,ymax)
                  tsec = tsec + ysec
                  terr = terr +yerr**2
                  teff = teff + yeff
               enddo
 92            maxit = k-1      !In case of error reading file
               xsec(i)=tsec/maxit
               xerr(i)=sqrt(terr)/maxit
               xmax(i)=tmax/xsec(i)
            endif
c            tmax
            xmax(i) = -1d0
            xsec(i) = xsec(i)*mfact(i)
            xerr(i) = xerr(i)*mfact(i)
            xerru(i) = xerru(i)*mfact(i)
            xerrc(i) = xerrc(i)*mfact(i)
            xlum(i) = xlum(i)/mfact(i)
            xtot = xtot+ xsec(i)
            eff(i)= xerr(i)*sqrt(real(nevents(i)))/(xsec(i)+1d-99)
            errtotu = errtotu+(xerru(i))**2
            errtotc = errtotc+(xerrc(i))
c            xtot = xtot+ xsec(i)*mfact(i)
c            eff(i)= xerr(i)*sqrt(real(nevents(i)))/xsec(i)
c            errtot = errtot+(mfact(i)*xerr(i))**2
            goto 95
 94         continue
c        There was an error reading an existing results.dat file
c        Stop generation with error message
            filename='../../error'
            INQUIRE(FILE="../../RunWeb", EXIST=file_exists)
            if(.not.file_exists) filename = '../' // filename
            open(unit=26,file=filename,status='unknown')
            write(26,*) 'Bad results.dat file for channel ',xi
 95         close(25)
c            write(*,*) i,maxit,xsec(i), eff(i)
         else
c            i=i-1   !This is for case w/ B.W. and optimization
         endif
      enddo
 99   close(15)
      errtot=sqrt(errtotc**2+errtotu)
      if ( .not. gen_events) then
         call write_bash(xsec,xerru,xerrc,xtot,mfact,err_goal,
     $        i,nevents,gname,nhel_refine)
      else
         open(unit=25,file='../results.dat',status='old',err=199)
         read(25,*) xtot
         write(*,'(a,e12.3)') 'Reading total xsection ',xtot
 199     close(25)
         if (gridpack) then
            call write_gen_grid(err_goal,dble(ngran),i,nevents,gname,
     $           xlum,xtot,mfact,xsec,nhel_refine)
         else
            call write_gen(err_goal,i,nevents,gname,xlum,xtot,mfact,
     $           xsec,xerr,nhel_refine)
         endif
      endif
      stop
 999  write(*,*) 'error'
      end


      subroutine write_bash(xsec,xerru,xerrc,xtot,
     $     mfact,err_goal,ng,jpoints,gn,nhel_refine)
c*****************************************************************************
c     Writes out bash commands for running each channel as needed.
c*****************************************************************************
      implicit none
c
c     Constants
c
      include 'maxparticles.inc'
      include 'run_config.inc'
      include 'maxconfigs.inc'

c      integer    max_np
c      parameter (max_np = 30)
c
c     global
c
      integer max_np,min_iter
      common/max_np/max_np,min_iter
c
c     Arguments
c
      double precision xsec(lmaxconfigs), xerru(lmaxconfigs),xerrc(lmaxconfigs)
      double precision err_goal,xtot
      integer mfact(lmaxconfigs),jpoints(lmaxconfigs),nhel_refine
      integer ng
      character*(80) gn(lmaxconfigs)
c
c     Local
c
      integer i,j,k, io(lmaxconfigs), npoints, ip, np
      double precision xt(lmaxconfigs),elimit
      double precision yerr,ysec,rerr
      logical fopened

c-----
c  Begin Code
c-----
      fopened = .false.
      k=0
      do j=1,ng
         if (mfact(j) .gt. 0) k=k+1
         io(j) = j
         xt(j)= sqrt((xerru(j)+xerrc(j)**2)*mfact(j))     !sort by error
      enddo
c
c     Let's redetermine err_goal based on luminosity
c
      write(*,*) 'Cross section pb',xtot
      write(*,*) 'Desired Goal',err_goal      
      write(*,*) 'Total Error',err_goal
c      elimit = err_goal*xtot/sqrt(real(k)) !Equal contributions from each
      elimit = err_goal*xtot/real(k) !Equal contributions from each

      call sort2(xt,io,ng)
      k=1
      xt(ng+1) = 0
      do while( xt(k) .gt. abs(elimit))  !abs is just in case elimit<0 by mistake
         k=k+1
      enddo
      k=k-1
      rerr=0d0
      do j=k+1,ng
c         rerr = rerr+xt(j)**2
         rerr = rerr+xt(j)
      enddo
      rerr=rerr**2
c      write(*,*) 'Number of diagrams to fix',k
c
c     Now readjust because most don't contribute
c
      elimit = sqrt((err_goal*xtot)**2 - rerr)/sqrt(real(k))

      
      np = max_np
      do i=1,k

c         yerr = xerr(io(i))*mfact(io(i))
         yerr = xt(i)
c         write(*,*) i,xt(i),elimit
         if (yerr .gt. elimit) then

         ysec = xsec(io(i)) + yerr
         npoints=(0.2d0)*jpoints(io(i))*(yerr/elimit)**2
         npoints = max(npoints,min_events)
         npoints = min(npoints,max_events)
c         np = np + 3*npoints
         np = np +1
         if (np .gt. max_np) then
            if (fopened) then
               call close_bash_file(26)
            endif
            fopened=.true.
            call open_bash_file(26)
c            np = 3*npoints
            np = 1
         endif

         ip = index(gn(io(i)),'/')
         write(*,*) 'Channel ',gn(io(i))(2:ip-1),
     $        yerr, jpoints(io(i)),npoints

         ip = index(gn(io(i)),'/')
         write(26,'(2a)') 'j=',gn(io(i))(1:ip-1)
c
c     Determine estimates for getting the desired accuracy
c

c
c     Now write the commands
c      
         write(26,20) 'if [[ ! -e $j ]]; then'
         write(26,25) 'mkdir $j'
         write(26,20) 'fi'
         write(26,20) 'cd $j'
         write(26,20) 'rm -f $k'
c         write(26,20) 'rm -f moffset.dat'

         write(26,'(5x,a,3i8,a)') 'echo "',npoints,max_iter,min_iter,
     $        '" >& input_sg.txt' 
         write(26,'(5x,a,f8.3,a)') 'echo "',max(elimit/ysec,0.001d0),
     $        '" >> input_sg.txt'
         write(26,'(5x,a)') 'echo "2" >> input_sg.txt'  !Grid
         write(26,'(5x,a)') 'echo "1" >> input_sg.txt'  !Suppress
         write(26,'(5x,a,i4,a)') 'echo "',nhel_refine,
     &        '"  >> input_sg.txt' !Helicity 
         write(26,'(5x,3a)')'echo "',gn(io(i))(2:ip-1),
     $        '" >>input_sg.txt'
      write(26,20) 'for((try=1;try<=16;try+=1)); '
      write(26,20) 'do'
         write(26,20) '../madevent >> $k <input_sg.txt'
         write(26,25) 'if [ -s $k ]'
         write(26,25) 'then'
         write(26,25) '    break'
         write(26,25) 'else'
         write(26,25) '    echo $try > fail.log '
         write(26,25) 'fi'
         write(26,25) 'done'
         write(26,20) 'rm ftn25 ftn26'
         write(26,20) 'cat $k >> log.txt'
         write(26,20) 'echo "" >> $k; echo "ls status:" >> $k; ls >> $k'
         write(26,20) 'cd ../'
      endif
      enddo  !Loop over diagrams
      if (fopened) then
         call close_bash_file(26)
      endif
      fopened=.false.
 15   format(a)
 20   format(5x,a)
 25   format(10x,a)
 999  close(26)
      end


      subroutine open_bash_file(lun)
c***********************************************************************
c     Opens bash file for looping including standard header info
c     which can be used with pbs, or stand alone
c***********************************************************************
      implicit none
c
c     Constants
c
      include 'maxparticles.inc'
      include 'run_config.inc'
c
c     Arguments
c
      integer lun
c
c     local
c
      character*30 fname
      integer ic, npos
      character*10 formstr

      data ic/0/
c-----
c  Begin Code
c-----
      ic=ic+1
      fname='ajob'
c     Write ic with correct number of digits
      npos=int(dlog10(dble(ic)))+1
      write(formstr,'(a,i1,a)') '(I',npos,')'
      write(fname(5:(5+npos-1)),formstr) ic

      write(*,*) 'Opening file ',fname
      open (unit=26, file = fname, status='unknown')
      write(26,15) '#!/bin/bash'
c      write(26,15) '#PBS -q ' // PBS_QUE
c      write(26,15) '#PBS -o /dev/null'
c      write(26,15) '#PBS -e /dev/null'
c      write(26,15) 'if [[ "$PBS_O_WORKDIR" != "" ]]; then' 
c      write(26,15) '    cd $PBS_O_WORKDIR'
c      write(26,15) 'fi'
      write(26,15) 'if [[ -e MadLoop5_resources.tar.gz && ! -e MadLoop5_resources ]]; then'
      write(26,15) 'tar -xzf MadLoop5_resources.tar.gz'
      write(26,15) 'fi'

      write(26,15) 'k=run1_app.log'
      write(lun,15) 'script=' // fname
c      write(lun,15) 'rm -f wait.$script >& /dev/null'
c      write(lun,15) 'touch run.$script'
 15   format(a)
      end

      subroutine close_bash_file(lun)
c***********************************************************************
c     Closes bash file
c***********************************************************************
      implicit none
c
c     Constants
c
c
c     Arguments
c
      integer lun
c
c     local
c
      character*30 fname
      integer ic

      data ic/0/
c-----
c  Begin Code
c-----

c      write(lun,'(a)') ')'
c
c     Now write the commands
c      
c      write(lun,20) 'j=G$i'
c      write(lun,20) 'if (! -e $j) then'
c      write(lun,25) 'mkdir $j'
c      write(lun,20) 'endif'
c      write(lun,20) 'cd $j'
c      write(lun,20) 'rm -f ftn25 ftn99'
c      write(lun,20) 'rm -f $k'
c      write(lun,20) 'cat ../input_app.txt >& input_app.txt'
c      write(lun,20) 'echo $i >> input_app.txt'
c      if (.false.) then
c         write(lun,20) 'cp ../../public.sh .'
c         write(lun,20) 'qsub -N $1$i public.sh >> ../../running_jobs'
c      else
c         write(lun,20) '../madevent > $k <input_app.txt'
c         write(lun,20) 'rm -f ftn25 ftn99'
c         write(lun,20) 'cp $k log.txt'
c      endif
c      write(lun,20) 'cd ../'
c      write(lun,15) 'end'
c      write(lun,15) 'rm -f run.$script >&/dev/null'
c      write(lun,15) 'touch done.$script >&/dev/null'
 15   format(a)
 20   format(5x,a)
 25   format(10x,a)
      close(lun)
      end



      subroutine write_gen(goal_lum,ng,jpoints,gn,xlum,xtot,mfact,xsec,
     $     xerr,nhel_refine)
c*****************************************************************************
c     Writes out scripts for achieving unweighted event goals
c*****************************************************************************
      implicit none
c
c     Constants
c
      include 'maxparticles.inc'
      include 'run_config.inc'
      include 'maxconfigs.inc'
c
c     global
c
      integer max_np,min_iter
      common/max_np/max_np,min_iter
c      integer    max_np     !now set in run_config.inc
c      parameter (max_np = 5)  !number of channels/job

c
c     Arguments
c
      double precision goal_lum, xlum(lmaxconfigs), xsec(lmaxconfigs),xtot
      double precision xerr(lmaxconfigs)
      integer jpoints(lmaxconfigs), mfact(lmaxconfigs)
      integer ng, np, nhel_refine
      character*(80) gn(lmaxconfigs)
c
c     Local
c
      integer i,j,k,kk, io(lmaxconfigs), npoints, ip, nfiles,ifile,npfile
      double precision xt(lmaxconfigs+1),elimit
      double precision yerr,ysec,rerr
      logical fopened
      character*26 cjobs
      integer mjobs,ijob,jc
      character*150 fname

      logical split_channels
      common /to_split/split_channels

      data cjobs/"abcdefghijklmnopqrstuvwxyz"/

c-----
c  Begin Code
c-----
      fopened=.false.
      write(*,*) 'Working on creating ', goal_lum, ' events.'
      goal_lum = goal_lum/(xtot*1000) !Goal luminosity in fb^-1
      write(*,*) 'Effective Luminosity', goal_lum, ' fb^-1.'
      k=0
      do j=1,ng
         io(j) = j
         xt(j)= goal_lum/(xlum(j)+1d-99)       !sort by events_needed/have.
         write(*,*) j,xlum(j),xt(j)
      enddo
c      write(*,*) 'Number of channels',ng,k

c    Reset multijob.dat for all channels
      do j=1,ng
        jc = index(gn(j),"/")
        fname = gn(j)(1:jc)// "multijob.dat"
        write(*,*) 'Resetting ' // fname
        open(unit=15,file=fname,status="unknown",err=10)
        write(15,*) 0
 10     close(15)
      enddo
c
c     Let's redetermine err_goal based on luminosity
c
      elimit = 1d0
      call sort2(xt,io,ng)
      k=1
      xt(ng+1) = 0
      do while( xt(k) .gt. abs(elimit)) !elimit should be >0
         write(*,*) 'Improving ',k,gn(io(k)),xt(k)
         k=k+1
      enddo
      kk=k
c     Check error for the rest of the channels - rerun if 
c     bigger than channel xsec and bigger than 1% of largest channel
      do while( kk .le. ng)
         if (xerr(io(kk)).gt.max(xsec(io(kk)),0.01*xsec(io(1)))) then
            write(*,*) 'Improving for error ',kk,gn(io(kk)),xt(kk),xsec(io(kk)),xerr(io(kk))
            io(k)=io(kk)
            xt(k)=xt(kk)
            k=k+1
         endif
         kk=kk+1
      enddo
      k=k-1
      write(*,*) 'Number of diagrams to fix',k
c
c     Now readjust because most don't contribute
c

c      np = max_np

c
c     Want to write channels so that heaviest one (with largest error)
c     gets grouped with least heavy channels. Complicated ordering for this
c     follows. np is the present process number.
c
      nfiles = k/max_np
      if(mod(k,max_np).gt.0) nfiles=nfiles+1
      ifile  = 0
      npfile = 0
      np = 1
      

      do i=1,k
         yerr = xt(np)
         npoints=0.2*jpoints(io(np))*(yerr/elimit)
         npoints = max(npoints,min_events)
         npoints = min(npoints,max_events)

         npfile=npfile+1
c         np = nfiles*npfile+1-ifile   !Fancy order for combining channels removed 12/6/2010 by tjs
         np = i
c
c     tjs 12/5/2010
c     Add loop to allow for multiple jobs on a single channel
c
         mjobs = (goal_lum*xsec(io(np))*1000 / MaxEventsPerJob + 0.9)
c         write(*,*) "Working on Channel ",i,io(np),xt(np), goal_lum*xsec(io(np))*1000 /maxeventsperjob
         if (mjobs .gt. 130)  then
            write(*,*) 'Error in gen_ximprove.f, too many events requested ',mjobs*maxeventsperjob
            mjobs=130
         endif
         if (mjobs .lt. 1 .or. .not. split_channels)  mjobs=1
c
c        write multijob.dat file for combine_runs.f 
c
         jc = index(gn(io(np)),"/")
         fname = gn(io(np))(1:jc)// "multijob.dat"
c            write(*,*) "Writing file ", fname
         open(unit=15,file=fname,status="unknown",err=11)
         if (mjobs .gt. 1) then
           write(15,*) mjobs
         else
           write(15,*) 0
         endif
 11      close(15)
         do ijob = 1, mjobs
c---
c tjs
c---
         if (npfile .gt. max_np .or. ifile.eq.0 .or. mjobs .gt. 1) then
            if (fopened) then
               call close_bash_file(26)
            endif
            fopened=.true.
            call open_bash_file(26)
            ifile=ifile+1
            npfile=1
c            if (ijob .eq. 1)  np = ifile !Only increment once / source channel
         endif
         ip = index(gn(io(np)),'/')
         write(*,*) 'Channel ',gn(io(np))(2:ip-1),
     $        yerr, jpoints(io(np)),npoints

         ip = index(gn(io(np)),'/')
         if (mjobs .gt. 1) then

            if (ip.eq.3) then
                write(26,'(a2,a2,a,i1)') 'j=',gn(io(np))(1:ip-1),cjobs(MODULO(ijob-1,26)+1:MODULO(ijob-1,26)+1),
     &                                              ijob/26
            else if(ip.eq.4) then
                write(26,'(a2,a3,a,i1)') 'j=',gn(io(np))(1:ip-1),cjobs(MODULO(ijob-1,26)+1:MODULO(ijob-1,26)+1),
     &                                              ijob/26
            else if(ip.eq.5) then
               write(26,'(a2,a4,a,i1)') 'j=',gn(io(np))(1:ip-1),cjobs(MODULO(ijob-1,26)+1:MODULO(ijob-1,26)+1),
     &                                              ijob/26
            else if(ip.eq.6) then
               write(26,'(a2,a5,a,i1)') 'j=',gn(io(np))(1:ip-1),cjobs(MODULO(ijob-1,26)+1:MODULO(ijob-1,26)+1),
     &                                              ijob/26
           else
               stop 1
           endif
         else
             write(26,'(3a)') 'j=',gn(io(np))(1:ip-1)
         endif
c
c     Now write the commands
c      
         write(26,20) 'if [[ ! -e $j ]]; then'
         write(26,25) 'mkdir $j'
         write(26,20) 'fi'
         write(26,20) 'cd $j'
         write(26,20) 'rm -f $k'
         write(26,20) 'rm -f moffset.dat >& /dev/null'
         write(26,*) '     echo ',ijob, ' > moffset.dat'

c
c     
c
c
c     Now I'll add a check to make sure the grid has been
c     adjusted  (ftn99 or ftn25 exist)
c
         write(26,20) 'if  [[ -e ftn26 ]]; then'
         write(26,25) 'cp ftn26 ftn25'
         write(26,20) 'fi'

         write(26,20) 'if [[ ! -e ftn25 ]]; then'


         write(26,'(9x,a,3i8,a)') 'echo "',npoints,max_iter,min_iter,
     $        '" >& input_sg.txt' 
c
c     tjs 8/7/2007-JA 8/17/11 Allow stop when have enough luminocity
c
         write(*,*) "Cross section",i,io(np),xsec(io(np)),mfact(io(np))
         write(26,'(9x,a,e13.5,a)') 'echo "',-goal_lum*1000/mjobs,
     $        '" >> input_sg.txt'                       !Luminocity
         write(26,'(9x,a)') 'echo "2" >> input_sg.txt'  !Grid Adjustment
         write(26,'(9x,a)') 'echo "1" >> input_sg.txt'  !Suppression
         write(26,'(9x,a,i4,a)') 'echo "',nhel_refine,
     &        ' " >> input_sg.txt' !Helicity 0=exact
         write(26,'(9x,3a)')'echo "',gn(io(np))(2:ip-1),
     $        '" >>input_sg.txt'
         write(26,25) 'for((try=1;try<=16;try+=1)); '
         write(26,25) 'do'
         write(26,25) '../madevent >> $k <input_sg.txt'
         write(26,25) 'if [ -s $k ]'
         write(26,25) 'then'
         write(26,25) '    break'
         write(26,25) 'else'
         write(26,25) '    echo $try > fail.log '
         write(26,25) 'fi'
         write(26,25) 'done'
         write(26,20) 'echo "" >> $k; echo "ls status:" >> $k; ls >> $k'
         write(26,25) 'cat $k >> log.txt'
         write(26,25) 'if [[ -e ftn26 ]]; then'
         write(26,25) '     cp ftn26 ftn25'
         write(26,25) 'fi'
         write(26,20) 'else'

         write(26,25) 'rm -f $k'

         write(26,'(9x,a,3i8,a)') 'echo "',npoints,max_iter,min_iter,
     $        '" >& input_sg.txt' 
c
c tjs 8/7/2007-JA 8/17/11    Change to request luminocity not accuracy
c
         write(26,'(9x,a,e13.5,a)') 'echo "',-goal_lum*1000/mjobs,
     $        '" >> input_sg.txt'                       !Luminocity
c         write(26,'(9x,a,e12.3,a)') 'echo "',-goal_lum*mfact(io(np)),
c     $        '" >> input_sg.txt'
         write(26,'(9x,a)') 'echo "0" >> input_sg.txt'
         write(26,'(9x,a)') 'echo "1" >> input_sg.txt'

         write(26,'(9x,a,i4,a)') 'echo "',nhel_refine,
     &        ' " >> input_sg.txt' !Helicity 0=exact

         write(26,'(9x,3a)')'echo "',gn(io(np))(2:ip-1),
     $        '" >>input_sg.txt'


c         write(26,'(9x,a)') 'echo "1" >> input_sg.txt' !Helicity 0=exact

c         write(26,'(5x,3a)')'echo "',gn(io(np))(2:ip-1),
c     $        '" >>input_sg.txt'
c         write(26,20) 'cp ../../public_sg.sh .'
c         write(26,20) 'qsub -N $1$j public_sg.sh >> ../../running_jobs'
         write(26,25) 'if [[ -e ftn26 ]]; then'
         write(26,25) '     cp ftn26 ftn25'
         write(26,25) 'fi'
                  write(26,25) 'for((try=1;try<=16;try+=1)); '
         write(26,25) 'do'
         write(26,25) '../madevent >> $k <input_sg.txt'
         write(26,25) 'if [ -s $k ]'
         write(26,25) 'then'
         write(26,25) '    break'
         write(26,25) 'else'
         write(26,25) '    echo $try > fail.log '
         write(26,25) 'fi'
         write(26,25) 'done'
         write(26,20) 'echo "" >> $k; echo "ls status:" >> $k; ls >> $k'
         write(26,25) 'cat $k >> log.txt'
         write(26,20) 'fi'
         write(26,20) 'cd ../'
c------
c  tjs  end loop over split process   
c------
      enddo  !(ijob, split channel)         

      enddo !(k  each channel)
      if (fopened) then
         call close_bash_file(26)
      endif
c      write(26,15) 'end'
 15   format(a)
 20   format(5x,a)
 25   format(10x,a)
 999  close(26)
      end


      subroutine write_gen_grid(goal_lum,ngran,ng,jpoints,gn,xlum,xtot,mfact,xsec,nhel_refine)
c*****************************************************************************
c     Writes out scripts for achieving unweighted event goals
c*****************************************************************************
      implicit none
c
c     Constants
c
      include 'maxparticles.inc'
      include 'run_config.inc'
      include 'maxconfigs.inc'
c
c   global
c
      integer max_np,min_iter
      common/max_np/max_np,min_iter
c
c     Arguments
c
      double precision goal_lum, xlum(lmaxconfigs), xsec(lmaxconfigs),xtot
      double precision ngran   !Granularity.... min # points from channel
      integer jpoints(lmaxconfigs), mfact(lmaxconfigs)
      integer ng, np, nhel_refine
      character*(80) gn(lmaxconfigs)
c
c     Local
c
      integer i,j,k, npoints, ip
      double precision xt(lmaxconfigs),elimit
      double precision yerr,ysec,rerr
      character*72 fname
      logical fopened
      double precision rvec
c-----
c  Begin Code
c-----

c      data ngran /10/
      fopened=.false.
c
c     These random #'s should be changed w/ run
c
c      ij=2134
c      kl = 4321
      rvec=0d0
      write(*,*) 'Working on creating ', goal_lum, ' events.'
      max_np = 1
      np = max_np   !Flag to open csh file
      do i=1,ng
         call ranmar(rvec)
         ip = index(gn(i),'/')
         fname = gn(i)(1:ip) // 'gscalefact.dat'
         open(unit=27,file=fname,status='unknown',err=91)
         if (goal_lum * xsec(i)/xtot  .ge. rvec*ngran ) then !need events
            write(*,*) 'Requesting events from ',gn(i)(1:ip-1),xsec(i),xtot/goal_lum
            if (xsec(i) .gt. xtot*ngran/goal_lum) then
               write(27,*) 1d0
            else
               write(27,*) xtot*ngran/xsec(i)/goal_lum
            endif
            npoints = goal_lum * xsec(i) / xtot
            if (npoints .lt. ngran) npoints = ngran
            np = np+1
            if (np .gt. max_np) then
               if (fopened) then
                  call close_bash_file(26)
               endif
               fopened=.true.
               call open_bash_file(26)
               np = 1
            endif
            ip = index(gn(i),'/')
            write(*,*) 'Channel ',gn(i)(2:ip-1), goal_lum * xsec(i) / xtot,
     $           npoints

            ip = index(gn(i),'/')
            write(26,'(2a)') 'j=',gn(i)(1:ip-1)
c
c           Now write the commands
c      
            write(26,20) 'if [[ ! -e $j ]]; then'
            write(26,25) 'mkdir $j'
            write(26,20) 'fi'
            write(26,20) 'cd $j'
            write(26,20) 'rm -f $k'
c
c     Now I'll add a check to make sure the grid has been
c     adjusted  (ftn99 or ftn25 exist)
c
            write(26,20) 'if  [[ -e ftn26 ]]; then'
            write(26,25) 'cp ftn26 ftn25'
            write(26,20) 'fi'

            write(26,20) 'if [[ ! -e ftn25 ]]; then'


            write(26,'(9x,a,3i8,a)') 'echo "',max(npoints,min_events),
     $           max_iter,min_iter,'" >& input_sg.txt' 
c
c     tjs 8/7/2007  Allow stop when have enough events
c
            write(*,*) "Cross section",i,xsec(i),mfact(i)
            write(26,'(9x,a,e13.5,a)') 'echo "',-npoints/xsec(i),
     $        '" >> input_sg.txt'                       !Luminocity
            write(26,'(9x,a)') 'echo "2" >> input_sg.txt' !Grid Adjustment
            write(26,'(9x,a)') 'echo "1" >> input_sg.txt' !Suppression
            write(26,'(9x,a,i4,a)') 'echo "',nhel_refine,
     &           ' " >> input_sg.txt' !Helicity 0=exact
            write(26,'(9x,3a)')'echo "',gn(i)(2:ip-1),
     $           '" >>input_sg.txt'
                  write(26,25) 'for((try=1;try<=16;try+=1)); '
         write(26,25) 'do'
            write(26,25) '../madevent >> $k <input_sg.txt'
         write(26,25) 'if [ -s $k ]'
         write(26,25) 'then'
         write(26,25) '    break'
         write(26,25) 'else'
         write(26,25) '    echo $try > fail.log '
         write(26,25) 'fi'
         write(26,25) 'done'
            write(26,20) 'echo "" >> $k; echo "ls status:" >> $k; ls >> $k'
            write(26,25) 'cat $k >> log.txt'
            write(26,25) 'if [[ -e ftn26 ]]; then'
            write(26,25) '     cp ftn26 ftn25'
            write(26,25) 'fi'
            write(26,20) 'else'

            write(26,25) 'rm -f $k'
            
            write(26,'(9x,a,3i8,a)') 'echo "',max(npoints,min_events),
     $           max_iter,min_iter,'" >& input_sg.txt' 
c
c tjs 8/7/2007    Change to request events not accuracy
c
            write(26,'(9x,a,e13.5,a)') 'echo "',-npoints / xsec(i),
     $           '" >> input_sg.txt' ! Luminocity
            write(26,'(9x,a)') 'echo "0" >> input_sg.txt'
            write(26,'(9x,a)') 'echo "1" >> input_sg.txt'

            write(26,'(9x,a,i4,a)') 'echo "',nhel_refine,
     &           ' " >> input_sg.txt' !Helicity 0=exact

            write(26,'(9x,3a)')'echo "',gn(i)(2:ip-1),
     $           '" >>input_sg.txt'

            write(26,25) 'if [[ -e ftn26 ]]; then'
            write(26,25) '     cp ftn26 ftn25'
            write(26,25) 'fi'
            write(26,25) 'for((try=1;try<=16;try+=1)); '
         write(26,25) 'do'
            write(26,25) '../madevent >> $k <input_sg.txt'
                     write(26,25) 'if [ -s $k ]'
         write(26,25) 'then'
         write(26,25) '    break'
         write(26,25) 'else'
         write(26,25) '    echo $try > fail.log '
         write(26,25) 'fi'
         write(26,25) 'done'
            write(26,20) 'echo "" >> $k; echo "ls status:" >> $k; ls >> $k'
            write(26,25) 'cat $k >> log.txt'
            write(26,20) 'fi'
            write(26,20) 'cd ../'
         else    !No events from this channel
            write(*,*) 'Skipping channel:',gn(i)(1:ip-1),xsec(i)*goal_lum/xtot,rvec
            write(27,*) 0d0
         endif
         close(27)
 91      cycle
      enddo
      call close_bash_file(26)
 15   format(a)
 20   format(5x,a)
 25   format(10x,a)
 999  close(26)
      close(27)
      end


      subroutine sort2(array,aux1,n)
      implicit none
! Arguments
      integer n
      integer aux1(n)
      double precision array(n)
!  Local Variables
      integer i,k
      double precision temp
      logical done

!-----------
! Begin Code
!-----------
      do i=n-1,1,-1
         done = .true.
         do k=1,i
            if (array(k) .lt. array(k+1)) then
               temp = array(k)
               array(k) = array(k+1)
               array(k+1) = temp
               temp = aux1(k)
               aux1(k) = aux1(k+1)
               aux1(k+1) = temp
               done = .false.
            end if
         end do
         if (done) return
      end do
      end 


      subroutine get_xsec_log(xsec,xerr,eff,xmax)
c*********************************************************************
c     Reads from output file, gets cross section and maxwgt from
c     first two iterations
c*********************************************************************
      implicit none
c
c     Arguments
c
      double precision xsec(2),xerr(2),eff(2),xmax(2)
c
c     Local
c     
      character*78 buff
      integer i
c-----
c  Begin Code
c-----
      xsec(1) = 0d0
      xerr(1) = 0d0
      xmax(1) = 0d0
      do while (.true.)
         read(25,'(a80)',err=99) buff
         if (buff(1:4) .eq. 'Iter') then
            read(buff(11:16),'(i5)') i
            if (i .eq. 1 .or. i .eq. 2) then
               read(buff(61:70),*) xmax(i)
               read(buff(21:33),*) xsec(i)
               xmax(i)=xmax(i)/xsec(i)
c               read(buff(48:59),*) xerr(i)
c               read(buff(48:59),*) xmax(i)
            endif
            read(25,'(a80)',err=99) buff
            read(buff(1:6),'(i5)') i
            if (i .eq. 1 .or. i .eq. 2) then
               read(buff(6:17),*) xsec(i)
               read(buff(20:31),*) xerr(i)
               read(buff(34:40),*) eff(i)               
            endif
            write(*,'(i4,4f12.3)') i,xsec(i),xerr(i),eff(i),xmax(i)
         endif
      enddo
 99   end




