      program improve
      implicit none
      logical done
      character*140 buffer
      character*140 DIRNAME
      integer lendirname,start,graph,start2,start3
      character*5 mode
      integer lmode
      common /run_mode/lmode,mode
      integer run_cluster
      common/c_run_mode/run_cluster

c$$$      write (*,'(a)') 'Enter "0" for local run, "1" for condor cluster'
      read  (*,*) run_cluster
      if (run_cluster.ne.0 .and. run_cluster .ne. 1) then
         write (*,*) "Invalid run mode", run_cluster
         stop
      endif
      
      done=.false.
      open (unit=1, file="dname.mg",status="old")
      read(1,'(a)') buffer
      read(buffer(9:index(buffer," ")-1),*) dirname
      lendirname=index(dirname," ")-1
      close(1)

      open (unit=1, file="../res.txt", status="old")
      do while (.not.done)
         read(1,'(a)',err=12,end=12) buffer
         start = index(buffer,DIRNAME(1:lendirname+1))
         if (index(buffer,DIRNAME(1:lendirname+1)).gt.0) then
            if (buffer(110:110).eq."Y") then
               start2=start+index(buffer(start:110)," ")
               do while (index(buffer(start2:110)," ").eq.1)
                  start2=start2+1
               enddo
               start3 = start2+index(buffer(start2:110),"G")
               lmode=start3-start2-2
               read (buffer(start2:start2+6),'(a)') mode
               read(buffer(start3:start3+index(buffer(start3:110),"  ")),*) graph
               write (*,*) mode(1:lmode),graph
               call open_bash_file(26)
               if (graph .lt. 10) then
                  write(26,'(i1$)') graph
               elseif (graph .lt. 100) then
                  write(26,'(i2$)') graph
               elseif (graph .lt. 1000) then
                  write(26,'(i3$)') graph
               elseif (graph .lt. 10000) then
                  write(26,'(i4$)') graph
               endif
               call close_bash_file(26)
            endif
         endif
      enddo
 12   continue
      close(1)

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
      include '../../Source/run_config.inc'
c
c     Arguments
c
      integer lun
c
c     Local
c
      integer ic

      data ic/0/

      character*30 mname
      character*30 fname
      common/to_close_bash_file/fname
      character*5 mode
      integer lmode
      common /run_mode/lmode,mode
      integer run_cluster
      common/c_run_mode/run_cluster

c-----
c  Begin Code
c-----
      ic=ic+1
      fname='ajob'//mode(1:lmode)
      mname='mg'//mode(1:lmode)
      if (ic .lt. 10) then
         write(fname(lmode+5:lmode+5),'(i1)') ic
         write(mname(lmode+3:lmode+7),'(i1,a4)') ic,".cmd"
      elseif (ic .lt. 100) then
         write(fname(lmode+5:lmode+6),'(i2)') ic
         write(mname(lmode+3:lmode+8),'(i2,a4)') ic,".cmd"
      elseif (ic .lt. 1000) then
         write(fname(lmode+5:lmode+7),'(i3)') ic
         write(mname(lmode+3:lmode+9),'(i3,a4)') ic,".cmd"
      endif

      open (unit=lun, file = fname, status='unknown')
      if (run_cluster.eq.0) then
         write(lun,15) '#!/bin/bash'
         write(lun,15) 'script=' // fname
         write(lun,15) 'rm -f wait.$script >& /dev/null'
         write(lun,15) 'touch run.$script'
         write(lun,15) 'echo $script'
         write(lun,'(a$)') 'for i in '
      elseif(run_cluster.eq.1) then
         write(lun,15) '#!/bin/bash'
         write(lun,15) 'if [ -n "$_CONDOR_SCRATCH_DIR" ]; then'
         write(lun,15) '    CONDOR_INITIAL_DIR=`pwd`'
         write(lun,15) '    cd $_CONDOR_SCRATCH_DIR'
         write(lun,15) 'fi'
         write(lun,15) 'mkdir lib'
         write(lun,15) 'mkdir Cards'
         write(lun,15) 'mkdir SubProcesses'
         write(lun,15) 'cp -a  $CONDOR_INITIAL_DIR/'//
     &        '../../MGMEVersion.txt .'
         write(lun,15) '#cp -ra  $CONDOR_INITIAL_DIR/'//
     &        '../../Source .'
         write(lun,15) 'cp -ra  $CONDOR_INITIAL_DIR/'//
     &        '../../Cards/* ./Cards/'
         write(lun,15) 'cp -ra  $CONDOR_INITIAL_DIR/'//
     &        '../../lib/Pdfdata ./lib/'
c         write(lun,15) 'cp -ra  $CONDOR_INITIAL_DIR/'//
c     &        '../../lib/PDFsets ./lib/'
c         write(lun,15) 'cd ./lib/PDFsets'
c         write(lun,15) 'gunzip *.gz'
c         write(lun,15) 'cd ../../'

         write(lun,15) "if [[ $1 == '0' ]]; then"
         write(lun,15) '    cp -a  $CONDOR_INITIAL_DIR/'//
     &           'madevent_vegas ./SubProcesses'
         write(lun,15) "elif [[ $1 == '1' ]]; then"
         write(lun,15) '    cp -a  $CONDOR_INITIAL_DIR/'//
     &           'madevent_mint ./SubProcesses'
         write(lun,15) "elif [[ $1 == '2' ]]; then"
         write(lun,15) '    cp -a  $CONDOR_INITIAL_DIR/'//
     &           'madevent_mintMC ./SubProcesses'
         write(lun,15) "fi"

         write(lun,15) 'cd SubProcesses'
         write(lun,'(a$)') 'for i in '
      endif

      if (run_cluster.eq.1) then
         open (unit=lun+1,file=mname,status='unknown')
         write(lun+1,15) 'universe = vanilla'
         write(lun+1,15) 'executable = '//fname
         write(lun+1,15) 'output = /dev/null'
         write(lun+1,15) 'error = /dev/null'
         write(lun+1,15) 'requirements = (MADGRAPH == True)'
         write(lun+1,15) 'log = /dev/null'
         if (mode(1:lmode).eq.'novi') then
            write(lun+1,15) 'Arguments = 0 novi grid'
         elseif (mode(1:lmode).eq.'virt0') then
            write(lun+1,15) 'Arguments = 0 virt0 born0'
         else
            write(lun+1,15) 'Arguments = 0 '//mode(1:lmode)
         endif
         write(lun+1,15) 'queue'
         write(lun+1,15) ''
         close(lun+1)
      endif

 15   format(a)
      end

      subroutine close_bash_file(lun)
c***********************************************************************
c     Opens bash file for looping including standard header info
c     which can be used with pbs, or stand alone
c***********************************************************************
      implicit none
c
c     Constants
c
c
c     Constants
c
c     Arguments
c
      integer lun
c
c     local
c
      integer ic,j

      data ic/0/

      character*30 fname
      common/to_close_bash_file/fname
      integer run_cluster
      common/c_run_mode/run_cluster

c-----
c  Begin Code
c-----

      write(lun,'(a)') '; do'
c
c     Now write the commands
c      
      if (run_cluster.eq.0) then
         write(lun,20) 'echo $i'
         write(lun,20) 'echo $i >& run.$script'
c madevent_vegas or madevent_mint
         write(lun,20) "if [[ $1 == '0' || $1 == '1' ]]; then"
         write(lun,25) 'j=$2\_G$i'
         write(lun,25) 'if [[ ! -e $j ]]; then'
         write(lun,30) 'mkdir $j'
         write(lun,25) 'fi'
         write(lun,25) 'cd $j'
         write(lun,25) 'if [[ "$3" != "" ]]; then'
         write(lun,30) 'if [[ -e ../$3\_G$i ]]; then'
         write(lun,35) "if [[ $1 == '0' ]]; then"
         write(lun,40) 'cp -f ../$3\_G$i/*.sv1 .'
         write(lun,35) "elif [[ $1 == '1' ]]; then"
         write(lun,40) 'cp -f ../$3\_G$i/mint_grids .'
         write(lun,35) "fi"
         write(lun,30) 'else'
         write(lun,35) 'echo "Cannot find direcotry ../$3\_G$i/"'//
     &        ' > log.txt'
         write(lun,35) 'exit'
         write(lun,30) 'fi'
         write(lun,25) 'fi'
c madevent_mintMC
         write(lun,20) "elif [[ $1 == '2' ]]; then"
         write(lun,25) 'j=G$2$i'
         write(lun,25) 'if [[ ! -e $j ]]; then'
         write(lun,30) 'mkdir $j'
         write(lun,25) 'fi'
         write(lun,25) 'cd $j'
         write(lun,25) 'if [[ "$3" != "" ]]; then'
         write(lun,30) 'if [[ -e ../G$3$i ]]; then'
         write(lun,35) 'cp -f ../G$3$i/mint_grids ./preset_mint_grids'
         write(lun,30) 'else'
         write(lun,35) 'echo "Cannot find direcotry ../G$3$i/"'//
     &        ' > log.txt'
         write(lun,35) 'exit'
         write(lun,30) 'fi'
         write(lun,25) 'fi'
c endif
         write(lun,20) "fi"

         write(lun,20) 'cp ../config.fks .'
         write(lun,20) 'cp ../integrate.fks .'
         write(lun,20) 'cp ../nbodyonly.fks .'
         write(lun,20) 'cp ../iproc.dat .'
         write(lun,20) 'if [[ -e ../../randinit ]]; then'
         write(lun,25) 'cp ../../randinit .'
         write(lun,20) 'fi'
         write(lun,20) 'cp ../symfact.dat .'
         write(lun,20) 'if [[ -e ../HelasNLO.input ]]; then'
         write(lun,25) 'cp -f ../HelasNLO.input .'
         write(lun,20) 'fi'
         write(lun,20) 'if [[ -e ../MadLoop.param ]]; then'
         write(lun,25) 'cp -f ../MadLoop.param .'
         write(lun,20) 'fi'
         write(lun,20) 'if [[ -e ../order.file ]]; then'
         write(lun,25) 'cp -f ../order.file .'
         write(lun,20) 'fi'
c madevent_vegas
         write(lun,20) "if [[ $1 == '0' ]]; then"
         write(lun,25) 'head -n 5 ../../madin.$2 >& input_app.txt'
         write(lun,25) 'echo $i >> input_app.txt'
         write(lun,25) 'tail -n 4 ../../madin.$2 >> input_app.txt'
         write(lun,25) 'time ../madevent_vegas > log.txt <input_app.txt'
c madevent_mint
         write(lun,20) "elif [[ $1 == '1' ]]; then"
         write(lun,25) 'head -n 5 ../../madinM.$2 >& input_app.txt'
         write(lun,25) 'echo $i >> input_app.txt'
         write(lun,25) 'tail -n 3 ../../madinM.$2 >> input_app.txt'
         write(lun,25) 'time ../madevent_mint > log.txt <input_app.txt'
c madevent_mintMC
         write(lun,20) "elif [[ $1 == '2' ]]; then"
         write(lun,25) 'head -n 6 ../../madinMMC_$2.2 >& input_app.txt'
         write(lun,25) 'echo $i >> input_app.txt'
         write(lun,25) 'tail -n 4 ../../madinMMC_$2.2 >> input_app.txt'
         write(lun,25)
     &        'time ../madevent_mintMC > log.txt <input_app.txt'
c endif
         write(lun,20) "fi"
         write(lun,20) 'cd ../'
         write(lun,15) 'done'
         write(lun,15) 'rm -f run.$script'
         write(lun,15) 'touch done.$script'

      elseif(run_cluster.eq.1) then
c madevent_vegas or madevent_mint
         write(lun,20) "if [[ $1 == '0' || $1 == '1' ]]; then"
         write(lun,25) 'j=$2\_G$i'
         write(lun,25) 'if [[ ! -e $j ]]; then'
         write(lun,30) 'mkdir $j'
         write(lun,25) 'fi'
         write(lun,25) 'cd $j'
         write(lun,25) 'cp -f $CONDOR_INITIAL_DIR/$j/* .'
         write(lun,25) 'if [[ "$3" != "" ]]; then'
         write(lun,30) 'if [[ -e $CONDOR_INITIAL_DIR/$3\_G$i ]]; then'
         write(lun,35) "if [[ $1 == '0' ]]; then"
         write(lun,40) 'cp -f $CONDOR_INITIAL_DIR/$3\_G$i/*.sv1 .'
         write(lun,35) "elif [[ $1 == '1' ]]; then"
         write(lun,40) 'cp -f $CONDOR_INITIAL_DIR/$3\_G$i/mint_grids .'
         write(lun,35) "fi"
         write(lun,30) 'else'
         write(lun,35) 'echo "Cannot find direcotry ../$3\_G$i/"'//
     &        ' > log2.txt'
         write(lun,30) 'fi'
         write(lun,25) 'fi'

c madevent_mintMC
         write(lun,20) "elif [[ $1 == '2' ]]; then"
         write(lun,25) 'j=G$2$i'
         write(lun,25) 'if [[ ! -e $j ]]; then'
         write(lun,30) 'mkdir $j'
         write(lun,25) 'fi'
         write(lun,25) 'cd $j'
         write(lun,25) 'cp -f $CONDOR_INITIAL_DIR/$j/* .'
         write(lun,25) 'if [[ "$3" != "" ]]; then'
         write(lun,30) 'if [[ -e $CONDOR_INITIAL_DIR/G$3$i ]]; then'
         write(lun,35) 'cp -f $CONDOR_INITIAL_DIR/G$3$i/mint_grids '//
     &        './preset_mint_grids'
         write(lun,30) 'else'
         write(lun,35) 'echo "Cannot find direcotry ../G$3$i/"'//
     &        ' > log.txt'
         write(lun,35) 'exit'
         write(lun,30) 'fi'
         write(lun,25) 'fi'
c endif
         write(lun,20) "fi"

         write(lun,20) 'cp -f $CONDOR_INITIAL_DIR/config.fks .'
         write(lun,20) 'cp -f $CONDOR_INITIAL_DIR/iproc.dat .'
         write(lun,20)
     &        'if [[ -e $CONDOR_INITIAL_DIR/../randinit ]]; then'
         write(lun,25) 'cp -f $CONDOR_INITIAL_DIR/../randinit .'
         write(lun,20) 'fi'
         write(lun,20) 'cp -f $CONDOR_INITIAL_DIR/integrate.fks .'
         write(lun,20) 'cp -f $CONDOR_INITIAL_DIR/nbodyonly.fks .'
         write(lun,20) 'cp -f $CONDOR_INITIAL_DIR/symfact.dat .'
         write(lun,20)
     &        'if [[ -e $CONDOR_INITIAL_DIR/HelasNLO.input ]]; then'
         write(lun,25) 'cp -f $CONDOR_INITIAL_DIR/HelasNLO.input .'
         write(lun,20) 'fi'
         write(lun,20)
     &        'if [[ -e $CONDOR_INITIAL_DIR/MadLoop.param ]]; then'
         write(lun,25) 'cp -f $CONDOR_INITIAL_DIR/MadLoop.param .'
         write(lun,20) 'fi'
         write(lun,20)
     &        'if [[ -e $CONDOR_INITIAL_DIR/order.file ]]; then'
         write(lun,25) 'cp -f $CONDOR_INITIAL_DIR/order.file .'
         write(lun,20) 'fi'

c madevent_vegas
         write(lun,20) "if [[ $1 == '0' ]]; then"
         write(lun,25) 'head -n 5 $CONDOR_INITIAL_DIR/'//
     &        '../madin.$2 >& input_app.txt'
         write(lun,25) 'echo $i >> input_app.txt'
         write(lun,25) 'tail -n 4 $CONDOR_INITIAL_DIR/'//
     &        '../madin.$2 >> input_app.txt'
         write(lun,25) 'time ../madevent_vegas > log.txt <input_app.txt'
c madevent_mint
         write(lun,20) "elif [[ $1 == '1' ]]; then"
         write(lun,25) 'head -n 5 $CONDOR_INITIAL_DIR/'//
     &        '../madinM.$2 >& input_app.txt'
         write(lun,25) 'echo $i >> input_app.txt'
         write(lun,25) 'tail -n 3 $CONDOR_INITIAL_DIR/'//
     &        '../madinM.$2 >> input_app.txt'
         write(lun,25) 'time ../madevent_mint > log.txt <input_app.txt'
c madevent_mintMC
         write(lun,20) "elif [[ $1 == '2' ]]; then"
         write(lun,25) 'head -n 6 $CONDOR_INITIAL_DIR/'//
     &        '../madinMMC_$2.2 >& input_app.txt'
         write(lun,25) 'echo $i >> input_app.txt'
         write(lun,25) 'tail -n 4 $CONDOR_INITIAL_DIR/'//
     &        '../madinMMC_$2.2 >> input_app.txt'
         write(lun,25)'time ../madevent_mintMC > log.txt <input_app.txt'
c endif
         write(lun,20) "fi"

         write(lun,20) 'cd ../'
         write(lun,15) 'done'
c madevent_vegas or madevent_mint
         write(lun,15) "if [[ $1 == '0' || $1 == '1' ]]; then"
         write(lun,20) 'cp -ar $2\_G* $CONDOR_INITIAL_DIR/'
c madevent_mintMC
         write(lun,15) "elif [[ $1 == '2' ]]; then"
         write(lun,20) 'cp -ar G$2* $CONDOR_INITIAL_DIR/'
         write(lun,20) 
c endif
         write(lun,15) "fi"
      endif
      
 15   format(a)
 20   format(4x,a)
 25   format(8x,a)
 30   format(12x,a)
 35   format(16x,a)
 40   format(20x,a)
      close(lun)
      end



      subroutine bw_increment_array(iarray,imax,ibase,force,done)
c************************************************************************
c     Increments iarray     
c************************************************************************
      implicit none
c
c     Arguments
c
      integer imax          !Input, number of elements in iarray
      integer ibase         !Base for incrementing, 0 is skipped
      logical force(imax)   !Force onshell BW, counting from -imax to -1
      integer iarray(imax)  !Output:Array of values being incremented
      logical done          !Output:Set when no more incrementing

c
c     Global
c
      include 'genps.inc'

c
c     Local
c
      integer i,j
      logical found
c-----
c  Begin Code
c-----
      found = .false.
      i = 1
      do while (i .le. imax .and. .not. found)
         if (iarray(i) .eq. 0) then    !don't increment this
            i=i+1
         elseif (iarray(i) .lt. ibase-1 .and. .not. force(imax+1-i)) then
            found = .true.
            iarray(i)=iarray(i)+1
         else
            iarray(i)=1
            i=i+1
         endif
      enddo
      done = .not. found
      end

c
c
c Dummy routines
c
c
      subroutine outfun(pp,www,iplot)
c      write(*,*)'This routine should not be called here'
c      stop
      end



      logical function pass_point(p)
      pass_point = .true.
      end

      LOGICAL FUNCTION PASSCUTS(P,rwgt)
      real*8 rwgt
      real*8 p(0:3,99)
      rwgt=1d0
      passcuts=.true.
      RETURN
      END

