
      subroutine open_bash_file(lun,fname,lname,mname)
c***********************************************************************
c     Opens bash file for looping including standard header info
c     which can be used with pbs, or stand alone
c***********************************************************************
      implicit none
c
c     Constants
c
c      include '../../Source/run_config.inc'
c
c     Arguments
c
      integer lun
c
c     Local
c
      integer ic

      data ic/0/

      integer lname
      character*30 mname
      character*30 fname
c$$$      common/to_close_bash_file/lname,fname

      integer run_cluster
      common/c_run_mode/run_cluster

c-----
c  Begin Code
c-----
      ic=ic+1
      if (ic .lt. 10) then
         write(fname(5:5),'(i1)') ic
         write(mname(3:7),'(i1,a4)') ic,".cmd"
         lname=lname+1
      elseif (ic .lt. 100) then
         write(fname(5:6),'(i2)') ic
         write(mname(3:8),'(i2,a4)') ic,".cmd"
         lname=lname+2
      elseif (ic .lt. 1000) then
         write(fname(5:7),'(i3)') ic
         write(mname(3:9),'(i3,a4)') ic,".cmd"
         lname=lname+3
      endif
      open (unit=lun, file = fname, status='unknown')
      if (run_cluster.eq.0) then
         write(lun,15) '#!/bin/bash'
         write(lun,15) 'script=' // fname(1:lname)//'.$1.$2.$4'
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
         write(lun,15) 'script=' // mname(1:lname+2)//'.$1.$2.$4'
         write(lun,15) 'rm -f $CONDOR_INITIAL_DIR/wait.$script '//
     &        '>& /dev/null'
         write(lun,15) 'touch $CONDOR_INITIAL_DIR/run.$script'
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
         write(lun,15) 'ln -s  $CONDOR_INITIAL_DIR/'//
     &        '../../lib/PDFsets ./lib/'
c         write(lun,15) 'cp -ra  $CONDOR_INITIAL_DIR/'//
c     &        '../../lib/PDFsets ./lib/'
c         write(lun,15) 'cd ./lib/PDFsets'
c         write(lun,15) 'gunzip *.gz'
c         write(lun,15) 'cd ../../'

         write(lun,15) 'cp -ra  $CONDOR_INITIAL_DIR/'//
     &        'MadLoopParams.dat .'
         write(lun,15) 'cp -ra  $CONDOR_INITIAL_DIR/'//
     &        'ColorDenomFactors.dat .'
         write(lun,15) 'cp -ra  $CONDOR_INITIAL_DIR/'//
     &        'ColorNumFactors.dat .'
         write(lun,15) 'cp -ra  $CONDOR_INITIAL_DIR/'//
     &        'HelConfigs.dat .'

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
         write(lun+1,15) ''
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
         write(lun,25) 'if [[ "$4" != "" ]]; then'
         write(lun,30) 'if [[ -e ../$4\_G$i ]]; then'
         write(lun,35) "if [[ $1 == '0' ]]; then"
         write(lun,40) 'cp -f ../$4\_G$i/*.sv1 .'
         write(lun,40)
     &        'cp -f ../$4\_G$i/grid.MC_integer . >/dev/null 2>&1'
         write(lun,35) "elif [[ $1 == '1' ]]; then"
         write(lun,40) 'cp -f ../$4\_G$i/mint_grids .'
         write(lun,40)
     &        'cp -f ../$4\_G$i/grid.MC_integer . >/dev/null 2>&1'
         write(lun,35) "fi"
         write(lun,30) 'else'
         write(lun,35) 'echo "Cannot find direcotry ../$4\_G$i/"'//
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
         write(lun,25) 'if [[ "$4" != "" ]]; then'
         write(lun,30) 'if [[ -e ../G$4$i ]]; then'
         write(lun,35) 'cp -f ../G$4$i/mint_grids ./preset_mint_grids'
         write(lun,35)
     &        'cp -f ../G$4$i/grid.MC_integer . >/dev/null 2>&1'
         write(lun,30) 'else'
         write(lun,35) 'echo "Cannot find direcotry ../G$4$i/"'//
     &        ' > log.txt'
         write(lun,35) 'exit'
         write(lun,30) 'fi'
         write(lun,25) 'fi'
c endif
         write(lun,20) "fi"

         write(lun,20) 'if [[ -e ../../randinit ]]; then'
         write(lun,25) 'cp -f ../../randinit .'
         write(lun,20) 'fi'
         write(lun,20) 'if [[ -e ../symfact.dat ]]; then'
         write(lun,25) 'ln -sf ../symfact.dat .'
         write(lun,20) 'fi'
         write(lun,20) 'if [[ -e ../MadLoopParams.dat ]]; then'
         write(lun,25) 'ln -sf ../MadLoopParams.dat .'
         write(lun,20) 'fi'
         write(lun,20)
     &        'if [[ -e ../ColorDenomFactors.dat ]]; then'
         write(lun,25) 'ln -sf ../ColorDenomFactors.dat .'
         write(lun,20) 'fi'
         write(lun,20) 'if [[ -e ../HelConfigs.dat ]]; then'
         write(lun,25) 'ln -sf ../HelConfigs.dat .'
         write(lun,20) 'fi'
         write(lun,20) 'if [[ -e ../ColorNumFactors.dat ]]; then'
         write(lun,25) 'ln -sf ../ColorNumFactors.dat .'
         write(lun,20) 'fi'
c madevent_vegas
         write(lun,20) "if [[ $1 == '0' ]]; then"
         write(lun,25) 'head -n 5 ../../madin.$2 >& input_app.txt'
         write(lun,25) 'echo $i >> input_app.txt'
         write(lun,25) 'tail -n 4 ../../madin.$2 >> input_app.txt'
         write(lun,25) 'T="$(date +%s)"'
         write(lun,25) 'time ../madevent_vegas > log.txt <input_app.txt'
         write(lun,25) 'T="$(($(date +%s)-T))"'
         write(lun,25) 'echo "Time in seconds: ${T}" >>log.txt'
c madevent_mint
         write(lun,20) "elif [[ $1 == '1' ]]; then"
         write(lun,25) 'head -n 5 ../../madinM.$2 >& input_app.txt'
         write(lun,25) 'echo $i >> input_app.txt'
         write(lun,25) 'tail -n 3 ../../madinM.$2 >> input_app.txt'
         write(lun,25) 'T="$(date +%s)"'
         write(lun,25) 'time ../madevent_mint > log.txt <input_app.txt'
         write(lun,25) 'T="$(($(date +%s)-T))"'
         write(lun,25) 'echo "Time in seconds: ${T}" >>log.txt'
c madevent_mintMC
         write(lun,20) "elif [[ $1 == '2' ]]; then"
         write(lun,25) "if [[ $3 == '0' || $3 == '2' ]]; then"
         write(lun,30) 'head -n 6 ../../madinMMC_$2.2 >& input_app.txt'
         write(lun,30) 'echo $i >> input_app.txt'
         write(lun,30) 'tail -n 4 ../../madinMMC_$2.2 >> input_app.txt'
         write(lun,25) "elif [[ $3 == '1' ]] ; then"
         write(lun,30) 'head -n 6 madinM1 >& input_app.txt'
         write(lun,30) 'echo $i >> input_app.txt'
         write(lun,30) 'tail -n 4 madinM1 >> input_app.txt'
         write(lun,25) "fi"
         write(lun,25) 'T="$(date +%s)"'
         write(lun,25)
     &        'time ../madevent_mintMC > log.txt <input_app.txt'
         write(lun,25) 'T="$(($(date +%s)-T))"'
         write(lun,25) 'echo "Time in seconds: ${T}" >>log.txt'
c endif
         write(lun,20) "fi"
         write(lun,20) 'cd ../'
         write(lun,15) 'done'
         write(lun,15) 'rm -f run.$script'
         write(lun,15) 'touch done.$script'

      elseif(run_cluster.eq.1) then
         write(lun,20) 'echo $i >& $CONDOR_INITIAL_DIR/run.$script'
         write(lun,20) 'runnumber=0'
c madevent_vegas or madevent_mint
         write(lun,20) "if [[ $1 == '0' || $1 == '1' ]]; then"
         write(lun,25) 'j=$2\_G$i'
         write(lun,25) 'if [[ ! -e $j ]]; then'
         write(lun,30) 'mkdir $j'
         write(lun,25) 'fi'
         write(lun,25) 'cd $j'
         write(lun,25) 'cp -f $CONDOR_INITIAL_DIR/$j/* .'
         write(lun,25) 'if [[ "$4" != "" ]]; then'
         write(lun,30) 'if [[ -e $CONDOR_INITIAL_DIR/$4\_G$i ]]; then'
         write(lun,35) "if [[ $1 == '0' ]]; then"
         write(lun,40) 'cp -f $CONDOR_INITIAL_DIR/$4\_G$i/*.sv1 .'
         write(lun,40) 'cp -f $CONDOR_INITIAL_DIR/$4\_G$i/'/
     &        /'grid.MC_integer . >/dev/null 2>&1'
         write(lun,35) "elif [[ $1 == '1' ]]; then"
         write(lun,40) 'cp -f $CONDOR_INITIAL_DIR/$4\_G$i/mint_grids .'
         write(lun,40) 'cp -f $CONDOR_INITIAL_DIR/$4\_G$i/'/
     &        /'grid.MC_integer . >/dev/null 2>&1'
         write(lun,35) "fi"
         write(lun,30) 'else'
         write(lun,35) 'echo "Cannot find direcotry ../$4\_G$i/"'//
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
         write(lun,25) 'if [[ "$4" != "" ]]; then'
         write(lun,30) 'if [[ "$4" == "H" ||"$4" == "S" ||'//
     &        ' "$4" == "V" || "$4" == "B" || "$4" == "F" ]]; then'

         write(lun,35) 'if [[ -e $CONDOR_INITIAL_DIR/G$4$i ]]; then'
         write(lun,40) 'cp -f $CONDOR_INITIAL_DIR/G$4$i/mint_grids '//
     &        './preset_mint_grids'
         write(lun,40) 'cp -f $CONDOR_INITIAL_DIR/G$4$i/'/
     &        /'grid.MC_integer . >/dev/null 2>&1'
         write(lun,35) 'else'
         write(lun,40) 'echo "Cannot find direcotry ../G$4$i/"'//
     &        ' > log.txt'
         write(lun,40) 'exit'
         write(lun,35) 'fi'
         write(lun,30) 'else'
         write(lun,35) 'runnumber=$4'
         write(lun,30) 'fi'
         write(lun,25) 'fi'
c endif
         write(lun,20) "fi"
         write(lun,20)
     &        'if [[ -e $CONDOR_INITIAL_DIR/../randinit ]]; then'
         write(lun,25) 'cp -f $CONDOR_INITIAL_DIR/../randinit .'
         write(lun,20) 'fi'
         write(lun,20)
     &        'if [[ -e $CONDOR_INITIAL_DIR/symfact.dat ]]; then'
         write(lun,25) 'cp -f $CONDOR_INITIAL_DIR/symfact.dat .'
         write(lun,20) 'fi'
         write(lun,20)
     &        'if [[ -e $CONDOR_INITIAL_DIR/MadLoopParams.dat ]]; then'
         write(lun,25) 'cp -f $CONDOR_INITIAL_DIR/MadLoopParams.dat .'
         write(lun,20) 'fi'
         write(lun,20)'if [[ -e $CONDOR_INITIAL_DIR'/
     &        /'/ColorDenomFactors.dat ]]; then'
         write(lun,25)
     &        'cp -f $CONDOR_INITIAL_DIR/ColorDenomFactors.dat .'
         write(lun,20) 'fi'
         write(lun,20)
     &        'if [[ -e $CONDOR_INITIAL_DIR/HelConfigs.dat ]]; then'
         write(lun,25) 'cp -f $CONDOR_INITIAL_DIR/HelConfigs.dat .'
         write(lun,20) 'fi'
         write(lun,20)
     &      'if [[ -e $CONDOR_INITIAL_DIR/ColorNumFactors.dat ]]; then'
         write(lun,25) 'cp -f $CONDOR_INITIAL_DIR/ColorNumFactors.dat .'
         write(lun,20) 'fi'

c madevent_vegas
         write(lun,20) "if [[ $1 == '0' ]]; then"
         write(lun,25) 'head -n 5 $CONDOR_INITIAL_DIR/'//
     &        '../madin.$2 >& input_app.txt'
         write(lun,25) 'echo $i >> input_app.txt'
         write(lun,25) 'tail -n 4 $CONDOR_INITIAL_DIR/'//
     &        '../madin.$2 >> input_app.txt'
         write(lun,25) 'T="$(date +%s)"'
         write(lun,25)
     &        'time ../madevent_vegas > log.txt <input_app.txt 2>&1'
         write(lun,25) 'T="$(($(date +%s)-T))"'
         write(lun,25) 'echo "Time in seconds: ${T}" >>log.txt'
c madevent_mint
         write(lun,20) "elif [[ $1 == '1' ]]; then"
         write(lun,25) 'head -n 5 $CONDOR_INITIAL_DIR/'//
     &        '../madinM.$2 >& input_app.txt'
         write(lun,25) 'echo $i >> input_app.txt'
         write(lun,25) 'tail -n 3 $CONDOR_INITIAL_DIR/'//
     &        '../madinM.$2 >> input_app.txt'
         write(lun,25) 'T="$(date +%s)"'
         write(lun,25)
     &        'time ../madevent_mint > log.txt <input_app.txt 2>&1'
         write(lun,25) 'T="$(($(date +%s)-T))"'
         write(lun,25) 'echo "Time in seconds: ${T}" >>log.txt'
c madevent_mintMC
         write(lun,20) "elif [[ $1 == '2' ]]; then"
         write(lun,25) 'if [[ $runnumber != 0 ]]; then'
         write(lun,30) 'mv -f nevts__$runnumber nevts'
         write(lun,30) 'source ./randinit'
         write(lun,30) 'r=`expr $r + $runnumber`'
         write(lun,30) 'echo "r=$r" >& randinit'
         write(lun,25) 'fi'
         write(lun,25) "if [[ $3 == '0' || $3 == '2' ]]; then"
         write(lun,30) 'head -n 6 $CONDOR_INITIAL_DIR/'//
     &        '../madinMMC_$2.2 >& input_app.txt'
         write(lun,30) 'echo $i >> input_app.txt'
         write(lun,30) 'tail -n 4 $CONDOR_INITIAL_DIR/'//
     &        '../madinMMC_$2.2 >> input_app.txt'
         write(lun,25) "elif [[ $3 == '1' ]]; then"
         write(lun,30) 'head -n 6 madinM1 >& input_app.txt'
         write(lun,30) 'echo $i >> input_app.txt'
         write(lun,30) 'tail -n 4 madinM1 >> input_app.txt'
         write(lun,25) "fi"
         write(lun,25) 'T="$(date +%s)"'
         write(lun,25)
     &        'time ../madevent_mintMC > log.txt <input_app.txt 2>&1'
         write(lun,25) 'T="$(($(date +%s)-T))"'
         write(lun,25) 'echo "Time in seconds: ${T}" >>log.txt'
c endif
         write(lun,20) "fi"

         write(lun,20) "if [[ $1 == '2' && $runnumber != 0 ]]; then"
         write(lun,25) 'cp -f events.lhe $CONDOR_INITIAL_DIR'//
     &        '/G$2$i/events_$runnumber.lhe'
         write(lun,25) 'cp -f log.txt $CONDOR_INITIAL_DIR'//
     &        '/G$2$i/log_$runnumber.txt'
         write(lun,20) 'fi'

         write(lun,20) 'cd ../'
         write(lun,15) 'done'
c madevent_vegas or madevent_mint
         write(lun,15) "if [[ $1 == '0' || $1 == '1' ]]; then"
         write(lun,20) 'cp -ar $2\_G* $CONDOR_INITIAL_DIR/'
c madevent_mintMC
         write(lun,15) "elif [[ $1 == '2' && $runnumber == 0 ]]; then"
         write(lun,20) 'cp -ar G$2* $CONDOR_INITIAL_DIR/'
         write(lun,20) 
c endif
         write(lun,15) "fi"
         write(lun,15) 'rm -f $CONDOR_INITIAL_DIR/run.$script'
         write(lun,15) 'touch $CONDOR_INITIAL_DIR/done.$script'
      endif
      
 15   format(a)
 20   format(4x,a)
 25   format(8x,a)
 30   format(12x,a)
 35   format(16x,a)
 40   format(20x,a)
 45   format(24x,a)
      close(lun)
      end

