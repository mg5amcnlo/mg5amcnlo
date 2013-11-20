      subroutine joinPath(str1,str2,path)

      character*(*) str1
      character*(*) str2
      character*(*) path

      integer i,j,k

      i =1
      do while (i.le.LEN(str1) .and. str1(i:i).ne.' ')
      path(i:i) = str1(i:i)
      i=i+1
      enddo
      j=1
      do while (j.le.LEN(str2) .and. str2(j:j).ne.' ')
      path(i-1+j:i-1+j) = str2(j:j)
      j=j+1
      enddo

      k=i+j-1
      do while (k.le.LEN(path))
      path(k:k) = ' '
      k=k+1
      enddo

      return

      end

      subroutine setMadLoopPath(path)

      character(512) path

      character(512) prefix,fpath
      character(17) nameToCheck
      parameter (nameToCheck='MadLoopParams.dat')

      LOGICAL ML_INIT
      DATA ML_INIT/.TRUE./
      common/ML_INIT/ML_INIT

      character(512) MLPath
      data MLPath/'[[NA]]'/      
      common/MLPATH/MLPath

      integer i

C     Just a dummy call for LD to pick up this function
C     when creating the BLHA2 dynamic library
      CALL SETPARA2(' ')

      if (LEN(path).ge.4 .and. path(1:4).eq.'auto') then
          if (MLPath(1:6).eq.'[[NA]]') then
C     Try to automatically find the path
          prefix='./'
          call joinPath(prefix,nameToCheck,fpath)
          OPEN(1, FILE=fpath, ERR=1, STATUS='OLD',      
     $    ACTION='READ')
          MLPath=prefix
          goto 10
1         continue
          close(1)
          prefix='./MadLoop5_resources/'
          call joinPath(prefix,nameToCheck,fpath)
          OPEN(1, FILE=fpath, ERR=2, STATUS='OLD',      
     $    ACTION='READ')
          MLPath=prefix
          goto 10
2         continue
          close(1)
c     We could not automatically find the auxiliary files
          write(*,*) '==='
          write(*,*) 'ERROR: MadLoop5 could not automatically find',
     $    ' the file MadLoopParams.dat.'
          write(*,*) '==='
          write(*,*) '(Try using <CALL setMadLoopPath(/my/path)>',
     $    ' (before your first call to MadLoop) in order to',
     $    ' set the directory where this file is located as well as',
     $    ' other auxiliary files, such as <xxx>_ColorNumFactors.dat',
     $    ', <xxx>_ColorDenomFactors.dat, etc..)'
          stop
10        continue
          close(1)
          return
          endif
      else
C     Use the one specified by the user
C     Make sure there is a separator added
      i =1
      do while (i.le.LEN(path) .and. path(i:i).ne.' ')
      i=i+1
      enddo
      if (path(i-1:i-1).ne.'/') then
          path(i:i) = '/'
      endif
      MLpath=path          
      endif

C     Check that the FilePath set is correct
      call joinPath(MLPath,nameToCheck,fpath)
      OPEN(1, FILE=fpath, ERR=3, STATUS='OLD',      
     $ACTION='READ')
      goto 11
3     continue
      close(1)
      write(*,*) '==='
      write(*,*) 'ERROR: The MadLoop5 auxiliary files could not',
     $' be found in ',MLPath
      write(*,*) '==='
      stop
11    continue
      close(1)

      end
