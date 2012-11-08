      subroutine ntuple(x,a,b,ii,jconfig)
c-------------------------------------------------------
c     Front to ranmar which allows user to easily
c     choose the seed.
c------------------------------------------------------
      implicit none
c
c     Arguments
c
      double precision x,a,b
      integer ii,jconfig
c
c     Local
c
      integer init, ioffset, joffset
      integer     ij, kl, iseed1,iseed2

c
c     Global
c
c-------
c     18/6/2012 tjs promoted to integer*8 to avoid overflow for iseed > 60K
c------
      integer*8       iseed
      common /to_seed/iseed
c
c     Data
c
      data init /1/
      save ij, kl
c-----
c  Begin Code
c-----
      if (init .eq. 1) then
         init = 0
         call get_offset(ioffset)
         if (iseed .eq. 0) call get_base(iseed)
c
c     TJS 3/13/2008
c     Modified to allow for more sequences 
c     iseed can be between 0 and 30081*30081
c     before pattern repeats
c
c
c     TJS 12/3/2010
c     multipied iseed to give larger values more likely to make change
c     get offset for multiple runs of single process
c
c
c     TJS 18/6/2012
c     Updated to better divide iseed among ij and kl seeds
c     Note it may still be possible to get identical ij,kl for
c     different iseed, if have exactly compensating joffset, ioffset, jconfig
c
         call get_moffset(joffset)
         joffset = joffset * 3157
         iseed = iseed * 31300       
         ij=1802+jconfig + mod(iseed,30081)
         kl=9373+(iseed/30081)+ioffset + joffset     !Switched to 30081  20/6/12 to avoid dupes in range 30082-31328
         write(*,'(a,i6,a3,i6)') 'Using random seed offsets',jconfig," : ",ioffset
         write(*,*) ' with seed', iseed/31300
         do while (ij .gt. 31328)
            ij = ij - 31328
         enddo
         do while (kl .gt. 30081)
            kl = kl - 30081
         enddo
        call rmarin(ij,kl)         
      endif
      call ranmar(x)
      do while (x .lt. 1d-16)
         call ranmar(x)
      enddo
      x = a+x*(b-a)
      end

      subroutine get_base(iseed)
c-------------------------------------------------------
c     Looks for file iproc.dat to offset random number gen
c------------------------------------------------------
      implicit none
c
c     Constants
c
      integer    lun
      parameter (lun=22)
c
c     Arguments
c
      integer*8 iseed
c
c     Local
c
      character*60 fname
      logical done
      integer i,level
c-----
c  Begin Code
c-----

      fname = 'randinit'
      done = .false.
      level = 1
      do while(.not. done .and. level .lt. 5)
         open(unit=lun,file=fname,status='old',err=15)
         done = .true.
 15      level = level+1
         fname = '../' // fname
         i=index(fname,' ')
         if (i .gt. 0) fname=fname(1:i-1)
      enddo
      if (done) then
         read(lun,'(a)',end=24,err=24) fname
         i = index(fname,'=')
         if (i .gt. 0) fname=fname(i+1:)
         read(fname,*,err=26,end=26) iseed
 24      close(lun)
c         write(*,*) 'Read iseed from randinit ',iseed
         return
 26      close(lun)
      endif
 25   iseed = 0
c      write(*,*) 'No base found using iseed=0'
      end

      subroutine get_offset(iseed)
c-------------------------------------------------------
c     Looks for file iproc.dat to offset random number gen
c------------------------------------------------------
      implicit none
c
c     Constants
c
      integer    lun
      parameter (lun=22)
c
c     Arguments
c
      integer iseed
c
c     Local
c
c-----
c  Begin Code
c-----

      open(unit=lun,file='./iproc.dat',status='old',err=15)
         read(lun,*,err=14) iseed
         close(lun)
         return
 14   close(lun)
 15   open(unit=lun,file='../iproc.dat',status='old',err=25)
         read(lun,*,err=24) iseed
         close(lun)
         return
 24   close(lun)
 25   iseed = 0
      end

      subroutine get_moffset(iseed)
c-------------------------------------------------------
c     Looks for file moffset.dat to offset random number gen
c------------------------------------------------------
      implicit none
c
c     Constants
c
      integer    lun
      parameter (lun=22)
c
c     Arguments
c
      integer iseed
c
c     Local
c
c-----
c  Begin Code
c-----

      open(unit=lun,file='./moffset.dat',status='old',err=25)
         read(lun,*,err=14) iseed
         write(*,*) "Got moffset",iseed
         close(lun)
         return
 14   close(lun)
 25   iseed = 0
      end

      subroutine ranmar(rvec)
*     -----------------
* universal random number generator proposed by marsaglia and zaman
* in report fsu-scri-87-50
* in this version rvec is a double precision variable.
      implicit real*8(a-h,o-z)
      common/ raset1 / ranu(97),ranc,rancd,rancm
      common/ raset2 / iranmr,jranmr
      save /raset1/,/raset2/
      uni = ranu(iranmr) - ranu(jranmr)
      if(uni .lt. 0d0) uni = uni + 1d0
      ranu(iranmr) = uni
      iranmr = iranmr - 1
      jranmr = jranmr - 1
      if(iranmr .eq. 0) iranmr = 97
      if(jranmr .eq. 0) jranmr = 97
      ranc = ranc - rancd
      if(ranc .lt. 0d0) ranc = ranc + rancm
      uni = uni - ranc
      if(uni .lt. 0d0) uni = uni + 1d0
      rvec = uni
      end
 
      subroutine rmarin(ij,kl)
*     -----------------
* initializing routine for ranmar, must be called before generating
* any pseudorandom numbers with ranmar. the input values should be in
* the ranges 0<=ij<=31328 ; 0<=kl<=30081
      implicit real*8(a-h,o-z)
      character*30 filename
      logical file_exists
      common/ raset1 / ranu(97),ranc,rancd,rancm
      common/ raset2 / iranmr,jranmr
      save /raset1/,/raset2/
* this shows correspondence between the simplified input seeds ij, kl
* and the original marsaglia-zaman seeds i,j,k,l.
* to get the standard values in the marsaglia-zaman paper (i=12,j=34
* k=56,l=78) put ij=1802, kl=9373
      write(*,*) "Ranmar initialization seeds",ij,kl
c
c    18/6/2012 TJS  Added check to ensure ij and kl are in range
c      
      if (ij .lt. 0 .or. ij .gt. 31328 .or.
     $     kl .lt. 0 .or. kl .gt. 30081) then
         filename='../../error'
         INQUIRE(FILE="../../RunWeb", EXIST=file_exists)
         if(.not.file_exists) filename = '../' // filename
         open(unit=26,file=filename,status='unknown')
         if (ij .lt. 0 .or. ij .gt. 31328) then
            write(26,*) 'Bad initialization value of ij in rmarin ', ij
            write(*,*) 'Bad initialization value of ij in rmarin ', ij
         elseif (kl .lt. 0 .or. kl .gt. 30081) then
            write(26,*) 'Bad initialization value of kl in rmarin ', kl
            write(*,*) 'Bad initialization value of kl in rmarin ', kl
         endif
         stop
      endif

      i = mod( ij/177 , 177 ) + 2
      j = mod( ij     , 177 ) + 2
      k = mod( kl/169 , 178 ) + 1
      l = mod( kl     , 169 )
      do 300 ii = 1 , 97
        s =  0d0
        t = .5d0
        do 200 jj = 1 , 24
          m = mod( mod(i*j,179)*k , 179 )
          i = j
          j = k
          k = m
          l = mod( 53*l+1 , 169 )
          if(mod(l*m,64) .ge. 32) s = s + t
          t = .5d0*t
  200   continue
        ranu(ii) = s
  300 continue
      ranc  =   362436d0 / 16777216d0
      rancd =  7654321d0 / 16777216d0
      rancm = 16777213d0 / 16777216d0
      iranmr = 97
      jranmr = 33
      end
