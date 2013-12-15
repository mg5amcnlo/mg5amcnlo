      subroutine ntuple(x,a,b,ii,jconfig)
C--------------------------------------------------------------
c
c
c     This is a modified version for use with multi-pole integrations
c     it allows you to get the same set of random numbers several times
c     (1 for each configuration) jj tells it what configuration to use, 
c     so if you always put jj=1, you basically get the old version of 
c     ntuple out. It is currently configued to have maxconfig=25.
c     Modified by Tim Stelzer July 5 1995
c
c
c       Abstract:
c
c      ntuple - returns sequences of quasi-uniform random numbers
c               in the interval (a,b) using scrambled radical inverse
c               sequences. These numbers are designed for use in
c               Monte Carlo integration routines.
c
c  Author: Bill Long, UW-Madison Phenomenology Institute, 19-MAR-1991
c               Based on the algorithm for an earlier routine, htuple,
c               written by George Weller.
c
c               Original reference for this algorithm:
c
c         E. Braaten and G. Weller, J. Comp. Phys. 33,249-258 (1979)
c
c  References for radical inverse sequences (and Halton sequences)
c               are ck+^Z7ed in the above article.
c
c    Usage Notes:
c
c 1) Sequence numbers, i, range from 1 to 25 and correspond
c   to scrambled radical inverse sequences based on the first
c   25 primes (2..97).  In general, when performing a multi-dimensional
c   integral, a separate value of i should be used for the values
c   along each axis.
c
c 2) The basic algorithm generates values in the range (0,1). These
c    are rescaled to the range (a,b) in the final statement, so
c    it is not necessary to have a < b.  If a=b, the returned value, x,
c    will always be equal to a.
c
c 3) The sequences for different values of i are independent. They
c    cycle with different periods ranging from ~ 4M numbers 
c    through 147M numbers, covering numbers of precision 23-27 bits
c    appropriate for the mantissa of a single precision real value.
c
c 4) Ntuple differs from htuple primarily in that ntuple sequences
c    cycle with periods that are roughly equal for each value of i.
c    Htuple sequences cycled too quickly for small i, and too
c    slowly for large i. Ntuple is also written in a more modern
c    style, and uses considerably less memory, helping execution
c    speed on cache-sensitive machines.
c
c
c--Argument Declarations
c
      real*8  x    !  OUT - quasi-random value returned
                   !        x is in the range (a,b)
      real*8  a    !  IN  - Lower bound of interval for x
      real*8  b    !  IN  - Upper bound of interval for x
      integer ii   !  IN  - Sequence number, restricted to 1 <= i <= MaxDim
      integer jconfig!IN  - Pole number, restricted to1<=jconfig<= MaxConfigs
c
c     Constants
c
      include "genps.inc"
      include 'maxconfigs.inc'

      integer ndim,kdim,mdim,maxconfig,ktot
      parameter (ndim = maxdim, kdim = 181, mdim = 1060)
c      parameter (maxconfig=maxconfigs, ktot=kdim*maxconfig)
      parameter (maxconfig=lmaxconfigs, ktot=kdim*maxconfig)

c
c--Local Variable Declarations
c
      integer base_minus1(ndim),mix(mdim),k(kdim,maxconfig)
      double precision accum(kdim,maxconfig)
      double precision pbase(kdim)
      integer offset,koffset(ndim),mix_offset(ndim),maxj(ndim)
      integer jj
      logical first_time

c
c--Fixed Data Initializations
c
      data (pbase(i), i = 1, 146)
     .     /2d0,4d0,8d0,16d0,32d0,64d0,128d0,256d0,512d0,1024d0,
     .     2048d0,4096d0,8192d0,16384d0,32768d0,65536d0,131072d0,
     .     262144d0,524288d0,1048576d0,2097152d0,4194304d0,8388608d0,
     .     16777216d0,0d0,3d0,9d0,27d0,81d0,243d0,729d0,2187d0,6561d0,
     .     19683d0,59049d0,177147d0,531441d0,1594323d0,4782969d0,
     .     14348907d0,0d0,5d0,25d0,125d0,625d0,3125d0,15625d0,78125d0,
     .     390625d0,1953125d0,9765625d0,0d0,7d0,49d0,343d0,2401d0,
     .     16807d0,117649d0,823543d0,5764801d0,0d0,11d0,121d0,1331d0,
     .     14641d0,161051d0,1771561d0,19487171d0,0d0,13d0,169d0,2197d0,
     .     28561d0,371293d0,4826809d0,0d0,17d0,289d0,4913d0,83521d0,
     .     1419857d0,24137569d0,0d0,19d0,361d0,6859d0,130321d0,
     .     2476099d0,47045881d0,0d0,23d0,529d0,12167d0,279841d0,
     .     6436343d0,0d0,29d0,841d0,24389d0,707281d0,20511149d0,0d0,
     .     31d0,961d0,29791d0,923521d0,28629151d0,0d0,37d0,1369d0,
     .     50653d0,1874161d0,69343957d0,0d0,41d0,1681d0,68921d0,
     .     2825761d0,115856201d0,0d0,43d0,1849d0,79507d0,3418801d0,
     .     147008443d0,0d0,47d0,2209d0,103823d0,4879681d0,0d0,53d0,
     .     2809d0,148877d0,7890481d0,0d0,59d0,3481d0,205379d0,
     .     12117361d0,0d0,61d0,3721d0,226981d0,13845841d0,0d0/
      data (pbase(i), i=147, 181)
     .     /67d0,4489d0,300763d0,20151121d0,0d0,71d0,5041d0,357911d0,
     .     25411681d0,0d0,73d0,5329d0,389017d0,28398241d0,0d0,79d0,
     .     6241d0,493039d0,38950081d0,0d0,83d0,6889d0,571787d0,
     .     47458321d0,0d0,89d0,7921d0,704969d0,62742241d0,0d0,97d0,
     .     9409d0,912673d0,88529281d0, 0d0/

      data base_minus1/
     .      1,    2,    4,    6,   10,   12,   16,   18,   22,   28,
     .     30,   36,   40,   42,   46,   52,   58,   60,   66,   70,
     .     72,   78,   82,   88,   96/

      data maxj/ 24, 15, 10, 8,    7,    6,    6,    6,    5,    5,
     .     5,    5,    5,    5,    4,    4,    4,    4,    4,    4,
     .     4,    4,    4,    4,    4/

      data koffset/
     .     0,   25,   41,   52,   61,   69,   76,   83,   90,   96,
     .     102,  108,  114,  120,  126,  131,  136,  141,  146,  151,
     .     156,  161,  166,  171,  176/  

      data mix_offset/
     .     0,    2,    5,   10,   17,   28,   41,   58,   77,  100,
     .     129,  160,  197,  238,  281,  328,  381,  440,  501,  568,
     .     639,  712,  791,  874,  963/

      data (mix(i), i = 1, 412)
     .     /1,0, 1,2,0, 3,1,4,2,0, 4,2,6,1,5,3,0, 5,8,2,10,3,6,1,9,
     .     7,4,0, 6,10,2,8,4,12,1,9,5,11,3,7,0, 8,13,3,11,5,16,1,10,7,
     .     14,4,12,2,15,6,9,0, 9,14,3,17,6,11,1,15,7,12,4,18,8,2,16,10,
     .     5,13,0, 11,17,4,20,7,13,2,22,9,15,5,18,1,14,10,21,6,16,3,19,
     .     8,12,0, 15,7,24,11,20,2,27,9,18,4,22,13,26,5,16,10,23,1,19,
     .     28,6,14,17,3,25,12,8,21,0,  15,23,5,27,9,18,2,29,12,20,7,25,
     .     11,17,3,30,14,22,1,21,8,26,10,16,28,4,19,6,24,13,0, 18,28,6,
     .     23,11,34,3,25,14,31,8,20,36,1,16,27,10,22,13,32,4,29,17,7,
     .     35,19,2,26,12,30,9,24,15,33,5,21,0, 20,31,7,26,12,38,3,23,
     .     34,14,17,29,5,40,10,24,1,35,18,28,9,33,15,21,4,37,13,30,8,
     .     39,22,2,27,16,32,11,25,6,36,19,0, 21,32,7,38,13,25,3,35,17,
     .     28,10,41,5,23,30,15,37,1,19,33,11,26,42,8,18,29,4,39,14,22,
     .     34,6,24,12,40,2,31,20,27,9,36,16,0, 24,12,39,6,33,20,44,3,
     .     29,16,36,10,42,22,8,31,26,14,46,1,35,18,28,5,40,19,37,11,25,
     .     43,4,30,15,34,9,45,21,2,32,17,41,13,27,7,38,23,0,
     .     26,40,9,33,16,49,4,36,21,45,12,29,6,51,23,38,14,43,1,30,19,
     .     47,10,34,24,42,3,27,52,15,18,39,7,46,31,11,35,20,48,2,28,41,
     .     8,22,50,13,32,17,44,5,37,25,0, 29,44,10,52,18,34,4,48,23,38,
     .     13,57,7,32,41,20,54,2,26,46,15,36,24,50,8,40,16,56,5,30,43/
      data (mix(i), i = 413, 803)
     .     /21,51,11,33,1,58,27,37,14,47,19,28,45,6,53,12,35,22,42,3,
     .     55,25,31,9,49,17,39,0,30,46,10,38,18,56,4,42,24,52,14,33,21,
     .     59,6,40,27,49,2,35,16,54,12,44,26,50,8,32,58,19,1,41,29,48,
     .     13,36,22,60,7,45,23,53,9,34,17,55,3,39,28,47,15,37,20,57,5,
     .     43,25,51,11,31,0, 33,50,11,59,20,39,5,54,26,44,15,64,23,36,
     .     2,57,30,47,9,62,18,41,13,52,28,37,4,66,24,46,8,55,31,17,60,
     .     34,1,48,21,43,63,12,38,25,53,7,49,16,58,29,6,42,65,19,35,10,
     .     51,27,56,3,40,32,61,14,45,22,0, 35,53,12,62,21,41,5,67,28,
     .     46,16,56,25,8,50,38,65,2,32,59,19,44,14,70,30,48,7,39,58,22,
     .     10,63,33,26,52,1,55,18,43,68,13,36,47,4,61,24,40,29,66,9,51,
     .     17,57,23,37,3,69,31,45,15,60,11,49,34,20,64,6,54,27,42,0,
     .     36,55,12,46,22,67,5,41,61,18,30,52,8,70,27,43,15,59,33,2,64,
     .     38,24,50,10,72,20,48,31,57,4,63,25,40,14,54,35,68,7,45,17,
     .     60,28,1,66,39,21,51,11,71,32,47,13,56,26,44,3,65,34,19,58,9,
     .     49,37,69,16,29,53,6,62,23,42,0, 39,59,13,69,24,46,6,74,31,
     .     51,18,63,9,42,55,27,77,2,35,65,21,48,15,71,33,53,4,61,29,43,
     .     17,75,37,10,67,49,22,57,7,72,26,40,56,1,64,30,45,14,78,20,
     .     52,34,11,68,41,60,5,36,73,23,50,16,62,28,3,76,44,25,58,12,
     .     66,38,19,54,32,70,8,47,0,41,62,14,73,25,48,6,67,32,54,19,80/
      data (mix(i), i = 804, 1060)
     .     /10,44,58,29,76,2,37,64,22,51,16,71,35,56,8,82,27,46,12,69,
     .     39,60,4,50,24,78,31,65,17,42,74,1,53,21,61,34,11,79,43,28,
     .     68,7,55,38,75,15,47,20,70,5,57,33,81,26,49,9,63,36,66,18,45,
     .     3,77,30,59,23,52,13,72,40,0, 44,67,15,56,27,82,6,50,74,22,
     .     36,63,10,86,33,53,18,77,40,2,70,47,29,80,12,60,38,65,20,88,
     .     4,51,31,72,24,58,8,78,42,46,16,84,34,62,1,69,26,55,19,76,41,
     .     11,83,49,30,66,7,59,37,87,14,54,25,73,21,68,43,3,79,35,57,
     .     13,81,45,28,64,5,75,32,52,17,85,39,9,61,71,23,48,0, 48,73,
     .     16,61,29,89,7,55,81,34,22,69,41,94,3,52,77,19,38,85,12,64,
     .     44,26,91,58,9,71,32,79,14,50,66,24,96,1,46,83,36,59,18,75,
     .     30,87,5,54,42,68,21,92,10,62,39,80,27,56,6,86,47,72,15,35,
     .     93,43,65,2,76,25,53,84,17,37,67,11,90,49,31,74,20,60,95,4,
     .     45,63,28,82,13,57,40,78,8,88,33,51,23,70,0/
c
c--Variable Data Initializations
c
      data accum /ktot*0d0/

      data k /ktot*0d0/
      data first_time /.true./
c
c--Code:
c
      if (first_time) then
         write(*,*) 'Warning htuple modified for 1 configuration only'
         write(*,*) 'Using htuple configuration ',jconfig
         first_time=.false.
c     
c        to use multiple configurations need to use line
        jj = jconfig
c
      endif
      jj = jconfig
c      jj=1      !use jj=jconfig for multiconfiguraion mode
      if (jj .lt. 1 .or. jj .gt. maxconfig) then
         print*,'Error in ntuple.  Invalid pole choice',jj
         stop
      endif
      i=ii
      i = ii+jj                   !This keeps us from generating same ran #'s
      do while (i .gt. ndim)
         i=i-ndim
      enddo
c      if (i .gt. ndim) i=i-ndim   !For different configurations
      j = 1
      offset = koffset(i)
      do while (k(offset+j,jj) .eq. base_minus1(i))
         k(offset+j,jj)=0
         j=j+1
         if (j.gt.maxj(i)) then
            do j=1,maxj(i)
               k(offset+j,jj)=0
               accum(offset+j,jj)=0d0
            end do
            j=1
         end if
      end do

      k(offset+j,jj)=k(offset+j,jj)+1
      accum(offset+j,jj) = accum(offset+j+1,jj) +
     .     mix(mix_offset(i)+k(offset+j,jj))/pbase(offset+j)
      do  jjj=2,j-1
         accum(offset+jjj,jj) = accum(offset+j,jj)
      end do
      x = a + (b-a) * accum(offset+j,jj)
c      write(*,'(2i6,1f15.8)') jj,i,x
      end

