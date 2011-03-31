cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                                                                      c
c  READ_SLHA_RIP(nin2,unimass,lowmass,width,                           c
c                bw,uu,vv,m_t,m_b,m_l,param_name,mz,mw)                c
c                                                                      c
c                      Written by Tilman Plehn                         c
c                      SM part by Johan Alwall                         c
c                                                                      c
c        now explicit stop, sbottom, stau mixing included              c
c                                                                      c
c   output  unimass(10)   listed below                                 c 
c           lowmass(0:99) listed below                                 c 
c           width(0:99)   same order of particles as in lowmass        c
c           bw            neutralino mixing matrix bino-wino basis     c
c           uu            chargino mixing matrix                       c
c           vv            chargino mixing matrix                       c  
c           m_t           stop mixing matrix                           c  
c           m_b           sbottom mixing matrix                        c  
c           m_l           stau mixing matrix                           c
c           param_name    name of teh param_card                       c  
c           mz,mw         ripped from ino matrices                     c  
c                                                                      c
c           note the different sign in the definition of A             c
c                                                                      c
c                       unimass(1)  universal scalar mass m_0(1,2)     c
c                       unimass(2)  additional mass term for m_0(1,2)  c
c                       unimass(3)  additional mass term for m_0(3)    c
c                       unimass(4)  additional mass term for m_0(Higgs)c
c                       unimass(5)  universal fermion mass m_1/2       c
c                       unimass(6)  additional mass term for m1        c
c                       unimass(7)  additional mass term for m2        c
c                       unimass(8)  additional mass term for m3        c
c                       unimass(9)  universal mixing parameter a_0     c
c                       unimass(10) tan(beta)                          c
c                       unimass(11) sign of higgs mass parameter mu    c
c                       .............                                  c
c                       unimass(20)                                    c
c                                                                      c
c                       lowmass(0)  mu                                 c
c                       lowmass(1)  m_1                                c
c                       lowmass(2)  m_2                                c
c                       lowmass(3)  m_3                                c
c                                                                      c
c                       lowmass(4)  gluino mass                        c
c                       lowmass(5)  \                                  c
c                       lowmass(6)   \                                 c
c                       lowmass(7)   /  neutralino masses [with sign]  c
c                       lowmass(8)  /                                  c
c                       lowmass(9)  \                                  c
c                       lowmass(10) /   chargino masses                c
c                                                                      c
c                       lowmass(11) sdown_l mass                       c
c                       lowmass(12) sdown_r mass                       c
c                       lowmass(13) sup_l mass                         c
c                       lowmass(14) sup_r mass                         c
c                       lowmass(46) sstrange_l mass                    c
c                       lowmass(47) sstrange_r mass                    c
c                       lowmass(48) scharm_l mass                      c
c                       lowmass(49) scharm_r mass                      c
c                       lowmass(15) degenerate squark mass (8)         c
c                       lowmass(16) degenerate squark mass (10)        c
c                       lowmass(17) sbottom-1 mass                     c
c                       lowmass(18) sbottom-2 mass                     c
c                       lowmass(19) stop-1 mass                        c
c                       lowmass(20) stop-2 mass                        c
c                                                                      c
c                       lowmass(21) a_b                                c
c                       lowmass(22) sin(2 theta_b)                     c
c                       lowmass(23) cos(2 theta_b)                     c
c                       lowmass(24) a_t                                c
c                       lowmass(25) sin(2 theta_t)                     c
c                       lowmass(26) cos(2 theta_t)                     c
c                                                                      c
c                       lowmass(30) selectron_l mass                   c
c                       lowmass(31) selectron_r mass                   c
c                       lowmass(32) selectron-neutrino mass            c
c                       lowmass(50) smu_l mass                         c
c                       lowmass(51) smu_r mass                         c
c                       lowmass(52) smu-neutrino mass                  c
c                       lowmass(33) stau_1 mass                        c
c                       lowmass(34) stau_2 mass                        c
c                       lowmass(35) stau-neutrino mass                 c
c                       lowmass(36) a_tau                              c
c                       lowmass(37) sin(2 theta_tau)                   c
c                       lowmass(38) cos(2 theta_tau)                   c
c                                                                      c
c                       lowmass(40) cp odd higgs mass                  c
c                       lowmass(41) light cp even higgs mass           c
c                       lowmass(42) heavy cp even higgs mass           c
c                       lowmass(43) charged higgs mass                 c
c                       lowmass(44) sin(alpha)                         c
c                       lowmass(45) cos(alpha)                         c
c                                                                      c
c                       lowmass(46) sstrange_l mass                    c
c                       lowmass(47) sstrange_r mass                    c
c                       lowmass(48) scharm_l mass                      c
c                       lowmass(49) scharm_r mass                      c
c                       lowmass(50) smu_l mass                         c
c                       lowmass(51) smu_r mass                         c
c                       lowmass(52) smu-neutrino mass                  c
c                                                                      c
c                       lowmass(80) unification scale                  c
c                       lowmass(81) unified coupling alpha(m_x)        c
c                                                                      c
c                       lowmass(91) trilinear higgs coupling lambda(1) c
c                       .......                                        c
c                       lowmass(97) lambda(7)                          c
c                                                                      c
c                       lowmass(99)                                    c
c                                                                      c
c                       bwmix neutralino mixing matrix (bino-wino)     c
c                       pzmix neutralino mixing matrix (photino-zino)  c
c                       uumix chargino mixing matrix u                 c
c                       vvmix chargino mixing matrix v                 c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      subroutine READ_SLHA_RIP(unimass,lowmass,width,
     &                         bw,uu,vv,m_t,m_b,m_l,param_name)!,mz,mw)

      implicit none

      character*(*) param_name
      integer   nin2,ii,j1,j2,check(1:20),check_final,check_sq(1:13)
      real*8    unimass(20),lowmass(0:99),bw(4,4),uu(2,2),vv(2,2)
      real*8    m_t(2,2),m_b(2,2),m_l(2,2)
      real*8    width(0:99),width_sq(1:10)
c      real*8    mw,mz,sw,cw,sb,cb
      character line1*6, line2*100
      logical   done
      logical fopened

c
c Zero parameter arrays
c
      unimass(10)=0d0
      do ii=0,99
         lowmass(ii)=0d0
      enddo

c
c     Read SUSY LHA file
c
      nin2=23
      call open_file_mdl(nin2,param_name,fopened)
      if(.not.fopened)
     &     stop 'Error: could not open file param_card.dat'

c                  make sure to start from beginning of file
      rewind(nin2)

c                  initialize the check arrays
      do j1=1,20,1
         check(j1) = 0
      end do

      do j1=1,13,1
         check_sq(j1) = 0 
      end do

c                  loop huge, has to exit
      do ii=1,10000,1

c                  check if routine can be left
ctp         check_final = 1
ctp         do j1=1,13
ctp            check_final = check_final * check(j1) 
ctp         end do
ctp         if (check_final.eq.1) then 
ctp            return
ctp         end if 
         
c                  read new line into. goto 190 at end-of-file.
         line1 = ' '
 10      read(nin2,'(a6,a100)',end=4400) line1,line2

         if(line1//line2.eq.'') goto 10
c                  rewrite line1(1:6) and line2(1:10) to all upper case  
         do j1=1,6
            if (line1(j1:j1).ne.'#') then 
               do j2=97,122
                  if (line1(j1:j1).eq.CHAR(j2)) line1(j1:j1)=CHAR(j2-32)
               end do
            end if
         end do

         do j1=1,20
            if (line2(j1:j1).ne.'#') then 
               do j2=97,122
                  if (line2(j1:j1).eq.CHAR(j2)) line2(j1:j1)=CHAR(j2-32)
               end do
            end if
         end do

c                  look for blocks and pick them one after the other
         if (line1(1:1).eq.'B') then

c                  look for block MINPAR
            if (line2(1:6).eq.'MINPAR') then 
c               call READ_BLOCK_MINPAR(nin2,unimass,done)
c               if (done) then 
c                  check(1) = 1
               cycle
c               else 
c                  print*, " READ_LES_HOUCHES: problem in MINPAR "
c                  call HARD_STOP
c               end if 

c                  look for block MASS
            else if (line2(1:4).eq.'MASS') then 
               call READ_BLOCK_MASS(nin2,lowmass,done)
               if (done) then 
                  check(2) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in MASS "
                  call HARD_STOP
               end if 

c                  look for block STOPMIX
            else if (line2(1:7).eq.'STOPMIX') then
               call READ_BLOCK_STOPMIX(nin2,m_t,done)
               if (done) then 
                  check(3) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in STOPMIX "
                  call HARD_STOP
               end if 

c                  look for block SBOTMIX
            else if (line2(1:7).eq.'SBOTMIX') then
               call READ_BLOCK_SBOTMIX(nin2,m_b,done)
               if (done) then 
                  check(4) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in SBOTMIX "
                  call HARD_STOP
               end if 

c                  look for block STAUMIX
            else if (line2(1:7).eq.'STAUMIX') then
               call READ_BLOCK_STAUMIX(nin2,m_l,done)
               if (done) then 
                  check(5) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in STAUMIX "
                  call HARD_STOP
               end if 

c                  look for block NMIX
            else if (line2(1:4).eq.'NMIX') then
               call READ_BLOCK_NMIX(nin2,bw,done)
               if (done) then 
                  check(6) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in NMIX "
                  call HARD_STOP
               end if 

c                  look for block UMIX
            else if (line2(1:4).eq.'UMIX') then
               call READ_BLOCK_UMIX(nin2,uu,done)
               if (done) then 
                  check(7) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in UMIX "
                  call HARD_STOP
               end if 

c                  look for block VMIX
            else if (line2(1:4).eq.'VMIX') then
               call READ_BLOCK_VMIX(nin2,vv,done)
               if (done) then 
                  check(8) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in VMIX "
                  call HARD_STOP
               end if 

c                  look for block ALPHA
            else if (line2(1:5).eq.'ALPHA') then
               call READ_BLOCK_ALPHA(nin2,lowmass,done)
               if (done) then 
                  check(9) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in ALPHA "
                  call HARD_STOP
               end if 

c                  look for block HMIX
            else if (line2(1:4).eq.'HMIX') then
               call READ_BLOCK_HMIX(nin2,lowmass,unimass,done)
               if (done) then 
                  check(10) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in HMIX "
                  call HARD_STOP
               end if 

c                  look for block AU
            else if (line2(1:2).eq.'AU') then
               call READ_BLOCK_AU(nin2,lowmass,done)
               if (done) then 
                  check(11) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in AU "
                  call HARD_STOP
               end if 

c                  look for block AD
            else if (line2(1:2).eq.'AD') then
               call READ_BLOCK_AD(nin2,lowmass,done)
               if (done) then 
                  check(12) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in AD "
                  call HARD_STOP
               end if 

c                  look for block AE
            else if (line2(1:2).eq.'AE') then
               call READ_BLOCK_AE(nin2,lowmass,done)
               if (done) then 
                  check(13) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in AE "
                  call HARD_STOP
               end if 
c                  look for block SMINPUTS
            else if (line2(1:8).eq.'SMINPUTS') then ! JA
               call READ_BLOCK_SMINPUTS(nin2,done)
               if (done) then 
                  check(14) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in SMINPUTS "
                  call HARD_STOP
               end if 

c                  look for block MSOFT (not crucial, though)
c            else if (line2(1:5).eq.'MSOFT') then
c               call READ_BLOCK_MSOFT(nin2,lowmass,done)
c               if (done) then 
c                  check(16) = 1
c                  cycle
c               else 
c                  print*, " READ_LES_HOUCHES: problem in MSOFT "
c                  call HARD_STOP
c               end if 

c                  look for block YU
            else if (line2(1:2).eq.'YU') then ! JA
               call READ_BLOCK_YU(nin2,lowmass,done)
               if (done) then 
                  check(17) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in YU "
                  call HARD_STOP
               end if 

c                  look for block YD
            else if (line2(1:2).eq.'YD') then ! JA
               call READ_BLOCK_YD(nin2,lowmass,done)
               if (done) then 
                  check(18) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in YD "
                  call HARD_STOP
               end if 

c                  look for block YE
            else if (line2(1:2).eq.'YE') then ! JA
               call READ_BLOCK_YE(nin2,lowmass,done)
               if (done) then 
                  check(19) = 1
                  cycle
               else 
                  print*, " READ_LES_HOUCHES: problem in YE "
                  call HARD_STOP
               end if 

c                  continue if block not interesting
            else 
               cycle 
            end if 

c                  look for decay lines and fill them one by one
         else if (line1(1:1).eq.'D') then
            call READ_DECAY(nin2,width,check_sq,width_sq)
            
            check_final = 1
            do j1=1,10
               check_final = check_final * check_sq(j1) 
            end do
            
            if (check_final.eq.1) then 
               width(15) = 0.D0
               do j1=1,8
                  width(15) = width(15) + width_sq(j1)
               end do
               width(15) = width(15)/8.D0
               
               width(16) = 0.D0
               do j1=1,10
                  width(16) = width(16) + width_sq(j1)
               end do
               width(16) = width(16)/10.D0
            
            end if 
            cycle

c                  continue if not a block statement
         else 
            cycle
         end if 

      end do
c            maximum number of lines exhausted
         print*, " READ_LES_HOUCHES: end of file not reached ",ii
         call HARD_STOP


 4400 continue

      close(nin2)
      
ctp      do j1=0,99
ctp         print*, "M,Gamma",j1,lowmass(j1),width(j1)
ctp      end do


c       extract the effective weak parameters from the mass matrices
c      if ( ( check(2)*check(6)*check(7)*check(8) ).eq.1) 
c     &    call EXTRACT_SLHA_WEAK(lowmass,bw,uu,vv,mw,mz,sw,cw,sb,cb)
                          
      do j1=11,13
        check_final = check_final * check_sq(j1) 
      end do
      if (check_final.eq.0) then 
        write(*,*) " READ_LES_HOUCHES: problem in DECAYS "
        write(*,*) " Standard Model decays not found (6,23,24)"
        call hard_stop
      endif
      
      check_final = 1
      do j1=2,14
         check_final = check_final * check(j1) 
      end do
      do j1=17,19
         check_final = check_final * check(j1) 
      end do
      if (check_final.eq.1) then 
        print*, " READ_LES_HOUCHES: done, all inputs read "
        return
      else 
        print*, " READ_LES_HOUCHES: input incomplete "
        do j1 = 1,15
          print*, check(j1)
        end do
        call HARD_STOP
      end if 

      return
      end

c----------------------
      subroutine READ_BLOCK_MINPAR(nin2,unimass,done)

      implicit none

      integer   nin2,ii,dumi
      real*8    dumr,unimass(20)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4401) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi,dumr

c            array defined in Xsugra.f
            if (dumi.eq.1) then 
               unimass(1) = dumr
            else if (dumi.eq.2) then 
               unimass(5) = dumr
            else if (dumi.eq.3) then 
               unimass(10) = dumr  ! FOR THE COMPARISON
c               write(6,*)'unimass(10) = ',unimass(10)
            else if (dumi.eq.4) then 
               unimass(11) = dumr
            else if (dumi.eq.5) then 
               unimass(9) = dumr
            end if 

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then
            goto 4401
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_MINPAR: block not terminated? "
      call HARD_STOP

 4401 backspace nin2
      done = .true.
      return
      
      end 

c----------------------
      subroutine READ_BLOCK_MASS(nin2,lowmass,done)

      implicit none

      integer   nin2,ii,dumi,j1
      real*8    dumr,lowmass(0:99),squark(1:10)
      character line1*1
      logical done

      include 'coupl.inc' ! JA

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4402) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi,dumr

c            array defined in Xsugra.f
            if (dumi.eq.1000021) then 
               lowmass(4) = dumr
            else if (dumi.eq.1000022) then 
               lowmass(5) = dumr
            else if (dumi.eq.1000023) then 
               lowmass(6) = dumr
            else if (dumi.eq.1000025) then 
               lowmass(7) = dumr
            else if (dumi.eq.1000035) then 
               lowmass(8) = dumr
            else if (dumi.eq.1000024) then 
               lowmass(9) = dumr
            else if (dumi.eq.1000037) then 
               lowmass(10) = dumr
            else if (dumi.eq.1000001) then 
               lowmass(11) = dumr
               squark(1)   = dumr
            else if (dumi.eq.2000001) then 
               lowmass(12) = dumr
               squark(2)   = dumr
            else if (dumi.eq.1000002) then 
               lowmass(13) = dumr
               squark(3)   = dumr
            else if (dumi.eq.2000002) then 
               lowmass(14) = dumr
               squark(4)   = dumr
            else if (dumi.eq.1000003) then 
               lowmass(46) = dumr
               squark(5)   = dumr
            else if (dumi.eq.2000003) then 
               lowmass(47) = dumr
               squark(6)   = dumr
            else if (dumi.eq.1000004) then 
               lowmass(48) = dumr
               squark(7)   = dumr
            else if (dumi.eq.2000004) then 
               lowmass(49) = dumr
               squark(8)   = dumr
            else if (dumi.eq.1000005) then 
               lowmass(17) = dumr
               squark(9)   = dumr
            else if (dumi.eq.2000005) then 
               lowmass(18) = dumr
               squark(10)  = dumr
            else if (dumi.eq.1000006) then 
               lowmass(19) = dumr
            else if (dumi.eq.2000006) then 
               lowmass(20) = dumr
            else if (dumi.eq.1000011) then 
               lowmass(30) = dumr
            else if (dumi.eq.2000011) then 
               lowmass(31) = dumr
            else if (dumi.eq.1000012) then 
               lowmass(32) = dumr
            else if (dumi.eq.1000013) then 
               lowmass(50) = dumr
            else if (dumi.eq.2000013) then 
               lowmass(51) = dumr
            else if (dumi.eq.1000014) then 
               lowmass(52) = dumr
            else if (dumi.eq.1000015) then 
               lowmass(33) = dumr
            else if (dumi.eq.2000015) then 
               lowmass(34) = dumr
            else if (dumi.eq.1000016) then 
               lowmass(35) = dumr
            else if (dumi.eq.36) then 
               lowmass(40) = dumr
            else if (dumi.eq.25) then 
               lowmass(41) = dumr
            else if (dumi.eq.35) then 
               lowmass(42) = dumr
            else if (dumi.eq.37) then 
               lowmass(43) = dumr
c            else if (dumi.eq.4) then ! JA
c               cmass   = dumr
            else if (dumi.eq.5) then 
               bmass   = dumr
            else if (dumi.eq.24) then 
               wmass   = dumr
            end if 

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then

c            degenerate squark masses
            lowmass(15) = 0.D0
            do j1=1,8
               lowmass(15) = lowmass(15) + squark(j1)
            end do
            lowmass(15) = lowmass(15)/8.D0

            lowmass(16) = 0.D0
            do j1=1,10
               lowmass(16) = lowmass(16) + squark(j1)
            end do
            lowmass(16) = lowmass(16)/10.D0
            
            goto 4402
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_MASS: block mass not terminated? "
      call HARD_STOP

 4402 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_STOPMIX(nin2,m_t,done)

      implicit none

      integer   nin2,ii,dumi1,dumi2
      real*8    dumr,m_t(2,2)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4403) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi1,dumi2,dumr

            if ( (dumi1.eq.1).and.(dumi2.eq.1) ) then 
               m_t(1,1) = dumr
            else if ( (dumi1.eq.1).and.(dumi2.eq.2) ) then 
               m_t(1,2) = dumr
            else if ( (dumi1.eq.2).and.(dumi2.eq.1) ) then 
               m_t(2,1) = dumr
            else if ( (dumi1.eq.2).and.(dumi2.eq.2) ) then 
               m_t(2,2) = dumr
            end if 

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then

            if ( (m_t(1,1).eq.+m_t(2,2)) .and.
     &           (m_t(1,2).eq.-m_t(2,1))      ) then 
c               print*, " READ_BLOCK_STOPMIX: rotation "
            else if ( (m_t(1,1).eq.-m_t(2,2)) .and.
     &                (m_t(1,2).eq.+m_t(2,1))      ) then 
c               print*, " READ_BLOCK_STOPMIX: rotation+flip "
            else 
               print*, " READ_BLOCK_STOPMIX: problem with matrix "
               call HARD_STOP
            end if 

            goto 4403
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_STOPMIX: block mass not terminated? "
      call HARD_STOP

 4403 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_SBOTMIX(nin2,m_b,done)

      implicit none

      integer   nin2,ii,dumi1,dumi2
      real*8    dumr,m_b(2,2)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4404) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi1,dumi2,dumr

            if ( (dumi1.eq.1).and.(dumi2.eq.1) ) then 
               m_b(1,1) = dumr
            else if ( (dumi1.eq.1).and.(dumi2.eq.2) ) then 
               m_b(1,2) = dumr
            else if ( (dumi1.eq.2).and.(dumi2.eq.1) ) then 
               m_b(2,1) = dumr
            else if ( (dumi1.eq.2).and.(dumi2.eq.2) ) then 
               m_b(2,2) = dumr
            end if 

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then

            if ( (m_b(1,1).eq.+m_b(2,2)) .and.
     &           (m_b(1,2).eq.-m_b(2,1))      ) then 
c               print*, " READ_BLOCK_SBOTMIX: rotation "
            else if ( (m_b(1,1).eq.-m_b(2,2)) .and.
     &                (m_b(1,2).eq.+m_b(2,1))      ) then 
c               print*, " READ_BLOCK_SBOTMIX: rotation+flip "
            else 
               print*, " READ_BLOCK_SBOTMIX: problem with matrix "
               call HARD_STOP
            end if 

            goto 4404 
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_SBOTMIX: block mass not terminated? "
      call HARD_STOP

 4404 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_STAUMIX(nin2,m_l,done)

      implicit none

      integer   nin2,ii,dumi1,dumi2
      real*8    dumr,m_l(2,2)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4405) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi1,dumi2,dumr

            if ( (dumi1.eq.1).and.(dumi2.eq.1) ) then 
               m_l(1,1) = dumr
            else if ( (dumi1.eq.1).and.(dumi2.eq.2) ) then 
               m_l(1,2) = dumr
            else if ( (dumi1.eq.2).and.(dumi2.eq.1) ) then 
               m_l(2,1) = dumr
            else if ( (dumi1.eq.2).and.(dumi2.eq.2) ) then 
               m_l(2,2) = dumr
            end if 

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then

            if ( (m_l(1,1).eq.+m_l(2,2)) .and.
     &           (m_l(1,2).eq.-m_l(2,1))      ) then 
c               print*, " READ_BLOCK_STAUMIX: rotation "
            else if ( (m_l(1,1).eq.-m_l(2,2)) .and.
     &                (m_l(1,2).eq.+m_l(2,1))      ) then 
c               print*, " READ_BLOCK_STAUMIX: rotation+flip "
            else 
               print*, " READ_BLOCK_STAUMIX: problem with matrix "
               call HARD_STOP
            end if 

            goto 4405
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_STAUMIX: block mass not terminated? "
      call HARD_STOP

 4405 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_NMIX(nin2,bw,done)

      implicit none

      integer   nin2,ii,dumi1,dumi2
      real*8    dumr,bw(4,4)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4406) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi1,dumi2,dumr

            if ( (dumi1.eq.1).and.(dumi2.eq.1) ) then 
               bw(1,1) = dumr
            else if ( (dumi1.eq.1).and.(dumi2.eq.2) ) then 
               bw(1,2) = dumr
            else if ( (dumi1.eq.1).and.(dumi2.eq.3) ) then 
               bw(1,3) = dumr
            else if ( (dumi1.eq.1).and.(dumi2.eq.4) ) then 
               bw(1,4) = dumr
            else if ( (dumi1.eq.2).and.(dumi2.eq.1) ) then 
               bw(2,1) = dumr
            else if ( (dumi1.eq.2).and.(dumi2.eq.2) ) then 
               bw(2,2) = dumr
            else if ( (dumi1.eq.2).and.(dumi2.eq.3) ) then 
               bw(2,3) = dumr
            else if ( (dumi1.eq.2).and.(dumi2.eq.4) ) then 
               bw(2,4) = dumr
            else if ( (dumi1.eq.3).and.(dumi2.eq.1) ) then 
               bw(3,1) = dumr
            else if ( (dumi1.eq.3).and.(dumi2.eq.2) ) then 
               bw(3,2) = dumr
            else if ( (dumi1.eq.3).and.(dumi2.eq.3) ) then 
               bw(3,3) = dumr
            else if ( (dumi1.eq.3).and.(dumi2.eq.4) ) then 
               bw(3,4) = dumr
            else if ( (dumi1.eq.4).and.(dumi2.eq.1) ) then 
               bw(4,1) = dumr
            else if ( (dumi1.eq.4).and.(dumi2.eq.2) ) then 
               bw(4,2) = dumr
            else if ( (dumi1.eq.4).and.(dumi2.eq.3) ) then 
               bw(4,3) = dumr
            else if ( (dumi1.eq.4).and.(dumi2.eq.4) ) then 
               bw(4,4) = dumr
            end if 

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then
            goto 4406
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_NMIX: block mass not terminated? "
      call HARD_STOP

 4406 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_UMIX(nin2,uu,done)

      implicit none

      integer   nin2,ii,dumi1,dumi2
      real*8    dumr,uu(2,2)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4407) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi1,dumi2,dumr

            if ( (dumi1.eq.1).and.(dumi2.eq.1) ) then 
               uu(1,1) = dumr
            else if ( (dumi1.eq.1).and.(dumi2.eq.2) ) then 
               uu(1,2) = dumr
            else if ( (dumi1.eq.2).and.(dumi2.eq.1) ) then 
               uu(2,1) = dumr
            else if ( (dumi1.eq.2).and.(dumi2.eq.2) ) then 
               uu(2,2) = dumr
            end if 

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then
            goto 4407
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_UMIX: block mass not terminated? "
      call HARD_STOP

 4407 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_VMIX(nin2,vv,done)

      implicit none

      integer   nin2,ii,dumi1,dumi2
      real*8    dumr,vv(2,2)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4408) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi1,dumi2,dumr

            if ( (dumi1.eq.1).and.(dumi2.eq.1) ) then 
               vv(1,1) = dumr
            else if ( (dumi1.eq.1).and.(dumi2.eq.2) ) then 
               vv(1,2) = dumr
            else if ( (dumi1.eq.2).and.(dumi2.eq.1) ) then 
               vv(2,1) = dumr
            else if ( (dumi1.eq.2).and.(dumi2.eq.2) ) then 
               vv(2,2) = dumr
            end if 

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then
            goto 4408
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_VMIX: block mass not terminated? "
      call HARD_STOP

 4408 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_ALPHA(nin2,lowmass,done)

      implicit none

      integer   nin2,ii
      real*8    dumr,lowmass(0:99)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4409) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumr
            
            lowmass(44) = sin( dumr )
            lowmass(45) = cos( dumr )
            
         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then
            goto 4409
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_ALPHA: block mass not terminated? "
      call HARD_STOP

 4409 backspace nin2
      done = .true.
      return

      end 


c----------------------
      subroutine READ_BLOCK_HMIX(nin2,lowmass,unimass,done)

      implicit none

      integer   nin2,ii,dumi
      real*8    dumr,unimass(20),lowmass(0:99)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4410) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi,dumr
            
            if (dumi.eq.1) then 
              lowmass(0) = dumr 
            else if (dumi.eq.2) then 
              unimass(10) = dumr ! USUALLY PREFERRED
            end if 
ctp            else 
ctp               print*, " READ_BLOCK_HMIX: problem with first entry "
ctp               call HARD_STOP

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then
            goto 4410
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_HMIX: block mass not terminated? "
      call HARD_STOP

 4410 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_AU(nin2,lowmass,done)

      implicit none

      integer   nin2,ii,dumi1,dumi2
      real*8    dumr,lowmass(0:99)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4411) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi1,dumi2,dumr
            
            if ( (dumi1.eq.3).and.(dumi2.eq.3) ) then 
               lowmass(24) = -dumr 
            end if 
ctp            else 
ctp               print*, " READ_BLOCK_AU: problem with first entry "
ctp               call HARD_STOP

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then
            goto 4411
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_AU: block mass not terminated? "
      call HARD_STOP

 4411 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_AD(nin2,lowmass,done)

      implicit none

      integer   nin2,ii,dumi1,dumi2
      real*8    dumr,lowmass(0:99)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4412) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi1,dumi2,dumr
            
            if ( (dumi1.eq.3).and.(dumi2.eq.3) ) then 
               lowmass(21) = -dumr 
            end if 
ctp            else 
ctp               print*, " READ_BLOCK_AD: problem with first entry "
ctp               call HARD_STOP

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then
            goto 4412
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_AD: block mass not terminated? "
      call HARD_STOP

 4412 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_AE(nin2,lowmass,done)

      implicit none

      integer   nin2,ii,dumi1,dumi2
      real*8    dumr,lowmass(0:99)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4413) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi1,dumi2,dumr
            
            if ( (dumi1.eq.3).and.(dumi2.eq.3) ) then 
               lowmass(36) = -dumr 
            end if 
ctp            else 
ctp               print*, " READ_BLOCK_AE: problem with first entry "
ctp               call HARD_STOP

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then
            goto 4413
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_AE: block mass not terminated? "
      call HARD_STOP

 4413 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_YU(nin2,lowmass,done)

      implicit none

      include 'sm_read_values.inc'

      integer   nin2,ii,dumi1,dumi2
      real*8    dumr,lowmass(0:99)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4413) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi1,dumi2,dumr
            
            if ( (dumi1.eq.3).and.(dumi2.eq.3) ) then 
              yt = dumr
            end if 
ctp            else 
ctp               print*, " READ_BLOCK_AE: problem with first entry "
ctp               call HARD_STOP

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then
            goto 4413
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_YU: block mass not terminated? "
      call HARD_STOP

 4413 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_YD(nin2,lowmass,done)

      implicit none

      include 'sm_read_values.inc'

      integer   nin2,ii,dumi1,dumi2
      real*8    dumr,lowmass(0:99)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4413) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi1,dumi2,dumr
            
            if ( (dumi1.eq.3).and.(dumi2.eq.3) ) then 
              yb = dumr
            end if 
ctp            else 
ctp               print*, " READ_BLOCK_AE: problem with first entry "
ctp               call HARD_STOP

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then
            goto 4413
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_YD: block mass not terminated? "
      call HARD_STOP

 4413 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_YE(nin2,lowmass,done)

      implicit none

      include 'sm_read_values.inc'

      integer   nin2,ii,dumi1,dumi2
      real*8    dumr,lowmass(0:99)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4413) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi1,dumi2,dumr
            
            if ( (dumi1.eq.3).and.(dumi2.eq.3) ) then 
              yl = dumr
            end if 
ctp            else 
ctp               print*, " READ_BLOCK_AE: problem with first entry "
ctp               call HARD_STOP

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then
            goto 4413
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_YE: block mass not terminated? "
      call HARD_STOP

 4413 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_SMINPUTS(nin2,done)

      implicit none

      integer   nin2,ii,icnt,dumi
      real*8    dumr
      character line1*1
      logical done

      include 'coupl.inc'
      include 'sm_read_values.inc'

      done = .false.
      icnt = 0

      do ii=1,200,1
         read(nin2,'(a1)',end=4413) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*) dumi,dumr
            
            if ( dumi.eq.1 ) then 
              alpha = dumr 
              alpha = 1d0/alpha
              icnt = icnt+1
            else if ( dumi.eq.2 ) then 
              gfermi = dumr 
              icnt = icnt+1
            else if ( dumi.eq.3 ) then 
              alfas = dumr 
              icnt = icnt+1
            else if ( dumi.eq.4 ) then 
              zmass = dumr 
              icnt = icnt+1
            else if ( dumi.eq.6 ) then 
              tmass = dumr 
              icnt = icnt+1
            else if ( dumi.eq.7 ) then 
              lmass = dumr 
              icnt = icnt+1
            end if 
ctp            else 
ctp               print*, " READ_BLOCK_AE: problem with first entry "
ctp               call HARD_STOP

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then
            goto 4413
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_SMINPUTS: block not terminated? "
      call HARD_STOP

 4413 backspace nin2
      if(icnt.eq.6) done = .true.
      return

      end 

c----------------------
      subroutine READ_BLOCK_MSOFT(nin2,lowmass,done)

      implicit none

      integer   nin2,ii,dumi1
      real*8    dumr,lowmass(0:99)
      character line1*1
      logical done

      done = .false.
      
      do ii=1,200,1
         read(nin2,'(a1)',end=4414) line1

c            decide what it is and read the line if mass 
         if (line1.eq.' ') then
            backspace nin2
            read(nin2,*)dumi1,dumr
            
            if (dumi1.eq.1) then 
               lowmass(1) = dumr 
            else if (dumi1.eq.2) then 
               lowmass(2) = dumr 
            else if (dumi1.eq.3) then 
               lowmass(3) = dumr 
            end if 
ctp            else 
ctp               print*, " READ_BLOCK_MSOFT: problem with first entry "
ctp               call HARD_STOP

         else if (line1.eq.'#') then
            cycle
         else if ((line1.eq.'D').or.(line1.eq.'B')) then
            goto 4414
         end if 

      end do
      
c            block should be terminated before
      print*, " READ_BLOCK_MSOFT: block mass not terminated? "
      call HARD_STOP

 4414 backspace nin2
      done = .true.
      return

      end 

c----------------------
      subroutine READ_DECAY(nin2,width,check_sq,width_sq)

      implicit none

      integer   nin2,dumi,check_sq(1:13)
      real*8    dumr,width(0:99),width_sq(1:10)
      character line1*5

      include 'coupl.inc' ! JA

      backspace nin2
      read(nin2,*)line1,dumi,dumr

      if (line1.ne.'DECAY') then 
         print*, " READ_DECAY: fuck slha! ",line1
         call HARD_STOP
      end if 

c            array defined in Xsugra.f
      if (dumi.eq.1000021) then 
         width(4) = dumr
      else if (dumi.eq.1000022) then 
         width(5) = dumr
      else if (dumi.eq.1000023) then 
         width(6) = dumr
      else if (dumi.eq.1000025) then 
         width(7) = dumr
      else if (dumi.eq.1000035) then 
         width(8) = dumr
      else if (dumi.eq.1000024) then 
         width(9) = dumr
      else if (dumi.eq.1000037) then 
         width(10) = dumr
      else if (dumi.eq.1000001) then 
         width(11) = dumr
         width_sq(1) = dumr
         check_sq(1) = 1 
      else if (dumi.eq.2000001) then 
         width(12) = dumr
         width_sq(2) = dumr
         check_sq(2) = 1 
      else if (dumi.eq.1000002) then 
         width(13) = dumr
         width_sq(3) = dumr
         check_sq(3) = 1 
      else if (dumi.eq.2000002) then 
         width(14) = dumr
         width_sq(4) = dumr
         check_sq(4) = 1 
      else if (dumi.eq.1000005) then 
         width(17) = dumr
         width_sq(9) = dumr
         check_sq(9) = 1 
      else if (dumi.eq.2000005) then 
         width(18) = dumr
         width_sq(10)= dumr
         check_sq(10)= 1 
      else if (dumi.eq.1000006) then 
         width(19) = dumr
      else if (dumi.eq.2000006) then 
         width(20) = dumr
      else if (dumi.eq.1000011) then 
         width(30) = dumr
      else if (dumi.eq.2000011) then 
         width(31) = dumr
      else if (dumi.eq.1000012) then 
         width(32) = dumr
      else if (dumi.eq.1000015) then 
         width(33) = dumr
      else if (dumi.eq.2000015) then 
         width(34) = dumr
      else if (dumi.eq.1000016) then 
         width(35) = dumr
      else if (dumi.eq.36) then 
         width(40) = dumr
      else if (dumi.eq.25) then 
         width(41) = dumr
      else if (dumi.eq.35) then 
         width(42) = dumr
      else if (dumi.eq.37) then 
         width(43) = dumr
      else if (dumi.eq.1000003) then 
         width_sq(5) = dumr
         check_sq(5) = 1 
      else if (dumi.eq.2000003) then 
         width_sq(6) = dumr
         check_sq(6) = 1 
      else if (dumi.eq.1000004) then 
         width_sq(7) = dumr
         check_sq(7) = 1 
      else if (dumi.eq.2000004) then 
         width_sq(8) = dumr      
         check_sq(8) = 1 
      else if (dumi.eq.6) then  ! JA
         twidth = dumr      
         check_sq(11) = 1 
      else if (dumi.eq.23) then 
         zwidth = dumr      
         check_sq(12) = 1 
      else if (dumi.eq.24) then 
         wwidth = dumr      
         check_sq(13) = 1 
      end if 

      end 


c----------------------
c note that there might be a problem with some signs of the angles
      subroutine EXTRACT_SLHA_WEAK(lowmass,bw,uu,vv,mw,mz,sw,cw,sb,cb)
      
      implicit none
      
      integer   i1, i2, i3
      real*8    lowmass(0:99),bw(4,4),uu(2,2),vv(2,2)
      real*8    mw,mz,sw,cw,sb,cb,tb
      real*8    diag_n(1:4,1:4),diag_c(1:2,1:2)
      real*8    dummy_n(1:4,1:4),dummy_c(1:2,1:2)
      real*8    mass_n(1:4,1:4),mass_c(1:2,1:2)
      real*8    zero, one, two
      parameter( zero = 0d0, one = 1d0, two = 2d0 )
      logical ldebug
      parameter( ldebug=.true. )

 111  format( 4(G16.10,2x) )
 112  format( a11,4(G16.10,2x) )

      do i1=1,4
      do i2=1,4
         diag_n(i1,i2) = zero
      end do
      end do

      do i1=1,4
         diag_n(i1,i1) = lowmass(i1+4)
      end do

      do i1=1,4
      do i2=1,4
         dummy_n(i1,i2) = zero
         do i3=1,4
            dummy_n(i1,i2) = dummy_n(i1,i2) + diag_n(i1,i3)*bw(i3,i2)
         end do
      end do
      end do

      do i1=1,4
      do i2=1,4
         mass_n(i1,i2) = zero
         do i3=1,4
            mass_n(i1,i2) = mass_n(i1,i2) + bw(i3,i1)*dummy_n(i3,i2)
         end do
      end do
      end do

      if (ldebug) then
         print*, 'EXTRACT_SLHA_WEAK: neutralino mass matrix '
         write(6,111) mass_n(1,1),mass_n(1,2),mass_n(1,3),mass_n(1,4)
         write(6,111) mass_n(2,1),mass_n(2,2),mass_n(2,3),mass_n(2,4)
         write(6,111) mass_n(3,1),mass_n(3,2),mass_n(3,3),mass_n(3,4)
         write(6,111) mass_n(4,1),mass_n(4,2),mass_n(4,3),mass_n(4,4)
      end if

      mz = sqrt(  mass_n(1,3)**2 + mass_n(1,4)**2
     &          + mass_n(2,3)**2 + mass_n(2,4)**2 )
      sw = sqrt( mass_n(1,3)**2 + mass_n(1,4)**2 )/mz
      cw = sqrt( mass_n(2,3)**2 + mass_n(2,4)**2 )/mz
      sb = sqrt( mass_n(1,4)**2 + mass_n(2,4)**2 )/mz
      cb = sqrt( mass_n(1,3)**2 + mass_n(2,3)**2 )/mz

      if (ldebug) then
         print*, 'EXTRACT_SLHA_WEAK: neutralino parameters '
         write(6,112) ' m_Z     = ',mz
         write(6,112) ' s2w,c2w = ',sw**2,cw**2,sw**2+cw**2
         write(6,112) ' beta    = ',sb,cb,sb/cb,sb**2+cb**2
      end if

      do i1=1,2
      do i2=1,2
         diag_c(i1,i2) = zero
      end do
      end do

      do i1=1,2
         diag_c(i1,i1) = lowmass(i1+8)
      end do

      do i1=1,2
      do i2=1,2
         dummy_c(i1,i2) = zero
         do i3=1,2
            dummy_c(i1,i2) = dummy_c(i1,i2) + diag_c(i1,i3)*vv(i3,i2)
         end do
      end do
      end do

      do i1=1,2
      do i2=1,2
         mass_c(i1,i2) = zero
         do i3=1,2
            mass_c(i1,i2) = mass_c(i1,i2) + uu(i3,i1)*dummy_c(i3,i2)
         end do
      end do
      end do

      if (ldebug) then
         print*, 'EXTRACT_SLHA_WEAK: chargino mass matrix '
         write(6,111) mass_c(1,1),mass_c(1,2)
         write(6,111) mass_c(2,1),mass_c(2,2)
      end if

      mw = sqrt(mass_c(1,2)**2+mass_c(2,1)**2)/sqrt(two)
      tb = mass_c(1,2)/mass_c(2,1)

      if (ldebug) then
         print*, 'EXTRACT_SLHA_WEAK: chargino parameters '
         write(6,112) ' m_W       = ',mw
         write(6,112) ' tan(beta) = ',tb
         print*, 'EXTRACT_SLHA_WEAK: theta_w from masses '
         write(6,112) ' s2w,c2w   = ',one-mw**2/mz**2,mw**2/mz**2
         print*
         write(6,*) 'DIFF BETWEEN Ni- and Xi-derived s^2(W) values:'
         write(6,*) sw**2 - (one-mw**2/mz**2)
         print*
      end if

      end 


           subroutine open_file_mdl(lun,filename,fopened)
c***********************************************************************
c     opens file input-card.dat in current directory or above
c***********************************************************************
      implicit none
c
c     Arguments
c
      integer lun
      logical fopened
      character*(*) filename
      character*90  tempname
      integer fine
      integer dirup,i

c-----
c  Begin Code
c-----
c
c     first check that we will end in the main directory
c
      tempname=filename
      fine=index(tempname,' ')
      if(fine.eq.0) fine=len(tempname)
      tempname=tempname(1:fine)
c
c         if I have to read a card
c
      if(index(filename,"_card").gt.0) then
         tempname='./Cards/'//tempname
      endif


      fopened=.false.
      do i=0,5
         open(unit=lun,file=tempname,status='old',ERR=30)
         fopened=.true.
         write(*,*) 'read model file',tempname
         exit
 30      tempname='../'//tempname
         if (i.eq.5)then
            write(*,*) 'Warning: file ',tempname,' is not correct'
            stop
         endif
      enddo


      return
      end
 

