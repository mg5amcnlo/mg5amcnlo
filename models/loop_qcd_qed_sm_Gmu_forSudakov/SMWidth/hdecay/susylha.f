c -------------------------------------------------------------------- c
c ------ This and the following subroutines read in the spectrum ----- c
c ------ file given in the SUSY Les Houches Accord format        ----- c
c ------ hep-ph/0311123.                                         ----- c
c ------ Thanks to Tilman Plehn for the first version which has  ----- c
c ------ been expanded and changed here.                         ----- c
c -------------------------------------------------------------------- c

      subroutine SLHA_read_leshouches_HDEC(ninlha)

      implicit double precision (a-h,m,o-z)
      double precision minval(1:20),smval(1:30),massval(1:50),
     .                 nmixval(4,4),umixval(2,2),vmixval(2,2),
     .                 stopmixval(2,2),sbotmixval(2,2),staumixval(2,2),
     .                 hmixval(1:10),gaugeval(1:3),msoftval(1:100),
     .                 auval(3,3),adval(3,3),aeval(3,3),yuval(3,3),
     .                 ydval(3,3),yeval(3,3),qvalue(1:20),extval(0:100)
      double precision tuval(3,3),tdval(3,3),teval(3,3)
      double precision vckmval(4)
      integer check(1:22),check_final,imod(1:2)
      character line1*6,line2*100,
     .          spinfo1*100,spinfo2*100,modselval*100,mincom(1:20)*20,
     .          extcom(0:100)*20,softcom(1:100)*20,hmixcom(1:10)*20
      logical done

      COMMON/SLHA_leshouches1_HDEC/spinfo1,spinfo2,modselval,mincom,
     .                             extcom,softcom,hmixcom
      COMMON/SLHA_leshouches2_HDEC/minval,extval,smval,massval,nmixval,
     .                      umixval,vmixval,stopmixval,sbotmixval,
     .                      staumixval,hmixval,gaugeval,msoftval,auval,
     .                      adval,aeval,yuval,ydval,yeval,alphaval,
     .                      qvalue,imod
      COMMON/SLHA_leshouches3_HDEC/vckmval
      COMMON/SLHA_checkval_HDEC/check

c -- start from the beginning of the file SLHA_leshouches.in --
      rewind(ninlha)

c -- initialization of the check array --
      do i1=1,22,1
         check(i1) = 0
      end do

c -- the scale Q -- 
      do i=1,20,1
         qvalue(i) = 0.D0
      end do

c a trick to jump over undefined parameters in subsequent writings
      unlikely = -123456789D0
      do i=1,100,1
         extval(i-1) = unlikely
      end do
      do i=1,20,1
         minval(i) = unlikely
      end do
      do i=1,10,1
         hmixval(i) = unlikely
      end do
      do i=1,100,1
         msoftval(i) = unlikely
      end do
      do i=1,3,1
       do j=1,3,1
         auval(i,j) = unlikely
         adval(i,j) = unlikely
         aeval(i,j) = unlikely
       end do
      end do
      do i=1,4,1
         vckmval(i) = unlikely
      end do

c ------------------------------------------------------------------- c
      do i=1,10000,1

c -- check if routine can be left --
         check_final = 1
         do i1=1,22,1
            check_final = check_final*check(i1)
         end do
         if(check_final.eq.1) then
            return
         endif

c -- read in new line --
         line1=' '
         read(ninlha,'(a6,a100)',end=9900,err=9900) line1,line2
         
c -- rewrite line1(1:6) and line2(1:20) to upper case --
         do j1=1,6,1
            if(line1(j1:j1).ne.'#') then
               do j2=97,122,1
                  if(line1(j1:j1).eq.char(j2)) line1(j1:j1)=char(j2-32)
               end do
            endif
         end do

         do j1=1,20,1
            if(line2(j1:j1).ne.'#') then
               do j2=97,122,1
                  if(line2(j1:j1).eq.char(j2)) line2(j1:j1)=char(j2-32)
               end do
            endif
         end do

c -- looks for blocks and reads them in one after the other --
         if(line1(1:1).eq.'B') then

c -- look for Block MODSEL --
            if(line2(1:6).eq.'MODSEL') then
               call SLHA_READ_MODSEL_HDEC(ninlha,modselval,imod,done)
               if (done) then
                  check(1) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in MODSEL'
               endif

c -- look for Block SMINPUTS --
            elseif(line2(1:8).eq.'SMINPUTS') then
               call SLHA_READ_SMINPUTS_HDEC(ninlha,smval,done)
               if (done) then
                  check(2) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in SMINPUTS'
               endif

c -- look for Block MINPAR --
            elseif(line2(1:6).eq.'MINPAR') then
               call SLHA_READ_MINPAR_HDEC(ninlha,minval,mincom,done)
               if (done) then
                  check(3) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in MINPAR'
               endif

c -- look for Block EXTPAR --
            elseif(line2(1:6).eq.'EXTPAR') then
               call SLHA_READ_EXTPAR_HDEC(ninlha,extval,extcom,done)
               if (done) then
                  check(4) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in EXTPAR'
               endif

c -- look for Block MASS --
            elseif(line2(1:4).eq.'MASS') then
               call SLHA_READ_MASS_HDEC(ninlha,massval,done)
               if (done) then
                  check(5) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in MASS'
               endif

c -- look for Block NMIX --
            elseif(line2(1:4).eq.'NMIX') then
               call SLHA_READ_NMIX_HDEC(ninlha,nmixval,done)
               if (done) then
                  check(6) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in NMIX'
               endif

c -- look for Block UMIX --
            elseif(line2(1:4).eq.'UMIX') then
               call SLHA_READ_UMIX_HDEC(ninlha,umixval,done)
               if (done) then
                  check(7) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in UMIX'
               endif

c -- look for Block VMIX --
            elseif(line2(1:4).eq.'VMIX') then
               call SLHA_READ_VMIX_HDEC(ninlha,vmixval,done)
               if (done) then
                  check(8) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in VMIX'
               endif

c -- look for Block STOPMIX --
            elseif(line2(1:7).eq.'STOPMIX') then
               call SLHA_READ_STOPMIX_HDEC(ninlha,stopmixval,done)
               if (done) then
                  check(9) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in STOPMIX' 
               endif

c -- look for Block SBOTMIX --
            elseif(line2(1:7).eq.'SBOTMIX') then
               call SLHA_READ_SBOTMIX_HDEC(ninlha,sbotmixval,done)
               if (done) then
                  check(10) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in SBOTMIX' 
               endif

c -- look for Block STAUMIX --
            elseif(line2(1:7).eq.'STAUMIX') then
               call SLHA_READ_STAUMIX_HDEC(ninlha,staumixval,done)
               if (done) then
                  check(11) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in STAUMIX' 
               endif

c -- look for Block ALPHA --
            elseif(line2(1:5).eq.'ALPHA') then
               call SLHA_READ_ALPHA_HDEC(ninlha,alphaval,done)
               if (done) then
                  check(12) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in ALPHA'
               endif

c -- look for Block HMIX --
            elseif(line2(1:4).eq.'HMIX') then
c              backspace ninlha
c              read(ninlha,'(16x,E16.8)') Qval
               call readQval(ninlha,Qval)
               qvalue(1) = Qval
               call SLHA_READ_HMIX_HDEC(ninlha,hmixval,hmixcom,done)
               if (done) then
                  check(13) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in HMIX'
               endif

c -- look for Block GAUGE --
            elseif(line2(1:5).eq.'GAUGE') then
c              backspace ninlha
c              read(ninlha,'(16x,E16.8)') Qval
               call readQval(ninlha,Qval)
               qvalue(2) = Qval
               call SLHA_READ_GAUGE_HDEC(ninlha,gaugeval,done)
               if (done) then
                  check(14) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in GAUGE'
               endif

c -- look for Block MSOFT --
            elseif(line2(1:5).eq.'MSOFT') then
c              backspace ninlha
c              read(ninlha,'(16x,E16.8)') Qval
               call readQval(ninlha,Qval)
               qvalue(3) = Qval
               call SLHA_READ_MSOFT_HDEC(ninlha,msoftval,softcom,done)
               if (done) then
                  check(15) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in MSOFT'
               endif

c -- look for Block AU --
            elseif(line2(1:2).eq.'AU') then
c              backspace ninlha
c              read(ninlha,'(13x,E16.8)') Qval
               call readQval(ninlha,Qval)
               qvalue(4) = Qval
               call SLHA_READ_AU_HDEC(ninlha,auval,done)
               if (done) then
                  check(16) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in AU'
               endif

c -- look for Block AD --
            elseif(line2(1:2).eq.'AD') then
c              backspace ninlha
c              read(ninlha,'(13x,E16.8)') Qval
               call readQval(ninlha,Qval)
               qvalue(5) = Qval
               call SLHA_READ_AD_HDEC(ninlha,adval,done)
               if (done) then
                  check(17) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in AD'
               endif

c -- look for Block AE --
            elseif(line2(1:2).eq.'AE') then
c              backspace ninlha
c              read(ninlha,'(13x,E16.8)') Qval
               call readQval(ninlha,Qval)
               qvalue(6) = Qval
               call SLHA_READ_AE_HDEC(ninlha,aeval,done)
               if (done) then
                  check(18) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in AE'
               endif

c -- look for Block YU --
            elseif(line2(1:2).eq.'YU') then
c              backspace ninlha
c              read(ninlha,'(13x,E16.8)') Qval
               call readQval(ninlha,Qval)
               qvalue(7) = Qval
               call SLHA_READ_YU_HDEC(ninlha,yuval,done)
               if (done) then
                  check(19) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in YU'
               endif

c -- look for Block YD --
            elseif(line2(1:2).eq.'YD') then
c              backspace ninlha
c              read(ninlha,'(13x,E16.8)') Qval
               call readQval(ninlha,Qval)
               qvalue(8) = Qval
               call SLHA_READ_YD_HDEC(ninlha,ydval,done)
               if (done) then
                  check(20) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in YD'
               endif

c -- look for Block YE --
            elseif(line2(1:2).eq.'YE') then
c              backspace ninlha
c              read(ninlha,'(13x,E16.8)') Qval
               call readQval(ninlha,Qval)
               qvalue(9) = Qval
               call SLHA_READ_YE_HDEC(ninlha,yeval,done)
               if (done) then
                  check(21) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in YE'
               endif

c -- look for Block SPINFO --
            elseif(line2(1:6).eq.'SPINFO') then
               call SLHA_READ_SPINFO_HDEC(ninlha,spinfo1,spinfo2,done)
               if (done) then
                  check(22) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in SPINFO'
               endif

c -- look for Block TUIN --
            elseif(line2(1:4).eq.'TUIN') then
c              backspace ninlha
c              read(ninlha,'(13x,E16.8)') Qval
               call readQval(ninlha,Qval)
               qvalue(4) = Qval
               call SLHA_READ_TUIN_HDEC(ninlha,tuval,done)
               auval(1,1) = tuval(1,1)/smval(22)
               auval(2,2) = tuval(2,2)/smval(24)
               auval(3,3) = tuval(3,3)/smval(6)
               do ii=1,3
                do jj=1,3
                 if(ii.ne.jj)auval(ii,jj)=0
                enddo
               enddo
               if (done) then
                  check(16) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in TUIN'
               endif

c -- look for Block TDIN --
            elseif(line2(1:4).eq.'TDIN') then
c              backspace ninlha
c              read(ninlha,'(13x,E16.8)') Qval
               call readQval(ninlha,Qval)
               qvalue(5) = Qval
               call SLHA_READ_TDIN_HDEC(ninlha,tdval,done)
               adval(1,1) = tdval(1,1)/smval(21)
               adval(2,2) = tdval(2,2)/smval(23)
               adval(3,3) = tdval(3,3)/smval(5)
               do ii=1,3
                do jj=1,3
                 if(ii.ne.jj)adval(ii,jj)=0
                enddo
               enddo
               if (done) then
                  check(17) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in TDIN'
               endif

c -- look for Block TEIN --
            elseif(line2(1:4).eq.'TEIN') then
c              backspace ninlha
c              read(ninlha,'(13x,E16.8)') Qval
               call readQval(ninlha,Qval)
               qvalue(6) = Qval
               call SLHA_READ_TEIN_HDEC(ninlha,teval,done)
               aeval(1,1) = teval(1,1)/smval(11)
               aeval(2,2) = teval(2,2)/smval(13)
               aeval(3,3) = teval(3,3)/smval(7)
               do ii=1,3
                do jj=1,3
                 if(ii.ne.jj)aeval(ii,jj)=0
                enddo
               enddo
               if (done) then
                  check(18) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in TEIN'
               endif

c -- look for Block VCKMIN --
            elseif(line2(1:6).eq.'VCKMIN') then
               call SLHA_READ_VCKMIN_HDEC(ninlha,vckmval,done)
               if (done) then
                  check(2) = 1
                  goto 1111
               else
                  print*,'SLHA_read_leshouches: problem in VCKMIN'
               endif

c -- continue if the Block is not interesting --
            else
               goto 1111
            endif

c -- continue if it is not a Block statement --
         else
            goto 1111
         endif

c -- maximum number of lines exhausted --
 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'

      return
      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_MODSEL_HDEC(ninlha,modselval,imod,done)

      implicit double precision (a-h,m,o-z)
      integer imod(1:2)
      character line1*1,line2*1,line3*100,modselval*100
      logical done

      done=.false.

      modselval = ' '

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,line2,line3

            if(idum1.eq.1) then
               imod(1) = idum1
               imod(2) = idum2
               modselval = line3
            endif

         elseif(line1.eq.'#') then
            go to 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done =.true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_SMINPUTS_HDEC(ninlha,smval,done)

      implicit double precision (a-h,m,o-z)
      double precision smval(30)
      character line1*1
      logical done

      done=.false.

      do i=1,30,1
         smval(i) = 0.D0
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum,val

c -- inverse EM coupling at the Z pole in the MS_bar scheme (with --
c -- five active flavours) --
            if(idum.eq.1) then
               smval(1) = val
c -- G_F, Fermi constant (in units of GeV^-2)
            elseif(idum.eq.2) then
               smval(2) = val
c -- Strong coupling at the Z pole in the MS_bar scheme (with five --
c -- active flavours) --
            elseif(idum.eq.3) then
               smval(3) = val
c -- M_Z, pole mass --
            elseif(idum.eq.4) then
               smval(4) = val
c -- mb(mb)^MS_bar. b quark running mass in the MS_bar scheme --
            elseif(idum.eq.5) then
               smval(5) = val
c -- mt, pole mass --
            elseif(idum.eq.6) then
               smval(6) = val
c -- mtau, pole mass --
            elseif(idum.eq.7) then
               smval(7) = val
            endif
            
         elseif(line1.eq.'#') then
            go to 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done =.true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_MINPAR_HDEC(ninlha,minval,mincom,done)

      implicit double precision (a-h,m,o-z)
      double precision minval(20)
      character line1*1,line2*1,line3*20,mincom(1:20)*20
      logical done

      done= .false.

      do i=1,20,1
         mincom(i) = ' '
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum,val,line2,line3

            do ii=1,6,1
               if(idum.eq.ii) then
                  minval(ii) = val
                  mincom(ii) = line3
               endif
            end do

c -- i=3: value for tanbeta(MZ) --
            
         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_EXTPAR_HDEC(ninlha,extval,extcom,done)

      implicit double precision (a-h,m,o-z)
      double precision extval(0:100)
      character line1*1,line2*1,line3*20,extcom(0:100)*20
      logical done

      done= .false.

      do i=1,100,1
         extcom(i-1) = ' '
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum,val,line2,line3

            do ii=1,54,1
               if(idum.eq.(ii-1)) then
                  extval(ii-1) = val
                  extcom(ii-1) = line3
               endif
            end do

         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_MASS_HDEC(ninlha,massval,done)

      implicit double precision (a-h,m,o-z)
      double precision massval(50)
      character line1*1
      logical done

      done= .false.

      do i=1,50,1
         massval(i) = 0.D0
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum,val

c -- value for M_W --
            if(idum.eq.24) then
               massval(1) = val
c -- value for M_h --
            elseif(idum.eq.25) then
               massval(2) = val
c -- value for M_H --
            elseif(idum.eq.35) then
               massval(3) = val
c -- value for M_A --
            elseif(idum.eq.36) then
               massval(4) = val
c -- value for M_H+/- --
            elseif(idum.eq.37) then
               massval(5) = val
c -- value for ~d_L --
            elseif(idum.eq.1000001) then
               massval(6) = val
c -- value for ~d_R --
            elseif(idum.eq.2000001) then
               massval(7) = val
c -- value for ~u_L --
            elseif(idum.eq.1000002) then
               massval(8) = val
c -- value for ~u_R --
            elseif(idum.eq.2000002) then
               massval(9) = val
c -- value for ~s_L --
            elseif(idum.eq.1000003) then
               massval(10) = val
c -- value for ~s_R --
            elseif(idum.eq.2000003) then
               massval(11) = val
c -- value for ~c_L --
            elseif(idum.eq.1000004) then
               massval(12) = val
c -- value for ~c_R --
            elseif(idum.eq.2000004) then
               massval(13) = val
c -- value for ~b_1 --
            elseif(idum.eq.1000005) then
               massval(14) = val
c -- value for ~b_2 --
            elseif(idum.eq.2000005) then
               massval(15) = val
c -- value for ~t_1 --
            elseif(idum.eq.1000006) then
               massval(16) = val
c -- value for ~t_2 --
            elseif(idum.eq.2000006) then
               massval(17) = val
c -- value for ~e_L --
            elseif(idum.eq.1000011) then
               massval(18) = val
c -- value for ~e_R --
            elseif(idum.eq.2000011) then
               massval(19) = val
c -- value for ~nu_eL --
            elseif(idum.eq.1000012) then
               massval(20) = val
c -- value for ~mu_L --
            elseif(idum.eq.1000013) then
               massval(21) = val
c -- value for ~mu_R --
            elseif(idum.eq.2000013) then
               massval(22) = val
c -- value for ~nu_muL --
            elseif(idum.eq.1000014) then
               massval(23) = val
c -- value for ~tau_1 --
            elseif(idum.eq.1000015) then
               massval(24) = val
c -- value for ~tau_2 --
            elseif(idum.eq.2000015) then
               massval(25) = val
c -- value for ~nu_tauL --
            elseif(idum.eq.1000016) then
               massval(26) = val
c -- value for ~g --
            elseif(idum.eq.1000021) then
               massval(27) = val
c -- value for ~chi_10 --
            elseif(idum.eq.1000022) then
               massval(28) = val
c -- value for ~chi_20 --
            elseif(idum.eq.1000023) then
               massval(29) = val
c -- value for ~chi_30 --
            elseif(idum.eq.1000025) then
               massval(30) = val
c -- value for ~chi_40 --
            elseif(idum.eq.1000035) then
               massval(31) = val
c -- value for ~chi_1+ --
            elseif(idum.eq.1000024) then
               massval(32) = val
c -- value for ~chi_2+ --
            elseif(idum.eq.1000037) then
               massval(33) = val
            endif
            
         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_NMIX_HDEC(ninlha,nmixval,done)

      implicit double precision (a-h,m,o-z)
      double precision nmixval(4,4)
      character line1*1
      logical done

      done= .false.

      do i=1,4,1
         do j=1,4,1
            nmixval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- the values for the neutralino mixing matrix --
            if(idum1.eq.1.and.idum2.eq.1) then
               nmixval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               nmixval(1,2) = val
            elseif(idum1.eq.1.and.idum2.eq.3) then
               nmixval(1,3) = val
            elseif(idum1.eq.1.and.idum2.eq.4) then
               nmixval(1,4) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               nmixval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               nmixval(2,2) = val
            elseif(idum1.eq.2.and.idum2.eq.3) then
               nmixval(2,3) = val
            elseif(idum1.eq.2.and.idum2.eq.4) then
               nmixval(2,4) = val
            elseif(idum1.eq.3.and.idum2.eq.1) then
               nmixval(3,1) = val
            elseif(idum1.eq.3.and.idum2.eq.2) then
               nmixval(3,2) = val
            elseif(idum1.eq.3.and.idum2.eq.3) then
               nmixval(3,3) = val
            elseif(idum1.eq.3.and.idum2.eq.4) then
               nmixval(3,4) = val
            elseif(idum1.eq.4.and.idum2.eq.1) then
               nmixval(4,1) = val
            elseif(idum1.eq.4.and.idum2.eq.2) then
               nmixval(4,2) = val
            elseif(idum1.eq.4.and.idum2.eq.3) then
               nmixval(4,3) = val
            elseif(idum1.eq.4.and.idum2.eq.4) then
               nmixval(4,4) = val
            endif
            
         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_UMIX_HDEC(ninlha,umixval,done)

      implicit double precision (a-h,m,o-z)
      double precision umixval(2,2)
      character line1*1
      logical done

      done= .false.

      do i=1,2,1
         do j=1,2,1
            umixval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- the values for the chargino mixing matrix U --
            if(idum1.eq.1.and.idum2.eq.1) then
               umixval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               umixval(1,2) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               umixval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               umixval(2,2) = val
            endif
            
         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_VMIX_HDEC(ninlha,vmixval,done)

      implicit double precision (a-h,m,o-z)
      double precision vmixval(2,2)
      character line1*1
      logical done

      done= .false.

      do i=1,2,1
         do j=1,2,1
            vmixval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- the values for the chargino mixing matrix V --
            if(idum1.eq.1.and.idum2.eq.1) then
               vmixval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               vmixval(1,2) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               vmixval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               vmixval(2,2) = val
            endif
            
         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_STOPMIX_HDEC(ninlha,stopmixval,done)

      implicit double precision (a-h,m,o-z)
      double precision stopmixval(2,2)
      character line1*1
      logical done

      done= .false.

      do i=1,2,1
         do j=1,2,1
            stopmixval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- the values for the stop mixing matrix: m(1,1) = cos(theta),     --
c -- m(1,2) = sin(theta), m(2,1) = -sin(theta), m(2,2) = cos(theta). --
            if(idum1.eq.1.and.idum2.eq.1) then
               stopmixval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               stopmixval(1,2) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               stopmixval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               stopmixval(2,2) = val
            endif
            
         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_SBOTMIX_HDEC(ninlha,sbotmixval,done)

      implicit double precision (a-h,m,o-z)
      double precision sbotmixval(2,2)
      character line1*1
      logical done

      done= .false.

      do i=1,2,1
         do j=1,2,1
            sbotmixval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- the values for the sbottom mixing matrix: m(1,1) = cos(theta),  --
c -- m(1,2) = sin(theta), m(2,1) = -sin(theta), m(2,2) = cos(theta). --
            if(idum1.eq.1.and.idum2.eq.1) then
               sbotmixval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               sbotmixval(1,2) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               sbotmixval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               sbotmixval(2,2) = val
            endif
            
         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_STAUMIX_HDEC(ninlha,staumixval,done)

      implicit double precision (a-h,m,o-z)
      double precision staumixval(2,2)
      character line1*1
      logical done

      done= .false.

      do i=1,2,1
         do j=1,2,1
            staumixval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- the values for the stau mixing matrix: m(1,1) = cos(theta),     --
c -- m(1,2) = sin(theta), m(2,1) = -sin(theta), m(2,2) = cos(theta). --
            if(idum1.eq.1.and.idum2.eq.1) then
               staumixval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               staumixval(1,2) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               staumixval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               staumixval(2,2) = val
            endif
            
         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_ALPHA_HDEC(ninlha,alphaval,done)

      implicit double precision (a-h,m,o-z)
      character line1*1
      logical done

      done= .false.

      alphaval = 0.D0

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) val

c -- value for the effective Higgs mixing parameter alpha --
c -- ! The given value is the 'best choice' solution. Depending on  ! --
c -- ! the spectrum calcluator it can be an on-shell parameter or   ! --
c -- ! can be given in the DR_bar definition at a certain           ! --
c -- ! characteristic scale etc. For details on the specific        ! --
c -- ! prescriptions see the manual of the particular spectrum      ! --
c -- ! calculator.                                                  ! --
            alphaval = val
            
         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_HMIX_HDEC(ninlha,hmixval,hmixcom,done)

      implicit double precision (a-h,m,o-z)
      double precision hmixval(1:10)
      character line1*1,line2*1,line3*20,hmixcom(1:10)*20
      logical done

      done= .false.

      do i=1,10,1
         hmixcom(i) = ' '
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum,val,line2,line3

            do ii=1,10,1
               if(idum.eq.ii) then
                  hmixval(ii) = val
                  hmixcom(ii) = line3
               endif
            end do

         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_GAUGE_HDEC(ninlha,gaugeval,done)

      implicit double precision (a-h,m,o-z)
      double precision gaugeval(1:3)
      character line1*1
      logical done

      done= .false.

      do i=1,3,1
         gaugeval(i) = 0.D0
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum,val

c -- The following parameters are the DR_bar running parameters at --
c -- the scale Q.                                                  --
c -- value for g_prime at the scale Q --
            if(idum.eq.1) then
               gaugeval(1) = val

c -- value for g at the scale Q --
            elseif(idum.eq.2) then
               gaugeval(2) = val

c -- value for g3 at the scale Q --
            elseif(idum.eq.3) then
               gaugeval(3) = val
            endif

         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_MSOFT_HDEC(ninlha,msoftval,softcom,done)

      implicit double precision (a-h,m,o-z)
      double precision msoftval(1:100)
      character line1*1,line2*1,line3*20,softcom(1:100)*20
      logical done

      done= .false.

      do i=1,100,1
         softcom(i)   = ' ' 
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum,val,line2,line3

            do ii=1,100,1
               if(idum.eq.ii) then
                  msoftval(ii) = val
                  softcom(ii)  = line3
               endif
            end do

         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_AU_HDEC(ninlha,auval,done)

      implicit double precision (a-h,m,o-z)
      double precision auval(3,3)
      character line1*1
      logical done

      done= .false.

      do i=1,3,1
         do j=1,3,1
            auval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- The following parameters are the DR_bar running parameters at --
c -- the scale Q.                                                  --
c -- values for AU(i,j) at the scale Q --
            if(idum1.eq.1.and.idum2.eq.1) then
               auval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               auval(1,2) = val
            elseif(idum1.eq.1.and.idum2.eq.3) then
               auval(1,3) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               auval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               auval(2,2) = val
            elseif(idum1.eq.2.and.idum2.eq.3) then
               auval(2,3) = val
            elseif(idum1.eq.3.and.idum2.eq.1) then
               auval(3,1) = val
            elseif(idum1.eq.3.and.idum2.eq.2) then
               auval(3,2) = val
            elseif(idum1.eq.3.and.idum2.eq.3) then
               auval(3,3) = val
            endif

         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_AD_HDEC(ninlha,adval,done)

      implicit double precision (a-h,m,o-z)
      double precision adval(3,3)
      character line1*1
      logical done

      done= .false.

      do i=1,3,1
         do j=1,3,1
            adval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- The following parameters are the DR_bar running parameters at --
c -- the scale Q.                                                  --
c -- values for AD(i,j) at the scale Q --
            if(idum1.eq.1.and.idum2.eq.1) then
               adval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               adval(1,2) = val
            elseif(idum1.eq.1.and.idum2.eq.3) then
               adval(1,3) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               adval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               adval(2,2) = val
            elseif(idum1.eq.2.and.idum2.eq.3) then
               adval(2,3) = val
            elseif(idum1.eq.3.and.idum2.eq.1) then
               adval(3,1) = val
            elseif(idum1.eq.3.and.idum2.eq.2) then
               adval(3,2) = val
            elseif(idum1.eq.3.and.idum2.eq.3) then
               adval(3,3) = val
            endif

         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_AE_HDEC(ninlha,aeval,done)

      implicit double precision (a-h,m,o-z)
      double precision aeval(3,3)
      character line1*1
      logical done

      done= .false.

      do i=1,3,1
         do j=1,3,1
            aeval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- The following parameters are the DR_bar running parameters at --
c -- the scale Q.                                                  --
c -- values for AE(i,j) at the scale Q --
            if(idum1.eq.1.and.idum2.eq.1) then
               aeval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               aeval(1,2) = val
            elseif(idum1.eq.1.and.idum2.eq.3) then
               aeval(1,3) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               aeval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               aeval(2,2) = val
            elseif(idum1.eq.2.and.idum2.eq.3) then
               aeval(2,3) = val
            elseif(idum1.eq.3.and.idum2.eq.1) then
               aeval(3,1) = val
            elseif(idum1.eq.3.and.idum2.eq.2) then
               aeval(3,2) = val
            elseif(idum1.eq.3.and.idum2.eq.3) then
               aeval(3,3) = val
            endif

         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_YU_HDEC(ninlha,yuval,done)

      implicit double precision (a-h,m,o-z)
      double precision yuval(3,3)
      character line1*1
      logical done

      done= .false.

      do i=1,3,1
         do j=1,3,1
            yuval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- The following parameters are the DR_bar running parameters at --
c -- the scale Q.                                                  --
c -- values for YU(i,j) at the scale Q --
            if(idum1.eq.1.and.idum2.eq.1) then
               yuval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               yuval(1,2) = val
            elseif(idum1.eq.1.and.idum2.eq.3) then
               yuval(1,3) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               yuval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               yuval(2,2) = val
            elseif(idum1.eq.2.and.idum2.eq.3) then
               yuval(2,3) = val
            elseif(idum1.eq.3.and.idum2.eq.1) then
               yuval(3,1) = val
            elseif(idum1.eq.3.and.idum2.eq.2) then
               yuval(3,2) = val
            elseif(idum1.eq.3.and.idum2.eq.3) then
               yuval(3,3) = val
            endif

         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_YD_HDEC(ninlha,ydval,done)

      implicit double precision (a-h,m,o-z)
      double precision ydval(3,3)
      character line1*1
      logical done

      done= .false.

      do i=1,3,1
         do j=1,3,1
            ydval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- The following parameters are the DR_bar running parameters at --
c -- the scale Q.                                                  --
c -- values for YD(i,j) at the scale Q --
            if(idum1.eq.1.and.idum2.eq.1) then
               ydval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               ydval(1,2) = val
            elseif(idum1.eq.1.and.idum2.eq.3) then
               ydval(1,3) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               ydval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               ydval(2,2) = val
            elseif(idum1.eq.2.and.idum2.eq.3) then
               ydval(2,3) = val
            elseif(idum1.eq.3.and.idum2.eq.1) then
               ydval(3,1) = val
            elseif(idum1.eq.3.and.idum2.eq.2) then
               ydval(3,2) = val
            elseif(idum1.eq.3.and.idum2.eq.3) then
               ydval(3,3) = val
            endif

         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_YE_HDEC(ninlha,yeval,done)

      implicit double precision (a-h,m,o-z)
      double precision yeval(3,3)
      character line1*1
      logical done

      done= .false.

      do i=1,3,1
         do j=1,3,1
            yeval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- The following parameters are the DR_bar running parameters at --
c -- the scale Q.                                                  --
c -- values for YE(i,j) at the scale Q --
            if(idum1.eq.1.and.idum2.eq.1) then
               yeval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               yeval(1,2) = val
            elseif(idum1.eq.1.and.idum2.eq.3) then
               yeval(1,3) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               yeval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               yeval(2,2) = val
            elseif(idum1.eq.2.and.idum2.eq.3) then
               yeval(2,3) = val
            elseif(idum1.eq.3.and.idum2.eq.1) then
               yeval(3,1) = val
            elseif(idum1.eq.3.and.idum2.eq.2) then
               yeval(3,2) = val
            elseif(idum1.eq.3.and.idum2.eq.3) then
               yeval(3,3) = val
            endif

         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_SPINFO_HDEC(ninlha,spinfo1,spinfo2,done)

      implicit double precision (a-h,m,o-z)
      character line1*1,line2*100,spinfo1*100,spinfo2*100
      logical done

      done= .false.

      spinfo1 = ' '
      spinfo2 = ' '

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,'(1x,i5,3x,a100)') idum,line2

c -- the name of the spectrum calculator --
            if(idum.eq.1) then
               spinfo1 = line2

c -- the version number of the spectrum calculator --
            elseif(idum.eq.2) then
               spinfo2 = line2
            endif

         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_TUIN_HDEC(ninlha,auval,done)

      implicit double precision (a-h,m,o-z)
      double precision auval(3,3)
      character line1*1
      logical done

      done= .false.

      do i=1,3,1
         do j=1,3,1
            auval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- The following parameters are the DR_bar running parameters at --
c -- the scale Q.                                                  --
c -- values for AU(i,j) at the scale Q --
            if(idum1.eq.1.and.idum2.eq.1) then
               auval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               auval(1,2) = val
            elseif(idum1.eq.1.and.idum2.eq.3) then
               auval(1,3) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               auval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               auval(2,2) = val
            elseif(idum1.eq.2.and.idum2.eq.3) then
               auval(2,3) = val
            elseif(idum1.eq.3.and.idum2.eq.1) then
               auval(3,1) = val
            elseif(idum1.eq.3.and.idum2.eq.2) then
               auval(3,2) = val
            elseif(idum1.eq.3.and.idum2.eq.3) then
               auval(3,3) = val
            endif

         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_TDIN_HDEC(ninlha,adval,done)

      implicit double precision (a-h,m,o-z)
      double precision adval(3,3)
      character line1*1
      logical done

      done= .false.

      do i=1,3,1
         do j=1,3,1
            adval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- The following parameters are the DR_bar running parameters at --
c -- the scale Q.                                                  --
c -- values for AD(i,j) at the scale Q --
            if(idum1.eq.1.and.idum2.eq.1) then
               adval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               adval(1,2) = val
            elseif(idum1.eq.1.and.idum2.eq.3) then
               adval(1,3) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               adval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               adval(2,2) = val
            elseif(idum1.eq.2.and.idum2.eq.3) then
               adval(2,3) = val
            elseif(idum1.eq.3.and.idum2.eq.1) then
               adval(3,1) = val
            elseif(idum1.eq.3.and.idum2.eq.2) then
               adval(3,2) = val
            elseif(idum1.eq.3.and.idum2.eq.3) then
               adval(3,3) = val
            endif

         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_TEIN_HDEC(ninlha,aeval,done)

      implicit double precision (a-h,m,o-z)
      double precision aeval(3,3)
      character line1*1
      logical done

      done= .false.

      do i=1,3,1
         do j=1,3,1
            aeval(i,j) = 0.D0
         end do
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum1,idum2,val

c -- The following parameters are the DR_bar running parameters at --
c -- the scale Q.                                                  --
c -- values for AE(i,j) at the scale Q --
            if(idum1.eq.1.and.idum2.eq.1) then
               aeval(1,1) = val
            elseif(idum1.eq.1.and.idum2.eq.2) then
               aeval(1,2) = val
            elseif(idum1.eq.1.and.idum2.eq.3) then
               aeval(1,3) = val
            elseif(idum1.eq.2.and.idum2.eq.1) then
               aeval(2,1) = val
            elseif(idum1.eq.2.and.idum2.eq.2) then
               aeval(2,2) = val
            elseif(idum1.eq.2.and.idum2.eq.3) then
               aeval(2,3) = val
            elseif(idum1.eq.3.and.idum2.eq.1) then
               aeval(3,1) = val
            elseif(idum1.eq.3.and.idum2.eq.2) then
               aeval(3,2) = val
            elseif(idum1.eq.3.and.idum2.eq.3) then
               aeval(3,3) = val
            endif

         elseif(line1.eq.'#') then
            goto 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done = .true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end

c -------------------------------------------------------------------- c

      subroutine SLHA_READ_VCKMIN_HDEC(ninlha,vckmval,done)

      implicit double precision (a-h,m,o-z)
      double precision vckmval(4)
      character line1*1
      logical done

      done=.false.

      do i=1,4,1
         vckmval(i) = 0.D0
      end do

      do i=1,200,1
         read(ninlha,'(a1)',end=9900) line1

c -- decide what it is and read the line if anything of interest --
         if (line1.eq.' ') then
            backspace ninlha
            read(ninlha,*) idum,val

c -- lambda
            if(idum.eq.1) then
               vckmval(1) = val
c -- A
            elseif(idum.eq.2) then
               vckmval(2) = val
c -- rhobar
            elseif(idum.eq.3) then
               vckmval(3) = val
c -- etabar
            elseif(idum.eq.4) then
               vckmval(4) = val
            endif
            
         elseif(line1.eq.'#') then
            go to 1111
         elseif(line1.eq.'b'.or.line1.eq.'B'.or.line1.eq.'d'.or.line1.eq
     ..'D') then
            backspace ninlha
            done =.true.
            return
         endif

 1111    continue
      end do

 9900 print*,'SLHA_read_leshouches: end of file'
      done = .true.

      end


      subroutine readQval(ninlha,Qval)
      character buff*100
      integer ninlha,i
      real*8 Qval
      backspace ninlha      
      read(ninlha,fmt='(A100)') buff
      do i=1,100 
       if(buff(i:i).eq.'=') then 
            read(buff(i+1:100),*) Qval
            return 
      endif
      enddo
      end
