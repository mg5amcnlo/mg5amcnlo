      subroutine decay_event(P5,wgt,ni,ic)
c********************************************************************
c     Decay particle(s) in the event 
c
c     input: P(0,MaxParticles) :four momenta
c            wgt               :event weight
c            ni                :number of particle in the event 
c                               before the decay
c            ic(6,MaxParticles):particle labels       
c
c     output:P(0,MaxParticles) :four momenta
c            wgt               :event weight
c            ic(6,MaxParticles):particle labels       
c
c     which particle has to be decayed is passed through
c     a common block DEC_ID in decay.inc
c
c********************************************************************
      implicit none
      include 'decay.inc'
c
c     Arguments
c      
      integer ni, ic(7,MaxParticles)
      double precision P5(0:4,MaxParticles),wgt
c
c     Local
c
      integer i,k,jj, jhel(MaxDecPart), idecay, idec(MaxParticles),j, im
      integer JD(MaxDecPart),ICOL(2,MaxDecPart)
      integer maxcol
      double precision pcms(0:3), pd(0:3,MaxDecPart),pswgt
      double precision p(0:3,MaxParticles)
      double precision totwidth, dwgt, goal_wgt
      double precision weight,r
      logical done
      real wgt_vegas
c
c     External
c
      real xran1
      integer iseed
      data iseed/1/  !no effect if already called
c      
c     Global
c
      data s_unw,s_wei,s_ove/3*0d0/
      data n_wei,n_ove/2*0/
c
      include 'coupl.inc'
      include 'calc_values.inc'
c
c     first map the momenta into objects 0:3
c
      do i=1,MaxParticles
         do j=0,3
            p(j,i)=p5(j,i)
         enddo
      enddo

c
c--   First find out how many particles there are in the 
c     event to decay
c
      call find_particle(ip,ni,ic,k,idec)         
      if (k.eq.0) return !no particle to decay
c
c--   Loop over particles to decay
c
      do jj=1,k
         idecay=idec(jj)
c--   find the largest label present for the color flow
         maxcol=max(ic(4,1),ic(5,1))
         do i=2,ni
            maxcol=max(maxcol,ic(4,i),ic(5,i))
         enddo   
         cindex=maxcol+1        !available color index
c--   setup color of the decay particle
         icol(1,1) =  ic(4,idecay) ! color      of particle
         icol(2,1) =  ic(5,idecay) ! anti-color of particle
c--   Boost to the momentum of particle to the CMS frame 
         do i=0,3
            pcms(i) = -p(i,1) - p(i,2)
         enddo
         pcms(0)=-pcms(0)
         call boostx(p(0,idecay),pcms,pd(0,1))
         jhel(1) =  ic(7,idecay) ! helicity   of particle 
c--   Unweigthed decay
         done=.false.
         do while (.not. done)
            CALL GET_X(X,WGT_VEGAS)
            CALL EVENT(PD(0,1),JHEL(1),JD(1),ICOL,WEIGHT)
            weight=weight*real(wgt_vegas)
            n_wei=n_wei+1
            s_wei=s_wei+weight
            r = xran1(iseed)
            if(weight / r .gt. mxwgt) then
               done=.true.         
               s_unw=s_unw+mxwgt
               if (weight .gt. mxwgt) then
                  s_ove = s_ove-mxwgt+weight
                  n_ove = n_ove + 1
               endif
            endif
         enddo
c--   Multiply by the branching ratio
         wgt = wgt * bratio     
c--   Boost back to lab frame and put momentum into P(0,x)
         do i=1,3
            pcms(i)=-pcms(i)
         enddo
         do i=2,MaxDecPart      !loop over decay products
            call boostx(pd(0,i),pcms,p(0,i-1+ni))
         enddo
c--   Set the new particle information
c--   The info on the decayed particles are kept intact but 
c--   status changed to decayed
         do i=1,ND
            ic(1,ni+i)=jd(i+1)  !particle's IDs
            ic(2,ni+i)=idecay   !Mother info
            ic(3,ni+i)=idecay   !Mother info
            ic(4,ni+i)=icol(1,i+1) !     color info
            ic(5,ni+i)=icol(2,i+1) !anti-color info
            ic(6,ni+i)=+1       !Final Particle
            ic(7,ni+i)=jhel(1+i) !Helicity info
         enddo
         ni = ni + nd           !Keep intermediate state decaying particle
         ic(6,idecay)=+2        !This is decayed particle      
c--   end loop over particles to decay
      enddo                     
c
c     finally map the P back to 0:4
c
      do i=1,ni
         do j=0,3
            p5(j,i)=p(j,i)
         enddo
c         p5(4,i)=pdgmass(ic(1,i))
      enddo

      do i=ni-nd,ni
          p5(4,i)=pdgmass(ic(1,i))
      enddo

      return
      end


      SUBROUTINE EVENT(PD,JHEL,JD,ICOL,DWGT)
c**********************************************************    
c     Calculation of the decay event quantities
c
c     input: pd(0:3,1) decay particle momentum
c
c     ouput: pd(0:3,MaxDecPart) momenta
c            jhel(MaxDecPart)   helicities
c            jd(MaxDecPart)     particle id's
c            icol(2,MaxDecPart) color labels
c            DWGT               pswgt*emmesq*factor
c
c**********************************************************
      implicit none
      include 'decay.inc'
c
c     Arguments
c
      real*8 pd(0:3,MaxDecPart)
      integer jhel(MaxDecPart),JD(MaxDecPart),ICOL(2,MaxDecPart)
      real*8 dwgt
c
c     Global
c
      include 'coupl.inc'
      include 'calc_values.inc'
c
c     local
c
      real*8 pswgt,emmesq,fudge,factor
      real*8 aa,aa1,aa2,aa3
      integer isign,jl,jvl,i,j
      logical hadro_dec
      integer multi_decay
      real*8 xprob1,xprob2
c
c     EXTERNAL
c
      integer get_hel
      real xran1
      integer iseed
      data iseed/1/ !no effect if already called

C------
C START
C------

      DWGT   =0d0
      emmesq =0d0
      isign  =ip/abs(ip)
      aa     =0d0
      aa1    =0d0
      aa2    =0d0
      do i=2,MaxDecPart
         icol(1,i)=0
         icol(2,i)=0
      enddo
      
c      write(*,*) 'from event: imode,ip',imode,ip

      If(abs(ip).eq.6) then
*-----------------------------------------------------
*     top decays
*-----------------------------------------------------

         
      If(imode.eq.1) then 
*------------------------
*     t  -> b  w+
*     t~ -> b~ w-
*------------------------

c--   masses
      M1=tmass
      M2=bmass
      M3=wmass
c--   id's
      jd(2)=isign*5  !b  or b~
      jd(3)=isign*24 !w+ or w-
c--   color
      icol(1,2)=icol(1,1)  !     color of the top
      icol(2,2)=icol(2,1)  !anti-color of the top
c--   couplings
      GXX(1)=gwf(1)
      GXX(2)=gwf(2)
c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*6d0    
c--   helicities 
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  get_hel(xran1(iseed),3)
c--   phase space
      call phasespace(pd,pswgt)     
      if(pswgt.eq.0d0) return
c--   matrix element
      if(isign.eq. 1) call emme_ffv(pd,jhel,emmesq)      
      if(isign.eq.-1) call emme_fxfv(pd,jhel,emmesq)      
c--   weight
      dwgt=pswgt*emmesq*factor
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)

      return
      endif !end imode=1

      if(imode.ge.2.and.imode.le.8) then 
*------------------------
*     t  -> b  vl l+  ,jj
*     t~ -> b~ l  vl~ ,jj
*------------------------

c--   masses
      M1=tmass
      M2=bmass
      M3=0d0
      M4=0d0
      MV=WMASS
      GV=WWIDTH
c--   color
      icol(1,2)=icol(1,1)  !     color of the top
      icol(2,2)=icol(2,1)  !anti-color of the top
c--   id's
      if(isign.eq.1) then       !t
         jl=4
         jvl=3
      else                      !t~
         jl=3
         jvl=4
      endif   
      jd(2)=isign*5             !b  or b~
c--   rnd number 
      if(imode.ge.5) aa =dble(xran1(iseed))
      if(imode.eq.8) aa1=dble(xran1(iseed))
c
c     various cases imode=2,3,4,5,6,7,8
c
      if(imode.eq.2) then
*-------------------
*     t  -> b  ve e+
*-------------------
         jd(jvl)=isign*12       !ve or ve~
         jd(jl) =isign*(-11)    !e+ or e-

      elseif(imode.eq.3) then   
*--------------------
*     t  -> b  vm mu+
*--------------------
         jd(jvl)=isign*14       !vm or vm~
         jd(jl) =isign*(-13)    !mu+ or mu-

      elseif(imode.eq.4) then
*-------------------
*     t  -> b  vt ta+
*-------------------
         if(isign.eq.1) then    !t
            M3=0D0
            M4=lmass
         else                   !t~
            M3=lmass
            M4=0d0
         endif   
         jd(jvl)=isign*16       !vt or vt~
         jd(jl) =isign*(-15)    !ta+ or ta-

      elseif(imode.eq.5) then
*-------------------
*     t -> b  vl l+  (e+mu) 
*-------------------
        if(aa.lt.Half) then
            jd(jvl)=isign*12    !ve or ve~
            jd(jl) =isign*(-11) !e+ or e-
         else
            jd(jvl)=isign*14    !vm or vm~
            jd(jl) =isign*(-13) !mu+ or mu-
         endif

      elseif(imode.eq.6) then
*-------------------
*     t -> b  vl l+  (e+mu+ta) 
*-------------------
         if(aa.lt.one/Three) then
            jd(jvl)=isign*12    !ve or ve~
            jd(jl) =isign*(-11) !e+ or e-
         elseif(aa.lt.two/three) then
            jd(jvl)=isign*14    !vm or vm~
            jd(jl) =isign*(-13) !mu+ or mu-
         else
            if(isign.eq.1) then !t
               M3=0D0
               M4=lmass
            else                !t~
               M3=lmass
               M4=0d0
            endif   
            jd(jvl)=isign*16    !vt or vt~
            jd(jl) =isign*(-15) !ta+ or ta-
         endif

      elseif(imode.eq.7) then
*-------------------
*     t -> b  j j   (ud,cs) 
*-------------------
c--   color
         icol(1,3)=cindex   !position 3 is always a particle  
         icol(2,3)=0
         icol(1,4)=0        !position 4 is always a anti-particle
         icol(2,4)=cindex

         if(aa.lt..5d0) then
            jd(jvl)=isign*2    !u  or u~
            jd(jl) =isign*(-1) !d~ or d
         else
            if(isign.eq.1) then !t
               M3=cmass
               M4=0d0
            else                !t~
               M3=0d0
               M4=cmass
            endif   
            jd(jvl)=isign*4     !c  or c~
            jd(jl) =isign*(-3)  !s~ or s
         endif

      elseif(imode.eq.8) then
*-------------------
*     t -> b  anything
*-------------------
c
c     first decide if W decays leptonically or hadronically
c
         if(aa1.lt.3d0*br_w_lv) then  !leptonic decay
            hadro_dec=.false.
            if(aa.lt.one/Three) then
               jd(jvl)=isign*12 !ve or ve~
               jd(jl) =isign*(-11) !e+ or e-
            elseif(aa.lt.two/three) then
               jd(jvl)=isign*14 !vm or vm~
               jd(jl) =isign*(-13) !mu+ or mu-
            else
               if(isign.eq.1) then !t
                  M3=0D0
                  M4=lmass
               else             !t~
                  M3=lmass
                  M4=0d0
               endif   
               jd(jvl)=isign*16 !vt or vt~
               jd(jl) =isign*(-15) !ta+ or ta-
            endif
            
         else                   ! hadronic decay
            hadro_dec=.true.
c--   color
            icol(1,3)=cindex    !position 3 is always a particle  
            icol(2,3)=0
            icol(1,4)=0         !position 4 is always a anti-particle
            icol(2,4)=cindex
            
            if(aa.lt..5d0) then
               jd(jvl)=isign*2  !u  or u~
               jd(jl) =isign*(-1) !d~ or d
            else
               if(isign.eq.1) then !t
                  M3=cmass
                  M4=0d0
               else             !t~
                  M3=0d0
                  M4=cmass
               endif   
               jd(jvl)=isign*4  !c  or c~
               jd(jl) =isign*(-3) !s~ or s
            endif
         endif

      endif !imode(from 2 to 8)
      endif !imode
 
c--   couplings
      GXX(1)=gwf(1)
      GXX(2)=gwf(2)
c--   color*(bose factor)*number of helicities
      factor=(3d0/3d0)*1d0*4d0    
      if(imode.eq.7) factor=factor*3d0 !quark color in jj
      if(imode.eq.8.and.hadro_dec) factor=factor*3d0 !quark color in jj
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  get_hel(xran1(iseed),2)
      jhel(4) =  -jhel(3)
c--   phase space
      if(GV.eq.0d0) call error_trap("GV")
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      if(isign.eq. 1) call emme_f3f(pd,jhel,emmesq) !t
      if(isign.eq.-1) call emme_fx3f(pd,jhel,emmesq) !t~
c--   weight
      dwgt=pswgt*emmesq*factor
      if(imode.eq.5) dwgt=dwgt*2d0 !l=e,mu
      if(imode.eq.6) dwgt=dwgt*3d0 !l=e,mu,tau
      if(imode.eq.7) dwgt=dwgt*2d0 !jj=ud,cs      
      if(imode.eq.8) then
         if(hadro_dec) then
            dwgt=dwgt*2d0/br_w_jj  !jj=ud,cs      
         else
            dwgt=dwgt*3d0/(3d0*br_w_lv)  !l=e,mu,tau
         endif
      endif
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return
      write(*,*) 'non-existing imode for top decays'
      endif ! top decays



      If(abs(ip).eq.15) then
*-----------------------------------------------------
*     tau decays
*-----------------------------------------------------


      if(imode.le.3) then 
*------------------------
*     ta -> vt vl l
*------------------------

c--   masses
      M1=lmass
      M2=0d0
      M3=0d0
      M4=0d0
      MV=WMASS
      GV=WWIDTH
c--   id's
      if(isign.eq.1) then       !tau-
         jl=3
         jvl=4
      else                      !tau+
         jl=4
         jvl=3
      endif   
      jd(2)=isign*16   !vt or vt~
c--   couplings
      GXX(1)=gwf(1)
      GXX(2)=gwf(2)
c--   rnd number 
      if(imode.eq.3) aa=dble(xran1(iseed))

      if    (imode.eq.1) then
*-------------------
*     ta  -> vt  e  ve
*-------------------
         jd(jvl)=isign*(-12)   !ve~ or ve
         jd(jl) =isign*(11)    !e-  or e+

      elseif(imode.eq.2) then   
*-----------------------
*     tau  -> vt  mu vm
*-----------------------
         jd(jvl)=isign*(-14)    !vm~ or vm
         jd(jl) =isign*( 13)    !mu- or mu+

      elseif(imode.eq.3) then
*----------------------
*     tau -> vt  l vl  (e+mu) 
*----------------------
        if(aa.lt.Half) then
           jd(jvl)=isign*(-12)  !ve~ or ve
           jd(jl) =isign*(11)   !e-  or e+
         else
            jd(jvl)=isign*(-14) !vm~ or vm
            jd(jl) =isign*( 13) !mu- or mu+
         endif

      endif   

c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*1d0    
c--   helicities: by definition of 3rd and 4th particle 
      jhel(2) =  -isign
      jhel(3) =  -1
      jhel(4) =   1
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      if(isign.eq. 1) call emme_f3f(pd,jhel,emmesq)  !tau-
      if(isign.eq.-1) call emme_fx3f(pd,jhel,emmesq) !tau+
c--   weight
      dwgt=pswgt*emmesq*factor
      if(imode.eq.3) dwgt=dwgt*2d0 !l=e,mu
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return


      elseif(imode.eq.4) then 
*------------------------
*     tau -> vt pi
*------------------------

c--   masses
      M1=lmass
      M2=0d0
      M3=0.1395d0
c--   id's
      jd(2)=isign*16            !vt or vt~
      jd(3)=isign*(-211)        !pi-  pi+
c--   couplings
      GXX(1)=gwf(1)
      GXX(2)=gwf(2)
c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*2d0    
c--   fudge to normalize
      fudge=lwidth*br_ta_pi/0.00373
      factor=factor*fudge
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  0              !not used
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      if(isign.eq. 1) call emme_ffs (pd,jhel,emmesq)
      if(isign.eq.-1) call emme_fxfs(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseIf(imode.eq.5) then 
*------------------------
*     tau -> vt rho
*------------------------

c--   masses
      M1=lmass
      M2=0d0
      M3=0.770d0
c--   id's
      jd(2)=isign*16            !vt or vt~
      jd(3)=isign*(-213)        !rho- rho +
c--   couplings
      GXX(1)=gwf(1)
      GXX(2)=gwf(2)
c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*6d0    
c--   fudge to normalize
      fudge=lwidth*br_ta_ro/0.0182
      factor=factor*fudge
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  get_hel(xran1(iseed),3)
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      if(isign.eq. 1) call emme_ffv (pd,jhel,emmesq)
      if(isign.eq.-1) call emme_fxfv(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return
      endif  !imodes
      write(*,*) 'non-existing imode for tau decays'
      endif !tau



      If(ip.eq.23) then
*-----------------------------------------------------
*     Z decays
*-----------------------------------------------------

c--   masses
      M1=zmass
      M2=0D0
      M3=0D0
c--   couplings
      GXX(1)=gzl(1)
      GXX(2)=gzl(2)
c--   rnd number for flavour sums
      aa =dble(xran1(iseed))
      aa1=dble(xran1(iseed))


      if    (imode.eq.1) then
*-------------------
*     z->e- e+
*-------------------
         jd(2) = 11   
         jd(3) =-11

      elseif(imode.eq.2) then   
*--------------------
*     z->mu- mu+
*--------------------
         jd(2) = 13   
         jd(3) =-13

      elseif(imode.eq.3) then
*-------------------
*     z->ta- ta+
*-------------------
         M2=lmass
         jd(2) = 15   
         jd(3) =-15

      elseif(imode.eq.4) then
*-------------------
*     z->e- e+,mu-mu+
*-------------------
         if(aa.lt.Half) then
            jd(2) = 11   
            jd(3) =-11
         else
            jd(2) = 13   
            jd(3) =-13
         endif

      elseif(imode.eq.5) then
*-------------------
*     z->e- e+,mu-mu+,ta-ta+
*-------------------
         if(aa.lt.one/Three) then
            jd(2) = 11   
            jd(3) =-11
         elseif(aa.lt.two/three) then
            jd(2) = 13   
            jd(3) =-13
         else
            M2=lmass
            jd(2) = 15   
            jd(3) =-15
         endif

      elseif(imode.eq.6) then
*------------------------
*     z->vl vl~
*------------------------

c--   couplings
         GXX(1)=gzn(1)
         GXX(2)=gzn(2)

         if(aa.lt.one/Three) then
            jd(2) = 12   
            jd(3) =-12
         elseif(aa.lt.two/three) then
            jd(2) = 14   
            jd(3) =-14
         else
            jd(2) = 16   
            jd(3) =-16
         endif


      elseif(imode.eq.7) then
*------------------------
*     z-> b b~
*------------------------
c--   color
         icol(1,2)=cindex   !position 2 is always a particle  
         icol(2,2)=0
         icol(1,3)=0        !position 3 is always a anti-particle
         icol(2,3)=cindex

c--   couplings
         GXX(1)=gzd(1)
         GXX(2)=gzd(2)
         M2=bmass
         jd(2) = 5  
         jd(3) =-5

      elseif(imode.eq.8) then
*------------------------
*     z-> c c~
*------------------------
c--   color
         icol(1,2)=cindex   !position 2 is always a particle  
         icol(2,2)=0
         icol(1,3)=0        !position 3 is always a anti-particle
         icol(2,3)=cindex

c--   couplings
         GXX(1)=gzu(1)
         GXX(2)=gzu(2)
         M2=cmass
         jd(2) = 4  
         jd(3) =-4


      elseif(imode.eq.9) then 
*------------------------
*     z-> u,d,c,s
*------------------------
c--   color
         icol(1,2)=cindex   !position 2 is always a particle  
         icol(2,2)=0
         icol(1,3)=0        !position 3 is always a anti-particle
         icol(2,3)=cindex

c-- probability of a uc or ds decay
         
         xprob1=w_z_uu/(w_z_dd + w_z_uu)

         if(aa1.lt.xprob1) then ! decay into ups
            multi_decay=1
            
            if(aa.lt. .5d0) then !u
               M2=0d0
               jd(2) = 2  
               jd(3) =-2
               GXX(1)=gzu(1)
               GXX(2)=gzu(2)
            else                !c
               jd(2) = 4  
               jd(3) =-4
               M2=cmass
               GXX(1)=gzu(1)
               GXX(2)=gzu(2)
            endif

         else                   !decay into downs
            multi_decay=2
            if(aa.lt..5d0) then !d
               M2=0d0
               jd(2) = 1  
               jd(3) =-1
               GXX(1)=gzd(1)
               GXX(2)=gzd(2)
            else                !s
               jd(2) = 3  
               jd(3) =-3
               M2=0d0
               GXX(1)=gzd(1)
               GXX(2)=gzd(2)
            endif
         
         endif                  !which decay: in ups or downs

      elseif(imode.eq.10) then 
*------------------------
*     z-> u,d,c,s,b
*------------------------
c--   color
         icol(1,2)=cindex   !position 2 is always a particle  
         icol(2,2)=0
         icol(1,3)=0        !position 3 is always a anti-particle
         icol(2,3)=cindex

c-- probability of a uc or ds decay
         
         xprob1=2d0*w_z_uu/(2d0*w_z_dd + 2d0*w_z_uu+w_z_bb)
         xprob2=xprob1+2d0*w_z_dd/(2d0*w_z_dd + 2d0*w_z_uu+w_z_bb)
         
         if(aa1.lt.xprob1) then ! decay into ups
            multi_decay=1
            if(aa.lt.0.5d0) then !u
               jd(2) = 2  
               jd(3) =-2
               M2=0d0
               GXX(1)=gzu(1)
               GXX(2)=gzu(2)
            else                !c 
               jd(2) = 4  
               jd(3) =-4
               M2=cmass
               GXX(1)=gzu(1)
               GXX(2)=gzu(2)
            endif
         
         elseif(aa1.lt.xprob2) then ! decay into downs
            multi_decay=2
            if(aa.lt. .5d0) then !d
               M2=0d0
               jd(2) = 1  
               jd(3) =-1
               GXX(1)=gzd(1)
               GXX(2)=gzd(2)
            else                !s      
               jd(2) = 3  
               jd(3) =-3
               M2=0d0
               GXX(1)=gzd(1)
               GXX(2)=gzd(2)
            endif
         else                   ! decay into b's
            multi_decay=3
            jd(2) = 5  
            jd(3) =-5
            M2=bmass
            GXX(1)=gzd(1)
            GXX(2)=gzd(2)
         endif   
         
      else
         write(*,*) 'non-existing imode for Z decays'
      endif                     ! imode
      
      M3=M2
c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*4d0    
      if(imode.ge.7) factor=factor*3d0 !quark color in jj
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  get_hel(xran1(iseed),2)
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_vff(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
      if(imode.eq.4)  dwgt=dwgt*2d0 !l=e,mu
      if(imode.eq.5)  dwgt=dwgt*3d0 !l=e,mu,tau
      if(imode.eq.6)  dwgt=dwgt*3d0 !vl=ve,vm,vt
c
      if(imode.eq.9) then       !j=u,d,c,s
         if(multi_decay.eq.1) then
            dwgt=dwgt*2d0/xprob1 !j=u,c 
         else
            dwgt=dwgt*2d0/(1.-xprob1) !j=d,s
         endif
      endif   
c     
      if(imode.eq.10) then      !j=u,d,c,s,b
         if(multi_decay.eq.1) then
            dwgt=dwgt*2d0/xprob1 !j=u,c 
         elseif(multi_decay.eq.2) then 
            dwgt=dwgt*2d0/(xprob2-xprob1) !j=d,s
         elseif(multi_decay.eq.3) then 
            dwgt=dwgt/(1.-xprob2) !j=b
         endif 
      endif   

      return
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      endif ! Z


      If(abs(ip).eq.24) then
*-----------------------------------------------------
*     W decays
*-----------------------------------------------------

c--   masses
      M1=wmass
      M2=0d0
      M3=0d0
c--   couplings
      GXX(1)=gwf(1)
      GXX(2)=gwf(2)
c--   id's
      if(isign.eq.1) then       !W+
         jl=3
         jvl=2
      else                      !W-
         jl=2
         jvl=3
      endif   
c--   rnd number 
      if(imode.ge.4) aa =dble(xran1(iseed))
      if(imode.eq.7) aa1=dble(xran1(iseed))


      if(imode.eq.1) then 
*------------------------
*     w-> e ve
*------------------------
         jd(jvl)=isign*12       !ve or ve~
         jd(jl) =isign*(-11)    !e+ or e-

      elseif(imode.eq.2) then   
*--------------------
*     w -> mu vm
*--------------------
         jd(jvl)=isign*14       !vm or vm~
         jd(jl) =isign*(-13)    !mu+ or mu-

      elseif(imode.eq.3) then
*-------------------
*     w -> ta vt
*-------------------
         if(isign.eq.1) then    !t
            M2=0D0
            M3=lmass
         else                   !t~
            M2=lmass
            M3=0d0
         endif   
         jd(jvl)=isign*16       !vt or vt~
         jd(jl) =isign*(-15)    !ta+ or ta-

      elseif(imode.eq.4) then
*-------------------
*     w -> vl l+  (e+mu) 
*-------------------
        if(aa.lt.Half) then
            jd(jvl)=isign*12    !ve or ve~
            jd(jl) =isign*(-11) !e+ or e-
         else
            jd(jvl)=isign*14    !vm or vm~
            jd(jl) =isign*(-13) !mu+ or mu-
         endif

      elseif(imode.eq.5) then
*-------------------
*     w -> vl l+  (e+mu+ta) 
*-------------------
         if(aa.lt.one/Three) then
            jd(jvl)=isign*12    !ve or ve~
            jd(jl) =isign*(-11) !e+ or e-
         elseif(aa.lt.two/three) then
            jd(jvl)=isign*14    !vm or vm~
            jd(jl) =isign*(-13) !mu+ or mu-
         else
            if(isign.eq.1) then !t
               M2=0D0
               M3=lmass
            else                !t~
               M2=lmass
               M3=0d0
            endif   
            jd(jvl)=isign*16    !vt or vt~
            jd(jl) =isign*(-15) !ta+ or ta-
         endif

      elseif(imode.eq.6) then
*-------------------
*     w -> j j   (ud,cs) 
*-------------------
c--   color
         icol(1,2)=cindex   !position 2 is always a particle  
         icol(2,2)=0
         icol(1,3)=0        !position 3 is always a anti-particle
         icol(2,3)=cindex

         if(aa.lt..5d0) then
            jd(jvl)=isign*2    !u  or u~
            jd(jl) =isign*(-1) !d~ or d
         else
            if(isign.eq.1) then !t
               M2=cmass
               M3=0d0
            else                !t~
               M2=0d0
               M3=cmass
            endif   
            jd(jvl)=isign*4     !c  or c~
            jd(jl) =isign*(-3)  !s~ or s
         endif

      elseif(imode.eq.7) then
*-------------------
*     w -> anything
*-------------------
c     
c     first decide if W decays leptonically or hadronically
c     
         if(aa1.lt.3d0*br_w_lv) then !leptonic decay
            hadro_dec=.false.
            if(aa.lt.one/Three) then
               jd(jvl)=isign*12 !ve or ve~
               jd(jl) =isign*(-11) !e+ or e-
            elseif(aa.lt.two/three) then
               jd(jvl)=isign*14 !vm or vm~
               jd(jl) =isign*(-13) !mu+ or mu-
            else
               if(isign.eq.1) then !t
                  M2=0D0
                  M3=lmass
               else             !t~
                  M2=lmass
                  M3=0d0
               endif   
               jd(jvl)=isign*16 !vt or vt~
               jd(jl) =isign*(-15) !ta+ or ta-
            endif
         else
            hadro_dec=.true.    !hadronic decay
c--   color
            icol(1,2)=cindex    !position 2 is always a particle  
            icol(2,2)=0
            icol(1,3)=0         !position 3 is always a anti-particle
            icol(2,3)=cindex
            
            if(aa.lt..5d0) then
               jd(jvl)=isign*2  !u  or u~
               jd(jl) =isign*(-1) !d~ or d
            else
               if(isign.eq.1) then !t
                  M2=cmass
                  M3=0d0
               else             !t~
                  M2=0d0
                  M3=cmass
               endif   
               jd(jvl)=isign*4  !c  or c~
               jd(jl) =isign*(-3) !s~ or s
            endif            
         endif
      endif
c
c     done with the modes: now I start the calculation
c      
     
c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*2d0    
      if(imode.eq.6) factor=factor*3d0 !quark color in jj
      if(imode.eq.7.and.hadro_dec) factor=factor*3d0 !quark color in jj
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  -jhel(2)
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_vff(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
      if(imode.eq.4)  dwgt=dwgt*2d0 !l=e,mu
      if(imode.eq.5)  dwgt=dwgt*3d0 !l=e,mu,tau
      if(imode.eq.6)  dwgt=dwgt*2d0 !j=ud+cs
      if(imode.eq.7) then
         if(hadro_dec) then
            dwgt=dwgt*2d0/br_w_jj        !jj=ud,cs      
         else
            dwgt=dwgt*3d0/(3d0*br_w_lv)  !l=e,mu,tau
         endif
      endif

c--   check that dwgt is a reasonable number
      call check_nan(dwgt)

      return
      endif ! w


      If(ip.eq.25) then
*-----------------------------------------------------
*     Higgs decays
*-----------------------------------------------------

      If(imode.eq.1) then 
*------------------------
*     h->b b~
*------------------------
c--   masses
      M1=hmass
      M2=bmass
      M3=bmass
c--   id's
      jd(2) = 5   
      jd(3) =-5
c--   color
      icol(1,2)=cindex          !position 2 is always a particle  
      icol(2,2)=0
      icol(1,3)=0               !position 3 is always a anti-particle
      icol(2,3)=cindex
c--   couplings
      GXX(1)=ghbot(1)
      GXX(2)=ghbot(2)
c--   color*(bose factor)*number of helicities
      factor=3d0*1d0*2d0    
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  jhel(2)
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_hff(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.2) then 
*------------------------
*     h->ta- ta+
*------------------------
c--   masses
      M1=hmass
      M2=lmass
      M3=lmass
c--   id's
      jd(2) = 15   
      jd(3) =-15
c--   couplings
      GXX(1)=ghtau(1)
      GXX(2)=ghtau(2)
c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*2d0    
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  jhel(2)
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_hff(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)

      return

      elseif(imode.eq.3) then 
*------------------------
*     h->mu- mu+
*------------------------
c--   masses
      M1=hmass
      M2=0d0
      M3=0d0
c--   id's
      jd(2) = 13   
      jd(3) =-13
c--   couplings
      GXX(1)=ghtau(1)/lmass*0.105658389d0
      GXX(2)=ghtau(2)/lmass*0.105658389d0
c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*2d0    
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  jhel(2)
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_hff(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.4) then 
*------------------------
*     h-> c c~
*------------------------
c--   masses
      M1=hmass
      M2=cmass
      M3=cmass
c--   id's
      jd(2) = 4   
      jd(3) =-4
c--   color
      icol(1,2)=cindex          !position 2 is always a particle  
      icol(2,2)=0
      icol(1,3)=0               !position 3 is always a anti-particle
      icol(2,3)=cindex
c--   couplings
      GXX(1)=ghcha(1)
      GXX(2)=ghcha(2)
c--   color*(bose factor)*number of helicities
      factor=3d0*1d0*4d0    
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  get_hel(xran1(iseed),2)
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_hff(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.5) then 
*------------------------
*     h-> t t~
*------------------------
c--   masses
      M1=hmass
      M2=tmass
      M3=tmass
c--   id's
      jd(2) = 6   
      jd(3) =-6
c--   couplings
      GXX(1)=ghtop(1)
      GXX(2)=ghtop(2)
c--   color*(bose factor)*number of helicities
      factor=3d0*1d0*4d0    
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  get_hel(xran1(iseed),2)
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_hff(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.6) then 
*------------------------
*     h-> g g 
*------------------------
c--   masses
      M1=hmass
      M2=0d0
      M3=0d0
c--   id's
      jd(2) = 21   
      jd(3) = 21
c--   color
      icol(1,2)=cindex 
      icol(2,2)=cindex+1
      icol(1,3)=cindex+1      
      icol(2,3)=cindex
c--   couplings
      GX=cmplx(1d0)
c--   color*(bose factor)*number of helicities
      factor=8d0*.5d0*4d0 
c--   fudge factor for normalization
      factor=factor*hmass*2d0*pi*(SMBRG*SMWDTH)
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  get_hel(xran1(iseed),2)
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_hvv(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.7) then 
*------------------------
*     h-> a a 
*------------------------

c--   masses
      M1=hmass
      M2=0d0
      M3=0d0
c--   id's
      jd(2) = 22   
      jd(3) = 22
c--   couplings
      GX=cmplx(1d0)
c--   color*(bose factor)*number of helicities
      factor=1d0*.5d0*4d0    
c--   fudge factor for normalization
      factor=factor*hmass*16d0*pi*(SMBRGA*SMWDTH)
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  get_hel(xran1(iseed),2)
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_hvv(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.8) then 
*------------------------
*     h-> z a   
*------------------------
c--   masses
      M1=hmass
      M2=zmass
      M3=0d0
c--   id's
      jd(2) = 23   
      jd(3) = 22
c--   couplings
      GX=cmplx(1d0)
c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*6d0    
c--   fudge factor for normalization
      factor=factor*SMBRZGA*SMWDTH*
     &8d0*pi*hmass/(1d0-(zmass/hmass)**2)
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),3)        !-1,0,1
      jhel(3) =  get_hel(xran1(iseed),2)
c--   phase space
      call twomom  (pd(0,1),pd(0,2),pd(0,3),pswgt)
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_hvv(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.9) then 
*------------------------
*     h-> w w
*------------------------
c--   masses
      M1=hmass
      M2=wmass
      M3=wmass
c--   id's
      jd(2) = 24   
      jd(3) =-24
c--   couplings
      GX=gwwh
c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*9d0    
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),3)
      jhel(3) =  get_hel(xran1(iseed),3)
      jhel(2) =  INT(3e0*xran1(iseed)) - 1
      jhel(3) =  INT(3e0*xran1(iseed)) - 1
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_hvv(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return


      elseif(imode.eq.10) then 
*--------------------------------
*     h -> w*  w -> l  vl  l' vl' (l,l'=e,mu)
*--------------------------------

c--   masses
      M1=hmass
      M2=0d0
      M3=0d0
      M4=0d0
      M5=0d0
      MV=WMASS
      GV=WWIDTH
c--   id's
      aa1=dble(xran1(iseed))
      aa2=dble(xran1(iseed))
c--   W*-
      if(aa1.lt.half) then
         jd(2) =  11            !e-  
         jd(3) = -12            !ve~ 
      else
         jd(2) =  13            !mu- 
         jd(3) = -14            !vm~ 
      endif
c--   W*+
      if(aa2.lt.half) then
         jd(4) =  12            !ve 
         jd(5) = -11            !e+ 
      else
         jd(4) =  14            !vm  
         jd(5) = -13            !mu+ 
      endif
c--   couplings
      GX     =gwwh
      GXX(1) =gwf(1)
      GXX(2) =gwf(2)
      GXX1(1)=gwf(1)
      GXX1(2)=gwf(2)
c--   color*(bose factor)*number of helicities
      factor=1d0*1d0 
c--   helicities
      jhel(2)=-1
      jhel(3)=1
      jhel(4)=-1
      jhel(5)=1
c--   phase space
      if(GV.eq.0d0) call error_trap("GV")
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_h4f(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   sum of flavours
      dwgt=dwgt*4d0
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.11) then 
*--------------------------------
*     h -> w*  w -> l  vl  l' vl' (l,l'=e,mu,ta)
*--------------------------------

c--   masses
      M1=hmass
      M2=0d0
      M3=0d0
      M4=0d0
      M5=0d0
      MV=WMASS
      GV=WWIDTH
c--   id's
      aa1=dble(xran1(iseed))
      aa2=dble(xran1(iseed))
c--   W*-
      if(aa1.lt.one/three) then
         jd(2) =  11            !e-  
         jd(3) = -12            !ve~ 
      elseif(aa1.lt.two/three) then
         jd(2) =  13            !mu- 
         jd(3) = -14            !vm~ 
      else
         M2    =  lmass
         jd(2) =  15            !ta- 
         jd(3) = -16            !vt~ 
      endif
c--   W*+
      if(aa2.lt.one/three) then
         jd(4) =  12            !ve 
         jd(5) = -11            !e+ 
      elseif(aa2.lt.two/three) then
         jd(4) =  14            !vm  
         jd(5) = -13            !mu+ 
      else
         M5    =  lmass
         jd(4) =  16            !vt 
         jd(5) = -15            !ta+ 
      endif

c--   couplings
      GX     =gwwh
      GXX(1) =gwf(1)
      GXX(2) =gwf(2)
      GXX1(1)=gwf(1)
      GXX1(2)=gwf(2)
c--   color*(bose factor)*number of helicities
      factor=1d0*1d0    
c--   helicities
      jhel(2)=-1
      jhel(3)=1
      jhel(4)=-1
      jhel(5)=1
c--   phase space
      if(GV.eq.0d0) call error_trap("GV")
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_h4f(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   sum of flavours
      dwgt=dwgt*9d0
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.12) then 
*--------------------------------
*     h -> w*  w -> j  j   l  vl  (jj=ud,cs;l=e,mu)
*--------------------------------

c--   masses
      M1=hmass
      M2=0d0
      M3=0d0
      M4=0d0
      M5=0d0
      MV=WMASS
      GV=WWIDTH
c--   color
      icol(1,2)=cindex          !position 2 is always a particle  
      icol(2,2)=0
      icol(1,3)=0               !position 3 is always a anti-particle
      icol(2,3)=cindex
c--   id's
      aa1=dble(xran1(iseed))
      aa2=dble(xran1(iseed))
      aa3=dble(xran1(iseed))

      if(aa1.lt.half) then
c--   W*-
         if(aa2.lt.half) then
            jd(2) =  1          !d 
            jd(3) = -2          !u~
         else
            M3    =  cmass
            jd(2) =  3          !s 
            jd(3) = -4          !c~ 
         endif         
c--   W*+
         if(aa3.lt.half) then
            jd(4) =  12         !ve 
            jd(5) = -11         !e+ 
         else
            jd(4) =  14         !vm  
            jd(5) = -13         !mu+ 
         endif

      else
c--   W*+
         if(aa2.lt.half) then
            jd(2) =  2          !u
            jd(3) = -1          !d~
         else
            M2    =  cmass
            jd(2) =  4          !c 
            jd(3) = -3          !s~ 
         endif         
c--   W*-
         if(aa3.lt.half) then
            jd(4) =  11         !e- 
            jd(5) = -12         !ve~ 
         else
            jd(4) =  13         !mu- 
            jd(5) = -14         !vm~ 
         endif
      endif

c--   couplings
      GX     =gwwh
      GXX(1) =gwf(1)
      GXX(2) =gwf(2)
      GXX1(1)=gwf(1)
      GXX1(2)=gwf(2)
c--   color*(bose factor)*number of helicities
      factor=3d0*1d0  
c--   helicities
      jhel(2)=-1
      jhel(3)=1
      jhel(4)=-1
      jhel(5)=1
c--   phase space
      if(GV.eq.0d0) call error_trap("GV")
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_h4f(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   sum of flavours
      dwgt=dwgt*4d0
c--   sum of W+ and W- possibilities
      dwgt=dwgt*2d0
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.13) then 
*--------------------------------
*     h -> w*  w -> j  j   l  vl  (jj=ud,cs;l=e,mu,ta)
*--------------------------------

c--   masses
      M1=hmass
      M2=0d0
      M3=0d0
      M4=0d0
      M5=0d0
      MV=WMASS
      GV=WWIDTH
c--   color
      icol(1,2)=cindex          !position 2 is always a particle  
      icol(2,2)=0
      icol(1,3)=0               !position 3 is always a anti-particle
      icol(2,3)=cindex
c--   id's
      aa1=dble(xran1(iseed))
      aa2=dble(xran1(iseed))
      aa3=dble(xran1(iseed))

      if(aa1.lt.half) then
c--   W*-
         if(aa2.lt.half) then
            jd(2) =  1          !d 
            jd(3) = -2          !u~
         else
            M3    =  cmass
            jd(2) =  3          !s 
            jd(3) = -4          !c~ 
         endif         
c--   W*+
         if(aa3.lt.one/three) then
            jd(4) =  12         !ve 
            jd(5) = -11         !e+ 
         elseif(aa2.lt.two/three) then
            jd(4) =  14         !vm  
            jd(5) = -13         !mu+ 
         else
            M5    =  lmass
            jd(4) =  16         !vt 
            jd(5) = -15         !ta+ 
         endif

      else
c--   W*+
         if(aa2.lt.half) then
            jd(2) =  2          !u
            jd(3) = -1          !d~
         else
            M2    =  cmass
            jd(2) =  4          !c 
            jd(3) = -3          !s~ 
         endif         
c--   W*-
         if(aa3.lt.one/three) then
            jd(4) =  11         !e-  
            jd(5) = -12         !ve~ 
         elseif(aa3.lt.two/three) then
            jd(4) =  13         !mu- 
            jd(5) = -14         !vm~ 
         else
            M4    =  lmass
            jd(4) =  15         !ta- 
            jd(5) = -16         !vt~ 
         endif
      endif

c--   couplings
      GX     =gwwh
      GXX(1) =gwf(1)
      GXX(2) =gwf(2)
      GXX1(1)=gwf(1)
      GXX1(2)=gwf(2)
c--   color*(bose factor)*number of helicities
      factor=3d0*1d0   
c--   helicities
      jhel(2)=-1
      jhel(3)=1
      jhel(4)=-1
      jhel(5)=1
c--   phase space
      if(GV.eq.0d0) call error_trap("GV")
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_h4f(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   sum of flavours
      dwgt=dwgt*6d0
c--   sum of W+ and W- possibilities
      dwgt=dwgt*2d0
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.14) then 
*------------------------
*     h-> z z 
*------------------------
c--   masses
      M1=hmass
      M2=zmass
      M3=zmass
c--   id's
      jd(2) = 23  
      jd(3) = 23
c--   couplings
      GX=gzzh
c--   color*(bose factor)*number of helicities
      factor=1d0*.5d0*9d0    
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),3)
      jhel(3) =  get_hel(xran1(iseed),3)
c--   phase space
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_hvv(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.15) then 
*---------------------------------
*     h -> z*  z -> l- l+  l-' l+'(l,l'=e,mu)
*---------------------------------

c--   masses
      M1=hmass
      M2=0d0
      M3=0d0
      M4=0d0
      M5=0d0
      MV=ZMASS
      GV=ZWIDTH
c--   id's
      aa1=dble(xran1(iseed))
      aa2=dble(xran1(iseed))
c     Z1
      if(aa1.lt.half) then
         jd(2) =  11            !e-  
         jd(3) = -11            !e+
      else
         jd(2) =  13            !mu- 
         jd(3) = -13            !mu+ 
      endif
c     Z2
      if(aa2.lt.half) then
         jd(4) =  11            !e- 
         jd(5) = -11            !e+ 
      else
         jd(4) =  13            !mu-  
         jd(5) = -13            !mu+ 
      endif
c--   couplings
      GX    =gzzh
      GXX(1)=gzl(1)
      GXX(2)=gzl(2)
      GXX1(1)=gzl(1)
      GXX1(2)=gzl(2)
c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*4d0/2d0   
c--   helicities
      jhel(2)=+1
      if(xran1(iseed) .gt. .5e0) jhel(2)=-1
      jhel(3)=-jhel(2)
      jhel(4)=+1
      if(xran1(iseed) .gt. .5e0) jhel(4)=-1
      jhel(5)=-jhel(4)
c--   phase space
      if(GV.eq.0d0) call error_trap("GV")
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_h4f(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   sum of flavours
      dwgt=dwgt*4d0
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return
      

      elseif(imode.eq.16) then 
*---------------------------------
*     h -> z*  z -> l- l+  l-' l+'(l,l'=e,mu,ta)
*---------------------------------

c--   masses
      M1=hmass
      M2=0d0
      M3=0d0
      M4=0d0
      M5=0d0
      MV=ZMASS
      GV=ZWIDTH
c--   id's
      aa1=dble(xran1(iseed))
      aa2=dble(xran1(iseed))
c     Z
      if(aa1.lt.one/three) then
         jd(2) =  11            !e-  
         jd(3) = -11            !e+
      elseif(aa1.lt.two/three) then
         jd(2) =  13            !mu- 
         jd(3) = -13            !mu+ 
      else
         M2    =  lmass
         M3    =  lmass
         jd(2) =  15            !ta- 
         jd(3) = -15            !ta+ 
      endif
c     Z
      if(aa2.lt.one/three) then
         jd(4) =  11            !e-
         jd(5) = -11            !e+ 
      elseif(aa2.lt.two/three) then
         jd(4) =  13            !mu-  
         jd(5) = -13            !mu+ 
      else
         M4    =  lmass
         M5    =  lmass
         jd(4) =  15            !ta-
         jd(5) = -15            !ta+ 
      endif
c--   couplings
      GX    =gzzh
      GXX(1)=gzl(1)
      GXX(2)=gzl(2)
      GXX1(1)=gzl(1)
      GXX1(2)=gzl(2)
c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*4d0/2d0    
c--   helicities
      jhel(2)=+1
      if(xran1(iseed) .gt. .5e0) jhel(2)=-1
      jhel(3)=-jhel(2)
      jhel(4)=+1
      if(xran1(iseed) .gt. .5e0) jhel(4)=-1
      jhel(5)=-jhel(4)
c--   phase space
      if(GV.eq.0d0) call error_trap("GV")
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_h4f(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   sum of flavours
      dwgt=dwgt*9d0
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.17) then 
*---------------------------------
*     h -> z*  z -> j  j~  l-  l+ (j=u,d,c,s;l=e,mu )
*---------------------------------

c--   masses
      M1=hmass
      M2=0d0
      M3=0d0
      M4=0d0
      M5=0d0
      MV=ZMASS
      GV=ZWIDTH
c--   color
      icol(1,2)=cindex          !position 2 is always a particle  
      icol(2,2)=0
      icol(1,3)=0               !position 3 is always a anti-particle
      icol(2,3)=cindex
c--   couplings
      GX    =gzzh
      GXX(1)=gzl(1)
      GXX(2)=gzl(2)
      GXX1(1)=gzl(1)
      GXX1(2)=gzl(2)
c--   id's
      aa =dble(xran1(iseed)) 
      aa1=dble(xran1(iseed))
      aa2=dble(xran1(iseed))
c     Z1
c-- probability of a uc or ds decay
         
         xprob1=w_z_uu/(w_z_dd + w_z_uu)

         if(aa1.lt.xprob1) then ! decay into ups
            multi_decay=1
            
            if(aa.lt. .5d0) then !u
               M2=0d0
               jd(2) = 2  
               jd(3) =-2
               GXX(1)=gzu(1)
               GXX(2)=gzu(2)
            else                !c
               jd(2) = 4  
               jd(3) =-4
               M2=cmass
               GXX(1)=gzu(1)
               GXX(2)=gzu(2)
            endif

         else                   !decay into downs
            multi_decay=2
            if(aa.lt..5d0) then !d
               M2=0d0
               jd(2) = 1  
               jd(3) =-1
               GXX(1)=gzd(1)
               GXX(2)=gzd(2)
            else                !s
               jd(2) = 3  
               jd(3) =-3
               M2=0d0
               GXX(1)=gzd(1)
               GXX(2)=gzd(2)
            endif
         
         endif                  !which decay: in ups or downs
      M3=M2
c     Z2
      if(aa2.lt.half) then
         jd(4) =  11            !e- 
         jd(5) = -11            !e+ 
      else
         jd(4) =  13            !mu-  
         jd(5) = -13            !mu+ 
      endif

c--   color*(bose factor)*number of helicities
      factor=3d0*1d0*4d0/2d0  
c--   helicities
      jhel(2)=+1
      if(xran1(iseed) .gt. .5e0) jhel(2)=-1
      jhel(3)=-jhel(2)
      jhel(4)=+1
      if(xran1(iseed) .gt. .5e0) jhel(4)=-1
      jhel(5)=-jhel(4)
c--   phase space
      if(GV.eq.0d0) call error_trap("GV")
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_h4f(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   sum of flavours
      dwgt=dwgt*2d0
      if(multi_decay.eq.1) then
         dwgt=dwgt*2d0/xprob1   !j=u,c 
      else
         dwgt=dwgt*2d0/(1.-xprob1) !j=d,s
      endif
c--   sum of two possibilities for the decay of a Z
      dwgt=dwgt*2d0
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.18) then 
*---------------------------------
*     h -> z*  z -> j  j~  l-  l+ (j=u,d,c,s;l=e,mu,ta)
*---------------------------------

c--   masses
      M1=hmass
      M2=0d0
      M3=0d0
      M4=0d0
      M5=0d0
      MV=ZMASS
      GV=ZWIDTH
c--   color
      icol(1,2)=cindex          !position 2 is always a particle  
      icol(2,2)=0
      icol(1,3)=0               !position 3 is always a anti-particle
      icol(2,3)=cindex
c--   couplings
      GX    =gzzh
      GXX(1)=gzl(1)
      GXX(2)=gzl(2)
      GXX1(1)=gzl(1)
      GXX1(2)=gzl(2)
c--   id's
      aa =dble(xran1(iseed))
      aa1=dble(xran1(iseed))
      aa2=dble(xran1(iseed))
c     Z1
      xprob1=w_z_uu/(w_z_dd + w_z_uu)
      if(aa1.lt.xprob1) then    ! decay into ups
         multi_decay=1
         
         if(aa.lt. .5d0) then   !u
            M2=0d0
            jd(2) = 2  
            jd(3) =-2
            GXX(1)=gzu(1)
            GXX(2)=gzu(2)
         else                   !c
            jd(2) = 4  
            jd(3) =-4
            M2=cmass
            GXX(1)=gzu(1)
            GXX(2)=gzu(2)
         endif
         
      else                      !decay into downs
         multi_decay=2
         if(aa.lt..5d0) then    !d
            M2=0d0
            jd(2) = 1  
            jd(3) =-1
            GXX(1)=gzd(1)
            GXX(2)=gzd(2)
         else                   !s
            jd(2) = 3  
            jd(3) =-3
            M2=0d0
            GXX(1)=gzd(1)
            GXX(2)=gzd(2)
         endif
         
      endif                     !which decay: in ups or downs
      M3=M2
c     Z2
      if(aa2.lt.one/three) then
         jd(4) =  11            !e-
         jd(5) = -11            !e+ 
      elseif(aa2.lt.two/three) then
         jd(4) =  13            !mu-  
         jd(5) = -13            !mu+ 
      else
         M4    =  lmass
         M5    =  lmass
         jd(4) =  15            !ta-
         jd(5) = -15            !ta+ 
      endif
      

c--   color*(bose factor)*number of helicities
      factor=3d0*1d0*4d0/2d0    
c--   helicities
      jhel(2)=+1
      if(xran1(iseed) .gt. .5e0) jhel(2)=-1
      jhel(3)=-jhel(2)
      jhel(4)=+1
      if(xran1(iseed) .gt. .5e0) jhel(4)=-1
      jhel(5)=-jhel(4)
c--   phase space
      if(GV.eq.0d0) call error_trap("GV")
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_h4f(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   sum of flavours
      dwgt=dwgt*3d0
      if(multi_decay.eq.1) then
         dwgt=dwgt*2d0/xprob1   !j=u,c 
      else
         dwgt=dwgt*2d0/(1.-xprob1) !j=d,s
      endif
c--   sum of two possibilities for the decay of a Z
      dwgt=dwgt*2d0
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return
      
      elseif(imode.eq.19) then 
*---------------------------------
*     h -> z*  z -> b  b~  l-  l+ (l=e,mu)
*---------------------------------

c--   masses
      M1=hmass
      M2=bmass
      M3=bmass
      M4=0d0
      M5=0d0
      MV=ZMASS
      GV=ZWIDTH
c--   color
      icol(1,2)=cindex          !position 2 is always a particle  
      icol(2,2)=0
      icol(1,3)=0               !position 3 is always a anti-particle
      icol(2,3)=cindex
c--   couplings
      GX     =gzzh
      GXX(1) =gzd(1)
      GXX(2) =gzd(2)
      GXX1(1)=gzl(1)
      GXX1(2)=gzl(2)
c--   id's
      jd(2) = 5  
      jd(3) =-5
      aa=dble(xran1(iseed))
c     Z2
      if(aa.lt.half) then
         jd(4) =  11            !e- 
         jd(5) = -11            !e+ 
      else
         jd(4) =  13            !mu-  
         jd(5) = -13            !mu+ 
      endif
       
c--   color*(bose factor)*number of helicities
      factor=3d0*1d0*8d0/2d0    
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  get_hel(xran1(iseed),2)
      jhel(4) =  get_hel(xran1(iseed),2)
      jhel(5) =  -jhel(4)
c--   phase space
      if(GV.eq.0d0) call error_trap("GV")
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_h4f(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   sum of flavours
      dwgt=dwgt*2d0
c--   sum of two possibilities for the decay of a Z
      dwgt=dwgt*2d0
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.20) then 
*---------------------------------
*     h -> z*  z -> b  b~  l-  l+ (l=e,mu,ta )
*---------------------------------

c--   masses
      M1=hmass
      M2=bmass
      M3=bmass
      M4=0d0
      M5=0d0
      MV=ZMASS
      GV=ZWIDTH
c--   color
      icol(1,2)=cindex          !position 2 is always a particle  
      icol(2,2)=0
      icol(1,3)=0               !position 3 is always a anti-particle
      icol(2,3)=cindex
c--   couplings
      GX     =gzzh
      GXX(1) =gzd(1)
      GXX(2) =gzd(2)
      GXX1(1)=gzl(1)
      GXX1(2)=gzl(2)
c--   color*(bose factor)*number of helicities
      factor=3d0*1d0*16d0    
c--   id's
      jd(2) = 5  
      jd(3) =-5
      aa=dble(xran1(iseed))
      if(aa.lt.one/three) then
         jd(4) =  11            !e-
         jd(5) = -11            !e+ 
      elseif(aa.lt.two/three) then
         jd(4) =  13            !mu-  
         jd(5) = -13            !mu+ 
      else
         M4    =  lmass
         M5    =  lmass
         jd(4) =  15            !ta-
         jd(5) = -15            !ta+ 
      endif
      
c--   color*(bose factor)*number of helicities
      factor=3d0*1d0*8d0/2d0    
c--   helicities
      jhel(2) =  get_hel(xran1(iseed),2)
      jhel(3) =  get_hel(xran1(iseed),2)
      jhel(4) =  get_hel(xran1(iseed),2)
      jhel(5) =  -jhel(4)
c--   phase space
      if(GV.eq.0d0) call error_trap("GV")
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_h4f(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   sum of flavours
      dwgt=dwgt*3d0
c--   sum of two possibilities for the decay of a Z
      dwgt=dwgt*2d0
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.21) then 
*---------------------------------
*     h -> z*  z -> vl vl~ l-' l+'(l=e,mu,ta,l'=e,mu)
*---------------------------------

c--   masses
      M1=hmass
      M2=0d0
      M3=0d0
      M4=0d0
      M5=0d0
      MV=ZMASS
      GV=ZWIDTH
c--   couplings
      GX    =gzzh
      GXX(1)=gzn(1)
      GXX(2)=gzn(2)
      GXX1(1)=gzl(1)
      GXX1(2)=gzl(2)
c--   id's
      aa1=dble(xran1(iseed))
      aa2=dble(xran1(iseed))
c     Z
      if(aa1.lt.one/three) then
         jd(2) =  12            !ve 
         jd(3) = -12            !ve~
      elseif(aa1.lt.two/three) then
         jd(2) =  14            !vm 
         jd(3) = -14            !vm~ 
      else
         jd(2) =  16            !vt 
         jd(3) = -16            !vt~ 
      endif
c     Z2
      if(aa2.lt.half) then
         jd(4) =  11            !e- 
         jd(5) = -11            !e+ 
      else
         jd(4) =  13            !mu-  
         jd(5) = -13            !mu+ 
      endif

c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*4d0/2d0    
c--   helicities
      jhel(2)=+1
      if(xran1(iseed) .gt. .5e0) jhel(2)=-1
      jhel(3)=-jhel(2)
      jhel(4)=+1
      if(xran1(iseed) .gt. .5e0) jhel(4)=-1
      jhel(5)=-jhel(4)
c--   phase space
      if(GV.eq.0d0) call error_trap("GV")
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_h4f(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   sum of flavours
      dwgt=dwgt*6d0
c--   sum of two possibilities for the decay of a Z
      dwgt=dwgt*2d0
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      elseif(imode.eq.22) then 
*---------------------------------
*     h -> z*  z -> vl vl~ l-' l+'(l,l'=e,mu,ta)
*---------------------------------

c--   masses
      M1=hmass
      M2=0d0
      M3=0d0
      M4=0d0
      M5=0d0
      MV=ZMASS
      GV=ZWIDTH
c--   couplings
      GX     =gzzh
      GXX(1) =gzn(1)
      GXX(2) =gzn(2)
      GXX1(1)=gzl(1)
      GXX1(2)=gzl(2)
c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*16d0    
c--   id's
      aa1=dble(xran1(iseed))
      aa2=dble(xran1(iseed))
c     Z
      if(aa1.lt.one/three) then
         jd(2) =  12            !ve 
         jd(3) = -12            !ve~
      elseif(aa1.lt.two/three) then
         jd(2) =  14            !vm 
         jd(3) = -14            !vm~ 
      else
         jd(2) =  16            !vt 
         jd(3) = -16            !vt~ 
      endif
c     Z
      if(aa2.lt.one/three) then
         jd(4) =  11            !e-
         jd(5) = -11            !e+ 
      elseif(aa2.lt.two/three) then
         jd(4) =  13            !mu-  
         jd(5) = -13            !mu+ 
      else
         M4    =  lmass
         M5    =  lmass
         jd(4) =  15            !ta-
         jd(5) = -15            !ta+ 
      endif


c--   color*(bose factor)*number of helicities
      factor=1d0*1d0*4d0/2d0    
c--   helicities
      jhel(2)=+1
      if(xran1(iseed) .gt. .5e0) jhel(2)=-1
      jhel(3)=-jhel(2)
      jhel(4)=+1
      if(xran1(iseed) .gt. .5e0) jhel(4)=-1
      jhel(5)=-jhel(4)
c--   phase space
      if(GV.eq.0d0) call error_trap("GV")
      call phasespace(pd,pswgt)
      if(pswgt.eq.0d0) return
c--   matrix element
      call emme_h4f(pd,jhel,emmesq)
c--   weight
      dwgt=pswgt*emmesq*factor
c--   sum of flavours
      dwgt=dwgt*9d0
c--   sum of two possibilities for the decay of a Z
      dwgt=dwgt*2d0
c--   check that dwgt is a reasonable number
      call check_nan(dwgt)
      return

      endif !imode

      endif  !higgs

      write(*,*) 'from decay:end of decay reached'
      
      return
      end




      INTEGER FUNCTION GET_HEL(RND,ID)
**********************************************************
* GIVEN THE RND NUMBER, RETURNS 
* +1,  -1 WHEN ID=2
* +1,0,-1 WHEN ID=3
**********************************************************
      implicit none
c
c     arguments
c
      real*4 rnd
      integer id
c----------
c     Begin
c----------      

      if(id.eq.2) then
         get_hel=-1
         if(rnd.gt.0.5e0) get_hel=1
      elseif(id.eq.3) then
         if(rnd.lt.0.333333e0)then
            get_hel=-1
         elseif(rnd.lt.0.666666e0) then
            get_hel=0
         else
            get_hel=1
         endif
      else
         write(*,*) 'choice not implemented in gen_hel'
      endif
      return
      end


      SUBROUTINE GET_X(X,WGT)
C-------------------------------------------------------
C     PROVIDES THE X(NDIM) AND THE WGT OBTAINED FROM
C     A PREVIOUS RUN OF VEGAS
C-------------------------------------------------------
C
      IMPLICIT NONE
C
C     PARAMETERS
C
      INTEGER NDMX,MXDIM
      PARAMETER (NDMX=50,MXDIM=10)
C
C     ARGUMENTS
C
      REAL*4 X(MXDIM),WGT
C
C     LOCAL
C
      INTEGER i,j,k,ia(MXDIM)
      REAL rc,xo,xn,rndn
C
C     GLOBAL
C
      INTEGER NG,NDIM   
      REAL   region(2*MXDIM),xi(NDMX,MXDIM),xnd,dx(MXDIM)
      COMMON /VEGAS_PAR1/NG,NDIM
      COMMON /VEGAS_PAR2/region,xi,xnd,dx
C
C     EXTERNAL
C
      REAL*4 XRAN1
      EXTERNAL XRAN1
      INTEGER IDUM
      DATA idum/1/ ! DOES NOT AFFECT PREVIOUS SETTING
c     
c--   start the loop over dimensions
c     
      wgt=1.
      do 17 j=1,ndim
c     fax: avoid random numbers exactly equal to 0. or 1.
 303     rndn=xran1(idum)
         if(rndn.eq.1e0.or.rndn.eq.0e0) goto 303
         xn=rndn*xnd+1.
         ia(j)=max(min(int(xn),NDMX),1)
         if(ia(j).gt.1)then
            xo=xi(ia(j),j)-xi(ia(j)-1,j)
            rc=xi(ia(j)-1,j)+(xn-ia(j))*xo
         else
            xo=xi(ia(j),j)
            rc=(xn-ia(j))*xo
         endif
         x(j)=region(j)+rc*dx(j)
         wgt=wgt*xo*xnd
 17   continue
     
      RETURN
      END
      
      


      SUBROUTINE ERROR_TRAP(STRING)
C-------------------------------------------------------
C     RETURNS INFORMATION ABOUT SOME ERROR THAT OCCURED
C-------------------------------------------------------
C
      IMPLICIT NONE
C
C     ARGUMENTS
C
      CHARACTER*20 STRING
C
C     BEGIN
C
      IF(STRING(1:2).EQ."GV") THEN
      write(*,*) '*****************************************************'
      write(*,*) '*                                                   *'
      write(*,*) '*          >>>>ERROR TRAP CALLED<<<<                *'
      write(*,*) '*                                                   *'
      write(*,*) '*   the width of the vector boson(s) entering       *'
      write(*,*) '*   the selected decay should be set >0 in          *'
      write(*,*) '*   the banner of the event file or in setpara.f.   *'
      write(*,*) '*                                                   *'
      write(*,*) '*            PROGRAM STOPS HERE                     *'
      write(*,*) '*****************************************************'
      ELSE
      write(*,*) '*****************************************************'
      write(*,*) '*                                                   *'
      write(*,*) '*          >>>>ERROR TRAP CALLED<<<<                *'
      write(*,*) '*                                                   *'
      write(*,*) '*             SOME ERROR OCCURRED                   *'
      write(*,*) '*                                                   *'
      write(*,*) '*****************************************************'
      ENDIF

      STOP

      RETURN
      END


      SUBROUTINE CHECK_NAN(x)
C-------------------------------------------------------
C     Check that x is real positive number
C-------------------------------------------------------
C
      IMPLICIT NONE
C
C     ARGUMENTS
C
      real*8 x
c
c     LOCAL
c
      integer n
      data n/0/
      SAVE
C
C     BEGIN
C
      if(.not.(x.gt.0d0).and.x.ne.0) then
         n=n+1
         x=0d0
         write(*,*) 'Found total: ',n,' errors in points in PS'  

c      open(unit=20,file='decay_error.log',status="old",err=100)
c      write(unit=20,fmt='(a50,1x,i5)') 'error in one point in PS',n
c      close(20)
c      return
c 100  open(unit=20,file='decay_error.log',status="new")
c      write(unit=20,fmt='(a50,1x,i5)') 'error in one point in PS',n
c      close(20)

      endif

      return
      end
