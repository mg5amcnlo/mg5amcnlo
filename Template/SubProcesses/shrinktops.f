      program ShrinkTops
c*******************************************************************************
c     Program that takes topologies from configs.inc and writes
c     out new topologies with last particle removed in file
c     configs-1.inc and props-1.inc
c******************************************************************************
      implicit none
c
c     Constants
c     
      include 'genps.inc'
      include 'nexternal.inc'
c
c     Local
c
      integer iconfig,igraph,i,jbranch,ibranch,jconfig
      integer isubprop, isubval
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer iforest2(2,-max_branch:-1,lmaxconfigs)
      integer t_chan
      integer            mapconfig(0:lmaxconfigs)
      integer            mapconfig2(0:lmaxconfigs)
      double precision      spole(maxinvar),swidth(maxinvar),bwjac
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      integer sprop2(-max_branch:-1,lmaxconfigs)
      integer tprid2(-max_branch:-1,lmaxconfigs)
      character*50 buff_pmass 
      character*50 buff_pwidth 
      character*50 buff_pow
      character*10 oniumtype
      integer nchars
c
c     external
c
      logical one_gluon_config
      logical two_gluon_config
c
c     data
c
      include 'configs.inc'
c-----
c  Begin Code
c----
      open(unit=35, file="configs_temp.inc",status="unknown",err=999)
      open(unit=36, file="props.inc",status="old",err=999)
      open(unit=37, file="props_temp.inc",status="unknown",err=999)

      open(unit=38, file="oniumtype.mg",status="unknown",err=999)
      read(38,'(a)') oniumtype
      close(38)
      call no_spaces(oniumtype,nchars)

      jconfig=0

      do iconfig = 1, mapconfig(0)     !Loop over all configurations
c
c     first quick check if we need to keep the conf.
c

      if(oniumtype(4:4).eq.'1') then
        if(one_gluon_config(iforest,
     & sprop,tprid,iconfig)) then
c
c      jump lines in props.inc
c
        ibranch = 0
         do while (ibranch .lt. nexternal-2)
            ibranch = ibranch + 1 
              read(36,'(a)') buff_pmass
              read(36,'(a)') buff_pwidth
              read(36,'(a)') buff_pow
         enddo
c
c     jump to the next config
c
        goto 12
        endif
      endif 

      if(oniumtype(1:1).eq.'3'.and.oniumtype(2:2).eq.'S') then
        if (two_gluon_config(iforest,
     & sprop,tprid,iconfig)) then
c
c      jump lines in props.inc
c
        ibranch = 0
         do while (ibranch .lt. nexternal-2)
            ibranch = ibranch + 1
              read(36,'(a)') buff_pmass
              read(36,'(a)') buff_pwidth
              read(36,'(a)') buff_pow
         enddo
c
c     jump to the next config
c
        goto 12
        endif

      endif

      jconfig=jconfig+1
      mapconfig2(jconfig)=mapconfig(iconfig)

c
c        Second  write out configuration # and graphs
c
         igraph = mapconfig(iconfig)       

         write(35,'(a,i6)') 'c   Graph ',igraph
         write(35,'(6x,a,i4,a,i4,a)')
     $        'data mapconfig(',jconfig,') /',igraph,'/'
c
c        Reset all parameters for configuration
c
         t_chan = 0
         isubprop = 0
         isubval  = 0
         jbranch = 0
         ibranch = 0
         do while (ibranch .lt. nexternal-2+t_chan)
            ibranch = ibranch + 1
c
            if (iforest(1,-ibranch,iconfig) .eq. 1) t_chan=1
c             sometimes there is 1 branch less in props.inc 
              if(t_chan.ne.1.or. ibranch.ne.nexternal-1) then
              read(36,'(a)') buff_pmass
              read(36,'(a)') buff_pwidth
              read(36,'(a)') buff_pow
c              write(*,'(i2,a)') ibranch, buff_pmass
              endif
            if (iforest(1,-ibranch,iconfig) .eq. (nexternal+1) .or.
     $          iforest(2,-ibranch,iconfig) .eq. (nexternal+1)) then   !Remove this one
c   isubprop records the index of the removed branch 
c   isubval records the index of the particle initially grouped with particle nexternal+1
c   isubprop is to be replaced by isubval later on
               isubprop  = -ibranch
               isubval = iforest(1,-ibranch,iconfig)+
     $              iforest(2,-ibranch,iconfig)-nexternal-1
            else   !write out this line
               jbranch=jbranch+1  !new ordering for branches
               do i=1,2
                  if (iforest(i,-ibranch,iconfig) .eq. isubprop) then
                     iforest(i,-ibranch,iconfig) = isubval
cPierre: here we have to add another condition
                  elseif(isubprop.ne.0.and. ! we have already met part. nexternal+1            
     &             isubprop.gt.iforest(i,-ibranch,iconfig)) then ! i.e iforest(i,-ibr) is an intermediate part with an index smaller then isubprop 
                     iforest(i,-ibranch,iconfig)=
     &               iforest(i,-ibranch,iconfig)+1
c end modif Pierre
                  endif
               enddo

              if(jbranch.ne.ibranch) then
                write(buff_pmass(13:15),'(i3)') -jbranch
                write(buff_pwidth(14:16),'(i3)') -jbranch
                write(buff_pow(11:13),'(i3)') -jbranch
              endif
               write(buff_pmass(17:20),'(i4)') jconfig
               write(buff_pwidth(18:21),'(i4)') jconfig
               write(buff_pow(15:18),'(i4)') jconfig


c write info in configs-1.inc
               write(35,99) -jbranch,jconfig,iforest(1,-ibranch,iconfig)
     $              ,iforest(2,-ibranch,iconfig),"?","?"
               iforest2(1,-jbranch,jconfig)=iforest(1,-ibranch,iconfig)
               iforest2(2,-jbranch,jconfig)=iforest(2,-ibranch,iconfig)
               if(t_chan.eq.0) then
               write(35,92) -jbranch,jconfig,sprop(-ibranch,iconfig) 
               sprop2(-jbranch,jconfig)=sprop(-ibranch,iconfig)
               elseif(jbranch.lt.nexternal-2) then
               write(35,93) -jbranch,jconfig,tprid(-ibranch,iconfig)  
               tprid2(-jbranch,jconfig)=tprid(-ibranch,iconfig)
               endif
c             here we should also write pmass,pwidth,pow
c             (sometimes there is 1 branch less in props.inc > condition on jbranch)
              if(t_chan.ne.1.or. ibranch.ne.nexternal-1) then

c               here break the loop in case we have just read the one-to-last branch 
c               and haven't met particle nexternal
                if(isubprop.eq.0.and.ibranch.eq.nexternal-2) then
                  goto 11 
                endif

               write(37,'(a)') buff_pmass
               write(37,'(a)') buff_pwidth
               write(37,'(a)') buff_pow
             endif
              endif
11          continue
         enddo

12    enddo
cPierre: add forgotten line
      write(35,'(6x,a,i4,a,i4,a)')
     $        'data mapconfig(',0,') /',jconfig,'/'
      mapconfig2(0)=jconfig
c end modif Pierre
      close(35)
      close(36)
      close(37)


c here we should remove equivalent configs
      call check_equivalent_configs(mapconfig2,iforest2,sprop2,tprid2)


 99   format(6x,'data(iforest(i,',i3,',',i4,'),i=1,2) /',i3,',',i3,'/',
     &     10x,'!  ',2a)


 92   format(6x,'data sprop(',i4,',',i4,') /',i8,'/')
 93   format(6x,'data tprid(',i4,',',i4,') /',i8,'/')



 999  continue
      end
         

      subroutine check_equivalent_configs(mapconfig,iforest,sprop,
     & tprid)
c*******************************************************************************
c     Program that removes redundant topologies from configs_temp.inc 
c     and props_temp.inc. Results written in configs-1.inc
c******************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'nexternal.inc'
c
c     argument
c
      integer            mapconfig(0:lmaxconfigs)
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
c
c     Local
c
      integer iconfig,igraph,i,jbranch,ibranch,temp_config
      integer t_chan,nb_configs
      character*50 buff_pmass(-nexternal:0,lmaxconfigs)
      character*50 buff_pwidth(-nexternal:0,lmaxconfigs)
      character*50 buff_pow(-nexternal:0,lmaxconfigs)
      character*50 buff_pmass_temp 
      character*50 buff_pwidth_temp
      character*50 buff_pow_temp
      logical foundmatch,foundmatch2
c
c     data
c
c-----
c  Begin Code
c----
      open(unit=35, file="configs-1.inc",status="unknown",err=998)
      open(unit=36, file="props_temp.inc",status="old",err=998)
      open(unit=37, file="props-1.inc",status="unknown",err=998)


      nb_configs=0

c      write(*,*) 'nb of configs before check equiv: ',mapconfig(0)

      do iconfig = 1, mapconfig(0)     !Loop over all configurations
        foundmatch2=.false.
        ibranch = 0
        t_chan = 0
      do while (ibranch .lt. nexternal-3+t_chan)
        ibranch = ibranch + 1
        if (iforest(1,-ibranch,iconfig) .eq. 1) t_chan=1
c       sometimes there is 1 branch less in props.inc
        if((t_chan.ne.1).or. (ibranch.lt.(nexternal-2))) then
          read(36,'(a)') buff_pmass(ibranch,iconfig)
          read(36,'(a)') buff_pwidth(ibranch,iconfig)
          read(36,'(a)') buff_pow(ibranch,iconfig)
        endif
      enddo

        temp_config=1
        do while (temp_config .lt. iconfig.and..not.foundmatch2)
         ibranch=0
         t_chan=0
            foundmatch=.true.
         do while (ibranch .lt. nexternal-3+t_chan)
            ibranch = ibranch + 1
            if (iforest(1,-ibranch,iconfig) .eq. 1) t_chan=1
            if(iforest(1,-ibranch, temp_config).ne.
     & iforest(1,-ibranch,iconfig).and.iforest(1,-ibranch,temp_config)
     & .ne. iforest(2,-ibranch,iconfig)) foundmatch=.false.

            if(iforest(2,-ibranch, temp_config).ne.
     & iforest(1,-ibranch,iconfig).and.iforest(2,-ibranch,temp_config)
     & .ne. iforest(2,-ibranch,iconfig)) foundmatch=.false.

            if (t_chan.eq.0) then
              if(sprop(-ibranch, temp_config).ne.
     & sprop(-ibranch,iconfig)) foundmatch=.false.
            else
              if(tprid(-ibranch, temp_config).ne.
     & tprid(-ibranch,iconfig)) foundmatch=.false.
            endif

c now check props.inc
            if(.false.) then
         if (buff_pmass(ibranch,temp_config)(23:50).ne.
     & buff_pmass(ibranch,iconfig)(23:50))  foundmatch=.false.
         if (buff_pwidth(ibranch,temp_config)(24:50).ne.
     & buff_pwidth(ibranch,iconfig)(24:50)) foundmatch=.false.
         if (buff_pow(ibranch,temp_config)(21:50).ne.
     & buff_pow(ibranch,iconfig)(21:50)) foundmatch=.false.

         if (buff_pmass(ibranch,temp_config)(1:16).ne.
     & buff_pmass(ibranch,iconfig)(1:16))  foundmatch=.false.
         if (buff_pwidth(ibranch,temp_config)(1:17).ne.
     & buff_pwidth(ibranch,iconfig)(1:17)) foundmatch=.false.
         if (buff_pow(ibranch,temp_config)(1:14).ne.
     & buff_pow(ibranch,iconfig)(1:14)) foundmatch=.false.
             endif           

         enddo  ! end loop over branches
         if (foundmatch) then 
          foundmatch2=.true.
c          write(*,*) 'Removing config ',iconfig
         endif

         temp_config=temp_config+1
        enddo ! inner loop over configs
        if(.not.foundmatch2) then !write config
         nb_configs=nb_configs+1
         igraph = mapconfig(iconfig)
         write(35,'(a,i6)') 'c   Graph ',igraph
         write(35,'(6x,a,i4,a,i4,a)')
     $        'data mapconfig(',nb_configs,') /',igraph,'/'
         ibranch=0
         t_chan=0
         do while (ibranch .lt. nexternal-3+t_chan)
            ibranch = ibranch + 1
            if (iforest(1,-ibranch,iconfig) .eq. 1) t_chan=1

            write(35,20) -ibranch,nb_configs,iforest(1,-ibranch,iconfig)
     $       ,iforest(2,-ibranch,iconfig),"?","?"
               if(t_chan.eq.0) then
               write(35,21) -ibranch,nb_configs,sprop(-ibranch,iconfig)
               elseif(ibranch.lt.nexternal-2) then
               write(35,22) -ibranch,nb_configs,tprid(-ibranch,iconfig)
               endif

              if(t_chan.ne.1.or. ibranch.lt.nexternal-2) then
               buff_pmass_temp=buff_pmass(ibranch,iconfig)             
               buff_pwidth_temp=buff_pwidth(ibranch,iconfig)             
               buff_pow_temp=buff_pow(ibranch,iconfig)    
               write(buff_pmass_temp(17:20),'(i4)') nb_configs        
               write(buff_pwidth_temp(18:21),'(i4)') nb_configs        
               write(buff_pow_temp(15:18),'(i4)') nb_configs       
c              
              write(37,'(a)') buff_pmass_temp
              write(37,'(a)') buff_pwidth_temp
              write(37,'(a)') buff_pow_temp
              endif 
         enddo
        endif

      enddo
      write(35,'(6x,a,i4,a,i4,a)')
     $        'data mapconfig(',0,') /',nb_configs,'/'

      close(35)
      close(36)
      close(37)

 21   format(6x,'data sprop(',i4,',',i4,') /',i8,'/')
 22   format(6x,'data tprid(',i4,',',i4,') /',i8,'/')

 20   format(6x,'data(iforest(i,',i3,',',i4,'),i=1,2) /',i3,',',i3,'/',
     &     10x,'!  ',2a)
 998  continue
      end

      logical function one_gluon_config(iforest,sprop,tprid,iconfig)

      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer iforest2(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      integer iconfig
c
c     local
c
      integer ibranch,t_chan

      one_gluon_config=.false.
         t_chan = 0
         ibranch = 0
c              write(*,*) 'iconfig',iconfig
         do while (ibranch .lt. nexternal-2+t_chan)
            ibranch = ibranch + 1
c
         if (iforest(1,-ibranch,iconfig) .eq. 1) t_chan=1
c
         if (iforest(1,-ibranch,iconfig).eq.(nexternal+1)) then
           if(iforest(2,-ibranch,iconfig).eq.(nexternal).and.
     & sprop(-ibranch,iconfig).eq.21) then
          one_gluon_config=.true.
          return
          endif
     
         endif    
c
         if (iforest(1,-ibranch,iconfig).eq.nexternal) then
           if(iforest(2,-ibranch,iconfig).eq.(nexternal+1).and.
     & sprop(-ibranch,iconfig).eq.21) then
          one_gluon_config=.true.
          return
          endif
           
         endif     
         enddo
      end

      logical function two_gluon_config(iforest,sprop,tprid,iconfig)

      implicit none
      include 'maxamps.inc'
      include 'genps.inc'
      include 'nexternal.inc'
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer iforest2(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      integer iconfig
      integer q_ass,qb_ass,prop_qb, prop_q
c
c     local
c
      integer ibranch,t_chan,i
      integer idup(nexternal,maxproc)
      integer mothup(2,nexternal,maxproc)
      integer icolup(2,nexternal,maxflow)
      include 'leshouche.inc'
      two_gluon_config=.false.

         t_chan = 0
         ibranch = 0
         do while (ibranch .lt. nexternal-2+t_chan)
            ibranch = ibranch + 1
c
         if (iforest(1,-ibranch,iconfig) .eq. 1) t_chan=1

            if (iforest(1,-ibranch,iconfig) .eq. (nexternal+1) .or.
     $          iforest(2,-ibranch,iconfig) .eq. (nexternal+1)) then
               prop_qb  = -ibranch
               qb_ass = iforest(1,-ibranch,iconfig)+
     $              iforest(2,-ibranch,iconfig)-nexternal-1
            endif

            if (iforest(1,-ibranch,iconfig) .eq. (nexternal) .or.
     $          iforest(2,-ibranch,iconfig) .eq. (nexternal)) then
               prop_q  = -ibranch
               q_ass = iforest(1,-ibranch,iconfig)+
     $              iforest(2,-ibranch,iconfig)-nexternal
            endif
         enddo

      if(qb_ass.eq.prop_q) then

        if(q_ass.gt.0) then
          if (idup(q_ass,1).ne.21) return     
        endif    

        if(q_ass.lt.0) then
          if(sprop(q_ass,iconfig).ne.21.and.tprid(q_ass,iconfig).ne.21) return         
        endif
       
        if(t_chan.eq.1.and.prop_qb.eq.(-nexternal+1) ) then
          if(idup(2,1).ne.21) return
        else
           if(sprop(prop_qb,iconfig).ne.21.and.tprid(prop_qb,iconfig).ne.21) return
        endif

      two_gluon_config=.true.
      endif

      if(q_ass.eq.prop_qb) then
        if(qb_ass.gt.0) then 
         if (idup(qb_ass,1).ne.21) return
        endif

        if(qb_ass.lt.0) then
          if(sprop(qb_ass,iconfig).ne.21.and.tprid(qb_ass,iconfig).ne.21) return
        endif

        if(t_chan.eq.1.and.prop_q.eq.(-nexternal+1) ) then
           if(idup(2,1).ne.21) return
        else
          if(sprop(prop_q,iconfig).ne.21.and.tprid(prop_q,iconfig).ne.21) return
        endif

       two_gluon_config=.true.
       endif

      return
      end


      subroutine no_spaces(buff,nchars)
c**********************************************************************
c     Given buff a buffer of words separated by spaces
c     returns it where all space are moved to the right
c     returns also the length of the single word.
c     maxlength is the length of the buffer
c     AUTHOR: FABIO MALTONI
c**********************************************************************
      implicit none
c
c     Constants
c
      integer    maxline
      parameter (maxline=10)
      character*1 null
      parameter  (null=' ')
c
c     Arguments
c
      character*(maxline) buff
      integer nchars,maxlength
c
c     Local
c
      integer i,j
      character*(maxline) temp
c-----
c  Begin Code
c-----
      nchars=0
c      write (*,*) "buff=",buff(1:maxlength)
      do i=1,maxline
         if(buff(i:i).ne.null) then
            nchars=nchars+1
            temp(nchars:nchars)=buff(i:i)
         endif
c         write(*,*) i,":",buff(1:maxlength),":",temp(1:nchars),":"
      enddo
      buff=temp
      end

