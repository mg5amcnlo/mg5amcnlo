      subroutine crossp(p1,p2,p)
c**************************************************************************
c     input:
c            p1, p2    vectors to cross
c**************************************************************************
      implicit none
      real*8 p1(0:3), p2(0:3), p(0:3)

      p(0)=0d0
      p(1)=p1(2)*p2(3)-p1(3)*p2(2)
      p(2)=p1(3)*p2(1)-p1(1)*p2(3)
      p(3)=p1(1)*p2(2)-p1(2)*p2(1)

      return 
      end


      subroutine rotate(p1,p2,n,nn2,ct,st,d)
c**************************************************************************
c     input:
c            p1        vector to be rotated
c            n         vector perpendicular to plane of rotation
c            nn2       squared norm of n to improve numerics
c            ct, st    cos/sin theta of rotation in plane 
c            d         direction: 1 there / -1 back
c     output:
c            p2        p1 rotated using defined rotation
c**************************************************************************
      implicit none
      real*8 p1(0:3), p2(0:3), n(0:3), at(0:3), ap(0:3), cr(0:3)
      double precision nn2, ct, st, na, nn
      integer d, i

      if (nn2.eq.0d0) then
         do i=0,3
            p2(i)=p1(i)
         enddo   
         return
      endif
      nn=dsqrt(nn2)
      na=(n(1)*p1(1)+n(2)*p1(2)+n(3)*p1(3))/nn2
      do i=1,3
         at(i)=n(i)*na
         ap(i)=p1(i)-at(i)
      enddo
      p2(0)=p1(0)
      call crossp(n,ap,cr)
      do i=1,3
         if (d.ge.0) then
            p2(i)=at(i)+ct*ap(i)+st/nn*cr(i)
         else 
            p2(i)=at(i)+ct*ap(i)-st/nn*cr(i)
         endif
      enddo
      
      return 
      end


      subroutine constr(p1,p2,n,nn2,ct,st)
c**************************************************************************
c     input:
c            p1, p2    p1 rotated onto p2 defines plane of rotation
c     output:
c            n         vector perpendicular to plane of rotation
c            nn2       squared norm of n to improve numerics
c            ct, st    cos/sin theta of rotation in plane 
c**************************************************************************
      implicit none
      real*8 p1(0:3), p2(0:3), n(0:3), tr(0:3)
      double precision nn2, ct, st, mct

      ct=p1(1)*p2(1)+p1(2)*p2(2)+p1(3)*p2(3)
      ct=ct/dsqrt(p1(1)**2+p1(2)**2+p1(3)**2)
      ct=ct/dsqrt(p2(1)**2+p2(2)**2+p2(3)**2)
      mct=ct
c     catch bad numerics
      if (mct-1d0>0d0) mct=0d0
      st=dsqrt(1d0-mct*mct)
      call crossp(p1,p2,n)
      nn2=n(1)**2+n(2)**2+n(3)**2
c     don't rotate if nothing to rotate
      if (nn2.le.1d-34) then
         nn2=0d0
         return
      endif
      return 
      end


      integer function combid(i,j)
c**************************************************************************
c     input:
c            i,j       legs to combine
c     output:
c            index of combined leg
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      integer i, j

c      combid=min(i+j,ishft(1,nexternal+1)-2-i-j)
      combid = i+j
     
      return
      end


      subroutine filprp(nFKSprocess,ignum,idij)
c**************************************************************************
c     Include graph ignum in list for propagator idij
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'cluster.inc'
      include 'message.inc'
      integer ignum, idij, nFKSprocess, i

      if(idij.gt.n_max_cl) return
      do i=1,id_cl(nFKSprocess,idij,0)
         if (id_cl(nFKSprocess,idij,i).eq.ignum) return
      enddo
      id_cl(nFKSprocess,idij,0)=id_cl(nFKSprocess,idij,0)+1
      id_cl(nFKSprocess,idij,id_cl(nFKSprocess,idij,0))=ignum
      if(btest(mlevel,5)) write(*,*)'Adding graph ',ignum,' to prop '
     $     ,idij,' for proc ',nFKSprocess
      return
      end

      logical function filgrp(ignum,ipnum,ipids)
c**************************************************************************
c     input:
c            ignum      number of graph to be analysed
c            ipnum      number of level to be analysed, 
c                       starting with nexternal
c            ipids      particle number, iforest number, 
c                       daughter1, daughter2
c     output:
c            true if no errors
c**************************************************************************
      implicit none
c Include
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'maxparticles.inc'
      include 'cluster.inc'
      include 'message.inc'
      double precision ZERO
      parameter (ZERO=0d0)
c Argument
      integer ignum,ipnum,ipids(nexternal,4,2:nexternal)
c Local
      integer i,j,k,l,icmp(2)
c Common
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer iforest(2,-max_branch:-1,n_max_cg)
      integer sprop(-max_branch:-1,n_max_cg)
      integer tprid(-max_branch:-1,n_max_cg)
      integer mapconfig(0:n_max_cg)
      common/c_configs_inc/iforest,sprop,tprid,mapconfig
      double precision prmass(-max_branch:nexternal,n_max_cg)
      double precision prwidth(-max_branch:-1,n_max_cg)
      integer prow(-max_branch:-1,n_max_cg)
      common/c_props_inc/prmass,prwidth,prow
c External
      integer combid
      logical isjet
      external combid,isjet

      if(btest(mlevel,4))
     $     write(*,*) 'graph,level: ',ignum,ipnum

      filgrp=.false.
C     Follow diagram tree down to last clustering
      do i=1,ipnum
         do j=i+1,ipnum
            if(btest(mlevel,4)) write(*,*)'at ids   (',ipids(i,1,ipnum)
     $           ,',',ipids(i,2,ipnum),'), (', ipids(j,1,ipnum),','
     $           ,ipids(j,2,ipnum),'), ',i,j
            do k=-nexternal+1,-1
               if ((iforest(1,k,ignum).eq.ipids(i,2,ipnum).and.
     &              iforest(2,k,ignum).eq.ipids(j,2,ipnum)).or.
     &             (iforest(2,k,ignum).eq.ipids(i,2,ipnum).and.
     &              iforest(1,k,ignum).eq.ipids(j,2,ipnum))) then
c     Add the combined propagator
                  icmp(1)=combid(ipids(i,1,ipnum),ipids(j,1,ipnum))
c     Add also the same propagator but from the other direction
                  icmp(2)=ishft(1,nexternal)-1-icmp(1)
c     Set pdg code for propagator
                  do l=1,2
c     If s-channel (this only works with compile flag 'fno-automatic')
                     if(sprop(k,ignum).ne.0)then
                        ipdgcl(icmp(l),ignum,nFKSprocess)=sprop(k,ignum)
c     If t-channel (this only works with compile flag 'fno-automatic')
                     else if(tprid(k,ignum).ne.0)then
                        ipdgcl(icmp(l),ignum,nFKSprocess)=tprid(k,ignum)
c     Final 2->1 process
                     else if(ipnum.eq.3)then
                        ipdgcl(icmp(l),ignum,nFKSprocess)=ipdgcl(2,ignum
     $                       ,nFKSprocess)
                     else
                        ipdgcl(icmp(l),ignum,nFKSprocess)=0
                     endif
                     if(btest(mlevel,4)) write(*,*)
     $                    'add table entry for (',ipids(i,1,ipnum), ','
     $                    ,ipids(j,1,ipnum),',',icmp(l),')', 'proc: '
     $                    ,nFKSprocess, ', pdg: ',ipdgcl(icmp(l),ignum
     $                    ,nFKSprocess)
                     call filprp(nFKSprocess,ignum,icmp(l))
c               Insert graph in list of propagators
                     if(prwidth(k,ignum).gt.ZERO) then
                     if(btest(mlevel,4))
     $                       write(*,*)'Adding resonance ',ignum,icmp(l)
                        resmap(icmp(l),ignum)=.true.
                     endif
                  enddo
c     proceed w/ next table, since there is no possibility,
c     to combine the same particle in another way in this graph
                  ipids(i,1,ipnum-1)=icmp(1)
                  ipids(i,2,ipnum-1)=k
                  ipids(i,3,ipnum-1)=i
                  ipids(i,4,ipnum-1)=j
                  ipnum=ipnum-1
                  do l=1,j-1
                    if(l.eq.i) cycle
                    ipids(l,1,ipnum)=ipids(l,1,ipnum+1)
                    ipids(l,2,ipnum)=ipids(l,2,ipnum+1)
                    ipids(l,3,ipnum)=l
                    ipids(l,4,ipnum)=0
                  enddo
                  do l=j,ipnum
                     ipids(l,1,ipnum)=ipids(l+1,1,ipnum+1)
                     ipids(l,2,ipnum)=ipids(l+1,2,ipnum+1)
                     ipids(l,3,ipnum)=l+1
                     ipids(l,4,ipnum)=0
                  enddo
                  if(ipnum.eq.2)then
c     Done with this diagram (need to do each diagram twice)
                     return
                  else
                     filgrp=.true.
                     return
                  endif
               endif
            enddo
         enddo
      enddo
      return
      end


      logical function filmap()
c**************************************************************************
c     Wrapper for the filgrp function. It loops over the diagrams. The
c     goal of this function is to fill the id_cl and ipdgcl with the
c     diagram (or better: configuration) information
c
c     output:
c            true if no errors
c**************************************************************************
      implicit none
c Include
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'cluster.inc'
      include 'message.inc'
      include 'genps.inc'
c Local
      integer start_config,end_config,i,j,iFKS,inpids
      integer ipids(nexternal,4,2:nexternal)
c Common
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer iforest(2,-max_branch:-1,n_max_cg)
      integer sprop(-max_branch:-1,n_max_cg)
      integer tprid(-max_branch:-1,n_max_cg)
      integer mapconfig(0:n_max_cg)
      common/c_configs_inc/iforest,sprop,tprid,mapconfig
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
c External
      logical filgrp
      external filgrp

c Set up which configuration should be included
      start_config=1
      end_config=mapconfig(0)
c Initialize all the clustering ID's to zero
      do i=1,n_max_cl
         id_cl(nFKSprocess,i,0)=0
      enddo
c Loop over the configurations
      do i=start_config,end_config
c         write (*,*) ' at graph ',i
         do j=1,nexternal
            ipids(j,1,nexternal)=ishft(1,j-1)
            ipids(j,2,nexternal)=j
            ipids(j,3,nexternal)=0
            ipids(j,4,nexternal)=0
            ipdgcl(ipids(j,1,nexternal),i,nFKSprocess)=idup(j,1)
         enddo
         inpids=nexternal
         if(btest(mlevel,3))
     $        write(*,*) 'Inserting graph ',i
 10      if (filgrp(i,inpids,ipids)) goto 10
      enddo
      filmap=.true.
      return
      end

      subroutine checkbw(p,nbw,ibwlist,isbw)
c**************************************************************************
c      Checks if any resonances are on the BW for this configuration
c**************************************************************************
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'ngraphs.inc'
      include 'run.inc'
      include 'real_from_born_configs.inc'
      double precision p(0:3,nexternal)
      integer nbw,ibwlist(2,nexternal),i,j,ida(2),idenpart
      logical isbw(*)
      logical OnBW(-nexternal:0),onshell
      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config
      integer iforest(2,-max_branch:-1,n_max_cg)
      integer sprop(-max_branch:-1,n_max_cg)
      integer tprid(-max_branch:-1,n_max_cg)
      integer mapconfig_dummy(0:n_max_cg)
      common/c_configs_inc/iforest,sprop,tprid,mapconfig_dummy
      integer icl(-(nexternal-3):nexternal)
      integer ibw,iconf
      double precision xp(0:3,-nexternal:nexternal),xmass
      double precision prmass(-max_branch:nexternal,n_max_cg)
      double precision prwidth(-max_branch:-1,n_max_cg)
      integer prow(-max_branch:-1,n_max_cg)
      common/c_props_inc/prmass,prwidth,prow
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      double precision dot
      external dot

c Check that momenta lead to an on-shell BW
      iconf=real_from_born_conf(this_config,nFKSprocess)
      do i=1,nexternal
         do j=0,3
            xp(j,i)=p(j,i)
         enddo
      enddo
      do i=-1,-(nexternal-3),-1
         onbw(i) = .false.
      enddo
      do i=-1,-(nexternal-3),-1 !Loop over propagators
c Skip the t-channels
         if (iforest(1,i,iconf).eq.1 .or. iforest(2,i,iconf).eq.1 .or.
     &       iforest(1,i,iconf).eq.2 .or. iforest(2,i,iconf).eq.2 ) exit
         do j=0,3
            xp(j,i) = xp(j,iforest(1,i,iconf))+xp(j,iforest(2,i,iconf))
         enddo
         if (prwidth(i,iconf) .gt. 0d0) then !This is B.W.
c
c     If the invariant mass is close to pole mass, set OnBW to true
c
            xmass = sqrt(dot(xp(0,i),xp(0,i)))
            onshell = ( abs(xmass-prmass(i,iconf)) .lt.
     &           bwcutoff*prwidth(i,iconf) )
            if(onshell)then
               OnBW(i) = .true.
c     If mother and daughter have the same ID, remove one of them
               idenpart=0
               do j=1,2
                  ida(j)=iforest(j,i,iconf)
                  if(    (ida(j).lt.0.and.
     &                        sprop(i,iconf).eq.sprop(ida(j),iconf))
     &               .or.(ida(j).gt.0.and.
     &                        sprop(i,iconf).eq.IDUP(ida(j),1)))
     &                 idenpart=ida(j)    ! mother and daugher have same ID
               enddo
c     Always remove if daughter final-state (and identical)
               if(idenpart.gt.0) then
                  OnBW(i)=.false.
c     Else remove either this resonance or daughter,
c                  whichever is closer to mass shell
               elseif(idenpart.lt.0.and.abs(xmass-prmass(i,iconf)).gt.
     $                 abs(sqrt(dot(xp(0,idenpart),xp(0,idenpart)))-
     $                 prmass(i,iconf))) then
                  OnBW(i)=.false.         ! mother off-shell
               elseif(idenpart.lt.0) then
                  OnBW(idenpart)=.false.  ! daughter off-shell
               endif
            endif
         endif
      enddo

      nbw=0
      do i=1,nexternal
        icl(i)=ishft(1,i-1)
      enddo
      do i=-1,-(nexternal-3),-1
        icl(i)=icl(iforest(1,i,iconf))+
     $     icl(iforest(2,i,iconf))
        isbw(icl(i))=.false.
        if(OnBW(i))then
          nbw=nbw+1
          ibwlist(1,nbw)=icl(i)
          ibwlist(2,nbw)=i
          isbw(icl(i))=.true.
c          print *,'Added BW for resonance ',i,icl(i),this_config
        endif
      enddo
      end


      logical function findmt(idij,icgs)
c**************************************************************************
c     input:
c            idij, icgs
c     output:
c            true if tree structure identified
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'maxconfigs.inc'
      include 'cluster.inc'
      include 'message.inc'
      include 'real_from_born_configs.inc'
      include 'run.inc'
      integer idij,icgs(0:n_max_cg)
      logical foundbw
      integer i, ii, j, jj, il, igsbk(0:n_max_cg)
      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
c
      findmt=.false.
c     if first clustering, set possible graphs
      if (icgs(0).eq.0) then
         ii=0
         do i=1,id_cl(nFKSprocess,idij,0)
c     if chcluster, allow only this config
           if(chcluster)then
              if(id_cl(nFKSprocess,idij ,i) .ne.
     $             real_from_born_conf(this_config,nFKSprocess)) cycle
           endif
c     check if we have constraint from onshell resonances
            foundbw=.true.
            do j=1,nbw
               if(resmap(ibwlist(1,j),id_cl(nFKSprocess,idij,i)))then
                  cycle
               endif
               foundbw=.false.
 10         enddo
            if((nbw.eq.0.or.foundbw))then
               ii=ii+1
               icgs(ii)=id_cl(nFKSprocess,idij,i)
            endif
         enddo
         icgs(0)=ii
         if (icgs(0).gt.0)then
           findmt=.true.
         endif
         if (btest(mlevel,5)) write (*,*)'findmt: ',findmt
     $        ,' NFKSPROCESS=',NFKSPROCESS,' icgs(0)=',icgs(0), ' icgs
     $        =',(icgs(i),i=1,icgs(0))
      else
c     Check for common graphs
         j=1
         ii=0
         do i=1,icgs(0)
            if(j.le.id_cl(nFKSprocess,idij,0).and.
     $           icgs(i).eq.id_cl(nFKSprocess,idij,j))then
               ii=ii+1
               icgs(ii)=id_cl(nFKSprocess,idij,j)
               j=j+1
            else if(j.le.id_cl(nFKSprocess,idij,0).and.
     $              icgs(i).gt.id_cl(nFKSprocess,idij,j)) then
               do while(icgs(i).gt.id_cl(nFKSprocess,idij,j).and.
     $              j.le.id_cl(nFKSprocess,idij,0))
                  j=j+1
               enddo
               if(j.le.id_cl(nFKSprocess,idij,0).and.
     $              icgs(i).eq.id_cl(nFKSprocess,idij,j))then
                  ii=ii+1
                  icgs(ii)=id_cl(nFKSprocess,idij,j)
               endif
            endif
         enddo
         icgs(0)=ii
         findmt=(icgs(0).gt.0)
      endif
      end


      logical function cluster(p)
c**************************************************************************
c     input:
c            p(0:3,i)           momentum of i'th parton
c     output:
c            true if tree structure identified
c**************************************************************************
      implicit none
c Include
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'genps.inc'
      include 'cluster.inc'
      include 'message.inc'
      include 'real_from_born_configs.inc'
      include 'run.inc'
c Argument
      double precision p(0:3,nexternal)
c Local
      integer i,j,k,n,idi,idj,idij,icgs(0:n_max_cg),nleft,iwin,jwin
     $     ,iwinp,imap(nexternal,2)
      double precision pcmsp(0:3),p1(0:3),pi(0:3),nr(0:3),nn2,ct,st
     $     ,minpt2ij,pt2ij(n_max_cl),zij(n_max_cl)
      double precision pz(0:3)
      data (pz(i),i=0,3)/1d0,0d0,0d0,1d0/
c Common
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config
c External
      integer combid
      logical findmt
      double precision dj, pydj, djb, pyjb, dot, SumDot, zclus
      external findmt,dj, pydj, djb, pyjb, dot, SumDot, zclus, combid

      if (btest(mlevel,1))
     $   write (*,*)'New event'

      cluster=.false.
      clustered=.false.
      do i=0,3
        pcmsp(i)=0d0
      enddo
c     Check if any resonances are on the BW, store results in to_checkbw
      call checkbw(p,nbw,ibwlist,isbw)
      if(btest(mlevel,4).and.nbw.gt.0)
     $     write(*,*) 'Found BWs: ',(ibwlist(1,i),i=1,nbw)

c     initialize index map
      do i=1,nexternal
         imap(i,1)=i
         imap(i,2)=ishft(1,i-1)
         mt2ij(i)=0
      enddo   
      mt2last=0d0
      minpt2ij=1.0d37
      do i=1,nexternal
c     initialize momenta
         idi=ishft(1,i-1)
         do j=0,3
            pcl(j,idi)=p(j,i)
         enddo
c     give mass to external particles
         pcl(4,idi)=max(dot(p(0,i),p(0,i)),0d0)
c     never combine the two beams
         if (i.gt.2) then
c     fill combine table, first pass only (others are done below),
c     determine all ptij
            do j=1,i-1
               idj=ishft(1,j-1)
               if (btest(mlevel,4)) then
                  write (*,*)'i = ',i,'(',idi,'), j = ',j,'(',idj,')'
               endif
c     cluster only combinable legs (acc. to diagrams)
               icgs(0)=0
               idij=combid(idi,idj)
               pt2ij(idij)=1.0d37
               if (findmt(idij,icgs)) then
                  if (btest(mlevel,4)) then
                     write(*,*)'diagrams: ',(icgs(k),k=1,icgs(0))
                  endif
                  if (j.ne.1.and.j.ne.2) then
c     final state clustering                     
                     if(isbw(idij))then
                       pt2ij(idij)=SumDot(pcl(0,idi),pcl(0,idj),1d0)
                       if (btest(mlevel,4))
     $                    write(*,*)'Mother ',idij,' has ptij ',
     $                    sqrt(pt2ij(idij))
                     else
                       if(ktscheme.eq.2)then
                         pt2ij(idij)=pydj(pcl(0,idi),pcl(0,idj))
                       else
                         pt2ij(idij)=dj(pcl(0,idi),pcl(0,idj))
                       endif
                     endif
                     zij(idij)=0d0
                  else
c     initial state clustering, only if hadronic collision
c     check whether 2->(n-1) process w/ cms energy > 0 remains
                     iwinp=imap(3-j,2);
                     if(ickkw.eq.2.or.ktscheme.eq.2)then
                        pt2ij(idij)=pyjb(pcl(0,idi), pcl(0,idj),pcl(0
     $                       ,iwinp),zij(idij))
                        zij(idij)=0d0
                     else
                        pt2ij(idij)=djb(pcl(0,idi))
                        zij(idij)=zclus(pcl(0,idi),pcl(0,idj),pcl(0
     $                       ,iwinp))
                     endif
c     prefer clustering when outgoing in direction of incoming
                     if(sign(1d0,pcl(3,idi)).ne.sign(1d0,pcl(3,idj)))
     $                    pt2ij(idij)=pt2ij(idij)*(1d0+1d-6)
                  endif
                  if (btest(mlevel,4)) then
                     write(*,*)'         ',idi,'&',idj,' part ',iwinp,
     &                              ' -> ',idij,' pt2ij = ',pt2ij(idij)
                     if(j.eq.1.or.j.eq.2)then
                       write(*,*)'     cf. djb: ',djb(pcl(0,idi))
                     endif
                  endif
c     Check if smallest pt2 ("winner")
                  if (pt2ij(idij).lt.minpt2ij) then
                     iwin=j
                     jwin=i
                     minpt2ij=pt2ij(idij)
                  endif                 
               endif
            enddo
         endif
      enddo
c     Take care of special 2 -> 1 case
      if (nexternal.eq.3.and.nincoming.eq.2) then
         n=1
c     Make sure that initial-state particles are daughters
         idacl(n,1)=imap(1,2)
         idacl(n,2)=imap(2,2)
         imocl(n)=imap(3,2)
         pt2ijcl(n)=pcl(4,imocl(n))
         zcl(n)=0.
c        Set info for LH clustering output
         icluster(1,n)=1
         icluster(2,n)=3
         icluster(3,n)=2
         igraphs(0)=1
         igraphs(1)=real_from_born_conf(this_config,nFKSprocess)
         cluster=.true.
         clustered=.true.
         return
      endif
c     initialize graph storage
      igraphs(0)=0
      nleft=nexternal
c     cluster 
      do n=1,nexternal-2
c     combine winner
         imocl(n)=imap(iwin,2)+imap(jwin,2)
         idacl(n,1)=imap(iwin,2)
         idacl(n,2)=imap(jwin,2)
         pt2ijcl(n)=minpt2ij
         zcl(n)=zij(imocl(n))
         if (btest(mlevel,2)) then
            write(*,*)'winner ',n,': ',idacl(n,1),'&',idacl(n,2),
     &           ' -> ',minpt2ij,', z = ',zcl(n)
            write (*,*) 'iwin,jwin:',iwin,jwin,imocl(n)
         endif
c        Set info for LH clustering output
         icluster(1,n)=imap(iwin,1)
         icluster(2,n)=imap(jwin,1)
         icluster(3,n)=0
         icluster(4,n)=0
         if (isbw(imocl(n))) then
            do i=1,nbw
               if(ibwlist(1,i).eq.imocl(n))then
                  icluster(4,n)=ibwlist(2,i)
                  exit
               endif
            enddo
         endif
c     Reset igraphs with new mother
         if (.not.findmt(imocl(n),igraphs)) then
            write(*,*) 'cluster.f: Error. Invalid combination.' 
            return
         endif
         if (btest(mlevel,4)) then
            write(*,*)'graphs: ',(igraphs(k),k=1,igraphs(0))
         endif
         if (iwin.lt.3) then
c     initial state clustering
c     Set mt2ij to m^2+pt^2 
            mt2ij(n)=djb(pcl(0,idacl(n,2)))
            if (btest(mlevel,1)) then
               write(*,*)'mtij(',n,') for ',idacl(n,2),' is '
     $              ,sqrt(mt2ij(n)),' (cf ',sqrt(pt2ijcl(n)),')'
            endif
            iwinp=imap(3-iwin,2);
c        Set partner info for LH clustering output
            icluster(3,n)=imap(3-iwin,1)
            do i=0,3
               pcl(i,imocl(n))=pcl(i,idacl(n,1))-pcl(i,idacl(n,2))
               pcmsp(i)=-pcl(i,imocl(n))-pcl(i,iwinp)
            enddo
            pcmsp(0)=-pcmsp(0)
            pcl(4,imocl(n))=0
            if((pcl(4,idacl(n,1)).gt.0.or.pcl(4,idacl(n,2)).gt.0).and.
     $         .not.(pcl(4,idacl(n,1)).gt.0.and.pcl(4,idacl(n,2)).gt.0))
     $         pcl(4,imocl(n))=max(pcl(4,idacl(n,1)),pcl(4,idacl(n,2)))

c       Don't boost if boost vector too lightlike or last vertex 
            if (pcmsp(0)**2-pcmsp(1)**2-pcmsp(2)**2-pcmsp(3)**2.gt.100d0
     $           .and.nleft.gt.4) then
               call boostx(pcl(0,imocl(n)),pcmsp(0),p1(0))
               call constr(p1(0),pz(0),nr(0),nn2,ct,st)
               do j=1,nleft
                  call boostx(pcl(0,imap(j,2)),pcmsp(0),p1(0))
                  call rotate(p1(0),pi(0),nr(0),nn2,ct,st,1)
                  do k=0,3
                     pcl(k,imap(j,2))=pi(k)
                  enddo
               enddo
               call boostx(pcl(0,imocl(n)),pcmsp(0),p1(0))
               call rotate(p1(0),pi(0),nr(0),nn2,ct,st,1)
               do k=0,3
                  pcl(k,imocl(n))=pi(k)
               enddo
            endif
         else
c     final state clustering
            do i=0,3
               pcl(i,imocl(n))=pcl(i,idacl(n,1))+pcl(i,idacl(n,2))
            enddo
            pcl(4,imocl(n))=0
            if(pcl(4,idacl(n,1)).gt.0.or.pcl(4,idacl(n,2)).gt.0.and..not.
     $           (pcl(4,idacl(n,1)).gt.0.and.pcl(4,idacl(n,2)).gt.0))
     $           pcl(4,imocl(n))=max(pcl(4,idacl(n,1)),pcl(4,idacl(n,2)))
            if(isbw(imocl(n)))then
               pcl(4,imocl(n))=pt2ijcl(n)
               if (btest(mlevel,4))
     $              write(*,*) 'Mother ',imocl(n),' has mass**2 ',
     $              pcl(4,imocl(n))
            endif
         endif
        
         nleft=nleft-1
c     create new imap
         imap(iwin,2)=imocl(n)
         do i=jwin,nleft
            imap(i,1)=imap(i+1,1)
            imap(i,2)=imap(i+1,2)
         enddo
         if (nleft.le.3) then
c     If last clustering is FS, store also average transverse mass
c     of the particles combined (for use if QCD vertex, e.g. tt~ or qq~)
            if(iwin.gt.2)then
               mt2last=
     &              sqrt(djb(pcl(0,idacl(n,1)))*djb(pcl(0,idacl(n,2))))
               if (btest(mlevel,3)) then
                  write(*,*)'Set mt2last to ',mt2last
               endif              
c     Boost and rotate back to get m_T for final particle
               if (pcmsp(0)**2-pcmsp(1)**2-pcmsp(2)**2-pcmsp(3)**2
     $              .gt.100d0) then
                  call rotate(pcl(0,imap(3,2)),p1(0),nr(0),nn2,ct,st,-1)
                  do k=1,3
                     pcmsp(k)=-pcmsp(k)
                  enddo
                  call boostx(p1(0),pcmsp(0),pcl(0,imap(3,2)))
               endif
            endif
c     Make sure that initial-state particle is always among daughters
            idacl(n+1,1)=imap(1,2)
            idacl(n+1,2)=imap(2,2)
            imocl(n+1)=imap(3,2)

            pt2ijcl(n+1)=djb(pcl(0,imap(3,2)))
c        Set info for LH clustering output
            icluster(1,n+1)=1
            icluster(2,n+1)=3
            icluster(3,n+1)=2
            icluster(4,n+1)=0
            if (btest(mlevel,3)) then
               write(*,*) 'Last vertex is',imap(1,2),imap(2,2),imap(3,2)
               write(*,*) '          ->',pt2ijcl(n+1),sqrt(pt2ijcl(n+1))
            endif
c     If present channel among graphs, use only this channel.
c     This is important when we have mixed QED-QCD
            do i=1,igraphs(0)
               if (igraphs(i).eq.
     &              real_from_born_conf(this_config,nFKSprocess)) then
                  igraphs(0)=1
                  igraphs(1)=
     &                 real_from_born_conf(this_config,nFKSprocess)
                  exit
               endif
            enddo
            zcl(n+1)=1
            cluster=.true.
            clustered=.true.
            return
         endif
c     calculate new ptij
c     recalculate all in this case due to rotation & boost
         minpt2ij=1.0d37
c     never combine the two beams so start loop at 3
         do i=3,nleft
            idi=imap(i,2)
c     determine all ptij
            do j=1,i-1
               idj=imap(j,2)
               if (btest(mlevel,4)) write (*,*)'i = ',i,'(',idi
     $              ,'), j = ',j,'(',idj,')'
c     Reset diagram list icgs
               do k=0,igraphs(0)
                  icgs(k)=igraphs(k)
               enddo
               if (btest(mlevel,4)) write (*,*)'Reset diagrams to: '
     $              ,(icgs(k),k=1,icgs(0))
               idij=combid(idi,idj)
               pt2ij(idij)=1.0d37
c     cluster only combinable legs (acc. to diagrams)
               if (.not.findmt(idij,icgs)) cycle
               if (btest(mlevel,4)) then
                  write(*,*)'diagrams: ',(icgs(k),k=1,icgs(0)),i,j,idi
     $                 ,idj
               endif
               if (j.ne.1.and.j.ne.2) then
c     final state clustering                     
                  if(isbw(idij))then
                     pt2ij(idij)=SumDot(pcl(0,idi),pcl(0,idj),1d0)
                     if (btest(mlevel,4)) write(*,*) 'Mother ',idij
     $                    ,' has ptij ', sqrt(pt2ij(idij))
                  else
                     if(ktscheme.eq.2)then
                        pt2ij(idij)=pydj(pcl(0,idi),pcl(0,idj))
                     else
                        pt2ij(idij)=dj(pcl(0,idi),pcl(0,idj))
                     endif
                  endif
                  zij(idij)=0d0
               else
c     initial state clustering, only if hadronic collision
c     check whether 2->(n-1) process w/ cms energy > 0 remains
                  iwinp=imap(3-j,2);
                  if(ickkw.eq.2.or.ktscheme.eq.2)then
                     pt2ij(idij)=pyjb(pcl(0,idi),pcl(0,idj),pcl(0
     $                    ,iwinp),zij(idij))
                  else
                     pt2ij(idij)=djb(pcl(0,idi))
                     zij(idij)=zclus(pcl(0,idi),pcl(0,idj),pcl(0
     $                    ,iwinp))
                  endif
c     prefer clustering when outgoing in direction of incoming
                  if(sign(1d0,pcl(3,idi)).ne.sign(1d0,pcl(3,idj)))
     $                 pt2ij(idij)=pt2ij(idij)*(1d0+1d-6)
               endif
               if (btest(mlevel,4)) then
                  write(*,*)'         ',idi,'&',idj,' part ',iwinp
     $                 ,' -> ',idij,' pt2ij = ',pt2ij(idij)
                  if(j.eq.1.or.j.eq.2)then
                     write(*,*)'     cf. djb: ',djb(pcl(0,idi))
                  endif
               endif
               if (pt2ij(idij).lt.minpt2ij) then
                  iwin=j
                  jwin=i
                  minpt2ij=pt2ij(idij)
               endif                 
            enddo
         enddo
      enddo

      return
      end
