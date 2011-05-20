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
c      write(*,*)'nn2 ',nn2,' ',nn,' ',na
c      write(*,*)'ap ',ap(1),',',ap(2),',',ap(3)
c      write(*,*)'at ',at(1),',',at(2),',',at(3)
      p2(0)=p1(0)
      call crossp(n,ap,cr)
c      write(*,*)'cr ',cr(1),',',cr(2),',',cr(3)
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

c     check rotation
c      call rotate(p1(0),tr(0),n(0),nn2,ct,st,1)
c      write(*,*)'p1 (',p1(0),',',p1(1),',',p1(2),',',p1(3),')'
c      write(*,*)'p2 (',p2(0),',',p2(1),',',p2(2),',',p2(3),')'
c      write(*,*)'nn (',n(0),',',n(1),',',n(2),',',n(3),')'
c      write(*,*)'nn (',n(0),',',n(1),',',n(2),',',n(3),')'
c      write(*,*)'nn2 = ',nn2,', ct = ',ct,', st = ',st
c      write(*,*)'tr (',tr(0),',',tr(1),',',tr(2),',',tr(3),')'
      
      return 
      end


      Subroutine mapids(ids,id)
c**************************************************************************
c     input:
c            ids       array of particle ids
c            id        compressed particle id
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      integer i, id, ids(nexternal)
      
      id=0
      do i=1,nexternal
         if (ids(i).ne.0) then
            id=id+ishft(1,i)
         endif
      enddo
c      write(*,*) 'cluster.f: compressed code is ',id

      return
      end


      subroutine mapid(id,ids)
c**************************************************************************
c     input:
c            id        compressed particle id
c            ids       array of particle ids
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      integer i, icd, id, ids(nexternal)
      
      icd=id
      do i=1,nexternal
         ids(i)=0
         if (btest(id,i)) then
            ids(i)=1
         endif
c         write(*,*) 'cluster.f: uncompressed code ',i,' is ',ids(i)
      enddo

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


      subroutine filprp(ignum,idij)
c**************************************************************************
c     Include graph ignum in list for propagator idij
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      include 'cluster.inc'
      integer ignum, idij, i

      if(idij.gt.n_max_cl) return
      do i=1,id_cl(idij,0)
         if (id_cl(idij,i).eq.ignum) return
      enddo
      id_cl(idij,0)=id_cl(idij,0)+1
      id_cl(idij,id_cl(idij,0))=ignum
c      print *,'Adding graph ',ignum,' to prop ',idij
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
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include 'cluster.inc'
      include 'coupl.inc'
      integer ignum, ipnum, ipids(nexternal,4,2:nexternal)
C $B$ IFOREST $B$ !this is a tag for MadWeight
      integer i, iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/iforest
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      common/to_sprop/sprop,tprid
C $E$ IFOREST $E$ !this is a tag for MadWeight
      INTEGER    n_max_cl_cg
      PARAMETER (n_max_cl_cg=n_max_cl*n_max_cg)
      data resmap/n_max_cl_cg*.false./
 

      Integer j, k, l, icmp(2)

      double precision ZERO
      parameter (ZERO=0d0)
      double precision pmass(-nexternal:0,lmaxconfigs)
      double precision pwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)
      logical first_time
      save pmass,pwidth,pow
      data first_time /.true./

      integer combid
      external combid

      if (first_time) then
         include 'props.inc'
         first_time=.false.
      endif

c      write(*,*) 'graph,level,iupdown: ',ignum,ipnum,iupdown

      filgrp=.false.
C   Follow diagram tree down to last clustering
      do i=1,ipnum
         do j=i+1,ipnum
c            write(*,*)'at ids   (',ipids(i,1,ipnum),',',ipids(i,2,ipnum),'), (',
c     &           ipids(j,1,ipnum),',',ipids(j,2,ipnum),'), ',i,j
            do k=-nexternal+1,-1
               if ((iforest(1,k,ignum).eq.ipids(i,2,ipnum).and.
     &              iforest(2,k,ignum).eq.ipids(j,2,ipnum)).or.
     &             (iforest(2,k,ignum).eq.ipids(i,2,ipnum).and.
     &              iforest(1,k,ignum).eq.ipids(j,2,ipnum))) then
c                 Add the combined propagator
                  icmp(1)=combid(ipids(i,1,ipnum),ipids(j,1,ipnum))
c                 Add also the same propagator but from the other direction
                  icmp(2)=ishft(1,nexternal)-1-icmp(1)
c     Set pdg code for propagator
                  do l=1,2
                     ipdgcl(icmp(l),ignum)=sprop(k,ignum)
                     if(ipdgcl(icmp(l),ignum).eq.0)
     $                    ipdgcl(icmp(l),ignum)=tprid(k,ignum)
c                  write(*,*) 'add table entry for (',ipids(i,1,ipnum),
c     &                 ',',ipids(j,1,ipnum),',',icmp,')','pdg: ',
c     $                 ipdgcl(icmp,ignum)
                     call filprp(ignum,icmp(l))

c               Insert graph in list of propagators
                     if(pwidth(k,ignum).gt.ZERO) then
c                    write(*,*)'Adding resonance ',ignum,icmp
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
c                  do l=1,ipnum
c                     write(*,*) 'new: ipids(',l,') = (',ipids(l,1,ipnum),
c     &                    ',',ipids(l,2,ipnum),',',ipids(l,3,ipnum),',',ipids(l,4,ipnum),')',
c     $                    ' pdg: ', ipdgcl(ipids(l,1,ipnum),ignum)
c                  enddo
                  if(ipnum.eq.2)then
c                 Done with this diagram
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
c     output:
c            true if no errors
c**************************************************************************
      implicit none
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include 'cluster.inc'
      include 'run.inc'
      include 'maxamps.inc'
C $B$ IFOREST $B$ !this is a tag for MadWeight
      integer mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config
C $E$ IFOREST $E$ !this is a tag for MadWeight
      integer i, j, inpids, ipids(nexternal,4,2:nexternal)
      integer start_config,end_config
      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'

      logical filgrp
      external filgrp

      if(chcluster) then
         start_config=this_config
         end_config=this_config
      else
         start_config=1
         end_config=mapconfig(0)
      endif
      do i=1,n_max_cl
         id_cl(i,0)=0
      enddo
      do i=start_config,end_config
c         write (*,*) ' at graph ',i
         do j=1,nexternal
            ipids(j,1,nexternal)=ishft(1,j-1)
            ipids(j,2,nexternal)=j
            ipids(j,3,nexternal)=0
            ipids(j,4,nexternal)=0
            ipdgcl(ipids(j,1,nexternal),i)=idup(j,1,1)
         enddo
         inpids=nexternal
c         print *,'Inserting graph ',i
 10      if (filgrp(i,inpids,ipids)) goto 10
      enddo
      filmap=.true.
      return
      end


      subroutine checkbw(nbw,ibwlist,isbw)
c**************************************************************************
c      Checks if any resonances are on the BW for this configuration
c**************************************************************************
      implicit none
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
C $B$ NGRAPHS $E$ !this is a tag for MadWeight

      integer nbw,ibwlist(nexternal)
      logical isbw(*)

      logical             OnBW(-nexternal:0)     !Set if event is on B.W.
      common/to_BWEvents/ OnBW
      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config
C $B$ IFOREST $B$ !this is a tag for MadWeight
      integer i, iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/ iforest
C $E$ IFOREST $E$ !this is a tag for MadWeight
C $B$ DECAYBW $E$ !this is a tag for MadWeight

      integer icl(-(nexternal-3):nexternal)
      integer ibw

      nbw=0
      do i=-1,-(nexternal-3),-1
C $B$ ONBW $B$ !this is a tag for MadWeight
        if(OnBW(i)) then 
C $E$ ONBW $E$ !this is a tag for MadWeight
           nbw=nbw+1
        endif
      enddo
      if(nbw.eq.0)then
c        print *,'No BW found'
        return
      endif

      do i=1,nexternal
        icl(i)=ishft(1,i-1)
      enddo
      ibw=0
      do i=-1,-(nexternal-3),-1
        icl(i)=icl(iforest(1,i,this_config))+
     $     icl(iforest(2,i,this_config))
        isbw(icl(i))=.false.
C $B$ ONBW $B$ !this is a tag for MadWeight
        if(OnBW(i))then
C $E$ ONBW $E$ !this is a tag for MadWeight
          ibw=ibw+1
          ibwlist(ibw)=icl(i)
          isbw(icl(i))=.true.
c          print *,'Added BW for resonance ',i,icl(i),this_config
          if(ibw.eq.nbw) return
        endif
      enddo
      
      end

      logical function findmt(idij,icgs,nbw,ibwlist)
c**************************************************************************
c     input:
c            idij, icgs
c     output:
c            true if tree structure identified
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      include 'cluster.inc'

      integer idij,nbw,ibwlist(nexternal),icgs(0:n_max_cg)
      logical foundbw
      integer i, ii, j, jj, il, igsbk(0:n_max_cg)

      findmt=.false.
c     if first clustering, set possible graphs
      if (icgs(0).eq.0) then
         ii=0
         do i=1,id_cl(idij,0)
c        check if we have constraint from onshell resonances
           foundbw=.true.
           do j=1,nbw
             if(resmap(ibwlist(j),id_cl(idij,i)))then
               cycle
             endif
             foundbw=.false.
 10        enddo
           if(nbw.eq.0.or.foundbw)then
              ii=ii+1
              icgs(ii)=id_cl(idij,i)
           endif
         enddo
         icgs(0)=ii
         if (icgs(0).gt.0)then
           findmt=.true.
         endif
         return
      else
c     Check for common graphs
         j=1
         ii=0
         do i=1,icgs(0)
            if(j.le.id_cl(idij,0).and.icgs(i).eq.id_cl(idij,j))then
               ii=ii+1
               icgs(ii)=id_cl(idij,j)
               j=j+1
            else if(j.le.id_cl(idij,0).and.icgs(i).gt.id_cl(idij,j)) then
               do while(icgs(i).gt.id_cl(idij,j).and.j.le.id_cl(idij,0))
                  j=j+1
               enddo
               if(j.le.id_cl(idij,0).and.icgs(i).eq.id_cl(idij,j))then
                  ii=ii+1
                  icgs(ii)=id_cl(idij,j)
               endif
            endif
         enddo
         icgs(0)=ii
         findmt=(icgs(0).gt.0)
         return
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
      include 'run.inc'
      include 'genps.inc'
      include 'nexternal.inc'
      include 'cluster.inc'
      include 'message.inc'
      include 'maxconfigs.inc'
      real*8 p(0:3,nexternal), pcmsp(0:3), p1(0:3)
      real*8 pi(0:3), nr(0:3), pz(0:3)
      integer i, j, k, n, idi, idj, idij, icgs(0:n_max_cg)
      integer nleft, iwin, jwin, iwinp, imap(nexternal,2) 
      double precision nn2,ct,st
      double precision minpt2ij,pt2ij(n_max_cl),zij(n_max_cl)

      integer mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config

      integer nbw,ibwlist(nexternal)
      logical isbw(n_max_cl)
      data isbw/n_max_cl*.false./

      data (pz(i),i=0,3)/1d0,0d0,0d0,1d0/

      integer combid
      logical findmt
      external findmt
      double precision dj, pydj, djb, pyjb, dot, SumDot, zclus
      external dj, pydj, djb, pyjb, dot, SumDot, zclus, combid

      if (btest(mlevel,1))
     $   write (*,*)'New event'

      cluster=.false.
      clustered=.false.
      do i=0,3
        pcmsp(i)=0
      enddo
c     Check if any resonances are on the BW, store results in to_checkbw
      call checkbw(nbw,ibwlist,isbw)

c     initialize index map
      do i=1,nexternal
         imap(i,1)=i
         imap(i,2)=ishft(1,i-1)
         mt2ij(i)=0
      enddo   
      mt2last=0
      minpt2ij=1.0d37
      do i=1,nexternal
c     initialize momenta
         idi=ishft(1,i-1)
         do j=0,3
            pcl(j,idi)=p(j,i)
         enddo
c     give mass to external particles
         pcl(4,idi)=dot(p(0,i),p(0,i))
c     never combine the two beams
         if (i.gt.2) then
c     fill combine table, first pass, determine all ptij
            do j=1,i-1
               idj=ishft(1,j-1)
               if (btest(mlevel,4))
     $              write (*,*)'i = ',i,'(',idi,'), j = ',j,'(',idj,')'
c     cluster only combinable legs (acc. to diagrams)
               icgs(0)=0
               idij=combid(idi,idj)
               pt2ij(idij)=1.0d37
               if (findmt(idij,icgs,nbw,ibwlist)) then
                  if (btest(mlevel,4)) then
                     write(*,*)'diagrams: ',(icgs(k),k=1,icgs(0))
                  endif
                  if (j.ne.1.and.j.ne.2) then
c     final state clustering                     
                     if(isbw(idij))then
                       pt2ij(idij)=SumDot(pcl(0,idi),pcl(0,idj),1d0)
                       if (btest(mlevel,4))
     $                    print *,'Mother ',idij,' has ptij ',
     $                    sqrt(pt2ij(idij))
                     else
                       if(ktscheme.eq.2)then
                         pt2ij(idij)=pydj(pcl(0,idi),pcl(0,idj))
                       else
                         pt2ij(idij)=dj(pcl(0,idi),pcl(0,idj))
                       endif
                     endif
                  else
c     initial state clustering, only if hadronic collision
c     check whether 2->(n-1) process w/ cms energy > 0 remains
                     iwinp=imap(3-j,2);
                     if(ickkw.eq.2.or.ktscheme.eq.2)then
                        pt2ij(idij)=pyjb(pcl(0,idi),
     $                    pcl(0,idj),pcl(0,iwinp),zij(idij))
                     else
                        pt2ij(idij)=djb(pcl(0,idi))
                        zij(idij)=zclus(pcl(0,idi),pcl(0,idj),pcl(0,iwinp))
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
         imocl(n)=idij
         idacl(n,1)=idi
         idacl(n,2)=idj
         pt2ijcl(n)=pcl(4,idi)
         zcl(n)=0.
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
         endif
c     Reset igraphs with new mother
         if (.not.findmt(imocl(n),igraphs,nbw,ibwlist)) then
            write(*,*) 'cluster.f: Error. Invalid combination.' 
            return
         endif
         if (btest(mlevel,4)) then
            write(*,*)'graphs: ',(igraphs(k),k=1,igraphs(0))
         endif
         if (iwin.lt.3) then
c     is clustering
c     Set mt2ij to m^2+pt^2 
            mt2ij(n)=djb(pcl(0,idacl(n,2)))
            if (btest(mlevel,3)) then
               write(*,*)'mtij(',n,') for ',idacl(n,2),' is ',sqrt(mt2ij(n)),
     $              ' (cf ',sqrt(pt2ijcl(n)),')'
            endif
            iwinp=imap(3-iwin,2);
            do i=0,3
               pcl(i,imocl(n))=pcl(i,idacl(n,1))-pcl(i,idacl(n,2))
c            enddo
c     set incoming particle on-shell
c            pcl(0,imocl(n))=sqrt(pcl(1,imocl(n))**2+
c     $         pcl(2,imocl(n))**2+pcl(3,imocl(n))**2)
c            do i=0,3
               pcmsp(i)=-pcl(i,imocl(n))-pcl(i,iwinp)
            enddo
            pcmsp(0)=-pcmsp(0)
            pcl(4,imocl(n))=0
            if(pcl(4,idacl(n,1)).gt.0.or.pcl(4,idacl(n,2)).gt.0.and..not.
     $         (pcl(4,idacl(n,1)).gt.0.and.pcl(4,idacl(n,2)).gt.0))
     $         pcl(4,imocl(n))=max(pcl(4,idacl(n,1)),pcl(4,idacl(n,2)))

c       Don't boost if boost vector too lightlike or last vertex 
            if (pcmsp(0)**2-pcmsp(1)**2-pcmsp(2)**2-pcmsp(3)**2.gt.100d0.and.
     $           nleft.gt.4) then
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
c     fs clustering
           do i=0,3
             pcl(i,imocl(n))=pcl(i,idacl(n,1))+pcl(i,idacl(n,2))
           enddo
           pcl(4,imocl(n))=0
           if(pcl(4,idacl(n,1)).gt.0.or.pcl(4,idacl(n,2)).gt.0.and..not.
     $        (pcl(4,idacl(n,1)).gt.0.and.pcl(4,idacl(n,2)).gt.0))
     $        pcl(4,imocl(n))=max(pcl(4,idacl(n,1)),pcl(4,idacl(n,2)))
           if(isbw(imocl(n)))then
             pcl(4,imocl(n))=pt2ijcl(n)
             if (btest(mlevel,4))
     $          print *,'Mother ',imocl(n),' has mass**2 ',
     $          pcl(4,imocl(n))
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
c           If last clustering is FS, store also average transverse mass
c           of the particles combined (for use if QCD vertex, e.g. tt~ or qq~)
            if(iwin.gt.2)then
               mt2last=sqrt(djb(pcl(0,idacl(n,1)))*djb(pcl(0,idacl(n,2))))
               if (btest(mlevel,3)) then
                  write(*,*)'Set mt2last to ',mt2last
               endif              
c         Boost and rotate back to get m_T for final particle
               if (pcmsp(0)**2-pcmsp(1)**2-pcmsp(2)**2-pcmsp(3)**2.gt.100d0) then
                  call rotate(pcl(0,imap(3,2)),p1(0),nr(0),nn2,ct,st,-1)
                  do k=1,3
                     pcmsp(k)=-pcmsp(k)
                  enddo
                  call boostx(p1(0),pcmsp(0),pcl(0,imap(3,2)))
               endif
            endif
c         Make sure that final-state particle is always among daughters
            idacl(n+1,1)=imap(1,2)
            idacl(n+1,2)=imap(3,2)
            imocl(n+1)=imap(2,2)
c         If mother is initial state leg (2nd beam), chose other leg
            if (imocl(n+1).eq.4)then
              idacl(n+1,1)=imap(2,2)
              imocl(n+1)=imap(1,2)
            endif

c            if(pcl(0,imocl(n)).gt.0d0)then
            pt2ijcl(n+1)=djb(pcl(0,imap(3,2)))
            if (btest(mlevel,3)) then
              write(*,*) 'Last vertex is ',imap(1,2),imap(2,2),imap(3,2)
              write(*,*) '            -> ',pt2ijcl(n+1),sqrt(pt2ijcl(n+1))
            endif
c     If present channel among graphs, use only this channel
c     This is important when we have mixed QED-QCD
            do i=1,igraphs(0)
               if (igraphs(i).eq.this_config) then
                  igraphs(0)=1
                  igraphs(1)=this_config
                  exit
               endif
            enddo
c            if(pt2ijcl(n).gt. pt2ijcl(n+1))then
c              pt2ijcl(n+1)=pt2ijcl(n)
c              if (btest(mlevel,3)) then
c                write(*,*)'Reset scale for vertex ',n+1,' to ',pt2ijcl(n+1)
c              endif              
c            endif
            zcl(n+1)=1
c            else
c              pt2ijcl(n+1)=pt2ijcl(n)
c            endif
c           Pick out the found graphs
c            print *,'Clustering succeeded, found graph ',igscl(1)
            cluster=.true.
            clustered=.true.
            return
         endif
c     calculate new ptij
c            write(*,*)'is case'
c     recalculate all in is case due to rotation & boost
         minpt2ij=1.0d37
            do i=1,nleft
               idi=imap(i,2)
c     never combine the two beams
               if (i.gt.2) then
c     determine all ptij
                  do j=1,i-1
                     idj=imap(j,2)
                     if (btest(mlevel,4))
     $                    write (*,*)'i = ',i,'(',idi,'), j = ',j,'(',idj,')'
c     Reset diagram list icgs
                     do k=0,igraphs(0)
                        icgs(k)=igraphs(k)
                     enddo
                     if (btest(mlevel,4))
     $                    write (*,*)'Reset diagrams to: ',(icgs(k),k=1,icgs(0))
c     cluster only combinable legs (acc. to diagrams)
                     idij=combid(idi,idj)
c                     write (*,*) 'RECALC !!! ',idij
                     pt2ij(idij)=1.0d37
                     if (findmt(idij,icgs,nbw,ibwlist)) then
                        if (btest(mlevel,4)) then
                           write(*,*)'diagrams: ',(icgs(k),k=1,icgs(0))
                       endif
                        if (j.ne.1.and.j.ne.2) then
c     final state clustering                     
                           if(isbw(idij))then
                             pt2ij(idij)=SumDot(pcl(0,idi),pcl(0,idj),1d0)
                             if (btest(mlevel,4))
     $                          print *,'Mother ',idij,' has ptij ',
     $                          sqrt(pt2ij(idij))
                           else
                             if(ktscheme.eq.2)then
                               pt2ij(idij)=pydj(pcl(0,idi),pcl(0,idj))
                             else
                               pt2ij(idij)=dj(pcl(0,idi),pcl(0,idj))
                             endif
                           endif
                        else
c     initial state clustering, only if hadronic collision
c     check whether 2->(n-1) process w/ cms energy > 0 remains
                          iwinp=imap(3-j,2);
                          do k=0,3
                             pcl(k,idij)=pcl(k,idj)-pcl(k,idi)
c                           pcmsp(k)=pcl(k,idij)+pcl(k,iwinp)
                          enddo
c                       ecms2=pcmsp(0)**2-pcmsp(1)**2-
c                       $                          pcmsp(2)**2-pcmsp(3)**2
c                       if (ecms2.gt.0.1d0.and.
c                       if ((nleft.eq.4.or.ecms2.gt.0.1d0).and.
c                         if((lpp(j).ne.0)) then
                            if(ickkw.eq.2.or.ktscheme.eq.2)then
                              pt2ij(idij)=pyjb(pcl(0,idi),
     $                           pcl(0,idj),pcl(0,iwinp),zij(idij))
                            else
                              pt2ij(idij)=djb(pcl(0,idi))
                              zij(idij)=zclus(pcl(0,idi),pcl(0,idj),pcl(0,iwinp))
                            endif
c                 prefer clustering when outgoing in direction of incoming
                            if(sign(1d0,pcl(3,idi)).ne.sign(1d0,pcl(3,idj)))
     $                         pt2ij(idij)=pt2ij(idij)*(1d0+1d-6)
c                          endif
                        endif
                        if (btest(mlevel,4)) then
                          write(*,*)'         ',idi,'&',idj,' part ',iwinp,' -> ',idij,
     &                       ' pt2ij = ',pt2ij(idij)
                           if(j.eq.1.or.j.eq.2)then
                             write(*,*)'     cf. djb: ',djb(pcl(0,idi))
                           endif
                        endif
                        if (pt2ij(idij).lt.minpt2ij) then
                           iwin=j
                           jwin=i
                           minpt2ij=pt2ij(idij)
                        endif                 

                     endif
                  enddo
               endif
            enddo
      enddo

      return
      end
