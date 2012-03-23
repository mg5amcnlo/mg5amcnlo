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

      combid=min(i+j,ishft(1,nexternal+1)-2-i-j)
      
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
      include "nexternal.inc"
      include 'cluster.inc'
      include 'coupl.inc'
      integer ignum, ipnum, ipids(nexternal,4,2:nexternal)
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/iforest
      INTEGER    n_max_cl_cg
      PARAMETER (n_max_cl_cg=n_max_cl*n_max_cg)
      data resmap/n_max_cl_cg*.false./
 

      Integer i, j, k, l, icmp, iupdown,inew(3:4),iold(3:4),itmp(3:4)
      data iupdown/1/

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

      filgrp=.false.
C   First follow diagram tree down to last clustering. Then need to
C   go back up, to allow for ISR from the other direction
      if(iupdown.eq.1)then
      do i=1,ipnum
         do j=i+1,ipnum
c            write(*,*)'at ids   (',ipids(i,1),',',ipids(i,2),'), (',
c     &           ipids(j,1),',',ipids(j,2),')'
            do k=-nexternal+1,-1
               if ((iforest(1,k,ignum).eq.ipids(i,2,ipnum).and.
     &              iforest(2,k,ignum).eq.ipids(j,2,ipnum)).or.
     &             (iforest(2,k,ignum).eq.ipids(i,2,ipnum).and.
     &              iforest(1,k,ignum).eq.ipids(j,2,ipnum))) then
                  icmp=combid(ipids(i,1,ipnum),ipids(j,1,ipnum))
c                  write(*,*) 'add table entry for (',ipids(i,1),
c     &                 ',',ipids(j,1),',',icmp,')' 
                  call filprp(ignum,ipids(i,1,ipnum))
                  call filprp(ignum,ipids(j,1,ipnum))
                  call filprp(ignum,ipids(i,1,ipnum)+ipids(j,1,ipnum))
                  call filprp(ignum,icmp)
c                  print *,'Resonance ',k,' has number ',icmp,
c     $               ipids(i,1,ipnum),ipids(j,1,ipnum)

c               Insert graph in list of propagators
                  if(pwidth(k,ignum).gt.ZERO) then
c                    print *,'Adding resonance ',ignum,ipids(i,1,ipnum)+ipids(j,1,ipnum)
                    resmap(ipids(i,1,ipnum)+ipids(j,1,ipnum),ignum)=.true.
                  endif
c     proceed w/ next table, since there is no possibility,
c     to combine the same particle in another way in this graph
                  ipids(i,1,ipnum-1)=ipids(i,1,ipnum)+ipids(j,1,ipnum)
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
                    iupdown=-1
                    ipids(i,1,ipnum)=ipids(6-i-j,1,ipnum)
                  endif
c                  do l=1,ipnum
c                     write(*,*) 'new: ipids(',l,') = (',ipids(l,1),
c     &                    ',',ipids(l,2),')'
c                  enddo
                  filgrp=.true.
                  return
               endif
            enddo
         enddo
      enddo
      else
C     Go back up, and look only at the daughters
        do i=1,ipnum
          if(ipids(i,4,ipnum).ne.0)then
            do j=3,4
              if(ipids(ipids(i,j,ipnum),2,ipnum+1).le.2)then
                inew(j)=ipids(i,1,ipnum)+
     $             ipids(ipids(i,7-j,ipnum),1,ipnum+1)
                call filprp(ignum,inew(j))
                iold(j)=ipids(ipids(i,j,ipnum),1,ipnum+1)
                itmp(j)=ipids(i,j,ipnum)
              endif
            enddo
C         Change to new particle number everywhere 
            do j=3,4
              if(ipids(ipids(i,j,ipnum),2,ipnum+1).le.2)then
                k=ipnum+1
                do while(ipids(itmp(j),1,k).eq.iold(j))
                  ipids(itmp(j),1,k)=inew(j)
                  itmp(j)=ipids(itmp(j),3,k)
                  k=k+1
                enddo
              endif
            enddo
            ipnum=ipnum+1
            if(ipnum.eq.nexternal)then
              iupdown=1
              filgrp=.false.
            else
              filgrp=.true.
            endif
            return
          endif
        enddo
      endif
      return
      end


      logical function filmap()
c**************************************************************************
c     output:
c            true if no errors
c**************************************************************************
      implicit none
      include 'genps.inc'
      include "nexternal.inc"
      include 'cluster.inc'
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/ iforest
      integer mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config
      integer i, j, inpids, ipids(nexternal,4,2:nexternal)

      logical filgrp
      external filgrp

      do i=1,n_max_cl
         id_cl(i,0)=0
      enddo
      do i=1,mapconfig(0)
c         write (*,*) ' at graph ',i
         do j=1,nexternal
            ipids(j,1,nexternal)=ishft(1,j)
            ipids(j,2,nexternal)=j
            ipids(j,3,nexternal)=0
            ipids(j,4,nexternal)=0
         enddo
         inpids=nexternal
c         print *,'Inserting graph ',i
 10      if (filgrp(mapconfig(i),inpids,ipids)) goto 10
      enddo
      filmap=.true.
      return
      end

      subroutine checkbw()
c**************************************************************************
c      Checks if any resonances are on the BW for this configuration
c**************************************************************************
      implicit none
      include 'genps.inc'
      include "nexternal.inc"

      logical             OnBW(-nexternal:0)     !Set if event is on B.W.
      common/to_BWEvents/ OnBW
      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/ iforest
      integer nbw,ibwlist(nexternal)
      common /to_checkbw/nbw,ibwlist

      integer icl(-(nexternal-3):nexternal)
      integer i,ibw

      nbw=0
      do i=-1,-(nexternal-3),-1
        if(OnBW(i)) nbw=nbw+1
      enddo
      if(nbw.eq.0)then
c        print *,'No BW found'
        return
      endif

      do i=1,nexternal
        icl(i)=ishft(1,i)
      enddo
      ibw=0
      do i=-1,-(nexternal-3),-1
        icl(i)=icl(iforest(1,i,mincfig))+
     $     icl(iforest(2,i,mincfig))
        if(OnBW(i))then
          ibw=ibw+1
          ibwlist(ibw)=icl(i)
c          print *,'Added BW for resonance ',i,icl(i),mincfig
          if(ibw.eq.nbw) return
        endif
      enddo
      
      end

      logical function findmt(idij,icgs)
c**************************************************************************
c     input:
c            p(0:3,i)           momentum of i'th parton
c     output:
c            true if tree structure identified
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      include 'cluster.inc'

      integer nbw,ibwlist(nexternal)
      common /to_checkbw/nbw,ibwlist

      integer i, ii, j, jj, il, idij, icgs(0:n_max_cg)
      logical foundbw

      findmt=.false.
c     if first clustering, set possible graphs
      if (icgs(0).eq.0) then
         icgs(0)=id_cl(idij,0)
         ii=0
         do i=1,icgs(0)
           ii=ii+1
           icgs(ii)=id_cl(idij,i)
c        check if we have constraint from onshell resonances
           foundbw=.true.
           do j=1,nbw
             if(resmap(ibwlist(j),icgs(ii)))then
               cycle
             endif
             foundbw=.false.
 10        enddo
           if(nbw.gt.0.and..not.foundbw) ii=ii-1
         enddo
         icgs(0)=ii
c         write(*,*)'no graph yet -> ',icgs(0)
         if (icgs(0).gt.0)then
c           print *,'Possible graphs: ',(icgs(i),i=1,icgs(0))
           findmt=.true.
         endif
         return
      else
c     look for common graph
         do i=1,icgs(0)
            do j=1,id_cl(idij,0)
               if (id_cl(idij,j).eq.icgs(i)) goto 20
            enddo
         enddo
c         write(*,*)'graphs but no match'
         return
c     accept combination, extract common graphs
 20      il=0
c         do i=1,icgs(0)
c            write(*,*)'arrive w/ ',idij,' ',i,': ',icgs(i)
c         enddo
c         do i=1,id_cl(idij,0)
c            write(*,*)'found ',i,': ',id_cl(idij,i)
c         enddo
         do i=1,icgs(0)
            do j=1,id_cl(idij,0)
               if (id_cl(idij,j).eq.icgs(i)) then
                  il=il+1
c                  write(*,*)'copy ',il,': ',icgs(i)
                  icgs(il)=icgs(i)
                  goto 30
               endif
            enddo
 30         continue
         enddo
         icgs(0)=il
c         write(*,*)'matching graphs -> ',icgs(0)
         findmt=.true.
         return
      endif

      return
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
      include "nexternal.inc"
      include 'cluster.inc'
      include 'message.inc'
      real*8 p(0:3,nexternal), pcmsp(0:3), p1(0:3)
      real*8 pi(0:3), nr(0:3), pz(0:3)
      integer i, j, k, n, idi, idj, idij, icgs(0:n_max_cg)
      integer nleft, iwin, jwin, iwinp, imap(nexternal,2) 
      double precision nn2, ct, st, minpt2ij, pt2ij(n_max_cl)

      integer nbw,ibwlist(nexternal)
      common /to_checkbw/nbw,ibwlist

      data (pz(i),i=0,3)/1d0,0d0,0d0,1d0/

      logical findmt
      external findmt
      double precision dj, djb, dot
      external dj, djb, dot

      cluster=.false.
      do i=0,3
        pcmsp(i)=0
      enddo
c     Check if any resonances are on the BW, stor results in to_checkbw
      call checkbw()

c     initialize index map
      do i=1,nexternal
         imap(i,1)=i
         imap(i,2)=ishft(1,i)
      enddo   
      do i=1,nexternal
c     initialize momenta
         idi=ishft(1,i)
         do j=0,3
            pcl(j,idi)=p(j,i)
         enddo
c     never combine the two beams
         if (i.gt.2) then
c     fill combine table, first pass, determine all ptij
            do j=1,i-1
               idj=ishft(1,j)
               if (btest(mlevel,4))
     $              write (*,*)'i = ',i,'(',idi,'), j = ',j,'(',idj,')'
c     cluster only combinable legs (acc. to diagrams)
               icgs(0)=0
               idij=idi+idj
               pt2ij(idij)=1.0d37
               if (findmt(idij,icgs)) then
                  if (btest(mlevel,4)) then
                     do k=1,icgs(0)
                        write(*,*)'icg(',k,') = ',icgs(k)
                     enddo
                  endif
                  if (j.ne.1.and.j.ne.2) then
c     final state clustering                     
                     pt2ij(idij)=dj(pcl(0,idi),pcl(0,idj))
                  else
c     initial state clustering, only if hadronic collision
c     check whether 2->(n-1) process w/ cms energy > 0 remains
                     iwinp=imap(3-j,2);
                     do k=0,3
                        pcl(k,idij)=pcl(k,idj)-pcl(k,idi)
c                        pcmsp(k)=pcl(k,idij)+pcl(k,iwinp)
                     enddo
c                     ecms2=pcmsp(0)**2-pcmsp(1)**2-
c     $                    pcmsp(2)**2-pcmsp(3)**2
c                     if (ecms2.gt.0.1d0.and.
                     if((lpp(1).ne.0.or.lpp(2).ne.0)) then
                        pt2ij(idij)=djb(pcl(0,idi))
c     prefer clustering when outgoing in direction of incoming
                        if(sign(1d0,pcl(3,idi)).ne.sign(1d0,pcl(3,idj)))
     $                     pt2ij(idij)=pt2ij(idij)*(1d0+1d-6)
                     endif
                  endif
                  if (btest(mlevel,4)) then
                     write(*,*)'         ',idi,'&',idj,' -> ',idij,
     &                    ' pt2ij = ',pt2ij(idij)
                  endif
               endif
            enddo
         endif
      enddo
c     initialize graph storage
      igscl(0)=0
      nleft=nexternal
c     cluster 
      do n=1,nexternal-3
c     determine winner
         minpt2ij=1.0d37
         do i=3,nleft
            do j=1,i-1
               idij=imap(i,2)+imap(j,2)
               icgs(0)=igscl(0)
               do k=1,icgs(0)
                  icgs(k)=igscl(k)
               enddo
               if (findmt(idij,icgs)) then 
                  if (btest(mlevel,4)) then
                     write(*,*)'check pt ',imap(i,2),'&',imap(j,2),' (',
     $                    idij,') -> ',pt2ij(idij),' vs. ',minpt2ij
                  endif
                  if (pt2ij(idij).lt.minpt2ij) then
                     iwin=j
                     jwin=i
                     minpt2ij=pt2ij(idij)
                  endif
               endif
            enddo
         enddo
c     combine winner
         imocl(n)=imap(iwin,2)+imap(jwin,2)
         idacl(n,1)=imap(iwin,2)
         idacl(n,2)=imap(jwin,2)
         pt2ijcl(n)=minpt2ij
         if (.not.findmt(imocl(n),igscl)) then
            write(*,*) 'cluster.f: Error. Invalid combination.' 
            return
         endif
         if (btest(mlevel,4)) then
            write(*,*),'winner ',n,': ',idacl(n,1),'&',idacl(n,2),
     &           ' -> ',minpt2ij
         endif
         if (iwin.lt.3) then
c     is clustering
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

c       Don't boost if boost vector too lightlike
            if (pcmsp(0)**2-pcmsp(1)**2-pcmsp(2)**2-pcmsp(3)**2.gt.100d0) then
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
c           do i=1,nbw
c             if(imocl(n).eq.ibwlist(i))then
c               pt2ijcl(n)=dot(pcl(0,imocl(n)),pcl(0,imocl(n)))
c               print *,'Mother ',imocl(n),' has ptij ',sqrt(pt2ijcl(n))
c               goto 10
c             endif
c           enddo
c 10        continue
         endif

         nleft=nleft-1
c     create new imap
         imap(iwin,2)=imocl(n)
         do i=jwin,nleft
            imap(i,1)=imap(i+1,1)
            imap(i,2)=imap(i+1,2)
         enddo
         if (nleft.eq.3) then
c           If last clustering is FS, store also average transverse mass
c           of the particles combined (for use if QCD vertex, e.g. tt~)
            if(iwin.gt.2)then
               mtlast=sqrt(djb(pcl(0,idacl(n,1)))*djb(pcl(0,idacl(n,2))))
            else
               mtlast=0
            endif
c         Boost and rotate back to get m_T for final particle
            if (pcmsp(0)**2-pcmsp(1)**2-pcmsp(2)**2-pcmsp(3)**2.gt.100d0) then
              call rotate(pcl(0,imap(3,2)),p1(0),nr(0),nn2,ct,st,-1)
              do k=1,3
                 pcmsp(k)=-pcmsp(k)
              enddo
              call boostx(p1(0),pcmsp(0),pcl(0,imap(3,2)))
            endif
            imocl(n+1)=imap(1,2)
            idacl(n+1,1)=imap(3,2)
            idacl(n+1,2)=imap(2,2)
c            if(pcl(0,imocl(n)).gt.0d0)then
            pt2ijcl(n+1)=djb(pcl(0,imap(3,2)))
c            else
c              pt2ijcl(n+1)=pt2ijcl(n)
c            endif
c            print *,'Clustering succeeded, found graph ',igscl(1)
            cluster=.true.
            return
         endif
c     calculate new ptij
c            write(*,*)'is case'
c     recalculate all in is case due to rotation & boost
            do i=1,nleft
               idi=imap(i,2)
c     never combine the two beams
               if (i.gt.2) then
c     determine all ptij
                  do j=1,i-1
                     idj=imap(j,2)
                     if (btest(mlevel,4))
     $                    write (*,*)'i = ',i,'(',idi,'), j = ',j,'(',idj,')'
c     cluster only combinable legs (acc. to diagrams)
                     icgs(0)=0
                     idij=idi+idj
c                     write (*,*) 'RECALC !!! ',idij
                     pt2ij(idij)=1.0d37
                     if (findmt(idij,icgs)) then
                       if (btest(mlevel,4)) then
                          do k=1,icgs(0)
                             write(*,*)'icg(',k,') = ',icgs(k)
                          enddo
                       endif
                        if (j.ne.1.and.j.ne.2) then
c     final state clustering                     
                           pt2ij(idij)=dj(pcl(0,idi),pcl(0,idj))
                        else
c     initial state clustering, only if hadronic collision
c     check whether 2->(n-1) process w/ cms energy > 0 remains
                           iwinp=imap(3-j,2);
                           do k=0,3
                              pcl(k,idij)=pcl(k,idj)-pcl(k,idi)
c                              pcmsp(k)=pcl(k,idij)+pcl(k,iwinp)
                           enddo
c                           ecms2=pcmsp(0)**2-pcmsp(1)**2-
c     $                          pcmsp(2)**2-pcmsp(3)**2
c                           if (ecms2.gt.0.1d0.and.
c                           if ((nleft.eq.4.or.ecms2.gt.0.1d0).and.
                           if((lpp(1).ne.0.or.lpp(2).ne.0)) then
                              pt2ij(idij)=djb(pcl(0,idi))
c     prefer clustering when outgoing in direction of incoming
                           if(sign(1d0,pcl(3,idi)).ne.sign(1d0,pcl(3,idj)))
     $                        pt2ij(idij)=pt2ij(idij)*(1d0+1d-6)
                           endif
                        endif
                        if (btest(mlevel,3)) then
                           write(*,*)'         ',idi,'&',idj,' -> ',idij,
     $                          ' pt2ij = ',pt2ij(idij)
                        endif
                     endif
                  enddo
               endif
            enddo
      enddo

      return
      end
