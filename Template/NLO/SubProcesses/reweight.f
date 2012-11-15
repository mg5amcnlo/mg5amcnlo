      double precision function gamma(q0)
c**************************************************
c   calculates the branching probability
c**************************************************
      implicit none
      include 'nexternal.inc'
      include 'message.inc'
      include 'cluster.inc'
      include 'sudakov.inc'
      include 'run.inc'
      integer i
      double precision q0, val, add, add2
      double precision qr,lf
      double precision alphas
      external alphas
      double precision pi
      parameter (pi=3.141592654d0)

      gamma=0.0d0

      if (Q1<m_qmass(iipdg)) return
      m_lastas=Alphas(alpsfact*q0)
      val=2d0*m_colfac(iipdg)*m_lastas/PI/q0
c   if (m_mode & bpm::power_corrs) then
      qr=q0/Q1
      if(m_pca(iipdg,iimode).eq.0)then
        lf=log(1d0/qr-1d0)
      else 
        lf=log(1d0/qr)
      endif
      val=val*(m_dlog(iipdg)*(1d0+m_kfac*m_lastas/(2d0*PI))*lf+m_slog(iipdg)
     $   +qr*(m_power(iipdg,1,iimode)+qr*(m_power(iipdg,2,iimode)
     $   +qr*m_power(iipdg,3,iimode))))
c   else
c   val=val*m_dlog*(1d0+m_kfac*m_lastas/(2d0*PI))*log(Q1/q0)+m_slog;
c   endif
      if(m_qmass(iipdg).gt.0d0)then
        val=val+m_colfac(iipdg)*m_lastas/PI/q0*(0.5-q0/m_qmass(iipdg)*
     $     atan(m_qmass(iipdg)/q0)-
     $     (1.0-0.5*(q0/m_qmass(iipdg))**2)*log(1.0+(m_qmass(iipdg)/q0)**2))
      endif
      val=max(val,0d0)
      if (iipdg.eq.21) then
        add=0d0
        do i=-6,-1
          if(m_qmass(abs(i)).gt.0d0)then
            add2=m_colfac(i)*m_lastas/PI/q0/
     $         (1.0+(m_qmass(abs(i))/q0)**2)*
     $         (1.0-1.0/3.0/(1.0+(m_qmass(abs(i))/q0)**2))
          else
            add2=2d0*m_colfac(i)*m_lastas/PI/q0*(m_slog(i)
     $         +qr*(m_power(i,1,iimode)+qr*(m_power(i,2,iimode)
     $         +qr*m_power(i,3,iimode))))
          endif
          add=add+max(add2,0d0)
        enddo
        val=val+add
      endif
      
      gamma = max(val,0d0)

      if (btest(mlevel,3)) then
        write(*,*)'       \\Delta^I_{',iipdg,'}(',
     &     q0,',',q1,') -> ',gamma
        write(*,*) val,m_lastas,m_dlog(iipdg),m_slog(iipdg)
        write(*,*) m_power(iipdg,1,iimode),m_power(iipdg,2,iimode),m_power(iipdg,3,iimode)
      endif

      return
      end

      double precision function sud(q0,Q11,ipdg,imode)
c**************************************************
c   actually calculates is sudakov weight
c**************************************************
      implicit none
      include 'message.inc'
      include 'nexternal.inc'
      include 'cluster.inc'      
      integer ipdg,imode
      double precision q0, Q11
      double precision gamma,DGAUSS
      external gamma,DGAUSS
      double precision eps
      parameter (eps=1d-5)
      
      sud=0.0d0

      Q1=Q11
      iipdg=iabs(ipdg)
      iimode=imode

      sud=exp(-DGAUSS(gamma,q0,Q1,eps))

      if (btest(mlevel,2)) then
        write(*,*)'       \\Delta^',imode,'_{',ipdg,'}(',
     &     2*log10(q0/q1),') -> ',sud
      endif

      return
      end

      double precision function sudwgt(q0,q1,q2,ipdg,imode)
c**************************************************
c   calculates is sudakov weight
c**************************************************
      implicit none
      include 'message.inc'
      integer ipdg,imode
      double precision q0, q1, q2
      double precision sud
      external sud
      
      sudwgt=1.0d0

      sudwgt=sud(q0,q2,ipdg,imode)/sud(q0,q1,ipdg,imode)

      if (btest(mlevel,2)) then
        write(*,*)'       \\Delta^',imode,'_{',ipdg,'}(',
     &     q0,',',q1,',',q2,') -> ',sudwgt
      endif

      return
      end

      logical function isqcd(ipdg)
c**************************************************
c   determines whether particle is qcd particle
c**************************************************
      implicit none
      integer ipdg, irfl

      isqcd=.true.

      irfl=iand(abs(ipdg),63)
      if (irfl.gt.8.and.irfl.ne.21) isqcd=.false.
c      write(*,*)'iqcd? pdg = ',ipdg,' -> ',irfl,' -> ',isqcd

      return
      end

      logical function isjet(ipdg)
c**************************************************
c   determines whether particle is qcd jet particle
c**************************************************
      implicit none
      integer ipdg, irfl

      isjet=.true.

      irfl=abs(ipdg)
      if (irfl.gt.5.and.irfl.ne.21) isjet=.false.
c      write(*,*)'isjet? pdg = ',ipdg,' -> ',irfl,' -> ',isjet

      return
      end


      subroutine ipartupdate(p,imo,ida1,ida2,idmo,idda1,idda2,ipart)
c**************************************************
c   Traces particle lines according to CKKW rules
c**************************************************
      implicit none

      include 'ncombs.inc'
      include 'nexternal.inc'

      double precision p(0:3,nexternal)
      integer imo,ida1,ida2,idmo,idda1,idda2,i
      integer ipart(2,n_max_cl)

      do i=1,2
        ipart(i,imo)=0
      enddo

c     IS clustering - just transmit info on incoming line
      if((ipart(1,ida1).ge.1.and.ipart(1,ida1).le.2).or.
     $   (ipart(1,ida2).ge.1.and.ipart(1,ida2).le.2))then
        if(ipart(1,ida2).ge.1.and.ipart(1,ida2).le.2)
     $     ipart(1,imo)=ipart(1,ida2)        
        if(ipart(1,ida1).ge.1.and.ipart(1,ida1).le.2)
     $     ipart(1,imo)=ipart(1,ida1)
        return
      endif        

c     FS clustering
      if(idmo.eq.21.and.idda1.eq.21.and.idda2.eq.21)then
c     gluon -> 2 gluon splitting: Choose hardest gluon
        if(p(1,ipart(1,ida1))**2+p(2,ipart(1,ida1))**2.gt.
     $     p(1,ipart(1,ida2))**2+p(2,ipart(1,ida2))**2) then
          ipart(1,imo)=ipart(1,ida1)
          ipart(2,imo)=ipart(2,ida1)
        else
          ipart(1,imo)=ipart(1,ida2)
          ipart(2,imo)=ipart(2,ida2)
        endif
      else if(idmo.eq.21.and.idda1.eq.-idda2)then
c     gluon -> quark anti-quark: use both, but take hardest as 1
        if(p(1,ipart(1,ida1))**2+p(2,ipart(1,ida1))**2.gt.
     $     p(1,ipart(1,ida2))**2+p(2,ipart(1,ida2))**2) then
          ipart(1,imo)=ipart(1,ida1)
          ipart(2,imo)=ipart(1,ida2)
        else
          ipart(1,imo)=ipart(1,ida2)
          ipart(2,imo)=ipart(1,ida1)
        endif
      else if(idmo.eq.idda1.or.idmo.eq.idda1+sign(1,idda2))then
c     quark -> quark-gluon or quark-Z or quark-h or quark-W
        ipart(1,imo)=ipart(1,ida1)
      else if(idmo.eq.idda2.or.idmo.eq.idda2+sign(1,idda1))then
c     quark -> gluon-quark or Z-quark or h-quark or W-quark
        ipart(1,imo)=ipart(1,ida2)
      endif
      
      return
      end
      
      logical function isqcdjetvx(imo,ida1,ida2,idmo,idda1,idda2,ipart)
c***************************************************
c   Checks if a qcd vertex generates a jet
c***************************************************
      implicit none

      include 'ncombs.inc'
      include 'nexternal.inc'

      integer imo,ida1,ida2,idmo,idda1,idda2,i
      integer ipart(2,n_max_cl)
      logical isqcd,isjet
      external isqcd,isjet

c     Check QCD vertex
      if(.not.isqcd(idmo).or..not.isqcd(idda1).or.
     &     .not.isqcd(idda2)) then
         isqcdjetvx = .false.
         return
      endif

c     IS clustering
      if((ipart(1,ida1).ge.1.and.ipart(1,ida1).le.2).or.
     $   (ipart(1,ida2).ge.1.and.ipart(1,ida2).le.2))then
c     Check if ida1 is outgoing parton or ida2 is outgoing parton
        if(ipart(1,ida2).ge.1.and.ipart(1,ida2).le.2.and.isjet(idda1).or.
     $        ipart(1,ida1).ge.1.and.ipart(1,ida1).le.2.and.isjet(idda2))then
           isqcdjetvx=.true.
        else
           isqcdjetvx=.false.
        endif
        return
      endif        

c     FS clustering
      if(isjet(idda1).or.isjet(idda2))then
         isqcdjetvx=.true.
      else
         isqcdjetvx=.false.
      endif
      
      return
      end

      double precision function rewgt(p)
c**************************************************
c   reweight the hard me according to ckkw
c   employing the information in common/cl_val/
c**************************************************
      implicit none

      include 'message.inc'
      include 'genps.inc'
      include 'nexternal.inc'
      include 'cluster.inc'
      include 'run.inc'
      include 'coupl.inc'
C   
C   ARGUMENTS 
C   
      DOUBLE PRECISION P(0:3,NEXTERNAL)

      integer i, j, idi, idj
      real*8 PI
      parameter( PI = 3.14159265358979323846d0 )

      integer mapconfig(0:lmaxconfigs), this_config
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      include 'configs.inc'
      real*8 xptj,xptb,xpta,xptl
      real*8 xetamin,xqcut,deltaeta
      common /to_specxpt/xptj,xptb,xpta,xptl,xetamin,xqcut,deltaeta
      integer    maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc)
      integer mothup(2,nexternal,maxproc)
      integer icolup(2,nexternal,maxflow)
      include 'leshouche.inc'
      double precision asref, ptprev(n_max_cl),ptmin
      integer n, icmp, ibeam(2), iqcd(0:2), ilast(0:nexternal)
      integer idfl, ipdg(n_max_cl), idmap(-nexternal:nexternal)
      integer ipart(2,n_max_cl)

      logical isqcd,isjet,isqcdjetvx,failed
      double precision alphas, sudwgt
      external isqcd, isjet, alphas, sudwgt, isqcdjetvx

      logical cluster
      external cluster

      rewgt=1.0d0

      if(ickkw.le.0.and.xqcut.le.0d0) return

c   
c   Cluster the configuration
c   
      
      if (cluster(p(0,1))) then
        if (xqcut.gt.0d0) then
          failed=.false.
          do i=1,nexternal-3
            if(pt2ijcl(i).lt.xqcut**2) failed=.true.
          enddo
          if(failed) then
            if (btest(mlevel,3)) then
              write(*,*)'q_min = ',pt2ijcl(1),' < ',xqcut**2
            endif
            rewgt=0d0
            return
          endif
        endif
      else
        write(*,*)'reweight.f: Error. Clustering failed.'
        rewgt=0d0
        return
      endif

c   Preparing graph particle information
      do i=1,nexternal
        ilast(i)=ishft(1,i)
        ptprev(ishft(1,i))=ptmin
        ptclus(i)=ptmin
        ipart(1,ishft(1,i))=i
        ipart(2,ishft(1,i))=0
      enddo
      ilast(0)=nexternal
      ibeam(1)=1
      ibeam(2)=2
      if (btest(mlevel,1)) then
        write(*,*)'identified tree {'
        do i=1,nexternal-3
          write(*,*)'  ',i,': ',idacl(i,1),'&',idacl(i,2),
     &       ' -> ',imocl(i),', ptij = ',dsqrt(pt2ijcl(i)) 
        enddo
        write(*,*)'  graphs (',igscl(0),'):',(igscl(i),i=1,igscl(0))
        write(*,*)'}'
c        write(*,*)'sudakov reweighting {'
      endif
c   fill particle information
      icmp=ishft(1,nexternal+1)-2
      do i=1,nexternal
        idmap(i)=ishft(1,i)
        ipdg(idmap(i))=idup(i,1)
c     write(*,*) i,' got id ',idmap(i),' -> ',ipdg(idmap(i))
      enddo
      do i=1,nexternal-3
        idi=iforest(1,-i,igscl(1))
        idj=iforest(2,-i,igscl(1))
        idmap(-i)=idmap(idi)+idmap(idj)
        idfl=sprop(-i,igscl(1))
        if (idfl.ne.0) then
          ipdg(idmap(-i))=idfl
          ipdg(icmp-idmap(-i))=idfl
        endif
        idfl=tprid(-i,igscl(1))
        if (idfl.ne.0) then
          ipdg(idmap(-i))=idfl
          ipdg(icmp-idmap(-i))=idfl
        endif
c     write(*,*) -i,' (',idi,',',idj,') got id ',idmap(-i),
c     &        ' -> ',ipdg(idmap(-i))
      enddo

c
c   Set factorization scale as for the MLM case
c
      if(ickkw.lt.2.and.xqcut.gt.0) then
cc     Using geometrical average of last two clusterings
c        q2fact(1)=scalefact**2*sqrt(pt2ijcl(nexternal-3)*pt2ijcl(nexternal-2))
c     Using last clustering value
        if(pt2ijcl(nexternal-2).lt.max(4d0,xqcut**2)
     $        .and.(.not.fixed_fac_scale.or..not.fixed_ren_scale))then
          write(*,*) 'Warning! Last clustering scale ',pt2ijcl(nexternal-2),
     $       ' set to ',pt2ijcl(nexternal-3)
          pt2ijcl(nexternal-2)=pt2ijcl(nexternal-3)
        endif
c     If last clustering is s-channel QCD (e.g. ttbar) use mtlast instead
c     (i.e. geom. average of transverse mass of t and t~)
        if(mtlast.gt.4d0 .and. isqcd(ipdg(idacl(nexternal-3,1)))
     $      .and. isqcd(ipdg(idacl(nexternal-3,2)))
     $      .and. isqcd(ipdg(imocl(nexternal-3))))then
           pt2ijcl(nexternal-2)=mtlast        
         endif
c     Set non-fixed factorization scale
        if(.not.fixed_fac_scale) then
           q2fact(1)=scalefact**2*pt2ijcl(nexternal-2)
           q2fact(2)=q2fact(1)
           if(q2fact(1).lt.4d0)then
              write(*,*)'Too small fact scale: ',q2fact(1)
              stop
           endif
        endif
        if(.not.fixed_ren_scale)then
c     In case of matching this is used for last vertex and all vertices
c     where a jet is not produced. 
           scale=scalefact*sqrt(pt2ijcl(nexternal-2))
           if(scale.lt.2d0)then
              write(*,*)'Error: Too small ren scale: ',scale
              stop
           endif
           G = SQRT(4d0*PI*ALPHAS(scale))
        endif
      endif

      if(ickkw.le.0) return

      if(sqrt(pt2ijcl(1)).lt.2d0)then
        write(*,*)'reweight.f: Error. Too small clustering variable, q = ',
     $     sqrt(pt2ijcl(1))
        write(*,*)'xqcut = ',xqcut
        rewgt=0d0
        return
      endif

c   
c   Set strong coupling used
c   
      asref=G**2/(4d0*PI)

c   Set mimimum kt scale, depending on highest mult or not
      if(hmult)then
        ptmin=dsqrt(pt2ijcl(1))
      else
        ptmin=xqcut
      endif

c   Set factorization scale to minimum kt scale, according to CKKW prescription
      if(ickkw.eq.2)then
        q2fact(1)=ptmin**2
        q2fact(2)=q2fact(1)
        if(q2fact(1).lt.4d0)then
           write(*,*)'Too small fact scale: ',q2fact(1)
           stop
        endif
      endif

c      njets=0
c   Perform alpha_s reweighting based on type of vertex
      do n=1,nexternal-2
        if (btest(mlevel,3)) then
          write(*,*)'  ',n,': ',idacl(n,1),'(',ipdg(idacl(n,1)),
     &       ')&',idacl(n,2),'(',ipdg(idacl(n,2)),') -> ',
     &       imocl(n),'(',ipdg(imocl(n)),'), ptij = ',
     &       dsqrt(pt2ijcl(n)) 
        endif
c     perform alpha_s reweighting only for vertices where a jet is produced
c     and not for the last clustering (use non-fixed ren. scale for these)
        if (n.lt.nexternal-2.and.isqcdjetvx(imocl(n),idacl(n,1),idacl(n,2),
     $     ipdg(imocl(n)),ipdg(idacl(n,1)),ipdg(idacl(n,2)),ipart)) then
c       alpha_s weight
          rewgt=rewgt*alphas(alpsfact*sqrt(pt2ijcl(n)))/asref
          if (btest(mlevel,3)) then
             write(*,*)' reweight vertex: ',ipdg(imocl(n)),ipdg(idacl(n,1)),ipdg(idacl(n,2))
            write(*,*)'       as: ',alphas(alpsfact*dsqrt(pt2ijcl(n))),
     &         '/',asref,' -> ',alphas(alpsfact*dsqrt(pt2ijcl(n)))/asref
            write(*,*)' and G=',SQRT(4d0*PI*ALPHAS(scale))
          endif
        endif
cc   Store qcd jet clustering values in ptjets vector
c        if (isjet(ipdg(idacl(n,1)))) then
c          njets=njets+1
c          ptjets(njets)=dsqrt(pt2ijcl(n))
c        endif
c   Update starting values for FS parton showering
        do i=1,2
          do j=1,2
            if(ipart(j,idacl(n,i)).gt.0)then
              ptclus(ipart(j,idacl(n,i)))=dsqrt(pt2ijcl(n))
            endif
          enddo
        enddo
c   Update particle tree map
        call ipartupdate(p,imocl(n),idacl(n,1),idacl(n,2),
     $     ipdg(imocl(n)),ipdg(idacl(n,1)),ipdg(idacl(n,2)),ipart)

c       establish relations
        do i=1,ilast(0)
          if (ilast(i).eq.idacl(n,1)) ilast(i)=imocl(n)
          if (ilast(i).eq.idacl(n,2)) then
            do j=i+1,ilast(0)
              ilast(j-1)=ilast(j)
            enddo
          endif
        enddo
        ilast(0)=ilast(0)-1
c       do i=1,ilast(0)
c       write(*,*)'last ',i,' -> ',ilast(i)
c       enddo
        if(ickkw.eq.2)then
c       Perform Sudakov reweighting
          do i=1,2
c         write(*,*)'weight ',idacl(n,i),', ptij=',ptprev(idacl(n,i))
            if (isqcd(ipdg(idacl(n,i)))) then
              do j=1,2
                if (idacl(n,i).eq.ibeam(j)) then
c               is sudakov weight
                  rewgt=rewgt*sudwgt(ptmin,ptprev(idacl(n,i)),
     &               dsqrt(pt2ijcl(n)),ipdg(idacl(n,i)),2)
                  ibeam(j)=imocl(n)
                  goto 10
                endif
              enddo
c           fs sudakov weight
              rewgt=rewgt*sudwgt(ptmin,ptprev(idacl(n,i)),
     &           dsqrt(pt2ijcl(n)),ipdg(idacl(n,i)),1)
            endif
 10         ptprev(imocl(n))=dsqrt(pt2ijcl(n))
          enddo
        endif
      enddo

      if (btest(mlevel,3)) then
        write(*,*)'} ->  w = ',rewgt
      endif
      return
      end
      
