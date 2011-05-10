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

      if (btest(mlevel,6)) then
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

      if (btest(mlevel,6)) then
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

      if(q2.le.q1)then
         if(q2.lt.q1.and.btest(mlevel,4))
     $        write(*,*)'Warning! q2 < q1 in sudwgt. Return 1.'
         return
      endif

      sudwgt=sud(q0,q2,ipdg,imode)/sud(q0,q1,ipdg,imode)

      if (btest(mlevel,5)) then
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

c     Assume that QCD particles have pdg codes that are (multiples of 1M) +
c     1-10 or 21
      irfl=mod(abs(ipdg),1000000)
      if (irfl.ge.11.and.irfl.ne.21) isqcd=.false.
c      write(*,*)'iqcd? pdg = ',ipdg,' -> ',irfl,' -> ',isqcd

      return
      end

      logical function isjet(ipdg)
c**************************************************
c   determines whether particle is qcd jet particle
c**************************************************
      implicit none

      include 'cuts.inc'

      integer ipdg, irfl

      isjet=.true.

      irfl=abs(ipdg)
      if (irfl.gt.maxjetflavor.and.irfl.ne.21) isjet=.false.
c      write(*,*)'isjet? pdg = ',ipdg,' -> ',irfl,' -> ',isjet

      return
      end

      logical function isparton(ipdg)
c**************************************************
c   determines whether particle is qcd jet particle
c**************************************************
      implicit none

      include 'cuts.inc'

      integer ipdg, irfl

      isparton=.true.

      irfl=abs(ipdg)
      if (irfl.gt.5.and.irfl.ne.21) isparton=.false.
c      write(*,*)'isparton? pdg = ',ipdg,' -> ',irfl,' -> ',isparton

      return
      end


      subroutine ipartupdate(p,imo,ida1,ida2,ipdg,ipart)
c**************************************************
c   Traces particle lines according to CKKW rules
c**************************************************
      implicit none

      include 'ncombs.inc'
      include 'nexternal.inc'
      include 'message.inc'

      double precision p(0:3,nexternal)
      integer imo,ida1,ida2,i,idmo,idda1,idda2
      integer ipdg(n_max_cl),ipart(2,n_max_cl)

      do i=1,2
        ipart(i,imo)=0
      enddo

      idmo=ipdg(imo)
      idda1=ipdg(ida1)
      idda2=ipdg(ida2)

      if (btest(mlevel,1)) then
        write(*,*) ' updating ipart for: ',ida1,ida2,imo
      endif

        if (btest(mlevel,1)) then
          write(*,*) ' daughters: ',(ipart(i,ida1),i=1,2),(ipart(i,ida2),i=1,2)
        endif

c     IS clustering - just transmit info on incoming line
      if((ipart(1,ida1).ge.1.and.ipart(1,ida1).le.2).or.
     $   (ipart(1,ida2).ge.1.and.ipart(1,ida2).le.2))then
        if(ipart(1,ida2).ge.1.and.ipart(1,ida2).le.2)
     $     ipart(1,imo)=ipart(1,ida2)        
        if(ipart(1,ida1).ge.1.and.ipart(1,ida1).le.2)
     $     ipart(1,imo)=ipart(1,ida1)
        if (btest(mlevel,1)) then
          write(*,*) ' -> ',(ipart(i,imo),i=1,2)
c     Set intermediate particle identity
          if(iabs(idmo).lt.6)then
            if(iabs(idda1).lt.6) ipdg(imo)=-idda1
            if(iabs(idda2).lt.6) ipdg(imo)=-idda2
            idmo=ipdg(imo)
            if (btest(mlevel,1)) then
              write(*,*) ' particle identities: ',idda1,idda2,idmo
            endif
          endif
        endif
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
      else
c     Color singlet
         ipart(1,imo)=ipart(1,ida1)
         ipart(2,imo)=ipart(1,ida2)
      endif
      
      if (btest(mlevel,1)) then
        write(*,*) ' -> ',(ipart(i,imo),i=1,2)
      endif

c     Set intermediate particle identity
      if(iabs(idmo).lt.6)then
        if(iabs(idda1).lt.6) ipdg(imo)=idda1
        if(iabs(idda2).lt.6) ipdg(imo)=idda2
        idmo=ipdg(imo)
        if (btest(mlevel,1)) then
          write(*,*) ' particle identities: ',idda1,idda2,idmo
        endif
      endif

      return
      end
      
      logical function isjetvx(imo,ida1,ida2,ipdg,ipart)
c***************************************************
c   Checks if a qcd vertex generates a jet
c***************************************************
      implicit none

      include 'ncombs.inc'
      include 'nexternal.inc'

      integer imo,ida1,ida2,idmo,idda1,idda2,i
      integer ipdg(n_max_cl),ipart(2,n_max_cl)
      logical isqcd,isjet
      external isqcd,isjet

      idmo=ipdg(imo)
      idda1=ipdg(ida1)
      idda2=ipdg(ida2)

c     Check QCD vertex
      if(.not.isqcd(idmo).or..not.isqcd(idda1).or.
     &     .not.isqcd(idda2)) then
         isjetvx = .false.
         return
      endif

c     IS clustering
      if((ipart(1,ida1).ge.1.and.ipart(1,ida1).le.2).or.
     $   (ipart(1,ida2).ge.1.and.ipart(1,ida2).le.2))then
c     Check if ida1 is outgoing parton or ida2 is outgoing parton
        if(ipart(1,ida2).ge.1.and.ipart(1,ida2).le.2.and.isjet(idda1).or.
     $        ipart(1,ida1).ge.1.and.ipart(1,ida1).le.2.and.isjet(idda2))then
           isjetvx=.true.
        else
           isjetvx=.false.
        endif
        return
      endif        

c     FS clustering
      if(isjet(idda1).or.isjet(idda2))then
         isjetvx=.true.
      else
         isjetvx=.false.
      endif
      
      return
      end

      logical function setclscales(p)
c**************************************************
c   reweight the hard me according to ckkw
c   employing the information in common/cl_val/
c**************************************************
      implicit none

      include 'message.inc'
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include 'cluster.inc'
      include 'run.inc'
      include 'coupl.inc'
C   
C   ARGUMENTS 
C   
      DOUBLE PRECISION P(0:3,NEXTERNAL)

C   local variables
      integer i, j, idi, idj
      real*8 PI
      parameter( PI = 3.14159265358979323846d0 )

      integer mapconfig(0:lmaxconfigs), this_config
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      real*8 xptj,xptb,xpta,xptl,xmtc
      real*8 xetamin,xqcut,deltaeta
      common /to_specxpt/xptj,xptb,xpta,xptl,xmtc,xetamin,xqcut,deltaeta
      real*8 q2bck(2)
      save q2bck
      include 'maxamps.inc'
      double precision asref, pt2prev(n_max_cl),pt2min
      integer n, ibeam(2), iqcd(0:2)!, ilast(0:nexternal)
      integer idfl, idmap(-nexternal:nexternal)
      integer ipart(2,n_max_cl)
      double precision xnow(2)
      integer jlast(2),jfirst(2),nwarning
      logical qcdline(2),qcdrad(2)
      logical failed,first
      data first/.true./
      data nwarning/0/

      logical isqcd,isjet,isparton,isjetvx,cluster
      double precision alphas
      external isqcd, isjet, isparton, isjetvx, cluster, alphas

      setclscales=.true.

      if(ickkw.le.0.and.xqcut.le.0d0.and.q2fact(1).gt.0.and.scale.gt.0) return

c   
c   Cluster the configuration
c   
      
      if (.not.cluster(p(0,1))) then
c        if (xqcut.gt.0d0) then
c          failed=.false.          
cc          if(pt2ijcl(1).lt.xqcut**2) failed=.true.
c          if(failed) then
c            if (btest(mlevel,3)) then
c              write(*,*)'q_min = ',pt2ijcl(1),' < ',xqcut**2
c            endif
c            setclscales=.false.
c            return
c          endif
c        endif
c      else
        write(*,*)'setclscales: Error. Clustering failed.'
        setclscales=.false.
        return
      endif

      if (btest(mlevel,1)) then
        write(*,*)'setclscales: identified tree {'
        do i=1,nexternal-2
          write(*,*)'  ',i,': ',idacl(i,1),'(',ipdgcl(idacl(i,1),igraphs(1)),')',
     $       '&',idacl(i,2),'(',ipdgcl(idacl(i,2),igraphs(1)),')',
     $       ' -> ',imocl(i),', ptij = ',dsqrt(pt2ijcl(i)),
     $       '(',ipdgcl(imocl(i),igraphs(1)),')'
        enddo
        write(*,*)'  graphs (',igraphs(0),'):',(igraphs(i),i=1,igraphs(0))
        write(*,*)'}'
      endif

cc
cc   Set factorization scale as for the MLM case
cc
c      if(xqcut.gt.0) then
cc     Using last clustering value
c        if(pt2ijcl(nexternal-2).lt.max(4d0,xqcut**2))then
c           setclscales=.false.
c           return
c        endif

c     If last clustering is s-channel QCD (e.g. ttbar) use mt2last instead
c     (i.e. geom. average of transverse mass of t and t~)
        if(mt2last.gt.4d0 .and. nexternal.gt.3 .and. isqcd(ipdgcl(idacl(nexternal-3,1),igraphs(1)))
     $      .and. isqcd(ipdgcl(idacl(nexternal-3,2),igraphs(1)))
     $      .and. isqcd(ipdgcl(imocl(nexternal-3),igraphs(1))))then
           mt2ij(nexternal-2)=mt2last
           mt2ij(nexternal-3)=mt2last
           if (btest(mlevel,3)) then
              write(*,*)' setclscales: set last vertices to mtlast: ',sqrt(mt2last)
           endif
        endif

C   If we have fixed factorization scale, for ickkw>0 means central
C   scale, i.e. last two scales (ren. scale for these vertices are
C   anyway already set by "scale" above
      if(ickkw.gt.0) then
         if(fixed_fac_scale.and.first)then
            q2bck(1)=q2fact(1)
            q2bck(2)=q2fact(2)
            first=.false.
         else if(fixed_fac_scale) then
            q2fact(1)=q2bck(1)
            q2fact(2)=q2bck(2)
         endif
      endif
      jfirst(1)=0
      jfirst(2)=0

      ibeam(1)=ishft(1,0)
      ibeam(2)=ishft(1,1)
      jlast(1)=0
      jlast(2)=0
      qcdline(1)=.false.
      qcdline(2)=.false.
      qcdrad(1)=.true.
      qcdrad(2)=.true.

c   Go through clusterings and set factorization scales for use in dsig
      do n=1,nexternal-2
        do i=1,2
          if (isqcd(ipdgcl(idacl(n,i),igraphs(1)))) then
            do j=1,2
              if ((isparton(ipdgcl(idacl(n,i),igraphs(1))).and.idacl(n,i).eq.ibeam(j).or.
     $           isparton(ipdgcl(imocl(n),igraphs(1))).and.imocl(n).eq.ibeam(j))
     $           .and.qcdrad(j)) then
c             is emission - this is what we want
c             Total pdf weight is f1(x1,pt2E)*fj(x1*z,Q)/fj(x1*z,pt2E)
c             f1(x1,pt2E) is given by DSIG, just need to set scale.
                 ibeam(j)=imocl(n)
                 if(ickkw.eq.0.or.jfirst(j).eq.0) jfirst(j)=n
                 jlast(j)=n
                 qcdline(j)=isqcd(ipdgcl(imocl(n),igraphs(1)))
                 if(n.lt.nexternal-2)then
                    qcdrad(j)=isqcd(ipdgcl(idacl(n,3-i),igraphs(1)))
                 endif
              else
                 qcdline(j)=isqcd(ipdgcl(imocl(n),igraphs(1)))
              endif
            enddo
          endif
        enddo
      enddo

      if (btest(mlevel,3))
     $     write(*,*) 'jfirst is ',jfirst(1),jfirst(2),
     $     ' and jlast is ',jlast(1),jlast(2)

c     Set central scale to mT2 and multiply with scalefact
      if(jlast(1).gt.0.and.mt2ij(jlast(1)).gt.0d0)
     $     pt2ijcl(jlast(1))=mt2ij(jlast(1))
      if(jlast(2).gt.0.and.mt2ij(jlast(2)).gt.0d0)
     $     pt2ijcl(jlast(2))=mt2ij(jlast(2))
      if(btest(mlevel,4))
     $     print *,'pt2ijcl is: ',jlast(1), sqrt(pt2ijcl(jlast(1))),
     $     jlast(2), sqrt(pt2ijcl(jlast(2)))
      if(qcdline(1).and.qcdline(2).and.jlast(1).ne.jlast(2)) then
c     If not WBF or similar, set uniform scale to be maximum
         pt2ijcl(jlast(1))=max(pt2ijcl(jlast(1)),pt2ijcl(jlast(2)))
         pt2ijcl(jlast(2))=pt2ijcl(jlast(1))
      endif
      if(jlast(1).gt.0) pt2ijcl(jlast(1))=scalefact**2*pt2ijcl(jlast(1))
      if(jlast(2).gt.0) pt2ijcl(jlast(2))=scalefact**2*pt2ijcl(jlast(2))
      if(lpp(1).eq.0.and.lpp(2).eq.0)then
         pt2ijcl(nexternal-2)=scalefact**2*pt2ijcl(nexternal-2)
         pt2ijcl(nexternal-3)=pt2ijcl(nexternal-2)
      endif

c     Check xqcut for vertices with jet daughters only
      if(xqcut.gt.0) then
         do n=1,nexternal-2
            if (n.lt.nexternal-2.and.n.ne.jlast(1).and.n.ne.jlast(2).and.
     $           (isjet(ipdgcl(idacl(n,1),igraphs(1))).or.isjet(ipdgcl(idacl(n,2),igraphs(1)))).and.
     $           sqrt(pt2ijcl(n)).lt.xqcut)then
               setclscales=.false.
               return
            endif
         enddo
      endif

c     JA: Check xmtc cut for central process
      if(pt2ijcl(jlast(1)).lt.xmtc**2.or.pt2ijcl(jlast(2)).lt.xmtc**2)then
         setclscales=.false.
         if(btest(mlevel,3)) write(*,*)'Failed xmtc cut ',
     $        sqrt(pt2ijcl(jlast(1))),sqrt(pt2ijcl(jlast(1))),' < ',xmtc
         return
      endif
      
      if(ickkw.eq.0.and.(fixed_fac_scale.or.q2fact(1).gt.0).and.
     $     (fixed_ren_scale.or.scale.gt.0)) return

c     Set renormalization scale to geom. aver. of factorization scales
      if(scale.eq.0d0) then
         if(jlast(1).gt.0.and.jlast(2).gt.0) then
            scale=(pt2ijcl(jlast(1))*pt2ijcl(jlast(2)))**0.25d0
         elseif(jlast(1).gt.0) then
            scale=sqrt(pt2ijcl(jlast(1)))
         elseif(jlast(2).gt.0) then
            scale=sqrt(pt2ijcl(jlast(2)))
         elseif(lpp(1).eq.0.and.lpp(2).eq.0) then
            scale=sqrt(pt2ijcl(nexternal-2))
         endif
         if(scale.gt.0)
     $        G = SQRT(4d0*PI*ALPHAS(scale))
      endif
      if (btest(mlevel,3))
     $     write(*,*) 'Set ren scale to ',scale

      if(ickkw.gt.0.and.q2fact(1).gt.0) then
c     Use the fixed or previously set scale for central scale
         if(jlast(1).gt.0) pt2ijcl(jlast(1))=q2fact(1)
         if(jlast(2).gt.0) pt2ijcl(jlast(2))=q2fact(2)
      endif
      
      if(lpp(1).eq.0.and.lpp(2).eq.0)then
         if(q2fact(1).gt.0)then
            pt2ijcl(nexternal-2)=q2fact(1)
            pt2ijcl(nexternal-3)=q2fact(1)
         else
            q2fact(1)=pt2ijcl(nexternal-2)
            q2fact(2)=q2fact(1)
         endif
      elseif(ickkw.eq.2.or.pdfwgt)then
c     Use the minimum scale found for fact scale in ME
         if(jlast(1).gt.0) q2fact(1)=min(pt2ijcl(jfirst(1)),pt2ijcl(jlast(1)))
         if(jlast(2).gt.0) q2fact(2)=min(pt2ijcl(jfirst(2)),pt2ijcl(jlast(2)))
      else if(q2fact(1).eq.0d0) then
         if(jlast(1).gt.0) q2fact(1)=pt2ijcl(jlast(1))
         if(jlast(2).gt.0) q2fact(2)=pt2ijcl(jlast(2))
      endif

      if(nexternal.eq.3.and.nincoming.eq.2) then
         if(q2fact(1).eq.0)
     $        q2fact(1)=pt2ijcl(nexternal-2)
         if(q2fact(2).eq.0)
     $        q2fact(2)=pt2ijcl(nexternal-2)
      endif

c     Check that factorization scale is >= 2 GeV
      if(lpp(1).ne.0.and.q2fact(1).lt.4d0.or.
     $   lpp(2).ne.0.and.q2fact(2).lt.4d0)then
         if(nwarning.le.10) then
             nwarning=nwarning+1
             write(*,*) 'Warning: Too low fact scales: ',
     $            sqrt(q2fact(1)), sqrt(q2fact(2))
          endif
         if(nwarning.eq.11) then
             nwarning=nwarning+1
             write(*,*) 'No more warnings written out this run.'
          endif
         setclscales=.false.
         return
      endif

      if (btest(mlevel,3))
     $     write(*,*) 'Set fact scales to ',sqrt(q2fact(1)),sqrt(q2fact(2))
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
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include 'cluster.inc'
      include 'run.inc'
      include 'coupl.inc'
      include 'maxamps.inc'
C   
C   ARGUMENTS 
C   
      DOUBLE PRECISION P(0:3,NEXTERNAL)

C   global variables
      integer              IPROC 
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SubProc/ PD, IPROC

C   local variables
      integer i, j, idi, idj
      real*8 PI
      parameter( PI = 3.14159265358979323846d0 )

      integer mapconfig(0:lmaxconfigs), this_config
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      include 'configs.inc'
      real*8 xptj,xptb,xpta,xptl,xmtc
      real*8 xetamin,xqcut,deltaeta
      common /to_specxpt/xptj,xptb,xpta,xptl,xmtc,xetamin,xqcut,deltaeta
      double precision asref, pt2prev(n_max_cl),pt2pdf(n_max_cl),pt2min
      integer n, ibeam(2), iqcd(0:2)!, ilast(0:nexternal)
      integer idfl, idmap(-nexternal:nexternal)
c     ipart gives external particle number chain
      integer ipart(2,n_max_cl)
      double precision xnow(2)
      double precision xtarget, tmp
      integer iseed,np
      data iseed/0/
      logical isvx

      logical isqcd,isjet,isparton,isjetvx
      double precision alphas,getissud,pdg2pdf, sudwgt
      real xran1
      external isqcd,isjet,isparton
      external alphas, isjetvx, getissud, pdg2pdf, xran1,  sudwgt

      rewgt=1.0d0

      if(ickkw.le.0) return

      if(.not.clustered)then
        write(*,*)'Error: No clustering done when calling rewgt!'
        stop
      endif
      clustered=.false.

c   Set mimimum kt scale, depending on highest mult or not
      if(hmult.or.ickkw.eq.1)then
        pt2min=0
      else
        pt2min=xqcut**2
      endif
      if (btest(mlevel,3))
     $     write(*,*) 'pt2min set to ',pt2min

c   Since we use pdf reweighting, need to know particle identities
      iprocset=1
      np = iproc
      xtarget=xran1(iseed)*pd(np)
      iprocset = 1
      do while (pd(iprocset) .lt. xtarget .and. iprocset .lt. np)
         iprocset=iprocset+1
      enddo
      if (btest(mlevel,1)) then
         write(*,*) 'Set process number ',iprocset
      endif

c   Preparing graph particle information (ipart, needed to keep track of
c   external particle clustering scales)
      do i=1,nexternal
c        ilast(i)=ishft(1,i)
         if(pt2min.gt.0)then
            pt2prev(ishft(1,i-1))=max(pt2min,p(0,i)**2-p(1,i)**2-p(2,i)**2-p(3,i)**2)
         else
            pt2prev(ishft(1,i-1))=0d0
         endif
         pt2pdf(ishft(1,i-1))=pt2prev(ishft(1,i-1))
         ptclus(i)=sqrt(pt2prev(ishft(1,i-1)))
         ipart(1,ishft(1,i-1))=i
         ipart(2,ishft(1,i-1))=0
      enddo
c      ilast(0)=nexternal
      ibeam(1)=ishft(1,0)
      ibeam(2)=ishft(1,1)
      if (btest(mlevel,1)) then
        write(*,*)'rewgt: identified tree {'
        do i=1,nexternal-2
          write(*,*)'  ',i,': ',idacl(i,1),'&',idacl(i,2),
     &       ' -> ',imocl(i),', ptij = ',dsqrt(pt2ijcl(i)) 
        enddo
        write(*,*)'  graphs (',igraphs(0),'):',(igraphs(i),i=1,igraphs(0))
        write(*,*)'}'
      endif
c     Set x values for the two sides, for IS Sudakovs
      do i=1,2
        xnow(i)=xbk(i)
      enddo
      if(btest(mlevel,3))then
        write(*,*) 'Set x values to ',xnow(1),xnow(2)
      endif

c     Prepare for resetting q2fact based on PDF reweighting
      if(ickkw.eq.2.or.pdfwgt)then
         q2fact(1)=0d0
         q2fact(2)=0d0
      endif
c   
c   Set strong coupling used
c   
      asref=G**2/(4d0*PI)

c   Perform alpha_s reweighting based on type of vertex
      do n=1,nexternal-2
        if (btest(mlevel,3)) then
          write(*,*)'  ',n,': ',idacl(n,1),'(',ipdgcl(idacl(n,1),igraphs(1)),
     &       ')&',idacl(n,2),'(',ipdgcl(idacl(n,2),igraphs(1)),') -> ',
     &       imocl(n),'(',ipdgcl(imocl(n),igraphs(1)),'), ptij = ',
     &       dsqrt(pt2ijcl(n)) 
        endif
c     perform alpha_s reweighting only for vertices where a jet is produced
c     and not for the last clustering (use non-fixed ren. scale for these)
        if (n.lt.nexternal-2.and.
     $     isjetvx(imocl(n),idacl(n,1),idacl(n,2),ipdgcl(1,igraphs(1)),ipart)) then
c       alpha_s weight
          rewgt=rewgt*alphas(alpsfact*sqrt(pt2ijcl(n)))/asref
          if (btest(mlevel,3)) then
             write(*,*)' reweight vertex: ',ipdgcl(imocl(n),igraphs(1)),ipdgcl(idacl(n,1),igraphs(1)),ipdgcl(idacl(n,2),igraphs(1))
            write(*,*)'       as: ',alphas(alpsfact*dsqrt(pt2ijcl(n))),
     &         '/',asref,' -> ',alphas(alpsfact*dsqrt(pt2ijcl(n)))/asref
            write(*,*)' and G=',SQRT(4d0*PI*ALPHAS(scale))
          endif
        endif
c   Update starting values for FS parton showering
        do i=1,2
          do j=1,2
            if(ipart(j,idacl(n,i)).gt.0)then
              ptclus(ipart(j,idacl(n,i)))=dsqrt(pt2ijcl(n))
            endif
          enddo
        enddo
c   Update particle tree map
        call ipartupdate(p,imocl(n),idacl(n,1),idacl(n,2),ipdgcl(1,igraphs(1)),ipart)
        if(ickkw.eq.2.or.pdfwgt) then
c       Perform PDF and, if ickkw=2, Sudakov reweighting
          isvx=.false.
          do i=1,2
c         write(*,*)'weight ',idacl(n,i),', ptij=',pt2prev(idacl(n,i))
            if (isqcd(ipdgcl(idacl(n,i),igraphs(1)))) then
               if(pt2min.eq.0d0) then
                  pt2min=pt2ijcl(n)
                  if (btest(mlevel,3))
     $                 write(*,*) 'pt2min set to ',pt2min
               endif
               if(pt2prev(idacl(n,i)).eq.0d0) pt2prev(idacl(n,i))=
     $              max(pt2min,p(0,i)**2-p(1,i)**2-p(2,i)**2-p(3,i)**2)
               do j=1,2
                  if (isparton(ipdgcl(idacl(n,i),igraphs(1))).and
     $                 .idacl(n,i).eq.ibeam(j)) then
c               is sudakov weight - calculate only once for each parton
c               line where parton line ends with change of parton id or
c               non-radiation vertex
                     isvx=.true.
                     ibeam(j)=imocl(n)
                     if(pt2pdf(idacl(n,i)).eq.0d0) pt2pdf(idacl(n,i))=pt2prev(idacl(n,i))
                     if(ickkw.eq.2.and.(ipdgcl(idacl(n,i),igraphs(1)).ne.
     $                    ipdgcl(imocl(n),igraphs(1)).or.
     $                    .not.isjetvx(imocl(n),idacl(n,1),idacl(n,2),
     $                    ipdgcl(1,igraphs(1)),ipart)).and.
     $                    pt2prev(idacl(n,i)).lt.pt2ijcl(n).and.zcl(n).gt.1d-20)then
                        tmp=min(1d0,max(getissud(ibeam(j),ipdgcl(idacl(n,i),
     $                       igraphs(1)),xnow(j),xnow(3-j),pt2ijcl(n)),1d-20)/
     $                       max(getissud(ibeam(j),ipdgcl(idacl(n,i),
     $                       igraphs(1)),xnow(j),xnow(3-j),pt2prev(idacl(n,i))),1d-20))
                        rewgt=rewgt*tmp
                        pt2prev(imocl(n))=pt2ijcl(n)
                        if (btest(mlevel,3)) then
                           write(*,*)' reweight line: ',ipdgcl(idacl(n,i),igraphs(1)), idacl(n,i)
                           write(*,*)'     pt2prev, pt2new, x1, x2: ',pt2prev(idacl(n,i)),pt2ijcl(n),xnow(j),xnow(3-j)
                           write(*,*)'           Sud: ',tmp
                           write(*,*)'        -> rewgt: ',rewgt
                        endif
                     else
                        pt2prev(imocl(n))=pt2prev(idacl(n,i))
                     endif
c               Total pdf weight is f1(x1,pt2E)*fj(x1*z,Q)/fj(x1*z,pt2E)
c               f1(x1,pt2E) is given by DSIG, already set scale for that
                     xnow(j)=xnow(j)*zcl(n)
                     if(q2fact(j).eq.0d0.and.ickkw.eq.2)then
                        q2fact(j)=pt2min ! Starting scale for PS
                        if (btest(mlevel,3)) then
                           write(*,*)' reweight: set fact scale ',j,' for PS scale to: ',q2fact(j)
                        endif
                     else if(q2fact(j).eq.0d0)then
                        q2fact(j)=pt2ijcl(n)
                     else if(pt2pdf(idacl(n,i)).lt.pt2ijcl(n).and.zcl(n).gt.1d-20) then
                        if(ickkw.eq.1) q2fact(j)=pt2ijcl(n)
                        rewgt=rewgt*max(pdg2pdf(abs(ibeam(j)),ipdgcl(idacl(n,i),
     $                       igraphs(1))*sign(1,ibeam(j)),xnow(j),sqrt(pt2ijcl(n))),1d-20)/
     $                       max(pdg2pdf(abs(ibeam(j)),ipdgcl(idacl(n,i),
     $                       igraphs(1))*sign(1,ibeam(j)),xnow(j),
     $                       sqrt(pt2pdf(idacl(n,i)))),1d-20)
                        if (btest(mlevel,3)) then
                           write(*,*)' reweight ',ipdgcl(idacl(n,i),igraphs(1)),' by pdfs: '
                           write(*,*)'     x, pt2prev, ptnew: ',xnow(j),pt2pdf(idacl(n,i)),pt2ijcl(n)
                           write(*,*)'           PDF: ',
     $                          pdg2pdf(abs(ibeam(j)),ipdgcl(idacl(n,i),
     $                          igraphs(1))*sign(1,ibeam(j)),xnow(j),sqrt(pt2ijcl(n))),' / ',
     $                          pdg2pdf(abs(ibeam(j)),ipdgcl(idacl(n,i),
     $                          igraphs(1))*sign(1,ibeam(j)),xnow(j),sqrt(pt2pdf(idacl(n,i))))
                           write(*,*)'        -> rewgt: ',rewgt
c                           write(*,*)'  (compare for glue: ',
c     $                          pdg2pdf(ibeam(j),21,xbk(j),sqrt(pt2pdf(idacl(n,i)))),' / ',
c     $                          pdg2pdf(ibeam(j),21,xbk(j),sqrt(pt2ijcl(n)))
c                           write(*,*)'       = ',pdg2pdf(ibeam(j),21,xbk(j),sqrt(pt2pdf(idacl(n,i))))/
c     $                          pdg2pdf(ibeam(j),21,xbk(j),sqrt(pt2ijcl(n)))
c                           write(*,*)'       -> ',pdg2pdf(ibeam(j),21,xbk(j),sqrt(pt2pdf(idacl(n,i))))/
c     $                          pdg2pdf(ibeam(j),21,xbk(j),sqrt(pt2ijcl(n)))*rewgt,' )'
                        endif
                     endif
c               End both Sudakov and pdf reweighting when we reach a
c               non-radiation vertex
                     if(isjetvx(imocl(n),idacl(n,1),idacl(n,2),ipdgcl(1,igraphs(1)),ipart)) then
                        pt2pdf(imocl(n))=pt2ijcl(n)
                     else
                        pt2pdf(imocl(n))=1d30
                        pt2prev(imocl(n))=1d30
                        if (btest(mlevel,3)) then
                           write(*,*)' rewgt: for vertex ',idacl(n,1),idacl(n,2),imocl(n),
     $                          ' with ids ',ipdgcl(idacl(n,1),igraphs(1)),
     $                          ipdgcl(idacl(n,2),igraphs(1)),ipdgcl(imocl(n),igraphs(1))
                           write(*,*)'    set pt2prev, pt2pdf: ',pt2prev(imocl(n)),pt2pdf(imocl(n))
                        endif
                     endif
                     goto 10
                  endif
               enddo
c           fs sudakov weight
               if(ickkw.eq.2.and.pt2prev(idacl(n,i)).lt.pt2ijcl(n).and.
     $              (isvx.or.ipdgcl(idacl(n,i),igraphs(1)).ne.ipdgcl(imocl(n),igraphs(1)).or.
     $              (ipdgcl(idacl(n,i),igraphs(1)).ne.
     $              ipdgcl(idacl(n,3-i),igraphs(1)).and.
     $              pt2prev(idacl(n,i)).gt.pt2prev(idacl(n,3-i))))) then
                  tmp=sudwgt(sqrt(pt2min),sqrt(pt2prev(idacl(n,i))),
     &                 dsqrt(pt2ijcl(n)),ipdgcl(idacl(n,i),igraphs(1)),1)
                  rewgt=rewgt*tmp
                  if (btest(mlevel,3)) then
                     write(*,*)' reweight fs line: ',ipdgcl(idacl(n,i),igraphs(1)), idacl(n,i)
                     write(*,*)'     pt2prev, pt2new: ',pt2prev(idacl(n,i)),pt2ijcl(n)
                     write(*,*)'           Sud: ',tmp
                     write(*,*)'        -> rewgt: ',rewgt
                  endif
                  pt2prev(imocl(n))=pt2ijcl(n)
               else
                  pt2prev(imocl(n))=pt2prev(idacl(n,i))
               endif 
            endif
 10         continue
          enddo
          if (ickkw.eq.2.and.n.eq.nexternal-2.and.isqcd(ipdgcl(imocl(n),igraphs(1))).and.
     $         pt2prev(imocl(n)).lt.pt2ijcl(n)) then
             tmp=sudwgt(sqrt(pt2min),sqrt(pt2prev(imocl(n))),
     &            dsqrt(pt2ijcl(n)),ipdgcl(imocl(n),igraphs(1)),1)
             rewgt=rewgt*tmp
             if (btest(mlevel,3)) then
                write(*,*)' reweight last fs line: ',ipdgcl(imocl(n),igraphs(1)), imocl(n)
                write(*,*)'     pt2prev, pt2new: ',pt2prev(imocl(n)),pt2ijcl(n)
                write(*,*)'           Sud: ',tmp
                write(*,*)'        -> rewgt: ',rewgt
             endif
          endif
        endif
      enddo

      if((ickkw.eq.2.or.pdfwgt).and.lpp(1).eq.0.and.lpp(2).eq.0)then
         q2fact(1)=pt2min
         q2fact(2)=q2fact(1)
      endif

      if (btest(mlevel,3)) then
        write(*,*)'} ->  w = ',rewgt
      endif
      return
      end
      
