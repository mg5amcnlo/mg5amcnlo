      double precision function gamma(q0)
c**************************************************
c   calculates the branching probability
c**************************************************
      implicit none
      include 'nexternal.inc'
      include 'message.inc'
      include 'maxamps.inc'
      include 'cluster.inc'
      include 'sudakov.inc'
      include 'maxparticles.inc'
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
      include 'maxamps.inc'
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
      integer get_color

      isqcd=(iabs(get_color(ipdg)).gt.1)

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
      include 'genps.inc'
      include 'run.inc'

      integer ipdg, irfl

      isparton=.true.

      irfl=abs(ipdg)
      if (irfl.gt.max(asrwgtflavor,maxjetflavor).and.irfl.ne.21)
     $     isparton=.false.
c      write(*,*)'isparton? pdg = ',ipdg,' -> ',irfl,' -> ',isparton

      return
      end


      subroutine ipartupdate(p,imo,ida1,ida2,ipdg,ipart)
c**************************************************
c   Traces particle lines according to CKKW rules
c**************************************************
c     ipart gives the external particle number corresponding to the present
c     quark or gluon line.
c     For t-channel lines, ipart(1) contains the connected beam.
c     For s-channel lines, it depends if it is quark or gluon line:
c     For quark lines, ipart(2) is 0 and ipart(1) connects to the corresponding
c     final-state quark. For gluons, if it splits into two gluons,
c     it connects to the hardest gluon. If it splits into qqbar, it ipart(1) is
c     the hardest and ipart(2) is the softest.
      implicit none

      include 'ncombs.inc'
      include 'nexternal.inc'
      include 'message.inc'

      double precision p(0:3,nexternal)
      integer imo,ida1,ida2,i,idmo,idda1,idda2
      integer ipdg(n_max_cl),ipart(2,n_max_cl)
      logical isjet
      external isjet
      integer iddgluon, iddother, idgluon, idother
      logical isqcd
      external isqcd

      idmo=ipdg(imo)
      idda1=ipdg(ida1)
      idda2=ipdg(ida2)

      if (btest(mlevel,4)) then
        write(*,*) ' updating ipart for: ',ida1,ida2,' -> ',imo
      endif

      if (btest(mlevel,4)) then
         write(*,*) ' daughters: ',(ipart(i,ida1),i=1,2),(ipart(i,ida2),i=1,2)
      endif

c     IS clustering - just transmit info on incoming line
      if((ipart(1,ida1).ge.1.and.ipart(1,ida1).le.2).or.
     $   (ipart(1,ida2).ge.1.and.ipart(1,ida2).le.2))then
         ipart(2,imo)=0
         if(ipart(1,ida1).le.2.and.ipart(1,ida2).le.2)then
c           This is last clustering - keep mother ipart
            ipart(1,imo)=ipart(1,imo)
         elseif(ipart(1,ida2).ge.1.and.ipart(1,ida2).le.2)then
            ipart(1,imo)=ipart(1,ida2)        
c           Transmit jet PDG code
            if(isjet(idmo)) then
               if(idda1.lt.21.and.isjet(idda1).and.
     $              (idda2.eq.21.or.idda2.eq.22))
     $              ipdg(imo)=-idda1
               if(idda2.lt.21.and.isjet(idda2).and.
     $              (idda1.eq.21.or.idda1.eq.22))
     $              ipdg(imo)=idda2
            endif
         elseif(ipart(1,ida1).ge.1.and.ipart(1,ida1).le.2)then
            ipart(1,imo)=ipart(1,ida1)
c           Transmit jet PDG code
            if(isjet(idmo)) then
               if(idda2.lt.21.and.isjet(idda2).and.
     $            (idda1.eq.21.or.idda1.eq.22))
     $              ipdg(imo)=-idda2
               if(idda1.lt.21.and.isjet(idda1).and.
     $            (idda2.eq.21.or.idda2.eq.22))
     $              ipdg(imo)=idda1
            endif
         endif
         if (btest(mlevel,4))
     $        write(*,*) ' -> ',(ipart(i,imo),i=1,2),
     $        ' (',ipdg(imo),')'
         return
      endif        

c     FS clustering
c     Transmit parton PDG code for parton vertex
      if(isjet(idmo)) then
         if(idda1.lt.21.and.isjet(idda1).and.
     $        (idda2.eq.21.or.idda2.eq.22))
     $        ipdg(imo)=idda1
         if(idda2.lt.21.and.isjet(idda2).and.
     $        (idda1.eq.21.or.idda1.eq.22))
     $        ipdg(imo)=idda2
         idmo=ipdg(imo)
      endif

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
      else if(idmo.eq.21 .and. abs(idda1).le.6 .and.
     $        abs(idda2).le.6) then
c     gluon -> quark anti-quark: use both, but take hardest as 1
        if(p(1,ipart(1,ida1))**2+p(2,ipart(1,ida1))**2.gt.
     $     p(1,ipart(1,ida2))**2+p(2,ipart(1,ida2))**2) then
          ipart(1,imo)=ipart(1,ida1)
          ipart(2,imo)=ipart(1,ida2)
        else
          ipart(1,imo)=ipart(1,ida2)
          ipart(2,imo)=ipart(1,ida1)
        endif
      else if(idmo.eq.21.and.(idda1.eq.21.or.idda2.eq.21))then
         if(idda1.eq.21) then
            iddgluon = idda1
            idgluon = ida1
            iddother = idda2
            idother = ida2
         else
            iddgluon = idda2
            iddother = idda1
            idgluon = ida2
            idother = ida1
         endif
         if (isqcd(iddother))then
c        gluon -> gluon + scalar octet Choose hardest one
            if(p(1,ipart(1,ida1))**2+p(2,ipart(1,ida1))**2.gt.
     $         p(1,ipart(1,ida2))**2+p(2,ipart(1,ida2))**2) then
               ipart(1,imo)=ipart(1,ida1)
               ipart(2,imo)=ipart(2,ida1)
            else
               ipart(1,imo)=ipart(1,ida2)
               ipart(2,imo)=ipart(2,ida2)
            endif
         else
c        gluon -> gluon + Higgs use the gluon one
               ipart(1,imo)=ipart(1,idgluon)
               ipart(2,imo)=ipart(2,idgluon)
         endif
      else if(idmo.eq.21) then
c     gluon > octet octet Choose hardest one
            if(p(1,ipart(1,ida1))**2+p(2,ipart(1,ida1))**2.gt.
     $         p(1,ipart(1,ida2))**2+p(2,ipart(1,ida2))**2) then
               ipart(1,imo)=ipart(1,ida1)
               ipart(2,imo)=ipart(2,ida1)
            else
               ipart(1,imo)=ipart(1,ida2)
               ipart(2,imo)=ipart(2,ida2)
            endif
      else if(idmo.eq.idda1.or.idmo.eq.idda1+sign(1,idda2))then
c     quark -> quark-gluon or quark-Z or quark-h or quark-W
        ipart(1,imo)=ipart(1,ida1)
        ipart(2,imo)=0
      else if(idmo.eq.idda2.or.idmo.eq.idda2+sign(1,idda1))then
c     quark -> gluon-quark or Z-quark or h-quark or W-quark
        ipart(1,imo)=ipart(1,ida2)
        ipart(2,imo)=0
      else
c     Color singlet
         ipart(1,imo)=ipart(1,ida1)
         ipart(2,imo)=ipart(1,ida2)
      endif
      
      if (btest(mlevel,4)) then
        write(*,*) ' -> ',(ipart(i,imo),i=1,2),' (',ipdg(imo),')'
      endif

      return
      end
      
      logical function isjetvx(imo,ida1,ida2,ipdg,ipart,islast)
c***************************************************
c   Checks if a qcd vertex generates a jet
c***************************************************
      implicit none

      include 'ncombs.inc'
      include 'nexternal.inc'

      integer imo,ida1,ida2,idmo,idda1,idda2,i
      integer ipdg(n_max_cl),ipart(2,n_max_cl)
      logical isqcd,isjet,islast
      external isqcd,isjet

      idmo=ipdg(imo)
      idda1=ipdg(ida1)
      idda2=ipdg(ida2)
c     Check QCD vertex
      if(islast.or..not.isqcd(idmo).or..not.isqcd(idda1).or.
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
      if((isjet(idda1).and.(isjet(idmo).or.idmo.eq.idda2)).or.
     $   (isjet(idda2).and.(isjet(idmo).or.idmo.eq.idda1))) then
         isjetvx=.true.
      else
         isjetvx=.false.
      endif
      return
      end

      logical function ispartonvx(imo,ida1,ida2,ipdg,ipart,islast)
c***************************************************
c   Checks if a qcd vertex generates a jet
c***************************************************
      implicit none

      include 'ncombs.inc'
      include 'nexternal.inc'

      integer imo,ida1,ida2,idmo,idda1,idda2,i
      integer ipdg(n_max_cl),ipart(2,n_max_cl)
      logical isqcd,isparton,islast
      external isqcd,isparton

      idmo=ipdg(imo)
      idda1=ipdg(ida1)
      idda2=ipdg(ida2)

c     Check QCD vertex
      if(.not.isqcd(idmo).or..not.isqcd(idda1).or.
     &     .not.isqcd(idda2)) then
         ispartonvx = .false.
         return
      endif

c     IS clustering
      if((ipart(1,ida1).ge.1.and.ipart(1,ida1).le.2).or.
     $   (ipart(1,ida2).ge.1.and.ipart(1,ida2).le.2))then
c     Check if ida1 is outgoing parton or ida2 is outgoing parton
         if(.not.islast.and.ipart(1,ida2).ge.1.and.ipart(1,ida2).le.2.and.isparton(idda1).or.
     $        ipart(1,ida1).ge.1.and.ipart(1,ida1).le.2.and.isparton(idda2))then
           ispartonvx=.true.
        else
           ispartonvx=.false.
        endif
        return
      endif        

c     FS clustering
      if(isparton(idda1).or.isparton(idda2))then
         ispartonvx=.true.
      else
         ispartonvx=.false.
      endif
      
      return
      end

      integer function ifsno(n,ipart)
c***************************************************
c   Returns the FS particle number corresponding to 
c   clustering number n (=ishft(ifsno) if FS)
c***************************************************
      implicit none
      
      include 'ncombs.inc'
      include 'nexternal.inc'
      integer n,ipart(2,n_max_cl)
      integer i
      ifsno=0
      if(ipart(1,n).gt.2.and.n.eq.ishft(1,ipart(1,n)-1))
     $     ifsno=ipart(1,n)
      return
      end

      logical function setclscales(p)
c**************************************************
c     Calculate dynamic scales based on clustering
c     Also perform xqcut and xmtc cuts
c**************************************************
      implicit none

      include 'message.inc'
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include 'maxamps.inc'
      include 'cluster.inc'
      include 'run.inc'
      include 'coupl.inc'
      include 'run_config.inc'
C   
C   ARGUMENTS 
C   
      DOUBLE PRECISION P(0:3,NEXTERNAL)

C   global variables
C     Present process number
      INTEGER IMIRROR,IPROC
      COMMON/TO_MIRROR/IMIRROR, IPROC
C     ICONFIG has this config number
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
c     Common block for reweighting info
c     q2bck holds the central q2fact scales
      integer jlast(2)
      integer njetstore(lmaxconfigs),iqjetstore(nexternal-2,lmaxconfigs)
      real*8 q2bck(2)
      common /to_rw/jlast,njetstore,iqjetstore,q2bck
      data njetstore/lmaxconfigs*-1/
      real*8 xptj,xptb,xpta,xptl,xmtc
      real*8 xetamin,xqcut,deltaeta
      common /to_specxpt/xptj,xptb,xpta,xptl,xmtc,xetamin,xqcut,deltaeta
      double precision stot,m1,m2
      common/to_stot/stot,m1,m2

C   local variables
      integer i, j, idi, idj, k
      real*8 PI
      parameter( PI = 3.14159265358979323846d0 )
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      double precision asref, pt2prev(n_max_cl),pt2min
      integer n, ibeam(2), iqcd(0:2)
      integer idfl, idmap(-nexternal:nexternal)
      integer ipart(2,n_max_cl)
      double precision xnow(2),etot
      integer jfirst(2),jcentral(2),nwarning
      logical qcdline(2),partonline(2)
      logical failed,first
      data first/.true./
      data nwarning/0/
      integer nqcd(lmaxconfigs)
      include 'config_nqcd.inc'

c     Variables for keeping track of jets
      logical goodjet(n_max_cl)
      integer njets,iqjets(nexternal)
      integer fsnum(2),ida(2),imo,jcode
      logical chclusold,fail,increasecode
      save chclusold
      integer tmpindex

      logical isqcd,isjet,isparton,cluster,isjetvx
      integer ifsno
      double precision alphas
      external isqcd, isjet, isparton, cluster, isjetvx, alphas, ifsno
      setclscales=.true.

      if(ickkw.le.0.and.xqcut.le.0d0.and.q2fact(1).gt.0.and.scale.gt.0) then
         if(use_syst)then
            s_scale=scale
            n_qcd=nqcd(iconfig)
            n_alpsem=0
            do i=1,2
               n_pdfrw(i)=0
            enddo
            s_rwfact=1d0
         endif
      return
      endif
c   
c   Cluster the configuration
c   
      
c     First time, cluster according to this config and store jets
c     (following times, only accept configurations if the same partons
c      are flagged as jets)
      if(njetstore(iconfig).eq.-1)then
         chclusold=chcluster
         chcluster=.true.
      endif
 100  clustered = cluster(p(0,1))
      if(.not.clustered) then
         open(unit=26,file='../../../error',status='unknown',err=999)
         write(26,*) 'Error: Clustering failed in cluster.f.'
         write(*,*) 'Error: Clustering failed in cluster.f.'
         stop
 999     write(*,*) 'error'
         setclscales=.false.
         clustered = .false.
         return
      endif
c     Reset chcluster to run_card value
      chcluster=chclusold

      if (btest(mlevel,1)) then
        write(*,*)'setclscales: identified tree {'
        do i=1,nexternal-2
          write(*,*)'  ',i,': ',idacl(i,1),'(',ipdgcl(idacl(i,1),igraphs(1),iproc),')',
     $       '&',idacl(i,2),'(',ipdgcl(idacl(i,2),igraphs(1),iproc),')',
     $       ' -> ',imocl(i),'(',ipdgcl(imocl(i),igraphs(1),iproc),')',
     $       ', ptij = ',dsqrt(pt2ijcl(i))
          write(*,*)'   icluster(',i,')=',(icluster(j,i),j=1,4)
        enddo
        write(*,*)'  process: ',iproc
        write(*,*)'  graphs (',igraphs(0),'):',(igraphs(i),i=1,igraphs(0))
        write(*,*)'}'
        write(*,*)'iconfig is ',iconfig
      endif

C   If we have fixed factorization scale, for ickkw>0 means central
C   scale, i.e. last two scales (ren. scale for these vertices are
C   anyway already set by "scale" above)
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

c   Preparing graph particle information (ipart, needed to keep track of
c   external particle clustering scales)

c   ipart gives the external particle number corresponding to the present
c   quark or gluon line. 
c   For t-channel lines, ipart(1) contains the connected beam. 
c   For s-channel lines, it depends if it is quark or gluon line:
c   For quark lines, ipart(2) is 0 and ipart(1) connects to the corresponding
c   final-state quark. For gluons, if it splits into two gluons, 
c   it connects to the hardest gluon. If it splits into qqbar, it ipart(1) is
c   the hardest and ipart(2) is the softest.

      do i=1,nexternal
         ipart(1,ishft(1,i-1))=i
         ipart(2,ishft(1,i-1))=0
      enddo
      do n=1,nexternal-3
        call ipartupdate(p,imocl(n),idacl(n,1),idacl(n,2),
     $       ipdgcl(1,igraphs(1),iproc),ipart)
      enddo

c     Prepare beam related variables for scale and jet determination
      do i=1,2
         ibeam(i)=ishft(1,i-1)
c        jfirst is first parton splitting on this side
         jfirst(i)=0
c        jlast is last parton on this side This means
c        the last cluster which is still QCD.
         jlast(i)=0
c        jcentral is the central scale vertex on this side. i.e it stops
c        when the T channel particles is not colored anymore.
         jcentral(i)=0
c        qcdline gives whether this IS line is QCD
         qcdline(i)=isqcd(ipdgcl(ibeam(i),igraphs(1),iproc))
c        partonline gives whether this IS line is parton (start out true for any QCD)
         partonline(i)=qcdline(i)
c        goodjet gives whether this cluster line is considered a jet
c        i.e. if all related/previous clustering are jet
         goodjet(ibeam(i))=partonline(i)
      enddo

      do i=3,nexternal
         j=ishft(1,i-1)
         goodjet(j)=isjet(ipdgcl(j,igraphs(1),iproc))
      enddo

c     Go through clusterings and set factorization scale points for use in dsig
c     as well as which FS particles count as jets (from jet vertices)
      do i=1,nexternal
         iqjets(i)=0
      enddo
      if (nexternal.eq.3) goto 10
c     jcode helps keep track of how many QCD/non-QCD flips we have gone through
      jcode=1
c     increasecode gives whether we should increase jcode at next vertex
      increasecode=.false.
      do n=1,nexternal-2
        do i=1,2 ! index of the child in the interaction
          do j=1,2 ! j index of the beam
            if(idacl(n,i).eq.ibeam(j))then
c             IS clustering
              ibeam(j)=imocl(n)
c             Determine which are beam particles based on n
              if(n.lt.nexternal-2) then
                 ida(i)=idacl(n,i)
                 ida(3-i)=idacl(n,3-i)
                 imo=imocl(n)
              else
                 ida(i)=idacl(n,i)
                 ida(3-i)=imocl(n)
                 imo=idacl(n,3-i)
              endif
c             
              if(partonline(j))then
c             If jfirst not set, set it
                 if(jfirst(j).eq.0) jfirst(j)=n
c             Stop fact scale where parton line stops
                 jlast(j)=n
                 partonline(j)=goodjet(ida(3-i)).and.
     $                isjet(ipdgcl(imo,igraphs(1),iproc))
              else if (jfirst(j).eq.0) then
                 jfirst(j) = n
                 goodjet(imo)=.false.
              else
                 goodjet(imo)=.false.
              endif
c             If not jet vertex, increase jcode. This is needed
c             e.g. in VBF if we pass over to the other side and hit
c             parton vertices again.
              if(.not.goodjet(ida(3-i)).or.
     $             .not.isjet(ipdgcl(ida(i),igraphs(1),iproc)).or.
     $             .not.isjet(ipdgcl(imo,igraphs(1),iproc))) then
                  jcode=jcode+1
                  increasecode=.true.
               else if(increasecode) then
                  jcode=jcode+1
                  increasecode=.false.
               endif
c             Consider t-channel jet radiations as jets only if
c             FS line is a jet line
              if(goodjet(ida(3-i))) then
                 if(partonline(j).or.
     $ ipdgcl(ida(3-i),igraphs(1),iproc).eq.21)then
c                   Need to include gluon to avoid soft singularity
                    iqjets(ipart(1,ida(3-i)))=1 ! 1 means for sure jet
                 else
                    iqjets(ipart(1,ida(3-i)))=jcode ! jcode means possible jet
                 endif
              endif
c             Trace QCD line through event
              if(qcdline(j))then
                 jcentral(j)=n
                 qcdline(j)=isqcd(ipdgcl(imo,igraphs(1),iproc))
              endif
            endif
          enddo
        enddo
        if (imocl(n).ne.ibeam(1).and.imocl(n).ne.ibeam(2)) then
c          FS clustering
c          Check QCD jet, take care so not a decay
           if(.not.isjetvx(imocl(n),idacl(n,1),idacl(n,2),
     $        ipdgcl(1,igraphs(1),iproc),ipart,n.eq.nexternal-2)) then
c          Remove non-gluon jets that lead up to non-jet vertices
           if(ipart(1,imocl(n)).gt.2)then ! ipart(1) set and not IS line
c          The ishft gives the FS particle corresponding to imocl
              if(ipdgcl(ishft(1,ipart(1,imocl(n))-1),igraphs(1),iproc).ne.21)then
                 iqjets(ipart(1,imocl(n)))=0
              else if (ipdgcl(imocl(n),igraphs(1),iproc).eq.21)then
c                special case for g > g h remove also the hardest gluon
                 iqjets(ipart(1,imocl(n)))=0
              endif
           endif
           if(ipart(2,imocl(n)).gt.2)then ! ipart(1) set and not IS line
c             The ishft gives the FS particle corresponding to imocl
              if(ipdgcl(ishft(1,ipart(2,imocl(n))-1),igraphs(1),iproc).ne.21.or.
     $                                   ipdgcl(imocl(n),igraphs(1),iproc).ne.21) then
c                 The second condition is to prevent the case of ggh where the gluon split in quark later.
c                 The first quark is already remove so we shouldn't remove this one.      
              iqjets(ipart(2,imocl(n)))=0
              endif
           endif
c          Set goodjet to false for mother
              goodjet(imocl(n))=.false.
              cycle
           endif

c          This is a jet vertex, so set jet flag for final-state jets
c          ifsno gives leg number if daughter is FS particle, otherwise 0
           fsnum(1)=ifsno(idacl(n,1),ipart)
           if(isjet(ipdgcl(idacl(n,1),igraphs(1),iproc)).and.
     $          fsnum(1).gt.0) then
              iqjets(fsnum(1))=1
           endif
           fsnum(1)=ifsno(idacl(n,2),ipart)
           if(isjet(ipdgcl(idacl(n,2),igraphs(1),iproc)).and.
     $          fsnum(1).gt.0) then
              iqjets(fsnum(1))=1
           endif
c          Flag mother as good jet if PDG is jet and both daughters are jets
           goodjet(imocl(n))=
     $          (isjet(ipdgcl(imocl(n),igraphs(1),iproc)).and.
     $          goodjet(idacl(n,1)).and.goodjet(idacl(n,2)))
        endif
      enddo

      if (btest(mlevel,4))then
         write(*,*) 'QCD jet status (before): ',(iqjets(i),i=3,nexternal)
      endif
c     Emissions with code 1 are always jets
c     Now take care of possible jets (i.e., with code > 1)
      if(.not. partonline(1).or..not.partonline(2))then
c       First reduce jcode by one if one remaining partonline
c       (in that case accept all jets with final jcode)
        if(partonline(1).or.partonline(2)) jcode=jcode-1
c       There parton emissions with code <= jcode are not jets
         do i=3,nexternal
            if(iqjets(i).gt.1.and.iqjets(i).le.jcode)then
               iqjets(i)=0
            endif
         enddo
      endif

 10   if(jfirst(1).le.0) jfirst(1)=jlast(1)
      if(jfirst(2).le.0) jfirst(2)=jlast(2)

      if (btest(mlevel,3))
     $     write(*,*) 'jfirst is ',jfirst(1),jfirst(2),
     $     ' jlast is ',jlast(1),jlast(2),
     $     ' and jcentral is ',jcentral(1),jcentral(2)

      if (btest(mlevel,3)) then
         write(*,'(a$)') 'QCD jets (final): '
         do i=3,nexternal
            if(iqjets(i).gt.0) write(*,'(i3$)') i
         enddo
         write(*,*)
      endif
      if(njetstore(iconfig).eq.-1) then
c     Store external jet numbers if first time
         njets=0
         do i=3,nexternal
            if(iqjets(i).gt.0)then
               njets=njets+1
               iqjetstore(njets,iconfig)=i
            endif
         enddo
         njetstore(iconfig)=njets
         if (btest(mlevel,4))
     $        write(*,*) 'Storing jets: ',(iqjetstore(i,iconfig),i=1,njets)
c     Recluster without requiring chcluster
         goto 100
      else
c     Otherwise, check that we have the right jets
c     if not, recluster according to iconfig
         fail=.false.
         njets=0
         do i=1,nexternal
            if(iqjets(i).gt.0)then
               njets=njets+1
c               if (iqjetstore(njets,iconfig).ne.i) fail=.true.
            endif
         enddo
         if(njets.ne.njetstore(iconfig)) fail=.true.
         if (fail) then
            if (igraphs(1).eq.iconfig) then
               open(unit=26,file='../../../error',status='unknown',err=999)
               write(*,*) 'Error: Failed despite same graph: ',iconfig
               write(*,*) 'Have jets (>0)',(iqjets(i),i=1,nexternal)
               write(*,*) 'Should be ',
     $              (iqjetstore(i,iconfig),i=1,njetstore(iconfig))
               write(26,*) 'Error: Failed despite same graph: ',iconfig,
     $              '. Have jets (>0)',(iqjets(i),i=1,nexternal),
     $              ', should be ',
     $              (iqjetstore(i,iconfig),i=1,njetstore(iconfig))
               stop
            endif
            if (btest(mlevel,3))
     $           write(*,*) 'Bad clustering, jets fail. Reclustering ',
     $           iconfig
            chcluster=.true.
c            goto 100 ! not
         endif
      endif
      
c     If last clustering is s-channel QCD (e.g. ttbar) use mt2last instead
c     (i.e. geom. average of transverse mass of t and t~)
        if(mt2last.gt.4d0 .and. nexternal.gt.3) then
           if(jlast(1).eq.nexternal-2.and.jlast(2).eq.nexternal-2.and.
     $        isqcd(ipdgcl(idacl(nexternal-3,1),igraphs(1),iproc)).and.
     $        isqcd(ipdgcl(idacl(nexternal-3,2),igraphs(1),iproc)).and.
     $        isqcd(ipdgcl(imocl(nexternal-3),igraphs(1),iproc)))then
              mt2ij(nexternal-2)=mt2last
              mt2ij(nexternal-3)=mt2last
              if (btest(mlevel,3)) then
                 write(*,*)' setclscales: set last vertices to mtlast: ',sqrt(mt2last)
              endif
           endif
        endif

c     Set central scale to mT2
      if(jcentral(1).gt.0) then
         if(mt2ij(jcentral(1)).gt.0d0)
     $        pt2ijcl(jcentral(1))=mt2ij(jcentral(1))
      endif
      if(jcentral(2).gt.0)then
         if(mt2ij(jcentral(2)).gt.0d0)
     $     pt2ijcl(jcentral(2))=mt2ij(jcentral(2))
      endif
      if(btest(mlevel,4))then
         write(*,*) 'jlast, jcentral: ',(jlast(i),i=1,2),(jcentral(i),i=1,2)
         if(jlast(1).gt.0) write(*,*)'pt(jlast 1): ', sqrt(pt2ijcl(jlast(1)))
         if(jlast(2).gt.0) write(*,*)'pt(jlast 2): ', sqrt(pt2ijcl(jlast(2)))
         if(jcentral(1).gt.0) write(*,*)'pt(jcentral 1): ', sqrt(pt2ijcl(jcentral(1)))
         if(jcentral(2).gt.0) write(*,*)'pt(jcentral 2): ', sqrt(pt2ijcl(jcentral(2)))
      endif
c     Check xqcut for vertices with jet daughters only
      ibeam(1)=ishft(1,0)
      ibeam(2)=ishft(1,1)
      if(xqcut.gt.0) then
         do n=1,nexternal-3
c        Check if any of vertex daughters among jets
            do i=1,2
c              ifsno gives leg number if daughter is FS particle, otherwise 0
               fsnum(1)=ifsno(idacl(n,i),ipart)
               if(fsnum(1).gt.0)then
                  if(iqjets(fsnum(1)).gt.0)then
c                    Daughter among jets - check xqcut
                     if(sqrt(pt2ijcl(n)).lt.xqcut)then
                        if (btest(mlevel,3))
     $                       write(*,*) 'Failed xqcut: ',n,
     $                       ipdgcl(idacl(n,1),igraphs(1),iproc),
     $                       ipdgcl(idacl(n,2),igraphs(1),iproc),
     $                       sqrt(pt2ijcl(n))
                        setclscales=.false.
                        clustered = .false.
                        return
                     endif
                  endif
               endif
            enddo
         enddo
      endif
c     JA: Check xmtc cut for central process
      if(xmtc**2.gt.0) then
         if(jcentral(1).gt.0.and.pt2ijcl(jcentral(1)).lt.xmtc**2
     $      .or.jcentral(2).gt.0.and.pt2ijcl(jcentral(2)).lt.xmtc**2)then
            setclscales=.false.
            clustered = .false.
            if(btest(mlevel,3)) write(*,*)'Failed xmtc cut ',
     $           sqrt(pt2ijcl(jcentral(1))),sqrt(pt2ijcl(jcentral(1))),
     $           ' < ',xmtc
            return
         endif
      endif
      
      if(ickkw.eq.0.and.(fixed_fac_scale.or.q2fact(1).gt.0).and.
     $     (fixed_ren_scale.or.scale.gt.0)) return

c     Ensure that last scales are at least as big as first scales
      if(jlast(1).gt.0)
     $     pt2ijcl(jlast(1))=max(pt2ijcl(jlast(1)),pt2ijcl(jfirst(1)))
      if(jlast(2).gt.0)
     $     pt2ijcl(jlast(2))=max(pt2ijcl(jlast(2)),pt2ijcl(jfirst(2)))

      if(ickkw.gt.0.and.q2fact(1).gt.0) then
c     Use the fixed or previously set scale for central scale
         if(jcentral(1).gt.0) pt2ijcl(jcentral(1))=q2fact(1)
         if(jcentral(2).gt.0.and.jcentral(2).ne.jcentral(1))
     $        pt2ijcl(jcentral(2))=q2fact(2)
      endif

      if(nexternal.eq.3.and.nincoming.eq.2.and.q2fact(1).eq.0) then
         q2fact(1)=pt2ijcl(nexternal-2)
         q2fact(2)=pt2ijcl(nexternal-2)
      endif

      if(q2fact(1).eq.0d0) then
c     Use the geom. average of central scale and first non-radiation vertex
         if(jlast(1).gt.0) q2fact(1)=sqrt(pt2ijcl(jlast(1))*pt2ijcl(jcentral(1)))
         if(jlast(2).gt.0) q2fact(2)=sqrt(pt2ijcl(jlast(2))*pt2ijcl(jcentral(2)))
         if(jcentral(1).gt.0.and.jcentral(1).eq.jcentral(2))then
c     We have a qcd line going through the whole event, use single scale
            q2fact(1)=max(q2fact(1),q2fact(2))
            q2fact(2)=q2fact(1)
         endif
      endif
      if(.not. fixed_fac_scale) then
         q2fact(1)=scalefact**2*q2fact(1)
         q2fact(2)=scalefact**2*q2fact(2)
         q2bck(1)=q2fact(1)
         q2bck(2)=q2fact(2)
         if (btest(mlevel,3))
     $      write(*,*) 'Set central fact scales to ',sqrt(q2bck(1)),sqrt(q2bck(2))
      endif
         
c     Set renormalization scale to geom. aver. of relevant scales
      if(scale.eq.0d0) then
         if(jlast(1).gt.0.and.jlast(2).gt.0)then
c           Use geom. average of last and central scales
            scale=(pt2ijcl(jlast(1))*pt2ijcl(jcentral(1))*
     $             pt2ijcl(jlast(2))*pt2ijcl(jcentral(2)))**0.125
         elseif(jlast(1).gt.0)then
c           Use geom. average of last and central scale
            scale=(pt2ijcl(jlast(1))*pt2ijcl(jcentral(1)))**0.25
         elseif(jlast(2).gt.0)then
c           Use geom. average of last and central scale
            scale=(pt2ijcl(jlast(2))*pt2ijcl(jcentral(2)))**0.25
         elseif(jcentral(1).gt.0.and.jcentral(2).gt.0) then
c           Use geom. average of central scales
            scale=(pt2ijcl(jcentral(1))*pt2ijcl(jcentral(2)))**0.25d0
         elseif(jcentral(1).gt.0) then
            scale=sqrt(pt2ijcl(jcentral(1)))
         elseif(jcentral(2).gt.0) then
            scale=sqrt(pt2ijcl(jcentral(2)))
         else
            scale=sqrt(pt2ijcl(nexternal-2))
         endif
         scale=scalefact*scale
         if(scale.gt.0)
     $        G = SQRT(4d0*PI*ALPHAS(scale))
      endif
      if (btest(mlevel,3))
     $     write(*,*) 'Set ren scale to ',scale


c     Take care of case when jcentral are zero
      if(jcentral(1).eq.0.and.jcentral(2).eq.0)then
         if(q2fact(1).gt.0)then
            pt2ijcl(nexternal-2)=q2fact(1)
            if(nexternal.gt.3) pt2ijcl(nexternal-3)=q2fact(1)
         else
            q2fact(1)=pt2ijcl(nexternal-2)
            q2fact(2)=q2fact(1)
         endif
      elseif(jcentral(1).eq.0)then
            q2fact(1) = pt2ijcl(jfirst(1))
      elseif(jcentral(2).eq.0)then
            q2fact(2) = pt2ijcl(jfirst(2))
      elseif(ickkw.eq.2.or.(pdfwgt.and.ickkw.gt.0))then
c     Total pdf weight is f1(x1,pt2E)*fj(x1*z,Q)/fj(x1*z,pt2E)
c     f1(x1,pt2E) is given by DSIG, just need to set scale.
c     Use the minimum scale found for fact scale in ME
         if(jlast(1).gt.0.and.jfirst(1).le.jlast(1))
     $        q2fact(1)=min(pt2ijcl(jfirst(1)),q2fact(1))
         if(jlast(2).gt.0.and.jfirst(2).le.jlast(2))
     $        q2fact(2)=min(pt2ijcl(jfirst(2)),q2fact(2))
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
         clustered = .false.
         return
      endif

      if (btest(mlevel,3))
     $     write(*,*) 'Set fact scales to ',sqrt(q2fact(1)),sqrt(q2fact(2))

c
c     Store jet info for matching
c
      etot=sqrt(stot)
      do i=1,nexternal
         ptclus(i)=0d0
      enddo

      do n=1,nexternal-2
         if(n.lt.nexternal-2) then
            ida(1)=idacl(n,1)
            ida(2)=idacl(n,2)
            imo=imocl(n)
         else
            ida(1)=idacl(n,1)
            ida(2)=imocl(n)
            imo=idacl(n,2)
         endif
         do i=1,2
            do j=1,2
c              First adjust goodjet based on iqjets
               if(goodjet(ida(i)).and.ipart(j,ida(i)).gt.2)then
                  if(iqjets(ipart(j,ida(i))).eq.0) goodjet(ida(i))=.false.
               endif
c              Now reset ptclus if jet vertex
               if(ipart(j,ida(i)).gt.2) then
                  if(isjetvx(imocl(n),idacl(n,1),idacl(n,2),
     $               ipdgcl(1,igraphs(1),iproc),ipart,n.eq.nexternal-2)
     $              .and.goodjet(ida(i))) then
                     ptclus(ipart(j,ida(i)))=
     $                    max(ptclus(ipart(j,ida(i))),dsqrt(pt2ijcl(n)))
                  else if(ptclus(ipart(j,ida(i))).eq.0d0) then
                     ptclus(ipart(j,ida(i)))=etot
                  endif
                  if (btest(mlevel,3))
     $                 write(*,*) 'Set ptclus for ',ipart(j,ida(i)),
     $                 ' to ', ptclus(ipart(j,ida(i))),ida(i),goodjet(ida(i))
               endif
            enddo
         enddo
      enddo
c
c     Store information for systematics studies
c

      if(use_syst)then
         s_scale=scale
         n_qcd=nqcd(igraphs(1))
         n_alpsem=0
         do i=1,2
            n_pdfrw(i)=0
         enddo
         s_rwfact=1d0
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
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include 'maxamps.inc'
      include 'cluster.inc'
      include 'run.inc'
      include 'coupl.inc'
      include 'run_config.inc'
C   
C   ARGUMENTS 
C   
      DOUBLE PRECISION P(0:3,NEXTERNAL)

C
C   global variables
C     Present process number
      INTEGER IMIRROR,IPROC
      COMMON/TO_MIRROR/IMIRROR, IPROC
      integer              IPSEL
      COMMON /SubProc/ IPSEL
      INTEGER SUBDIAG(MAXSPROC),IB(2)
      COMMON/TO_SUB_DIAG/SUBDIAG,IB
      data IB/1,2/
C     ICONFIG has this config number
      INTEGER MCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MCONFIG, ICONFIG
c     Common block for reweighting info
c     q2bck holds the central q2fact scales
      integer jlast(2)
      integer njetstore(lmaxconfigs),iqjetstore(nexternal-2,lmaxconfigs)
      real*8 q2bck(2)
      common /to_rw/jlast,njetstore,iqjetstore,q2bck
      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'

C   local variables
      integer i, j, idi, idj,ijet
      real*8 PI
      parameter( PI = 3.14159265358979323846d0 )

      integer mapconfig(0:lmaxconfigs), this_config
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(maxsproc,-max_branch:-1,lmaxconfigs)
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
      double precision xtarget,tmp,pdfj1,pdfj2,q2now
      integer iseed,np
      data iseed/0/
      logical isvx
      logical goodjet(n_max_cl)

      logical isqcd,isjet,isparton,isjetvx,ispartonvx
      double precision alphas,getissud,pdg2pdf, sudwgt
      real xran1
      external isqcd,isjet,isparton,ispartonvx
      external alphas, isjetvx, getissud, pdg2pdf, xran1,  sudwgt

      rewgt=1.0d0
      clustered=.false.

      if(ickkw.le.0.and..not.use_syst) return

c   Set mimimum kt scale, depending on highest mult or not
      if(hmult.or.ickkw.eq.1)then
        pt2min=0
      else
        pt2min=xqcut**2
      endif
      if (btest(mlevel,3))
     $     write(*,*) 'pt2min set to ',pt2min

c   Since we use pdf reweighting, need to know particle identities
      if (btest(mlevel,1)) then
         write(*,*) 'Set process number ',ipsel
      endif

      if (use_syst.and.igraphs(1).eq.0) igraphs(1) = iconfig ! happens if use_syst=T BUT fix scale
c     Set incoming particle identities
      ipdgcl(1,igraphs(1),iproc)=idup(1,ipsel,iproc)
      ipdgcl(2,igraphs(1),iproc)=idup(2,ipsel,iproc)
      if (btest(mlevel,2)) then
         write(*,*) 'Set particle identities: ',
     $        1,ipdgcl(1,igraphs(1),iproc),
     $        ' and ',
     $        2,ipdgcl(2,igraphs(1),iproc)

      endif
      
c     Store pdf information for systematics studies (initial)
      if(use_syst)then
         do j=1,2
            n_pdfrw(j)=1
            i_pdgpdf(1,j)=ipdgcl(j,igraphs(1),iproc)
            s_xpdf(1,j)=xbk(ib(j))
            s_qpdf(1,j)=sqrt(q2fact(j))
         enddo
      endif

      if(ickkw.le.0)then
         asref=0 ! usefull for syscalc
         goto 100
      endif
c   Preparing graph particle information (ipart, needed to keep track of
c   external particle clustering scales)
      do i=1,nexternal
c        ilast(i)=ishft(1,i)
         pt2prev(ishft(1,i-1))=0d0
         if (ickkw.eq.2) then
            if(pt2min.gt.0)then
               pt2prev(ishft(1,i-1))=
     $              max(pt2min,p(0,i)**2-p(1,i)**2-p(2,i)**2-p(3,i)**2)
            endif
            pt2pdf(ishft(1,i-1))=pt2prev(ishft(1,i-1))
         else if(pdfwgt.and.ickkw.gt.0) then
            pt2pdf(ishft(1,i-1))=0d0
         endif
         ipart(1,ishft(1,i-1))=i
         ipart(2,ishft(1,i-1))=0
         if (btest(mlevel,4))
     $        write(*,*) 'Set ipart for ',ishft(1,i-1),' to ',
     $        ipart(1,ishft(1,i-1)),ipart(2,ishft(1,i-1))
      enddo
c      ilast(0)=nexternal
      ibeam(1)=ishft(1,0)
      ibeam(2)=ishft(1,1)
      if (btest(mlevel,1)) then
        write(*,*)'rewgt: identified tree {'
        do i=1,nexternal-2
          write(*,*)'  ',i,': ',idacl(i,1),'(',ipdgcl(idacl(i,1),igraphs(1),iproc),')',
     $       '&',idacl(i,2),'(',ipdgcl(idacl(i,2),igraphs(1),iproc),')',
     $       ' -> ',imocl(i),'(',ipdgcl(imocl(i),igraphs(1),iproc),')',
     $       ', ptij = ',dsqrt(pt2ijcl(i))
        enddo
        write(*,*)'  graphs (',igraphs(0),'):',(igraphs(i),i=1,igraphs(0))
        write(*,*)'}'
      endif
c     Set x values for the two sides, for IS Sudakovs
      do i=1,2
        xnow(i)=xbk(ib(i))
      enddo
      if(btest(mlevel,3))then
        write(*,*) 'Set x values to ',xnow(1),xnow(2)
      endif

c     Prepare for resetting q2fact based on PDF reweighting
      if(ickkw.eq.2)then
         q2fact(1)=0d0
         q2fact(2)=0d0
      endif

c     Prepare checking for parton vertices
      ijet=1
      do i=1,nexternal
         j=ishft(1,i-1)
c        Set jet identities according to chosen subprocess
         if(isjet(idup(i,ipsel,iproc)))
     $        ipdgcl(j,igraphs(1),iproc)=idup(i,ipsel,iproc)
         if (btest(mlevel,2))
     $        write(*,*) 'Set particle identities: ',
     $        i,ipdgcl(j,igraphs(1),iproc)
         if(i.le.2)then
            goodjet(j)=isparton(ipdgcl(j,igraphs(1),iproc))
         elseif(ijet.le.njetstore(iconfig).and.
     $        i.eq. iqjetstore(ijet,iconfig)) then
            goodjet(j)=.true.
            ijet=ijet+1
         elseif(isparton(ipdgcl(j,igraphs(1),iproc)).and.
     $          .not.isjet(ipdgcl(j,igraphs(1),iproc))) then
            goodjet(j)=.true.            
         else
            goodjet(j)=.false.
         endif
      if(btest(mlevel,4)) print *,'Set goodjet ',j,goodjet(j)
      enddo
c   
c   Set strong coupling used
c   
      asref=G**2/(4d0*PI)

c   Perform alpha_s reweighting based on type of vertex
      do n=1,nexternal-2
c       scale for alpha_s reweighting
        q2now=pt2ijcl(n)
        if(n.eq.nexternal-2) then
           q2now = scale**2
        endif
        if (btest(mlevel,3)) then
          write(*,*)'  ',n,': ',idacl(n,1),'(',ipdgcl(idacl(n,1),igraphs(1),iproc),
     &       ')&',idacl(n,2),'(',ipdgcl(idacl(n,2),igraphs(1),iproc),
     &       ') -> ',imocl(n),'(',ipdgcl(imocl(n),igraphs(1),iproc),
     &       '), ptij = ',dsqrt(q2now) 
        endif
c   Update particle tree map
        call ipartupdate(p,imocl(n),idacl(n,1),idacl(n,2),
     $       ipdgcl(1,igraphs(1),iproc),ipart)
c     perform alpha_s reweighting only for vertices where a parton is produced
c     and not for the last clustering (use non-fixed ren. scale for these)
        if (n.lt.nexternal-2)then
c          Use goodjet to trace allowed parton lines.
c          For ISR, allow only splittings where all particles are along
c          good parton lines; for FSR, just require one FS particle to be good
           goodjet(imocl(n))=isparton(ipdgcl(imocl(n),igraphs(1),iproc))
     $          .and.goodjet(idacl(n,1)).and.goodjet(idacl(n,2))
           if(btest(mlevel,4))
     $          write(*,*)'Set goodjet ',imocl(n),' to ',goodjet(imocl(n))
           if(ipart(1,imocl(n)).le.2.and.goodjet(imocl(n)).or.  ! ISR
     $        ipart(1,imocl(n)).gt.2.and.                       ! FSR
     $          ispartonvx(imocl(n),idacl(n,1),idacl(n,2),
     $          ipdgcl(1,igraphs(1),iproc),ipart,.false.).and.
     $        (goodjet(idacl(n,1)).or.goodjet(idacl(n,2)))) then
c       alpha_s weight
              rewgt=rewgt*alphas(alpsfact*sqrt(q2now))/asref
c             Store information for systematics studies
              if(use_syst)then
                 n_alpsem=n_alpsem+1
                 s_qalps(n_alpsem)=sqrt(q2now)
              endif
              if (btest(mlevel,3)) then
                 write(*,*)' reweight vertex: ',ipdgcl(imocl(n),igraphs(1),iproc),
     $                ipdgcl(idacl(n,1),igraphs(1),iproc),ipdgcl(idacl(n,2),igraphs(1),iproc)
                 write(*,*)'       as: ',alphas(alpsfact*dsqrt(q2now)),
     &                '/',asref,' -> ',alphas(alpsfact*dsqrt(q2now))/asref
                 write(*,*)' and G=',SQRT(4d0*PI*ALPHAS(scale))
              endif
           endif
        endif
        if(ickkw.eq.2.or.(pdfwgt.and.ickkw.gt.0)) then
c       Perform PDF and, if ickkw=2, Sudakov reweighting
          isvx=.false.
          do i=1,2
c         write(*,*)'weight ',idacl(n,i),', ptij=',pt2prev(idacl(n,i))
            if (isqcd(ipdgcl(idacl(n,i),igraphs(1),iproc))) then
               if(ickkw.eq.2.and.pt2min.eq.0d0) then
                  pt2min=pt2ijcl(n)
                  if (btest(mlevel,3))
     $                 write(*,*) 'pt2min set to ',pt2min
               endif
               if(ickkw.eq.2.and.pt2prev(idacl(n,i)).eq.0d0)
     $              pt2prev(idacl(n,i))=
     $              max(pt2min,p(0,i)**2-p(1,i)**2-p(2,i)**2-p(3,i)**2)
               do j=1,2
                  if (isparton(ipdgcl(idacl(n,i),igraphs(1),iproc)).and
     $                 .idacl(n,i).eq.ibeam(j)) then
c               is sudakov weight - calculate only once for each parton
c               line where parton line ends with change of parton id or
c               non-radiation vertex
                     isvx=.true.
                     ibeam(j)=imocl(n)
c                    Perform Sudakov reweighting if ickkw=2
                     if(ickkw.eq.2.and.(ipdgcl(idacl(n,i),igraphs(1),iproc).ne.
     $                    ipdgcl(imocl(n),igraphs(1),iproc).or.
     $                    .not.isjetvx(imocl(n),idacl(n,1),idacl(n,2),
     $                    ipdgcl(1,igraphs(1),iproc),ipart,n.eq.nexternal-2)).and.
     $                    pt2prev(idacl(n,i)).lt.pt2ijcl(n))then
                        tmp=min(1d0,max(getissud(ibeam(j),ipdgcl(idacl(n,i),
     $                       igraphs(1),iproc),xnow(j),xnow(3-j),pt2ijcl(n)),1d-20)/
     $                       max(getissud(ibeam(j),ipdgcl(idacl(n,i),
     $                       igraphs(1),iproc),xnow(j),xnow(3-j),pt2prev(idacl(n,i))),1d-20))
                        rewgt=rewgt*tmp
                        pt2prev(imocl(n))=pt2ijcl(n)
                        if (btest(mlevel,3)) then
                           write(*,*)' reweight line: ',ipdgcl(idacl(n,i),igraphs(1),iproc), idacl(n,i)
                           write(*,*)'     pt2prev, pt2new, x1, x2: ',pt2prev(idacl(n,i)),pt2ijcl(n),xnow(j),xnow(3-j)
                           write(*,*)'           Sud: ',tmp
                           write(*,*)'        -> rewgt: ',rewgt
                        endif
                     else if(ickkw.eq.2) then
                        pt2prev(imocl(n))=pt2prev(idacl(n,i))
                     endif
c                 End Sudakov reweighting when we reach a non-radiation vertex
                     if(ickkw.eq.2.and..not.
     $                    ispartonvx(imocl(n),idacl(n,1),idacl(n,2),
     $                    ipdgcl(1,igraphs(1),iproc),ipart,n.eq.nexternal-2)) then
                        pt2prev(imocl(n))=1d30
                        if (btest(mlevel,3)) then
                          write(*,*)' rewgt: ending reweighting for vx ',
     $                          idacl(n,1),idacl(n,2),imocl(n),
     $                          ' with ids ',ipdgcl(idacl(n,1),igraphs(1),iproc),
     $                          ipdgcl(idacl(n,2),igraphs(1),iproc),ipdgcl(imocl(n),igraphs(1),iproc)
                        endif
                     endif
c               PDF reweighting
c               Total pdf weight is f1(x1,pt2E)*fj(x1*z,Q)/fj(x1*z,pt2E)
c               f1(x1,pt2E) is given by DSIG, already set scale for that
                     if (zcl(n).gt.0d0.and.zcl(n).lt.1d0) then
                        xnow(j)=xnow(j)*zcl(n)
                     endif
c                    PDF scale
                     q2now=min(pt2ijcl(n), q2bck(j))
c                    Set PDF scale to central factorization scale
c                    if non-radiating vertex or last 2->2
                     if(n.eq.jlast(j)) then
                        q2now=q2bck(j)
                     endif
                     if (btest(mlevel,3))
     $                    write(*,*)' set q2now for pdf to ',sqrt(q2now)
                     if(q2fact(j).eq.0d0.and.ickkw.eq.2)then
                        q2fact(j)=pt2min ! Starting scale for PS
                        pt2pdf(imocl(n))=q2now
                        if (btest(mlevel,3))
     $                       write(*,*)' set fact scale ',j,
     $                          ' for PS scale to: ',sqrt(q2fact(j))
                     else if(pt2pdf(idacl(n,i)).eq.0d0)then
                        pt2pdf(imocl(n))=q2now
                        if (btest(mlevel,3))
     $                       write(*,*)' set pt2pdf for ',imocl(n),
     $                          ' to: ',sqrt(pt2pdf(imocl(n)))
                     else if(pt2pdf(idacl(n,i)).lt.q2now.and.
     $                       n.le.jlast(j))then
                        pdfj1=pdg2pdf(abs(lpp(IB(j))),ipdgcl(idacl(n,i),
     $                       igraphs(1),iproc)*sign(1,lpp(IB(j))),
     $                       xnow(j),sqrt(q2now))
                        pdfj2=pdg2pdf(abs(lpp(IB(j))),ipdgcl(idacl(n,i),
     $                       igraphs(1),iproc)*sign(1,lpp(IB(j))),
     $                       xnow(j),sqrt(pt2pdf(idacl(n,i))))
                        if(pdfj2.lt.1d-10)then
c                          Scale too low for heavy quark
                           rewgt=0d0
                           if (btest(mlevel,3))
     $                        write(*,*) 'Too low scale for quark pdf: ',
     $                        sqrt(pt2pdf(idacl(n,i))),pdfj2,pdfj1
                           return
                        endif
                        rewgt=rewgt*pdfj1/pdfj2
c     Store information for systematics studies
                        if(use_syst)then
                           n_pdfrw(j)=n_pdfrw(j)+1
                           i_pdgpdf(n_pdfrw(j),j)=ipdgcl(idacl(n,i),igraphs(1),iproc)
                           if (zcl(n).gt.0d0.and.zcl(n).lt.1d0) then
                              s_xpdf(n_pdfrw(j),j)=xnow(j)/zcl(n)
                           else
                              s_xpdf(n_pdfrw(j),j)=xnow(j) 
                           endif
                           s_qpdf(n_pdfrw(j),j)=sqrt(q2now)
                        endif
                        if (btest(mlevel,3)) then
                           write(*,*)' reweight ',n,i,ipdgcl(idacl(n,i),igraphs(1),iproc),' by pdfs: '
                           write(*,*)'     x, ptprev, ptnew: ',xnow(j),
     $                          sqrt(pt2pdf(idacl(n,i))),sqrt(q2now)
                           write(*,*)'           PDF: ',pdfj1,' / ',pdfj2
                           write(*,*)'        -> rewgt: ',rewgt
c                           write(*,*)'  (compare for glue: ',
c     $                          pdg2pdf(lpp(j),21,xbk(j),sqrt(pt2pdf(idacl(n,i)))),' / ',
c     $                          pdg2pdf(lpp(j),21,xbk(j),sqrt(pt2ijcl(n)))
c                           write(*,*)'       = ',pdg2pdf(ibeam(j),21,xbk(j),sqrt(pt2pdf(idacl(n,i))))/
c     $                          pdg2pdf(lpp(j),21,xbk(j),sqrt(pt2ijcl(n)))
c                           write(*,*)'       -> ',pdg2pdf(ibeam(j),21,xbk(j),sqrt(pt2pdf(idacl(n,i))))/
c     $                          pdg2pdf(lpp(j),21,xbk(j),sqrt(pt2ijcl(n)))*rewgt,' )'
                        endif
c                       Set scale for mother as this scale
                        pt2pdf(imocl(n))=q2now                           
                     else if(pt2pdf(idacl(n,i)).ge.q2now) then
c                    If no reweighting, just copy daughter scale for mother
                        pt2pdf(imocl(n))=pt2pdf(idacl(n,i))
                     endif
                     goto 10
                  endif
               enddo
c           fs sudakov weight
               if(ickkw.eq.2.and.pt2prev(idacl(n,i)).lt.pt2ijcl(n).and.
     $              (isvx.or.ipdgcl(idacl(n,i),igraphs(1),iproc).ne.ipdgcl(imocl(n),igraphs(1),iproc).or.
     $              (ipdgcl(idacl(n,i),igraphs(1),iproc).ne.
     $              ipdgcl(idacl(n,3-i),igraphs(1),iproc).and.
     $              pt2prev(idacl(n,i)).gt.pt2prev(idacl(n,3-i))))) then
                  tmp=sudwgt(sqrt(pt2min),sqrt(pt2prev(idacl(n,i))),
     &                 dsqrt(pt2ijcl(n)),ipdgcl(idacl(n,i),igraphs(1),iproc),1)
                  rewgt=rewgt*tmp
                  if (btest(mlevel,3)) then
                     write(*,*)' reweight fs line: ',ipdgcl(idacl(n,i),igraphs(1),iproc), idacl(n,i)
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
          if (ickkw.eq.2.and.n.eq.nexternal-2.and.isqcd(ipdgcl(imocl(n),igraphs(1),iproc)).and.
     $         pt2prev(imocl(n)).lt.pt2ijcl(n)) then
             tmp=sudwgt(sqrt(pt2min),sqrt(pt2prev(imocl(n))),
     &            dsqrt(pt2ijcl(n)),ipdgcl(imocl(n),igraphs(1),iproc),1)
             rewgt=rewgt*tmp
             if (btest(mlevel,3)) then
                write(*,*)' reweight last fs line: ',ipdgcl(imocl(n),igraphs(1),iproc), imocl(n)
                write(*,*)'     pt2prev, pt2new: ',pt2prev(imocl(n)),pt2ijcl(n)
                write(*,*)'           Sud: ',tmp
                write(*,*)'        -> rewgt: ',rewgt
             endif
          endif
        endif
      enddo

      if(ickkw.eq.2.and.lpp(1).eq.0.and.lpp(2).eq.0)then
         q2fact(1)=pt2min
         q2fact(2)=q2fact(1)
      else if (ickkw.gt.0.and.pdfwgt) then
         q2fact(1)=q2bck(1)
         q2fact(2)=q2bck(2)         
         if (btest(mlevel,3))
     $        write(*,*)' set fact scales for PS to ',
     $        sqrt(q2fact(1)),sqrt(q2fact(2))
      endif

      if (btest(mlevel,3)) then
        write(*,*)'} ->  w = ',rewgt
      endif

 100  continue

c     Set reweight factor for systematics studies
      if(use_syst)then
         s_rwfact = rewgt
         
c     Need to multiply by: initial PDF, alpha_s^n_qcd to get
c     factor in front of matrix element
         do i=1,2
            if (lpp(IB(i)).ne.0) then
                s_rwfact=s_rwfact*pdg2pdf(abs(lpp(IB(i))),
     $           i_pdgpdf(1,i)*sign(1,lpp(IB(i))),
     $           s_xpdf(1,i),s_qpdf(1,i))
            endif
         enddo
         if (asref.gt.0d0.and.n_qcd.le.nexternal)then
            s_rwfact=s_rwfact*asref**n_qcd
c         else
c            s_rwfact=0d0
         endif
      endif

      return
      end
      
