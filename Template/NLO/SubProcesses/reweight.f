      double precision function gamma(q0)
c*******************************************************
c   Calculates the argument of the exponent
c   o. It takes only terms logarithmic in log(Q/q0) and 
c      constants into account (up to NLL) (iimode=2).
c   o. For iimode=1, only the O(alphaS) expansion is
c      included, with the alphaS stripped off.
c******************************************************
      implicit none
c Include
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'nFKSconfigs.inc'
      include 'cluster.inc'
c Constants
      double precision ZERO,PI,CA,CF,kappa
      parameter (ZERO=0d0)
      parameter (PI = 3.14159265358979323846d0)
      parameter (CA = 3d0)
      parameter (CF = 4d0/3d0)
      parameter (kappa = CA*(67d0/18d0 - pi**2/6d0) - NF*5d0/9d0)
c Argument
      double precision q0
c Local
      logical isgluon
      integer i
      double precision alphasq0,mu,qom,moq2
c External
      double precision alphas
      external alphas
c Data 
      double precision mass(6)
      data (mass(i),i=1,6) /0d0,0d0,0d0,0d0,4.5d0,173d0/

      gamma=0.0d0
      if (q0.ge.Q1) then
         return
      endif

c Compute alphaS at the scale of the branching; with a freeze-out at 0.5
c GeV
      if (iimode.eq.2) then
         alphasq0=alphas(max(q0,0.5d0))
      elseif(iimode.eq.1) then
         alphasq0=1d0
      else
         write (*,*) 'Unknown imode for Sudakov',iimode
         stop
      endif

      if (iipdg.eq.21) then
c Gluon sudakov         
c     g->gg contribution
         gamma=CA*alphasq0*log(Q1/q0)/pi                     ! A1
     &        -CA*alphasq0*11d0/(12d0*pi)                    ! B1
         if (iimode.eq.2) then
             gamma=gamma+CA*alphasq0**2*log(Q1/q0)*kappa/(2d0*pi**2)   ! A2
          endif
c     g->qqbar contribution
         do i=1,6
            if (i.le.NF) then   ! massless splitting
               gamma=gamma+alphasq0/(6d0*pi)   ! B1
            else                ! massive splitting
               moq2=(mass(i)/q0)**2
               gamma=gamma+alphasq0/(4d0*pi)/(1d0+moq2)*
     &              (1d0 - 1d0/(3d0*(1+moq2)))  ! B1
            endif
         enddo
      elseif(abs(iipdg).le.6) then
c Quark Sudakov
         gamma=CF*alphasq0*log(Q1/q0)/pi                   ! A1
     &        -CF*alphasq0*3d0/(4d0*pi)                    ! B1
         if (iimode.eq.2) then
            gamma=gamma+CF*alphasq0**2*log(Q1/q0)*kappa/(2d0*pi**2) ! A2
         endif
         if (iipdg.gt.NF) then ! include mass effects
            qom=q0/mass(iipdg)
            gamma=gamma+CF*alphasq0/pi/2d0*( 0.5d0 - qom*atan(1d0/qom) -
     $           (1d0-0.5d0*qom**2)*log(1d0+1d0/qom**2) )
         endif
      else
         write (*,*) 'ERROR in reweight.f: do not know'/
     $        /' which Sudakov to compute',iipdg
         write (*,*) 'FxFx is not supported for models'/
     $        /' with coloured BSM particles'
         stop 1
      endif
c Integration is over dq^2/q^2 = 2*dq/q, so factor 2. Also, include
c already the minus sign here
      gamma=-gamma*2d0/q0
      return
      end


      double precision function sud(q0,Q11,ipdg,imode)
c**************************************************
c   actually calculates is sudakov weight
c**************************************************
      implicit none
      include 'message.inc'
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
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

      sud=exp(DGAUSS(gamma,q0,Q1,eps))

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

      sudwgt=sud(q0,q2,ipdg,2)/sud(q0,q1,ipdg,2)

      if (btest(mlevel,5)) then
        write(*,*)'       \\Delta^',imode,'_{',ipdg,'}(',
     &     q0,',',q1,',',q2,') -> ',sudwgt
      endif

      return
      end

      double precision function sud_exp(q0,Q11,ipdg,imode)
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'cluster.inc'      
      integer ipdg,imode
      double precision q0, Q11
      double precision gamma,DGAUSS
      external gamma,DGAUSS
      double precision eps
      parameter (eps=1d-5)
      sud_exp=0.0d0
      Q1=Q11
      iipdg=iabs(ipdg)
      iimode=imode
      sud_exp=DGAUSS(gamma,q0,Q1,eps)
      return
      end


      double precision function sudwgt_exp(q0,q1,q2,ipdg,imode)
c**************************************************
c   calculates is sudakov weight expanded up to order alpha_s
c   (not including the alpha_s factor itself).
c**************************************************
      implicit none
      include 'message.inc'
      integer ipdg,imode
      double precision q0, q1, q2
      double precision sud_exp
      external sud_exp
      
      sudwgt_exp=0d0

      if(q2.le.q1)then
         if(q2.lt.q1.and.btest(mlevel,4))
     $        write(*,*)'Warning! q2 < q1 in sudwgt. Return 0.'
         return
      endif

      sudwgt_exp=sud_exp(q0,q2,ipdg,imode) - sud_exp(q0,q1,ipdg,imode)

      if (btest(mlevel,5)) then
        write(*,*)'       \\Delta^',imode,'_{',ipdg,'}(',
     &     q0,',',q1,',',q2,')|alphs_s -> ',sudwgt_exp
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
         write(*,*) ' daughters: ',(ipart(i,ida1),i=1,2),
     &        (ipart(i,ida2),i=1,2)
      endif

c     Initial State clustering - just transmit info on incoming line
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

c     Final State clustering
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
         if ( p(1,ipart(1,ida1))**2+p(2,ipart(1,ida1))**2.gt.
     $        p(1,ipart(1,ida2))**2+p(2,ipart(1,ida2))**2 ) then
            ipart(1,imo)=ipart(1,ida1)
            ipart(2,imo)=ipart(2,ida1)
         else
            ipart(1,imo)=ipart(1,ida2)
            ipart(2,imo)=ipart(2,ida2)
         endif
      else if(idmo.eq.21 .and. abs(idda1).le.6 .and.
     $        abs(idda2).le.6)then
c     gluon -> quark anti-quark: use both, but take hardest as 1
         if ( p(1,ipart(1,ida1))**2+p(2,ipart(1,ida1))**2.gt.
     $        p(1,ipart(1,ida2))**2+p(2,ipart(1,ida2))**2 ) then
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
         if (isqcd(idother))then
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

c     Initinal State clustering
      if((ipart(1,ida1).ge.1.and.ipart(1,ida1).le.2).or.
     $   (ipart(1,ida2).ge.1.and.ipart(1,ida2).le.2))then
c     Check if ida1 is outgoing parton or ida2 is outgoing parton
         if((ipart(1,ida2).ge.1 .and.ipart(1,ida2).le.2
     $        .and.isjet(idda1)).or.
     $      (ipart(1,ida1).ge.1 .and.ipart(1,ida1).le.2
     $        .and.isjet(idda2)))then
           isjetvx=.true.
        else
           isjetvx=.false.
        endif
        return
      endif        
c     Final State clustering
      if((isjet(idda1).and.(isjet(idmo).or.idmo.eq.idda2)).or.(
     $   isjet(idda2).and.(isjet(idmo).or.idmo.eq.idda1))) then
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
         if(.not.islast.and.ipart(1,ida2).ge.1
     $        .and.ipart(1,ida2).le.2.and.isparton(idda1).or.ipart(1
     $        ,ida1).ge.1.and.ipart(1
     $        ,ida1).le.2.and.isparton(idda2))then
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
c     Calculate dynamic scales used in the central
c     hard (lowest-multiplicity) process.
c**************************************************
      implicit none
c Include
      include 'nexternal.inc'
      include 'maxparticles.inc'
      include 'maxconfigs.inc'
      include 'run.inc'
      include 'nFKSconfigs.inc'
      include 'cluster.inc'
      include 'message.inc'
      include 'coupl.inc'
      include 'real_from_born_configs.inc'
      include 'cuts.inc'
      double precision ZERO,PI
      parameter (ZERO=0d0)
      parameter( PI = 3.14159265358979323846d0 )
c Argument
      double precision p(0:3,nexternal)
c Local
      integer i,j,n,ibeam(2),jfirst(2),jcentral(2),ipart(2
     $     ,n_max_cl),iconf,nqcd(n_max_cl)
      logical partonline(2),qcdline(2),first
      data first /.true./
      double precision pt_tmp,etot
      integer nwarning
      data nwarning /0/
c     Variables for keeping track of jets
      logical goodjet(n_max_cl)
      integer njets,iqjets(nexternal)
      integer fsnum(2),ida(2),imo,jcode
      logical chclusold,fail,increasecode
      save chclusold
c Common
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config
      integer njetstore(fks_configs,lmaxconfigs),iqjetstore(nexternal-2
     $     ,fks_configs,lmaxconfigs),jlast(2)
      integer tot_conf
      parameter (tot_conf=lmaxconfigs*fks_configs)
      data njetstore/tot_conf*-1/
      real*8 q2bck(2)
      common /to_rw/jlast,njetstore,iqjetstore,q2bck
      logical use_syst
      DOUBLE PRECISION s_scale
      INTEGER n_qcd,n_alpsem
      DOUBLE PRECISION s_qalps(max_particles-2)
      INTEGER n_pdfrw(2),i_pdgpdf(max_particles-2,2)
      DOUBLE PRECISION s_xpdf(max_particles-2,2),s_qpdf(max_particles-2,2)
      DOUBLE PRECISION s_rwfact
      COMMON/TO_SYST/use_syst,n_qcd,n_alpsem,n_pdfrw,i_pdgpdf,
     $               s_scale,s_qalps,s_xpdf,s_qpdf,s_rwfact
c External
      logical cluster,isqcd,isparton,isjetvx,isjet,ispartonvx
      integer ifsno
      double precision alphas
      external cluster,isqcd,isparton,isjetvx,isjet,alphas,ispartonvx
     $     ,ifsno
c FxFx
      integer nFxFx_ren_scales
      double precision FxFx_ren_scales(0:nexternal),FxFx_fac_scale(2)
      common/c_FxFx_scales/FxFx_ren_scales,nFxFx_ren_scales
     $     ,FxFx_fac_scale
c
      setclscales=.true.

      if(ickkw.le.0.and.xqcut.le.0d0.and.q2fact(1).gt.0.and.scale.gt.0)
     $     return

      if (ickkw.eq.3) then
         use_syst=.false.
         chcluster=.false.
      endif
c   
c   Cluster the configuration
c   
      iconf=real_from_born_conf(this_config,nFKSprocess)
c     First time, cluster according to this config and store jets
c     (following times, only accept configurations if the same partons
c      are flagged as jets)
      if(njetstore(nFKSprocess,iconf).eq.-1)then
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
            write(*,*)'  ',i,': ',idacl(i,1),'(',ipdgcl(idacl(i,1)
     $           ,igraphs(1),nFKSprocess),')','&',idacl(i,2),'('
     $           ,ipdgcl(idacl(i,2),igraphs(1),nFKSprocess),')',' -> '
     $           ,imocl(i),'(',ipdgcl(imocl(i),igraphs(1),nFKSprocess)
     $           ,')' ,', ptij = ',dsqrt(pt2ijcl(i))
          write(*,*)'   icluster(',i,')=',(icluster(j,i),j=1,4)
         enddo
         write(*,*)'  process: ',nFKSprocess
         write(*,*)'  graphs (',igraphs(0),'):',(igraphs(i),i=1
     $        ,igraphs(0))
         write(*,*)'}'
        write(*,*)'iconf is ',iconf,nFKSprocess
        do i=1,nexternal
           write (*,*) i,(p(j,i),j=0,3)
        enddo
      endif

C     If we have fixed factorization scale, for ickkw>0 means central
C     scale, i.e. last two scales (ren. scale for these vertices are
C     anyway already set by "scale" above) (Do not do this for FxFx
C     merging)
        if(ickkw.gt.0 .and. ickkw.ne.3) then
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
     $       ipdgcl(1,igraphs(1),nFKSprocess),ipart)
      enddo
      
c     Prepare beam related variables for scale and jet determination
      do i=1,2
         ibeam(i)=ishft(1,i-1)
c     jfirst is first parton splitting on this side
         jfirst(i)=0
c     jlast is last parton on this side
         jlast(i)=0
c     jcentral is the central scale vertex on this side
         jcentral(i)=0
c     qcdline gives whether this IS line is QCD
         qcdline(i)=isqcd(ipdgcl(ibeam(i),igraphs(1),nFKSprocess))
c     partonline gives whether this IS line is parton (start out true for any QCD)
         partonline(i)=qcdline(i)
c     goodjet gives whether this cluster line is considered a jet
         goodjet(ibeam(i))=partonline(i)
      enddo

      do i=3,nexternal
         j=ishft(1,i-1)
         goodjet(j)=isjet(ipdgcl(j,igraphs(1),nFKSprocess))
      enddo

c     For FxFx merging, we need to know what the first QCD clustering is
c     (this one is non-physical for the S-events) because that one
c     should be skipped.
      ifxfx(1)=-1
      ifxfx(2)=-1
      pt_tmp=99d9
      do n=1,nexternal-2        ! loop over cluster nodes
         if (ispartonvx(imocl(n),idacl(n,1),idacl(n,2),ipdgcl(1
     $        ,igraphs(1),nFKSprocess),ipart,.false.)) then
            if (ifxfx(1).lt.0 .or. pt2ijcl(n).lt.pt_tmp) then
               ifxfx(1)=n
               pt_tmp=pt2ijcl(n)
            endif
         endif
      enddo
      pt_tmp=99d9
      do n=1,nexternal-2        ! loop over cluster nodes
         if (ispartonvx(imocl(n),idacl(n,1),idacl(n,2),ipdgcl(1
     $        ,igraphs(1),nFKSprocess),ipart,.false.)) then
            if ((ifxfx(2).lt.0 .or. pt2ijcl(n).lt.pt_tmp) .and.
     $           n.ne.ifxfx(1)) then
               ifxfx(2)=n
               pt_tmp=pt2ijcl(n)
            endif
         endif
      enddo
      if (btest(mlevel,3)) then
         write(*,*)'setclscales: ifxfx has been found to be',ifxfx(1)
     $        ,ifxfx(2)
      endif

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
      do n=1,nexternal-2        ! loop over cluster nodes
         do i=1,2               ! loop over daughters in clusterings
            do j=1,2            ! loop over the two initial state partons
               if (idacl(n,i).eq.ibeam(j)) then
c     Initial State clustering
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
                  if(partonline(j))then
c     If jfirst not set, set it
                     if((jfirst(j).eq.0 .and. ickkw.ne.3) .or.
     $                    (jfirst(j).eq.0 .and. ickkw.eq.3 .and.
     $                    ifxfx(1).ne.n))jfirst(j)=n
c     Stop fact scale where parton line stops
                     jlast(j)=n
                     partonline(j)=goodjet(ida(3-i)).and.
     $                isjet(ipdgcl(imo,igraphs(1),nFKSprocess))
                  else
                     goodjet(imo)=.false.
                  endif
c     If not jet vertex, increase jcode. This is needed e.g. in VBF if
c     we pass over to the other side and hit parton vertices again.
                  if(.not.goodjet(ida(3-i)).or. .not.isjet(ipdgcl(ida(i)
     $                 ,igraphs(1),nFKSprocess)).or.
     $                 .not.isjet(ipdgcl(imo,igraphs(1),nFKSprocess)))
     $                 then
                     jcode=jcode+1
                     increasecode=.true.
                  else if(increasecode) then
                     jcode=jcode+1
                     increasecode=.false.
                  endif
c     Consider t-channel jet radiations as jets only if FS line is a jet
c     line
                  if(goodjet(ida(3-i))) then
                     if(partonline(j).or.
     $                    ipdgcl(ida(3-i),igraphs(1),nFKSprocess).eq.21)then
c     Need to include gluon to avoid soft singularity
                        iqjets(ipart(1,ida(3-i)))=1 ! 1 means for sure jet
                     else
                        iqjets(ipart(1,ida(3-i)))=jcode ! jcode means possible jet
                     endif
                  endif
c     Trace QCD line through event
                  if(qcdline(j))then
                     jcentral(j)=n
                     qcdline(j)=isqcd(ipdgcl(imo,igraphs(1),nFKSprocess))
                  endif
               endif
            enddo
         enddo
         if (imocl(n).ne.ibeam(1).and.imocl(n).ne.ibeam(2)) then
c     Final State clustering
c           Check QCD jet, take care so not a decay
            if(.not.isjetvx(imocl(n),idacl(n,1),idacl(n,2),
     $           ipdgcl(1,igraphs(1),nFKSprocess),ipart,n.eq.nexternal-2)) then
c     Remove non-gluon jets that lead up to non-jet vertices
               if(ipart(1,imocl(n)).gt.2)then ! ipart(1) set and not IS line
c     The ishft gives the FS particle corresponding to imocl
                  if(.not.isjet(ipdgcl(ishft(1,ipart(1,imocl(n))-1)
     $                 ,igraphs(1),nFKSprocess)))
     &                 iqjets(ipart(1,imocl(n)))=0
               endif
               if(ipart(2,imocl(n)).gt.2)then ! ipart(2) set and not IS line
c     The ishft gives the FS particle corresponding to imocl
                  if(.not.isjet(ipdgcl(ishft(1,ipart(2,imocl(n))-1)
     $                 ,igraphs(1),nFKSprocess))) 
     &                 iqjets(ipart(2,imocl(n)))=0
               endif
c     Set goodjet to false for mother
               goodjet(imocl(n))=.false.
               cycle
            endif
c     This is a jet vertex, so set jet flag for final-state jets ifsno
c     gives leg number if daughter is FS particle, otherwise 0
            fsnum(1)=ifsno(idacl(n,1),ipart)
            if(isjet(ipdgcl(idacl(n,1),igraphs(1),nFKSprocess)).and.
     $           fsnum(1).gt.0)  iqjets(fsnum(1))=1
            fsnum(1)=ifsno(idacl(n,2),ipart)
            if(isjet(ipdgcl(idacl(n,2),igraphs(1),nFKSprocess)).and.
     $           fsnum(1).gt.0) iqjets(fsnum(1))=1
c     Flag mother as good jet if PDG is jet and both daughters are jets
            goodjet(imocl(n))=
     $           (isjet(ipdgcl(imocl(n),igraphs(1),nFKSprocess)).and.
     $           goodjet(idacl(n,1)).and.goodjet(idacl(n,2)))
         endif
      enddo

      if (btest(mlevel,4))then
         write(*,*)'QCD jet status (before): ',(iqjets(i),i=3,nexternal)
      endif
c     Emissions with code 1 are always jets Now take care of possible
c     jets (i.e., with code > 1)
      if(.not. partonline(1).or..not.partonline(2))then
c     First reduce jcode by one if one remaining partonline (in that
c     case accept all jets with final jcode)
         if(partonline(1).or.partonline(2)) jcode=jcode-1
c     There parton emissions with code <= jcode are not jets
         do i=3,nexternal
            if(iqjets(i).gt.1.and.iqjets(i).le.jcode) iqjets(i)=0
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
      if(njetstore(nFKSprocess,iconf).eq.-1) then
c     Store external jet numbers if first time
         njets=0
         do i=3,nexternal
            if(iqjets(i).gt.0)then
               njets=njets+1
               iqjetstore(njets,nFKSprocess,iconf)=i
            endif
         enddo
         njetstore(nFKSprocess,iconf)=njets
         if (btest(mlevel,4)) write(*,*)'Storing jets: ',(iqjetstore(i
     $        ,nFKSprocess,iconf),i=1,njets)
c     Recluster without requiring chcluster
         goto 100
      else
c     Otherwise, check that we have the right jets
c     if not, recluster according to iconf
         fail=.false.
         njets=0
         do i=1,nexternal
            if(iqjets(i).gt.0)then
               njets=njets+1
               if (iqjetstore(njets,nFKSprocess,iconf).ne.i) fail=.true.
            endif
         enddo
         if(njets.ne.njetstore(nFKSprocess,iconf)) fail=.true.
         if (fail) then
            if (igraphs(1).eq.iconf) then
               open(unit=26,file='../../../error',status='unknown',err=999)
               write(*,*) 'Error: Failed despite same graph: ',iconf
     $              ,nFKSprocess
               write(*,*) 'Have jets (>0)',(iqjets(i),i=1,nexternal)
               write(*,*) 'Should be ', (iqjetstore(i,nFKSprocess,iconf)
     $              ,i=1,njetstore(nFKSprocess,iconf))
               write(26,*) 'Error: Failed despite same graph: ',iconf
     $              ,nFKSprocess,'. Have jets (>0)',(iqjets(i),i=1
     $              ,nexternal),', should be ', (iqjetstore(i
     $              ,nFKSprocess,iconf),i=1,njetstore(nFKSprocess
     $              ,iconf))
               
               stop
            endif
            if (btest(mlevel,3))
     $           write(*,*) 'Bad clustering, jets fail. Reclustering ',
     $           iconf
            chcluster=.true.
            goto 100
         endif
      endif
c     If last clustering is s-channel QCD (e.g. ttbar) use mt2last instead
c     (i.e. geom. average of transverse mass of t and t~)
      if(mt2last.gt.4d0 .and. nexternal.gt.3) then
         if(jlast(1).eq.nexternal-2.and.jlast(2).eq.nexternal-2.and.
     $        isqcd(ipdgcl(idacl(nexternal-3,1),igraphs(1),nFKSprocess)).and. 
     $        isqcd(ipdgcl(idacl(nexternal-3,2),igraphs(1),nFKSprocess)).and. 
     $        isqcd(ipdgcl(imocl(nexternal-3),igraphs(1),nFKSprocess)))then
            mt2ij(nexternal-2)=mt2last
            mt2ij(nexternal-3)=mt2last
            if (btest(mlevel,3)) then
               write(*,*)' setclscales: set last vertices to mtlast: '
     $              ,sqrt(mt2last)
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
     $        pt2ijcl(jcentral(2))=mt2ij(jcentral(2))
      endif
      if(btest(mlevel,4))then
         write (*,*)'jlast, jcentral: ',(jlast(i),i=1,2),(jcentral(i),i
     $        =1,2)
         if(jlast(1).gt.0) write(*,*)'pt(jlast 1): ',
     $        sqrt(pt2ijcl(jlast(1)))
         if(jlast(2).gt.0) write(*,*)'pt(jlast 2): ',
     $        sqrt(pt2ijcl(jlast(2)))
         if(jcentral(1).gt.0) write(*,*)'pt(jcentral 1): ',
     $        sqrt(pt2ijcl(jcentral(1)))
         if(jcentral(2).gt.0) write(*,*)'pt(jcentral 2): ',
     $        sqrt(pt2ijcl(jcentral(2)))
      endif
c     Check xqcut for vertices with jet daughters only
      ibeam(1)=ishft(1,0)
      ibeam(2)=ishft(1,1)
c     For FxFx skip the cut on xqcut
      if(xqcut.gt.0 .and. ickkw.ne.3) then
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
     $                       ipdgcl(idacl(n,1),igraphs(1),nFKSprocess),
     $                       ipdgcl(idacl(n,2),igraphs(1),nFKSprocess),
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
c     For FxFx skip the cut on xmtc
      if(xmtc**2.gt.0 .and. ickkw.ne.3) then
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
      if(jlast(1).gt.0) pt2ijcl(jlast(1))=max(pt2ijcl(jlast(1))
     $     ,pt2ijcl(jfirst(1)))
      if(jlast(2).gt.0) pt2ijcl(jlast(2))=max(pt2ijcl(jlast(2))
     $     ,pt2ijcl(jfirst(2)))

      if(ickkw.gt.0.and.q2fact(1).gt.0 .and. ickkw.ne.3) then
c     Use the fixed or previously set scale for central scale
         if(jcentral(1).gt.0) pt2ijcl(jcentral(1))=q2fact(1)
         if(jcentral(2).gt.0.and.jcentral(2).ne.jcentral(1))
     $        pt2ijcl(jcentral(2))=q2fact(2)
      endif

      if ( (nexternal.eq.3.and.nincoming.eq.2.and.ickkw.eq.3) .or.
     &     (nexternal.eq.3.and.nincoming.eq.2.and.
     &                           q2fact(1).eq.0.and.ickkw.ne.3)) then
         q2fact(1)=pt2ijcl(nexternal-2)
         q2fact(2)=pt2ijcl(nexternal-2)
      endif

      if(q2fact(1).eq.0d0 .or. ickkw.eq.3) then
c     Use the geom. average of central scale and first non-radiation vertex
         if(jlast(1).gt.0)
     &        q2fact(1)=sqrt(pt2ijcl(jlast(1))*pt2ijcl(jcentral(1)))
         if(jlast(2).gt.0)
     &        q2fact(2)=sqrt(pt2ijcl(jlast(2))*pt2ijcl(jcentral(2)))
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
      if(scale.eq.0d0 .or. ickkw.eq.3) then
         if(jlast(1).gt.0.and.jlast(2).gt.0)then
c     Use geom. average of last and central scales
            scale=(pt2ijcl(jlast(1))*pt2ijcl(jcentral(1))*
     $           pt2ijcl(jlast(2))*pt2ijcl(jcentral(2)))**0.125
         elseif(jlast(1).gt.0)then
c     Use geom. average of last and central scale
            scale=(pt2ijcl(jlast(1))*pt2ijcl(jcentral(1)))**0.25
         elseif(jlast(2).gt.0)then
c     Use geom. average of last and central scale
            scale=(pt2ijcl(jlast(2))*pt2ijcl(jcentral(2)))**0.25
         elseif(jcentral(1).gt.0.and.jcentral(2).gt.0) then
c     Use geom. average of central scales
            scale=(pt2ijcl(jcentral(1))*pt2ijcl(jcentral(2)))**0.25d0
         elseif(jcentral(1).gt.0) then
            scale=sqrt(pt2ijcl(jcentral(1)))
         elseif(jcentral(2).gt.0) then
            scale=sqrt(pt2ijcl(jcentral(2)))
         else
            scale=sqrt(pt2ijcl(nexternal-2))
         endif
         FxFx_ren_scales(0)=scale ! do not include scalefact here
         scale=scalefact*scale
         if(scale.gt.0 .and. ickkw.ne.3)
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
      elseif(ickkw.ge.2.or.pdfwgt)then
c     Total pdf weight is f1(x1,pt2E)*fj(x1*z,Q)/fj(x1*z,pt2E)
c     f1(x1,pt2E) is given by DSIG, just need to set scale.
c     Use the minimum scale found for fact scale in ME
         if(jlast(1).gt.0.and.jfirst(1).le.jlast(1))
     $        q2fact(1)=min(pt2ijcl(jfirst(1)),q2fact(1))
         if(jlast(2).gt.0.and.jfirst(2).le.jlast(2))
     $        q2fact(2)=min(pt2ijcl(jfirst(2)),q2fact(2))
      endif

c FxFx factorization scales (divide scalefact here, because it was
c included above and it should not be included for the FxFx scales)
      FxFx_fac_scale(1)=sqrt(q2fact(1))/scalefact
      FxFx_fac_scale(2)=sqrt(q2fact(2))/scalefact
      
      if (ickkw.ne.3) then  ! For FxFx, this is done in setscales.f
c     Check that factorization scale is >= 2 GeV
         if ( lpp(1).ne.0.and.q2fact(1).lt.4d0.or.
     $        lpp(2).ne.0.and.q2fact(2).lt.4d0    )then
            if(nwarning.le.10) then
               nwarning=nwarning+1
               write(*,*) 'Warning: Too low fact scales: ',
     $              sqrt(q2fact(1)), sqrt(q2fact(2))
               write (*,*) pt2ijcl,ifxfx
            endif
            if(nwarning.eq.11) then
               nwarning=nwarning+1
               write(*,*) 'No more warnings written out this run.'
            endif
            setclscales=.false.
            clustered = .false.
            return
         endif
      endif

      if (btest(mlevel,3))
     $     write(*,*) 'Set fact scales to ',sqrt(q2fact(1)),sqrt(q2fact(2))

c
c     Store jet info for matching
c
      etot=sqrt(4d0*ebeam(1)*ebeam(2))
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
c     First adjust goodjet based on iqjets
               if(goodjet(ida(i)).and.ipart(j,ida(i)).gt.2)then
                  if(iqjets(ipart(j,ida(i))).eq.0) goodjet(ida(i))=.false.
               endif
c     Now reset ptclus if jet vertex
               if(ipart(j,ida(i)).gt.2) then
                  if(isjetvx(imocl(n),idacl(n,1),idacl(n,2),
     $               ipdgcl(1,igraphs(1),nFKSprocess),ipart,n.eq.nexternal-2)
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
      

      double precision function rewgt(p,rewgt_exp)
c**************************************************
c   reweight the hard me according to ckkw
c   employing the information in common/cl_val/
c**************************************************
      implicit none

c Include
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'cluster.inc'
      include 'message.inc'
      include 'run.inc'
      include 'coupl.inc'
      include 'real_from_born_configs.inc'
      double precision ZERO,PI
      parameter (ZERO=0d0)
      parameter( PI = 3.14159265358979323846d0 )
c Argument
      double precision p(0:3,nexternal),rewgt_exp
c Local
      logical isvx,goodjet(n_max_cl)
      integer i,j,n,k,l,ipart(2,n_max_cl),ibeam(2),iconf,ijet
      double precision pt2min,pt2prev(n_max_cl),pt2pdf(n_max_cl)
     $     ,xnow(2),asref,q2now,tmp,tmp2,pdfj1,pdfj2,xqcut
      integer ib(2)
      data ib /1,2/
c Common
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config
      integer njetstore(fks_configs,lmaxconfigs),iqjetstore(nexternal-2
     $     ,fks_configs,lmaxconfigs),jlast(2)
      real*8 q2bck(2)
      common /to_rw/jlast,njetstore,iqjetstore,q2bck
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      logical use_syst
      DOUBLE PRECISION s_scale
      INTEGER n_qcd,n_alpsem
      DOUBLE PRECISION s_qalps(max_particles-2)
      INTEGER n_pdfrw(2),i_pdgpdf(max_particles-2,2)
      DOUBLE PRECISION s_xpdf(max_particles-2,2),s_qpdf(max_particles-2,2)
      DOUBLE PRECISION s_rwfact
      COMMON/TO_SYST/use_syst,n_qcd,n_alpsem,n_pdfrw,i_pdgpdf,
     $               s_scale,s_qalps,s_xpdf,s_qpdf,s_rwfact
c External
      logical ispartonvx,isqcd,isparton,isjetvx,isjet
      double precision alphas,getissud,pdg2pdf,sudwgt,sudwgt_exp
      external ispartonvx,alphas,isqcd,isparton,isjetvx,getissud
     $     ,pdg2pdf,isjet,sudwgt,sudwgt_exp
c FxFx
      integer nFxFx_ren_scales
      double precision FxFx_ren_scales(0:nexternal)
      common/c_FxFx_scales/FxFx_ren_scales,nFxFx_ren_scales

      iconf=real_from_born_conf(this_config,nFKSprocess)
      rewgt=1.0d0
      rewgt_exp=0.0d0
      clustered=.false.
      if (ickkw.eq.3) then
         use_syst=.false.
         hmult=.true.
      endif
      if(ickkw.le.0) return

c   Set mimimum kt scale, depending on highest mult or not
      if(hmult.or.ickkw.eq.1)then
        pt2min=0d0
      else
        pt2min=xqcut**2
      endif
c For FxFx merging, pt2min should be set equal to the 2nd smallest (QCD)
c clustering scale
      if (ifxfx(2).gt.0 .and. pt2ijcl(ifxfx(2)).gt.0d0) then
         pt2min=pt2ijcl(ifxfx(2))
      else
         pt2min=1.0d37
      endif

      if (btest(mlevel,3))
     $     write(*,*) 'pt2min set to ',pt2min

c   Since we use pdf reweighting, need to know particle identities
      if (btest(mlevel,1)) then
         write(*,*) 'Set process number ',nFKSprocess
      endif
c     Set incoming particle identities
      ipdgcl(1,igraphs(1),nFKSprocess)=idup(1,1)
      ipdgcl(2,igraphs(1),nFKSprocess)=idup(2,1)
      if (btest(mlevel,2)) then
         write(*,*) 'Set particle identities: ',
     $        1,ipdgcl(1,igraphs(1),nFKSprocess),
     $        ' and ',
     $        2,ipdgcl(2,igraphs(1),nFKSprocess)
      endif
      
c     Store pdf information for systematics studies (initial)
      if(use_syst)then
         do j=1,2
            n_pdfrw(j)=1
            i_pdgpdf(1,j)=ipdgcl(j,igraphs(1),nFKSprocess)
            s_xpdf(1,j)=xbk(ib(j))
            s_qpdf(1,j)=sqrt(q2fact(j))
         enddo
      endif

      if(ickkw.le.0) goto 100

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
         pt2prev(ishft(1,i-1))=0d0
         if (ickkw.ge.2) then
            if(pt2min.gt.0)then
               pt2prev(ishft(1,i-1))=
     $              max(pt2min,p(0,i)**2-p(1,i)**2-p(2,i)**2-p(3,i)**2)
            endif
            pt2pdf(ishft(1,i-1))=pt2prev(ishft(1,i-1))
         else if(pdfwgt) then
            pt2pdf(ishft(1,i-1))=0d0
         endif
         ipart(1,ishft(1,i-1))=i
         ipart(2,ishft(1,i-1))=0
         if (btest(mlevel,4))
     $        write(*,*) 'Set ipart for ',ishft(1,i-1),' to ',
     $        ipart(1,ishft(1,i-1)),ipart(2,ishft(1,i-1))
      enddo
      ibeam(1)=ishft(1,0)
      ibeam(2)=ishft(1,1)
      if (btest(mlevel,1)) then
         write(*,*)'rewgt: identified tree {'
         do i=1,nexternal-2
            write(*,*)'  ',i,': ',idacl(i,1),'(',ipdgcl(idacl(i,1)
     $           ,igraphs(1),nFKSprocess),')','&',idacl(i,2),'('
     $           ,ipdgcl(idacl(i ,2),igraphs(1),nFKSprocess),')',' -> '
     $           ,imocl(i),'(' ,ipdgcl(imocl(i),igraphs(1),nFKSprocess)
     $           ,')' ,', ptij = ' ,dsqrt(pt2ijcl(i))
         enddo
         write(*,*)'  graphs (',igraphs(0),'):',(igraphs(i),i=1
     $        ,igraphs(0))
         write(*,*)'}'
      endif
c     Set x values for the two sides, for Initial State Sudakovs
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
         if(isjet(idup(i,1)))
     $        ipdgcl(j,igraphs(1),nFKSprocess)=idup(i,1)
         if (btest(mlevel,2))
     $        write(*,*) 'Set particle identities: ',
     $        i,ipdgcl(j,igraphs(1),nFKSprocess)
         if(i.le.2)then
            goodjet(j)=isparton(ipdgcl(j,igraphs(1),nFKSprocess))
         elseif(ijet.le.njetstore(nFKSprocess,iconf).and.
     $           i.eq. iqjetstore(ijet,nFKSprocess,iconf)) then
            goodjet(j)=.true.
            ijet=ijet+1
         elseif(isparton(ipdgcl(j,igraphs(1),nFKSprocess)).and.
     $           .not.isjet(ipdgcl(j,igraphs(1),nFKSprocess))) then
            goodjet(j)=.true.            
         else
            goodjet(j)=.false.
         endif
         if(btest(mlevel,4)) print *,'Set goodjet ',j,goodjet(j)
      enddo
c   
c     Set strong coupling used
c   
      if (ickkw.ne.3) then      ! Don't need asref for FxFx
         asref=G**2/(4d0*PI)
      endif

      nFxFx_ren_scales=0

c     Perform alpha_s reweighting based on type of vertex
      do n=1,nexternal-2
c        scale for alpha_s reweighting 
         q2now=pt2ijcl(n)
         if(n.eq.nexternal-2) then
            q2now = scale**2
         endif
         if (btest(mlevel,3)) then
            write(*,*)'  ',n,': ',idacl(n,1),'(',ipdgcl(idacl(n,1)
     $           ,igraphs(1),nFKSprocess),')&',idacl(n,2),'('
     $           ,ipdgcl(idacl(n ,2) ,igraphs(1),nFKSprocess),') -> '
     $           ,imocl(n),'(' ,ipdgcl(imocl(n) ,igraphs(1),nFKSprocess)
     $           ,'), ptij = ' ,dsqrt(q2now) 
         endif
c   Update particle tree map
        call ipartupdate(p,imocl(n),idacl(n,1),idacl(n,2),
     $       ipdgcl(1,igraphs(1),nFKSprocess),ipart)
c     perform alpha_s reweighting only for vertices where a parton is produced
c     and not for the last clustering (use non-fixed ren. scale for these)
         if (n.lt.nexternal-2)then
c     Use goodjet to trace allowed parton lines.  For ISR, allow only
c     splittings where all particles are along good parton lines; for
c     FSR, just require one FS particle to be good
            goodjet(imocl(n))=isparton(ipdgcl(imocl(n),igraphs(1),nFKSprocess))
     $           .and.goodjet(idacl(n,1)).and.goodjet(idacl(n,2))
            if(btest(mlevel,4))
     $           write(*,*)'Set goodjet ',imocl(n),' to ',goodjet(imocl(n))
            if(ipart(1,imocl(n)).le.2.and.goodjet(imocl(n)).or. ! ISR
     $           ipart(1,imocl(n)).gt.2.and. ! FSR
     $           ispartonvx(imocl(n),idacl(n,1),idacl(n,2),
     $           ipdgcl(1,igraphs(1),nFKSprocess),ipart,.false.).and.
     $           (goodjet(idacl(n,1)).or.goodjet(idacl(n,2)))) then
c       alpha_s weight (skip for FxFx)
               if (ickkw.ne.3) then
                  rewgt=rewgt*alphas(alpsfact*sqrt(q2now))/asref
c     Store information for systematics studies
                  if(use_syst)then
                     n_alpsem=n_alpsem+1
                     s_qalps(n_alpsem)=sqrt(q2now)
                  endif
               else
                  nFxFx_ren_scales=nFxFx_ren_scales+1
                  FxFx_ren_scales(nFxFx_ren_scales)=sqrt(q2now)
               endif
               if (btest(mlevel,3)) then
                  write(*,*)' reweight vertex: ',ipdgcl(imocl(n)
     $                 ,igraphs(1),nFKSprocess),ipdgcl(idacl(n,1)
     $                 ,igraphs(1) ,nFKSprocess),ipdgcl(idacl(n,2)
     $                 ,igraphs(1),nFKSprocess),sqrt(q2now)
                  if (ickkw.ne.3) then
                     write(*,*)'       as: ',alphas(alpsfact*dsqrt(q2now)),
     &                    '/',asref,' -> ',alphas(alpsfact*dsqrt(q2now))/asref
                     write(*,*)' and G=',SQRT(4d0*PI*ALPHAS(scale))
                  endif
               endif
            endif
         endif
         if(.not.(ickkw.ge.2.or.pdfwgt)) cycle
c     Perform PDF and, if ickkw=2, Sudakov reweighting
         isvx=.false.
         do i=1,2  ! loop over the 2 daughters
c     write(*,*)'weight ',idacl(n,i),', ptij=',pt2prev(idacl(n,i))
            if (isqcd(ipdgcl(idacl(n,i),igraphs(1),nFKSprocess))) then
               if(ickkw.eq.2.and.pt2min.eq.0d0) then
                  pt2min=pt2ijcl(n)
                  if (btest(mlevel,3))
     $                 write(*,*) 'pt2min set to ',pt2min
               endif
               if(ickkw.eq.2.and.pt2prev(idacl(n,i)).eq.0d0)
     $              pt2prev(idacl(n,i))=
     $              max(pt2min,p(0,i)**2-p(1,i)**2-p(2,i)**2-p(3,i)**2)
               do j=1,2         ! loop over 2 beams
c     Check which beam belongs to the clustering (and cycle if non-QCD
c     or not correct beam)
                  if ( .not. (isparton(ipdgcl(idacl(n,i),igraphs(1)
     $                 ,nFKSprocess)).and.idacl(n,i).eq.ibeam(j))) cycle
c     is sudakov weight - calculate only once for each parton line where
c     parton line ends with change of parton id or non-radiation vertex
                  isvx=.true.
                  ibeam(j)=imocl(n)
c     Perform Sudakov reweighting if ickkw=2
                  if(ickkw.ge.2.and.(ipdgcl(idacl(n,i),igraphs(1),nFKSprocess).ne.
     $                 ipdgcl(imocl(n),igraphs(1),nFKSprocess).or.
     $                 .not.isjetvx(imocl(n),idacl(n,1),idacl(n,2),
     $                 ipdgcl(1,igraphs(1),nFKSprocess),ipart,n.eq.nexternal-2)).and.
     $                 pt2prev(idacl(n,i)).lt.pt2ijcl(n))then
                     if (ickkw.ne.3) then
c Sudakov including PDFs:
                        tmp=min(1d0,max(
     &                       getissud(ibeam(j),
     &                       ipdgcl(idacl(n,i),igraphs(1),nFKSprocess),
     &                       xnow(j),xnow(3-j),pt2ijcl(n)) ,
     &                       1d-20 ) /
     $                       max(
     &                       getissud(ibeam(j),
     &                       ipdgcl(idacl(n,i),igraphs(1),nFKSprocess),
     &                       xnow(j),xnow(3-j),pt2prev(idacl(n,i))) ,
     &                       1d-20 ))
                        tmp2=0d0
                     else
c     Sudakov excluding PDFs:
                        tmp=sudwgt(sqrt(pt2min),sqrt(pt2prev(idacl(n,i))),
     $                       dsqrt(pt2ijcl(n)),ipdgcl(idacl(n,i),igraphs(1)
     $                       ,nFKSprocess),1)
                        tmp2=sudwgt_exp(sqrt(pt2min)
     $                       ,sqrt(pt2prev(idacl(n,i))),dsqrt(pt2ijcl(n))
     $                       ,ipdgcl(idacl(n,i),igraphs(1),nFKSprocess),1)
                     endif
                     rewgt=rewgt*tmp
                     rewgt_exp=rewgt_exp+tmp2
                     pt2prev(imocl(n))=pt2ijcl(n)
                     if (btest(mlevel,3)) then
                        write(*,*)' reweight line: ' ,ipdgcl(idacl(n,i)
     $                       ,igraphs(1) ,nFKSprocess), idacl(n,i)
                        write(*,*)'     pt2prev, pt2new, x1, x2: '
     $                       ,pt2prev(idacl(n,i)),pt2ijcl(n) ,xnow(j)
     $                       ,xnow(3-j)
                        write(*,*)'           Sud: ',tmp
                        write(*,*)'        -> rewgt: ',rewgt
                     endif
                  else if(ickkw.eq.2) then
                     pt2prev(imocl(n))=pt2prev(idacl(n,i))
                  endif
c     End Sudakov reweighting when we reach a non-radiation vertex
                  if(ickkw.eq.2.and..not. ispartonvx(imocl(n)
     $                 ,idacl(n,1),idacl(n,2), ipdgcl(1,igraphs(1)
     $                 ,nFKSprocess),ipart,n.eq.nexternal-2)) then
                     pt2prev(imocl(n))=1d30
                     if (btest(mlevel,3)) then
                        write(*,*) ' rewgt: ending reweighting for vx ',
     $                       idacl(n,1),idacl(n,2),imocl(n),
     $                       ' with ids ',ipdgcl(idacl(n,1) ,igraphs(1)
     $                       ,nFKSprocess), ipdgcl(idacl(n,2),igraphs(1)
     $                       ,nFKSprocess),ipdgcl(imocl(n) ,igraphs(1)
     $                       ,nFKSprocess)
                     endif
                  endif
c     PDF reweighting
c     Total pdf weight is f1(x1,pt2E)*fj(x1*z,Q)/fj(x1*z,pt2E)
c     f1(x1,pt2E) is given by DSIG, already set scale for that
                  if (zcl(n).gt.0d0.and.zcl(n).lt.1d0) then
                     xnow(j)=xnow(j)*zcl(n)
                  endif
c     For FxFx: Skip PDF reweighting and skip the Final state Sudakov
                  if (ickkw.eq.3) goto 10 
c     PDF scale
                  q2now=min(pt2ijcl(n), q2bck(j))
c     Set PDF scale to central factorization scale if non-radiating
c     vertex or last 2->2
                  if(n.eq.jlast(j)) then
                     q2now=q2bck(j)
                  endif
                  if (btest(mlevel,3))
     $                 write(*,*)' set q2now for pdf to ',sqrt(q2now)
                  if(q2fact(j).eq.0d0.and.ickkw.eq.2)then
                     q2fact(j)=pt2min ! Starting scale for PS
                     pt2pdf(imocl(n))=q2now
                     if (btest(mlevel,3))
     $                    write(*,*)' set fact scale ',j,
     $                    ' for PS scale to: ',sqrt(q2fact(j))
                  else if(pt2pdf(idacl(n,i)).eq.0d0)then
                     pt2pdf(imocl(n))=q2now
                     if (btest(mlevel,3))
     $                    write(*,*)' set pt2pdf for ',imocl(n),
     $                    ' to: ',sqrt(pt2pdf(imocl(n)))
                  else if(pt2pdf(idacl(n,i)).lt.q2now.and.
     $                    n.le.jlast(j))then
                     pdfj1=pdg2pdf(abs(lpp(IB(j))),
     $                    ipdgcl(idacl(n,i),
     $                    igraphs(1),nFKSprocess)*sign(1,lpp(IB(j))),
     $                    xnow(j),sqrt(q2now))
                     pdfj2=pdg2pdf(abs(lpp(IB(j))),
     $                    ipdgcl(idacl(n,i),
     $                    igraphs(1),nFKSprocess)*sign(1,lpp(IB(j))),
     $                    xnow(j),sqrt(pt2pdf(idacl(n,i))))
                     if(pdfj2.lt.1d-10)then
c     Scale too low for heavy quark
                        rewgt=0d0
                        if (btest(mlevel,3))
     $                       write(*,*) 'Too low scale for quark pdf: ',
     $                       sqrt(pt2pdf(idacl(n,i))),pdfj2,pdfj1
                        return
                     endif
                     rewgt=rewgt*pdfj1/pdfj2
c     Store information for systematics studies
                     if(use_syst)then
                        n_pdfrw(j)=n_pdfrw(j)+1
                        i_pdgpdf(n_pdfrw(j),j)=ipdgcl(idacl(n,i),igraphs(1),nFKSprocess)
                        if (zcl(n).gt.0d0.and.zcl(n).lt.1d0) then
                           s_xpdf(n_pdfrw(j),j)=xnow(j)/zcl(n)
                        else
                           s_xpdf(n_pdfrw(j),j)=xnow(j) 
                        endif
                        s_qpdf(n_pdfrw(j),j)=sqrt(q2now)
                     endif
                     if (btest(mlevel,3)) then
                        write(*,*)' reweight ',n,i,ipdgcl(idacl(n,i),igraphs(1),nFKSprocess),' by pdfs: '
                        write(*,*)'     x, ptprev, ptnew: ',xnow(j),
     $                       sqrt(pt2pdf(idacl(n,i))),sqrt(q2now)
                        write(*,*)'           PDF: ',pdfj1,' / ',pdfj2
                        write(*,*)'        -> rewgt: ',rewgt
                     endif
c     Set scale for mother as this scale
                     pt2pdf(imocl(n))=q2now                           
                  else if(pt2pdf(idacl(n,i)).ge.q2now) then
c     If no reweighting, just copy daughter scale for mother
                     pt2pdf(imocl(n))=pt2pdf(idacl(n,i))
                  endif
                  goto 10       !  Skip the Final state Sudakov
               enddo
c     Final State sudakov weight
               if(ickkw.ge.2.and.pt2prev(idacl(n
     $              ,i)).lt.pt2ijcl(n).and.(isvx.or.ipdgcl(idacl(n,i)
     $              ,igraphs(1),nFKSprocess).ne.ipdgcl(imocl(n)
     $              ,igraphs(1),nFKSprocess).or.(ipdgcl(idacl(n,i)
     $              ,igraphs(1),nFKSprocess).ne.ipdgcl(idacl(n,3-i)
     $              ,igraphs(1),nFKSprocess).and.pt2prev(idacl(n
     $              ,i)).gt.pt2prev(idacl(n,3-i))))) then
                  tmp=sudwgt(sqrt(pt2min),sqrt(pt2prev(idacl(n,i))),
     $                 dsqrt(pt2ijcl(n)),ipdgcl(idacl(n,i),igraphs(1)
     $                 ,nFKSprocess),1)
                  tmp2=sudwgt_exp(sqrt(pt2min),sqrt(pt2prev(idacl(n,i))),
     $                 dsqrt(pt2ijcl(n)),ipdgcl(idacl(n,i),igraphs(1)
     $                 ,nFKSprocess),1)
                  rewgt=rewgt*tmp
                  rewgt_exp=rewgt_exp+tmp2
                  if (btest(mlevel,3)) then
                     write(*,*)' reweight fs line: ',ipdgcl(idacl(n,i)
     $                    ,igraphs(1),nFKSprocess), idacl(n,i)
                     write(*,*)'     pt2prev, pt2new: ',pt2prev(idacl(n
     $                    ,i)),pt2ijcl(n)
                     write(*,*)'           Sud: ',tmp
                     write(*,*)'           Sud_exp: ',tmp2
                     write(*,*)'        -> rewgt: ',rewgt
                     write(*,*)'        -> rewgt_exp: ',rewgt_exp
                  endif
                  pt2prev(imocl(n))=pt2ijcl(n)
               else
                  pt2prev(imocl(n))=pt2prev(idacl(n,i))
               endif 
            endif
 10         continue
         enddo
         if (ickkw.ge.2.and.n.eq.nexternal-2.and.
     $        isqcd(ipdgcl(imocl(n),igraphs(1),nFKSprocess)).and.
     $        pt2prev(imocl(n)).lt.pt2ijcl(n))then
            tmp=sudwgt(sqrt(pt2min),sqrt(pt2prev(imocl(n))),
     $           dsqrt(pt2ijcl(n)),ipdgcl(imocl(n),igraphs(1)
     $           ,nFKSprocess),1)
            tmp2=sudwgt_exp(sqrt(pt2min),sqrt(pt2prev(imocl(n))),
     $           dsqrt(pt2ijcl(n)),ipdgcl(imocl(n),igraphs(1)
     $           ,nFKSprocess),1)
            rewgt=rewgt*tmp
            rewgt_exp=rewgt_exp+tmp2
            if (btest(mlevel,3)) then
               write(*,*)' reweight last fs line: ',ipdgcl(imocl(n)
     $              ,igraphs(1),nFKSprocess), imocl(n)
               write(*,*)'     pt2prev, pt2new: ',pt2prev(imocl(n))
     $              ,pt2ijcl(n)
               write(*,*)'           Sud: ',tmp
               write(*,*)'           Sud_exp: ',tmp2
               write(*,*)'        -> rewgt: ',rewgt
               write(*,*)'        -> rewgt_exp: ',rewgt_exp
            endif
         endif
      enddo

      if(ickkw.eq.2.and.lpp(1).eq.0.and.lpp(2).eq.0)then
         q2fact(1)=pt2min
         q2fact(2)=q2fact(1)
      else if (ickkw.eq.1.and.pdfwgt) then
         q2fact(1)=q2bck(1)
         q2fact(2)=q2bck(2)         
         if (btest(mlevel,3))
     $        write(*,*)' set fact scales for PS to ',
     $        sqrt(q2fact(1)),sqrt(q2fact(2))
      endif

      if (btest(mlevel,3)) then
        write(*,*)'} ->  w = ',rewgt,' w_exp=',rewgt_exp
      endif

 100  continue

c     Set reweight factor for systematics studies
      if(use_syst)then
         s_rwfact = rewgt
c     Need to multiply by: initial PDF, alpha_s^n_qcd to get
c     factor in front of matrix element
         do i=1,2
            s_rwfact=s_rwfact*pdg2pdf(abs(lpp(IB(i))),
     $           i_pdgpdf(1,i)*sign(1,lpp(IB(i))),
     $           s_xpdf(1,i),s_qpdf(1,i))
         enddo
         s_rwfact=s_rwfact*asref**n_qcd
      endif

      return
      end
      
