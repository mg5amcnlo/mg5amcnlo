************************************************************************
*
*     Analysis routines for lepton-pair production.
*
************************************************************************
      subroutine analysis_begin(nwgt,weights_info)
*
      implicit none
*
      include "dbook.inc"
*
      integer nwgt
      character*(*) weights_info(*)
      integer j,kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
      character*5 cc(2)
      data cc/"     "," Born"/
*
*     Initialize histograms
*
      call inihist
*
*     Check whether the user is trying to fill to many histograms
*
      nwgt_analysis = nwgt
      if (nwgt_analysis*5.gt.nplots/2) then
         write(*,*) "error in analysis_begin: ",
     1              "too many histograms, increase NPLOTS to",
     2               nwgt_analysis * 5 * 2
         call exit(-10)
      endif
*
*     Define the observables
*
      do j=1,2
         do kk=1,nwgt_analysis
            l = ( kk - 1 ) * 4 + ( j - 1 ) * 2
            call bookup(l+1,"y_ll  "//weights_info(kk)//cc(j),
     1                  0.5d0,-3d0,3d0)
            call bookup(l+2,"m_ll  "//weights_info(kk)//cc(j),
     1                  5d0,70d0,170d0)
         enddo
      enddo
*
      return
      end
*
************************************************************************
      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
*
      implicit none
*
      include "nexternal.inc"
*
      integer i,kk,l
      integer ibody
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      double precision wgts(*)
      integer nwgt_analysis
      common/c_analysis/nwgt_analysis
      double precision www
      double precision yll,mll
      double precision getrapidity
      external getrapidity
*
*     All information can be found in analysis_td_template.f
*
*     The ibody variable is:
*     ibody = 1 : (n+1)-body contribution
*     ibody = 2 : n-body contribution (excluding the Born)
*     ibody = 3 : Born contribution
*
      if(ibody.ne.1.and.ibody.ne.2.and.ibody.ne.3)then
         write(6,*) "Error: Invalid value for ibody =",ibody
         call exit(-10)
      endif
*
*     Check that the number of outcoming particles is correct
*
      if(nexternal.ne.5)then
         write(*,*) "error in analysis_fill: "//
     1              "only for lepton-pair production"
         call exit(-10)
      endif
*
*     Check that the third and the forth particle are leptons
*
      if (abs(ipdg(3)).lt.11.or.abs(ipdg(3)).gt.16)then
         write(*,*) "error in analysis_fill: "//
     1              "The first outcoming particle must be a lepton",
     2               "ID =",ipdg(3)
         call exit(-10)
      endif
      if (abs(ipdg(4)).lt.11.or.abs(ipdg(4)).gt.16)then
         write(*,*) "error in analysis_fill: "//
     1              "The second outcoming particle must be a lepton",
     2              "ID =",ipdg(4)
         call exit(-10)
      endif
*
*     Compute observables
*
*     Rapidity of the pair
      yll = getrapidity(p(0,3)+p(0,4),p(3,3)+p(3,4))
*     Invariant mass of the pair
      mll = dsqrt(( p(0,3) + p(0,4) )**2d0 - ( p(1,3) + p(1,4) )**2d0 
     1          - ( p(2,3) + p(2,4) )**2d0 - ( p(3,3) + p(3,4) )**2d0)
*
*     Fill histograms
*
      do i=1,2
         do kk=1,nwgt_analysis
            www = wgts(kk)
            l = ( kk - 1 ) * 4 + ( i - 1 ) * 2
*     Only fill Born histograms for ibody=3
            if (ibody.ne.3.and.i.eq.2) cycle
*
            call mfill(l+1,yll,www)
            call mfill(l+2,mll,www)
         enddo
      enddo
*
      return      
      end
*
************************************************************************
      subroutine analysis_end(xnorm)
*
      implicit none
*
      include "dbook.inc"
*
      character*14 ytit
      double precision xnorm
      integer i
      integer kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
*
      call open_topdrawer_file
      call mclear
*
      do i=1,NPLOTS
         call mopera(i,"+",i,i,xnorm,0.d0)
         call mfinal(i)
      enddo
      ytit = "sigma per bin "
      do i=1,2
         do kk=1,nwgt_analysis
            l = ( kk - 1 ) * 4 + ( i - 1 ) * 2
            call multitop(l+1,3,2,"y_ll  "," ","LIN")
            call multitop(l+2,3,2,"m_ll  "," ","LIN")
         enddo
      enddo
*
      call close_topdrawer_file
*
      return                
      end
*
************************************************************************
*
*     Auxiliary functions for the kinematics
*
************************************************************************
      function getrapidity(en,pl)
*
      implicit none
*
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y
      parameter (tiny=1d-8)
*
      xplus  = en + pl
      xminus = en - pl
*
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
         if((xplus/xminus).gt.tiny.and.(xminus/xplus).gt.tiny)then
            y = 0.5d0 * log( xplus / xminus  )
         else
            y = sign(1d0,pl) * tiny
         endif
      else
         y = sign(1d0,pl) * tiny
      endif
*
      getrapidity = y
*
      return
      end
