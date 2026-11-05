c
c Fixed-order analysis for "p p > t t~ [QCD]" with the four-fermion SMEFT
c operators, used for the toy fit of arXiv:2212.05067 (Table 4) at NLO.
c
c The histograms are filled separately for each NP order of the squared
c matrix element (orders_tag_plot): NP^2=0 (SM), 2 (interference), 4
c (squared dimension-six). They are meant to be rebinned to the bins of
c the data sets afterwards:
c   m_tt     : 5 GeV bins, 250 to 3500 GeV (all events, inclusive xsec = sum)
c   y_tt     : |y_tt|, 0.05 bins, 0 to 3, separately for |y_t|>|y_tbar| (+)
c              and |y_t|<|y_tbar| (-): A_C = (N+ - N-)/(N+ + N-)
c   beta_z   : |p_z|/E of the tt system, 0.02 bins, 0 to 1, same (+)/(-)
c The tt system is built from the top and antitop only (the radiated
c parton of the real emission is not included), as in the parton-level
c A_C definitions.
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_begin(nwgt,weights_info)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      integer inp,l
      character*12 cnp(0:2)
      data cnp/'NP2=0 SM    ','NP2=2 int   ','NP2=4 quad  '/
      call HwU_inithist(nwgt,weights_info)
      do inp=0,2
        l=inp*10
        call HwU_book(l+1,'mtt   '//cnp(inp),650,250.d0,3500.d0)
        call HwU_book(l+2,'ytt+  '//cnp(inp),60,0.d0,3.d0)
        call HwU_book(l+3,'ytt-  '//cnp(inp),60,0.d0,3.d0)
        call HwU_book(l+4,'btt+  '//cnp(inp),50,0.d0,1.d0)
        call HwU_book(l+5,'btt-  '//cnp(inp),50,0.d0,1.d0)
        call HwU_book(l+6,'mtt+  '//cnp(inp),650,250.d0,3500.d0)
        call HwU_book(l+7,'mtt-  '//cnp(inp),650,250.d0,3500.d0)
      enddo
      return
      end


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_end(dummy)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      double precision dummy
      call HwU_write_file
      return
      end


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      double precision wgts(*)
      integer ibody
      integer orders_tag_plot
      common /corderstagplot/ orders_tag_plot
      integer i,it,iat,inp,l,iord,np
      double precision ptt(0:3),mtt,ytt,btt,yt,yat,dely
c find the (anti)top
      it=0
      iat=0
      do i=nincoming+1,nexternal
        if (ipdg(i).eq.6) it=i
        if (ipdg(i).eq.-6) iat=i
      enddo
      if (it.eq.0 .or. iat.eq.0) return
c the NP power of the squared matrix element of this contribution
      np=-1
      do iord=1,nsplitorders
        if (trim(adjustl(ordernames(iord))).eq.'NP') then
          np=mod(orders_tag_plot/100**(iord-1),100)
        endif
      enddo
      if (np.lt.0) np=0
c squared orders NP^2=0,2,4 -> histogram set 0,1,2
      inp=np/2
      if (inp.lt.0 .or. inp.gt.2) return
      l=inp*10
      do i=0,3
        ptt(i)=p(i,it)+p(i,iat)
      enddo
      mtt=dsqrt(max(0d0,ptt(0)**2-ptt(1)**2-ptt(2)**2-ptt(3)**2))
      ytt=0.5d0*dlog((ptt(0)+ptt(3))/(ptt(0)-ptt(3)))
      btt=dabs(ptt(3))/ptt(0)
      yt=0.5d0*dlog((p(0,it)+p(3,it))/(p(0,it)-p(3,it)))
      yat=0.5d0*dlog((p(0,iat)+p(3,iat))/(p(0,iat)-p(3,iat)))
      dely=dabs(yt)-dabs(yat)
      call HwU_fill(l+1,mtt,wgts)
      if (dely.gt.0d0) then
        call HwU_fill(l+2,dabs(ytt),wgts)
        call HwU_fill(l+4,btt,wgts)
        call HwU_fill(l+6,mtt,wgts)
      else
        call HwU_fill(l+3,dabs(ytt),wgts)
        call HwU_fill(l+5,btt,wgts)
        call HwU_fill(l+7,mtt,wgts)
      endif
      return
      end
