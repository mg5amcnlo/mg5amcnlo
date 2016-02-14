      subroutine initialise(lpp, pdfid)
      implicit none
      integer lpp(2)
      integer pdfid
CF2PY INTENT(IN) :: LPP(2)
CF2PY INTENT(IN) :: PDF

      integer save_lpp(2)
      common/beam_type/save_lpp
      double precision value(20)
      character*20 parm(20)
c       Save the beam type information        
        save_lpp(1) = lpp(1)
        save_lpp(2) = lpp(2)

c       Initialise the pdf
        if (abs(lpp(1)).eq.1.or.abs(lpp(2)).eq.1)then
           value(1)=pdfid
           parm(1)='DEFAULT'
           call pdfset(parm,value)
        endif
        return
        end

      subroutine test_pdf()
      implicit none 
      integer lpp(2)
      double precision pdf, PDG2PDF
      lpp(1)=1
      lpp(2) =1
      call initialise(lpp, 230000)
      pdf = PDG2PDF(1,1,0.1,100)
      write(*,*) pdf
      end
     


      subroutine get_wgt(scales2, pdg, bjx, wgt, g, qcdpower, ymur, 
     &                   ymuf, n_ctr, new_wgt, all_wgt)
      implicit none
      double precision scales2(3,n_ctr)
      integer pdg(2, n_ctr)
      double precision bjx(2,n_ctr)
      double precision g(n_ctr)
      integer qcdpower(n_ctr)
      double precision wgt(3, n_ctr)
      double precision ymur, ymuf
      integer n_ctr
      double precision new_wgt
      double precision all_wgt(n_ctr)

CF2PY INTENT(IN) :: scales2(3, n_ctr)
CF2PY INTENT(IN) :: pdg(2, n_ctr)
CF2PY INTENT(IN) :: bjx(2, n_ctr)
CF2PY INTENT(IN) :: wgt(3, n_ctr)
CF2PY INTENT(IN) :: g(n_ctr)
CF2PY INTENT(IN) :: qcdpower(n_ctr)
CF2PY INTENT(IN) :: ymur
CF2PY INTENT(IN) :: ymuf
CF2PY depend(n_ctr) :: scales2, pdg, bjx, qcdpower
CF2PY INTENT(OUT) :: new_wgt
CF2PY INTENT(OUT) :: all_wgt(n_ctr)


      integer save_lpp(2)
      common/beam_type/save_lpp

c     LOCAL      
      integer i,kr
      double precision mu2_q
      double precision mu2_r, mu2_f
      double precision xlum
      integer lp, pd
      double precision pi
      parameter (pi=3.14159265358979323846d0)
c     debug
      double precision gs,add_wgt
c     external
      double precision alphas, pdg2pdf
      external alphas, pdg2pdf
      new_wgt = 0d0

      do i=1, n_ctr
         mu2_q=scales2(1,i)
         mu2_r=scales2(2,i)*ymuR**2
c Update the strong coupling
         mu2_f=scales2(3,i)*ymuF**2
c         gs = sqrt(4d0*pi*alphas(sqrt(mu2_r)))
c call the PDFs
         xlum=1d0
         LP=SIGN(1,save_LPP(1))
         pd=pdg(1,i)
         if (pd.eq.21) pd=0
         xlum=xlum*PDG2PDF(ABS(save_LPP(1)),pd*LP,bjx(1,i)
     &           ,DSQRT(mu2_f))
         LP=SIGN(1,save_LPP(2))
         pd=pdg(2,i)
         if (pd.eq.21) pd=0
         xlum=xlum*PDG2PDF(ABS(save_LPP(2)),pd*LP,bjx(2,i)
     &           ,DSQRT(mu2_f))
c         write(*,*) "xlum", xlum
c add the weights to the array
         add_wgt =  xlum * (wgt(1,i)+wgt(2,i)*log(mu2_r
     &              /mu2_q)+wgt(3,i)*log(mu2_f/mu2_q))*g(i)
     &              **QCDpower(i)
c         write(*,*) i, add_wgt
         all_wgt(i) = add_wgt
         new_wgt = new_wgt + add_wgt
c         wgts(iwgt,i)=wgts(iwgt,i)
c     &              *rwgt_muR_dep_fac(sqrt(mu2_r(kr)))
      enddo
      return
      end

