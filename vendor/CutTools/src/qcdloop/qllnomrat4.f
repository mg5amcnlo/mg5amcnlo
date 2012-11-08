      double complex function qllnomrat4(ms1,ms2,mt1,mt2)
************************************************************************
*     Author: R.K. Ellis                                               *
*     August, 2007.                                                    *
*     Lnomrat4(-s1,-s2,-t1,-t2)=                                       *
*     ln(1-((ms1-i*ep)*(ms2-i*ep))/((mt1-i*ep)/(mt2-i*ep)))            *
*     this function is hard-wired for sign of epsilon we must adjust   *
*     sign of s1,s2,t1,t2 to get the right sign for epsilon            *
************************************************************************
      implicit none
      include 'qlconstants.f'
      double precision ms1,ms2,mt1,mt2,prod,htheta
C--- define Heaviside theta function (=1 for x>0) and (0 for x < 0)
      htheta(ms1)=half+half*sign(one,ms1)
      prod=1d0-(ms1*ms2)/(mt1*mt2)
      qllnomrat4=dcmplx(dlog(abs(prod)))
     . -half*impi*dcmplx(htheta(-prod))
     . *dcmplx(htheta(-ms1)+htheta(-ms2)-htheta(-mt1)-htheta(-mt2))

      return
      end

