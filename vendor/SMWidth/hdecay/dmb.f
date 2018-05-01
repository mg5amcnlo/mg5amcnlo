 
      double precision function t134p_hdec(am1,am2,mu)
      implicit double precision (a-h,o-z)
      double precision m1,m2,mu,ll1,ll2
      complex*16 li2_hdec
      sp(a) = dreal(li2_hdec(dcmplx(a,0.d0)))
      pi = 4*datan(1.d0)
      zeta2 = pi**2/6
      if(am1.lt.am2)then
       m1 = am1
       m2 = am2
      else
       m1 = am2
       m2 = am1
      endif
      ll1 = dlog(mu**2/m1**2)
      ll2 = dlog(mu**2/m2**2)
      if(m1.eq.m2)then
       dummy = 7*(m1**2+m2**2)/2
     .       + m1**2*(ll1**2+3*ll1) + m2**2*(ll2**2+3*ll2)
     .       - m1**2/2*dlog(m1**2/m2**2)**2
      else
       dummy = 7*(m1**2+m2**2)/2
     .       + m1**2*(ll1**2+3*ll1) + m2**2*(ll2**2+3*ll2)
     .       + (m1**2-m2**2)*(dlog(m1**2/m2**2)*dlog(1-m1**2/m2**2)
     .                       + sp(m1**2/m2**2)-zeta2)
     .       - m1**2/2*dlog(m1**2/m2**2)**2
      endif
      t134p_hdec = dummy/mu**2
      return
      end
 
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
      double precision function t134_hdec(am1,am2,am3,mu)
      implicit double precision (a-h,o-z)
      double precision m1,m2,m3,mu,ll1,ll2,ll3
      complex*16 sp,li2_hdec,lam,ll,cx,cy,phi
      sp(cx) = li2_hdec(cx)
      lam(cx,cy) = (1-cx-cy)**2-4*cx*cy
      phi(cx,cy,ll) = (2*cdlog((1+cx-cy-ll)/2)*cdlog((1-cx+cy-ll)/2)
     .              -cdlog(cx)*cdlog(cy)+2*zeta2
     .              -2*sp((1+cx-cy-ll)/2)-2*sp((1-cx+cy-ll)/2))/ll
      eps = 1.d-15
      rim = dcmplx(1.d0,eps)
      pi = 4*datan(1.d0)
      zeta2 = pi**2/6
      m1 = dmin1(am1,am2,am3)
      m3 = dmax1(am1,am2,am3)
      if(m1.eq.am2.and.m3.eq.am3.or.m1.eq.am3.and.m3.eq.am2) m2 = am1
      if(m1.eq.am1.and.m3.eq.am3.or.m1.eq.am3.and.m3.eq.am1) m2 = am2
      if(m1.eq.am1.and.m3.eq.am2.or.m1.eq.am2.and.m3.eq.am1) m2 = am3
      cx = m1**2/m3**2*rim
      cy = m2**2/m3**2*rim
      ll1 = dlog(mu**2/m1**2)
      ll2 = dlog(mu**2/m2**2)
      ll3 = dlog(mu**2/m3**2)
      ll = cdsqrt(lam(cx,cy))
      dummy = 7*(m1**2+m2**2+m3**2)/2
     .      + m1**2*(ll1**2+3*ll1) + m2**2*(ll2**2+3*ll2)
     .      + m3**2*(ll3**2+3*ll3)
     .      +  (m1**2-m2**2-m3**2)/4*dlog(m2**2/m3**2)**2
     .      + (-m1**2+m2**2-m3**2)/4*dlog(m1**2/m3**2)**2
     .      + (-m1**2-m2**2+m3**2)/4*dlog(m1**2/m2**2)**2
     .      + m3**2/2*lam(cx,cy)*phi(cx,cy,ll)
      t134_hdec = dummy/mu**2
      return
      end
 
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
      double precision function
     .       felw_hdec(scale,amu,amg,amsb1,amsb2,amst1,amst2,amt)
      implicit double precision (b-h,o-q,s-z), complex*16 (a,r)
      double precision amg,amsb1,amsb2,amst1,amst2,amu,amt
      double precision mg,mb1,mb2,mt1,mt2,mu,mt
      double precision anomalous
      double precision a
      complex*16 sp,li2_hdec,xgl,xt1,xt2,xt,xb1,xb2
      sp(r) = li2_hdec(r)
      fi(a,b,c) = (a*b*log(a/b)+b*c*log(b/c)+c*a*log(c/a))
     .          / (a-b)/(b-c)/(a-c)
      t134p(a,b,c)  = t134p_hdec(a,b,c)
      t134(a,b,c,d) = t134_hdec(a,b,c,d)

      eps = 1.d-15
      pi = 4*datan(1.d0)
      zeta2 = pi**2/6

      fnorm = 1/amu**2

      cf = 4/3.d0

      rim = dcmplx(1.d0,eps)

      mt  = amt
      mu  = dabs(amu)
      mg  = amg
      mt1 = amst1
      mt2 = amst2
      mb1 = amsb1
      mb2 = amsb2

      xt  = amt**2/amu**2   * rim
      xgl = amg**2/amu**2   * rim
      xt1 = amst1**2/amu**2 * rim
      xt2 = amst2**2/amu**2 * rim
      xb1 = amsb1**2/amu**2 * rim
      xb2 = amsb2**2/amu**2 * rim

      r22=(4*log(xgl)**2*(-xt1**2*xt2+xt1*xt2**2+xt1-xt2)+4*log(xgl)*
     . log(xt1)*xt1*(-xt2**2+2*xt2-1)+4*log(xgl)*log(xt2)*xt2*(xt1**2
     . -2*xt1+1)+12*log(xgl)*(-xt1**2*xt2+xt1*xt2**2+xt1-xt2)+log(xt1
     . )**2*xt1*(xt2**2-2*xt2+1)+2*log(xt1)*xt1*(-xt1*xt2**2+2*xt1*
     . xt2-xt1-2*xt2**2+4*xt2-2)+log(xt2)**2*xt2*(-xt1**2+2*xt1-1)+2*
     . log(xt2)*xt2*(xt1**2*xt2+2*xt1**2-2*xt1*xt2-4*xt1+xt2+2)+t134p
     . (mu,mt1,mg)*xgl*(-xt1*xt2**2+2*xt1*xt2-xt1-xt2**2+2*xt2-1)+
     . t134p(mu,mt2,mg)*xgl*(xt1**2*xt2+xt1**2-2*xt1*xt2-2*xt1+xt2+1)
     . +14*(-xt1**2*xt2+xt1*xt2**2+xt1-xt2))/(2*(xt1**3*xt2**2-2*xt1
     . **3*xt2+xt1**3-xt1**2*xt2**3+3*xt1**2*xt2-2*xt1**2+2*xt1*xt2**
     . 3-3*xt1*xt2**2+xt1-xt2**3+2*xt2**2-xt2))

      r42=(8*log(xgl)**2*(xt1**2*xt2-xt1*xt2**2-xt1+xt2)+8*log(xgl)*
     . log(xt1)*xt1*(xt2**2-2*xt2+1)+8*log(xgl)*log(xt2)*xt2*(-xt1**2
     . +2*xt1-1)+24*log(xgl)*(xt1**2*xt2-xt1*xt2**2-xt1+xt2)+2*log(
     . xt1)**2*xt1*(-xt2**2+2*xt2-1)+log(xt1)*xt1*(5*xt1*xt2**2-10*
     . xt1*xt2+5*xt1+7*xt2**2-14*xt2+7)+2*log(xt2)**2*xt2*(xt1**2-2*
     . xt1+1)+log(xt2)*xt2*(-5*xt1**2*xt2-7*xt1**2+10*xt1*xt2+14*xt1-
     . 5*xt2-7)+2*t134p(mt1,mu,mg)*xgl*(xt1*xt2**2-2*xt1*xt2+xt1+xt2
     . **2-2*xt2+1)+2*t134p(mt2,mu,mg)*xgl*(-xt1**2*xt2-xt1**2+2*xt1*
     . xt2+2*xt1-xt2-1)+28*(xt1**2*xt2-xt1*xt2**2-xt1+xt2))/(2*(xt1**
     . 3*xt2**2-2*xt1**3*xt2+xt1**3-xt1**2*xt2**3+3*xt1**2*xt2-2*xt1
     . **2+2*xt1*xt2**3-3*xt1*xt2**2+xt1-xt2**3+2*xt2**2-xt2))

      ans3=t134(mt2,mg,mt,mg)*xgl*(-xb1*xgl*xt1+xb1*xgl-xb1*xt*xt1+
     . xb1*xt+xb1*xt1*xt2-xb1*xt2-xb2*xgl*xt1+xb2*xgl-xb2*xt*xt1+xb2*
     . xt+xb2*xt1*xt2-xb2*xt2+2*xgl**2*xt1-2*xgl**2+2*xgl*xt*xt1-2*
     . xgl*xt-2*xgl*xt1*xt2+2*xgl*xt2)+t134(mu,mb1,mt,mg)*xgl*(-xb1*
     . xb2*xt1+xb1*xb2*xt2+xb1*xgl*xt1-xb1*xgl*xt2-xb2*xt*xt1+xb2*xt*
     . xt2+xb2*xt1-xb2*xt2+xgl*xt*xt1-xgl*xt*xt2-xgl*xt1+xgl*xt2)+
     . t134(mu,mb2,mt,mg)*xgl*(-xb1*xb2*xt1+xb1*xb2*xt2-xb1*xt*xt1+
     . xb1*xt*xt2+xb1*xt1-xb1*xt2+xb2*xgl*xt1-xb2*xgl*xt2+xgl*xt*xt1-
     . xgl*xt*xt2-xgl*xt1+xgl*xt2)+t134(mu,mg,mt,mg)*xgl*(xb1*xgl*xt1
     . -xb1*xgl*xt2+xb1*xt*xt1-xb1*xt*xt2-xb1*xt1+xb1*xt2+xb2*xgl*xt1
     . -xb2*xgl*xt2+xb2*xt*xt1-xb2*xt*xt2-xb2*xt1+xb2*xt2-2*xgl**2*
     . xt1+2*xgl**2*xt2-2*xgl*xt*xt1+2*xgl*xt*xt2+2*xgl*xt1-2*xgl*xt2
     . )
      ans2=t134(mt1,mb1,mt,mg)*xgl*(-xb1*xb2*xt2+xb1*xb2+xb1*xgl*xt2-
     . xb1*xgl-xb2*xt*xt2+xb2*xt+xb2*xt1*xt2-xb2*xt1+xgl*xt*xt2-xgl*
     . xt-xgl*xt1*xt2+xgl*xt1)+t134(mt1,mb2,mt,mg)*xgl*(-xb1*xb2*xt2+
     . xb1*xb2-xb1*xt*xt2+xb1*xt+xb1*xt1*xt2-xb1*xt1+xb2*xgl*xt2-xb2*
     . xgl+xgl*xt*xt2-xgl*xt-xgl*xt1*xt2+xgl*xt1)+t134(mt1,mg,mt,mg)*
     . xgl*(xb1*xgl*xt2-xb1*xgl+xb1*xt*xt2-xb1*xt-xb1*xt1*xt2+xb1*xt1
     . +xb2*xgl*xt2-xb2*xgl+xb2*xt*xt2-xb2*xt-xb2*xt1*xt2+xb2*xt1-2*
     . xgl**2*xt2+2*xgl**2-2*xgl*xt*xt2+2*xgl*xt+2*xgl*xt1*xt2-2*xgl*
     . xt1)+t134(mt2,mb1,mt,mg)*xgl*(xb1*xb2*xt1-xb1*xb2-xb1*xgl*xt1+
     . xb1*xgl+xb2*xt*xt1-xb2*xt-xb2*xt1*xt2+xb2*xt2-xgl*xt*xt1+xgl*
     . xt+xgl*xt1*xt2-xgl*xt2)+t134(mt2,mb2,mt,mg)*xgl*(xb1*xb2*xt1-
     . xb1*xb2+xb1*xt*xt1-xb1*xt-xb1*xt1*xt2+xb1*xt2-xb2*xgl*xt1+xb2*
     . xgl-xgl*xt*xt1+xgl*xt+xgl*xt1*xt2-xgl*xt2)+ans3
      ans1=log(xb1)*log(xt1)*xb1*xt1*(-xb2*xt2+xb2+xgl*xt2-xgl)+log(
     . xb1)*log(xt2)*xb1*xt2*(xb2*xt1-xb2-xgl*xt1+xgl)+log(xb2)*log(
     . xt1)*xb2*xt1*(-xb1*xt2+xb1+xgl*xt2-xgl)+log(xb2)*log(xt2)*xb2*
     . xt2*(xb1*xt1-xb1-xgl*xt1+xgl)+log(xgl)*log(xt1)*xt1*(4*xb1*xb2
     . *xt2-4*xb1*xb2-3*xb1*xgl*xt2+3*xb1*xgl-3*xb2*xgl*xt2+3*xb2*xgl
     . +2*xgl**2*xt2-2*xgl**2)+log(xgl)*log(xt2)*xt2*(-4*xb1*xb2*xt1+
     . 4*xb1*xb2+3*xb1*xgl*xt1-3*xb1*xgl+3*xb2*xgl*xt1-3*xb2*xgl-2*
     . xgl**2*xt1+2*xgl**2)+log(xt1)**2*xt1*(-xb1*xb2*xt2+xb1*xb2+xb1
     . *xgl*xt2-xb1*xgl+xb2*xgl*xt2-xb2*xgl-xgl**2*xt2+xgl**2)+4*log(
     . xt1)*xt1*(xb1*xb2*xt2-xb1*xb2-xb1*xgl*xt2+xb1*xgl-xb2*xgl*xt2+
     . xb2*xgl+xgl**2*xt2-xgl**2)+log(xt2)**2*xt2*(xb1*xb2*xt1-xb1*
     . xb2-xb1*xgl*xt1+xb1*xgl-xb2*xgl*xt1+xb2*xgl+xgl**2*xt1-xgl**2)
     . +4*log(xt2)*xt2*(-xb1*xb2*xt1+xb1*xb2+xb1*xgl*xt1-xb1*xgl+xb2*
     . xgl*xt1-xb2*xgl-xgl**2*xt1+xgl**2)+ans2
      r52=ans1/(4*(xb1*xb2*xt1**2*xt2-xb1*xb2*xt1**2-xb1*xb2*xt1*xt2
     . **2+xb1*xb2*xt1+xb1*xb2*xt2**2-xb1*xb2*xt2-xb1*xgl*xt1**2*xt2+
     . xb1*xgl*xt1**2+xb1*xgl*xt1*xt2**2-xb1*xgl*xt1-xb1*xgl*xt2**2+
     . xb1*xgl*xt2-xb2*xgl*xt1**2*xt2+xb2*xgl*xt1**2+xb2*xgl*xt1*xt2
     . **2-xb2*xgl*xt1-xb2*xgl*xt2**2+xb2*xgl*xt2+xgl**2*xt1**2*xt2-
     . xgl**2*xt1**2-xgl**2*xt1*xt2**2+xgl**2*xt1+xgl**2*xt2**2-xgl**
     . 2*xt2))

      ans1=8*log(xgl)**2*(xt1**2*xt2-xt1**2-xt1*xt2**2+xt1+xt2**2-xt2
     . )+4*log(xgl)*log(xt1)*xt1*(4*xt1*xt2**2-8*xt1*xt2+4*xt1-3*xt2
     . **2+6*xt2-3)+4*log(xgl)*log(xt2)*xt2*(-4*xt1**2*xt2+3*xt1**2+8
     . *xt1*xt2-6*xt1-4*xt2+3)+12*log(xgl)*(xt1**2*xt2-xt1**2-xt1*xt2
     . **2+xt1+xt2**2-xt2)+log(xt1)**2*xt1*(-8*xt1*xt2**2+16*xt1*xt2-
     . 8*xt1+5*xt2**2-10*xt2+5)+4*log(xt1)*xt1*(3*xt1*xt2**2-6*xt1*
     . xt2+3*xt1-2*xt2**2+4*xt2-2)+log(xt2)**2*xt2*(8*xt1**2*xt2-5*
     . xt1**2-16*xt1*xt2+10*xt1+8*xt2-5)+4*log(xt2)*xt2*(-3*xt1**2*
     . xt2+2*xt1**2+6*xt1*xt2-4*xt1-3*xt2+2)+4*t134p(mt1,mt1,mg)*xgl*
     . (xt1*xt2**2-2*xt1*xt2+xt1+xt2**2-2*xt2+1)+4*t134p(mt2,mt2,mg)*
     . xgl*(-xt1**2*xt2-xt1**2+2*xt1*xt2+2*xt1-xt2-1)+4*t134p(mu,mt1,
     . mg)*xgl*(-xt1*xt2**2+2*xt1*xt2-xt1-xt2**2+2*xt2-1)+4*t134p(mu,
     . mt2,mg)*xgl*(xt1**2*xt2+xt1**2-2*xt1*xt2-2*xt1+xt2+1)+8*(xt1**
     . 2*xt2-xt1**2-xt1*xt2**2+xt1+xt2**2-xt2)
      r72=ans1/(8*(xt1**3*xt2**2-2*xt1**3*xt2+xt1**3-xt1**2*xt2**3+3*
     . xt1**2*xt2-2*xt1**2+2*xt1*xt2**3-3*xt1*xt2**2+xt1-xt2**3+2*xt2
     . **2-xt2))

      ans5=2*((2*((14*xt2-15+14*xt1+5*xt)*xt-((xt2-3)*xt2+xt1**2+(4*
     . xt2-3)*xt1))*xgl**3+(4*xt2-3+4*xt1-8*xt)*xgl**4-((2*xt+3)*(xt-
     . xt1)**2*(xt-xt2)**2+2*xgl**5)-(2*(2*((3*xt2-2)*xt2+3*xt1**2)+(
     . 5*xt2-4)*xt1)*xt-((4*(xt2-3)*xt1-3*xt2)*xt2+(4*xt2-3)*xt1**2)-
     . 2*(24*xt2-23+24*xt1)*xt**2-10*xt**3)*xgl**2-2*((2*((3*xt2-2)*
     . xt2+3*xt1**2)+(5*xt2-4)*xt1)*xt**2+(3*((xt2-1)*xt1**2-xt2**2)+
     . (3*xt2+5)*xt1*xt2)*xt-(14*xt2-15+14*xt1)*xt**3+((xt2-3)*xt1-3*
     . xt2)*xt1*xt2+4*xt**4)*xgl)*(xt1-1)*(xt2-1)-(2*(xt+xt1)*xgl-(xt
     . -xt1)**2-xgl**2)*(2*(xt+xt2)*xgl-(xt-xt2)**2-xgl**2)*(xt-1+xgl
     . )*(xt2-2+xt1)*t134(mu,mt,mg,mg)*xgl)*(xt1-xt2)
      ans4=-2*((2*((xt-xt1)**2*(xt-xt2)**2+xgl**4-(6*xt**2+10*xt*xt1+
     . 10*xt*xt2-12*xt-xt1**2-4*xt1*xt2-xt2**2)*xgl**2-2*(xt1+xt2-xt)
     . *xgl**3+2*(((2*xt2-3)*xt2+2*xt1**2+(7*xt2-3)*xt1)*xt-(5*xt2-6+
     . 5*xt1)*xt**2-(xt1+xt2)*xt1*xt2+xt**3)*xgl)*(xt1-1)*(xt2-1)+(2*
     . (xt+xt1)*xgl-(xt-xt1)**2-xgl**2)*(2*(xt+xt2)*xgl-(xt-xt2)**2-
     . xgl**2)*(xt2-2+xt1)*log(xgl))*(xt1-xt2)+(2*(xt+xt1)*xgl+(xt-
     . xt1)**2*(xt1-2)-xgl**2*xt1)*(2*(xt+xt2)*xgl-(xt-xt2)**2-xgl**2
     . )*(log(xgl)-log(xt1))*(xt2-1)**2-(2*(xt+xt1)*xgl-(xt-xt1)**2-
     . xgl**2)*(2*(xt+xt2)*xgl+(xt-xt2)**2*(xt2-2)-xgl**2*xt2)*(log(
     . xgl)-log(xt2))*(xt1-1)**2)*(log(xgl)-log(xt))*xt+ans5
      ans3=-(4*((xt-xt2**2+xt2)*(xt-xt2)**2+xgl**3-((xt2+1)*xt2+xt)*
     . xgl**2-((xt+4*xt2**2)*xt-(2*xt2-1)*xt2**2)*xgl)+(((xt+xt2)*(xt
     . -xt2)*(xt2-2)-4*xt*xt2)*xgl+(xt+2*xt2)*(xt2-2)*xgl**2-(xt-xt2)
     . **2*(xt2-2)*xt-(xt2-2)*xgl**3)*(log(xgl)-log(xt2)))*(2*(xt+xt1
     . )*xgl-(xt-xt1)**2-xgl**2)*(log(xgl)-log(xt2))*(xt1-1)**2+(4*((
     . xt-xt1**2+xt1)*(xt-xt1)**2+xgl**3-((xt1+1)*xt1+xt)*xgl**2-((xt
     . +4*xt1**2)*xt-(2*xt1-1)*xt1**2)*xgl)+(((xt+xt1)*(xt-xt1)*(xt1-
     . 2)-4*xt*xt1)*xgl+(xt+2*xt1)*(xt1-2)*xgl**2-(xt-xt1)**2*(xt1-2)
     . *xt-(xt1-2)*xgl**3)*(log(xgl)-log(xt1)))*(2*(xt+xt2)*xgl-(xt-
     . xt2)**2-xgl**2)*(log(xgl)-log(xt1))*(xt2-1)**2+ans4
      ans2=(2*(((xt+2*xt2)*xt+(xt2-4)*xt2)*xgl+(xt2+2+xt)*xgl**2-(xt+
     . xt2-2)*(xt-xt2)**2-xgl**3)*(xt1-1)**2*t134(mt2,mt,mg,mg)*xgl-(
     . 2*(xt+xt2)*xgl-(xt-xt2)**2-xgl**2)*(xt2-2+xt1)*(log(xgl)+4)*(
     . xgl+xt)*(xt1-xt2)*log(xgl))*(2*(xt+xt1)*xgl-(xt-xt1)**2-xgl**2
     . )-2*(((xt+2*xt1)*xt+(xt1-4)*xt1)*xgl+(xt1+2+xt)*xgl**2-(xt+xt1
     . -2)*(xt-xt1)**2-xgl**3)*(2*(xt+xt2)*xgl-(xt-xt2)**2-xgl**2)*(
     . xt2-1)**2*t134(mt1,mt,mg,mg)*xgl+((2*(xt1+xt2)-xgl)*xgl**3-(xt
     . -xt1)**2*(xt-xt2)**2+(2*xt**2+6*xt*xt1+6*xt*xt2-8*xt-xt1**2-4*
     . xt1*xt2-xt2**2)*xgl**2-2*(((xt2-2)*xt2+xt1**2+2*(3*xt2-1)*xt1)
     . *xt-(3*xt2-4+3*xt1)*xt**2-(xt1+xt2)*xt1*xt2)*xgl)*(log(xgl)-
     . log(xt))**2*(xt1-xt2)*(xt1-1)*(xt2-1)*xt+ans3
      ans1=-ans2
      r82=ans1/(4*((xt-xt1)**2+xgl**2-2*(xt+xt1)*xgl)*((xt-xt2)**2+
     . xgl**2-2*(xt+xt2)*xgl)*(xt1-xt2)*(xt1-1)**2*(xt2-1)**2)

      ans2=4*log(xt2)*xt2*(-3*xt1**3*xt2+2*xt1**3+3*xt1**2*xt2**2+4*
     . xt1**2*xt2-4*xt1**2-6*xt1*xt2**2+xt1*xt2+2*xt1+3*xt2**2-2*xt2)
     . +4*(xt1**3*xt2-xt1**3-2*xt1**2*xt2**2+xt1**2*xt2+xt1**2+xt1*
     . xt2**3+xt1*xt2**2-2*xt1*xt2-xt2**3+xt2**2)
      ans1=4*log(xgl)*log(xt1)*xt1*(2*xt1**2*xt2**2-4*xt1**2*xt2+2*
     . xt1**2-2*xt1*xt2**3+3*xt1*xt2**2-xt1+xt2**3-2*xt2**2+xt2)+4*
     . log(xgl)*log(xt2)*xt2*(-2*xt1**3*xt2+xt1**3+2*xt1**2*xt2**2+3*
     . xt1**2*xt2-2*xt1**2-4*xt1*xt2**2+xt1+2*xt2**2-xt2)+4*log(xgl)*
     . (xt1**3*xt2-xt1**3-2*xt1**2*xt2**2+xt1**2*xt2+xt1**2+xt1*xt2**
     . 3+xt1*xt2**2-2*xt1*xt2-xt2**3+xt2**2)+log(xt1)**2*xt1*(-6*xt1
     . **2*xt2**2+12*xt1**2*xt2-6*xt1**2+2*xt1*xt2**3-xt1*xt2**2-4*
     . xt1*xt2+3*xt1+xt2**3-2*xt2**2+xt2)+4*log(xt1)*log(xt2)*xt1*xt2
     . *(xt1**2*xt2-xt1**2+xt1*xt2**2-4*xt1*xt2+3*xt1-xt2**2+3*xt2-2)
     . +4*log(xt1)*xt1*(3*xt1**2*xt2**2-6*xt1**2*xt2+3*xt1**2-3*xt1*
     . xt2**3+4*xt1*xt2**2+xt1*xt2-2*xt1+2*xt2**3-4*xt2**2+2*xt2)+log
     . (xt2)**2*xt2*(2*xt1**3*xt2+xt1**3-6*xt1**2*xt2**2-xt1**2*xt2-2
     . *xt1**2+12*xt1*xt2**2-4*xt1*xt2+xt1-6*xt2**2+3*xt2)+ans2
      r92=ans1/(8*(xt1**4*xt2**2-2*xt1**4*xt2+xt1**4-2*xt1**3*xt2**3+
     . 2*xt1**3*xt2**2+2*xt1**3*xt2-2*xt1**3+xt1**2*xt2**4+2*xt1**2*
     . xt2**3-6*xt1**2*xt2**2+2*xt1**2*xt2+xt1**2-2*xt1*xt2**4+2*xt1*
     . xt2**3+2*xt1*xt2**2-2*xt1*xt2+xt2**4-2*xt2**3+xt2**2))

      ans6=-sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2)*(((xt1-xt2)*xt1+
     . 4*xt**2)*(log(xt+xt1-xgl-sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl
     . **2))-log(xt+xt1-xgl+sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))
     . )+4*log(xt-xt1-xgl+sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))*
     . xt**2-4*log(xt-xt1-xgl-sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2
     . ))*xt**2-(xt1-xt2)*log(-(xt-xt1+xgl+sqrt(-2*(xt+xt1)*xgl+(xt-
     . xt1)**2+xgl**2)))*xt1+(xt1-xt2)*log(-(xt-xt1+xgl-sqrt(-2*(xt+
     . xt1)*xgl+(xt-xt1)**2+xgl**2)))*xt1)*(xt-xt1+xgl)*xt2
      ans5=-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)*(((xt1-xt2)*xt2-
     . 4*xt**2)*(log(xt+xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl
     . **2))-log(xt+xt2-xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))
     . )-4*log(xt-xt2-xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))*
     . xt**2+4*log(xt-xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2
     . ))*xt**2-(xt1-xt2)*log(-(xt-xt2+xgl+sqrt(-2*(xt+xt2)*xgl+(xt-
     . xt2)**2+xgl**2)))*xt2+(xt1-xt2)*log(-(xt-xt2+xgl-sqrt(-2*(xt+
     . xt2)*xgl+(xt-xt2)**2+xgl**2)))*xt2)*(xt-xt2+xgl)*xt1+ans6
      ans4=4*(xgl+xt-xt1)*(xgl-xt-xt1)*log(xt-xt1-xgl+sqrt(-2*(xt+xt1
     . )*xgl+(xt-xt1)**2+xgl**2))*xt**2*xt2+4*(xgl+xt-xt1)*(xgl-xt-
     . xt1)*log(xt-xt1-xgl-sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))*
     . xt**2*xt2+(xgl+xt-xt2)*(xgl-xt-xt2)*(xt1-xt2)*log(-(xt-xt2+xgl
     . +sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)))*xt1*xt2+(xgl+xt-
     . xt2)*(xgl-xt-xt2)*(xt1-xt2)*log(-(xt-xt2+xgl-sqrt(-2*(xt+xt2)*
     . xgl+(xt-xt2)**2+xgl**2)))*xt1*xt2+(xgl+xt-xt1)*(xgl-xt-xt1)*(
     . xt1-xt2)*log(-(xt-xt1+xgl+sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl
     . **2)))*xt1*xt2+(xgl+xt-xt1)*(xgl-xt-xt1)*(xt1-xt2)*log(-(xt-
     . xt1+xgl-sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2)))*xt1*xt2+
     . ans5
      ans3=(2*((xt1+xt2-2*xt-2*xgl)*(xt1-xt2)+(xt1-xt2-8*xt)*log(xt2)
     . *xt2+(xt1-xt2+8*xt)*log(xt1)*xt1-6*(xt1-xt2)*log(xt)*xt-(xt1+
     . xt2+2*xt)*(xt1-xt2)*log(xgl))*xt*xt2+(xgl+xt-xt2)*(xgl-xt-xt2)
     . *(4*xt**2-xt1*xt2+xt2**2)*log(xt+xt2-xgl+sqrt(-2*(xt+xt2)*xgl+
     . (xt-xt2)**2+xgl**2))+(xgl+xt-xt2)*(xgl-xt-xt2)*(4*xt**2-xt1*
     . xt2+xt2**2)*log(xt+xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+
     . xgl**2)))*xt1-(xgl+xt-xt1)*(xgl-xt-xt1)*(4*xt**2+xt1**2-xt1*
     . xt2)*log(xt+xt1-xgl+sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))*
     . xt2-(xgl+xt-xt1)*(xgl-xt-xt1)*(4*xt**2+xt1**2-xt1*xt2)*log(xt+
     . xt1-xgl-sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))*xt2-4*(xgl+
     . xt-xt2)*(xgl-xt-xt2)*log(xt-xt2-xgl+sqrt(-2*(xt+xt2)*xgl+(xt-
     . xt2)**2+xgl**2))*xt**2*xt1-4*(xgl+xt-xt2)*(xgl-xt-xt2)*log(xt-
     . xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))*xt**2*xt1+
     . ans4
      ans7=((xt1-1)*log(xt2)*xt2-(xt2-1)*log(xt1)*xt1)
      ans2=ans3*ans7
      ans1=-ans2
      rat2=ans1/(16*(xt1-xt2)**2*(xt1-1)*(xt2-1)*xt**2*xt1*xt2)

      ans5=((log(-(xt-xt2+xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2
     . )))-log(xt+xt2-xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))+
     . log(-(xt-xt2+xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)))-
     . log(xt+xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)))*(xt+
     . xt2-xgl)*(xt-xt2+xgl)-2*(xt1+xt2-2*xgl)*xt+(log(-(xt-xt1+xgl-
     . sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2)))-log(xt+xt1-xgl+sqrt
     . (-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))+log(-(xt-xt1+xgl+sqrt(-2
     . *(xt+xt1)*xgl+(xt-xt1)**2+xgl**2)))-log(xt+xt1-xgl-sqrt(-2*(xt
     . +xt1)*xgl+(xt-xt1)**2+xgl**2)))*(xt+xt1-xgl)*(xt-xt1+xgl))*(
     . xt1-xt2)*log(xgl)
      ans4=sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)*(log(xt+xt2-xgl-
     . sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))-log(xt+xt2-xgl+sqrt(
     . -2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))-log(-(xt-xt2+xgl+sqrt(-2*
     . (xt+xt2)*xgl+(xt-xt2)**2+xgl**2)))+log(-(xt-xt2+xgl-sqrt(-2*(
     . xt+xt2)*xgl+(xt-xt2)**2+xgl**2))))*(xt-xt2+xgl)*((xt1-1)*log(
     . xt2)*xt2-(xt2-1)*log(xt1)*xt1)+2*((log(xgl)-log(xt2))**2*(2*xt
     . -xt2)*(xt1-1)*xt2-2*(xt1-xt2)*log(xgl)**2*xt-(log(xgl)-log(xt1
     . ))**2*(2*xt-xt1)*(xt2-1)*xt1)*xt+sqrt(-2*(xt+xt1)*xgl+(xt-xt1)
     . **2+xgl**2)*(log(xt+xt1-xgl-sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+
     . xgl**2))-log(xt+xt1-xgl+sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**
     . 2))-log(-(xt-xt1+xgl+sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))
     . )+log(-(xt-xt1+xgl-sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))))
     . *(xt-xt1+xgl)*((xt1-1)*log(xt2)*xt2-(xt2-1)*log(xt1)*xt1)+ans5
      ans3=-ans4
      ans2=-(((log(-(xt-xt2+xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl
     . **2)))-log(xt+xt2-xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)
     . )+log(-(xt-xt2+xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)))-
     . log(xt+xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)))*(xt+
     . xt2-xgl)*(xt-xt2+xgl)-2*(xt1+xt2-2*xgl)*xt+(log(-(xt-xt1+xgl-
     . sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2)))-log(xt+xt1-xgl+sqrt
     . (-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))+log(-(xt-xt1+xgl+sqrt(-2
     . *(xt+xt1)*xgl+(xt-xt1)**2+xgl**2)))-log(xt+xt1-xgl-sqrt(-2*(xt
     . +xt1)*xgl+(xt-xt1)**2+xgl**2)))*(xt+xt1-xgl)*(xt-xt1+xgl))*(
     . xt2-1)-2*((log(xgl)-log(xt2))*xt2-log(xgl))*(xt1-xt2)*xt)*(log
     . (xgl)-log(xt1))*xt1+ans3
      ans1=(((log(-(xt-xt2+xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**
     . 2)))-log(xt+xt2-xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))+
     . log(-(xt-xt2+xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)))-
     . log(xt+xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)))*(xt+
     . xt2-xgl)*(xt-xt2+xgl)-2*(xt1+xt2-2*xgl)*xt+(log(-(xt-xt1+xgl-
     . sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2)))-log(xt+xt1-xgl+sqrt
     . (-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))+log(-(xt-xt1+xgl+sqrt(-2
     . *(xt+xt1)*xgl+(xt-xt1)**2+xgl**2)))-log(xt+xt1-xgl-sqrt(-2*(xt
     . +xt1)*xgl+(xt-xt1)**2+xgl**2)))*(xt+xt1-xgl)*(xt-xt1+xgl))*(
     . xt1-1)-2*(xt1-xt2)*log(xgl)*xt)*(log(xgl)-log(xt2))*xt2+ans2
      rlt2=ans1/(8*(xt1-xt2)*(xt1-1)*(xt2-1)*xt**2)

      ans3=(2*(((log(xgl)-log(xt2))*(xt1-1)**2*xt2-(xt1-xt2)**2*log(
     . xgl)+(xt2+1-7*xt)*(xt2-1)*xt1)*xt1-(7*xt1**2-3*xt2-2*(xt2+1)*
     . xt1)*(xt2-1)*xgl)*xt1+(2*((2*(xt2+1)*xt1+3*xt2)*xt-2*xt1**3)*
     . xt1-(log(xt-xt1-xgl-sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))-
     . log(xt+xt1-xgl-sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))+log(
     . xt-xt1-xgl+sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))-log(xt+
     . xt1-xgl+sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2)))*(xt+xt1-xgl
     . )*(xt-xt1+xgl)*(xt1**2-xt2))*(xt2-1))*(log(xgl)-log(xt1))
      ans2=sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2)*(log(xt+xt1-xgl-
     . sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))-log(xt+xt1-xgl+sqrt(
     . -2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))+log(xt-xt1-xgl+sqrt(-2*(
     . xt+xt1)*xgl+(xt-xt1)**2+xgl**2))-log(xt-xt1-xgl-sqrt(-2*(xt+
     . xt1)*xgl+(xt-xt1)**2+xgl**2)))*((xt1**2-xt2)*(xt2-1)*log(xt1)-
     . (xt1-1)**2*log(xt2)*xt2-(xt1-xt2)*(xt1-1)*(xt2-1))*(xt-xt1+xgl
     . )+((log(xt-xt1-xgl-sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))-
     . log(xt+xt1-xgl-sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))+log(
     . xt-xt1-xgl+sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))-log(xt+
     . xt1-xgl+sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2)))*(xt+xt1-xgl
     . )*(xt-xt1+xgl)+2*(5*xt+xt1+5*xgl)*xt1)*((log(xgl)-log(xt2))*(
     . xt1-1)**2*xt2-(xt1-xt2)**2*log(xgl))+ans3
      ans1=2*(((log(xgl)-log(xt2))**2*(xt1-1)**2*xt2-(xt1-xt2)**2*log
     . (xgl)**2)*(xgl+xt)+(xt2+1-5*xt)*(xt2-1)*xt1**2)*xt1-(log(xt-
     . xt1-xgl-sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))-log(xt+xt1-
     . xgl-sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))+log(xt-xt1-xgl+
     . sqrt(-2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2))-log(xt+xt1-xgl+sqrt(
     . -2*(xt+xt1)*xgl+(xt-xt1)**2+xgl**2)))*(xt+xt1-xgl)*(xt-xt1+xgl
     . )*(xt1-xt2)*(xt1-1)*(xt2-1)-2*(xt+xt1+xgl)*(log(xgl)-log(xt1))
     . **2*(xt1**2-xt2)*(xt2-1)*xt1-2*(5*xt*xt2+xt1**3+5*(xt1-xt2)*(
     . xt1-1)*xgl-(5*(xt2+1)*xt-xt2)*xt1)*(xt2-1)*xt1-2*((log(xgl)-
     . log(xt1))*(xt1**2-xt2)*(xt2-1)-(log(xgl)-log(xt2))*(xt1-1)**2*
     . xt2+((xt1-xt2)*log(xgl)+(xt1-1)*(xt2-1))*(xt1-xt2))*(log(xgl)-
     . log(xt))*xt*xt1+ans2
      rmt12=ans1/(4*(xt1-xt2)**2*(xt1-1)**2*(xt2-1)*xt1)

      ans3=((log(xt-xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))
     . -log(xt+xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))+log(
     . xt-xt2-xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))-log(xt+
     . xt2-xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)))*(xt+xt2-xgl
     . )*(xt-xt2+xgl)+2*(log(xgl)-log(xt2))*xt2**2+2*(5*xt+xt2+5*xgl)
     . *xt2)*(log(xgl)-log(xt1))*(xt2-1)**2*xt1+(2*(((2*xt2+3)*xt+xt2
     . **2)*xt1**2-(xt1-xt2)**2*log(xgl)*xt2-((7*xt2**2+3)*xt+2*xt2**
     . 3)*xt1)*xt2+(log(xt-xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+
     . xgl**2))-log(xt+xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**
     . 2))+log(xt-xt2-xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))-
     . log(xt+xt2-xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)))*(xt+
     . xt2-xgl)*(xt-xt2+xgl)*(xt1-xt2**2)*(xt1-1)+2*(((7*xt-1)*xt2-2*
     . (xt-xt2**2))*xt2-((7*xt2-2)*xt2-(2*xt2+3)*xt1)*(xt1-1)*xgl)*
     . xt2)*(log(xgl)-log(xt2))
      ans2=-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)*(log(xt+xt2-xgl-
     . sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))-log(xt+xt2-xgl+sqrt(
     . -2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))+log(xt-xt2-xgl+sqrt(-2*(
     . xt+xt2)*xgl+(xt-xt2)**2+xgl**2))-log(xt-xt2-xgl-sqrt(-2*(xt+
     . xt2)*xgl+(xt-xt2)**2+xgl**2)))*((xt1-xt2**2)*(xt1-1)*log(xt2)+
     . (xt2-1)**2*log(xt1)*xt1-(xt1-xt2)*(xt1-1)*(xt2-1))*(xt-xt2+xgl
     . )-((log(xt-xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))-
     . log(xt+xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))+log(
     . xt-xt2-xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))-log(xt+
     . xt2-xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)))*(xt+xt2-xgl
     . )*(xt-xt2+xgl)+2*(5*xt+xt2+5*xgl)*xt2)*(xt1-xt2)**2*log(xgl)+
     . ans3
      ans1=(log(xt-xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))-
     . log(xt+xt2-xgl-sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))+log(
     . xt-xt2-xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2))-log(xt+
     . xt2-xgl+sqrt(-2*(xt+xt2)*xgl+(xt-xt2)**2+xgl**2)))*(xt+xt2-xgl
     . )*(xt-xt2+xgl)*(xt1-xt2)*(xt1-1)*(xt2-1)+2*((log(xgl)-log(xt1)
     . )**2*(xt2-1)**2*xt1-(xt1-xt2)**2*log(xgl)**2)*(xgl+xt)*xt2+2*(
     . xt+xt2+xgl)*(log(xgl)-log(xt2))**2*(xt1-xt2**2)*(xt1-1)*xt2+2*
     . (5*xt+xt2+5*xgl)*(xt1-xt2)*(xt1-1)*(xt2-1)*xt2+2*((log(xgl)-
     . log(xt1))*(xt2-1)**2*xt1+(log(xgl)-log(xt2))*(xt1-xt2**2)*(xt1
     . -1)-((xt1-xt2)*log(xgl)-(xt1-1)*(xt2-1))*(xt1-xt2))*(log(xgl)-
     . log(xt))*xt*xt2+ans2
      rmt22=ans1/(4*(xt1-xt2)**2*(xt1-1)*(xt2-1)**2*xt2)

c     relw = r22+r42+r52+r72+r82+r92+rat2+rlt2+rmt12+rmt22
      relw = r22+r42+r52+r72+r82+r92+rlt2+rmt12+rmt22
      bo  = fi(amst1**2,amst2**2,amu**2)
      relw = cf*relw/bo

      anomalous =-cf
      finscale = 2*dlog(scale**2/amg**2)
      felw_hdec = dreal(relw)*fnorm + anomalous + finscale

c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)'dat: ',dreal(cf*rat2/bo)*fnorm
c     write(6,*)'tot: ',felw_hdec
c     write(6,*)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)
c     write(6,*)'ind: ',dreal(r22),dreal(r42),dreal(r52),dreal(r72),
c    . dreal(r82),dreal(r92),dreal(rat2),dreal(rlt2),dreal(rmt12),
c    . dreal(rmt22)
c     write(6,*)'     ',-1/4.d0,3/4.d0,-1/2.d0,-1/12.d0,-1/3.d0,1/6.d0,
c    .       (5+3*dlog(amg**2/amt**2))/8,0.d0,1/2.d0,1/2.d0
c     write(6,*)
c     write(6,*)'dat: ',dreal(cf*rat2/bo)*fnorm
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      return
      end
 
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
      double precision function fqcd_hdec(scale,amt,amg,amsb1,amsb2,
     .                         amst1,amst2,amsu1,amsu2,amsd1,amsd2,xxt)
      implicit double precision (b-h,o-q,s-z), complex*16 (a,r)
      double precision amt,amg,amsb1,amsb2,amst1,amst2,
     .                 amsu1,amsu2,amsd1,amsd2
      double precision mt,mg,mb1,mb2,mt1,mt2,ms1,ms2,mu
      complex*16 sp,li2_hdec,xt,xgl,xt1,xt2,xb1,xb2,xs1,xs2,xq
      double precision m,mq
      double precision anomalous
      double precision a
      complex*16 xp,xm
      sp(r) = li2_hdec(r)
      xp(m) = (amg**2+amt**2-m**2)/2/amg**2/rim
     . +cdsqrt(((amg**2+amt**2-m**2)/2/amg**2/rim)**2-amt**2/amg**2/rim)
      xm(m) = (amg**2+amt**2-m**2)/2/amg**2/rim
     . -cdsqrt(((amg**2+amt**2-m**2)/2/amg**2/rim)**2-amt**2/amg**2/rim)
      fi(a,b,c) = (a*b*log(a/b)+b*c*log(b/c)+c*a*log(c/a))
     .          / (a-b)/(b-c)/(a-c)
      t134p(a,b,c)  = t134p_hdec(a,b,c)
      t134(a,b,c,d) = t134_hdec(a,b,c,d)

      eps = 1.d-15
      pi = 4*datan(1.d0)
      zeta2 = pi**2/6

      ca = 3
      cf = 4/3.d0
      tr = 1/2.d0
      nu = 2
      nd = 2
      nf = nu+nd+1

      fnorm = 4/amg**2

      rim = dcmplx(1.d0,eps)

      mq  = amt
      mt  = amt
      mg  = amg
      mu  = amg
      mt1 = amsb1
      mt2 = amsb2
      mb1 = amsb1
      mb2 = amsb2

      xq  = amt**2/amg**2   * rim
      xt  = amt**2/amg**2   * rim
      xgl = amg**2/amg**2   * rim
      xt1 = amsb1**2/amg**2 * rim
      xt2 = amsb2**2/amg**2 * rim
      xb1 = amsb1**2/amg**2 * rim
      xb2 = amsb2**2/amg**2 * rim

c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c--comparison with luminita
c     amsu1 =  998.540d0
c     amsu2 =  999.385d0
c     amsd1 = 1001.770d0
c     amsd2 = 1001.310d0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      r12=(log(xb1)**2*xb1*(-2*xb1*xb2**2+4*xb1*xb2-2*xb1+xb2**2-2*
     . xb2+1)+4*log(xb1)*xb1*(xb1*xb2**2-2*xb1*xb2+xb1-xb2**2+2*xb2-1
     . )+log(xb2)**2*xb2*(2*xb1**2*xb2-xb1**2-4*xb1*xb2+2*xb1+2*xb2-1
     . )+4*log(xb2)*xb2*(-xb1**2*xb2+xb1**2+2*xb1*xb2-2*xb1-xb2+1)+4*
     . t134p(mb1,mb1,mg)*xb1*(xb2**2-2*xb2+1)+4*t134p(mb1,mg,mg)*(-
     . xb1*xb2**2+2*xb1*xb2-xb1-xb2**2+2*xb2-1)+4*t134p(mb2,mb2,mg)*
     . xb2*(-xb1**2+2*xb1-1)+4*t134p(mb2,mg,mg)*(xb1**2*xb2+xb1**2-2*
     . xb1*xb2-2*xb1+xb2+1)+4*t134p(mg,mg,mg)*(-xb1**2+2*xb1+xb2**2-2
     . *xb2))/(16*(xb1**3*xb2**2-2*xb1**3*xb2+xb1**3-xb1**2*xb2**3+3*
     . xb1**2*xb2-2*xb1**2+2*xb1*xb2**3-3*xb1*xb2**2+xb1-xb2**3+2*xb2
     . **2-xb2))

      r22=(4*log(xgl)**2*(-xt1**2*xt2+xt1*xt2**2+xt1-xt2)+4*log(xgl)*
     . log(xt1)*xt1*(-xt2**2+2*xt2-1)+4*log(xgl)*log(xt2)*xt2*(xt1**2
     . -2*xt1+1)+12*log(xgl)*(-xt1**2*xt2+xt1*xt2**2+xt1-xt2)+log(xt1
     . )**2*xt1*(xt2**2-2*xt2+1)+2*log(xt1)*xt1*(-xt1*xt2**2+2*xt1*
     . xt2-xt1-2*xt2**2+4*xt2-2)+log(xt2)**2*xt2*(-xt1**2+2*xt1-1)+2*
     . log(xt2)*xt2*(xt1**2*xt2+2*xt1**2-2*xt1*xt2-4*xt1+xt2+2)+t134p
     . (mu,mt1,mg)*xgl*(-xt1*xt2**2+2*xt1*xt2-xt1-xt2**2+2*xt2-1)+
     . t134p(mu,mt2,mg)*xgl*(xt1**2*xt2+xt1**2-2*xt1*xt2-2*xt1+xt2+1)
     . +14*(-xt1**2*xt2+xt1*xt2**2+xt1-xt2))/(2*(xt1**3*xt2**2-2*xt1
     . **3*xt2+xt1**3-xt1**2*xt2**3+3*xt1**2*xt2-2*xt1**2+2*xt1*xt2**
     . 3-3*xt1*xt2**2+xt1-xt2**3+2*xt2**2-xt2))

      r32=(log(xb1)**2*xb1*(-xb2**2+2*xb2-1)+2*log(xb1)*xb1*(2*xb1*
     . xb2**2-4*xb1*xb2+2*xb1+xb2**2-2*xb2+1)+log(xb2)**2*xb2*(xb1**2
     . -2*xb1+1)+2*log(xb2)*xb2*(-2*xb1**2*xb2-xb1**2+4*xb1*xb2+2*xb1
     . -2*xb2-1)+2*t134p(mb1,mg,mg)*(2*xb1*xb2**2-4*xb1*xb2+2*xb1-xb2
     . **2+2*xb2-1)+2*t134p(mb2,mg,mg)*(-2*xb1**2*xb2+xb1**2+4*xb1*
     . xb2-2*xb1-2*xb2+1)+3*t134p(mg,mg,mg)*(xb1**2*xb2-xb1**2-xb1*
     . xb2**2+xb1+xb2**2-xb2)+14*(xb1**2*xb2-xb1*xb2**2-xb1+xb2))/(8*
     . (xb1**3*xb2**2-2*xb1**3*xb2+xb1**3-xb1**2*xb2**3+3*xb1**2*xb2-
     . 2*xb1**2+2*xb1*xb2**3-3*xb1*xb2**2+xb1-xb2**3+2*xb2**2-xb2))

      r42=(8*log(xgl)**2*(xt1**2*xt2-xt1*xt2**2-xt1+xt2)+8*log(xgl)*
     . log(xt1)*xt1*(xt2**2-2*xt2+1)+8*log(xgl)*log(xt2)*xt2*(-xt1**2
     . +2*xt1-1)+24*log(xgl)*(xt1**2*xt2-xt1*xt2**2-xt1+xt2)+2*log(
     . xt1)**2*xt1*(-xt2**2+2*xt2-1)+log(xt1)*xt1*(5*xt1*xt2**2-10*
     . xt1*xt2+5*xt1+7*xt2**2-14*xt2+7)+2*log(xt2)**2*xt2*(xt1**2-2*
     . xt1+1)+log(xt2)*xt2*(-5*xt1**2*xt2-7*xt1**2+10*xt1*xt2+14*xt1-
     . 5*xt2-7)+2*t134p(mt1,mu,mg)*xgl*(xt1*xt2**2-2*xt1*xt2+xt1+xt2
     . **2-2*xt2+1)+2*t134p(mt2,mu,mg)*xgl*(-xt1**2*xt2-xt1**2+2*xt1*
     . xt2+2*xt1-xt2-1)+28*(xt1**2*xt2-xt1*xt2**2-xt1+xt2))/(2*(xt1**
     . 3*xt2**2-2*xt1**3*xt2+xt1**3-xt1**2*xt2**3+3*xt1**2*xt2-2*xt1
     . **2+2*xt1*xt2**3-3*xt1*xt2**2+xt1-xt2**3+2*xt2**2-xt2))

      r52=(log(xb1)**2*xb1*(2*xb1*xb2**2-4*xb1*xb2+2*xb1-xb2**2+2*xb2
     . -1)+4*log(xb1)*xb1*(-xb1*xb2**2+2*xb1*xb2-xb1+xb2**2-2*xb2+1)+
     . log(xb2)**2*xb2*(-2*xb1**2*xb2+xb1**2+4*xb1*xb2-2*xb1-2*xb2+1)
     . +4*log(xb2)*xb2*(xb1**2*xb2-xb1**2-2*xb1*xb2+2*xb1+xb2-1)+2*
     . t134p(mb1,mb2,mg)*(xb1**2*xb2-xb1**2-xb1*xb2**2+xb1+xb2**2-xb2
     . )+2*t134p(mb1,mg,mg)*(-xb1**2*xb2+xb1**2+2*xb1*xb2-2*xb1-xb2+1
     . )+2*t134p(mb2,mg,mg)*(xb1*xb2**2-2*xb1*xb2+xb1-xb2**2+2*xb2-1)
     . )/(16*(xb1**3*xb2**2-2*xb1**3*xb2+xb1**3-xb1**2*xb2**3+3*xb1**
     . 2*xb2-2*xb1**2+2*xb1*xb2**3-3*xb1*xb2**2+xb1-xb2**3+2*xb2**2-
     . xb2))

      r62=(log(xb1)**2*xb1*(xb2**2-2*xb2+1)+2*log(xb1)*xb1*(-xb2**2+2
     . *xb2-1)+log(xb2)**2*xb2*(-xb1**2+2*xb1-1)+2*log(xb2)*xb2*(xb1
     . **2-2*xb1+1)+2*t134p(mb1,mg,mg)*(xb1*xb2**2-2*xb1*xb2+xb1+xb2
     . **2-2*xb2+1)+2*t134p(mb2,mg,mg)*(-xb1**2*xb2-xb1**2+2*xb1*xb2+
     . 2*xb1-xb2-1)+2*t134p(mg,mg,mg)*(xb1**2*xb2+xb1**2-xb1*xb2**2-3
     . *xb1-xb2**2+3*xb2))/(8*(xb1**3*xb2**2-2*xb1**3*xb2+xb1**3-xb1
     . **2*xb2**3+3*xb1**2*xb2-2*xb1**2+2*xb1*xb2**3-3*xb1*xb2**2+xb1
     . -xb2**3+2*xb2**2-xb2))

      ans1=8*log(xgl)**2*(xt1**2*xt2-xt1**2-xt1*xt2**2+xt1+xt2**2-xt2
     . )+4*log(xgl)*log(xt1)*xt1*(4*xt1*xt2**2-8*xt1*xt2+4*xt1-3*xt2
     . **2+6*xt2-3)+4*log(xgl)*log(xt2)*xt2*(-4*xt1**2*xt2+3*xt1**2+8
     . *xt1*xt2-6*xt1-4*xt2+3)+12*log(xgl)*(xt1**2*xt2-xt1**2-xt1*xt2
     . **2+xt1+xt2**2-xt2)+log(xt1)**2*xt1*(-8*xt1*xt2**2+16*xt1*xt2-
     . 8*xt1+5*xt2**2-10*xt2+5)+4*log(xt1)*xt1*(3*xt1*xt2**2-6*xt1*
     . xt2+3*xt1-2*xt2**2+4*xt2-2)+log(xt2)**2*xt2*(8*xt1**2*xt2-5*
     . xt1**2-16*xt1*xt2+10*xt1+8*xt2-5)+4*log(xt2)*xt2*(-3*xt1**2*
     . xt2+2*xt1**2+6*xt1*xt2-4*xt1-3*xt2+2)+4*t134p(mt1,mt1,mg)*xgl*
     . (xt1*xt2**2-2*xt1*xt2+xt1+xt2**2-2*xt2+1)+4*t134p(mt2,mt2,mg)*
     . xgl*(-xt1**2*xt2-xt1**2+2*xt1*xt2+2*xt1-xt2-1)+4*t134p(mu,mt1,
     . mg)*xgl*(-xt1*xt2**2+2*xt1*xt2-xt1-xt2**2+2*xt2-1)+4*t134p(mu,
     . mt2,mg)*xgl*(xt1**2*xt2+xt1**2-2*xt1*xt2-2*xt1+xt2+1)+8*(xt1**
     . 2*xt2-xt1**2-xt1*xt2**2+xt1+xt2**2-xt2)
      r72=ans1/(8*(xt1**3*xt2**2-2*xt1**3*xt2+xt1**3-xt1**2*xt2**3+3*
     . xt1**2*xt2-2*xt1**2+2*xt1*xt2**3-3*xt1*xt2**2+xt1-xt2**3+2*xt2
     . **2-xt2))

      r82=(log(xb1)**2*(xb1*xb2**2-2*xb1*xb2+xb1-2*xb2**2+4*xb2-2)+4*
     . log(xb1)*(-xb1**2*xb2**2+2*xb1**2*xb2-xb1**2+xb1*xb2**2-2*xb1*
     . xb2+xb1+xb2**2-2*xb2+1)+log(xb2)**2*(-xb1**2*xb2+2*xb1**2+2*
     . xb1*xb2-4*xb1-xb2+2)+4*log(xb2)*(xb1**2*xb2**2-xb1**2*xb2-xb1
     . **2-2*xb1*xb2**2+2*xb1*xb2+2*xb1+xb2**2-xb2-1)+2*t134p(mb1,mg,
     . mg)*(-xb1*xb2**2+2*xb1*xb2-xb1+xb2**2-2*xb2+1)+2*t134p(mb2,mg,
     . mg)*(xb1**2*xb2-xb1**2-2*xb1*xb2+2*xb1+xb2-1)+10*(-xb1**2*xb2+
     . xb1**2+xb1*xb2**2-xb1-xb2**2+xb2))/(8*(xb1**3*xb2**2-2*xb1**3*
     . xb2+xb1**3-xb1**2*xb2**3+3*xb1**2*xb2-2*xb1**2+2*xb1*xb2**3-3*
     . xb1*xb2**2+xb1-xb2**3+2*xb2**2-xb2))

      ans2=4*log(xt2)*xt2*(-3*xt1**3*xt2+2*xt1**3+3*xt1**2*xt2**2+4*
     . xt1**2*xt2-4*xt1**2-6*xt1*xt2**2+xt1*xt2+2*xt1+3*xt2**2-2*xt2)
     . +4*(xt1**3*xt2-xt1**3-2*xt1**2*xt2**2+xt1**2*xt2+xt1**2+xt1*
     . xt2**3+xt1*xt2**2-2*xt1*xt2-xt2**3+xt2**2)
      ans1=4*log(xgl)*log(xt1)*xt1*(2*xt1**2*xt2**2-4*xt1**2*xt2+2*
     . xt1**2-2*xt1*xt2**3+3*xt1*xt2**2-xt1+xt2**3-2*xt2**2+xt2)+4*
     . log(xgl)*log(xt2)*xt2*(-2*xt1**3*xt2+xt1**3+2*xt1**2*xt2**2+3*
     . xt1**2*xt2-2*xt1**2-4*xt1*xt2**2+xt1+2*xt2**2-xt2)+4*log(xgl)*
     . (xt1**3*xt2-xt1**3-2*xt1**2*xt2**2+xt1**2*xt2+xt1**2+xt1*xt2**
     . 3+xt1*xt2**2-2*xt1*xt2-xt2**3+xt2**2)+log(xt1)**2*xt1*(-6*xt1
     . **2*xt2**2+12*xt1**2*xt2-6*xt1**2+2*xt1*xt2**3-xt1*xt2**2-4*
     . xt1*xt2+3*xt1+xt2**3-2*xt2**2+xt2)+4*log(xt1)*log(xt2)*xt1*xt2
     . *(xt1**2*xt2-xt1**2+xt1*xt2**2-4*xt1*xt2+3*xt1-xt2**2+3*xt2-2)
     . +4*log(xt1)*xt1*(3*xt1**2*xt2**2-6*xt1**2*xt2+3*xt1**2-3*xt1*
     . xt2**3+4*xt1*xt2**2+xt1*xt2-2*xt1+2*xt2**3-4*xt2**2+2*xt2)+log
     . (xt2)**2*xt2*(2*xt1**3*xt2+xt1**3-6*xt1**2*xt2**2-xt1**2*xt2-2
     . *xt1**2+12*xt1*xt2**2-4*xt1*xt2+xt1-6*xt2**2+3*xt2)+ans2
      r92=ans1/(8*(xt1**4*xt2**2-2*xt1**4*xt2+xt1**4-2*xt1**3*xt2**3+
     . 2*xt1**3*xt2**2+2*xt1**3*xt2-2*xt1**3+xt1**2*xt2**4+2*xt1**2*
     . xt2**3-6*xt1**2*xt2**2+2*xt1**2*xt2+xt1**2-2*xt1*xt2**4+2*xt1*
     . xt2**3+2*xt1*xt2**2-2*xt1*xt2+xt2**4-2*xt2**3+xt2**2))

      mt1 = amst1
      mt2 = amst2
      xt1 = amst1**2/amg**2 * rim
      xt2 = amst2**2/amg**2 * rim

      ralsca2=(-((9*log(xb1)-10)*(xb2-1)*log(xb1)*xb1-(9*log(xb2)-10)
     . *(xb1-1)*log(xb2)*xb2))/(48*(xb1-xb2)*(xb1-1)*(xb2-1))

      ralscf2=(-((xb1-1)*log(xb2)*xb2-(xb2-1)*log(xb1)*xb1))/(8*(xb1-
     . xb2)*(xb1-1)*(xb2-1))

      rmb12=(-((((2*xb1**3-3*xb2-(xb2-6)*xb1**2-2*(xb2+1)*xb1)*xb1+(
     . xb1**2-xb2)*(xb1-1)**2*log(-(xb1-1)))*(xb2-1)+(xb1-1)**2*log(
     . xb2)*xb1**2*xb2)*log(xb1)-(((xb1**2-xb2)*(xb1+1)*(xb2-1)*log(
     . xb1)**2-(xb1-1)**2*log(xb2)**2*xb2)*xb1+((xb1-xb2)*(xb2-1)+(
     . xb1-1)*log(xb2)*xb2)*((xb1+5)*xb1+(xb1-1)**2*log(-(xb1-1)))*(
     . xb1-1))))/(4*(xb1-xb2)**2*(xb1-1)**2*(xb2-1)*xb1)

      rmb22=(-(((xb1-xb2**2)*(xb1-1)*(xb2+1)*log(xb2)**2+(xb2-1)**2*
     . log(xb1)**2*xb1)*xb2+((xb2+5)*xb2+(xb2-1)**2*log(-(xb2-1)))*(
     . xb1-xb2)*(xb1-1)*(xb2-1)-((xb2-1)**2*log(-(xb2-1))-log(xb2)*
     . xb2**2+(xb2+5)*xb2)*(xb2-1)**2*log(xb1)*xb1+((2*(xb2**2+3*xb2-
     . 1)*xb2-(xb2**2+2*xb2+3)*xb1)*xb2-(xb1-xb2**2)*(xb2-1)**2*log(-
     . (xb2-1)))*(xb1-1)*log(xb2)))/(4*(xb1-xb2)**2*(xb1-1)*(xb2-1)**
     . 2*xb2)

      rmgca2=(-((3*log(xb1)-14)*(xb1+1)*(xb2-1)**2*log(xb1)*xb1-(3*
     . log(xb2)-14)*(xb1-1)**2*(xb2+1)*log(xb2)*xb2-28*(xb1-xb2)*(xb1
     . -1)*(xb2-1)))/(16*(xb1-xb2)*(xb1-1)**2*(xb2-1)**2)

      ms1 = amsu1
      ms2 = amsu2
      xs1 = amsu1**2/amg**2 * rim
      xs2 = amsu2**2/amg**2 * rim

      ans2=2*t134p(mb2,ms1,mg)*(xb1**2*xb2-xb1**2*xs1-2*xb1*xb2+2*xb1
     . *xs1+xb2-xs1)+2*t134p(mb2,ms2,mg)*(xb1**2*xb2-xb1**2*xs2-2*xb1
     . *xb2+2*xb1*xs2+xb2-xs2)+2*t134p(mg,ms1,mg)*(-2*xb1**2*xb2+xb1
     . **2*xs1+xb1**2+2*xb1*xb2**2-2*xb1*xs1-xb2**2*xs1-xb2**2+2*xb2*
     . xs1)+2*t134p(mg,ms2,mg)*(-2*xb1**2*xb2+xb1**2*xs2+xb1**2+2*xb1
     . *xb2**2-2*xb1*xs2-xb2**2*xs2-xb2**2+2*xb2*xs2)+12*(xb1**2*xb2*
     . xs1+xb1**2*xb2*xs2+xb1**2*xb2-xb1**2*xs1-xb1**2*xs2-xb1**2-xb1
     . *xb2**2*xs1-xb1*xb2**2*xs2-xb1*xb2**2+xb1*xs1+xb1*xs2+xb1+xb2
     . **2*xs1+xb2**2*xs2+xb2**2-xb2*xs1-xb2*xs2-xb2)
      ans1=log(xb1)**2*xb1*(-xb2**2*xs1-xb2**2*xs2+2*xb2*xs1+2*xb2*
     . xs2-xs1-xs2)+2*log(xb1)*log(xs1)*xb1*xs1*(-xb2**2+2*xb2-1)+2*
     . log(xb1)*log(xs2)*xb1*xs2*(-xb2**2+2*xb2-1)+4*log(xb1)*xb1*(
     . xb2**2*xs1+xb2**2*xs2-2*xb2*xs1-2*xb2*xs2+xs1+xs2)+log(xb2)**2
     . *xb2*(xb1**2*xs1+xb1**2*xs2-2*xb1*xs1-2*xb1*xs2+xs1+xs2)+2*log
     . (xb2)*log(xs1)*xb2*xs1*(xb1**2-2*xb1+1)+2*log(xb2)*log(xs2)*
     . xb2*xs2*(xb1**2-2*xb1+1)+4*log(xb2)*xb2*(-xb1**2*xs1-xb1**2*
     . xs2+2*xb1*xs1+2*xb1*xs2-xs1-xs2)+log(xs1)**2*xs1*(xb1**2*xb2-
     . xb1**2-xb1*xb2**2+xb1+xb2**2-xb2)+8*log(xs1)*xs1*(-xb1**2*xb2+
     . xb1**2+xb1*xb2**2-xb1-xb2**2+xb2)+log(xs2)**2*xs2*(xb1**2*xb2-
     . xb1**2-xb1*xb2**2+xb1+xb2**2-xb2)+8*log(xs2)*xs2*(-xb1**2*xb2+
     . xb1**2+xb1*xb2**2-xb1-xb2**2+xb2)+2*t134p(mb1,ms1,mg)*(-xb1*
     . xb2**2+2*xb1*xb2-xb1+xb2**2*xs1-2*xb2*xs1+xs1)+2*t134p(mb1,ms2
     . ,mg)*(-xb1*xb2**2+2*xb1*xb2-xb1+xb2**2*xs2-2*xb2*xs2+xs2)+ans2
      ru102=ans1/(8*(xb1**3*xb2**2-2*xb1**3*xb2+xb1**3-xb1**2*xb2**3+3
     . *xb1**2*xb2-2*xb1**2+2*xb1*xb2**3-3*xb1*xb2**2+xb1-xb2**3+2*
     . xb2**2-xb2))

      rualstr2=(-((log(xs2)-6+log(xs1))*((xb1-1)*log(xb2)*xb2-(xb2-1)*
     . log(xb1)*xb1)+3*((xb1-1)*log(xb2)**2*xb2-(xb2-1)*log(xb1)**2*
     . xb1)))/(24*(xb1-xb2)*(xb1-1)*(xb2-1))

      rumgtr2=((xs2-6+xs1+log(xs2)+log(xs1)+(log(xs2-1)-log(xs2))*(xs2
     . -1)**2+(log(xs1-1)-log(xs1))*(xs1-1)**2)*((xb1+1)*(xb2-1)**2*
     . log(xb1)*xb1-(xb1-1)**2*(xb2+1)*log(xb2)*xb2)+2*((log(xs2-1)-
     . log(xs2))*(xs2-1)**2+log(xs1)+log(xs2)+(log(xs1-1)-log(xs1))*(
     . xs1-1)**2)*(xb1-xb2)*(xb1-1)*(xb2-1)+(xb1+1)*(xb2-1)**2*log(
     . xb1)**2*xb1-(xb1-1)**2*(xb2+1)*log(xb2)**2*xb2+2*(xs2-6+xs1)*(
     . xb1-xb2)*(xb1-1)*(xb2-1))/(8*(xb1-xb2)*(xb1-1)**2*(xb2-1)**2)

      ms1 = amsd1
      ms2 = amsd2
      xs1 = amsd1**2/amg**2 * rim
      xs2 = amsd2**2/amg**2 * rim

      ans2=2*t134p(mb2,ms1,mg)*(xb1**2*xb2-xb1**2*xs1-2*xb1*xb2+2*xb1
     . *xs1+xb2-xs1)+2*t134p(mb2,ms2,mg)*(xb1**2*xb2-xb1**2*xs2-2*xb1
     . *xb2+2*xb1*xs2+xb2-xs2)+2*t134p(mg,ms1,mg)*(-2*xb1**2*xb2+xb1
     . **2*xs1+xb1**2+2*xb1*xb2**2-2*xb1*xs1-xb2**2*xs1-xb2**2+2*xb2*
     . xs1)+2*t134p(mg,ms2,mg)*(-2*xb1**2*xb2+xb1**2*xs2+xb1**2+2*xb1
     . *xb2**2-2*xb1*xs2-xb2**2*xs2-xb2**2+2*xb2*xs2)+12*(xb1**2*xb2*
     . xs1+xb1**2*xb2*xs2+xb1**2*xb2-xb1**2*xs1-xb1**2*xs2-xb1**2-xb1
     . *xb2**2*xs1-xb1*xb2**2*xs2-xb1*xb2**2+xb1*xs1+xb1*xs2+xb1+xb2
     . **2*xs1+xb2**2*xs2+xb2**2-xb2*xs1-xb2*xs2-xb2)
      ans1=log(xb1)**2*xb1*(-xb2**2*xs1-xb2**2*xs2+2*xb2*xs1+2*xb2*
     . xs2-xs1-xs2)+2*log(xb1)*log(xs1)*xb1*xs1*(-xb2**2+2*xb2-1)+2*
     . log(xb1)*log(xs2)*xb1*xs2*(-xb2**2+2*xb2-1)+4*log(xb1)*xb1*(
     . xb2**2*xs1+xb2**2*xs2-2*xb2*xs1-2*xb2*xs2+xs1+xs2)+log(xb2)**2
     . *xb2*(xb1**2*xs1+xb1**2*xs2-2*xb1*xs1-2*xb1*xs2+xs1+xs2)+2*log
     . (xb2)*log(xs1)*xb2*xs1*(xb1**2-2*xb1+1)+2*log(xb2)*log(xs2)*
     . xb2*xs2*(xb1**2-2*xb1+1)+4*log(xb2)*xb2*(-xb1**2*xs1-xb1**2*
     . xs2+2*xb1*xs1+2*xb1*xs2-xs1-xs2)+log(xs1)**2*xs1*(xb1**2*xb2-
     . xb1**2-xb1*xb2**2+xb1+xb2**2-xb2)+8*log(xs1)*xs1*(-xb1**2*xb2+
     . xb1**2+xb1*xb2**2-xb1-xb2**2+xb2)+log(xs2)**2*xs2*(xb1**2*xb2-
     . xb1**2-xb1*xb2**2+xb1+xb2**2-xb2)+8*log(xs2)*xs2*(-xb1**2*xb2+
     . xb1**2+xb1*xb2**2-xb1-xb2**2+xb2)+2*t134p(mb1,ms1,mg)*(-xb1*
     . xb2**2+2*xb1*xb2-xb1+xb2**2*xs1-2*xb2*xs1+xs1)+2*t134p(mb1,ms2
     . ,mg)*(-xb1*xb2**2+2*xb1*xb2-xb1+xb2**2*xs2-2*xb2*xs2+xs2)+ans2
      rd102=ans1/(8*(xb1**3*xb2**2-2*xb1**3*xb2+xb1**3-xb1**2*xb2**3+3
     . *xb1**2*xb2-2*xb1**2+2*xb1*xb2**3-3*xb1*xb2**2+xb1-xb2**3+2*
     . xb2**2-xb2))

      rdalstr2=(-((log(xs2)-6+log(xs1))*((xb1-1)*log(xb2)*xb2-(xb2-1)*
     . log(xb1)*xb1)+3*((xb1-1)*log(xb2)**2*xb2-(xb2-1)*log(xb1)**2*
     . xb1)))/(24*(xb1-xb2)*(xb1-1)*(xb2-1))

      rdmgtr2=((xs2-6+xs1+log(xs2)+log(xs1)+(log(xs2-1)-log(xs2))*(xs2
     . -1)**2+(log(xs1-1)-log(xs1))*(xs1-1)**2)*((xb1+1)*(xb2-1)**2*
     . log(xb1)*xb1-(xb1-1)**2*(xb2+1)*log(xb2)*xb2)+2*((log(xs2-1)-
     . log(xs2))*(xs2-1)**2+log(xs1)+log(xs2)+(log(xs1-1)-log(xs1))*(
     . xs1-1)**2)*(xb1-xb2)*(xb1-1)*(xb2-1)+(xb1+1)*(xb2-1)**2*log(
     . xb1)**2*xb1-(xb1-1)**2*(xb2+1)*log(xb2)**2*xb2+2*(xs2-6+xs1)*(
     . xb1-xb2)*(xb1-1)*(xb2-1))/(8*(xb1-xb2)*(xb1-1)**2*(xb2-1)**2)

      ms1 = amsb1
      ms2 = amsb2
      xs1 = amsb1**2/amg**2 * rim
      xs2 = amsb2**2/amg**2 * rim

      ans1=log(xb1)**2*xb1*(xb1**2*xb2-xb1**2-4*xb1*xb2**2+6*xb1*xb2-
     . 2*xb1-xb2**3+3*xb2**2-2*xb2)+2*log(xb1)*log(xb2)*xb1*xb2*(xb1
     . **2-2*xb1-xb2**2+2*xb2)+4*log(xb1)*xb1*(-2*xb1**2*xb2+2*xb1**2
     . +3*xb1*xb2**2-2*xb1*xb2-xb1+xb2**3-4*xb2**2+3*xb2)+log(xb2)**2
     . *xb2*(xb1**3+4*xb1**2*xb2-3*xb1**2-xb1*xb2**2-6*xb1*xb2+2*xb1+
     . xb2**2+2*xb2)+4*log(xb2)*xb2*(-xb1**3-3*xb1**2*xb2+4*xb1**2+2*
     . xb1*xb2**2+2*xb1*xb2-3*xb1-2*xb2**2+xb2)+2*t134p(mb1,mb2,mg)*(
     . -xb1**3+xb1**2*xb2+2*xb1**2-xb1*xb2**2-2*xb1+xb2**3-2*xb2**2+2
     . *xb2)+2*t134p(mb1,mg,mg)*(xb1**3-2*xb1**2*xb2-xb1**2+xb1*xb2**
     . 2+2*xb1*xb2-xb2**2)+2*t134p(mb2,mg,mg)*(-xb1**2*xb2+xb1**2+2*
     . xb1*xb2**2-2*xb1*xb2-xb2**3+xb2**2)+12*(xb1**3*xb2-xb1**3-xb1*
     . xb2**3+xb1+xb2**3-xb2)
      rb102=ans1/(8*(xb1**3*xb2**2-2*xb1**3*xb2+xb1**3-xb1**2*xb2**3+3
     . *xb1**2*xb2-2*xb1**2+2*xb1*xb2**3-3*xb1*xb2**2+xb1-xb2**3+2*
     . xb2**2-xb2))

      rbalstr2=(-((log(xs2)-6+log(xs1))*((xb1-1)*log(xb2)*xb2-(xb2-1)*
     . log(xb1)*xb1)+3*((xb1-1)*log(xb2)**2*xb2-(xb2-1)*log(xb1)**2*
     . xb1)))/(24*(xb1-xb2)*(xb1-1)*(xb2-1))

      rbmgtr2=((xs2-6+xs1+log(xs2)+log(xs1)+(log(xs2-1)-log(xs2))*(xs2
     . -1)**2+(log(xs1-1)-log(xs1))*(xs1-1)**2)*((xb1+1)*(xb2-1)**2*
     . log(xb1)*xb1-(xb1-1)**2*(xb2+1)*log(xb2)*xb2)+2*((log(xs2-1)-
     . log(xs2))*(xs2-1)**2+log(xs1)+log(xs2)+(log(xs1-1)-log(xs1))*(
     . xs1-1)**2)*(xb1-xb2)*(xb1-1)*(xb2-1)+(xb1+1)*(xb2-1)**2*log(
     . xb1)**2*xb1-(xb1-1)**2*(xb2+1)*log(xb2)**2*xb2+2*(xs2-6+xs1)*(
     . xb1-xb2)*(xb1-1)*(xb2-1))/(8*(xb1-xb2)*(xb1-1)**2*(xb2-1)**2)

      ms1 = amst1
      ms2 = amst2
      xs1 = amst1**2/amg**2 * rim
      xs2 = amst2**2/amg**2 * rim

      ans14=-8*log(xs2)*xb1*xb2**2*xs2-4*log(xs2)*xb1*xq**2*xs2-4*log
     . (xs2)*xb1*xq*xs2**2-4*log(xs2)*xb1*xq*xs2+8*log(xs2)*xb1*xs2**
     . 3-16*log(xs2)*xb1*xs2**2+8*log(xs2)*xb1*xs2-4*log(xs2)*xb2**2*
     . xq**2*xs2-4*log(xs2)*xb2**2*xq*xs2**2-4*log(xs2)*xb2**2*xq*xs2
     . +8*log(xs2)*xb2**2*xs2**3-16*log(xs2)*xb2**2*xs2**2+8*log(xs2)
     . *xb2**2*xs2+4*log(xs2)*xb2*xq**2*xs2+4*log(xs2)*xb2*xq*xs2**2+
     . 4*log(xs2)*xb2*xq*xs2-8*log(xs2)*xb2*xs2**3+16*log(xs2)*xb2*
     . xs2**2-8*log(xs2)*xb2*xs2
      ans13=-log(xs2)**2*xb1*xb2**2*xq**2*xs2+log(xs2)**2*xb1*xb2**2*
     . xs2**3-2*log(xs2)**2*xb1*xb2**2*xs2**2+log(xs2)**2*xb1*xb2**2*
     . xs2+log(xs2)**2*xb1*xq**2*xs2-log(xs2)**2*xb1*xs2**3+2*log(xs2
     . )**2*xb1*xs2**2-log(xs2)**2*xb1*xs2+log(xs2)**2*xb2**2*xq**2*
     . xs2-log(xs2)**2*xb2**2*xs2**3+2*log(xs2)**2*xb2**2*xs2**2-log(
     . xs2)**2*xb2**2*xs2-log(xs2)**2*xb2*xq**2*xs2+log(xs2)**2*xb2*
     . xs2**3-2*log(xs2)**2*xb2*xs2**2+log(xs2)**2*xb2*xs2-4*log(xs2)
     . *xb1**2*xb2*xq**2*xs2-4*log(xs2)*xb1**2*xb2*xq*xs2**2-4*log(
     . xs2)*xb1**2*xb2*xq*xs2+8*log(xs2)*xb1**2*xb2*xs2**3-16*log(xs2
     . )*xb1**2*xb2*xs2**2+8*log(xs2)*xb1**2*xb2*xs2+4*log(xs2)*xb1**
     . 2*xq**2*xs2+4*log(xs2)*xb1**2*xq*xs2**2+4*log(xs2)*xb1**2*xq*
     . xs2-8*log(xs2)*xb1**2*xs2**3+16*log(xs2)*xb1**2*xs2**2-8*log(
     . xs2)*xb1**2*xs2+4*log(xs2)*xb1*xb2**2*xq**2*xs2+4*log(xs2)*xb1
     . *xb2**2*xq*xs2**2+4*log(xs2)*xb1*xb2**2*xq*xs2-8*log(xs2)*xb1*
     . xb2**2*xs2**3+16*log(xs2)*xb1*xb2**2*xs2**2+ans14
      ans12=-4*log(xb2)**2*xb1*xb2*xq*xs1-8*log(xb2)**2*xb1*xb2*xq*
     . xs2**2+4*log(xb2)**2*xb1*xb2*xq*xs2-4*log(xb2)**2*xb1*xb2*xq+2
     . *log(xb2)**2*xb1*xb2*xs1*xs2**2-4*log(xb2)**2*xb1*xb2*xs1*xs2+
     . 2*log(xb2)**2*xb1*xb2*xs1+2*log(xb2)**2*xb1*xb2*xs2**3-4*log(
     . xb2)**2*xb1*xb2*xs2**2+2*log(xb2)**2*xb1*xb2*xs2+2*log(xb2)**2
     . *xb2*xq**3-log(xb2)**2*xb2*xq**2*xs1-5*log(xb2)**2*xb2*xq**2*
     . xs2-4*log(xb2)**2*xb2*xq**2+2*log(xb2)**2*xb2*xq*xs1*xs2+2*log
     . (xb2)**2*xb2*xq*xs1+4*log(xb2)**2*xb2*xq*xs2**2-2*log(xb2)**2*
     . xb2*xq*xs2+2*log(xb2)**2*xb2*xq-log(xb2)**2*xb2*xs1*xs2**2+2*
     . log(xb2)**2*xb2*xs1*xs2-log(xb2)**2*xb2*xs1-log(xb2)**2*xb2*
     . xs2**3+2*log(xb2)**2*xb2*xs2**2-log(xb2)**2*xb2*xs2+log(xs2)**
     . 2*xb1**2*xb2*xq**2*xs2-log(xs2)**2*xb1**2*xb2*xs2**3+2*log(xs2
     . )**2*xb1**2*xb2*xs2**2-log(xs2)**2*xb1**2*xb2*xs2-log(xs2)**2*
     . xb1**2*xq**2*xs2+log(xs2)**2*xb1**2*xs2**3-2*log(xs2)**2*xb1**
     . 2*xs2**2+log(xs2)**2*xb1**2*xs2+ans13
      ans11=4*log(xb1)**2*xb1*xq**2-2*log(xb1)**2*xb1*xq*xs1*xs2-2*
     . log(xb1)**2*xb1*xq*xs1-4*log(xb1)**2*xb1*xq*xs2**2+2*log(xb1)
     . **2*xb1*xq*xs2-2*log(xb1)**2*xb1*xq+log(xb1)**2*xb1*xs1*xs2**2
     . -2*log(xb1)**2*xb1*xs1*xs2+log(xb1)**2*xb1*xs1+log(xb1)**2*xb1
     . *xs2**3-2*log(xb1)**2*xb1*xs2**2+log(xb1)**2*xb1*xs2+2*log(xb2
     . )**2*xb1**2*xb2*xq**3-log(xb2)**2*xb1**2*xb2*xq**2*xs1-5*log(
     . xb2)**2*xb1**2*xb2*xq**2*xs2-4*log(xb2)**2*xb1**2*xb2*xq**2+2*
     . log(xb2)**2*xb1**2*xb2*xq*xs1*xs2+2*log(xb2)**2*xb1**2*xb2*xq*
     . xs1+4*log(xb2)**2*xb1**2*xb2*xq*xs2**2-2*log(xb2)**2*xb1**2*
     . xb2*xq*xs2+2*log(xb2)**2*xb1**2*xb2*xq-log(xb2)**2*xb1**2*xb2*
     . xs1*xs2**2+2*log(xb2)**2*xb1**2*xb2*xs1*xs2-log(xb2)**2*xb1**2
     . *xb2*xs1-log(xb2)**2*xb1**2*xb2*xs2**3+2*log(xb2)**2*xb1**2*
     . xb2*xs2**2-log(xb2)**2*xb1**2*xb2*xs2-4*log(xb2)**2*xb1*xb2*xq
     . **3+2*log(xb2)**2*xb1*xb2*xq**2*xs1+10*log(xb2)**2*xb1*xb2*xq
     . **2*xs2+8*log(xb2)**2*xb1*xb2*xq**2-4*log(xb2)**2*xb1*xb2*xq*
     . xs1*xs2+ans12
      ans10=5*log(xb1)**2*xb1*xb2**2*xq**2*xs2+4*log(xb1)**2*xb1*xb2
     . **2*xq**2-2*log(xb1)**2*xb1*xb2**2*xq*xs1*xs2-2*log(xb1)**2*
     . xb1*xb2**2*xq*xs1-4*log(xb1)**2*xb1*xb2**2*xq*xs2**2+2*log(xb1
     . )**2*xb1*xb2**2*xq*xs2-2*log(xb1)**2*xb1*xb2**2*xq+log(xb1)**2
     . *xb1*xb2**2*xs1*xs2**2-2*log(xb1)**2*xb1*xb2**2*xs1*xs2+log(
     . xb1)**2*xb1*xb2**2*xs1+log(xb1)**2*xb1*xb2**2*xs2**3-2*log(xb1
     . )**2*xb1*xb2**2*xs2**2+log(xb1)**2*xb1*xb2**2*xs2+4*log(xb1)**
     . 2*xb1*xb2*xq**3-2*log(xb1)**2*xb1*xb2*xq**2*xs1-10*log(xb1)**2
     . *xb1*xb2*xq**2*xs2-8*log(xb1)**2*xb1*xb2*xq**2+4*log(xb1)**2*
     . xb1*xb2*xq*xs1*xs2+4*log(xb1)**2*xb1*xb2*xq*xs1+8*log(xb1)**2*
     . xb1*xb2*xq*xs2**2-4*log(xb1)**2*xb1*xb2*xq*xs2+4*log(xb1)**2*
     . xb1*xb2*xq-2*log(xb1)**2*xb1*xb2*xs1*xs2**2+4*log(xb1)**2*xb1*
     . xb2*xs1*xs2-2*log(xb1)**2*xb1*xb2*xs1-2*log(xb1)**2*xb1*xb2*
     . xs2**3+4*log(xb1)**2*xb1*xb2*xs2**2-2*log(xb1)**2*xb1*xb2*xs2-
     . 2*log(xb1)**2*xb1*xq**3+log(xb1)**2*xb1*xq**2*xs1+5*log(xb1)**
     . 2*xb1*xq**2*xs2+ans11
      ans9=-4*t134(mb1,mq,ms2,mg)*xb1*xb2+2*t134(mb1,mq,ms2,mg)*xb1*
     . xq**2-4*t134(mb1,mq,ms2,mg)*xb1*xq*xs2-4*t134(mb1,mq,ms2,mg)*
     . xb1*xq+2*t134(mb1,mq,ms2,mg)*xb1*xs2**2-4*t134(mb1,mq,ms2,mg)*
     . xb1*xs2+2*t134(mb1,mq,ms2,mg)*xb1+2*t134(mb1,mq,ms2,mg)*xb2**2
     . *xq**3-6*t134(mb1,mq,ms2,mg)*xb2**2*xq**2*xs2-4*t134(mb1,mq,
     . ms2,mg)*xb2**2*xq**2+6*t134(mb1,mq,ms2,mg)*xb2**2*xq*xs2**2+2*
     . t134(mb1,mq,ms2,mg)*xb2**2*xq-2*t134(mb1,mq,ms2,mg)*xb2**2*xs2
     . **3+4*t134(mb1,mq,ms2,mg)*xb2**2*xs2**2-2*t134(mb1,mq,ms2,mg)*
     . xb2**2*xs2-4*t134(mb1,mq,ms2,mg)*xb2*xq**3+12*t134(mb1,mq,ms2,
     . mg)*xb2*xq**2*xs2+8*t134(mb1,mq,ms2,mg)*xb2*xq**2-12*t134(mb1,
     . mq,ms2,mg)*xb2*xq*xs2**2-4*t134(mb1,mq,ms2,mg)*xb2*xq+4*t134(
     . mb1,mq,ms2,mg)*xb2*xs2**3-8*t134(mb1,mq,ms2,mg)*xb2*xs2**2+4*
     . t134(mb1,mq,ms2,mg)*xb2*xs2+2*t134(mb1,mq,ms2,mg)*xq**3-6*t134
     . (mb1,mq,ms2,mg)*xq**2*xs2-4*t134(mb1,mq,ms2,mg)*xq**2+6*t134(
     . mb1,mq,ms2,mg)*xq*xs2**2+2*t134(mb1,mq,ms2,mg)*xq-2*t134(mb1,
     . mq,ms2,mg)*xs2**3+4*t134(mb1,mq,ms2,mg)*xs2**2-2*t134(mb1,mq,
     . ms2,mg)*xs2-2*log(xb1)**2*xb1*xb2**2*xq**3+log(xb1)**2*xb1*xb2
     . **2*xq**2*xs1+ans10
      ans8=4*t134(mb1,mq,ms1,mg)*xb2*xq**2*xs1+8*t134(mb1,mq,ms1,mg)*
     . xb2*xq**2*xs2+8*t134(mb1,mq,ms1,mg)*xb2*xq**2-8*t134(mb1,mq,
     . ms1,mg)*xb2*xq*xs1*xs2-8*t134(mb1,mq,ms1,mg)*xb2*xq*xs1-4*t134
     . (mb1,mq,ms1,mg)*xb2*xq*xs2**2+8*t134(mb1,mq,ms1,mg)*xb2*xq*xs2
     . -4*t134(mb1,mq,ms1,mg)*xb2*xq+4*t134(mb1,mq,ms1,mg)*xb2*xs1*
     . xs2**2-8*t134(mb1,mq,ms1,mg)*xb2*xs1*xs2+4*t134(mb1,mq,ms1,mg)
     . *xb2*xs1+2*t134(mb1,mq,ms1,mg)*xq**3-2*t134(mb1,mq,ms1,mg)*xq
     . **2*xs1-4*t134(mb1,mq,ms1,mg)*xq**2*xs2-4*t134(mb1,mq,ms1,mg)*
     . xq**2+4*t134(mb1,mq,ms1,mg)*xq*xs1*xs2+4*t134(mb1,mq,ms1,mg)*
     . xq*xs1+2*t134(mb1,mq,ms1,mg)*xq*xs2**2-4*t134(mb1,mq,ms1,mg)*
     . xq*xs2+2*t134(mb1,mq,ms1,mg)*xq-2*t134(mb1,mq,ms1,mg)*xs1*xs2
     . **2+4*t134(mb1,mq,ms1,mg)*xs1*xs2-2*t134(mb1,mq,ms1,mg)*xs1+2*
     . t134(mb1,mq,ms2,mg)*xb1*xb2**2*xq**2-4*t134(mb1,mq,ms2,mg)*xb1
     . *xb2**2*xq*xs2-4*t134(mb1,mq,ms2,mg)*xb1*xb2**2*xq+2*t134(mb1,
     . mq,ms2,mg)*xb1*xb2**2*xs2**2-4*t134(mb1,mq,ms2,mg)*xb1*xb2**2*
     . xs2+2*t134(mb1,mq,ms2,mg)*xb1*xb2**2-4*t134(mb1,mq,ms2,mg)*xb1
     . *xb2*xq**2+8*t134(mb1,mq,ms2,mg)*xb1*xb2*xq*xs2+8*t134(mb1,mq,
     . ms2,mg)*xb1*xb2*xq-4*t134(mb1,mq,ms2,mg)*xb1*xb2*xs2**2+8*t134
     . (mb1,mq,ms2,mg)*xb1*xb2*xs2+ans9
      ans7=2*t134(mb1,mq,ms1,mg)*xb1*xb2**2*xq**2-4*t134(mb1,mq,ms1,
     . mg)*xb1*xb2**2*xq*xs2-4*t134(mb1,mq,ms1,mg)*xb1*xb2**2*xq+2*
     . t134(mb1,mq,ms1,mg)*xb1*xb2**2*xs2**2-4*t134(mb1,mq,ms1,mg)*
     . xb1*xb2**2*xs2+2*t134(mb1,mq,ms1,mg)*xb1*xb2**2-4*t134(mb1,mq,
     . ms1,mg)*xb1*xb2*xq**2+8*t134(mb1,mq,ms1,mg)*xb1*xb2*xq*xs2+8*
     . t134(mb1,mq,ms1,mg)*xb1*xb2*xq-4*t134(mb1,mq,ms1,mg)*xb1*xb2*
     . xs2**2+8*t134(mb1,mq,ms1,mg)*xb1*xb2*xs2-4*t134(mb1,mq,ms1,mg)
     . *xb1*xb2+2*t134(mb1,mq,ms1,mg)*xb1*xq**2-4*t134(mb1,mq,ms1,mg)
     . *xb1*xq*xs2-4*t134(mb1,mq,ms1,mg)*xb1*xq+2*t134(mb1,mq,ms1,mg)
     . *xb1*xs2**2-4*t134(mb1,mq,ms1,mg)*xb1*xs2+2*t134(mb1,mq,ms1,mg
     . )*xb1+2*t134(mb1,mq,ms1,mg)*xb2**2*xq**3-2*t134(mb1,mq,ms1,mg)
     . *xb2**2*xq**2*xs1-4*t134(mb1,mq,ms1,mg)*xb2**2*xq**2*xs2-4*
     . t134(mb1,mq,ms1,mg)*xb2**2*xq**2+4*t134(mb1,mq,ms1,mg)*xb2**2*
     . xq*xs1*xs2+4*t134(mb1,mq,ms1,mg)*xb2**2*xq*xs1+2*t134(mb1,mq,
     . ms1,mg)*xb2**2*xq*xs2**2-4*t134(mb1,mq,ms1,mg)*xb2**2*xq*xs2+2
     . *t134(mb1,mq,ms1,mg)*xb2**2*xq-2*t134(mb1,mq,ms1,mg)*xb2**2*
     . xs1*xs2**2+4*t134(mb1,mq,ms1,mg)*xb2**2*xs1*xs2-2*t134(mb1,mq,
     . ms1,mg)*xb2**2*xs1-4*t134(mb1,mq,ms1,mg)*xb2*xq**3+ans8
      ans15=(xq**2-2*xq*xs1-2*xq+xs1**2-2*xs1+1)
      ans6=ans7*ans15
      ans5=-ans6
      ans4=(4*(2*(xs1-1)**2-xq**2-(xs1+1)*xq)-(xs1-1+xq)*(xs1-1-xq)*
     . log(xs1))*(2*(xs2+1)*xq-(xs2-1)**2-xq**2)*(xb1-xb2)*(xb1-1)*(
     . xb2-1)*log(xs1)*xs1+ans5
      ans3=-ans4
      ans17=4*(3*((xs1+1)*xs1+xs2**2+xs2-1)+2*(3*xq**2+2)*xq+((2*xq**
     . 2-5*xq+8)*xq-3*(xs1+xs2)**2)*(xs1+xs2)-(4*xq-3*xs1*xs2)*(2*xs1
     . **2+3*xs1*xs2+2*xs2**2)+2*(3*xs1**2-8*xs1*xs2+3*xs2**2)*(xs1+
     . xs2)*xq-(13*xq**2+4*xs1*xs2)*xq**2+((3*xs1**2-2*xs1*xs2+3*xs2
     . **2)*xq-2*(xs1+xs2)*xq**2-3*(xs1+xs2)*xs1*xs2+6*xq**3)*(xq-xs1
     . )*(xq-xs2))*(xb1-xb2)*(xb1-1)*(xb2-1)+4*(2*((xs2-1)**2+xs1**2+
     . 2*(2*xs2-1)*xs1+(5*xq**2-1)*xq)-(xs1**2-4*xs1*xs2+xs2**2)*xq-2
     . *(xq**2+2*xs1*xs2)*(xs1+xs2)+(xs1-1-xq)*(2*(xs2+1)*xq-(xs2-1)
     . **2-xq**2)*log(xs1)*xs1+(2*(xs1+1)*xq-(xs1-1)**2-xq**2)*(xs2-1
     . -xq)*log(xs2)*xs2+(xs1+xs2-6*xq)*xq+((xs1+xs2)*xq-4*xq**2+2*
     . xs1*xs2)*(xq-xs1)*(xq-xs2))*(xb1-xb2)*(xb1-1)*(xb2-1)*log(xq)*
     . xq
      ans16=-2*(2*((xs1-1)**2*xs1-xq**3-(3*xs1-1)*xq*xs1+(3*xs1+1)*xq
     . **2)+(xq**2-2*xq*xs1+xs1**2-2*xs1+1)*(xq-xs1-1)*xb2-(2*((xs1+1
     . )*xq-(xs1-1)**2)*xb2-(xq**2-2*xq*xs1+xs1**2-2*xs1+1)*(xq-xs1-1
     . ))*xb1)*(2*(xs2+1)*xq-(xs2-1)**2-xq**2)*(xb1-xb2)*t134(mg,mq,
     . ms1,mg)-2*((xs2-1)**2+xs1**2+2*(2*xs2-1)*xs1+2*(xq+1)*(xq-1)*
     . xq+(xq-2*xs1*xs2)*(xs1+xs2)-(xs1**2-4*xs1*xs2+xs2**2)*xq-(xq**
     . 2-xs1*xs2)*(xq-xs1)*(xq-xs2))*(xb1-xb2)*(xb1-1)*(xb2-1)*log(xq
     . )**2*xq-2*((2*((xs2-1)**2*xs2-xq**3-(3*xs2-1)*xq*xs2+(3*xs2+1)
     . *xq**2)+(xq**2-2*xq*xs2+xs2**2-2*xs2+1)*(xq-xs2-1)*xb2-(2*((
     . xs2+1)*xq-(xs2-1)**2)*xb2-(xq**2-2*xq*xs2+xs2**2-2*xs2+1)*(xq-
     . xs2-1))*xb1)*(xb1-xb2)*t134(mg,mq,ms2,mg)+((xq-xs1+xb2)*t134(
     . mb2,mq,ms1,mg)+(xq-xs2+xb2)*t134(mb2,mq,ms2,mg))*(2*(xs2+1)*xq
     . -(xs2-1)**2-xq**2)*(xb1-1)**2)*(2*(xs1+1)*xq-(xs1-1)**2-xq**2)
     . +ans17
      ans2=2*(2*(xs1+xs2-2*xq)-log(xs2)*xs2-log(xs1)*xs1+2*log(xq)*xq
     . )*((xb1-1)**2*log(xb2)*xb2-(xb2-1)**2*log(xb1)*xb1)*(2*(xs1+1)
     . *xq-(xs1-1)**2-xq**2)*(2*(xs2+1)*xq-(xs2-1)**2-xq**2)+ans3+
     . ans16
      ans1=-ans2
      rt102=ans1/(8*((xs1-1)**2+xq**2-2*(xs1+1)*xq)*((xs2-1)**2+xq**2-
     . 2*(xs2+1)*xq)*(xb1-xb2)*(xb1-1)**2*(xb2-1)**2)

      rtalstr2=(-((log(xs2)-6+log(xs1))*((xb1-1)*log(xb2)*xb2-(xb2-1)*
     . log(xb1)*xb1)+3*((xb1-1)*log(xb2)**2*xb2-(xb2-1)*log(xb1)**2*
     . xb1)))/(24*(xb1-xb2)*(xb1-1)*(xb2-1))

      rtmgtr2=((xs2-6+xs1-2*xq+(xq+1)*log(xs2)+(xq+1)*log(xs1)-2*log(
     . xq)*xq-(xs2-1-xq)*log(xp(ms2))*xp(ms2)-(xs1-1-xq)*log(xp(ms1))
     . *xp(ms1)-(xs2-1-xq)*log(xm(ms2))*xm(ms2)-(xs1-1-xq)*log(xm(ms1
     . ))*xm(ms1)+(xs2-1-xq)*log(xp(ms2)-1)*xp(ms2)+(xs1-1-xq)*log(xp
     . (ms1)-1)*xp(ms1)+(xs2-1-xq)*log(xm(ms2)-1)*xm(ms2)+(xs1-1-xq)*
     . log(xm(ms1)-1)*xm(ms1))*((xb1+1)*(xb2-1)**2*log(xb1)*xb1-(xb1-
     . 1)**2*(xb2+1)*log(xb2)*xb2)+2*((log(xs1)+log(xs2))*(xq+1)-2*
     . log(xq)*xq)*(xb1-xb2)*(xb1-1)*(xb2-1)+(xb1+1)*(xb2-1)**2*log(
     . xb1)**2*xb1-(xb1-1)**2*(xb2+1)*log(xb2)**2*xb2+2*((log(xm(ms2)
     . -1)-log(xm(ms2)))*xm(ms2)+(log(xp(ms2)-1)-log(xp(ms2)))*xp(ms2
     . ))*(xs2-1-xq)*(xb1-xb2)*(xb1-1)*(xb2-1)+2*((log(xm(ms1)-1)-log
     . (xm(ms1)))*xm(ms1)+(log(xp(ms1)-1)-log(xp(ms1)))*xp(ms1))*(xs1
     . -1-xq)*(xb1-xb2)*(xb1-1)*(xb2-1)+2*(xs2-6+xs1-2*xq)*(xb1-xb2)*
     . (xb1-1)*(xb2-1))/(8*(xb1-xb2)*(xb1-1)**2*(xb2-1)**2)

c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ans2=(((3*xs1-1)*(xs1-1)+3*xq**2-2*(3*xs1+2)*xq-(xq-xs1+1)*(xq-
     . xs1-1)*xb2-(((xs1-1)*(xs1-3)+xq**2-2*(xs1+2)*xq)*xb2+(xq-xs1+1
     . )*(xq-xs1-1))*xb1)*(2*(xs2+1)*xq-(xs2-1)**2-xq**2)*t134(mg,mq,
     . ms1,mg)-2*(2*(2*xs2-1+2*xq-(xq-xs2)**2)*log(xs1)*xs1-3*((xs2+2
     . +xs1)*xq+(xs1-1)*(xs2-1)-3*xq**2)*(xs1-xs2)-2*(2*xs1-1+2*xq-(
     . xq-xs1)**2)*log(xs2)*xs2)*(xb1-1)*(xb2-1)*log(xq)*xq)*(xb1-xb2
     . )
      ans1=((2*(xs1+1)*xq-(xs1-1)**2-xq**2)*(xs2-1+xq)*(log(xs2)-6)*
     . log(xs2)*xs2-(xs1-1+xq)*(2*(xs2+1)*xq-(xs2-1)**2-xq**2)*(log(
     . xs1)-6)*log(xs1)*xs1)*(xb1-xb2)*(xb1-1)*(xb2-1)+((t134(mb1,mq,
     . ms1,mg)-t134(mb1,mq,ms2,mg))*(xb1+1)*(xb2-1)**2-(t134(mb2,mq,
     . ms1,mg)-t134(mb2,mq,ms2,mg))*(xb1-1)**2*(xb2+1))*(2*(xs1+1)*xq
     . -(xs1-1)**2-xq**2)*(2*(xs2+1)*xq-(xs2-1)**2-xq**2)-((xs2+2+xs1
     . )*xq+(xs1-1)*(xs2-1)-3*xq**2)*(xb1-xb2)*(xb1-1)*(xb2-1)*(xs1-
     . xs2)*log(xq)**2*xq+14*(xs2-1+xs1-(xq+xs1)*(xq+xs2)+2*(xq**2-
     . xs1*xs2)*xq)*(xb1-xb2)*(xb1-1)*(xb2-1)*(xs1-xs2)-((3*xs2-1)*(
     . xs2-1)+3*xq**2-2*(3*xs2+2)*xq-(xq-xs2+1)*(xq-xs2-1)*xb2-(((xs2
     . -1)*(xs2-3)+xq**2-2*(xs2+2)*xq)*xb2+(xq-xs2+1)*(xq-xs2-1))*xb1
     . )*(2*(xs1+1)*xq-(xs1-1)**2-xq**2)*(xb1-xb2)*t134(mg,mq,ms2,mg)
     . +ans2
      rtp102=ans1/(2*((xs1-1)**2+xq**2-2*(xs1+1)*xq)*((xs2-1)**2+xq**2-
     . 2*(xs2+1)*xq)*(xb1-xb2)*(xb1-1)**2*(xb2-1)**2*(xs1-xs2))

      rmgtrt2=(-((log(xp(ms2)-1)-log(xp(ms2)))*xp(ms2)+log(xs1)-log(
     . xs2)-(log(xp(ms1)-1)-log(xp(ms1)))*xp(ms1)+(log(xm(ms2)-1)-log
     . (xm(ms2)))*xm(ms2)-(log(xm(ms1)-1)-log(xm(ms1)))*xm(ms1))*((
     . xb1+1)*(xb2-1)**2*log(xb1)*xb1-(xb1-1)**2*(xb2+1)*log(xb2)*xb2
     . +2*(xb1-xb2)*(xb1-1)*(xb2-1)))/(4*(xb1-xb2)*(xb1-1)**2*(xb2-1)
     . **2)

      fact = amt**2*xxt/amg**3
      rtp102 = fact * rtp102
      fact = 2*amt**2*xxt/(amst1**2-amst2**2)/amg
      rmgtrt2 = fact * rmgtrt2
      rtp102 = rtp102 + rmgtrt2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      rctca = ralsca2 + rmgca2
      rctcf = ralscf2 + rmb12 + rmb22
      ructtr = rualstr2 + rumgtr2
      rdcttr = rdalstr2 + rdmgtr2
      rbcttr = rbalstr2 + rbmgtr2
      rtcttr = rtalstr2 + rtmgtr2

      bo  =-2*fi(amsb1**2,amsb2**2,amg**2)

      rca = r12 + r22/4 + r32 + r52 + r62 + rctca
      rcf = -r22/2 - r42/2 - 2*r52 - r72/2 + r82 - r92/2 + rctcf
      rtr = nu*(ru102 + ructtr) + nd*(rd102 + rdcttr)
      rtrb= rb102 + rbcttr
      rtrt= rt102 + rtp102 + rtcttr + bo/3*dlog(amg**2/amt**2)/fnorm

      rqcd = ca*rca + cf*rcf + tr*(rtr+rtrb+rtrt)
      rqcd = rqcd/bo
      anomalous = - cf/4
      finscale = (11*ca-4*tr*nf)/12*dlog(scale**2/amg**2)
      fqcd_hdec = dreal(rqcd)*fnorm + anomalous + finscale

c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     fac = fnorm/bo
c     write(6,*)'par: ',scale,amg,amsb1,amsb2,amst1,amst2
c     write(6,*)'C_A: ',dreal(ca*r12*fac),dreal(ca*r22/4*fac),
c    .                  dreal(ca*r32*fac),dreal(ca*r42*fac),
c    .                  dreal(ca*r62*fac),dreal(ca*rctca*fac)
c     write(6,*)'C_F: ',dreal(-cf*r22/2*fac),dreal(-cf*r42/2*fac),
c    .  dreal(-2*cf*r52*fac),dreal(cf*r72/2*fac),dreal(cf*r82*fac),
c    .                  dreal(-cf*r92/2*fac),dreal(cf*rctcf*fac)
c     write(6,*)'T_R: ',dreal(tr*rtr*fac),dreal(tr*rtrb*fac),
c    .                  dreal(tr*rtrt*fac)
c     write(6,*)'rtrt ',dreal(tr*rt102*fac),dreal(tr*rtp102*fac),
c    .                  dreal(tr*rtcttr*fac),3*dlog(amg**2/amt**2)
c     write(6,*)'QCD: ',dreal(ca*rca*fac),dreal(cf*rcf*fac),
c    .                  dreal(tr*(rtr+rtrb+rtrt)*fac),anomalous,finscale
c     write(6,*)'sum: ',fqcd_hdec
c     write(6,*)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     fac = fnorm/bo
c     ysu = tr*dreal(nu*(ru102+ructtr)*fac
c    .              -4*nu/12.d0*dlog(scale**2/amg**2))
c     ysd = tr*dreal(nd*(rd102+rdcttr)*fac
c    .              -4*nd/12.d0*dlog(scale**2/amg**2))
c     ysb = tr*dreal(rtrb*fac-4/12.d0*dlog(scale**2/amg**2))
c     yst = tr*dreal(rtrt-rtp102)*fac
c     ystp= tr*dreal(rtp102/xxt)*fac
c     fac = 0.7705157934627561d0*0.088374016132112168d0/pi
c     ysu  = ysu * fac
c     ysd  = ysd * fac
c     ysb  = ysb * fac
c     yst  = yst * fac
c     ystp = ystp* fac
c     write(6,*)'dmb: '
c     write(6,*)'full ',dreal(rqcd)*fnorm*fac,anomalous*fac,finscale*fac
c     write(6,*)'CA = ',ca*(dreal(rca)*fnorm/bo
c    .                     +11/12.d0*dlog(scale**2/amg**2))*fac
c     write(6,*)'CF = ',cf*(dreal(rcf)*fnorm/bo-1/4.d0)*fac
c     write(6,*)'TR = ',tr*(dreal(rtr+rtrb+rtrt)*fnorm/bo
c    .                     -4*nf/12.d0*dlog(scale**2/amg**2))*fac
c     write(6,*)'su,sd,sb,st: ',ysu,ysd,ysb,yst,ystp*xxt,ystp
c     write(6,*)'xt = ',xxt
c     write(6,*)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      return
      end
 
