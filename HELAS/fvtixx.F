      subroutine fvtixx(fi,vc,tc,gc,gt,fmass,fwidth , fvti)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-IN external fermion, a gauge boson and a tensor boson.
c
c input:
c       complex fi(6)          : flow-in fermion                    |fi>
c       complex vc(6)          : input   vector                        v
c       complex tc(18)         : input   tensor                        T
c       complex gc(2)          : coupling constants                  gvf
c       complex gt             : coupling constant      gtfv=-1/Lambda/2
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c
c output:
c       complex fvti(6)        : off-shell fermion           |f':v,T,fi>
c
c- by Q.Li - OCT. 2006
c
      implicit none
      double complex fi(6), vc(6), tc(18), gc(2), gt, fvti(6)
      double precision  fmass, fwidth

      double complex ft(6,4)
      double complex d, T00, T12, T13, T14, T23, T24, T34
      double precision po(4), po2
      integer i
      
      double precision rZero, rTwo
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      double complex cone
      parameter( cone = ( 0.0d0, 1.0d0 ))

      
      ft(1,1) = tc(1)
      ft(1,2) = tc(2)
      ft(1,3) = tc(3)
      ft(1,4) = tc(4)
      ft(2,1) = tc(5)
      ft(2,2) = tc(6)
      ft(2,3) = tc(7)
      ft(2,4) = tc(8)
      ft(3,1) = tc(9)
      ft(3,2) = tc(10)
      ft(3,3) = tc(11)
      ft(3,4) = tc(12)
      ft(4,1) = tc(13)
      ft(4,2) = tc(14)
      ft(4,3) = tc(15)
      ft(4,4) = tc(16)
      ft(5,1) = tc(17)
      ft(6,1) = tc(18)

      fvti(5) = fi(5) - ft(5,1) -vc(5)
      fvti(6) = fi(6) - ft(6,1) -vc(6)

      po(1) = dreal(fvti(5))
      po(2) = dreal(fvti(6))
      po(3) = dimag(fvti(6))
      po(4) = dimag(fvti(5))
 
      po2=po(1)**2-po(2)**2-po(3)**2-po(4)**2
     
      T00 = ft(1,1)-ft(2,2)-ft(3,3)-ft(4,4)
      T12 = ft(1,2) + ft(2,1)
      T13 = ft(1,3) + ft(3,1)
      T14 = ft(1,4) + ft(4,1)
      T23 = ft(2,3) + ft(3,2)
      T24 = ft(2,4) + ft(4,2)
      T34 = ft(3,4) + ft(4,3)
   
      if ( fmass.gt.rZero ) then
         d =  -1.0d0/dcmplx( po2-fmass**2, fmass*fwidth )
      else
         d =  -1.0d0/dcmplx( po2, rZero )
      end if

      fvti(1) =  fmass*fi(4)*gc(2)*(T12*vc(1) - cone*T13*vc(1)
     & - T23*(-(cone*vc(2)) + vc(3)) + 2.d0*T00*(-vc(2) + cone*vc(3))
     & - T24*vc(4) + cone*T34*vc(4)) + fmass*fi(3)*gc(2)*
     &     (T12*vc(2) - T24*vc(2) + T13*vc(3) - T34*vc(3)
     &+ 2.d0*T00*(vc(1) - vc(4)) + T14*(vc(1) + vc(4))) + 
     &    fi(1)*gc(1)*((-po(2) + cone*po(3))*(-(T12*vc(1))
     & - cone*T13*vc(1) - T23*(-(cone*vc(2)) - vc(3)) + 
     &          2.d0*T00*(vc(2) + cone*vc(3)) + T24*vc(4) 
     &+ cone*T34*vc(4)) + 
     &       (po(1) - po(4))*(T12*vc(2) + T24*vc(2) 
     &+ T13*vc(3) + T34*vc(3) + T14*(-vc(1) + vc(4)) 
     &+ 2.d0*T00*(vc(1) + vc(4)))) + 
     &    fi(2)*gc(1)*((po(1) - po(4))*(-(T12*vc(1))
     & + cone*T13*vc(1) - T23*(cone*vc(2) - vc(3)) 
     &+ 2.d0*T00*(vc(2) - cone*vc(3)) + 
     &          T24*vc(4) - cone*T34*vc(4)) 
     &+ (-po(2) + cone*po(3))*
     &        (T12*vc(2) - T24*vc(2) + T13*vc(3) 
     &- T34*vc(3) + 2.d0*T00*(vc(1) - vc(4)) + T14*(vc(1) + vc(4))))

      fvti(2) = fmass*fi(3)*gc(2)*(T12*vc(1) 
     &+ cone*T13*vc(1) - T23*(cone*vc(2) + vc(3))
     & + 2.d0*T00*(-vc(2) - cone*vc(3)) - 
     &       T24*vc(4) - cone*T34*vc(4)) + fmass*fi(4)*gc(2)*
     &     (T12*vc(2) + T24*vc(2) + T13*vc(3) + T34*vc(3)
     & + T14*(-vc(1) + vc(4)) + 2.d0*T00*(vc(1) + vc(4))) + 
     &    fi(1)*gc(1)*((po(1) + po(4))*(-(T12*vc(1)) - cone*T13*vc(1)
     & - T23*(-(cone*vc(2)) - vc(3)) + 
     &          2.d0*T00*(vc(2) + cone*vc(3))
     & + T24*vc(4) + cone*T34*vc(4)) + 
     &       (-po(2) - cone*po(3))*(T12*vc(2) + T24*vc(2)
     & + T13*vc(3) + T34*vc(3) + T14*(-vc(1) + vc(4)) + 
     &          2.d0*T00*(vc(1) + vc(4)))) + fi(2)*gc(1)*
     &     ((-po(2) - cone*po(3))*(-(T12*vc(1)) + cone*T13*vc(1)
     & - T23*(cone*vc(2) - vc(3)) + 2.d0*T00*(vc(2) - cone*vc(3)) + 
     &          T24*vc(4) - cone*T34*vc(4)) + (po(1) + po(4))*
     &        (T12*vc(2) - T24*vc(2) + T13*vc(3) - T34*vc(3) 
     &+ 2.d0*T00*(vc(1) - vc(4)) + T14*(vc(1) + vc(4))))

      fvti(3) = fmass*fi(2)*gc(1)*(-(T12*vc(1)) + cone*T13*vc(1)
     & - T23*(cone*vc(2) - vc(3)) + 2.d0*T00*(vc(2) - cone*vc(3)) + 
     &       T24*vc(4) - cone*T34*vc(4)) + fmass*fi(1)*gc(1)*
     &     (T12*vc(2) + T24*vc(2) + T13*vc(3) + T34*vc(3)
     & + T14*(-vc(1) + vc(4)) + 2.d0*T00*(vc(1) + vc(4))) + 
     &    fi(4)*gc(2)*((po(1) + po(4))*(T12*vc(1) - cone*T13*vc(1)
     & - T23*(-(cone*vc(2)) + vc(3)) + 2.d0*T00*(-vc(2) + cone*vc(3))  
     &       -T24*vc(4) + cone*T34*vc(4)) + (po(2) - cone*po(3))*
     &        (T12*vc(2) + T24*vc(2) + T13*vc(3) + T34*vc(3)
     & + T14*(-vc(1) + vc(4)) + 2.d0*T00*(vc(1) + vc(4)))) + 
     &    fi(3)*gc(2)*((po(2) - cone*po(3))*(T12*vc(1)
     & + cone*T13*vc(1) - T23*(cone*vc(2) + vc(3)) + 
     &          2.d0*T00*(-vc(2) - cone*vc(3))
     & - T24*vc(4) - cone*T34*vc(4)) + 
     &       (po(1) + po(4))*(T12*vc(2) - T24*vc(2)
     & + T13*vc(3) - T34*vc(3) + 2.d0*T00*(vc(1) - vc(4)) 
     & + T14*(vc(1) + vc(4))))

      fvti(4) = fmass*fi(1)*gc(1)*(-(T12*vc(1)) - cone*T13*vc(1)
     &	 - T23*(-(cone*vc(2)) - vc(3)) 
     & + 2.d0*T00*(vc(2) + cone*vc(3)) + 
     &       T24*vc(4) + cone*T34*vc(4)) + fmass*fi(2)*gc(1)*
     &     (T12*vc(2) - T24*vc(2) + T13*vc(3) - T34*vc(3)
     & + 2.d0*T00*(vc(1) - vc(4)) + T14*(vc(1) + vc(4))) + 
     &    fi(4)*gc(2)*((po(2) + cone*po(3))*(T12*vc(1) 
     & - cone*T13*vc(1) - T23*(-(cone*vc(2)) + vc(3)) + 
     &          2.d0*T00*(-vc(2) + cone*vc(3)) - T24*vc(4)
     & + cone*T34*vc(4)) + 
     &       (po(1) - po(4))*(T12*vc(2) + T24*vc(2)
     & + T13*vc(3) + T34*vc(3) + T14*(-vc(1) + vc(4))
     & + 2.d0*T00*(vc(1) + vc(4)))) + 
     &    fi(3)*gc(2)*((po(1) - po(4))*(T12*vc(1) 
     &+ cone*T13*vc(1) - T23*(cone*vc(2) + vc(3))
     & + 2.d0*T00*(-vc(2) - cone*vc(3)) - 
     &          T24*vc(4) - cone*T34*vc(4))
     & + (po(2) + cone*po(3))*
     &        (T12*vc(2) - T24*vc(2) + T13*vc(3) 
     &- T34*vc(3) + 2.d0*T00*(vc(1) - vc(4)) + T14*(vc(1) + vc(4))))  

      fvti(1) = fvti(1)
     &	-2*fmass*fi(4)*gc(2)*(ft(2,2)*vc(2) - cone*ft(3,3)*vc(3))
     & - 2*fmass*fi(3)*gc(2)*(ft(1,1)*vc(1) + ft(4,4)*vc(4)) + 
     &    fi(1)*gc(1)*(-2*(-po(2) + cone*po(3))*(-(ft(2,2)*vc(2))
     & - cone*ft(3,3)*vc(3)) - 
     &       2*(po(1) - po(4))*(ft(1,1)*vc(1) - ft(4,4)*vc(4))) + 
     &    fi(2)*gc(1)*(-2*(po(1) - po(4))*(-(ft(2,2)*vc(2)) 
     &+ cone*ft(3,3)*vc(3)) - 
     &       2*(-po(2) + cone*po(3))*(ft(1,1)*vc(1) + ft(4,4)*vc(4)))

      fvti(2) = fvti(2)
     &-2*fmass*fi(3)*gc(2)*(ft(2,2)*vc(2) + cone*ft(3,3)*vc(3))
     & - 2*fmass*fi(4)*gc(2)*(ft(1,1)*vc(1) - ft(4,4)*vc(4)) + 
     &   fi(1)*gc(1)*(-2*(po(1) + po(4))*(-(ft(2,2)*vc(2))
     & - cone*ft(3,3)*vc(3)) - 
     &       2*(-po(2) - cone*po(3))*(ft(1,1)*vc(1) - ft(4,4)*vc(4))) + 
     &    fi(2)*gc(1)*(-2*(-po(2) - cone*po(3))*(-(ft(2,2)*vc(2))
     & + cone*ft(3,3)*vc(3)) - 
     &       2*(po(1) + po(4))*(ft(1,1)*vc(1) + ft(4,4)*vc(4)))
     	
      fvti(3) = fvti(3)
     &-2*fmass*fi(2)*gc(1)*(-(ft(2,2)*vc(2)) + cone*ft(3,3)*vc(3))
     & - 2*fmass*fi(1)*gc(1)*(ft(1,1)*vc(1) - ft(4,4)*vc(4)) + 
     &    fi(4)*gc(2)*(-2*(po(1) + po(4))*(ft(2,2)*vc(2) 
     &- cone*ft(3,3)*vc(3)) - 
     &       2*(po(2) - cone*po(3))*(ft(1,1)*vc(1) - ft(4,4)*vc(4))) + 
     &    fi(3)*gc(2)*(-2*(po(2) - cone*po(3))*(ft(2,2)*vc(2) 
     &+ cone*ft(3,3)*vc(3)) - 
     &       2*(po(1) + po(4))*(ft(1,1)*vc(1) + ft(4,4)*vc(4)))	
	
      fvti(4) = fvti(4)
     &-2*fmass*fi(1)*gc(1)*(-(ft(2,2)*vc(2)) - cone*ft(3,3)*vc(3))
     & - 2*fmass*fi(2)*gc(1)*(ft(1,1)*vc(1) + ft(4,4)*vc(4)) + 
     &    fi(4)*gc(2)*(-2*(po(2) + cone*po(3))*(ft(2,2)*vc(2)
     & - cone*ft(3,3)*vc(3)) - 
     &       2*(po(1) - po(4))*(ft(1,1)*vc(1) - ft(4,4)*vc(4))) + 
     &    fi(3)*gc(2)*(-2*(po(1) - po(4))*(ft(2,2)*vc(2)
     & + cone*ft(3,3)*vc(3)) - 
     &       2*(po(2) + cone*po(3))*(ft(1,1)*vc(1) + ft(4,4)*vc(4)))	

      do i = 1,4
         fvti(i) = -fvti(i)*d*gt
      end do

      return
      end
