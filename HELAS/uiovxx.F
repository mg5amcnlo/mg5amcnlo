      subroutine uiovxx(fi,fo,vc,gc,gt,tmass,twidth , uiov)
c
c This subroutine computes an off-shell tensor boson 
c wavefunction from a flowing-out fermion, a flowing-in fermion and 
c a gauge vector boson.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex vc(6)          : input    vector                       v
c       complex gc(2)          : coupling constants                  gvf
c       complex gt             : coupling constant      gtfv=-1/Lambda/2
c       real    tmass          : mass  of output tensor T
c       real    twidth         : width of output tensor T
c
c output:
c       complex uiov(18)       : tensor boson       j^mu^nu(<fo|v,T|fi>)
c
c- by Q.Li - OCT. 2006
c- Added massless tensor - P. de Aquino - Oct. 2009 
c
      implicit none
      double complex fi(6), fo(6), vc(6), gc(2), gt, uiov(18)
      double precision  tmass, twidth

      double complex yiov(6,4)
      double complex d
      double precision pt(4), pt2, KTVC
      double precision MET(4,4)
      integer i,j
      
      double precision rZero, rTwo,r4,r8,r3
      parameter(rZero=0.0d0, rTwo=2.0d0, r4=4.0d0, r8=8.0d0, r3=3.0d0)
      double complex cone
      parameter( cone = ( 0.0d0, 1.0d0 ))


      yiov(5,1) = fo(5) + vc(5) -fi(5)
      yiov(6,1) = fo(6) + vc(6) -fi(6)

      pt(1) = dreal(yiov(5,1))
      pt(2) = dreal(yiov(6,1))
      pt(3) = dimag(yiov(6,1))
      pt(4) = dimag(yiov(5,1))
      
      pt2 = pt(1)**2-pt(2)**2-pt(3)**2-pt(4)**2
      KTVC = pt(1)*vc(1)-pt(2)*vc(2)-pt(3)*vc(3)-pt(4)*vc(4)
      
      do i=1,4
         do j=1,4
            MET(i,j) = 0.0d0
         enddo 
      enddo
      MET(1,1) =  1.0d0
      MET(2,2) = -1.0d0
      MET(3,3) = -1.0d0
      MET(4,4) = -1.0d0
      
      if ( tmass.eq.rZero ) then
         d =  -1.0d0/dcmplx( pt2, rZero )

         do i = 1,4
	    do j=1,4
            yiov(i,j) = fi(4)*gc(2)*(fo(1)*MET(i,j)*(vc(2) - cone*vc(3)) 
     &+ fo(2)*MET(i,j)*(-vc(1) - vc(4))) 
     &+ fi(1)*gc(1)*(fo(4)*MET(i,j)*(-vc(2) 
     &- cone*vc(3)) + fo(3)*MET(i,j)*(-vc(1) - vc(4))) 
     &+ fi(3)*gc(2)*(fo(2)*MET(i,j)*(vc(2) 
     &+ cone*vc(3)) + fo(1)*MET(i,j)*(-vc(1) + vc(4))) 
     &+ fi(2)*gc(1)*(fo(3)*MET(i,j)*(-vc(2) + cone*vc(3)) 
     &+ fo(4)*MET(i,j)*(-vc(1) + vc(4)))
   
         end do
        end do 
   
         yiov(1,1) = yiov(1,1)
     &- 2*fi(1)*fo(3)*gc(1)*vc(1) 
     &- 2*fi(2)*fo(4)*gc(1)*vc(1) 
     &- 2*fi(3)*fo(1)*gc(2)*vc(1) 
     &- 2*fi(4)*fo(2)*gc(2)*vc(1)
   
         yiov(2,2) = yiov(2,2)
     &+	2*fi(2)*fo(3)*gc(1)*vc(2) 
     &+ 2*fi(1)*fo(4)*gc(1)*vc(2) 
     &- 2*fi(4)*fo(1)*gc(2)*vc(2) 
     &- 2*fi(3)*fo(2)*gc(2)*vc(2)
   
         yiov(3,3) = yiov(3,3)
     &- 2*cone*fi(2)*fo(3)*gc(1)*vc(3) 
     &+ 2*cone*fi(1)*fo(4)*gc(1)*vc(3) 
     &+ 2*cone*fi(4)*fo(1)*gc(2)*vc(3) 
     &- 2*cone*fi(3)*fo(2)*gc(2)*vc(3)
   
         yiov(4,4) = yiov(4,4)
     &+ 2*fi(1)*fo(3)*gc(1)*vc(4) 
     &- 2*fi(2)*fo(4)*gc(1)*vc(4) 
     &- 2*fi(3)*fo(1)*gc(2)*vc(4) 
     &+ 2*fi(4)*fo(2)*gc(2)*vc(4)
   
         yiov(1,2) = yiov(1,2)
     &+ fi(3)*gc(2)*(-(fo(2)*vc(1)) - fo(1)*vc(2)) 
     &+ fi(4)*gc(2)*(-(fo(1)*vc(1)) - fo(2)*vc(2)) 
     &+ fi(1)*gc(1)*(fo(4)*vc(1) - fo(3)*vc(2)) 
     &+ fi(2)*gc(1)*(fo(3)*vc(1) - fo(4)*vc(2))
     
         yiov(1,3) = yiov(1,3)
     &+	fi(3)*gc(2)*(-(cone*fo(2)*vc(1)) - fo(1)*vc(3)) 
     &+ fi(4)*gc(2)*(cone*fo(1)*vc(1) - fo(2)*vc(3)) 
     &+ fi(1)*gc(1)*(cone*fo(4)*vc(1) - fo(3)*vc(3)) 
     &+ fi(2)*gc(1)*(-(cone*fo(3)*vc(1)) - fo(4)*vc(3))
     
         yiov(1,4) = yiov(1,4)
     &+ fi(2)*fo(4)*gc(1)*(-vc(1) - vc(4)) 
     &+ fi(3)*fo(1)*gc(2)*(-vc(1) - vc(4)) 
     &+ fi(1)*fo(3)*gc(1)*(vc(1) - vc(4)) 
     &+ fi(4)*fo(2)*gc(2)*(vc(1) - vc(4))
   
         yiov(2,3) = yiov(2,3)
     &+ fi(3)*fo(2)*gc(2)*(-(cone*vc(2)) - vc(3)) 
     &+ fi(4)*fo(1)*gc(2)*(cone*vc(2) - vc(3)) 
     &+ fi(2)*fo(3)*gc(1)*(-(cone*vc(2)) + vc(3)) 
     &+ fi(1)*fo(4)*gc(1)*(cone*vc(2) + vc(3))
   
         yiov(2,4) = yiov(2,4)
     &+ fi(4)*gc(2)*(fo(2)*vc(2) - fo(1)*vc(4)) 
     &+ fi(3)*gc(2)*(-(fo(1)*vc(2)) - fo(2)*vc(4)) 
     &+ fi(2)*gc(1)*(-(fo(4)*vc(2)) + fo(3)*vc(4)) 
     &+ fi(1)*gc(1)*(fo(3)*vc(2) + fo(4)*vc(4))

         yiov(3,4) = yiov(3,4)
     &+fi(4)*gc(2)*(fo(2)*vc(3) + cone*fo(1)*vc(4)) 
     &+ fi(3)*gc(2)*(-(fo(1)*vc(3)) - cone*fo(2)*vc(4)) 
     &+ fi(2)*gc(1)*(-(fo(4)*vc(3)) - cone*fo(3)*vc(4)) 
     &+ fi(1)*gc(1)*(fo(3)*vc(3) + cone*fo(4)*vc(4))
   
         yiov(2,1) = yiov(1,2)
         yiov(3,1) = yiov(1,3)
         yiov(4,1) = yiov(1,4)
         yiov(3,2) = yiov(2,3)
         yiov(4,2) = yiov(2,4)
         yiov(4,3) = yiov(3,4)
   			
         do i = 1,4
   	    do j=1,4
            yiov(i,j) = -yiov(i,j)*d*gt
         end do
        end do
      
      else
         if ( tmass.gt.rZero ) then
            d =  -1.0d0/dcmplx( pt2-tmass**2, tmass*twidth )
   
   
            do i = 1,4
   	       do j=1,4
                  yiov(i,j) = fi(3)*gc(2)*(fo(2)*(-((KTVC*r4*MET(i,j)*
     &      (-pt(2) - cone*pt(3)))/(r3*Tmass**2)) - 
     &      (KTVC*r8*(-pt(2) - cone*pt(3))*pt(i)*pt(j))/(r3*Tmass**4) +
     &          (pt2*r4*MET(i,j)*(-vc(2) - cone*vc(3)))/(r3*Tmass**2) + 
     &          (pt2*r8*pt(i)*pt(j)*(-vc(2) - cone*vc(3)))/(r3*Tmass**4)
     & - (r4*pt(i)*pt(j)*(-vc(2) - cone*vc(3)))/Tmass**2 + 
     &    (rtwo*(-pt(2) - cone*pt(3))*(pt(j)*vc(i) + pt(i)*vc(j)))
     &/Tmass**2) + 
     &       fo(1)*(-((KTVC*r4*MET(i,j)*(pt(1) - pt(4)))/(r3*Tmass**2))
     & - (KTVC*r8*(pt(1) - pt(4))*pt(i)*pt(j))/(r3*Tmass**4) + 
     &          (pt2*r4*MET(i,j)*(vc(1) - vc(4)))/(r3*Tmass**2)
     & + (pt2*r8*pt(i)*pt(j)*(vc(1) - vc(4)))/(r3*Tmass**4) - 
     &          (r4*pt(i)*pt(j)*(vc(1) - vc(4)))/Tmass**2
     & + (rtwo*(pt(1) - pt(4))*(pt(j)*vc(i) + pt(i)*vc(j)))/Tmass**2))  
     &   + fi(2)*gc(1)*(fo(3)*(-((KTVC*r4*MET(i,j)*(pt(2) 
     &- cone*pt(3)))/(r3*Tmass**2)) - 
     &     (KTVC*r8*(pt(2) - cone*pt(3))*pt(i)*pt(j))/(r3*Tmass**4) + 
     &          (pt2*r4*MET(i,j)*(vc(2) - cone*vc(3)))/(r3*Tmass**2) + 
     &          (pt2*r8*pt(i)*pt(j)*(vc(2) - cone*vc(3)))/(r3*Tmass**4)
     & - (r4*pt(i)*pt(j)*(vc(2) - cone*vc(3)))/Tmass**2 + 
     &    (rtwo*(pt(2) - cone*pt(3))*(pt(j)*vc(i) 
     &+ pt(i)*vc(j)))/Tmass**2) + 
     &       fo(4)*(-((KTVC*r4*MET(i,j)*(pt(1) - pt(4)))/(r3*Tmass**2)) 
     &- (KTVC*r8*(pt(1) - pt(4))*pt(i)*pt(j))/(r3*Tmass**4) + 
     &          (pt2*r4*MET(i,j)*(vc(1) - vc(4)))/(r3*Tmass**2) 
     &+ (pt2*r8*pt(i)*pt(j)*(vc(1) - vc(4)))/(r3*Tmass**4) - 
     &          (r4*pt(i)*pt(j)*(vc(1) - vc(4)))/Tmass**2 
     &+ (rtwo*(pt(1) - pt(4))*(pt(j)*vc(i) + pt(i)*vc(j)))/Tmass**2)) + 
     &    fi(4)*gc(2)*(fo(1)*(-((KTVC*r4*MET(i,j)*(-pt(2) + cone*pt(3)))
     &/(r3*Tmass**2)) - 
     &      (KTVC*r8*(-pt(2) + cone*pt(3))*pt(i)*pt(j))/(r3*Tmass**4) + 
     &          (pt2*r4*MET(i,j)*(-vc(2) + cone*vc(3)))/(r3*Tmass**2) + 
     &          (pt2*r8*pt(i)*pt(j)*(-vc(2) + cone*vc(3)))/(r3*Tmass**4) 
     &- (r4*pt(i)*pt(j)*(-vc(2) + cone*vc(3)))/Tmass**2 + 
     &          (rtwo*(-pt(2) + cone*pt(3))*(pt(j)*vc(i) 
     &+ pt(i)*vc(j)))/Tmass**2) + 
     &       fo(2)*(-((KTVC*r4*MET(i,j)*(pt(1) + pt(4)))/(r3*Tmass**2))
     & - (KTVC*r8*(pt(1) + pt(4))*pt(i)*pt(j))/(r3*Tmass**4) + 
     &          (pt2*r4*MET(i,j)*(vc(1) + vc(4)))/(r3*Tmass**2)
     & + (pt2*r8*pt(i)*pt(j)*(vc(1) + vc(4)))/(r3*Tmass**4) - 
     &          (r4*pt(i)*pt(j)*(vc(1) + vc(4)))/Tmass**2
     & + (rtwo*(pt(1) + pt(4))*(pt(j)*vc(i) + pt(i)*vc(j)))/Tmass**2)) + 
     &    fi(1)*gc(1)*(fo(4)*(-((KTVC*r4*MET(i,j)*(pt(2) + cone*pt(3)))
     &/(r3*Tmass**2)) - 
     &     (KTVC*r8*(pt(2) + cone*pt(3))*pt(i)*pt(j))/(r3*Tmass**4) + 
     &          (pt2*r4*MET(i,j)*(vc(2) + cone*vc(3)))/(r3*Tmass**2) + 
     &          (pt2*r8*pt(i)*pt(j)*(vc(2) + cone*vc(3)))/(r3*Tmass**4)
     & - (r4*pt(i)*pt(j)*(vc(2) + cone*vc(3)))/Tmass**2 + 
     &          (rtwo*(pt(2) + cone*pt(3))*(pt(j)*vc(i)
     & + pt(i)*vc(j)))/Tmass**2) + 
     &       fo(3)*(-((KTVC*r4*MET(i,j)*(pt(1) + pt(4)))/(r3*Tmass**2))
     & - (KTVC*r8*(pt(1) + pt(4))*pt(i)*pt(j))/(r3*Tmass**4) + 
     &          (pt2*r4*MET(i,j)*(vc(1) + vc(4)))/(r3*Tmass**2) 
     &+ (pt2*r8*pt(i)*pt(j)*(vc(1) + vc(4)))/(r3*Tmass**4) - 
     &          (r4*pt(i)*pt(j)*(vc(1) + vc(4)))/Tmass**2 
     &+ (rtwo*(pt(1) + pt(4))*(pt(j)*vc(i) + pt(i)*vc(j)))/Tmass**2))
   
               end do
            end do 
   
            yiov(1,1) = yiov(1,1)
     & +fi(1)*fo(3)*gc(1)*((4*KTVC*pt(1))/Tmass**2 - 2*rtwo*vc(1)) + 
     &  fi(2)*fo(4)*gc(1)*((4*KTVC*pt(1))/Tmass**2 - 2*rtwo*vc(1)) + 
     &  fi(3)*fo(1)*gc(2)*((4*KTVC*pt(1))/Tmass**2 - 2*rtwo*vc(1)) 
     &+ fi(4)*fo(2)*gc(2)*((4*KTVC*pt(1))/Tmass**2 - 2*rtwo*vc(1))
   
            yiov(2,2) = yiov(2,2)
     &	+fi(4)*fo(1)*gc(2)*((4*KTVC*pt(2))/Tmass**2 - 2*rtwo*vc(2)) + 
     &    fi(3)*fo(2)*gc(2)*((4*KTVC*pt(2))/Tmass**2 - 2*rtwo*vc(2)) + 
     &    fi(2)*fo(3)*gc(1)*((-4*KTVC*pt(2))/Tmass**2 + 2*rtwo*vc(2)) + 
     &    fi(1)*fo(4)*gc(1)*((-4*KTVC*pt(2))/Tmass**2 + 2*rtwo*vc(2))

            yiov(3,3) = yiov(3,3)
     &+fi(2)*fo(3)*gc(1)*((4*cone*KTVC*pt(3))/Tmass**2
     & - 2*cone*rtwo*vc(3)) + 
     &    fi(3)*fo(2)*gc(2)*((4*cone*KTVC*pt(3))/Tmass**2 
     &- 2*cone*rtwo*vc(3)) + 
     &    fi(1)*fo(4)*gc(1)*((-4*cone*KTVC*pt(3))/Tmass**2
     & + 2*cone*rtwo*vc(3)) + 
     &    fi(4)*fo(1)*gc(2)*((-4*cone*KTVC*pt(3))/Tmass**2
     & + 2*cone*rtwo*vc(3))
   
            yiov(4,4) = yiov(4,4)
     &+fi(2)*fo(4)*gc(1)*((4*KTVC*pt(4))/Tmass**2 - 2*rtwo*vc(4)) + 
     &    fi(3)*fo(1)*gc(2)*((4*KTVC*pt(4))/Tmass**2 - 2*rtwo*vc(4)) + 
     &    fi(1)*fo(3)*gc(1)*((-4*KTVC*pt(4))/Tmass**2 + 2*rtwo*vc(4)) + 
     &    fi(4)*fo(2)*gc(2)*((-4*KTVC*pt(4))/Tmass**2 + 2*rtwo*vc(4))
   
            yiov(1,2) = yiov(1,2)
     &	+fi(3)*gc(2)*(fo(2)*((2*KTVC*pt(1))/Tmass**2 - rtwo*vc(1))
     & + fo(1)*((2*KTVC*pt(2))/Tmass**2 - rtwo*vc(2))) + 
     &    fi(4)*gc(2)*(fo(1)*((2*KTVC*pt(1))/Tmass**2 - rtwo*vc(1)) 
     &+ fo(2)*((2*KTVC*pt(2))/Tmass**2 - rtwo*vc(2))) + 
     &    fi(1)*gc(1)*(fo(4)*((-2*KTVC*pt(1))/Tmass**2 + rtwo*vc(1))
     & + fo(3)*((2*KTVC*pt(2))/Tmass**2 - rtwo*vc(2))) + 
     &    fi(2)*gc(1)*(fo(3)*((-2*KTVC*pt(1))/Tmass**2 + rtwo*vc(1))
     & + fo(4)*((2*KTVC*pt(2))/Tmass**2 - rtwo*vc(2)))
   
            yiov(1,3) = yiov(1,3)
     &	+fi(3)*gc(2)*(fo(2)*((2*cone*KTVC*pt(1))/Tmass**2 
     &- cone*rtwo*vc(1)) + 
     &       fo(1)*((2*KTVC*pt(3))/Tmass**2 - rtwo*vc(3))) + 
     &    fi(4)*gc(2)*(fo(1)*((-2*cone*KTVC*pt(1))/Tmass**2 
     &+ cone*rtwo*vc(1)) + fo(2)*((2*KTVC*pt(3))/Tmass**2 
     &- rtwo*vc(3))) + 
     &    fi(1)*gc(1)*(fo(4)*((-2*cone*KTVC*pt(1))/Tmass**2
     & + cone*rtwo*vc(1)) + fo(3)*((2*KTVC*pt(3))/Tmass**2 
     &- rtwo*vc(3))) + 
     &    fi(2)*gc(1)*(fo(3)*((2*cone*KTVC*pt(1))/Tmass**2
     & -cone*rtwo*vc(1))+ fo(4)*((2*KTVC*pt(3))/Tmass**2 - rtwo*vc(3)))
   
            yiov(1,4) = yiov(1,4)
     &+fi(1)*fo(3)*gc(1)*((2*KTVC*(-pt(1) + pt(4)))/Tmass**2
     & - rtwo*(-vc(1) + vc(4))) + 
     &     fi(4)*fo(2)*gc(2)*((2*KTVC*(-pt(1) + pt(4)))/Tmass**2
     & - rtwo*(-vc(1) + vc(4))) + 
     &    fi(2)*fo(4)*gc(1)*((2*KTVC*(pt(1) + pt(4)))/Tmass**2 
     &- rtwo*(vc(1) + vc(4))) + 
     &    fi(3)*fo(1)*gc(2)*((2*KTVC*(pt(1) + pt(4)))/Tmass**2 
     &- rtwo*(vc(1) + vc(4)))
   
            yiov(2,3) = yiov(2,3)
     &+fi(1)*fo(4)*gc(1)*((2*KTVC*(-(cone*pt(2)) - pt(3)))/Tmass**2
     & - rtwo*(-(cone*vc(2)) - vc(3))) + 
     &    fi(2)*fo(3)*gc(1)*((2*KTVC*(cone*pt(2) - pt(3)))/Tmass**2
     & - rtwo*(cone*vc(2) - vc(3))) + 
     &    fi(4)*fo(1)*gc(2)*((2*KTVC*(-(cone*pt(2)) + pt(3)))/Tmass**2
     & - rtwo*(-(cone*vc(2)) + vc(3))) + 
     &    fi(3)*fo(2)*gc(2)*((2*KTVC*(cone*pt(2) + pt(3)))/Tmass**2 
     &- rtwo*(cone*vc(2) + vc(3)))
   
            yiov(2,4) = yiov(2,4)
     &+fi(4)*gc(2)*(fo(2)*((-2*KTVC*pt(2))/Tmass**2 + rtwo*vc(2))
     & + fo(1)*((2*KTVC*pt(4))/Tmass**2 - rtwo*vc(4))) + 
     &    fi(3)*gc(2)*(fo(1)*((2*KTVC*pt(2))/Tmass**2 - rtwo*vc(2))
     & + fo(2)*((2*KTVC*pt(4))/Tmass**2 - rtwo*vc(4))) + 
     &    fi(2)*gc(1)*(fo(4)*((2*KTVC*pt(2))/Tmass**2 - rtwo*vc(2)) 
     &+ fo(3)*((-2*KTVC*pt(4))/Tmass**2 + rtwo*vc(4))) + 
     &    fi(1)*gc(1)*(fo(3)*((-2*KTVC*pt(2))/Tmass**2 + rtwo*vc(2))
     & + fo(4)*((-2*KTVC*pt(4))/Tmass**2 + rtwo*vc(4)))

            yiov(3,4) = yiov(3,4)
     &	+fi(3)*gc(2)*(fo(1)*((2*KTVC*pt(3))/Tmass**2 - rtwo*vc(3)) + 
     &       fo(2)*((2*cone*KTVC*pt(4))/Tmass**2 - cone*rtwo*vc(4))) + 
     &    fi(2)*gc(1)*(fo(4)*((2*KTVC*pt(3))/Tmass**2 - rtwo*vc(3)) 
     &+ fo(3)*((2*cone*KTVC*pt(4))/Tmass**2 - cone*rtwo*vc(4))) + 
     &    fi(4)*gc(2)*(fo(2)*((-2*KTVC*pt(3))/Tmass**2 + rtwo*vc(3)) + 
     &       fo(1)*((-2*cone*KTVC*pt(4))/Tmass**2 + cone*rtwo*vc(4))) + 
     &    fi(1)*gc(1)*(fo(3)*((-2*KTVC*pt(3))/Tmass**2 + rtwo*vc(3)) 
     &+ fo(4)*((-2*cone*KTVC*pt(4))/Tmass**2 + cone*rtwo*vc(4)))

            yiov(2,1) = yiov(1,2)
            yiov(3,1) = yiov(1,3)
            yiov(4,1) = yiov(1,4)
            yiov(3,2) = yiov(2,3)
            yiov(4,2) = yiov(2,4)
            yiov(4,3) = yiov(3,4)
			
             do i = 1,4
            do j=1,4
               yiov(i,j) = -yiov(i,j)*d/2.0d0*gt
             end do
            end do
         
         else
            write(*,*) 'invalid tensor mass'
         end if
      end if
  
      uiov(1) = yiov(1,1)
      uiov(2) = yiov(1,2)
      uiov(3) = yiov(1,3)
      uiov(4) = yiov(1,4)
      uiov(5) = yiov(2,1)
      uiov(6) = yiov(2,2)
      uiov(7) = yiov(2,3)
      uiov(8) = yiov(2,4)
      uiov(9) = yiov(3,1)
      uiov(10) = yiov(3,2)
      uiov(11) = yiov(3,3)
      uiov(12) = yiov(3,4)
      uiov(13) = yiov(4,1)
      uiov(14) = yiov(4,2)
      uiov(15) = yiov(4,3)
      uiov(16) = yiov(4,4)
      uiov(17) = yiov(5,1)
      uiov(18) = yiov(6,1)

      return
      end
