subroutine FFVM_3(F1, F2, COUP, M3, W3,V3)
use aloha_object
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 COUP
 type(aloha) F1
 integer flv_index1 
 type(aloha) F2
 integer flv_index2 
 real*8 M3
 real*8 OM3
 real*8 P3(0:3)
 complex*16 TMP0
 type(aloha) V3
 real*8 W3
 complex*16 denom
    OM3 = 0d0
    if (M3.ne.0d0) OM3=1d0/M3**2
    V3%P(:) = +F1%P(:)+F2%P(:)
P3(:) = -V3 % P (:)
   flv_index1 = F1 %flv_index
   flv_index2 = F2 %flv_index
   if(flv_index1.ne.flv_index2.or.flv_index1.eq.0d0)then  
 V3%W(:) = (0d0,0d0)
  return
endif
 TMP0 = (F1 % W(3)*(F2 % W(1)*(P3(0)+P3(3))+F2 % W(2)*(P3(1)-CI*(P3(2))))+F1 % W(4)*(F2 % W(1)*(P3(1)+CI*(P3(2)))+F2 % W(2)*(P3(0)-P3(3))))
    denom = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI* W3))
    V3%W(1)= denom*(-CI)*(F2 % W(1)*F1 % W(3)+F2 % W(2)*F1 % W(4)-P3(0)*OM3*TMP0)
    V3%W(2)= denom*(-CI)*(-F2 % W(2)*F1 % W(3)-F2 % W(1)*F1 % W(4)-P3(1)*OM3*TMP0)
    V3%W(3)= denom*(-CI)*(+CI*(F2 % W(2)*F1 % W(3))-CI*(F2 % W(1)*F1 % W(4))-P3(2)*OM3*TMP0)
    V3%W(4)= denom*(-CI)*(-F2 % W(1)*F1 % W(3)-P3(3)*OM3*TMP0+F2 % W(2)*F1 % W(4))
 end


subroutine MP_FFVM_3(F1, F2, COUP, M3, W3,V3)
use aloha_object
implicit none
 complex*32 CI
 parameter (CI=(0q0,1q0))
 complex*32 COUP
 type(aloha) F1
 integer flv_index1 
 type(aloha) F2
 integer flv_index2 
 real*16 M3
 real*16 OM3
 real*16 P3(0:3)
 complex*32 TMP0
 type(aloha) V3
 real*16 W3
 complex*32 denom
    OM3 = 0q0
    if (M3.ne.0q0) OM3=1q0/M3**2
    V3%P(:) = +F1%P(:)+F2%P(:)
P3(:) = -V3 % P (:)
   flv_index1 = F1 %flv_index
   flv_index2 = F2 %flv_index
   if(flv_index1.ne.flv_index2.or.flv_index1.eq.0d0)then  
 V3%W(:) = (0d0,0d0)
  return
endif
 TMP0 = (F1 % W(3)*(F2 % W(1)*(P3(0)+P3(3))+F2 % W(2)*(P3(1)-CI*(P3(2))))+F1 % W(4)*(F2 % W(1)*(P3(1)+CI*(P3(2)))+F2 % W(2)*(P3(0)-P3(3))))
    denom = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI* W3))
    V3%W(1)= denom*(-CI)*(F2 % W(1)*F1 % W(3)+F2 % W(2)*F1 % W(4)-P3(0)*OM3*TMP0)
    V3%W(2)= denom*(-CI)*(-F2 % W(2)*F1 % W(3)-F2 % W(1)*F1 % W(4)-P3(1)*OM3*TMP0)
    V3%W(3)= denom*(-CI)*(+CI*(F2 % W(2)*F1 % W(3))-CI*(F2 % W(1)*F1 % W(4))-P3(2)*OM3*TMP0)
    V3%W(4)= denom*(-CI)*(-F2 % W(1)*F1 % W(3)-P3(3)*OM3*TMP0+F2 % W(2)*F1 % W(4))
 end


