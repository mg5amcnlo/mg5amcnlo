subroutine FFV1_3(F1, F2, COUP, M3, W3,V3)
use aloha_object
use model_object
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 COUP
 type(aloha) F1
 integer flv_index1 
 type(aloha) F2
 integer flv_index2 
 real*8 M3
 real*8 P3(0:3)
 type(aloha) V3
 real*8 W3
 complex*16 denom
    V3%%P(:) = +F1%%P(:)+F2%%P(:)
P3(:) = -V3 %% P (:)
   flv_index1 = F1 %%flv_index
   flv_index2 = F2 %%flv_index
   if(flv_index1.ne.flv_index2.or.flv_index1.eq.0d0)then  
 V3%%W(:) = (0d0,0d0)
  return
endif
    denom = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI* W3))
    V3%%W(1)= denom*(-CI)*(F2 %% W(3)*F1 %% W(1)+F2 %% W(4)*F1 %% W(2)+F2 %% W(1)*F1 %% W(3)+F2 %% W(2)*F1 %% W(4))
    V3%%W(2)= denom*(-CI)*(-F2 %% W(4)*F1 %% W(1)-F2 %% W(3)*F1 %% W(2)+F2 %% W(2)*F1 %% W(3)+F2 %% W(1)*F1 %% W(4))
    V3%%W(3)= denom*(-CI)*(-CI*(F2 %% W(4)*F1 %% W(1)+F2 %% W(1)*F1 %% W(4))+CI*(F2 %% W(3)*F1 %% W(2)+F2 %% W(2)*F1 %% W(3)))
    V3%%W(4)= denom*(-CI)*(-F2 %% W(3)*F1 %% W(1)-F2 %% W(2)*F1 %% W(4)+F2 %% W(4)*F1 %% W(2)+F2 %% W(1)*F1 %% W(3))
 end


