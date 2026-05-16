subroutine FFV1C1_1(F1, V3, COUP, M2, W2,F2)
use aloha_object
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 COUP
 type(aloha) F1
 integer flv_index1 
 type(aloha) F2
 integer flv_index2 
 complex*16 FCT1
 real*8 M2
 real*8 P2(0:3)
 real*8 P3(0:3)
 complex*16 TMP0
 type(aloha) V3
 real*8 W2
 complex*16 denom
 complex*16 mymdl_VEC
 external mymdl_VEC
P3(:) = V3 % P (:)
    F2%P(:) = +F1%P(:)+V3%P(:)
P2(:) = -F2 % P (:)
  F2 % FLV_INDEX = F1 % FLV_INDEX
 TMP0 = (P3(0)*P3(0)-P3(1)*P3(1)-P3(2)*P3(2)-P3(3)*P3(3))
 FCT1 = mymdl_VEC(TMP0)
    denom = COUP/(P2(0)**2-P2(1)**2-P2(2)**2-P2(3)**2 - M2 * (M2 -CI* W2))
    F2%W(1)= denom*(-CI )* FCT1*(F1 % W(1)*(P2(0)*(-V3 % W(1)+V3 % W(4))+(P2(1)*(V3 % W(2)-CI*(V3 % W(3)))+(P2(2)*(+CI*(V3 % W(2))+V3 % W(3))+P2(3)*(-V3 % W(1)+V3 % W(4)))))+(F1 % W(2)*(P2(0)*(V3 % W(2)+CI*(V3 % W(3)))+(P2(1)*(-1d0)*(V3 % W(1)+V3 % W(4))+(P2(2)*(-1d0)*(+CI*(V3 % W(1)+V3 % W(4)))+P2(3)*(V3 % W(2)+CI*(V3 % W(3))))))+M2*(F1 % W(3)*(V3 % W(1)+V3 % W(4))+F1 % W(4)*(V3 % W(2)+CI*(V3 % W(3))))))
    F2%W(2)= denom*CI * FCT1*(F1 % W(1)*(P2(0)*(-V3 % W(2)+CI*(V3 % W(3)))+(P2(1)*(V3 % W(1)-V3 % W(4))+(P2(2)*(-CI*(V3 % W(1))+CI*(V3 % W(4)))+P2(3)*(V3 % W(2)-CI*(V3 % W(3))))))+(F1 % W(2)*(P2(0)*(V3 % W(1)+V3 % W(4))+(P2(1)*(-1d0)*(V3 % W(2)+CI*(V3 % W(3)))+(P2(2)*(+CI*(V3 % W(2))-V3 % W(3))-P2(3)*(V3 % W(1)+V3 % W(4)))))+M2*(F1 % W(3)*(-V3 % W(2)+CI*(V3 % W(3)))+F1 % W(4)*(-V3 % W(1)+V3 % W(4)))))
    F2%W(3)= denom*CI * FCT1*(F1 % W(3)*(P2(0)*(V3 % W(1)+V3 % W(4))+(P2(1)*(-V3 % W(2)+CI*(V3 % W(3)))+(P2(2)*(-1d0)*(+CI*(V3 % W(2))+V3 % W(3))-P2(3)*(V3 % W(1)+V3 % W(4)))))+(F1 % W(4)*(P2(0)*(V3 % W(2)+CI*(V3 % W(3)))+(P2(1)*(-V3 % W(1)+V3 % W(4))+(P2(2)*(-CI*(V3 % W(1))+CI*(V3 % W(4)))-P2(3)*(V3 % W(2)+CI*(V3 % W(3))))))+M2*(F1 % W(1)*(-V3 % W(1)+V3 % W(4))+F1 % W(2)*(V3 % W(2)+CI*(V3 % W(3))))))
    F2%W(4)= denom*(-CI )* FCT1*(F1 % W(3)*(P2(0)*(-V3 % W(2)+CI*(V3 % W(3)))+(P2(1)*(V3 % W(1)+V3 % W(4))+(P2(2)*(-1d0)*(+CI*(V3 % W(1)+V3 % W(4)))+P2(3)*(-V3 % W(2)+CI*(V3 % W(3))))))+(F1 % W(4)*(P2(0)*(-V3 % W(1)+V3 % W(4))+(P2(1)*(V3 % W(2)+CI*(V3 % W(3)))+(P2(2)*(-CI*(V3 % W(2))+V3 % W(3))+P2(3)*(-V3 % W(1)+V3 % W(4)))))+M2*(F1 % W(1)*(-V3 % W(2)+CI*(V3 % W(3)))+F1 % W(2)*(V3 % W(1)+V3 % W(4)))))
 end


