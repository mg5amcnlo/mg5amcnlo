subroutine FFV1C1_2(F2, V3, COUP, M1, W1,F1)
use aloha_object
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 COUP
 type(aloha) F1
 integer flv_index1 
 type(aloha) F2
 integer flv_index2 
 real*8 M1
 real*8 P1(0:3)
 type(aloha) V3
 real*8 W1
 complex*16 denom
    F1%P(:) = +F2%P(:)+V3%P(:)
P1(:) = -F1 % P (:)
  F1 % FLV_INDEX = F2 % FLV_INDEX
    denom = COUP/(P1(0)**2-P1(1)**2-P1(2)**2-P1(3)**2 - M1 * (M1 -CI* W1))
    F1%W(1)= denom*(-CI)*(F2 % W(1)*(P1(0)*(V3 % W(1)+V3 % W(4))+(P1(1)*(-1d0)*(V3 % W(2)+CI*(V3 % W(3)))+(P1(2)*(+CI*(V3 % W(2))-V3 % W(3))-P1(3)*(V3 % W(1)+V3 % W(4)))))+(F2 % W(2)*(P1(0)*(V3 % W(2)-CI*(V3 % W(3)))+(P1(1)*(-V3 % W(1)+V3 % W(4))+(P1(2)*(+CI*(V3 % W(1))-CI*(V3 % W(4)))+P1(3)*(-V3 % W(2)+CI*(V3 % W(3))))))+M1*(F2 % W(3)*(V3 % W(1)-V3 % W(4))+F2 % W(4)*(-V3 % W(2)+CI*(V3 % W(3))))))
    F1%W(2)= denom*CI*(F2 % W(1)*(P1(0)*(-1d0)*(V3 % W(2)+CI*(V3 % W(3)))+(P1(1)*(V3 % W(1)+V3 % W(4))+(P1(2)*(+CI*(V3 % W(1)+V3 % W(4)))-P1(3)*(V3 % W(2)+CI*(V3 % W(3))))))+(F2 % W(2)*(P1(0)*(-V3 % W(1)+V3 % W(4))+(P1(1)*(V3 % W(2)-CI*(V3 % W(3)))+(P1(2)*(+CI*(V3 % W(2))+V3 % W(3))+P1(3)*(-V3 % W(1)+V3 % W(4)))))+M1*(F2 % W(3)*(V3 % W(2)+CI*(V3 % W(3)))-F2 % W(4)*(V3 % W(1)+V3 % W(4)))))
    F1%W(3)= denom*CI*(F2 % W(3)*(P1(0)*(-V3 % W(1)+V3 % W(4))+(P1(1)*(V3 % W(2)+CI*(V3 % W(3)))+(P1(2)*(-CI*(V3 % W(2))+V3 % W(3))+P1(3)*(-V3 % W(1)+V3 % W(4)))))+(F2 % W(4)*(P1(0)*(V3 % W(2)-CI*(V3 % W(3)))+(P1(1)*(-1d0)*(V3 % W(1)+V3 % W(4))+(P1(2)*(+CI*(V3 % W(1)+V3 % W(4)))+P1(3)*(V3 % W(2)-CI*(V3 % W(3))))))+M1*(F2 % W(1)*(-1d0)*(V3 % W(1)+V3 % W(4))+F2 % W(2)*(-V3 % W(2)+CI*(V3 % W(3))))))
    F1%W(4)= denom*(-CI)*(F2 % W(3)*(P1(0)*(-1d0)*(V3 % W(2)+CI*(V3 % W(3)))+(P1(1)*(V3 % W(1)-V3 % W(4))+(P1(2)*(+CI*(V3 % W(1))-CI*(V3 % W(4)))+P1(3)*(V3 % W(2)+CI*(V3 % W(3))))))+(F2 % W(4)*(P1(0)*(V3 % W(1)+V3 % W(4))+(P1(1)*(-V3 % W(2)+CI*(V3 % W(3)))+(P1(2)*(-1d0)*(+CI*(V3 % W(2))+V3 % W(3))-P1(3)*(V3 % W(1)+V3 % W(4)))))+M1*(F2 % W(1)*(V3 % W(2)+CI*(V3 % W(3)))+F2 % W(2)*(V3 % W(1)-V3 % W(4)))))
 end


