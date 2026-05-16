subroutine VVS4PZ1_2(V1, S3, COUP, M2, W2,V2)
use aloha_object
implicit none
 include "../MODEL/input.inc"
 include "../MODEL/coupl.inc"
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 COUP
 complex*16 FCT1
 real*8 M2
 real*8 P1(0:3)
 real*8 P2(0:3)
 type(aloha) S3
 complex*16 TMP0
 complex*16 TMP1
 complex*16 TMP2
 type(aloha) V1
 type(aloha) V2
 real*8 W2
 complex*16 denom
P1(:) = V1 % P (:)
    V2%P(:) = +V1%P(:)+S3%P(:)
P2(:) = -V2 % P (:)
 TMP0 = (P2(0)*P2(0)-P2(1)*P2(1)-P2(2)*P2(2)-P2(3)*P2(3))
 TMP1 = (P2(0)*V1 % W(1)-P2(1)*V1 % W(2)-P2(2)*V1 % W(3)-P2(3)*V1 % W(4))
 TMP2 = (P2(0)*P1(0)-P2(1)*P1(1)-P2(2)*P1(2)-P2(3)*P1(3))
 FCT1 = ((M2*(-M2+CI*(W2))+TMP0))**(2d0)
    denom = COUP/(FCT1)
    V2%W(1)= denom*M2*S3 % W(1)*mdl_dWZ*(-P1(0)*TMP1+V1 % W(1)*TMP2)
    V2%W(2)= denom*M2*S3 % W(1)*mdl_dWZ*(-P1(1)*TMP1+V1 % W(2)*TMP2)
    V2%W(3)= denom*M2*S3 % W(1)*mdl_dWZ*(-P1(2)*TMP1+V1 % W(3)*TMP2)
    V2%W(4)= denom*M2*S3 % W(1)*mdl_dWZ*(-P1(3)*TMP1+V1 % W(4)*TMP2)
 end


