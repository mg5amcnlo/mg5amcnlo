subroutine RFSC1_1(R1, S3, COUP, M2, W2,F2)
use aloha_object
use model_object
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 COUP
 type(aloha) F2
 integer flv_index2 
 real*8 M2
 real*8 P2(0:3)
 type(aloha2d) R1
 type(aloha) S3
 real*8 W2
 complex*16 denom
    F2%%P(:) = +R1%%P(:)+S3%%P(:)
P2(:) = -F2 %% P (:)
    denom = COUP/(P2(0)**2-P2(1)**2-P2(2)**2-P2(3)**2 - M2 * (M2 -CI* W2))
    F2%%W(1)= denom*CI * S3 %% W(1)*(P2(0)*(-1d0)*(-R1 %% W(1)+R1 %% W(6)+R1 %% W(13)+CI*(R1 %% W(10)))+(P2(1)*(R1 %% W(2)+R1 %% W(14)-R1 %% W(5)+CI*(R1 %% W(9)))+(P2(2)*(+CI*(R1 %% W(2)+R1 %% W(14))-CI*(R1 %% W(5))-R1 %% W(9))-P2(3)*(-R1 %% W(1)+R1 %% W(6)+R1 %% W(13)+CI*(R1 %% W(10))))))
    F2%%W(2)= denom*CI * S3 %% W(1)*(P2(0)*(R1 %% W(2)+R1 %% W(14)-R1 %% W(5)+CI*(R1 %% W(9)))+(P2(1)*(-1d0)*(-R1 %% W(1)+R1 %% W(6)+R1 %% W(13)+CI*(R1 %% W(10)))+(P2(2)*(-CI*(R1 %% W(1))+CI*(R1 %% W(6)+R1 %% W(13))-R1 %% W(10))-P2(3)*(R1 %% W(2)+R1 %% W(14)-R1 %% W(5)+CI*(R1 %% W(9))))))
    F2%%W(3)= denom*CI * M2*S3 %% W(1)*(-R1 %% W(1)+R1 %% W(6)+R1 %% W(13)+CI*(R1 %% W(10)))
    F2%%W(4)= denom*(-CI )* M2*S3 %% W(1)*(R1 %% W(2)+R1 %% W(14)-R1 %% W(5)+CI*(R1 %% W(9)))
 end


