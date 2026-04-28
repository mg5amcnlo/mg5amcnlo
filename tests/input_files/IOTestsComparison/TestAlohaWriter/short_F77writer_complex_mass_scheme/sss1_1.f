subroutine SSS1_1(S2, S3, COUP, M1,S1)
use aloha_object
use model_object
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 COUP
 complex*16 M1
 real*8 P1(0:3)
 type(aloha) S1
 type(aloha) S2
 type(aloha) S3
 complex*16 denom
entry SSS1_2(S2, S3, COUP, M1,S1)

entry SSS1_3(S2, S3, COUP, M1,S1)

    S1%%P(:) = +S2%%P(:)+S3%%P(:)
P1(:) = -S1 %% P (:)
    denom = COUP/(P1(0)**2-P1(1)**2-P1(2)**2-P1(3)**2 - M1**2)
    S1%%W(1)= denom*CI * S3 %% W(1)*S2 %% W(1)
 end




