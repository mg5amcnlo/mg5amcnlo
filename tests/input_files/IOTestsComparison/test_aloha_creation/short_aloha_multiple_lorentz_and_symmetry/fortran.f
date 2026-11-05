subroutine VVS1_1(V2, S3, COUP, M1, W1,V1)
use aloha_object
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 COUP
 real*8 M1
 real*8 OM1
 real*8 P1(0:3)
 type(aloha) S3
 complex*16 TMP0
 type(aloha) V1
 type(aloha) V2
 real*8 W1
 complex*16 denom
entry VVS1_2(V2, S3, COUP, M1, W1,V1)

    OM1 = 0d0
    if (M1.ne.0d0) OM1=1d0/M1**2
    V1%P(:) = +V2%P(:)+S3%P(:)
P1(:) = -V1 % P (:)
 TMP0 = (V2 % W(1)*P1(0)-V2 % W(2)*P1(1)-V2 % W(3)*P1(2)-V2 % W(4)*P1(3))
    denom = COUP/(P1(0)**2-P1(1)**2-P1(2)**2-P1(3)**2 - M1 * (M1 -CI* W1))
    V1%W(1)= denom*S3 % W(1)*(-CI*(V2 % W(1))+CI*(P1(0)*OM1*TMP0))
    V1%W(2)= denom*S3 % W(1)*(-CI*(V2 % W(2))+CI*(P1(1)*OM1*TMP0))
    V1%W(3)= denom*S3 % W(1)*(-CI*(V2 % W(3))+CI*(P1(2)*OM1*TMP0))
    V1%W(4)= denom*S3 % W(1)*(-CI*(V2 % W(4))+CI*(P1(3)*OM1*TMP0))
 end



subroutine VVS1_2_1(V2, S3, COUP1, COUP2, M1, W1,V1)
use aloha_object
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 COUP1
 complex*16 COUP2
 real*8 M1
 real*8 OM1
 real*8 P1(0:3)
 type(aloha) S3
 type(aloha) V1
 type(aloha) V2
 type(aloha) Vtmp
 real*8 W1
 complex*16 denom
 integer*4 i
entry VVS1_2_2(V2, S3, COUP1, COUP2, M1, W1,V1)

    call VVS1_1(V2,S3,COUP1,M1,W1,V1)
    call VVS2_1(V2,S3,COUP2,M1,W1,Vtmp)
 do i = 1, 4
        V1 %W(i) = V1%W(i) + Vtmp%W(i)
 enddo
 end

