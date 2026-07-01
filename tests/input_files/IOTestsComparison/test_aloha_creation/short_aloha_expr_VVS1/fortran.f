subroutine VVS1_0(V1, V2, S3, COUP,vertex)
use aloha_object
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 COUP
 real*8 P1(0:3)
 type(aloha) S3
 complex*16 TMP0
 complex*16 TMP1
 complex*16 TMP2
 complex*16 TMP3
 type(aloha) V1
 type(aloha) V2
 complex*16 vertex
P1(:) = V1 % P (:)
 TMP0 = (V2 % W(1)*P1(0)-V2 % W(2)*P1(1)-V2 % W(3)*P1(2)-V2 % W(4)*P1(3))
 TMP1 = (P1(0)*V1 % W(1)-P1(1)*V1 % W(2)-P1(2)*V1 % W(3)-P1(3)*V1 % W(4))
 TMP2 = (V2 % W(1)*V1 % W(1)-V2 % W(2)*V1 % W(2)-V2 % W(3)*V1 % W(3)-V2 % W(4)*V1 % W(4))
 TMP3 = (P1(0)*P1(0)-P1(1)*P1(1)-P1(2)*P1(2)-P1(3)*P1(3))
 vertex = COUP*S3 % W(1)*(-CI*(TMP0*TMP1)+CI*(TMP2*TMP3))
 end


