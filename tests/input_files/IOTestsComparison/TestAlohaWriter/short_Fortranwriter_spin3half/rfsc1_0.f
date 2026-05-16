subroutine RFSC1_0(F2, R1, S3, COUP,vertex)
use aloha_object
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 COUP
 type(aloha) F2
 integer flv_index2 
 type(aloha2d) R1
 type(aloha) S3
 complex*16 TMP0
 complex*16 vertex
 TMP0 = (F2 % W(3)*(-R1 % W(1)+R1 % W(6)+R1 % W(13)+CI*(R1 % W(10)))-F2 % W(4)*(R1 % W(2)+R1 % W(14)-R1 % W(5)+CI*(R1 % W(9))))
 vertex = COUP*(-CI * TMP0*S3 % W(1))
 end


