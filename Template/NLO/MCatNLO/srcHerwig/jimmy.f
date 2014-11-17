      SUBROUTINE JMABEND
      WRITE(*,1000)
      RETURN
 1000 FORMAT(33H JMABEND ROUTINE HAS BEEN CALLED.)
      END
*
* $Id: jmdbesi0.F 9 2005-07-29 16:15:46Z jmb $
*
* $Log$
* Revision 1.1  2005/07/29 16:15:46  jmb
* Include the various CERNLIB functions jimmy needs
*
* Revision 1.1.1.1  1996/04/01 15:02:00  mclareni
* Mathlib gen
*
*
      DOUBLE PRECISION FUNCTION JMDBESI0(X)
      IMPLICIT DOUBLE PRECISION (A-H,J,O-Z)
      SAVE

      LOGICAL LEX
      CHARACTER NAME0*(*),NAME1*(*),NAME0E*(*),NAME1E*(*)
      CHARACTER*80 ERRTXT
      DIMENSION CI(0:24,0:1),CK(0:16,0:1)

      PARAMETER (NAME0 = 'JMBESK0/JMDBESK0', NAME0E =
     $     'JMEBESK0/JMDBESK0')
      PARAMETER (NAME1 = 'JMBESK1/JMDBESK1', NAME1E =
     $     'JMEBESK1/JMDBESK1')

      PARAMETER (EPS=1D-15)
      PARAMETER (Z1 = 1, HF = Z1/2)
      PARAMETER (PI = 3.14159 26535 89793 24D0)
      PARAMETER (CE = 0.57721 56649 01532 86D0)
      PARAMETER (PIH = PI/2, RPIH = 2/PI, RPI2 = 1/(2*PI))

      DATA CI( 0,0) /+1.00827 92054 58740 032D0/
      DATA CI( 1,0) /+0.00844 51226 24920 943D0/
      DATA CI( 2,0) /+0.00017 27006 30777 567D0/
      DATA CI( 3,0) /+0.00000 72475 91099 959D0/
      DATA CI( 4,0) /+0.00000 05135 87726 878D0/
      DATA CI( 5,0) /+0.00000 00568 16965 808D0/
      DATA CI( 6,0) /+0.00000 00085 13091 223D0/
      DATA CI( 7,0) /+0.00000 00012 38425 364D0/
      DATA CI( 8,0) /+0.00000 00000 29801 672D0/
      DATA CI( 9,0) /-0.00000 00000 78956 698D0/
      DATA CI(10,0) /-0.00000 00000 33127 128D0/
      DATA CI(11,0) /-0.00000 00000 04497 339D0/
      DATA CI(12,0) /+0.00000 00000 01799 790D0/
      DATA CI(13,0) /+0.00000 00000 00965 748D0/
      DATA CI(14,0) /+0.00000 00000 00038 604D0/
      DATA CI(15,0) /-0.00000 00000 00104 039D0/
      DATA CI(16,0) /-0.00000 00000 00023 950D0/
      DATA CI(17,0) /+0.00000 00000 00009 554D0/
      DATA CI(18,0) /+0.00000 00000 00004 443D0/
      DATA CI(19,0) /-0.00000 00000 00000 859D0/
      DATA CI(20,0) /-0.00000 00000 00000 709D0/
      DATA CI(21,0) /+0.00000 00000 00000 087D0/
      DATA CI(22,0) /+0.00000 00000 00000 112D0/
      DATA CI(23,0) /-0.00000 00000 00000 012D0/
      DATA CI(24,0) /-0.00000 00000 00000 018D0/

      DATA CI( 0,1) /+0.97580 06023 26285 926D0/
      DATA CI( 1,1) /-0.02446 74429 63276 385D0/
      DATA CI( 2,1) /-0.00027 72053 60763 829D0/
      DATA CI( 3,1) /-0.00000 97321 46728 020D0/
      DATA CI( 4,1) /-0.00000 06297 24238 640D0/
      DATA CI( 5,1) /-0.00000 00659 61142 154D0/
      DATA CI( 6,1) /-0.00000 00096 13872 919D0/
      DATA CI( 7,1) /-0.00000 00014 01140 901D0/
      DATA CI( 8,1) /-0.00000 00000 47563 167D0/
      DATA CI( 9,1) /+0.00000 00000 81530 681D0/
      DATA CI(10,1) /+0.00000 00000 35408 148D0/
      DATA CI(11,1) /+0.00000 00000 05102 564D0/
      DATA CI(12,1) /-0.00000 00000 01804 409D0/
      DATA CI(13,1) /-0.00000 00000 01023 594D0/
      DATA CI(14,1) /-0.00000 00000 00052 678D0/
      DATA CI(15,1) /+0.00000 00000 00107 094D0/
      DATA CI(16,1) /+0.00000 00000 00026 120D0/
      DATA CI(17,1) /-0.00000 00000 00009 561D0/
      DATA CI(18,1) /-0.00000 00000 00004 713D0/
      DATA CI(19,1) /+0.00000 00000 00000 829D0/
      DATA CI(20,1) /+0.00000 00000 00000 743D0/
      DATA CI(21,1) /-0.00000 00000 00000 080D0/
      DATA CI(22,1) /-0.00000 00000 00000 117D0/
      DATA CI(23,1) /+0.00000 00000 00000 011D0/
      DATA CI(24,1) /+0.00000 00000 00000 019D0/

      DATA CK( 0,0) /+0.98840 81742 30825 800D0/
      DATA CK( 1,0) /-0.01131 05046 46928 281D0/
      DATA CK( 2,0) /+0.00026 95326 12762 724D0/
      DATA CK( 3,0) /-0.00001 11066 85196 665D0/
      DATA CK( 4,0) /+0.00000 06325 75108 500D0/
      DATA CK( 5,0) /-0.00000 00450 47337 641D0/
      DATA CK( 6,0) /+0.00000 00037 92996 456D0/
      DATA CK( 7,0) /-0.00000 00003 64547 179D0/
      DATA CK( 8,0) /+0.00000 00000 39043 756D0/
      DATA CK( 9,0) /-0.00000 00000 04579 936D0/
      DATA CK(10,0) /+0.00000 00000 00580 811D0/
      DATA CK(11,0) /-0.00000 00000 00078 832D0/
      DATA CK(12,0) /+0.00000 00000 00011 360D0/
      DATA CK(13,0) /-0.00000 00000 00001 727D0/
      DATA CK(14,0) /+0.00000 00000 00000 275D0/
      DATA CK(15,0) /-0.00000 00000 00000 046D0/
      DATA CK(16,0) /+0.00000 00000 00000 008D0/

      DATA CK( 0,1) /+1.03595 08587 72358 331D0/
      DATA CK( 1,1) /+0.03546 52912 43331 114D0/
      DATA CK( 2,1) /-0.00046 84750 28166 889D0/
      DATA CK( 3,1) /+0.00001 61850 63810 053D0/
      DATA CK( 4,1) /-0.00000 08451 72048 124D0/
      DATA CK( 5,1) /+0.00000 00571 32218 103D0/
      DATA CK( 6,1) /-0.00000 00046 45554 607D0/
      DATA CK( 7,1) /+0.00000 00004 35417 339D0/
      DATA CK( 8,1) /-0.00000 00000 45757 297D0/
      DATA CK( 9,1) /+0.00000 00000 05288 133D0/
      DATA CK(10,1) /-0.00000 00000 00662 613D0/
      DATA CK(11,1) /+0.00000 00000 00089 048D0/
      DATA CK(12,1) /-0.00000 00000 00012 726D0/
      DATA CK(13,1) /+0.00000 00000 00001 921D0/
      DATA CK(14,1) /-0.00000 00000 00000 305D0/
      DATA CK(15,1) /+0.00000 00000 00000 050D0/
      DATA CK(16,1) /-0.00000 00000 00000 009D0/

      NU=0
      LEX=.FALSE.
      GO TO 6

      ENTRY JMDEBSI0(X)

      NU=0
      LEX=.TRUE.
      GO TO 6

      ENTRY JMDBESI1(X)

      NU=1
      LEX=.FALSE.
      GO TO 6

      ENTRY JMDEBSI1(X)

      NU=1
      LEX=.TRUE.

    6 V=ABS(X)
      IF(V .LT. 8) THEN
       Y=(HF*V)**2
       XL=NU+2
       A0=1
       A1=1+2*Y/((XL+1)*(XL-1))
       A2=1+Y*(4+3*Y/((XL+2)*XL))/((XL+3)*(XL-1))
       B0=1
       B1=1-Y/(XL+1)
       B2=1-Y*(1-Y/(2*(XL+2)))/(XL+3)
       W1=3+XL
       V1=3-XL
       V3=XL-1
       V2=V3+V3
       C=0
       DO 3 N = 3,30
       C0=C
       FN=N
       W1=W1+2
       W2=W1-1
       W3=W2-1
       W4=W3-1
       W5=W4-1
       W6=W5-1
       V1=V1+1
       V2=V2+1
       V3=V3+1
       U1=FN*W4
       E=V3/(U1*W3)
       U2=E*Y
       F1=1+Y*V1/(U1*W1)
       F2=(1+Y*V2/(V3*W2*W5))*U2
       F3=-Y*Y*U2/(W4*W5*W5*W6)
       A=F1*A2+F2*A1+F3*A0
       B=F1*B2+F2*B1+F3*B0
       C=A/B
       IF(ABS(C0-C) .LT. EPS*ABS(C)) GO TO 4
       A0=A1
       A1=A2
       A2=A
       B0=B1
       B1=B2
       B2=B
    3  CONTINUE
    4  H=C
       IF(NU .EQ. 1) H=HF*X*H
       IF(LEX) H=EXP(-V)*H
      ELSE
       R=1/V
       H=16*R-1
       ALFA=H+H
       B1=0
       B2=0
       DO 1 I = 24,0,-1
       B0=CI(I,NU)+ALFA*B1-B2
       B2=B1
    1  B1=B0
       H=SQRT(RPI2*R)*(B0-H*B2)
       IF(NU*X .LT. 0) H=-H
       IF(.NOT.LEX) H=EXP(V)*H
      ENDIF
      GO TO 9

      ENTRY JMDBESK0(X)

      NU=0
      LEX=.FALSE.
      GO TO 8

      ENTRY JMDEBSK0(X)

      NU=0
      LEX=.TRUE.
      GO TO 8

      ENTRY JMDBESK1(X)

      NU=1
      LEX=.FALSE.
      GO TO 8

      ENTRY JMDEBSK1(X)

      NU=1
      LEX=.TRUE.

    8 IF(X .LE. 0) THEN
       H=0
       WRITE(ERRTXT,101) X
       IF(NU .EQ. 0 .AND. .NOT.LEX) CALL JMMLPT(NAME0 ,'C313.1',ERRTXT)
       IF(NU .EQ. 0 .AND.      LEX) CALL JMMLPT(NAME0E,'C313.1',ERRTXT)
       IF(NU .EQ. 1 .AND. .NOT.LEX) CALL JMMLPT(NAME1 ,'C313.1',ERRTXT)
       IF(NU .EQ. 1 .AND.      LEX) CALL JMMLPT(NAME1E,'C313.1',ERRTXT)
      ELSEIF(X .LT. 1) THEN
       B=HF*X
       BK=-(LOG(B)+CE)
       F=BK
       P=HF
       Q=HF
       C=1
       D=B**2
       BK1=P
       DO 11 N = 1,15
       FN=N
       RFN=1/FN
       P=P*RFN
       Q=Q*RFN
       F=(F+P+Q)*RFN
       C=C*D*RFN
       G=C*(P-FN*F)
       H=C*F
       BK=BK+H
       BK1=BK1+G
       IF(BK1*H+ABS(G)*BK .LE. EPS*BK*BK1) GO TO 12
   11  CONTINUE
   12  H=BK
       IF(NU .EQ. 1) H=BK1/B
       IF(LEX) H=EXP(X)*H
      ELSEIF(X .LE. 5) THEN
       XN=4*NU**2
       A=9-XN
       B=25-XN
       C=768*X**2
       C0=48*X
       A0=1
       A1=(16*X+7+XN)/A
       A2=(C+C0*(XN+23)+XN*(XN+62)+129)/(A*B)
       B0=1
       B1=(16*X+9-XN)/A
       B2=(C+C0*B)/(A*B)+1
       C=0
       DO 24 N = 3,30
       C0=C
       FN=N
       FN2=FN+FN
       FN1=FN2-1
       FN3=FN1/(FN2-3)
       FN4=12*FN**2-(1-XN)
       FN5=16*FN1*X
       RAN=1/((FN2+1)**2-XN)
       F1=FN3*(FN4-20*FN)+FN5
       F2=28*FN-FN4-8+FN5
       F3=FN3*((FN2-5)**2-XN)
       A=(F1*A2+F2*A1+F3*A0)*RAN
       B=(F1*B2+F2*B1+F3*B0)*RAN
       C=A/B
       IF(ABS(C0-C) .LT. EPS*ABS(C)) GO TO 25
       A0=A1
       A1=A2
       A2=A
       B0=B1
       B1=B2
       B2=B
   24  CONTINUE
   25  H=C/SQRT(RPIH*X)
       IF(.NOT.LEX) H=EXP(-X)*H
      ELSE
       R=1/X
       H=10*R-1
       ALFA=H+H
       B1=0
       B2=0
       DO 23 I = 16,0,-1
       B0=CK(I,NU)+ALFA*B1-B2
       B2=B1
   23  B1=B0
       H=SQRT(PIH*R)*(B0-H*B2)
       IF(.NOT.LEX) H=EXP(-X)*H
      ENDIF
    9 CONTINUE
      JMDBESI0=H
      RETURN
  101 FORMAT(' NON-POSITIVE ARGUMENT X = ',1P,E15.6)
      END
*
* $Id: jmdbs4.F 9 2005-07-29 16:15:46Z jmb $
*
* $Log$
* Revision 1.1  2005/07/29 16:15:46  jmb
* Include the various CERNLIB functions jimmy needs
*
* Revision 1.1.1.1  1996/04/01 15:02:03  mclareni
* Mathlib gen
*
*
      FUNCTION JMDBS4(X,NU)
      IMPLICIT DOUBLE PRECISION (A-H,J,O-Z)
      SAVE
      CHARACTER*80 ERRTXT
      CHARACTER NAMEI*(*),NAMEK*(*),NAMEIE*(*),NAMEKE*(*)
      PARAMETER (NAMEI = 'BSIR4/JMDBS4', NAMEIE = 'EBSIR4/DEBIR4')
      PARAMETER (NAMEK = 'BSKR4/DBSKR4', NAMEKE = 'EBSKR4/DEBKR4')

      LOGICAL LEX

      DIMENSION BC(0:23,2),CC(0:15,2),PP(-3:3),GG(-3:3)

      PARAMETER (EPS = 1D-14)
      PARAMETER (Z1 = 1, HF =Z1/2)
      PARAMETER (PI = 3.14159 26535 89793 24D0)
      PARAMETER (W2 = 1.41421 35623 73095 05D0)
      PARAMETER (G1 = 3.62560 99082 21908 31D0)
      PARAMETER (G3 = 1.22541 67024 65177 65D0)
      PARAMETER (PIH = PI/2, RPIH = 2/PI, RPI = 1/PI, RW2 = 1/W2)
      PARAMETER (C1 = PI/(2*W2))
      PARAMETER (GM = 2*(1/G3-4/G1), GP = (4/G1+1/G3)/2)

      DATA GG(-3) /0.27581 56628 30209 31D0/, PP(-3) /-0.75D0/
      DATA GG(-1) /0.81604 89390 98262 98D0/, PP(-1) /-0.25D0/
      DATA GG( 1) /1.10326 26513 20837 26D0/, PP( 1) / 0.25D0/
      DATA GG( 3) /1.08806 52521 31017 31D0/, PP( 3) / 0.75D0/

      DATA BC( 0,1) / 1.00619 92270 14122 57D0/
      DATA BC( 1,1) / 0.00631 99620 31140 72D0/
      DATA BC( 2,1) / 0.00012 56131 27965 64D0/
      DATA BC( 3,1) / 0.00000 52052 40761 57D0/
      DATA BC( 4,1) / 0.00000 03591 84411 39D0/
      DATA BC( 5,1) / 0.00000 00355 85362 89D0/
      DATA BC( 6,1) / 0.00000 00036 05011 66D0/
      DATA BC( 7,1) /-0.00000 00001 26294 10D0/
      DATA BC( 8,1) /-0.00000 00002 96595 12D0/
      DATA BC( 9,1) /-0.00000 00001 18337 70D0/
      DATA BC(10,1) /-0.00000 00000 21655 68D0/
      DATA BC(11,1) / 0.00000 00000 03032 04D0/
      DATA BC(12,1) / 0.00000 00000 03041 10D0/
      DATA BC(13,1) / 0.00000 00000 00530 77D0/
      DATA BC(14,1) /-0.00000 00000 00204 53D0/
      DATA BC(15,1) /-0.00000 00000 00105 49D0/
      DATA BC(16,1) / 0.00000 00000 00005 50D0/
      DATA BC(17,1) / 0.00000 00000 00014 36D0/
      DATA BC(18,1) / 0.00000 00000 00001 14D0/
      DATA BC(19,1) /-0.00000 00000 00001 87D0/
      DATA BC(20,1) /-0.00000 00000 00000 32D0/
      DATA BC(21,1) / 0.00000 00000 00000 26D0/
      DATA BC(22,1) / 0.00000 00000 00000 06D0/
      DATA BC(23,1) /-0.00000 00000 00000 04D0/

      DATA BC( 0,2) / 0.98980 19115 24008 91D0/
      DATA BC( 1,2) /-0.01035 09365 14827 02D0/
      DATA BC( 2,2) /-0.00015 85263 84973 08D0/
      DATA BC( 3,2) /-0.00000 60527 21962 69D0/
      DATA BC( 4,2) /-0.00000 04158 38597 31D0/
      DATA BC( 5,2) /-0.00000 00487 99346 57D0/
      DATA BC( 6,2) /-0.00000 00089 86835 44D0/
      DATA BC( 7,2) /-0.00000 00019 83283 58D0/
      DATA BC( 8,2) /-0.00000 00003 58969 60D0/
      DATA BC( 9,2) /-0.00000 00000 08766 62D0/
      DATA BC(10,2) / 0.00000 00000 25819 45D0/
      DATA BC(11,2) / 0.00000 00000 09780 24D0/
      DATA BC(12,2) / 0.00000 00000 00565 05D0/
      DATA BC(13,2) /-0.00000 00000 00851 66D0/
      DATA BC(14,2) /-0.00000 00000 00270 25D0/
      DATA BC(15,2) / 0.00000 00000 00040 96D0/
      DATA BC(16,2) / 0.00000 00000 00040 50D0/
      DATA BC(17,2) / 0.00000 00000 00001 11D0/
      DATA BC(18,2) /-0.00000 00000 00005 25D0/
      DATA BC(19,2) /-0.00000 00000 00000 70D0/
      DATA BC(20,2) / 0.00000 00000 00000 70D0/
      DATA BC(21,2) / 0.00000 00000 00000 14D0/
      DATA BC(22,2) /-0.00000 00000 00000 10D0/
      DATA BC(23,2) /-0.00000 00000 00000 02D0/

      DATA CC( 0,1) / 0.99128 81656 75147 07D0/
      DATA CC( 1,1) /-0.00850 62567 20022 24D0/
      DATA CC( 2,1) / 0.00019 70491 57408 35D0/
      DATA CC( 3,1) /-0.00000 80377 10166 54D0/
      DATA CC( 4,1) / 0.00000 04554 01498 43D0/
      DATA CC( 5,1) /-0.00000 00323 27352 82D0/
      DATA CC( 6,1) / 0.00000 00027 16130 28D0/
      DATA CC( 7,1) /-0.00000 00002 60644 07D0/
      DATA CC( 8,1) / 0.00000 00000 27882 69D0/
      DATA CC( 9,1) /-0.00000 00000 03267 69D0/
      DATA CC(10,1) / 0.00000 00000 00414 09D0/
      DATA CC(11,1) /-0.00000 00000 00056 17D0/
      DATA CC(12,1) / 0.00000 00000 00008 09D0/
      DATA CC(13,1) /-0.00000 00000 00001 23D0/
      DATA CC(14,1) / 0.00000 00000 00000 20D0/
      DATA CC(15,1) /-0.00000 00000 00000 03D0/

      DATA CC( 0,2) / 1.01476 24350 64637 87D0/
      DATA CC( 1,2) / 0.01449 34617 87809 66D0/
      DATA CC( 2,2) /-0.00025 87162 07241 80D0/
      DATA CC( 3,2) / 0.00000 96912 18911 49D0/
      DATA CC( 4,2) /-0.00000 05261 29313 99D0/
      DATA CC( 5,2) / 0.00000 00363 96854 29D0/
      DATA CC( 6,2) /-0.00000 00030 05472 76D0/
      DATA CC( 7,2) / 0.00000 00002 84827 80D0/
      DATA CC( 8,2) /-0.00000 00000 30182 91D0/
      DATA CC( 9,2) / 0.00000 00000 03511 10D0/
      DATA CC(10,2) /-0.00000 00000 00442 27D0/
      DATA CC(11,2) / 0.00000 00000 00059 70D0/
      DATA CC(12,2) /-0.00000 00000 00008 56D0/
      DATA CC(13,2) / 0.00000 00000 00001 30D0/
      DATA CC(14,2) /-0.00000 00000 00000 21D0/
      DATA CC(15,2) / 0.00000 00000 00000 03D0/

      LEX=.FALSE.
      GO TO 8

      ENTRY JMDEB4(X,NU)

      LEX=.TRUE.

    8 MU=ABS(NU)
      IF(MU .NE. 1 .AND. MU .NE. 2 .AND. MU .NE. 3 .OR.
     1   NU .LT. 0 .AND. X .LE. 0 .OR. NU .GT. 0 .AND. X .LT. 0) THEN
       S=0
       WRITE(ERRTXT,101) X,NU
       IF(.NOT.LEX) CALL JMMLPT(NAMEI ,'C327.1',ERRTXT)
       IF(     LEX) CALL JMMLPT(NAMEIE,'C327.1',ERRTXT)
      ELSEIF(X .EQ. 0) THEN
       S=0
      ELSEIF(NU .EQ. -2) THEN
       IF(LEX) THEN
        S=HF*(1+EXP(-X-X))/SQRT(PIH*X)
       ELSE
        S=COSH(X)/SQRT(PIH*X)
       ENDIF
      ELSEIF(NU .EQ. 2) THEN
       IF(LEX) THEN
        IF(X .LT. HF) THEN
         S=SINH(X)*EXP(-X)/SQRT(PIH*X)
        ELSE
         S=HF*(1-EXP(-X-X))/SQRT(PIH*X)
        ENDIF
       ELSE
        S=SINH(X)/SQRT(PIH*X)
       ENDIF
      ELSEIF(X .LT. 8) THEN
       Y=(HF*X)**2
       XN=PP(NU)
       XL=XN+2
       A0=1
       A1=1+2*Y/((XL+1)*(XN+1))
       A2=1+Y*(4+3*Y/((XL+2)*(XN+2)))/((XL+3)*(XN+1))
       B0=1
       B1=1-Y/(XL+1)
       B2=1-Y*(1-Y/(2*(XL+2)))/(XL+3)
       T1=3+XL
       V1=3-XL
       V3=XL-1
       V2=V3+V3
       C=0
       DO 33 N = 3,30
       C0=C
       T1=T1+2
       T2=T1-1
       T3=T2-1
       T4=T3-1
       T5=T4-1
       T6=T5-1
       V1=V1+1
       V2=V2+1
       V3=V3+1
       U1=N*T4
       E=V3/(U1*T3)
       U2=E*Y
       F1=1+Y*V1/(U1*T1)
       F2=(1+Y*V2/(V3*T2*T5))*U2
       F3=-Y*Y*U2/(T4*T5*T5*T6)
       A=F1*A2+F2*A1+F3*A0
       B=F1*B2+F2*B1+F3*B0
       C=A/B
       IF(ABS(C0-C) .LT. EPS*ABS(C)) GO TO 34
       A0=A1
       A1=A2
       A2=A
       B0=B1
       B1=B2
       B2=B
   33  CONTINUE
   34  S=GG(NU)*(HF*X)**PP(NU)*C
       IF(LEX) S=EXP(-X)*S
      ELSE
       K=(MU+1)/2
       R=1/X
       W=SQRT(RPI*R)
       H=16*R-1
       ALFA=H+H
       B1=0
       B2=0
       DO 2 I = 23,0,-1
       B0=BC(I,K)+ALFA*B1-B2
       B2=B1
    2  B1=B0
       S=RW2*W*(B0-H*B2)
       IF(.NOT.LEX) S=EXP(X)*S
       T=0
       IF(NU .LT. 0) THEN
        H=10*R-1
        ALFA=H+H
        B1=0
        B2=0
        DO 3 I = 15,0,-1
        B0=CC(I,K)+ALFA*B1-B2
        B2=B1
    3   B1=B0
        R=EXP(-X)
        T=W*R*(B0-H*B2)
        IF(LEX) T=R*T
       ENDIF
       S=S+T
      ENDIF
      GO TO 99

      ENTRY JMDBSKR4(X,NU)

      LEX=.FALSE.
      GO TO 9

      ENTRY JMDEBKR4(X,NU)

      LEX=.TRUE.

    9 MU=ABS(NU)
      IF(MU .NE. 1 .AND. MU .NE. 2 .AND. MU .NE. 3 .OR. X .LE. 0) THEN
       S=0
       WRITE(ERRTXT,101) X,NU
       IF(.NOT.LEX) CALL JMMLPT(NAMEK ,'C327.1',ERRTXT)
       IF(     LEX) CALL JMMLPT(NAMEKE,'C327.1',ERRTXT)
      ELSEIF(MU .EQ. 2) THEN
       S=SQRT(PIH/X)
       IF(.NOT.LEX) S=EXP(-X)*S
      ELSEIF(X .LE. 1) THEN
       A0=PP(-1)
       B=HF*X
       D=-LOG(B)
       F=A0*D
       E=EXP(F)
       G=(GM*A0+GP)*E
       BK=C1*(HF*GM*(E+1/E)+GP*D*SINH(F)/F)
       F=BK
       E=A0**2
       P=HF*C1*G
       Q=HF/G
       C=1
       D=B**2
       BK1=P
       DO 11 N = 1,15
       FN=N
       F=(FN*F+P+Q)/(FN**2-E)
       C=C*D/FN
       P=P/(FN-A0)
       Q=Q/(FN+A0)
       G=C*(P-FN*F)
       H=C*F
       BK=BK+H
       BK1=BK1+G
       IF(H*BK1+ABS(G)*BK .LE. EPS*BK*BK1) GO TO 12
   11  CONTINUE
   12  S=BK
       IF(MU .EQ. 3) S=BK1/B
       IF(LEX) S=EXP(X)*S
      ELSEIF(X .LE. 5) THEN
       XN=4*PP(MU)**2
       A=9-XN
       B=25-XN
       C=768*X**2
       C0=48*X
       A0=1
       A1=(16*X+7+XN)/A
       A2=(C+C0*(XN+23)+XN*(XN+62)+129)/(A*B)
       B0=1
       B1=(16*X+9-XN)/A
       B2=(C+C0*B)/(A*B)+1
       C=0
       DO 24 N = 3,30
       C0=C
       FN=N
       FN2=FN+FN
       FN1=FN2-1
       FN3=FN1/(FN2-3)
       FN4=12*FN**2-(1-XN)
       FN5=16*FN1*X
       RAN=1/((FN2+1)**2-XN)
       F1=FN3*(FN4-20*FN)+FN5
       F2=28*FN-FN4-8+FN5
       F3=FN3*((FN2-5)**2-XN)
       A=(F1*A2+F2*A1+F3*A0)*RAN
       B=(F1*B2+F2*B1+F3*B0)*RAN
       C=A/B
       IF(ABS(C0-C) .LT. EPS*ABS(C)) GO TO 25
       A0=A1
       A1=A2
       A2=A
       B0=B1
       B1=B2
       B2=B
   24  CONTINUE
   25  S=C/SQRT(RPIH*X)
       IF(.NOT.LEX) S=EXP(-X)*S
      ELSE
       K=(MU+1)/2
       R=1/X
       Y=5*R
       H=10*R-1
       ALFA=H+H
       B1=0
       B2=0
       DO 13 I = 15,0,-1
       B0=CC(I,K)+ALFA*B1-B2
       B2=B1
   13  B1=B0
       S=SQRT(PIH*R)*(B0-H*B2)
       IF(.NOT.LEX) S=EXP(-X)*S
      ENDIF
   99 JMDBS4=S

      RETURN
  101 FORMAT('INCORRECT ARGUMENT OR INDEX, X = ',1P,E15.6,5X,'NU = ',I5)
      END
*
* $Id: jmdbsir3.F 9 2005-07-29 16:15:46Z jmb $
*
* $Log$
* Revision 1.1  2005/07/29 16:15:46  jmb
* Include the various CERNLIB functions jimmy needs
*
* Revision 1.1.1.1  1996/04/01 15:02:07  mclareni
* Mathlib gen
*
*
      FUNCTION JMDBSIR3(X,NU)
      IMPLICIT DOUBLE PRECISION (A-H,J,O-Z)
      SAVE
      CHARACTER*80 ERRTXT
      CHARACTER NAMEI*(*),NAMEK*(*),NAMEIE*(*),NAMEKE*(*)
      PARAMETER (NAMEI = 'JMBSIR3/JMDBSIR3', NAMEIE =
     $     'JMEBSIR3/JMDEBIR3')
      PARAMETER(NAMEK = 'JMBSKR3/DBSKR3', NAMEKE = 'JMEBSKR3/JMDEBKR3')

      LOGICAL LEX

      DIMENSION BC(0:23,2),CC(0:15,2),PP(-2:2),GG(-2:2)

      PARAMETER (EPS = 1D-15)
      PARAMETER (Z1 = 1, HF = Z1/2)
      PARAMETER (PI = 3.14159 26535 89793 24D0)
      PARAMETER (W3 = 1.73205 08075 68877 29D0)
      PARAMETER (G1 = 2.67893 85347 07747 63D0)
      PARAMETER (G2 = 1.35411 79394 26400 42D0)
      PARAMETER (PIH = PI/2, RPIH = 2/PI, RPI2 = 1/(2*PI))
      PARAMETER (C1 = 2*PI/(3*W3))
      PARAMETER (GM = 3*(1/G2-3/G1)/2, GP = (3/G1+1/G2)/2)

      DATA PP(-2) /-0.66666 66666 66666 67D0/
      DATA PP(-1) /-0.33333 33333 33333 33D0/
      DATA PP( 1) / 0.33333 33333 33333 33D0/
      DATA PP( 2) / 0.66666 66666 66666 67D0/

      DATA GG(-2) / 0.37328 21739 07395 23D0/
      DATA GG(-1) / 0.73848 81116 21648 31D0/
      DATA GG( 1) / 1.11984 65217 22185 68D0/
      DATA GG( 2) / 1.10773 21674 32472 47D0/

      DATA BC( 0,1) / 1.00458 61710 93207 35D0/
      DATA BC( 1,1) / 0.00467 34791 99873 60D0/
      DATA BC( 2,1) / 0.00009 08034 04815 04D0/
      DATA BC( 3,1) / 0.00000 37262 16110 59D0/
      DATA BC( 4,1) / 0.00000 02520 73237 90D0/
      DATA BC( 5,1) / 0.00000 00227 82110 77D0/
      DATA BC( 6,1) / 0.00000 00012 91332 28D0/
      DATA BC( 7,1) /-0.00000 00006 11915 16D0/
      DATA BC( 8,1) /-0.00000 00003 75616 85D0/
      DATA BC( 9,1) /-0.00000 00001 16415 46D0/
      DATA BC(10,1) /-0.00000 00000 14443 25D0/
      DATA BC(11,1) / 0.00000 00000 05373 69D0/
      DATA BC(12,1) / 0.00000 00000 03074 27D0/
      DATA BC(13,1) / 0.00000 00000 00297 66D0/
      DATA BC(14,1) /-0.00000 00000 00265 20D0/
      DATA BC(15,1) /-0.00000 00000 00091 37D0/
      DATA BC(16,1) / 0.00000 00000 00015 52D0/
      DATA BC(17,1) / 0.00000 00000 00014 12D0/
      DATA BC(18,1) /-0.00000 00000 00000 23D0/
      DATA BC(19,1) /-0.00000 00000 00001 98D0/
      DATA BC(20,1) /-0.00000 00000 00000 13D0/
      DATA BC(21,1) / 0.00000 00000 00000 29D0/
      DATA BC(22,1) / 0.00000 00000 00000 03D0/
      DATA BC(23,1) /-0.00000 00000 00000 05D0/

      DATA BC( 0,2) / 0.99363 49867 16925 14D0/
      DATA BC( 1,2) /-0.00646 71526 00616 03D0/
      DATA BC( 2,2) /-0.00010 60188 22351 55D0/
      DATA BC( 3,2) /-0.00000 41406 57716 24D0/
      DATA BC( 4,2) /-0.00000 02916 95418 21D0/
      DATA BC( 5,2) /-0.00000 00365 71574 33D0/
      DATA BC( 6,2) /-0.00000 00075 81590 37D0/
      DATA BC( 7,2) /-0.00000 00019 23008 52D0/
      DATA BC( 8,2) /-0.00000 00004 20438 80D0/
      DATA BC( 9,2) /-0.00000 00000 39372 04D0/
      DATA BC(10,2) / 0.00000 00000 19007 44D0/
      DATA BC(11,2) / 0.00000 00000 10137 64D0/
      DATA BC(12,2) / 0.00000 00000 01331 30D0/
      DATA BC(13,2) /-0.00000 00000 00676 92D0/
      DATA BC(14,2) /-0.00000 00000 00311 72D0/
      DATA BC(15,2) / 0.00000 00000 00011 87D0/
      DATA BC(16,2) / 0.00000 00000 00040 21D0/
      DATA BC(17,2) / 0.00000 00000 00004 78D0/
      DATA BC(18,2) /-0.00000 00000 00004 74D0/
      DATA BC(19,2) /-0.00000 00000 00001 16D0/
      DATA BC(20,2) / 0.00000 00000 00000 59D0/
      DATA BC(21,2) / 0.00000 00000 00000 21D0/
      DATA BC(22,2) /-0.00000 00000 00000 08D0/
      DATA BC(23,2) /-0.00000 00000 00000 03D0/

      DATA CC( 0,1) / 0.99353 64122 76093 39D0/
      DATA CC( 1,1) /-0.00631 44392 60798 63D0/
      DATA CC( 2,1) / 0.00014 30095 80961 13D0/
      DATA CC( 3,1) /-0.00000 57870 60592 03D0/
      DATA CC( 4,1) / 0.00000 03265 50333 20D0/
      DATA CC( 5,1) /-0.00000 00231 23231 95D0/
      DATA CC( 6,1) / 0.00000 00019 39555 14D0/
      DATA CC( 7,1) /-0.00000 00001 85897 89D0/
      DATA CC( 8,1) / 0.00000 00000 19868 42D0/
      DATA CC( 9,1) /-0.00000 00000 02326 79D0/
      DATA CC(10,1) / 0.00000 00000 00294 68D0/
      DATA CC(11,1) /-0.00000 00000 00039 95D0/
      DATA CC(12,1) / 0.00000 00000 00005 75D0/
      DATA CC(13,1) /-0.00000 00000 00000 87D0/
      DATA CC(14,1) / 0.00000 00000 00000 14D0/
      DATA CC(15,1) /-0.00000 00000 00000 02D0/

      DATA CC( 0,2) / 1.00914 95380 72789 40D0/
      DATA CC( 1,2) / 0.00897 12068 42483 60D0/
      DATA CC( 2,2) /-0.00017 13895 98261 54D0/
      DATA CC( 3,2) / 0.00000 65547 92549 82D0/
      DATA CC( 4,2) /-0.00000 03595 19190 49D0/
      DATA CC( 5,2) / 0.00000 00250 24412 19D0/
      DATA CC( 6,2) /-0.00000 00020 74924 13D0/
      DATA CC( 7,2) / 0.00000 00001 97223 67D0/
      DATA CC( 8,2) /-0.00000 00000 20946 47D0/
      DATA CC( 9,2) / 0.00000 00000 02440 93D0/
      DATA CC(10,2) /-0.00000 00000 00307 91D0/
      DATA CC(11,2) / 0.00000 00000 00041 61D0/
      DATA CC(12,2) /-0.00000 00000 00005 97D0/
      DATA CC(13,2) / 0.00000 00000 00000 91D0/
      DATA CC(14,2) /-0.00000 00000 00000 14D0/
      DATA CC(15,2) / 0.00000 00000 00000 02D0/

      LEX=.FALSE.
      GO TO 8

      ENTRY JMDEBIR3(X,NU)

      LEX=.TRUE.

    8 MU=ABS(NU)
      IF(MU .NE. 1 .AND. MU .NE. 2 .OR. NU .LT. 0 .AND. X .LE. 0
     1   .OR. NU .GT. 0 .AND. X .LT. 0) THEN
       S=0
       WRITE(ERRTXT,101) X,NU
       IF(.NOT.LEX) CALL JMMLPT(NAMEI ,'C340.1',ERRTXT)
       IF(     LEX) CALL JMMLPT(NAMEIE,'C340.1',ERRTXT)
      ELSEIF(X .EQ. 0) THEN
       S=0
      ELSEIF(X .LT. 8) THEN
       Y=(HF*X)**2
       XN=PP(NU)
       XL=XN+2
       A0=1
       A1=1+2*Y/((XL+1)*(XN+1))
       A2=1+Y*(4+3*Y/((XL+2)*(XN+2)))/((XL+3)*(XN+1))
       B0=1
       B1=1-Y/(XL+1)
       B2=1-Y*(1-Y/(2*(XL+2)))/(XL+3)
       T1=3+XL
       V1=3-XL
       V3=XL-1
       V2=V3+V3
       C=0
       DO 33 N = 3,30
       C0=C
       T1=T1+2
       T2=T1-1
       T3=T2-1
       T4=T3-1
       T5=T4-1
       T6=T5-1
       V1=V1+1
       V2=V2+1
       V3=V3+1
       U1=N*T4
       E=V3/(U1*T3)
       U2=E*Y
       F1=1+Y*V1/(U1*T1)
       F2=(1+Y*V2/(V3*T2*T5))*U2
       F3=-Y*Y*U2/(T4*T5*T5*T6)
       A=F1*A2+F2*A1+F3*A0
       B=F1*B2+F2*B1+F3*B0
       C=A/B
       IF(ABS(C0-C) .LT. EPS*ABS(C)) GO TO 34
       A0=A1
       A1=A2
       A2=A
       B0=B1
       B1=B2
       B2=B
   33  CONTINUE
   34  S=GG(NU)*(HF*X)**PP(NU)*C
       IF(LEX) S=EXP(-X)*S
      ELSE
       R=1/X
       W=SQRT(RPI2*R)
       H=16*R-1
       ALFA=H+H
       B1=0
       B2=0
       DO 2 I = 23,0,-1
       B0=BC(I,MU)+ALFA*B1-B2
       B2=B1
    2  B1=B0
       S=W*(B0-H*B2)
       IF(.NOT.LEX) S=EXP(X)*S
       T=0
       IF(NU .LT. 0) THEN
        H=10*R-1
        ALFA=H+H
        B1=0
        B2=0
        DO 3 I = 15,0,-1
        B0=CC(I,MU)+ALFA*B1-B2
        B2=B1
    3   B1=B0
        R=EXP(-X)
        T=W3*W*R*(B0-H*B2)
        IF(LEX) T=R*T
       END IF
       S=S+T
      END IF
      GO TO 99

      ENTRY JMDBSKR3(X,NU)

      LEX=.FALSE.
      GO TO 9

      ENTRY JMDEBKR3(X,NU)

      LEX=.TRUE.

    9 MU=ABS(NU)
      IF(MU .NE. 1 .AND. MU .NE. 2 .OR. X .LE. 0) THEN
       S=0
       WRITE(ERRTXT,101) X,NU
       IF(.NOT.LEX) CALL JMMLPT(NAMEK ,'C340.1',ERRTXT)
       IF(     LEX) CALL JMMLPT(NAMEKE,'C340.1',ERRTXT)
      ELSEIF(X .LE. 1) THEN
       A0=PP(-1)
       B=HF*X
       D=-LOG(B)
       F=A0*D
       E=EXP(F)
       G=(GM*A0+GP)*E
       BK=C1*(HF*GM*(E+1/E)+GP*D*SINH(F)/F)
       F=BK
       E=A0**2
       P=HF*C1*G
       Q=HF/G
       C=1
       D=B**2
       BK1=P
       DO 11 N = 1,15
       FN=N
       F=(FN*F+P+Q)/(FN**2-E)
       C=C*D/FN
       P=P/(FN-A0)
       Q=Q/(FN+A0)
       G=C*(P-FN*F)
       H=C*F
       BK=BK+H
       BK1=BK1+G
       IF(H*BK1+ABS(G)*BK .LE. EPS*BK*BK1) GO TO 12
   11  CONTINUE
   12  S=BK
       IF(MU .EQ. 2) S=BK1/B
       IF(LEX) S=EXP(X)*S
      ELSEIF(X .LE. 5) THEN
       XN=4*PP(MU)**2
       A=9-XN
       B=25-XN
       C=768*X**2
       C0=48*X
       A0=1
       A1=(16*X+7+XN)/A
       A2=(C+C0*(XN+23)+XN*(XN+62)+129)/(A*B)
       B0=1
       B1=(16*X+9-XN)/A
       B2=(C+C0*B)/(A*B)+1
       C=0
       DO 24 N = 3,30
       C0=C
       FN=N
       FN2=FN+FN
       FN1=FN2-1
       FN3=FN1/(FN2-3)
       FN4=12*FN**2-(1-XN)
       FN5=16*FN1*X
       RAN=1/((FN2+1)**2-XN)
       F1=FN3*(FN4-20*FN)+FN5
       F2=28*FN-FN4-8+FN5
       F3=FN3*((FN2-5)**2-XN)
       A=(F1*A2+F2*A1+F3*A0)*RAN
       B=(F1*B2+F2*B1+F3*B0)*RAN
       C=A/B
       IF(ABS(C0-C) .LT. EPS*ABS(C)) GO TO 25
       A0=A1
       A1=A2
       A2=A
       B0=B1
       B1=B2
       B2=B
   24  CONTINUE
   25  S=C/SQRT(RPIH*X)
       IF(.NOT.LEX) S=EXP(-X)*S
      ELSE
       R=1/X
       H=10*R-1
       ALFA=H+H
       B1=0
       B2=0
       DO 13 I = 15,0,-1
       B0=CC(I,MU)+ALFA*B1-B2
       B2=B1
   13  B1=B0
       S=SQRT(PIH*R)*(B0-H*B2)
       IF(.NOT.LEX) S=EXP(-X)*S
      END IF

   99 JMDBSIR3=S

      RETURN
  101 FORMAT('INCORRECT ARGUMENT OR INDEX, X = ',1P,E15.6,' NU = ',I5)
      END
*
* $Id: jmdbska.F 22 2006-11-19 13:14:22Z jmb $
*
* $Log$
* Revision 1.1  2005/07/29 16:15:46  jmb
* Include the various CERNLIB functions jimmy needs
*
* Revision 1.1.1.1  1996/04/01 15:02:08  mclareni
* Mathlib gen
*
*
      SUBROUTINE JMDBSKA(X,IA,JA,NL,B)
      IMPLICIT DOUBLE PRECISION (A-H,J,O-Z)
      SAVE
      LOGICAL LEX
      INTEGER JA

      CHARACTER NAME*(*),ENAM*(*)
      CHARACTER*80 ERRTXT
      PARAMETER (NAME = 'BSKA/JMDBSKA', ENAM = 'EBSKA/DEBKA')
 
      PARAMETER (Z1 = 1, Z2 = 2, Z3 = 3, Z4 = 4)
      PARAMETER (Z12 = Z1/Z2, Z13 = Z1/Z3, Z14 = Z1/Z4, Z23 = Z2/Z3)
      PARAMETER (Z34 = Z3/Z4)
 
      DIMENSION B(0:*)
 
      PARAMETER (PI = 3.14159 26535 89793D0, PIV = PI/4)
 
      LEX=.FALSE.
      GO TO 9
 
      ENTRY JMDEBKA(X,IA,JA,NL,B)
      LEX=.TRUE.
 
    9 MODE=10*IA+JA
      N=NL-1
      U=2/X
      IF(LEX) THEN
       IF(X .LE. 0) THEN
        N=0
        WRITE(ERRTXT,101) X
        CALL JMMLPT(ENAM,'C341.1',ERRTXT)
       ELSEIF(NL .LT. 0 .OR. NL .GT. 100) THEN
        N=0
        WRITE(ERRTXT,103) NL
        CALL JMMLPT(ENAM,'C341.3',ERRTXT)
       ELSEIF(IA .EQ. 0) THEN
        A=0
        B(0)=JMDEBSK0(X)
        B(1)=JMDEBSK1(X)
       ELSEIF(MODE .EQ. 12) THEN
        A=Z12
        B(0)=SQRT(PIV*U)
        B(1)=B(0)*(1+A*U)
       ELSEIF(MODE .EQ. 13) THEN
        A=Z13
        B(0)=JMDEBKR3(X,1)
        B(1)=JMDEBKR3(X,2)+A*U*B(0)
       ELSEIF(MODE .EQ. 14) THEN
        A=Z14
        B(0)=JMDEBKR4(X,1)
        B(1)=JMDEBKR4(X,3)+A*U*B(0)
       ELSEIF(MODE .EQ. 23) THEN
        A=Z23
        B(0)=JMDEBKR3(X,2)
        B(1)=JMDEBKR3(X,1)+A*U*B(0)
       ELSEIF(MODE .EQ. 34) THEN
        A=Z34
        B(0)=JMDEBKR4(X,3)
        B(1)=JMDEBKR4(X,1)+A*U*B(0)
       ELSE
        N=0
        WRITE(ERRTXT,102) IA,JA
        CALL JMMLPT(ENAM,'C341.2',ERRTXT)
       ENDIF
      ELSE
       IF(X .LE. 0) THEN
        N=0
        WRITE(ERRTXT,101) X
        CALL JMMLPT(NAME,'C341.1',ERRTXT)
       ELSEIF(NL .LT. 0 .OR. NL .GT. 100) THEN
        N=0
        WRITE(ERRTXT,103) NL
        CALL JMMLPT(NAME,'C341.3',ERRTXT)
       ELSEIF(IA .EQ. 0) THEN
        A=0
        B(0)=JMDBESK0(X)
        B(1)=JMDBESK1(X)
       ELSEIF(MODE .EQ. 12) THEN
        A=Z12
        B(0)=EXP(-X)*SQRT(PIV*U)
        B(1)=B(0)*(1+A*U)
       ELSEIF(MODE .EQ. 13) THEN
        A=Z13
        B(0)=JMDBSKR3(X,1)
        B(1)=JMDBSKR3(X,2)+A*U*B(0)
       ELSEIF(MODE .EQ. 14) THEN
        A=Z14
        B(0)=JMDBSKR4(X,1)
        B(1)=JMDBSKR4(X,3)+A*U*B(0)
       ELSEIF(MODE .EQ. 23) THEN
        A=Z23
        B(0)=JMDBSKR3(X,2)
        B(1)=JMDBSKR3(X,1)+A*U*B(0)
       ELSEIF(MODE .EQ. 34) THEN
        A=Z34
        B(0)=JMDBSKR4(X,3)
        B(1)=JMDBSKR4(X,1)+A*U*B(0)
       ELSE
        N=0
        WRITE(ERRTXT,102) IA,JA
        CALL JMMLPT(NAME,'C341.2',ERRTXT)
       ENDIF
      ENDIF
      DO 1 IJ = 1,N
      A=A+1
    1 B(IJ+1)=B(IJ-1)+A*U*B(IJ)
      RETURN
  101 FORMAT('NON-POSITIVE ARGUMENT  X = ',E15.6)
  102 FORMAT('PAIR (IA,JA) = (',I5,I5,')  ILLEGAL')
  103 FORMAT('ILLEGAL  NL =',I5)
      END
      FUNCTION JMLNOC (CHV)
C
C CERN PROGLIB# M507    LENOCC          .VERSION KERNFOR  4.21  890323
C ORIG. March 85, A.Petrilli, re-write 21/02/89, JZ
C
C-    Find last non-blank character in CHV
 
      CHARACTER    CHV*(*)
 
      N = LEN(CHV)
 
      DO 17  JJ= N,1,-1
      IF (CHV(JJ:JJ).NE.' ') GO TO 99
 17    CONTINUE
      JJ = 0
 
 99    JMLNOC = JJ
      RETURN
      END
*
* $Id: jmmlpt.F 9 2005-07-29 16:15:46Z jmb $
*
* $Log$
* Revision 1.1  2005/07/29 16:15:46  jmb
* Include the various CERNLIB functions jimmy needs
*
* Revision 1.1.1.1  1996/04/01 15:02:52  mclareni
* Mathlib gen
*
      SUBROUTINE JMMLPT(NAME,ERC,TEXT)
      CHARACTER*(*) NAME,ERC,TEXT
      LOGICAL LMF,LRF
      SAVE

      IF(ERC(5:6).NE.'.0') THEN
        CALL JMMLMR(ERC,MLG,LMF,LRF)
      ELSE
        LMF=.TRUE.
        LRF=.FALSE.
      ENDIF
      IF(LMF) THEN
        LT=JMLNOC(TEXT)
        IF(MLG .LT. 1) WRITE(  *,100) ERC(1:4),NAME,ERC,TEXT(1:LT)
        IF(MLG .GE. 1) WRITE(MLG,100) ERC(1:4),NAME,ERC,TEXT(1:LT)
      ENDIF
      IF(.NOT.LRF) CALL JMABEND
      RETURN
100   FORMAT(7X,'***** CERN ',A,1X,A,' ERROR ',A,': ',A)
      END
*
* $Id: jmmlst.F 9 2005-07-29 16:15:46Z jmb $
*
* $Log$
* Revision 1.1  2005/07/29 16:15:46  jmb
* Include the various CERNLIB functions jimmy needs
*
* Revision 1.1.1.1  1996/04/01 15:02:53  mclareni
* Mathlib gen
*
      SUBROUTINE JMMLST(ERC,NLG,MXM,MXR)

      PARAMETER (KTE = 132)
      CHARACTER*6 ERC,CODE(KTE)
      LOGICAL LMF,LRF
      DIMENSION KNTM(KTE),KNTR(KTE)

      SAVE
      DATA ILG /0/

C     renumber the data statements after putting new codes in Unix with:
C     awk -F'[()]' '{ printf"%s(%s)%s(%s)%s(%s)%s\n",$1,NR,$3,NR,$5,NR,$7 }'
C     and modify KTE to the number of lines below

      DATA CODE(1),KNTM(1),KNTR(1) / 'B100.1', 255, 255 /
      DATA CODE(2),KNTM(2),KNTR(2) / 'B300.1', 255, 255 /
      DATA CODE(3),KNTM(3),KNTR(3) / 'B300.2', 255, 255 /
      DATA CODE(4),KNTM(4),KNTR(4) / 'C200.0', 255, 255 /
      DATA CODE(5),KNTM(5),KNTR(5) / 'C200.1', 255, 255 /
      DATA CODE(6),KNTM(6),KNTR(6) / 'C200.2', 255, 255 /
      DATA CODE(7),KNTM(7),KNTR(7) / 'C200.3', 255, 255 /
      DATA CODE(8),KNTM(8),KNTR(8) / 'C201.0', 255, 255 /
      DATA CODE(9),KNTM(9),KNTR(9) / 'C202.0', 255, 255 /
      DATA CODE(10),KNTM(10),KNTR(10) / 'C202.1', 255, 255 /
      DATA CODE(11),KNTM(11),KNTR(11) / 'C202.2', 255, 255 /
      DATA CODE(12),KNTM(12),KNTR(12) / 'C205.1', 255, 255 /
      DATA CODE(13),KNTM(13),KNTR(13) / 'C205.2', 255, 255 /
      DATA CODE(14),KNTM(14),KNTR(14) / 'C207.0', 255, 255 /
      DATA CODE(15),KNTM(15),KNTR(15) / 'C208.0', 255, 255 /
      DATA CODE(16),KNTM(16),KNTR(16) / 'C209.0', 255, 255 /
      DATA CODE(17),KNTM(17),KNTR(17) / 'C209.1', 255, 255 /
      DATA CODE(18),KNTM(18),KNTR(18) / 'C209.2', 255, 255 /
      DATA CODE(19),KNTM(19),KNTR(19) / 'C209.3', 255, 255 /
      DATA CODE(20),KNTM(20),KNTR(20) / 'C210.1', 255, 255 /
      DATA CODE(21),KNTM(21),KNTR(21) / 'C302.1', 255, 255 /
      DATA CODE(22),KNTM(22),KNTR(22) / 'C303.1', 255, 255 /
      DATA CODE(23),KNTM(23),KNTR(23) / 'C304.1', 255, 255 /
      DATA CODE(24),KNTM(24),KNTR(24) / 'C305.1', 255, 255 /
      DATA CODE(25),KNTM(25),KNTR(25) / 'C306.1', 255, 255 /
      DATA CODE(26),KNTM(26),KNTR(26) / 'C307.1', 255, 255 /
      DATA CODE(27),KNTM(27),KNTR(27) / 'C312.1', 255, 255 /
      DATA CODE(28),KNTM(28),KNTR(28) / 'C313.1', 255, 255 /
      DATA CODE(29),KNTM(29),KNTR(29) / 'C315.1', 255, 255 /
      DATA CODE(30),KNTM(30),KNTR(30) / 'C316.1', 255, 255 /
      DATA CODE(31),KNTM(31),KNTR(31) / 'C316.2', 255, 255 /
      DATA CODE(32),KNTM(32),KNTR(32) / 'C320.1', 255, 255 /
      DATA CODE(33),KNTM(33),KNTR(33) / 'C321.1', 255, 255 /
      DATA CODE(34),KNTM(34),KNTR(34) / 'C323.1', 255, 255 /
      DATA CODE(35),KNTM(35),KNTR(35) / 'C327.1', 255, 255 /
      DATA CODE(36),KNTM(36),KNTR(36) / 'C328.1', 255, 255 /
      DATA CODE(37),KNTM(37),KNTR(37) / 'C328.2', 255, 255 /
      DATA CODE(38),KNTM(38),KNTR(38) / 'C328.3', 255, 255 /
      DATA CODE(39),KNTM(39),KNTR(39) / 'C330.1', 255, 255 /
      DATA CODE(40),KNTM(40),KNTR(40) / 'C330.2', 255, 255 /
      DATA CODE(41),KNTM(41),KNTR(41) / 'C330.3', 255, 255 /
      DATA CODE(42),KNTM(42),KNTR(42) / 'C331.1', 255, 255 /
      DATA CODE(43),KNTM(43),KNTR(43) / 'C331.2', 255, 255 /
      DATA CODE(44),KNTM(44),KNTR(44) / 'C334.1', 255, 255 /
      DATA CODE(45),KNTM(45),KNTR(45) / 'C334.2', 255, 255 /
      DATA CODE(46),KNTM(46),KNTR(46) / 'C334.3', 255, 255 /
      DATA CODE(47),KNTM(47),KNTR(47) / 'C334.4', 255, 255 /
      DATA CODE(48),KNTM(48),KNTR(48) / 'C334.5', 255, 255 /
      DATA CODE(49),KNTM(49),KNTR(49) / 'C334.6', 255, 255 /
      DATA CODE(50),KNTM(50),KNTR(50) / 'C336.1', 255, 255 /
      DATA CODE(51),KNTM(51),KNTR(51) / 'C337.1', 255, 255 /
      DATA CODE(52),KNTM(52),KNTR(52) / 'C338.1', 255, 255 /
      DATA CODE(53),KNTM(53),KNTR(53) / 'C340.1', 255, 255 /
      DATA CODE(54),KNTM(54),KNTR(54) / 'C343.1', 255, 255 /
      DATA CODE(55),KNTM(55),KNTR(55) / 'C343.2', 255, 255 /
      DATA CODE(56),KNTM(56),KNTR(56) / 'C343.3', 255, 255 /
      DATA CODE(57),KNTM(57),KNTR(57) / 'C343.4', 255, 255 /
      DATA CODE(58),KNTM(58),KNTR(58) / 'C344.1', 255, 255 /
      DATA CODE(59),KNTM(59),KNTR(59) / 'C344.2', 255, 255 /
      DATA CODE(60),KNTM(60),KNTR(60) / 'C344.3', 255, 255 /
      DATA CODE(61),KNTM(61),KNTR(61) / 'C344.4', 255, 255 /
      DATA CODE(62),KNTM(62),KNTR(62) / 'C345.1', 255, 255 /
      DATA CODE(63),KNTM(63),KNTR(63) / 'C346.1', 255, 255 /
      DATA CODE(64),KNTM(64),KNTR(64) / 'C346.2', 255, 255 /
      DATA CODE(65),KNTM(65),KNTR(65) / 'C346.3', 255, 255 /
      DATA CODE(66),KNTM(66),KNTR(66) / 'C347.1', 255, 255 /
      DATA CODE(67),KNTM(67),KNTR(67) / 'C347.2', 255, 255 /
      DATA CODE(68),KNTM(68),KNTR(68) / 'C347.3', 255, 255 /
      DATA CODE(69),KNTM(69),KNTR(69) / 'C347.4', 255, 255 /
      DATA CODE(70),KNTM(70),KNTR(70) / 'C347.5', 255, 255 /
      DATA CODE(71),KNTM(71),KNTR(71) / 'C347.6', 255, 255 /
      DATA CODE(72),KNTM(72),KNTR(72) / 'C348.1', 255, 255 /
      DATA CODE(73),KNTM(73),KNTR(73) / 'C349.1', 255, 255 /
      DATA CODE(74),KNTM(74),KNTR(74) / 'C349.2', 255, 255 /
      DATA CODE(75),KNTM(75),KNTR(75) / 'C349.3', 255, 255 /
      DATA CODE(76),KNTM(76),KNTR(76) / 'D101.1', 255, 255 /
      DATA CODE(77),KNTM(77),KNTR(77) / 'D103.1', 255, 255 /
      DATA CODE(78),KNTM(78),KNTR(78) / 'D104.1', 255, 255 /
      DATA CODE(79),KNTM(79),KNTR(79) / 'D104.2', 255, 255 /
      DATA CODE(80),KNTM(80),KNTR(80) / 'D105.1', 255, 255 /
      DATA CODE(81),KNTM(81),KNTR(81) / 'D105.2', 255, 255 /
      DATA CODE(82),KNTM(82),KNTR(82) / 'D107.1', 255, 255 /
      DATA CODE(83),KNTM(83),KNTR(83) / 'D110.0', 255, 255 /
      DATA CODE(84),KNTM(84),KNTR(84) / 'D110.1', 255, 255 /
      DATA CODE(85),KNTM(85),KNTR(85) / 'D110.2', 255, 255 /
      DATA CODE(86),KNTM(86),KNTR(86) / 'D110.3', 255, 255 /
      DATA CODE(87),KNTM(87),KNTR(87) / 'D110.4', 255, 255 /
      DATA CODE(88),KNTM(88),KNTR(88) / 'D110.5', 255, 255 /
      DATA CODE(89),KNTM(89),KNTR(89) / 'D110.6', 255, 255 /
      DATA CODE(90),KNTM(90),KNTR(90) / 'D113.1', 255, 255 /
      DATA CODE(91),KNTM(91),KNTR(91) / 'D201.1', 255, 255 /
      DATA CODE(92),KNTM(92),KNTR(92) / 'D202.1', 255, 255 /
      DATA CODE(93),KNTM(93),KNTR(93) / 'D401.1', 255, 255 /
      DATA CODE(94),KNTM(94),KNTR(94) / 'D601.1', 255, 255 /
      DATA CODE(95),KNTM(95),KNTR(95) / 'E210.1', 255, 255 /
      DATA CODE(96),KNTM(96),KNTR(96) / 'E210.2', 255, 255 /
      DATA CODE(97),KNTM(97),KNTR(97) / 'E210.3', 255, 255 /
      DATA CODE(98),KNTM(98),KNTR(98) / 'E210.4', 255, 255 /
      DATA CODE(99),KNTM(99),KNTR(99) / 'E210.5', 255, 255 /
      DATA CODE(100),KNTM(100),KNTR(100) / 'E210.6', 255, 255 /
      DATA CODE(101),KNTM(101),KNTR(101) / 'E210.7', 255, 255 /
      DATA CODE(102),KNTM(102),KNTR(102) / 'E211.0', 255, 255 /
      DATA CODE(103),KNTM(103),KNTR(103) / 'E211.1', 255, 255 /
      DATA CODE(104),KNTM(104),KNTR(104) / 'E211.2', 255, 255 /
      DATA CODE(105),KNTM(105),KNTR(105) / 'E211.3', 255, 255 /
      DATA CODE(106),KNTM(106),KNTR(106) / 'E211.4', 255, 255 /
      DATA CODE(107),KNTM(107),KNTR(107) / 'E406.0', 255, 255 /
      DATA CODE(108),KNTM(108),KNTR(108) / 'E406.1', 255, 255 /
      DATA CODE(109),KNTM(109),KNTR(109) / 'E407.0', 255, 255 /
      DATA CODE(110),KNTM(110),KNTR(110) / 'E408.0', 255, 255 /
      DATA CODE(111),KNTM(111),KNTR(111) / 'E408.1', 255, 255 /
      DATA CODE(112),KNTM(112),KNTR(112) / 'F500.0', 255, 255 /
      DATA CODE(113),KNTM(113),KNTR(113) / 'F500.1', 255, 255 /
      DATA CODE(114),KNTM(114),KNTR(114) / 'F500.2', 255, 255 /
      DATA CODE(115),KNTM(115),KNTR(115) / 'F500.3', 255, 255 /
      DATA CODE(116),KNTM(116),KNTR(116) / 'G100.1', 255, 255 /
      DATA CODE(117),KNTM(117),KNTR(117) / 'G100.2', 255, 255 /
      DATA CODE(118),KNTM(118),KNTR(118) / 'G101.1', 255, 255 /
      DATA CODE(119),KNTM(119),KNTR(119) / 'G101.2', 255, 255 /
      DATA CODE(120),KNTM(120),KNTR(120) / 'G105.1', 255, 255 /
      DATA CODE(121),KNTM(121),KNTR(121) / 'G106.1', 255, 255 /
      DATA CODE(122),KNTM(122),KNTR(122) / 'G106.2', 255, 255 /
      DATA CODE(123),KNTM(123),KNTR(123) / 'G116.1', 255, 255 /
      DATA CODE(124),KNTM(124),KNTR(124) / 'G116.2', 255, 255 /
      DATA CODE(125),KNTM(125),KNTR(125) / 'H101.0', 255, 255 /
      DATA CODE(126),KNTM(126),KNTR(126) / 'H101.1', 255, 255 /
      DATA CODE(127),KNTM(127),KNTR(127) / 'H101.2', 255, 255 /
      DATA CODE(128),KNTM(128),KNTR(128) / 'H301.1', 255, 255 /
      DATA CODE(129),KNTM(129),KNTR(129) / 'U501.1', 255, 255 /
      DATA CODE(130),KNTM(130),KNTR(130) / 'V202.1', 255, 255 /
      DATA CODE(131),KNTM(131),KNTR(131) / 'V202.2', 255, 255 /
      DATA CODE(132),KNTM(132),KNTR(132) / 'V202.3', 255, 255 /


      ILG=NLG
      L=0
      IF(ERC .NE. ' ') THEN
       DO 10 L = 1,6
       IF(ERC(1:L) .EQ. ERC) GOTO 12
   10  CONTINUE
   12  CONTINUE
      ENDIF
      DO 14 I = 1,KTE
      IF(L .EQ. 0 .OR. CODE(I)(1:L) .EQ. ERC(1:L)) THEN
       IF(MXM .GE. 0) KNTM(I)=MXM
       IF(MXR .GE. 0) KNTR(I)=MXR
      ENDIF
   14 CONTINUE
      RETURN

      ENTRY JMMLMR(ERC,MLG,LMF,LRF)

      MLG=ILG
      DO 20 I = 1,KTE
      IF(ERC .EQ. CODE(I))  GOTO 21
   20 CONTINUE
      WRITE(*,100) ERC
      CALL JMABEND
      RETURN

   21 LMF=KNTM(I) .GE. 1
      LRF=KNTR(I) .GE. 1
      IF(LMF .AND. KNTM(I) .LT. 255)  KNTM(I)=KNTM(I)-1
      IF(LRF .AND. KNTR(I) .LT. 255)  KNTR(I)=KNTR(I)-1
      IF(.NOT.LRF) THEN
       IF(ILG .LT. 1) WRITE(  *,101) CODE(I)
       IF(ILG .GE. 1) WRITE(ILG,101) CODE(I)
      ENDIF
      RETURN
  100 FORMAT(7X,'***** CERN N002 JMMLST ... ERROR N002: ',
     1'ERROR CODE ',A6,' NOT RECOGNIZED BY ERROR MONITOR. RUN ABORTED.')
  101 FORMAT(7X,'***** CERN N002 JMMLST ... ERROR NOO2.1: ',
     1'RUN TERMINATED BY LIBRARY ERROR CONDITION ',A6)
      END
*
* $Id: jmucpy.F 9 2005-07-29 16:15:46Z jmb $
*
* $Log$
* Revision 1.1  2005/07/29 16:15:46  jmb
* Include the various CERNLIB functions jimmy needs
*
* Revision 1.1.1.1  1996/02/15 17:50:19  mclareni
* Kernlib
*
*
      SUBROUTINE JMUCPY (A,B,N)
C
C CERN PROGLIB# V301    UCOPY           .VERSION KERNFOR  4.40  940929
C ORIG. 01/01/65 JZ
C
      REAL*8  A(*),B(*)
C
C--                NO OVERLAP  OR  BEGINNING OF A ON END OF B
      IF (N.EQ.0) RETURN
         DO 21 I=1,N
   21 B(I)=A(I)
      RETURN
      END
*CMZ :          23/08/93  13.30.11  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE ADDBND (N,NFREE,IFREE,NACTV,IACTV,INEW,IDNEW,NADIM,AHES
     1S,XTEMP,GFREE,GNORM)
      INTEGER N, NFREE, NACTV, INEW, IDNEW, NADIM
      INTEGER IFREE(N), IACTV(N)
      DOUBLE PRECISION GNORM
      DOUBLE PRECISION AHESS(NADIM, N), XTEMP(N), GFREE(N)
      INTEGER I, INPOS, NFRM1
      DOUBLE PRECISION RNEW
      SAVE
      CALL DELETE(NFREE,IFREE,INEW,INPOS)
      IF(INPOS.EQ.0) RETURN
      NACTV=NACTV+1
      IACTV(NACTV)=IDNEW*INEW
      RNEW=GNORM*GNORM-GFREE(INPOS)**2
      IF(RNEW.LT.0.0D+0) RNEW=0.0D+0
      GNORM=SQRT(RNEW)
      CALL SHRNK(NFREE,NADIM,AHESS,INPOS,XTEMP)
      IF(INPOS.EQ.NFREE) GOTO 20
      NFRM1=NFREE-1
      DO 10 I=INPOS,NFRM1
      GFREE(I)=GFREE(I+1)
 10   CONTINUE
 20   NFREE=NFREE-1
      RETURN
      END
*CMZ :          23/08/93  13.30.11  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE BFGS (N,NADIM,AHESS,EPSMCH,G,GNEW,P,ALPHA,Y)
      INTEGER N, NADIM
      DOUBLE PRECISION EPSMCH, ALPHA
      DOUBLE PRECISION AHESS(NADIM, N), G(N), GNEW(N), P(N), Y(N)
      INTEGER I, IFAIL, IP1, J, NM1
      DOUBLE PRECISION CONST, DVDOT, GTP, YTP
      SAVE
      DO 10 I=1,N
      Y(I)=GNEW(I)-G(I)
 10   CONTINUE
      YTP=DVDOT(N,Y,P)
      IF(YTP.LE.0.0D+0) RETURN
      GTP=DVDOT(N,G,P)
      CONST=ALPHA*YTP
      IF(CONST.LT.1.0D-10) RETURN
      CONST=1.0D+0/CONST
      CALL MODCHL(N,NADIM,AHESS,CONST,Y,IFAIL)
      IF(GTP.GE.0.0D+0) RETURN
      CONST=1.0D+0/SQRT(-GTP)
      DO 20 I=1,N
      Y(I)=G(I)*CONST
 20   CONTINUE
      CALL NMDCHL(N,NADIM,AHESS,EPSMCH,Y,P)
      NM1=N-1
      IF(NM1.EQ.0) RETURN
      DO 40 I=1,NM1
      IP1=I+1
      DO 30 J=IP1,N
      AHESS(I,J)=AHESS(J,I)
 30   CONTINUE
 40   CONTINUE
      RETURN
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE BNDOPT (N,XMIN,XMAX,XLOW,XHI,FMIN,FMAX,FLOBD,FUPBD,X,XT
     1EMP,GFREE,PFREE,GACTV,GNEW,AHESS,DELTA,FTOL,GTOL,ETA,ALFMAX,MAXFUN
     2,NFCNT,IFREE,IACTV,IPRINT,IRESLT)
      INTEGER N, MAXFUN, NFCNT, IPRINT, IRESLT
      INTEGER IFREE(N), IACTV(N)
      DOUBLE PRECISION FMIN, FMAX, FLOBD, FUPBD, DELTA, FTOL, GTOL,
     1                 ETA, ALFMAX
      DOUBLE PRECISION XMIN(N), XMAX(N), XLOW(N), XHI(N), X(N),
     1                 XTEMP(N), GFREE(N), PFREE(N), GACTV(N), GNEW(N),
     2                 AHESS(N,N)
      INTEGER I, IBDEL, IBTRUE, IDNEW, II, IIMIN, ILOC, INEW, IPCNT,
     1        ITER, ITEST, J, NACTV, NADIM, NFREE
      LOGICAL LOCAL, NEGMUL, XALTER
      DOUBLE PRECISION  A, ALPHA, ALRAT, B, BTOL, B1, D, DEL, DFUN,
     1 DVDOT, DXNORM, E, EPSMCH, FA, FBD, FBEST, FDIFF,
     2                  FOLD, FRAT, FTEST, FTRUE, FU, FV, FW, FY, FZ,
     3                  GNORM, GTEST1, GTEST2, GTP, GU, OLDF, PE,
     4                  PNORM, QEPS, R, RR, RTEPS, SCXBD, SFTBND, SS,
     5                  STEPMX, T, TOL, U, XBEST, XLAMDA, XNORM, XRAT,
     6                  XV, XW, ZTOL1, ZTOL2
      DATA EPSMCH/2.22D-16/
      SAVE
      RTEPS=SQRT(EPSMCH)
      QEPS=SQRT(RTEPS)
      ZTOL1=ABS(FTOL)
      NADIM=N
      NFCNT=0
      DO 10 I=1,N
      IFREE(I)=0
      IACTV(I)=0
 10   CONTINUE
      IF(IPRINT.LT.0) GOTO 20
      WRITE(6,310)
      WRITE(6,320) (XLOW(I),I=1,N)
      WRITE(6,330) (XHI(I),I=1,N)
 20   DO 300 IIMIN=1,2
      IF(IIMIN.EQ.2) GOTO 30
      CALL DVCOPY(N,XMIN,X)
      FTRUE=FMIN
      FBEST=FTRUE
      FBD=FLOBD
      GOTO 40
 30   CALL DVCOPY(N,XMAX,X)
      FTRUE=FMAX
      FBEST=-FTRUE
      FBD=FUPBD
 40   ITER=0
      GNORM=0.0D+0
      IPCNT=0
      FOLD=FBEST
      LOCAL=.FALSE.
      DO 60 J=1,N
      DO 50 I=1,N
      AHESS(I,J)=0.0D+0
 50   CONTINUE
      AHESS(J,J)=1.0D+0
 60   CONTINUE
      BTOL=QEPS
 70   CALL BNDTST(N,X,XLOW,XHI,BTOL,NFREE,IFREE,NACTV,IACTV,XALTER)
      IF(.NOT.XALTER) GOTO 80
      FTRUE=DFUN(N,X)
      FBEST=FTRUE
      IF(IIMIN.EQ.2) FBEST=-FTRUE
      FOLD=FBEST
 80   IF(IPRINT.LT.0) GOTO 90
      IF(IIMIN.EQ.1) WRITE(6,340) FTRUE
      IF(IIMIN.EQ.2) WRITE(6,350) FTRUE
      IF(LOCAL) WRITE(6,450) NFCNT
      WRITE(6,360) (X(I),I=1,N)
      WRITE(6,370) NFREE,NACTV
      IF(NFREE.GT.0.AND.NFREE.LT.N) WRITE(6,380) (IFREE(I),I=1,NFREE)
      IF(NACTV.GT.0.AND.NACTV.LT.N) WRITE(6,390) (IACTV(I),I=1,NACTV)
 90   FTEST=RTEPS*(1.0D+0+ABS(FBD))
      IF(ABS(FBD-FBEST).LT.FTEST) GOTO 250
      IF(NFREE.EQ.0) GOTO 250
      CALL GRDCMP(N,NFREE,IFREE,X,FTRUE,DELTA,XHI,XTEMP,GFREE)
      NFCNT=NFCNT+NFREE
      IF(IIMIN.EQ.1) GOTO 110
      DO 100 I=1,NFREE
      GFREE(I)=-GFREE(I)
 100  CONTINUE
 110  CALL RLEN(NFREE,GFREE,GNORM)
      IF(GNORM.GT.GTOL.OR.LOCAL) GOTO 120
      IF(IPRINT.GT.0) WRITE(6,440)
      DEL=DELTA
      IF(DELTA.LT.1.0D+0) DEL=SQRT(DELTA)
      ZTOL2=10.0D+0*SQRT(RTEPS*DEL)
      IF(ZTOL2.LT.RTEPS) ZTOL2=RTEPS
      LOCAL=.TRUE.
      CALL LOCSCH(IIMIN,N,NFREE,IFREE,X,FBEST,XLOW,XHI,DEL,ZTOL2,NFCNT,
     1GNEW,FY,XTEMP,FZ,PFREE)
      FTEST=RTEPS*(1.0D+0+ABS(FBEST))
      IF(FZ.GE.FBEST.OR.ABS(FZ-FBEST).LT.FTEST) GOTO 250
      CALL DVCOPY(N,XTEMP,X)
      FBEST=FZ
      FTRUE=FZ
      IF(IIMIN.EQ.2) FTRUE=-FZ
      GOTO 70
 120  ITER=ITER+1
      IF(NFCNT.GT.MAXFUN) GOTO 270
      IF(IPCNT.GE.IPRINT) IPCNT=0
      IPCNT=IPCNT+1
      IF(NFREE.EQ.0) GOTO 250
      CALL LDLSOL(NFREE,NADIM,AHESS,GFREE,PFREE)
      DO 130 I=1,NFREE
      PFREE(I)=-PFREE(I)
 130  CONTINUE
      CALL RLEN(NFREE,PFREE,PNORM)
      PE=PNORM+RTEPS
      CALL FEASMV(N,NFREE,IFREE,X,PFREE,XLOW,XHI,EPSMCH,STEPMX,INEW,IDN
     1EW)
      IF((STEPMX*PE).GT.DELTA) GOTO 140
      CALL ADDBND(N,NFREE,IFREE,NACTV,IACTV,INEW,IDNEW,NADIM,AHESS,XTEM
     1P,GFREE,GNORM)
      ITER=ITER-1
      GOTO 120
 140  GTP=DVDOT(NFREE,GFREE,PFREE)
      XLAMDA=MIN(STEPMX,ALFMAX)
      U=MIN(1.0D+0,XLAMDA)
      FU=FBEST
      GU=GTP
      ILOC=1
      SFTBND=DELTA/PE
      T=RTEPS/PE
      DO 150 I=1,N
      XTEMP(I)=X(I)
 150  CONTINUE
 160  CALL NEWPTQ(RTEPS,T,ETA,SFTBND,XLAMDA,U,FU,GU,XBEST,FBEST,XW,FW,X
     1V,FV,A,FA,B,OLDF,B1,SCXBD,E,D,RR,SS,GTEST1,GTEST2,TOL,ILOC,ITEST)
      IF(ITEST.NE.1) GOTO 180
      R=XBEST+U
      DO 170 I=1,NFREE
      II=IFREE(I)
      XTEMP(II)=X(II)+R*PFREE(I)
 170  CONTINUE
      FU=DFUN(N,XTEMP)
      NFCNT=NFCNT+1
      IF(IIMIN.EQ.2) FU=-FU
      GOTO 160
 180  IF(ITEST.NE.0) GOTO 250
      ALPHA=XBEST
      DO 190 I=1,NFREE
      II=IFREE(I)
      X(II)=X(II)+XBEST*PFREE(I)
 190  CONTINUE
      FTRUE=FBEST
      IF(IIMIN.EQ.2) FTRUE=-FBEST
      IF(IPRINT.LE.0.OR.IPCNT.LT.IPRINT) GOTO 200
      WRITE(6,400) ITER,NFCNT,FBEST
      WRITE(6,360) (X(I),I=1,N)
      WRITE(6,370) NFREE,NACTV
      IF(NFREE.GT.0.AND.NFREE.LT.N) WRITE(6,380) (IFREE(I),I=1,NFREE)
      IF(NACTV.GT.0.AND.NACTV.LT.N) WRITE(6,390) (IACTV(I),I=1,NACTV)
 200  CALL GRDCMP(N,NFREE,IFREE,X,FTRUE,DELTA,XHI,XTEMP,GNEW)
      CALL RLEN(NFREE,GNEW,GNORM)
      IF(IPRINT.GT.0.AND.IPCNT.EQ.IPRINT) WRITE(6,460) GNORM
      IF(IIMIN.EQ.1) GOTO 220
      DO 210 I=1,NFREE
      GNEW(I)=-GNEW(I)
 210  CONTINUE
 220  NFCNT=NFCNT+NFREE
      CALL BFGS(NFREE,NADIM,AHESS,EPSMCH,GFREE,GNEW,PFREE,ALPHA,XTEMP)
      DO 230 I=1,NFREE
      GFREE(I)=GNEW(I)
 230  CONTINUE
      ALRAT=ABS(ALPHA-STEPMX)/STEPMX
      IF(ALRAT.GE.QEPS) GOTO 240
      CALL ADDBND(N,NFREE,IFREE,NACTV,IACTV,INEW,IDNEW,NADIM,AHESS,XTEM
     1P,GFREE,GNORM)
      GOTO 120
 240  FDIFF=FOLD-FBEST
      FRAT=FDIFF/(1.0D+0+ABS(FBEST))
      IF(FTOL.LT.0.0D+0) FRAT=FDIFF/MAX(ABS(FBEST),EPSMCH)
      CALL RLEN(N,X,XNORM)
      DXNORM=ALPHA*PNORM
      XRAT=DXNORM/(1.0D+0+XNORM)
      IF(FTOL.LT.0.0D+0) XRAT=DXNORM/MAX(XNORM,EPSMCH)
      FOLD=FBEST
      IF(FRAT.GT.ZTOL1.OR.XRAT.GT.ZTOL1) GOTO 120
 250  IF(NACTV.EQ.0) GOTO 270
      CALL MULCHK(N,NACTV,IACTV,IIMIN,RTEPS,X,XHI,FTRUE,DELTA,XTEMP,GAC
     1TV,NEGMUL,IBDEL,IBTRUE)
      NFCNT=NFCNT+NACTV
      IF(IPRINT.LE.0.OR.IPCNT.LT.IPRINT) GOTO 260
      WRITE(6,420) ITER,NACTV
      WRITE(6,390) (IACTV(I),I=1,NACTV)
      WRITE(6,430) (GACTV(I),I=1,NACTV)
 260  IF(.NOT.NEGMUL) GOTO 270
      CALL DELBND(N,NACTV,IACTV,NFREE,IFREE,IBDEL,IBTRUE,GACTV,NADIM,AH
     1ESS,GFREE,GNORM)
      GOTO 120
 270  IF(IPRINT.LT.0) GOTO 280
      WRITE(6,410) ITER,NFCNT,FTRUE
      WRITE(6,360) (X(I),I=1,N)
      WRITE(6,370) NFREE,NACTV
      IF(NFREE.GT.0.AND.NFREE.LT.N) WRITE(6,380) (IFREE(I),I=1,NFREE)
      IF(NACTV.GT.0.AND.NACTV.LT.N) WRITE(6,390) (IACTV(I),I=1,NACTV)
 280  IF(IIMIN.EQ.2) GOTO 290
      FMIN=FBEST
      CALL DVCOPY(N,X,XMIN)
      GOTO 300
 290  FMAX=-FBEST
      CALL DVCOPY(N,X,XMAX)
 300  CONTINUE
      RETURN
 310  FORMAT('-START OF OPTIMIZATION')
 320  FORMAT(' LOWER BOUNDS', 6(1PD16.6))
 330  FORMAT(' UPPER BOUNDS', 6(1PD16.6))
 340  FORMAT(' MINIMIZATION STEP -- INITIAL FMIN =', 1PD16.6)
 350  FORMAT('0MAXIMIZATION STEP -- INITIAL FMAX =', 1PD16.6)
 360  FORMAT(' X ARRAY'/6(1PD16.6))
 370  FORMAT(1X,I5,' FREE VARIABLES',I5,' FIXED VARIABLES')
 380  FORMAT(' INDICES OF FREE  VARIABLES'/10I8)
 390  FORMAT(' INDICES OF FIXED VARIABLES'/10I8)
 400  FORMAT('0 AFTER ITERATION',I5,' AND',I5,
     1  ' FUNCTION EVALUATIONS, THE FUNCTION VALUE IS',1PD16.6)
 410  FORMAT('0*** FINAL RESULT ***'/' AFTER',I5,
     1    ' ITERATIONS AND', I6,' FUNCTION EVALUATIONS',
     2      ' THE BEST FUNCTION VALUE IS', 1PD17.7)
 420  FORMAT('0 AT ITERATION',I6,' CHECK MULTIPLIERS',
     1    ' FOR THE',I5,' FIXED VARIABLES')
 430  FORMAT(' GRADIENT WITH RESPECT TO FIXED VARIABLES'/
     1             1X,6(1PD16.6))
 440  FORMAT(' EXECUTE LOCAL SEARCH')
 450  FORMAT(1X,I5,' FUNCTION EVALUATIONS AFTER LOCAL SEARCH')
 460  FORMAT(' NORM OF PROJECTED GRADIENT =', 1PD16.6)
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE BNDTST (N,X,XLOW,XHI,BNDTOL,NFREE,IFREE,NACTV,IACTV,XAL
     1TER)
      INTEGER N, NFREE, NACTV
      INTEGER IFREE(N), IACTV(N)
      LOGICAL XALTER
      DOUBLE PRECISION BNDTOL
      DOUBLE PRECISION X(N), XLOW(N), XHI(N)
      INTEGER I
CMM   DOUBLE PRECISION DABS, XDENOM
      DOUBLE PRECISION       XDENOM
      SAVE
      XALTER=.FALSE.
      NFREE=0
      NACTV=0
      DO 30 I=1,N
      XDENOM=1.0D+0+ABS(XHI(I))
      IF((X(I)-XHI(I))/XDENOM.GE.(-BNDTOL)) GOTO 10
      XDENOM=1.0D+0+ABS(XLOW(I))
      IF((X(I)-XLOW(I))/XDENOM.LE.BNDTOL) GOTO 20
      NFREE=NFREE+1
      IFREE(NFREE)=I
      GOTO 30
 10   X(I)=XHI(I)
      XALTER=.TRUE.
      NACTV=NACTV+1
      IACTV(NACTV)=-I
      GOTO 30
 20   X(I)=XLOW(I)
      XALTER=.TRUE.
      NACTV=NACTV+1
      IACTV(NACTV)=I
 30   CONTINUE
      RETURN
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE BOUNDS (BUCKET,PARENT,TREE,PARTN,UMINUS,UPLUS)
      INTEGER BUCKET,PARENT,TREE(4,1000),JTREE
      REAL PARTN(18001),UMINUS(10),UPLUS(10)
      SAVE
 10   JTREE=TREE(4,PARENT)
      IF((BUCKET.GT.TREE(1,PARENT).OR.JTREE.GE.0).AND.(BUCKET.LE.TREE(1
     1,PARENT).OR.JTREE.LE.0)) GOTO 30
      UPLUS(ABS(JTREE))=PARTN(PARENT)
      IF(TREE(2,PARENT).GE.0) GOTO 20
      IF(BUCKET.NE.-TREE(2,PARENT)) GOTO 60
      GOTO 50
 20   PARENT=TREE(2,PARENT)
      GOTO 10
 30   UMINUS(ABS(JTREE))=PARTN(PARENT)
      IF(TREE(3,PARENT).GE.0) GOTO 40
      IF(BUCKET.NE.-TREE(3,PARENT)) GOTO 80
      GOTO 50
 40   PARENT=TREE(3,PARENT)
      GOTO 10
 50   RETURN
 60   WRITE(6,70) BUCKET,TREE(2,PARENT)
 70   FORMAT(' LOOKING FOR LEFT BUCKET ',I5,'  BUT FOUND ',I6)
      STOP
 80   WRITE(6,90) BUCKET,TREE(3,PARENT)
 90   FORMAT(' LOOKING FOR RIGHT BUCKET ',I5,' BUT FOUND ',I6)
      STOP
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE BUCMVE (NEWSTR,NEWEND,OLDSTR,SCR,BUCKTS,ISCR,IBUC)
      INTEGER NEWSTR,NEWEND,OLDSTR
      REAL SCR(ISCR),BUCKTS(IBUC)
      COMMON /BUKSZE/ MAXWRD
      INTEGER MAXWRD
      SAVE
      DO 70 I=1,MAXWRD
      DO 10 J=NEWSTR,NEWEND
      SCR(J)=BUCKTS(I+MAXWRD*(OLDSTR+J-NEWSTR-1))
 10   CONTINUE
      J=NEWEND+1
      GOTO 30
 20   J=J+1
 30   IF((J).GT.(OLDSTR+NEWEND-NEWSTR-1)) GOTO 40
      SCR(J)=BUCKTS(I+MAXWRD*(NEWSTR+J-NEWEND-1))
      GOTO 20
 40   J=NEWSTR
      GOTO 60
 50   J=J+1
 60   IF((J).GT.(OLDSTR+NEWEND-NEWSTR-1)) GOTO 70
      BUCKTS(I+MAXWRD*(J-1))=SCR(J)
      GOTO 50
 70   CONTINUE
      RETURN
      END
*CMZ :          04/09/93  16.53.11  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE BUFOPT (N,RXMIN,RXMAX,RXLOW,RXHI,RFMIN,RFMAX,RFLOBD,RFU
     1PBD,W,LENW,IW,LENIW,NFCNT,IRESLT)
      INTEGER N, LENW, LENIW, NFCNT, IRESLT
      INTEGER IW(LENIW)
      REAL RXMIN(N), RXMAX(N), RXLOW(N), RXHI(N), RFMIN, RFMAX,
     1     RFLOBD, RFUPBD
      DOUBLE PRECISION W(LENW)
      INTEGER I, IPRINT, IXHI, IXLOW, IXMAX, JAHESS, JGACTV, JGFREE,
     1        JGNEW, JIACTV, JIFREE, JPFREE, JX, JXHI, JXLOW, JXMAX,
     2        JXMIN, JXTEMP, MAXFUN
      REAL RFDIF, SNGL
      DOUBLE PRECISION ALFMAX, DELTA, ETA, FLOBD, FMAX, FMIN,
     1   FTOL, FUPBD, GTOL
      SAVE
      JXMIN=1
      JXMAX=JXMIN+N
      JXLOW=JXMAX+N
      JXHI=JXLOW+N
      DO 10 I=1,N
      W(I)=DBLE(RXMIN(I))
      IXMAX=JXMAX-1+I
      W(IXMAX)=DBLE(RXMAX(I))
      IXLOW=JXLOW-1+I
      W(IXLOW)=DBLE(RXLOW(I))
      IXHI=JXHI-1+I
      W(IXHI)=DBLE(RXHI(I))
 10   CONTINUE
      FMAX=DBLE(RFMAX)
      FMIN=DBLE(RFMIN)
      FLOBD=DBLE(RFLOBD)
      FUPBD=DBLE(RFUPBD)
      CALL SETTOL(N,FTOL,GTOL,DELTA,ETA,ALFMAX,MAXFUN,IPRINT)
      IF(IPRINT.LT.0) GOTO 20
      WRITE(6,70) (RXLOW(I),I=1,N)
      WRITE(6,80) (RXHI(I),I=1,N)
 20   JX=JXHI+N
      JXTEMP=JX+N
      JGFREE=JXTEMP+N
      JPFREE=JGFREE+N
      JGACTV=JPFREE+N
      JGNEW=JGACTV+N
      JAHESS=JGNEW+N
      JIFREE=1
      JIACTV=JIFREE+N
      CALL BNDOPT(N,W(JXMIN),W(JXMAX),W(JXLOW),W(JXHI),FMIN,FMAX,FLOBD,
     1FUPBD,W(JX),W(JXTEMP),W(JGFREE),W(JPFREE),W(JGACTV),W(JGNEW),W(JAH
     2ESS),DELTA,FTOL,GTOL,ETA,ALFMAX,MAXFUN,NFCNT,IW(JIFREE),IW(JIACTV)
     3,IPRINT,IRESLT)
      DO 30 I=1,N
      RXMIN(I)=SNGL(W(I))
      IXMAX=N+I
      RXMAX(I)=SNGL(W(IXMAX))
 30   CONTINUE
** JMB      RFMIN=SNGL(FMIN)
******      RFMAX=SNGL(FMAX)
      IF (ABS(FMIN).GT.1.E-36) THEN
        RFMIN=SNGL(FMIN)
      ELSE
        RFMIN=0.0
      ENDIF
      IF (ABS(FMAX).GT.1.E-36) THEN
        RFMAX=SNGL(FMAX)
      ELSE
        RFMAX=0.0
      ENDIF
**
      RFDIF=RFMAX-RFMIN
      IF(IPRINT.LT.0) RETURN
      WRITE(6,40) RFDIF,RFMAX,RFMIN,NFCNT
      WRITE(6,50) (RXMIN(I),I=1,N)
      WRITE(6,60) (RXMAX(I),I=1,N)
      RETURN
 40   FORMAT('0 END OF OPTIMIZATION -- FDIF =', 1PE15.5, 4X,
     1  'FMAX =',1PE15.5,4X,'FMIN =',1PE15.5,4X,'NFCNT =',I7)
 50   FORMAT(' XMIN',5(1PE15.5))
 60   FORMAT(' XMAX',5(1PE15.5))
 70   FORMAT(' LOWER BOUNDS',5(1PE15.5))
 80   FORMAT(' UPPER BOUNDS',5(1PE15.5))
      END
*CMZ :          23/08/93  13.30.11  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE DELBND (N,NACTV,IACTV,NFREE,IFREE,IBDEL,IBTRUE,GACTV,NA
     1DIM,AHESS,GFREE,GNORM)
      INTEGER N, NACTV, NFREE, IBDEL, IBTRUE, NADIM
      INTEGER IACTV(N), IFREE(N)
      DOUBLE PRECISION GNORM
      DOUBLE PRECISION AHESS(NADIM, N), GFREE(N), GACTV(N)
      INTEGER IPOS, J
      DOUBLE PRECISION RVAL
      SAVE
      CALL DELETE(NACTV,IACTV,IBTRUE,IPOS)
      NACTV=NACTV-1
      IBTRUE=ABS(IBTRUE)
      IFREE(NFREE+1)=IBTRUE
      IF(NFREE.EQ.0) GOTO 20
      DO 10 J=1,NFREE
      AHESS(NFREE+1,J)=0.0D+0
 10   CONTINUE
 20   NFREE=NFREE+1
      AHESS(NFREE,NFREE)=1.0D+0
      GFREE(NFREE)=GACTV(IBDEL)
      RVAL=GNORM*GNORM+GFREE(NFREE)**2
      GNORM=SQRT(RVAL)
      RETURN
      END
*CMZ :          23/08/93  13.30.11  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE DELETE (NLEN,IARRAY,IVALUE,IPOS)
      INTEGER NLEN, IVALUE, IPOS
      INTEGER IARRAY(NLEN)
      INTEGER I, NLENM1
      SAVE
      IPOS=0
      DO 10 I=1,NLEN
      IF(IVALUE.NE.IARRAY(I)) GOTO 10
      IPOS=I
      GOTO 20
 10   CONTINUE
      RETURN
 20   IF(IPOS.EQ.NLEN) RETURN
      NLENM1=NLEN-1
      DO 30 I=IPOS,NLENM1
      IARRAY(I)=IARRAY(I+1)
 30   CONTINUE
      RETURN
      END
*CMZ :          26/11/93  11.31.36  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE DELSLV(N,FMAJOR,FMINOR,LMAX,FRACT,X,XLOW,XUP,VOL,NCUT,N
     1CDIM,ICUT,DELPLS,DELNEG,REGTOL,FTOL,FORIG,DFORIG,FNEW,FNLIN,FNLROW
     2,DIAGJ,SPDIAG,SOL,Z,NFCNT)
      INTEGER N, NCUT, NCDIM, NFCNT
      INTEGER ICUT(NCDIM)
      DOUBLE PRECISION FMAJOR, FMINOR, FRACT, REGTOL, FTOL
      DOUBLE PRECISION X(N), XUP(N), XLOW(N), DELPLS(N), DELNEG(N)
      DOUBLE PRECISION FORIG(NCDIM), FNLIN(NCDIM)
      DOUBLE PRECISION DFORIG(NCDIM), FNEW(NCDIM), FNLROW(NCDIM)
      DOUBLE PRECISION DIAGJ(NCDIM), SPDIAG(NCDIM), SOL(NCDIM), Z(N)
      LOGICAL LMAX
      INTEGER I, IAB, IBACK, II, ISAVE, ITRY,
     1        NCUTM1, NEAR
      DOUBLE PRECISION BIG, DEL, DELMAX, DELMIN, DFNEW, DFUN,
     1        FDIF, FGAM, FNORM, FNRMNW, FOMX, FRAT, FZ, GAMMA,
     2        GAMNEW, PROD, RATGAM, REGINV, SINGTL, TSTVAL,
     3        VAL, VOL, XMULT, YDI, YDMIN
CMM   INTEGER IABS
      DATA SINGTL/ 1.0D-4/
      DATA BIG/ 1.0D+10/  
      SAVE
*      write(*,*) 'SOL',sol
*      write(*,*) 'FNLIN',fnlin
      NFCNT=0
      FDIF=FMAJOR-FMINOR
      PROD=1.0D+0
      DO 10 I=1,N
C ** JMB
      IF (ABS(DELPLS(I)).LT.1.D-16) DELPLS(I)=0.D0
      IF (ABS(DELNEG(I)).LT.1.D-16) DELNEG(I)=0.D0
C -- JMB
      PROD=PROD*(DELPLS(I)+DELNEG(I))
 10   CONTINUE
C ** JMB
      IF (VOL.NE.0.D0) THEN
        GAMMA=PROD/VOL
        FGAM=GAMMA*FMAJOR+(1.0D+0-GAMMA)*FMINOR
      ELSE
        GAMMA=1.0D0
        FGAM = 0.0D0
      ENDIF
C -- JMB
      DO 20 I=1,N
      Z(I)=X(I)
 20   CONTINUE
      DO 50 I=1,NCUT
      II=ICUT(I)
      IAB=ABS(II)
      IF(II.LT.0) GOTO 30
      Z(IAB)=X(IAB)+DELPLS(IAB)
      GOTO 40
 30   Z(IAB)=X(IAB)-DELNEG(IAB)
 40   FNEW(I)=DFUN(N,Z)
      Z(IAB)=X(IAB)
      NFCNT=NFCNT+1
 50   CONTINUE
   60 YDMIN= 1.0D+30
      NEAR=0
      DO 70 I=1,NCUT
      YDI=ABS(FMAJOR-FNEW(I))
      IF(YDI.GT.YDMIN) GOTO 70
      YDMIN=YDI
      NEAR=I
 70   CONTINUE
      IF(NEAR.EQ.0) RETURN
      IF((LMAX.AND.FNEW(NEAR).LT.FGAM).OR.
     1(.NOT.LMAX.AND.FNEW(NEAR).GT.FGAM)) GOTO 130
      ISAVE=ICUT(NEAR)
      IF(NEAR.EQ.NCUT.OR.NCUT.EQ.1) GOTO 90
      NCUTM1=NCUT-1
      DO 80 I=NEAR,NCUTM1
      ICUT(I)=ICUT(I+1)
      FNEW(I)=FNEW(I+1)
 80   CONTINUE
 90   NCUT=NCUT-1
      IF(ISAVE.LT.0) GOTO 100
      DELPLS(ISAVE)=XUP(ISAVE)-X(ISAVE)
*      write(*,*) 'A:delpls(isave),isave,xup(isave),x(isave)'
*     &     ,delpls(isave),isave,xup(isave),x(isave)
      GOTO 110
 100  IAB=ABS(ISAVE)
      DELNEG(IAB)=X(IAB)-XLOW(IAB)
*      write(*,*) 'delneg(iab),iab,x(iab),xlow(iab)',delneg(iab),iab
*     &     ,x(iab),xlow(iab)
 110  PROD=1.0D+0
      DO 120 I=1,N
C ** JMB
      IF (ABS(DELPLS(I)).LT.1.D-16) DELPLS(I)=0.D0
      IF (ABS(DELNEG(I)).LT.1.D-16) DELNEG(I)=0.D0
C -- JMB
      PROD=PROD*(DELPLS(I)+DELNEG(I))
 120  CONTINUE
C ** JMB
      IF (VOL.NE.0.D0) THEN
        GAMMA=PROD/VOL
        FGAM=GAMMA*FMAJOR+(1.0D+0-GAMMA)*FMINOR
      ELSE
        GAMMA=1.0D0
        FGAM = 0.0D0
      ENDIF
C -- JMB
      IF(NCUT.EQ.0) RETURN
      GOTO 60
 130  DO 160 I=1,NCUT
      II=ICUT(I)
      IAB=ABS(II)
      IF(II.LT.0) GOTO 140
      DEL=DELPLS(IAB)
      GOTO 150
 140  DEL=DELNEG(IAB)
 150  FORIG(I)=FNEW(I)
      IF (DEL.NE.0.D0) THEN
         DFORIG(I)=(FNEW(I)-FMAJOR)/DEL
      ELSE
         DFORIG(I)=0.D0         
      ENDIF
 160  CONTINUE
      CALL FEQN(NCUT,FORIG,FGAM,FNLIN)
      CALL RLEN(NCUT,FNLIN,FNORM)
 170  DO 180 I=1,NCUT
      FNLIN(I)=-FNLIN(I)
 180  CONTINUE
      IF(NCUT.EQ.1) GOTO 200
      DIAGJ(1)=DFORIG(1)
      SPDIAG(1)=-DFORIG(2)
      NCUTM1=NCUT-1
      DO 190 I=1,NCUTM1
      DIAGJ(I)=DFORIG(I)
      SPDIAG(I)=-DFORIG(I+1)
 190  CONTINUE
 200  DO 210 I=1,NCUT
      II=ICUT(I)
      II=ABS(II)
      IF (GAMMA.NE.0) THEN
        IF (ABS(DELPLS(II)+DELNEG(II)).NE.0.D0) THEN
           FNLROW(I)=-GAMMA*FDIF/(DELPLS(II)+DELNEG(II))
        ELSE
*           write(*,440) 'DELPLS(II)+DELNEG(II)='
*           write(*,*) DELPLS(II)+DELNEG(II)
           FNLROW(I)=0
        ENDIF
      ELSE
        FNLROW(I)=0.0
      ENDIF
 210  CONTINUE
      FNLROW(1)=DFORIG(1)+FNLROW(1)
      IF(NCUT.EQ.1) GOTO 230
      DO 220 I=1,NCUTM1
      XMULT=0.0D+0
      IF (ABS(FNLROW(I)).LT.BIG*ABS(DIAGJ(I))) THEN
         IF (DIAGJ(I).NE.0.D0) THEN
            XMULT=FNLROW(I)/DIAGJ(I)
         ELSE
*            write(*,440) 'diag(j)=0!'
            XMULT=0.D0
         ENDIF
      ENDIF
      FNLROW(I+1)=FNLROW(I+1)-XMULT*SPDIAG(I)
      FNLIN(NCUT)=FNLIN(NCUT)-XMULT*FNLIN(I)
 220  CONTINUE
 230  SOL(NCUT)=FNLIN(NCUT)
      IF (ABS(FNLROW(NCUT)).LT.BIG*ABS(FNLIN(NCUT))) THEN
         IF (FNLROW(NCUT).NE.0.D0) THEN
            SOL(NCUT)=FNLIN(NCUT)/FNLROW(NCUT)
         ELSE
            SOL(NCUT)=0.D0
         ENDIF
      ENDIF
*      write(*,*) 'FNLIN(NCUT),ncut,fnlrow(ncut)',FNLIN(NCUT),ncut
*     &     ,fnlrow(ncut)
      IF(NCUT.EQ.1) GOTO 250
      DO 240 I=2,NCUT
      IBACK=NCUT-I+1
      VAL=FNLIN(IBACK)-SOL(IBACK+1)*SPDIAG(IBACK)
      SOL(IBACK)=VAL
*     write(*,*) 'sol(iback),iback,val',sol(iback),iback,val
      IF(ABS(DIAGJ(IBACK)).LT.BIG*ABS(VAL)) THEN 
         IF (DIAGJ(IBACK).NE.0.D0) THEN
            SOL(IBACK)=VAL/DIAGJ(IBACK)
         ELSE
            SOL(IBACK)=0.D0
         ENDIF
      ENDIF
 240  CONTINUE
 250  ITRY=0
      DO 280 I=1,NCUT
      II=ICUT(I)
      IAB=ABS(II)
      IF(II.LT.0) GOTO 260
      DELMAX=FRACT*(XUP(IAB)-X(IAB)-DELPLS(IAB))
      DELMIN=-DELPLS(IAB)
      GOTO 270
 260  DELMAX=FRACT*(X(IAB)-XLOW(IAB)-DELNEG(IAB))
      DELMIN=-DELNEG(IAB)
 270  IF(SOL(I).GT.DELMAX) SOL(I)=0.75D+0*DELMAX
      IF(SOL(I).LT.DELMIN) SOL(I)=0.75D+0*DELMIN
 280  CONTINUE
 290  DO 310 I=1,NCUT
      II=ICUT(I)
      IAB=ABS(II)
      IF(II.LT.0) GOTO 300
      DELPLS(IAB)=DELPLS(IAB)+SOL(I)
*      write(*,*) 'A:delpls(iab),iab,sol(i),i,delmax,delmin,',delpls(iab)
*     &     ,iab,sol(i),i,delmax,delmin
      GOTO 310
 300  DELNEG(IAB)=DELNEG(IAB)+SOL(I)
*      write(*,*) 'A:delneg(iab),iab,sol(i),i',delneg(iab),iab
*     &    ,sol(i),i
 310  CONTINUE
      DO 340 I=1,NCUT
      II=ICUT(I)
      IAB=ABS(II)
      IF(II.LT.0) GOTO 320
      Z(IAB)=X(IAB)+DELPLS(IAB)
      GOTO 330
 320  Z(IAB)=X(IAB)-DELNEG(IAB)
 330  CONTINUE
*      write(*,*) 'iab,x(iab),z(iab),delpls,delneg',iab,x(iab),z(iab)
*     &     ,delpls,delneg
      FZ=DFUN(N,Z)
      NFCNT=NFCNT+1
      FNEW(I)=FZ
      Z(IAB)=X(IAB)
 340  CONTINUE
      PROD=1.0D+0
      DO 350 I=1,N
C ** JMB
      IF (ABS(DELPLS(I)).LT.1.D-16) DELPLS(I)=0.D0
      IF (ABS(DELNEG(I)).LT.1.D-16) DELNEG(I)=0.D0
C -- JMB
      PROD=PROD*(DELPLS(I)+DELNEG(I))
 350  CONTINUE
      IF (VOL.NE.0.D0) THEN
         GAMNEW=PROD/VOL
      ELSE
         GAMNEW=0.D0
      ENDIF
      FGAM=GAMNEW*FMAJOR+(1.0D+0-GAMNEW)*FMINOR
      CALL FEQN(NCUT,FNEW,FGAM,FNLIN)
      CALL RLEN(NCUT,FNLIN,FNRMNW)
      IF(FNRMNW.GT.FNORM) GOTO 380
      FOMX=0.0D+0
      DO 360 I=1,NCUT
      IF(ABS(FNEW(I)).GT.FOMX) FOMX=ABS(FNEW(I))
      DFNEW=FNEW(I)-FORIG(I)
      TSTVAL=1.0D+0
      IF (ABS(SOL(I)).LT.BIG*ABS(DFNEW)) THEN
         IF (SOL(I).NE.0.D0) THEN
            TSTVAL=DFNEW/SOL(I)
         ELSE
            TSTVAL=0.D0
         ENDIF
      ENDIF
      IF(ABS(TSTVAL).LT.SINGTL*ABS(DFORIG(I))) TSTVAL=SINGTL*DFORIG(I
     1)
      DFORIG(I)=TSTVAL
      FORIG(I)=FNEW(I)
 360  CONTINUE
      FNORM=FNRMNW
      FOMX=MAX(FOMX,ABS(FGAM))
      FRAT=FNORM/(1.0D+0+FOMX)
      REGINV=1.0D+0/REGTOL
      RATGAM=GAMNEW/GAMMA
      GAMMA=GAMNEW
C---  Activate to do debugging
C     WRITE(6,420) FNORM,FOMX,FRAT
C     WRITE(6,430) FGAM,RATGAM
      FGAM=GAMMA*FMAJOR+(1.0D+0-GAMMA)*FMINOR
      IF(RATGAM.GT.REGTOL.AND.RATGAM.LT.REGINV) GOTO 370
      IF(FRAT.LT.FTOL) GOTO 370
      GOTO 170
 370  RETURN
 380  ITRY=ITRY+1
      IF(ITRY.GT.2) RETURN
      DO 410 I=1,NCUT
      II=ICUT(I)
      IAB=ABS(II)
      IF(II.GE.0) THEN
         DELPLS(IAB)=DELPLS(IAB)-SOL(I)
*         write(*,*) 'B:delpls(iab),iab,sol(i),i',delpls(iab),iab
*     &        ,sol(i),i
      ELSE
         DELNEG(IAB)=DELNEG(IAB)-SOL(I)
*         write(*,*) 'B:delneg(iab),iab,sol(i),i',delneg(iab),iab
*     &        ,sol(i),i
      ENDIF
      SOL(I)=SOL(I)*0.25D+0
 410  CONTINUE
      GOTO 290
 420  FORMAT(' FNORM, FOMX, FRAT', 3(1PD15.5))
 430  FORMAT(' FGAM, RATGAM', 2(1PD15.5))
 440  FORMAT(A)
      END


*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE DIVON (D,UMINUS,UPLUS,ERROR,NMAX,FUN1,ERRINT)
      INTEGER D,NMAX
      REAL UMINUS(D),UPLUS(D),ERROR
      COMMON /ANSWER/ FINTGL,SPRD,DUMMY(5),NRGN,MAXRGN /FUNN/ NFUN,MO(2)
      COMMON /Z0001/ ERR,NMIN,MCOUNT /SAMPLE/ NPT /PRSTOP/ NSTOP
      COMMON /D151DT/ IDATE /PRINT/ IPRINT /ZEETRM/ ITRMF
      REAL*8 IDATE
      DATA FAC /1.0/
      SAVE
C         INITIALISATION OF CONSTANTS
      CALL DVNBKD
C
      ITRMF=0
      IF(NMAX.GT.0) GOTO 30
      IF(IPRINT.LE.0) GOTO 20
      WRITE(6,10) NMAX
 10   FORMAT('0--- DIVON --- ',I5,' FUNCTION EVALUATIONS SPECIFIED.')
 20   RETURN
 30   IF(IPRINT.LE.0) GOTO 80
      WRITE(6,40) IDATE,D,ERROR,NMAX
 40   FORMAT('1--- DIVON --- DIVONNE4 (',A8,')  MULTIPLE INTEGRATION'/
     11X,I2,'  DIMENSIONS, ERROR =',G13.5,',  NMAX = ',I8)
      WRITE(6,50)
 50   FORMAT('0INTEGRATION LIMITS :')
      DO 70 I=1,D
      WRITE(6,60) I,UMINUS(I),UPLUS(I)
 60   FORMAT(' X(',I2,') :',G13.5,'   TO',G14.5)
 70   CONTINUE
 80   NMIN=9999999
      MCOUNT=NSTOP
      ERR=ERROR
      ITRMF=1
      CALL PARTN(D,UMINUS,UPLUS,0.0,NMAX)
      IF(NFUN+2*NPT*NRGN.LE.NMAX) GOTO 110
      IF(IPRINT.LE.0) GOTO 100
      WRITE(6,90)
 90   FORMAT('0--- DIVON --- NOT ENOUGH FUNCTION EVALUATIONS LEFT'
     1/' FOR MORE PRECISE INTEGRAL ESTIMATE.')
 100  FUN1=FINTGL
      ERRINT=SPRD/NPT
      RETURN
 110  ERRR=ABS(ERR)
      IF(ERR.GT.0.0) ERRR=ERRR*FINTGL
      NINT=MIN(MAX(2.0*NPT,SPRD*FAC/ERRR),REAL(NMAX-NFUN)/NRGN)+.5
      IF(IPRINT.LE.0) GOTO 130
      WRITE(6,120) NINT
 120  FORMAT('0--- DIVON --- INTEGRAL AND ERROR ESTIMATE WITH',I6,
     1 ' POINTS / REGION :')
 130  CALL INTGRL(D,0,NINT,FUN1,ERRINT)
      RETURN
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE DVCOPY(N,A,B)
      INTEGER N
      DOUBLE PRECISION A(N), B(N)
      INTEGER I
      DO 10 I=1,N
      B(I)=A(I)
 10   CONTINUE
      RETURN
      END
*CMZ :          23/08/93  13.30.11  by  Jonathan Butterworth
*-- Author :
      DOUBLE PRECISION FUNCTION DVDOT(N, A, B)
      INTEGER N
      DOUBLE PRECISION A(N), B(N)
      INTEGER I
      DOUBLE PRECISION SUM
      DVDOT=0.0D+0
      IF(N.LE.0) RETURN
      SUM=0.0D+0
      DO 10 I=1,N
      SUM=SUM+A(I)*B(I)
 10   CONTINUE
      DVDOT=SUM
      RETURN
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE DVNBKD
C  *******************************************************************
C  *                                                                 *
C  *                                                                 *
C  *     -   D I V O N N E 4  -  PROGRAM FOR MULTIPLE INTEGRATION    *
C  *                                           AND                   *
C  *                               ADAPTIVE IMPORTANCE SAMPLING      *
C  *                                            BY                   *
C  *                                   NESTED PARTITIONING.          *
C  *                                                                 *
C  *                                                                 *
C  *                                                                 *
C  *                                                                 *
C  *     CODED BY:     J. H. FRIEDMAN, SLAC/CERN.                    *
C  *                                  AND                            *
C  *                   M. H. WRIGHT, STANFORD UNIVERSITY             *
C  *     MODIFIED FOR CERN BY T.LINDELOF, JULY 1981                  *
C  *     FORTRAN 77 UPDATE BY T.LINDELOF, MAY 1982                   *
C  *******************************************************************
C
C   INITIALIZATION ROUTINE TO 'SIMULATE' BLOCK DATA
C     BLOCK DATA
      COMMON /D151DT/ IDATE
      REAL*8 IDATE
      COMMON /PRINT/ IPRINT
      COMMON /ISTRGE/ MXRGNS , ISTOR(12000)
      COMMON /RSTRGE/ RSTSZE,RSTOR(18001)
      INTEGER RSTSZE
      COMMON /QUADRE/ IDEG
      COMMON /START/ ISTART
      COMMON /EXFILE/ NFILE
      COMMON /DISPOS/ IDISP
      COMMON /DEPTHS/ ISTDPH , INCDPH
      COMMON /SAMPLE/ NPOINT
      COMMON /CUTOLS/ BNDTOL, FRACT, REGNTL, FNLTOL
      COMMON /BNDLMT/ FLOBD,FUPBD
      COMMON /PRSTOP/ NSTOP
      COMMON /ZEETRM/ ITRMF
C
      DATA INITL/0/
      SAVE INITL
      SAVE
      IF(INITL.NE.0) RETURN
      INITL=1
C
      CALL JMUCPY(8H17/12/80,IDATE,2)
      IPRINT=1
      MXRGNS=3000
      RSTSZE=18001
      IDEG=0
      ISTART=1
      NFILE=1
      IDISP=0
      ISTDPH=3
      INCDPH=5
      NPOINT=50
      BNDTOL=.05
      FRACT=.5
      REGNTL=.9
      FNLTOL=.1
      FLOBD=-9.9E37
      FUPBD= 9.9E37
      NSTOP=5
      ITRMF=0
C
      CALL DVNOPT
C
      RETURN
      END
*CMZ :          13/01/95  14.03.47  by  Jonathan Butterworth
*-- Author :
      INTEGER FUNCTION EXMBUC(NUMBER, NDIM, BUCKTS, GOOD, MAXFUN, MAXDPH
     1, IRM)
      INTEGER NUMBER, NDIM, MAXFUN, MAXDPH
      REAL GOOD, BUCKTS(IRM)
      COMMON /PRINT/ IPRINT
      INTEGER IPRINT
      COMMON /ANSWER/ INTGRL,ERROR,ERRMAX,GEFF,Q2,Q3,Q5,NUMBR,MBUC
      COMMON /DEPTHS/ FSTDPH,INCDPH
      INTEGER FSTDPH,INCDPH,NTREES
      REAL INTGRL,VAR,ERROR,ERRMAX,VARMAX,VARMX2,PRCNT
      REAL Q2,Q3,Q5,GEFF,SEFF,FMAX,FMIN,TOTVOL
      COMMON /QUADRE/ DEGREE
      INTEGER DEGREE
      COMMON /MAXERR/ ERRPCT,ERRABS
      REAL ERRPCT,ERRABS
      COMMON /FUNN/ NFUN, NFOPT, NFCUT
      INTEGER NFUN, NFOPT, NFCUT
      COMMON /SIGSPL/ COORD,PLACE,FSTENT,DOSPLT
      COMMON /BUKSZE/ MAXWRD
      INTEGER MAXWRD
      COMMON /LIMITS/ GMINUS(10),GPLUS(10)
      LOGICAL FSTENT,DOSPLT,UTERM,USRTRM
      REAL COORD,PLACE
      INTEGER NUMBR,PTR
      SAVE
      NUMBR=NUMBER
      IF(NUMBER.NE.1) GOTO 40
      NTREES=0
      ERRPCT=GOOD
      ERRABS=0
      MAXDPH=FSTDPH
      FSTENT=.TRUE.
      NFUN=0
      NFOPT=0
      NFCUT=0
      DOSPLT=.FALSE.
      GUUD=ABS(GOOD)
      IF(GOOD.GE.0) GOTO 10
      ERRABS=GUUD
      ERRPCT=0
 10   IF(DEGREE.NE.1.AND.DEGREE.NE.2.AND.DEGREE.NE.3.AND.DEGREE.NE.5) D
     1EGREE=0
      IF(DEGREE.NE.1) GOTO 30
      TOTVOL=1.0E+0
      DO 20 I=1,NDIM
      TOTVOL=TOTVOL*(GPLUS(I)-GMINUS(I))
 20   CONTINUE
 30   EXMBUC=0
      RETURN
 40   NTREES=NTREES+1
      MBUC=0
      PTR=MBUC
      INTGRL=0.0E+0
      VAR=INTGRL
      VARMAX=VAR
      VARMX2=VARMAX
      Q2=VARMX2
      Q3=Q2
      Q5=Q3
      GEFF=Q5
      FMAX=GEFF
      FMIN=FMAX
      DO 80 IBUC=1,NUMBER
      INTGRL=INTGRL+BUCKTS(PTR+1)
      VAR=VAR+BUCKTS(PTR+2)
      IF(BUCKTS(PTR+2).LE.VARMAX) GOTO 50
      VARMX2=VARMAX
      VARMAX=BUCKTS(PTR+2)
      MBUC=IBUC
      GOTO 60
 50   IF(BUCKTS(PTR+2).LE.VARMX2) GOTO 60
      VARMX2=BUCKTS(PTR+2)
 60   IF(DEGREE.NE.1) GOTO 70
      GEFF=GEFF+(BUCKTS(PTR+5)-BUCKTS(PTR+6))*BUCKTS(PTR+7)
      FMAX=MAX(FMAX,BUCKTS(PTR+5))
      FMIN=MIN(FMIN,BUCKTS(PTR+6))
 70   IF(DEGREE.GE.2) Q2=Q2+BUCKTS(PTR+5)
      IF(DEGREE.GE.3) Q3=Q3+BUCKTS(PTR+6)
      IF(DEGREE.EQ.5) Q5=Q5+BUCKTS(PTR+7)
      PTR=PTR+MAXWRD
 80   CONTINUE
** JMB
      IF (VAR.GT.1.0E-20) THEN
        ERROR=SQRT(VAR)
      ELSE
*        WRITE(6,*) 'EXMBUC:VAR=',VAR,':Set to 0'
        ERROR=0.
        VAR=0.
      ENDIF
      IF (VARMAX.GT.1.0E-20) THEN
        ERRMAX=SQRT(VARMAX)
      ELSE
*        WRITE(6,*) 'EXMBUC:VARMAX=',VARMAX,':Set to 0'
        VARMAX=0.
        ERRMAX=0.
      ENDIF
      IF(DEGREE.EQ.1) THEN
        IF (GEFF.NE.0.D0) THEN
          GEFF=INTGRL/GEFF
        ELSE IF (INTGRL.NE.0.D0) THEN
          WRITE(6,400) 'EXMBUC ERROR! INTGRL='
          WRITE(6,401) INTGRL
	  INTGRL=0.D0
        ENDIF
      ENDIF
**
      IF(GOOD.LE.0) GOTO 100
      IF(INTGRL.EQ.0.0E+0) GOTO 90
      PRCNT=ERROR/ABS(INTGRL)
      GOTO 110
 90   PRCNT=0.0E+0
      GOTO 110
 100  PRCNT=ERROR
 110  UTERM=USRTRM(NTREES)
      IF(IPRINT.LE.0) GOTO 220
      IF(MOD(NTREES,IPRINT).NE.0 .AND. NFUN.LT.MAXFUN .AND. PRCNT.GT
     1.GUUD .AND. .NOT.UTERM) GOTO 220
      WRITE(6,120) NTREES,NUMBER,INTGRL,ERROR,ERRMAX,MBUC
 120  FORMAT(///' ITERATION ',I5,'.',I10,' REGIONS'/
     1 ' APPROXIMATE INTEGRAL = ',G13.5,'  WITH TOTAL RSS SPREAD ',
     2 G13.5/' THE LARGEST SINGLE SPREAD IS ',G13.5,
     3 '  IN REGION ',I5)
      IF(DEGREE.NE.1) GOTO 140
      SEFF=INTGRL/((FMAX-FMIN)*TOTVOL)
      WRITE(6,130) GEFF,SEFF
  130 FORMAT(' ESTIMATED RANGEN EFFICIENCY =',G13.5/
     *' SIMPLE ACCEPT/REJECT =',G13.5)
 140  IF(DEGREE.LT.2) GOTO 160
      WRITE(6,150) Q2
 150  FORMAT(' 2ND DEGREE QUADRATURE =  ',G13.5)
 160  IF(DEGREE.LT.3) GOTO 180
      WRITE(6,170) Q3
 170  FORMAT(' 3RD DEGREE QUADRATURE =  ',G13.5)
 180  IF(DEGREE.NE.5) GOTO 200
      WRITE(6,190) Q5
 190  FORMAT(' 5TH DEGREE QUADRATURE =  ',G13.5)
 200  WRITE(6,210) NFUN,NFOPT,NFCUT
 210  FORMAT(1X,I10,' INTEGRAND EVALUATIONS SO FAR'/1X,I10,
     1 ' IN OPTIMIZATION, ',I10,' IN FINDING CUTS')
 220  IF(NFUN.LT.MAXFUN) GOTO 250
      IF(IPRINT.LE.0) GOTO 240
      WRITE(6,230) MAXFUN
 230  FORMAT(' THIS EXCEEDES SPECIFIED LIMIT OF',I10)
 240  EXMBUC=0
      RETURN
 250  IF (IPRINT.GE.1) WRITE(6,402) 'PRCNT=',PRCNT
      IF(PRCNT.GT.GUUD) GOTO 280
      IF(IPRINT.LE.0) GOTO 270
      WRITE(6,260) ERROR
 260  FORMAT(' TOTAL RSS SPREAD ',G13.5,' IS BELOW SPECIFIED MAXIMUM')
 270  EXMBUC=0
      RETURN
 280  IF(.NOT.(UTERM)) GOTO 310
      IF(IPRINT.LE.0) GOTO 300
      WRITE(6,290)
 290  FORMAT(' USER REQUESTED TERMINATION')
 300  EXMBUC=0
      RETURN
 310  EXMBUC=MBUC
      MAXDPH=INCDPH
      ERRABS=SQRT(VARMX2)
      DOSPLT=.TRUE.
      PTR=MAXWRD*(MBUC-1)
      COORD=BUCKTS(PTR+3)
      PLACE=BUCKTS(PTR+4)
      RETURN
 400  FORMAT(A)
 402  FORMAT(A,F10.5)
 401  FORMAT(F10.4)
      END
*CMZ :          23/08/93  13.30.11  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE FEASMV (N,NFREE,IFREE,X,PFREE,XLOW,XHI,TOL,STEPMX,INEW,
     1IDNEW)
      INTEGER N, NFREE, INEW, IDNEW
      INTEGER IFREE(NFREE)
      DOUBLE PRECISION TOL, STEPMX
      DOUBLE PRECISION X(N), PFREE(NFREE), XLOW(N), XHI(N)
      INTEGER I, IDIR, II
      DOUBLE PRECISION STEP
      SAVE
C*NS  DOUBLE PRECISION DABS
      STEPMX=1.0D+30
      DO 30 I=1,NFREE
      II=IFREE(I)
      IF(ABS(PFREE(I)).LT.TOL) GOTO 30
      IF(PFREE(I).GT.0.0D+0) GOTO 10
      STEP=(XLOW(II)-X(II))/PFREE(I)
      IDIR=1
      GOTO 20
 10   STEP=(XHI(II)-X(II))/PFREE(I)
      IDIR=-1
 20   IF(STEP.GE.STEPMX) GOTO 30
      STEPMX=STEP
      INEW=II
      IDNEW=IDIR
 30   CONTINUE
      RETURN
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE FEQN (NCUT,F,FGAM,FNLIN)
      INTEGER NCUT
      DOUBLE PRECISION FGAM
      DOUBLE PRECISION F(NCUT), FNLIN(NCUT)
      INTEGER I, NCUTM1
      SAVE
      IF(NCUT.EQ.1) GOTO 20
      NCUTM1=NCUT-1
      DO 10 I=1,NCUTM1
      FNLIN(I)=F(I)-F(I+1)
 10   CONTINUE
 20   FNLIN(NCUT)=F(1)-FGAM
      RETURN
      END
*CMZ :          04/09/93  16.19.09  by  Jonathan Butterworth
*-- Author :
      REAL FUNCTION FUN(N,RX)
      DOUBLE PRECISION XDB(15), DFUN, FUNX
      REAL RX(N),SNGL
      EXTERNAL DFUN
      SAVE
      DO 10 I=1,N
      XDB(I)=DBLE(RX(I))
 10   CONTINUE
      FUNX=DFUN(N,XDB)
** JMB
      IF (ABS(FUNX).LT.1.E-36) FUNX=0.D0
      FUN=SNGL(FUNX)
      RETURN
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE GENPNT (NDIM,X,WT)
      REAL X(10),WWT,WT,FUN
      COMMON /ISTRGE/ MXRGNS,ISTOR(12000)
      COMMON /RSTRGE/ RSTSZE,RSTOR(18001)
      INTEGER RSTSZE
      COMMON /BUKSZE/ MAXWRD
      INTEGER MAXWRD
      COMMON /TRESZE/ ENTREE,ENTBUC
      INTEGER ENTREE,ENTBUC
      COMMON /LIMITS/ GMINUS(10),GPLUS(10)
      COMMON /FUNN/ NFUN, NFOPT, NFCUT /GENINL/ JNLGEN
      COMMON /QUADRE/ IDEG
      INTEGER IDEG
      INTEGER NFUN,JNLGEN,INLGEN
      INTEGER PARENT,PNTR,NTIMES,MODE
      REAL UPLUS(10),UMINUS(10),XSAVE(10),WWTSVE
      SAVE
      MODE=1
      ENTRY RANGEN(NDIM,X)
C*UL 10   INLGEN=JNLGEN
      INLGEN=JNLGEN
      IF(INLGEN.EQ.0) GOTO 90
      INLGEN=0
      NTIMES=0
      IF(ENTBUC.GT.1) GOTO 30
      WRITE(6,20)
 20   FORMAT(' GENPNT/RANGEN CALLED BEFORE PARTN')
      STOP
 30   IF(IDEG.EQ.1) GOTO 50
      WRITE(6,40)
 40   FORMAT(' GENPNT/RANGEN CALLED WITH IDEG NE 1')
      STOP
 50   IF(MAXWRD.EQ.7) GOTO 70
      WRITE(6,60)
 60   FORMAT(' GENPNT/RANGEN CALLED WITH IMPROPER BUCKET STORAGE')
      STOP
 70   ISCR=MXRGNS*(MAXWRD+1)
      RSTOR(ISCR+1)=0.0E+0
      PNTR=MXRGNS+1
      DO 80 J=1,ENTBUC
      RSTOR(ISCR+J+1)=RSTOR(ISCR+J)+RSTOR(PNTR+4)*RSTOR(PNTR+6)
      PNTR=PNTR+MAXWRD
 80   CONTINUE
 90   IF(MODE.EQ.1) GOTO 100
      IF(NTIMES.GT.0) GOTO 180
 100  CALL RANUMS(R,1)
      R=R*RSTOR(ISCR+ENTBUC+1)
      NL=1
      NH=ENTBUC+1
 110  IF(NH.LE.NL+1) GOTO 130
      NX=(NH+NL)/2
      IF(R.GT.RSTOR(ISCR+NX)) GOTO 120
      NH=NX
      GOTO 110
 120  NL=NX
      GOTO 110
 130  PARENT=1
      DO 140 J=1,NDIM
      UMINUS(J)=GMINUS(J)
      UPLUS(J)=GPLUS(J)
 140  CONTINUE
      CALL BOUNDS(NL,PARENT,ISTOR,RSTOR,UMINUS,UPLUS)
      PNTR=MXRGNS+1+MAXWRD*(NL-1)
      CALL RANUMS(X,NDIM)
      DO 150 J=1,NDIM
      X(J)=(UPLUS(J)-UMINUS(J))*X(J)+UMINUS(J)
 150  CONTINUE
      IF(MODE.NE.1) GOTO 160
      WT=FUN(NDIM,X)/RSTOR(PNTR+4)
      NFUN=NFUN+1
      RETURN
 160  CALL RANUMS(R,1)
      IF(R.LE.RSTOR(PNTR+5)/RSTOR(PNTR+4)) RETURN
      WWT=FUN(NDIM,X)/RSTOR(PNTR+4)
      NFUN=NFUN+1
      IF(R.GT.WWT) GOTO 100
      IF(WWT.LE.1.0E+0) RETURN
      NTIMES=INT(WWT)
      WWTSVE=WWT
      DO 170 I=1,NDIM
      XSAVE(I)=X(I)
 170  CONTINUE
      RETURN
 180  IF(NTIMES.LE.1) GOTO 200
      NTIMES=NTIMES-1
      DO 190 I=1,NDIM
      X(I)=XSAVE(I)
 190  CONTINUE
      RETURN
 200  NTIMES=0
      WWT=WWTSVE-INT(WWTSVE)
      IF(R.LT.WWT) RETURN
      GOTO 100
      END
*CMZ :          23/08/93  13.30.11  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE GRDCMP (N,NCOMP,INDEX,X,FVAL,DELTA,XHI,XTEMP,GRAD)
      INTEGER N, NCOMP
      INTEGER INDEX(NCOMP)
      DOUBLE PRECISION FVAL, DELTA
      DOUBLE PRECISION X(N), XHI(N), XTEMP(N), GRAD(NCOMP)
      INTEGER I, II
      DOUBLE PRECISION DELX, DFUN, FDIF, FTEMP
      SAVE
CMM   INTEGER IABS
      IF(NCOMP.LE.0) RETURN
      DO 10 I=1,N
      XTEMP(I)=X(I)
 10   CONTINUE
      DO 20 I=1,NCOMP
      DELX=DELTA
      II=ABS(INDEX(I))
      IF((X(II)+DELTA).GT.XHI(II)) DELX=-DELTA
      XTEMP(II)=X(II)+DELX
      FTEMP=DFUN(N,XTEMP)
      XTEMP(II)=X(II)
      FDIF=(FTEMP-FVAL)/DELX
      GRAD(I)=FDIF
 20   CONTINUE
      RETURN
      END
*CMZ :          09/01/95  16.40.40  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE INTGRL (NDIM,INTDEG,NPOINT,FUNINT,ERROR)
C*NS  INTEGER NDIM, INTDEG, INTPNT
      INTEGER NDIM, INTDEG
      REAL ERROR
      COMMON /ISTRGE/ MXRGNS,ISTOR(12000)
      COMMON /RSTRGE/ RSTSZE,RSTOR(18001)
      INTEGER RSTSZE
      COMMON /BUKSZE/ MAXWRD
      INTEGER MAXWRD
      COMMON /TRESZE/ ENTREE,ENTBUC
      INTEGER ENTREE,ENTBUC
      COMMON /LIMITS/ GMINUS(10),GPLUS(10)
      COMMON /PRINT/ IPRINT
      COMMON /FUNN/ NFUN, NFOPT, NFCUT
      INTEGER NFUN, NFOPT, NFCUT
      COMMON /QUADRE/ DEGREE
      INTEGER DEGREE
      REAL UMINUS(10),UPLUS(10)
      INTEGER PARENT,PTR,NTOT,NIBUC,BUCPTR
      DOUBLE PRECISION FINT,ERRSQ
      EXTERNAL FUN
      SAVE
      IF(ENTBUC.GT.1) GOTO 20
      WRITE(6,10)
 10   FORMAT(' FUNINT CALLED BEFORE PARTN.')
      STOP
 20   IF(INTDEG.LE.1) GOTO 30
      IF(INTDEG.EQ.2) NFUN=NFUN+ENTBUC*(NDIM+1)
      IF(INTDEG.EQ.3) NFUN=NFUN+ENTBUC*2*NDIM
      IF(INTDEG.EQ.5) NFUN=NFUN+ENTBUC*(2*NDIM**2+1)
 30   ISCR=MXRGNS*(MAXWRD+1)
      FUNINT=0.0E+0
      ERROR=FUNINT
      BUCPTR=MXRGNS+1
      IF(INTDEG.NE.1) GOTO 50
      NTOT=NPOINT*ENTBUC
      J=BUCPTR+1
      ERRTOT=0.0E+0
      DO 40 I=1,ENTBUC
      ERRTOT=ERRTOT+SQRT(RSTOR(J))
      J=J+MAXWRD
 40   CONTINUE
 50   DO 180 IBUC=1,ENTBUC
      PARENT=1
      DO 60 J=1,NDIM
      UMINUS(J)=GMINUS(J)
      UPLUS(J)=GPLUS(J)
 60   CONTINUE
      CALL BOUNDS(IBUC,PARENT,ISTOR,RSTOR,UMINUS,UPLUS)
      IF(INTDEG.GE.0) GOTO 70
      CALL USRINT(UMINUS,UPLUS,RSTOR(BUCPTR),RSTOR(BUCPTR+1),RFINT,RERR
     1SQ)
      FUNINT=FUNINT+RFINT
      ERROR=ERROR+RERRSQ
      BUCPTR=BUCPTR+MAXWRD
      GOTO 180
 70   IF(DEGREE.NE.1) GOTO 80
      CELVOL=RSTOR(BUCPTR+6)
      GOTO 100
 80   CELVOL=1.0E+0
      DO 90 J=1,NDIM
      CELVOL=CELVOL*(UPLUS(J)-UMINUS(J))
 90   CONTINUE
 100  IF(INTDEG.LE.1) GOTO 110
      FUNINT=FUNINT+QUAD(NDIM,INTDEG,UMINUS,UPLUS,FUN)*CELVOL
      GOTO 180
 110  IF(INTDEG.NE.1) GOTO 120
      NIBUC=INT(SQRT(RSTOR(BUCPTR+1))*NTOT/ERRTOT+0.5)
      IF(NIBUC.GE.5) GOTO 130
      FUNINT=FUNINT+RSTOR(BUCPTR)
      ERROR=ERROR+RSTOR(BUCPTR+1)
      BUCPTR=BUCPTR+MAXWRD
      GOTO 180
 120  NIBUC=NPOINT
      CALL QUASI(XX,NDIM,NIBUC,-NPOINT)
 130  PTR=ISCR
      FINT=0.0E+0
      ERRSQ=FINT
      DO 170 J=1,NIBUC
      IF(INTDEG.NE.1) GOTO 140
      CALL RANUMS(RSTOR(PTR+1),NDIM)
      GOTO 150
 140  CALL QUASI(RSTOR(PTR+1),NDIM,1,NIBUC)
 150  DO 160 I=1,NDIM
      RSTOR(I+PTR)=(UPLUS(I)-UMINUS(I))*RSTOR(I+PTR)+UMINUS(I)
 160  CONTINUE
      F=FUN(NDIM,RSTOR(PTR+1))
** JMB
      IF(ABS(F).LT.1.0E-17) F=0.0
      FINT=FINT+F
      ERRSQ=ERRSQ+F**2
 170  CONTINUE
      FINT=FINT/NIBUC
      ERRSQ=ERRSQ/NIBUC
      ERRSQ=ERRSQ-FINT**2
      FINT=FINT*CELVOL
      ERRSQ=ERRSQ*CELVOL**2
      IF(INTDEG.EQ.0) ERRSQ=RSTOR(BUCPTR+1)/NIBUC
      ERRSQ=ERRSQ/NIBUC
      FUNINT=FUNINT+FINT
      ERROR=ERROR+ERRSQ
      NFUN=NFUN+NIBUC
** JMB
      IF (ABS(FINT).GT.1.0E-36) THEN
        RSTOR(BUCPTR)=SNGL(FINT)
      ELSE
        RSTOR(BUCPTR)=0.0
      ENDIF
      BUCPTR=BUCPTR+MAXWRD
 180  CONTINUE
      IF(ERROR.GT.0) ERROR=SQRT(ERROR)
      IF(IPRINT.LE.0) GOTO 200
      WRITE(6,190) FUNINT,ERROR,NFUN
 190  FORMAT(' INTEGRAL ESTIMATE = ',G13.5,'   +/-',G13.5/1X,
     1 I10,' TOTAL INTEGRAND EVALUATIONS')
 200  RETURN
      END







*CMZ :          23/08/93  13.30.11  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE LDLSOL (N,NADIM,CHOL,B,X)
      INTEGER N, NADIM
      DOUBLE PRECISION CHOL(NADIM, N)
      DOUBLE PRECISION B(N), X(N)
      INTEGER I, IM1, J, JJ, JP1
      DOUBLE PRECISION SUM
      SAVE
      X(1)=B(1)
      IF(N.EQ.1) GOTO 30
      DO 20 I=2,N
      SUM=B(I)
      IM1=I-1
      DO 10 J=1,IM1
      SUM=SUM-CHOL(I,J)*X(J)
 10   CONTINUE
      X(I)=SUM
 20   CONTINUE
 30   IF(CHOL(N,N).LE.0.0D+0) RETURN
      X(N)=X(N)/CHOL(N,N)
      IF(N.EQ.1) RETURN
      DO 50 JJ=2,N
      J=N-JJ+1
      IF(CHOL(J,J).LE.0.0D+0) RETURN
      SUM=X(J)/CHOL(J,J)
      JP1=J+1
      DO 40 I=JP1,N
      SUM=SUM-CHOL(I,J)*X(I)
 40   CONTINUE
      X(J)=SUM
 50   CONTINUE
      RETURN
      END
*CMZ :          23/08/93  13.30.12  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE LOCSCH (IIMIN,N,NFREE,IFREE,X,FX,XLOW,XHI,DELTA,SUFTOL,
     1NFCNT,Y,FY,Z,FZ,P)
      INTEGER IIMIN, N, NFREE, NFCNT
      INTEGER IFREE(N)
      DOUBLE PRECISION FX, DELTA, SUFTOL, FY, FZ
      DOUBLE PRECISION X(N), XLOW(N), XHI(N), Y(N), Z(N), P(N)
      INTEGER I, II, ILOC, ITEST
      DOUBLE PRECISION A, B, B1, D, DELX, DFUN, E, EPSMCH, ETA, FA,
     1                 FBEST, FSAV, FTEST, FU, FV, FW, GTEST1, GTEST2,
     2                 GTP, GU, OLDF, PE, PNORM, R, RR, RTEPS,
     3                 SCXBD, SFTBND, SMAX, SNMAX, SOPP, SPMAX, SS,
     4                 SSAV, STP, STPDN, STPNEG, STPPOS, STPUP, T,
     5                 TOL, U, XBEST, XLAMDA, XV, XW, YSAV
      SAVE
      DATA EPSMCH/2.22D-16/
      RTEPS=SQRT(EPSMCH)
      DELX=DELTA
      DO 10 I=1,N
      Y(I)=X(I)
      Z(I)=X(I)
 10   CONTINUE
      FZ=FX
      FY=FX
      SMAX=1.0D+30
      DO 40 I=1,NFREE
      II=IFREE(I)
      STPUP=XHI(II)-X(II)
      STPDN=X(II)-XLOW(II)
      IF(STPUP.LT.STPDN) GOTO 20
      P(I)=1.0D+0
      STP=STPUP
      GOTO 30
 20   P(I)=-1.0D+0
      STP=STPDN
 30   IF(STP.LT.SMAX) SMAX=STP
 40   CONTINUE
      SMAX=0.9D+0*SMAX
 50   IF(DELX.GT.SMAX) DELX=SMAX
      DO 60 I=1,NFREE
      II=IFREE(I)
      Y(II)=X(II)+DELX*P(I)
 60   CONTINUE
      FY=DFUN(N,Y)
      IF(IIMIN.EQ.2) FY=-FY
      NFCNT=NFCNT+1
      FTEST=SUFTOL*(1.0D+0+ABS(FX))
      IF(ABS(FX-FY).GT.FTEST.OR.DELX.EQ.SMAX) GOTO 70
      DELX=5.0D+0*DELX
      GOTO 50
 70   DELX=DELTA
      CALL ORTHVC(N,NFREE,IFREE,X,FX,XLOW,XHI,Y,FY,P,SPMAX,SNMAX)
      SMAX=0.9D+0*SPMAX
      SOPP=0.9D+0*SNMAX
      IF(SMAX.GE.SNMAX.OR.NFREE.EQ.1) GOTO 90
      DO 80 I=1,NFREE
      P(I)=-P(I)
 80   CONTINUE
      SSAV=SMAX
      SMAX=SOPP
      SOPP=SSAV
 90   IF(DELX.GT.SMAX) DELX=SMAX
 100  DO 110 I=1,NFREE
      II=IFREE(I)
      Z(II)=Y(II)+DELX*P(I)
 110  CONTINUE
      FZ=DFUN(N,Z)
      IF(IIMIN.EQ.2) FZ=-FZ
      NFCNT=NFCNT+1
      FTEST=SUFTOL*(1.0D+0+ABS(FY))
      IF(ABS(FY-FZ).GT.FTEST.OR.(5.0D+0*DELX).GT.SMAX) GOTO 120
      DELX=5.0D+0*DELX
      GOTO 100
 120  IF(FY.EQ.FZ) GOTO 210
      IF(FY.LT.FZ) GOTO 130
      GTP=(FZ-FY)/DELX
      XLAMDA=SMAX/0.9D+0
      U=MIN(2.0D+0*DELX,SMAX)
      GOTO 160
 130  DO 140 I=1,NFREE
      P(I)=-P(I)
 140  CONTINUE
      U=MIN(2.0D+0*DELX,SOPP)
      XLAMDA=SOPP/0.9D+0+DELX
      GTP=(FY-FZ)/DELX
      DO 150 I=1,N
      YSAV=Y(I)
      Y(I)=Z(I)
      Z(I)=YSAV
 150  CONTINUE
      FSAV=FY
      FY=FZ
      FZ=FSAV
 160  CALL RLEN(NFREE,P,PNORM)
      PE=PNORM+RTEPS
      ILOC=1
      FU=FY
      GU=GTP
      SFTBND=0.0D+0
      ETA=RTEPS
      T=RTEPS/PE
 170  CALL NEWPTQ(RTEPS,T,ETA,SFTBND,XLAMDA,U,FU,GU,XBEST,FBEST,XW,FW,X
     1V,FV,A,FA,B,OLDF,B1,SCXBD,E,D,RR,SS,GTEST1,GTEST2,TOL,ILOC,ITEST)
      IF(ITEST.NE.1) GOTO 190
      R=XBEST+U
      DO 180 I=1,NFREE
      II=IFREE(I)
      Z(II)=Y(II)+R*P(I)
 180  CONTINUE
      FU=DFUN(N,Z)
      IF(IIMIN.EQ.2) FU=-FU
      NFCNT=NFCNT+1
      FZ=FU
      GOTO 170
 190  IF(ITEST.NE.0) GOTO 210
      DO 200 I=1,NFREE
      II=IFREE(I)
      Z(II)=Y(II)+XBEST*P(I)
 200  CONTINUE
      FZ=FBEST
 210  IF(FZ.EQ.FX) RETURN
      DO 220 I=1,NFREE
      II=IFREE(I)
      P(I)=Z(II)-X(II)
 220  CONTINUE
      SPMAX=1.0D+30
      SNMAX=1.0D+30
      DO 230 I=1,NFREE
      II=IFREE(I)
      CALL MXSTEP(X(II),XLOW(II),XHI(II),P(I),STPPOS,STPNEG)
      IF(STPPOS.LT.SPMAX) SPMAX=STPPOS
      IF(STPNEG.LT.SNMAX) SNMAX=STPNEG
 230  CONTINUE
      CALL RLEN(NFREE,P,PNORM)
      GTP=FZ-FX
      U=MIN(2.0D+0,0.9D+0*SPMAX)
      XLAMDA=SPMAX
      IF(FZ.LT.FX) GOTO 270
      DELX=DELTA/(PNORM+RTEPS)
      IF(DELX.GT.0.9D+0*SNMAX) DELX=0.9D+0*SNMAX
      DO 240 I=1,NFREE
      II=IFREE(I)
      Y(II)=X(II)-DELX*P(I)
 240  CONTINUE
      FY=DFUN(N,Y)
      IF(IIMIN.EQ.2) FY=-FY
      NFCNT=NFCNT+1
      IF(FY.LT.FX) GOTO 250
      IF(DELX.LT.1.0D+0) GTP=(FX-FY)/DELX
      GOTO 270
 250  GTP=(FY-FX)/DELX
      DO 260 I=1,NFREE
      P(I)=-P(I)
 260  CONTINUE
      U=MIN(2.0D+0*DELX,0.9D+0*SNMAX)
      XLAMDA=SNMAX
 270  ILOC=1
      PE=PNORM+RTEPS
      FU=FX
      GU=GTP
      SFTBND=0.0D+0
      ETA=RTEPS
      T=RTEPS/PE
 280  CALL NEWPTQ(RTEPS,T,ETA,SFTBND,XLAMDA,U,FU,GU,XBEST,FBEST,XW,FW,X
     1V,FV,A,FA,B,OLDF,B1,SCXBD,E,D,RR,SS,GTEST1,GTEST2,TOL,ILOC,ITEST)
      IF(ITEST.NE.1) GOTO 300
      R=XBEST+U
      DO 290 I=1,NFREE
      II=IFREE(I)
      Y(II)=X(II)+R*P(I)
 290  CONTINUE
      FU=DFUN(N,Y)
      IF(IIMIN.EQ.2) FU=-FU
      NFCNT=NFCNT+1
      GOTO 280
 300  IF(ITEST.NE.0) RETURN
      DO 310 I=1,NFREE
      II=IFREE(I)
      Z(II)=X(II)+XBEST*P(I)
 310  CONTINUE
      FZ=FBEST
      RETURN
      END
*CMZ :          23/08/93  13.30.11  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE MODCHL (N,NADIM,AHESS,ALPHA,Z,IFAIL)
      INTEGER N, NADIM, IFAIL
      DOUBLE PRECISION ALPHA
      DOUBLE PRECISION AHESS(NADIM,N), Z(N)
      INTEGER I, IB, IP1, J, K
      DOUBLE PRECISION A, BETA, DB, DI, GAMMA, P1, RMAX, T, W
      SAVE
      RMAX= 1.0D+37
      IFAIL=0
      A=ALPHA
      K=0
      DO 50 I=1,N
      P1=Z(I)
      DI=AHESS(I,I)
      T=A*P1
      DB=DI+T*P1
      AHESS(I,I)=DB
      IF(DB.GE.1.0D+0) GOTO 10
      IF(DB.GT.0.0D+0.AND.DI.LE.RMAX*DB) GOTO 10
      IFAIL=1
      RETURN
 10   GAMMA=DI/DB
      BETA=T/DB
      A=A*GAMMA
      K=K+I
      J=K
      IF(I.EQ.N) GOTO 50
      IP1=I+1
      IF(GAMMA.GE.2.5D-1) GOTO 30
      DO 20 IB=IP1,N
      T=AHESS(IB,I)
      AHESS(IB,I)=T*GAMMA+BETA*Z(IB)
      Z(IB)=Z(IB)-P1*T
 20   CONTINUE
      GOTO 50
 30   DO 40 IB=IP1,N
      T=AHESS(IB,I)
      W=Z(IB)-P1*T
      Z(IB)=W
      AHESS(IB,I)=BETA*W+T
 40   CONTINUE
 50   CONTINUE
      RETURN
      END
*CMZ :          23/08/93  13.30.11  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE MULCHK (N,NACTV,IACTV,IIMIN,TOL,X,XHI,FTRUE,DELTA,XTEMP
     1,GACTV,NEGMUL,IBDEL,IBTRUE)
      INTEGER N, NACTV, IIMIN, IBDEL, IBTRUE
      INTEGER IACTV(NACTV)
      LOGICAL NEGMUL
      DOUBLE PRECISION FTRUE, DELTA, TOL
      DOUBLE PRECISION X(N), XHI(N), XTEMP(N), GACTV(NACTV)
      INTEGER I
      DOUBLE PRECISION SIG, XLTEST, XMULOW
      SAVE
      CALL GRDCMP(N,NACTV,IACTV,X,FTRUE,DELTA,XHI,XTEMP,GACTV)
      IF(IIMIN.EQ.1) GOTO 20
      DO 10 I=1,NACTV
      GACTV(I)=-GACTV(I)
 10   CONTINUE
 20   NEGMUL=.FALSE.
      IBDEL=0
      IBTRUE=0
      XMULOW=0.0D+0
      DO 30 I=1,NACTV
      SIG=1.0D+0
      IF(IACTV(I).LT.0) SIG=-1.0D+0
      XLTEST=GACTV(I)*SIG
      IF(XLTEST.GE.(-TOL)) GOTO 30
      NEGMUL=.TRUE.
      IF(XLTEST.GE.XMULOW) GOTO 30
      XMULOW=XLTEST
      IBDEL=I
      IBTRUE=IACTV(I)
 30   CONTINUE
      RETURN
      END
*CMZ :          23/08/93  13.30.12  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE MXSTEP (X,XLOW,XHI,P,STPPOS,STPNEG)
      DOUBLE PRECISION X, XLOW, XHI, P, STPPOS, STPNEG
      SAVE
      IF(P.EQ.0.0D+0) RETURN
      IF(P.GT.0.0D+0) GOTO 10
      STPPOS=-(X-XLOW)/P
      STPNEG=-(XHI-X)/P
      RETURN
 10   STPPOS=(XHI-X)/P
      STPNEG=(X-XLOW)/P
      RETURN
      END
*CMZ :          23/08/93  13.30.12  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE NEWPTQ (EPS,T,ETA,SFTBND,XLAMDA,U,FU,GU,XMIN,FMIN,XW,FW
     1,XV,FV,A,FA,B,OLDF,B1,SCXBD,E,D,RR,SS,GTEST1,GTEST2,TOL,ILOC,ITEST
     2)
      INTEGER ILOC, ITEST
      DOUBLE PRECISION EPS, T, ETA, SFTBND, XLAMDA, U, FU, GU,
     1   XMIN, FMIN, XW, FW, XV, FV, A, FA, B, OLDF,
     2   B1, SCXBD, E, D, RR, SS, GTEST1, GTEST2, TOL
      DOUBLE PRECISION A1, D1, D2, Q, R, S, T2, XM
      SAVE
      GOTO (10,20,20,230,220),ILOC
 10   ITEST=2
      TOL=T
      T2=TOL+TOL
      IF(U.LE.0.0D+0.OR.XLAMDA.LE.T2.OR.GU.GT.0.0D+0) RETURN
      ITEST=1
      XMIN=0.0D+0
      XW=0.0D+0
      XV=0.0D+0
      A=0.0D+0
      OLDF=FU
      FMIN=FU
      FW=FU
      FV=FU
      FA=FU
      D=U
      SCXBD=EPS*ABS(XLAMDA)+T
      B=XLAMDA+SCXBD
      E=B
      B1=B
      SCXBD=XLAMDA-SCXBD/(1.0D+0+EPS)
      GTEST1=-1.0D-4*GU
      GTEST2=-ETA*GU
      ILOC=2
      GOTO 190
 20   IF(FU.GT.FMIN) GOTO 50
      IF(U.LT.0.0D+0) GOTO 30
      A=0.0D+0
      FA=FMIN
      GOTO 40
 30   B=0.0D+0
 40   XV=XW
      FV=FW
      FW=FMIN
      FMIN=FU
      XMIN=XMIN+U
      A=A-U
      B=B-U
      XV=XV-U
      XW=0.0D+0-U
      SCXBD=SCXBD-U
      TOL=EPS*ABS(XMIN)+T
      T2=TOL+TOL
      GOTO 90
 50   IF(U.GE.0.0D+0) GOTO 60
      A=U
      FA=FU
      GOTO 70
 60   B=U
 70   IF(FU.GT.FW.AND.XW.NE.0.0D+0) GOTO 80
      XV=XW
      FV=FW
      XW=U
      FW=FU
      GOTO 90
 80   IF(FU.GT.FV.AND.XV.NE.0.0D+0.AND.XV.NE.XW) GOTO 90
      XV=U
      FV=FU
 90   XM=5.0D-1*(A+B)
      IF(ABS(XM).LE.T2-5.0D-1*(B-A).OR.XMIN+B.LE.SFTBND.OR.FA-FMIN.LE.
     1ABS(A)*GTEST2.AND.FMIN.LT.OLDF.AND.(ABS(XMIN-XLAMDA).GT.TOL.OR.S
     2CXBD.LT.B)) GOTO 210
      R=0.0D+0
      Q=0.0D+0
      S=0.0D+0
      IF(ABS(E).LE.TOL) GOTO 120
      IF(ILOC.NE.2) GOTO 100
      Q=2.0D+0*(FW-FMIN-XW*GU)
      S=GU*XW*XW
      IF(XMIN.NE.0.0D+0) S=(2.0D+0*(FMIN-FW)+XW*GU)*XW
      GOTO 110
 100  R=XW*(FV-FMIN)
      Q=XV*(FW-FMIN)
      S=R*XW-Q*XV
      Q=2.0D+0*(Q-R)
 110  IF(Q.GT.0.0D+0) S=-S
      IF(Q.LE.0.0D+0) Q=-Q
      R=E
      IF(D.NE.B1.OR.B.LE.SCXBD) E=D
 120  A1=A
      B1=B
      IF(XMIN.NE.A) GOTO 130
      D=XM
      GOTO 160
 130  IF(B.LE.SCXBD) GOTO 140
      D=-4.0D+0*A
      IF(D.GE.B) D=SCXBD
      GOTO 160
 140  D1=A
      D2=B
      IF(ABS(D2).GT.TOL.AND.(XW.LE.0.0D+0.OR.ABS(D1).LE.TOL)) GOTO 1
     150
      U=D1
      D1=D2
      D2=U
 150  U=-D1/D2
      D=5.0D+0*D2*(1.0D-1+1.0D+0/U)/1.1D+1
      IF(U.LT.1.0D+0) D=5.0D-1*D2*SQRT(U)
 160  IF(D.LE.0.0D+0) A1=D
      IF(D.GT.0.0D+0) B1=D
      IF(ABS(S).GE.ABS(5.0D-1*Q*R).OR.S.LE.Q*A1.OR.S.GE.Q*B1) GOTO 1
     170
      D=S/Q
      IF(D-A.GE.T2.AND.B-D.GE.T2) GOTO 180
      D=TOL
      IF(XM.LE.0.0D+0) D=-TOL
      GOTO 180
 170  E=B
      IF(XM.LE.0.0D+0) E=A
 180  ILOC=3
 190  IF(D.LT.SCXBD) GOTO 200
      D=SCXBD
      SCXBD=SCXBD*(1.0D+0+7.5D-1*EPS)+7.5D-1*TOL
 200  U=D
      IF(ABS(D).LT.TOL.AND.D.LE.0.0D+0) U=-TOL
      IF(ABS(D).LT.TOL.AND.D.GT.0.0D+0) U=TOL
      ITEST=1
      RETURN
 210  RR=XMIN
      SS=5.0D-1
      FU=FMIN
 220  IF(ABS(XMIN-XLAMDA).GE.TOL.OR.XMIN.EQ.T) GOTO 230
      XMIN=XLAMDA
      IF(SCXBD.LE.B) GOTO 230
      U=0.0D+0
      ILOC=4
      ITEST=1
      RETURN
 230  IF(XMIN+B.GT.SFTBND) GOTO 240
      ITEST=4
      RETURN
 240  IF(OLDF-FU.LE.GTEST1*XMIN) GOTO 250
      FMIN=FU
      ITEST=0
      RETURN
 250  IF(XMIN.NE.T) GOTO 260
      ITEST=3
      RETURN
 260  XMIN=RR*SS
      SS=SS*SS
      IF(XMIN.LT.T) XMIN=T
      ITEST=1
      U=0.0D+0
      ILOC=5
      RETURN
      END
*CMZ :          23/08/93  13.30.11  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE NMDCHL (N,NADIM,AHESS,EPSMCH,Z,P)
      INTEGER N, NADIM
      DOUBLE PRECISION EPSMCH
      DOUBLE PRECISION Z(N), AHESS(NADIM,N), P(N)
C*NS  INTEGER I, IB, IQ, J, JN1, JP1, K
C*NS  INTEGER I, IB,     J, JN1, JP1, K
      DOUBLE PRECISION BETA, DJ, G, GAMMA, GAMMA1, PJ, T
      SAVE
      GAMMA=0.0D+0
      J=1
      DO 30 I=1,N
      T=Z(I)
      IF(I.EQ.1) GOTO 20
      K=I-1
      DO 10 IB=1,K
      T=T-P(IB)*AHESS(I,IB)
 10   CONTINUE
 20   P(I)=T
      GAMMA=GAMMA+T*T/AHESS(I,I)
 30   CONTINUE
      GAMMA1=1.0D+0-GAMMA
      GAMMA=EPSMCH
      IF(GAMMA1.GT.EPSMCH) GAMMA=GAMMA1
      IF(-GAMMA1.GT.EPSMCH) GAMMA=-GAMMA1
      JN1=N+1
      DO 50 I=1,N
      J=JN1-I
      PJ=P(J)
      DJ=AHESS(J,J)
      T=PJ/DJ
      Z(J)=PJ
      BETA=-T/GAMMA
      G=GAMMA+PJ*T
      AHESS(J,J)=DJ*GAMMA/G
      GAMMA=G
      IF(J.EQ.N) GOTO 50
      JP1=J+1
      DO 40 IB=JP1,N
      T=AHESS(IB,J)
      AHESS(IB,J)=T+BETA*Z(IB)
      Z(IB)=Z(IB)+PJ*T
 40   CONTINUE
 50   CONTINUE
      RETURN
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE NOCUT (N,XLOW,XUP,Z,IDSCNT,IDSCRM,PARTN,NFCNT)
      INTEGER N, IDSCNT, IDSCRM, NFCNT
      DOUBLE PRECISION PARTN
      DOUBLE PRECISION XLOW(N), XUP(N), Z(N)
      INTEGER I, MOD
CMM   DOUBLE PRECISION DABS, DEVMAX, DFUN, XSAVE, YMID, YLOW, YUP, YDIF
      DOUBLE PRECISION       DEVMAX, DFUN, XSAVE, YMID, YLOW, YUP, YDIF
      SAVE
      DEVMAX=0.0D+0
      DO 10 I=1,N
      Z(I)=0.5D+0*(XLOW(I)+XUP(I))
 10   CONTINUE
      YMID=DFUN(N,Z)
      DO 30 I=1,N
      XSAVE=Z(I)
      Z(I)=XLOW(I)
      YLOW=DFUN(N,Z)
      Z(I)=XUP(I)
      YUP=DFUN(N,Z)
      YDIF=ABS(YMID-0.5D+0*(YLOW+YUP))
      IF(YDIF.LT.DEVMAX) GOTO 20
      DEVMAX=YDIF
      PARTN=XSAVE
      IDSCRM=I
 20   Z(I)=XSAVE
 30   CONTINUE
      NFCNT=NFCNT+2*N+1
      IF(DEVMAX.NE.0.0D+0) GOTO 40
      IDSCRM=MOD(IDSCNT,N)+1
      PARTN=Z(IDSCRM)
      IDSCNT=IDSCNT+1
      RETURN
 40   IDSCNT=0
      RETURN
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE NODAUD (PARENT,TNODE,NEWNOD,NEWLFT,TREE)
      INTEGER PARENT,TNODE,NEWNOD,TREE(4,1000),SON,NEWLFT
      INTEGER SNODES(100)
      INTEGER PNODES,MNODES
      SAVE
      IF(100.GT.100) GOTO 200
      PNODES=0
      MNODES=100
      PNODES=PNODES+1
      IF(PNODES.GT.MNODES) GOTO 160
      SNODES(PNODES)=PARENT
 10   IF(TNODE.GT.TREE(1,PARENT)) GOTO 40
      IF(PNODES.LE.0) GOTO 150
      PARENT=SNODES(PNODES)
      IF(TREE(4,PARENT).GE.0) GOTO 20
      SON=TREE(2,PARENT)
      GOTO 30
 20   SON=TREE(3,PARENT)
 30   IF(SON.LE.0) GOTO 70
      PARENT=SON
      PNODES=PNODES+1
      IF(PNODES.GT.MNODES) GOTO 160
      SNODES(PNODES)=PARENT
      GOTO 70
 40   IF(PNODES.LE.0) GOTO 150
      PARENT=SNODES(PNODES)
      IF(TREE(4,PARENT).LE.0) GOTO 50
      SON=TREE(2,PARENT)
      GOTO 60
 50   SON=TREE(3,PARENT)
 60   IF(SON.LE.0) GOTO 70
      PARENT=SON
      PNODES=PNODES+1
      IF(PNODES.GT.MNODES) GOTO 160
      SNODES(PNODES)=PARENT
 70   IF(SON.GE.0) GOTO 10
      IF(TNODE.NE.-SON) GOTO 220
      GOTO 80
 80   IF(TREE(4,PARENT).GE.0) GOTO 100
      IF(SON.NE.TREE(3,PARENT)) GOTO 90
      TREE(3,PARENT)=NEWNOD
      GOTO 120
 90   TREE(2,PARENT)=NEWNOD
      GOTO 120
 100  IF(SON.NE.TREE(2,PARENT)) GOTO 110
      TREE(2,PARENT)=NEWNOD
      GOTO 120
 110  TREE(3,PARENT)=NEWNOD
 120  SON=NEWNOD
 130  IF(TREE(2,PARENT).EQ.SON.AND.TREE(4,PARENT).LT.0.OR.TREE(3,PARENT
     1).EQ.SON.AND.TREE(4,PARENT).GT.0) GOTO 140
      SON=PARENT
      PNODES=PNODES-1
      IF(PNODES.LT.0) GOTO 180
      IF(PNODES.LE.0) GOTO 150
      PARENT=SNODES(PNODES)
      GOTO 130
 140  TREE(1,PARENT)=NEWLFT
 150  RETURN
 160  WRITE(6,170)
 170  FORMAT(' NODAUD STACK OVERFLOW')
      STOP
 180  WRITE(6,190)
 190  FORMAT(' NODAUD STACK UNDERFLOW')
      STOP
 200  WRITE(6,210)
 210  FORMAT(' NODAUD IMPOSSIBLE')
      STOP
 220  WRITE(6,230) TNODE,TREE(SON,PARENT)
 230  FORMAT(' LOOKING FOR BUCKET ',I10,'  BUT FOUND ',I10)
      STOP
      END
*CMZ :          23/08/93  13.30.12  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE ORTHVC (N,NFREE,IFREE,X,FX,XLOW,XHI,Y,FY,P,SPMAX,SNMAX)
      INTEGER N, NFREE
      INTEGER IFREE(NFREE)
      DOUBLE PRECISION FX, FY, SPMAX, SNMAX
      DOUBLE PRECISION X(N), XLOW(N), XHI(N), Y(N), P(N)
      INTEGER I, II, KTEST, NFRM1, NTEMP
      LOGICAL EVEN
      DOUBLE PRECISION STPNEG, STPPOS
      SAVE
      KTEST=2*(NFREE/2)
      EVEN=.FALSE.
      IF(KTEST.EQ.NFREE) EVEN=.TRUE.
      NTEMP=NFREE
      IF(.NOT.EVEN) NTEMP=NFREE-1
      IF(NTEMP.GT.0) GOTO 10
      IF(FY.GT.FX) P(1)=-P(1)
      II=IFREE(1)
      CALL MXSTEP(Y(II),XLOW(II),XHI(II),P(1),SPMAX,SNMAX)
      RETURN
 10   SPMAX=1.0D+30
      SNMAX=1.0D+30
      DO 20 I=1,NTEMP,2
      II=IFREE(I)
      CALL MXSTEP(Y(II),XLOW(II),XHI(II),P(I),STPPOS,STPNEG)
      IF(STPPOS.LT.SPMAX) SPMAX=STPPOS
      IF(STPNEG.LT.SNMAX) SNMAX=STPNEG
      II=IFREE(I+1)
      P(I+1)=-P(I+1)
      CALL MXSTEP(Y(II),XLOW(II),XHI(II),P(I+1),STPPOS,STPNEG)
      IF(STPPOS.LT.SPMAX) SPMAX=STPPOS
      IF(STPNEG.LT.SNMAX) SNMAX=STPNEG
 20   CONTINUE
      IF(EVEN) RETURN
      NFRM1=NFREE-1
      P(NFRM1)=P(NFRM1)/2.0D+0
      P(NFREE)=-P(NFREE)/2.0D+0
      II=IFREE(NFREE)
      CALL MXSTEP(Y(II),XLOW(II),XHI(II),P(NFREE),STPPOS,STPNEG)
      IF(STPPOS.LT.SPMAX) SPMAX=STPPOS
      IF(STPNEG.LT.SNMAX) SNMAX=STPNEG
      RETURN
      END
*CMZ :          26/08/93  16.47.55  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE PARTN (NDIM,GMINUS,GPLUS,GOOD,MAXFUN)
      INTEGER NDIM, NPOINT, MAXFUN
      REAL FLOBD, FUPBD, GOOD
      REAL GMINUS(10), GPLUS(10)
      COMMON /D151DT/ IDATE /PRINT/ IPRINT
      REAL*8 IDATE
      COMMON /ISTRGE/ MXRGNS,TREE(4,1),DUMMY1(11996)
      INTEGER MXRGNS, TREE
      COMMON /RSTRGE/ RSTSZE,PRTNS(18001)
      INTEGER RSTSZE
      COMMON /MLIMIT/ MFLAG
      LOGICAL MFLAG
      COMMON /TRESZE/ ENTREE,ENTBUC
      INTEGER ENTREE, ENTBUC
      COMMON /START/ ISTART
      INTEGER ISTART
      COMMON /EXFILE/ NFILE
      INTEGER NFILE
      COMMON /DISPOS/ IDISP
      INTEGER IDISP
      COMMON /QUADRE/ IDEG
      INTEGER IDEG
      COMMON /BUKSZE/ MAXWRD
      INTEGER MAXWRD
      COMMON /GENINL/ INLGEN
      INTEGER INLGEN
      COMMON /LIMITS/ QMINUS(10),QPLUS(10)
      COMMON /SAMPLE/ NPOINT
      COMMON /BNDLMT/ FLOBD,FUPBD
      REAL UMINUS(10),UPLUS(10)
      INTEGER MAXDPH,PARENT
      INTEGER TARGET,MAXBUC,NEWENT,OLDSTR,NEWBUC,NEWEND,EXMBUC
      INTEGER NXRGNS,NEEDST,NMOVE
      SAVE

C         INITIALISATION OF CONSTANTS
      CALL DVNBKD
C
      IF(NDIM.LE.10) GOTO 20
      WRITE(6,10) NDIM
 10   FORMAT('0DIMENSION = ',I5,'  IS LARGER THAN UPPER LIMIT SET AT',
     1' COMPILE TIME.')
      STOP
 20   DO 30 I=1,NDIM
      QMINUS(I)=GMINUS(I)
      QPLUS(I)=GPLUS(I)
 30   CONTINUE
      MAXWRD=4
      IF(IDEG.EQ.1) MAXWRD=7
      IF(IDEG.GE.2) MAXWRD=MAXWRD+1
      IF(IDEG.GE.3) MAXWRD=MAXWRD+1
      IF(IDEG.EQ.5) MAXWRD=MAXWRD+1
      NEEDST=MXRGNS*(MAXWRD+1)+MAX((NDIM+1)*(NPOINT+5),MXRGNS)+1
      IF(NEEDST.LE.RSTSZE) GOTO 40
      NEEDST=RSTSZE-1
      MXRGNS=NEEDST/(MAXWRD+2)
      IF(MXRGNS.LT.(NDIM+1)*(NPOINT+5)) MXRGNS=(NEEDST-(NDIM+1)*(NPOINT
     1+5))/(MAXWRD+1)
 40   IF(ISTART.NE.2.AND.ISTART.NE.3) ISTART=1
      INLGEN=NPOINT
      IF(IPRINT.EQ.0) GOTO 120
      WRITE(6,50) IDATE
 50   FORMAT('1PARTN VERSION OF ',A8)
      WRITE(6,60) NDIM,GOOD,MAXFUN
 60   FORMAT(1X,I2,' DIMENSIONS. MAXIMUM RSS SPREAD OF',G13.5/
     1 '  WITH A MAXIMUM OF  ',I6,'  INTEGRAND EVALUATIONS.')
      IF(ISTART.NE.1) GOTO 80
      WRITE(6,70)
 70   FORMAT(' BEGIN PARTITIONING.')
      GOTO 120
 80   IF(ISTART.NE.3) GOTO 100
      WRITE(6,90) NFILE
 90   FORMAT(' CONTINUE PARTITIONING READ FROM TAPE',I2)
      GOTO 120
 100  IF(ISTART.NE.2) GOTO 120
      WRITE(6,110)
 110  FORMAT(' PARTITIONING CONTINUES.')
 120  MFLAG=.FALSE.
      IRM=18001-MXRGNS
      I=EXMBUC(1,NDIM,PRTNS(MXRGNS+1),GOOD,MAXFUN,MAXDPH,IRM)
      IF(ISTART.NE.1) GOTO 140
      ENTREE=1
      ENTBUC=ENTREE
      MXWDSV=MAXWRD
      NXRGNS=MXRGNS
      DO 130 I=1,NDIM
      UPLUS(I)=GPLUS(I)
      UMINUS(I)=GMINUS(I)
 130  CONTINUE
      IMR=18001-MXRGNS
      CALL RECPARZ
     1(NDIM,UMINUS,UPLUS,FLOBD,FUPBD,MAXDPH,ENTREE,TREE,PRTN
     1S,ENTBUC,PRTNS(MXRGNS+1),IMR)
      GOTO 160
 140  IF(ISTART.NE.3) GOTO 160
      READ (NFILE) ENTREE,INFO,NXRGNS,MXWDSV,((TREE(I,J),I=1,4),J=1,ENTR
     1EE),(PRTNS(J),J=1,INFO)
      ENTBUC=(INFO-NXRGNS)/MXWDSV
      IF(ENTREE.EQ.ENTBUC-1) GOTO 160
      WRITE(6,150) NFILE
 150  FORMAT(' INCONSISTENT INFORMATION ON TAPE',I2)
      STOP
 160  IF(NXRGNS.EQ.MXRGNS) GOTO 230
      NMOVE=MXWDSV*NXRGNS
      IF(NXRGNS.LE.MXRGNS) GOTO 200
      IF(ENTBUC.LT.MXRGNS) GOTO 180
      WRITE(6,170) MXRGNS,ENTBUC
 170  FORMAT(' MAXIMUM NUMBER OF REGIONS ',I5,' IS TOO SMALL.'/
     1 ' RESET TO GREATER THAN ',I5,'.')
      STOP
 180  DO 190 I=1,NMOVE
      PRTNS(I+MXRGNS)=PRTNS(I+NXRGNS)
 190  CONTINUE
      GOTO 230
 200  I=NMOVE
      GOTO 220
 210  I=I+(-1)
 220  IF((-1)*((I)-(1)).GT.0) GOTO 230
      PRTNS(I+MXRGNS)=PRTNS(I+NXRGNS)
      GOTO 210
 230  IF(MXWDSV.EQ.MAXWRD) GOTO 300
      IF(MXWDSV.GE.MAXWRD) GOTO 270
      I=ENTBUC
      GOTO 250
 240  I=I+(-1)
 250  IF((-1)*((I)-(1)).GT.0) GOTO 300
      DO 260 J=1,MXWDSV
      PRTNS(MAXWRD*(I-1)+J+MXRGNS)=PRTNS(MXWDSV*(I-1)+J+MXRGNS)
 260  CONTINUE
      GOTO 240
 270  DO 290 I=1,ENTBUC
      DO 280 J=1,MAXWRD
      PRTNS(MAXWRD*(I-1)+J+MXRGNS)=PRTNS(MXWDSV*(I-1)+J+MXRGNS)
 280  CONTINUE
 290  CONTINUE
 300  TARGET=EXMBUC(ENTBUC,NDIM,PRTNS(MXRGNS+1),GOOD,MAXFUN,MAXDPH,IRM)
      IF(TARGET.EQ.0) GOTO 350
      PARENT=1
      DO 310 I=1,NDIM
      UPLUS(I)=GPLUS(I)
      UMINUS(I)=GMINUS(I)
 310  CONTINUE
      CALL BOUNDS(TARGET,PARENT,TREE,PRTNS,UMINUS,UPLUS)
      NEWENT=ENTREE+1
      NEWBUC=ENTBUC+1
      IMR=18001-MXRGNS
      CALL RECPARZ
     1(NDIM,UMINUS,UPLUS,FLOBD,FUPBD,MAXDPH,NEWENT,TREE,PRTN
     1S,NEWBUC,PRTNS(MXRGNS+1),IMR)
      IF(.NOT.(MFLAG)) GOTO 330
      WRITE(6,320) MXRGNS
 320  FORMAT(' STORAGE LIMIT  ',I6,' REACHED.')
      GOTO 350
 330  IF(NEWBUC.EQ.ENTBUC+1) GOTO 350
      MAXBUC=NEWBUC
      PARENT=ENTREE+1
      OLDSTR=ENTBUC+1
      NEWBUC=TARGET
      CALL TREAUD(PARENT,OLDSTR,NEWBUC,MAXBUC-ENTBUC+NEWBUC-1,TREE)
      NEWEND=NEWBUC
      IF(TARGET.GE.ENTBUC) GOTO 340
      PARENT=1
      OLDSTR=TARGET+1
      NEWBUC=NEWBUC+1
      CALL TREAUD(PARENT,OLDSTR,NEWBUC,MAXBUC-1,TREE)
 340  PARENT=1
      CALL NODAUD(PARENT,TARGET,ENTREE+1,NEWEND,TREE)
      I11=18001-MAXWRD*MXRGNS-MXRGNS
      I12=18001-MXRGNS
      CALL BUCMVE(TARGET,NEWEND,ENTBUC+1,PRTNS(MAXWRD*MXRGNS+MXRGNS+1),
     1PRTNS(MXRGNS+1),I11,I12)
      ENTBUC=MAXBUC-1
      ENTREE=NEWENT
      GOTO 300
 350  MXWDSV=MAXWRD
      NXRGNS=MXRGNS
      IF(IDISP.EQ.0) RETURN
      INFO=MXRGNS+MAXWRD*ENTBUC
      REWIND NFILE
      WRITE(NFILE) ENTREE,INFO,MXRGNS,MAXWRD,((TREE(I,J),I=1,4),J=1,ENT
     1REE),(PRTNS(J),J=1,INFO)
      END FILE NFILE
      IF(IPRINT.EQ.0) GOTO 370
      WRITE(6,360) NFILE
 360  FORMAT(' INFORMATION FOR RESTART WRITTEN ON TAPE',I2)
 370  RETURN
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      REAL FUNCTION QUAD(D,DEG,LOWER,UPPER,FUN)
      INTEGER D,DEG
      REAL LOWER(D),UPPER(D),FUN
      EXTERNAL FUN
      INTEGER D0,DP1,TWOD,PT,NPTS,TIMES,KMAX
      REAL X(10,202),Y(10),Z(10),R
      DOUBLE PRECISION SUM0,SUM1,SUM2,B0,B1,B2
      REAL PI,TWOPI
      DATA PI,TWOPI /3.14159E+0,6.28319E+0/
      DATA D0 /0/
** JMB
      SAVE
      IF(D.EQ.D0) GOTO 130
      D0=D
      SQ23=SQRT(2.0E+0/3.0E+0)
      SQ13I=1.0E+0/SQRT(3.0E+0)
      DP1=D+1
      TWOD=2*D
      KMAX=D/2
      DO 20 K=1,KMAX
      DO 10 J=1,DP1
      I=J-1
      X(2*K-1,J)=SQ23*COS(TWOPI*I*K/DP1)
      X(2*K,J)=SQ23*SIN(TWOPI*I*K/DP1)
 10   CONTINUE
 20   CONTINUE
      IF(MOD(D,2).NE.1) GOTO 40
      DO 30 I=1,DP1
      X(D,I)=-SQ13I*(-1)**I
 30   CONTINUE
 40   DO 60 K=1,KMAX
      DO 50 I=1,TWOD
      X(2*K-1,I+DP1)=SQ23*COS((2*K-1)*I*PI/D)
      X(2*K,I+DP1)=SQ23*SIN((2*K-1)*I*PI/D)
 50   CONTINUE
 60   CONTINUE
      IF(MOD(D,2).NE.1) GOTO 80
      DO 70 I=1,TWOD
      X(D,I+DP1)=SQ13I*(-1)**I
 70   CONTINUE
 80   I=1
      GOTO 100
 90   I=I+1
 100  IF((I).GT.(DP1+TWOD)) GOTO 120
      DO 110 J=1,D
      X(J,I)=0.5E+0*(X(J,I)+1.0E+0)
 110  CONTINUE
      GOTO 90
 120  R=0.5E+0*SQRT(3.0E+0/5.0E+0)
      B0=(25.0E+0*D**2-115.0E+0*D+162.0E+0)/162.0E+0
      B1=(70.0E+0-25.0E+0*D)/162.0E+0
      B2=25.0E+0/324.0E+0
 130  IF(DEG.NE.2) GOTO 140
      PT=0
      NPTS=DP1
      GOTO 180
 140  IF(DEG.NE.3) GOTO 150
      PT=DP1
      NPTS=TWOD
      GOTO 180
 150  IF(DEG.NE.5) GOTO 160
      GOTO 210
 160  WRITE(6,170) DEG
 170  FORMAT(' ILLEGAL DEGREE =',G13.5)
      STOP
 180  QUAD=0.0E+0
      DO 200 I=1,NPTS
      DO 190 J=1,D
      Y(J)=(UPPER(J)-LOWER(J))*X(J,I+PT)+LOWER(J)
 190  CONTINUE
      QUAD=QUAD+FUN(D,Y)
 200  CONTINUE
      QUAD=QUAD/NPTS
      RETURN
 210  SUM0=0.0E+0
      SUM1=SUM0
      SUM2=SUM1
      DO 220 J=1,D
      Y(J)=0.5E+0*(UPPER(J)+LOWER(J))
 220  CONTINUE
      SUM0=FUN(D,Y)
      DO 230 J=1,D
      Z(J)=(UPPER(J)-LOWER(J))*R
 230  CONTINUE
      DO 240 I=1,D
      Y(I)=Y(I)+Z(I)
      SUM1=SUM1+FUN(D,Y)
      Y(I)=Y(I)-2.0E+0*Z(I)
      SUM1=SUM1+FUN(D,Y)
      Y(I)=Y(I)+Z(I)
 240  CONTINUE
      DO 270 I=2,D
      Y(I)=Y(I)+Z(I)
      JMAX=I-1
      DO 260 TIMES=1,2
      DO 250 J=1,JMAX
      Y(J)=Y(J)+Z(J)
      SUM2=SUM2+FUN(D,Y)
      Y(J)=Y(J)-2.0E+0*Z(J)
      SUM2=SUM2+FUN(D,Y)
      Y(J)=Y(J)+Z(J)
 250  CONTINUE
      Y(I)=Y(I)-2.0E+0*Z(I)
 260  CONTINUE
      Y(I)=Y(I)+3.0E+0*Z(I)
 270  CONTINUE
      QUAD=B0*SUM0+B1*SUM1+B2*SUM2
      RETURN
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE QUASI (X,NDIM,NGIVE,NF)
C--      CORRECTED 830401 CERN GEN PAM 1.05
C--      INITIAL VALUE OF I CHANGED FROM 1 TO 2 FOR CASE OF NPT LT 7
C--      CORRECTED 850503 CERN GEN PAM 1.16
C--      17'TH IN LIST OF PRIME NUMBERS CORRECTED FROM 1197 TO 1187
      INTEGER NDIM,N,NGIVE
      REAL X(10,NGIVE)
      INTEGER PRIME(21),AA(20,9)
      REAL THETA(10)
      INTEGER D0,N0,A,P,IPT
      DATA D0,N0 /2*0/
** JMB
      SAVE
      DATA PRIME / 7,19,29,37,47,97,149,199,293,397,499,599,691,797,887,
     1997,1187, 1499,1789,1999,9999999 /
      DATA AA / 2,7,12,27,29,61,44,76,81,163,209,165,390,346,192,705, 65
     16,629,1037,878, 2,3,20,13,30,78,67,73,103,274,405,248,585,305,674,
     2650, 358,526,1540,1082, 2,9,21,16,37,64,29,114,211,157,241,472,371
     3,477,113,252, 736,1205,853,1486, 2,2,27,35,44,8,110,42,258,104,445
     4,21,640,665,678,535, 1003,56,1634,1781, 2,17,15,29,38,76,85,183,28
     5,356,452,522,655,787,841,697, 1073,1025,1025,962, 2,17,2,2,2,67,12
     68,94,257,393,395,395,355,341,397,361, 383,367,383,398, 2,17,2,19,2
     7,79,139,149,170,165,169,161,157,171,149,167, 157,171,133,151, 2,17
     8,2,19,45,2,139,67,170,87,169,85,157,89,81,167,157,75,89,83, 2,17,2
     97,35,45,47,51,45,170,87,53,51,157,89,81,167,157,53,53,53 /
      N=ABS(NF)
      IF(NGIVE.LT.1.OR.NGIVE.GT.N) NGIVE=N
      IF(N.EQ.N0 .AND. NDIM.EQ.D0) GOTO 60
      N0=N
      D0=NDIM
      IPT=0
C--      INITIAL VALUE CHANGED FROM 1 TO 2, PAM 1.05
      I=2
      GOTO 20
 10   I=I+1
 20   IF((I).GT.(21)) GOTO 30
      IF(PRIME(I).LE.N) GOTO 10
 30   P=PRIME(I-1)
      A=AA(I-1,NDIM-1)
      THETA(1)=1.0E+0
      THETA(2)=A
      DO 40 I=3,NDIM
      THETA(I)=MOD(THETA(2)*THETA(I-1),REAL(P))
 40   CONTINUE
      DO 50 I=1,NDIM
      THETA(I)=THETA(I)/P
 50   CONTINUE
 60   IF(IPT+NGIVE.GT.P) NGIVE=P-IPT
      IF(NF.LT.0) RETURN
      DO 80 K=1,NGIVE
      IPT=IPT+1
      DO 70 I=1,NDIM
      TERM=IPT*THETA(I)
      X(I,K)=ABS(2.0E+0*MOD(TERM,1.0E+0)-1.0E+0)
 70   CONTINUE
 80   CONTINUE
      IF(IPT.EQ.P) IPT=0
      RETURN
      END
*CMZ :          23/08/93  13.30.12  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE RANUMS (X,N)
      REAL X(N)
      INTEGER IA, IC, ITWO, IY, M2, M
      DOUBLE PRECISION HALFM
      DATA M2 / 0 /, ITWO / 2 /, IY /123456789/
      SAVE
      IF(M2.NE.0) GOTO 20
      M=1
 10   M2=M
      M=ITWO*M2
      IF(M.GT.M2) GOTO 10
      HALFM=M2
      IA=8*INT(HALFM*ATAN(1.0D0)/8.0D0)+5
      IC=2*INT(HALFM*(0.5D0-SQRT(3.0D0)/6.0D0))+1
      S=0.5/HALFM
 20   DO 30 I=1,N
      IY=IY*IA+IC
      IF(IY/2.GT.M2) IY=(IY-M2)-M2
      IF(IY.LT.0) IY=(IY+M2)+M2
 30   X(I)=IY*S
      RETURN
      END
*CMZ :          26/08/93  17.27.26  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE RECPARZ
     1(NDIM,UMINUS,UPLUS,FLOBD,FUPBD,MAXDPH,ENTREE,TRE
     1E,PARTN,ENTBUC,BUCKTS,IBUC)
      INTEGER NDIM,ENTREE,ENTBUC,TREE(4,1000),MAXDPH
      REAL UMINUS(NDIM),UPLUS(NDIM),PARTN(18001),BUCKTS(IBUC)
      COMMON /MLIMIT/ MFLAG
      LOGICAL MFLAG
      COMMON /ISTRGE/ BUKMAX, DUMMY1(12000)
      COMMON /BUKSZE/ MAXWRD
C*NS  INTEGER MAXWRD,TREMAX,BUKMAX,PARENT,COORD
      INTEGER MAXWRD,TREMAX,BUKMAX,PARENT
      LOGICAL TERMNL
      INTEGER SNODES(100)
      INTEGER PNODES,MNODES
      REAL SPRTNS(100)
      INTEGER PPRTNS,MPRTNS
      INTEGER SAXES(100)
      INTEGER PAXES,MAXES
      SAVE
      IF(100.GT.100) GOTO 420
      PNODES=0
      MNODES=100
      IF(100.GT.100) GOTO 440
      PPRTNS=0
      MPRTNS=100
      IF(100.GT.100) GOTO 460
      PAXES=0
      MAXES=100
      TREMAX=BUKMAX-1
      TERMNL=.FALSE.
      IRBUC=IBUC-(MAXWRD*(ENTBUC-1))
      IRBUC=IBUC-(MAXWRD*(ENTBUC-1))
      CALL SPLIT(NDIM,UMINUS,UPLUS,FLOBD,FUPBD,TERMNL,ITREE,PARTN(ENTRE
     1E),BUCKTS(MAXWRD*(ENTBUC-1)+1),IRBUC)
      TREE(4,ENTREE)=-ITREE
      IF(TERMNL) RETURN
      PNODES=PNODES+1
      IF(PNODES.GT.MNODES) GOTO 320
      SNODES(PNODES)=ENTREE
      TREE(1,ENTREE)=0
 10   ENTREE=ENTREE+1
      IF(ENTREE.GT.TREMAX) GOTO 380
      TERMNL=PNODES.GE.MAXDPH
      IF(PNODES.LE.0) GOTO 210
      PARENT=SNODES(PNODES)
      JTREE=TREE(4,PARENT)
      PAXES=PAXES+1
      IF(PAXES.GT.MAXES) GOTO 360
      SAXES(PAXES)=JTREE
      IF(TREE(1,PARENT).NE.0) GOTO 90
      IF(JTREE.GE.0) GOTO 20
      PPRTNS=PPRTNS+1
      IF(PPRTNS.GT.MPRTNS) GOTO 340
      SPRTNS(PPRTNS)=UPLUS(-JTREE)
      UPLUS(-JTREE)=PARTN(PARENT)
      GOTO 30
 20   PPRTNS=PPRTNS+1
      IF(PPRTNS.GT.MPRTNS) GOTO 340
      SPRTNS(PPRTNS)=UMINUS(JTREE)
      UMINUS(JTREE)=PARTN(PARENT)
 30   IRBUC=IBUC-(MAXWRD*(ENTBUC-1))
      IRBUC=IBUC-(MAXWRD*(ENTBUC-1))
      CALL SPLIT(NDIM,UMINUS,UPLUS,FLOBD,FUPBD,TERMNL,ITREE,PARTN(ENTRE
     1E),BUCKTS(MAXWRD*(ENTBUC-1)+1),IRBUC)
      TREE(4,ENTREE)=-ITREE
      IF(.NOT.(.NOT.TERMNL)) GOTO 50
      IF(JTREE.GE.0) GOTO 40
      TREE(2,PARENT)=ENTREE
      TREE(1,ENTREE)=0
      PNODES=PNODES+1
      IF(PNODES.GT.MNODES) GOTO 320
      SNODES(PNODES)=ENTREE
      GOTO 10
 40   TREE(3,PARENT)=ENTREE
      TREE(1,ENTREE)=0
      PNODES=PNODES+1
      IF(PNODES.GT.MNODES) GOTO 320
      SNODES(PNODES)=ENTREE
      GOTO 10
 50   ENTREE=ENTREE-1
      IF(ENTREE.LT.1) GOTO 400
      IF(JTREE.GE.0) GOTO 60
      TREE(2,PARENT)=-ENTBUC
      GOTO 70
 60   TREE(3,PARENT)=-ENTBUC
 70   TREE(1,PARENT)=ENTBUC
      ENTBUC=ENTBUC+1
      IF(ENTBUC.GT.BUKMAX) GOTO 390
      IF(PAXES.LE.0) GOTO 220
      JTREE=SAXES(PAXES)
      PAXES=PAXES-1
      IF(PAXES.LT.0) GOTO 280
      IF(JTREE.GE.0) GOTO 80
      IF(PPRTNS.LE.0) GOTO 240
      UPLUS(-JTREE)=SPRTNS(PPRTNS)
      PPRTNS=PPRTNS-1
      IF(PPRTNS.LT.0) GOTO 300
      GOTO 10
 80   IF(PPRTNS.LE.0) GOTO 240
      UMINUS(JTREE)=SPRTNS(PPRTNS)
      PPRTNS=PPRTNS-1
      IF(PPRTNS.LT.0) GOTO 300
      GOTO 10
 90   IF(JTREE.LE.0) GOTO 100
      PPRTNS=PPRTNS+1
      IF(PPRTNS.GT.MPRTNS) GOTO 340
      SPRTNS(PPRTNS)=UPLUS(JTREE)
      UPLUS(JTREE)=PARTN(PARENT)
      GOTO 110
 100  PPRTNS=PPRTNS+1
      IF(PPRTNS.GT.MPRTNS) GOTO 340
      SPRTNS(PPRTNS)=UMINUS(-JTREE)
      UMINUS(-JTREE)=PARTN(PARENT)
 110  IRBUC=IBUC-(MAXWRD*(ENTBUC-1))
      IRBUC=IBUC-(MAXWRD*(ENTBUC-1))
      CALL SPLIT(NDIM,UMINUS,UPLUS,FLOBD,FUPBD,TERMNL,ITREE,PARTN(ENTRE
     1E),BUCKTS(MAXWRD*(ENTBUC-1)+1),IRBUC)
      TREE(4,ENTREE)=-ITREE
      IF(.NOT.(.NOT.TERMNL)) GOTO 130
      IF(JTREE.LE.0) GOTO 120
      TREE(2,PARENT)=ENTREE
      TREE(1,ENTREE)=0
      PNODES=PNODES+1
      IF(PNODES.GT.MNODES) GOTO 320
      SNODES(PNODES)=ENTREE
      GOTO 10
 120  TREE(3,PARENT)=ENTREE
      TREE(1,ENTREE)=0
      PNODES=PNODES+1
      IF(PNODES.GT.MNODES) GOTO 320
      SNODES(PNODES)=ENTREE
      GOTO 10
 130  ENTREE=ENTREE-1
      IF(ENTREE.LT.1) GOTO 400
      IF(JTREE.LE.0) GOTO 140
      TREE(2,PARENT)=-ENTBUC
      GOTO 150
 140  TREE(3,PARENT)=-ENTBUC
 150  IF(TREE(1,PARENT).EQ.0) GOTO 180
      IF(PAXES.LE.0) GOTO 220
      JTREE=SAXES(PAXES)
      PAXES=PAXES-1
      IF(PAXES.LT.0) GOTO 280
      IF(JTREE.LE.0) GOTO 160
      IF(PPRTNS.LE.0) GOTO 240
      UPLUS(JTREE)=SPRTNS(PPRTNS)
      PPRTNS=PPRTNS-1
      IF(PPRTNS.LT.0) GOTO 300
      GOTO 170
 160  IF(PPRTNS.LE.0) GOTO 240
      UMINUS(-JTREE)=SPRTNS(PPRTNS)
      PPRTNS=PPRTNS-1
      IF(PPRTNS.LT.0) GOTO 300
 170  PNODES=PNODES-1
      IF(PNODES.LT.0) GOTO 260
      IF(PNODES.LE.0) GOTO 210
      PARENT=SNODES(PNODES)
      GOTO 150
 180  IF(PAXES.LE.0) GOTO 220
      JTREE=SAXES(PAXES)
      PAXES=PAXES-1
      IF(PAXES.LT.0) GOTO 280
      IF(JTREE.GE.0) GOTO 190
      IF(PPRTNS.LE.0) GOTO 240
      UPLUS(-JTREE)=SPRTNS(PPRTNS)
      PPRTNS=PPRTNS-1
      IF(PPRTNS.LT.0) GOTO 300
      GOTO 200
 190  IF(PPRTNS.LE.0) GOTO 240
      UMINUS(JTREE)=SPRTNS(PPRTNS)
      PPRTNS=PPRTNS-1
      IF(PPRTNS.LT.0) GOTO 300
 200  TREE(1,PARENT)=ENTBUC
      ENTBUC=ENTBUC+1
      IF(ENTBUC.GT.BUKMAX) GOTO 390
      GOTO 10
 210  RETURN
 220  WRITE(6,230)
 230  FORMAT(' STACK AXES EMPTY')
      STOP
 240  WRITE(6,250)
 250  FORMAT(' STACK PRTNS EMPTY')
      STOP
 260  WRITE(6,270)
 270  FORMAT(' STACK NODES UNDERFLOW')
      STOP
 280  WRITE(6,290)
 290  FORMAT(' STACK AXES UNDERFLOW')
      STOP
 300  WRITE(6,310)
 310  FORMAT(' STACK PRTNS UNDERFLOW')
      STOP
 320  WRITE(6,330)
 330  FORMAT(' STACK NODES OVERFLOW')
      STOP
 340  WRITE(6,350)
 350  FORMAT(' STACK PRTNS OVERFLOW')
      STOP
 360  WRITE(6,370)
 370  FORMAT(' STACK AXES OVERFLOW')
      STOP
 380  MFLAG=.TRUE.
      RETURN
 390  MFLAG=.TRUE.
      RETURN
 400  WRITE(6,410)
 410  FORMAT(' ENTREE DECREMENTED PAST ONE')
      STOP
 420  WRITE(6,430) MAXDPH
 430  FORMAT(' MAXDPH = ',I10,' TOO LARGE')
      STOP
 440  WRITE(6,450) MAXDPH
 450  FORMAT(' MAXDPH = ',I10,' TOO LARGE')
      STOP
 460  WRITE(6,470) MAXDPH
 470  FORMAT(' MAXDPH = ',I10,' TOO LARGE')
      STOP
      END
*CMZ :          23/08/93  13.30.12  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE RLEN (LEN,VEC,VLEN)
      INTEGER   LEN
      DOUBLE PRECISION   VEC(LEN), VLEN
      INTEGER  I
      DOUBLE PRECISION   ABSV, RATIO, RMIN, TOL, TVMX, VMAX
      SAVE
      DATA TOL / 1.0D-20 /
      DATA RMIN / 1.0D-34/
      VMAX=0.0D+0
      DO 10 I=1,LEN
      ABSV=ABS(VEC(I))
      IF(ABSV.GT.VMAX) VMAX=ABSV
 10   CONTINUE
      VLEN=0.0D+0
      IF(VMAX.EQ.0.0D+0) RETURN
      TVMX=0.0D+0
      IF(VMAX.GT.RMIN) TVMX=TOL*VMAX
      DO 20 I=1,LEN
      IF(ABS(VEC(I)).LE.TVMX) GOTO 20
      RATIO=VEC(I)/VMAX
      VLEN=VLEN+RATIO*RATIO
 20   CONTINUE
      VLEN=VMAX*SQRT(VLEN)
      RETURN
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE SETTOL (N,FTOL,GTOL,DELTA,ETA,ALFMAX,MAXFUN,IPRINT)
      INTEGER N, MAXFUN, IPRINT
      DOUBLE PRECISION FTOL, GTOL, DELTA, ETA, ALFMAX
      SAVE
      FTOL=5.0D-2
      GTOL=1.0D-2
      DELTA=1.0D-5
      ETA=0.2D+0
      ALFMAX=1.0D+2
      MAXFUN=50*N
      IPRINT=-1
      RETURN
      END
*CMZ :          23/08/93  13.30.11  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE SHRNK (N,NADIM,AHESS,IOUT,VEC)
      INTEGER N, NADIM, IOUT
      DOUBLE PRECISION AHESS(NADIM, N), VEC(N)
      INTEGER I, IFAIL, IM1, IOM1, J, NM1
      DOUBLE PRECISION GAMMA
      SAVE
      IF(N.EQ.IOUT.OR.N.EQ.1) RETURN
      IF(IOUT.EQ.1) GOTO 20
      IOM1=IOUT-1
      DO 10 I=1,IOM1
      VEC(I)=0.0D+0
 10   CONTINUE
 20   NM1=N-1
      GAMMA=AHESS(IOUT,IOUT)
      DO 30 I=IOUT,NM1
      VEC(I)=AHESS(I+1,IOUT)
      AHESS(I,I)=AHESS(I+1,I+1)
 30   CONTINUE
      DO 50 I=IOUT,NM1
      IM1=I-1
      IF(IM1.EQ.0) GOTO 50
      DO 40 J=1,IM1
      AHESS(I,J)=AHESS(I+1,J)
 40   CONTINUE
 50   CONTINUE
      CALL MODCHL(NM1,NADIM,AHESS,GAMMA,VEC,IFAIL)
      AHESS(N,N)=0.0D+0
      DO 60 J=1,NM1
      AHESS(N,J)=0.0D+0
 60   CONTINUE
      RETURN
      END
*CMZ :          04/09/93  17.07.27  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE SPLIT (NDIM,UMINUS,UPLUS,FLOBD,FUPBD,TERMNL,DISCRM,PART
     1N,BUCKTS,IRBUC)
      INTEGER NDIM, DISCRM, IRBUC
      LOGICAL TERMNL
      REAL FLOBD, FUPBD, PARTN
      REAL UMINUS(NDIM), UPLUS(NDIM), BUCKTS(IRBUC)
      INTEGER MPOINT
      COMMON /SAMPLE/ MPOINT
      INTEGER MAXWRD
      COMMON /BUKSZE/ MAXWRD
      REAL ERRPCT, ERRABS
      COMMON /MAXERR/ ERRPCT, ERRABS
      INTEGER NFUN, NFOPT, NFCUT
      COMMON /FUNN/ NFUN, NFOPT, NFCUT
      LOGICAL FSTENT, DOSPLT
      REAL COORD, PLACE
      COMMON /SIGSPL/ COORD, PLACE, FSTENT, DOSPLT
      INTEGER DEGREE
      COMMON /QUADRE/ DEGREE
      INTEGER MXRGNS, ISTOR
      COMMON /ISTRGE/ MXRGNS, ISTOR(12000)
      INTEGER RSTSZE
      REAL RSTOR
      COMMON /RSTRGE/ RSTSZE,RSTOR(18001)
      REAL BNDTOL, FRACT, REGNTL, FNLTOL
      COMMON /CUTOLS/ BNDTOL, FRACT, REGNTL, FNLTOL
      INTEGER MAJOR, MINOR
      INTEGER ICUT(20), IWORK(20)
      INTEGER DCMSVE(20)
      INTEGER MAXJ, MINJ, NPOINT, NFCNT
      INTEGER DISCNT
      INTEGER NCUT
      EXTERNAL FUN
      LOGICAL LMAX
      REAL DIFEXT
      REAL X(10, 202), Y(202)
      REAL PARSVE(20)
C*NS  REAL DEVMAX, DEVMIN, YMAX, YMIN, SUM, CELVOL, YBAR, ERROR, ERRSQ
C*NS  DOUBLE PRECISION DPARTN, DFLOAT, DBNDTL, DFRACT, DREGTL, DFNLTL
      REAL         DEVMIN, YMAX, YMIN,      CELVOL, YBAR, ERROR, ERRSQ
      DOUBLE PRECISION DPARTN,         DBNDTL, DFRACT, DREGTL, DFNLTL
      DOUBLE PRECISION FMAJOR, FMINOR, VOL
      DOUBLE PRECISION DELPLS(10), DELNEG(10)
      DOUBLE PRECISION XLOW(10), XUP(10)
      DOUBLE PRECISION Z(10), WORK(200)
      DATA DISCNT /0/
      DATA NCUT /0/
* JMB
      SAVE
      IF(.NOT.(DOSPLT)) GOTO 10
      DOSPLT=.FALSE.
      FSTENT=DOSPLT
      DISCRM=COORD
      PARTN=PLACE
      RETURN
 10   IF(NCUT.EQ.0) GOTO 20
      DISCRM=DCMSVE(NCUT)
      PARTN=PARSVE(NCUT)
      NCUT=NCUT-1
      DOSPLT=.FALSE.
      FSTENT=DOSPLT
      TERMNL=FSTENT
      RETURN
 20   ISCR=MXRGNS*(MAXWRD+1)+1
      NPOINT=MPOINT
      CALL QUASI(X,NDIM,NPOINT,MPOINT)
      DO 40 J=1,NPOINT
      DO 30 I=1,NDIM
      X(I,J)=(UPLUS(I)-UMINUS(I))*X(I,J)+UMINUS(I)
 30   CONTINUE
      Y(J)=FUN(NDIM,X(1,J))
 40   CONTINUE
      NFUN=NFUN+NPOINT
      CELVOL=1.0E+0
      DO 50 I=1,NDIM
      CELVOL=CELVOL*(UPLUS(I)-UMINUS(I))
 50   CONTINUE
      YMAX= -9.9E34
      YMIN=  9.9E34
      DO 70 J=1,NPOINT
      IF(Y(J).GE.YMIN) GOTO 60
      YMIN=Y(J)
      MINJ=J
 60   IF(Y(J).LE.YMAX) GOTO 70
      YMAX=Y(J)
      MAXJ=J
 70   CONTINUE
      DO 80 I=1,NDIM
      X(I,NPOINT+1)=X(I,MAXJ)
      X(I,NPOINT+2)=X(I,MINJ)
 80   CONTINUE
      Y(NPOINT+1)=YMAX
      Y(NPOINT+2)=YMIN
      CALL BUFOPT(NDIM,X(1,NPOINT+2),X(1,NPOINT+1),UMINUS,UPLUS,Y(NPOIN
     1T+2),Y(NPOINT+1),FLOBD,FUPBD,WORK,200,IWORK,20,NFCNT,IRESLT)
      NFUN=NFUN+NFCNT
      NFOPT=NFOPT+NFCNT
      DIFEXT=Y(NPOINT+1)-Y(NPOINT+2)
      ERROR=DIFEXT*CELVOL*0.5E+0
      YBAR=0.0E+0
      DO 90 I=1,NPOINT
      YBAR=YBAR+Y(I)
 90   CONTINUE
      YBAR=YBAR/NPOINT
      IF (YBAR.LT.1.E-36) YBAR=0.E+0
      FBAR=YBAR*CELVOL
      IF(ABS(Y(NPOINT+2)-YBAR).LE.ABS(Y(NPOINT+1)-YBAR)) GOTO 100
      MAJOR=NPOINT+2
      MINOR=NPOINT+1
      GOTO 110
 100  MAJOR=NPOINT+1
      MINOR=NPOINT+2
 110  FMAJOR=Y(MAJOR)
      FMINOR=Y(MINOR)
      I=1
      GOTO 130
 120  I=I+1
 130  IF((I).GT.(NDIM)) GOTO 140
      Z(I)=X(I,MAJOR)
      GOTO 120
 140  I=1
      GOTO 160
 150  I=I+1
 160  IF((I).GT.(NDIM)) GOTO 170
      XLOW(I)=UMINUS(I)
      XUP(I)=UPLUS(I)
      GOTO 150
 170  NCDIM=2*NDIM
      DFRACT=FRACT
      DBNDTL=BNDTOL
      CALL TSTEXT(NDIM,Z,XLOW,XUP,DBNDTL,DFRACT,NCUT,NCDIM,ICUT,DELPLS,
     1DELNEG)
      IF(NCUT.NE.0) GOTO 180
      CALL NOCUT(NDIM,XLOW,XUP,WORK(1),DISCNT,DISCRM,DPARTN,NFUN)
      PARTN=DPARTN
      GOTO 240
 180  LMAX=.TRUE.
      IF(FMAJOR.GE.FMINOR) GOTO 190
      LMAX=.FALSE.
 190  VOL=CELVOL
      DREGTL=REGNTL
      DFNLTL=FNLTOL
      CALL DELSLV(NDIM,FMAJOR,FMINOR,LMAX,DFRACT,Z,XLOW,XUP,VOL,NCUT,NCD
     1IM,ICUT,DELPLS,DELNEG,DREGTL,DFNLTL,WORK(1),WORK(NCDIM+1),WORK(2*N
     2CDIM+1),WORK(3*NCDIM+1),WORK(4*NCDIM+1),WORK(5*NCDIM+1),WORK(6*NCD
     3IM+1),WORK(7*NCDIM+1),WORK(8*NCDIM+1),NFCNT)
      NFUN=NFUN+NFCNT
      NFCUT=NFCUT+NFCNT
      IF(NCUT.NE.0) GOTO 200
      CALL NOCUT(NDIM,XLOW,XUP,WORK(1),DISCNT,DISCRM,DPARTN,NFUN)
      PARTN=DPARTN
      GOTO 240
 200  I=1
      GOTO 220
 210  I=I+1
 220  IF((I).GT.(NCUT)) GOTO 240
      DCMSVE(I)=ICUT(I)
      II=ABS(ICUT(I))
      IF(ICUT(I).LE.0) GOTO 230
      PARSVE(I)=X(II,MAJOR)+DELPLS(II)
      GOTO 210
 230  PARSVE(I)=X(II,MAJOR)-DELNEG(II)
      GOTO 210
 240  NCUTSV=NCUT
      IF(NCUT.LE.0) GOTO 250
      DISCRM=ICUT(NCUT)
      PARTN=PARSVE(NCUT)
      NCUT=NCUT-1
 250  IF(.NOT.(FSTENT)) GOTO 260
      FSTENT=.FALSE.
      TERMNL=FSTENT
      RETURN
 260  IF(ERROR.EQ.0.0E+0) GOTO 270
      TERMNL=TERMNL.OR.ERROR.LT.ERRABS
      IF(FBAR.NE.0.0E+0) TERMNL=TERMNL.OR.ERROR/ABS(FBAR).LE.ERRPCT
 270  IF(.NOT.(TERMNL)) GOTO 330
      NCUT=0
      IF(NCUTSV.LE.0) GOTO 290
      DEVMIN= 9.9E34
      DO 280 I=1,NCUTSV
      J=ABS(ICUT(I))
      XX=ABS(X(J,MAJOR)-PARSVE(I))
      IF(XX.GE.DEVMIN) GOTO 280
      DEVMIN=XX
      PARTN=PARSVE(I)
      DISCRM=ICUT(I)
 280  CONTINUE
*** JMB 290  ERRSQ=ERROR**2
 290  IF (ABS(ERROR).GT.1.E-16) THEN
        ERRSQ=ERROR**2
      ELSE
        ERRSQ=0.0
      ENDIF
      ERRABS=MAX(ERRABS,ERROR)
      BUCKTS(1)=FBAR
      BUCKTS(2)=ERRSQ
      BUCKTS(3)=DISCRM
      BUCKTS(4)=PARTN
      IF(DEGREE.NE.1) GOTO 300
      BUCKTS(5)=Y(NPOINT+1)
      BUCKTS(6)=Y(NPOINT+2)
      BUCKTS(7)=CELVOL
 300  IF(DEGREE.LT.2) GOTO 310
      NFUN=NFUN+NDIM+1
      BUCKTS(5)=QUAD(NDIM,2,UMINUS,UPLUS,FUN)*CELVOL
 310  IF(DEGREE.LT.3) GOTO 320
      NFUN=NFUN+NDIM+NDIM
      BUCKTS(6)=QUAD(NDIM,3,UMINUS,UPLUS,FUN)*CELVOL
 320  IF(DEGREE.NE.5) GOTO 330
      NFUN=NFUN+2*NDIM**2+1
      BUCKTS(7)=QUAD(NDIM,5,UMINUS,UPLUS,FUN)*CELVOL
 330  RETURN
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE TREAUD (PARENT,OLDSTR,NEWBUC,NEWMAX,TREE)
      INTEGER PARENT,OLDSTR,NEWBUC,NEWMAX,TREE(4,1000),SON
      INTEGER SNODES(100)
      INTEGER PNODES,MNODES
      SAVE
      IF(100.GT.100) GOTO 250
      PNODES=0
      MNODES=100
      PNODES=PNODES+1
      IF(PNODES.GT.MNODES) GOTO 210
      SNODES(PNODES)=PARENT
 10   IF(OLDSTR.GT.TREE(1,PARENT)) GOTO 40
      IF(PNODES.LE.0) GOTO 200
      PARENT=SNODES(PNODES)
      IF(TREE(4,PARENT).GE.0) GOTO 20
      SON=TREE(2,PARENT)
      GOTO 30
 20   SON=TREE(3,PARENT)
 30   IF(SON.LE.0) GOTO 70
      PARENT=SON
      PNODES=PNODES+1
      IF(PNODES.GT.MNODES) GOTO 210
      SNODES(PNODES)=PARENT
      GOTO 70
 40   IF(PNODES.LE.0) GOTO 200
      PARENT=SNODES(PNODES)
      IF(TREE(4,PARENT).LE.0) GOTO 50
      SON=TREE(2,PARENT)
      GOTO 60
 50   SON=TREE(3,PARENT)
 60   IF(SON.LE.0) GOTO 70
      PARENT=SON
      PNODES=PNODES+1
      IF(PNODES.GT.MNODES) GOTO 210
      SNODES(PNODES)=PARENT
 70   IF(SON.GE.0) GOTO 10
      IF(OLDSTR.NE.-SON) GOTO 290
      GOTO 80
 80   IF(SON.LT.0) GOTO 110
      IF(PNODES.LE.0) GOTO 200
      PARENT=SNODES(PNODES)
      IF(TREE(4,PARENT).GE.0) GOTO 90
      SON=TREE(2,PARENT)
      GOTO 100
 90   SON=TREE(3,PARENT)
 100  IF(SON.LE.0) GOTO 80
      PARENT=SON
      PNODES=PNODES+1
      IF(PNODES.GT.MNODES) GOTO 210
      SNODES(PNODES)=PARENT
      GOTO 80
 110  IF(TREE(4,PARENT).GE.0) GOTO 130
      IF(SON.NE.TREE(3,PARENT)) GOTO 120
      TREE(3,PARENT)=-NEWBUC
      GOTO 150
 120  TREE(2,PARENT)=-NEWBUC
      GOTO 150
 130  IF(SON.NE.TREE(2,PARENT)) GOTO 140
      TREE(2,PARENT)=-NEWBUC
      GOTO 150
 140  TREE(3,PARENT)=-NEWBUC
 150  SON=-NEWBUC
 160  IF(TREE(2,PARENT).EQ.SON.AND.TREE(4,PARENT).LT.0.OR.TREE(3,PARENT
     1).EQ.SON.AND.TREE(4,PARENT).GT.0) GOTO 170
      SON=PARENT
      PNODES=PNODES-1
      IF(PNODES.LT.0) GOTO 230
      IF(PNODES.LE.0) GOTO 200
      PARENT=SNODES(PNODES)
      GOTO 160
 170  TREE(1,PARENT)=NEWBUC
      NEWBUC=NEWBUC+1
      IF(NEWBUC.GT.NEWMAX) GOTO 270
      IF(PNODES.LE.0) GOTO 200
      PARENT=SNODES(PNODES)
      IF(TREE(4,PARENT).LE.0) GOTO 180
      SON=TREE(2,PARENT)
      GOTO 190
 180  SON=TREE(3,PARENT)
 190  IF(SON.LE.0) GOTO 80
      PARENT=SON
      PNODES=PNODES+1
      IF(PNODES.GT.MNODES) GOTO 210
      SNODES(PNODES)=PARENT
      GOTO 80
 200  RETURN
 210  WRITE(6,220)
 220  FORMAT(' STACKSIZE TOO SMALL')
      STOP
 230  WRITE(6,240)
 240  FORMAT(' STACK NODES UNDERFLOW')
      STOP
 250  WRITE(6,260)
 260  FORMAT(' IMPOSSIBLE')
      STOP
 270  WRITE(6,280) NEWBUC,NEWMAX
 280  FORMAT(' NEWBUC =',I10,' EXCEEDES',I10)
      STOP
 290  WRITE(6,300) OLDSTR,TREE(SON,PARENT)
 300  FORMAT(' LOOKING FOR BUCKET',I10,' BUT FOUND',I10)
      STOP
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE TREDMP(NDIM)
      COMMON /ISTRGE/ MXRGNS,TREE(4,1),DUMMY1(11996)
      COMMON /RSTRGE/ RSTSZE,PARTN(18001)
      INTEGER RSTSZE
      INTEGER TREE
      REAL PARTN
      COMMON /TRESZE/ ENTREE,ENTBUC
      INTEGER ENTREE,ENTBUC
      COMMON /LIMITS/ GMINUS(10),GPLUS(10)
      COMMON /BUKSZE/ MAXWRD
      INTEGER MAXWRD
      REAL UMINUS(10),UPLUS(10)
      INTEGER PARENT
      WRITE(6,10) ENTREE
 10   FORMAT('1TREE HAS ',I5,' NONTERMINAL NODES')
      DO 30 I=1,ENTREE
      WRITE(6,20) I,(TREE(J,I),J=1,4),PARTN(I)
 20   FORMAT(' NONTERMINAL NODE(',I5,') = ',4I5,G13.5)
 30   CONTINUE
      RETURN
      ENTRY BUKDMP(NDIM)
      WRITE(6,40) ENTBUC
 40   FORMAT('1PARTITIONING HAS ',I5,' TOTAL REGIONS')
      K=MXRGNS
      DO 100 I=1,ENTBUC
      WRITE(6,50) I
 50   FORMAT(' REGION',I6)
      PARENT=1
      DO 60 J=1,NDIM
      UMINUS(J)=GMINUS(J)
      UPLUS(J)=GPLUS(J)
 60   CONTINUE
      CALL BOUNDS(I,PARENT,TREE,PARTN,UMINUS,UPLUS)
      DO 80 J=1,NDIM
      WRITE(6,70) UMINUS(J),J,UPLUS(J)
 70   FORMAT(1X,G13.5,' .LT. X(',I2,') .LE. ',G13.5)
 80   CONTINUE
      WRITE(6,90) (PARTN(K+J),J=1,MAXWRD)
 90   FORMAT(' REGION INFORMATION = ',8G13.5/1X,9G13.5/1X,
     1 9G13.5/1X,9G13.5)
      K=K+MAXWRD
 100  CONTINUE
      RETURN
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE TSTEXT (N,XEXTR,XLOW,XUP,EDGFAC,FRACT,NCUT,NCDIM,ICUT,D
     1ELPLS,DELNEG)
      INTEGER N, NCDIM, NCUT
      INTEGER ICUT(NCDIM)
      DOUBLE PRECISION EDGFAC, FRACT
      DOUBLE PRECISION XEXTR(N), XLOW(N), XUP(N), DELPLS(N), DELNEG(N)
      INTEGER I
      DOUBLE PRECISION DIFX
      SAVE
      NCUT=0
      DO 10 I=1,N
      DIFX=XUP(I)-XEXTR(I)
      DELPLS(I)=DIFX
*      write(*,*) 'delpls(i),i,xup(i),xextr(i)',delpls(i),i,xup(i)
*     &     ,xextr(i)
      IF(DIFX.LT.EDGFAC*(XUP(I)-XLOW(I))) GOTO 10
      DELPLS(I)=FRACT*DIFX
*      write(*,*) 'delpls(i),i,fract',delpls(i),i,fract
      NCUT=NCUT+1
      ICUT(NCUT)=I
 10   CONTINUE
      DO 20 I=1,N
      DIFX=XEXTR(I)-XLOW(I)
      DELNEG(I)=DIFX
*      write(*,*) 'delneg(i),i,xup(i),xextr(i)',delneg(i),i,xup(i)
*     &     ,xextr(i)
      IF(DIFX.LT.EDGFAC*(XUP(I)-XLOW(I))) GOTO 20
      DELNEG(I)=FRACT*DIFX
*      write(*,*) 'delpls(i),i,fract',delpls(i),i,fract
      NCUT=NCUT+1
      ICUT(NCUT)=-I
 20   CONTINUE
      RETURN
      END








*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE USRINT (UMINUS,UPLUS,INTIN,SPRDIN,INTOUT,ERROUT)
      REAL INTIN,INTOUT
      SAVE
      WRITE(6,10)
 10   FORMAT(' DUMMY USER INTEGRATION SUBROUTINE CALLED.',
     1' INCONSISTENT INPUT.')
      STOP
      END
*CMZ :          23/08/93  13.30.10  by  Jonathan Butterworth
*-- Author :
      LOGICAL FUNCTION USRTRM(ITER)
      COMMON /ANSWER/ FINTGL,SPRD,DUMMY(5),NRGN,MAXRGN /FUNN/ NFUN,MO(2)
      COMMON /Z0001/ ERR,NMIN,MCOUNT /PRINT/ IPR /ZEETRM/ ITRMF
      INTEGER COUNT
      SAVE
      DATA FAC /1.0/
      USRTRM=.FALSE.
      IF(ITRMF.EQ.0) RETURN
      ERROR=ABS(ERR)
      IF(ERR.GT.0.0) ERROR=ERROR*FINTGL
      NPR=SPRD*FAC/ERROR+0.5
      IF(NPR.GT.1999) RETURN
      N=NFUN+NPR*NRGN
      IF(N.GE.NMIN) GOTO 10
      NMIN=N
      COUNT=0
      RETURN
 10   COUNT=COUNT+1
      IF(COUNT.GT.MCOUNT) USRTRM=.TRUE.
      IF(.NOT.(USRTRM).OR.IPR.LE.0) GOTO 30
      WRITE(6,20)
 20   FORMAT('0--- DIVON --- PARTITIONING TERMINATION.')
 30   RETURN
      END
      DOUBLE PRECISION FUNCTION DFUN( N, xy )
C ----------------------------------------------------------------------------
C           Top level function call: integrated by DIVON4.
C           What function is actually called depends upon the value of
C           FN_TYPE in the JMFLAG common block.
C
C           3 = Eikonalising gamma-p xsec
C           4 = Calculating xsec for there being N (& only N) scatters.
C         101 = Calculating the total (uneikonalised) xsec.
C ----------------------------------------------------------------------------

      IMPLICIT NONE

      include 'JIMMY.INC'


      INTEGER N,i
      DOUBLE PRECISION XY(N)
      DOUBLE PRECISION JMKERN, EIKFUN, JMSNFN

      IF (FN_TYPE.EQ.101) THEN

C --    Uneikonalised cross section
        DFUN = JMKERN(N,XY)

      ELSE IF (FN_TYPE.EQ.3) THEN

C --    Eikonalising total xsec
        DFUN = EIKFUN(N,XY)

      ELSE IF (FN_TYPE.EQ.4) THEN

C --    Calculating xsec for N scatters
        DFUN = JMSNFN(N,XY)

      ELSE

C --    Illegal call.
        WRITE (JMOUT,*) 'DFUN:Illegal value of FN_TYPE!'
        STOP

      ENDIF

      RETURN
      END
*CMZ :          05/01/95  11.18.45  by  Jonathan Butterworth
*-- Author :
      SUBROUTINE DVNOPT
C --------------------------------------------------------------------------
C      CERNLIB INTEGRATION OPTIONS
C --------------------------------------------------------------------------

      include 'JIMMY.INC'


      COMMON/QUADRE/IDEG
      COMMON/PRINT/IPRINT
      COMMON/BNDLMT/FLOW,FHIGH
C
      IDEG   = 1
C
      IF (JMBUG.LT.1) THEN
        IPRINT = 0
      ELSE
        IPRINT = 1
      ENDIF
C
      FLOW = 0.0
C
      RETURN
      END
*CMZ :          25/04/95  15.01.42  by  Unknown
*-- Author : Jeff Forshaw
      DOUBLE PRECISION FUNCTION EIKFUN(NDIM,V)
C ----------------------------------------------------------------------------
C  Purpose: Used when eikonalising the total gamma-p xsec.
C ----------------------------------------------------------------------------

      include 'HERWIG65.INC'
      include 'JIMMY.INC'


      INTEGER NDIM

      DOUBLE PRECISION XSECN,V(NDIM),B2MAX,B2,FAC1
      DOUBLE PRECISION JMAREA, AREA
      COMMON/BLOCK1/XSECN

      B2MAX=5.D2
      FAC1=B2MAX
      B2=FAC1*V(1)
      AREA = JMAREA(B2)
      EIKFUN=PIFAC/PHAD*(1.D0-DEXP(-AREA*XSECN*PHAD))*FAC1

      RETURN
      END
      DOUBLE PRECISION FUNCTION EIKON( DUMMY )
C     --------------------------------------------------------------------
C     Purpose: To eikonalise the total hadronic cross section pass the
C     cross section to be eikonalised (XSECN) and the routine returns
C     the eikonalised version (EIKON)
C     --------------------------------------------------------------------
      IMPLICIT NONE

      include 'JIMMY.INC'

      DOUBLE PRECISION EPS, XSECN, DUMMY
      INTEGER NDIM, MAXPTS, MINPTS, LENWRK, I
      PARAMETER(NDIM=2,MAXPTS=10000*NDIM,
     & LENWRK=(NDIM+2)*(1+MAXPTS/(2**NDIM+2*NDIM*NDIM+2*NDIM+1)))
      REAL A(NDIM), B(NDIM), RESULT, ERROR

      COMMON/BLOCK1/XSECN               !COMMON local to eikfun

      XSECN = DUMMY
      DO 10 I=1,NDIM
        A(I)=0.0
        B(I)=1.0
  10  CONTINUE

      MINPTS = 0
      EPS    = 1.D-3
      FN_TYPE = 3
      CALL PARTN( NDIM, A, B, 1.0, 10000 )
      CALL INTGRL( NDIM, 0, 200, RESULT, ERROR )
      EIKON = DBLE(RESULT)

      RETURN
      END


      DOUBLE PRECISION FUNCTION EPFUN( Y, I )
C     -----------------------------------------------------------------
C     Purpose: To construct the inclusive & eikonalised hadronic cross
C     section at the given Z. The bremstralung factor for photons in not
C     included.
C     -----------------------------------------------------------------
      include 'HERWIG65.INC'
      include 'JIMMY.INC'

      INTEGER I, J
      DOUBLE PRECISION Y, Z, JMSHAT, BREMFAC, JMSN
      DOUBLE PRECISION EIKON, QCDRES, ARG1, ARG2, JMFWW

      Z = Y
      JMSHAT=Z*2.D0*(EBEAM1*EBEAM2+PBEAM1*PBEAM2)

c     Uneikonalised cross section
      ARG1=QCDRES( Z )

c     Eikonalise the cross section
      ARG2=EIKON( ARG1 )

C --  Store the p(n) values too.
      DO J=1,MAXMS
        IF (ARG2.GT.1.0D-17) THEN
           IF (JMUEO.NE.0) THEN
              JMARRY( 4+J, I ) = JMSN( ARG1, J )/ARG1
           ELSE
              JMARRY( 4+J, I ) = JMSN( ARG1, J )/ARG2
           ENDIF
        ELSE
          JMARRY( 4+J, I ) = 0.D0
        ENDIF
      ENDDO

      IF (I.LE.NPSIMP) THEN

C       Store (and write out) intermediate results
C       for the xsecn.
        WRITE(JMOUT,801) ARG1*GEV2NB/1000.0,ARG2*GEV2NB/1000.0, JMSHAT
        JMARRY( 1, I ) = Z
C       Store uneikonalised cross section.
        JMARRY( 2, I ) = ARG1
C       Store eikonalised cross section.
        JMARRY( 3, I ) = ARG2

      ELSE

        WRITE(JMOUT,*) 'TOO MANY CALLS TO EPFUN!-FATAL'
        STOP

      ENDIF

      EPFUN = ARG2

      IF (JMBUG.GT.2) THEN
c         WRITE(JMOUT,*) Z,ARG1*BREMFAC,ARG2*BREMFAC
         WRITE(JMOUT,*) Z,ARG1,ARG2
     &,JMARRY(5,I),JMARRY(6,I),JMARRY(7,I),JMARRY(8,I),JMARRY(9,I)
      ENDIF

      RETURN
 801  FORMAT(2X,'HADRONIC C-S: UNEIKONALIZED=',G8.2
     &     ,'ub, EIKONALIZED=',G8.2,'ub, at s =',G8.2,' GeV2')
      END


      SUBROUTINE HWMSCT(ABORT)
C -----------------------------------------------------------------
C Administer multiple scattering
C
C If the returned argument (ABORT) is TRUE then the event has been
C vetoed by the eikonalisation, and should be abandoned.
C -----------------------------------------------------------------
      include 'HERWIG65.INC'
      include 'JIMMY.INC'

      LOGICAL ABORT
      INTEGER N, I, J, REPORT, CHECKSUM, CHECKEV, COUNTER
      INTEGER TMPPR
      LOGICAL FIRST
      DATA FIRST /.TRUE./
      SAVE FIRST, CHECKSUM, CHECKEV

      IF (FIRST) THEN
         CHECKSUM=0.
         CHECKEV=0.
         FIRST=.FALSE.
      ENDIF

C -- Abort the event if there are any errors.
      IF (IERROR.NE.0) THEN
        ABORT=.TRUE.
        CALL HWUFNE
        RETURN
      ENDIF

      ABORT = .FALSE.

C     Decide how many hard scatters
C     If NSCAT=0, this event was rejected (this can happen because the 
C     cross section at a given centre-of-mass energy changes with 
C     eikonalisation).
C     J returns the index of the upper S bound.
      CALL HWNSCT(J)

      IF (JMBUG.GT.0) THEN
         WRITE(*,*) 'HWMSCT:NUMBER OF SCATTERS REQUESTED =',NSCAT    
         CHECKSUM=CHECKSUM+NSCAT
         IF (NSCAT.GT.0) THEN
            CHECKEV=CHECKEV+1
            WRITE(*,*) 'RUNNING AVERAGE=',FLOAT(CHECKSUM)/FLOAT(CHECKEV)
         ENDIF
      ENDIF

      IF (NSCAT.EQ.0) THEN
C --     Finish event (making sure aborted events aren't printed)
         TMPPR=MAXPR
         MAXPR=0
         CALL HWUFNE
         MAXPR=TMPPR
         ABORT = .TRUE.
         NEVHEP=NEVHEP-1
         RETURN
      ENDIF

      REPORT = 0
      COUNTER = 0
      N = NSCAT

      DO I = 2, N

 10      CONTINUE

         IF (REPORT.EQ.0) THEN
            
            CALL HWHSCT(REPORT,(I.EQ.2),JMUEO,PTJIM)

            IF (REPORT.EQ.5) THEN
               WRITE(*,*) 'FATAL ERROR'
               RETURN
            ENDIF

c     If there's an error which isn't a fatal error, retry.
            IF (REPORT.NE.0) THEN

               IF (JMBUG.GT.0) WRITE(*,*) 'Report=',report
               REPORT=0
               IF (COUNTER.LT.MAXMSTRY) THEN
                  COUNTER=COUNTER+1
                  GOTO 10
               ELSE
c     Avoid infinite loops. If there's really no phase space, reduce
c     the number of scatters.     
                  NSCAT=NSCAT-1
                  IF (JMBUG.GT.0) WRITE(*,*) 'LOST A SCATTER'
                  COUNTER=0
               ENDIF
            ENDIF

            IF (ANOMOFF) THEN
               
               IF (ANOMSC(1,1).NE.0.OR.ANOMSC(1,2).NE.0)THEN
                  WRITE(JMOUT,*) 'Anom. scat',ANOMSC(1,1),ANOMSC(1,2)
                  NSCAT=NSCAT-1
                  REPORT=6
               ENDIF
               
            ENDIF
            
         ENDIF
         
      ENDDO

      IF (JMBUG.GT.0) THEN
         WRITE(*,*) 'HWMSCT:NUMBER OF SCATTERS GENERATED =',NSCAT
      ENDIF

 100  CONTINUE

C     =====================================================================
C     Store the number of events lost from the HERWIG cross section.
C     (which equals the number of "multiple" scatters)
      TOTSCAT = TOTSCAT + NSCAT
      IF (JMUEO.EQ.0) THEN
         NLOST   = NLOST + (NSCAT - 1.D0)
      ENDIF
C     Get the S-hat distribution correct. To do this we must alter
C     the amount of vetoing of events, as this is based initially on
C     the simple eikonal model without taking into account "lost"
C     scatters.
C     We should decrease the chances of an event being rejected by
C     a factor of NSCAT(S)/N(S) where NSCAT is the number of scatters
C     actually generated at this S, and N(S) is the number that would have
C     been generated so far by the "simple" model.
C     Store these numbers in JMARRY(MAXMS+5,I) and JMARRY(MAXMS+6,I)

c     Turn this feature off if the process being generated is not
c     eikonalised itself (i.e. MI are being used only for the
c     underlying event.
      JMARRY(MAXMS+5,J)   = JMARRY(MAXMS+5,J)+FLOAT(NSCAT)
      JMARRY(MAXMS+6,J)   = JMARRY(MAXMS+6,J)+FLOAT(N)

      RETURN
      END




*CMZ :          17/07/95  11.18.11  by  Jonathan Butterworth
*-- Author : JMB  03/12/92
      SUBROUTINE HWNSCT(J)
C --------------------------------------------------------------------------
C Purpose: Choose  how many scatters in this event (NSCAT).
C          If the number of scatters returned is 0, the event
C          should be rejected due to the reductiion of the cross
C          by eikonalisation.
C          J (return argument) is the index of the upper edge of the
C          bin in s in which we lie.
C          If the chosen particles are at a fixed cms, J is always 1
C --------------------------------------------------------------------------

      include 'HERWIG65.INC'
      include 'JIMMY.INC'

      DOUBLE PRECISION Z, EIK, UNEIK, JMGAMZ, TEST
      DOUBLE PRECISION JMRNDM, DICE, SOFAR, PROBS(MAXMS)
      INTEGER I, J, DUM, IERR
      LOGICAL FIRST
      DATA FIRST /.TRUE./
      SAVE FIRST
      REAL tmp1, tmp2

      NSCAT=0
      IF (IERROR.NE.0) RETURN

      IERR = 0

C --  Z is the fraction of the beam-beam centre-of-mass energy (squared)
C     available in the subcollision. 
      IF (JCMVAR.EQ.0) THEN
        Z=-1.D0
        J=1
      ELSE
        Z=PHEP(5,3)**2/(2.D0*(EBEAM1*EBEAM2+PBEAM1*PBEAM2))
      ENDIF
C
C --  Stage one: Keep the event?
      IF (JMUEO.NE.0) THEN
         TEST = 1.
      ELSE
         EIK=JMGAMZ(Z,2,J)
         UNEIK=JMGAMZ(Z,1,J)
         TEST=EIK/UNEIK      
C     Adjust the probability of zero scatters to the fact that some
C     scatters are lost in event generation.
         IF (JMARRY(5+MAXMS,J).GT.0.AND.JMUEO.EQ.0) THEN
            TEST=TEST*JMARRY(6+MAXMS,J)/JMARRY(5+MAXMS,J)
         ENDIF
         DICE = JMRNDM(2,DUM)
         
         IF (DICE.GT.TEST) THEN
C     Event is rejected
            NSCAT = 0
            RETURN
         ENDIF
      ENDIF

C --  Stage two: Decide how many scatters in this event.
C     First find the respective probabilities.
      CALL JMPN( PROBS, Z, J )
      DICE = JMRNDM(2,DUM)
      SOFAR = 0.D0
      IF (JMBUG.GT.0) THEN
         IF (FIRST) THEN
            tmp1=0.
            tmp2=0.
            do i=1,maxms
               write(*,*) 'HWNSCT:Prob ',i,'=',probs(i)
               tmp1=tmp1+probs(i)
               tmp2=tmp2+probs(i)*float(i)
            enddo
            write(*,*) 'Summed prob =',tmp1
            write(*,*) 'Weighted average prob =',tmp2/float(maxms)
            first=.false.
         ENDIF
         IF (JMBUG.GT.2) write(*,*) 'HWNSCT: dice=',dice
      ENDIF
      DO I = 1, MAXMS
        SOFAR = SOFAR + PROBS(I)
        IF (DICE.LT.SOFAR) THEN
          NSCAT = I
          GOTO 11
        ENDIF
      ENDDO
      WRITE(JMOUT,*)
     &    '*** HWNSCT WARNING:TOO MANY SCATTERS IN EVENT.'
      NSCAT = MAXMS

 11   CONTINUE

      END



      SUBROUTINE JIMMIN
C --------------------------------------------------------------
C Initialise default values for JIMMY. Call just after HWIGIN.
C Author :    Jonathan Butterworth   08/01/95
C --------------------------------------------------------------
      include 'HERWIG65.INC'
      include 'JIMMY.INC'

      INTEGER I

 8900 FORMAT(A)

C     Set defaults
      PHAD=300.0

C     Minimum PT of the secondary scatters
      PTJIM=2.5

C     Defualt debug level to lowest.
      JMBUG=0

C     Number of attempts to regenerate a scatter which
C     is initially vetoed by energy conservation 
      MAXMSTRY=100

C     Master flag for MI
      MSFLAG=1

      DO I=1,264
        JMRAD(I)=0.D0
      ENDDO
C     Radii for particles.
C     (zero if not known - i.e. most of them,
C      e+ e- get the photon radius.)
C     Photon
      JMRAD(59)=4.7D-1
C     Proton and Neutron and antis
      JMRAD(73)=7.1D-1
      JMRAD(75)=7.1D-1
      JMRAD(91)=7.1D-1
      JMRAD(93)=7.1D-1
C     e+e-
      JMRAD(121)=JMRAD(59)
      JMRAD(127)=JMRAD(59)

      ANOMOFF=.TRUE.
      JMUEO=1

      RETURN
      END



      DOUBLE PRECISION FUNCTION JMAREA( B2 )
c ----------------------------------------------------------------------
c   Purpose: Returns area overlap A(B)
c      Mods: 20-Aug-1993 JMB. Allow use of CERNLIB.
c      Mods: 27-Jan-1995  RW. Allow e+e- (not NAGLIB!)
c ----------------------------------------------------------------------

      include 'HERWIG65.INC'
      include 'JIMMY.INC'

      INTEGER NPHOTONS
      DOUBLE PRECISION B2,AREA
      DOUBLE PRECISION JMDBESK0, JMDBESK1, BESSEL(0:3)
      DOUBLE PRECISION X1, X2

      IF (B2.LE.0.D0) THEN
        IF (JMBUG.GT.11) WRITE(JMOUT,*)'**** JMAREA:B2=',B2
        JMAREA = 0.D0
        RETURN
      ENDIF
      X1=(JMV2*B2)**.5
      X2=(JMU2*B2)**.5

      NPHOTONS=0
      IF ((ABS(IDPDG(IPART1)).EQ.11).OR.(ABS(IDPDG(IPART1)).EQ.22))
     $     NPHOTONS=NPHOTONS+1
      IF ((ABS(IDPDG(IPART2)).EQ.11).OR.(ABS(IDPDG(IPART2)).EQ.22))
     $     NPHOTONS=NPHOTONS+1

      IF (NPHOTONS.EQ.1) THEN
C     `ep' type run
C     Modified Bessel function K0
         AREA = JMDBESK0(X1)
         AREA = (AREA-JMDBESK0(X2))*JMU2/(JMU2-JMV2)

C       Modified Bessel function K1
         AREA=(AREA-X2/TWO*JMDBESK1(X2))*JMU2*JMV2/(JMU2-JMV2)/
     &        (TWO*PIFAC)
         
      ELSE IF (NPHOTONS.EQ.2) THEN
         
C     `e+e-' type run.
C     Modified Bessel function K1
         AREA = JMV2*X1*JMDBESK1(X1)/(4.D0*PIFAC)
         
         
      ELSE IF (NPHOTONS.EQ.0) THEN
         
C     `pp' type run.
C     Modified Bessel function K3
         CALL JMDBSKA(X2,0,1,3,BESSEL)
         AREA = JMU2/96/PIFAC*(X2**3)*BESSEL(3)
         
      ENDIF

      JMAREA = AREA

      RETURN
      END
*-- Author : JMB
      SUBROUTINE JMEFIN
C -----------------------------------------------------------------
C     Adjust HERWIG cross section in light of Multiple Scattering.
C     Must be called after (from?) HWEFIN.
C -----------------------------------------------------------------
      include 'HERWIG65.INC'
      include 'JIMMY.INC'

      WRITE (JMOUT,1)
    1 FORMAT(/10X,'MODIFIED OUTPUT ON ELEMENTARY PROCESS'/)

      IF (JMUEO.EQ.0) THEN

         IF (NLOST.EQ.0) THEN
            WRITE (6,10)
         ELSE
            AVWGT=AVWGT*(TOTSCAT-NLOST)/TOTSCAT
            WRITE (6,11) INT(TOTSCAT),INT(NLOST),1000.*AVWGT

         ENDIF
      ELSE
         WRITE (6,13)
         WRITE (6,12) INT(TOTSCAT)         
      ENDIF
         
 10   FORMAT(10X,
     &     'NO MULTIPLE SCATTERS:NO MODIFICATION NECESSARY')
 11      FORMAT(1P,
     &        10X,'NUMBER OF SCATTERS = ',I20/
     &        10X,'NO.OF MULT.SCATTERS= ',I20/
     &        10X,'FINAL C-S (PB)     =',G12.4)
 12      FORMAT(1P,
     &        10X,'NUMBER OF SCATTERS = ',I20)
 13      FORMAT(1P,
     &        10X,'MULTIPLE SCATTERS USED FOR UNDERLYING EVENT'/
     &        10X,'NO CHANGE TO TOTAL CROSS SECTION.')
      END
*CMZ :          31/08/93  17.51.25  by  Jonathan Butterworth
*-- Author :
      DOUBLE PRECISION FUNCTION JMFACT( N )
C --------------------------------------------------------------------------
C Factorial of N, given as double precision to allow for very large numbers.
C --------------------------------------------------------------------------
      IMPLICIT NONE
      INTEGER N, I
      DOUBLE PRECISION TOTAL

      TOTAL = 1.D0
      DO I = 2, N
        TOTAL = TOTAL*DBLE(I)
      ENDDO
      JMFACT = TOTAL
      RETURN
      END
*-- Author : Jon Butterworth
      DOUBLE PRECISION FUNCTION JMGAMZ( Z, WHICH, J )
C -------------------------------------------------------------------
C  Purpose: Returns total gamma-p cross section at given z, by
C           interpolating logarithmically between the points stored
C           during initialisation.
C           If the incoming particles are at fixed CM energy,
C           (e.g. pp) then input should be Z=-1.0, and J is set to 1
C           WHICH = 1 returns uneikonalised xsec
C           WHICH = 2 returns eikonalised xsec
C -------------------------------------------------------------------
      include 'HERWIG65.INC'
      include 'JIMMY.INC'


      DOUBLE PRECISION Z
      DOUBLE PRECISION ZSIN, LOG1, LOG2, Z1, Z2, LOGSIG, M, C
      INTEGER I, J, WHICH, LOC

      JMGAMZ = 0.D0

      IF (IERROR.NE.0) RETURN

      IF (WHICH.NE.1.AND.WHICH.NE.2) THEN
         WRITE(JMOUT,*) 'JMGAMZ called with illegal WHICH flag',WHICH
         STOP
      ENDIF

      LOC = WHICH+1

      IF (Z.LT.-0.5D0) THEN
        J=1
        JMGAMZ=JMARRY(LOC,J)
        RETURN
      ENDIF

      ZSIN = Z
      DO 1, I = 1, NPSIMP

        IF (ZSIN.LE.JMARRY( 1, I )) GOTO 2

 1    CONTINUE

      WRITE(JMOUT,*) 'JMGAMZ: Illegal Z input!', ZSIN, JMARRY(1,NPSIMP)
      IERROR = 200
      RETURN

C     We are interpolating between JMARRY(LOC,I-1) and JMARRY(LOC,I)
 2    CONTINUE

      J=I

C     Do a logarithmic interpolation if possible/sensible.
      IF (JMARRY(LOC,I-1).GT.0.AND.JMARRY(LOC,I).GT.0) THEN

        LOG1 = DLOG( JMARRY(LOC,I-1) )
        LOG2 = DLOG( JMARRY(LOC,I) )
        Z1 = JMARRY(1,I-1)
        Z2 = JMARRY(1,I)

        M = (LOG2-LOG1)/(Z2-Z1)
        C = LOG1-M*Z1

        LOGSIG = M*ZSIN + C

        JMGAMZ = EXP( LOGSIG )

      ELSE

C       Otherwise do a linear one.
        LOG1 = JMARRY(LOC,I-1)
        LOG2 = JMARRY(LOC,I)
        Z1 = JMARRY(1,I-1)
        Z2 = JMARRY(1,I)

        M = (LOG2-LOG1)/(Z2-Z1)
        C = LOG1-M*Z1

        LOGSIG = M*ZSIN + C

        JMGAMZ = LOGSIG

      ENDIF

      RETURN
      END
      SUBROUTINE JMGRID
C     ------------------------------------------------------------------------
C     Purpose: Calculate the total hadronic cross section grid with
C     eikonalisation. This routine uses the functions QCDRES (to
C     calculate the uneikonalised gp cross section) and EIKON (to
C     eikonalise it).
C     ------------------------------------------------------------------------

      include 'HERWIG65.INC'
      include 'JIMMY.INC'

      INTEGER I
      DOUBLE PRECISION XSECN, EPFUN, H, Y, SUM, TMP, DELTA

      DELTA = 0.005D0

      IF (JCMVAR.GT.0) THEN

c     Step-size
         H = (ONE - JMZMIN)/FLOAT(NPSIMP)
         
c     Lower limit of integration
         Y = JMZMIN+DELTA
         
         SUM = EPFUN( Y, 0 )
         
         Y = JMZMIN
         
         DO 2, I=1, NPSIMP/2
            
            Y = Y + H
            TMP = EPFUN(Y, 2*I-1 )*FOUR
            SUM = SUM + TMP
            
            Y = Y + H
            TMP = EPFUN(Y, 2*I )*TWO
            SUM = SUM + TMP
            
 2       CONTINUE
         
C     Get the weight for the end point correct
         SUM = SUM - TMP/TWO
         XSECN = SUM*H/THREE

      ELSE
         
         SUM   = EPFUN( 1.D0, 1 )
         XSECN = SUM
         
      ENDIF
         
      RETURN
   50 FORMAT(8F10.5)
      END



*CMZ :          12/05/95  01.10.55  by  Jonathan Butterworth
*-- Author : JMB 03/12/92
      SUBROUTINE JMINIT
C --------------------------------------------------------------------------
C Purpose:  Run initialisation specific to Jimmy. Call from HERWIG
C           run initialisation (HWEINI).
C --------------------------------------------------------------------------

      include 'HERWIG65.INC'
      include 'JIMMY.INC'


      INTEGER I
      EXTERNAL DFUN, DVNOPT
      DOUBLE PRECISION PHADTMP, SMIN

      NLOST=0
      TOTSCAT=0

C     If MI are not switched on, do nothing.
      IF (MSFLAG.EQ.0) RETURN 



C --  Title page.
      WRITE (JMOUT,1001)

c removed this check for ATLAS (JMB)

C     If MI are not implemented for this process, print warning and set
C     MSFLAG to 0. This is the list of processes with either an incoming
C     lepton or an incoming pointlike photon
      IF ((ABS(IPRO).EQ.10).OR.(ABS(IPRO).EQ.50).OR.
     &    (ABS(IPRO).EQ.51).OR.(ABS(IPRO).EQ.53).OR.
     &    (ABS(IPRO).EQ.55).OR.(ABS(IPRO).EQ.60).OR.
     &    (ABS(IPRO).EQ.90).OR.(ABS(IPRO).EQ.91).OR.
     &    (ABS(IPRO).EQ.95)) THEN
         
         WRITE(JMOUT,1006) 
     &        'JMINIT: Multiparton interactions make no sense'
     &        ,'        for this process: Turning them off.'
         MSFLAG=0
         RETURN

      ELSE IF (ABS(IPRO).EQ.15) THEN
         
c     MI fine and can be used either in the full eikonal or in the
c     underlying event approximation, depending on the users setting of
c     JMUEO.

         WRITE(JMOUT,1006) 'Multiparton interactions being used. '

         IF (JMUEO.EQ.0) THEN
            PTJIM=PTMIN
         ELSE 
            JMUEO=1
            IF (2.0*PTJIM.GT.PTMIN) THEN
               WRITE(6,*)
     $              'JMINIT: WARNING. PTJIM AND PTMIN ARE TOO SIMILAR.'
               WRITE(6,*)
     $              'JMINIT: RECOMMEND YOU USE JMUEO=0'
            ENDIF

            WRITE(JMOUT,1006)
     &      'JMINIT: Multiparton interactions being used in the '
            WRITE(JMOUT,1006)
     &      '        underlying event under high ET jets'
            
         ENDIF

      ELSE

c     MI approx.
         WRITE(JMOUT,1006)
     &      'JMINIT: Multiparton interactions being used in the '
         WRITE(JMOUT,1006)
     &     '         underlying event only.'
         WRITE(JMOUT,1006)
     &     '         Main process is not eikonalised. '

         JMUEO=2

      ENDIF

      WRITE(JMOUT,1008) PTJIM

C --  Set up the radii appropriately (unless they have been set by the user).
      IF (JMV2.EQ.0) JMV2=JMRAD(IPART1)
      IF (JMU2.EQ.0) JMU2=JMRAD(IPART2)
      IF (JMV2.EQ.0) THEN
        WRITE(JMOUT,8001) 1
	WRITE(JMOUT,*) IPART1,IPART2
	WRITE(JMOUT,*) JMRAD
        JMV2=JMRAD(127)
      ENDIF
      IF (JMU2.EQ.0) THEN
        WRITE(JMOUT,8001) 2
        JMU2=JMRAD(127)
      ENDIF

C --  Print out radii to be used.
      IF (IPART1.EQ.91.OR.IPART1.EQ.73) THEN
         IF (IPART2.EQ.91.OR.IPART2.EQ.73) THEN
            WRITE(JMOUT,1009) JMU2
         ELSE
            WRITE(JMOUT,1009) JMV2
            WRITE(JMOUT,1010) JMU2
         ENDIF
      ELSE IF (IPART2.EQ.91.OR.IPART2.EQ.73) THEN
         WRITE(JMOUT,1009) JMU2
         WRITE(JMOUT,1010) JMV2
      ELSE 
         WRITE(JMOUT,1010) JMV2
      ENDIF


C --  Decide whether the energy of 0,1 or 2 of the beams is varying.
C     This assumes that if the beam is an electron, we are dealing with
C     a spectrum of photon energies, but if it is a photon it
C     is monoenergetic. All other beams are monoenergetic hadrons.
      JCMVAR=0
      PHADTMP=PHAD
      PHAD=1.0
      IF (ABS(IDPDG(IPART1)).EQ.11) THEN
         JCMVAR=JCMVAR+1
         PHAD=PHAD*PHADTMP
      ELSE IF (IDPDG(IPART1).EQ.22) THEN
         PHAD=PHAD*PHADTMP
      ENDIF
      IF (ABS(IDPDG(IPART2)).EQ.11) THEN
         JCMVAR=JCMVAR+1
         PHAD=PHAD*PHADTMP
      ELSE IF (IDPDG(IPART2).EQ.22) THEN
         PHAD=PHAD*PHADTMP
      ENDIF

C     Calculate cross section(s)
      WRITE(JMOUT,1005)

c     Minimum allowed Z=S(hadhad)/S(beambeam).
      IF (JCMVAR.EQ.0) THEN
         JMZMIN=1.0
      ELSE
         CALL HWEGAS(SMIN)
         IF (JCMVAR.EQ.1) THEN
            JMZMIN=MAX(YWWMIN,SMIN/
     &           (2.0*(EBEAM1*EBEAM2+PBEAM1*PBEAM2)))
         ELSE IF (JCMVAR.EQ.2) THEN
C     Both particle energies are varying.
            JMZMIN=MAX(YWWMIN**2,SMIN/
     &           (2.0*(EBEAM1*EBEAM2+PBEAM1*PBEAM2)))
         ENDIF
      ENDIF

      CALL JMGRID

c     Restore the photon/hadron probability.
      PHAD=PHADTMP

      IF (JMBUG.GT.0) THEN
        WRITE(JMOUT,1006) 'Z values'
        DO I = 1, NPSIMP
          WRITE(JMOUT,1007) JMARRY(1,I)
        ENDDO
        WRITE(JMOUT,1006) 'UnEikonalised xsec'
        DO I = 1, NPSIMP
          WRITE(JMOUT,1007) JMARRY(2,I)*GEV2NB
        ENDDO
        WRITE(JMOUT,1006) 'Eikonalised xsec'
        DO I = 1, NPSIMP
          WRITE(JMOUT,1007) JMARRY(3,I)*GEV2NB
        ENDDO
      ENDIF

      RETURN

 1001 FORMAT( 8X,'----------------------------------------------'/,
     &        8X,'                J  I  M  M  Y               '/,
     &        8X,'   Inclusion of  Eikonal multiple scattering '/,
     &        8X,'                Version 4.3            '/,
     &        8X,'    Requires HERWIG 6.510 or above   '/,
     &        8X,'----------------------------------------------')
 1005 FORMAT( 8X,'       GENERATING CROSS-SECTION GRID         ')
 1006 FORMAT(8X,A)
 1007 FORMAT(F20.5)
 1008 FORMAT(8X,'MINIMUM PT FOR SECONDARY SCATTERS=',G8.2,'GeV')
 1009 FORMAT(8X,'1/RADIUS OF PROTON AND ANTIPROTON=',G8.2,'GeV')
 1010 FORMAT(8X,'1/RADIUS OF PHOTON               =',G8.2,'GeV')

 8001 FORMAT(8X,'JIMMY:NO RADIUS FOR PARTICLE ',I2,':USING PHOTON')

 999  CONTINUE
      END
*-- Author : JRF/JMB
      DOUBLE PRECISION FUNCTION JMKERN( NDIM, V )
C -------------------------------------------------------------
C   Purpose: The kernel for the integral over PT,X1,X2
C            The integral goes from pt2tmp -> ptmax2.
C            When using this function in the initialsation,
C            pt2tmp == ptmin**2 and ptmax2 = the kinematic limit.
C            Otherwise they are the pt2 of the current
C            hard scatter and of the last, respectively.
C -------------------------------------------------------------

      include 'HERWIG65.INC'
      include 'JIMMY.INC'


      INTEGER NDIM
      DOUBLE PRECISION PT2, PT2TMP, PTMAX2, V(NDIM)
      DOUBLE PRECISION JMXS1, FAC1, FAC2
      DOUBLE PRECISION FAC3, LX1, LX2, LPT2
      DOUBLE PRECISION X1,X2,JMS

c     Beam-beam centre-of-mass energy.
      JMS = 2.D0*(EBEAM1*EBEAM2+PBEAM1*PBEAM2)

      IF (FN_TYPE.NE.101) THEN
        WRITE(JMOUT,8902) 'JMKERN:THIS SHOULD NEVER HAPPEN!'
        RETURN
      ENDIF

      PT2TMP = PTJIM**2
C     (This value is never used
C     - it is always greater than the kinematic limit)
      PTMAX2 = JMS/4.D0 - RMASS(1)**2

C     Define region of integration
      FAC1 = -DLOG( 4.D0*PT2TMP/(YGAMMA*JMS) )
      IF ( FAC1 .LE. 0.D0 ) THEN
        JMKERN = 0.D0
        GOTO 8900
      ENDIF

      IF (JMBUG.GT.2) THEN
        write(JMOUT,8902) 'v(1),ygamma,jms,pt2tmp,fac1'
        write(JMOUT,*) v(1),ygamma,jms,pt2tmp
     &     ,fac1
      ENDIF

      LX1 = DLOG( 4.D0*PT2TMP/(YGAMMA*JMS)) +V(1)*FAC1
      X1 = DEXP( LX1 )
      IF ( X1 .GE. ONE ) THEN
        JMKERN = 0.D0
        GOTO 8900
      ENDIF

      FAC2 = -DLOG( 4.D0*PT2TMP/(YGAMMA*X1*JMS) )
      IF ( FAC2 .LE. 0.D0 ) THEN
        JMKERN = 0.D0
        GOTO 8900
      ENDIF

      LX2 = DLOG( 4.D0*PT2TMP/(YGAMMA*JMS*X1) ) + V(2)*FAC2
      X2 = DEXP( LX2 )
      IF ( X2 .GE. 1.D0 ) THEN
        JMKERN = 0.D0
        GOTO 8900
      ENDIF

      IF (NDIM.EQ.3) THEN
        PTMAX2 = MIN(JMS*YGAMMA*X1*X2/4.D0,PTMAX2)
        FAC3 = DLOG( PTMAX2 ) - DLOG( PT2TMP )
        IF ( FAC3 .LE. SMALL ) THEN
          JMKERN = 0.D0
          GOTO 8900
        ENDIF

        LPT2 = DLOG( PT2TMP ) + V(3)*FAC3
        PT2  = DEXP( LPT2 )
        IF (JMBUG.GT.2) THEN
          write(JMOUT,8902) 'lx1,x1,lx2,x2,pt2'
          write(JMOUT,*) lx1,x1,lx2,x2,pt2
        ENDIF
        JMKERN = X1*X2*PT2*FAC1*FAC2*FAC3*
     &      PHAD * JMXS1(X1,X2,PT2,0,0)

	IF (JMBUG.GT.2) THEN
          write(JMOUT,8902) 'JMKERN,X1,X2,PT2,FAC1,FAC2,FAC3,PHAD'
          write(JMOUT,*) JMKERN,X1,X2,PT2,FAC1,FAC2,FAC3,PHAD
	ENDIF

        IF ( JMKERN .LE. 0.D-30 ) JMKERN = 0.D0

      ELSE

         WRITE(JMOUT,8902) 'JMKERN ERROR'

      ENDIF

 8900 CONTINUE

      IF (JMBUG.GT.2) THEN
        WRITE(JMOUT,8901) JMKERN,X1,X2,PT2
      ENDIF
 8901 FORMAT
     & (1X,'XS=',F12.6,' X1=',F10.8,' X2=',F10.8,' PT2=',F10.3)
 8902 FORMAT(A)
      RETURN
      END
      SUBROUTINE JMKNIF(X1,X2,PT2,VETO)
C --------------------------------------------------------------------------
C Enforce virtual mass cutoffs of the structure function values
C generated.
C --------------------------------------------------------------------------
      include 'HERWIG65.INC'
      include 'JIMMY.INC'


      INTEGER I
      DOUBLE PRECISION HWBVMC
      DOUBLE PRECISION JMS, X1, X2, PT2, QL1, QL2
      LOGICAL VETO

      VETO = .FALSE.

      JMS=2.D0*(EBEAM1*EBEAM2+PBEAM1*PBEAM2)

C     Demand that there's enough energy to produce at least a
C     pair of quarks. We can do this seperately for each subprocess
C     based upon their final state masses, but this will do for now.
      IF (PT2+RMASS(1)**2.GT.(X1*X2*YGAMMA*JMS)/4.D0) THEN
        VETO = .TRUE.
	IF (JMBUG.GT.3) THEN
          WRITE(JMOUT,8900) 'JMVETO: Not enough energy. Vetoing.'
	  WRITE(JMOUT,*) PT2,RMASS(1),(X1*X2*YGAMMA*JMS)/4.D0
	ENDIF
        RETURN
      ENDIF

C     Now apply the initial state cutoffs, process by process.
      QL1 = (1.D0-X1)*EMSCA
      QL2 = (1.D0-X2)*EMSCA

      DO I = 1, 13
        JMVETO(1,I) = 1.
        JMVETO(2,I) = 1.
        IF (QL1.LT.HWBVMC(I)) JMVETO(1,I)=0.
        IF (QL2.LT.HWBVMC(I)) JMVETO(2,I)=0.
      ENDDO

 8900 FORMAT(A)
      RETURN
      END



*CMZ :          17/07/95  11.08.55  by  Jonathan Butterworth
*-- Author : J. M. Butterworth
      SUBROUTINE JMPN( PROBS, ZDUB, K )
C --------------------------------------------------------------------------
C  Purpose: Returns the probabilities of N scatters at a given z, by
C           interpolating between the points stored during initialisation.
C           Returned arg. J is the lower edge of the s bin we are in.
C --------------------------------------------------------------------------
      include 'HERWIG65.INC'
      include 'JIMMY.INC'


      DOUBLE PRECISION ZDUB
      DOUBLE PRECISION Z, PROBS( MAXMS )
      DOUBLE PRECISION JMSC1, JMSC2, Z1, Z2, SCSIG, M, C
      INTEGER I, J, K

      Z=ZDUB
      DO I=1,MAXMS
         PROBS(I)=0.D0
      ENDDO

      IF (IERROR.NE.0) RETURN

      IF (ZDUB.LT.-0.5D0) THEN
        DO J=1,MAXMS
           PROBS(J)=JMARRY(4+J,1)
        ENDDO
        RETURN
      ENDIF

      DO 1, I=1, NPSIMP

        IF (Z.LE.JMARRY( 1, I )) GOTO 2

 1    CONTINUE

      WRITE(JMOUT,*) 'JMPN: ILLEGAL Z INPUT! ',Z
      IERROR=200
      RETURN

C     We are interpolating between JMARRY(J,I-1) and JMARRY(J,I)
C     Linear interpolation.
 2    CONTINUE

      K=I

      DO J=1, MAXMS

        IF (JMARRY(4+J,I-1).GT.0.0) THEN
          JMSC1 = JMARRY(4+J,I-1)
        ENDIF

        IF (JMARRY(4+J,I).GT.0.D0) THEN
          JMSC2 = JMARRY(4+J,I)
        ENDIF

        Z1 = JMARRY(1,I-1)
        Z2 = JMARRY(1,I)

        M = (JMSC2-JMSC1)/(Z2-Z1)
        C = JMSC1-M*Z1

        SCSIG = M*Z + C

        PROBS(J) = SCSIG

      ENDDO

      RETURN
      END
      DOUBLE PRECISION FUNCTION JMRNDM(WOT,SEED)
C ------------------------------------------------------------------------
C  Random number generator interface.
C  1) SET  2) GENERATE UNIFORM RANDOM NUMBER 3) RETURN CURRENT SEEDS
C ------------------------------------------------------------------------

      IMPLICIT NONE

      include 'JIMMY.INC'


      INTEGER WOT, SEED(2)
      DOUBLE PRECISION HWRGEN, HWRGET, HWRSET

      IF (WOT.EQ.1) THEN

C       Initialise
        JMRNDM = HWRSET(SEED)

      ELSE IF (WOT.EQ.2) THEN

C       Generate a random number
        JMRNDM = HWRGEN(SEED)

      ELSE IF (WOT.EQ.3) THEN

C       Return current seed.
        JMRNDM = HWRGET(SEED)

      ELSE

        WRITE(JMOUT,*) 'JMRNDM CALLED WITH ILLEGAL WOT=',WOT
        STOP

      ENDIF
      RETURN
      END
      DOUBLE PRECISION FUNCTION JMSN( DUMMY1, DUMMY2 )
C --------------------------------------------------------------------
C  Purpose: To calculate the cross section for N and
C           only N scatters.
C --------------------------------------------------------------------

      IMPLICIT NONE

      include 'JIMMY.INC'


      INTEGER N, DUMMY2
      DOUBLE PRECISION JMSNFN, EPS, XSECN, DUMMY1
      INTEGER NDIM, MAXPTS, MINPTS, LENWRK, I
      PARAMETER(NDIM=2,MAXPTS=10000*NDIM,
     *   LENWRK=(NDIM+2)*(1+MAXPTS/(2**NDIM+2*NDIM*NDIM+2*NDIM+1)))

      REAL A(NDIM), B(NDIM), RESULT, ERROR

      EXTERNAL JMSNFN
      COMMON/BLOCK2/ XSECN, N       ! COMMON local to JMSNFN

      XSECN = DUMMY1
      N = DUMMY2
      DO 10 I=1,NDIM
        A(I)=0.0
        B(I)=1.0
  10  CONTINUE

      MINPTS = 0
      EPS    = 1.D-3
      FN_TYPE = 4
      CALL PARTN( NDIM, A, B, 1.0, 10000 )
      CALL INTGRL( NDIM, 0, 200, RESULT, ERROR )
      IF (JMBUG.GT.0) write(*,*) 'jmsn: result, error',result,error
      JMSN = DBLE(RESULT)

      RETURN
      END
      DOUBLE PRECISION FUNCTION JMSNFN(NDIM,V)
C ----------------------------------------------------------------------------
C  Purpose: Used when eikonalising the total hadronic xsec.
C ----------------------------------------------------------------------------

      include 'HERWIG65.INC'
      include 'JIMMY.INC'


      INTEGER NDIM, N
      DOUBLE PRECISION XSECN, V(NDIM), B2MAX, B2, FAC1
      DOUBLE PRECISION CHI, JMAREA, JMFACT
      COMMON/BLOCK2/XSECN, N

      B2MAX = 5.D2
      FAC1  = B2MAX
      B2    = FAC1*V(1)

      CHI = JMAREA(B2)*XSECN*PHAD

      IF (JMBUG.GT.10) THEN
         WRITE(*,*) 'JMSNFN:XSECN,CHI',XSECN,CHI
      ENDIF

      IF (JMUEO.EQ.0) THEN
c     Standard (i.e. useless!) JIMMY. Generating QCD 2->2 events
c     totally democratically.
         JMSNFN = PIFAC/PHAD*FAC1*CHI**N/JMFACT(N)*DEXP(-CHI)
      ELSE
c     JIMMY being used to simulate the underlying event to a special
c     (low cross section) process.
         JMSNFN = N*PIFAC/PHAD*FAC1*CHI**N/JMFACT(N)*DEXP(-CHI)         
      ENDIF

      IF (JMBUG.GT.10) THEN
         WRITE(*,*) 'JMSNFN:N,JMFACT(N),JMSNFN',N,JMFACT(N),JMSNFN
      ENDIF

      RETURN
      END
      DOUBLE PRECISION FUNCTION JMXS1( X1, X2, PT2, FLAG1, FLAG2 )
c ------------------------------------------------------------------
c  Purpose: The fully differential cross section dsigma/dx1dx2dt
c           for E-P resolved scattering WITHOUT eikonal integral.
c
c   Inputs: x1, x2  -> "Bjorken" x's of photon and proton,
c                       respectively
c           pt2     ->  pt2 of the partons
c           flag1   ->  If flag1=0 then no YGAMMA included,
c           flag2   ->  If flag2=0 calculate dxsec/d(pt2)
c
c Author: JRF/JMB
c   Mods: JRF Oct '93 - Include different parton processes.
c ------------------------------------------------------------------
      include 'HERWIG65.INC'
      include 'JIMMY.INC'


      INTEGER I, J, NF, FLAG1, FLAG2, PID(4)
      DOUBLE PRECISION Y, PT2, X1, X2, S, T, U, Q2, G1
      DOUBLE PRECISION U1, UU1, D1, DD1, S1, SS1, G2, U2, UU2
      DOUBLE PRECISION D2, DD2, S2, SS2, SCL2, ALPHA, F, F1, F2
      DOUBLE PRECISION X, C1, CC1, C2, CC2, JMPTOT
      DOUBLE PRECISION SMIN, SCALE
      DOUBLE PRECISION DIST(13), HWUALF
      LOGICAL VETO

      IF (JMBUG.GT.3) THEN
        WRITE(JMOUT,8900) 'JMXS1 called'
      ENDIF

c     Centre-of-mass energy for this parton-parton collision.
      S = 2.D0*(EBEAM1*EBEAM2+PBEAM1*PBEAM2)*YGAMMA*X1*X2

c     Appropriate kinematics
      Y = 4.*PT2/S

c     Define Mandelstam variables in parton subprocess
      IF (Y .GE. .99D0) THEN
        IF (FLAG2.EQ.0) THEN
c         We are calculating dxsec/d(pt2) and are at the pole.
	  IF (JMBUG.GT.3) THEN
            WRITE(JMOUT,8900) 'JMXS1 at the pole. ZERO'
          ENDIF
          JMXS1 = 0.D0
          RETURN
        ELSE
          T = -S/2.D0
        ENDIF
      ELSE
        T = -S/2.*(1.0D0-(1.0D0-Y)**.5)
      ENDIF

      U=-S-T
c     Select Scale
      EMSCA = SQRT(TWO*S*T*U/(S*S+T*T+U*U))
      SCALE = EMSCA
      Q2    = EMSCA**2

      IF ((X1 .GE. 0.99D0).OR.(X2 .GE. 0.99D0)) THEN
	IF (JMBUG.GT.3) THEN
          WRITE(JMOUT,8900) 'JMXS1 zero: x out of upper range'
	ENDIF
        JMXS1=0.D0
        RETURN
      ENDIF
      IF ((X1 .EQ. 0.0D0).OR.(X2 .EQ. 0.D0)) THEN
	IF (JMBUG.GT.3) THEN
          WRITE(JMOUT,8900) 'JMXS1 zero: x out of lower range'
	ENDIF
        JMXS1=0.D0
        RETURN
      ENDIF

c     Apply detailed phase space cutoffs.
      CALL JMKNIF(X1,X2,PT2,VETO)
      IF (VETO) THEN
	IF (JMBUG.GT.3) THEN
          WRITE(JMOUT,8900) 'JMXS1 vetoed by jmknif'
	ENDIF
        JMXS1 = 0.D0
        RETURN
      ENDIF

c     BEAM1 pdfs, with applied cutoffs on x1
      X = X1
      IF (JMBUG.GT.3) THEN
        write(JMOUT,8900) 'x1,x,scale'
        write(JMOUT,*) x1,x,scale
      ENDIF
      IF (IPART1.EQ.121.OR.IPART1.EQ.127) THEN
C       photon PDF
        CALL HWSFUN(X,SCALE,59,NSTRU,DIST,1)
      ELSE
        CALL HWSFUN(X,SCALE,IPART1,NSTRU,DIST,1)
      ENDIF
      IF (JMBUG.GT.3) THEN
        write(JMOUT,8900) 'x1,x,scale,dist'
        write(JMOUT,*) x2,x,scale,dist
      ENDIF
      G1  = DIST(13)/X1*JMVETO(1,13)
      U1  = DIST(2)/X1*JMVETO(1,2)
      UU1 = DIST(8)/X1*JMVETO(1,8)
      D1  = DIST(1)/X1*JMVETO(1,1)
      DD1 = DIST(7)/X1*JMVETO(1,7)
      S1  = DIST(3)/X1*JMVETO(1,3)
      SS1 = DIST(9)/X1*JMVETO(1,9)
      C1  = DIST(4)/X1*JMVETO(1,4)
      CC1 = DIST(10)/X1*JMVETO(1,10)

c     BEAM2 pdfs, with applied cutoffs on x2
      X = X2
*      write(*,*) 'x2,x,scale',x2,x,scale
      IF (IPART2.EQ.121.OR.IPART2.EQ.127) THEN
C       photon PDF
        CALL HWSFUN(X,SCALE,59,NSTRU,DIST,2)
      ELSE
C       proton PDF
        CALL HWSFUN(X,SCALE,IPART2,NSTRU,DIST,2)
      ENDIF
*      write(*,*) 'x2,x,scale,dist',x2,x,scale,dist
      G2  = DIST(13)/X2*JMVETO(2,13)
      U2  = DIST(2)/X2*JMVETO(2,2)
      UU2 = DIST(8)/X2*JMVETO(2,8)
      D2  = DIST(1)/X2*JMVETO(2,1)
      DD2 = DIST(7)/X2*JMVETO(2,7)
      S2  = DIST(3)/X2*JMVETO(2,3)
      SS2 = DIST(9)/X2*JMVETO(2,9)
      C2  = DIST(4)/X2*JMVETO(2,4)
      CC2 = DIST(10)/X2*JMVETO(2,10)

C     Define LAMBDA_QCD, ALPHA_S and number of flavours.
c      NF = NFLAV
c      SCL2  = QCDLAM**2
c      ALPHA = 4.D0*PIFAC/(11.D0-2.D0/3.D0*NF)/DLOG(Q2/SCL2)
c      write(*,*) 'alpha 1',alpha
      ALPHA = HWUALF(1,EMSCA)
c      write(*,*) 'alpha 2',alpha

C********************************************************************
C**** JMPROC(1:117) CONTAINS THE PROBABILITIES FOR ALL 117         **
C**** SUBPROCESSES.                                                **
C**** NOTATION TO BE USED IN REST OF THIS COMMENT IS:              **
C**** P(i) = ij-kl means that P(i) is the probability for ij=>kl   **
C**** Where i,j,k and l are parton types (u,d,s,c for quarks,      **
C**** U,D,S,C for antiquarks and g for gluons) and i refers to     **
C**** partons originating from the photon whilst j refers to those **
C**** originating from the proton. Each probability is colour and  **
C**** spin averaged and the Mandelstam variable, t, is defined by  **
C**** the momenta of i and k (or j and l). The hard scale must be  **
C**** t-u symmetric since we always use t-u symmetrised cross      **
C**** sections (nb: p_T is t-u symmetric)                          **
C********************************************************************
C
C**** JMPROC(1) = gg-gg
C
C**** JMPROC(2) to JMPROC(5)  =  gg-qQ  (all equal)
C
C**** JMPROC(6) to JMPROC(13) = qq-qq, QQ-QQ (i.e. uu-uu,dd-dd,..,UU-UU,..,CC-CC)
C
C**** JMPROC(14) to JMPROC(61) = qq'-qq', qQ'-qQ', Qq'-Qq', QQ'-QQ'
C
C**** JMPROC(62) to JMPROC(93) = qQ-(uU, dD, sS, cC), Qq-(uU, dD, sS, cC)
C
C**** JMPROC(94) to JMPROC(101)  = qQ-gg, Qq-gg
C
C**** JMPROC(102) to JMPROC(117) = gq-gq, qg-gq, gQ-gQ, Qg-gQ
C
C**** THE SUM OF THE JMPROC(i)'s RESIDES IN JMPTOT
C********************************************************************
C       gg-gg
      JMPROC(1)=9./2.*(3.0D0-t*u/s/s-s*u/t/t-s*t/u/u)*g1*g2
C       gg-qQ
      JMPROC(2)=(t/u+u/t-9./4.*(t*t+u*u)/s/s)/3.d0*g1*g2
      JMPROC(3)=JMPROC(2)
      JMPROC(4)=JMPROC(2)
      JMPROC(5)=JMPROC(2)
C       qq-qq,QQ-QQ
        f=4./9.*((s*s+u*u)/t/t+(s*s+t*t)/u/u)-8./27.*s*s/t/u
      JMPROC(6)=u1*u2*f
      JMPROC(7)=d1*d2*f
      JMPROC(8)=s1*s2*f
      JMPROC(9)=c1*c2*f
      JMPROC(10)=uu1*uu2*f
      JMPROC(11)=dd1*dd2*f
      JMPROC(12)=ss1*ss2*f
      JMPROC(13)=cc1*cc2*f
C       qq'-qq', qQ'-qQ', Qq'-Qq', QQ'-QQ'
        f=4./9.*((s*s+u*u)/t/t+(s*s+t*t)/u/u)
      JMPROC(14)=u1*d2*f
      JMPROC(15)=u1*s2*f
      JMPROC(16)=u1*c2*f
      JMPROC(17)=d1*u2*f
      JMPROC(18)=d1*s2*f
      JMPROC(19)=d1*c2*f
      JMPROC(20)=s1*u2*f
      JMPROC(21)=s1*d2*f
      JMPROC(22)=s1*c2*f
      JMPROC(23)=c1*u2*f
      JMPROC(24)=c1*d2*f
      JMPROC(25)=c1*s2*f
      JMPROC(26)=u1*dd2*f
      JMPROC(27)=u1*ss2*f
      JMPROC(28)=u1*cc2*f
      JMPROC(29)=d1*uu2*f
      JMPROC(30)=d1*ss2*f
      JMPROC(31)=d1*cc2*f
      JMPROC(32)=s1*uu2*f
      JMPROC(33)=s1*dd2*f
      JMPROC(34)=s1*cc2*f
      JMPROC(35)=c1*uu2*f
      JMPROC(36)=c1*dd2*f
      JMPROC(37)=c1*ss2*f
      JMPROC(38)=uu1*d2*f
      JMPROC(39)=uu1*s2*f
      JMPROC(40)=uu1*c2*f
      JMPROC(41)=dd1*u2*f
      JMPROC(42)=dd1*s2*f
      JMPROC(43)=dd1*c2*f
      JMPROC(44)=ss1*u2*f
      JMPROC(45)=ss1*d2*f
      JMPROC(46)=ss1*c2*f
      JMPROC(47)=cc1*u2*f
      JMPROC(48)=cc1*d2*f
      JMPROC(49)=cc1*s2*f
      JMPROC(50)=uu1*dd2*f
      JMPROC(51)=uu1*ss2*f
      JMPROC(52)=uu1*cc2*f
      JMPROC(53)=dd1*uu2*f
      JMPROC(54)=dd1*ss2*f
      JMPROC(55)=dd1*cc2*f
      JMPROC(56)=ss1*uu2*f
      JMPROC(57)=ss1*dd2*f
      JMPROC(58)=ss1*cc2*f
      JMPROC(59)=cc1*uu2*f
      JMPROC(60)=cc1*dd2*f
      JMPROC(61)=cc1*ss2*f
C       qQ-(uU, dD, sS, cC), Qq-(uU, dD, sS, cC)
        f1=4./9.*(u*u+t*t)/s/s*2.d0
        f2=f1+4./9.*((s*s+u*u)/t/t+(s*s+t*t)/u/u)-8./27.*
     *     (u*u/s/t+t*t/s/u)
      JMPROC(62)=u1*uu2*f2
      JMPROC(63)=u1*uu2*f1
      JMPROC(64)=JMPROC(63)
      JMPROC(65)=JMPROC(63)
      JMPROC(66)=d1*dd2*f1
      JMPROC(67)=d1*dd2*f2
      JMPROC(68)=JMPROC(66)
      JMPROC(69)=JMPROC(66)
      JMPROC(70)=s1*ss2*f1
      JMPROC(71)=JMPROC(70)
      JMPROC(72)=s1*ss2*f2
      JMPROC(73)=JMPROC(70)
      JMPROC(74)=c1*cc2*f1
      JMPROC(75)=JMPROC(74)
      JMPROC(76)=JMPROC(74)
      JMPROC(77)=c1*cc2*f2
      JMPROC(78)=uu1*u2*f2
      JMPROC(79)=uu1*u2*f1
      JMPROC(80)=JMPROC(79)
      JMPROC(81)=JMPROC(79)
      JMPROC(82)=dd1*d2*f1
      JMPROC(83)=dd1*d2*f2
      JMPROC(84)=JMPROC(82)
      JMPROC(85)=JMPROC(82)
      JMPROC(86)=ss1*s2*f1
      JMPROC(87)=JMPROC(86)
      JMPROC(88)=ss1*s2*f2
      JMPROC(89)=JMPROC(86)
      JMPROC(90)=cc1*c2*f1
      JMPROC(91)=JMPROC(90)
      JMPROC(92)=JMPROC(90)
      JMPROC(93)=cc1*c2*f2
C       qQ-gg, Qq-gg
        f=32./27.*(t/u+u/t)-8./3.*(t*t+u*u)/s/s
      JMPROC(94)=u1*uu2*f
      JMPROC(95)=d1*dd2*f
      JMPROC(96)=s1*ss2*f
      JMPROC(97)=c1*cc2*f
      JMPROC(98)=uu1*u2*f
      JMPROC(99)=dd1*d2*f
      JMPROC(100)=ss1*s2*f
      JMPROC(101)=cc1*c2*f
C       gq-gq, qg-gq, gQ-gQ, Qg-gQ
        f=(s*s+u*u)/t/t+(s*s+t*t)/u/u-4./9.*(s/u+u/s+s/t+t/s)
      JMPROC(102)=g1*u2*f
      JMPROC(103)=g1*d2*f
      JMPROC(104)=g1*s2*f
      JMPROC(105)=g1*c2*f
      JMPROC(106)=g2*u1*f
      JMPROC(107)=g2*d1*f
      JMPROC(108)=g2*s1*f
      JMPROC(109)=g2*c1*f
      JMPROC(110)=g1*uu2*f
      JMPROC(111)=g1*dd2*f
      JMPROC(112)=g1*ss2*f
      JMPROC(113)=g1*cc2*f
      JMPROC(114)=g2*uu1*f
      JMPROC(115)=g2*dd1*f
      JMPROC(116)=g2*ss1*f
      JMPROC(117)=g2*cc1*f

      JMPTOT=0.D0
      DO I=1,117
C       Parton type from photon.
*        PID(1) = JMPTYP(I)/1000000
C       Parton type from proton.
*        PID(2) = MOD(JMPTYP(I)/10000,100)
C       Scattered partons
        PID(3) = MOD(JMPTYP(I)/100,100)
        PID(4) = MOD(JMPTYP(I),100)
        DO J=3,4
          IF (PID(J).EQ.0) THEN
            PID(J)=13
          ELSE IF (PID(J).GT.10) THEN
            PID(J) = PID(J)-4
          ENDIF
        ENDDO
        SMIN = RMASS(PID(3))**2 + RMASS(PID(4))**2+ TWO*PTJIM**2+TWO
     $       * SQRT(PTJIM**2+RMASS(PID(3))**2)*SQRT(PTJIM**2+RMASS(PID(4
     $       ))**2)
        IF (S.LT.SMIN) THEN
          JMPROC(I)=0.0
        ELSE
          JMPTOT=JMPTOT+JMPROC(I)
        ENDIF
      ENDDO

      IF (FLAG2.EQ.0) THEN
C       Calculate dxsec/d(pt2)
        JMXS1=JMPTOT*PIFAC*ALPHA**2/(S**2 * SQRT(1.D0-Y))
      ELSE
        WRITE(JMOUT,8900) 'JMXS1 Called with illegal flag2'
        WRITE(JMOUT,*) flag2
        JMXS1=0.D0
      ENDIF

C     WEISZACKER-WILLIAMS flux factor.
C     Only Want This When NOT integrating (i.e. never again)
C     (flag=0 otherwise)
      IF (FLAG1.NE.0) THEN
        WRITE(JMOUT,8900) 'JMXS1 ERROR'
      ENDIF

 8900 FORMAT(A)

      RETURN
      END
      DOUBLE PRECISION FUNCTION QCDRES( Z )
C     ---------------------------------------------------------------------
C     Purpose: To calculate the inclusive hadron(or photon)+hadron(or
C     photon) => JETS 
C     ---------------------------------------------------------------------
      include 'HERWIG65.INC'
      include 'JIMMY.INC'

      INTEGER I, MINPTS, NDIM, MAXPTS, LENWRK

C     Set parameters for integration
      PARAMETER(NDIM=3,MAXPTS=10000*NDIM,
     *   LENWRK=(NDIM+2)*(1+MAXPTS/(2**NDIM+2*NDIM*NDIM+2*NDIM+1)))
      DOUBLE PRECISION Z, EPS
      EXTERNAL JMKERN
      REAL A(NDIM), B(NDIM)
      REAL RESULT, ERROR

C     INITIALISE INTEGRATION LIMITS (ALL 0 TO 1)
      DO 10 I=1, NDIM
        A(I)=0.0
        B(I)=1.0
  10  CONTINUE

      MINPTS=0
      EPS=1.0D-3
C     Store the current z value
      YGAMMA=Z

      FN_TYPE=101
      CALL PARTN( NDIM, A, B, 1.0, 10000 )
      CALL INTGRL( NDIM, 0, 200, RESULT, ERROR )
      QCDRES=DBLE(RESULT)

C     Re-include the PHAD factor (JMKERN divides out the probability of the
C     photon becoming a hadronic = 1/phad)
      QCDRES = QCDRES/PHAD

      RETURN

      END







