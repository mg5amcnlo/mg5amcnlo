MODULE special_fun
CONTAINS
  SUBROUTINE HYGFZ(A,B,C,Z,ZHF)
!                                                                                                                                                                                                         
!       ======================================================                                                                                                                                            
!       Purpose: Compute the hypergeometric function for a                                                                                                                                                
!                complex argument, F(a,b,c,z)                                                                                                                                                             
!       Input :  a --- Parameter                                                                                                                                                                          
!                b --- Parameter                                                                                                                                                                          
!                c --- Parameter,  c <> 0,-1,-2,...                                                                                                                                                       
!                z --- Complex argument                                                                                                                                                                   
!       Output:  ZHF --- F(a,b,c,z)                                                                                                                                                                       
!       Routines called:                                                                                                                                                                                  
!            (1) GAMMA for computing gamma function                                                                                                                                                       
!            (2) PSI for computing psi function                                                                                                                                                           
!       ======================================================                                                                                                                                            
!                                                                                                                                                                                                         
    IMPLICIT NONE
    LOGICAL::L0,L1,L2,L3,L4,L5,L6
    REAL(KIND(1d0)),INTENT(INOUT)::A,B,C
    COMPLEX(KIND(1d0)),INTENT(INOUT)::Z
    COMPLEX(KIND(1d0)),INTENT(OUT)::ZHF
    REAL(KIND(1d0))::X,Y,EPS,AA,BB,A0,PI,EL,CA,CB,GAB,GABC,GBA
    REAL(KIND(1d0))::GC,GA,GB,GCAB,GCA,GCB,G0,G1,G2,G3,GCBK
    REAL(KIND(1d0))::X1,GM,GAM,GBM,PA,PB,RM,SM,SP0,SP,PAC,PCA
    INTEGER::NM,K,MCAB,M,J,MAB,NCA,NCB
    COMPLEX(KIND(1d0))::ZR,Z1,ZC0,ZR0,ZW,ZF0,Z00,ZC1,ZF1,ZP,ZP0,ZR1
    REAL(KIND(1d0))::RK1,RK2,SJ1,SJ2,SQ,T0,W0,WS
    X=REAL(Z)
    Y=DIMAG(Z)
    EPS=1.0D-15
    L0=C.EQ.INT(C).AND.C.LT.0.0D0
    L1=DABS(1.0D0-X).LT.EPS.AND.Y.EQ.0.0D0.AND.C-A-B.LE.0.0D0
    L2=CDABS(Z+1.0D0).LT.EPS.AND.DABS(C-A+B-1.0D0).LT.EPS
    L3=A.EQ.INT(A).AND.A.LT.0.0D0
    L4=B.EQ.INT(B).AND.B.LT.0.0D0
    L5=C-A.EQ.INT(C-A).AND.C-A.LE.0.0D0
    L6=C-B.EQ.INT(C-B).AND.C-B.LE.0.0D0
    AA=A
    BB=B
    A0=CDABS(Z)
    IF (A0.GT.0.95D0) EPS=1.0D-8
    PI=3.141592653589793D0
    EL=.5772156649015329D0
    IF (L0.OR.L1) THEN
       WRITE(*,*)'The hypergeometric series is divergent'
       RETURN
    ENDIF
    IF (A0.EQ.0.0D0.OR.A.EQ.0.0D0.OR.B.EQ.0.0D0) THEN
       ZHF=(1.0D0,0.0D0)
    ELSE IF (Z.EQ.1.0D0.AND.C-A-B.GT.0.0D0) THEN
       CALL GAMMA(C,GC)
       CALL GAMMA(C-A-B,GCAB)
       CALL GAMMA(C-A,GCA)
       CALL GAMMA(C-B,GCB)
       ZHF=GC*GCAB/(GCA*GCB)
    ELSE IF (L2) THEN
       G0=DSQRT(PI)*2.0D0**(-A)
       CALL GAMMA(C,G1)
       CALL GAMMA(1.0D0+A/2.0D0-B,G2)
       CALL GAMMA(0.5D0+0.5D0*A,G3)
       ZHF=G0*G1/(G2*G3)
    ELSE IF (L3.OR.L4) THEN
       IF (L3) NM=INT(ABS(A))
       IF (L4) NM=INT(ABS(B))
       ZHF=(1.0D0,0.0D0)
       ZR=(1.0D0,0.0D0)
       DO K=1,NM
          ZR=ZR*(A+K-1.0D0)*(B+K-1.0D0)/(K*(C+K-1.0D0))*Z
          ZHF=ZHF+ZR ! 10
       ENDDO
    ELSE IF (L5.OR.L6) THEN
       IF (L5) NM=INT(ABS(C-A))
       IF (L6) NM=INT(ABS(C-B))
       ZHF=(1.0D0,0.0D0)
       ZR=(1.0D0,0.0D0)
       DO K=1,NM
          ZR=ZR*(C-A+K-1.0D0)*(C-B+K-1.0D0)/(K*(C+K-1.0D0))*Z
          ZHF=ZHF+ZR ! 15
       ENDDO
       ZHF=(1.0D0-Z)**(C-A-B)*ZHF
    ELSE IF (A0.LE.1.0D0) THEN
       IF (X.LT.0.0D0) THEN
          Z1=Z/(Z-1.0D0)
          IF (C.GT.A.AND.B.LT.A.AND.B.GT.0.0) THEN
             A=BB
             B=AA
          ENDIF
          ZC0=1.0D0/((1.0D0-Z)**A)
          ZHF=(1.0D0,0.0D0)
          ZR0=(1.0D0,0.0D0)
          DO K=1,500
             ZR0=ZR0*(A+K-1.0D0)*(C-B+K-1.0D0)/(K*(C+K-1.0D0))*Z1
             ZHF=ZHF+ZR0
             IF (CDABS(ZHF-ZW).LT.CDABS(ZHF)*EPS)EXIT ! GO TO 25
             ZW=ZHF ! 20
          ENDDO
          ZHF=ZC0*ZHF ! 25
       ELSE IF (A0.GE.0.90D0) THEN
          GM=0.0D0
          MCAB=INT(C-A-B+EPS*DSIGN(1.0D0,C-A-B))
          IF (DABS(C-A-B-MCAB).LT.EPS) THEN
             M=INT(C-A-B)
             CALL GAMMA(A,GA)
             CALL GAMMA(B,GB)
             CALL GAMMA(C,GC)
             CALL GAMMA(A+M,GAM)
             CALL GAMMA(B+M,GBM)
             CALL PSI(A,PA)
             CALL PSI(B,PB)
             IF (M.NE.0) GM=1.0D0
             DO J=1,ABS(M)-1
                GM=GM*J ! 30
             ENDDO
             RM=1.0D0
             DO J=1,ABS(M)
                RM=RM*J ! 35
             ENDDO
             ZF0=(1.0D0,0.0D0)
             ZR0=(1.0D0,0.0D0)
             ZR1=(1.0D0,0.0D0)
             SP0=0.D0
             SP=0.0D0
             IF (M.GE.0) THEN
                ZC0=GM*GC/(GAM*GBM)
                ZC1=-GC*(Z-1.0D0)**M/(GA*GB*RM)
                DO K=1,M-1
                   ZR0=ZR0*(A+K-1.D0)*(B+K-1.D0)/(K*(K-M))*(1.D0-Z)
                   ZF0=ZF0+ZR0 ! 40
                ENDDO
                DO K=1,M
                   SP0=SP0+1.0D0/(A+K-1.0D0)+1.0/(B+K-1.0D0)-1.D0/K ! 45
                ENDDO
                ZF1=PA+PB+SP0+2.0D0*EL+CDLOG(1.0D0-Z)
                DO K=1,500
                   SP=SP+(1.0D0-A)/(K*(A+K-1.0D0))+(1.0D0-B)/&
                        (K*(B+K-1.0D0))
                   SM=0.0D0
                   DO J=1,M
                      SM=SM+(1.0D0-A)/((J+K)*(A+J+K-1.0D0)) &
                           +1.0D0/(B+J+K-1.0D0)
                      !    CONTINUE 50
                   ENDDO
                   ZP=PA+PB+2.0D0*EL+SP+SM+CDLOG(1.0D0-Z)
                   ZR1=ZR1*(A+M+K-1.0D0)*(B+M+K-1.0D0)/(K*(M+K)) &
                        *(1.0D0-Z)
                   ZF1=ZF1+ZR1*ZP
                   IF (CDABS(ZF1-ZW).LT.CDABS(ZF1)*EPS)EXIT ! GO TO 60
                   ZW=ZF1 ! 55
                ENDDO
                ZHF=ZF0*ZC0+ZF1*ZC1 ! 60
             ELSE IF (M.LT.0) THEN
                M=-M
                ZC0=GM*GC/(GA*GB*(1.0D0-Z)**M)
                ZC1=-(-1)**M*GC/(GAM*GBM*RM)
                DO K=1,M-1
                   ZR0=ZR0*(A-M+K-1.0D0)*(B-M+K-1.0D0)/(K*(K-M))&
                        *(1.0D0-Z)
                   ZF0=ZF0+ZR0 ! 65
                ENDDO
                DO K=1,M
                   SP0=SP0+1.0D0/K ! 70
                ENDDO
                ZF1=PA+PB-SP0+2.0D0*EL+CDLOG(1.0D0-Z)
                DO K=1,500
                   SP=SP+(1.0D0-A)/(K*(A+K-1.0D0))+(1.0D0-B)/(K*&
                        (B+K-1.0D0))
                   SM=0.0D0
                   DO J=1,M
                      SM=SM+1.0D0/(J+K) ! 75
                   ENDDO
                   ZP=PA+PB+2.0D0*EL+SP-SM+CDLOG(1.0D0-Z)
                   ZR1=ZR1*(A+K-1.D0)*(B+K-1.D0)/(K*(M+K))*(1.D0-Z)
                   ZF1=ZF1+ZR1*ZP
                   IF (CDABS(ZF1-ZW).LT.CDABS(ZF1)*EPS)EXIT ! GO TO 85
                   ZW=ZF1 ! 80
                ENDDO
                ZHF=ZF0*ZC0+ZF1*ZC1 ! 85
             ENDIF
          ELSE
             CALL GAMMA(A,GA)
             CALL GAMMA(B,GB)
             CALL GAMMA(C,GC)
             CALL GAMMA(C-A,GCA)
             CALL GAMMA(C-B,GCB)
             CALL GAMMA(C-A-B,GCAB)
             CALL GAMMA(A+B-C,GABC)
             ZC0=GC*GCAB/(GCA*GCB)
             ZC1=GC*GABC/(GA*GB)*(1.0D0-Z)**(C-A-B)
             ZHF=(0.0D0,0.0D0)
             ZR0=ZC0
             ZR1=ZC1
             DO K=1,500
                ZR0=ZR0*(A+K-1.D0)*(B+K-1.D0)/(K*(A+B-C+K))*(1.D0-Z)
                ZR1=ZR1*(C-A+K-1.0D0)*(C-B+K-1.0D0)/(K*(C-A-B+K))&
                     *(1.0D0-Z)
                ZHF=ZHF+ZR0+ZR1
                IF (CDABS(ZHF-ZW).LT.CDABS(ZHF)*EPS)EXIT ! GO TO 95
                ZW=ZHF ! 90
             ENDDO
             ZHF=ZHF+ZC0+ZC1 ! 95
          ENDIF
       ELSE
          Z00=(1.0D0,0.0D0)
          IF (C-A.LT.A.AND.C-B.LT.B) THEN
             Z00=(1.0D0-Z)**(C-A-B)
             A=C-A
             B=C-B
          ENDIF
          ZHF=(1.0D0,0.D0)
          ZR=(1.0D0,0.0D0)
          DO K=1,1500
             ZR=ZR*(A+K-1.0D0)*(B+K-1.0D0)/(K*(C+K-1.0D0))*Z
             ZHF=ZHF+ZR
             IF(CDABS(ZHF-ZW).LE.CDABS(ZHF)*EPS)EXIT !  GO TO 105
             ZW=ZHF ! 100
          ENDDO
          ZHF=Z00*ZHF ! 105
       ENDIF
    ELSE IF (A0.GT.1.0D0) THEN
       MAB=INT(A-B+EPS*DSIGN(1.0D0,A-B))
       IF (DABS(A-B-MAB).LT.EPS.AND.A0.LE.1.1D0) B=B+EPS
       IF (DABS(A-B-MAB).GT.EPS) THEN
          CALL GAMMA(A,GA)
          CALL GAMMA(B,GB)
          CALL GAMMA(C,GC)
          CALL GAMMA(A-B,GAB)
          CALL GAMMA(B-A,GBA)
          CALL GAMMA(C-A,GCA)
          CALL GAMMA(C-B,GCB)
          ZC0=GC*GBA/(GCA*GB*(-Z)**A)
          ZC1=GC*GAB/(GCB*GA*(-Z)**B)
          ZR0=ZC0
          ZR1=ZC1
          ZHF=(0.0D0,0.0D0)
          DO K=1,500
             ZR0=ZR0*(A+K-1.0D0)*(A-C+K)/((A-B+K)*K*Z)
             ZR1=ZR1*(B+K-1.0D0)*(B-C+K)/((B-A+K)*K*Z)
             ZHF=ZHF+ZR0+ZR1
             IF (CDABS((ZHF-ZW)/ZHF).LE.EPS)EXIT ! GO TO 115
             ZW=ZHF ! 110
          ENDDO
          ZHF=ZHF+ZC0+ZC1 ! 115
       ELSE
          IF (A-B.LT.0.0D0) THEN
             A=BB
             B=AA
          ENDIF
          CA=C-A
          CB=C-B
          NCA=INT(CA+EPS*DSIGN(1.0D0,CA))
          NCB=INT(CB+EPS*DSIGN(1.0D0,CB))
          IF (DABS(CA-NCA).LT.EPS.OR.DABS(CB-NCB).LT.EPS) C=C+EPS
          CALL GAMMA(A,GA)
          CALL GAMMA(C,GC)
          CALL GAMMA(C-B,GCB)
          CALL PSI(A,PA)
          CALL PSI(C-A,PCA)
          CALL PSI(A-C,PAC)
          MAB=INT(A-B+EPS)
          ZC0=GC/(GA*(-Z)**B)
          CALL GAMMA(A-B,GM)
          ZF0=GM/GCB*ZC0
          ZR=ZC0
          DO K=1,MAB-1
             ZR=ZR*(B+K-1.0D0)/(K*Z)
             T0=A-B-K
             CALL GAMMA(T0,G0)
             CALL GAMMA(C-B-K,GCBK)
             ZF0=ZF0+ZR*G0/GCBK ! 120
          ENDDO
          IF (MAB.EQ.0) ZF0=(0.0D0,0.0D0)
          ZC1=GC/(GA*GCB*(-Z)**A)
          SP=-2.0D0*EL-PA-PCA
          DO J=1,MAB
             SP=SP+1.0D0/J ! 125
          ENDDO
          ZP0=SP+CDLOG(-Z)
          SQ=1.0D0
          DO J=1,MAB
             SQ=SQ*(B+J-1.0D0)*(B-C+J)/J ! 130
          ENDDO
          ZF1=(SQ*ZP0)*ZC1
          ZR=ZC1
          RK1=1.0D0
          SJ1=0.0D0
          DO K=1,10000
             ZR=ZR/Z
             RK1=RK1*(B+K-1.0D0)*(B-C+K)/(K*K)
             RK2=RK1
             DO J=K+1,K+MAB
                RK2=RK2*(B+J-1.0D0)*(B-C+J)/J ! 135
             ENDDO
             SJ1=SJ1+(A-1.0D0)/(K*(A+K-1.0D0))+(A-C-1.0D0)/&
                  (K*(A-C+K-1.0D0))
             SJ2=SJ1
             DO J=K+1,K+MAB
                SJ2=SJ2+1.0D0/J ! 140
             ENDDO
             ZP=-2.0D0*EL-PA-PAC+SJ2-1.0D0/(K+A-C)&
                  -PI/DTAN(PI*(K+A-C))+CDLOG(-Z)
             ZF1=ZF1+RK2*ZR*ZP
             WS=CDABS(ZF1)
             IF (DABS((WS-W0)/WS).LT.EPS)EXIT ! GO TO 150
             W0=WS ! 145
          ENDDO
          ZHF=ZF0+ZF1 ! 150
       ENDIF
    ENDIF
    A=AA ! 155
    B=BB
    IF (K.GT.150) WRITE(*,160)
160 FORMAT(1X,'Warning! You should check the accuracy in HYGFZ')
    RETURN
  END SUBROUTINE HYGFZ

  SUBROUTINE HYGFX(A,B,C,X,HF)
! 
!       ==================================================== 
!       Purpose: Compute hypergeometric function F(a,b,c,x),2F1                                                                                                
!       Input :  a --- Parameter 
!                b --- Parameter  
!                c --- Parameter, c <> 0,-1,-2,...      
!                x --- Argument   ( x < 1 )     
!       Output:  HF --- F(a,b,c,x) 
!       Routines called: 
!            (1) GAMMA for computing gamma function 
!            (2) PSI for computing psi function 
!       ==================================================== 
!
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(INOUT)::A,B,C,X
    REAL(KIND(1d0)),INTENT(OUT)::HF
    LOGICAL::L0,L1,L2,L3,L4,L5
    REAL(KIND(1d0)),PARAMETER::PI=3.141592653589793D0
    REAL(KIND(1d0)),PARAMETER::EL=.5772156649015329D0
    REAL(KIND(1d0))::EPS,GC,GB,GCAB,GCA,GCB,GABC,G1,G2,G3,R
    REAL(KIND(1d0))::AA,BB,X1,GM,GAM,GBM,RM,F0,R0,R1,SP0,SP
    REAL(KIND(1d0))::A0,SM,C0,C1,F1,G0,GA,HW,PA,PB,RP
    INTEGER::NM,K,M,J
    L0=C.EQ.INT(C).AND.C.LT.0.0
    L1=1.0D0-X.LT.1.0D-15.AND.C-A-B.LE.0.0
    L2=A.EQ.INT(A).AND.A.LT.0.0
    L3=B.EQ.INT(B).AND.B.LT.0.0
    L4=C-A.EQ.INT(C-A).AND.C-A.LE.0.0
    L5=C-B.EQ.INT(C-B).AND.C-B.LE.0.0
    IF (L0.OR.L1) THEN
       WRITE(*,*)'The hypergeometric series is divergent'
       RETURN
    ENDIF
    EPS=1.0D-15
    IF (X.GT.0.95)EPS=1.0D-8
    IF (X.EQ.0.0.OR.A.EQ.0.0.OR.B.EQ.0.0) THEN
       HF=1.0D0
       RETURN
    ELSE IF (1.0D0-X.EQ.EPS.AND.C-A-B.GT.0.0) THEN
       CALL GAMMA(C,GC)
       CALL GAMMA(C-A-B,GCAB)
       CALL GAMMA(C-A,GCA)
       CALL GAMMA(C-B,GCB)
       HF=GC*GCAB/(GCA*GCB)
       RETURN
    ELSE IF (1.0D0+X.LE.EPS.AND.DABS(C-A+B-1.0).LE.EPS) THEN
       G0=DSQRT(PI)*2.0D0**(-A)
       CALL GAMMA(C,G1)
       CALL GAMMA(1.0D0+A/2.0-B,G2)
       CALL GAMMA(0.5D0+0.5*A,G3)
       HF=G0*G1/(G2*G3)
       RETURN
    ELSE IF (L2.OR.L3) THEN
       IF (L2) NM=INT(ABS(A))
       IF (L3) NM=INT(ABS(B))
       HF=1.0D0
       R=1.0D0
       DO K=1,NM
          R=R*(A+K-1.0D0)*(B+K-1.0D0)/(K*(C+K-1.0D0))*X
          HF=HF+R ! 10
       ENDDO
       RETURN
    ELSE IF (L4.OR.L5) THEN
       IF (L4) NM=INT(ABS(C-A))
       IF (L5) NM=INT(ABS(C-B))
       HF=1.0D0
       R=1.0D0
       DO K=1,NM
          R=R*(C-A+K-1.0D0)*(C-B+K-1.0D0)/(K*(C+K-1.0D0))*X
          HF=HF+R ! 15
       ENDDO
       HF=(1.0D0-X)**(C-A-B)*HF
       RETURN
    ENDIF
    AA=A
    BB=B
    X1=X
    IF (X.LT.0.0D0) THEN
       X=X/(X-1.0D0)
       IF (C.GT.A.AND.B.LT.A.AND.B.GT.0.0) THEN
          A=BB
          B=AA
       ENDIF
       B=C-B
    ENDIF
    IF (X.GE.0.75D0) THEN
       GM=0.0D0
       IF (DABS(C-A-B-INT(C-A-B)).LT.1.0D-15) THEN
          M=INT(C-A-B)
          CALL GAMMA(A,GA)
          CALL GAMMA(B,GB)
          CALL GAMMA(C,GC)
          CALL GAMMA(A+M,GAM)
          CALL GAMMA(B+M,GBM)
          CALL PSI(A,PA)
          CALL PSI(B,PB)
          IF (M.NE.0) GM=1.0D0
          DO J=1,ABS(M)-1
             GM=GM*J ! 30
          ENDDO
          RM=1.0D0
          DO J=1,ABS(M)
             RM=RM*J ! 35
          ENDDO
          F0=1.0D0
          R0=1.0D0
          R1=1.0D0
          SP0=0.D0
          SP=0.0D0
          IF (M.GE.0) THEN
             C0=GM*GC/(GAM*GBM)
             C1=-GC*(X-1.0D0)**M/(GA*GB*RM)
             DO K=1,M-1
                R0=R0*(A+K-1.0D0)*(B+K-1.0)/(K*(K-M))*(1.0-X)
                F0=F0+R0 ! 40
             ENDDO
             DO K=1,M
                SP0=SP0+1.0D0/(A+K-1.0)+1.0/(B+K-1.0)-1.0/K ! 45
             ENDDO
             F1=PA+PB+SP0+2.0D0*EL+DLOG(1.0D0-X)
             DO K=1,250
                SP=SP+(1.0D0-A)/(K*(A+K-1.0))+(1.0-B)/(K*(B+K-1.0))
                SM=0.0D0
                DO J=1,M
                   SM=SM+(1.0D0-A)/((J+K)*(A+J+K-1.0))+1.0/&
                        (B+J+K-1.0)  ! 50
                ENDDO
                RP=PA+PB+2.0D0*EL+SP+SM+DLOG(1.0D0-X)
                R1=R1*(A+M+K-1.0D0)*(B+M+K-1.0)/(K*(M+K))*(1.0-X)
                F1=F1+R1*RP
                IF (DABS(F1-HW).LT.DABS(F1)*EPS)EXIT ! GO TO 60
                HW=F1 ! 55
             ENDDO
             HF=F0*C0+F1*C1 ! 60
          ELSE IF (M.LT.0) THEN
             M=-M
             C0=GM*GC/(GA*GB*(1.0D0-X)**M)
             C1=-(-1)**M*GC/(GAM*GBM*RM)
             DO K=1,M-1
                R0=R0*(A-M+K-1.0D0)*(B-M+K-1.0)/(K*(K-M))*(1.0-X)
                F0=F0+R0 ! 65
             ENDDO
             DO K=1,M
                SP0=SP0+1.0D0/K ! 70
             ENDDO
             F1=PA+PB-SP0+2.0D0*EL+DLOG(1.0D0-X)
             DO K=1,250
                SP=SP+(1.0D0-A)/(K*(A+K-1.0))+(1.0-B)/(K*(B+K-1.0))
                SM=0.0D0
                DO J=1,M
                   SM=SM+1.0D0/(J+K) ! 75
                ENDDO
                RP=PA+PB+2.0D0*EL+SP-SM+DLOG(1.0D0-X)
                R1=R1*(A+K-1.0D0)*(B+K-1.0)/(K*(M+K))*(1.0-X)
                F1=F1+R1*RP
                IF (DABS(F1-HW).LT.DABS(F1)*EPS)EXIT ! GO TO 85
                HW=F1 ! 80
             ENDDO
             HF=F0*C0+F1*C1 ! 85
          ENDIF
       ELSE
          CALL GAMMA(A,GA)
          CALL GAMMA(B,GB)
          CALL GAMMA(C,GC)
          CALL GAMMA(C-A,GCA)
          CALL GAMMA(C-B,GCB)
          CALL GAMMA(C-A-B,GCAB)
          CALL GAMMA(A+B-C,GABC)
          C0=GC*GCAB/(GCA*GCB)
          C1=GC*GABC/(GA*GB)*(1.0D0-X)**(C-A-B)
          HF=0.0D0
          R0=C0
          R1=C1
          DO K=1,250
             R0=R0*(A+K-1.0D0)*(B+K-1.0)/(K*(A+B-C+K))*(1.0-X)
             R1=R1*(C-A+K-1.0D0)*(C-B+K-1.0)/(K*(C-A-B+K))&
                  *(1.0-X)
             HF=HF+R0+R1
             IF (DABS(HF-HW).LT.DABS(HF)*EPS)EXIT ! GO TO 95
             HW=HF ! 90
          ENDDO
          HF=HF+C0+C1 ! 95
       ENDIF
    ELSE
       A0=1.0D0
       IF (C.GT.A.AND.C.LT.2.0D0*A.AND.&
            C.GT.B.AND.C.LT.2.0D0*B) THEN
          A0=(1.0D0-X)**(C-A-B)
          A=C-A
          B=C-B
       ENDIF
       HF=1.0D0
       R=1.0D0
       DO K=1,250
          R=R*(A+K-1.0D0)*(B+K-1.0D0)/(K*(C+K-1.0D0))*X
          HF=HF+R
          IF (DABS(HF-HW).LE.DABS(HF)*EPS)EXIT ! GO TO 105
          HW=HF ! 100
       ENDDO
       HF=A0*HF
    ENDIF
    IF (X1.LT.0.0D0) THEN
       X=X1
       C0=1.0D0/(1.0D0-X)**AA
       HF=C0*HF
    ENDIF
    A=AA
    B=BB
    IF (K.GT.120) WRITE(*,115)
115 FORMAT(1X,'Warning! You should check the accuracy in HYGFX')
    RETURN
  END SUBROUTINE HYGFX

  SUBROUTINE GAMMA(X,GA)
!
!       ================================================== 
!       Purpose: Compute gamma function â(x)  
!       Input :  x  --- Argument of â(x)  
!                       ( x is not equal to 0,-1,-2,úúú) 
!       Output:  GA --- â(x)
!       ==================================================  
!    
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::X
    REAL(KIND(1d0)),INTENT(OUT)::GA
    REAL(KIND(1d0)),DIMENSION(26)::G
    REAL(KIND(1d0)),PARAMETER::PI=3.141592653589793D0
    REAL(KIND(1d0))::Z,R,GR
    INTEGER::M1,K,M
    IF (X.EQ.INT(X)) THEN
       IF (X.GT.0.0D0) THEN
          GA=1.0D0
          M1=X-1
          DO K=2,M1
             GA=GA*K ! 10
          ENDDO
       ELSE
          GA=1.0D+300
       ENDIF
    ELSE
       IF (DABS(X).GT.1.0D0) THEN
          Z=DABS(X)
          M=INT(Z)
          R=1.0D0
          DO K=1,M
             R=R*(Z-K) ! 15
          ENDDO
          Z=Z-M
       ELSE
          Z=X
       ENDIF
       G(1:26)=(/1.0D0,0.5772156649015329D0,&
            -0.6558780715202538D0, -0.420026350340952D-1,&
            0.1665386113822915D0,-.421977345555443D-1,&
            -.96219715278770D-2, .72189432466630D-2,&
            -.11651675918591D-2, -.2152416741149D-3,&
            .1280502823882D-3, -.201348547807D-4,&
            -.12504934821D-5, .11330272320D-5,&
            -.2056338417D-6, .61160950D-8,&
            .50020075D-8, -.11812746D-8,&
            .1043427D-9, .77823D-11,&
            -.36968D-11, .51D-12,&
            -.206D-13, -.54D-14, .14D-14, .1D-15/)
       GR=G(26)
       DO K=25,1,-1
          GR=GR*Z+G(K) ! 20
       ENDDO
       GA=1.0D0/(GR*Z)
       IF (DABS(X).GT.1.0D0) THEN
          GA=GA*R
          IF (X.LT.0.0D0) GA=-PI/(X*GA*DSIN(PI*X))
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE GAMMA
  
  SUBROUTINE PSI(X,PS)
!
!  ====================================== 
!      Purpose: Compute Psi function
!      Input :  x  --- Argument of psi(x) 
!      Output:  PS --- psi(x)                                                                                                                                                                              
! ======================================                                                                                                                                                              
!
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::X
    REAL(KIND(1d0)),INTENT(OUT)::PS
    REAL(KIND(1d0))::PI,EL,S,XA
    INTEGER::N,K
    REAL(KIND(1d0))::A1,A2,A3,A4,A5,A6,A7,A8,X2
    XA=DABS(X)
    PI=3.141592653589793D0
    EL=.5772156649015329D0
    S=0.0D0
    IF (X.EQ.INT(X).AND.X.LE.0.0) THEN
       PS=1.0D+300
       RETURN
    ELSE IF (XA.EQ.INT(XA)) THEN
       N=XA
       DO K=1 ,N-1
          S=S+1.0D0/K
       ENDDO
       PS=-EL+S
    ELSE IF (XA+.5.EQ.INT(XA+.5)) THEN
       N=XA-.5
       DO K=1,N
          S=S+1.0/(2.0D0*K-1.0D0) ! 20
       ENDDO
       PS=-EL+2.0D0*S-1.386294361119891D0
    ELSE
       IF (XA.LT.10.0) THEN
          N=10-INT(XA)
          DO K=0,N-1
             S=S+1.0D0/(XA+K)
          ENDDO
          XA=XA+N
       ENDIF
       X2=1.0D0/(XA*XA)
       A1=-.8333333333333D-01
       A2=.83333333333333333D-02
       A3=-.39682539682539683D-02
       A4=.41666666666666667D-02
       A5=-.75757575757575758D-02
       A6=.21092796092796093D-01
       A7=-.83333333333333333D-01
       A8=.4432598039215686D0
       PS=DLOG(XA)-.5D0/XA+X2*(((((((A8*X2+A7)*X2+&
            A6)*X2+A5)*X2+A4)*X2+A3)*X2+A2)*X2+A1)
       PS=PS-S
    ENDIF
    IF (X.LT.0.0) PS=PS-PI*DCOS(PI*X)/DSIN(PI*X)-1.0D0/X
    RETURN
  END SUBROUTINE PSI
END MODULE SPECIAL_FUN
