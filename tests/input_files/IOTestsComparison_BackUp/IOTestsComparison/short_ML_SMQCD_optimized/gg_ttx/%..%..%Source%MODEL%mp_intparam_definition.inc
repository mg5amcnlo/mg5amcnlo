ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

C     Parameters that should not be recomputed event by event.
C     
      IF(READLHA) THEN

        MP__G = 2 * SQRT(MP__AS*MP__PI)  ! for the first init

        MP__MDL_CONJG__CKM3X3 = 1.000000E+00_16

        MP__MDL_CKM22 = 1.000000E+00_16

        MP__MDL_LHV = 1.000000E+00_16

        MP__MDL_CKM3X3 = 1.000000E+00_16

        MP__MDL_CONJG__CKM22 = 1.000000E+00_16

        MP__MDL_CONJG__CKM33 = 1.000000E+00_16

        MP__MDL_NCOL = 3.000000E+00_16

        MP__MDL_CA = 3.000000E+00_16

        MP__MDL_TF = 5.000000E-01_16

        MP__MDL_CF = (4.000000E+00_16/3.000000E+00_16)

        MP__MDL_COMPLEXI = (0.000000E+00_16,1.000000E+00_16)

        MP__MDL_MZ__EXP__2 = MP__MDL_MZ**2

        MP__MDL_MZ__EXP__4 = MP__MDL_MZ**4

        MP__MDL_SQRT__2 = SQRT(2.000000E+00_16)

        MP__MDL_MH__EXP__2 = MP__MDL_MH**2

        MP__MDL_NCOL__EXP__2_M_1 = MP__MDL_NCOL**2-1.000000E+00_16

        MP__MDL_MB__EXP__2 = MP__MDL_MB**2

        MP__MDL_MT__EXP__2 = MP__MDL_MT**2

        MP__MDL_NCOL__EXP__2_M_1_0 = MP__MDL_NCOL**2-1.000000E+00_16

        MP__MDL_AEW = 1.000000E+00_16/MP__AEWM1

        MP__MDL_MW = SQRT(MP__MDL_MZ__EXP__2/2.000000E+00_16+SQRT(MP__M
     $   DL_MZ__EXP__4/4.000000E+00_16-(MP__MDL_AEW*MP__PI*MP__MDL_MZ__
     $   EXP__2)/(MP__MDL_GF*MP__MDL_SQRT__2)))

        MP__MDL_SQRT__AEW = SQRT(MP__MDL_AEW)

        MP__MDL_EE = 2.000000E+00_16*MP__MDL_SQRT__AEW*SQRT(MP__PI)

        MP__MDL_MW__EXP__2 = MP__MDL_MW**2

        MP__MDL_SW2 = 1.000000E+00_16-MP__MDL_MW__EXP__2/MP__MDL_MZ__EX
     $   P__2

        MP__MDL_CW = SQRT(1.000000E+00_16-MP__MDL_SW2)

        MP__MDL_SQRT__SW2 = SQRT(MP__MDL_SW2)

        MP__MDL_SW = MP__MDL_SQRT__SW2

        MP__MDL_G1 = MP__MDL_EE/MP__MDL_CW

        MP__MDL_GW = MP__MDL_EE/MP__MDL_SW

        MP__MDL_V = (2.000000E+00_16*MP__MDL_MW*MP__MDL_SW)/MP__MDL_EE

        MP__MDL_V__EXP__2 = MP__MDL_V**2

        MP__MDL_LAM = MP__MDL_MH__EXP__2/(2.000000E+00_16*MP__MDL_V__EX
     $   P__2)

        MP__MDL_YB = (MP__MDL_YMB*MP__MDL_SQRT__2)/MP__MDL_V

        MP__MDL_YT = (MP__MDL_YMT*MP__MDL_SQRT__2)/MP__MDL_V

        MP__MDL_YTAU = (MP__MDL_YMTAU*MP__MDL_SQRT__2)/MP__MDL_V

        MP__MDL_MUH = SQRT(MP__MDL_LAM*MP__MDL_V__EXP__2)

        MP__MDL_AXIALZUP = (3.000000E+00_16/2.000000E+00_16)*(
     $   -(MP__MDL_EE*MP__MDL_SW)/(6.000000E+00_16*MP__MDL_CW))
     $   -(1.000000E+00_16/2.000000E+00_16)*((MP__MDL_CW*MP__MDL_EE)
     $   /(2.000000E+00_16*MP__MDL_SW))

        MP__MDL_AXIALZDOWN = (-1.000000E+00_16/2.000000E+00_16)*(
     $   -(MP__MDL_CW*MP__MDL_EE)/(2.000000E+00_16*MP__MDL_SW))
     $   +(-3.000000E+00_16/2.000000E+00_16)*(-(MP__MDL_EE*MP__MDL_SW)
     $   /(6.000000E+00_16*MP__MDL_CW))

        MP__MDL_VECTORZUP = (1.000000E+00_16/2.000000E+00_16)
     $   *((MP__MDL_CW*MP__MDL_EE)/(2.000000E+00_16*MP__MDL_SW))
     $   +(5.000000E+00_16/2.000000E+00_16)*(-(MP__MDL_EE*MP__MDL_SW)
     $   /(6.000000E+00_16*MP__MDL_CW))

        MP__MDL_VECTORZDOWN = (1.000000E+00_16/2.000000E+00_16)*(
     $   -(MP__MDL_CW*MP__MDL_EE)/(2.000000E+00_16*MP__MDL_SW))
     $   +(-1.000000E+00_16/2.000000E+00_16)*(-(MP__MDL_EE*MP__MDL_SW)
     $   /(6.000000E+00_16*MP__MDL_CW))

        MP__MDL_VECTORAUP = (2.000000E+00_16*MP__MDL_EE)/3.000000E
     $   +00_16

        MP__MDL_VECTORADOWN = -(MP__MDL_EE)/3.000000E+00_16

        MP__MDL_VECTORWMDXU = (1.000000E+00_16/2.000000E+00_16)
     $   *((MP__MDL_EE)/(MP__MDL_SW*MP__MDL_SQRT__2))

        MP__MDL_AXIALWMDXU = (-1.000000E+00_16/2.000000E+00_16)
     $   *((MP__MDL_EE)/(MP__MDL_SW*MP__MDL_SQRT__2))

        MP__MDL_VECTORWPUXD = (1.000000E+00_16/2.000000E+00_16)
     $   *((MP__MDL_EE)/(MP__MDL_SW*MP__MDL_SQRT__2))

        MP__MDL_AXIALWPUXD = -(1.000000E+00_16/2.000000E+00_16)
     $   *((MP__MDL_EE)/(MP__MDL_SW*MP__MDL_SQRT__2))

        MP__MDL_I1X33 = MP__MDL_YB*MP__MDL_CONJG__CKM3X3

        MP__MDL_I2X33 = MP__MDL_YT*MP__MDL_CONJG__CKM3X3

        MP__MDL_I3X33 = MP__MDL_CKM3X3*MP__MDL_YT

        MP__MDL_I4X33 = MP__MDL_CKM3X3*MP__MDL_YB

        MP__MDL_VECTOR_TBGP = MP__MDL_I1X33-MP__MDL_I2X33

        MP__MDL_AXIAL_TBGP = -MP__MDL_I2X33-MP__MDL_I1X33

        MP__MDL_VECTOR_TBGM = MP__MDL_I3X33-MP__MDL_I4X33

        MP__MDL_AXIAL_TBGM = -MP__MDL_I4X33-MP__MDL_I3X33

        MP__MDL_GW__EXP__2 = MP__MDL_GW**2

        MP__MDL_CW__EXP__2 = MP__MDL_CW**2

        MP__MDL_EE__EXP__2 = MP__MDL_EE**2

        MP__MDL_SW__EXP__2 = MP__MDL_SW**2

        MP__MDL_YB__EXP__2 = MP__MDL_YB**2

        MP__MDL_YT__EXP__2 = MP__MDL_YT**2

      ENDIF
C     
C     Parameters that should be recomputed at an event by even basis.
C     
      MP__AS = MP__G**2/4/MP__PI

      MP__MDL_MU_R__EXP__2 = MP__MU_R**2

      MP__MDL_SQRT__AS = SQRT(MP__AS)

      MP__MDL_G__EXP__2 = MP__G**2

      MP__MDL_G__EXP__3 = MP__G**3

      MP__MDL_G__EXP__4 = MP__G**4

C     
C     Definition of the EW coupling used in the write out of aqed
C     
      MP__GAL(1) = 2 * SQRT(MP__PI/MP__AEWM1)
      MP__GAL(2) = 1D0

