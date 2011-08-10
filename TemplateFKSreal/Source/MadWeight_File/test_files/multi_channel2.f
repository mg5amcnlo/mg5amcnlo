C+-----------------------------------------------------------------------+
C|                MULTI-CHANNEL WEIGHT FOR MADWEIGHT                     |
C|                                                                       |
C|     Author: Pierre Artoisenet (UCL-CP3)                               |
C|             Olivier Mattelaer (UCL-CP3)                               |
C+-----------------------------------------------------------------------+
C|     This file is generated automaticly by MADWEIGHT-ANALYZER          |
C+-----------------------------------------------------------------------+

       double precision function multi_channel_weight(config)
C+-----------------------------------------------------------------------+
C|     routine returnings the multi channel weight linked to the         |
C|       change of variable 'config'                                     |
C+-----------------------------------------------------------------------+
       	implicit none

       	integer config
       	include 'coupl.inc'
           include 'd_choices.inc'
c    double precision prov1,prov2,prov3,prov4,prov5,prov6,prov7,prov8,prov9
c    double precision prov10,prov11,prov12,prov13,prov14,prov15,prov16
c    double precision prov17,prov18,prov19,prov20,prov21,prov22,prov23 
           double precision num,den
           double precision zero,one
           parameter (zero=0d0)
           parameter (one=1d0)
        double precision local_1
        double precision tf_E_for_7

        external tf_E_for_7

        double precision local_2
        double precision tf_E_for_3

        external tf_E_for_3

        double precision local_3
        double precision tf_E_for_6

        external tf_E_for_6

        double precision local_4
        double precision  Breit_Wigner_for_part
        external  Breit_Wigner_for_part
        double precision local_5
        double precision tf_E_for_part

        external tf_E_for_part

        double precision local_6
        double precision local_7
        double precision local_8
        double precision local_9
        double precision tf_E_for_5

        external tf_E_for_5

        double precision local_10
        double precision local_11
        double precision tf_E_for_4

        external tf_E_for_4


        if (config.eq.1) then
        local_1 = tf_E_for_7()
        local_2 = tf_E_for_3()
        local_3 = tf_E_for_6()
        local_4 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_5 = tf_E_for_part(first_d_4_5)
        local_6 = tf_E_for_part(first_d_7_8)
        local_7 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_8 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_9 = tf_E_for_5()
        local_10 = tf_E_for_part(second_d_4_5)
        local_11 = tf_E_for_4()

        num = 1d0 * 1d0 * local_1 * local_4 * local_8 * local_9 * local_10 * local_11
        den = 0d0 + 1d0 * local_1 * local_4 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_4 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_10 * local_11 +
     & 1d0 * local_4 * local_6 * local_7 * local_9 * local_10 * local_11 +
     & 1d0 * local_2 * local_4 * local_6 * local_7 * local_9 * local_11 +
     & 1d0 * local_2 * local_5 * local_6 * local_9 * local_10 * local_11
        multi_channel_weight = num/den

        elseif (config.eq.2) then
        local_1 = tf_E_for_7()
        local_2 = tf_E_for_3()
        local_3 = tf_E_for_6()
        local_4 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_5 = tf_E_for_part(first_d_4_5)
        local_6 = tf_E_for_part(first_d_7_8)
        local_7 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_8 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_9 = tf_E_for_5()
        local_10 = tf_E_for_part(second_d_4_5)
        local_11 = tf_E_for_4()

        num = 1d0 * 1d0 * local_1 * local_5 * local_8 * local_9 * local_10 * local_11
        den = 0d0 + 1d0 * local_1 * local_4 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_4 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_10 * local_11 +
     & 1d0 * local_4 * local_6 * local_7 * local_9 * local_10 * local_11 +
     & 1d0 * local_2 * local_4 * local_6 * local_7 * local_9 * local_11 +
     & 1d0 * local_2 * local_5 * local_6 * local_9 * local_10 * local_11
        multi_channel_weight = num/den

        elseif (config.eq.3) then
        local_1 = tf_E_for_7()
        local_2 = tf_E_for_3()
        local_3 = tf_E_for_6()
        local_4 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_5 = tf_E_for_part(first_d_4_5)
        local_6 = tf_E_for_part(first_d_7_8)
        local_7 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_8 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_9 = tf_E_for_5()
        local_10 = tf_E_for_part(second_d_4_5)
        local_11 = tf_E_for_4()

        num = 1d0 * 1d0 * local_1 * local_4 * local_6 * local_9 * local_10 * local_11
        den = 0d0 + 1d0 * local_1 * local_4 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_4 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_10 * local_11 +
     & 1d0 * local_4 * local_6 * local_7 * local_9 * local_10 * local_11 +
     & 1d0 * local_2 * local_4 * local_6 * local_7 * local_9 * local_11 +
     & 1d0 * local_2 * local_5 * local_6 * local_9 * local_10 * local_11
        multi_channel_weight = num/den

        elseif (config.eq.4) then
        local_1 = tf_E_for_7()
        local_2 = tf_E_for_3()
        local_3 = tf_E_for_6()
        local_4 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_5 = tf_E_for_part(first_d_4_5)
        local_6 = tf_E_for_part(first_d_7_8)
        local_7 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_8 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_9 = tf_E_for_5()
        local_10 = tf_E_for_part(second_d_4_5)
        local_11 = tf_E_for_4()

        num = 1d0 * 1d0 * local_1 * local_5 * local_6 * local_9 * local_10 * local_11
        den = 0d0 + 1d0 * local_1 * local_4 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_4 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_10 * local_11 +
     & 1d0 * local_4 * local_6 * local_7 * local_9 * local_10 * local_11 +
     & 1d0 * local_2 * local_4 * local_6 * local_7 * local_9 * local_11 +
     & 1d0 * local_2 * local_5 * local_6 * local_9 * local_10 * local_11
        multi_channel_weight = num/den

        elseif (config.eq.5) then
        local_1 = tf_E_for_7()
        local_2 = tf_E_for_3()
        local_3 = tf_E_for_6()
        local_4 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_5 = tf_E_for_part(first_d_4_5)
        local_6 = tf_E_for_part(first_d_7_8)
        local_7 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_8 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_9 = tf_E_for_5()
        local_10 = tf_E_for_part(second_d_4_5)
        local_11 = tf_E_for_4()

        num = 1d0 * 1d0 * local_1 * local_3 * local_5 * local_8 * local_9 * local_10
        den = 0d0 + 1d0 * local_1 * local_4 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_4 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_10 * local_11 +
     & 1d0 * local_4 * local_6 * local_7 * local_9 * local_10 * local_11 +
     & 1d0 * local_2 * local_4 * local_6 * local_7 * local_9 * local_11 +
     & 1d0 * local_2 * local_5 * local_6 * local_9 * local_10 * local_11
        multi_channel_weight = num/den

        elseif (config.eq.6) then
        local_1 = tf_E_for_7()
        local_2 = tf_E_for_3()
        local_3 = tf_E_for_6()
        local_4 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_5 = tf_E_for_part(first_d_4_5)
        local_6 = tf_E_for_part(first_d_7_8)
        local_7 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_8 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_9 = tf_E_for_5()
        local_10 = tf_E_for_part(second_d_4_5)
        local_11 = tf_E_for_4()

        num = 1d0 * 1d0 * local_1 * local_3 * local_5 * local_6 * local_9 * local_10
        den = 0d0 + 1d0 * local_1 * local_4 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_4 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_10 * local_11 +
     & 1d0 * local_4 * local_6 * local_7 * local_9 * local_10 * local_11 +
     & 1d0 * local_2 * local_4 * local_6 * local_7 * local_9 * local_11 +
     & 1d0 * local_2 * local_5 * local_6 * local_9 * local_10 * local_11
        multi_channel_weight = num/den

        elseif (config.eq.7) then
        local_1 = tf_E_for_7()
        local_2 = tf_E_for_3()
        local_3 = tf_E_for_6()
        local_4 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_5 = tf_E_for_part(first_d_4_5)
        local_6 = tf_E_for_part(first_d_7_8)
        local_7 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_8 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_9 = tf_E_for_5()
        local_10 = tf_E_for_part(second_d_4_5)
        local_11 = tf_E_for_4()

        num = 1d0 * 1d0 * local_1 * local_3 * local_5 * local_8 * local_10 * local_11
        den = 0d0 + 1d0 * local_1 * local_4 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_4 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_10 * local_11 +
     & 1d0 * local_4 * local_6 * local_7 * local_9 * local_10 * local_11 +
     & 1d0 * local_2 * local_4 * local_6 * local_7 * local_9 * local_11 +
     & 1d0 * local_2 * local_5 * local_6 * local_9 * local_10 * local_11
        multi_channel_weight = num/den

        elseif (config.eq.8) then
        local_1 = tf_E_for_7()
        local_2 = tf_E_for_3()
        local_3 = tf_E_for_6()
        local_4 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_5 = tf_E_for_part(first_d_4_5)
        local_6 = tf_E_for_part(first_d_7_8)
        local_7 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_8 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_9 = tf_E_for_5()
        local_10 = tf_E_for_part(second_d_4_5)
        local_11 = tf_E_for_4()

        num = 1d0 * 1d0 * local_1 * local_3 * local_5 * local_6 * local_10 * local_11
        den = 0d0 + 1d0 * local_1 * local_4 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_4 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_10 * local_11 +
     & 1d0 * local_4 * local_6 * local_7 * local_9 * local_10 * local_11 +
     & 1d0 * local_2 * local_4 * local_6 * local_7 * local_9 * local_11 +
     & 1d0 * local_2 * local_5 * local_6 * local_9 * local_10 * local_11
        multi_channel_weight = num/den

        elseif (config.eq.9) then
        local_1 = tf_E_for_7()
        local_2 = tf_E_for_3()
        local_3 = tf_E_for_6()
        local_4 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_5 = tf_E_for_part(first_d_4_5)
        local_6 = tf_E_for_part(first_d_7_8)
        local_7 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_8 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_9 = tf_E_for_5()
        local_10 = tf_E_for_part(second_d_4_5)
        local_11 = tf_E_for_4()

        num = 1d0 * 1d0 * local_4 * local_6 * local_7 * local_9 * local_10 * local_11
        den = 0d0 + 1d0 * local_1 * local_4 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_4 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_10 * local_11 +
     & 1d0 * local_4 * local_6 * local_7 * local_9 * local_10 * local_11 +
     & 1d0 * local_2 * local_4 * local_6 * local_7 * local_9 * local_11 +
     & 1d0 * local_2 * local_5 * local_6 * local_9 * local_10 * local_11
        multi_channel_weight = num/den

        elseif (config.eq.10) then
        local_1 = tf_E_for_7()
        local_2 = tf_E_for_3()
        local_3 = tf_E_for_6()
        local_4 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_5 = tf_E_for_part(first_d_4_5)
        local_6 = tf_E_for_part(first_d_7_8)
        local_7 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_8 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_9 = tf_E_for_5()
        local_10 = tf_E_for_part(second_d_4_5)
        local_11 = tf_E_for_4()

        num = 1d0 * 1d0 * local_2 * local_4 * local_6 * local_7 * local_9 * local_11
        den = 0d0 + 1d0 * local_1 * local_4 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_4 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_10 * local_11 +
     & 1d0 * local_4 * local_6 * local_7 * local_9 * local_10 * local_11 +
     & 1d0 * local_2 * local_4 * local_6 * local_7 * local_9 * local_11 +
     & 1d0 * local_2 * local_5 * local_6 * local_9 * local_10 * local_11
        multi_channel_weight = num/den

        elseif (config.eq.11) then
        local_1 = tf_E_for_7()
        local_2 = tf_E_for_3()
        local_3 = tf_E_for_6()
        local_4 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_5 = tf_E_for_part(first_d_4_5)
        local_6 = tf_E_for_part(first_d_7_8)
        local_7 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_8 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_9 = tf_E_for_5()
        local_10 = tf_E_for_part(second_d_4_5)
        local_11 = tf_E_for_4()

        num = 1d0 * 1d0 * local_2 * local_5 * local_6 * local_9 * local_10 * local_11
        den = 0d0 + 1d0 * local_1 * local_4 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_8 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_4 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_5 * local_6 * local_9 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_9 * local_10 +
     & 1d0 * local_1 * local_3 * local_5 * local_8 * local_10 * local_11 +
     & 1d0 * local_1 * local_3 * local_5 * local_6 * local_10 * local_11 +
     & 1d0 * local_4 * local_6 * local_7 * local_9 * local_10 * local_11 +
     & 1d0 * local_2 * local_4 * local_6 * local_7 * local_9 * local_11 +
     & 1d0 * local_2 * local_5 * local_6 * local_9 * local_10 * local_11
        multi_channel_weight = num/den
       	   endif
       	   return
       	   end

