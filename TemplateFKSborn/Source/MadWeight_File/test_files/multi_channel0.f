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
        double precision  Breit_Wigner_for_part
        external  Breit_Wigner_for_part
        double precision local_2
        double precision local_3
        double precision local_4
        double precision tf_E_for_part

        external tf_E_for_part

        double precision local_5
        double precision tf_E_for_6

        external tf_E_for_6

        double precision local_6
        double precision local_7

        if (config.eq.1) then
        local_1 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_2 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_3 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_4 = tf_E_for_part(first_d_7_8)
        local_5 = tf_E_for_6()
        local_6 = tf_E_for_part(second_d_7_8)
        local_7 = Breit_Wigner_for_part( -4, TMASS, TWIDTH)

        num = 1d0 * 1d0 * local_1 * local_2 * local_6 * local_7
        den = 0d0 + 1d0 * local_1 * local_2 * local_6 * local_7 +
     & 1d0 * local_1 * local_2 * local_5 * local_7 +
     & 1d0 * local_2 * local_4 * local_5 * local_6 +
     & 1d0 * local_1 * local_3 * local_6 * local_7 +
     & 1d0 * local_1 * local_3 * local_5 * local_7 +
     & 1d0 * local_3 * local_4 * local_5 * local_6
        multi_channel_weight = num/den

        elseif (config.eq.2) then
        local_1 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_2 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_3 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_4 = tf_E_for_part(first_d_7_8)
        local_5 = tf_E_for_6()
        local_6 = tf_E_for_part(second_d_7_8)
        local_7 = Breit_Wigner_for_part( -4, TMASS, TWIDTH)

        num = 1d0 * 1d0 * local_1 * local_2 * local_5 * local_7
        den = 0d0 + 1d0 * local_1 * local_2 * local_6 * local_7 +
     & 1d0 * local_1 * local_2 * local_5 * local_7 +
     & 1d0 * local_2 * local_4 * local_5 * local_6 +
     & 1d0 * local_1 * local_3 * local_6 * local_7 +
     & 1d0 * local_1 * local_3 * local_5 * local_7 +
     & 1d0 * local_3 * local_4 * local_5 * local_6
        multi_channel_weight = num/den

        elseif (config.eq.3) then
        local_1 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_2 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_3 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_4 = tf_E_for_part(first_d_7_8)
        local_5 = tf_E_for_6()
        local_6 = tf_E_for_part(second_d_7_8)
        local_7 = Breit_Wigner_for_part( -4, TMASS, TWIDTH)

        num = 1d0 * 1d0 * local_2 * local_4 * local_5 * local_6
        den = 0d0 + 1d0 * local_1 * local_2 * local_6 * local_7 +
     & 1d0 * local_1 * local_2 * local_5 * local_7 +
     & 1d0 * local_2 * local_4 * local_5 * local_6 +
     & 1d0 * local_1 * local_3 * local_6 * local_7 +
     & 1d0 * local_1 * local_3 * local_5 * local_7 +
     & 1d0 * local_3 * local_4 * local_5 * local_6
        multi_channel_weight = num/den

        elseif (config.eq.4) then
        local_1 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_2 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_3 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_4 = tf_E_for_part(first_d_7_8)
        local_5 = tf_E_for_6()
        local_6 = tf_E_for_part(second_d_7_8)
        local_7 = Breit_Wigner_for_part( -4, TMASS, TWIDTH)

        num = 1d0 * 1d0 * local_1 * local_3 * local_6 * local_7
        den = 0d0 + 1d0 * local_1 * local_2 * local_6 * local_7 +
     & 1d0 * local_1 * local_2 * local_5 * local_7 +
     & 1d0 * local_2 * local_4 * local_5 * local_6 +
     & 1d0 * local_1 * local_3 * local_6 * local_7 +
     & 1d0 * local_1 * local_3 * local_5 * local_7 +
     & 1d0 * local_3 * local_4 * local_5 * local_6
        multi_channel_weight = num/den

        elseif (config.eq.5) then
        local_1 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_2 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_3 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_4 = tf_E_for_part(first_d_7_8)
        local_5 = tf_E_for_6()
        local_6 = tf_E_for_part(second_d_7_8)
        local_7 = Breit_Wigner_for_part( -4, TMASS, TWIDTH)

        num = 1d0 * 1d0 * local_1 * local_3 * local_5 * local_7
        den = 0d0 + 1d0 * local_1 * local_2 * local_6 * local_7 +
     & 1d0 * local_1 * local_2 * local_5 * local_7 +
     & 1d0 * local_2 * local_4 * local_5 * local_6 +
     & 1d0 * local_1 * local_3 * local_6 * local_7 +
     & 1d0 * local_1 * local_3 * local_5 * local_7 +
     & 1d0 * local_3 * local_4 * local_5 * local_6
        multi_channel_weight = num/den

        elseif (config.eq.6) then
        local_1 = Breit_Wigner_for_part( -2, WMASS, WWIDTH)
        local_2 = Breit_Wigner_for_part( -1, WMASS, WWIDTH)
        local_3 = Breit_Wigner_for_part( -3, TMASS, TWIDTH)
        local_4 = tf_E_for_part(first_d_7_8)
        local_5 = tf_E_for_6()
        local_6 = tf_E_for_part(second_d_7_8)
        local_7 = Breit_Wigner_for_part( -4, TMASS, TWIDTH)

        num = 1d0 * 1d0 * local_3 * local_4 * local_5 * local_6
        den = 0d0 + 1d0 * local_1 * local_2 * local_6 * local_7 +
     & 1d0 * local_1 * local_2 * local_5 * local_7 +
     & 1d0 * local_2 * local_4 * local_5 * local_6 +
     & 1d0 * local_1 * local_3 * local_6 * local_7 +
     & 1d0 * local_1 * local_3 * local_5 * local_7 +
     & 1d0 * local_3 * local_4 * local_5 * local_6
        multi_channel_weight = num/den
       	   endif
       	   return
       	   end

