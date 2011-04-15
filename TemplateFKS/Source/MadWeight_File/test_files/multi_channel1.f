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

        if (config.eq.1) then
        multi_channel_weight = 1.0
       	   endif
       	   return
       	   end

