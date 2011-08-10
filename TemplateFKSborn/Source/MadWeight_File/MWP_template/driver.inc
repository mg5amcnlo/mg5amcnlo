C+-----------------------------------------------------------------------+
C|                   DEFINITION FOR MADWEIGHT DRIVER                     |
C|                                                                       |
C|     Author: Pierre Artoisenet (UCL-CP3)                               |
C|             Olivier Mattelaer (UCL-CP3)                               |
C+-----------------------------------------------------------------------+

c     channel position
c
      integer config_pos,perm_pos
      common /to_config/config_pos,perm_pos
c
c
      integer max_channel
      parameter (max_channel=1000)
c
c
c
      double precision cross_perm(2,max_channel) ! maximum of first step integral
      integer          l_cross_max(max_channel)
      common /to_cross_estimation/l_cross_max, cross_perm
c                                                                         
