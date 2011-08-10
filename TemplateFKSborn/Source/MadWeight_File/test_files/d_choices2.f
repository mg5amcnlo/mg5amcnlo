      
C+-----------------------------------------------------------------------+
C|                         D CHOICES FOR MADWEIGHT                       |
C|                                                                       |
C|     Author: Pierre Artoisenet (UCL-CP3)                               |
C|             Olivier Mattelaer (UCL-CP3)                               |
C+-----------------------------------------------------------------------+
C|     This file is generated automaticly by MADWEIGHT-ANALYZER          |
C+-----------------------------------------------------------------------+     



         subroutine init_d_assignement()
        include 'd_choices.inc'


C+-----------------------------------------------------------------------+
C|                                                                       |
C|            variable for block d containing:                           |
C|            4 5 -1                                                     |
C|                                                                       |
C+-----------------------------------------------------------------------+




        call init_block_d_alignment(4,5,first_d_4_5, second_d_4_5)


C+-----------------------------------------------------------------------+
C|                                                                       |
C|            variable for block d containing:                           |
C|            7 8 -2                                                     |
C|                                                                       |
C+-----------------------------------------------------------------------+





        call init_block_d_alignment(7,8,first_d_7_8, second_d_7_8)


        return
        end


