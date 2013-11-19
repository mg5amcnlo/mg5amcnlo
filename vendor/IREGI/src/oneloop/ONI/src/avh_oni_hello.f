************************************************************************
* This is the file  avh_oni_hello.f  of the package                    *
*                                                                      *
*                  Oneloop with Numerical Integration                  *
*                                                                      *
* for the evaluation of 1-loop scalar 1-, 2-, 3- and 4-point functions *
*                                                                      *
* author: Andreas van Hameren <hamerenREMOVETHIS@ifj.edu.pl>           *
*   date: 15-12-2010                                                   *
************************************************************************
*
* This package needs access to the routine "cuhre" from the Cuba
* package by Thomas Hahn.
*
* Execute  make  to create a library,
* or just compile the following 11 source files:
*   avh_oni_hello.o
*   avh_oni_3div.f
*   avh_oni_3fin.f
*   avh_oni_dfam.f
*   avh_oni_4div.f
*   avh_oni_cmplx.f
*   avh_oni_func.f
*   avh_oni_cuba.f
*   avh_oni_dqagpe.f
*   avh_oni_d1mach.f
*   avh_oni_real.f
*
* Make sure you choose the right version family for Cuba in the
* makefile, or copy the right .h-file on  avh_oni_cuba.h .
*
*
* The following routines will then be available:
*      subroutine avh_oni_a0m( rslt ,mm )
*      subroutine avh_oni_a0c( rslt ,mm )
*      subroutine avh_oni_b0m( rslt ,p1,m1,m2 )
*      subroutine avh_oni_b0c( rslt ,p1,m1,m2 )
*      subroutine avh_oni_c0m( rslt ,p1,p2,p3 ,m1,m2,m3 )
*      subroutine avh_oni_c0c( rslt ,p1,p2,p3 ,m1,m2,m3 )
*      subroutine avh_oni_d0m( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
*      subroutine avh_oni_d0c( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
*      subroutine avh_oni_mu_set( mu )
*      subroutine avh_oni_onshell( thrs )
*      subroutine avh_oni_unit( unit )
*      subroutine avh_oni_printall( unit )
*      subroutine avh_oni_maxeval_set( maxeval )
*      subroutine avh_oni_unitcuba_set( unit )
*
* 1-point function:
*
*      subroutine avh_oni_a0m( rslt ,mm )
*        input:  double precision mm
*      subroutine avh_oni_a0c( rslt ,mm )
*        input:  double complex mm
*
*        output: double complex rslt(0) = eps^0   -coefficient
*                               rslt(1) = eps^(-1)-coefficient
*                               rslt(2) = eps^(-2)-coefficient
*
* 2-point function:
*
*      subroutine avh_oni_b0m( rslt ,p1,m1,m2 )
*        input:  double precision p1,m1,m2
*      subroutine avh_oni_b0c( rslt ,p1,m1,m2 )
*        input:  double complex p1,m1,m2
*
*        output: double complex rslt(0) = eps^0   -coefficient
*                               rslt(1) = eps^(-1)-coefficient
*                               rslt(2) = eps^(-2)-coefficient
*
* 3-point function:
*
*      subroutine avh_oni_c0m( rslt ,p1,p2,p3 ,m1,m2,m3 )
*        input:  double precision p1,p2,p3 ,m1,m2,m3
*      subroutine avh_oni_c0c( rslt ,p1,p2,p3 ,m1,m2,m3 )
*        input:  double complex p1,p2,p3 ,m1,m2,m3
*
*        output: double complex rslt(0) = eps^0   -coefficient
*                               rslt(1) = eps^(-1)-coefficient
*                               rslt(2) = eps^(-2)-coefficient
*
* 4-point function:
*
*      subroutine avh_oni_d0m( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
*        input:  double precision p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
*      subroutine avh_oni_d0c( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
*        input:  double complex p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
*
*        output: double complex rslt(0) = eps^0   -coefficient
*                               rslt(1) = eps^(-1)-coefficient
*                               rslt(2) = eps^(-2)-coefficient
*
*
* To set the renormalization scale:
*
*       subroutine avh_oni_mu_set(mu)
*         input: double precision mu , has unit of mass (so is not mu^2)
*       If this routine is not called, mu is set to the default mu=1d0.
*
*
* To set the threshold to distinguish between IR-divergent and IR-finite
* cases:
*       subroutine avh_oni_onshell(thrs)
*         input: double precision thrs
*       If this routine is not called, thrs is considered to be 0d0
*
*
* Messages are sent to unit=6 by default. You can change this with
*       subroutine avh_oni_unit( your_unit )
*         input: integer your_unit
*       If input is smaller than 1, no messages will be printed at all.
*
*
* All input and accompanying output is printed to unit "unit" in case
* you call
*       subroutine avh_oni_printall( unit )
*         input: integer unit
*       If input is smaller than 1, nothing will be printed
*
*
* The number of function evaluations for the 2- and 3-dim integrals
* performed with Cuba can be set with
*       subroutine avh_oni_maxeval_set( maxeval )
*         input: integer maxeval
*       If this routine is not called, then maxeval=1000000
*
*
* Messages related to the numerical integration are sent to unit=6
* by default. You can change this with
*       subroutine avh_oni_unitcuba_set( unit )
*         input: integer your_unit
*       If input is smaller than 1, no messages will be printed at all.
*
*
* Check the comments in the routines themselves for more details.
*
***********************************************************************

      subroutine avh_oni_hello
*  ********************************************************************
*  ********************************************************************
      implicit none
      logical init
      data init/.true./
      save init
      if (init) then
        init = .false.
        write(*,'(a36,a36)') '####################################'
     &                      ,'####################################'
        write(*,'(a36,a36)') '#                                   '
     &                      ,'                                   #'
        write(*,'(a36,a36)') '#        You are using Oneloop with '
     &                      ,'Numerical Integration 1.1.1        #'
        write(*,'(a36,a36)') '#                                   '
     &                      ,'                                   #'
        write(*,'(a36,a36)') '# for the evaluation of 1-loop scala'
     &                      ,'r 1-, 2-, 3- and 4-point functions #'
        write(*,'(a36,a36)') '#                                   '
     &                      ,'                                   #'
        write(*,'(a36,a36)') '# author: Andreas van Hameren <hamer'
     &                      ,'enREMOVETHIS@ifj.edu.pl>           #'
        write(*,'(a36,a36)') '#   date: 15-12-2010                '
     &                      ,'                                   #'
        write(*,'(a36,a36)') '#                                   '
     &                      ,'                                   #'
        write(*,'(a36,a36)') '# Please cite                       '
     &                      ,'                                   #'
        write(*,'(a36,a36)') '#    A. van Hameren, arXiv:1007.4716'
     &                      ,' [hep-ph]                          #'
        write(*,'(a36,a36)') '# in publications with results obtai'
     &                      ,'ned with the help of this program. #'
        write(*,'(a36,a36)') '#                                   '
     &                      ,'                                   #'
        write(*,'(a36,a36)') '####################################'
     &                      ,'####################################'
      endif
      end
