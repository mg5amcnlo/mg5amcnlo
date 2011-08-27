************************************************************************
* This is the file  avh_olo_hello.f  of the package                    *
*                                                                      *
*                               OneLOop                                *
*                                                                      *
* for the evaluation of 1-loop scalar 1-, 2-, 3- and 4-point functions *
*                                                                      *
* author: Andreas van Hameren <hamerenREMOVETHIS@ifj.edu.pl>           *
*   date: 09-09-2010                                                   *
************************************************************************
*
* Execute  make  to create a library,
* or just compile the following 10 source files:
*   avh_olo_hello.f
*   avh_olo_cmplx.f
*   avh_olo_real.f
*   avh_olo_func.f
*   avh_olo_3div.f
*   avh_olo_3fin.f
*   avh_olo_4div.f
*   avh_olo_4fin.f
*   avh_olo_b11.f
*   avh_olo_cd0.f
*
* The following routines will then be available:
*      subroutine avh_olo_a0m( rslt ,mm )
*      subroutine avh_olo_a0c( rslt ,mm )
*      subroutine avh_olo_b0m( rslt ,p1,m1,m2 )
*      subroutine avh_olo_b0c( rslt ,p1,m1,m2 )
*      subroutine avh_olo_b11m( b11,b00,b1,b0 ,p1,m1,m2 )
*      subroutine avh_olo_b11c( b11,b00,b1,b0 ,p1,m1,m2 )
*      subroutine avh_olo_c0m( rslt ,p1,p2,p3 ,m1,m2,m3 )
*      subroutine avh_olo_c0c( rslt ,p1,p2,p3 ,m1,m2,m3 )
*      subroutine avh_olo_d0m( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
*      subroutine avh_olo_d0c( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
*      subroutine avh_olo_mu_set( mu )
*      subroutine avh_olo_onshell( thrs )
*      subroutine avh_olo_unit( your_unit )
*      subroutine avh_olo_printall( unit )
*
* 1-point function:
*
*      subroutine avh_olo_a0m( rslt ,mm )
*        input:  double precision mm
*      subroutine avh_olo_a0c( rslt ,mm )
*        input:  double complex mm
*
*        output: double complex rslt(0) = eps^0   -coefficient
*                               rslt(1) = eps^(-1)-coefficient
*                               rslt(2) = eps^(-2)-coefficient
*
* 2-point function:
*
*      subroutine avh_olo_b0m( rslt ,p1,m1,m2 )
*        input:  double precision p1,m1,m2
*      subroutine avh_olo_b0c( rslt ,p1,m1,m2 )
*        input:  double complex p1,m1,m2
*
*        output: double complex rslt(0) = eps^0   -coefficient
*                               rslt(1) = eps^(-1)-coefficient
*                               rslt(2) = eps^(-2)-coefficient
*
* 2-point Passarino-Veltman functions:
*
*      subroutine avh_olo_b11m( b11,b00,b1,b0 ,p1,m1,m2 )
*        input:  double precision p1,m1,m2
*      subroutine avh_olo_b11c( b11,b00,b1,b0 ,p1,m1,m2 )
*        input:  double complex p1,m1,m2
*
*        output: double complex bX(0) = eps^0   -coefficient
*                               bX(1) = eps^(-1)-coefficient
*                               bX(2) = eps^(-2)-coefficient
*                for X={11,00,1,0} defined such that
*                B{mu,nu} = g{mu,nu}*b00 + p1{mu}*p1{nu}*b11
*                   B{mu} = p1{mu}*b1
*
* 3-point function:
*
*      subroutine avh_olo_c0m( rslt ,p1,p2,p3 ,m1,m2,m3 )
*        input:  double precision p1,p2,p3 ,m1,m2,m3
*      subroutine avh_olo_c0c( rslt ,p1,p2,p3 ,m1,m2,m3 )
*        input:  double complex p1,p2,p3 ,m1,m2,m3
*
*        output: double complex rslt(0) = eps^0   -coefficient
*                               rslt(1) = eps^(-1)-coefficient
*                               rslt(2) = eps^(-2)-coefficient
*
* 4-point function:
*
*      subroutine avh_olo_d0m( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
*        input:  double precision p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
*      subroutine avh_olo_d0c( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
*        input:  double complex p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
*
*        output: double complex rslt(0) = eps^0   -coefficient
*                               rslt(1) = eps^(-1)-coefficient
*                               rslt(2) = eps^(-2)-coefficient
*
* to set the renormalization scale:
*
*       subroutine avh_olo_mu_set(mu)
*         input: double precision mu , has unit of mass (so is not mu^2)
*       If this routine is not called, mu is set to the default mu=1d0.
*
* to set the threshold to distinguish between IR-divergent and IR-finite
* cases:
*       subroutine avh_olo_onshell(thrs)
*         input: double precision thrs
*       If this routine is not called, thrs is considered to be 0d0
*
* Messages are sent to unit=6 by default. You can change this with
*       subroutine avh_olo_unit( your_unit )
*         input: integer your_unit
*       If input is smaller than 1, no messages will be printed at all.
*
* All input and accompanying output is printed to unit "unit" in case
* you call
*       subroutine avh_olo_printall( unit )
*         input: integer unit
*       If input is smaller than 1, nothing will be printed
*
* Check the comments in the routines themselves for more details.
*
*
* Routines for IR-divergent functions with all internal masses equal zero
* based on  G. Duplancic and B. Nizic,
*           Eur.Phys.J.C20:357-370,2001 (arXiv:hep-ph/0006249)
* and on  Z. Bern, L.J. Dixon and D.A. Kosower,
*         Nucl.Phys.B412,751(1994) (arXiv:hep-ph/9306240) 
*
* Routines for IR-divergent functions with non-zero internal masses
* based on  R. Keith Ellis and G. Zanderighi,
*           JHEP 0802:002,2008 (arXiv:0712.1851)
* and on  W. Beenakker, H. Kuijf, W.L. van Neerven, J. Smith,
*         Phys.Rev.D40,54(1989)
* and on  W. Beenakker, S. Dittmaier, M. Kramer, B. Plumper, M. Spira,
*         P.M. Zerwas, Nucl.Phys.B653:151-203,2003 (arXiv:hep-ph/0211352)
* and on  E.L. Berger, M. Klasen, T.M.P. Tait,
*         Phys.Rev.D62:095014,2000. (arXiv:hep-ph/0005196)
* and on  W. Beenakker and D. Denner, Nucl.Phys.B338,349(1990)
*
* Routines for finite 4-point functions with real masses based on
*   A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
*
* Routines for finite 3-point functions obtained from these by sending
* one mass to infinity.
*
* Routines for finite 4-point functions with complex masses based on
* based on Dao Thi Nhung and Le Duc Ninh,
*          Comput.Phys.Commun.180:2258-2267,2009, arXiv:0902.0325 [hep-ph]
*   and on G. 't Hooft and M.J.G. Veltman, Nucl.Phys.B153:365-401,1979 
* 
* Routines for full 2-point functions based on
*   A. Denner, Fortsch.Phys.41:307-420,1993 (arXiv:0709.1075)
*
***********************************************************************

      subroutine avh_olo_hello
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
        write(*,'(a36,a36)') '#                      You are using'
     &                      ,' OneLOop 1.1                       #'
        write(*,'(a36,a36)') '#                                   '
     &                      ,'                                   #'
        write(*,'(a36,a36)') '# for the evaluation of 1-loop scala'
     &                      ,'r 1-, 2-, 3- and 4-point functions #'
        write(*,'(a36,a36)') '#                                   '
     &                      ,'                                   #'
        write(*,'(a36,a36)') '# author: Andreas van Hameren <hamer'
     &                      ,'enREMOVETHIS@ifj.edu.pl>           #'
        write(*,'(a36,a36)') '#   date: 09-09-2010                '
     &                      ,'                                   #'
        write(*,'(a36,a36)') '#                                   '
     &                      ,'                                   #'
        write(*,'(a36,a36)') '# Please cite                       '
     &                      ,'                                   #'
        write(*,'(a36,a36)') '#    A. van Hameren, arXiv:1007.4716'
     &                      ,' [hep-ph]                          #'
        write(*,'(a36,a36)') '#    A. van Hameren, C.G. Papadopoul'
     &                      ,'os and R. Pittau,                  #'
        write(*,'(a36,a36)') '#    JHEP 0909:106,2009, arXiv:0903.'
     &                      ,'4665 [hep-ph]                      #'
        write(*,'(a36,a36)') '# in publications with results obtai'
     &                      ,'ned with the help of this program. #'
        write(*,'(a36,a36)') '#                                   '
     &                      ,'                                   #'
        write(*,'(a36,a36)') '####################################'
     &                      ,'####################################'
      endif
      end
