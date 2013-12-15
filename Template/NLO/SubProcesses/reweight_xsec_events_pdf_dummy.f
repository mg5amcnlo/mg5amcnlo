      subroutine initpdf(i)
      implicit none
      integer i
      write (*,*) 'ERROR from dummy "initpdf()": cannot do'//
     &     ' pdf reweighting without LHAPDF'
      stop
      end

      subroutine pdfset(parm,value)
      implicit none
      character*20 parm(20)
      double precision value
      write (*,*) 'ERROR from dummy "setpdf()": cannot do'//
     &     ' pdf reweighting without LHAPDF'
      stop
      end
