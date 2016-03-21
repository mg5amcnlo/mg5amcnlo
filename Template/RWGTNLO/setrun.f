C-------------------------------------------------
C   GET_PDFUP
C   Convert MadEvent pdf name to LHAPDF number
C-------------------------------------------------

      subroutine get_pdfup(pdfin,pdfgup,pdfsup,lhaid)
      implicit none

      character*(*) pdfin
      integer mpdf
      integer npdfs,i,pdfgup(2),pdfsup(2),lhaid

      parameter (npdfs=16)
      character*7 pdflabs(npdfs)
      data pdflabs/
     $   'none',
     $   'mrs02nl',
     $   'mrs02nn',
     $   'cteq4_m',
     $   'cteq4_l',
     $   'cteq4_d',
     $   'cteq5_m',
     $   'cteq5_d',
     $   'cteq5_l',
     $   'cteq5m1',
     $   'cteq6_m',
     $   'cteq6_l',
     $   'cteq6l1',     
     $   'nn23lo',
     $   'nn23lo1',
     $   'nn23nlo'/
      integer numspdf(npdfs)
      data numspdf/
     $   00000,
     $   20250,
     $   20270,
     $   19150,
     $   19170,
     $   19160,
     $   19050,
     $   19060,
     $   19070,
     $   19051,
     $   10000,
     $   10041,
     $   10042,
     $   246800,
     $   247000,
     $   244600/


      if(pdfin.eq."lhapdf") then
        write(*,*)'using LHAPDF'
        do i=1,2
           pdfgup(i)=-1
           pdfsup(i)=lhaid
        enddo
        return
      endif

      
      mpdf=-1
      do i=1,npdfs
        if(pdfin(1:len_trim(pdfin)) .eq. pdflabs(i))then
          mpdf=numspdf(i)
        endif
      enddo

      if(mpdf.eq.-1) then
        write(*,*)'ERROR: pdf ',pdfin,' not implemented in get_pdfup.'
        write(*,*)'known pdfs are'
        write(*,*) pdflabs
        open(unit=26,file='../../error',status='unknown')
        write(26,*)'ERROR: pdf ',pdfin,' not implemented in get_pdfup.'
        write(26,*)'known pdfs are'
        write(26,*) pdflabs
        stop 1
c$$$        write(*,*)'using ',pdflabs(12)
c$$$        mpdf=numspdf(12)
      endif

      do i=1,2
        pdfgup(i)=-1
        pdfsup(i)=mpdf
      enddo

      return
      end
