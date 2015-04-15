      implicit none
      integer uin,uouts,uouts_cen,uouts_min,uouts_max,
     &        uoutpdf,uoutpdf_cen,uoutpdf_min,uoutpdf_max,uoutpdf_avg
      parameter (uin=99,uouts=98,uoutpdf=97,uouts_cen=96,
     &           uoutpdf_cen=95,uouts_min=94,uoutpdf_min=93,
     &           uouts_max=92,uoutpdf_max=91,uoutpdf_avg=90)
      integer i,j,nng,nps,lb,ipdfmin,ipdfmax,iscalevar,iPDFvar,
     &        npdf,i0,i1
      logical done
      character*10 str
      character*100 infile,outfile_s,outfile_pdf,outfilebase,var
      character*100 outfile_s_central,outfile_s_min,outfile_s_max
      character*100 outfile_pdf_central,outfile_pdf_min,outfile_pdf_max,outfile_pdf_average
      double precision xs(0:1000,10),ys(1000,10)
      double precision xpdf(0:1000,100),ypdf(1000,1000),
     &     ysmin(1000),ysmax(1000),ypdfmin(1000),ypdfmax(1000),ypdfavg(1000),
     &      deltapdf(100)
      integer nc
      parameter (nc=15)
      character*(nc) scale(10),pdf(100)
      integer orders(100)
      data scale/'central value  ',
     &           'muR=1.0 muF=1.0','muR=1.0 muF=2.0','muR=1.0 muF=0.5',
     &           'muR=2.0 muF=1.0','muR=2.0 muF=2.0','muR=2.0 muF=0.5',
     &           'muR=0.5 muF=1.0','muR=0.5 muF=2.0','muR=0.5 muF=0.5'/
c
      write (*,*) 'Give input .top file'
      read(*,'(a)') infile
      open(unit=uin,file=infile,status='old')
      outfilebase=infile(1:index(infile,'.top')-1)
      lb=index(outfilebase,' ')-1
c
      write (*,*) 'Type 1 for scale variations, 0 not'
      read (*,*) iscalevar
      if(iscalevar.ne.0.and.iscalevar.ne.1)then
         write(*,*)'You should type 0 or 1, not ',iscalevar
         stop
      endif
c
      write (*,*) 
     $   'Type 1 for PDF variations (Hessian), 2 for MC (NNPDF), 0 not'
      read (*,*) iPDFvar
      if(iPDFvar.lt.0.or.iPDFvar.gt.2)then
         write(*,*)'You should type 0, 1 or 2, not ',iPDFvar
         stop
      endif
c
      npdf=0
      str='no PDF'
      if(iPDFvar.eq.1.or.iPDFvar.eq.2)then
         write (*,*) 'Give minimum and maximum uncertainty PDF sets'
         write (*,*)'(like e.g. 21101 21140)'
         read(*,*)ipdfmin,ipdfmax
         npdf=abs(ipdfmax-ipdfmin)+1
         do j=1,npdf
            write(str,'(i8)')ipdfmin-1+j
            pdf(j)='PDF='//str
         enddo
      endif
c
      if(iscalevar.eq.1)then
         outfile_s=outfilebase(1:lb)//'_scale.top'
         outfile_s_central=outfilebase(1:lb)//'_scale_central.top'
         outfile_s_min=outfilebase(1:lb)//'_scale_min.top'
         outfile_s_max=outfilebase(1:lb)//'_scale_max.top'
         open(unit=uouts,file=outfile_s,status='unknown')
         open(unit=uouts_cen,file=outfile_s_central,status='unknown')
         open(unit=uouts_min,file=outfile_s_min,status='unknown')
         open(unit=uouts_max,file=outfile_s_max,status='unknown')
         call write_top_start(uouts_cen)
         call write_top_start(uouts_min)
         call write_top_start(uouts_max)
         call write_top_start(uouts)
      endif
      if(iPDFvar.eq.1)then
         outfile_pdf=outfilebase(1:lb)//'_pdf.top'
         outfile_pdf_central=outfilebase(1:lb)//'_pdf_central.top'
         outfile_pdf_min=outfilebase(1:lb)//'_pdf_min.top'
         outfile_pdf_max=outfilebase(1:lb)//'_pdf_max.top'
         open(unit=uoutpdf,file=outfile_pdf,status='unknown')
         open(unit=uoutpdf_cen,file=outfile_pdf_central,status=
     &                                                     'unknown')
         open(unit=uoutpdf_min,file=outfile_pdf_min,status='unknown')
         open(unit=uoutpdf_max,file=outfile_pdf_max,status='unknown')
         call write_top_start(uoutpdf_cen)
         call write_top_start(uoutpdf_min)
         call write_top_start(uoutpdf_max)
         call write_top_start(uoutpdf)
      else if (iPDFvar.eq.2) then
         outfile_pdf=outfilebase(1:lb)//'_pdf.top'
         outfile_pdf_central=outfilebase(1:lb)//'_pdf_central.top'
         outfile_pdf_average=outfilebase(1:lb)//'_pdf_average.top'
         outfile_pdf_min=outfilebase(1:lb)//'_pdf_min.top'
         outfile_pdf_max=outfilebase(1:lb)//'_pdf_max.top'
         open(unit=uoutpdf,file=outfile_pdf,status='unknown')
         open(unit=uoutpdf_cen,file=outfile_pdf_central,status=
     &                                                     'unknown')
         open(unit=uoutpdf_avg,file=outfile_pdf_average,status=
     &                                                     'unknown')
         open(unit=uoutpdf_min,file=outfile_pdf_min,status='unknown')
         open(unit=uoutpdf_max,file=outfile_pdf_max,status='unknown')
         call write_top_start(uoutpdf_cen)
         call write_top_start(uoutpdf_avg)
         call write_top_start(uoutpdf_min)
         call write_top_start(uoutpdf_max)
         call write_top_start(uoutpdf)
      endif
c
      var='first variable to find'
      done=.false.
      do
         rewind(uin)
         call get_reference_var(uin,var,done,i0,i1)
         write(*,*) var
         if(done) exit
         xs(0,1)=0d0
         call get_data(uin,xs(0,1),ys(1,1))
c Scale uncertainties
         if(iscalevar.eq.1)then
            do i=2,10
               call get_var(uin,' (  '//var(1:i1-5)//scale(i),i0,i1)
               xs(0,i)=0d0
               call get_data(uin,xs(0,i),ys(1,i))
               if (xs(0,i).ne.xs(0,1)) then
                  write (*,*) 'data not consistent... skipping plot'
                  cycle
               endif
            enddo
         endif
c PDF uncertainties
         if(iPDFvar.eq.1.or.iPDFvar.eq.2)then
            do i=1,npdf
               call get_var(uin,' (  '//var(1:i1-5)//pdf(i),i0,i1)
               xpdf(0,i)=0d0
               call get_data(uin,xpdf(0,i),ypdf(1,i))
               if (xpdf(0,i).ne.xs(0,1)) then
                  write (*,*) 'data not consistent... skipping plot'
                  cycle
               endif
            enddo
         endif
c Now I should have all data
c Compute max/min scale
         if(iscalevar.eq.1)then
            do i=1,nint(xs(0,1))
               ysmax(i)=-1d99
               ysmin(i)=+1d99
               do j=2,10
                  ysmax(i)=max( ysmax(i),ys(i,j) )
                  ysmin(i)=min( ysmin(i),ys(i,j) )
               enddo
            enddo
            call write_top_header(uouts_cen,var)
            call write_top_header(uouts_min,var)
            call write_top_header(uouts_max,var)
            call write_top_header(uouts,var)
            call write_top_data(uouts_cen,xs(0,1),ys(1,1))
            call write_top_data(uouts_min,xs(0,1),ysmin)
            call write_top_data(uouts_max,xs(0,1),ysmax)
            call write_top_data2(uouts,xs(0,1),ys(1,1),ysmax,ysmin)
         endif
c Compute max/min pdf according to hessian method
         if(iPDFvar.eq.1)then
            do j=1,nint(xpdf(0,1))
               ypdfmax(j)=0d0
               ypdfmin(j)=0d0
               do i=1,20
                  nps=2*i-1
                  nng=2*i
                  ypdfmin(j)=ypdfmin(j)+( max(0.d0,
     &                 ys(j,1)-ypdf(j,nps),
     &                 ys(j,1)-ypdf(j,nng)) )**2
                  ypdfmax(j)=ypdfmax(j)+( max(0.d0,
     &                 ypdf(j,nps)-ys(j,1),
     &                 ypdf(j,nng)-ys(j,1)) )**2
               enddo
               ypdfmin(j)=ys(j,1)-sqrt(ypdfmin(j))
               ypdfmax(j)=ys(j,1)+sqrt(ypdfmax(j))
            enddo
            call write_top_header(uoutpdf_cen,var)
            call write_top_header(uoutpdf_min,var)
            call write_top_header(uoutpdf_max,var)
            call write_top_header(uoutpdf,var)
            call write_top_data(uoutpdf_cen,xs(0,1),ys(1,1))
            call write_top_data(uoutpdf_min,xs(0,1),ypdfmin)
            call write_top_data(uoutpdf_max,xs(0,1),ypdfmax)
            call write_top_data2(uoutpdf,xs(0,1),ys(1,1),ypdfmax,
     &                                                   ypdfmin)
        else if (iPDFvar.eq.2) then
            do j=1,nint(xpdf(0,1))
               ypdfmax(j)=0d0
               ypdfmin(j)=0d0
               ypdfavg(j)=0d0
               do i=1, npdf
                  ypdfavg(j)=ypdfavg(j) + ypdf(j,i)/dble(npdf)
               enddo
               do i=1, npdf
                  deltapdf(i)=abs(ypdfavg(j) - ypdf(j,i))
               enddo
               call sortrx(100, deltapdf, orders)

               ypdfmin(j)=ys(j,1)-deltapdf(orders(68))
               ypdfmax(j)=ys(j,1)+deltapdf(orders(68))
            enddo
            call write_top_header(uoutpdf_cen,var)
            call write_top_header(uoutpdf_avg,var)
            call write_top_header(uoutpdf_min,var)
            call write_top_header(uoutpdf_max,var)
            call write_top_header(uoutpdf,var)
            call write_top_data(uoutpdf_cen,xs(0,1),ys(1,1))
            call write_top_data(uoutpdf_min,xs(0,1),ypdfmin)
            call write_top_data(uoutpdf_avg,xs(0,1),ypdfavg)
            call write_top_data(uoutpdf_max,xs(0,1),ypdfmax)
            call write_top_data2(uoutpdf,xs(0,1),ys(1,1),ypdfmax,
     &                                                   ypdfmin)
         endif
      enddo
c
      close (uin)
      if(iscalevar.eq.1)close (uouts)
      if(iPDFvar.eq.1)close (uoutpdf)
c
      return
      end



      subroutine get_reference_var(uin,var,done,i0,i1)
      implicit none
      integer uin
      character*100 buff,var
      logical found_original,done
      logical verbose
      parameter (verbose=.false.)
      integer i0,i1
      character*100 lines(1000)
      integer nlines
      common/to_lines/lines, nlines
      nlines=0
      if (var.eq.'first variable to find') then
         found_original=.true.
      else
         found_original=.false.
      endif
      do
         read(uin,'(a)',end=999) buff
         i0=index(buff,'(')
         i1=index(buff,'central')
         if (found_original) then
            if (i0.ne.0.and.i1.ne.0)then
               var=buff(i0+3:i1-1)
               exit
            else
               nlines = nlines+1
               lines(nlines) = buff(1:100)
               if (index(buff,'PLOT').ne.0) nlines = 0
               if (index(buff,'NEW PLOT').ne.0) nlines = 1
            endif
         else
            if(buff(1:i1-1).eq.' (  '//var(1:i1-5))then
               found_original=.true.
            else
               nlines = nlines+1
               lines(nlines) = buff(1:100)
               if (index(buff,'PLOT').ne.0) nlines = 0
               if (index(buff,'NEW PLOT').ne.0) nlines = 1
            endif
         endif
      enddo
      read(uin,'(a)') buff
      return
 999  done=.true.
      return
      end



      subroutine get_var(uin,var,i0,i1)
      implicit none
      integer uin,i0,i1
      integer nc
      parameter(nc=15)
      logical verbose
      parameter (verbose=.false.)
      character*100 var,buff
      do
         read(uin,'(a)',end=998,err=998) buff
         if (buff(1:i1+nc-1).eq.var(1:i1+nc-1)) exit
      enddo
      read(uin,'(a)') buff
      return
 998  write (*,*) 'ERROR var not found'
      if(verbose) write (*,*) var
      stop
      end
      


      subroutine get_data(uin,x,y)
      implicit none
      integer uin
      double precision x(0:1000),y(1000),dy(1000)
      character*100 buff
      do 
         read(uin,'(a)')buff
         if(index(buff,'HIST').eq.0) then
            x(0)=x(0)+1d0
            read(buff,*),x(nint(x(0))),y(nint(x(0))),dy(nint(x(0)))
         else
            exit
         endif
      enddo
      return
      end



      subroutine write_top_start(unit)
      implicit none
      integer unit
      write (unit,'(a)') 'SET DEVICE POSTSCRIPT ORIENT=3'
      write (unit,'(a)') 'SET FONT DUPLEX'
      return
      end



      subroutine write_top_header(unit,var)
      implicit none
      integer unit
      character*100 var
      character*100 lines(1000)
      logical verbose
      parameter (verbose=.false.)
      integer nlines,i
      common/to_lines/lines, nlines
      write(unit,'(a)') 'PLOT'
      do i = 1, nlines
        if(verbose) write(*,*) lines(i)
        write(unit,'(a)') trim(lines(i))
      enddo
      write(unit,'(a)') trim(' (  '//var)
      return
      end



      subroutine write_top_data(unit,x,y)
      implicit none
      integer unit,i
      double precision x(0:1000),y(1000)
      do i=1,nint(x(0))
         write(unit,999) x(i),y(i),0d0
      enddo
      write (unit,'(a)') 'HIST SOLID'
      return
999   format(3(d16.6))
      end



      subroutine write_top_data2(unit,x,y,ymax,ymin)
      implicit none
      integer unit,i
      double precision x(0:1000),y(1000),ymax(1000),ymin(1000)
      do i=1,nint(x(0))
         write(unit,999) x(i),y(i),0d0
      enddo
      write (unit,'(a)') 'HIST SOLID RED'
      do i=1,nint(x(0))
         write(unit,999) x(i),ymax(i),0d0
      enddo
      write (unit,'(a)') 'HIST SOLID'
      do i=1,nint(x(0))
         write (unit,999) x(i),ymin(i),0d0
      enddo
      write (unit,'(a)') 'HIST SOLID'
      write (unit,'(a)') ''
      return
999   format(3(d16.6))
      end



C Here's a hybrid QuickSort I wrote a number of years ago.  It's
C based on suggestions in Knuth, Volume 3, and performs much better
C than a pure QuickSort on short or partially ordered input arrays.  

      SUBROUTINE SORTRX(N,DATA,INDEX)
C===================================================================
C
C     SORTRX -- SORT, Real input, indeX output
C
C
C     Input:  N     INTEGER
C             DATA  REAL
C
C     Output: INDEX INTEGER (DIMENSION N)
C
C This routine performs an in-memory sort of the first N elements of
C array DATA, returning into array INDEX the indices of elements of
C DATA arranged in ascending order.  Thus,
C
C    DATA(INDEX(1)) will be the smallest number in array DATA;
C    DATA(INDEX(N)) will be the largest number in DATA.
C
C The original data is not physically rearranged.  The original order
C of equal input values is not necessarily preserved.
C
C===================================================================
C
C SORTRX uses a hybrid QuickSort algorithm, based on several
C suggestions in Knuth, Volume 3, Section 5.2.2.  In particular, the
C "pivot key" [my term] for dividing each subsequence is chosen to be
C the median of the first, last, and middle values of the subsequence;
C and the QuickSort is cut off when a subsequence has 9 or fewer
C elements, and a straight insertion sort of the entire array is done
C at the end.  The result is comparable to a pure insertion sort for
C very short arrays, and very fast for very large arrays (of order 12
C micro-sec/element on the 3081K for arrays of 10K elements).  It is
C also not subject to the poor performance of the pure QuickSort on
C partially ordered data.
C
C Created:  15 Jul 1986  Len Moss
C
C===================================================================
 
      INTEGER   N,INDEX(N)
      REAL*8      DATA(N)
 
      INTEGER   LSTK(31),RSTK(31),ISTK
      INTEGER   L,R,I,J,P,INDEXP,INDEXT
      REAL*8      DATAP
 
C     QuickSort Cutoff
C
C     Quit QuickSort-ing when a subsequence contains M or fewer
C     elements and finish off at end with straight insertion sort.
C     According to Knuth, V.3, the optimum value of M is around 9.
 
      INTEGER   M
      PARAMETER (M=9)
 
C===================================================================
C
C     Make initial guess for INDEX
 
      DO 50 I=1,N
         INDEX(I)=I
   50    CONTINUE
 
C     If array is short, skip QuickSort and go directly to
C     the straight insertion sort.
 
      IF (N.LE.M) GOTO 900
 
C===================================================================
C
C     QuickSort
C
C     The "Qn:"s correspond roughly to steps in Algorithm Q,
C     Knuth, V.3, PP.116-117, modified to select the median
C     of the first, last, and middle elements as the "pivot
C     key" (in Knuth's notation, "K").  Also modified to leave
C     data in place and produce an INDEX array.  To simplify
C     comments, let DATA[I]=DATA(INDEX(I)).
 
C Q1: Initialize
      ISTK=0
      L=1
      R=N
 
  200 CONTINUE
 
C Q2: Sort the subsequence DATA[L]..DATA[R].
C
C     At this point, DATA[l] <= DATA[m] <= DATA[r] for all l < L,
C     r > R, and L <= m <= R.  (First time through, there is no
C     DATA for l < L or r > R.)
 
      I=L
      J=R
 
C Q2.5: Select pivot key
C
C     Let the pivot, P, be the midpoint of this subsequence,
C     P=(L+R)/2; then rearrange INDEX(L), INDEX(P), and INDEX(R)
C     so the corresponding DATA values are in increasing order.
C     The pivot key, DATAP, is then DATA[P].
 
      P=(L+R)/2
      INDEXP=INDEX(P)
      DATAP=DATA(INDEXP)
 
      IF (DATA(INDEX(L)) .GT. DATAP) THEN
         INDEX(P)=INDEX(L)
         INDEX(L)=INDEXP
         INDEXP=INDEX(P)
         DATAP=DATA(INDEXP)
      ENDIF
 
      IF (DATAP .GT. DATA(INDEX(R))) THEN
         IF (DATA(INDEX(L)) .GT. DATA(INDEX(R))) THEN
            INDEX(P)=INDEX(L)
            INDEX(L)=INDEX(R)
         ELSE
            INDEX(P)=INDEX(R)
         ENDIF
         INDEX(R)=INDEXP
         INDEXP=INDEX(P)
         DATAP=DATA(INDEXP)
      ENDIF
 
C     Now we swap values between the right and left sides and/or
C     move DATAP until all smaller values are on the left and all
C     larger values are on the right.  Neither the left or right
C     side will be internally ordered yet; however, DATAP will be
C     in its final position.
 
  300 CONTINUE
 
C Q3: Search for datum on left >= DATAP
C
C     At this point, DATA[L] <= DATAP.  We can therefore start scanning
C     up from L, looking for a value >= DATAP (this scan is guaranteed
C     to terminate since we initially placed DATAP near the middle of
C     the subsequence).
 
         I=I+1
         IF (DATA(INDEX(I)).LT.DATAP) GOTO 300
 
  400 CONTINUE
 
C Q4: Search for datum on right <= DATAP
C
C     At this point, DATA[R] >= DATAP.  We can therefore start scanning
C     down from R, looking for a value <= DATAP (this scan is guaranteed
C     to terminate since we initially placed DATAP near the middle of
C     the subsequence).
 
         J=J-1
         IF (DATA(INDEX(J)).GT.DATAP) GOTO 400
 
C Q5: Have the two scans collided?
 
      IF (I.LT.J) THEN
 
C Q6: No, interchange DATA[I] <--> DATA[J] and continue
 
         INDEXT=INDEX(I)
         INDEX(I)=INDEX(J)
         INDEX(J)=INDEXT
         GOTO 300
      ELSE
 
C Q7: Yes, select next subsequence to sort
C
C     At this point, I >= J and DATA[l] <= DATA[I] == DATAP <= DATA[r],
C     for all L <= l < I and J < r <= R.  If both subsequences are
C     more than M elements long, push the longer one on the stack and
C     go back to QuickSort the shorter; if only one is more than M
C     elements long, go back and QuickSort it; otherwise, pop a
C     subsequence off the stack and QuickSort it.
 
         IF (R-J .GE. I-L .AND. I-L .GT. M) THEN
            ISTK=ISTK+1
            LSTK(ISTK)=J+1
            RSTK(ISTK)=R
            R=I-1
         ELSE IF (I-L .GT. R-J .AND. R-J .GT. M) THEN
            ISTK=ISTK+1
            LSTK(ISTK)=L
            RSTK(ISTK)=I-1
            L=J+1
         ELSE IF (R-J .GT. M) THEN
            L=J+1
         ELSE IF (I-L .GT. M) THEN
            R=I-1
         ELSE
C Q8: Pop the stack, or terminate QuickSort if empty
            IF (ISTK.LT.1) GOTO 900
            L=LSTK(ISTK)
            R=RSTK(ISTK)
            ISTK=ISTK-1
         ENDIF
         GOTO 200
      ENDIF
 
  900 CONTINUE
 
C===================================================================
C
C Q9: Straight Insertion sort
 
      DO 950 I=2,N
         IF (DATA(INDEX(I-1)) .GT. DATA(INDEX(I))) THEN
            INDEXP=INDEX(I)
            DATAP=DATA(INDEXP)
            P=I-1
  920       CONTINUE
               INDEX(P+1) = INDEX(P)
               P=P-1
               IF (P.GT.0) THEN
                  IF (DATA(INDEX(P)).GT.DATAP) GOTO 920
               ENDIF
            INDEX(P+1) = INDEXP
         ENDIF
  950    CONTINUE
 
C===================================================================
C
C     All done
 
      END
