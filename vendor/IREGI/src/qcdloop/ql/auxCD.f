* auxCD.F
* auxiliary functions used by the three- and four-point integrals
* these functions are adapted from Ansgar Denner's bcanew.f
* to the conventions of LoopTools;
* they are used for double-checking the results of FF
* last modified 25 Oct 05 th

      double complex function ln(x, isig)
      implicit none
      include 'qlconstants.f'
      double precision x, isig
      if( x .gt. 0 ) then
        ln = DCMPLX(log(x),0d0)
      else
       ln = dcmplx(log(-x)) + DCMPLX(0D0, sign(pi, isig))
      endif
      end

************************************************************************

      double complex function cln(z, isig)
      implicit none
      include 'qlconstants.f'
      double complex z
      double precision isig

      if( DIMAG(z) .eq. 0d0 .and. DBLE(z) .le. 0d0 ) then
         cln = log(-z) + DCMPLX(0D0, sign(pi, isig))
      else
         cln = log(z)
      endif
      end

************************************************************************
      
      double complex function denspence(z, isig)
      implicit none
      double complex z
      double precision isig

      include 'qlconstants.f'
      include 'qlonshellcutoff.f'

      double complex z1
      double precision az1,acc

      double complex li2series, cln
      external li2series, cln
      acc=qlonshellcutoff

      z1 = cone - z
      az1 = abs(z1)

      if( isig .eq. 0d0 .and. 
     &    DIMAG(z) .eq. 0d0 .and. abs(DBLE(z1)) .lt. acc )
     &    print *, "denspence: argument on cut"

      if( az1 .lt. 1D-15 ) then
        denspence = dcmplx(pisqo6)
      else if( DBLE(z) .lt. .5D0 ) then
        if( abs(z) .lt. 1d0 ) then
          denspence = li2series(z, isig)
        else
          denspence = -dcmplx(pisqo6) -
     &        .5D0*cln(-z, -isig)**2 - li2series(1d0/z, -isig)
        endif
      else
        if( az1 .lt. 1d0 ) then
          denspence = dcmplx(pisqo6) -
     &        cln(z, isig)*cln(z1, -isig) - li2series(z1, -isig)
        else
          denspence = 2d0*dcmplx(pisqo6) +
     &        .5D0*cln(-z1, -isig)**2 - cln(z, isig)*cln(z1, -isig) +
     &        li2series(1d0/z1, isig)
        endif
      endif
      end

************************************************************************

      double complex function li2series(z, isig)
      implicit none
      include 'qlconstants.f'
      double complex z
      double precision isig

      double complex xm, x2, new
      integer j

      double complex cln
      external cln

* these are the even-n Bernoulli numbers, already divided by (n + 1)!
* as in Table[BernoulliB[n]/(n + 1)!, {n, 2, 50, 2}]
      double precision b(25)
      data b /
     &    0.02777777777777777777777777777777777777777778774D0, 
     &    -0.000277777777777777777777777777777777777777777778D0, 
     &    4.72411186696900982615268329554043839758125472D-6, 
     &    -9.18577307466196355085243974132863021751910641D-8, 
     &    1.89788699889709990720091730192740293750394761D-9, 
     &    -4.06476164514422552680590938629196667454705711D-11, 
     &    8.92169102045645255521798731675274885151428361D-13, 
     &    -1.993929586072107568723644347793789705630694749D-14, 
     &    4.51898002961991819165047655285559322839681901D-16, 
     &    -1.035651761218124701448341154221865666596091238D-17, 
     &    2.39521862102618674574028374300098038167894899D-19, 
     &    -5.58178587432500933628307450562541990556705462D-21, 
     &    1.309150755418321285812307399186592301749849833D-22, 
     &    -3.087419802426740293242279764866462431595565203D-24, 
     &    7.31597565270220342035790560925214859103339899D-26, 
     &    -1.740845657234000740989055147759702545340841422D-27, 
     &    4.15763564461389971961789962077522667348825413D-29, 
     &    -9.96214848828462210319400670245583884985485196D-31, 
     &    2.394034424896165300521167987893749562934279156D-32, 
     &    -5.76834735536739008429179316187765424407233225D-34, 
     &    1.393179479647007977827886603911548331732410612D-35, 
     &    -3.372121965485089470468473635254930958979742891D-37, 
     &    8.17820877756210262176477721487283426787618937D-39, 
     &    -1.987010831152385925564820669234786567541858996D-40, 
     &    4.83577851804055089628705937311537820769430091D-42 /

      xm = -cln(cone - z, -isig)
      x2 = xm**2
      li2series = xm - x2/4D0
      do j = 1, 25
        xm = xm*x2
        new = li2series + xm*b(j)
        if( new .eq. li2series ) return
        li2series = new
      enddo
      print *, "li2series: bad convergence"
      end

