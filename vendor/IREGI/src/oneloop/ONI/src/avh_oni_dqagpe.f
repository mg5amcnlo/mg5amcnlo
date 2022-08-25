************************************************************************
* This is the file  avh_oni_dqagpe.f  of the package                   *
*                                                                      *
*                  Oneloop with Numerical Integration                  *
*                                                                      *
* for the evaluation of 1-loop scalar 1-, 2-, 3- and 4-point functions *
*                                                                      *
* author: Andreas van Hameren <hamerenREMOVETHIS@ifj.edu.pl>           *
*   date: 15-12-2010                                                   *
************************************************************************
*                                                                      *
* Have a look at the file  avh_oni_hello.f  for more information.      *
*                                                                      *
************************************************************************
*                                                                      *
* Contains  subroutine dqagpe  plus dependencies from quadpack,        *
* adapted for complex integrands. Original obtained from netlib.org    *
*                                                                      *
* All function and subroutine names prepended with "avh_oni_"          *
*                                                                      *
************************************************************************

      subroutine avh_oni_dqagpe(fncn,a,b,npts2,points,epsabs,epsrel,
     *   limit,result,abserr,neval,ier)
C     *   ,ier,alist,blist,rlist,elist,pts,iord,level,ndin,
C     *   last)
c***begin prologue  dqagpe
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a2a1
c***keywords  automatic integrator, general-purpose,
c             singularities at user specified points,
c             extrapolation, globally adaptive.
c***author  piessens,robert ,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  the routine calculates an approximation result to a given
c            definite integral i = integral of f over (a,b), hopefully
c            satisfying following claim for accuracy abs(i-result).le.
c            max(epsabs,epsrel*abs(i)). break points of the integration
c            interval, where local difficulties of the integrand may
c            occur(e.g. singularities,discontinuities),provided by user.
c***description
c
c        computation of a definite integral
c        standard fortran subroutine
c        double precision version
c
c        parameters
c         on entry
c            f      - double precision
c                     function subprogram defining the integrand
c                     function f(x). the actual name for f needs to be
c                     declared e x t e r n a l in the driver program.
c
c            a      - double precision
c                     lower limit of integration
c
c            b      - double precision
c                     upper limit of integration
c
c            npts2  - integer
c                     number equal to two more than the number of
c                     user-supplied break points within the integration
c                     range, npts2.ge.2.
c                     if npts2.lt.2, the routine will end with ier = 6.
c
c            points - double precision
c                     vector of dimension npts2, the first (npts2-2)
c                     elements of which are the user provided break
c                     points. if these points do not constitute an
c                     ascending sequence there will be an automatic
c                     sorting.
c
c            epsabs - double precision
c                     absolute accuracy requested
c            epsrel - double precision
c                     relative accuracy requested
c                     if  epsabs.le.0
c                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                     the routine will end with ier = 6.
c
c            limit  - integer
c                     gives an upper bound on the number of subintervals
c                     in the partition of (a,b), limit.ge.npts2
c                     if limit.lt.npts2, the routine will end with
c                     ier = 6.
c
c         on return
c            result - double precision
c                     approximation to the integral
c
c            abserr - double precision
c                     estimate of the modulus of the absolute error,
c                     which should equal or exceed abs(i-result)
c
c            neval  - integer
c                     number of integrand evaluations
c
c            ier    - integer
c                     ier = 0 normal and reliable termination of the
c                             routine. it is assumed that the requested
c                             accuracy has been achieved.
c                     ier.gt.0 abnormal termination of the routine.
c                             the estimates for integral and error are
c                             less reliable. it is assumed that the
c                             requested accuracy has not been achieved.
c            error messages
c                     ier = 1 maximum number of subdivisions allowed
c                             has been achieved. one can allow more
c                             subdivisions by increasing the value of
c                             limit (and taking the according dimension
c                             adjustments into account). however, if
c                             this yields no improvement it is advised
c                             to analyze the integrand in order to
c                             determine the integration difficulties. if
c                             the position of a local difficulty can be
c                             determined (i.e. singularity,
c                             discontinuity within the interval), it
c                             should be supplied to the routine as an
c                             element of the vector points. if necessary
c                             an appropriate special-purpose integrator
c                             must be used, which is designed for
c                             handling the type of difficulty involved.
c                         = 2 the occurrence of roundoff error is
c                             detected, which prevents the requested
c                             tolerance from being achieved.
c                             the error may be under-estimated.
c                         = 3 extremely bad integrand behaviour occurs
c                             at some points of the integration
c                             interval.
c                         = 4 the algorithm does not converge.
c                             roundoff error is detected in the
c                             extrapolation table. it is presumed that
c                             the requested tolerance cannot be
c                             achieved, and that the returned result is
c                             the best which can be obtained.
c                         = 5 the integral is probably divergent, or
c                             slowly convergent. it must be noted that
c                             divergence can occur with any other value
c                             of ier.gt.0.
c                         = 6 the input is invalid because
c                             npts2.lt.2 or
c                             break points are specified outside
c                             the integration range or
c                             (epsabs.le.0 and
c                              epsrel.lt.max(50*rel.mach.acc.,0.5d-28))
c                             or limit.lt.npts2.
c                             result, abserr, neval, last, rlist(1),
c                             and elist(1) are set to zero. alist(1) and
c                             blist(1) are set to a and b respectively.
c
c            alist  - double precision
c                     vector of dimension at least limit, the first
c                      last  elements of which are the left end points
c                     of the subintervals in the partition of the given
c                     integration range (a,b)
c
c            blist  - double precision
c                     vector of dimension at least limit, the first
c                      last  elements of which are the right end points
c                     of the subintervals in the partition of the given
c                     integration range (a,b)
c
c            rlist  - double precision
c                     vector of dimension at least limit, the first
c                      last  elements of which are the integral
c                     approximations on the subintervals
c
c            elist  - double precision
c                     vector of dimension at least limit, the first
c                      last  elements of which are the moduli of the
c                     absolute error estimates on the subintervals
c
c            pts    - double precision
c                     vector of dimension at least npts2, containing the
c                     integration limits and the break points of the
c                     interval in ascending sequence.
c
c            level  - integer
c                     vector of dimension at least limit, containing the
c                     subdivision levels of the subinterval, i.e. if
c                     (aa,bb) is a subinterval of (p1,p2) where p1 as
c                     well as p2 is a user-provided break point or
c                     integration limit, then (aa,bb) has level l if
c                     abs(bb-aa) = abs(p2-p1)*2**(-l).
c
c            ndin   - integer
c                     vector of dimension at least npts2, after first
c                     integration over the intervals (pts(i)),pts(i+1),
c                     i = 0,1, ..., npts2-2, the error estimates over
c                     some of the intervals may have been increased
c                     artificially, in order to put their subdivision
c                     forward. if this happens for the subinterval
c                     numbered k, ndin(k) is put to 1, otherwise
c                     ndin(k) = 0.
c
c            iord   - integer
c                     vector of dimension at least limit, the first k
c                     elements of which are pointers to the
c                     error estimates over the subintervals,
c                     such that elist(iord(1)), ..., elist(iord(k))
c                     form a decreasing sequence, with k = last
c                     if last.le.(limit/2+2), and k = limit+1-last
c                     otherwise
c
c            last   - integer
c                     number of subintervals actually produced in the
c                     subdivisions process
c
c***references  (none)
c***routines called  d1mach,dqelg,dqk21,dqpsrt
c***end prologue  dqagpe
      double complex fncn,result,area,area1,area12,area2,reseps,rlist,
     *  rlist2,res3la,zero
      parameter( zero=(0d0,0d0) )
      double precision a,abseps,abserr,alist,a1,
     *  a2,b,blist,b1,b2,correc,dabs,defabs,defab1,defab2,dmax1,dmin1,
     *  dres,avh_oni_d1mach,elist,epmach,epsabs,epsrel,erlarg,erlast,
     *  errbnd,
     *  errmax,error1,erro12,error2,errsum,ertest,oflow,points,pts,
     *  resa,resabs,sign,temp,uflow
      integer i,id,ier,ierro,ind1,ind2,iord,ip1,iroff1,iroff2,iroff3,j,
     *  jlow,jupbnd,k,ksgn,ktmin,last,levcur,level,levmax,limit,maxerr,
     *  ndin,neval,nint,nintp1,npts,npts2,nres,nrmax,numrl2
      logical extrap,noext
c
c
      dimension alist(limit),blist(limit),elist(limit),iord(limit),
     *  level(limit),ndin(npts2),points(npts2),pts(npts2),res3la(3),
     *  rlist(limit),rlist2(52)
c
      external fncn
c
c            the dimension of rlist2 is determined by the value of
c            limexp in subroutine epsalg (rlist2 should be of dimension
c            (limexp+2) at least).
c
c
c            list of major variables
c            -----------------------
c
c           alist     - list of left end points of all subintervals
c                       considered up to now
c           blist     - list of right end points of all subintervals
c                       considered up to now
c           rlist(i)  - approximation to the integral over
c                       (alist(i),blist(i))
c           rlist2    - array of dimension at least limexp+2
c                       containing the part of the epsilon table which
c                       is still needed for further computations
c           elist(i)  - error estimate applying to rlist(i)
c           maxerr    - pointer to the interval with largest error
c                       estimate
c           errmax    - elist(maxerr)
c           erlast    - error on the interval currently subdivided
c                       (before that subdivision has taken place)
c           area      - sum of the integrals over the subintervals
c           errsum    - sum of the errors over the subintervals
c           errbnd    - requested accuracy max(epsabs,epsrel*
c                       abs(result))
c           *****1    - variable for the left subinterval
c           *****2    - variable for the right subinterval
c           last      - index for subdivision
c           nres      - number of calls to the extrapolation routine
c           numrl2    - number of elements in rlist2. if an appropriate
c                       approximation to the compounded integral has
c                       been obtained, it is put in rlist2(numrl2) after
c                       numrl2 has been increased by one.
c           erlarg    - sum of the errors over the intervals larger
c                       than the smallest interval considered up to now
c           extrap    - logical variable denoting that the routine
c                       is attempting to perform extrapolation. i.e.
c                       before subdividing the smallest interval we
c                       try to decrease the value of erlarg.
c           noext     - logical variable denoting that extrapolation is
c                       no longer allowed (true-value)
c
c            machine dependent constants
c            ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c           oflow is the largest positive magnitude.
c
c***first executable statement  dqagpe
      epmach = avh_oni_d1mach(4)
c
c            test on validity of parameters
c            -----------------------------
c
      ier = 0
      neval = 0
      last = 0
      result = zero
      abserr = 0.0d+00
      alist(1) = a
      blist(1) = b
      rlist(1) = zero
      elist(1) = 0.0d+00
      iord(1) = 0
      level(1) = 0
      npts = npts2-2
      if(npts2.lt.2.or.limit.le.npts.or.(epsabs.le.0.0d+00.and.
     *  epsrel.lt.dmax1(0.5d+02*epmach,0.5d-28))) ier = 6
      if(ier.eq.6) go to 999
c
c            if any break points are provided, sort them into an
c            ascending sequence.
c
      sign = 1.0d+00
      if(a.gt.b) sign = -1.0d+00
      pts(1) = dmin1(a,b)
      if(npts.eq.0) go to 15
      do 10 i = 1,npts
        pts(i+1) = points(i)
   10 continue
   15 pts(npts+2) = dmax1(a,b)
      nint = npts+1
      a1 = pts(1)
      if(npts.eq.0) go to 40
      nintp1 = nint+1
      do 20 i = 1,nint
        ip1 = i+1
        do 20 j = ip1,nintp1
          if(pts(i).le.pts(j)) go to 20
          temp = pts(i)
          pts(i) = pts(j)
          pts(j) = temp
   20 continue
      if(pts(1).ne.dmin1(a,b).or.pts(nintp1).ne.dmax1(a,b)) ier = 6
      if(ier.eq.6) go to 999
c
c            compute first integral and error approximations.
c            ------------------------------------------------
c
   40 resabs = 0.0d+00
      do 50 i = 1,nint
        b1 = pts(i+1)
        call avh_oni_dqk21(fncn,a1,b1,area1,error1,defabs,resa)
        abserr = abserr+error1
        result = result+area1
        ndin(i) = 0
        if(error1.eq.resa.and.error1.ne.0.0d+00) ndin(i) = 1
        resabs = resabs+defabs
        level(i) = 0
        elist(i) = error1
        alist(i) = a1
        blist(i) = b1
        rlist(i) = area1
        iord(i) = i
        a1 = b1
   50 continue
      errsum = 0.0d+00
      do 55 i = 1,nint
        if(ndin(i).eq.1) elist(i) = abserr
        errsum = errsum+elist(i)
   55 continue
c
c           test on accuracy.
c
      last = nint
      neval = 21*nint
      dres = cdabs(result) !CDABSyes
      errbnd = dmax1(epsabs,epsrel*dres)
      if(abserr.le.0.1d+03*epmach*resabs.and.abserr.gt.errbnd) ier = 2
      if(nint.eq.1) go to 80
      do 70 i = 1,npts
        jlow = i+1
        ind1 = iord(i)
        do 60 j = jlow,nint
          ind2 = iord(j)
          if(elist(ind1).gt.elist(ind2)) go to 60
          ind1 = ind2
          k = j
   60   continue
        if(ind1.eq.iord(i)) go to 70
        iord(k) = iord(i)
        iord(i) = ind1
   70 continue
      if(limit.lt.npts2) ier = 1
   80 if(ier.ne.0.or.abserr.le.errbnd) go to 210
c
c           initialization
c           --------------
c
      rlist2(1) = result
      maxerr = iord(1)
      errmax = elist(maxerr)
      area = result
      nrmax = 1
      nres = 0
      numrl2 = 1
      ktmin = 0
      extrap = .false.
      noext = .false.
      erlarg = errsum
      ertest = errbnd
      levmax = 1
      iroff1 = 0
      iroff2 = 0
      iroff3 = 0
      ierro = 0
      uflow = avh_oni_d1mach(1)
      oflow = avh_oni_d1mach(2)
      abserr = oflow
      ksgn = -1
      if(dres.ge.(0.1d+01-0.5d+02*epmach)*resabs) ksgn = 1
c
c           main do-loop
c           ------------
c
      do 160 last = npts2,limit
c
c           bisect the subinterval with the nrmax-th largest error
c           estimate.
c
        levcur = level(maxerr)+1
        a1 = alist(maxerr)
        b1 = 0.5d+00*(alist(maxerr)+blist(maxerr))
        a2 = b1
        b2 = blist(maxerr)
        erlast = errmax
        call avh_oni_dqk21(fncn,a1,b1,area1,error1,resa,defab1)
        call avh_oni_dqk21(fncn,a2,b2,area2,error2,resa,defab2)
c
c           improve previous approximations to integral
c           and error and test for accuracy.
c
        neval = neval+42
        area12 = area1+area2
        erro12 = error1+error2
        errsum = errsum+erro12-errmax
        area = area+area12-rlist(maxerr)
        if(defab1.eq.error1.or.defab2.eq.error2) go to 95
        if(cdabs(rlist(maxerr)-area12).gt.0.1d-04*cdabs(area12) !CDABSyes
     *  .or.erro12.lt.0.99d+00*errmax) go to 90
        if(extrap) iroff2 = iroff2+1
        if(.not.extrap) iroff1 = iroff1+1
   90   if(last.gt.10.and.erro12.gt.errmax) iroff3 = iroff3+1
   95   level(maxerr) = levcur
        level(last) = levcur
        rlist(maxerr) = area1
        rlist(last) = area2
        errbnd = dmax1(epsabs,epsrel*cdabs(area)) !CDABSyes
c
c           test for roundoff error and eventually set error flag.
c
        if(iroff1+iroff2.ge.10.or.iroff3.ge.20) ier = 2
        if(iroff2.ge.5) ierro = 3
c
c           set error flag in the case that the number of
c           subintervals equals limit.
c
        if(last.eq.limit) ier = 1
c
c           set error flag in the case of bad integrand behaviour
c           at a point of the integration range
c
        if(dmax1(dabs(a1),dabs(b2)).le.(0.1d+01+0.1d+03*epmach)*
     *  (dabs(a2)+0.1d+04*uflow)) ier = 4
c
c           append the newly-created intervals to the list.
c
        if(error2.gt.error1) go to 100
        alist(last) = a2
        blist(maxerr) = b1
        blist(last) = b2
        elist(maxerr) = error1
        elist(last) = error2
        go to 110
  100   alist(maxerr) = a2
        alist(last) = a1
        blist(last) = b1
        rlist(maxerr) = area2
        rlist(last) = area1
        elist(maxerr) = error2
        elist(last) = error1
c
c           call subroutine dqpsrt to maintain the descending ordering
c           in the list of error estimates and select the subinterval
c           with nrmax-th largest error estimate (to be bisected next).
c
  110  call avh_oni_dqpsrt(limit,last,maxerr,errmax,elist,iord,nrmax)
c ***jump out of do-loop
        if(errsum.le.errbnd) go to 190
c ***jump out of do-loop
        if(ier.ne.0) go to 170
        if(noext) go to 160
        erlarg = erlarg-erlast
        if(levcur+1.le.levmax) erlarg = erlarg+erro12
        if(extrap) go to 120
c
c           test whether the interval to be bisected next is the
c           smallest interval.
c
        if(level(maxerr)+1.le.levmax) go to 160
        extrap = .true.
        nrmax = 2
  120   if(ierro.eq.3.or.erlarg.le.ertest) go to 140
c
c           the smallest interval has the largest error.
c           before bisecting decrease the sum of the errors over
c           the larger intervals (erlarg) and perform extrapolation.
c
        id = nrmax
        jupbnd = last
        if(last.gt.(2+limit/2)) jupbnd = limit+3-last
        do 130 k = id,jupbnd
          maxerr = iord(nrmax)
          errmax = elist(maxerr)
c ***jump out of do-loop
          if(level(maxerr)+1.le.levmax) go to 160
          nrmax = nrmax+1
  130   continue
c
c           perform extrapolation.
c
  140   numrl2 = numrl2+1
        rlist2(numrl2) = area
        if(numrl2.le.2) go to 155
        call avh_oni_dqelg(numrl2,rlist2,reseps,abseps,res3la,nres)
        ktmin = ktmin+1
        if(ktmin.gt.5.and.abserr.lt.0.1d-02*errsum) ier = 5
        if(abseps.ge.abserr) go to 150
        ktmin = 0
        abserr = abseps
        result = reseps
        correc = erlarg
        ertest = dmax1(epsabs,epsrel*cdabs(reseps)) !CDABSyes
c ***jump out of do-loop
        if(abserr.lt.ertest) go to 170
c
c           prepare bisection of the smallest interval.
c
  150   if(numrl2.eq.1) noext = .true.
        if(ier.ge.5) go to 170
  155   maxerr = iord(1)
        errmax = elist(maxerr)
        nrmax = 1
        extrap = .false.
        levmax = levmax+1
        erlarg = errsum
  160 continue
c
c           set the final result.
c           ---------------------
c
c
  170 if(abserr.eq.oflow) go to 190
      if((ier+ierro).eq.0) go to 180
      if(ierro.eq.3) abserr = abserr+correc
      if(ier.eq.0) ier = 3
      if(result.ne.zero.and.area.ne.zero)go to 175
      if(abserr.gt.errsum)go to 190
      if(area.eq.zero) go to 210
      go to 180
  175 if(abserr/cdabs(result).gt.errsum/cdabs(area))go to 190 !CDABSyes
c
c           test on divergence.
c
  180 if(ksgn.eq.(-1).and.dmax1(cdabs(result),cdabs(area)).le. !CDABSyes
     *  resabs*0.1d-01) go to 210
      if(0.1d-01.gt.dreal(result/area).or.
     *  dreal(result/area).gt.0.1d+03.or.
     *  errsum.gt.cdabs(area)) ier = 6 !CDABSyes
      go to 210
c
c           compute global integral sum.
c
  190 result = zero
      do 200 k = 1,last
        result = result+rlist(k)
  200 continue
      abserr = errsum
  210 if(ier.gt.2) ier = ier-1
      result = result*sign
  999 return
      end
      subroutine avh_oni_dqelg(n,epstab,result,abserr,res3la,nres)
c***begin prologue  dqelg
c***refer to  dqagie,dqagoe,dqagpe,dqagse
c***routines called  d1mach
c***revision date  830518   (yymmdd)
c***keywords  epsilon algorithm, convergence acceleration,
c             extrapolation
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math & progr. div. - k.u.leuven
c***purpose  the routine determines the limit of a given sequence of
c            approximations, by means of the epsilon algorithm of
c            p.wynn. an estimate of the absolute error is also given.
c            the condensed epsilon table is computed. only those
c            elements needed for the computation of the next diagonal
c            are preserved.
c***description
c
c           epsilon algorithm
c           standard fortran subroutine
c           double precision version
c
c           parameters
c              n      - integer
c                       epstab(n) contains the new element in the
c                       first column of the epsilon table.
c
c              epstab - double precision
c                       vector of dimension 52 containing the elements
c                       of the two lower diagonals of the triangular
c                       epsilon table. the elements are numbered
c                       starting at the right-hand corner of the
c                       triangle.
c
c              result - double precision
c                       resulting approximation to the integral
c
c              abserr - double precision
c                       estimate of the absolute error computed from
c                       result and the 3 previous results
c
c              res3la - double precision
c                       vector of dimension 3 containing the last 3
c                       results
c
c              nres   - integer
c                       number of calls to the routine
c                       (should be zero at first call)
c
c***end prologue  dqelg
c
      double complex epstab,e0,e1,e2,e3,res,result,res3la
      double precision abserr,dabs,delta1,delta2,delta3,dmax1,
     *  avh_oni_d1mach,
     *  epmach,epsinf,error,err1,err2,err3,e1abs,
     *  oflow,ss,tol1,tol2,tol3
      integer i,ib,ib2,ie,indx,k1,k2,k3,limexp,n,newelm,nres,num
      dimension epstab(52),res3la(3)
c
c           list of major variables
c           -----------------------
c
c           e0     - the 4 elements on which the computation of a new
c           e1       element in the epsilon table is based
c           e2
c           e3                 e0
c                        e3    e1    new
c                              e2
c           newelm - number of elements to be computed in the new
c                    diagonal
c           error  - error = abs(e1-e0)+abs(e2-e1)+abs(new-e2)
c           result - the element in the new diagonal with least value
c                    of error
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           oflow is the largest positive magnitude.
c           limexp is the maximum number of elements the epsilon
c           table can contain. if this number is reached, the upper
c           diagonal of the epsilon table is deleted.
c
c***first executable statement  dqelg
      epmach = avh_oni_d1mach(4)
      oflow = avh_oni_d1mach(2)
      nres = nres+1
      abserr = oflow
      result = epstab(n)
      if(n.lt.3) go to 100
      limexp = 50
      epstab(n+2) = epstab(n)
      newelm = (n-1)/2
      epstab(n) = dcmplx(oflow)
      num = n
      k1 = n
      do 40 i = 1,newelm
        k2 = k1-1
        k3 = k1-2
        res = epstab(k1+2)
        e0 = epstab(k3)
        e1 = epstab(k2)
        e2 = res
        e1abs = cdabs(e1) !CDABSyes
        delta2 = e2-e1
        err2 = dabs(delta2)
        tol2 = dmax1(cdabs(e2),e1abs)*epmach !CDABSyes
        delta3 = e1-e0
        err3 = dabs(delta3)
        tol3 = dmax1(e1abs,cdabs(e0))*epmach !CDABSyes
        if(err2.gt.tol2.or.err3.gt.tol3) go to 10
c
c           if e0, e1 and e2 are equal to within machine
c           accuracy, convergence is assumed.
c           result = e2
c           abserr = abs(e1-e0)+abs(e2-e1)
c
        result = res
        abserr = err2+err3
c ***jump out of do-loop
        go to 100
   10   e3 = epstab(k1)
        epstab(k1) = e1
        delta1 = e1-e3
        err1 = dabs(delta1)
        tol1 = dmax1(e1abs,cdabs(e3))*epmach !CDABSyes
c
c           if two elements are very close to each other, omit
c           a part of the table by adjusting the value of n
c
        if(err1.le.tol1.or.err2.le.tol2.or.err3.le.tol3) go to 20
        ss = 0.1d+01/delta1+0.1d+01/delta2-0.1d+01/delta3
        epsinf = cdabs(ss*e1) !CDABSyes
c
c           test to detect irregular behaviour in the table, and
c           eventually omit a part of the table adjusting the value
c           of n.
c
        if(epsinf.gt.0.1d-03) go to 30
   20   n = i+i-1
c ***jump out of do-loop
        go to 50
c
c           compute a new element and eventually adjust
c           the value of result.
c
   30   res = e1+dcmplx(0.1d+01/ss)
        epstab(k1) = res
        k1 = k1-2
        error = err2+cdabs(res-e2)+err3 !CDABSyes
        if(error.gt.abserr) go to 40
        abserr = error
        result = res
   40 continue
c
c           shift the table.
c
   50 if(n.eq.limexp) n = 2*(limexp/2)-1
      ib = 1
      if((num/2)*2.eq.num) ib = 2
      ie = newelm+1
      do 60 i=1,ie
        ib2 = ib+2
        epstab(ib) = epstab(ib2)
        ib = ib2
   60 continue
      if(num.eq.n) go to 80
      indx = num-n+1
      do 70 i = 1,n
        epstab(i)= epstab(indx)
        indx = indx+1
   70 continue
   80 if(nres.ge.4) go to 90
      res3la(nres) = result
      abserr = oflow
      go to 100
c
c           compute error estimate
c
   90 abserr = cdabs(result-res3la(3))+cdabs(result-res3la(2)) !CDABSyes
     *  +cdabs(result-res3la(1)) !CDABSyes
      res3la(1) = res3la(2)
      res3la(2) = res3la(3)
      res3la(3) = result
  100 abserr = dmax1(abserr,0.5d+01*epmach*cdabs(result)) !CDABSyes
      return
      end
      subroutine avh_oni_dqk21(fncn,a,b,result,abserr,resabs,resasc)
c***begin prologue  dqk21
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a2
c***keywords  21-point gauss-kronrod rules
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  to compute i = integral of f over (a,b), with error
c                           estimate
c                       j = integral of abs(f) over (a,b)
c***description
c
c           integration rules
c           standard fortran subroutine
c           double precision version
c
c           parameters
c            on entry
c              f      - double precision
c                       function subprogram defining the integrand
c                       function f(x). the actual name for f needs to be
c                       declared e x t e r n a l in the driver program.
c
c              a      - double precision
c                       lower limit of integration
c
c              b      - double precision
c                       upper limit of integration
c
c            on return
c              result - double precision
c                       approximation to the integral i
c                       result is computed by applying the 21-point
c                       kronrod rule (resk) obtained by optimal addition
c                       of abscissae to the 10-point gauss rule (resg).
c
c              abserr - double precision
c                       estimate of the modulus of the absolute error,
c                       which should not exceed abs(i-result)
c
c              resabs - double precision
c                       approximation to the integral j
c
c              resasc - double precision
c                       approximation to the integral of abs(f-i/(b-a))
c                       over (a,b)
c
c***references  (none)
c***routines called  avh_oni_d1mach
c***end prologue  avh_oni_dqk21
c
      double complex fncn,fc,fval1,fval2,resk,reskh,fv1,fv2,fsum,resg,
     *  result
      double precision a,absc,abserr,b,centr,dabs,dhlgth,dmax1,dmin1,
     *  avh_oni_d1mach,epmach,hlgth,
     *  resabs,resasc,
     *  uflow,wg,wgk,xgk
      integer j,jtw,jtwm1
      external fncn
c
      dimension fv1(10),fv2(10),wg(5),wgk(11),xgk(11)
c
c           the abscissae and weights are given for the interval (-1,1).
c           because of symmetry only the positive abscissae and their
c           corresponding weights are given.
c
c           xgk    - abscissae of the 21-point kronrod rule
c                    xgk(2), xgk(4), ...  abscissae of the 10-point
c                    gauss rule
c                    xgk(1), xgk(3), ...  abscissae which are optimally
c                    added to the 10-point gauss rule
c
c           wgk    - weights of the 21-point kronrod rule
c
c           wg     - weights of the 10-point gauss rule
c
c
c gauss quadrature weights and kronron quadrature abscissae and weights
c as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
c bell labs, nov. 1981.
c
      data wg  (  1) / 0.0666713443 0868813759 3568809893 332 d0 /
      data wg  (  2) / 0.1494513491 5058059314 5776339657 697 d0 /
      data wg  (  3) / 0.2190863625 1598204399 5534934228 163 d0 /
      data wg  (  4) / 0.2692667193 0999635509 1226921569 469 d0 /
      data wg  (  5) / 0.2955242247 1475287017 3892994651 338 d0 /
c
      data xgk (  1) / 0.9956571630 2580808073 5527280689 003 d0 /
      data xgk (  2) / 0.9739065285 1717172007 7964012084 452 d0 /
      data xgk (  3) / 0.9301574913 5570822600 1207180059 508 d0 /
      data xgk (  4) / 0.8650633666 8898451073 2096688423 493 d0 /
      data xgk (  5) / 0.7808177265 8641689706 3717578345 042 d0 /
      data xgk (  6) / 0.6794095682 9902440623 4327365114 874 d0 /
      data xgk (  7) / 0.5627571346 6860468333 9000099272 694 d0 /
      data xgk (  8) / 0.4333953941 2924719079 9265943165 784 d0 /
      data xgk (  9) / 0.2943928627 0146019813 1126603103 866 d0 /
      data xgk ( 10) / 0.1488743389 8163121088 4826001129 720 d0 /
      data xgk ( 11) / 0.0000000000 0000000000 0000000000 000 d0 /
c
      data wgk (  1) / 0.0116946388 6737187427 8064396062 192 d0 /
      data wgk (  2) / 0.0325581623 0796472747 8818972459 390 d0 /
      data wgk (  3) / 0.0547558965 7435199603 1381300244 580 d0 /
      data wgk (  4) / 0.0750396748 1091995276 7043140916 190 d0 /
      data wgk (  5) / 0.0931254545 8369760553 5065465083 366 d0 /
      data wgk (  6) / 0.1093871588 0229764189 9210590325 805 d0 /
      data wgk (  7) / 0.1234919762 6206585107 7958109831 074 d0 /
      data wgk (  8) / 0.1347092173 1147332592 8054001771 707 d0 /
      data wgk (  9) / 0.1427759385 7706008079 7094273138 717 d0 /
      data wgk ( 10) / 0.1477391049 0133849137 4841515972 068 d0 /
      data wgk ( 11) / 0.1494455540 0291690566 4936468389 821 d0 /
c
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the interval
c           hlgth  - half-length of the interval
c           absc   - abscissa
c           fval*  - function value
c           resg   - result of the 10-point gauss formula
c           resk   - result of the 21-point kronrod formula
c           reskh  - approximation to the mean value of f over (a,b),
c                    i.e. to i/(b-a)
c
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  dqk21
      epmach = avh_oni_d1mach(4)
      uflow = avh_oni_d1mach(1)
c
      centr = 0.5d+00*(a+b)
      hlgth = 0.5d+00*(b-a)
      dhlgth = dabs(hlgth)
c
c           compute the 21-point kronrod approximation to
c           the integral, and estimate the absolute error.
c
      resg = dcmplx(0.0d+00)
      fc = fncn(centr)
      resk = wgk(11)*fc
      resabs = cdabs(resk) !CDABSyes
      do 10 j=1,5
        jtw = 2*j
        absc = hlgth*xgk(jtw)
        fval1 = fncn(centr-absc)
        fval2 = fncn(centr+absc)
        fv1(jtw) = fval1
        fv2(jtw) = fval2
        fsum = fval1+fval2
        resg = resg+wg(j)*fsum
        resk = resk+wgk(jtw)*fsum
        resabs = resabs+wgk(jtw)*(cdabs(fval1)+cdabs(fval2)) !CDABSyes
   10 continue
      do 15 j = 1,5
        jtwm1 = 2*j-1
        absc = hlgth*xgk(jtwm1)
        fval1 = fncn(centr-absc)
        fval2 = fncn(centr+absc)
        fv1(jtwm1) = fval1
        fv2(jtwm1) = fval2
        fsum = fval1+fval2
        resk = resk+wgk(jtwm1)*fsum
        resabs = resabs+wgk(jtwm1)*(cdabs(fval1)+cdabs(fval2)) !CDABSyes
   15 continue
      reskh = resk*dcmplx(0.5d+00)
      resasc = wgk(11)*cdabs(fc-reskh) !CDABSyes
      do 20 j=1,10
        resasc = resasc+wgk(j)*(cdabs(fv1(j)-reskh)+cdabs(fv2(j)-reskh)) !CDABSyes
   20 continue
      result = resk*hlgth
      resabs = resabs*dhlgth
      resasc = resasc*dhlgth
      abserr = cdabs((resk-resg)*hlgth) !CDABSyes
      if(resasc.ne.0.0d+00.and.abserr.ne.0.0d+00)
     *  abserr = resasc*dmin1(0.1d+01,(0.2d+03*abserr/resasc)**1.5d+00)
      if(resabs.gt.uflow/(0.5d+02*epmach)) abserr = dmax1
     *  ((epmach*0.5d+02)*resabs,abserr)
      return
      end
      subroutine avh_oni_dqpsrt(limit,last,maxerr,ermax,elist,iord
     *,nrmax)
c***begin prologue  dqpsrt
c***refer to  dqage,dqagie,dqagpe,dqawse
c***routines called  (none)
c***revision date  810101   (yymmdd)
c***keywords  sequential sorting
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  this routine maintains the descending ordering in the
c            list of the local error estimated resulting from the
c            interval subdivision process. at each call two error
c            estimates are inserted using the sequential search
c            method, top-down for the largest error estimate and
c            bottom-up for the smallest error estimate.
c***description
c
c           ordering routine
c           standard fortran subroutine
c           double precision version
c
c           parameters (meaning at output)
c              limit  - integer
c                       maximum number of error estimates the list
c                       can contain
c
c              last   - integer
c                       number of error estimates currently in the list
c
c              maxerr - integer
c                       maxerr points to the nrmax-th largest error
c                       estimate currently in the list
c
c              ermax  - double precision
c                       nrmax-th largest error estimate
c                       ermax = elist(maxerr)
c
c              elist  - double precision
c                       vector of dimension last containing
c                       the error estimates
c
c              iord   - integer
c                       vector of dimension last, the first k elements
c                       of which contain pointers to the error
c                       estimates, such that
c                       elist(iord(1)),...,  elist(iord(k))
c                       form a decreasing sequence, with
c                       k = last if last.le.(limit/2+2), and
c                       k = limit+1-last otherwise
c
c              nrmax  - integer
c                       maxerr = iord(nrmax)
c
c***end prologue  avh_oni_dqpsrt
c
      double precision elist,ermax,errmax,errmin
      integer i,ibeg,ido,iord,isucc,j,jbnd,jupbn,k,last,limit,maxerr,
     *  nrmax
      dimension elist(last),iord(last)
c
c           check whether the list contains more than
c           two error estimates.
c
c***first executable statement  dqpsrt
      if(last.gt.2) go to 10
      iord(1) = 1
      iord(2) = 2
      go to 90
c
c           this part of the routine is only executed if, due to a
c           difficult integrand, subdivision increased the error
c           estimate. in the normal case the insert procedure should
c           start after the nrmax-th largest error estimate.
c
   10 errmax = elist(maxerr)
      if(nrmax.eq.1) go to 30
      ido = nrmax-1
      do 20 i = 1,ido
        isucc = iord(nrmax-1)
c ***jump out of do-loop
        if(errmax.le.elist(isucc)) go to 30
        iord(nrmax) = isucc
        nrmax = nrmax-1
   20    continue
c
c           compute the number of elements in the list to be maintained
c           in descending order. this number depends on the number of
c           subdivisions still allowed.
c
   30 jupbn = last
      if(last.gt.(limit/2+2)) jupbn = limit+3-last
      errmin = elist(last)
c
c           insert errmax by traversing the list top-down,
c           starting comparison from the element elist(iord(nrmax+1)).
c
      jbnd = jupbn-1
      ibeg = nrmax+1
      if(ibeg.gt.jbnd) go to 50
      do 40 i=ibeg,jbnd
        isucc = iord(i)
c ***jump out of do-loop
        if(errmax.ge.elist(isucc)) go to 60
        iord(i-1) = isucc
   40 continue
   50 iord(jbnd) = maxerr
      iord(jupbn) = last
      go to 90
c
c           insert errmin by traversing the list bottom-up.
c
   60 iord(i-1) = maxerr
      k = jbnd
      do 70 j=i,jbnd
        isucc = iord(k)
c ***jump out of do-loop
        if(errmin.lt.elist(isucc)) go to 80
        iord(k+1) = isucc
        k = k-1
   70 continue
      iord(i) = last
      go to 90
   80 iord(k+1) = last
c
c           set maxerr and ermax.
c
   90 maxerr = iord(nrmax)
      ermax = elist(maxerr)
      return
      end
