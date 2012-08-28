c Set the following variables to .true. if the corresponding
c uncertainty has to be computed at the parton-level NLO
      logical doNLOscaleunc,doNLOPDFunc
      parameter (doNLOscaleunc=.false.)
      parameter (doNLOPDFunc=.false.)

c Number of possible scale choices for uncertainty studies, including 
c the central one
      integer numscales_init
      parameter (numscales_init=3)

c ID numbers (LHAPDF) of the central set, and first and last error sets
      integer idefPDF,ifirstPDF,ilastPDF
      parameter (idefPDF=21100)
      parameter (ifirstPDF=21101)
      parameter (ilastPDF=21140)

c Central value and lower and upper limits of the scale variation range,
c for the factorization scale. These are fractions of the default
c scale used in the computation. The cross section will be computed
c numscales times, thus including the central value
      double precision yfactF0,yfactFlow,yfactFupp
      parameter (yfactF0=1d0)
      parameter (yfactFlow=0.5d0)
      parameter (yfactFupp=2.d0)

c Central value and lower and upper limits of the scale variation range,
c for the renormalization scale. These are fractions of the default
c scale used in the computation. The cross section will be computed
c numscales times, thus including the central value
      double precision yfactR0,yfactRlow,yfactRupp
      parameter (yfactR0=1d0)
      parameter (yfactRlow=0.5d0)
      parameter (yfactRupp=2.d0)

c Same as mu[R/F/Q]_over_ref, but used only in the context of reweighting
      double precision ymuR_over_ref,ymuF1_over_ref,ymuF2_over_ref,
     #                 yQES_over_ref
      common/cNLOrwgt1/ymuR_over_ref,ymuF1_over_ref,ymuF2_over_ref,
     #                 yQES_over_ref

c Fractions of reference scales, to be used for the computation
c of scale uncertainties
      double precision yfactR(maxscales),yfactF(maxscales)
      common/cNLOrwgt2/yfactR,yfactF

c IDs (LHAPDF) of PDFs used for the computation of PDF uncertainties.
c The central set idpdf(0) presently coincides with the default set used 
c in the computation, but this condition may be relaxed
      integer idpdf(0:maxPDFs)
      common/cNLOrwgt3/idpdf

c Total number of PDFs used in the computation (ilastPDF-ifirstPDF+2),
c including the central one
      integer numPDFs
      common/cNLOrwgt4/numPDFs
