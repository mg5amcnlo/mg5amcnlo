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
