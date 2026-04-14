c************************************************************************
c**                                                                    **
c**           MadGraph/MadEvent Interface to FeynRules                 **
c**                                                                    **
c**          C. Duhr (Louvain U.) - M. Herquet (NIKHEF)                **
c**                                                                    **
c************************************************************************

c Formats for printout output

c Simple real
 1    format(1x,a15,e13.5)
c Simple Complex
 2    format(1x,a15,e13.5,1x,e13.5)
c Real with mass dimension
 3    format(1x,a15,f11.5,' GeV')
c Chiral couplings
 4    format(1x,a15,e13.5,1x,e13.5,a15,e13.5,1x,e13.5)
 
 
c Formats for helas_coupling output

c Real
 11   format(a10,e13.5)
c Complex
 12   format(a10,e13.5,1x,e13.5 )
c Chiral
 13   format(a10,e13.5,1x,e13.5,1x,e13.5,1x,e13.5 )


