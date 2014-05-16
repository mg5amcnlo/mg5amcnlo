c************************************************************************
c**                                                                    **
c**           MadGraph/MadEvent Interface to FeynRules                 **
c**                                                                    **
c**          C. Duhr (Louvain U.) - M. Herquet (NIKHEF)                **
c**                                                                    **
c************************************************************************

      subroutine printout
      implicit none

      include 'coupl.inc'
      include 'input.inc'
      
      include 'formats.inc'

      write(*,*) '*****************************************************'
      write(*,*) '*               MadGraph/MadEvent                   *'
      write(*,*) '*        --------------------------------           *'
      write(*,*) '*          http://madgraph.hep.uiuc.edu             *'	
      write(*,*) '*          http://madgraph.phys.ucl.ac.be           *'
      write(*,*) '*          http://madgraph.roma2.infn.it            *'
      write(*,*) '*        --------------------------------           *'	
      write(*,*) '*                                                   *'
      write(*,*) '*          PARAMETER AND COUPLING VALUES            *'
      write(*,*) '*                                                   *'
      write(*,*) '*****************************************************'
      write(*,*)
     
      include 'param_write.inc'
      include 'coupl_write.inc'
 
      return
      end
