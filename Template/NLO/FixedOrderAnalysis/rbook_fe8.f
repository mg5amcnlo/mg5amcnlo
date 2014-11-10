c     --------------------------------------------------------------------
c     RBOOK -- A simple fortran interface to ROOT histogramming
c
c     This file implements the FORTRAN front-end functions that call 
c     the C++ back-end functions
c
c     (C) 31/08/2006 NIKHEF / Wouter Verkerke
c     -------------------------------------------------------------------
c
c Modified on 22/11/2013 by SF to use double precision
c
c The following routines allow booking and filling of histograms
c
c - rinit(filename)
c    character*(*) filename
c      Opens a file with name filename; this is were root stuff will go
c
c - rwrit()
c      Writes to root file opened with rinit
c
c - rclos()
c      Closes the root file; frees memory
c
c - rbook(id,histname,binsize,xlo,xhi)
c    character*(*) histname
c    integer id
c    double precision binsize,xlo,xhi
c      Books a one-dimensional histogram, identified by integer id and
c      tagged with name histname. The x axis ranges from xlo to xhi, with
c      bin size equal to binsize
c
c - rbook2(id,histname,xbinsize,xlo,xhi,ybinsize,ylo,yhi)
c    character*(*) histname
c    integer id
c    double precision xbinsize,xlo,xhi,ybinsize,ylo,yhi
c      Analogous to rbook(), but relevant to a two-dimensional histogram
c
c - rfill(id,xval,wgt)
c    integer id
c    double precision xval,wgt
c      Fills one-dimensional histogram identified by id [opened with 
c      rbook(id,...)]; x value equal to xval, weight equal to wgt
c
c - rfill2(id,xval,yval,wgt)
c    integer id
c    double precision xval,yval,wgt
c      Analogous to rfill(), but relevant to a two-dimensional histogram
c

c      ----------------------------------------------------
c      RINIT front end -- Open a ROOT file with given name
c      ----------------------------------------------------
       subroutine rinit(name)
       character*(*) name

c      -- Call C++ back end function --
       call rinit_be_(name,len(name)) 

       return
       end


c      --------------------------------------
c      RWRIT front end -- Write to ROOT file
c      --------------------------------------
       subroutine rwrit()
       
c      -- Call C++ back end function --
       call rwrit_be_

       return 
       end




c      --------------------------------------
c      RCLOSE front end -- Close the ROOT file
c      --------------------------------------
       subroutine rclos()
       
c      -- Call C++ back end function --
       call rclos_be_

       return 
       end



c      --------------------------------------------
c      RBOOK front end -- Create ROOT TH1 histogram
c      --------------------------------------------
       subroutine rbook(id,name,xbin,xlo,xhi)
       character*(*) name
       integer id,nbin
       double precision xbin,xlo,xhi

c      APPLgrid commons
       include "reweight_appl.inc"
       include "appl_common.inc"
       integer iappl
       common /for_applgrid/ iappl

       nbin = int((xhi-xlo)/(xbin*0.9999d0))

c      Initialize the grids only if the switch "iappl" is different from zero
c      and if the title string containes the word "central" and does not contain
c      the word "Born".
       if(iappl.ne.0.and.index(name,"central").ne.0.and.
     1                   index(name,"Born").eq.0)then
c      Observable parameters
          appl_obs_nbins = nbin
          appl_obs_min   = xlo
          appl_obs_max   = xhi
c      Initialize APPLgrid routines
          call APPL_init
c      Keep track of the position of this histogram
          nh_obs = nh_obs + 1
          ih_obs(nh_obs) = id
       endif

c      -- Call C++ back end function --
       call rbook_be_(id,name,len(name),nbin,xlo,xhi)
      
       return
       end


c      --------------------------------------------
c      RBOOK2 front end -- Create ROOT TH2 histogram
c      --------------------------------------------
       subroutine rbook2(id,name,xbin,xlo,xhi,ybin,ylo,yhi)
       character*(*) name
       integer id,nbinx,nbiny
       double precision xbin,xlo,xhi,ybin,ylo,yhi

       nbinx = int((xhi-xlo)/(xbin*0.9999d0))
       nbiny = int((yhi-ylo)/(ybin*0.9999d0))
c      -- Call C++ back end function --
       call rbook2_be_(id,name,len(name),nbinx,xlo,xhi,nbiny,ylo,yhi)
      
       return
       end


c      ------------------------------------------
c      RFILL front end -- Fill ROOT TH1 histogram
c      ------------------------------------------
       subroutine rfill(ihisto,xval,wgt)
       integer ihisto
       double precision xval,wgt

c      APPLgrid commons
       include "reweight_appl.inc"
       include "appl_common.inc"
       integer iappl
       common /for_applgrid/ iappl
       integer j
c
       if(iappl.ne.0)then
          do j=1,nh_obs
             if(ihisto.eq.ih_obs(j))then
                appl_obs_num   = j
                appl_obs_histo = xval
c      Fill the reference APPLgrid histograms
                call APPL_fill_ref
c      Fill the APPLgrid files
                call APPL_fill
             endif
          enddo
       endif

c      -- Call C++ back end function --
       call rfill_be_(ihisto,xval,wgt)

       return
       end


c      ------------------------------------------
c      RFILL2 front end -- Fill ROOT TH2 histogram
c      ------------------------------------------
       subroutine rfill2(ihisto,xval,yval,wgt)
       integer ihisto
       double precision xval,yval,wgt

c      -- Call C++ back end function --
       call rfill2_be_(ihisto,xval,yval,wgt)

       return
       end

c     -------------------------------------------
c     ROPERA frond end -- Manipulate histograms
c     ------------------------------------------
      subroutine ropera(ih1,oper,ih2,ih3,x,y)
      integer ih1,ih2,ih3
      character*(*) oper
      double precision x,y

c     APPLgrid commons
      include "reweight_appl.inc"
      include "appl_common.inc"
      integer iappl
      common /for_applgrid/ iappl
      integer j
      if(iappl.ne.0)then
         do j=1,nh_obs
            if(ih1.eq.ih_obs(j))then
               appl_obs_num = j
               call APPL_fill_ref_out
               call APPL_term
            endif
         enddo
      endif

c     -- Call C++ back end function --
      call ropera_be_(ih1,oper,len(oper),ih2,ih3,x,y)
      
      return
      end

c     -------------------------------------------
c     ROPERA frond end -- Copy histograms
c     ------------------------------------------
      subroutine rcopy(ih1,ih2) 
      integer ih1,ih2

c     -- Call C++ back end function --
      call rcopy_be_(ih1,ih2)
      
      return
      end


C This is a dummy subroutine that is called during integration, but
C should not do anything when ROOT is linked.
      subroutine accum(incl)
      logical incl
      end
      subroutine addfil(string)
      character*(*) string
      end
