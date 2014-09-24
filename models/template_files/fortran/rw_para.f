c************************************************************************
c**                                                                    **
c**           MadGraph/MadEvent Interface to FeynRules                 **
c**                                                                    **
c**          C. Duhr (Louvain U.) - M. Herquet (NIKHEF)                **
c**                                                                    **
c************************************************************************

      subroutine setpara(param_name)
      implicit none

      character*(*) param_name
      logical readlha

      %(includes)s

      integer maxpara
      parameter (maxpara=5000)
      
      integer npara
      character*20 param(maxpara),value(maxpara)

      %(load_card)s
      include 'param_read.inc'
      call coup()

      return

      end

      subroutine setParamLog(OnOff)

      logical OnOff
      logical WriteParamLog
      data WriteParamLog/.TRUE./
      common/IOcontrol/WriteParamLog

      WriteParamLog = OnOff

      end

      subroutine setpara2(param_name)
      implicit none

      character(512) param_name

      integer k
      logical found

      character(512) ParamCardPath
      common/ParamCardPath/ParamCardPath

      if (param_name(1:1).ne.' ') then
        ! Save the basename of the param_card for the ident_card.
        ! If no absolute path was used then this ParamCardPath
        ! remains empty
        ParamCardPath = '.'
        k = LEN(param_name)
        found = .False.
        do while (k.ge.1.and..not.found)
          if (param_name(k:k).eq.'/') then
              found=.True.
          endif
          k=k-1
        enddo
        if (k.ge.1) then
          ParamCardPath(1:k)=param_name(1:k)
        endif
        call setpara(param_name)
      endif
      if (param_name(1:1).eq.'*') then
         ! Dummy call to printout so that it is available in the
         ! dynamic library for MadLoop BLHA2
         ! In principle the --whole-archive option of ld could be 
         ! used but it is not always supported
         call printout()
         call setParamLog(.True.)
      endif
      return

      end
