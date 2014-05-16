c Write the showered events in stdhep format. Requires the stdhep (and
c mcfio) libraries. The event file will be identical to the Les Houches
c parton-level input event file with ".hep" appended.
C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END

C----------------------------------------------------------------------
      SUBROUTINE PYABEG
C     USER''S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      implicit none
      integer lok,niter,maxevv
      character*80 output_file
      integer istrstd
      common/canalysis/istrstd
      CHARACTER*50 QQIN         ! LH event file
      COMMON/VVJIN/QQIN
      COMMON/CMAXEVV/MAXEVV
c For the stdhep event file, use the input file with ".hep" appended
      output_file=QQIN
      write(output_file(index(QQIN,' '):),'(a)') '.hep'
      
      NITER = MAXEVV !number of events
C...Open STDHEP output file
      call stdxwinit(output_file,'PYTHIA file',NITER,istrstd,lok)
      
      if (lok.ne.0) then
         print *,'Failed to open output file ',output_file,
     &        '. Quitting!'
         stop
      endif
C   Store Pythia run info in STDHEP common block
      call stdflpyxsec(NITER)
C   Write STDHEP begin run record
      call stdxwrt(100,istrstd,lok)
      if (lok.ne.0) then
         print '('' Write failure on'', '' begin run record: '',i5)',lok
         stop
      endif
      PRINT *,'stdhep initialized'
 999  END


C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVT)
C     USER''S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      implicit none
      integer lok,IEVT
      integer istrstd
      common/canalysis/istrstd
C...Finalize STDHEP file
      call stdflpyxsec(IEVT)
      call stdxwrt(200,istrstd,lok) ! write STDHEP end run record
      if (lok.ne.0) then
         print '('' Write failure on'', '' end run record: '',i5)',lok
      endif
      call stdxend(istrstd)
 999  END

C----------------------------------------------------------------------
      SUBROUTINE PYANAL
C     USER''S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      implicit none
      integer lok
      integer istrstd
      common/canalysis/istrstd
C...Convert pythia event to stdhep format
      call lunhep(1)
C...Write event to STDHEP output file
      call stdxwrt(4,istrstd,lok)
      if (lok.ne.0) then
         print '('' Failure writing event '',i5)',lok
      endif
 999  END
