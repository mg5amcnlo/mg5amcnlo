      subroutine cite(key, context)
c***********************************************************************
c     Record that the reference identified by the INSPIRE texkey `key`
c     was used by this run, optionally for the purpose described by the
c     free-text `context`.
c
c     Each call appends a single line
c         key<TAB>context
c     to the per-process file
c         $MG5_CITATION_DIR/cite.<host>.<pid>.log
c     (de-duplicated within the process).  The orchestrating Python layer
c     collects every such file at the end of the run and turns them into a
c     ready-to-use citations.bib together with a human-readable summary.
c
c     A per-process file name means there is never a cross-process write
c     race, on any filesystem.  When MG5_CITATION_DIR is unset the routine
c     is a silent no-op, so it is safe to call unconditionally.  Any I/O
c     failure is swallowed: citation tracking must never abort a run.
c***********************************************************************
      implicit none
c
c     Arguments
c
      character*(*) key, context
c
c     Local parameters
c
      integer    maxcite
      parameter (maxcite=500)
      integer    reclen
      parameter (reclen=320)
c
c     Saved per-process state (the keys already written)
c
      character*(reclen) seen(maxcite)
      integer nseen
      save seen, nseen
      data nseen /0/
c
c     Local variables
c
      character*512  cdir
      character*1024 fname
      character*(reclen) record
      character*256  host
      integer dirlen, st, pid, i, lun
      logical used
c
c-----
c  Begin Code
c-----
c     enabled only when MG5_CITATION_DIR points somewhere
      call get_environment_variable('MG5_CITATION_DIR',
     &     cdir, dirlen, st)
      if (dirlen .le. 0) return
      if (dirlen .gt. len(cdir)) return
c
c     the de-duplication record is  key<TAB>context
      record = trim(key)//char(9)//trim(context)
c
c     guard the shared state/file against OpenMP threads of this process
c$omp critical (mg5_cite)
      used = .false.
      do i = 1, nseen
         if (seen(i) .eq. record) used = .true.
      enddo
c
      if (.not. used) then
         if (nseen .lt. maxcite) then
            nseen = nseen + 1
            seen(nseen) = record
         endif
c        build  <dir>/cite.<host>.<pid>.log
         host = 'localhost'
         call hostnm(host, st)
         pid = getpid()
         write(fname, '(a,a,a,a,i0,a)') cdir(1:dirlen),
     &        '/cite.', trim(host), '.', pid, '.log'
c        append the record, swallowing any failure
         lun = 87
         open(unit=lun, file=fname, status='unknown',
     &        position='append', iostat=st)
         if (st .eq. 0) then
            write(lun, '(a)', iostat=st) trim(record)
            close(lun)
         endif
      endif
c$omp end critical (mg5_cite)
c
      return
      end
