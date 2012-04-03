      program write_ajob_basic
      implicit none
      integer lname
      character*30 mname
      character*30 fname
      integer run_cluster
      common/c_run_mode/run_cluster
      run_cluster=1
      fname='genE'
      lname=4
      mname='me'
      call open_bash_file(26,fname,lname,mname)
      write(26,'(a$)') ' <<TAG>>  '
      call close_bash_file(26)
      end
