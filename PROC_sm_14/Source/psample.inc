c
c Global variables used by psample
c
      integer                ihost(max_host),npnts,nans,icpu(max_host)
      integer                 ierror(max_host)
      character*30                   hostname(max_host),program
      common /sample_machine/ ihost,icpu,ierror,npnts,nans, 
     &                        hostname,program

