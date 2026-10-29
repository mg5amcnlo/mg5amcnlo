program audit
  implicit none
  double precision :: mcmass(21),pysudakov,q,m,s0,s1,r
  integer :: im,iq
  double precision :: masses(2)=(/500d0,1000d0/),cuts(5)=(/10d0,20d0,50d0,100d0,173d0/)
  external pysudakov
  mcmass=0d0
  include 'MCmasses_PYTHIA8.inc'
  do im=1,2
    m=masses(im)
    s0=pysudakov(200d0,m,21,1,mcmass)
    do iq=1,5
      q=cuts(iq);s1=pysudakov(q,m,21,1,mcmass);r=s1/s0
      write(*,'(2F8.1,4ES23.14)') m,q,s1,s0,r,r**4
    end do
  end do
end program
