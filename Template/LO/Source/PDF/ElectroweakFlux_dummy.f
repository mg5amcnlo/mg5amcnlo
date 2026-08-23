       double precision function eva_get_pdf_by_PID(vPID,fPID,vpol,fLpol,x,mu2)
       implicit none
       integer vPID,fPID,vpol
       double precision fLpol,x,mu2
       write(*,*) "EWFlux_dummy: WRONG PDF linked"
       eva_get_pdf_by_PID = 1.0
       stop 1
       return
       end
