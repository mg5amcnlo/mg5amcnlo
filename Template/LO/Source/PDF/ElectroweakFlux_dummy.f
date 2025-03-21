       double precision function eva_get_pdf_by_PID(vPID,fPID,vpol,fLpol,x,mu2,ievo)
       implicit none
       integer ievo ! =0 for evolution by q^2 (!=0 for evolution by pT^2)
       integer vPID,fPID,vpol
       double precision fLpol,x,mu2
       write(*,*) "WRONG PDF linked"
       eva_get_pdf_by_PID = 1.0
       stop 1
       return
       end
