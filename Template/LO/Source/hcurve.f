C--------------------------------------------
C
C		Routine to dump histogram data to a file
C
	subroutine hcurve(id,filename)	
C
C		Dumps current histogram number id to file 'filename' and
C		clears histogram id.
C
	include 'hbook.inc'
	character*(*) filename
        real sum,npts

	if (nhist .eq. 0) return
	open (unit=69,name=filename,status='unknown')
	do i = 1, nhist
		if (id .eq. idnumber(i)) go to 10
		end do
	return
10	continue
	k = pointer(i)
	nx = int(data(k)+.1)
	xmin = data(k+1)
	xmax = data(k+2)
	xbinsize = (xmax-xmin)/nx
	if (single dim(i)) then
           sum=0
           npts=0
           do m=1,nx
              sum=sum+data(k+2+m)
              npts=npts+npoints(k+2+m)
           enddo
           write (69,300) label(i)(1:labelleng(label(i)))
           write (69,700) (xmin+(m-.5)*xbinsize,
     $          data(k+2+m),sqrt(abs(error(k+2+m))),
     $          npoints(k+2+m)/(npts*sum+1e-23),m=1,nx)
	else
		ny = int(data(k+3) + .1)
		ymin = data(k+4)
		ymax = data(k+5)
		ybinsize = (ymax-ymin)/ny
		write (69,300) label(i)(1:labelleng(label(i)))
		k = k + 5
		do n=1,ny
                   fixed y =  ymin + (n-.5)*ybinsize
                   write (69,500) (xmin+(m-.5)*xbinsize,fixed y,
     $                  data(k+m),m=1,nx)
		   write(69,*) 
                   k = k + nx
                end do
		end if
	close (unit=69)
	return
300	format ('# Histogram ',a)
400	format (1x,2g15.6)
500	format (1x,3g15.6)
700	format (1x,4g15.6)
	end
C
C
C
C
	function labelleng(string)
	character*(*) string

	do i=len(string),1,-1
		if (string(i:i) .ne. ' ') then
			labelleng=i
			return
			end if
		end do
	labelleng=1
	return
	end
