C-----------------------------------------------
C
C	Routine to initialize a one-independent-variable histogram
C
	subroutine hbook1(id,inlabel,nx,xmin,xmax,zinitial)
C
C		id = integer used to identify histogram to HFILL
C		inlabel = label to be written on the output by the plotting
C				program (character of len <=40)
C		nx = number of x bins (integer)
C		xmin = min x value (real)
C		xmax = max x value (real)
C		zinitial = initial value for each bin (real)
C
	include 'hbook.inc'
	character*(*) inlabel

	if (nhist .eq. nhistmax) then
		print*,' Maximum number of histograms exceeded'
	else
		nhist = nhist+1
		label(nhist) = inlabel
		id number(nhist) = id
		single dim(nhist) = .true.
		k=pointer(nhist)
		pointer(nhist+1) = nx+3+k
		data(k)=nx
		data(k+1)=xmin
		data(k+2)=xmax
		do i=k+3,pointer(nhist+1)-1
			data(i)=zinitial
			error(i)=zinitial**2
			end do
		end if
	return
	end
