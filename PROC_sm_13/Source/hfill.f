C----------------------------------------------
C
C		Routine to add zincrement to a bin in a histogram
C
	subroutine hfill(id,x,y,zincrement)
C
C		id = integer associated with the histogram
C		x  = x value to locate bin (real)
C		y  = y value to locate bin (real) [ignored for 1-dim histo]
C		zincrement = value to be added to bin specified by (x,y)
C
	include 'hbook.inc'
	data nhist/0/,pointer(1)/1/

	do i=1,nhist
		if (id number(i) .eq. id) go to 10
		end do
	print*,' id number ',id,' does not belong to a current histogram'
	return
10	continue
	k = pointer(i)
	nx=data(k)+.1
	ixoff = (x-data(k+1))/(data(k+2)-data(k+1))*data(k)+1
	if (ixoff .le. 0 .or. ixoff .gt. nx) return
	if (single dim(i)) then
		data(k+2+ixoff)=data(k+2+ixoff)+zincrement
		error(k+2+ixoff)=error(k+2+ixoff)+zincrement**2
		npoints(k+2+ixoff)=npoints(k+2+ixoff)+1.
	else
		ny=data(k+3)+.1
		iyoff = (y-data(k+4))/(data(k+5)-data(k+4))*data(k+3)+1
		if (iyoff .le. 0 .or. iyoff .gt. ny) return
		ioff = nx*(iyoff-1)+ixoff
		data(k+5+ioff)=data(k+5+ioff)+zincrement
		end if
	return
	end
