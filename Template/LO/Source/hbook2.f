        subroutine hbook2(id,inlabel,nx,xmin,xmax,ny,ymin,ymax,zinitial)
C
C               id = integer used to identify histogram to HFILL
C               inlabel = label to be written on the output by the plotting
C                               program (character of len <=40)
C               nx = number of x bins (integer)
C               xmin = min x value (real)
C               xmax = max x value (real)
C               ny,ymin,ymax = same for y values
C               zinitial = initial value for each bin (real)
C
        include 'hbook.inc'
        character*(*) inlabel

        if (nhist .eq. nhistmax) then
                print*,' Maximum number of histograms exceeded'
        else
                nhist = nhist+1
                label(nhist) = inlabel
                id number(nhist) = id
                single dim(nhist) = .false.
                k=pointer(nhist)
                pointer(nhist+1) = nx*ny+6+k
                data(k)=nx
                data(k+1)=xmin
                data(k+2)=xmax
                data(k+3)=ny
                data(k+4)=ymin
                data(k+5)=ymax
                do i=k+6,pointer(nhist+1)-1
                        data(i)=zinitial
                        end do
                end if
        return
        end
