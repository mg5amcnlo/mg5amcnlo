subroutine extra_cnt(p, icnt, cnts)
C call the extra cnt corresponding to icnt
C may be a dummy function, depending on the process
implicit none
integer icnt, i
include 'nexternal.inc'
double precision p(0:3, nexternal-1)
include 'orders.inc'
double complex cnts(2,nsplitorders)

do i = 1, nsplitorders
cnts(1,i) = dcmplx(0d0,0d0)
cnts(2,i) = dcmplx(0d0,0d0)
enddo
%(iflines)s
return
end


integer function get_extra_cnt_color(icnt,ipart)
C return the color of the ipart-th particle of 
C counterterm icnt
integer icnt ipart
include 'nexternal.inc'
integer nexternalb
parameter(nexternalb=nexternal-1)
integer ncnt,i
parameter (ncnt=%(ncnt)d)
integer cnt_color(ncnt,nexternalb)
%(cnt_color_lines)s

if (icnt.gt.ncnt.or.icnt.le.0) then
write(*,*) 'ERROR#1 in get_extra_cnt_color', icnt
stop
endif
if (ipart.ge.nexternal.or.ipart.le.0) then
write(*,*) 'ERROR#2 in get_extra_cnt_color', ipart
stop
endif

get_extra_cnt_color=cnt_color(icnt,ipart)

return
end


integer function get_extra_cnt_pdg(icnt,ipart)
C return the pdg id of the ipart-th particle of 
C counterterm icnt
integer icnt ipart
include 'nexternal.inc'
integer nexternalb
parameter(nexternalb=nexternal-1)
integer ncnt,i
parameter (ncnt=%(ncnt)d)
integer cnt_pdg(ncnt,nexternalb)
%(cnt_pdg_lines)s

if (icnt.gt.ncnt.or.icnt.le.0) then
write(*,*) 'ERROR#1 in get_extra_cnt_pdg', icnt
stop
endif
if (ipart.ge.nexternal.or.ipart.le.0) then
write(*,*) 'ERROR#2 in get_extra_cnt_pdg', ipart
stop
endif

get_extra_cnt_pdg=cnt_pdg(icnt,ipart)

return
end


double precision function get_extra_cnt_charge(icnt,ipart)
C return the charge id of the ipart-th particle of 
C counterterm icnt
integer icnt ipart
include 'nexternal.inc'
integer nexternalb
parameter(nexternalb=nexternal-1)
integer ncnt,i
parameter (ncnt=%(ncnt)d)
double precision cnt_charge(ncnt,nexternalb)
%(cnt_charge_lines)s

if (icnt.gt.ncnt.or.icnt.le.0) then
write(*,*) 'ERROR#1 in get_extra_cnt_charge', icnt
stop
endif
if (ipart.ge.nexternal.or.ipart.le.0) then
write(*,*) 'ERROR#2 in get_extra_cnt_charge', ipart
stop
endif

get_extra_cnt_charge=cnt_charge(icnt,ipart)

return
end
