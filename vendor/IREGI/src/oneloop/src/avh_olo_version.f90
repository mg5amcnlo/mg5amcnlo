!
! Copyright (C) 2014 Andreas van Hameren. 
!
! This file is part of OneLOop-3.4.
!
! OneLOop-3.4 is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! OneLOop-3.4 is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with OneLOop-3.4.  If not, see <http://www.gnu.org/licenses/>.
!


module avh_olo_version
  implicit none
  private
  public :: olo_version
  logical ,save :: done=.false.
contains
  subroutine olo_version
  if (done) return ;done=.true.
  write(*,'(a72)') '########################################################################'
  write(*,'(a72)') '#                                                                      #'
  write(*,'(a72)') '#                      You are using OneLOop-3.4                       #'
  write(*,'(a72)') '#                                                                      #'
  write(*,'(a72)') '# for the evaluation of 1-loop scalar 1-, 2-, 3- and 4-point functions #'
  write(*,'(a72)') '#                                                                      #'
  write(*,'(a72)') '# author: Andreas van Hameren <hamerenREMOVETHIS@ifj.edu.pl>           #'
  write(*,'(a72)') '#   date: 02-01-2014                                                   #'
  write(*,'(a72)') '#                                                                      #'
  write(*,'(a72)') '# Please cite                                                          #'
  write(*,'(a72)') '#    A. van Hameren,                                                   #'
  write(*,'(a72)') '#      Comput.Phys.Commun. 182 (2011) 2427-2438, arXiv:1007.4716       #'
  write(*,'(a72)') '#    A. van Hameren, C.G. Papadopoulos and R. Pittau,                  #'
  write(*,'(a72)') '#      JHEP 0909:106,2009, arXiv:0903.4665                             #'
  write(*,'(a72)') '# in publications with results obtained with the help of this program. #'
  write(*,'(a72)') '#                                                                      #'
  write(*,'(a72)') '########################################################################'
  end subroutine
end module
