      module StringCast

        integer max_length
        parameter (max_length = 300)

        interface toStr
          module procedure toStr_int
          module procedure toStr_real 
          module procedure toStr_real_with_ndig
          module procedure toStr_real_with_format
          module procedure toStr_char_array
        end interface toStr

        contains

!       This removes all blank character at the head of the size 100
!       charachter array and places them at the end.
        function trim_ahead(input)
          character(max_length)      :: input
          character(max_length)      :: trim_ahead
          integer                    :: i
          integer                    :: first_char_index
          
          first_char_index = max_length
          do i=1,max_length
            if (input(i:i).ne.' ') then
              first_char_index = i
              EXIT 
            endif
          enddo
          
          do i=first_char_index,max_length
            trim_ahead((i-first_char_index+1):(i-first_char_index+1))=
     &                                                       input(i:i)
          enddo
          do i=(max_length-first_char_index+2),max_length
            trim_ahead(i:i)=' '
          enddo
        end function trim_ahead

!       Just to cast the max_width parameter to a string for formatting
        function get_width()
          character(max_length)        :: get_width 
          write(get_width,'(i20.20)') max_length
        end function get_width

        function toStr_char_array(input)
          character, dimension(:), intent(in) :: input
          character(max_length)               :: toStr_char_array
          integer i
          do i=1,max_length
            if (i.le.size(input)) then
               toStr_char_array(i:i)=input(i)
            else
               toStr_char_array(i:i)=' '
            endif
          enddo
        end function toStr_char_array

        function toStr_int(input)
          integer, intent(in)        :: input
          character(max_length)      :: toStr_int
          character(max_length)      :: tmp, tmp2
          integer                    :: i
          
          write(tmp,'(i'//get_width()//')') input
          toStr_int = trim_ahead(tmp)

        end function toStr_int

        function toStr_real(input)
          real*8, intent(in)           :: input
          character(max_length)        :: toStr_real

          toStr_real = toStr_real_with_ndig(input,16)
        end function toStr_real

!       The width will be automatically replaced, so leav it to 'w'
!       in the format specifier.
!       Example of call:  toStr_real_with_format(1.223204d0,'Fw.4') 
        function toStr_real_with_format(input, chosen_format)
          real*8, intent(in)            :: input
          character(len=*), intent(in)  :: chosen_format 
          character(max_length)         :: toStr_real_with_format
          character(max_length)         :: format_spec
          integer :: i, w_index
          
          w_index = -1
          do i=1,len(chosen_format)
           if (chosen_format(i:i).eq.'w') then
             w_index = i
             exit
           endif
          enddo
          if (w_index.eq.-1.or.w_index.eq.1) then
            write(toStr_real_with_format,'('//chosen_format//')') input 
          else
            write(toStr_real_with_format,'('//chosen_format(1:i-1)//
     &    TRIM(toStr(max_length))//chosen_format(i+1:len(chosen_format))
     &                                                      //')') input
          endif
          toStr_real_with_format = trim_ahead(toStr_real_with_format)
        end function toStr_real_with_format

        function toStr_real_with_ndig(input, n_digits)
          real*8, intent(in)         :: input
          integer, intent(in)        :: n_digits
          character(max_length)      :: toStr_real_with_ndig
          character(max_length)      :: format_spec
          
          format_spec = '(F'//TRIM(toStr(max_length))//'.'//
     &                               TRIM(toStr(n_digits))//')'

          write(toStr_real_with_ndig,format_spec) input 
          toStr_real_with_ndig = trim_ahead(toStr_real_with_ndig)
        end function toStr_real_with_ndig

      end module StringCast
