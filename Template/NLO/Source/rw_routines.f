      subroutine case_trap2(name)
c**********************************************************    
c change the string to lowercase if the input is not
c**********************************************************
      implicit none
c
c     ARGUMENT
c      
      character*(*) name
c
c     LOCAL
c
      integer i,k

      do i=1,len(name)
         k=ichar(name(i:i))
         if(k.ge.65.and.k.le.90) then  !upper case A-Z
            k=ichar(name(i:i))+32   
            name(i:i)=char(k)        
         endif
      enddo

      return
      end


      subroutine to_upper(name)
c**********************************************************    
c change the string to uppercase if the input is not
c**********************************************************
      implicit none
c
c     ARGUMENT
c      
      character*(*) name
c
c     LOCAL
c
      integer i,k

      do i=1,len(name)
         k=ichar(name(i:i))
         if(k.gt.90) then  !upper case A-Z
            k=ichar(name(i:i))-32   
            name(i:i)=char(k)        
         endif
      enddo

      return
      end
