c-----------------------------------------------------------
c Chapter 7: Next permutation of n letters(p59) 
c-----------------------------------------------------------
c   Name of subroutine: NEXPER
c
c   Algorithm:Generating next permutation of 1,2,...n.
c
c   input:    n=5
c   complier: f77 nexper_2.f
c-----------------------------------------------------------

c      parameter(n=6)
c      integer a(n),b(n),i
c      logical mtc,even
c      mtc=.false.
c  10  call nexper(n,a,mtc,even)
c      write(*,*)(a(i),i=1,n)
c      if(mtc)goto 10
c      stop
c      end

c-----Subroutine begins here--------------------------------
      subroutine nexper(n,a,mtc,even)
c next permutation of {1,...,n}. Ref NW p 59.
      integer a(*),s,d
      logical mtc,even
      integer i,j,n,i1,nm3,ia
      nm3=n-3
      if(mtc)goto 10
      do 1 i=1,n
    1 a(i)=i
      mtc=.true.
    5 even=.true.
      if(n.eq.1)goto 8
    6 if(a(n).ne.1.or.a(1).ne.2+mod(n,2))return
      if(n.le.3)goto 8
      do 7 i=1,nm3
      if(a(i+1).ne.a(i)+1)return
    7 continue
    8 mtc=.false.
      return
   10 if(n.eq.1)goto 27
      if(.not.even)goto 20
      ia=a(1)
      a(1)=a(2)
      a(2)=ia
      even=.false.
      goto 6
   20 s=0
      do 26 i1=2,n
   25 ia=a(i1)
      i=i1-1
      d=0
      do 30 j=1,i
   30 if(a(j).gt.ia) d=d+1
      s=d+s
      if(d.ne.i*mod(s,2)) goto 35
   26 continue
   27 a(1)=0
      goto 8
   35 m=mod(s+1,2)*(n+1)
      do 40 j=1,i
      if(isign(1,a(j)-ia).eq.isign(1,a(j)-m))goto 40
      m=a(j)
      l=j
   40 continue
      a(l)=ia
      a(i1)=m
      even=.true.
      return
      end
c-----------------------------------------------------------
c
c    Calucul de la factorielle    
c
c-----------------------------------------------------------
c-----Subroutine begins here--------------------------------
      integer function factoriel(n)
	integer res,n,a
	a=n
        res=1
	do while (n.GT.1)
		res=res*n
		n=n-1
	enddo
	factoriel=res
        n=a
	return
	end


