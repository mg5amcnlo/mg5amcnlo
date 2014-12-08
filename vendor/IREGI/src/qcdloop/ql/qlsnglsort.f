      subroutine qlsnglsort(n,arr)
C---order on the basis of abs value
      INTEGER n,i,j
      DOUBLE PRECISION arr(n),a
      do 12 j=2,n
        a=arr(j)
        do 11 i=j-1,1,-1
          if(abs(arr(i)).le. abs(a))goto 10
          arr(i+1)=arr(i)
11      continue
        i=0
10      arr(i+1)=a
12    continue
      return
      END
