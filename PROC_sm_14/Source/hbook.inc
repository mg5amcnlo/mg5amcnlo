C		Internal common blocks for pheno/hbook routines
C
C		LABELS(i) = label for histogram i
C		nhist	  = number of histograms (starts as 0, max is 20)
C		idnumber(i) = code number to identify histograms in HFILL
C		pointer(i)  = index of beginning of data for histo # i
C		single dim(i) = .true. if single variable, .false. if double
C
	parameter (nhistmax=20,nhistmax1=nhistmax+1)
	real data(10000),error(10000),npoints(10000)
	integer pointer(nhistmax1),id number(nhistmax)
	logical single dim(nhistmax)
	character*40 label(nhistmax)

       common /hbooklabel/ label
       common /hbookarrays/nhist,id number,pointer,
     $       single dim,data,error,npoints
