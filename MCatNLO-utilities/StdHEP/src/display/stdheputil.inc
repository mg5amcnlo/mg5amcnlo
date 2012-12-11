c
c	stdheputil.inc: common block for stdHepFortranUtil.F 
c
	integer file_type, xdr_stream, init_zebra, zeb_file_pointer 
	common /stdheputil/  file_type, xdr_stream, init_zebra,
     &       zeb_file_pointer
	save stdheputil
