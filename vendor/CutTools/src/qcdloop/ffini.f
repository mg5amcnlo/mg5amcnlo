*	$Id: ffini.f,v 1.1 1996/03/27 08:05:18 gj Exp $
*
*	glue routine for older versions of FF:
*	define ffinit, ffexit to be equal to ffini, ffexi
*	
*	when using CERN libs do *not* include this file, because ffinit 
*	already exists in packlib.
*
*	All programs written after 17-mar-1996 should work without this file
*	
 	subroutine ffinit
	call ffini
	end
 	subroutine ffexit
	call ffexi
	end
