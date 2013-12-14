	PROGRAM GENSKETCH1
c
c	Very simple program to generate a little detector sketch, 
c	made of line segments. One segment consist of two sets of xyz 
c	coordinates. 
c	Format is extremly simple : one line per segment, e.g., 6 f.p. numbers.
c	
c	This particular detector is square, 10 cm. in size, regularly spaced
c	in Z with Forward extension "logarytmic spaced" for uniform hit 
c	density.

	IMPLICIT NONE
	INTEGER I,J,K, IZ, IS
	REAL X1(8), X2(8), Y1(8), Y2(8)
	REAL ZP, ETA, TETA
	
	DATA X1/-10., 10., 10., -10., -2., 2., 2., -2./
	DATA X2/10., 10., -10., -10., 2., 2., -2., -2./
	DATA Y1/10., 10., -10., -10., 2., 2., -2., -2./
	DATA Y2/10., -10., -10., 10., 2., -2., -2., 2./
c
c	Central region : 10 planes, centered adound 0. Spacing is 5 cm.
c
	OPEN (UNIT=9, FILE = 'sketch1.det', form = 'FORMATTED', 
     &        status = 'NEW')
        write (9,*) ' Sketch detector 1 '
        WRITE (9,*) 'Central region '
	DO IZ  = -5, 5
	 ZP = IZ * 5.0
	 DO  k = 1,8
	  write (9,901) X1(K), Y1(K), ZP, X2(K), Y2(K), ZP
	 END DO
	END DO
	
        WRITE (9,*) 'Fordward region '
	DO IS = -1, 1, 2
	  DO J = 0, 5
	  ETA = 1.5 + J * 0.5
	  TETA = 2.0 * ATAN(EXP(-ETA))
	  ZP = IS * ( 25.0 + 10.0 / TETA)
	  DO  k = 1,8
	    write (9,901) X1(K), Y1(K), ZP, X2(K), Y2(K), ZP
	   END DO
	 END DO
	END DO
	WRITE (9,*) 'End of detector '  
	close(9)
	 
 901     format(6(1x, f10.5))
	stop
	end	
	 
