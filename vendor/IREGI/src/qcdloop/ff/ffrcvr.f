*###[ ffrcvr:
	subroutine ffrcvr(isig)
    	integer isig,ier,nold,ncall
	save nold
	include 'ff.h'
	data nold /0/
	data ncall /0/
	if ( isig .ne. 8 ) then
		print *,'ffrcvr: Somebody shot a signal ',isig,' at me'
		stop
	endif
*	Only give the message once per event
	if ( nevent .eq. nold ) then
		ncall = ncall + 1
		if ( ncall .lt. 100 ) then
*			return
		else
			print *,'ffrcvr: error:  more than 100 calls'
			stop
		endif
	else
		nold = nevent
		ncall = 0
	endif
	ner = ner + 100
	ier = 0
	call fferr(100,ier)
*###] ffrcvr: 
	end
