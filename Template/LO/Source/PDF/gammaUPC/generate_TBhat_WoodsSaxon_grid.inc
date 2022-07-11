
       WRITE(*,*)"INFO: generate B grid for TAhat in Woods-Saxon (may take a few seconds)"
       IF(store_grid)OPEN(UNIT=30307,FILE=TRIM(nuclear_dir)//'grid/TBhat_WoodsSaxon.grid')
       R_common=RR
       w_common=w
       aa_common=aa
       A_common=A
       ds=10d0*RR/NSTEPS
       IF(store_grid)THEN
          WRITE(30307,*)NSTEPS+1,R_common,w_common,aa_common
       ELSE
          NB=NSTEPS+1
	  ALLOCATE(XB(NB))
	  ALLOCATE(YB(NB))
       ENDIF
       DO i=0,NSTEPS
          ss=i*ds
          ss_common=ss
	  CALL trapezoid_integration(10000,TAhat_fxn_WoodsSaxon,50d0*RR,TAhat_WoodsSaxon)
	  TAhat_WoodsSaxon=TAhat_WoodsSaxon/A
          IF(store_grid)THEN
             WRITE(30307,*)ss,TAhat_WoodsSaxon
          ELSE
	     XB(i+1)=ss
	     YB(i+1)=TAhat_WoodsSaxon
	  ENDIF
       ENDDO
       IF(store_grid)CLOSE(UNIT=30307)