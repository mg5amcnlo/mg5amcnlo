      integer jpart(7,-nexternal+3:2*nexternal-3)
      double precision pb(0:4,-nexternal+3:2*nexternal-3)
      integer isym(nexternal,99),jsym, npart
      double precision sscale,aaqcd,aaqed
      character*1000 buff
      character*(s_bufflen) s_buff(7)
      integer nclus
      character*(clus_bufflen) buffclus(nexternal)
	  character*(maxEventLength) event_record
      logical AlreadySetInBiasModule

	  common/to_lhe_event_info/jpart,pb,s_buff,buff,nclus,buffclus,event_record,
     &  sscale,aaqcd,aaqed,isym,jsym,npart,AlreadySetInBiasModule

      integer ngroup
      common/to_group/ngroup
