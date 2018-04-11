      subroutine read_event(lun,P,wgt,nexternal,ic,ievent,sscale,
     $                      aqcd,aqed,buff,u_syst,s_buff,nclus,buffclus,
     $                      done)
c********************************************************************
c     Reads one event from data file #lun
c     ic(*,1) = Particle ID
c     ic(*,2) = Mothup(1)
c     ic(*,3) = Mothup(2)
c     ic(*,4) = ICOLUP(1)
c     ic(*,5) = ICOLUP(2)
c     ic(*,6) = ISTUP   -1=initial state +1=final  +2=decayed
c     ic(*,7) = Helicity
c********************************************************************
      implicit none
      include 'maxparticles.inc'
      include 'run_config.inc'
      include 'run.inc'
      double precision pi
      parameter (pi = 3.1415926d0)
c
c     Arguments
c      
      integer lun
      integer nexternal, ic(7,*)
      logical done
      double precision P(0:4,*),wgt,aqcd,aqed,sscale
      integer ievent
      character*(*) buff
      logical u_syst
      character*(s_bufflen) s_buff(*)
      integer nclus
      character*(clus_bufflen) buffclus(*)
c
c     Local
c
      integer i,j,k
      character*(s_bufflen) buftmp
      double precision xdum1,xdum2
c
c     Global
c
      logical banner_open
      integer lun_ban
      common/to_banner/banner_open, lun_ban

      data lun_ban/37/
      data banner_open/.false./

      double precision bias_weight
      logical impact_xsec
      common/bias/bias_weight,impact_xsec
c-----
c  Begin Code
c-----     
      buff=' '
      done=.false.
      if (.not. banner_open) then
         open (unit=lun_ban, status='scratch')
         banner_open=.true.
      endif
 11   read(lun,'(a300)',end=99,err=99) buftmp
      do while(index(buftmp,"<event") .eq. 0)
         write(lun_ban,'(a)') buftmp
         read(lun,'(a300)',end=99,err=99) buftmp
      enddo
      read(lun,*,err=11, end=11) nexternal,ievent,wgt,sscale,aqed,aqcd
      do i=1,nexternal
         read(lun,*,err=99,end=99) ic(1,i),ic(6,i),(ic(j,i),j=2,5),
     $     (p(j,i),j=1,3),p(0,i),p(4,i),xdum1,xdum2
         ic(7,i)=xdum2
      enddo
      
c     Clustering scales
      read(lun,'(a)',end=99,err=99) buff
      if (lhe_version.lt.3d0)then
        if(buff(1:1).ne.'#') then
         backspace(lun)
         buff=''
        endif
      else
        if(buff(1:7).ne.'<scales') then
         backspace(lun)
         buff=''
        endif
      endif

c     Reading the bias weight (if present)
      read(lun,'(a300)',end=99,err=99) buftmp
      if(buftmp(1:6).ne.'<rwgt>') then
         backspace(lun)
         bias_weight = 1.0d0
      else
         do while(buftmp(1:7).ne.'</rwgt>')
           read(lun,'(a300)',end=99,err=99) buftmp
           if (buftmp(1:16).eq."<wgt id='bias'> ") then
              read(buftmp(17:31),'(1e15.7)') bias_weight
           endif
         enddo
      endif

c     Systematics info
      read(lun,'(a)',end=99,err=99) s_buff(1)
      if(s_buff(1).ne.'<mgrwt>') then
         s_buff(1)=' '
         backspace(lun)
         u_syst=.false.
      else
         i=1
         do while(s_buff(i).ne.'</mgrwt>')
            i=i+1
            read(lun,'(a)',end=99,err=99) s_buff(i)
         enddo
         u_syst=.true.
      endif
c     Clustering info
      read(lun,'(a)',end=99,err=99) buffclus(1)
      if(buffclus(1).ne.'<clustering>') then
         buffclus(1)=' '
         backspace(lun)
         nclus=0
      else
         i=1
         do while(buffclus(i).ne.'</clustering>')
            i=i+1
            read(lun,'(a)',end=99,err=99) buffclus(i)
         enddo
         nclus=i
      endif
      return
 99   done=.true.
      return
 55   format(i3,5e19.11)         
      end

      subroutine write_event_to_stream(evt_record,P,wgt,nexternal,ic,
     &     ievent,scale,aqcd, aqed,buff,u_syst,s_buff,nclus,buffclus)
c********************************************************************
C     This an *exact* copy of write_event, except that it writes it 
C     to a character array argument as opposed to an I/O stream.
c********************************************************************
      implicit none

      include 'maxparticles.inc'
      include 'run_config.inc'
c
c     parameters
c
      double precision pi
      parameter (pi = 3.1415926d0)
c
c     Arguments
c      
      character*(maxEventLength) evt_record
      integer ievent
      integer nexternal, ic(7,*)
      double precision P(0:4,*),wgt
      double precision aqcd, aqed, scale
      character*1000 buff
      logical u_syst
      character*(s_bufflen) s_buff(*)
      integer nclus
      character*(clus_bufflen) buffclus(*)
c
c     Local
c
      integer i,j,k
      character*(maxEventLength) largeBuff
c
c     Global
c
      double precision bias_weight
      logical impact_xsec
      common/bias/bias_weight,impact_xsec

c-----
c  Begin Code
c-----     
c      aqed= gal(1)*gal(1)/4d0/pi
c      aqcd = g*g/4d0/pi
      write(largeBuff,'(a)') '<event>'
      evt_record=trim(evt_record)//trim(largeBuff)
      write(largeBuff,'(i2,i5,e16.7e3,3e15.7)') nexternal,ievent,wgt,scale,
     $                                   aqed,aqcd
      evt_record=trim(evt_record)//CHAR(13)//CHAR(10)//trim(largeBuff)
      do i=1,nexternal
         write(largeBuff,51) ic(1,i),ic(6,i),(ic(j,i),j=2,5),
     $     (p(j,i),j=1,3),p(0,i),p(4,i),0.,real(ic(7,i))
         evt_record=trim(evt_record)//CHAR(13)//CHAR(10)//trim(largeBuff)
      enddo
      if(buff(1:7).eq.'<scales') then
        write(largeBuff,'(a)') buff(1:len_trim(buff))
        evt_record=trim(evt_record)//CHAR(13)//CHAR(10)//trim(largeBuff)
      endif
      if(buff(1:1).eq.'#') then
        write(largeBuff,'(a)') buff(1:len_trim(buff))
        evt_record=trim(evt_record)//CHAR(13)//CHAR(10)//trim(largeBuff)
      endif
      if(.not.impact_xsec) then
          write(largeBuff,'(a)') '<rwgt>'
          evt_record=trim(evt_record)//CHAR(13)//CHAR(10)//trim(largeBuff)
          write(largeBuff,'(a16,1e15.7,a6)') "<wgt id='bias'> ",
     $                                              bias_weight,"</wgt>"
          evt_record=trim(evt_record)//CHAR(13)//CHAR(10)//trim(largeBuff)
          write(largeBuff,'(a)') '</rwgt>'
          evt_record=trim(evt_record)//CHAR(13)//CHAR(10)//trim(largeBuff)
      endif 
      if(u_syst)then
         do i=1,7
            write(largeBuff,'(a)') s_buff(i)(1:len_trim(s_buff(i)))
            evt_record=trim(evt_record)//CHAR(13)//CHAR(10)//trim(largeBuff)
         enddo
      endif
      do i=1,nclus
         write(largeBuff,'(a)') buffclus(i)(1:len_trim(buffclus(i)))
         evt_record=trim(evt_record)//CHAR(13)//CHAR(10)//trim(largeBuff)
      enddo
      write(largeBuff,'(a)') '</event>'
      evt_record=trim(evt_record)//CHAR(13)//CHAR(10)//trim(largeBuff)
      return
 51   format(i11,5i5,5e19.11,f3.0,f4.0)
      end


      subroutine write_event(lun,P,wgt,nexternal,ic,ievent,scale,aqcd,
     $     aqed,buff,u_syst,s_buff,nclus,buffclus)
c********************************************************************
c
c /!\ When making changes to this subroutine, make sure to accordingly
c     update write_event_to_stream
c
c********************************************************************
c     Writes one event from data file #lun according to LesHouches
c     ic(1,*) = Particle ID
c     ic(2.*) = Mothup(1)
c     ic(3,*) = Mothup(2)
c     ic(4,*) = ICOLUP(1)
c     ic(5,*) = ICOLUP(2)
c     ic(6,*) = ISTUP   -1=initial state +1=final  +2=decayed
c     ic(7,*) = Helicity
c********************************************************************
      implicit none

      include 'maxparticles.inc'
      include 'run_config.inc'
c
c     parameters
c
      double precision pi
      parameter (pi = 3.1415926d0)
c
c     Arguments
c      
      integer lun, ievent
      integer nexternal, ic(7,*)
      double precision P(0:4,*),wgt
      double precision aqcd, aqed, scale
      character*1000 buff
      logical u_syst
      character*(s_bufflen) s_buff(*)
      integer nclus
      character*(clus_bufflen) buffclus(*)
c
c     Local
c
      integer i,j,k
c
c     Global
c
      double precision bias_weight
      logical impact_xsec
      common/bias/bias_weight,impact_xsec

c-----
c  Begin Code
c-----     
c      aqed= gal(1)*gal(1)/4d0/pi
c      aqcd = g*g/4d0/pi

      write(lun,'(a)') '<event>'
      write(lun,'(i2,i5,e16.7e3,3e15.7)') nexternal,ievent,wgt,scale,aqed,aqcd
      do i=1,nexternal
         write(lun,51) ic(1,i),ic(6,i),(ic(j,i),j=2,5),
     $     (p(j,i),j=1,3),p(0,i),p(4,i),0.,real(ic(7,i))
      enddo
      if(buff(1:7).eq.'<scales') write(lun,'(a)') buff(1:len_trim(buff))
      if(buff(1:1).eq.'#') write(lun,'(a)') buff(1:len_trim(buff))      
      if(.not.impact_xsec) then
          write(lun,'(a)') '<rwgt>'
          write(lun,'(a16,1e15.7,a6)') "<wgt id='bias'> ",bias_weight,
     $                                                          "</wgt>"
          write(lun,'(a)') '</rwgt>'
      endif 
      if(u_syst)then
         do i=1,7
            write(lun,'(a)') s_buff(i)(1:len_trim(s_buff(i)))
         enddo
      endif
      do i=1,nclus
         write(lun,'(a)') buffclus(i)(1:len_trim(buffclus(i)))
      enddo
      write(lun,'(a)') '</event>'
      return
 51   format(i11,5i5,5e19.11,f3.0,f4.0)
      end

      subroutine write_comments(lun)
c********************************************************************
c     Outputs all of the banner comment lines back at the top of
c     the file lun.
c********************************************************************
      implicit none
c
c     Arguments
c
      integer lun
c
c     Local
c
      character*(200) buff
c
c     Global
c
      logical banner_open
      integer lun_ban
      common/to_banner/banner_open, lun_ban

c-----
c  Begin Code
c-----     
c      write(*,*) 'Writing comments'
      if (banner_open) then
         rewind(lun_ban)
         do while (.true.) 
            read(lun_ban,'(a)',end=99,err=99) buff
            write(lun,'(a)') buff
c            write(*,*) buff
         enddo
 99      close(lun_ban)
         banner_open = .false.
      endif
      end

