      subroutine transform(p,q,ip,jp,kp,ijp,kpp,mass_i,mass_j,mass_k)
************************************************************************
*     Author: R.K. Ellis                                               *
*     September, 1999.                                                 *
*     Given p (p1 + p2 --> p3 ... px .. p_(nexternal))                 *
*     produce q (q1 + q2 --> q3 ... qx .. q_(nexternal-1))             *
*                                                                      *
*     ip is the emitter, jp the unresolved and kp is the spectator     *
*     after branching.                                                 *
*     ijp and kpp are the emitter and spectator before branching       *
*                                                                      *
*     Correct branch chosen automatically                              *
*                                                                      *
*     Adapted for MadGraph/MadEvent by RF and NG, June 2008            *
*                                                                      *
************************************************************************
      implicit none 
      include 'nexternal.inc'
      double precision p(0:3,nexternal),q(0:3,nexternal),x,omx,y,omy,
     . k(0:3),kt(0:3),ks(0:3),kDk,ksDks,
     . kDp(nincoming+1:nexternal),ksDp(nincoming+1:nexternal)
      integer ip,kp,j,nu,jp,ipart,inp,fip,ijp,kpp

      double precision dot,dotij,dotik,dotjk,mass_i,mass_j,mass_k,mu_i,mu_j
      double precision mu_k,mu_ij,qtot(0:3),qsq,pij(0:3),lambda_tr,msq_ij
      double precision msq_k,msq_i,msq_j,porig(0:3,nexternal),beta,gamma
      external dot,lambda_tr
      double precision one, two
      parameter (one=1d0,two=2d0)
      logical firsttime
      data firsttime /.false./

      if (firsttime) then
         write (*,*)'Using the "transform(..)" subroutine, '//
     &        'originally written by R.K. Ellis, Sep 1999.'
         write (*,*)'Adapted for MadDipole by '//
     &        'N. Greiner & R. Frederix Jun. 2008.'
         firsttime=.false.
      endif
      
      do j=1,nexternal
         do nu=0,3
            q(nu,j)=0d0
c            porig(nu,j)=p(nu,j)
         enddo
      enddo
      
      dotij=dot(p(0,ip),p(0,jp))
      dotik=dot(p(0,ip),p(0,kp))
      dotjk=dot(p(0,jp),p(0,kp))

      if((ijp.ge.nexternal).or.(ijp.le.0).or.
     .   (kpp.ge.nexternal).or.(kpp.le.0).or.
     .   (ip .gt.nexternal).or.(ip .le.0).or.
     .   (jp .gt.nexternal).or.(jp .le.0).or.
     .   (kp .gt.nexternal).or.(kp .le.0)) then
         write (*,*) 'ERROR in momentum mapping.'
         write (*,*) 'ip, jp, kp, ijp and/or kpp too large: '
     .                ,ip,jp,kp,ijp,kpp
         return
      endif
      


      if(mass_i.lt.1d-5.and. mass_k .lt.1d-5) then
c  massles particles
      if ((ip .le. nincoming) .and. (kp .le. nincoming)) then
c---  initial-initial

         x=one-(dotij+dotjk)/dotik

         do nu=0,3
            q(nu,ijp)=  x*p(nu,ip)
            q(nu,kpp)=  p(nu,kp)
            k(nu)    =  p(nu,ip)+p(nu,kp)-p(nu,jp)
            kt(nu)   =  q(nu,ijp)+p(nu,kp)
            ks(nu)   =  k(nu)+kt(nu)
         enddo
         
         kDk  = dot(k(0),k(0))
         ksDks= dot(ks(0),ks(0))
         
         ipart=nincoming+1
         do j=nincoming+1,nexternal
            if (j .eq. jp) then
               go to 19
            else
               kDp(j) = dot(k(0),p(0,j))
               ksDp(j)= dot(ks(0),p(0,j))
               do nu=0,3
                  q(nu,ipart)=p(nu,j)-two*ksDp(j)*ks(nu)/ksDks
     .                               +two*kDp(j)*kt(nu)/kDk
               enddo
               ipart=ipart+1
               if(ipart.gt.nexternal)then
                  write(*,*) 'ERROR, ipart .ge. nexternal in ii',ipart
               endif
            endif
 19         continue
         enddo

         return
      
      elseif (((ip .le. nincoming) .and. (kp .gt. nincoming)) .or.
     .        ((ip .gt. nincoming) .and. (kp .le. nincoming))) then
c---initial-final or final-initial
         if (ip .le. nincoming) then
            inp=ip
            fip=kp  
            x = one - dotjk/(dotik+dotij)
         else
            inp=kp
            fip=ip
            x = one - dotij/(dotik+dotjk)
         endif

         ipart=1
         omx=one-x
         do j=1,nexternal
            do nu=0,3
               if (j.eq.inp) then
                  q(nu,ipart)=x*p(nu,inp)
                  if ((ipart.ne.ijp).and.(ipart.ne.kpp))then
                     write(*,*)'ERROR, problem with momentum mapping fi'
                  endif
               elseif (j.eq.jp) then
                  goto 20
               elseif (j.eq.fip) then
                  q(nu,ipart)=p(nu,jp)+p(nu,fip)-omx*p(nu,inp)
                  if ((ipart.ne.ijp).and.(ipart.ne.kpp))then
                     write(*,*)'ERROR, problem with momentum mapping fi'
                  endif
               else
                  q(nu,ipart)=p(nu,j)
               endif
            enddo
            ipart=ipart+1
            if(ipart.gt.nexternal)then
               write(*,*) 'ERROR, ipart .ge. nexternal in fi'
            endif
 20         continue
         enddo

         return
         
      elseif ((ip .gt. nincoming) .and. (kp .gt. nincoming)) then
c---  final-final
         ipart=1
         y=dotij/(dotij+dotjk+dotik)
         omy=one-y
         do j=1,nexternal
            do nu=0,3
               if (j.eq.ip) then
                  q(nu,ipart)=p(nu,jp)+p(nu,ip)-y/omy*p(nu,kp)
                  if (ipart.ne.ijp)then
                     write(*,*)'ERROR, problem with momentum mapping ff'
                  endif
               elseif (j.eq.jp) then
                  goto 21
               elseif (j.eq.kp) then
                  q(nu,ipart)=p(nu,kp)/omy
                  if (ipart.ne.kpp)then
                     write(*,*)'ERROR, problem with momentum mapping ff'
                  endif
               else
                  q(nu,ipart)=p(nu,j)
               endif
            enddo
            ipart=ipart+1
            if(ipart.gt.nexternal)then
               write(*,*) 'ERROR, ipart .ge. nexternal in ff'
            endif
 21         continue
         enddo

         return
      endif

      else

c massive case

      msq_i=mass_i**2
      msq_k=mass_k**2
      msq_j=mass_j**2
      if(mass_i .eq.mass_j) then
       msq_ij=0d0
      else
       msq_ij=msq_i
      endif

      if((ip.le.nincoming) .and. (kp.le.nincoming)) then
c --- initial-initial
       write(*,*) 'ERROR, no massive particles in initial state allowed'

      elseif (((ip .le. nincoming) .and. (kp .gt. nincoming)) .or.
     .        ((ip .gt. nincoming) .and. (kp .le. nincoming))) then
c---initial-final or final-initial
         if (ip .le. nincoming) then
            inp=ip
            fip=kp  
            x = one - dotjk/(dotik+dotij)
         else
            inp=kp
            fip=ip
            x = one - (dotij-0.5d0*(msq_ij-msq_i-msq_j))/(dotik+dotjk)
         endif

         ipart=1
         omx=one-x
         do j=1,nexternal
            do nu=0,3
               if (j.eq.inp) then
                  q(nu,ipart)=x*p(nu,inp)
                  if ((ipart.ne.ijp).and.(ipart.ne.kpp))then
                     write(*,*)'ERROR, problem with momentum mapping fi'
                  endif
               elseif (j.eq.jp) then
                  goto 23
               elseif (j.eq.fip) then
                  q(nu,ipart)=p(nu,jp)+p(nu,fip)-omx*p(nu,inp)
                  if ((ipart.ne.ijp).and.(ipart.ne.kpp))then
                     write(*,*)'ERROR, problem with momentum mapping fi'
                  endif
               else
                  q(nu,ipart)=p(nu,j)
               endif
            enddo
            ipart=ipart+1
            if(ipart.gt.nexternal)then
               write(*,*) 'ERROR, ipart .ge. nexternal in fi'
            endif
 23         continue
         enddo

         return

      elseif ((ip .gt. nincoming) .and. (kp .gt. nincoming)) then
c---  final-final
       ipart=1
       do nu=0,3
        qtot(nu)=p(nu,ip)+p(nu,jp)+p(nu,kp)
        pij(nu)=p(nu,ip)+p(nu,jp)
       enddo
       qsq=dot(qtot,qtot)
         do j=1,nexternal
            do nu=0,3
               if (j.eq.ip) then
                 q(nu,ipart)=qtot(nu)-(sqrt(lambda_tr(qsq,msq_ij,msq_k))/
     &                 sqrt(lambda_tr(qsq,dot(pij,pij),msq_k))
     &                        *(p(nu,kp)-dot(qtot,p(0,kp))/qsq*qtot(nu))
     &                        +(qsq+msq_k-msq_ij)/two/qsq*qtot(nu))
                  if (ipart.ne.ijp)then
                     write(*,*)'ERROR, problem with momentum mapping ff'
                  endif
               elseif (j.eq.jp) then
                  goto 24
               elseif (j.eq.kp) then
c                  print*, msq_ij,msq_k,dot(pij,pij),(2d0*dot(p(0,ip),p(0,jp)))
                  q(nu,ipart)=sqrt(lambda_tr(qsq,msq_ij,msq_k))/
     &                 sqrt(lambda_tr(qsq,dot(pij,pij),msq_k))
     &                        *(p(nu,kp)-dot(qtot,p(0,kp))/qsq*qtot(nu))
     &                        +(qsq+msq_k-msq_ij)/two/qsq*qtot(nu)
                  if (ipart.ne.kpp)then
                     write(*,*)'ERROR, problem with momentum mapping ff'
                  endif
               else
                  q(nu,ipart)=p(nu,j)
               endif
            enddo
            ipart=ipart+1
            if(ipart.gt.nexternal)then
               write(*,*) 'ERROR, ipart .ge. nexternal in ff'
            endif
 24         continue
         enddo

         return

      endif
      endif

      end


      REAL*8 function lambda_tr(x,y,z)
c  triangular function
      implicit none

c   Arguments
      real*8 x,y,z

      lambda_tr=x**2+y**2+z**2-2d0*x*y-2d0*x*z-2d0*y*z

      end
