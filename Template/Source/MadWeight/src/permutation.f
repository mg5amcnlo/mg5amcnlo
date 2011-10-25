ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine give_permut(perm)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC
CC     INPUT/OUTPUT
CC     cccccccc
CC     from one permut of 1,2,....., Ntot give the next one
CC     this permut, is in fact 3 permut on jet, muon and electron  
CC
CC     global needed(not modified)
CC     ccccccccccccccccccccccccccc
CC     
CC      /to_identical/ (init by identical_perm)
Cc      /num_part/     (init by info_final_part)
CC
CC	global needed  (modified!!!)
CC      cccccccccccccccccccccccccccc
CC
CC	/to_control_perm/ (init all to false to start, control permutation)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      implicit none
      include 'nexternal.inc'
     
      integer perm(nexternal-2)
      
      integer iden_jet(nexternal-1)
      integer iden_bjet(nexternal-1)
      integer iden_mu(nexternal-1)
      integer iden_amu(nexternal-1)
      integer iden_e(nexternal-1)
      integer iden_ae(nexternal-1)
      integer iden_ta(nexternal-1)
      integer iden_ata(nexternal-1)
      common/to_identical/iden_jet,iden_bjet,iden_e,iden_ae,iden_mu,iden_amu,iden_ta,iden_ata

C     global variable for permutations

      integer num_inv,num_jet,num_bjet,num_e,num_ae,num_mu,num_amu,
     +   num_ta,num_ata   !number of jet,elec,muon, undetectable
      COMMON/num_part/num_inv,num_jet,num_bjet,num_e,num_ae,num_mu,
     +  num_amu,num_ta,num_ata !particle in the final state      

C     global variable for control in nexper
      
      logical mtc_j,mtc_b,mtc_m,mtc_e,mtc_am,mtc_ae,mtc_t,mtc_at
      logical even_j,even_b,even_m,even_e,even_am,even_ae,even_t,even_at
      common/to_control_perm/mtc_j,mtc_b,mtc_e,mtc_ae,mtc_m,mtc_am,mtc_t,mtc_at,
     & even_j,even_b,even_e,even_ae,even_m,even_am,even_t,even_at

C     local variable
      integer perm_prov(nexternal-2)
      integer i,j
CC     
CC    next permutation for anti-tau
CC   

 10   if(num_ata.ne.0)then
         do i=1,num_ata
            perm_prov(i)=perm(num_bjet+num_jet+num_e+num_ae+num_mu+num_amu+num_ta+i)
     &     -num_bjet-num_jet-num_e-num_ae-num_mu-num_amu-num_ata
         enddo
         
         call nexper(num_ata,perm_prov,mtc_at,even_at)
c     assignation fo the permutation
         do j=1,num_ata
            perm(num_bjet+num_jet+num_e+num_ae+num_mu+num_amu+num_ta+j)=
     &     num_bjet+num_jet+num_e+num_ae+num_mu+num_amu+num_ta+perm_prov(j)
         enddo
c     control on invalid permutation
         do i=1,num_ata/2
            if(iden_ata(2*i).eq.0) goto 24 !break
            if(perm_prov(iden_ata(2*i)).lt.perm_prov(iden_ata(2*i-1))) 
     &           goto 10        !repeat step: invalid permutation
         enddo
 24      continue
      else
         mtc_at=.false.          !for init next perm
      endif

CC
CC    next permutation for tau
CC

 11   if(num_ta.ne.0)then
       if(.not.mtc_at)then
         do i=1,num_ta
            perm_prov(i)=perm(num_bjet+num_jet+num_e+num_ae+num_mu+num_amu+i)
     &     -num_bjet-num_jet-num_e-num_ae-num_mu-num_amu
         enddo

         call nexper(num_ta,perm_prov,mtc_t,even_t)
c     assignation fo the permutation
         do j=1,num_ta
            perm(num_bjet+num_jet+num_e+num_ae+num_mu+num_amu+j)=
     &     num_bjet+num_jet+num_e+num_ae+num_mu+num_amu+perm_prov(j)
         enddo
c     control on invalid permutation
         do i=1,num_ta/2
            if(iden_ta(2*i).eq.0) goto 22 !break
            if(perm_prov(iden_ta(2*i)).lt.perm_prov(iden_ta(2*i-1)))
     &           goto 11        !repeat step: invalid permutation
         enddo
 22      continue
       endif
      else
         mtc_t=.false.          !for init next perm
      endif

CC     
CC    next permutation for anti-muon
CC   

 12   if(num_amu.ne.0)then
       if(.not.mtc_at.and..not.mtc_t)then
         do i=1,num_amu
            perm_prov(i)=perm(num_bjet+num_jet+num_e+num_ae+num_mu+i)
     &     -num_bjet-num_jet-num_e-num_ae-num_mu
         enddo
         
         call nexper(num_amu,perm_prov,mtc_am,even_am)
c     assignation fo the permutation
         do j=1,num_amu
            perm(num_bjet+num_jet+num_e+num_ae+num_mu+j)=
     &     num_bjet+num_jet+num_e+num_ae+num_mu+perm_prov(j)
         enddo
c     control on invalid permutation
         do i=1,num_amu/2
            if(iden_amu(2*i).eq.0) goto 21 !break
            if(perm_prov(iden_amu(2*i)).lt.perm_prov(iden_amu(2*i-1))) 
     &           goto 12        !repeat step: invalid permutation
         enddo
 21      continue
       endif
      else
         mtc_am=.false.          !for init next perm
      endif

CC
CC    next permutation for muon
CC

 13   if(num_mu.ne.0)then
       if(.not.mtc_at.and..not.mtc_t.and..not.mtc_am)then
         do i=1,num_mu
            perm_prov(i)=perm(num_bjet+num_jet+num_e+num_ae+i)
     &     -num_bjet-num_jet-num_e-num_ae
         enddo

         call nexper(num_mu,perm_prov,mtc_m,even_m)
c     assignation fo the permutation
         do j=1,num_mu
            perm(num_bjet+num_jet+num_e+num_ae+j)=
     &     num_bjet+num_jet+num_e+num_ae+perm_prov(j)
         enddo
c     control on invalid permutation
         do i=1,num_mu/2
            if(iden_mu(2*i).eq.0) goto 23 !break
            if(perm_prov(iden_mu(2*i)).lt.perm_prov(iden_mu(2*i-1)))
     &           goto 13        !repeat step: invalid permutation
         enddo
 23      continue
         endif
      else
         mtc_m=.false.          !for init next perm
      endif

C     C     
C     C    next permutation for positron
C     C     
 14   if(num_ae.ne.0)then
       if(.not.mtc_at.and..not.mtc_t.and..not.mtc_am.and..not.mtc_m)then
            do i=1,num_ae
               perm_prov(i)=perm(num_bjet+num_jet+num_e+i)-num_jet-num_bjet-num_e
            enddo

            call nexper(num_ae,perm_prov,mtc_ae,even_ae)
           
c     assignation fo the permutation
            do j=1,num_ae
               perm(num_bjet+num_jet+num_e+j)=num_bjet+num_jet+num_e+perm_prov(j)
            enddo
           
c     control on invalid permutation
            do i=1,num_ae/2
               if(iden_ae(2*i).eq.0)goto 25 !break
               if(perm_prov(iden_ae(2*i)).lt.perm_prov(iden_ae(2*i-1))) 
     &              goto 14     !repeat step: invalid permutation
            enddo
 25         continue   
         endif
      else 
         mtc_ae=.false.          !for init next perm       
      endif
      
C     C
C     C    next permutation for electron
C     C
 15   if(num_e.ne.0)then
       if(.not.mtc_at.and..not.mtc_t.and..not.mtc_am.and..not.mtc_m.and..not.mtc_ae)then
            do i=1,num_e
               perm_prov(i)=perm(num_bjet+num_jet+i)-num_jet-num_bjet
            enddo

            call nexper(num_e,perm_prov,mtc_e,even_e)

c     assignation fo the permutation
            do j=1,num_e
               perm(num_bjet+num_jet+j)=num_bjet+num_jet+perm_prov(j)
            enddo

c     control on invalid permutation
            do i=1,num_e/2
               if(iden_e(2*i).eq.0)goto 27 !break
               if(perm_prov(iden_e(2*i)).lt.perm_prov(iden_e(2*i-1)))
     &              goto 15     !repeat step: invalid permutation
            enddo
 27         continue
         endif
      else
         mtc_e=.false.          !for init next perm
      endif
CC     
CC    next permutation for bjet
CC 
 17   if(num_bjet.ne.0)then
       if(.not.mtc_at.and..not.mtc_t.and..not.mtc_am.and..not.mtc_m.and.
     +.not.mtc_ae.and..not.mtc_e)then
            do i=1,num_bjet
               perm_prov(i)=perm(num_jet+i)
            enddo
            call nexper(num_bjet,perm_prov,mtc_b,even_b)
c     assignation fo the permutation         
            do j=1,num_bjet
               perm(num_jet+j)=perm_prov(j)+num_jet
            enddo           
c     control on invalid permutation
            do i=1,num_bjet/2
               if(iden_bjet(2*i).eq.0)goto 31 !break
               if(perm_prov(iden_bjet(2*i)).lt.perm_prov(iden_bjet(2*i-1))) 
     &              goto 17     !repeat step: invalid permutation
            enddo
 31         continue
         endif
      else
         mtc_b=.false.          !for init next perm  
      endif
CC     
CC    next permutation for jet
CC 
 16   if(num_jet.ne.0)then
       if(.not.mtc_at.and..not.mtc_t.and..not.mtc_am.and..not.mtc_m.and.
     +.not.mtc_ae.and..not.mtc_e.and..not.mtc_b)then
            do i=1,num_jet
               perm_prov(i)=perm(i)
            enddo
            call nexper(num_jet,perm_prov,mtc_j,even_j)
c     assignation fo the permutation         
            do j=1,num_jet
               perm(j)=perm_prov(j)
            enddo           
c     control on invalid permutation
            do i=1,num_jet/2
               if(iden_jet(2*i).eq.0)goto 29 !break
               if(perm_prov(iden_jet(2*i)).lt.perm_prov(iden_jet(2*i-1))) 
     &              goto 16     !repeat step: invalid permutation
            enddo
 29         continue
         endif
      else
         mtc_j=.false.          !for init next perm  
      endif
CC
CC     Complete permutation with invisible particle 
CC
      do j=1,num_inv
         perm(num_jet+num_bjet+num_mu+num_amu+num_e+num_ae+num_ta+num_ata+j)
     &=num_jet+num_bjet+num_mu+num_amu+num_e+num_ae+num_ta+num_ata+j
      enddo

 99   end
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine get_num_per(num_per)
c
c	compute the total number of permutations	
c      need COMMON/to_identical (init by identical_perm())
c      need COMMON/num_part/ (init by info_final_part())
c
c    output: num_per= number of permutation
c
      implicit none
      include 'nexternal.inc'

      integer num_per

      integer iden_jet(nexternal-1) !we need one zero in last position to
      integer iden_bjet(nexternal-1)  !treat not even case
      integer iden_mu(nexternal-1)
      integer iden_amu(nexternal-1)
      integer iden_e(nexternal-1)
      integer iden_ae(nexternal-1)
      integer iden_ta(nexternal-1)
      integer iden_ata(nexternal-1)
      common/to_identical/iden_jet,iden_bjet,iden_e,iden_ae,iden_mu,iden_amu,iden_ta,iden_ata

      integer num_inv,num_jet,num_bjet,num_e,num_ae,num_mu,num_amu,num_ta,num_ata   !number of jet,elec,muon, undetectable
      COMMON/num_part/num_inv,num_jet,num_bjet,num_e,num_ae,num_mu,num_amu,num_ta,num_ata !particle in the final state
 
      integer i
c
c     FONCTION
c
      integer factoriel
      external factoriel   

      num_per=factoriel(num_bjet)*factoriel(num_jet)
     & *factoriel(num_mu)*factoriel(num_e)*factoriel(num_amu)*factoriel(num_ae)
     & *factoriel(num_ata)*factoriel(num_ta)
c---
c TEMPORY: initialization of iden_?
c---

      do i=1,nexternal-1
       iden_bjet(i)=0
       iden_jet(i)=0
       iden_mu(i)=0
       iden_amu(i)=0
       iden_e(i)=0
       iden_ae(i)=0
       iden_ta(i)=0
       iden_ata(i)=0
      enddo

      do i=1,(nexternal-2)/2
         if(iden_bjet(2*i).ne.0) num_per=num_per/2
         if(iden_jet(2*i).ne.0)  num_per=num_per/2
         if(iden_mu(2*i).ne.0) num_per=num_per/2
         if(iden_amu(2*i).ne.0) num_per=num_per/2
         if(iden_ae(2*i).ne.0) num_per=num_per/2        
         if(iden_e(2*i).ne.0) num_per=num_per/2        
         if(iden_ta(2*i).ne.0) num_per=num_per/2
         if(iden_ata(2*i).ne.0) num_per=num_per/2
      enddo

      write(*,*) "The number of parton-jet assignements is", num_per

      end


C**************************************************************************************
      subroutine init_perm()
                                                                                                                                                     
c control permutation
      logical mtc_j,mtc_b,mtc_m,mtc_e,mtc_am,mtc_ae,mtc_t,mtc_at  ! logicals to control permutations
      logical even_j,even_b,even_m,even_e,even_am,even_ae,even_t,even_at ! false=reinit permutations
      common/to_control_perm/mtc_j,mtc_b,mtc_e,mtc_ae,mtc_m,mtc_am,mtc_t,mtc_at,
     & even_j,even_b,even_m,even_e,even_am,even_ae,even_t,even_at
                                                                                                                                                     
      mtc_j=.false.
      mtc_b=.false.
      mtc_am=.false.
      mtc_ae=.false.
      mtc_m=.false.
      mtc_e=.false.
      mtc_t=.false.
      mtc_at=.false.
      return
      end
C**************************************************************************************
      subroutine assign_perm(perm_id)
C
C
      include 'nexternal.inc'
      integer perm_id(nexternal-2)     !permutation of 1,2,...,nexternal-2      
C
      integer i,j

      double precision pexp_init(0:3,nexternal)  !impulsion in original configuration
      common/to_pexp_init/pexp_init
      double precision pexp(0:3,nexternal)      
      common/to_pexp/pexp

      integer tag_lhco(3:nexternal)
      common/lhco_order/tag_lhco
      integer tag_init(3:nexternal),type(nexternal),run_number,trigger
      double precision eta_init(nexternal),phi_init(nexternal),
     &pt_init(nexternal),j_mass(nexternal),ntrk(nexternal),
     &btag(nexternal),had_em(nexternal),dummy1(nexternal),
     &dummy2(nexternal)
      common/LHCO_input/eta_init,phi_init,pt_init,
     &j_mass,ntrk,btag,had_em,dummy1,dummy2,tag_init,type,run_number,
     &trigger

      integer matching_type_part(3:nexternal) !modif/link between our order by type for permutation
      integer inv_matching_type_part(3:nexternal)
      common/madgraph_order_type/matching_type_part,
     & inv_matching_type_part

      do j=3,nexternal
         do i=0,3
            pexp(i,j)=pexp_init(i,inv_matching_type_part(
     & 2+perm_id(matching_type_part(j)-2)))
         enddo
         tag_lhco(j)=tag_init(inv_matching_type_part(
     & 2+perm_id(matching_type_part(j)-2)))
      enddo

      end 
