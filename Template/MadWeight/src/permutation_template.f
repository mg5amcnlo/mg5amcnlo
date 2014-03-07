C**************************************************************************************
      subroutine assign_perm(perm_id)
C
C
      include 'nexternal.inc'
      include "phasespace.inc"
      integer perm_id(nexternal-2)     !permutation of 1,2,...,nexternal-2      
C
      integer i,j

      double precision pexp_init(0:3,nexternal)  !impulsion in original configuration
      common/to_pexp_init/pexp_init
      double precision pexp(0:3,nexternal)      
      common/to_pexp/pexp

      integer tag_lhco(3:nexternal)
      common/lhco_order/tag_lhco
      integer tag_init(3:max_particles),type(max_particles),run_number,trigger
      double precision eta_init(max_particles),phi_init(max_particles),
     &pt_init(max_particles),j_mass(max_particles),ntrk(max_particles),
     &btag(max_particles),had_em(max_particles),dummy1(max_particles),
     &dummy2(max_particles)
      common/LHCO_input/eta_init,phi_init,pt_init,
     &j_mass,ntrk,btag,had_em,dummy1,dummy2,tag_init,type,run_number,
     &trigger

      integer matching_type_part(3:max_particles) !modif/link between our order by type for permutation
      integer inv_matching_type_part(3:max_particles)
      common/madgraph_order_type/matching_type_part,
     & inv_matching_type_part

      do j=3,nexternal
         do i=0,3
            pexp(i,j)=pexp_init(i, 2+perm_id(j-2))
         enddo
         tag_lhco(j)=tag_init(2+perm_id(j-2))
      enddo

      end 
