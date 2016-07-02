C ************************************************************
C Source for the library implementing a bias function that 
C populates the large pt tale of the leading jet. 
C
C The two options of this subroutine, that can be set in
C the run card are:
C    > (double precision) ptj_bias_target_ptj : target ptj value
C    > (double precision) ptj_bias_enhancement_power : exponent
C
C Schematically, the functional form of the enhancement is
C    bias_wgt = [ptj(evt)/mean_ptj]^enhancement_power
C ************************************************************
      
      subroutine bias_wgt(p, bias_weight)
          implicit none
C
C Parameters
C
          include '../maxparticles.inc'          
          include '../nexternal.inc'
          include '../run.inc'
C
C Arguments
C
          double precision p(0:3,nexternal)
          double precision bias_weight
C
C local variables
C
          integer i
          double precision ptj(nexternal)
          double precision max_ptj
C
C Global variables
C
          logical is_a_j(nexternal),is_a_l(nexternal),
     &            is_a_b(nexternal),is_a_a(nexternal),
     &            is_a_onium(nexternal),is_a_nu(nexternal),
     &            is_heavy(nexternal),do_cuts(nexternal)
          common/to_specisa/is_a_j,is_a_a,is_a_l,is_a_b,is_a_nu,
     &                      is_heavy,is_a_onium,do_cuts
C --------------------
C BEGIN IMPLEMENTATION
C --------------------
          
          do i=1,nexternal
            ptj(i)=-1.0d0
            if (is_a_j(i)) then
              ptj(i)=sqrt(p(1,i)**2+p(2,i)**2)
            endif
          enddo

          max_ptj=-1.0d0
          do i=1,nexternal
            max_ptj = max(max_ptj,ptj(i))
          enddo
          if (max_ptj.lt.0.0d0) then
            bias_weight = 1.0d0
            return
          endif

          bias_weight = (max_ptj/ptj_bias_target_ptj)
     &                                      **ptj_bias_enhancement_power
          return

      end subroutine bias_wgt
