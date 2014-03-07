      subroutine setTF
      implicit none
c     
      include 'nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'

      call Init_MET_LHCO

      include 'transfer_card.inc'

      return
      end
