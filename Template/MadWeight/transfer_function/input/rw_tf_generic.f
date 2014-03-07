      subroutine setTF
      implicit none
c     
      include 'nexternal.inc'
      include 'nb_tf.inc'
      include 'TF_param.inc'


      integer maxpara
      parameter (maxpara=1000)
c
c     local
c
      integer npara
      character*20 param(maxpara),value(maxpara)

      call Init_MET_LHCO
      call load_tf_para(npara,param,value,"transfer_card.dat")

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
c
C     user parameter for transfer functions
c
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


$$ADD_HERE$$
      


      return
      end



