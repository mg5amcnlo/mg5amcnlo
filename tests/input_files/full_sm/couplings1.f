      subroutine coup1(readlha)

      implicit none
      logical readlha

      include 'input.inc'
      include 'coupl.inc'
      include 'intparam_definition.inc'


c Interactions associated with 1
      MGVX1 = G
      MGVX2 = GGT1


      GG(1) = -G
      GG(2) = -G

      return
      end
