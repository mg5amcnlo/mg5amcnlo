        ! The orders to be integrated for the Born and at NLO
      INTEGER NSPLITORDERS
      PARAMETER (NSPLITORDERS=2)
      CHARACTER*3 ORDERNAMES(NSPLITORDERS)
      DATA ORDERNAMES / 'QCD', 'QED' /
      INTEGER BORN_ORDERS(NSPLITORDERS), NLO_ORDERS(NSPLITORDERS)
        ! the order of the coupling orders is QCD, QED
      DATA BORN_ORDERS / 4, 0 /
      DATA NLO_ORDERS / 4, 2 /
        ! The position of the QCD /QED orders in the array
      INTEGER QCD_POS, QED_POS
        ! if = -1, then it is not in the split_orders
      PARAMETER (QCD_POS = 1)
      PARAMETER (QED_POS = 2)
        ! this is to keep track of the various 
        ! coupling combinations entering each ME
      INTEGER AMP_SPLIT_SIZE, AMP_SPLIT_SIZE_BORN
      PARAMETER (AMP_SPLIT_SIZE = 4)
        ! the first entries in the next line in amp_split are for the born 
      PARAMETER (AMP_SPLIT_SIZE_BORN = 1)
      DOUBLE PRECISION AMP_SPLIT(AMP_SPLIT_SIZE)
      DOUBLE COMPLEX AMP_SPLIT_CNT(AMP_SPLIT_SIZE,2,NSPLITORDERS)
      COMMON /TO_AMP_SPLIT/AMP_SPLIT, AMP_SPLIT_CNT

