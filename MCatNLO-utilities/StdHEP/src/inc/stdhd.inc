c -------------------------------------------------------------
c This header collects the mcfio initial information
c
c   date      - MCFIO_CREATIONDATE: creation date
c   title     - MCFIO_TITLE: title
c   comment   - MCFIO_COMMENT: comment
c   dlen      - actual lenght of date
c   tlen      - actual lenght of title
c   clen      - actual lenght of comment
c   numblocks - MCFIO_NUMBLOCKS: number of blocks per event
c   blkids    - MCFIO_BLOCKIDS: list of block types
c
c use: write(*,1001) title(1:tlen),date(1:dlen),comment(1:clen)
c      1001  format(10x,'title: ',a60/10x,'date:  ',a60/10x,a70)
c
      character*255 date, title, comment
      integer dlen, tlen, clen, numblocks, blkids
      common /stdhd1/ date, title, comment
      common /stdhd2/ dlen, tlen, clen, numblocks, blkids(50)
      save /stdhd1/
      save /stdhd2/
c
c -------------------------------------------------------------
