      subroutine mcfio_FPrintDictionary(ilun)

c*******************************************************************************
c									       *
c mcfio_FPrintDictionary.F -- Fortran version of PrintDictionary               *
c									       *
c Copyright (c) 1994 Universities Research Association, Inc.		       *
c All rights reserved.							       *
c 									       *
c This material resulted from work developed under a Government Contract and   *
c is subject to the following license:  The Government retains a paid-up,      *
c nonexclusive, irrevocable worldwide license to reproduce, prepare derivative *
c works, perform publicly and display publicly by or for the Government,       *
c including the right to distribute to other Government contractors.  Neither  *
c the United States nor the United States Department of Energy, nor any of     *
c their employees, makes any warranty, express or implied, or assumes any      *
c legal liability or responsibility for the accuracy, completeness, or         *
c usefulness of any information, apparatus, product, or process disclosed, or  *
c represents that its use would not infringe privately owned rights.           *
c                                        				       *
c									       *
c Written by Paul Lebrun, Lynn Garren					       *
c									       *
c									       *
c*******************************************************************************

      integer ilun
      write(ilun,1001)
1001  format(//
     1 ' Mcfast I/o Dictionary for Key words used in mcfio_Info',
     2 ' routines'/
     3 ' --------------------------------------------------------',
     4 '-------')
   
      write(ilun,1002)
1002  format(/
     1 ' For Streams '/
     2 ' -------------- '/
     3 '   MCFIO_STATUS: The current status of the file;  '/
     4 '                 the answer can be set to: '/
     5 '   MCFIO_BOF : at beginning of file '/
     6 '   MCFIO_EOF : at the end of file '/
     7 '   MCFIO_RUNNING: At least a valid file header has been',
     8 ' read or written'/
     9 '   MCFIO_READORWRITE: if set MCFIO_READ, open for read only '/
     1 '                      if set MCFIO_WRITE, open for write only '/
     2 '   MCFIO_DIRECTORSEQUENTIAL: if set MCFIO_DIRECT, accessing a',
     3 ' UNIX file '/
     4 '                           : if set MCFIO_SEQUENTIAL,',
     5 ' accessing a tape '/
     6 '   MCFIO_NUMEVTS : Total number of events encode/decoded',
     7 ' so far.'/
     8 '   MCFIO_NUMBLOCK: The number of blocks defined in the file.')
    
      write(ilun,1003)
1003  format(
     1 '   MCFIO_BLOCKIDS: The I.D. of the block defined in the file.'/
     2 '   MCFIO_NUMWORDS: Total number of 4-bytes words',
     2 ' encoded/decoded so far. '/
     3 '   MCFIO_EFFICIENCY: The overhead in blocking and XDR',
     3 ' (*10000 ) '/
     4 '   MCFIO_CREATIONDATE: The date (30 Character) when the file',
     4 ' was written '/
     5 '   MCFIO_TITLE: The title (255 Characters max) for the job '/
     6 '   MCFIO_COMMENT: The comment (255 Characters max) for the job')
 
      write(ilun,1004)
1004  format(/
     1 ' For Sequential Access only '/
     2 ' -------------------------- '/
     2 '   MCFIO_FILENUMBER : The Sequential file number currently',
     2 ' accessed.'/
     3 '   MCFIO_MAXLREC: Maximum Record length'/
     4 '   MCFIO_MINLREC: Minumum Record length'/
     5 '   MCFIO_NUMRECORDS: The number of records in the current',
     5 ' event'/
     6 '   MCFIO_RECORDLENGTHS: The record lengths for the current', 
     6 ' event'/
     7 '   MCFIO_DEVICENAME: The device name opened by the stream '/
     8 '                     (character string, 255 l.)')
 
      write(ilun,1005)
1005  format(/
     1 ' For Direct Access only '/
     2 ' ----------------------- '/
     2 '   MCFIO_FILENAME: The UNIX file name opened by the stream '/
     3 '                   (character string, 255 l.)')
   
      write(ilun,1006)
1006  format(/
     1 ' For Events '/
     2 ' -------------- '/
     3 '   MCFIO_NUMBLOCK: The number of blocks defined in the event.'/
     4 '   MCFIO_BLOCKIDS: The I.D. of the block defined in the event.'/
     5 '   MCFIO_EVENTNUMBER: The Event Number for this event. '/
     6 '   MCFIO_STORENUMBER: The Store Number for this event. '/
     7 '   MCFIO_RUNNUMBER: The Run Number for this event. '/
     8 '   MCFIO_TRIGGERMASK: The Trigger Mask for this event. '/
     9 '   MCFIO_VERSION: The 4-Character version of the event header')
   
      write(ilun,1007)
1007  format(/
     1 ' For Blocks '/
     2 ' -------------- '/
     3 '   MCFIO_VERSION: The 4-Character version of a particular',
     4 ' block'/)
      return
      end
