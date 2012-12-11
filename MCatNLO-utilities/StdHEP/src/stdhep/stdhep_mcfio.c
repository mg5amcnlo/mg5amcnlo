/*******************************************************************************
*									       *
* stdhep_mcfio.c -- C version of mcfio interface routines                      *
*									       *
* Copyright (c) 1995 Universities Research Association, Inc.		       *
* All rights reserved.							       *
* 									       *
* This material resulted from work developed under a Government Contract and   *
* is subject to the following license:  The Government retains a paid-up,      *
* nonexclusive, irrevocable worldwide license to reproduce, prepare derivative *
* works, perform publicly and display publicly by or for the Government,       *
* including the right to distribute to other Government contractors.  Neither  *
* the United States nor the United States Department of Energy, nor any of     *
* their employees, makes any warranty, express or implied, or assumes any      *
* legal liability or responsibility for the accuracy, completeness, or         *
* usefulness of any information, apparatus, product, or process disclosed, or  *
* represents that its use would not infringe privately owned rights.           *
*                                        				       *
*									       *
* Written by Lynn Garren    					       	       *
*									       *
*									       *
*******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <rpc/types.h>
#include <rpc/xdr.h>
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
/* 
*   mcfio/StdHep definitions and include files
*/
#include "mcfio_Dict.h"
#include "stdhep.h"
#include "hepev4.h"
#include "hepeup.h"
#include "heprup.h"
#include "stdhd.h"
#include "stdcnt.h"
#include "stdlun.h"
#include "stdhep_mcfio.h"

struct hepevt hepevt_;
struct hepev2 hepev2_;
struct hepev3 hepev3_;
struct hepev4 hepev4_;
struct hepev5 hepev5_;
struct hepeup hepeup_;
struct heprup heprup_;
struct stdcnt stdcnt_;
struct heplun heplun_;
struct stdstr stdstr_;
struct stdhd1 stdhd1_;
struct stdhd2 stdhd2_;

extern int xdr_stdhep_();
extern int xdr_stdhep_multi_();
extern int xdr_stdhep_4_();
extern int xdr_stdhep_4_multi_();
extern int xdr_stdhep_cm1_();
extern int xdr_hepeup_();
extern int xdr_heprup_();

int StdHepXdrReadInit(char *filename, int ntries, int ist)
{
    int ierr;
    
    mcfioC_Init();
    ierr = StdHepXdrReadOpen(filename, ntries, ist);
    return ierr;
}
int StdHepXdrReadOpen(char *filename, int ntries, int ist)
{
    int istream, iblk;
    int numblocks, blkids[50];

    istream =  mcfioC_OpenReadDirect(filename);
    stdstr_.ixdrstr[ist] = istream;
    if (istream == -1) {
        fprintf(stderr," StdHepXdrReadOpen: cannot open output file \n");
        return -1;
        }
    mcfioC_InfoStreamChar(istream, MCFIO_CREATIONDATE, stdhd1_.date, &stdhd2_.dlen);
    mcfioC_InfoStreamChar(istream, MCFIO_TITLE, stdhd1_.title, &stdhd2_.tlen);
    mcfioC_InfoStreamChar(istream, MCFIO_COMMENT, stdhd1_.comment, &stdhd2_.clen);
    mcfioC_InfoStreamInt(istream, MCFIO_NUMEVTS, &ntries);
    mcfioC_InfoStreamInt(istream, MCFIO_NUMBLOCKS, &numblocks);
    mcfioC_InfoStreamInt(istream, MCFIO_BLOCKIDS, blkids);

    stdhd2_.numblocks = numblocks; 
    for ( iblk=0; iblk < numblocks; ++iblk ) {
        stdhd2_.blkids[iblk] = blkids[iblk];
    }

    stdcnt_.nstdrd = 0;
    stdcnt_.nlhrd = 0;
    fprintf(stdout,
       " StdHepXdrReadOpen: successfully opened input stream %d\n",istream);
    fprintf(stdout,"          title: %s\n",stdhd1_.title);
    fprintf(stdout,"          date: %s\n",stdhd1_.date);
    fprintf(stdout,"                    %d events\n",ntries);
    fprintf(stdout,"                    %d blocks per event\n",stdhd2_.numblocks);
    return 0;
}
int StdHepXdrRead(int *ilbl, int ist)
{
/* Purpose: to read a buffer or an event from the standard common block.
C
C       returns ilbl
C
C		ilbl = 1   - standard HEPEVT common block
C		ilbl = 2   - standard HEPEVT common block and HEPEV2
C		ilbl = 3   - stdevent struct
C	        ilbl = 4   - standard HEPEVT common block with Les Houches
C	        ilbl = 5   - standard HEPEVT common block with Les Houches
C                               and multiple collisions
C		ilbl = 11  -  HEPEUP common block
C		ilbl = 12  -  HEPRUP common block
C		ilbl = 100 - STDHEP begin run record
C		ilbl = 200 - STDHEP end run record
C   */

    int istat;
    int i, numblocks, blkids[50];

    int istream = stdstr_.ixdrstr[ist];
    if(mcfioC_NextEvent(istream) != MCFIO_RUNNING) {
        mcfioC_InfoStreamInt(istream, MCFIO_STATUS, &istat);
        if(istat == MCFIO_EOF) {
            fprintf(stderr,"     StdHepXdrRead: end of file found\n");
            return 1;
            }
        else {
            fprintf(stderr,"     StdHepXdrRead: unrecognized status - stop\n");
            return 2;
            }
        }
    mcfioC_InfoStreamInt(istream, MCFIO_NUMBLOCKS, &numblocks);
    mcfioC_InfoStreamInt(istream, MCFIO_BLOCKIDS, blkids);

    for (i = 0; i < numblocks; i++) {
        if (blkids[i] == MCFIO_STDHEP) {
            StdHepZero();
            if (mcfioC_Block(istream,MCFIO_STDHEP,xdr_stdhep_) != -1) {
                *ilbl = 1;
                if (StdHepTempCopy(2,istream) == 0)
                    stdcnt_.nstdrd = stdcnt_.nstdrd + 1;
                return 0;
                }
            }
        else if (blkids[i] == MCFIO_STDHEPM) {
            StdHepZero();
            if (mcfioC_Block(istream,MCFIO_STDHEPM,xdr_stdhep_multi_) != -1) {
                *ilbl = 2;
                stdcnt_.nstdrd = stdcnt_.nstdrd + 1;
                return 0;
                }
            }
        else if (blkids[i] == MCFIO_STDHEP4) {
            StdHepZero();
            if (mcfioC_Block(istream,MCFIO_STDHEP4,xdr_stdhep_4_) != -1) {
                *ilbl = 4;
                if (StdHepTempCopy(2,istream) == 0)
                    stdcnt_.nstdrd = stdcnt_.nstdrd + 1;
                return 0;
                }
            }
        else if (blkids[i] == MCFIO_STDHEP4M) {
            StdHepZero();
            if (mcfioC_Block(istream,MCFIO_STDHEP4M,xdr_stdhep_4_multi_) != -1) {
                *ilbl = 5;
                stdcnt_.nstdrd = stdcnt_.nstdrd + 1;
                return 0;
                }
            }
        else if (blkids[i] == MCFIO_STDHEPBEG) {
            if (mcfioC_Block(istream,MCFIO_STDHEPBEG,xdr_stdhep_cm1_) != -1) {
                *ilbl = 100;
                return 0;
                }
            }
        else if (blkids[i] == MCFIO_STDHEPEND) {
            if (mcfioC_Block(istream,MCFIO_STDHEPEND,xdr_stdhep_cm1_) != -1) {
                *ilbl = 200;
                return 0;
                }
            }
        else if (blkids[i] == MCFIO_HEPEUP) {
            if (mcfioC_Block(istream,MCFIO_HEPEUP,xdr_hepeup_) != -1) {
                *ilbl = 11;
                stdcnt_.nlhrd = stdcnt_.nlhrd + 1;
                return 0;
                }
            }
        else if (blkids[i] == MCFIO_HEPRUP) {
            if (mcfioC_Block(istream,MCFIO_HEPRUP,xdr_heprup_) != -1) {
                *ilbl = 12;
                stdcnt_.nlhrd = stdcnt_.nlhrd + 1;
                return 0;
                }
            }
        }
    return 1;
}
int StdHepXdrReadMulti(int *ilbl, int ist)
{
/* Purpose: to read a buffer or an event from the standard common block
            this routine handles multiple input streams
C
C       return ilbl
C
C		ilbl = 1   - standard HEPEVT common block
C		ilbl = 2   - standard HEPEVT common block and HEPEV2
C		ilbl = 100 - STDHEP begin run record
C		ilbl = 200 - STDHEP end run record
C   */

    int istat;
    int i, numblocks, blkids[50];

    int istream = stdstr_.ixdrstr[ist];
    if(mcfioC_NextEvent(istream) != MCFIO_RUNNING) {
        mcfioC_InfoStreamInt(istream, MCFIO_STATUS, &istat);
        if(istat == MCFIO_EOF) {
            fprintf(stderr,"     StdHepXdrReadMulti: end of file found\n");
            return 1;
            }
        else {
            fprintf(stderr,
              "     StdHepXdrReadMulti: unrecognized status - stop\n");
            return 2;
            }
        }
    mcfioC_InfoStreamInt(istream, MCFIO_NUMBLOCKS, &numblocks);
    mcfioC_InfoStreamInt(istream, MCFIO_BLOCKIDS, blkids);
    for (i = 0; i < numblocks; i++) {
        if (blkids[i] == MCFIO_STDHEP) {
           if (mcfioC_Block(istream,MCFIO_STDHEP,xdr_stdhep_) == -1) {
                fprintf(stderr,
                  "     StdHepXdrReadMulti: unable to read xdr block\n");
                return 1;
                }
            *ilbl = 1;
            if (StdHepTempCopy(2,istream) == 0)
                stdcnt_.nstdrd = stdcnt_.nstdrd + 1;
            }
        else if (blkids[i] == MCFIO_STDHEPM) {
            fprintf(stderr,
          "    StdHepXdrRead: multiple interaction event - HEPEVT is zeroed\n");
            StdHepZero();
            if (mcfioC_Block(istream,MCFIO_STDHEPM,xdr_stdhep_multi_) == -1) {
                fprintf(stderr,
                  "     StdHepXdrReadMulti: unable to read xdr block\n");
                return 1;
                }
            *ilbl = 2;
            stdcnt_.nstdrd = stdcnt_.nstdrd + 1;
            }
        else if (blkids[i] == MCFIO_STDHEP4) {
           if (mcfioC_Block(istream,MCFIO_STDHEP4,xdr_stdhep_4_) == -1) {
                fprintf(stderr,
                  "     StdHepXdrReadMulti: unable to read xdr block\n");
                return 1;
                }
            *ilbl = 4;
            if (StdHepTempCopy(2,istream) == 0)
                stdcnt_.nstdrd = stdcnt_.nstdrd + 1;
            }
        else if (blkids[i] == MCFIO_STDHEP4M) {
            fprintf(stderr,
          "    StdHepXdrRead: multiple interaction event - HEPEVT is zeroed\n");
            StdHepZero();
            if (mcfioC_Block(istream,MCFIO_STDHEP4M,xdr_stdhep_4_multi_) == -1) {
                fprintf(stderr,
                  "     StdHepXdrReadMulti: unable to read xdr block\n");
                return 1;
                }
            *ilbl = 5;
            stdcnt_.nstdrd = stdcnt_.nstdrd + 1;
            }
        }
    return 0;
}
int StdHepXdrWriteInit(char *filename, char *title, int ntries, int ist)
{
    int ierr;

    mcfioC_Init();
    ierr = StdHepXdrWriteOpen(filename, title, ntries, ist);
    return ierr;
}
int StdHepXdrWriteOpen(char *filename, char *title, int ntries, int ist)
{
    int istream, iblk;
    int numblocks = 8;
    int blkids[50];
    char *comment = '\0';

    blkids[0] = MCFIO_STDHEP;
    blkids[1] = MCFIO_STDHEPM;
    blkids[2] = MCFIO_STDHEPBEG;
    blkids[3] = MCFIO_STDHEPEND;
    blkids[4] = MCFIO_STDHEP4;
    blkids[5] = MCFIO_STDHEP4M;
    blkids[6] = MCFIO_HEPEUP;
    blkids[7] = MCFIO_HEPRUP;

    strncpy(stdhd1_.title,title,255);
    stdhd2_.numblocks = numblocks;
    for ( iblk=0; iblk < numblocks; ++iblk ) {
        stdhd2_.blkids[iblk] = blkids[iblk];
    }
 
    istream =  mcfioC_OpenWriteDirect(filename, title, comment,
                    ntries, blkids, numblocks);
    stdstr_.ixdrstr[ist] = istream;
    if (istream == -1) {
        fprintf(stderr," StdHepXdrWriteOpen: cannot open output file \n");
        return -1;
        }
    fprintf(stdout," StdHepXdrWriteOpen: I/O initialized for StdHep only\n");
    return 0;
}
int StdHepXdrWrite(int ilbl, int ist)
{
    int iret = 0;

    if ((ilbl == 1) || (ilbl == 2))
        iret = StdHepXdrWriteEvent(ilbl, ist);
    else if ((ilbl == 4) || (ilbl == 5))
        iret = StdHepXdrWriteEventLH(ilbl, ist);
    else if (ilbl == 11) 
        iret = StdHepXdrWriteEventEUP(ilbl, ist);
    else if (ilbl == 12)
        iret = StdHepXdrWriteEventRUP(ilbl, ist);
    else if ((ilbl == 100) || (ilbl == 200))
        iret = StdHepXdrWriteCM(ilbl, ist);
    else
        fprintf(stderr,
      "     StdHepXdrWrite: don't know what to do with record type %d\n", ilbl);
    return iret;
}
int StdHepXdrWriteCM(int ilbl, int ist)
{
    int istream = stdstr_.ixdrstr[ist];
    if (ilbl == 100) {
        if (mcfioC_Block(istream, MCFIO_STDHEPBEG, xdr_stdhep_cm1_) == -1) {
            fprintf(stderr,
              "     StdHepXdrWriteCM: error filling stdhep cm1 common block\n");
            return 2;
            }
        }
    else if (ilbl == 200) {
        if (mcfioC_Block(istream, MCFIO_STDHEPEND, xdr_stdhep_cm1_) == -1) {
            fprintf(stderr,
              "     StdHepXdrWriteCM: error filling stdhep cm1 common block\n");
            return 2;
            }
        }
    else {
        fprintf(stderr,
           "     StdHepXdrWriteCM: called with improper label %d\n",ilbl);
        return 3;
        }
    if (mcfioC_NextEvent(istream) == -1) {
        fprintf(stderr,
          "     StdHepXdrWriteCM: error writing stdhep cm1 xdr block\n");
        return 1;
        }
    return 0;
}
int StdHepXdrWriteEvent(int ilbl, int ist)
{
    int istream = stdstr_.ixdrstr[ist];
    if ((ilbl != 1) && (ilbl != 2)) {
        fprintf(stderr,
          "     StdHepXdrWriteEvent: called with illegal label %d\n",
                            ilbl);
        return 3;
        }
    else if (hepevt_.nhep <= 0) {
        fprintf(stderr,
          "     StdHepXdrWriteEvent: event %d is empty\n", hepevt_.nevhep);
        return 0;
        }
    else if (ilbl == 1) {
        if (StdHepTempCopy(1,istream) != 0) {
            fprintf(stderr,
              "     StdHepXdrWriteEvent: copy failed - event not written\n");
            return 4;
            }
        if (mcfioC_Block(istream, MCFIO_STDHEP, xdr_stdhep_) == -1) {
            fprintf(stderr,
          "     StdHepXdrWriteEvent: error filling stdhep block for event %d\n",
                     hepevt_.nevhep);
            return 2;
            }
        mcfioC_SetEventInfo(istream, MCFIO_STORENUMBER, &hepevt_.nevhep);
        }
    else if (ilbl == 2) {
        if (mcfioC_Block(istream, MCFIO_STDHEPM, xdr_stdhep_multi_) == -1) {
            fprintf(stderr,
          "     StdHepXdrWriteEvent: error filling stdhep block for event %d\n",
                     hepevt_.nevhep);
            return 2;
            }
        mcfioC_SetEventInfo(istream, MCFIO_STORENUMBER, &hepevt_.nevhep);
        }
    if (mcfioC_NextEvent(istream) == -1) {
        fprintf(stderr,"     StdHepXdrWriteCM: error writing event %d\n",
                         hepevt_.nevhep);
        return 1;
        }
    stdcnt_.nstdwrt = stdcnt_.nstdwrt + 1;
    return 0;
}
int StdHepXdrWriteEventLH(int ilbl, int ist)
{
    int istream = stdstr_.ixdrstr[ist];
    if ((ilbl != 4) && (ilbl != 5)) {
        fprintf(stderr,
          "     StdHepXdrWriteEventLH: called with illegal label %d\n",
                            ilbl);
        return 3;
        }
    else if (hepevt_.nhep <= 0) {
        fprintf(stderr,
          "     StdHepXdrWriteEventLH: event %d is empty\n", hepevt_.nevhep);
        return 0;
        }
    else if (ilbl == 4) {
        if (StdHepTempCopy(1,istream) != 0) {
            fprintf(stderr,
              "     StdHepXdrWriteEventLH: copy failed - event not written\n");
            return 4;
            }
        if (mcfioC_Block(istream, MCFIO_STDHEP4, xdr_stdhep_4_) == -1) {
            fprintf(stderr,
          "     StdHepXdrWriteEventLH: error filling stdhep block for event %d\n",
                     hepevt_.nevhep);
            return 2;
            }
        mcfioC_SetEventInfo(istream, MCFIO_STORENUMBER, &hepevt_.nevhep);
        }
    else if (ilbl == 5) {
        if (mcfioC_Block(istream, MCFIO_STDHEP4M, xdr_stdhep_4_multi_) == -1) {
            fprintf(stderr,
          "     StdHepXdrWriteEventLH: error filling stdhep block for event %d\n",
                     hepevt_.nevhep);
            return 2;
            }
        mcfioC_SetEventInfo(istream, MCFIO_STORENUMBER, &hepevt_.nevhep);
        }
    if (mcfioC_NextEvent(istream) == -1) {
        fprintf(stderr,"     StdHepXdrWriteLH: error writing event %d\n",
                         hepevt_.nevhep);
        return 1;
        }
    stdcnt_.nstdwrt = stdcnt_.nstdwrt + 1;
    return 0;
}
int StdHepXdrWriteEventEUP(int ilbl, int ist)
{
    int istream = stdstr_.ixdrstr[ist];
    if ( ilbl != 11 ) {
        fprintf(stderr,
          "     StdHepXdrWriteEventEUP: called with illegal label %d\n",
                            ilbl);
        return 3;
        }
    else if (hepeup_.nup <= 0) {
        fprintf(stderr,
          "     StdHepXdrWriteEventEUP: event is empty\n");
        return 0;
        }
    else if (ilbl == 11) {
        if (mcfioC_Block(istream, MCFIO_HEPEUP, xdr_hepeup_) == -1) {
            fprintf(stderr,
          "     StdHepXdrWriteEventEUP: error filling stdhep block for event\n");
            return 2;
            }
        }
    if (mcfioC_NextEvent(istream) == -1) {
        fprintf(stderr,"     StdHepXdrWriteEUP: error writing event\n");
        return 1;
        }
    stdcnt_.nlhwrt = stdcnt_.nlhwrt + 1;
    return 0;
}
int StdHepXdrWriteEventRUP(int ilbl, int ist)
{
    int istream = stdstr_.ixdrstr[ist];
    if ( ilbl != 12 ) {
        fprintf(stderr,
          "     StdHepXdrWriteEventRUP: called with illegal label %d\n",
                            ilbl);
        return 3;
        }
    else if (ilbl == 12) {
        if (mcfioC_Block(istream, MCFIO_HEPRUP, xdr_heprup_) == -1) {
            fprintf(stderr,
          "     StdHepXdrWriteEventRUP: error filling stdhep block for event\n");
            return 2;
            }
        }
    if (mcfioC_NextEvent(istream) == -1) {
        fprintf(stderr,"     StdHepXdrWriteRUP: error writing event\n");
        return 1;
        }
    stdcnt_.nlhwrt = stdcnt_.nlhwrt + 1;
    return 0;
}
void StdHepXdrEnd(int ist)
{
    int inum, ieff;

    int istream = stdstr_.ixdrstr[ist];
    mcfioC_InfoStreamInt(istream, MCFIO_NUMWORDS, &inum);
    mcfioC_InfoStreamInt(istream, MCFIO_EFFICIENCY, &ieff);
    mcfioC_Close(istream);
    fprintf(stdout,
       "          StdHepXdrEnd: %d words i/o with %d efficiency\n",inum,ieff);
}
void StdHepPrintHeader( )
{
    fprintf(stdout," StdHep MCFio header information:\n");
    fprintf(stdout,"          title: %s\n",stdhd1_.title);
    fprintf(stdout,"          date:  %s\n",stdhd1_.date);
    fprintf(stdout,"          %s\n",stdhd1_.comment);
    fprintf(stdout,"                    %d blocks per event\n",stdhd2_.numblocks);
}
