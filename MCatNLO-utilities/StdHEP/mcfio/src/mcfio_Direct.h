/*******************************************************************************
*									       *
* mcfio_Direct.h --  Include file for mcfast Direct i/o layer. 		       *
*									       *
* Copyright (c) 1994 Universities Research Association, Inc.		       *
* All rights reserved.							       *
*									       *
*******************************************************************************/
#define INITIATE 3
#define FLUSH 4
int mcfioC_OpenReadDirect(char *filename);
int mcfioC_OpenReadMapped(char *filename);
int mcfioC_OpenWriteDirect(char *filename, char *title, char *comment, 
                           int numevts_pred, int *blkIds, u_int nBlocks);
int mcfioC_NextEvent(int stream);
int mcfioC_SpecificEvent(int stream, int ievt,
                             int istore, int irun, int itrig);
int mcfioC_NextSpecificEvent(int stream, int ievt,
                             int istore, int irun, int itrig);
void mcfioC_CloseDirect(int jstr);
void mcfioC_RewindDirect(int jstr);
int  mcfioC_WrtEvt(mcfStream *str, int mode);    
int  mcfioC_Wrttable(mcfStream *str, int mode);     
int  mcfioC_Wrtfhead(mcfStream *str, int mode);
