/*******************************************************************************
*									       *
* mcfio.h --  Include file for mcfast I/O subsystem, C interface.      	       *
*									       *
* Copyright (c) 1994 Universities Research Association, Inc.		       *
* All rights reserved.							       *
*									       *
*******************************************************************************/
/*
** 	This file is a summary of various other .h files, assembled for 
** 	the ease of use of the yet undocumented C API for MCFIO. 
** 	Please refer to detailed documentation from the the F77 API.
**
**	Rule : mcfio_*   ==> F77 API
**	       mcfioC_*  ==> C API
*/
#define MCFIO_VERSION 100
#define MCFIO_STATUS 101
#define MCFIO_RUNNING 102
#define MCFIO_BOF 103
#define MCFIO_EOF 104
#define MCFIO_NUMBLOCKS 501
#define MCFIO_READORWRITE 502
#define MCFIO_READ 1
#define MCFIO_WRITE 2
#define MCFIO_DIRECTORSEQUENTIAL 503
#define MCFIO_DIRECT 1
#define MCFIO_SEQUENTIAL 2
#define MCFIO_MEMMAPPED 3
#define MCFIO_BLOCKIDS 504
#define MCFIO_NUMWORDS 505
#define MCFIO_EFFICIENCY 506
#define MCFIO_NUMEVTS 507
#define MCFIO_FILENUMBER 508
#define MCFIO_MAXREC 509
#define MCFIO_MINREC 510
#define MCFIO_NUMRECORDS 511
#define MCFIO_RECORDLENGTHS 512
#define MCFIO_TITLE 1001
#define MCFIO_COMMENT 1002
#define MCFIO_CREATIONDATE 1003
#define MCFIO_CLOSINGDATE 1013
#define MCFIO_FILENAME 1004
#define MCFIO_DEVICENAME 1005
#define MCFIO_EVENTNUMBER 2001
#define MCFIO_STORENUMBER 2002
#define MCFIO_RUNNUMBER 2003
#define MCFIO_TRIGGERMASK 2004
#define MCFIO_NUMNTUPLES 4001
#define MCFIO_NTUPLESLIST 4002
/*
** Block definition now. Start counting at 101 See also mcfioC_GetBlockNames
*/
#define MCFIO_STDHEP 101
#define MCFIO_OFFTRACKARRAYS 102
#define MCFIO_OFFTRACKSTRUCT 103
#define MCFIO_TRACEARRAYS    104
#define MCFIO_STDHEPM 105
#define MCFIO_STDHEPBEG 106
#define MCFIO_STDHEPEND 107
#define MCFIO_STDHEPCXX 108
#define MCFIO_STDHEP4 201
#define MCFIO_STDHEP4M 202
#define MCFIO_HEPEUP 203
#define MCFIO_HEPRUP 204

extern "C" void mcfioC_Init(void);
extern "C" void mcfioC_Close(int istream);
extern "C" void mcfioC_PrintDictionary(void);
extern "C" unsigned int mcfioC_InfoNumSream(int *istreams, unsigned int nmax);
extern "C" void mcfioC_InfoStreamInt(int istream, int key, int *value);
extern "C" void mcfioC_InfoStreamChar(int istream, int key, char *answer, int *lret);
extern "C" void mcfioC_InfoEventInt(int istream, int key, int *value);
extern "C" void mcfioC_InfoEventChar(int istream, int key, char *answer, int *lret);
extern "C" void mcfioC_SetEventInfo(int istream, int key, int *value);
extern "C" void mcfioC_InfoBlockChar(int stream, int blk, int key,
                            char *answer, int *lret);
extern "C" void mcfioC_GetBlockName(int blkId, char *answer);
extern "C" int mcfioC_OpenReadSequential(char *device, char *label, int filenumber);
extern "C" int mcfioC_OpenWriteSequential(char *device, char *label, char *title, 
               char *comment, int numevts_pred, 
                int *blkIds, unsigned int nBlocks);
extern "C" int mcfioC_OpenReadDirect(char *filename);
extern "C" int mcfioC_OpenReadMapped(char *filename);
extern "C" int mcfioC_OpenWriteDirect(char *filename, char *title, char *comment, 
                           int numevts_pred, int *blkIds, u_int nBlocks);
extern "C" int mcfioC_NextEvent(int stream);
extern "C" int mcfioC_SpecificEvent(int stream, int ievt,
                             int istore, int irun, int itrig);
extern "C" int mcfioC_NextSpecificEvent(int stream, int ievt,
                             int istore, int irun, int itrig);
extern "C" void mcfioC_CloseDirect(int jstr);
extern "C" void mcfioC_RewindDirect(int jstr);
extern "C" int mcfioC_Block(int stream, int blkid, 
  bool_t xdr_filtercode(XDR *xdrs, int *blockid, int *ntot, char **version));
extern "C" int mcfioC_NTuple(int stream, int nTupleid, char * version); 
extern "C" int mcfioC_NTupleMult(int stream, int nTupleid, char * version);
extern "C" int mcfioC_NTupleVar(int stream, int nTupleid, int ivar, char * version);
extern "C" int mcfioC_NTupleSubVar(int stream, int nTupleid, int ivar, int multIndex,
                           char * version);
extern "C" int mcfioC_NTupleSubStruct(int stream, int nTupleid, int multIndex,
                           char * version);
extern "C" void mcfioC_DefineUserBlock(int blkNum, char *descr); 
extern "C" char *mcfioC_UserBlockDescript(int blkn); 


