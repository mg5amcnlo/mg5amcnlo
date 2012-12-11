/*******************************************************************************
*									       *
* mcfio_dict.h --  Dictionary for Key words used in Info routines.             *
*									       *
* Copyright (c) 1994 Universities Research Association, Inc.		       *
* All rights reserved.							       *
*									       *
*******************************************************************************/
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
