/*******************************************************************************
*									       *
* mcfio_Block.h --  Include file for mcfast Direct i/o layer. 		       *
*									       *
* Copyright (c) 1994 Universities Research Association, Inc.		       *
* All rights reserved.							       *
*									       *
*******************************************************************************/
int mcfioC_Block(int stream, int blkid, 
 bool_t xdr_filtercode(XDR *xdrs, int *blockid, int *ntot, char **version));
int mcfioC_NTuple(int stream, int nTupleid, char * version); 
int mcfioC_NTupleMult(int stream, int nTupleid, char * version);
int mcfioC_NTupleVar(int stream, int nTupleid, int ivar, char * version);
int mcfioC_NTupleSubVar(int stream, int nTupleid, int ivar, int multIndex,
                           char * version);
int mcfioC_NTupleSubStruct(int stream, int nTupleid, int multIndex,
                           char * version);
 

