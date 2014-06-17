/*******************************************************************************
*									       *
* mcfio_Block.c --  Utility routines for the McFast Monte-Carlo                  *
*		The routine to encode/decode a block 	                       *
*									       *
* Copyright (c) 1994 Universities Research Association, Inc.		       *
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
* Written by Paul Lebrun						       *
*									       *
*									       *
*******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <sys/param.h>
#include <rpc/types.h>
#include <sys/types.h>
#include <rpc/xdr.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#ifdef SUNOS
#include <floatingpoint.h>
#else /* SUNOS */
#include <float.h>
#endif /* SUNOS */
#include "mcf_nTupleDescript.h"
#include "mcf_xdr.h"
#include "mcf_xdr_Ntuple.h"
#include "mcfio_Dict.h"
#include "mcfio_Util1.h"
#include "mcf_NTuIOUtils.h"
#include "mcfio_Direct.h"
#include "mcfio_Block.h"
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

int mcfioC_Block(int stream, int blkid, 
 bool_t xdr_filtercode(XDR *xdrs, int *blockid, int *ntot, char **version))
/*
** Routine to decode or encode a particular Block. Return 1 if O.K, 
** -1 if a problem or unknow block.  
**
** Adding Ntuple instances ... October 1995.
*/
{ 
  int i, j, jstr, idtmp, ntot, nbuff;
  bool_t ok;
  off_t p1;
  mcfStream *str;
   
  if (McfStreamPtrList == NULL) { 
     fprintf(stderr,
  " mcfio_Block: You must first initialize by calling mcfio_Init.\n"); 
     return -1;
  }
  jstr = stream-1;
  if (McfStreamPtrList[jstr] == NULL) { 
     fprintf(stderr,
 " mcfio_Block: First, declare the stream by calling mcfio_Open...\n"); 
     return -1;
  }
  str = McfStreamPtrList[jstr];
  if ((str->row == MCFIO_WRITE) && 
      (str->fhead->nBlocks == str->ehead->nBlocks)) {
     fprintf(stderr,
 " mcfio_Block: Maximum number of Blocks reached for stream %d ...\n", stream);
     fprintf(stderr,
 "              Please upgrade the declaration mcfio_Open statement \n");
     return -1;
  }
     
  if (str->row == MCFIO_READ) {
      for(i=0, j=-1; i<str->ehead->nBlocks; i++) {
           if (str->ehead->blockIds[i] == blkid) j = i;
        }
      if (j == -1) {
        fprintf(stderr,
 " mcfio_Block: Unable to find block i.d. %d in Stream %d \n", blkid, stream);
          return -1;  
      }
      if (fseeko(str->filePtr,str->ehead->ptrBlocks[j],SEEK_SET) != 0) {
        fprintf(stderr,
         " mcfio_Block: Unable to position stream at block %d \n", blkid);
          return -1;  
      }
      str->currentPos = str->ehead->ptrBlocks[j];
  } else if (str->row == MCFIO_WRITE)  {
      idtmp = blkid;
      /*
      ** if to Sequential media, one first has to make sure we have 
      ** enough room in the buffer.
      */
      if (str->dos == MCFIO_SEQUENTIAL) {
         str->xdr->x_op = XDR_MCFIOCODE;
         ok = xdr_filtercode(str->xdr, &idtmp, &ntot, McfGenericVersion);
         str->xdr->x_op = XDR_ENCODE;
         if ((str->currentPos + 4*(ntot + 1)) > str->bufferSize) {
          /*
          ** Once again, I don't trust realloc, got to copy to the second 
          ** buffer. 
          */
             nbuff = 1 + 
                    (((4*(ntot + 1)) + (str->currentPos - str->firstPos))/
                       str->maxlrec);
             str->buffer2 = 
                 (char *) malloc (sizeof(char) * (str->maxlrec *nbuff));
             memcpy(str->buffer2, str->buffer, 
                       (str->currentPos - str->firstPos));
             free(str->buffer);
             str->buffer = str->buffer2;
             str->buffer2 = NULL;
             str->bufferSize = str->maxlrec * nbuff;
             xdrmem_create(str->xdr, str->buffer, str->bufferSize, XDR_ENCODE);
             if (fseeko(str->filePtr,str->currentPos,SEEK_SET) != 0) {
                 fprintf(stderr,
             " mcfio_Block:\n\
 Unable to position stream %d at block %d after realocation.\n", stream, blkid);
                 return -1; 
             } 
          }
       }
   }
   p1 = str->currentPos;
   ok = xdr_filtercode(str->xdr, &idtmp, &ntot, McfGenericVersion);
   if (ok == FALSE) {
        fprintf(stderr,
         " mcfio_Block: Unable to encode or decode block I.D. %d \n", blkid);
         j = str->ehead->nBlocks;
         if (fseeko(str->filePtr,p1,SEEK_SET) != 0) 
           fprintf(stderr,
         " mcfio_Block: Unable to position stream at block %d \n", blkid);
         return -1;
      }
   if(blkid != idtmp) {
        fprintf(stderr,
         " mcfio_Block: Unexpected I.D = %d found instead of I.D. %d \n",
              idtmp, blkid);
        return -1;
      }
    if (str->row == MCFIO_WRITE)  {  
      str->ehead->blockIds[str->ehead->nBlocks] = blkid;
      str->ehead->ptrBlocks[str->ehead->nBlocks] = p1;
      str->ehead->nBlocks++; 
    }
    str->currentPos = ftello(str->filePtr);    
    str->numWordsC += (ntot/4);
    str->numWordsT += ((str->currentPos-p1)/4);
    return 1;
        
}
int mcfioC_NTuple(int stream, int nTupleId, char * version)
{ 
  int i, j, jstr, idtmp, ntot, nbuff;
  bool_t ok;
  off_t p1;
  mcfStream *str;
  nTuDDL *ddl;
  descrGenNtuple *dNTu;
     
  if (McfStreamPtrList == NULL) { 
     fprintf(stderr,
  " mcfio_NTuple: You must first initialize by calling mcfio_Init.\n"); 
     return -1;
  }
  jstr = stream-1;
  if (McfStreamPtrList[jstr] == NULL) { 
     fprintf(stderr,
 " mcfio_NTuple: First, declare the stream by calling mcfio_Open...\n"); 
     return -1;
  }
  
  ddl = mcf_GetNTuByStreamID(stream, nTupleId);
  if (ddl == NULL) {
     fprintf(stderr,
 " mcfio_NTuple: Illegal or inexistant NTuple Id %d for stream %d \n", 
     nTupleId, stream); 
     return -1;
  }
  if (ddl->reference == NULL) dNTu = ddl->descrNtu;
  else dNTu = ddl->reference->descrNtu;
  str = McfStreamPtrList[jstr];
  if ((str->row == MCFIO_WRITE) && 
      (str->fhead->nNTuples == str->ehead->nNTuples)) {
     fprintf(stderr,
" mcfio_NTuple: Maximum number of NTuples reached for stream %d ...\n", stream);
     fprintf(stderr,
 "              Please upgrade the Ntuple declarations statements. \n");
     return -1;
  }
     
  if (str->row == MCFIO_READ) {
      for(i=0, j=-1; i<str->ehead->nNTuples; i++) {
           if (str->ehead->nTupleIds[i] == ddl->seqNTuId) j = i;
        }
      if (j == -1) {
        fprintf(stderr,
 " mcfio_NTuple: Unable to find NTuple i.d. %d in Stream %d \n",
          nTupleId, stream);
          return -1;  
      }
      if (fseeko(str->filePtr,str->ehead->ptrNTuples[j],SEEK_SET) != 0) {
        fprintf(stderr,
         " mcfio_NTuple: Unable to position stream at NTuple %d \n", nTupleId);
          return -1;  
      }
      str->currentPos = str->ehead->ptrNTuples[j];
  } else if (str->row == MCFIO_WRITE)  {
      /*
      ** if to Sequential media, one first has to make sure we have 
      ** enough room in the buffer.
      */
      if (str->dos == MCFIO_SEQUENTIAL) {
         str->xdr->x_op = XDR_MCFIOCODE;
         ok = xdr_mcfast_NTuple(str->xdr, dNTu, &ntot,
                                 ddl->seqNTuId, version);
         str->xdr->x_op = XDR_ENCODE;
         if (ok == FALSE) {
             fprintf(stderr,
 "mcfio_NTuple: can not Encode or Decode Ntuple id % on Seq. Stream %d ", 
             nTupleId, stream);
             return -1;
         }
         if ((str->currentPos + 4*(ntot + 1)) > str->bufferSize) {
          /*
          ** Once again, I don't trust realloc, got to copy to the second 
          ** buffer. 
          */
             nbuff = 1 + 
                    (((4*(ntot + 1)) + (str->currentPos - str->firstPos))/
                       str->maxlrec);
             str->buffer2 = 
                 (char *) malloc (sizeof(char) * (str->maxlrec *nbuff));
             memcpy(str->buffer2, str->buffer, 
                       (str->currentPos - str->firstPos));
             free(str->buffer);
             str->buffer = str->buffer2;
             str->buffer2 = NULL;
             str->bufferSize = str->maxlrec * nbuff;
             xdrmem_create(str->xdr, str->buffer, str->bufferSize, XDR_ENCODE);
             if (fseeko(str->filePtr,str->currentPos,SEEK_SET) != 0) {
                 fprintf(stderr,
             " mcfio_NTuple:\n\
 Unable to position stream %d at Ntuple %d after realocation.\n",
                 stream, nTupleId);
                 return -1; 
             } 
          }
       }
   }
   p1 = str->currentPos;
   ok = xdr_mcfast_NTuple(str->xdr, dNTu, &ntot, ddl->seqNTuId, version);
   if (ok == FALSE) {
        fprintf(stderr,
         " mcfio_NTuple: Unable to encode or decode NTuple I.D. %d \n",
             nTupleId);
         j = str->ehead->nNTuples;
         if (fseeko(str->filePtr,p1,SEEK_SET) != 0) 
           fprintf(stderr,
         " mcfio_NTuple: Unable to position stream at NTuple %d \n", nTupleId);
         return -1;
      }
    if (str->row == MCFIO_WRITE)  {  
      str->ehead->nTupleIds[str->ehead->nNTuples] = ddl->seqNTuId;
      str->ehead->ptrNTuples[str->ehead->nNTuples] = p1;
      str->ehead->nNTuples++; 
    }
    str->currentPos = ftello(str->filePtr);
    str->numWordsC += (ntot/4);
    str->numWordsT += ((str->currentPos-p1)/4);
    return 1;
        
}
/*
** Optimized version used exclusively to read the multiplicity value 
** within an NTuple. It is assumed that the stream is open read direct 
** access (No checks!), and the event table is available, and the 
** NTuple is accessible.  Once again, No checks! Use at your onw risk.
** Also, we do not keep record of the number of byte Read.  
*/
int mcfioC_NTupleMult(int stream, int nTupleId, char * version)
{ 
  int i, j, jstr, idtmp, ntot, nbuff;
  bool_t ok;
  mcfStream *str;
  nTuDDL *ddl;
  descrGenNtuple *dNTu;
     
  jstr = stream-1;
  ddl = mcf_GetNTuByStreamID(stream, nTupleId);
  if (ddl->reference == NULL) dNTu = ddl->descrNtu;
  else dNTu = ddl->reference->descrNtu;
  str = McfStreamPtrList[jstr];
  for(i=0, j=-1; i<str->ehead->nNTuples; i++) {
           if (str->ehead->nTupleIds[i] == ddl->seqNTuId) j = i;
   }
  if (fseeko(str->filePtr,str->ehead->ptrNTuples[j],SEEK_SET) != 0) {
        fprintf(stderr,
    " mcfio_NTupleMult: Unable to position stream at NTuple %d \n", nTupleId);
          return -1;  
      }
  str->currentPos = str->ehead->ptrNTuples[j];
  if (dNTu->multXDROffset == 0) 
      ok = xdr_mcfast_NTupleXDRPtr(str->xdr, dNTu, &ntot,
                                   ddl->seqNTuId, version);
   else ok = xdr_mcfast_NTupleMult(str, dNTu, version);
   if (ok == FALSE) {
        fprintf(stderr,
         " mcfio_NTuple: Unable to encode or decode NTuple I.D. %d \n",
             nTupleId);
         j = str->ehead->nNTuples;
         if (fseeko(str->filePtr,str->currentPos,SEEK_SET) != 0) 
           fprintf(stderr,
         " mcfio_NTuple: Unable to position stream at NTuple %d \n", nTupleId);
         return -1;
      }
      /*
      ** This probably could be optimized away. Note the that the current 
      ** position of the stream strored in str->currentPos is no longer 
      ** valied exiting this routine. However, there is enough redundancy 
      ** in the data structure to figure out where we could go..
      */
     /*  xdr_setpos(str->xdr, str->currentPos);   */ 
    return TRUE;
        
}
            
/*
** Optimized version used exclusively to read a specific variable  
** within an NTuple. Valid only if the variable is of fixed size 
** (e.g. not indexed by multiplicity) or if the data structure organization is
** of type parallel array. It is assumed that the stream is open read direct 
** access (No checks!), and the event table is available, and the 
** NTuple is accessible.  Once again, No checks! Use at your own risk.
*/
int mcfioC_NTupleVar(int stream, int nTupleId, int ivar, char * version)
{ 
  int i, j, jstr, idtmp, ntot, nbuff;
  bool_t ok;
  mcfStream *str;
  nTuDDL *ddl;
  descrGenNtuple *dNTu;
     
  jstr = stream-1;
  ddl = mcf_GetNTuByStreamID(stream, nTupleId);
  if (ddl->reference == NULL) dNTu = ddl->descrNtu;
  else dNTu = ddl->reference->descrNtu;
  str = McfStreamPtrList[jstr];
  for(i=0, j=-1; i<str->ehead->nNTuples; i++) {
           if (str->ehead->nTupleIds[i] == ddl->seqNTuId) j = i;
   }
  if (fseeko(str->filePtr,str->ehead->ptrNTuples[j],SEEK_SET) != 0) {
        fprintf(stderr,
    " mcfio_NTupleVar: Unable to position stream at NTuple %d \n", nTupleId);
          return -1;  
      }
  str->currentPos = str->ehead->ptrNTuples[j];
  if (dNTu->multXDROffset == 0) 
      ok = xdr_mcfast_NTupleXDRPtr(str->xdr, dNTu, &ntot,
                                   ddl->seqNTuId, version);
   else ok = xdr_mcfast_NTupleVar(str, dNTu, ivar, version);
   if (ok == FALSE) {
        fprintf(stderr,
         " mcfio_NTuple: Unable to encode or decode NTuple I.D. %d \n",
             nTupleId);
         j = str->ehead->nNTuples;
         if (fseeko(str->filePtr,str->currentPos,SEEK_SET) != 0) 
           fprintf(stderr,
         " mcfio_NTuple: Unable to position stream at NTuple %d \n", nTupleId);
         return -1;
      }
    return TRUE;
        
}
/*
** Optimized version used exclusively to read a specific variable within a  
** substructure within an NTuple. Valid only if of type indexed  
** and if the data structure organization is
** of type VAX FORTRAN d/s. It is assumed that the stream is open read direct 
** access (No checks!), and the event table is available, and the 
** NTuple is accessible.  Once again, No checks! Use at your own risk.
*/
int mcfioC_NTupleSubVar(int stream, int nTupleId, int ivar, int multIndex,
                               char * version)
{ 
  int i, j, jstr, idtmp, ntot, nbuff;
  bool_t ok;
  mcfStream *str;
  nTuDDL *ddl;
  descrGenNtuple *dNTu;
     
  jstr = stream-1;
  ddl = mcf_GetNTuByStreamID(stream, nTupleId);
  if (ddl->reference == NULL) dNTu = ddl->descrNtu;
  else dNTu = ddl->reference->descrNtu;
  str = McfStreamPtrList[jstr];
  for(i=0, j=-1; i<str->ehead->nNTuples; i++) {
           if (str->ehead->nTupleIds[i] == ddl->seqNTuId) j = i;
   }
  if (fseeko(str->filePtr,str->ehead->ptrNTuples[j],SEEK_SET) != 0) {
        fprintf(stderr,
    " mcfio_NTupleVar: Unable to position stream at NTuple %d \n", nTupleId);
          return -1;  
      }
  str->currentPos = str->ehead->ptrNTuples[j];
  if (dNTu->multXDROffset == 0) 
      ok = xdr_mcfast_NTupleXDRPtr(str->xdr, dNTu, &ntot,
                                   ddl->seqNTuId, version);
   else ok = xdr_mcfast_NTupleSubVar(str, dNTu, ivar, multIndex, version);
   if (ok == FALSE) {
        fprintf(stderr,
         " mcfio_NTuple: Unable to encode or decode NTuple I.D. %d \n",
             nTupleId);
         j = str->ehead->nNTuples;
         if (fseeko(str->filePtr,str->currentPos,SEEK_SET) != 0) 
           fprintf(stderr,
         " mcfio_NTuple: Unable to position stream at NTuple %d \n", nTupleId);
         return -1;
      }
    return TRUE;
        
}
/*
** Optimized version used exclusively to read a specific   
** substructure within an NTuple. Valid only if of type indexed  
** and if the data structure organization is
** of type VAX FORTRAN d/s. It is assumed that the stream is open read direct 
** access (No checks!), and the event table is available, and the 
** NTuple is accessible.  Once again, No checks! Use at your own risk.
*/
int mcfioC_NTupleSubStruct(int stream, int nTupleId, int multIndex,
                               char * version)
{ 
  int i, j, jstr, idtmp, ntot, nbuff;
  bool_t ok;
  mcfStream *str;
  nTuDDL *ddl;
  descrGenNtuple *dNTu;
     
  jstr = stream-1;
  ddl = mcf_GetNTuByStreamID(stream, nTupleId);
  if (ddl->reference == NULL) dNTu = ddl->descrNtu;
  else dNTu = ddl->reference->descrNtu;
  str = McfStreamPtrList[jstr];
  for(i=0, j=-1; i<str->ehead->nNTuples; i++) {
           if (str->ehead->nTupleIds[i] == ddl->seqNTuId) j = i;
   }
  if (fseeko(str->filePtr,str->ehead->ptrNTuples[j],SEEK_SET) != 0) {
        fprintf(stderr,
    " mcfio_NTupleVar: Unable to position stream at NTuple %d \n", nTupleId);
          return -1;  
      }
  str->currentPos = str->ehead->ptrNTuples[j];
  if (dNTu->multXDROffset == 0) 
      ok = xdr_mcfast_NTupleXDRPtr(str->xdr, dNTu, &ntot,
                                   ddl->seqNTuId, version);
   else ok = xdr_mcfast_NTupleSubStruct(str, dNTu, multIndex, version);
   if (ok == FALSE) {
        fprintf(stderr,
         " mcfio_NTuple: Unable to encode or decode NTuple I.D. %d \n",
             nTupleId);
         j = str->ehead->nNTuples;
         if (fseeko(str->filePtr,str->currentPos,SEEK_SET) != 0) 
           fprintf(stderr,
         " mcfio_NTuple: Unable to position stream at NTuple %d \n", nTupleId);
         return -1;
      }
    return TRUE;
        
}
