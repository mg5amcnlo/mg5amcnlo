/*******************************************************************************
*									       *
* mcf_evt_xdr.c -- XDR Utility routines for the McFast Monte-Carlo             *
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
#ifdef SUNOS
#include <floatingpoint.h>
#else /* SUNOS */
#include <float.h>
#endif /* SUNOS */
#include <stdlib.h>
#include <time.h>
#include "mcf_nTupleDescript.h"
#include "mcf_xdr.h"
#include "mcf_xdr_Ntuple.h"
#include "mcf_NTuIOFiles.h"
#include "mcf_NTuIOUtils.h"
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif


static bool_t xdr_mcfast_NTuDDL(XDR *xdrs, char *version, nTuDDL *ddl);
static bool_t xdr_mcfast_descrNTU(XDR *xdrs, char *version,
                                    descrGenNtuple *dNTu);
static bool_t xdr_mcfast_varDescrNTU(XDR *xdrs, char *version,
                                    varGenNtuple *var);

extern nTuDDL **NTuDDLList;
extern int NumOfNTuples;

bool_t xdr_mcfast_generic(XDR *xdrs, int *blockid,
 				 int *ntot, char** version, char** data)
{
/*  Translate a Generic mcfFast block. This module will allocate memory 
    for the data. */
        
    unsigned int nn;
    
    if (xdrs->x_op == XDR_ENCODE) {
      nn = strlen(*data);  
      *ntot = 12+nn;
       strcpy(*version, "0.00");
       } else if (xdrs->x_op == XDR_FREE) {
          free(*data);
          return 1;
       }
      
     if (( xdr_int(xdrs, blockid) && 
     	      xdr_int(xdrs, ntot) &&
     	      xdr_string(xdrs, version, MCF_XDR_VERSION_LENGTH)) 
     	         == FALSE) return FALSE;
     nn = *ntot - 12;	      
     if (xdrs->x_op == XDR_DECODE) *data = NULL; 
     return (xdr_string(xdrs, data, nn));     	
}   


bool_t xdr_mcfast_headerBlock(XDR *xdrs, int *blockid,
 				 int *ntot, char** version)
{
/*  Translate a Generic mcfFast block. This module will allocate memory 
    for the data. */
        
    unsigned int nn;
    
    if (xdrs->x_op == XDR_ENCODE) {
       printf ("xdr_mcfast_headerBlock: Internal error \n");
       return FALSE;
       }
      
     return ( xdr_int(xdrs, blockid) && 
     	      xdr_int(xdrs, ntot) &&
     	      xdr_string(xdrs, version, MCF_XDR_VERSION_LENGTH));
}   
bool_t xdr_mcfast_fileheader(XDR *xdrs, int *blockid,
 		 int *ntot, char** version, mcfxdrFileHeader **mcf,
 		  int streamId)
{
/*  Translate a mcf FileHeader block.  This subroutine will allocate
	the memory needed if the stream is DECODE */
        
    int i;
    unsigned int nn, oldNumOfNTuples;
    char **ctmp;
    char *atmp, *btmp, *dtmp;
    int *itmp;
    bool_t ok;
    mcfxdrFileHeader *mcftmp;
    nTuDDL *ddl;
    float fv;
    
    
    mcftmp = *mcf;
    if (xdrs->x_op == XDR_ENCODE) {
      *ntot = sizeof(mcfxdrFileHeader) - sizeof(int *) - sizeof(char **) 
              + 2 * sizeof(int) * mcftmp->nBlocks 
              - sizeof(char) * MCF_XDR_F_TITLE_LENGTH
              + sizeof(char) * strlen(mcftmp->title) + 
              + sizeof(char) * strlen(mcftmp->comment) ;
      for (i=0, ctmp = mcftmp->blockNames; 
             i< mcftmp->nBlocks; i++, ctmp++) *ntot += strlen(*ctmp);  
       strcpy(*version, "2.01");
     }  else if (xdrs->x_op == XDR_FREE) {
          mcfioC_Free_FileHeader(mcf);
          return 1;
     } else if((xdrs->x_op == XDR_DECODE) && (*mcf == NULL)) {
          mcftmp = (mcfxdrFileHeader *) malloc(sizeof(mcfxdrFileHeader));
          *mcf = mcftmp;
     } 
        

       
     if (( xdr_int(xdrs, blockid) && 
     	      xdr_int(xdrs, ntot) &&
     	      xdr_string(xdrs, version, MCF_XDR_VERSION_LENGTH))
     	                  == FALSE) return FALSE;
     
     /*
     ** Code valid for version 1.00
     */
     if (strcmp(*version, "1.00") == 0) {
         atmp = &(mcftmp->title[0]);
         btmp = &(mcftmp->comment[0]);
         dtmp = &(mcftmp->date[0]);
     	      
        if ((xdr_string(xdrs, &atmp, MCF_XDR_F_TITLE_LENGTH) &&
             xdr_string(xdrs,&btmp, MCF_XDR_F_TITLE_LENGTH) &&
             xdr_string(xdrs,&dtmp, 30)) == FALSE) return FALSE;
	
        if ((xdr_u_int(xdrs,&(mcftmp->numevts_expect)) &&
             xdr_u_int(xdrs,&(mcftmp->numevts)) &&
             xdr_u_int(xdrs,&(mcftmp->firstTable)) &&
             xdr_u_int(xdrs,&(mcftmp->dimTable)) &&
             xdr_u_int(xdrs,&(mcftmp->nBlocks))) == FALSE) return FALSE;
        if(xdrs->x_op == XDR_DECODE) {
           mcftmp->blockIds = (int *) malloc(sizeof(int) * mcftmp->nBlocks);
           mcftmp->blockNames = 
           	(char**) malloc(sizeof(char *) * mcftmp->nBlocks);
           for (i=0; i<mcftmp->nBlocks; i++) 
                mcftmp->blockNames[i] =
                  (char *) malloc(sizeof(char) * (MCF_XDR_B_TITLE_LENGTH +1));
        }
        itmp = mcftmp->blockIds;
        if (xdrs->x_op == XDR_ENCODE) nn = mcftmp->nBlocks;
	if (xdr_array(xdrs, (char **) &itmp, &nn, 
	             mcftmp->nBlocks, sizeof(int), xdr_int) == FALSE) 
	              return FALSE;
	for (i=0; i<mcftmp->nBlocks; i++) {
	       if (xdr_string(xdrs, &(mcftmp->blockNames[i]), 
	               MCF_XDR_B_TITLE_LENGTH) == FALSE) return FALSE; 
	    }	              
	 mcftmp->nNTuples = 0;  
     } else if (strncmp(*version, "2.",2) == 0){
         sscanf(*version, "%f", &fv);
     /*
     ** Code valid for version 2.xx, adding the NTuples
     */
         atmp = &(mcftmp->title[0]);
         btmp = &(mcftmp->comment[0]);
         dtmp = &(mcftmp->date[0]);
     	      
        if ((xdr_string(xdrs, &atmp, MCF_XDR_F_TITLE_LENGTH) &&
             xdr_string(xdrs,&btmp, MCF_XDR_F_TITLE_LENGTH) &&
             xdr_string(xdrs,&dtmp, 30)) == FALSE) return FALSE;
             
         if (fv == 2.) strcpy(mcftmp->closingDate, mcftmp->date);
         else {
             atmp = &(mcftmp->closingDate[0]);
            if (xdr_string(xdrs, &atmp, 30) == FALSE) return FALSE; 
     	}      
        if ((xdr_u_int(xdrs,&(mcftmp->numevts_expect)) &&
             xdr_u_int(xdrs,&(mcftmp->numevts)) &&
             xdr_u_int(xdrs,&(mcftmp->firstTable)) &&
             xdr_u_int(xdrs,&(mcftmp->dimTable)) &&
             xdr_u_int(xdrs,&(mcftmp->nBlocks)) &&
             xdr_u_int(xdrs,&(mcftmp->nNTuples))) == FALSE) return FALSE;
        if((xdrs->x_op == XDR_DECODE) && (mcftmp->nBlocks > 0)) {
           mcftmp->blockIds = (int *) malloc(sizeof(int) * mcftmp->nBlocks);
           mcftmp->blockNames = 
           	(char**) malloc(sizeof(char *) * mcftmp->nBlocks);
           for (i=0; i<mcftmp->nBlocks; i++) 
                mcftmp->blockNames[i] =
                  (char *) malloc(sizeof(char) * (MCF_XDR_B_TITLE_LENGTH +1));
        }
        itmp = mcftmp->blockIds;
        if (xdrs->x_op == XDR_ENCODE) nn = mcftmp->nBlocks;
        if (mcftmp->nBlocks > 0) {
	    if (xdr_array(xdrs, (char **) &itmp, &nn, 
	             mcftmp->nBlocks, sizeof(int), xdr_int) == FALSE) 
	              return FALSE;
	    for (i=0; i<mcftmp->nBlocks; i++) {
	          if (xdr_string(xdrs, &(mcftmp->blockNames[i]), 
	               MCF_XDR_B_TITLE_LENGTH) == FALSE) return FALSE; 
	    }
	  } else {
	   mcftmp->blockNames = NULL;
	   mcftmp->blockIds = NULL;
	}
	/*
	** Now take care of the Ntuples
	*/
        if((xdrs->x_op == XDR_DECODE) && (mcftmp->nNTuples > 0)) {
	   oldNumOfNTuples = NumOfNTuples;
           for (i=0; i<mcftmp->nNTuples; i++) {
                ddl = (nTuDDL * ) malloc(sizeof(nTuDDL));
                AddNTuDDLtoList(ddl);
                if (xdr_mcfast_NTuDDL(xdrs, *version, ddl) == FALSE) 
                                                        return FALSE;
           }
        }  else if ((xdrs->x_op == XDR_ENCODE)  && (mcftmp->nNTuples > 0)) {  
            for (i=0; i<NumOfNTuples; i++) {
                ddl =mcf_GetNTuByPtrID(i+1);
                if ((ddl->streamId == streamId) &&  
                    (xdr_mcfast_NTuDDL(xdrs, *version, ddl) == FALSE)) 
                                                        return FALSE;
           }                                             
       }                                                  
		              
     } else return FALSE; /* Other Futur version encoded here. */
     return TRUE;
     	      
}   

bool_t xdr_mcfast_eventtable(XDR *xdrs, int *blockid,
 		 int *ntot, char** version, mcfxdrEventTable **mcf)
{
/*  Translate a mcf EventTable block.  This subroutine will allocate
	the memory needed if the stream is DECODE */
        
    int i, *idat;
    unsigned int nn, nnold, *uidat;
    char **ctmp;
    mcfxdrEventTable *mcftmp;
    
    
    mcftmp = *mcf;
    if (xdrs->x_op == XDR_ENCODE) {
      *ntot = sizeof(mcfxdrEventTable) + 4 * sizeof(int)* mcftmp->dim
              + sizeof(unsigned int)* mcftmp->dim - 2 * sizeof(int)
              - 4 * sizeof(int *) - sizeof(u_int *);
       strcpy(*version, "1.00");
     }  else if (xdrs->x_op == XDR_FREE) {
          mcfioC_Free_EventTable(mcf);
          return 1;
     } else if((xdrs->x_op == XDR_DECODE) && ( mcftmp == NULL)) {
          mcftmp = (mcfxdrEventTable *) malloc(sizeof(mcfxdrEventTable));
          *mcf = mcftmp;
     } 
        

       
     if (( xdr_int(xdrs, blockid) && 
     	      xdr_int(xdrs, ntot) &&
     	      xdr_string(xdrs, version, MCF_XDR_VERSION_LENGTH)) 
     	                 == FALSE) return FALSE;
     
     /*
     ** Code valid for version 1.00
     */
     if (strcmp(*version, "1.00") == 0) {
     	      
        if((xdrs->x_op == XDR_DECODE) && (mcftmp->evtnums != NULL))
             nnold = mcftmp->previousnumevts;
          else nnold = 0;
        idat = &mcftmp->nextLocator;
        uidat = (u_int *) &mcftmp->numevts;  
        if ((xdr_int(xdrs,idat) && xdr_u_int(xdrs,uidat )) == FALSE)
                  return FALSE; 
        if(xdrs->x_op == XDR_DECODE) {
           if ((mcftmp->evtnums == NULL) || (mcftmp->numevts > nnold)) {
           if (mcftmp->evtnums != NULL) {
            /*
            ** I don't trust realloc.. just alloc again.. 
            */
            free(mcftmp->evtnums); free(mcftmp->storenums); 
            free(mcftmp->runnums); free(mcftmp->trigMasks);
            free(mcftmp->ptrEvents);
            }  
           mcftmp->evtnums = (int *) malloc(sizeof(int) * mcftmp->dim);
           mcftmp->storenums = (int *) malloc(sizeof(int) * mcftmp->dim);
           mcftmp->runnums = (int *) malloc(sizeof(int) * mcftmp->dim);
           mcftmp->trigMasks = (int *) malloc(sizeof(int) * mcftmp->dim);
           mcftmp->ptrEvents = 
            (unsigned int *) malloc(sizeof(unsigned int) * mcftmp->dim);
            mcftmp->previousnumevts = mcftmp->dim;
           }
        }
        if (xdrs->x_op == XDR_ENCODE) nn = mcftmp->dim;
        idat = mcftmp->evtnums;
	if (xdr_array(xdrs, (char **) &idat, &nn, 
	              mcftmp->dim, sizeof(int), xdr_int) == FALSE) 
	              return FALSE;
        idat = mcftmp->storenums;
	if (xdr_array(xdrs, (char **) &idat, &nn, 
	              mcftmp->dim, sizeof(int), xdr_int) == FALSE) 
	              return FALSE;
        idat = mcftmp->runnums;
	if (xdr_array(xdrs, (char **) &idat, &nn, 
	              mcftmp->dim, sizeof(int), xdr_int) == FALSE) 
	              return FALSE;
        idat = mcftmp->trigMasks;
	if (xdr_array(xdrs, (char **) &idat, &nn, 
	              mcftmp->dim, sizeof(int), xdr_int) == FALSE) 
	              return FALSE;
        uidat = mcftmp->ptrEvents;
	if (xdr_array(xdrs, (char **) &uidat, &nn, 
	              mcftmp->dim, sizeof(int), xdr_u_int) == FALSE) 
	              return FALSE;
     } else return FALSE; /* Future version encoded here. */
     return TRUE;
     	      
}
   
bool_t xdr_mcfast_seqheader(XDR *xdrs, int *blockid,
 		 int *ntot, char** version, mcfxdrSequentialHeader **mcf)
{
/*  Translate a mcf EventTable block.  This subroutine will allocate
	the memory needed if the stream is DECODE */
        
    int i;
    unsigned int nn;
    char **ctmp;
    mcfxdrSequentialHeader *mcftmp;
    
    
    if (xdrs->x_op == XDR_ENCODE) {
      mcftmp = *mcf;
      *ntot = sizeof(mcfxdrSequentialHeader);
       strcpy(*version, "1.00");
     }  else if (xdrs->x_op == XDR_FREE) {
          mcfioC_Free_SeqHeader(mcf);
          return 1;
     } else if(xdrs->x_op == XDR_DECODE) {
          if (*mcf == NULL) {
              mcftmp = (mcfxdrSequentialHeader *) 
                        malloc(sizeof(mcfxdrSequentialHeader));
              *mcf = mcftmp;
          } else mcftmp = *mcf;
          
     } 
        

       
/*     if (( xdr_int(xdrs, blockid) && 
     	      xdr_int(xdrs, ntot) &&
     	      xdr_string(xdrs, version, MCF_XDR_VERSION_LENGTH)) 
     	                 == FALSE) return FALSE;
*/
      if (xdr_int(xdrs,blockid) == FALSE) return FALSE;
      if (xdr_int(xdrs,ntot) == FALSE) return FALSE; 
      if (xdr_string(xdrs, version, MCF_XDR_VERSION_LENGTH) 
     	                 == FALSE) return FALSE;    
     /*
     ** Code valid for version 1.00
     */
     if (strcmp(*version, "1.00") == 0) {
     	      
        if (xdr_u_int(xdrs,&(mcftmp->nRecords)) == FALSE) return FALSE; 
     } else return FALSE; /* Futur version encoded here. */
     return TRUE;
     	      
}

bool_t xdr_mcfast_eventheader(XDR *xdrs, int *blockid,
 		 int *ntot, char** version, mcfxdrEventHeader **mcf)
{
/*  Translate a mcf Event header block.  This subroutine will allocate
	the memory needed if the stream is DECODE */
        
    int i, *itmp;
    unsigned int nn, nnold, nNTuOld, *uitmp;
    char **ctmp;
    mcfxdrEventHeader *mcftmp;
    
    
    mcftmp = *mcf;
    if (xdrs->x_op == XDR_ENCODE) {
      *ntot = sizeof(mcfxdrEventHeader)
              + sizeof(unsigned int)* mcftmp->nBlocks
              + sizeof(int ) * mcftmp->nBlocks 
              - sizeof(int *)  - sizeof(u_int *) ;
       strcpy(*version, "2.00");
     }  else if (xdrs->x_op == XDR_FREE) {
          mcfioC_Free_EventHeader(mcf);
          return 1;
     } else if((xdrs->x_op == XDR_DECODE) && (mcftmp == NULL)) {
          mcftmp =
           (mcfxdrEventHeader *) malloc(sizeof(mcfxdrEventHeader));
          *mcf = mcftmp;
          mcftmp->blockIds = NULL;
          mcftmp->ptrBlocks = NULL;
     } 
        

       
     if (( xdr_int(xdrs, blockid) && 
     	      xdr_int(xdrs, ntot) &&
     	      xdr_string(xdrs, version, MCF_XDR_VERSION_LENGTH)) 
     	                  == FALSE) return FALSE;
     
     /*
     ** Code valid for version 1.00
     */
     if (strcmp(*version, "1.00") == 0) {
        if((xdrs->x_op == XDR_DECODE) && (mcftmp->blockIds != NULL))
             nnold = mcftmp->dimBlocks;  
     	else nnold = 0;      
        if ((xdr_int(xdrs,&(mcftmp->evtnum)) &&
             xdr_int(xdrs,&(mcftmp->storenum)) &&
             xdr_int(xdrs,&(mcftmp->runnum)) &&
             xdr_int(xdrs,&(mcftmp->trigMask)) &&
             xdr_u_int(xdrs,&(mcftmp->nBlocks)) &&
             xdr_u_int(xdrs,&(mcftmp->dimBlocks))) == FALSE) return FALSE; 
        if(xdrs->x_op == XDR_DECODE) {
           if ((mcftmp->blockIds == NULL) || (mcftmp->dimBlocks > nnold)) {
           if (mcftmp->blockIds != NULL) {
            /*
            ** I don't trust realloc.. just alloc again.. 
            */
            free(mcftmp->blockIds); free(mcftmp->ptrBlocks); 
            }  
           mcftmp->blockIds =
             (int *) malloc(sizeof(unsigned int) * mcftmp->dimBlocks);
           mcftmp->ptrBlocks =
             (unsigned int *) malloc(sizeof(unsigned int) * mcftmp->dimBlocks);
           }
        }
        if (xdrs->x_op == XDR_ENCODE)  nn = mcftmp->dimBlocks;
        itmp = mcftmp->blockIds;
	if (xdr_array(xdrs, (char **) &itmp, &nn, 
	              mcftmp->dimBlocks, sizeof(int), xdr_int) == FALSE) 
	              return FALSE;
	uitmp = mcftmp->ptrBlocks;              
	if (xdr_array(xdrs, (char **) &uitmp, &nn, 
	              mcftmp->dimBlocks, sizeof(u_int), xdr_u_int) == FALSE) 
	              return FALSE;
     } else if (strcmp(*version, "2.00") == 0) {
        if (xdrs->x_op == XDR_DECODE) {
           nnold = 0;
           if (mcftmp->blockIds != NULL)  nnold = mcftmp->dimBlocks;
           nNTuOld = 0;
           if (mcftmp->nTupleIds != NULL)  nNTuOld = mcftmp->dimNTuples;
        }  
        if ((xdr_int(xdrs,&(mcftmp->evtnum)) &&
             xdr_int(xdrs,&(mcftmp->storenum)) &&
             xdr_int(xdrs,&(mcftmp->runnum)) &&
             xdr_int(xdrs,&(mcftmp->trigMask)) &&
             xdr_u_int(xdrs,&(mcftmp->nBlocks)) &&
             xdr_u_int(xdrs,&(mcftmp->dimBlocks)) &&
             xdr_u_int(xdrs,&(mcftmp->nNTuples)) &&
             xdr_u_int(xdrs,&(mcftmp->dimNTuples))) == FALSE) return FALSE;
        if(xdrs->x_op == XDR_DECODE) {
           if ((mcftmp->blockIds == NULL) || (mcftmp->dimBlocks > nnold)) {
           if (mcftmp->blockIds != NULL) { 
               free(mcftmp->blockIds);
               free(mcftmp->ptrBlocks);
           }     
           mcftmp->blockIds =
             (int *) malloc(sizeof(unsigned int) * mcftmp->dimBlocks);
           mcftmp->ptrBlocks =
             (unsigned int *) malloc(sizeof(unsigned int) * mcftmp->dimBlocks);
           }
           if ((mcftmp->nTupleIds == NULL) || (mcftmp->dimNTuples > nNTuOld)) {
           if (mcftmp->nTupleIds != NULL) { 
               free(mcftmp->nTupleIds);
               free(mcftmp->ptrNTuples);
           }     
           mcftmp->nTupleIds =
             (int *) malloc(sizeof(unsigned int) * mcftmp->dimNTuples);
           mcftmp->ptrNTuples =
             (unsigned int *) malloc(sizeof(unsigned int) * mcftmp->dimNTuples);
           }
        }
        if (mcftmp->dimBlocks > 0) {
            if (xdrs->x_op == XDR_ENCODE)  nn = mcftmp->dimBlocks;
            itmp = mcftmp->blockIds;
	    if (xdr_array(xdrs, (char **) &itmp, &nn, 
	              mcftmp->dimBlocks, sizeof(int), xdr_int) == FALSE) 
	              return FALSE;
	    uitmp = mcftmp->ptrBlocks;              
	    if (xdr_array(xdrs, (char **) &uitmp, &nn, 
	              mcftmp->dimBlocks, sizeof(u_int), xdr_u_int) == FALSE) 
	              return FALSE;
        }
        if (mcftmp->dimNTuples > 0) {
            if (xdrs->x_op == XDR_ENCODE)  nn = mcftmp->dimNTuples;
            itmp = mcftmp->nTupleIds;
	    if (xdr_array(xdrs, (char **) &itmp, &nn, 
	              mcftmp->dimNTuples, sizeof(int), xdr_int) == FALSE) 
	              return FALSE;
	    uitmp = mcftmp->ptrNTuples;              
	    if (xdr_array(xdrs, (char **) &uitmp, &nn, 
	              mcftmp->dimNTuples, sizeof(u_int), xdr_u_int) == FALSE) 
	              return FALSE;
	}              
     } else 
      return FALSE; /* Futur version encoded here. */
     return TRUE;
     	      
}

static bool_t xdr_mcfast_NTuDDL(XDR *xdrs, char *version, nTuDDL *ddl)
{
    int i, nc_title, nc_category, idRef;
    descrGenNtuple *dNTu;
    
    
    /*
    ** This is the first version, let us not get too compilcated..
    */
    if (xdrs->x_op == XDR_ENCODE) {
           nc_title = strlen(ddl->title);
           nc_category = strlen(ddl->category);
           idRef = -1;
           /*
           ** Cross reference is only valid within the same stream.
           */
           if ((ddl->reference != NULL) && 
               (ddl->streamId == ddl->reference->streamId )) { 
               /*
               ** compute the rerefence token. This is the sequential 
               ** number of the reference Ntuple for this stream.
               */
               for (i=0, idRef=0; i<NumOfNTuples; i++) { 
                   if (NTuDDLList[i]->streamId == ddl->reference->streamId)
                       idRef++;
                   if (NTuDDLList[i]->id == ddl->reference->id) break;
               }    
          }
    }      
    if (xdr_int(xdrs, &nc_title) == FALSE) return FALSE;
    if (xdr_int(xdrs, &nc_category) == FALSE) return FALSE;
    if (xdr_int(xdrs, &idRef) == FALSE) return FALSE;
    if (xdrs->x_op == XDR_DECODE) {
       ddl->title = (char *) malloc(sizeof(char) * (nc_title +1));
       ddl->category = (char *) malloc(sizeof(char) * (nc_category +1));
       ddl->dbinFileName = NULL;
       ddl->streamId = -1;
    }   
    if (xdr_int(xdrs,&(ddl->uid)) == FALSE) return FALSE;
    if (xdr_string(xdrs, &(ddl->title), nc_title) == FALSE) return FALSE;
    if (xdr_string(xdrs, &(ddl->category), 
                      nc_category) == FALSE) return FALSE;
    if (idRef == -1) {                  
        if (xdrs->x_op == XDR_DECODE) 
           ddl->descrNtu = (descrGenNtuple *) malloc (sizeof(descrGenNtuple));
         if (ddl->descrNtu == NULL) dNTu = ddl->reference->descrNtu;
           else dNTu = ddl->descrNtu; 
        if (xdr_mcfast_descrNTU(xdrs, version, dNTu) == FALSE) 
            return FALSE;
        if (xdrs->x_op == XDR_DECODE) ddl->reference = NULL; 
    } else {
        if (xdrs->x_op == XDR_DECODE) {
              ddl->descrNtu = NULL;
              ddl->referenceId = idRef;
              /* we will set the reference pointer in mcfio_Direct */
        }
    }        
    return TRUE; 
    	      
}

static bool_t xdr_mcfast_descrNTU(XDR *xdrs, char *version,
                                    descrGenNtuple *dNTu)
{
    int i, nc_desc, nc_title;
    u_int nn;
    char *tc;
    /*
    ** This is the first version, let us not get too compilcated..
    */
    
    if (xdr_int(xdrs,&(dNTu->numVariables)) == FALSE) return FALSE;
    dNTu->numAvailable = dNTu->numVariables;
    if (xdr_int(xdrs,&(dNTu->maxMultiplicity)) == FALSE) return FALSE;
    if (xdr_int(xdrs,&(dNTu->orgStyle)) == FALSE)return FALSE;
    if (xdr_int(xdrs,&(dNTu->firstIndexed)) == FALSE) return FALSE;
    if (xdrs->x_op == XDR_ENCODE)  nc_title = strlen(dNTu->title);
    if (xdr_int(xdrs, &nc_title) == FALSE) return FALSE;
    if (xdrs->x_op == XDR_ENCODE)  nc_desc = strlen(dNTu->description);
    if (xdr_int(xdrs, &nc_desc) == FALSE) return FALSE;
    if (xdrs->x_op == XDR_DECODE) {
        dNTu->title = (char *) malloc(sizeof(char) * (nc_title+1));
        dNTu->subXDROffset = NULL;
        dNTu->description = (char *) malloc(sizeof(char) * (nc_desc+1));
        dNTu->varOrdering = (int *) malloc(sizeof(int) * dNTu->numVariables);
        for (i=0; i<dNTu->numVariables; i++) dNTu->varOrdering[i] = i;
        if (dNTu->orgStyle == PARALLEL_ARRAY_NTU) {
           dNTu->subXDROffset = NULL;
           dNTu->subOffset = NULL;  
        } else {
           dNTu->subOffset = 
          (long *) malloc(sizeof(long) * dNTu->maxMultiplicity);
           dNTu->subXDROffset = 
          (u_int *) malloc(sizeof(long) * dNTu->maxMultiplicity);
        }  
        dNTu->variables =
        (varGenNtuple **) malloc(sizeof(varGenNtuple *) * dNTu->numVariables);  
        for (i=0; i<dNTu->numVariables; i++)
            dNTu->variables[i] = (varGenNtuple *) malloc(sizeof(varGenNtuple));
     }
     tc = dNTu->nameIndex;
     if (xdr_string(xdrs, &tc, 31) == FALSE) return FALSE;
     if (xdr_string(xdrs, 
        (char **) &(dNTu->title), nc_title) == FALSE) return FALSE;
     if (xdr_string(xdrs,
           &(dNTu->description), nc_desc) == FALSE) return FALSE;
     tc =  dNTu->version;     
     if (xdr_string(xdrs,  &tc, 7) == FALSE) return FALSE;
     if (xdr_long(xdrs,  &(dNTu->multOffset)) == FALSE) return FALSE;
     if (xdr_long(xdrs,  &(dNTu->fenceOffset)) == FALSE) return FALSE;
     nn = dNTu->maxMultiplicity;
     if (dNTu->orgStyle != PARALLEL_ARRAY_NTU) { 
        if (xdr_array(xdrs, 
      (char **) &(dNTu->subOffset), &nn, nn, sizeof(long), xdr_long) == FALSE) 
           return FALSE;
     }      
     for (i=0; i<dNTu->numVariables; i++) 
	if (xdr_mcfast_varDescrNTU(xdrs, version, dNTu->variables[i]) == FALSE)
	        return FALSE;
     return TRUE;
}
static bool_t xdr_mcfast_varDescrNTU(XDR *xdrs, char *version,
                                    varGenNtuple *var)
{
    int i, nc_name, nc_desc, *pdim;
    u_int nn;
    
    
    
    if (xdrs->x_op == XDR_ENCODE)  nc_name = strlen(var->name);
    if (xdr_int(xdrs, &nc_name) == FALSE) return FALSE;
    if (xdrs->x_op == XDR_ENCODE) {
         if (var->description == NULL) nc_desc = 0;
             else nc_desc = strlen(var->description);
    }     
    if (xdr_int(xdrs, &nc_desc) == FALSE) return FALSE;
    if (xdrs->x_op == XDR_DECODE) {
        var->name = (char *) malloc(sizeof(char) * (nc_name+1));
        if (nc_desc>0) 
           var->description = (char *) malloc(sizeof(char) * (nc_desc+1));
        else    var->description = NULL;
        var->nameBlank = FALSE;
     }
  
     if (xdr_string(xdrs, &(var->name), nc_name) == FALSE) return FALSE;
     if (nc_desc > 0) 
        if (xdr_string(xdrs, &(var->description), nc_desc) == FALSE) 
             return FALSE;
     if (xdr_int(xdrs,&(var->type)) == FALSE) return FALSE;
     if (xdr_char(xdrs,&(var->isFixedSize)) == FALSE) return FALSE;
     if (xdr_int(xdrs,&(var->numDim)) == FALSE) return FALSE;
     nn = var->numDim;
     pdim = var->dimensions;
     if ((nn > 0) && (xdr_array(xdrs, 
        (char **) &pdim, &nn, nn, sizeof(int), xdr_int)) == FALSE) 
           return FALSE;
     if (xdrs->x_op == XDR_ENCODE) nn = (u_int) var->lengthB;  
     if (xdr_u_int(xdrs,&(nn)) == FALSE) return FALSE;
     if (xdrs->x_op == XDR_DECODE) var->lengthB = (size_t) nn;
     if (xdrs->x_op == XDR_ENCODE) nn = (u_int) var->lengthW;  
     if (xdr_u_int(xdrs,&(nn)) == FALSE) return FALSE;
     if (xdrs->x_op == XDR_DECODE) var->lengthW = (size_t) nn;  
     if (xdr_long(xdrs,&(var->offset)) == FALSE) return FALSE;
     return TRUE;
}
/*
** Generalized NTuple XDR filter
*/
bool_t xdr_mcfast_NTuple(XDR *xdrs, descrGenNtuple *dNTu,
 		 int *pnTot, int nTupleId, char* version)
{
    int i, j, id, nm, lastFixed;
    u_int nn;
    char *vv, *cDat, *start;
    int *pnMult;
    void *pnFence;
    int *ipnFence;
    void *end, *pt;
    bool_t ok;
/*
** Upon write, check that the version token is identical to the one stored 
** in the ddl. 
*/
     start = version;
     if(dNTu->firstIndexed == -1) lastFixed = dNTu->numVariables;
        else lastFixed = dNTu->firstIndexed;
     if ((xdrs->x_op == XDR_ENCODE) || (xdrs->x_op == XDR_MCFIOCODE)) {
         nn = strlen(dNTu->version);
         if (strncmp(version, dNTu->version, (size_t) nn ) != 0) {
              fprintf (stderr, "mcfio_NTuple: version mismatch! \n\
          Version used in the Event loop = %s\n\
                  ...  in the DDl template = %s\n", version,dNTu->version);
              return FALSE;
         }
         id = nTupleId;
/*
**   Compute the total length 
*/
         cDat = start; cDat +=  dNTu->multOffset;
         pnMult = (int *) cDat;
         nm = *pnMult;
         for (i=0, nn=0; i<lastFixed; i++)
               nn += dNTu->variables[i]->lengthB;
         if(dNTu->firstIndexed != -1) 
             for(i=dNTu->firstIndexed; i<dNTu->numVariables; i++) 
                nn += (dNTu->variables[i]->lengthB * nm);
         *pnTot = 6 + nn/4;      
     }
     if (xdr_int(xdrs, &id) == FALSE) return FALSE;
     if (xdr_int(xdrs, pnTot) == FALSE) return FALSE;
     if (xdrs->x_op == XDR_ENCODE) {
         vv = dNTu->version;
         if (xdr_string(xdrs, &vv, 11) == FALSE) return FALSE;
     } else  if (xdrs->x_op == XDR_DECODE) {
         if (xdr_string(xdrs, &version, 11) == FALSE) return FALSE;
         if (strcmp(version, dNTu->version) != 0) {
              fprintf (stderr, "mcfio_NTuple: version mismatch! \n\
          Version used in the Event loop = %s\n\
                  ...  in the DDl template = %s\n", version,dNTu->version);
              return FALSE;
         }
         if (id != nTupleId) {
              fprintf (stderr, "mcfio_NTuple: internal error! \n\
          Unexpected NTuple identifier % instead of %d\n", id, nTupleId);
          return FALSE;
          }
     }
         
     cDat = start; cDat +=  dNTu->multOffset;
     pnMult = (int *) cDat;
     if (xdr_int(xdrs, pnMult) == FALSE) return FALSE;
     /*
     ** Close the fence now, we will check it upon DECODE at the end
     */
     cDat = start; cDat +=  dNTu->fenceOffset;
     pnFence = (void *) cDat;
     if (xdrs->x_op == XDR_ENCODE) memcpy(pnFence, pnTot, sizeof(int));
     if (xdr_int(xdrs, (int *) pnFence) == FALSE) return FALSE;
     nm = *pnMult;
     for (i=0; i<lastFixed; i++) {
        if (dNTu->variables[i]->lengthW == 1) {
           cDat = start; cDat += dNTu->variables[i]->offset;
           pt = (void *) cDat;
           switch (dNTu->variables[i]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      ok = xdr_char(xdrs, (char *) pt);
                      break;
                   case INTEGER2_NTU:
                      ok = xdr_short(xdrs, (short *) pt);
                      break;
                   case LOGICAL_NTU: case INTEGER_NTU:
                      ok = xdr_int(xdrs, (int *) pt);
                      break;
                   case REAL_NTU:
                      ok = xdr_float(xdrs, (float *) pt); 
                      break;
                   case DBL_PRECISION_NTU:
                      ok = xdr_double(xdrs, (double *) pt);
                      break;
                   case COMPLEX_NTU:
                      nn =2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_COMPLEX_NTU:
                      nn =2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case POINTER_NTU:
                      ok = xdr_long(xdrs, (long *) pt);
                      break;
                   default :
                       fprintf (stderr, "mcfio_NTuple: internal error! \n\
          Unexpected variables type %d on NTuple \n", 
                       dNTu->variables[i]->type, nTupleId);
                       break;
              }
        }      
        else if (dNTu->variables[i]->lengthW > 0) {
           cDat = start; cDat +=  dNTu->variables[i]->offset;
           pt = (void *) cDat;
           nn = dNTu->variables[i]->lengthW;
           switch (dNTu->variables[i]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      ok = xdr_bytes(xdrs, (char **) &pt, &nn, nn);
                      break;
                   case INTEGER2_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(short), xdr_short);
                      break;
                   case LOGICAL_NTU: case INTEGER_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(int), xdr_int);
                      break;
                   case REAL_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_PRECISION_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case COMPLEX_NTU:
                      nn = nn*2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_COMPLEX_NTU:
                      nn = nn*2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case POINTER_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(long), xdr_long);
                      break;
                   default :
                       fprintf (stderr, "mcfio_NTuple: internal error! \n\
          Unexpected variables type %d on NTuple \n", 
                       dNTu->variables[i]->type, nTupleId);
                       break;
              }
          if (ok == FALSE) return FALSE;    
         }
      }
      if (dNTu->firstIndexed != -1) {
         if (dNTu->orgStyle == PARALLEL_ARRAY_NTU) {
          for (i=dNTu->firstIndexed; i<dNTu->numVariables; i++) {
                 cDat = start; cDat +=  dNTu->variables[i]->offset;
                 pt = (void *) cDat;
                 nn = nm * dNTu->variables[i]->lengthW;
                 switch (dNTu->variables[i]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      vv = (char *) pt;
                      ok = xdr_bytes(xdrs, (char **) &pt, &nn, nn);
                      break;
                   case INTEGER2_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(short), xdr_short);
                      break;
                   case LOGICAL_NTU: case INTEGER_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(int), xdr_int);
                      break;
                   case REAL_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_PRECISION_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case COMPLEX_NTU:
                      nn = nn*2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_COMPLEX_NTU:
                      nn = nn*2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case POINTER_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(long), xdr_long);
                      break;
                   default :
                       fprintf (stderr, "mcfio_NTuple: internal error! \n\
          Unexpected variables type %d on NTuple \n", 
                       dNTu->variables[i]->type, nTupleId);
                       break;
              }
          if (ok == FALSE) return FALSE;    
        }
     } else { /*dump the substructures one a time */
     for (j=0; j<nm; j++) {
       for (i=dNTu->firstIndexed; i<dNTu->numVariables; i++) {
        cDat = start; 
        cDat += (dNTu->subOffset[j] + dNTu->variables[i]->offset);
        pt = (void *) cDat;
        if (dNTu->variables[i]->lengthW == 1) {
           switch (dNTu->variables[i]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      ok = xdr_char(xdrs, (char *) pt);
                      break;
                   case INTEGER2_NTU:
                      ok = xdr_short(xdrs, (short *) pt);
                      break;
                   case LOGICAL_NTU: case INTEGER_NTU:
                      ok = xdr_int(xdrs, (int *) pt);
                      break;
                   case REAL_NTU:
                      ok = xdr_float(xdrs, (float *) pt); 
                      break;
                   case DBL_PRECISION_NTU:
                      ok = xdr_double(xdrs, (double *) pt);
                      break;
                   case COMPLEX_NTU:
                      nn =2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_COMPLEX_NTU:
                      nn =2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case POINTER_NTU:
                      ok = xdr_long(xdrs, (long *) pt);
                      break;
                   default :
                       fprintf (stderr, "mcfio_NTuple: internal error! \n\
          Unexpected variables type %d on NTuple \n", 
                       dNTu->variables[i]->type, nTupleId);
                       break;
              }
        }      
        else if (dNTu->variables[i]->lengthW > 0) {
           nn = dNTu->variables[i]->lengthW;
           switch (dNTu->variables[i]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      ok = xdr_bytes(xdrs, (char **) &pt, &nn, nn);
                      break;
                   case INTEGER2_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(short), xdr_short);
                      break;
                   case LOGICAL_NTU: case INTEGER_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(int), xdr_int);
                      break;
                   case REAL_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_PRECISION_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case COMPLEX_NTU:
                      nn = nn*2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_COMPLEX_NTU:
                      nn = nn*2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case POINTER_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(long), xdr_long);
                      break;
                   default :
                       fprintf (stderr, "mcfio_NTuple: internal error! \n\
          Unexpected variables type %d on NTuple \n", 
                       dNTu->variables[i]->type, nTupleId);
                       break;
              }
          if (ok == FALSE) return FALSE;    
         }
        } /*end of i loop */
       } /*end of j loop */
      } /* End of orgStyle clause */
      } /* End of firstIndexed clause */
      /*
      ** Check the fence.. 
      */
      ipnFence = (int *) pnFence;
      if ((xdrs->x_op == XDR_DECODE) && (*ipnFence != *pnTot)) {
              fprintf (stderr, "mcfio_NTuple: Suspected Data Overwrite! \n\
          Fence content found on the input stream is = %d\n\
                  ...  while we expect %d\n", *ipnFence, *pnTot);
              return FALSE;
      }
      return TRUE;
}

/*
** Generalized NTuple XDR filter, for DECODE only, used exclusively 
** to establish the relative XDR pointers.
*/
bool_t xdr_mcfast_NTupleXDRPtr(XDR *xdrs, descrGenNtuple *dNTu,
 		 int *pnTot, int nTupleId, char* version)
{
    int i, j, id, nm, lastFixed;
    u_int nn, startXDR;
    char *vv, *cDat;
    int *pnMult, *pnFence;
    void *start, *end, *pt;
    bool_t ok;
    
    /*
    ** Allocate memory for supointer array if need be.
    */
     if(dNTu->firstIndexed == -1) lastFixed = dNTu->numVariables;
        else lastFixed = dNTu->firstIndexed;
        
     if (dNTu->subXDROffset != NULL) free(dNTu->subXDROffset);
     dNTu->subXDROffset = 
          (u_int *) malloc (sizeof(u_int) * dNTu->maxMultiplicity);
     start = (void *) version;
     startXDR = xdr_getpos(xdrs);
     if (xdr_int(xdrs, &id) == FALSE) return FALSE;
      if (xdr_int(xdrs, pnTot) == FALSE) return FALSE;

      if (xdr_string(xdrs, &version, 11) == FALSE) return FALSE;
      if (id != nTupleId) {
              fprintf (stderr, "mcfio_NTuple: internal error! \n\
          Unexpected NTuple identifier % instead of %d\n", id, nTupleId);
          return FALSE;
      }
     cDat = start; cDat +=  dNTu->multOffset;
     pnMult = (int *) cDat;
     dNTu->multXDROffset = xdr_getpos(xdrs) - startXDR;
     if (xdr_int(xdrs, pnMult) == FALSE) return FALSE;
     /*
     ** Close the fence now, we will check it upon DECODE at the end
     */
     cDat = start; cDat += dNTu->fenceOffset;
     pnFence = (int *) cDat;
     dNTu->fenceXDROffset = xdr_getpos(xdrs) - startXDR;
     if (xdr_int(xdrs, (int *) pnFence) == FALSE) return FALSE;
     nm = *pnMult;
     for (i=0; i<lastFixed; i++) {
        dNTu->variables[i]->offsetXDR = 0;
        if (dNTu->variables[i]->lengthW == 1) {
           cDat = start; cDat +=  dNTu->variables[i]->offset;
           pt = (void *) cDat;
          dNTu->variables[i]->offsetXDR = xdr_getpos(xdrs) - startXDR;
           switch (dNTu->variables[i]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      ok = xdr_char(xdrs, (char *) pt);
                      break;
                   case INTEGER2_NTU:
                      ok = xdr_short(xdrs, (short *) pt);
                      break;
                   case LOGICAL_NTU: case INTEGER_NTU:
                      ok = xdr_int(xdrs, (int *) pt);
                      break;
                   case REAL_NTU:
                      ok = xdr_float(xdrs, (float *) pt); 
                      break;
                   case DBL_PRECISION_NTU:
                      ok = xdr_double(xdrs, (double *) pt);
                      break;
                   case COMPLEX_NTU:
                      nn =2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_COMPLEX_NTU:
                      nn =2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case POINTER_NTU:
                      ok = xdr_long(xdrs, (long *) pt);
                      break;
                   default :
                       fprintf (stderr, "mcfio_NTuple: internal error! \n\
          Unexpected variables type %d on NTuple \n", 
                       dNTu->variables[i]->type, nTupleId);
                       break;
              }
        }      
        else if (dNTu->variables[i]->lengthW > 0) {
           cDat = start;  cDat += dNTu->variables[i]->offset;
           pt = (void *) cDat;
           nn = dNTu->variables[i]->lengthW;
           dNTu->variables[i]->offsetXDR = xdr_getpos(xdrs) - startXDR;
           switch (dNTu->variables[i]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      ok = xdr_bytes(xdrs, (char **) &pt, &nn, nn);
                      break;
                   case INTEGER2_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(short), xdr_short);
                      break;
                   case LOGICAL_NTU: case INTEGER_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(int), xdr_int);
                      break;
                   case REAL_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_PRECISION_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case COMPLEX_NTU:
                      nn = nn*2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_COMPLEX_NTU:
                      nn = nn*2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case POINTER_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(long), xdr_long);
                      break;
                   default :
                       fprintf (stderr, "mcfio_NTuple: internal error! \n\
          Unexpected variables type %d on NTuple \n", 
                       dNTu->variables[i]->type, nTupleId);
                       break;
              }
          if (ok == FALSE) return FALSE;    
         }
      }
      if (dNTu->firstIndexed != -1) {
      if (dNTu->orgStyle == PARALLEL_ARRAY_NTU) {
          for (i=dNTu->firstIndexed; i<dNTu->numVariables; i++) {
                 cDat =start;  cDat += dNTu->variables[i]->offset;
                 pt = (void *) cDat;
                 nn = nm * dNTu->variables[i]->lengthW;
                 dNTu->variables[i]->offsetXDR = xdr_getpos(xdrs) - startXDR;
                 switch (dNTu->variables[i]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      vv = (char *) pt;
                      ok = xdr_bytes(xdrs, (char **) &pt, &nn, nn);
                      break;
                   case INTEGER2_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(short), xdr_short);
                      break;
                   case LOGICAL_NTU: case INTEGER_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(int), xdr_int);
                      break;
                   case REAL_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_PRECISION_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case COMPLEX_NTU:
                      nn = nn*2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_COMPLEX_NTU:
                      nn = nn*2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case POINTER_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(long), xdr_long);
                      break;
                   default :
                       fprintf (stderr, "mcfio_NTuple: internal error! \n\
          Unexpected variables type %d on NTuple \n", 
                       dNTu->variables[i]->type, nTupleId);
                       break;
              }
          if (ok == FALSE) return FALSE;    
        }
     } else { /*dump the substructure one a time */
     for (j=0; j<nm; j++) {
       dNTu->subXDROffset[j] = xdr_getpos(xdrs) - startXDR;
       for (i=dNTu->firstIndexed; i<dNTu->numVariables; i++) {
        cDat = start; 
        cDat += (dNTu->subOffset[j] + dNTu->variables[i]->offset);
        pt = (void *) cDat;
        if (j == 0) dNTu->variables[i]->offsetXDR = 0;
        if (dNTu->variables[i]->lengthW == 1) {
          if (j == 0)  dNTu->variables[i]->offsetXDR =
                 xdr_getpos(xdrs) - startXDR- dNTu->subXDROffset[j];
           switch (dNTu->variables[i]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      ok = xdr_char(xdrs, (char *) pt);
                      break;
                   case INTEGER2_NTU:
                      ok = xdr_short(xdrs, (short *) pt);
                      break;
                   case LOGICAL_NTU: case INTEGER_NTU:
                      ok = xdr_int(xdrs, (int *) pt);
                      break;
                   case REAL_NTU:
                      ok = xdr_float(xdrs, (float *) pt); 
                      break;
                   case DBL_PRECISION_NTU:
                      ok = xdr_double(xdrs, (double *) pt);
                      break;
                   case COMPLEX_NTU:
                      nn =2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_COMPLEX_NTU:
                      nn =2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case POINTER_NTU:
                      ok = xdr_long(xdrs, (long *) pt);
                      break;
                   default :
                       fprintf (stderr, "mcfio_NTuple: internal error! \n\
          Unexpected variables type %d on NTuple \n", 
                       dNTu->variables[i]->type, nTupleId);
                       break;
              }
        }      
        else if (dNTu->variables[i]->lengthW > 0) {
           nn = dNTu->variables[i]->lengthW;
           if (j == 0) dNTu->variables[i]->offsetXDR =
                 xdr_getpos(xdrs) - startXDR - dNTu->subXDROffset[0];
           switch (dNTu->variables[i]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      ok = xdr_bytes(xdrs, (char **) &pt, &nn, nn);
                      break;
                   case INTEGER2_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(short), xdr_short);
                      break;
                   case LOGICAL_NTU: case INTEGER_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(int), xdr_int);
                      break;
                   case REAL_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_PRECISION_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case COMPLEX_NTU:
                      nn = nn*2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_COMPLEX_NTU:
                      nn = nn*2;
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case POINTER_NTU:
                      ok = xdr_array(xdrs, 
                         (char **) &pt, &nn, nn, sizeof(long), xdr_long);
                      break;
                   default :
                       fprintf (stderr, "mcfio_NTuple: internal error! \n\
          Unexpected variables type %d on NTuple \n", 
                       dNTu->variables[i]->type, nTupleId);
                       break;
              }
          if (ok == FALSE) return FALSE;    
         }
        } /*end of i loop */
       } /*end of j loop */
      } /* End of orgStyle clause */
      } /* End of firstIndexed clause */
      /*
      ** Check the fence.. 
      */
      if (*pnFence != *pnTot) {
              fprintf (stderr, "mcfio_NTuple: Suspected Data Overwrite! \n\
          Fence content found on the input stream is = %d\n\
                  ...  while we expect %d\n", *pnFence, *pnTot);
              return FALSE;
      }
      return TRUE;
}
/*
** Generalized NTuple XDR filter, used for Decode only. 
** Simply decode the multiplicty value. No checks whatsoever!
*/
bool_t xdr_mcfast_NTupleMult(mcfStream *str, descrGenNtuple *dNTu,
 		char* version)
{
    char *cDat;
    
     cDat = version;
     cDat +=  dNTu->multOffset;     
     xdr_setpos(str->xdr, (str->currentPos + dNTu->multXDROffset) );
     return  (xdr_int(str->xdr, ((int *) cDat)));
}

/*
** Generalized NTuple XDR filter, used for Decode only. 
** Simply decode one variable (scalar) or array value. No checks whatsoever!
** Not applicable if the structure organization style is VAX FORTRAN d/s 
** and the index corresponds to an indexed variable.
*/
bool_t xdr_mcfast_NTupleVar(mcfStream *str, descrGenNtuple *dNTu,
 		int ivar, char* version)
{
    char *cDat;
    u_int nn;
    void *pt;
    int ivarP;
    
     ivarP = ivar;
     while (dNTu->variables[ivarP]->lengthW == 0) ivarP--;
     cDat = version;
     cDat += dNTu->variables[ivarP]->offset;
     pt = (void *) cDat;     
     xdr_setpos(str->xdr, 
              (str->currentPos + dNTu->variables[ivarP]->offsetXDR));
     if ((dNTu->variables[ivarP]->lengthW == 1) &&
         (ivarP < dNTu->firstIndexed)) {
          switch (dNTu->variables[ivarP]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      return  xdr_char(str->xdr, (char *) pt);
                   case INTEGER2_NTU:
                      return  xdr_short(str->xdr, (short *) pt);
                   case LOGICAL_NTU: case INTEGER_NTU:
                      return  xdr_int(str->xdr, (int *) pt);
                   case REAL_NTU:
                      return  xdr_float(str->xdr, (float *) pt); 
                   case DBL_PRECISION_NTU:
                      return  xdr_double(str->xdr, (double *) pt);
                   case COMPLEX_NTU:
                      nn =2;
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                   case DBL_COMPLEX_NTU:
                      nn =2;
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                   case POINTER_NTU:
                      return  xdr_long(str->xdr, (long *) pt);
                   default :
                      return FALSE;
              }
        } else {
           nn = dNTu->variables[ivarP]->lengthW;
           switch (dNTu->variables[ivarP]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      return  xdr_bytes(str->xdr, (char **) &pt, &nn, nn);
                   case INTEGER2_NTU:
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(short), xdr_short);
                   case LOGICAL_NTU: case INTEGER_NTU:
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(int), xdr_int);
                   case REAL_NTU:
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                   case DBL_PRECISION_NTU:
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                   case COMPLEX_NTU:
                      nn = nn*2;
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                   case DBL_COMPLEX_NTU:
                      nn = nn*2;
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                   case POINTER_NTU:
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(long), xdr_long);
                   default :
                       return FALSE;
              }
         }
}
/*
** Generalized NTuple XDR filter, used for Decode only. 
** Simply decode one variable (scalar) or array value. No checks whatsoever!
** Not applicable if the structure organization style is parallel array
** or the index corresponds to a fixed size variable.
*/
bool_t xdr_mcfast_NTupleSubVar(mcfStream *str, descrGenNtuple *dNTu,
 		int ivar, int multIndex, char* version)
{
    char *cDat;
    u_int nn;
    void *pt;
    int ivarP;
    
     ivarP = ivar;
     while (dNTu->variables[ivarP]->lengthW == 0) ivarP--;
     cDat = version;
     cDat += dNTu->subOffset[multIndex];
     cDat += dNTu->variables[ivarP]->offset;
     pt = (void *) cDat;     
     xdr_setpos(str->xdr, 
              (str->currentPos +  dNTu->subXDROffset[multIndex] +
              dNTu->variables[ivarP]->offsetXDR));
     if (dNTu->variables[ivarP]->lengthW == 1) { 
          switch (dNTu->variables[ivarP]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      return  xdr_char(str->xdr, (char *) pt);
                   case INTEGER2_NTU:
                      return  xdr_short(str->xdr, (short *) pt);
                   case LOGICAL_NTU: case INTEGER_NTU:
                      return  xdr_int(str->xdr, (int *) pt);
                   case REAL_NTU:
                      return  xdr_float(str->xdr, (float *) pt); 
                   case DBL_PRECISION_NTU:
                      return  xdr_double(str->xdr, (double *) pt);
                   case COMPLEX_NTU:
                      nn =2;
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                   case DBL_COMPLEX_NTU:
                      nn =2;
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                   case POINTER_NTU:
                      return  xdr_long(str->xdr, (long *) pt);
                   default :
                      return FALSE;
              }
        } else {
           nn = dNTu->variables[ivarP]->lengthW;
           switch (dNTu->variables[ivarP]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      return  xdr_bytes(str->xdr, (char **) &pt, &nn, nn);
                   case INTEGER2_NTU:
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(short), xdr_short);
                   case LOGICAL_NTU: case INTEGER_NTU:
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(int), xdr_int);
                   case REAL_NTU:
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                   case DBL_PRECISION_NTU:
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                   case COMPLEX_NTU:
                      nn = nn*2;
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                   case DBL_COMPLEX_NTU:
                      nn = nn*2;
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                   case POINTER_NTU:
                      return  xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(long), xdr_long);
                   default :
                       return FALSE;
              }
         }
}
/*
** Generalized NTuple XDR filter, used for Decode only. 
** Simply decode a sub-structure given a value for the multiplicity index.
** Not applicable if the structure organization style is parallel array.
** No check whatsover!
*/
bool_t xdr_mcfast_NTupleSubStruct(mcfStream *str, descrGenNtuple *dNTu,
 		int multIndex, char* version)
{
    char *cDat;
    u_int nn;
    void *pt;
    int iv;
    bool_t ok;
    
     xdr_setpos(str->xdr, 
              (str->currentPos +  dNTu->subXDROffset[multIndex]));
     for (iv=dNTu->firstIndexed; iv<dNTu->numVariables; iv++) {          
        cDat = version;
        cDat += 
             dNTu->subOffset[multIndex] + dNTu->variables[iv]->offset;
        pt = (void *) cDat;
        if (dNTu->variables[iv]->lengthW == 1) {
              switch (dNTu->variables[iv]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      ok =  xdr_char(str->xdr, (char *) pt);
                      break;
                   case INTEGER2_NTU:
                      ok = xdr_short(str->xdr, (short *) pt);
                      break;
                   case LOGICAL_NTU: case INTEGER_NTU:
                      ok = xdr_int(str->xdr, (int *) pt);
                      break;
                   case REAL_NTU:
                      ok = xdr_float(str->xdr, (float *) pt); 
                      break;
                   case DBL_PRECISION_NTU:
                      ok = xdr_double(str->xdr, (double *) pt);
                      break;
                   case COMPLEX_NTU:
                      nn =2;
                      ok = xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_COMPLEX_NTU:
                      nn =2;
                      ok = xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case POINTER_NTU:
                      ok = xdr_long(str->xdr, (long *) pt);
                   default :
                      return FALSE;
              }
        } else if (dNTu->variables[iv]->lengthW > 1){
           nn = dNTu->variables[iv]->lengthW;
           switch (dNTu->variables[iv]->type) {
                   case BYTE_NTU: case CHARACTER_NTU:
                      ok = xdr_bytes(str->xdr, (char **) &pt, &nn, nn);
                      break;
                   case INTEGER2_NTU:
                      ok = xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(short), xdr_short);
                      break;
                   case LOGICAL_NTU: case INTEGER_NTU:
                      ok = xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(int), xdr_int);
                      break;
                   case REAL_NTU:
                      ok = xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_PRECISION_NTU:
                      ok = xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case COMPLEX_NTU:
                      nn = nn*2;
                      ok = xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(float), xdr_float);
                      break;
                   case DBL_COMPLEX_NTU:
                      nn = nn*2;
                      ok = xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(double), xdr_double);
                      break;
                   case POINTER_NTU:
                      ok = xdr_array(str->xdr, 
                         (char **) &pt, &nn, nn, sizeof(long), xdr_long);
                      break;
                   default :
                       return FALSE;
              }
           }
       }
       return TRUE;
}
