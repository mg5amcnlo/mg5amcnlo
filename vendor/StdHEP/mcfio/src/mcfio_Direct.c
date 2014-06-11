/*******************************************************************************
*									       *
* mcfio_Direct.c --  Utility routines for the McFast Monte-Carlo                 *
*		Direct Access I/O core routines 	                       *
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
#include <sys/stat.h>
#include <rpc/xdr.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include <fcntl.h>
#ifdef SUNOS
#include <floatingpoint.h>
#else /* SUNOS */
#include <float.h>
#endif /* SUNOS */
#include "mcf_nTupleDescript.h"
#include "mcf_xdr.h"
#include "mcfio_Dict.h"
#include "mcfio_Util1.h"
#include "mcfio_Direct.h"
#include "mcf_NTuIOUtils.h"
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef MAP_FILE 
#define MAP_FILE 0
#endif

extern nTuDDL **NTuDDLList;
extern int NumOfNTuples;


/* Static routine used in this module */

static int mcfioC_gofornextevent(mcfStream *str);   
static int  mcfioC_nextspecevt(mcfStream *str, int inum, int istore, 
                                       int irun, int itrig); 
static int openReadDirect(char*filename, int mode);


int mcfioC_OpenReadDirect(char *filename)
{                                            
/*
** Routine to open and read the header file for a Direct access Stream, 
** Standard Unix I/O 
*/
    return openReadDirect(filename, MCFIO_DIRECT);
}

int mcfioC_OpenReadMapped(char *filename)
{                                            
/*
** Routine to open and read the header file for a Direct access Stream, 
** Standard Unix I/O 
*/
    return openReadDirect(filename, MCFIO_MEMMAPPED);
}

static int openReadDirect(char *filename, int mode)
/*
** Routine to open and read the header file for a Direct access Stream.
*/
{
   int i, j, jstr, idtmp, ntot, ll1, jdRef, oldNumOfNTuples;
   int iff;
   off_t p1, p2;
   FILE *ff;
   mcfStream *str;
   nTuDDL *ddl, *ddlRef;
   struct stat statbuf;
   char *srcFile;
   
   
  if (McfStreamPtrList == NULL) mcfioC_Init(); 
   
  if (McfNumOfStreamActive >= MCF_STREAM_NUM_MAX) {
     fprintf(stderr,
  " mcfio_OpenReadDirect: Too many streams opened simultaneously.\n"); 
     return -1;
   }
   jstr = -1; i=0;
   while ((jstr == -1) && (i<MCF_STREAM_NUM_MAX)) {
          if (McfStreamPtrList[i] == NULL) jstr=i;
          i++;
          }
   if(jstr == -1) {
     fprintf(stderr,
  " mcfio_OpenReadDirect: Internal error, please report \n"); 
     return -1;
   }
   if ((filename == NULL) || (strlen(filename) > 255)) {
     fprintf(stderr,
  " mcfio_OpenReadDirect: You must give a valid UNIX filename.\n"); 
     return -1;
   }
   /*
   ** Now we can try to open this file.... 
   */
   if (mode == MCFIO_DIRECT) {
       ff = fopen(filename, "r");
       if (ff == NULL) {
           fprintf(stderr,
    " mcfio_OpenReadDirect: Problem opening file %s, message \n", filename);
           perror ("mcfio_OpenReadDirect"); 
           return -1;
       }
   } else { 
      /*
      ** Using memory mapped i/o
      */
      iff = open(filename, O_RDONLY);
          if (iff < NULL) {
          fprintf(stderr,
  " mcfio_OpenReadMapped: Problem opening file %s, message \n", filename);
          perror ("mcfio_OpenReadMapped"); 
          return -1;
      }
   }
   McfStreamPtrList[jstr] = (mcfStream *) malloc(sizeof(mcfStream));
   str = McfStreamPtrList[jstr];
   str->xdr = (XDR *) malloc(sizeof(XDR));
   str->id = jstr+1;
   str->row = MCFIO_READ;
   str->dos = mode;
   str->numWordsC = 0;
   str->numWordsT = 0;
   ll1 = strlen(filename) + 1;
   str->filename = (char *) malloc(sizeof(char) * ll1);
   strcpy(str->filename,filename);
   if (mode == MCFIO_DIRECT) {
       str->filePtr = ff;
       xdrstdio_create(str->xdr, ff, XDR_DECODE);
       str->fileDescr = 0;
       str->fileAddr = NULL;
       str->fileLen = 0; 
   } else {
      /*
      ** Use memory mapped I/O 
      */
      if (fstat(iff, &statbuf) < 0) {
          fprintf (stderr,
  " mcfio_OpenReadMapped: Problem getting file length for %s \n", filename);
          perror ("mcfio_OpenReadMapped"); 
          return -1;
      }
      if ((srcFile =
        mmap(0, statbuf.st_size, PROT_READ, MAP_FILE | MAP_SHARED, iff, 0 )) 
        == (caddr_t) -1) {
       fprintf (stderr,
  " mcfio_OpenReadMapped: Problem with memory mapping for %s \n", filename);
       perror ("mcfio_OpenReadMapped"); 
       return -1;
      }
      str->filePtr = (FILE *) NULL;
      str->fileDescr = iff;
      str->fileAddr = srcFile;
      str->fileLen = (size_t) statbuf.st_size;
      xdrmem_create(str->xdr, srcFile, statbuf.st_size,  XDR_DECODE);          
   }         
   str->device = NULL;
   str->vsn = NULL;
   str->filenumber = -1;
   str->minlrec = -1;
   str->maxlrec = -1;
   str->shead = NULL;
   str->ehead = NULL;
   str->table = NULL;
   str->buffer = NULL;
   str->buffer2 = NULL;
   p1 = ftello(str->filePtr);
   str->firstPos = p1;
   str->status = MCFIO_BOF;
   str->fhead = NULL;
   oldNumOfNTuples = NumOfNTuples;
   if (xdr_mcfast_fileheader(str->xdr, &idtmp,
                &ntot, McfGenericVersion, &(str->fhead), str->id) == FALSE) {
       fprintf (stderr, 
               "mcfio_OpenReadDirect: Unable to decode fileheader \n");
       mcfioC_FreeStream(&McfStreamPtrList[jstr]);
       mcfioC_Close(jstr+1);
       return -1;
   }
   if (idtmp != FILEHEADER) {
       fprintf (stderr, 
            "mcfio_OpenReadDirect: First Structure not the header \n");
      
       fprintf (stderr, 
            "                    : Further accesses probably suspicious \n");
       mcfioC_FreeStream(&McfStreamPtrList[jstr]);
       mcfioC_Close(jstr+1);
       return -1;
   }    
   p2 = ftello(str->filePtr);
   str->numWordsC += (ntot/4);
   /*
   ** Check if new these Ntuple template are not reference, if so,
   ** set the reference pointer accordingly, conversely, recompute the 
   ** offsets and length if requested.  We also fill the sequential 
   ** id number for the descriptors.  Note: those are trivial for 
   ** input streams, but we still fill them for consitency.
   */
   for (i=0; i<str->fhead->nNTuples; i++) {
      ddl = mcf_GetNTuByPtrID((oldNumOfNTuples+i+1));
      if (ddl == NULL) continue;
      ddl->streamId = (jstr+1);
      ddl->seqNTuId = (i+1);
      if (ddl->descrNtu == NULL) {
          for (j=0, jdRef=1; j<i; j++, jdRef++) {
             if (jdRef == ddl->referenceId) { 
               ddlRef = mcf_GetNTuByPtrID((oldNumOfNTuples+j+1));
               /*
               ** back up in the linked list if need be, until we 
                ** a fully documented descriptor.
                */
               while (ddlRef->descrNtu == NULL) ddlRef = ddlRef->reference;
                 ddl->reference = ddlRef;
                      break;
             }
        }
      } else {
          if (McfNTuPleSaveDecoding == TRUE) {
             mcf_ComputeNTuOffsets(ddl);    
             mcf_ComputeNTuLengths(ddl);
          }   
      }           
   }
   str->currentPos = p2;
   str->fhead->firstTable = p2;
    /* presumably correct , assume standard direct acces file config. */
   str->numWordsT += ((p2-p1)/4);
   str->status = MCFIO_RUNNING;
   str->table = (mcfxdrEventTable *) malloc(sizeof(mcfxdrEventTable));
   str->table->nextLocator = -1;
   str->table->dim = str->fhead->dimTable;
   str->table->numevts = 0;
   str->table->previousnumevts = 0;
   str->table->evtnums = NULL;
   str->table->storenums = NULL;
   str->table->runnums = NULL;
   str->table->trigMasks = NULL;
   str->table->ptrEvents = NULL;
   str->ehead = (mcfxdrEventHeader *) malloc(sizeof(mcfxdrEventHeader));
   str->ehead->dimBlocks = str->fhead->nBlocks;
   str->ehead->blockIds = NULL;
   str->ehead->ptrBlocks = NULL;
   str->ehead->dimNTuples = str->fhead->nNTuples;
   str->ehead->nTupleIds = NULL;
   str->ehead->ptrNTuples = NULL;
   McfNumOfStreamActive++;
   return (jstr+1);
}
    
int mcfioC_OpenWriteDirect(char *filename, char *title, char *comment,
                           int numevts_pred, int *blkIds, u_int nBlocks)

/*
** Routine to open and write the header file for a Direct access Stream.
*/
{
   int i, jstr, idtmp, ntot;
   off_t p1, p2;
   FILE *ff;
   mcfStream *str;
   
  if (McfStreamPtrList == NULL) { 
     fprintf(stderr,
  " mcfio_OpenWriteDirect: We will first initialize by calling mcfio_Init.\n"); 
     mcfioC_Init();
  }
  if (McfNumOfStreamActive >= MCF_STREAM_NUM_MAX) {
     fprintf(stderr,
  " mcfio_OpenWriteDirect: Too many streams opened simultaneously.\n"); 
     return -1;
   }
   jstr = -1; i=0;
   while ((jstr == -1) && (i<MCF_STREAM_NUM_MAX)) {
          if (McfStreamPtrList[i] == NULL) jstr=i;
          i++;
          }
   if(jstr == -1) {
     fprintf(stderr,
  " mcfio_OpenWriteDirect: Internal error, please report \n"); 
     return -1;
   }
   if ((filename == NULL) || (strlen(filename) > 255)) {
     fprintf(stderr,
  " mcfio_OpenWriteDirect: You must give a valid UNIX filename.\n"); 
     return -1;
   }
   if ((title != NULL) && (strlen(title) > 255)) {
     fprintf(stderr,
  " mcfio_OpenWriteDirect: Title is too long\n"); 
     return -1;
   }
     
   if ((comment != NULL) && (strlen(comment) > 255)) {
     fprintf(stderr,
  " mcfio_OpenWriteDirect: comment is too long\n"); 
     return -1;
   }
      
   /*
   ** Now we can try to open this file.... 
   */
   ff = fopen(filename, "w");
   if (ff == NULL) {
     fprintf(stderr,
  " mcfio_OpenWriteDirect: Problem opening file %s, message \n", filename);
     perror ("mcfio_OpenWriteDirect"); 
     return -1;
   }
   McfStreamPtrList[jstr] = (mcfStream *) malloc(sizeof(mcfStream));
   str = McfStreamPtrList[jstr];
   str->xdr = (XDR *) malloc(sizeof(XDR));
   str->id = jstr+1;
   str->row = MCFIO_WRITE;
   str->dos = MCFIO_DIRECT;
   str->numWordsC = 0;
   str->numWordsT = 0;
   str->filename = (char *) malloc(sizeof(char) * ( strlen(filename) +1) );
   strcpy(str->filename,filename); 
   str->filePtr = ff;
   str->device = NULL;
   str->vsn = NULL;
   str->filenumber = -1;
   str->minlrec = -1;
   str->maxlrec = -1;
   str->shead = NULL;
   str->ehead = NULL;
   str->table = NULL;
   str->buffer = NULL;
   str->buffer2 = NULL;
   xdrstdio_create(str->xdr, ff, XDR_ENCODE);
   p1 = ftello(str->filePtr);
   str->firstPos = p1;
   str->currentPos = p1;
   str->status = MCFIO_BOF;
   str->fhead = (mcfxdrFileHeader *) malloc(sizeof(mcfxdrFileHeader));
   /*
   ** Fill the file header, additional info will be written on tape
   */
   if (title == NULL) strcpy(str->fhead->title,"No Title given");
    else strcpy(str->fhead->title,title);
    
   if (comment == NULL) strcpy(str->fhead->comment,"No comment");
    else strcpy(str->fhead->comment, comment);
   str->fhead->numevts_expect = numevts_pred;
   str->fhead->numevts = 0;
   /* 
   ** Futur expansion : make this a tunable parameter.
   */
   str->fhead->dimTable = MCF_DEFAULT_TABLE_SIZE;
   str->fhead->firstTable = -1;
   str->fhead->nBlocks = nBlocks;
   if (nBlocks > 0) {
      str->fhead->blockIds = (int *) malloc(sizeof(int) * nBlocks);
      str->fhead->blockNames = (char**) malloc(sizeof(char *) * nBlocks);
   } else {
      str->fhead->blockIds = NULL;
      str->fhead->blockNames = NULL;
   }     
   for (i=0; i<nBlocks; i++) {
     str->fhead->blockIds[i] = blkIds[i];
     str->fhead->blockNames[i] = 
     (char *) malloc(sizeof(char) * (MCF_XDR_B_TITLE_LENGTH + 1));
     mcfioC_GetBlockName(blkIds[i], str->fhead->blockNames[i]);
   }
   str->fhead->nNTuples = 0; /* Will be filled later */ 
   if (mcfioC_Wrtfhead(str, INITIATE) == FALSE){
       mcfioC_FreeStream(&McfStreamPtrList[jstr]);
       fclose(ff);
       return -1;
   }
   str->table = (mcfxdrEventTable *) malloc(sizeof(mcfxdrEventTable));
   str->table->numevts=-1;
   str->table->nextLocator = -1;
   str->table->evtnums =   (int *) malloc(sizeof(int) * str->fhead->dimTable);
   str->table->storenums = (int *) malloc(sizeof(int) * str->fhead->dimTable);
   str->table->runnums = (int *) malloc(sizeof(int) * str->fhead->dimTable);
   str->table->trigMasks = (int *) malloc(sizeof(int) * str->fhead->dimTable);
   str->table->ptrEvents = 
         (off_t *) calloc(str->fhead->dimTable, sizeof(off_t));
   /*
   ** Write the first dummy table 
   */
   if (mcfioC_Wrttable(str, INITIATE) == FALSE) return -1;
   str->ehead = (mcfxdrEventHeader *) malloc(sizeof(mcfxdrEventHeader));
   str->ehead->dimBlocks = str->fhead->nBlocks;
   str->ehead->nBlocks = 0;
   str->ehead->dimNTuples = 0;
   str->ehead->nNTuples = 0;
   str->ehead->evtnum = 0;
   str->ehead->previousevtnum = 0;
   str->ehead->storenum = 0;
   str->ehead->runnum = 0;
   str->ehead->trigMask = 0;
   str->ehead->nTupleIds = NULL;
   str->ehead->ptrNTuples = NULL;
   if (nBlocks > 0) {
      str->ehead->blockIds = 
          (int *) malloc(sizeof(int) * str->fhead->nBlocks);
      str->ehead->ptrBlocks =
         (off_t *) calloc(str->fhead->nBlocks, sizeof(off_t));
   } else {
       str->ehead->blockIds = NULL;
       str->ehead->ptrBlocks = NULL; 
   }       
   /*
   ** Write the first dummy event header
   */
   if (mcfioC_WrtEvt(str, INITIATE) == FALSE) return -1;
   str->ehead->evtnum = 0;
   str->status = MCFIO_RUNNING;
   McfNumOfStreamActive++;
   return (jstr+1);

}

int mcfioC_NextEvent(int stream)
/*
** The Core routine for getting or setting the next event d.s. from/to 
**  a stream. 
**
*/
{
   int i, jstr, idtmp, ntot, nn1;
   off_t p_evt, p2;
   mcfStream *str;
   
  if (McfStreamPtrList == NULL) { 
     fprintf(stderr,
  " mcfio_NextEvent: You must first initialize by calling mcfio_Init.\n"); 
     return -1;
  }
  jstr = stream-1;
  if (McfStreamPtrList[jstr] == NULL) { 
     fprintf(stderr,
 " mcfio_NextEvent: First, declare the stream by calling mcfio_Open...\n"); 
     return -1;
  }
  str = McfStreamPtrList[jstr];
  if (str->dos == MCFIO_SEQUENTIAL) return mcfioC_NextEventSequential(stream);
  if (str->row == MCFIO_READ) {
  /*
  ** Read the next event, hunt for either an event or a table of event
  **  if event table not available.
  */
      if ((str->table == NULL) || 
         ((str->table != NULL)&& (str->table->evtnums == NULL))) { 
                idtmp = mcfioC_gofornextevent(str);
                if (idtmp != EVENTTABLE) {
                    if (str->table !=NULL) 
                       mcfioC_Free_EventTable(&(str->table));
                    if (idtmp == NOTHING) return -1;
                    p_evt = str->currentPos;
                 } else {
                  if( xdr_mcfast_eventtable(str->xdr, &idtmp,
 		     &ntot, McfGenericVersion, &(str->table)) == FALSE) {
                           fprintf(stderr,
 " mcfio_NextEvent: XDR Error decoding the EventTable \n"); 
 		            return -1;
 		    }
                    p2 = ftello(str->filePtr);
                    str->numWordsC += (ntot/4);
                    str->numWordsT += ((p2-str->currentPos)/4);
                    str->currentPos = p2;
                    str->table->ievt = 0;
                    /* 
                    ** If table empty, cal this routine recursively to get 
                    **   the next event 
                    */
                    if (str->table->numevts <= 0) {
                      if (str->table->nextLocator == -1) 
                       mcfioC_Free_EventTable(&(str->table));
                       return mcfioC_NextEvent(str->id);
                    }     
                    p_evt = str->table->ptrEvents[0];
                } 
      } else {
           if (str->table->ievt < str->table->numevts) {
                 p_evt = str->table->ptrEvents[str->table->ievt];
           } else {
           /*
           ** decode the next table, if valid. If not, scrap the 
           ** existing table and call next event recursively.
           */
              if (str->table->nextLocator == -2) {
                  /* 
                  ** Stream is at EOF
                  */
                   str->status = MCFIO_EOF;
                   return MCFIO_EOF;
              } else if (str->table->nextLocator == -1) { 
                           fprintf(stderr,
 " mcfio_NextEvent: Corrupted Event Table \n"); 
 		            return -1;
                }
                if (fseeko(str->filePtr,str->table->nextLocator,SEEK_SET) != 0) {
                           fprintf(stderr,
 " mcfio_NextEvent: Error Repositioning stream \n"); 
 		            return -1;
 		 }
                 if( xdr_mcfast_eventtable(str->xdr, &idtmp,
 		     &ntot, McfGenericVersion, &(str->table)) == FALSE) {
                           fprintf(stderr,
 " mcfio_NextEvent: XDR Error decoding the EventTable \n"); 
 		            return -1;
 		    }
                    p2 = ftello(str->filePtr);
                    str->numWordsC += (ntot/4);
                    str->numWordsT += ((p2-str->currentPos)/4);
                    str->currentPos = p2;
                    str->table->ievt = 0;
                    p_evt = str->table->ptrEvents[0];
                  }
       }
       /* 
       ** we should be pointing to a good event header here. 
       */
       if (fseeko(str->filePtr,p_evt,SEEK_SET) != 0) return -1;
       if( xdr_mcfast_eventheader(str->xdr, &idtmp,
	&ntot, McfGenericVersion, &(str->ehead)) == FALSE) return -1;
        str->currentPos = ftello(str->filePtr);
        str->numWordsC += (ntot/4);
        str->numWordsT += ((str->currentPos - p_evt)/4);
        if (str->table != NULL) str->table->ievt ++;              
        return MCFIO_RUNNING;
  } else {
    /*
    ** Writing Code here.
    */
    str->table->numevts++;
    str->fhead->numevts++;
    if (str->ehead->previousevtnum == str->ehead->evtnum) str->ehead->evtnum++;
     /*
     ** Write the current event header, normal case. First Flush the current
     **  event,  then initiate the next one event. Note that wrtevt will
     ** reposition the stream after rewriting the event header, if FLUSH. 
     ** e.g. ready to initiate either a new table or a new event.
     */
     if (mcfioC_WrtEvt(str, FLUSH) == FALSE) return -1;
     str->ehead->previousevtnum = str->ehead->evtnum;
     if (str->table->numevts == (str->fhead->dimTable - 1)) {
      /*
      ** The Event table is now full. Flush it. Then initiate a new table. 
      */ 
       str->table->nextLocator = ftello(str->filePtr);
       if (mcfioC_Wrttable(str, FLUSH) == FALSE) return -1;
       if (mcfioC_Wrttable(str, INITIATE) == FALSE) return -1;
     }
     str->ehead->nBlocks = 0;
     str->ehead->nNTuples = 0;
     nn1 = str->ehead->evtnum;
     if (mcfioC_WrtEvt(str, INITIATE) == FALSE) return -1;
     str->ehead->evtnum = nn1;
     return MCFIO_RUNNING;
  }
}

int mcfioC_SpecificEvent(int stream, int ievt,
                             int istore, int irun, int itrig)
{
   int i, jstr, idtmp, ntot, ok, nn1;
   off_t p1, p2;
   mcfStream *str;
   
  if (McfStreamPtrList == NULL) { 
     fprintf(stderr,
  " mcfio_SpecificEvent: You must first initialize by calling mcfio_Init.\n"); 
     return -1;
  }
  jstr = stream-1;
  if (McfStreamPtrList[jstr] == NULL) { 
     fprintf(stderr,
 " mcfio_SpecificEvent: First, declare the stream by calling mcfio_Open...\n"); 
     return -1;
  }
  str = McfStreamPtrList[jstr];
  if ((str->row != MCFIO_READ) || (str->dos == MCFIO_SEQUENTIAL)) {
     fprintf(stderr,
" mcfio_SpecificEvent: Only valid for INPUT, DIRECT ACCESS \
 or Memory Mapped \n"); 
     return -1;
     }
  if (fseeko(str->filePtr,str->fhead->firstTable,SEEK_SET) != 0) {
       fprintf(stderr,
 " mcfio_SpecificEvent:  Could not reposition Direct Access Stream %d \n",
         (jstr+1)) ;
    return -1;
   }
   str->currentPos = str->fhead->firstTable;
   
   ok = mcfioC_nextspecevt(str, ievt, istore, irun, itrig);
   if (ok == FALSE) {
      mcfioC_RewindDirect(jstr);
      if (fseeko(str->filePtr,str->fhead->firstTable,SEEK_SET) != 0) {
           fprintf(stderr,
     " mcfio_SpecificEvent:  Could not reposition Direct Access Stream %d \n",
         (jstr+1)) ;
         return -1;
      }
      str->currentPos = str->fhead->firstTable;
      ok = mcfioC_nextspecevt(str, ievt, istore, irun, itrig);
    }
    if (ok == FALSE)   return -1;
   return ok;
    
}	                             
int mcfioC_NextSpecificEvent(int stream, int ievt,
                             int istore, int irun, int itrig)
{
   int i, jstr, idtmp, ntot, ok, nn1;
   off_t p1, p2;
   mcfStream *str;
   
  if (McfStreamPtrList == NULL) { 
     fprintf(stderr,
  " mcfio_NextSpecific: You must first initialize by calling mcfio_Init.\n"); 
     return -1;
  }
  jstr = stream-1;
  if (McfStreamPtrList[jstr] == NULL) { 
     fprintf(stderr,
 " mcfio_NextSpecific: First, declare the stream by calling mcfio_Open...\n"); 
     return -1;
  }
  str = McfStreamPtrList[jstr];
  if ((str->row != MCFIO_READ) || (str->dos == MCFIO_SEQUENTIAL)) {
     fprintf(stderr,
 " mcfio_NextSpecificEvent: Only valid for INPUT, DIRECT ACCESS\
 or memory mapped I/O  \n"); 
     return -1;
     }
   ok = mcfioC_nextspecevt(str, ievt, istore, irun, itrig);
   if (ok == FALSE) return -1;
   return ok;
    
}	                             


void mcfioC_CloseDirect(int jstr)
/*
** Close a direct access stream, Standard I/O or Memory Mapped
**
*/
{
   int i, idtmp, ntot;
   u_int *p_ptr;
   off_t p1, p2;
   FILE *ff;
   mcfStream *str;
   nTuDDL *ddl;
      
   str =  McfStreamPtrList[jstr];
   if (str->row == MCFIO_WRITE) {
       /*
       **  Flush the event header, and the last table header. 
       */
       if (str->status == MCFIO_RUNNING) { 
         str->table->numevts++;
         str->ehead->evtnum++;
         if (mcfioC_WrtEvt(str, FLUSH) == FALSE) return;
         str->table->nextLocator = -2;
         str->table->numevts--; /* Decrement, the table is incomplete at 
         				this point */
         if (mcfioC_Wrttable(str, FLUSH) == FALSE) return;
         if (mcfioC_Wrtfhead(str, FLUSH) == FALSE) return;
       }
     }
     xdr_destroy(str->xdr);
     if (str->dos == MCFIO_DIRECT) { 
         fclose(str->filePtr);
     } else {
         /*
         ** Memory mapped I/O, one has to unmapped.. 
         */
         munmap((caddr_t) str->fileAddr, str->fileLen);
         close(str->fileDescr);
     }
     /*
     ** One must declare the Ntuples obsolete for this stream. 
     ** Do not release the memory, just flag these Ntuple with an obsolete 
     ** stream
     */
     for (i=0; i<NumOfNTuples; i++) {
         ddl = mcf_GetNTuByPtrID((i+1));
         if ((ddl != NULL) && (ddl->streamId == (jstr+1)))
               ddl->streamId = -1;
    }
}
       
void mcfioC_RewindDirect(int jstr)
/*
** Rewind a direct access stream, open for Read only
**
*/
{
    mcfStream *str;
    
    str =  McfStreamPtrList[jstr];
    if (fseeko(str->filePtr,str->fhead->firstTable,SEEK_SET) != 0)
       fprintf(stderr,
       " mcfio_Rewind:  Could not reposition Direct Access Stream %d \n",
         (jstr+1)) ;
    str->currentPos = str->fhead->firstTable;
    if (str->table != NULL) {
        str->table->nextLocator = str->fhead->firstTable;
        str->table->numevts = 0;
        str->table->previousnumevts = 0;
    }    
    if (str->ehead != NULL) {
        str->ehead->evtnum = 0;
        str->ehead->previousevtnum = 0;
    }
    return;
}  
   
int  mcfioC_Wrtfhead(mcfStream *str, int mode)     
/*
** Write the file header. 
**  IF Mode = INITIATE, write the dummy information, at the current location.
**  IF mode = Flush, rewite all the information, this time with the 
**  correct number of events.
**
*/
{
   int idtmp, ntot;
   off_t p1, p0;
   int k;
    time_t clock;
   
   idtmp = FILEHEADER;
   if (mode == FLUSH) {
     time(&clock);
     strcpy(str->fhead->closingDate, ctime(&clock));
     if(fseeko(str->filePtr,str->firstPos,SEEK_SET) != 0) return FALSE; 
     if (xdr_mcfast_fileheader(str->xdr, &idtmp,
          &ntot, McfGenericVersion, &(str->fhead), str->id) == FALSE) {
       fprintf (stderr, 
               "mcfio_OpenCloseDirect: Unable to reencode file head \n");
       return FALSE;
      }
      /*
      ** The version of MCFIO is still at this point v2.0 
      */
   } else if (mode == INITIATE) {
       /* Put the current date/time in a string */
     time(&clock);
     strcpy(str->fhead->date, ctime(&clock));
     /*
     ** We obviously do not have the closing times stamp yet (Causality)
     ** So we put ?, however, we have to put the right number of them, 
     ** the we do not screw up the XDR pointers..
     */
     for (k=0; k<strlen(ctime(&clock)); k++) str->fhead->closingDate[k] = '?';
     str->fhead->closingDate[strlen(ctime(&clock))] = '\0';
     p0 = str->currentPos;
     if (xdr_mcfast_fileheader(str->xdr, &idtmp,
               &ntot, McfGenericVersion, &(str->fhead), str->id) == FALSE) {
       fprintf (stderr, 
               "mcfio_OpenWriteDirect: Unable to encode fileheader \n");
       return FALSE;
      } 
      p1 = ftello(str->filePtr);
      str->numWordsC += (ntot/4);
      str->numWordsT += ((p1-p0)/4);
      str->currentPos = p1;
      return TRUE;
   } else {
     fprintf(stderr," mcfioC_Wrtfhead: Internal error, lost mode \n");
     return FALSE;
   }
   return TRUE;
}
             
   
int  mcfioC_WrtEvt(mcfStream *str, int mode)     
/*
** Write an event header, and update the table. Presumably, we have room 
**  in this table to do so.
**  IF Mode = INITIATE, write the dummy event header, at the current location.
**   Do not fill the element table.
**  If mode = FLUSH write the real event header and also
**     fill the Table elements. 
**
*/
{
   int i, idtmp, ntot;
   off_t p1, p0;
   
   idtmp = EVENTHEADER;
   if (mode == FLUSH) {
    str->table->evtnums[str->table->numevts] = str->ehead->evtnum;             
    str->table->storenums[str->table->numevts] = str->ehead->storenum;             
    str->table->runnums[str->table->numevts] = str->ehead->runnum;             
    str->table->trigMasks[str->table->numevts] = str->ehead->trigMask;
    str->table->ptrEvents[str->table->numevts] = str->evtPos;
    p0 = str->currentPos;
    if(fseeko(str->filePtr,str->evtPos,SEEK_SET) != 0) return FALSE; 
    p1 = str->evtPos;
    if(xdr_mcfast_eventheader(str->xdr, &idtmp,
            &ntot, McfGenericVersion, &(str->ehead)) == FALSE) return FALSE;
    str->currentPos = ftello(str->filePtr); 
    str->numWordsC += (ntot/4);
    str->numWordsT += ((str->currentPos-p1)/4);
    if(fseeko(str->filePtr,p0,SEEK_SET) != 0) return FALSE;
    str->currentPos = p0;
    str->ehead->nBlocks = 0;
    str->ehead->nNTuples = 0;
    return TRUE;
   } else if (mode == INITIATE) {
    str->ehead->nBlocks = 0; /*do not initialize nNTuples, already done */
    str->ehead->evtnum = -1;
    str->evtPos = ftello(str->filePtr);
    
    if(xdr_mcfast_eventheader(str->xdr, &idtmp,
            &ntot, McfGenericVersion, &(str->ehead)) == FALSE) return FALSE;
    str->currentPos = ftello(str->filePtr);
    return TRUE;
   } else {
     fprintf(stderr," mcfioC_WrtEvt: Internal error, lost mode \n");
     return FALSE;
   }
}
             
int  mcfioC_Wrttable(mcfStream *str, int mode)     
/*
** Write an event table. 
**  IF Mode = INITIATE, write the dummy event table, at the current location.
**   Do not fill the element table.
**  If mode = FLUSH write the real event header and also
**     fill the Table elements. 
**
*/
{
   int idtmp, ntot;
   off_t p1, p0;
   
   idtmp = EVENTTABLE;
   str->table->dim = str->fhead->dimTable;
   if (mode == FLUSH) {
    p0 = str->currentPos;
    if(fseeko(str->filePtr,str->tablePos,SEEK_SET) != 0) return FALSE; 
    p1 = str->tablePos;
    str->table->numevts++;
    if(xdr_mcfast_eventtable(str->xdr, &idtmp,
            &ntot, McfGenericVersion, &(str->table)) == FALSE) return FALSE;
    str->currentPos = ftello(str->filePtr); 
    str->numWordsC += (ntot/4);
    str->numWordsT += ((str->currentPos-p1)/4);
    if(fseeko(str->filePtr,p0,SEEK_SET) != 0) return FALSE;
    str->currentPos = p0;
    str->tablePos = -1;
    str->table->nextLocator = -1;
    str->table->numevts=-1;
    return TRUE;
   } else if (mode == INITIATE) {
    str->tablePos = ftello(str->filePtr);
    str->table->nextLocator = -1;
    if(xdr_mcfast_eventtable(str->xdr, &idtmp,
            &ntot, McfGenericVersion, &(str->table)) == FALSE) return FALSE;
    str->currentPos = ftello(str->filePtr);
    return TRUE;
   } else {
     fprintf(stderr," mcfioC_Wrttable: Internal error, lost mode \n");
     return FALSE;
   }
}

static int mcfioC_gofornextevent(mcfStream *str)   
/*
** Move in the direct access file to the next event or event table, 
** whatever comes first. The XDR current position is set to the beginning 
** of the event header or event table, if search sucessfull.
** We position the stream to the last Block or Ntuple defined in 
** the current event. 
*/
{
   off_t p1, p2;
   int id, ntot, go;
   
   go = TRUE;
   
   while (go == TRUE) {
     p1 = ftello(str->filePtr);
     if (xdr_mcfast_headerBlock(str->xdr, &id, &ntot, McfGenericVersion)
            == FALSE)  return NOTHING;
     if ((id == EVENTTABLE) || (id == EVENTHEADER)) {
         str->currentPos = p1;
         if(fseeko(str->filePtr,p1,SEEK_SET) != 0) return NOTHING;
         return id;
     }
   }
   return NOTHING; /* This statement is to make the compiler happy */
}  
             
static int  mcfioC_nextspecevt(mcfStream *str, int inum, int istore, 
                                       int irun, int itrig)
/*
** For Input, Direct access streams, hunt for a psecific event
**
*/  
{
   int i, jstr, j, idtmp, ntot, found;
   off_t p_evt, p2;

   if ((str->table == NULL) || 
         ((str->table != NULL)&& (str->table->evtnums == NULL))) { 
                idtmp = mcfioC_gofornextevent(str);
                if (idtmp != EVENTTABLE) {
                  fprintf(stderr,
 " mcfio_SpecificEvent: No event table on stream %d \n", str->id);
                  return FALSE;
                 } else {
                  if( xdr_mcfast_eventtable(str->xdr, &idtmp,
 		     &ntot, McfGenericVersion, &(str->table)) == FALSE) {
                           fprintf(stderr,
 " mcfio_SpecificEvent: XDR Error decoding the EventTable \n"); 
 		            return FALSE;
 		    }
                    p2 = ftello(str->filePtr);
                    str->numWordsC += (ntot/4);
                    str->numWordsT += ((p2-str->currentPos)/4);
                    str->currentPos = p2;
                    str->table->ievt = 0;
                    /* 
                    ** If table empty, cal this routine recursively to get 
                    **   the next event 
                    */
                    str->table->ievt = 0;
                } 
      }
      found = FALSE;
      while (found == FALSE){
           j =  str->table->ievt;    
           if (str->table->ievt < str->table->numevts) {
             if (((inum == 0)
                 || ( inum != 0 && (str->table->evtnums[j] == inum))) &&
                 (((istore == 0) 
                 || (istore != 0) && (str->table->storenums[j] == istore))) &&
                 (((irun == 0) 
                 || (irun != 0) && (str->table->runnums[j] == irun))) &&
                 (((itrig == 0) 
                 || (itrig != 0) && (str->table->trigMasks[j] == itrig))))
                  found = TRUE;
                  p_evt = str->table->ptrEvents[str->table->ievt];
                  str->table->ievt++;
           } else {
           /*
           ** decode the next table, if valid. If not, scrap the 
           ** existing table and call next event recursively.
           */
              if (str->table->nextLocator == -2) {
                  /* 
                  ** Stream is at EOF
                  */
                   str->status = MCFIO_EOF;
		   
                   return FALSE;
		   
              } else  if (str->table->nextLocator == -1) {
                           fprintf(stderr,
 " mcfio_NextEvent: Next EventTable corrupted, abandoning search \n"); 
 		            return FALSE;
              }
              if (fseeko(str->filePtr,str->table->nextLocator,SEEK_SET)
                      != 0) { fprintf(stderr,
 " mcfio_NextEvent: XDR Error repositioning to the next EventTable \n"); 
 		            return FALSE;
              } else  {
                     if( xdr_mcfast_eventtable(str->xdr, &idtmp,
 		     &ntot, McfGenericVersion, &(str->table)) == FALSE) {
                           fprintf(stderr,
 " mcfio_NextEvent: XDR Error decoding the EventTable \n"); 
 		            return FALSE;
 		    }
 	       }
               p2 = ftello(str->filePtr);
               str->numWordsC += (ntot/4);
               str->numWordsT += ((p2-str->currentPos)/4);
               str->currentPos = p2;
               str->table->ievt = 0;
               p_evt = str->table->ptrEvents[0];
           }
       }
       if (found == FALSE) return FALSE;
       /* 
       ** we should be pointing to a good event header here. 
       */
       if (fseeko(str->filePtr,p_evt,SEEK_SET) != 0) return FALSE;
       if( xdr_mcfast_eventheader(str->xdr, &idtmp,
	&ntot, McfGenericVersion, &(str->ehead)) == FALSE) return FALSE;
        str->currentPos = ftello(str->filePtr);
        str->numWordsC += (ntot/4);
        str->numWordsT += ((str->currentPos - p_evt)/4);
        return MCFIO_RUNNING;
        
}
