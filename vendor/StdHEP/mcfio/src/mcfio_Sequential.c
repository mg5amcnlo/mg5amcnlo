/*******************************************************************************
*									       *
* mcfio_Sequential.c --  Utility routines for the McFast Monte-Carlo           *
*	Real Sequential routines, based on the RBIO package.                   *
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
* Written by Paul Lebrun						       *
* May 1995								       *
*									       *
*******************************************************************************/
#include <stdio.h>
#include <string.h>
#include "mcfio_Sequential.h"
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
#include "mcf_xdr.h"
#include "mcfio_Dict.h"
#include "mcfio_Util1.h"
#include "mcfio_Sequential.h"
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
#define INITIATE 3

int mcfioC_OpenReadSequential(char *device, char *vsn, int filenumber)
{
   int i, jstr, idtmp, ntot, ll, iost, jfn, ldat, lrdat;
   u_int p1, p2;
   mcfStream *str;
   char *fileRbio;
/*
** Prolog, as in mcfio_Direct
*/   
  if (McfStreamPtrList == NULL) mcfioC_Init(); 
  if (McfNumOfStreamActive >= MCF_STREAM_NUM_MAX) {
     fprintf(stderr,
  " mcfio_OpenReadSequential: Too many streams opened simultaneously.\n"); 
     return -1;
   }
/*
** Check that this device is not already in used, if so, check that the 
**  status is MCFIO_EOF, if so, assign the jstr to that stream.
*/
   for (i=0, jstr = -1; i<McfNumOfStreamActive; i++) {
       if (McfStreamPtrList[i] != NULL) {
           str = McfStreamPtrList[i];
           if ((str->row == MCFIO_READ) && (str->dos == MCFIO_SEQUENTIAL)) {
              if (strcmp(device, str->device) == 0) {
                  if (str->status != MCFIO_EOF) 
                      mcfioC_CloseSequentialFile((i+1));
                  jstr = i;
              }     
           }
       }
   }  
   while ((jstr == -1) && (i<MCF_STREAM_NUM_MAX)) {
          if (McfStreamPtrList[i] == NULL) jstr=i;
          i++;
          }
   if(jstr == -1) {
     fprintf(stderr,
  " mcfio_OpenReadSequential: Internal error, please report \n"); 
     return -1;
   }
/*
** building the filename string for rbio.  Note that we do not repeat the 
** label if the device has already been opened prior to this call.
**
*/
   jfn = jstr+1;
   if (McfStreamPtrList[jstr] == NULL) {
       ll = strlen(device) + strlen(vsn) + 40;
       fileRbio = (char *) malloc (sizeof(char)*ll);
       if (strcmp(vsn,"None") == 0)                                          
           sprintf(fileRbio,"%s:S=%d", 
               device, filenumber);
        else if (strcmp(vsn,"Disk") == 0) 
           sprintf(fileRbio,"%s:UFORT", device);
        else 
           sprintf(fileRbio,"%s:VSN=%s:S=%d", 
               device, vsn, filenumber);
       ll = strlen(fileRbio);
       rbfopen_(&jfn, fileRbio, "R", &iost, ll, 1);
   } else {
       ll = strlen(device) + 12;
       fileRbio = (char *) malloc (sizeof(char)*ll);
       sprintf(fileRbio,"%s:S=%d", device, filenumber);
       ll = strlen(fileRbio);
       rbopen_(&jfn, fileRbio, &iost, ll, 1);
   }       
   free(fileRbio);
   if (iost != 0) { 
           fprintf(stderr,
 " mcfio_OpenReadSequential: Problem opening device %s, \
 VSN %s, file %d \n", device, vsn, filenumber);
     return -1;
   }
   
   if (McfStreamPtrList[jstr] == NULL) {
      McfStreamPtrList[jstr] = (mcfStream *) malloc(sizeof(mcfStream));
      str = McfStreamPtrList[jstr];
      str->xdr = (XDR *) malloc(sizeof(XDR));
      str->id = jstr+1;
      str->row = MCFIO_READ;
      str->dos = MCFIO_SEQUENTIAL;
      str->filename = NULL;
      str->filePtr = 0;
      ll = strlen(device) + 1;
      str->device = (char*) malloc(sizeof(char) * ll);
      strcpy(str->device, device);
      str->filenumber = filenumber;
      ll = strlen(vsn) + 1;
      str->vsn = (char*) malloc(sizeof(char) * ll);
      strcpy(str->vsn, vsn);
      str->minlrec = MCF_XDR_MINLREC;
      if (strcmp(vsn,"Disk") == 0)  
          ll = MCF_XDR_MAXLREC;
      else     
         rblklen_(&jfn, &ll, &iost);
      str->maxlrec = ll;
      str->shead = NULL;
      str->ehead = NULL;
      str->fhead = NULL;
      str->table = NULL;
      str->buffer = NULL;
      str->buffer2 = NULL;
    } else {
      str->filenumber = filenumber;
      str->minlrec = MCF_XDR_MINLREC;
      rblklen_(&jfn, &ll, &iost);
      str->maxlrec = ll;
      str->bufferSize = 0;
      if (str->buffer != NULL) free(str->buffer);
      if (str->buffer2 != NULL) free(str->buffer2);
    }
    str->numWordsC = 0;
    str->numWordsT = 0;
/*
** decode the first buffer, the Sequential header. 
*/   
   if (str->buffer == NULL) {
       str->bufferSize = str->maxlrec; 
       str->buffer = (char *) malloc(sizeof(char) * (str->maxlrec + 1));
   }    
   ldat = str->maxlrec;
   rbread_(&jfn, str->buffer, &ldat, &lrdat, &iost);
   if (iost != 0) { 
     fprintf(stderr,
 " mcfio_OpenReadSequential: Problem reading first record on \n device %s, \
 VSN %s, file %d \n", device, vsn, filenumber);
     mcfioC_FreeStream(&McfStreamPtrList[jstr]);
     rbfclose_(&jfn, &iost);
     return -1;
   }
   str->numWordsT += lrdat/4;
   xdrmem_create(str->xdr, str->buffer, str->maxlrec, XDR_DECODE);
   p1 = xdr_getpos(str->xdr);
   str->firstPos = p1;
   str->status = MCFIO_BOF;

   if (xdr_mcfast_fileheader(str->xdr, &idtmp,
                &ntot, McfGenericVersion, &(str->fhead), str->id) == FALSE) {
       fprintf (stderr, 
               "mcfio_OpenReadSequential: Unable to decode seqheader \n");
       mcfioC_FreeStream(&McfStreamPtrList[jstr]);
       rbfclose_(&jfn, &iost);
       return -1;
   }
   if (idtmp != FILEHEADER) {
       fprintf (stderr, 
            "mcfio_OpenReadSequential: First Structure not the file header \n");
      
       fprintf (stderr, 
            "                    : Further accesses probably suspicious \n");
       mcfioC_FreeStream(&McfStreamPtrList[jstr]);
       rbfclose_(&jfn, &iost);
       return -1;
   }
   xdr_setpos(str->xdr, p1);
   str->currentPos = p1;   
   str->numWordsC += (ntot/4);
   str->status = MCFIO_RUNNING;
   if (str->ehead == NULL) 
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


int mcfioC_OpenWriteSequential(char *device, char *vsn, char *title,
             char *comment, int numevts_pred,
              int *blkIds, unsigned int nBlocks)
{
   int i, jstr, idtmp, ntot, ll, iost, jfn, ldat, lwdat, filenumber;
   u_int p1, p2;
   mcfStream *str;
   char *fileRbio;
/*
** Prolog, as in mcfio_Direct
*/   
  if (McfStreamPtrList == NULL) { 
     fprintf(stderr,
" mcfio_OpenWriteSequential: We will first initialize by calling mcfio_Init.\n"); 
     mcfioC_Init();
  }
  if (McfNumOfStreamActive >= MCF_STREAM_NUM_MAX) {
     fprintf(stderr,
  " mcfio_OpenWriteSequential: Too many streams opened simultaneously.\n"); 
     return -1;
   }
/*
** Check that this device is not already in used, if so, check that the 
**  status is MCFIO_EOF, if so, assign the jstr to that stream.
*/
   for (i=0, jstr = -1; i<McfNumOfStreamActive; i++) {
       if (McfStreamPtrList[i] != NULL) {
           str = McfStreamPtrList[i];
           if ((str->row == MCFIO_WRITE) && (str->dos == MCFIO_SEQUENTIAL)) {
              if (strcmp(device, str->device) == 0) {
                  if (str->status != MCFIO_EOF) 
                      mcfioC_CloseSequentialFile((i+1));
                  jstr = i;
              }     
           }
       }
   }  
   while ((jstr == -1) && (i<MCF_STREAM_NUM_MAX)) {
          if (McfStreamPtrList[i] == NULL) jstr=i;
          i++;
          }
   if(jstr == -1) {
     fprintf(stderr,
  " mcfio_OpenWriteSequential: Internal error, please report \n"); 
     return -1;
   }
   if ((title != NULL) && (strlen(title) > 255)) {
     fprintf(stderr,
  " mcfio_OpenWriteSequential: Title is too long\n"); 
     return -1;
   }
     
   if ((comment != NULL) && (strlen(comment) > 255)) {
     fprintf(stderr,
  " mcfio_OpenWriteSequential: comment is too long\n"); 
     return -1;
   }
      
/*
** building the filename string for rbio
*/
   jfn = jstr+1;
   lwdat = MCF_XDR_MAXLREC;
   if (McfStreamPtrList[jstr] == NULL) {
       filenumber = 1;
       ll = strlen(device) + strlen(vsn) + 80; /* more than 10 digits for 
                                               ** filenumber block length O.K.
                                               */ 
       fileRbio = (char *) malloc (sizeof(char)*ll);
       if (strcmp(vsn,"None") == 0)                                          
           sprintf(fileRbio,"%s:D=mcfio_%d.dat:S=%d:F=U:B=%d", 
               device, filenumber, filenumber, lwdat);
        else if (strcmp(vsn,"Disk") == 0) 
           sprintf(fileRbio,"%s:UFORT", device);
        else 
           sprintf(fileRbio,"%s:D=mcfio_%d.dat:VSN=%s:S=%d:F=U:B=%d", 
               device, filenumber,vsn, filenumber, lwdat);
       ll = strlen(fileRbio);
       rbfopen_(&jfn, fileRbio, "W", &iost, ll, 1);
   } else {
       str->filenumber++;
       filenumber = str->filenumber;
       ll = strlen(device) + 80;
       fileRbio = (char *) malloc (sizeof(char)*ll);
       sprintf(fileRbio,"%s:D=mcfio_%d.dat:S=%d:F=U:B=%d", 
                device, filenumber, filenumber, lwdat);
       ll = strlen(fileRbio);
       rbopen_(&jfn, fileRbio, &iost, ll, 1);
   }       
   free(fileRbio);
   if (iost != 0) { 
           fprintf(stderr,
 " mcfio_OpenWriteSequential: \n\
    Problem opening device %s, \
 VSN %s, file %d, Rbio status =  \n", device, vsn, filenumber, iost);
     return -1;
   }
   
   if (McfStreamPtrList[jstr] == NULL) {
      McfStreamPtrList[jstr] = (mcfStream *) malloc(sizeof(mcfStream));
      str = McfStreamPtrList[jstr];
      str->xdr = (XDR *) malloc(sizeof(XDR));
      str->id = jstr+1;
      str->row = MCFIO_WRITE;
      str->dos = MCFIO_SEQUENTIAL;
      str->filename = NULL;
      str->filePtr = 0;
      ll = strlen(device) + 1;
      str->device = (char*) malloc(sizeof(char) * ll);
      strcpy(str->device, device);
      ll = strlen(vsn) + 1;
      str->vsn = (char*) malloc(sizeof(char) * ll);
      strcpy(str->vsn, vsn);
      str->filenumber = 1;
      str->minlrec = MCF_XDR_MINLREC;
      if (strcmp(vsn,"Disk") == 0)  
          ll = MCF_XDR_MAXLREC;
      else     
         rblklen_(&jfn, &ll, &iost);
      str->maxlrec = ll;
      str->shead = NULL;
      str->ehead = NULL;
      str->fhead = NULL;
      str->table = NULL;
      str->buffer = NULL;
      str->buffer2 = NULL;
    } else {
      str->minlrec = MCF_XDR_MINLREC;
      rblklen_(&jfn, &ll, &iost);
      str->maxlrec = ll;
      str->bufferSize = 0;
      if (str->buffer != NULL) free(str->buffer);
      if (str->buffer2 != NULL) free(str->buffer2);
    }
    str->numWordsC = 0;
    str->numWordsT = 0;
/*
** encode the first buffer, the file header. 
*/   
    if (str->buffer == NULL) {
       str->bufferSize =  str->maxlrec;
       str->buffer = (char *) malloc(sizeof(char) * (str->maxlrec + 1));
    }    
    ldat = str->maxlrec;
   
    xdrmem_create(str->xdr, str->buffer, str->maxlrec, XDR_ENCODE);
    str->firstPos = xdr_getpos(str->xdr);
    str->status = MCFIO_BOF;

    if (str->fhead == NULL) {
       str->fhead = (mcfxdrFileHeader *) malloc(sizeof(mcfxdrFileHeader));
       str->fhead->blockIds = NULL;
       str->fhead->blockNames = NULL;
    }
   /*
   ** Fill the file header, additional info will be written on tape
   */
    if (title == NULL) strcpy(str->fhead->title,"No Title given");
      else strcpy(str->fhead->title,title);
    
    if (comment == NULL) strcpy(str->fhead->comment,"No comment");
       else strcpy(str->fhead->comment, comment);
    str->fhead->numevts_expect = numevts_pred;
    str->fhead->numevts = 0;
    str->fhead->dimTable = 0;
    str->fhead->firstTable = 0;
    str->fhead->nBlocks = nBlocks;
    str->fhead->nNTuples = 0;
    if (str->fhead->blockIds != NULL) free(str->fhead->blockIds);
    str->fhead->blockIds = (int *) malloc(sizeof(int) * nBlocks);
    if (str->fhead->blockNames != NULL) {
      for (i=0; i<nBlocks; i++) if (str->fhead->blockNames[i] != NULL) { 
             free(str->fhead->blockNames[i]);
             str->fhead->blockNames[i] = NULL;
      }
      free(str->fhead->blockNames);
    }    
    str->fhead->blockNames = (char**) malloc(sizeof(char*) * nBlocks);
    for (i=0; i<nBlocks; i++) {
      str->fhead->blockIds[i] = blkIds[i];
      str->fhead->blockNames[i] = 
        (char *) malloc(sizeof(char) * (MCF_XDR_B_TITLE_LENGTH + 1));
      mcfioC_GetBlockName(blkIds[i], str->fhead->blockNames[i]);
    } 
     
    p1 = xdr_getpos(str->xdr);
    idtmp = FILEHEADER;
    if (xdr_mcfast_fileheader(str->xdr, &idtmp,
              &ntot, McfGenericVersion, &(str->fhead), str->id) == FALSE) {
       fprintf (stderr, 
               "mcfio_OpenWriteSequential: Unable to encode fileheader \n");
       mcfioC_FreeStream(&McfStreamPtrList[jstr]);
       rbfclose_(&jfn, &iost);
       return -1;
   }
/*
** write this buffer.
*/ 
    str->numWordsC += (ntot/4);
    p2 = xdr_getpos(str->xdr);
    ldat = p2 - p1;
    if (ldat < 512) ldat = 512;  
    rbwrite_(&jfn, str->buffer, &ldat, &lwdat, &iost); 
    if (iost != 0) { 
     fprintf(stderr,
 " mcfio_OpenWriteSequential: Problem writing first record on \n device %s, \
 VSN %s, file %d \n", device, vsn, filenumber);
     fprintf(stderr, " Status from Rbio %d \n", iost); 
     mcfioC_FreeStream(&McfStreamPtrList[jstr]);
     rbfclose_(&jfn, &iost);
     return -1;
   }
   str->numWordsT += (lwdat/4);
   /* printf(" Wrote in Open Seq. %d \n" , lwdat); */
   xdr_setpos(str->xdr, str->firstPos);
   str->currentPos = str->firstPos ;
   str->status = MCFIO_RUNNING;
/*
** Compose the dummy Sequential header to the first buffer.
*/
   if (str->shead == NULL) 
       str->shead = (mcfxdrSequentialHeader *)
                    malloc(sizeof(mcfxdrSequentialHeader));
   str->shead->nRecords = 1;
   idtmp = SEQUENTIALHEADER;
   if (xdr_mcfast_seqheader(str->xdr, &idtmp,
                             &ntot, McfGenericVersion, &(str->shead)) == FALSE) {
       fprintf (stderr, 
               "mcfio_OpenWriteSequential: Unable to encode fileheader \n");
       mcfioC_FreeStream(&McfStreamPtrList[jstr]);
       rbfclose_(&jfn, &iost);
       return -1;
   }
   
/*
** Compose the first dummy event header into memory. This will overwritten
**   with the complete information upon closure of the event.
*/   
   str->ehead = (mcfxdrEventHeader *) malloc(sizeof(mcfxdrEventHeader));
   str->ehead->dimBlocks = str->fhead->nBlocks;
   str->ehead->nBlocks = 0;
   str->ehead->nNTuples = 0;
   str->ehead->dimNTuples = str->fhead->nNTuples;
   str->ehead->evtnum = 0;
   str->ehead->previousevtnum = 0;
   str->ehead->storenum = 0;
   str->ehead->runnum = 0;
   str->ehead->trigMask = 0;
   str->ehead->blockIds = (int *) malloc(sizeof(int) * str->fhead->nBlocks);
   str->ehead->ptrBlocks = (u_int *) malloc(sizeof(int) * str->fhead->nBlocks);
   str->ehead->nTupleIds = (int *) malloc(sizeof(int) * str->fhead->nNtuples);
   str->ehead->ptrNTuples = 
       (u_int *) malloc(sizeof(int) * str->fhead->nNtuples);
   if (mcfioC_WrtEvt(str, INITIATE) == FALSE) {
       fprintf (stderr, 
               "mcfio_OpenWriteSequential: Unable to encode Evtheader \n");
       mcfioC_FreeStream(&McfStreamPtrList[jstr]);
       rbfclose_(&jfn, &iost);
       return -1;
   }    
   str->ehead->evtnum = 0;
   McfNumOfStreamActive++;
   return (jstr+1);

}

int mcfioC_NextEventSequential(int stream)
{
   int i, j, jstr, idtmp, ntot, ok, jfn, lrdat, lwdat, ldat, iost, lrem;
   u_int p1;
   mcfStream *str;
   char *tmpbuf;
   
  jfn = stream;
  jstr = stream -1;
  str = McfStreamPtrList[jstr];
  
/*
** Branching code Here, INPUT vs OUTPUT 
*/
   if (str->row == MCFIO_READ) {
/*
** Read the first record. Decode the Sequential header. Read more records 
**   if need be.
*/
      if (str->buffer2  == NULL) 
          str->buffer2 = (char *) malloc(sizeof(char)* (str->maxlrec+1)); 
      rbread_(&jfn, str->buffer2, &(str->maxlrec), &lrdat, &iost);
      if (iost != 0) { 
         fprintf(stderr,
                " mcfio_NextEventSequential: \n\
   Problem reading first record for this evt, stream %d, Rbio status %d \n", 
                       stream, iost);
          mcfioC_FreeStream(&McfStreamPtrList[jstr]);
          rbfclose_(&jfn, &iost);
          return -1;
       }
/*
** Read the Sequential header to get the number of buffers.
*/
      if (str->shead == NULL) 
          str->shead = 
            (mcfxdrSequentialHeader *) malloc(sizeof(mcfxdrSequentialHeader));
      xdrmem_create(str->xdr, str->buffer2, str->maxlrec, XDR_DECODE);
      str->firstPos = xdr_getpos(str->xdr); 
      if (xdr_mcfast_seqheader(str->xdr, &idtmp,
                             &ntot, McfGenericVersion, &(str->shead)) == FALSE) {
          fprintf (stderr, 
               "mcfio_NextEventSequential: Unable to decode seqheader \n");
          mcfioC_FreeStream(&McfStreamPtrList[jstr]);
          rbfclose_(&jfn, &iost);
          return -1;
      }
      str->currentPos = xdr_getpos(str->xdr);
      str->numWordsC += (ntot/4);
      if (str->shead->nRecords > 1) {
         if (str->bufferSize < (str->shead->nRecords * str->maxlrec)) {
             if (str->buffer != NULL) free(str->buffer);
             str->bufferSize = (str->shead->nRecords * str->maxlrec);
             str->buffer = (char *) malloc(sizeof(char) * (str->bufferSize+1));
         }
         tmpbuf = str->buffer;
         memcpy(str->buffer, str->buffer2,lrdat);
         tmpbuf += lrdat;
         for (i=1; i<str->shead->nRecords; i++) {
             rbread_(&jfn, tmpbuf, &(str->maxlrec), &lrdat, &iost);
             if (iost != 0) { 
                 fprintf(stderr,
                " mcfio_NextEventSequential: \n\
     Problem reading first record for this evt, stream %d, Rbio status\n",
               stream, iost);
              mcfioC_FreeStream(&McfStreamPtrList[jstr]);
              rbfclose_(&jfn, &iost);
              return -1;
             }
             tmpbuf += lrdat;
         }
         /*
         ** We have to reposition ourselves.. 
         */
         xdrmem_create(str->xdr, str->buffer, str->bufferSize, XDR_DECODE);
         xdr_setpos(str->xdr, str->currentPos);
       }
      if (xdr_mcfast_eventheader(str->xdr, &idtmp,
                          &ntot, McfGenericVersion, &(str->ehead)) == FALSE) {
          fprintf (stderr, 
               "mcfio_NextEventSequential: Unable to decode evtheader \n");
          mcfioC_FreeStream(&McfStreamPtrList[jstr]);
          rbfclose_(&jfn, &iost);
          return -1;
      }
      str->currentPos = xdr_getpos(str->xdr);
      str->numWordsC += (ntot/4);
    } else {  /* Writing an event.. */
    /*
    ** The entire buffer is at str->buffer. it has a size of
    ** str->bufferSize bytes. First, we have to compute the number of 
    ** records, rewrite the Sequential header, and the Event header.
    */
       p1 = str->currentPos;
       str->shead->nRecords = 1 + (p1 - str->firstPos)/str->maxlrec;
       if (((p1 - str->firstPos) % str->maxlrec) == 0) str->shead->nRecords--;
       xdr_setpos(str->xdr, str->firstPos);
       idtmp = SEQUENTIALHEADER;
       xdr_mcfast_seqheader(str->xdr, &idtmp,
                          &ntot, McfGenericVersion, &(str->shead));
       str->numWordsC += (ntot/4);
       str->currentPos = xdr_getpos(str->xdr);
       str->ehead->evtnum++;
       idtmp = EVENTHEADER;
       xdr_mcfast_eventheader(str->xdr, &idtmp,
            &ntot, McfGenericVersion, &(str->ehead));
       str->currentPos = xdr_getpos(str->xdr); 
       str->numWordsC += (ntot/4);
    /*
    ** The buffer is ready to be written to sequential media
    **
    */
       
       lrem = p1 - str->firstPos;
       for (i=0, tmpbuf=str->buffer; i<str->shead->nRecords; i++) {
           if (i == (str->shead->nRecords - 1)) 
               ldat = (lrem > 512) ? lrem : 512;
           else 
               ldat = str->maxlrec;
           rbwrite_(&jfn, tmpbuf, &ldat, &lwdat, &iost);
           /* printf(" Wrote in evt loop %d \n" , lwdat); */
           if (iost != 0) { 
              fprintf(stderr,
                  " mcfio_NextEventSequential: \n\
     Problem writing record on Stream %d, Rbio status %d ", stream, iost);
            mcfioC_FreeStream(&McfStreamPtrList[jstr]);
            rbfclose_(&jfn, &iost);
            return -1;
           }
           str->numWordsT += lwdat/4;
           lrem -= ldat;
           tmpbuf += ldat;
        }    
        /*
        ** Initiate a new bufffer
        */
        str->shead->nRecords = 1;
        xdr_setpos(str->xdr, str->firstPos);
        idtmp = SEQUENTIALHEADER;
        if (xdr_mcfast_seqheader(str->xdr, &idtmp,
                        &ntot, McfGenericVersion, &(str->shead)) == FALSE) {
            fprintf (stderr, 
               "mcfio_OpenWriteSequential: Unable to encode fileheader \n");
            mcfioC_FreeStream(&McfStreamPtrList[jstr]);
            rbfclose_(&jfn, &iost);
            return -1;
        }
        str->currentPos = xdr_getpos(str->xdr);
        str->ehead->nBlocks = 0;
        str->ehead->previousevtnum = str->ehead->evtnum;
        if (mcfioC_WrtEvt(str, INITIATE) == FALSE) {
           fprintf (stderr, 
               "mcfio_OpenWriteSequential: Unable to encode Evtheader \n");
           mcfioC_FreeStream(&McfStreamPtrList[jstr]);
           rbfclose_(&jfn, &iost);
           return -1;
        }
        str->ehead->evtnum = str->ehead->previousevtnum;
            
    }
    str->status = MCFIO_RUNNING;
    return MCFIO_RUNNING;
}

void mcfioC_CloseSequentialFile(int stream)
{
   mcfStream *str;
   int jfn, iost, jstr;
   
   jstr = stream -1;
   str =  McfStreamPtrList[jstr];
   str->status = MCFIO_EOF;
   jfn = jstr + 1;
   rbclose_(&jfn, &iost);
   if (iost != 0) {  
      fprintf(stderr,
                  " mcfio_CloseSequentialFile: \n\
     Problem closing file Stream %d, Rbio status %d ", jstr, iost);
     mcfioC_FreeStream(&McfStreamPtrList[jstr]);
  }
}

void mcfioC_CloseSequentialTape(int stream)
{
   mcfStream *str;
   int jfn, iost, jstr;
   
   jstr = stream -1;
   str =  McfStreamPtrList[jstr];
   jfn = jstr + 1;
   rbfclose_(&jfn, &iost);
   /*
   ** This suppose to work, 
   ** but does not..
   **
     if (strcmp(str->vsn,"Disk") != 0) rbrewind_(&jfn, &iost);
      rbumount_(&jfn, &iost);
   */
   if (iost != 0)  
      fprintf(stderr,
                  " mcfio_CloseSequentialTape: \n\
   Problem closing file Stream %d, Rbio status %d ", stream, iost);
   mcfioC_FreeStream(&McfStreamPtrList[jstr]);

}
