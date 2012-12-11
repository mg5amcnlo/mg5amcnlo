/*******************************************************************************
*									       *
* mcfio_Init.c -- Utility routines for the McFast Monte-Carlo                    *
*		Initialisation & info routines                                 *
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
#ifdef SUNOS
#include <floatingpoint.h>
#else /* SUNOS */
#include <float.h>
#endif /* SUNOS */
#include <time.h>
#include "mcf_nTupleDescript.h"
#include "mcf_xdr.h"
#include "mcfio_Util1.h"
#include "mcfio_Direct.h"
#include "mcfio_Sequential.h"
#include "mcfio_Dict.h"
#include "mcf_ntubld_db.h"
#include "mcf_NTuIOFiles.h"
#include "mcf_NTuIOUtils.h"
#include "mcf_NTuIOUtils.h"
#include "mcfio_UserDictionary.h"
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

mcfStream **McfStreamPtrList=NULL;
unsigned int McfNumOfStreamActive=0;
char **McfGenericVersion=NULL;

/*
** This stuff is needed for dbin utilities...
*/
struct line_title_c line_title_c_;
struct header_c header_c_;
struct variable_c variable_c_;
/*
** Names of variable types for Ntuple utilities
*/ 
char *VarTypesNamesF77[N_VAR_TYPES];
char *VarTypesNamesC[N_VAR_TYPES];
/*
** Ntuple global list
*/
extern nTuDDL **NTuDDLList;
extern int NumOfNTuples;


void mcfioC_Init(void)
/* Global Initialisation routine. Simply set the 
**
*/
{
  int i;
  char *env, *line;
  FILE *Ffp;
  
/*
** This is no longer needed... 

    env = NULL;
    env = getenv("MCFIO_DIR");
    if (env == NULL) { 
       printf ("You must first set the environment variable MCFIO_DIR\n");
       printf ("  by either setting up mcfio (Fermi UPS), or setting \n");
       printf
        ("  this env. variable to the place where mcf_NTuBld.db resides.\n");
       exit(0);
    } */
    
    
    /*
    ** Check now that the master template exist.
    
    line = (char *) malloc(sizeof(char) * (FILENAME_MAX+1));
    sprintf(line,"%s/mcf_NTuBld.db", env);
    Ffp = fopen(line, "r");
    if (Ffp == NULL) {
       printf ("The file %s could not be opened. \n", line);
       printf (" Please check MCFIO installation. \n"); 
       exit(0);
    }
    fclose(Ffp);
    free(line);
    
*/
  /*
  ** Use only one version for now.  Possible extension here.
  */
  McfGenericVersion = (char **) malloc(sizeof(char *));
  *McfGenericVersion = (char *) malloc(sizeof(char) * 8);
  
  VarTypesNamesF77[0]= "Byte         ";
  VarTypesNamesF77[1]= "Character    ";
  VarTypesNamesF77[2]= "Integer*2    ";
  VarTypesNamesF77[3]= "Logical      ";
  VarTypesNamesF77[4]= "Integer      ";
  VarTypesNamesF77[5]= "Real         ";
  VarTypesNamesF77[6]= "Double Precision";
  VarTypesNamesF77[7]= "Complex      ";
  VarTypesNamesF77[8]= "Double Complex  ";
  VarTypesNamesF77[9]= "Pointer      ";
        
  VarTypesNamesC[0]= "char         ";
  VarTypesNamesC[1]= "char         ";
  VarTypesNamesC[2]= "short        ";
  VarTypesNamesC[3]= "int          ";
  VarTypesNamesC[4]= "int          ";
  VarTypesNamesC[5]= "float        ";
  VarTypesNamesC[6]= "double       ";
  VarTypesNamesC[7]= "float[2]     ";
  VarTypesNamesC[8]= "double[2]    ";
  VarTypesNamesC[9]= "void *       ";
  
  if (NTuDDLList != NULL) {
      for (i=0; i<NumOfNTuples; i++) DestroyNTuDDL(NTuDDLList[i]);
      free(NTuDDLList);
  }
  NTuDDLList = (nTuDDL **) malloc(sizeof(int *)* NTU_START_LIST_SIZE);
  NumOfNTuples = 0;
  
  if (McfStreamPtrList == NULL) {  
	McfStreamPtrList = (mcfStream **)
		 malloc(sizeof(mcfStream *) * MCF_STREAM_NUM_MAX);
        for (i=0; i< MCF_STREAM_NUM_MAX; i++) McfStreamPtrList[i] = NULL;
        return;
   } 
  for (i=0; i< MCF_STREAM_NUM_MAX; i++) McfStreamPtrList[i] = NULL;
  mcfioC_Close(0); 
  McfNumOfStreamActive=0;
  
}

void mcfioC_Close(int istream) 
/* 
** Closing a Stream istream is the F77 index to the array of mcf Streams.
*/
{
   int i;
   
   if (McfStreamPtrList == NULL) return;
   if ((istream < 0) || (istream >  MCF_STREAM_NUM_MAX)) {
      fprintf (stderr, "mcf_close, Illegal argument, stream = %d \n", istream);
      return;
   }   
   if (istream == 0) {
       for (i=0; i<MCF_STREAM_NUM_MAX; i++) {
          if (McfStreamPtrList[i] != NULL) {
               switch (McfStreamPtrList[i]->dos) {
                   case MCFIO_DIRECT: case MCFIO_MEMMAPPED:
                      mcfioC_CloseDirect(i);
                      break;
                   case MCFIO_SEQUENTIAL: 
                      mcfioC_CloseSequentialTape(i);
                      break;
		   default:
		      fprintf
		       (stderr," mcf_close, Internal Error, please report \n");
		      break;
		      }
             mcfioC_FreeStream(&McfStreamPtrList[i]);
         }
      }
      return;
   }
   i = istream -1;
   if (McfStreamPtrList[i] != NULL) {
                switch (McfStreamPtrList[i]->dos) {
                   case MCFIO_DIRECT: case MCFIO_MEMMAPPED:
                      mcfioC_CloseDirect(i);
                      break;
                   case MCFIO_SEQUENTIAL: 
                      mcfioC_CloseSequentialTape(i);
                      break;
		   default:
		      fprintf
		       (stderr," mcf_close, Internal Error, please report \n");
		      break;
		      }
          mcfioC_FreeStream(&McfStreamPtrList[i]); 
       }
}

void mcfioC_Rewind(int istream) 
/* 
** Closing a Stream istream is the F77 index to the array of mcf Streams.
*/
{
   int i;
   
   if (McfStreamPtrList == NULL) return;
   if ((istream <= 0) || (istream >  MCF_STREAM_NUM_MAX)) {
      fprintf (stderr, "mcfio_Rewind, Illegal argument, stream = %d \n",
                         istream);
      return;
   }   
   i = istream -1;
   
   if (McfStreamPtrList[i] != NULL) {
                if(McfStreamPtrList[i]->row == MCFIO_WRITE) {
		     fprintf
	   (stderr," mcf_Rewind, Not support for Output Stream \n");
		      return;
		   }
                switch (McfStreamPtrList[i]->dos) {
                   case MCFIO_DIRECT: case MCFIO_MEMMAPPED:
                      mcfioC_RewindDirect(i);
                      break;
                   case MCFIO_SEQUENTIAL: 
		      fprintf
    (stderr," mcf_Rewind, Sequential, done by a close Sequential File\n\
    Then reopening a stream on the same sequential  media  \n");
                      break;
		   default:
		      fprintf
		       (stderr," mcf_Rewind, Internal Error, please report \n");
		      break;
		      }
		McfStreamPtrList[i]->numWordsC = 0;
		McfStreamPtrList[i]->numWordsT = 0;
       }
}

void mcfioC_Free_FileHeader(mcfxdrFileHeader **p)
{
   int i;
   mcfxdrFileHeader *head = *p;
   
   if (head == NULL) return;
   for (i=0; i<head->nBlocks; i++) 
     if (head->blockNames[i] != NULL) free(head->blockNames[i]);
   if (head->blockNames != NULL) free (head->blockNames); 
   if (head->blockIds != NULL) free(head->blockIds);
   free(head);
   *p = NULL;
} 

void mcfioC_Free_SeqHeader(mcfxdrSequentialHeader **p)
{
   mcfxdrSequentialHeader *head = *p;
   
   if (head == NULL) return;
   free(head);
   *p = NULL;
} 

void mcfioC_Free_EventHeader(mcfxdrEventHeader **p)
{
   mcfxdrEventHeader *head = *p;
   
   if (head == NULL) return;
   if (head->ptrBlocks != NULL) free(head->ptrBlocks);
   if (head->blockIds != NULL) free(head->blockIds);
   if (head->ptrNTuples != NULL) free(head->ptrNTuples);
   if (head->nTupleIds != NULL) free(head->nTupleIds);
   free(head);
   *p = NULL;
}

void mcfioC_Free_EventTable(mcfxdrEventTable **p)
{
   mcfxdrEventTable *table = *p;
   
   if (table == NULL) return;
   if (table->evtnums != NULL) free(table->evtnums);
   if (table->storenums != NULL) free(table->storenums);
   if (table->runnums != NULL) free(table->runnums);
   if (table->trigMasks != NULL) free(table->trigMasks);
   if (table->ptrEvents != NULL) free(table->ptrEvents);
   free(table);
   *p = NULL;
}
 
void mcfioC_FreeStream(mcfStream **stream)
{
   mcfStream *str = *stream;
   if (str == NULL) return;
   if (str->filename != NULL) free (str->filename);
   if (str->device != NULL) free (str->device);
   if (str->vsn != NULL) free (str->vsn);
   if (str->fhead != NULL) mcfioC_Free_FileHeader(&(str->fhead)); 
   if (str->shead != NULL) mcfioC_Free_SeqHeader(&(str->shead));
   if (str->ehead != NULL) mcfioC_Free_EventHeader(&(str->ehead));
   if (str->table != NULL) mcfioC_Free_EventTable(&(str->table));
   if (str->buffer != NULL) free (str->buffer);
   if (str->buffer2 != NULL) free (str->buffer2);
   free(str);
   *stream = NULL;
   McfNumOfStreamActive--;
}


void mcfioC_PrintDictionary(void)
{
   printf (" \n");
   printf
    (" Mcfast I/o Dictionary for Key words used in mcfio_Info routines \n");
   
   printf
    (" --------------------------------------------------------------- \n");
   printf (" \n");
   printf (" For Streams \n");
   printf (" -------------- \n");
   printf (" MCFIO_STATUS: The current status of the file;  \n");
   printf ("               the answer can be set to: \n");
   printf
 ("   MCFIO_BOF : at beginning of file \n");
   printf
 ("   MCFIO_EOF : at the end of file \n");
   printf 
 ("   MCFIO_RUNNING: At least a valid file header has been read or written\n");
   
   printf
    (" MCFIO_READORWRITE: if set MCFIO_READ, open for  read only \n");
   printf
    ("                    if set MCFIO_WRITE, open for  write only \n");
   printf
   (" MCFIO_DIRECTORSEQUENTIAL: if set MCFIO_DIRECT, accessing a UNIX file \n");
   printf
   ("                         : if set MCFIO_SEQUENTIAL, accessing a tape \n");
   printf
 (" MCFIO_NUMEVTS : Total number of events encode/decoded so far. \n");
   printf
    (" MCFIO_NUMBLOCK: The number of blocks defined in the file. \n");
    
   printf
   (" MCFIO_BLOCKIDS: The I.D. of the block defined in the file.\n");
   printf
 (" MCFIO_NUMWORDS: Total number of 4-bytes words encode/decoded so far. \n");
   printf
 (" MCFIO_EFFICIENCY: The overhead in blocking and XDR (*10000 ) \n");
   printf
 (" MCFIO_CREATIONDATE: The date (30 Character) when the file was opened \n");
   printf
 (" MCFIO_CLOSINGDATE: The date (30 Character) when the file was closed \n");
   printf
 (" MCFIO_TITLE: The title (255 Characters max) for the job \n");
   printf
 (" MCFIO_COMMENT: The comment (255 Characters max) for the job \n");
 
   printf (" \n");
   printf (" For Sequential Access only \n");
   printf 
   ("    MCFIO_FILENUMBER : The Sequential file number currently accessed.\n");
   printf ("    MCFIO_MAXLREC: Maximum Record length\n");
   printf ("    MCFIO_MINLREC: Minumum Record length\n");
   printf
   ("    MCFIO_NUMRECORDS: The number of records in the current event\n");
   printf 
   ("    MCFIO_RECORDLENGTHS: The record lengths for the current event\n"); 
   printf ("    MCFIO_DEVICENAME: The device name opened by the stream\n ");
   printf ("                    (character string, 255 l.)\n");
   printf (" \n");
   printf (" For Direct Access only \n");
   printf ("    MCFIO_FILENAME: The UNIX file name opened by the stream\n ");
   printf ("                    (character string, 255 l.)\n");
   
   printf (" \n");
   printf (" For Events \n");
   printf (" -------------- \n");
   printf
    (" MCFIO_NUMBLOCK: The number of blocks defined in the event.\n");
    
   printf
   (" MCFIO_BLOCKIDS: The I.D. of the block defined in the event.\n");
   printf
   (" MCFIO_EVENTNUMBER: The Event Number for this event. \n");
   printf
   (" MCFIO_STORENUMBER: The Store Number for this event. \n");
   printf
   (" MCFIO_RUNNUMBER: The Run Number for this event. \n");
   printf
   (" MCFIO_TRIGGERMASK: The Trigger Mask for this event. \n");
   printf (" MCFIO_VERSION: The 4-Character version of the event header \n ");
   
   printf (" \n");
   printf (" For Blocks \n");
   printf (" -------------- \n");
   printf (" MCFIO_VERSION: The 4-Character version of a particular block \n ");
   
   printf (" \n");
   printf (" For NTuples \n");
   printf (" -------------- \n");
   printf (" MCFIO_NUMNTUPLES: The number of defined NTuples on a stream \n ");
   printf (" See also mcfio_GetNTupleIds, mcfio_GetNTupleUID, \n");
   printf ("           mcfio_GetNTupleCategory, mcfio_GetNTupleTitle and \n");
   printf ("           mcfio_GetNTupleName \n");
   
}
   
unsigned int mcfioC_InfoNumStream(int *istreams,  unsigned int nmax)
/*
** Returns in the arrary istream the list of active stream indices.
** 
*/
{
   int i,j;
   
   if (nmax >= MCF_STREAM_NUM_MAX) {
     fprintf(stderr, "mcfio_Info, Illegal size of Stream Pointer array \n");
     return 0;
   }   
   for (i=0,j=0; i<MCF_STREAM_NUM_MAX; i++) {
   	if (McfStreamPtrList[i] != NULL) { 
          if (j < nmax) istreams[j] = McfStreamPtrList[i]->id;
   	  j++;  
          }
   } 
   return  McfNumOfStreamActive;
}

void mcfioC_InfoStreamInt(int stream, int key, int *values)
/*
** Information routine for the Stream.  Based on key, return in *values 
** the requested information 
*/
{
    int i, num, jstr;
    float a;
    mcfStream *str;
    jstr = stream - 1;
    if ((jstr <0) || (jstr >= MCF_STREAM_NUM_MAX)) {
      fprintf(stderr,"mcfio_InfoStream: Stream id %d is illegal \n",
                     stream);
      return;
    }
    str = McfStreamPtrList[jstr];
    if (str == NULL) {
      fprintf(stderr,"mcfio_InfoStream: Stream id %d is inactive \n",
                     stream);
      return;
    }
    switch (key) {
      case MCFIO_STATUS: 
           *values = str->status;
           break;
      case MCFIO_READORWRITE:
            *values = str->row;
            break;
      case MCFIO_DIRECTORSEQUENTIAL:
            *values = str->dos;
            break;
      case MCFIO_NUMWORDS:
            *values = str->numWordsT;
            break;
      case MCFIO_EFFICIENCY:
            a = ((float ) (str->numWordsC))/ (float) (str->numWordsT); 
            *values = (int) (10000. * a);
            break;
      case MCFIO_NUMEVTS:
           if(str->fhead != NULL)  *values = str->fhead->numevts;
           break;
      case MCFIO_NUMBLOCKS:
            if(str->fhead != NULL) *values = str->fhead->nBlocks;
            break;
      case MCFIO_BLOCKIDS:
      	/* 
      	** Crash bug possibility here, if the dimension is wrong ! 
      	*/
      	  if(str->fhead != NULL) {
            for (i=0; i<str->fhead->nBlocks; i++)
              values[i] = str->fhead->blockIds[i];
            }
            break;
        /*
        ** Now the specific items for Sequential stuff
        */
      case MCFIO_FILENUMBER:
          if (str->dos != MCFIO_SEQUENTIAL) {
             fprintf(stderr,
 "mcfio_InfoStream: Meaningless request for stream %d, key MCFIO_FILENUMBER\n", 
   stream);
   	     return;
   	    }
            *values = str->filenumber;
            break;
       case MCFIO_MAXREC:
          if (str->dos != MCFIO_SEQUENTIAL) {
             fprintf(stderr,
   "mcfio_InfoStream: Meaningless request for stream %d, key MCFIO_MAXREC\n", 
   stream);
   	     return;
   	   }
           *values = str->maxlrec;
   	   break;
       case MCFIO_MINREC:
          if (str->dos != MCFIO_SEQUENTIAL) {
             fprintf(stderr,
   "mcfio_InfoStream: Meaningless request for stream %d, key MCFIO_MINREC \n", 
   stream);
   	     return;
   	   }
           *values = str->minlrec;
   	   break;   	              
       case MCFIO_NUMRECORDS:
          if ((str->dos != MCFIO_SEQUENTIAL) || (str->shead == NULL) ) {
             fprintf(stderr,
 "mcfio_InfoStream: Meaningless request for stream %d, key MCFIO_NUMRECORDS \n", 
   stream);
   	     return;
   	   }
           *values = str->shead->nRecords;
   	   break;
       case MCFIO_RECORDLENGTHS:
          if ((str->dos != MCFIO_SEQUENTIAL) || (str->shead == NULL) ) {
             fprintf(stderr,
 "mcfio_InfoStream: Meaningless request for stream %d, key MCFIO_RECORDLENGTHS \n", 
   stream);
   	     return;
   	   }
   	   *values = str->maxlrec;
   	   break;
       case MCFIO_NUMNTUPLES:
          for (i=0, num=0; i<NumOfNTuples; i++)
               if (NTuDDLList[i]->streamId == stream) num++; 
          *values = num;
          break;            
      default:
            fprintf(stderr,
             "mcfio_InfoStream: Unrecognized Keyword %d\n", key);
    }
}

void mcfioC_InfoStreamChar(int stream, int key, char *answer, int *lret)
/*
** Information routine for the Stream.  Based on key, return in *values 
** the requested information 
*/
{
    int i, jstr;
    float a;
    mcfStream *str;
    jstr = stream - 1;
    if ((jstr <0) || (jstr >= MCF_STREAM_NUM_MAX)) {
      fprintf(stderr,"mcfio_InfoStream: Stream id %d is illegal \n",
                     stream);
      *lret = 0;
      return;
    }
    str = McfStreamPtrList[jstr];
    if (str == NULL) {
      fprintf(stderr,"mcfio_InfoStream: Stream id %d is inactive \n",
                     stream);
      *lret = 0;
      return;
    }
    switch (key) {
      case MCFIO_TITLE:
            if (str->fhead != NULL) strcpy(answer,str->fhead->title);
            break;
      case MCFIO_COMMENT:
            if (str->fhead != NULL) strcpy(answer,str->fhead->comment);
            break;
      case MCFIO_CREATIONDATE:
            if (str->fhead != NULL) strcpy(answer,str->fhead->date);
            break;
      case MCFIO_CLOSINGDATE:
            if (str->fhead != NULL) strcpy(answer,str->fhead->closingDate);
            break;
      case MCFIO_FILENAME:
          if (str->dos == MCFIO_SEQUENTIAL) {
             fprintf(stderr,
   "mcfio_InfoStream: Meaningless request for stream %d, key MCFIO_FILENAME \n", 
   stream);
             *lret = 0;
   	     return;
   	   }
            strcpy(answer,str->filename);
            break;
      case MCFIO_DEVICENAME:
          if (str->dos != MCFIO_SEQUENTIAL) {
             fprintf(stderr,
 "mcfio_InfoStream: Meaningless request for stream %d, key MCFIO_DEVICENAME \n", 
   stream);
             *lret = 0;
   	     return;
   	   }
            strcpy(answer,str->device);
            break;
      default:
            fprintf(stderr,
             "mcfio_InfoStream: Unrecognized Keyword %d\n", key);
             *lret = 0;
   	     return;
                           
    }
    *lret = strlen(answer);
} 
void mcfioC_InfoEventInt(int stream, int key, int *values)
/*
** Information routine for the current Event.  
**   Based on key, return in *values the requested information 
*/
{
    int i, jstr;
    float a;
    mcfStream *str;
    jstr = stream - 1;
    if ((jstr <0) || (jstr >= MCF_STREAM_NUM_MAX)) {
      fprintf(stderr,"mcfio_InfoEvent: Stream id %d is illegal \n",
                     stream);
      return;
    }
    str = McfStreamPtrList[jstr];
    if (str == NULL) {
      fprintf(stderr,"mcfio_InfoEvent: Stream id %d is inactive \n",
                     stream);
      return;
    }
    if (str->ehead ==NULL) {
      fprintf(stderr,"mcfio_InfoEvent: Stream id %d is at beginning \n",
                     stream);
      return;
    }  
    switch (key) {
      case MCFIO_EVENTNUMBER:
       *values = str->ehead->evtnum;
       break;
      case MCFIO_STORENUMBER:
       *values = str->ehead->storenum;
       break;
      case MCFIO_RUNNUMBER:
       *values = str->ehead->runnum;
       break;
      case MCFIO_TRIGGERMASK:
       *values = str->ehead->trigMask;
       break;
      case MCFIO_NUMBLOCKS:
       *values = str->ehead->nBlocks;
       break;
      case MCFIO_BLOCKIDS:
       for(i=0; i<str->ehead->nBlocks; i++) 
               values[i] = str->ehead->blockIds[i];
       break;        
      case MCFIO_NUMNTUPLES:
        *values = str->ehead->nNTuples;
        break;
      case MCFIO_NTUPLESLIST:
       for(i=0; i<str->ehead->nNTuples; i++) 
               values[i] = str->ehead->nTupleIds[i];
       break;
      default:
            fprintf(stderr,
             "mcfio_InfoEvent: Unrecognized Keyword %d\n", key);
                      
    }
}
 
void mcfioC_SetEventInfo(int stream, int key, int *values)
/*
** Set anciallary information for the current Event.  
**   Based on key, return in *values the requested information 
**   Only valid for Output Streams. 
*/
{
    int i, jstr;
    float a;
    mcfStream *str;
    jstr = stream - 1;
    if ((jstr <0) || (jstr >= MCF_STREAM_NUM_MAX)) {
      fprintf(stderr,"mcfio_InfoStream: Stream id %d is illegal \n",
                     stream);
      return;
    }
    str = McfStreamPtrList[jstr];
    if (str == NULL) {
      fprintf(stderr,"mcfio_SetEvent: Stream id %d is inactive \n",
                     stream);
      return;
    }
    if (str->ehead ==NULL) {
      fprintf(stderr,"mcfio_SetEvent: Stream id %d is at beginning \n",
                     stream);
      return;
    }
    if (str->row != MCFIO_WRITE) {  
      fprintf(stderr,
      "mcfio_SetEvent: Stream id %d must be an Output stream \n",
                     stream);
      return;
    }
    switch (key) {
      case MCFIO_EVENTNUMBER:
       str->ehead->evtnum = *values;
       break;
      case MCFIO_STORENUMBER:
       str->ehead->storenum = *values;
       break;
      case MCFIO_RUNNUMBER:
       str->ehead->runnum = *values;
       break;
      case MCFIO_TRIGGERMASK:
       str->ehead->trigMask = *values;
       break;
      case MCFIO_NUMBLOCKS: case MCFIO_BLOCKIDS:
      fprintf(stderr,
  "mcfio_SetEvent: Blocks and Block contents are set by mcfio_Blocks\n" );
      return;
      default:
            fprintf(stderr,
             "mcfio_SetEvent: Unrecognized Keyword %d\n", key);
                      
    }
} 

void mcfioC_InfoEventChar(int stream, int key, char *answer, int *lret)
/*
** Information routine for the current Event.  
**   Based on key, return in *values the requested information 
*/
{
    int i, jstr;
    float a;
    mcfStream *str;
    jstr = stream - 1;
    if ((jstr <0) || (jstr >= MCF_STREAM_NUM_MAX)) {
      fprintf(stderr,"mcfio_InfoStream: Stream id %d is illegal \n",
                     stream);
      *lret = 0;
      return;
    }
    str = McfStreamPtrList[jstr];
    if (str == NULL) {
      fprintf(stderr,"mcfio_InfoStream: Stream id %d is inactive \n",
                     stream);
      *lret = 0;
      return;
    }
    if (str->ehead ==NULL) {
      fprintf(stderr,"mcfio_InfoStream: Stream id %d is at beginning \n",
                     stream);
      *lret = 0;
      return;
    }  
    switch (key) {
      case MCFIO_VERSION:
       strcpy(answer, str->ehead->version);
       break;
      
      default:
            fprintf(stderr,
             "mcfio_InfoEvent: Unrecognized Keyword %d\n", key);
            *lret = 0;
            return;
                      
    }
    *lret = strlen(answer);
} 

void mcfioC_InfoBlockChar(int stream, int blkid,
                            int key, char *answer, int *lret)
/*
** Information routine for a particular block within the current Event.  
**   Based on key, return the requested information in string answer.
*/
{
    int i, jstr, itmp, nn;
    u_int pos;
    bool_t tt;
    XDR *xx;
    char* data, *vv;
    mcfStream *str;
    jstr = stream - 1;
    if ((jstr <0) || (jstr >= MCF_STREAM_NUM_MAX)) {
      fprintf(stderr,"mcfio_InfoStream: Stream id %d is illegal \n",
                     stream);
      *lret = 0;               
      return;
    }
    str = McfStreamPtrList[jstr];
    if (str == NULL) {
      fprintf(stderr,"mcfio_InfoStream: Stream id %d is inactive \n",
                     stream);
      *lret = 0;               
      return;
    }
    if (str->ehead ==NULL) {
      fprintf(stderr,"mcfio_InfoStream: Stream id %d is at beginning \n",
                     stream);
      *lret = 0;               
      return;
    }
    pos = 0;
    if(str->xdr != NULL) for(i=0; i<str->ehead->nBlocks; i++) 
      if( str->ehead->blockIds[i] == blkid) pos = str->ehead->ptrBlocks[i];
    if (pos == 0) {
      fprintf(stderr,
      "mcfio_InfoStream: Stream id %d event %d does not contain block %d \n",
             stream,str->ehead->evtnum, blkid );
      *lret = 0;               
      return;
    }
     
    switch (key) {
      case MCFIO_VERSION:
       tt = xdr_setpos(str->xdr, pos); 
       tt = xdr_mcfast_generic(str->xdr, &itmp, &nn, &vv, &data);
       xdr_free(xdr_string, data); 
       strcpy(answer, vv);
       break;
      
      default:
            fprintf(stderr,
             "mcfio_InfoEvent: Unrecognized Keyword %d\n", key);
            *lret = 0;
            return;               
                      
    }
    *lret = strlen(answer);
} 
void mcfioC_GetBlockName(int blkId, char *answer)
/*
**  Get a Block name from the dictionary..It is assume that answer has 
**     pre-malloc, size MCF_XDR_B_TITLE_LENGTH 
*/
{
   char *uDescr;
    switch (blkId) {
       case MCFIO_STDHEP:
         strcpy(answer,
         " Standard HEP COMMON block, see STDHEP Product");
         break;
         
       case MCFIO_STDHEPM:
         strcpy(answer,
         " Standard HEP COMMON block with multiple interaction, see STDHEP Product");
         break;
         
       case MCFIO_STDHEP4:
         strcpy(answer,
         " Standard HEP COMMON block with Les Houches, see STDHEP Product");
         break;
         
       case MCFIO_STDHEP4M:
         strcpy(answer,
         " Standard HEP COMMON block with Les Houches and multiple interaction");
         break;
         
       case MCFIO_HEPEUP:
         strcpy(answer,
         " Les Houches HEPEUP common block");
         break;
         
       case MCFIO_HEPRUP:
         strcpy(answer,
         " Les Houches HEPRUP common block");
         break;
         
       case MCFIO_STDHEPCXX:
         strcpy(answer,
         " StdHep::Event class, see StdHepC++ Product");
         break;
         
       case MCFIO_STDHEPBEG:
         strcpy(answer,
         " Stdhep begin run record, see STDHEP Product");
         break;
         
       case MCFIO_STDHEPEND:
         strcpy(answer,
         " Stdhep end run record, see STDHEP Product");
         break;
         
       case MCFIO_OFFTRACKARRAYS:
         strcpy(answer,
         " The mcfast Offline Tracks, saved into parallel arrays");
         break;
         
       case MCFIO_OFFTRACKSTRUCT:
         strcpy(answer,
         " The mcfast Offline Tracks, saved as the structure");
         break;
       default:          
         sprintf(answer, " Private User Block number %d ", blkId );
	 uDescr = mcfioC_UserBlockDescript(blkId);
	 if (uDescr == NULL) fprintf(stderr,
          "mcfio_GetBlockName: Warning Unrecognized block I.D. %d\n", blkId);
         else answer = uDescr;             
  }         
	
}



