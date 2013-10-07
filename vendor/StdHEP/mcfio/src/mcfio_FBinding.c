/*******************************************************************************
*									       *
* mcfio_FBinding.c --  Utility routines for the McFast Monte-Carlo               *
*		Fortran Application Interface.  	                       *
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
* The mallocNCopyMcfio and CleanFortranString have been borrowed from the           *
*  Nirvana project. 				     	                       *
*									       *
*******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/param.h>
#include <rpc/types.h>
#include <sys/types.h>
#include <rpc/xdr.h>
#include "mcf_nTupleDescript.h"
#include "mcf_xdr.h"
#include "mcfio_Dict.h"
#include "mcfio_Util1.h"
#include "mcfio_Direct.h"
#include "mcfio_Sequential.h"
#include "mcfio_Block.h"

char *mallocNCopyMcfio(char *string, int length);
static void cleanFortranString(char *string, int length);

void mcfio_init_(void)
{
     mcfioC_Init();
}

void mcfio_rewind_(int *stream)
{
     mcfioC_Rewind(*stream);
}

void mcfio_close_(int *stream)
{
     mcfioC_Close(*stream);
}

void mcfio_closesequentialfile_(int *stream)
{
     mcfioC_CloseSequentialFile(*stream);
}

void mcfio_closesequentialtape_(int *stream)
{
     mcfioC_CloseSequentialTape(*stream);
}

void mcfio_printdictionary_(void)
{
     mcfioC_PrintDictionary();
}

unsigned int mcfio_infonumstream_(int *istreams, int *nmax)
{
     return mcfioC_InfoNumStream(istreams, (unsigned int) *nmax);
}

void mcfio_infostreamint_(int *stream, int *key, int *values)
{
    mcfioC_InfoStreamInt(*stream, *key, values);
}

void mcfio_infostreamchar_(int *stream, int *key,
                           char *answer, int *lret, int length)
{
    char *aString;
    int ll1;
    /* Rely on the fact that the maximum char. length is probably FILENAME_MAX
       for information content, that is, 1024 on IRIX and other systems */
    aString = (char *)  malloc((FILENAME_MAX + 1) * sizeof(char));
    mcfioC_InfoStreamChar(*stream, *key, aString, lret);
    ll1 = *lret;
    if (ll1 > length) ll1 = length;
    strncpy(answer, aString, ll1); 
    free(aString);
}

void mcfio_infoeventint_(int *Event, int *key, int *values)
{
    mcfioC_InfoEventInt(*Event, *key, values);
}

void mcfio_infoeventchar_(int *Event, int *key, char *answer,
                                         int *lret, int length)
{
    char *aString;
    int ll1;
    
    aString = (char *)  malloc(sizeof(char) * (FILENAME_MAX + 1));
    mcfioC_InfoEventChar(*Event, *key, aString, lret);
    ll1 = *lret;
    if (ll1 > length) ll1 = length;
    strncpy(answer, aString, ll1); 
    free(aString);
}

void mcfio_seteventinfo_(int *Event, int *key, int *values)
{
    mcfioC_SetEventInfo(*Event, *key, values);
}

void mcfio_infoblockchar_(int *Event, int *blkId,
                          int *key, char *answer, int *lret, int length)
{
    char *aString;
    
    int ll1;
    
    aString = (char *)  malloc(sizeof(char) * (FILENAME_MAX + 1));
    mcfioC_InfoBlockChar(*Event, *blkId, *key, aString, lret);
    ll1 = *lret;
    if (ll1 > length) ll1 = length;
    strncpy(answer, aString, ll1); 
    free(aString);
}

void mcfio_getblockname_(int *blkId, char *answer, int length)
{
    char *aString;
    
    aString = (char *)  malloc(sizeof(char) * (FILENAME_MAX + 1));
    mcfioC_GetBlockName(*blkId, aString);
    strncpy(answer, aString, length); 
    free(aString);
}

void mcfio_defineuserblock_(int *blkId, char *name,
                   bool_t xdr_filter, int *current_size, int length)
{
    char *aString;
    
    aString = mallocNCopyMcfio(name, length);
    mcfioC_DefineUserBlock(*blkId, aString, xdr_filter, current_size);
    free(aString);
}

int mcfio_declarentuple_(int *uid, char*title, char *category,
                         int *stream, char *filename, int la, int lb, int lc)
{
    char *aString, *bString, *cString;
    int iret;
    
    aString = mallocNCopyMcfio(title, la);
    bString = mallocNCopyMcfio(category, lb);
    cString = mallocNCopyMcfio(filename, lc); 
    iret = mcfioC_DeclareNtuple(*uid, aString, bString, *stream, cString);
    free(aString); free(bString); free(cString);
    return iret;    
}

int mcfio_enddeclntuples_(int *stream)
{
   return mcfioC_EndDeclNTuples(*stream);
}


int mcfio_getntupleids_(int *stream, int *ids, int *max)
{    
    return mcfioC_GetNTupleIds(*stream, ids, *max);
}    

int mcfio_getntupleuid_(int *stream, int *id)
{    
    return mcfioC_GetNTupleUID(*stream, *id);
}
    
int mcfio_getntuplecategory_(int *stream, int *id, char *category, int ll)
{ 
    int iret, lret;
    char *aString;
    mcfioC_GetNTupleCategory(*stream, *id, &aString);
    lret = strlen(aString);
    if (ll > lret) strcpy(category, aString);
    else strncpy(category, aString, (ll-1));
    return lret;    
}    

int mcfio_getntupletitle_(int *stream, int *id, char *title, int ll)
{ 
    int iret, lret;
    char *aString;
    mcfioC_GetNTupleTitle(*stream, *id, &aString);
    lret = strlen(aString);
    if (ll > lret) strcpy(title, aString);
    else strncpy(title, aString, (ll-1));
    return lret;    
}    
   
int mcfio_getntuplename_(int *stream, int *id, char *name, int ll)
{ 
    int iret, lret;
    char *aString;
    mcfioC_GetNTupleName(*stream, *id, &aString);
    lret = strlen(aString);
    if (ll > lret) strcpy(name, aString);
    else strncpy(name, aString, (ll-1));
    return lret;    
}
    
int mcfio_openreaddirect_(char *filename, int length)
{
    char *aString;
    int iret;
    
    aString = mallocNCopyMcfio(filename, length);
    iret =  mcfioC_OpenReadDirect(aString);
    if (aString != NULL) free(aString);
    return iret;
}

int mcfio_openreadmapped_(char *filename, int length)
{
    char *aString;
    int iret;
    
    aString = mallocNCopyMcfio(filename, length);
    iret =  mcfioC_OpenReadMapped(aString);
    if (aString != NULL) free(aString);
    return iret;
}

int mcfio_openreadsequential_(char *device, char *vsn, 
                                 int *filenumber, int l1, int l2)
{
    char *aString, *bString;
    int iret;
    
    aString = mallocNCopyMcfio(device, l1);
    bString = mallocNCopyMcfio(vsn,l2);
    iret =  mcfioC_OpenReadSequential(aString, bString, *filenumber);
    if (aString != NULL) free(aString);
    if (bString != NULL) free(bString);
    return iret;
}

void mcfio_setforsavedecoding_(int *value)
{
  mcfioC_SetForSaveDecoding(*value);
}     

int mcfio_openwritedirect_(char *filename, char *title, char *comment, 
                           int *numevts_pred, int *blkIds, int *nBlocks,
                           int ll1, int ll2, int ll3)
{
    char *aString, *bString, *cString;
    int iret;
    
    aString = mallocNCopyMcfio(filename, ll1);
    bString = mallocNCopyMcfio(title, ll2);
    cString = mallocNCopyMcfio(comment, ll3);
    iret =  mcfioC_OpenWriteDirect(aString, bString, cString, 
                     *numevts_pred, blkIds, (unsigned int) *nBlocks);
    if (aString != NULL) free(aString); 
    if (bString != NULL) free(bString);
    if (cString != NULL) free(cString);
    return iret;
}
                 
int mcfio_openwritesequential_(char *device, char *vsn, char *title, 
                           char *comment, int *numevts_pred, int *blkIds, 
                           int *nBlocks, int ll1, int ll2, int ll3, int ll4)
{
    char *aString, *bString, *cString, *dString;
    int iret;
    
    aString = mallocNCopyMcfio(device, ll1);
    bString = mallocNCopyMcfio(vsn, ll2);
    cString = mallocNCopyMcfio(title, ll3);
    dString = mallocNCopyMcfio(comment, ll4);
    iret =  mcfioC_OpenWriteSequential(aString, bString, cString, dString, 
                     *numevts_pred, blkIds, (unsigned int) *nBlocks);
    if (aString != NULL) free(aString); 
    if (bString != NULL) free(bString);
    if (cString != NULL) free(cString);
    if (dString != NULL) free(dString); 
     return iret;
}

int mcfio_nextevent_(int *stream)
{
    return mcfioC_NextEvent(*stream);
}    

int mcfio_specificevent_(int *stream, int * ievt,
                             int *istore, int *irun, int *itrig)
{
    return mcfioC_SpecificEvent(*stream, *ievt, *istore, *irun, *itrig);
}
    
int mcfio_nextspecificevent_(int *stream,int * ievt,
                             int *istore, int *irun, int *itrig)
{
    return mcfioC_NextSpecificEvent(*stream, 
                                     *ievt, *istore, *irun, *itrig);
}

int mcfio_block_(int *stream, int *blkid,
  bool_t xdr_filtercode(XDR *xdrs, int *blockid, int *ntot, char **version))
{
    return mcfioC_Block(*stream, *blkid, xdr_filtercode);
}

int mcfio_ntuple_(int *stream, int *ntupleid, char *version, int ll)
{
/*
** Note we do not copy the string this time, as we are interested in the 
** address, not the content
*/
    return mcfioC_NTuple(*stream, *ntupleid, version);
}
    
     
int mcfio_ntuplemult_(int *stream, int *ntupleid, char *version, int ll)
{
    return mcfioC_NTupleMult(*stream, *ntupleid, version);
}

int mcfio_ntuplevar_(int *stream, int *ntupleid, int *ivar, 
                       char *version, int ll)
{
   int ivarF;
   ivarF = (*ivar) -1;
    return mcfioC_NTupleVar(*stream, *ntupleid, ivarF, version);
}

int mcfio_ntuplesubvar_(int *stream, int *ntupleid,
                        int *ivar, int *multIndex, char *version, int ll)
{
   int ivarF, multIndexF;
   ivarF = (*ivar) -1;
   multIndexF = (*multIndex) -1;
    return mcfioC_NTupleSubVar(*stream, *ntupleid,
                                 ivarF, multIndexF, version);
}

int mcfio_ntuplesubstruct_(int *stream, int *ntupleid,
                           int *multIndex, char *version, int ll)
{
   int multIndexF;
   multIndexF = (*multIndex) -1;
   return mcfioC_NTupleSubStruct(*stream, *ntupleid, 
                                  multIndexF, version);
}


char *mallocNCopyMcfio(char *string, int length)
{
    char *str;
    
    if (string == NULL) return NULL;
    if (length < 0) {
    	fprintf(stderr, 
    	"hs: Error. Character argument has negative length, setting to null\n");
    	length = 0;
    }
    str = malloc(length + 1);
    if (length > 0)
    	strncpy(str, string, length);
    str[length] = '\0';
    /* printf(" string = /%s/, length: %d (before)\n", str, strlen(str)); */
    cleanFortranString(str, length);
    /* printf(" string = /%s/, length: %d (cleaned)\n", str, strlen(str)); */
    return str;
}


/*
** clean junk out of fortran strings that might not have been
** fully initialized or have trailing blanks.  Assumes enough
** room to append a null at the end of the string.
**
** This cleanFortranString continues after detecting a junk character and
** includes a few more valid characters than other cleanFortranString's.
*/
static void cleanFortranString(char *string, int length)
{
    static char validChars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ\
abcdefghijklmnopqrstuvwxyz1234567890/~!@#$%^&*()_+=-`\"\'?><,.\\[]{}:; \t";
    char *inPtr, *outPtr;
    int i;
    
    /* remove junk characters */
    inPtr = outPtr = string;
    for (i=1; i<=length; i++) {
    	if (strchr(validChars, *inPtr))
    	    *outPtr++ = *inPtr++;
    	else
    	    inPtr++;
    }
    
    /* remove trailing spaces */
    for (outPtr--; outPtr>=string; outPtr--)
    	if (*outPtr != ' ' && *outPtr != '\0')
    	    break;
    
    /* add a null to terminate the string */
    ++outPtr;
    *outPtr = '\0';
}


