/*******************************************************************************
*									       *
* mcf_NTuIOUtils.c -- Utilities to manipulate files within the MCFIO Gen.      *
*        				Ntuple schema                          *
*									       *
*	P. Lebrun, September 1995.					       *
*									       *
*******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/param.h>
#include <limits.h>
#include <rpc/types.h>
#include <sys/types.h>
#include <rpc/xdr.h>
#include "mcf_nTupleDescript.h"
#include "mcf_xdr.h"
#include "mcfio_Dict.h"
#include "mcf_NTuIOFiles.h"
#include "mcf_NTuIOUtils.h"
#include "mcf_ntubld_db.h"
#ifndef False
#define False 0
#endif
#ifndef True
#define True 1
#endif

extern nTuDDL **NTuDDLList;
extern int NumOfNTuples;

nTuDDL *mcf_GetNTuByPtrID(int id)
{
    int **ip;
    
    if ( (id < 1) || (id > NumOfNTuples)) return NULL;
    ip = (int **) NTuDDLList;
    ip += (id-1);
    return (nTuDDL *) *ip; 
}
     
nTuDDL *mcf_GetNTuByStreamID(int stream, int id)
{
     int i, num;
     nTuDDL *ddl;
     
     for (i=0, num=0; i<NumOfNTuples; i++) { 
          ddl = NTuDDLList[i];
          if ((ddl->streamId == stream) && (ddl->seqNTuId == id)) return ddl;
     }
     return NULL;
}   
int mcf_NTuId(int uid, char *category)
/* 
	uid		Unique User id 
	category	Category name, must be an exact match

	Returns:	Macfio_Ntuple id, or -1 if no items matched, or if 
			Category is illegal..
*/
{
	int i, j, **ip;
	nTuDDL *item;
	char *cat;
	
     if (!mcf_CheckValidCat(category, FALSE))  return -1;
     ip = (int **) NTuDDLList;
     cat = mcf_ValidStr(category, NTU_MAX_CATEGORY_LENGTH, "category");
     for (i=0; i< NumOfNTuples; i++, ip++) {
	 item = (nTuDDL *) *ip;
	 if (item->uid == uid) { /* Look first at uid, if match, */
	         		    /* Confirm with Category */
	    if ((category == NULL) && (item->category == NULL)) 
	      				return (item->id);
	    if (strcmp(category, item->category) == 0)
	    				return (item->id);
            j = strspn(category, " ");	
	    if (strcmp((category+j), item->category) == 0)
	                                return (item->id); 
	 }
     }
     return -1;
}

int mcfioC_GetNTupleIds(int stream, int *ids, int max)
{
     int i, num;
     nTuDDL *ddl;
     
     for (i=0, num=0; i<NumOfNTuples; i++) { 
          ddl = NTuDDLList[i];
          if (ddl->streamId == stream) {
              if (num < max ) ids[num] = ddl->id;
              num++;
          }
     }
     return num;
}   

int mcfioC_GetNTupleUID(int stream, int id)
{
   nTuDDL *ddl = mcf_GetNTuByStreamID(stream, id);
   return ddl->uid;
}

void mcfioC_GetNTupleCategory(int stream, int id, char **answer)
{
   nTuDDL *ddl = mcf_GetNTuByStreamID(stream, id);
   *answer = ddl->category;
}
 
void mcfioC_GetNTupleTitle(int stream, int id, char **answer)
{
   nTuDDL *ddl = mcf_GetNTuByStreamID(stream, id);
   *answer = ddl->title;
}

void mcfioC_GetNTupleName(int stream, int id, char **answer)
{
   nTuDDL *ddl = mcf_GetNTuByStreamID(stream, id);
   if (ddl->reference == NULL) 
       *answer = ddl->descrNtu->title;
   else *answer = ddl->reference->descrNtu->title;    
}

/*
** Copy utility routine for a General Ntuple Variable descriptor d/s
** It is the responsability of the usr to allocate memory for the 
** structure where data will be copied to.
*/           
void CopyVarGenNtuple(varGenNtuple *vFrom, varGenNtuple *vTo)
{
    char *string, *tc, *tc2;
    int i, ll;

    if ((vTo == NULL)  || (vFrom == NULL)) return;
    vTo->nameBlank = vFrom->nameBlank;
    if (vTo->name != NULL) {
        free(vTo->name);
        vTo->name = NULL;
    }
    if (vFrom->name != NULL) {
        ll = (1 + strlen(vFrom->name));
        vTo->name = 
           (char *) malloc(sizeof(char) * ll);
        strcpy(vTo->name, vFrom->name);
    }
    if (vTo->description != NULL) {
         free(vTo->description);
         vTo->description = NULL;
    }
    if (vFrom->description != NULL) {
        vTo->description = 
           (char *) malloc(sizeof(char) * (1 + strlen(vFrom->description)));
        strcpy(vTo->description, vFrom->description);
    }
    vTo->type = vFrom->type;
    vTo->isFixedSize = vFrom->isFixedSize;
    vTo->numDim = vFrom->numDim;
    if (vFrom->numDim > 0)  {
           for (i=0; i<vFrom->numDim; i++)
              vTo->dimensions[i] = vFrom->dimensions[i]; 
    }
    vTo->offset = vFrom->offset;
    vTo->offsetXDR = vFrom->offsetXDR;
}
/*
** insert this ddl into the Global List, expand the list if need be.
** Also increment the number of NTuples defined (don't do it twice!). 
*/
void AddNTuDDLtoList(nTuDDL *ddl)
{
    int i, **ipo;
    
    NumOfNTuples++;
    ddl->id = NumOfNTuples;
    /*
    ** insert this ddl into the Global List, expand the list if need be
    */
    if( (NumOfNTuples - (NumOfNTuples/NTU_START_LIST_SIZE)*NTU_START_LIST_SIZE)
                     == 1 && (NumOfNTuples != 1)) {
            ipo = (int **) NTuDDLList;
    	    NTuDDLList  = (nTuDDL **) malloc(sizeof(int *)*
              ((NumOfNTuples/NTU_START_LIST_SIZE + 1)*NTU_START_LIST_SIZE));
            memcpy(NTuDDLList, ipo, (sizeof(int *)*(NumOfNTuples-1)));
    	    free (ipo);
    }
    NTuDDLList[NumOfNTuples-1] = ddl;
            
}
/*
** Free the memory for a Ntuple Data Descrp. Lang (DDL).  
*/
void DestroyNTuDDL(nTuDDL *ddl)
{
   int i;
   if (ddl->title != NULL) free(ddl->title);
   if (ddl->category != NULL) free(ddl->category);
   if (ddl->dbinFileName != NULL) free(ddl->dbinFileName);
   DestroyGenNtuple(ddl->descrNtu);
   free(ddl);
} 
/*
** Free the memory for a Description NTuple
** Note : the pointer to adrresses are lost, the user will have to give 
** them to this application back..
*/
void DestroyGenNtuple(descrGenNtuple *dNTu)
{
    int i;

    if (dNTu == NULL) return;
    if (dNTu->title != NULL) free(dNTu->title);
    if (dNTu->description != NULL) free(dNTu->description);
    if (dNTu->varOrdering != NULL) free(dNTu->varOrdering);
    if (dNTu->subOffset != NULL) free(dNTu->subOffset);
    if (dNTu->subXDROffset != NULL) free(dNTu->subXDROffset);
    for (i=0; i<dNTu->numAvailable; i++)
         DestroyVarGenNtuple(dNTu->variables[i]);
    free(dNTu->variables);
    free(dNTu);     
}    


void DestroyVarGenNtuple(varGenNtuple *var)
{

    if (var == NULL) return;
    if (var->name != NULL) free(var->name);
    if (var->description != NULL) free(var->description);
    free(var);
}    
/*
 * ValidStr - Validate strings supplied by user
 *
 *	      returns: pointer to valid same or new truncated string
 *
 *  Note: ** copy string returned, if needed, before calling ValidStr again **
 */
char *mcf_ValidStr(char *string, int max_length, char *strKind)
{
    static char str[NTU_MAX_CATEGORY_LENGTH+1];	     /* make longest string */
    static char str1[1] = "";
    
    if (string == NULL)
    	return str1;			   /* return empty string	    */
    if (strlen(string) <= max_length)
    	return string;			   /* return pointer to same string */
    fprintf(stderr,
      "Mcfio_Ntuple: Error. Specified %s string is too long, truncating\n     ->%s\n",
    	 strKind, string);
    memset(str, 0, NTU_MAX_CATEGORY_LENGTH+1);
    return strncpy(str, string, max_length); /* return ptr to trunc. string */
}
/*
** Based on the HistoScope Check Category 
*/      
int mcf_CheckValidCat(char *category, int dotDotDot)
{
    static char validChars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ\
abcdefghijklmnopqrstuvwxyz1234567890/~!@#$%^&*()_+=-`\"\'\t?><,. ";
    char *strDots, *error = NULL;
    int len;
    
    if (category == NULL)
    	return 1;
    len = strlen(category);
    strDots = strstr(category, "...");
    if (len >= NTU_MAX_CATEGORY_LENGTH)
    	error = "is too long";
    else if (strspn(category, validChars) != len)
    	error = "contains invalid characters";
    else if (strstr(category, "//") != NULL)
    	error = "contains \"//\"";
    else if (category[0] == '/')
    	error = "contains leading slash";
    else if (category[len-1] == '/')
    	error = "contains trailing slash";
    else if ((dotDotDot == 0 && strDots != NULL) 
    	  || (dotDotDot != 0 && strDots != NULL && strDots != category + len-3))
    	error = "contains invalid \"...\"";
    	
    if (error != NULL) {
    	fprintf(stderr, "Error in declared category %s: %s\n",
    		error, category);
    	return 0;
    } else {
    	return (strDots == NULL ? 1 : -1);
    }
}

