/*
** A small container to hold a set of user block declaration
**
* Written by Paul Lebrun, Aug 2001
*/
#include <stdio.h>
#include <string.h>
#include <sys/param.h>
#include <rpc/types.h>
#include <sys/types.h>
#include <rpc/xdr.h>
#include <limits.h>
#include <stdlib.h>
#include "mcfio_UserDictionary.h"

#define NUMUSERBLOCKDEFAULT 100

allMCFIO_UserBlockDecl *AllMCFIO_UserBlockDecl = NULL;


char *mcfioC_UserBlockDescript(int blkn)
{
   int i;
   if (AllMCFIO_UserBlockDecl == NULL) return NULL;
   for (i=0; i<AllMCFIO_UserBlockDecl->num; i++) {
     if (AllMCFIO_UserBlockDecl->decls[i]->blkNum == blkn) 
        return AllMCFIO_UserBlockDecl->decls[i]->title;
   }
   return NULL;
}

void mcfioC_DefineUserBlock(int blkN, char *descr){
   int i;
   aUserBlockDecl *abd;
   
   if (AllMCFIO_UserBlockDecl == NULL) {
   
      AllMCFIO_UserBlockDecl = (allMCFIO_UserBlockDecl *) malloc (
                             sizeof(allMCFIO_UserBlockDecl));
      AllMCFIO_UserBlockDecl->numPreAlloc = NUMUSERBLOCKDEFAULT;
      AllMCFIO_UserBlockDecl->num = 0;
      AllMCFIO_UserBlockDecl->decls = (aUserBlockDecl **) malloc( 
                            NUMUSERBLOCKDEFAULT * sizeof(aUserBlockDecl *));
   }
   if (AllMCFIO_UserBlockDecl->num == AllMCFIO_UserBlockDecl->numPreAlloc) {
       AllMCFIO_UserBlockDecl->numPreAlloc += NUMUSERBLOCKDEFAULT;
       AllMCFIO_UserBlockDecl->decls = 
	(aUserBlockDecl **) realloc (((void *) AllMCFIO_UserBlockDecl->decls), 
	   (AllMCFIO_UserBlockDecl->numPreAlloc  * sizeof(aUserBlockDecl *)));
   }
   AllMCFIO_UserBlockDecl->decls[AllMCFIO_UserBlockDecl->num] = 
     (aUserBlockDecl *) malloc (sizeof(aUserBlockDecl));
   abd = AllMCFIO_UserBlockDecl->decls[AllMCFIO_UserBlockDecl->num];
   AllMCFIO_UserBlockDecl->num++;
   abd->blkNum = blkN;
   abd->title = (char *) malloc (sizeof(char) * (strlen(descr) + 1));
   strcpy(abd->title, descr);
}    
