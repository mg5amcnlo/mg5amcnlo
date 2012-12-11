/*******************************************************************************
*									       *
* mcf_stdcm1_xdr.c -- XDR Utility routines for the Block stdcm1 filters        *
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
* Written by Paul Lebrun, Lynn Garren					       *
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
#include "mcfio_Dict.h"
#include "mcf_xdr.h"
#include "stdcm1.h"
#include "stdver.h"
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

struct stdver stdver_;
struct stdcm1 stdcm1_;
struct stdcm2 stdcm2_;

bool_t xdr_stdhep_cm1_(XDR *xdrs, int *blockid,
 				 int *ntot, char** version)

{
/*  Translate the STDCM1 COMMON block from the STDHEP package to/from
    an XDR stream. Note that we do not allocate memory, because we fill
    directly the COMMON.  Also, mcfio will allocate the space for the 
    string version.  */
    
    unsigned int nn, nn1, nn2;
    int *idat;
    double *dat;
    char *vers;
    char *cdat;
    
    if ((xdrs->x_op == XDR_ENCODE) || (xdrs->x_op == XDR_MCFIOCODE))  {
       if (( *blockid != MCFIO_STDHEPEND)&&( *blockid != MCFIO_STDHEPBEG)) {
          fprintf (stderr, "mcf_Stdhep_cm1_xdr: Inconsistent Blockid %d \n ", 
           (*blockid));
          return FALSE;
       }
       nn = sizeof(int) * stdcm1_.nevtreq;
       nn1 = sizeof(float) * stdcm1_.nevtreq;
       nn2 = sizeof(double) * stdcm1_.nevtreq;
       *ntot = 3 * sizeof(int) + 3 * nn + 2 * nn1 +  2 * nn2
               + 2 * sizeof(char) * ( MCF_XDR_STDCM2_LENGTH + 1 );
       if (xdrs->x_op == XDR_MCFIOCODE) return TRUE;
       strncpy(version[0],stdver_.stdhep_ver, 4);
     } 
      
     if     ( (xdr_int(xdrs, blockid) &&
     	      xdr_int(xdrs, ntot) &&
     	      xdr_string(xdrs, version, MCF_XDR_VERSION_LENGTH) )
              == FALSE) return FALSE;
              
     if ((xdrs->x_op == XDR_DECODE) && 
          (( *blockid != MCFIO_STDHEPEND)&&( *blockid != MCFIO_STDHEPBEG))) {
          fprintf (stderr, "mcf_Stdhep_cm1_xdr: Inconsistent Blockid %d \n ", 
           (*blockid));
          return FALSE;
     }
     if ( xdr_int(xdrs, &(stdcm1_.nevtreq) ) == FALSE) return FALSE;
     if ( xdr_int(xdrs, &(stdcm1_.nevtgen) ) == FALSE) return FALSE;
     if ( xdr_int(xdrs, &(stdcm1_.nevtwrt) ) == FALSE) return FALSE;
     if ( xdr_float(xdrs, &(stdcm1_.stdecom) ) == FALSE) return FALSE;
     if ( xdr_float(xdrs, &(stdcm1_.stdxsec) ) == FALSE) return FALSE;
     if ( xdr_double(xdrs, &(stdcm1_.stdseed1) ) == FALSE) return FALSE;
     if ( xdr_double(xdrs, &(stdcm1_.stdseed2) ) == FALSE) return FALSE;
     /*
     ** V5.01 Upgrade : adding stdcm2 
     */ 
     vers = *version;          
     if (((strcmp(vers,"1.") == 0) || (strcmp(vers,"2.") == 0) ||
          (strcmp(vers,"3.") == 0) || (strcmp(vers,"4.") == 0) ||
	  (strcmp(vers,"5.00") == 0) ) && (xdrs->x_op == XDR_DECODE)) {
	   strncpy(stdcm2_.generatorname, " ", MCF_XDR_STDCM2_LENGTH);
	   strncpy(stdcm2_.pdfname, " ", MCF_XDR_STDCM2_LENGTH);
           return TRUE;
     }
/* 
  allocate memory and deal with encoding and decoding separately
*/
     cdat = malloc(MCF_XDR_STDCM2_LENGTH+1);
     if( (xdrs->x_op == XDR_DECODE) ) {
	 strncpy(stdcm2_.generatorname, " ", MCF_XDR_STDCM2_LENGTH);
	 strncpy(stdcm2_.pdfname, " ", MCF_XDR_STDCM2_LENGTH);
	 cdat = NULL;
	 if ( xdr_string(xdrs, &cdat, MCF_XDR_STDCM2_LENGTH+1 ) == FALSE) return FALSE;
         strncpy(stdcm2_.generatorname,cdat,MCF_XDR_STDCM2_LENGTH);
	 cdat = NULL;
	 if ( xdr_string(xdrs, &cdat, MCF_XDR_STDCM2_LENGTH+1 ) == FALSE) return FALSE;
         strncpy(stdcm2_.pdfname,cdat,MCF_XDR_STDCM2_LENGTH);
     } else {
	 strncpy(cdat, stdcm2_.generatorname, MCF_XDR_STDCM2_LENGTH);
         /* some compilers do not properly append the null terminator */
	 cdat[MCF_XDR_STDCM2_LENGTH]='\0';
	 if ( xdr_string(xdrs, &cdat, MCF_XDR_STDCM2_LENGTH+1 ) == FALSE) return FALSE;
	 strncpy(cdat, stdcm2_.pdfname, MCF_XDR_STDCM2_LENGTH);
	 cdat[MCF_XDR_STDCM2_LENGTH]='\0';
	 if ( xdr_string(xdrs, &cdat, MCF_XDR_STDCM2_LENGTH+1 ) == FALSE) return FALSE;
     }
     free(cdat);

     /*
     ** V5.02 Upgrade : add nevtlh to stdcm1 
     ** note that we cannot get here unless the version is 5.00 or greater
     */ 
     if (((strcmp(vers,"5.00") == 0) || (strcmp(vers,"5.01") == 0))
	  && (xdrs->x_op == XDR_DECODE)) {
	   stdcm1_.nevtlh = 0;
           return TRUE;
     }
     if ( xdr_int(xdrs, &(stdcm1_.nevtlh) ) == FALSE) return FALSE;

     return TRUE;
}   

