/*******************************************************************************
*									       *
* mcf_Stdhep_xdr.c -- XDR Utility routines for the Block Stdhep filters        *
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
#include "stdhep.h"
#include "stdtmp.h"
#include "stdver.h"
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

struct stdver stdver_;

bool_t xdr_stdhep_(XDR *xdrs, int *blockid,
 				 int *ntot, char** version)

{
/*  Translate the HEPEVT temporary COMMON block from the STDHEP package to/from
    an XDR stream. Note that we do not allocate memory, because we fill
    directly the COMMON.  Also, mcfio will allocate the space for the 
    string version.  */
    
    unsigned int nn, nn2, nn4, nn5, nnw, nnw2, nnw4, nnw5;
    int *idat;
    double *dat;
    
    if ((xdrs->x_op == XDR_ENCODE) || (xdrs->x_op == XDR_MCFIOCODE))  {
       if (*blockid != MCFIO_STDHEP) {
          fprintf (stderr, "mcf_Stdhep_xdr: Inconsistent Blockid %d \n ", 
           (*blockid));
          return FALSE;
       }

       nn =  (unsigned int) stdtmp_.nhept;   /* Number of elements in isthep or idhep     */
       nn2 = 2*(unsigned int) stdtmp_.nhept; /* Number of elements in jmohep or jdahep    */
       nn4 = 4*(unsigned int) stdtmp_.nhept; /* Number of elements in vhep                */
       nn5 = 5*(unsigned int) stdtmp_.nhept; /* Number of elements in phep                */
       nnw = (unsigned int) stdtmp_.nhept;
       nnw2 = 2 * nnw;
       nnw4 = 4 * nnw;
       nnw5 = 5 * nnw;

       /* Total length in bytes include blockid, ntot, version, nevhept and nhept as well
          as the arrays remembering doubles are longer than ints.                         */
       *ntot = 5*sizeof(int) + sizeof(int)*(2*nn + 2*nn2) + sizeof(double)*(nn4 + nn5);

       if (xdrs->x_op == XDR_MCFIOCODE) return TRUE;
       strncpy(version[0],stdver_.stdhep_ver, 4);
       } 
      
     if     ( (xdr_int(xdrs, blockid) &&
     	      xdr_int(xdrs, ntot) &&
     	      xdr_string(xdrs, version, MCF_XDR_VERSION_LENGTH) &&
     	      xdr_int(xdrs, &(stdtmp_.nevhept)) &&
              xdr_int(xdrs, &(stdtmp_.nhept))) == FALSE) return FALSE;
              
     if ((xdrs->x_op == XDR_DECODE) && ( *blockid != MCFIO_STDHEP) ) {
          fprintf (stderr, "mcf_Stdhep_xdr: Inconsistent Blockid %d \n ", 
           (*blockid));
          return FALSE;
     }
     idat = stdtmp_.isthept;
     if     ( xdr_array(xdrs, (char **)  &idat,
         &nnw, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = stdtmp_.idhept;
     if     ( xdr_array(xdrs, (char **)   &idat,
         &nnw, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = (int *) stdtmp_.jmohept;
     if     ( xdr_array(xdrs, (char **)   &idat,
                &nnw2, 2*NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = (int *) stdtmp_.jdahept;
     if     ( xdr_array(xdrs,  (char **)   &idat,
                &nnw2, 2*NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     dat = (double *)  stdtmp_.phept;
     if     ( xdr_array(xdrs,  (char **)   &dat,
                &nnw5, 5*NMXHEP, sizeof(double), xdr_double) == FALSE) return FALSE; 
     dat = (double *) stdtmp_.vhept;
     if     ( xdr_array(xdrs, (char **)    &dat,
                &nnw4, 4*NMXHEP, sizeof(double), xdr_double) == FALSE) return FALSE;
     return TRUE;
}   

bool_t xdr_stdhep_multi_(XDR *xdrs, int *blockid,
 				 int *ntot, char** version)

{
/*  Translate the HEPEVT COMMON block from the STDHEP package to/from
    an XDR stream. Note that we do not allocate memory, because we fill
    directly the COMMON.  Also, mcfio will allocate the space for the 
    string version. 
    Also translate the HEPEV2 COMMON block from the STDHEP package to/from
    an XDR stream. HEPEV2 contains multiple interaction information */
    
    unsigned int nn, nn2, nn4, nn5, nnw, nnw2, nnw4, nnw5, nmlt, nnmlt;
    int i;
    int *idat;
    char *vers;
    double *dat;
    
    if ((xdrs->x_op == XDR_ENCODE) || (xdrs->x_op == XDR_MCFIOCODE))  {
       if (*blockid != MCFIO_STDHEPM) {
          fprintf (stderr, "mcf_Stdhep_xdr: Inconsistent Blockid %d \n ", 
           (*blockid));
          return FALSE;
       }
       nn = sizeof(int) * hepevt_.nhep;
       nn2 = 2 * sizeof(int) * hepevt_.nhep;
       nn4 = 4 * sizeof(double) * hepevt_.nhep;
       nn5 = 5 * sizeof(double) * hepevt_.nhep;
       nmlt = sizeof(int) * hepev2_.nmulti;
       nnw = (unsigned int) hepevt_.nhep;
       nnw2 = 2 * nnw;
       nnw4 = 4 * nnw;
       nnw5 = 5 * nnw;
       nnmlt = (unsigned int) hepev2_.nmulti;
       *ntot = 6 * sizeof(int) + 3 * nn + 2 * nn2 + nn4 + nn5 + 3 * nmlt;
       if (xdrs->x_op == XDR_MCFIOCODE) return TRUE;
       strncpy(version[0],stdver_.stdhep_ver, 4);
       } 
      
     if     ( (xdr_int(xdrs, blockid) &&
     	      xdr_int(xdrs, ntot) &&
     	      xdr_string(xdrs, version, MCF_XDR_VERSION_LENGTH) &&
     	      xdr_int(xdrs, &(hepevt_.nevhep)) &&
              xdr_int(xdrs, &(hepevt_.nhep))) == FALSE) return FALSE;
              
     if ((xdrs->x_op == XDR_DECODE) && ( *blockid != MCFIO_STDHEPM) ) {
          fprintf (stderr, "mcf_Stdhep_xdr: Inconsistent Blockid %d \n ", 
           (*blockid));
          return FALSE;
     }
     idat = hepevt_.isthep;
     if     ( xdr_array(xdrs, (char **)  &idat,
            &nnw, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = hepevt_.idhep;
     if     ( xdr_array(xdrs, (char **)   &idat,
            &nnw, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = (int *) hepevt_.jmohep;
     if     ( xdr_array(xdrs, (char **)   &idat,
                &nnw2, 2*NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = (int *) hepevt_.jdahep;
     if     ( xdr_array(xdrs,  (char **)   &idat,
                &nnw2, 2*NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     dat = (double *)  hepevt_.phep;
     if     ( xdr_array(xdrs,  (char **)   &dat,
                &nnw5, 5*NMXHEP, sizeof(double), xdr_double) == FALSE) return FALSE; 
     dat = (double *) hepevt_.vhep;
     if     ( xdr_array(xdrs, (char **)    &dat,
                &nnw4, 4*NMXHEP, sizeof(double), xdr_double) == FALSE) return FALSE;
     /*
     ** V2.02 Upgrade : adding Multiple interactions. 
     */ 
     vers = *version;          
     if ((strcmp(vers,"1.05") == 0)  && (xdrs->x_op == XDR_DECODE)) {
           hepev2_.nmulti = -1;
           return TRUE;
     }      
     if ( xdr_int(xdrs, &(hepev2_.nmulti)) == FALSE) return FALSE;
     idat = hepev2_.jmulti;
     if     ( xdr_array(xdrs, (char **)   &idat,
         &nnw, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     /*
     ** V4.04 Upgrade : adding more Multiple interaction information
     */ 
     if (((strcmp(vers,"2.") > 0) || (strcmp(vers,"3.") > 0)) 
            && (xdrs->x_op == XDR_DECODE)) {
	   for (i = 0; i <= NMXMLT; i++) {
	       hepev3_.nevmulti[i] = 0;
	       hepev3_.itrkmulti[i] = 0;
	       hepev3_.mltstr[i] = 0;
	   }
           return TRUE;
     }      
     if (((strcmp(vers,"4.00") == 0) || (strcmp(vers,"4.01") == 0) ||
          (strcmp(vers,"4.02") == 0) || (strcmp(vers,"4.03") == 0) )  
            && (xdrs->x_op == XDR_DECODE)) {
	   for (i = 0; i <= NMXMLT; i++) {
	       hepev3_.nevmulti[i] = 0;
	       hepev3_.itrkmulti[i] = 0;
	       hepev3_.mltstr[i] = 0;
	   }
           return TRUE;
     }      
     idat = hepev3_.nevmulti;
     if     ( xdr_array(xdrs, (char **)   &idat,
            &nnmlt, NMXMLT, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = hepev3_.itrkmulti;
     if     ( xdr_array(xdrs, (char **)   &idat,
            &nnmlt, NMXMLT, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = hepev3_.mltstr;
     if     ( xdr_array(xdrs, (char **)   &idat,
            &nnmlt, NMXMLT, sizeof(int), xdr_int) == FALSE) return FALSE;
     return TRUE;
}   


