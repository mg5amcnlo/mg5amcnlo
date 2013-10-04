/*******************************************************************************
*									       *
* mcf_Stdhep_xdr.c -- XDR Utility routines for the McFast Monte-Carlo	       *
*				 filters for Block Stdhep.		       *
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
#include "mcfio_Dict.h"
#include "mcf_xdr.h"
#include "stdhep.h"
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

bool_t xdr_mcfast_stdhep_(XDR *xdrs, int *blockid,
 				 int *ntot, char** version)

{
/*  Translate the HEPEVT COMMON block from the STDHEP package to/from
    an XDR stream. Note that we do not allocate memory, because we fill
    directly the COMMON.  Also, the mcfio will allocate the space for the 
    string version.  */
    
    unsigned int nn, nn2, nn4, nn5;
    int *idat;
    float *dat;
    
    if ((xdrs->x_op == XDR_ENCODE) || (xdrs->x_op == XDR_MCFIOCODE)) {
       if (*blockid != MCFIO_STDHEP) {
          fprintf (stderr, "mcf_Stdhep_xdr: Inconsistent Blockid %d \n ", 
           (*blockid));
          return FALSE;
       }
       nn = (unsigned int) hepevt_.nhep;
       nn2 = 2 *(unsigned int) hepevt_.nhep;
       nn4 = 4 *(unsigned int) hepevt_.nhep;
       nn5 = 5 *(unsigned int) hepevt_.nhep;
       *ntot = 20 + 4 * (2 * nn + 2* nn2 + nn4 + nn5);
       if (xdrs->x_op == XDR_MCFIOCODE) return TRUE;
       strcpy(*version, "1.05");
       } 
      
     if     ( (xdr_int(xdrs, blockid) &&
     	      xdr_int(xdrs, ntot) &&
     	      xdr_string(xdrs, version, 4) &&
     	      xdr_int(xdrs, &(hepevt_.nevhep)) &&
              xdr_int(xdrs, &(hepevt_.nhep))) == FALSE) return FALSE;
              
     if ((xdrs->x_op == XDR_DECODE) && ( *blockid != MCFIO_STDHEP) ) {
          fprintf (stderr, "mcf_Stdhep_xdr: Inconsistent Blockid %d \n ", 
           (*blockid));
          return FALSE;
     }
     idat = hepevt_.isthep;
     if     ( xdr_array(xdrs, (char **)  &idat,
                &nn, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = hepevt_.idhep;
     if     ( xdr_array(xdrs, (char **)   &idat,
                &nn, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = (int *) hepevt_.jmohep;
     if     ( xdr_array(xdrs, (char **)   &idat,
                &nn2, 2*NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = (int *) hepevt_.jdahep;
     if     ( xdr_array(xdrs,  (char **)   &idat,
                &nn2, 2*NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     dat = (float *)  hepevt_.phep;
     if     ( xdr_array(xdrs,  (char **)   &dat,
                &nn5, 5*NMXHEP, sizeof(int), xdr_float) == FALSE) return FALSE; 
     dat = (float *) hepevt_.vhep;
     if     ( xdr_array(xdrs, (char **)    &dat,
                &nn4, 4*NMXHEP, sizeof(int), xdr_float) == FALSE) return FALSE;
     return TRUE;
}   

