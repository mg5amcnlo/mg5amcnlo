/*******************************************************************************
*									       *
* mcf_StdHep_cxx.c -- XDR Utility routines for the Block Stdhep filters        *
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
* Written by Lynn Garren					       *
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
#include "stdevent.h"
#include "stdver.h"
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

struct stdver stdver_;

bool_t xdr_stdhep_cxx_(XDR *xdrs, int *blockid,
 				 int *ntot, char** version)

{
/*  Translate a copy of the StdHep::Event class to/from
    an XDR stream. Note that we do not allocate memory, because we fill
    directly the COMMON.  Also, mcfio will allocate the space for the 
    string version.  */
    
    unsigned int nn, nnint, nndbl, nnw, nnw4, nnw5, nmlt, nnmlt;
    int i;
    int *idat;
    double *dat;
    
    if ((xdrs->x_op == XDR_ENCODE) || (xdrs->x_op == XDR_MCFIOCODE))  {
	if (*blockid != MCFIO_STDHEPCXX) {
           fprintf (stderr, "mcf_StdHep_cxx: Inconsistent Blockid %d \n ", 
            (*blockid));
           return FALSE;
	}
	nnint = 8 * sizeof(int) * stdevent_.nhep;
	nndbl = 16 * sizeof(double) * stdevent_.nhep;
	nmlt = 3 * sizeof(int) * stdevent_.ncol;
	nnw = (unsigned int) stdevent_.nhep;
	nnw4 = 4 * nnw;
	nnmlt = (unsigned int) stdevent_.ncol;
	*ntot = 6 * sizeof(int) + nnint + nndbl + nmlt;
	if (xdrs->x_op == XDR_MCFIOCODE) return TRUE;
	strncpy(version[0],stdver_.stdhep_ver, 4);
     } 
      
     if     ( (xdr_int(xdrs, blockid) &&
     	      xdr_int(xdrs, ntot) &&
     	      xdr_string(xdrs, version, MCF_XDR_VERSION_LENGTH) &&
     	      xdr_int(xdrs, &(stdevent_.nevhep)) &&
     	      xdr_int(xdrs, &(stdevent_.nhep)) &&
              xdr_int(xdrs, &(stdevent_.ncol))) == FALSE) return FALSE;
              
     if ((xdrs->x_op == XDR_DECODE) && ( *blockid != MCFIO_STDHEPCXX) ) {
          fprintf (stderr, "mcf_StdHep_cxx: Inconsistent Blockid %d \n ", 
           (*blockid));
          return FALSE;
     }
     idat = stdevent_.nevcol;
     if     ( xdr_array(xdrs, (char **)   &idat,
            &nnmlt, NMXMLT, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = stdevent_.iostr;
     if     ( xdr_array(xdrs, (char **)   &idat,
            &nnmlt, NMXMLT, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = stdevent_.nhepcol;
     if     ( xdr_array(xdrs, (char **)   &idat,
            &nnmlt, NMXMLT, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = stdevent_.isthep;
     if     ( xdr_array(xdrs, (char **)  &idat,
            &nnw, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = stdevent_.idhep;
     if     ( xdr_array(xdrs, (char **)   &idat,
            &nnw, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = stdevent_.jmohep1;
     if     ( xdr_array(xdrs, (char **)   &idat,
                &nnw, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = stdevent_.jmohep2;
     if     ( xdr_array(xdrs, (char **)   &idat,
                &nnw, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = stdevent_.jdahep1;
     if     ( xdr_array(xdrs, (char **)   &idat,
                &nnw, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = stdevent_.jdahep2;
     if     ( xdr_array(xdrs, (char **)   &idat,
                &nnw, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = stdevent_.color;
     if     ( xdr_array(xdrs, (char **)   &idat,
                &nnw, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = stdevent_.jcol;
     if     ( xdr_array(xdrs, (char **)   &idat,
                &nnw, NMXHEP, sizeof(int), xdr_int) == FALSE) return FALSE;
     dat = (double *)  stdevent_.phep;
     if     ( xdr_array(xdrs,  (char **)   &dat,
                &nnw4, 4*NMXHEP, sizeof(double), xdr_double) == FALSE) return FALSE; 
     dat = stdevent_.mass;
     if     ( xdr_array(xdrs,  (char **)   &dat,
                &nnw, NMXHEP, sizeof(double), xdr_double) == FALSE) return FALSE; 
     dat = stdevent_.helicity;
     if     ( xdr_array(xdrs,  (char **)   &dat,
                &nnw, NMXHEP, sizeof(double), xdr_double) == FALSE) return FALSE; 
     dat = (double *) stdevent_.vcr;
     if     ( xdr_array(xdrs, (char **)    &dat,
                &nnw4, 4*NMXHEP, sizeof(double), xdr_double) == FALSE) return FALSE;
     dat = (double *) stdevent_.vdcy;
     if     ( xdr_array(xdrs, (char **)    &dat,
                &nnw4, 4*NMXHEP, sizeof(double), xdr_double) == FALSE) return FALSE;
     if ( xdr_int(xdrs, &(stdevent_.ncol)) == FALSE) return FALSE;

     return TRUE;
}   
