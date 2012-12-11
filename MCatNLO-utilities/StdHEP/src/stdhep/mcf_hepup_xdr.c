/*******************************************************************************
*									       *
* mcf_hepup_xdr.c -- XDR Utility routines for the Block Stdhep filters        *
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
#include "hepeup.h"
#include "heprup.h"
#include "stdver.h"
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

struct stdver stdver_;

bool_t xdr_hepeup_(XDR *xdrs, int *blockid,
 				 int *ntot, char** version)

{
/*  Translate the HEPEUP COMMON block from the STDHEP package to/from
    an XDR stream. Note that we do not allocate memory, because we fill
    directly the COMMON.  Also, mcfio will allocate the space for the 
    string version.  */
    
    unsigned int nn, nn2, nn5;
    int *idat;
    double *dat;
    
    if ((xdrs->x_op == XDR_ENCODE) || (xdrs->x_op == XDR_MCFIOCODE))  {
       if (*blockid != MCFIO_HEPEUP) {
          fprintf (stderr, "mcf_hepup_xdr: Inconsistent Blockid %d \n ", 
           (*blockid));
          return FALSE;
       }

       nn =  (unsigned int) hepeup_.nup;   /* Number of elements in idup, istup, vtimup, spinup */
       nn2 = 2*(unsigned int) hepeup_.nup; /* Number of elements in mothup, icolup    */
       nn5 = 5*(unsigned int) hepeup_.nup; /* Number of elements in pup               */

       /* Total length in bytes include blockid, ntot, version, as well
          as the common block entries                                     */
       *ntot = sizeof(int)*(5 + 2*nn + 2*nn2) + sizeof(double)*(4 + 2*nn + nn5);

       if (xdrs->x_op == XDR_MCFIOCODE) return TRUE;
       strncpy(version[0],stdver_.stdhep_ver, 4);
       } 
      
     if     ( (xdr_int(xdrs, blockid) &&
     	      xdr_int(xdrs, ntot) &&
     	      xdr_string(xdrs, version, MCF_XDR_VERSION_LENGTH) &&
     	      xdr_int(xdrs, &(hepeup_.nup)) &&
              xdr_int(xdrs, &(hepeup_.idprup))) == FALSE) return FALSE;
              
     if ((xdrs->x_op == XDR_DECODE) && ( *blockid != MCFIO_HEPEUP) ) {
          fprintf (stderr, "mcf_hepup_xdr: Inconsistent Blockid %d \n ", 
           (*blockid));
          return FALSE;
     }

     if ( xdr_double(xdrs, &(hepeup_.xwgtup) ) == FALSE) return FALSE;
     if ( xdr_double(xdrs, &(hepeup_.scalup) ) == FALSE) return FALSE;
     if ( xdr_double(xdrs, &(hepeup_.aqedup) ) == FALSE) return FALSE;
     if ( xdr_double(xdrs, &(hepeup_.aqcdup) ) == FALSE) return FALSE;
     idat = hepeup_.idup;
     if     ( xdr_array(xdrs, (char **)  &idat,
         &nn, MAXNUP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = hepeup_.istup;
     if     ( xdr_array(xdrs, (char **)  &idat,
         &nn, MAXNUP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = hepeup_.mothup;
     if     ( xdr_array(xdrs, (char **)  &idat,
         &nn2, 2*MAXNUP, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = hepeup_.icolup;
     if     ( xdr_array(xdrs, (char **)  &idat,
         &nn2, 2*MAXNUP, sizeof(int), xdr_int) == FALSE) return FALSE;
     dat = (double *)  hepeup_.pup;
     if     ( xdr_array(xdrs,  (char **)   &dat,
         &nn5, 5*MAXNUP, sizeof(double), xdr_double) == FALSE) return FALSE; 
     dat = (double *)  hepeup_.vtimup;
     if     ( xdr_array(xdrs,  (char **)   &dat,
         &nn, MAXNUP, sizeof(double), xdr_double) == FALSE) return FALSE; 
     dat = (double *)  hepeup_.spinup;
     if     ( xdr_array(xdrs,  (char **)   &dat,
         &nn, MAXNUP, sizeof(double), xdr_double) == FALSE) return FALSE; 
     return TRUE;
}   

bool_t xdr_heprup_(XDR *xdrs, int *blockid,
 				 int *ntot, char** version)

{
/*  Translate the HEPRUP COMMON block from the STDHEP package to/from
    an XDR stream. Note that we do not allocate memory, because we fill
    directly the COMMON.  Also, mcfio will allocate the space for the 
    string version.        */
    
    unsigned int nn, n2;
    int i;
    int *idat;
    char *vers;
    double *dat;
    
    if ((xdrs->x_op == XDR_ENCODE) || (xdrs->x_op == XDR_MCFIOCODE))  {
       if (*blockid != MCFIO_HEPRUP) {
          fprintf (stderr, "mcf_hepup_xdr: Inconsistent Blockid %d \n ", 
           (*blockid));
          return FALSE;
       }

       nn =  (unsigned int) heprup_.nprup;   /* Number of elements in xsecup, xerrup, xmaxup, lprup */
       n2 =  (unsigned int) 2;   /* Number of elements in idbmup, ebmup, pdfgup, pdfsup */

       /* Total length in bytes include blockid, ntot, version, as well
          as the common block entries                                     */
       *ntot = sizeof(int)*(5 + 3*n2 + nn) + sizeof(double)*(n2 + 3*nn);

       if (xdrs->x_op == XDR_MCFIOCODE) return TRUE;
       strncpy(version[0],stdver_.stdhep_ver, 4);
       } 
      
     if     ( (xdr_int(xdrs, blockid) &&
     	      xdr_int(xdrs, ntot) &&
     	      xdr_string(xdrs, version, MCF_XDR_VERSION_LENGTH) &&
     	      xdr_int(xdrs, &(heprup_.idwtup)) &&
              xdr_int(xdrs, &(heprup_.nprup))) == FALSE) return FALSE;
              
     if ((xdrs->x_op == XDR_DECODE) && ( *blockid != MCFIO_HEPRUP) ) {
          fprintf (stderr, "mcf_hepup_xdr: Inconsistent Blockid %d \n ", 
           (*blockid));
          return FALSE;
     }
     idat = heprup_.idbmup;
     if     ( xdr_array(xdrs, (char **)  &idat,
            &n2, 2, sizeof(int), xdr_int) == FALSE) return FALSE;
     dat = (double *)  heprup_.ebmup;
     if     ( xdr_array(xdrs,  (char **)   &dat,
                &n2, 2, sizeof(double), xdr_double) == FALSE) return FALSE; 
     idat = heprup_.pdfgup;
     if     ( xdr_array(xdrs, (char **)  &idat,
            &n2, 2, sizeof(int), xdr_int) == FALSE) return FALSE;
     idat = heprup_.pdfsup;
     if     ( xdr_array(xdrs, (char **)  &idat,
            &n2, 2, sizeof(int), xdr_int) == FALSE) return FALSE;
     dat = (double *)  heprup_.xsecup;
     if     ( xdr_array(xdrs,  (char **)   &dat,
                &nn, MAXPUP, sizeof(double), xdr_double) == FALSE) return FALSE; 
     dat = (double *)  heprup_.xerrup;
     if     ( xdr_array(xdrs,  (char **)   &dat,
                &nn, MAXPUP, sizeof(double), xdr_double) == FALSE) return FALSE; 
     dat = (double *)  heprup_.xmaxup;
     if     ( xdr_array(xdrs,  (char **)   &dat,
                &nn, MAXPUP, sizeof(double), xdr_double) == FALSE) return FALSE; 
     idat = heprup_.lprup;
     if     ( xdr_array(xdrs, (char **)  &idat,
            &nn, MAXPUP, sizeof(int), xdr_int) == FALSE) return FALSE;
     
     return TRUE;
}   


