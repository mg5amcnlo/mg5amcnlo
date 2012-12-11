/*******************************************************************************
*									       *
* mcf_xdr_Ntuple.h --  Include file for mcfast Xdrlayer used in the 	       *
*       Ntuple code. Refers to a bunch of structure not included in this file. *	 Specifies the headers     *
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
* their employees, makes any warrenty, express or implied, or assumes any      *
* legal liability or responsibility for the accuracy, completeness, or         *
* usefulness of any information, apparatus, product, or process disclosed, or  *
* represents that its use would not infringe privately owned rights.           *
*									       *
*******************************************************************************/
bool_t xdr_mcfast_NTuple(XDR *xdrs, descrGenNtuple *dNTu,
 		 int *ntot, int nTupleId,  char* version);
bool_t xdr_mcfast_NTupleXDRPtr(XDR *xdrs, descrGenNtuple *dNTu,
 		 int *ntot, int nTupleId,  char* version);
bool_t xdr_mcfast_NTupleMult(mcfStream *str,
                             descrGenNtuple *dNTu, char* version); 		 
bool_t xdr_mcfast_NTupleVar(mcfStream *str,
                           descrGenNtuple *dNTu, int ivar, char* version); 		 
bool_t xdr_mcfast_NTupleSubVar(mcfStream *str,
             descrGenNtuple *dNTu, int ivar, int multIndex, char* version);
bool_t xdr_mcfast_NTupleSubStruct(mcfStream *str,
             descrGenNtuple *dNTu, int multIndex, char* version);
