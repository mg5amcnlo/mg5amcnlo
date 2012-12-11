/*******************************************************************************
*									       *
* vmsUtils.c - Utility routines for VMS systems.			       *
*									       *
*	       This file contains the following functions:		       *
*									       *
*	 StrDescToNul      - Convert VMS String Descriptor to C-like null-     *
*			       terminated string.  Returns the address of      *
*			       the malloc'd string.  (Call FreeNulStr() when   *
*			       done with the string.)			       *
*	 NulStrToDesc	   - Convert C-like null-terminated string to VMS      *
*			       String Descriptor.  Returns the address of      *
*			       the malloc'd descriptor, which should be free'd *
*			       when done by calling FreeStrDesc().  	       *
*	 NulStrWrtDesc	   - Convert C-like null-terminated string to VMS      *
*			       String Descriptor for writing into.  The C      *
*			       String should already be allocated and the      *
*			       length passed as the second parameter.  Returns *
*			       the address of the malloc'd descriptor, which   *
*			       should be free'd when done via FreeStrDesc().   *
*	 FreeNulStr	   - Frees null-terminated strings created by 	       *
*			       StrDescToNul(). 				       *
*	 FreeStrDesc 	   - Frees VMS String Descriptors created by 	       *
*			       NulStrToDesc() and NulStrWrtDesc(). 	       *
*	 ConvertVMSCommandLine	- Convert an argument vector representing a    *
*			       VMS-style command line to something Unix-like.  *
*			       Limitations: no abbreviations, some syntax      *
*			       information is lost so some errors will yield   *
*			       strange results.				       * 
*									       *
*        rint              - Returns the integer (represented as a double      *
*                              precision number) nearest its double argument.  *
*                                                                              *
*        ProcAlive 	   - See if a process (identified by pID) is still     *
*			       alive on VMS.				       *
*                                                                              *
* Copyright (c) 1993 Universities Research Association, Inc.		       *
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
* Fermilab Nirvana GUI Library						       *
* February 22, 1993							       *
*									       *
* Written by Joy Kyriakopulos						       *
*									       *
*******************************************************************************/
/* SCCS ID: vmsUtils.h 1.7 7/7/94 */

#ifdef VMS
#ifndef __DESCRIP_LOADED
#include descrip
#endif /*__DESCRIP_LOADED*/

#define INCLUDE_FNF 0
#define EXCLUDE_FNF 1
#define NOT_ERR_FNF 2

char *StrDescToNul(struct dsc$descriptor_s *vmsString);
struct dsc$descriptor_s *NulStrToDesc(char *nulTString);
struct dsc$descriptor_s *NulStrWrtDesc(char *nulTString, int strLen);
void FreeNulStr(char *nulTString);
void FreeStrDesc(struct dsc$descriptor_s *vmsString);
double rint(double dnum);
void ConvertVMSCommandLine(int *argc, char **argv[]);
int VMSFileScan(char *dirname, char *(*namelist[]), int (*select)(), int fnf);
void VMSFileScanDone(void);
int ProcAlive(const unsigned int pID);

#endif /*VMS*/
