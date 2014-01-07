/*******************************************************************************
*									       *
* mcf_BrowseUtil2.c -- Utilities and auxillary Panels for NTuple Browser.      *
*	Gives a short listing of the characteristic of the file header         *
* Copyright (c) 1995, 1996 Universities Research Association, Inc.	       *
* All rights reserved.							       *
* 									       *
*******************************************************************************/
#include <sys/param.h>
/*
** An array of such things, running parallel to the NTuDDLList
*/   
void mcfioC_ShowBrowserHeadDump();
void mcfioC_ShowBrowserEvtHeadDump();
#ifdef HISTO
void mcfioC_ShowBrowserOneDHist();
void mcfioC_OneDHistUpdateNTupleContent();
#endif
void mcfioC_RemoveWhiteSpace(char *str);


