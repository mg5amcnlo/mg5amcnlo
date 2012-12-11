/*******************************************************************************
*									       *
* printUtils.h -- Fermilab Nirvana library Printer Menu	& Printing Routines    *
*									       *
* Copyright (c) 1991 Universities Research Association, Inc.		       *
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
* April 20, 1992							       *
*									       *
* Written by Arnulfo Zepeda-Navratil				               *
*            Centro de Investigacion y Estudio Avanzados ( CINVESTAV )         *
*            Dept. Fisica - Mexico                                             *
*            BITNET: ZEPEDA@CINVESMX                                           *
*									       *
*******************************************************************************/
/* SCCS ID: printUtils.h 1.4 8/4/93 */

/* Maximum length of an error returned by IssuePrintCommand() */
#define MAX_PRINT_ERROR_LENGTH 1024

#define DESTINATION_REMOTE 1
#define DESTINATION_LOCAL  2

void LoadPrintPreferences(XrmDatabase prefDB, char *appName, char *appClass,
	int lookForFlpr);

#ifdef VMS
void PrintFile(Widget parent, char *PrintFileName, char *jobName, int delete);
#else
void PrintFile(Widget parent, char *PrintFileName, char *jobName);
#endif /*VMS*/
