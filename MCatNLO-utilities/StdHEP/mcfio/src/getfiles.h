/*************************************************************************
*                                                                        *
*                  Getfiles.h                                            *
*                                                                        *
* Copyright (c) 1993 Universities Research Association, Inc.             *
* All rights reserved.                                                   *
*                                                                        *
* Fermilab Nirvana Project                                               *
* May 23, 1991                                                           *
*                                                                        *
* Written by Donna Reid                                                  *
*                                                                        *
* modified 11/6/91 JMK - updated function prototypes			 *
* modified 3/22/93 JMK - added prototype for HandleCustomExistFileSB	 *
*                                                                        *
*************************************************************************/
/* SCCS ID: getfiles.h 1.7 1/11/96 */

#define GFN_OK		1               /* Get Filename OK constant     */
#define GFN_CANCEL	2               /* Get Filename Cancel constant */

int GetNewFilename (Widget parent, char *promptString, char *filename);
int GetExistingFilename (Widget parent, char *promptString, char *filename);
int HandleCustomExistFileSB(Widget existFileSB, char *filename);
char *GetFileDialogDefaultDirectory(void);
char *GetFileDialogDefaultPattern(void);
void SetFileDialogDefaultDirectory(char *dir);
void SetFileDialogDefaultPattern(char *pattern);
void SetGetExistingFilenameTextFieldRemoval(int state);
