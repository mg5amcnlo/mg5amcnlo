#ifndef STDHEP_MCFIO_H
#define STDHEP_MCFIO_H

/*******************************************************************************
*									       *
* stdhep_mcfio.h -- header for C version of mcfio interface routines                      *
*									       *
* Copyright (c) 1995 Universities Research Association, Inc.		       *
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
* Written by Lynn Garren    					       	       *
*									       *
*									       *
*******************************************************************************/

/*   prototypes */
#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif


int StdHepXdrReadInit(char *filename, int ntries, int ist);
int StdHepXdrReadOpen(char *filename, int ntries, int ist);
int StdHepXdrRead(int *ilbl, int ist);
int StdHepXdrReadMulti(int *ilbl, int ist);
int StdHepXdrWriteInit(char *filename, char *title, int ntries, int ist);
int StdHepXdrWriteOpen(char *filename, char *title, int ntries, int ist);
int StdHepXdrWrite(int ilbl, int ist);
int StdHepXdrWriteCM(int ilbl, int ist);
int StdHepXdrWriteEvent(int ilbl, int ist);
int StdHepXdrWriteEventLH(int ilbl, int ist);
int StdHepXdrWriteEventEUP(int ilbl, int ist);
int StdHepXdrWriteEventRUP(int ilbl, int ist);
void StdHepXdrEnd(int ist);
void StdHepPrintHeader( );

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif /* STDHEP_MCFIO_H */
