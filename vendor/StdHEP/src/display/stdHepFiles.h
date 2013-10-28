/*******************************************************************************
*									       *
* stdHepFiles.h -- Nirvana Phase Space Event Display, Accessing the StdHep     *
*                  files and records					       *   
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
* their employees, makes any warrenty, express or implied, or assumes any      *
* legal liability or responsibility for the accuracy, completeness, or         *
* usefulness of any information, apparatus, product, or process disclosed, or  *
* represents that its use would not infringe privately owned rights.           *
*                                        				       *
* Fermilab Nirvana GUI Library						       *
* January 28 1994							       *
*									       *
* Written by Paul Lebrun						       *
*									       *
*******************************************************************************/

void OpenGetStdHep(StdHepWindow *window, int nevt);
void LoadColorCodeEvent(StdHepWindow *window);
void GetDetectorSketch(SpaceWindow *window);

#include "stdhep.h"
#include "hepev4.h"
