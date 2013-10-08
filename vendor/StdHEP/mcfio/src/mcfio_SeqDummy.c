/*******************************************************************************
*									       *
* mcfio_SeqDummy.c --  Utility routines for the McFast Monte-Carlo               *
*	Dummy Sequential routines, for the library without Sequential          *
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
#include "mcfio_Sequential.h"

int mcfioC_OpenReadSequential(char *device, char *label, int filenumber)
{
	fprintf(stderr,
	"mcfioC_OpenReadSequential: Not available in this library. \n");
	return -1;
}


int mcfioC_OpenWriteSequential(char *device, char *label, char *title,
             char *comment, int numevts_pred,
              int *blkIds, unsigned int nBlocks)
{
	fprintf(stderr,
	"mcfioC_OpenWriteSequential: Not available in this library. \n");
	return -1;
}

int mcfioC_NextEventSequential(int stream)
{
	fprintf(stderr,
	"mcfioC_NextEventSequential: Not available in this library. \n");
	return -1;
}

void mcfioC_CloseSequentialFile(int jstr)
{
	fprintf(stderr,
	"mcfioC_CloseSequentialFile: Not available in this library. \n");
	return;
}

void mcfioC_CloseSequentialTape(int jstr)
{
	fprintf(stderr,
	"mcfioC_CloseSequentialTape: Not available in this library. \n");
	return;
}
