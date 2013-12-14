/*******************************************************************************
*									       *
* mc_Sequential.h --  Include file for mcfast Sequential i/o layer.            *
*									       *
* Copyright (c) 1994 Universities Research Association, Inc.		       *
* All rights reserved.							       *
*									       *
*******************************************************************************/
int mcfioC_OpenReadSequential(char *device, char *label, int filenumber);
int mcfioC_OpenWriteSequential(char *device, char *label, char *title, 
                char *comment, int numevts_pred, 
                int *blkIds, unsigned int nBlocks);
int mcfioC_NextEventSequential(int stream);
void mcfioC_CloseSequentialFile(int stream);
void mcfioC_CloseSequentialTape(int stream);
