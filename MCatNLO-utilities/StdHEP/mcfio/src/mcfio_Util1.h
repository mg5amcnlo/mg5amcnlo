/*******************************************************************************
*									       *
* mcfio_Util1.h --  Include file for mcfast initialisation & info i/o layer.   *
*									       *
* Copyright (c) 1994 Universities Research Association, Inc.		       *
* All rights reserved.							       *
*									       *
*******************************************************************************/
void mcfioC_Init(void);
void mcfioC_Close(int istream);
void mcfioC_PrintDictionary(void);
unsigned int mcfioC_InfoNumSream(int *istreams, unsigned int nmax);
void mcfioC_InfoStreamInt(int istream, int key, int *value);
void mcfioC_InfoStreamChar(int istream, int key, char *answer, int *lret);
void mcfioC_InfoEventInt(int istream, int key, int *value);
void mcfioC_InfoEventChar(int istream, int key, char *answer, int *lret);
void mcfioC_SetEventInfo(int istream, int key, int *value);
void mcfioC_Free_FileHeader(mcfxdrFileHeader **p);
void mcfioC_Free_SeqHeader(mcfxdrSequentialHeader **p);
void mcfioC_Free_EventHeader(mcfxdrEventHeader **p);
void mcfioC_Free_EventTable(mcfxdrEventTable **p);
void mcfioC_FreeStream(mcfStream **stream);
void mcfioC_InfoBlockChar(int stream, int blk, int key,
                            char *answer, int *lret);
void mcfioC_GetBlockName(int blkId, char *answer);
