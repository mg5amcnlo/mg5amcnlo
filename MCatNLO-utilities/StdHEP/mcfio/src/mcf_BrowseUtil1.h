/*******************************************************************************
*									       *
* mcf_BrowseUtil1.h -- Utilities and auxillary Panels for NTuple Browser.      *			       *
*									       *
* Copyright (c) 1995, 1996 Universities Research Association, Inc.	       *
* All rights reserved.							       *
* 									       *
*******************************************************************************/
#include <sys/param.h>
/*
** a bunch of data structure to keep information to be able to manage 
** volatile histograms in the Browser. 
*/
typedef struct nTuBroHsGeneralRec{
    int type;     /* The HistoScope type of item (1D Hist, 2D hist, Ntuple) */
    int id;      /* The HistoScope id of the created item */
}  nTuBroHsGeneral;

typedef struct nTuBroHs1DRec{
    int type;     /* The HistoScope type of item (1D Hist) */
    int id;      /* The HistoScope id of the created item */
    int varNumber; /* The variable number (-1 = Multiplicity) */
    int subBlock; /* The subblock number to be histogram -1 = all */
    long *lDat;   /* The addresses to be histogrammed ( or starting address) */
    int *varIndices;
       /* An array containing the array index if var. multi-dimensional */ 
}  nTuBroHs1D;

typedef struct nTuBroHs2DRec{
    int type;     /* The HistoScope type of item (1D Hist) */
    int id;      /* The HistoScope id of the created item */
    int xVarNumber; /* The variable number (-1 = Multiplicity) for X axis */
    int yVarNumber; /* The variable number (-1 = Multiplicity) for Y axis */
    int xSubBlock; /* The subblock number to be histogram -1 = all, X axis */
    int ySubBlock; /* The subblock number to be histogram -1 = all, Y axis */
    long *xLDat;  /* The addresses to be histogrammed ( or starting address) */
    long *yLDat;  /* The addresses to be histogrammed ( or starting address) */
    int *xVarIndices;
       /* An array containing the array index if var. multi-dimensional */ 
    int *yVarIndices;
       /* An array containing the array index if var. multi-dimensional */ 
}  nTuBroHs2D;

/*
** A Data structure to hold analysis information for the NTuple Browser
*/

typedef struct nTuBrowserInfoRec {
    int id;             /* Id, pointing back by mcf_GetNTuByPtrID */
    nTuBuildWindow *templateW;   /* The top structure for viewing the DDL */
    Widget dumpDataW;   /* The Text widget for viewing the data content */
    Widget dumpDataFormW; /* It's parent form */
    Widget dumpDataShellW;   /* The shell widget for viewing the data content */
    Widget dumpDataBriefW; /* A button to control brief vs full dump */
    Widget fromDataTextW;  /* Widget to specify specific substructure ranges */
    Widget toDataTextW;   /* in the dump utility */
    int dumpFrom;        /* Starting index for the dump */
    int dumpTo;          /* Ending index for the dump */
    int nHistoItems;    /* The number of HistoScope items related to this */
    int nHisto1D;	/* The number of 1D histo */
    int nHisto2D;       /* The number of 2D histo */
    int nHistoNtuples;  /* The number of Ntuples */
    int sizeOfLists;    /* The size of the the following lists */
    nTuBroHsGeneral **hsItemList;  
                 /* The list of structures containing info for hs histograms */
    int currentData;    /* A flag stating if the data is current */
    void *data;		/* A pointer to the temporary user data area */
} nTuBrowserInfo;
 
/*
** An array of such things, running parallel to the NTuDDLList
*/   
extern  nTuBrowserInfo **NTupleBrowserList;

void mcfioC_CreateBrowserAnalysis();
void mcfioC_ExtendBrowserAnalysis(nTuBrowserInfo *nTuBr);
void mcfioC_createBrowserData(nTuBrowserInfo *nTuBr);
void mcfioC_DestroyBrowserAnalysis();
void mcfioC_ShowBrowserDataDump(nTuBrowserInfo * nTuBr);


