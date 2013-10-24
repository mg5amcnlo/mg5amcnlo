/*******************************************************************************
*									       *
* mcf_xdr.h --  Include file for mcfast Xdr layer. Specifies the headers     *
*                 ( Block, event, table and files) 			       *	       *
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
* their employees, makes any warrenty, express or implied, or assumes any      *
* legal liability or responsibility for the accuracy, completeness, or         *
* usefulness of any information, apparatus, product, or process disclosed, or  *
* represents that its use would not infringe privately owned rights.           *
*									       *
*******************************************************************************/
#define MCF_XDR_F_TITLE_LENGTH 255
#define MCF_XDR_B_TITLE_LENGTH 80
#define MCF_XDR_MAXLREC 32000
#define MCF_XDR_MINLREC 512
#define MCF_XDR_VERSION "v0.0"
#define MCF_STREAM_NUM_MAX 20
#define MCF_DEFAULT_TABLE_SIZE 100
#define MCF_XDR_VERSION_LENGTH 4
#define MCF_XDR_STDCM2_LENGTH 20
#define XDR_MCFIOCODE 1025 /* Private code to be passed to the encoding  
				filter to estimate the length prior to encode
				in memory */ 

typedef enum _mcfxdrBlockType { 
	GENERIC, FILEHEADER, EVENTTABLE, SEQUENTIALHEADER, 
	EVENTHEADER, NOTHING
} mcfxdrBlockType;


typedef struct _mcfxdrGeneric{
	int id;		/* Identifier for this item =  FILEHEADER */
	int length;     /* The length of data body, byte count, excluding 
				the id and version, and this word */
	char version[MCF_XDR_VERSION_LENGTH+1];
	                /* The version of this particular block */
	int *data;	/* The data block */
} mcfxdrGeneric;

typedef struct _mcfxdrFileHeader{
	int id;		/* Identifier for this item =  FILEHEADER */
	int length;     /* The length of data body, byte count, excluding 
				the id and version, and this word */
	char version[MCF_XDR_VERSION_LENGTH+1];
	                /* The version of this particular block */
	char title[MCF_XDR_F_TITLE_LENGTH+1];
		        /* The title length */
	char comment[MCF_XDR_F_TITLE_LENGTH+1]; /* The comment ..*/
	char date[30];
	char closingDate[30];
	unsigned int numevts_expect;    /* The number of event expected */
	unsigned int numevts;    /* The number of evts really written on tape */
	unsigned int firstTable; /* The XDR locator for the first table */
	unsigned int dimTable; /* The number of events listed in the fixed-sized 
	                           event table */
	unsigned int nBlocks;		
			/* The maximum number of Block types in the file
				( excluding File headers and Event Tables) */
	int *blockIds;     /* The list of Block identifiers */
	
	char **blockNames; /* The list of names ( Titles) for these blocks */
	unsigned int nNTuples;
	                /* The maximum number of Ntuples defined for this 
	                	stream */
	
} mcfxdrFileHeader;

typedef struct _mcfxdrEventTable{
	int id;		/* Identifier for this item =  EVENTTABLE */
	int length;     /* The length of data body, byte count, excluding 
				the id and version, and this word */
	char version[MCF_XDR_VERSION_LENGTH+1];
	                 /* The version of this particular block */
	int nextLocator; /*The Locator for the next Event Table. */
	int previousnumevts; /* The size of the previous Table */
        int numevts;	/* The number of events in this chunk */
        unsigned int dim; /* The dimension of the arrays listed below */
	unsigned int ievt;     /* The current index in the list */
	int *evtnums;	/* The List of event numbers, within a store */
	int *storenums; /* The list of Store number within a Run */
	int *runnums;   /* The list of run numbers */
	int *trigMasks; /* The list of user-defined Trigger masks */
	unsigned int *ptrEvents;
			/* The list of XDR pointers for these events */ 
} mcfxdrEventTable;

typedef struct _mcfxdrSequentialHeader{
	int id;		/* Identifier for this item =  SEQUENTIALHEADER */
	int length;     /* The length of data body, byte count, excluding 
				the id and version, and this word */
	char version[MCF_XDR_VERSION_LENGTH+1];
	               /* The version of this particular block */
	unsigned int nRecords; /* The number of records (including this one) 
				in the logical event */			
} mcfxdrSequentialHeader;
 
typedef struct _mcfxdrEventHeader{
	int id;		/* Identifier for this item =  CHUNKHEADER */
	int length;     /* The length of data body, byte count, excluding 
				the id and version, and this word */
	char version[MCF_XDR_VERSION_LENGTH+1];
	               /* The version of this particular block */
	int previousevtnum; /* The previous event number */
	int evtnum;	/* The event numbers, within a store */
	int storenum;   /* The Store number within a Run */
	int runnum;     /* The Run numbers */
	int trigMask;  /* The Trigger masks */
	unsigned int nBlocks;  /* The number of Blocks  */
	unsigned int dimBlocks; /* The dimension of the two following arrays */
	int *blockIds;     /* The list of Block identifiers */
	unsigned int *ptrBlocks;
			/* The list of XDR pointers for these blocks */ 
	unsigned int nNTuples;
	                /* The number of Ntuples defined for this event */
	                
	unsigned int dimNTuples; /* The dimension of the two following arrays */
	int *nTupleIds; /* The list of Ntuple identifiers, pointing to the 
				global list array */                				
	unsigned int *ptrNTuples;
			/* The list of XDR pointers for these NTuples */ 
	
} mcfxdrEventHeader;

typedef struct _mcfStream{
	int id; 	/* Id of the Stream */
	int row;	/* Read or Write */
	int dos;	/* Direct, Memory Mapped I/O or Sequential */
	int status;     /* The Stream status, either at BOF, RUNNING, EOF 
	                   or simply declared, and needs to be opened 
	                   (NTuple usage) */
	int numWordsC;  /* The number of words read or written, Content */
	int numWordsT;  /* The number of words read or written, Total */
	mcfxdrFileHeader *fhead; /* The File header */
	mcfxdrEventHeader *ehead; /* The current Event Header */  
	unsigned int currentPos; /* The XDR current position */
	unsigned int evtPos;     /* The XDR position for the begingin of evt */
	unsigned int tablePos;   /* The XDR position for the table */
	unsigned int firstPos;   /* The XDR position just before file header */
	XDR *xdr;       /* The XDR stream */
	char *filename; /* Filename */
	FILE *filePtr;  /* The file pointer */
	int fileDescr;      /* File descriptor if Memory Mapped */
	char *fileAddr;  /* Address in virtual memory if Memory Mapped */
	size_t fileLen;   /* The file length */
	mcfxdrEventTable *table; /* The event table */
	char *device;	/* The device name, if any */
	char *vsn;      /* The Visual S. number, e.g., the tape label */
	int filenumber; /* The sequential file number, if any */
	int minlrec;    /* The minimum record length for this stream */
	int maxlrec;    /* The maximum record length for this stream */
	int bufferSize; /* The current size of the primary buffer */	
	mcfxdrSequentialHeader *shead; /* The Sequential header */
	char *buffer;   /*  A pointer to a generic data buffer, to get the 
				data from tape and then decode it */
	char *buffer2; /* A secondary buffer, to hold the event 
				as the event grows */
} mcfStream;

extern mcfStream **McfStreamPtrList;
extern char **McfGenericVersion;
extern unsigned int McfNumOfStreamActive;	 
extern bool_t McfNTuPleSaveDecoding;

bool_t xdr_mcfast_generic(XDR *xdrs, int *blockid,
 				 int *ntot, char** version, char** data);
bool_t xdr_mcfast_headerBlock(XDR *xdrs, int *blockid,
 				 int *ntot, char** version);
bool_t xdr_mcfast_fileheader(XDR *xdrs, int *blockid,
 		 int *ntot, char** version, mcfxdrFileHeader **mcf, 
 		 int streamId);
bool_t xdr_mcfast_eventtable(XDR *xdrs, int *blockid,
 		 int *ntot, char** version, mcfxdrEventTable **mcf);
bool_t xdr_mcfast_seqheader(XDR *xdrs, int *blockid,
 		 int *ntot, char** version, mcfxdrSequentialHeader **mcf);
bool_t xdr_mcfast_eventheader(XDR *xdrs, int *blockid,
 		 int *ntot, char** version, mcfxdrEventHeader **mcf);
