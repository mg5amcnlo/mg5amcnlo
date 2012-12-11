/*******************************************************************************
*									       *
* mcf_nTupleDescript.h -- Include file for mcfast generalized nTuple           *
*    descriptors.  This is a genric structres that can hold info about	       *
*    specficic instances of a generalized nTuple. 			       *
*									       *
*	P. Lebrun, September 1995.					       *
*									       *
*******************************************************************************/
/*
** Information concerning a generic variable within an Ntuple
*/
enum varTypes {BYTE_NTU, CHARACTER_NTU, INTEGER2_NTU, LOGICAL_NTU,
               INTEGER_NTU, REAL_NTU,
               DBL_PRECISION_NTU, COMPLEX_NTU, DBL_COMPLEX_NTU, POINTER_NTU};

enum orgStyles {PARALLEL_ARRAY_NTU, DATA_STRUCT_NTU};               
               
#define N_VAR_TYPES 10
#define MAX_VAR_NAME  31
#define MAX_NTU_TITLE 80
#define MAX_VAR_DESCRIP 1023
#define MAX_VAR_DIMENSIONS 4
#define NUM_START_VARIABLES 10
#define NTU_MAX_TITLE_LENGTH 80
#define NTU_MAX_CATEGORY_LENGTH 255
#define NTU_MAX_CATEGORY_DEPTH 40
#define NTU_START_LIST_SIZE 20

typedef struct {
    char nameBlank;   /* flag indicating that the variable does not exist. */
    char *name;       /* Mnemonic name of the variable. */
    char *description;/* description for the variable */
    int  type;        /* Variable type (int, ...) see above enum varTypes */
    char isFixedSize; /* Variable is or is not indexed by nTuple multiplicity*/
    int  numDim;      /* The variable dimensions, not counting mult. one */
    int  dimensions[MAX_VAR_DIMENSIONS+1];
                      /* Variable dims, not counting the multiplicity one*/
    size_t lengthW;   /* Used in XDR filtering, length in words */
    size_t lengthB; /* Used in XDR filtering, length in byte */                
    long offset;    /* The variable virtual address for a given instance */
    u_int offsetXDR; /* The variable relative address within the struct. */
} varGenNtuple;

typedef struct {
    int numVariables;    /* The total number of variables in the structure */
    int numAvailable;    /* The number of available var. in var. array  */
    char nameIndex[32];  /* The name for the Ntuple single index */
    int maxMultiplicity; /* The maximum multiplicity for any instances  */
    char *title;	 /* Title for the structure */
    char *description;   /* Description of this structure. */
    char version[8];	 /* The version string */
    int orgStyle;	 /* The organization of the indexed variables */
    void *address;	 /* Virtual address of a particular instance */
    long multOffset;      /* Offset for the multiplicity offset */
    u_int multXDROffset;   /* Adress for the multiplicity offset */
    long fenceOffset;     /* Offset for the fence */
    u_int fenceXDROffset;  /* XDR offset for the fence */
    long  *subOffset;     /* Offset for the sub structures */
    u_int  *subXDROffset;  /* XDR offset for the sub structures */
    varGenNtuple **variables; /* The variable descriptions */
    int *varOrdering;    /* Ordering of the variables for the dbin, .h..  file*/
    int firstIndexed;    /* Once ordered, the first indexed for indexed part */
} descrGenNtuple; 

/*
** A Data structure to hold a DDL, without MOTIF widget, to be used in 
** stand alone mode in mcfio.
*/

typedef struct nTuDDLRec {
    int id;             /* The id of the NTuple, as returned to the user */
    int seqNTuId;	/* The sequential number for this particular stream */
    int uid;            /* The user Id, Unique (within a Category) id */    
    char *category;
    char *title;
    char *dbinFileName; /* dbin filename, not guarantted to be there. */
    int streamId;       /* The stream on which this ddl is assigned to */
    int referenceId;     
    struct nTuDDLRec *reference;
                       /* the reference in case a similar ddl has already 
    			   been installed in the running image. */
    			   
    descrGenNtuple *descrNtu; /* The Ntuple Descriptor */
} nTuDDL;    

