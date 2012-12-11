/* ntuBuild
** Creation Date : Sat Oct 28 14:02:58 1995
**  User Comments
**  
*/    
#define nglob_max 12
typedef struct {
    char           v5_pad[2]; /* ?  */
    char           missingetphi[30]; /* ?  */
    short          V6_new[80]; /* ?  */
    float          v5_new[6][5]; /* ?  */
    float          v7_now2; /* ?  */
    float          missingetnomuon; /* ?  */
    double         v7_now; /* ?  */
    double         v7_now_last; /* ?  */
} glob_v_struct; 
/* ----- */  
typedef struct {
    char version[12]; /* Version token */
    int nglob; /* Generalized Ntuple Multiplicity value */ 
    char           eventnumber[8]; /* ?  */
    int            triggerbit1; /* this is a trigger bit field  */
    int            runnumber; /* ?  */
    int            No_way; /* ?  */
    float          missinget; /* ?  */
    double         something; /* this is junk variable  */
    glob_v_struct var[nglob_max]; /* The array of substructures */
    int fence; 
} glob_struct; 
