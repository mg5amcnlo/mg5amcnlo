/* ntuBuild
** Creation Date : Tue Oct 17 22:09:51 1995
**  User Comments
**  
*/    
#define nglob_max 12
typedef struct {
    char version[8]; /* Version token */
    int nglob; /* Generalized Ntuple Multiplicity value */ 
    char           eventnumber; /* ?  */
    short          triggerbit1; /* this is a trigger bit field  */
    int            runnumber; /* ?  */
    int            No_way; /* ?  */
    float          missinget; /* ?  */
    double         something; /* this is junk variable  */
    char           missingetphi[nglob_max][30]; /* ?  */
    char           V6_new[nglob_max][80]; /* ?  */
    float          v5_new[nglob_max][6][5]; /* ?  */
    float          v7_now2[nglob_max]; /* ?  */
    float          missingetnomuon[nglob_max]; /* ?  */
    double         v7_now[nglob_max]; /* ?  */
    double         v7_now_last[nglob_max]; /* ?  */
    int fence; 
} glob_struct; 
