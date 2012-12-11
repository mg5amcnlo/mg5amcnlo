/*
** STDHEP begin/end run COMMON block
** See product StDhep
*/
extern struct stdcm1 {
float stdecom;   /*   STDECOM  - center-of-mass energy */
float stdxsec;   /*   STDXSEC  - cross-section */
double stdseed1; /*   STDSEED1 - random number seed */
double stdseed2; /*   STDSEED2 - random number seed */
int nevtreq;     /*   NEVTREQ  - number of events to be generated */
int nevtgen;     /*   NEVTGEN  - number of events actually generated */
int nevtwrt;     /*   NEVTWRT  - number of events written to output file */
int nevtlh;      /*   NEVTLH  - number of Les Houches events written to output file */
} stdcm1_;

extern struct stdcm2 {
char generatorname[20];		/* name of Monte Carlo generator */
char pdfname[20];     		/* name of PDF method used */
} stdcm2_;
