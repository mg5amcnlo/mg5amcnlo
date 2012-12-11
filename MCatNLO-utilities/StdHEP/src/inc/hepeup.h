/*
C...User process event common block.
*/

#define MAXNUP 500
extern struct hepeup {
  int nup;		/* number of particles */
  int idprup;
  double xwgtup;
  double scalup;
  double aqedup;
  double aqcdup;
  int idup[MAXNUP];
  int istup[MAXNUP];
  int mothup[MAXNUP][2];
  int icolup[MAXNUP][2];
  double pup[MAXNUP][5];
  double vtimup[MAXNUP];
  double spinup[MAXNUP];
} hepeup_;

