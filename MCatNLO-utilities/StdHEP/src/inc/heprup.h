/*
C...User process initialization commonblock.
*/

#define MAXPUP 100
extern struct heprup {
  int idbmup[2];
  double ebmup[2];
  int pdfgup[2];
  int pdfsup[2];
  int idwtup;
  int nprup;
  double xsecup[MAXPUP];
  double xerrup[MAXPUP];
  double xmaxup[MAXPUP];
  int lprup[MAXPUP];
} heprup_;
