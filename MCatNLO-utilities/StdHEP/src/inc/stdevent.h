/*
  This is a copy of the StdHep::Event class for use with mcfio
*/
/*  note that to avoid alignment problems, structures and common blocks
    should be in the order: double precision, real, integer.
*/
#define NMXHEP 4000
#define NMXMLT 16
struct stdevent {
  double phep[NMXHEP][4];   /* 4-Momentum */
  double mass[NMXHEP];      /* mass */
  double helicity[NMXHEP];  /* helicity */
  double vcr[NMXHEP][4];    /* creation vertex information */
  double vdcy[NMXHEP][4];   /* decay vertex information */
  int nevhep;		    /* The event number */
  int nhep;		    /* The number of entries in this event */
  int ncol;		    /* The number of collisions in this event */
  int nevcol[NMXMLT];       /* event number of original collision */
  int iostr[NMXMLT];        /* stream this event is from */
  int nhepcol[NMXMLT];      /* The number of entries in this collision */
  int isthep[NMXHEP]; 	    /* The Particle id */
  int idhep[NMXHEP];        /* The particle id */
  int jmohep1[NMXHEP];      /* The position of the first mother particle */
  int jmohep2[NMXHEP];      /* The position of the second mother particle */
  int jdahep1[NMXHEP];      /* Position of the first daughter */
  int jdahep2[NMXHEP];      /* Position of the last daughter */
  int color[NMXHEP];        /* color */
  int jcol[NMXHEP];         /* collision number */
} stdevent_;
