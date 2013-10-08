/*
    StdHep counting common block
*/
extern struct stdcnt {
    int nstdwrt;	/* number of events written */
    int nstdrd;		/* number of events read */
    int nlhwrt;		/* number of Les Houches events written */
    int nlhrd;		/* number of Les Houches events read */
} stdcnt_;
