/* Hepev4 holds Les Houches information */
/*  note that to avoid alignment problems, structures and common blocks
    should be in the order: double precision, real, integer.
*/
extern struct hepev4 {
  double eventweightlh;			/* event weight */
  double alphaqedlh; 			/* QED coupling alpha_em */
  double alphaqcdlh;			/* QCD coupling alpha_s */
  double scalelh[10]; 			/* Scale Q of the event */
  double spinlh[NMXHEP][3]; 		/* spin information */
  int    icolorflowlh[NMXHEP][2]; 	/* (Anti-)Colour flow */
  int    idruplh;			/* ID, as given by LPRUP codes */
} hepev4_;

extern struct hepev5 {
  double eventweightmulti[NMXMLT];	/* original event weight */
  double alphaqedmulti[NMXMLT]; 	/* original QED coupling alpha_em */
  double alphaqcdmulti[NMXMLT];		/* original QCD coupling alpha_s */
  double scalemulti[NMXMLT][10]; 	/* original Scale Q of the event */
  int    idrupmulti[NMXMLT];		/* original ID, as given by LPRUP codes */
} hepev5_;
