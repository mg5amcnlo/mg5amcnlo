//
// The suffixes indicate the type of input as follows:
//
// ccr: complex momenta, complex masses, real scale
// rcr: real momenta   , complex masses, real scale
// rrr: real momenta   , real masses   , real scale
// cc : complex momenta, complex masses, no scale
// rc : real momenta   , complex masses, no scale
// rr : real momenta   , real masses   , no scale
//
// If the routines without scale are called, the default scale is used,
// which can be set with OLO_SCALE.
//
// D0-routines
#define OLO_D0ccr   __avh_olo_dp_MOD_d0ccr
#define OLO_D0cc    __avh_olo_dp_MOD_d0cc
#define OLO_D0rcr   __avh_olo_dp_MOD_d0rcr
#define OLO_D0rc    __avh_olo_dp_MOD_d0rc
#define OLO_D0rrr   __avh_olo_dp_MOD_d0rrr
#define OLO_D0rr    __avh_olo_dp_MOD_d0rr
// C0-routines
#define OLO_C0ccr   __avh_olo_dp_MOD_c0ccr
#define OLO_C0cc    __avh_olo_dp_MOD_c0cc
#define OLO_C0rcr   __avh_olo_dp_MOD_c0rcr
#define OLO_C0rc    __avh_olo_dp_MOD_c0rc
#define OLO_C0rrr   __avh_olo_dp_MOD_c0rrr
#define OLO_C0rr    __avh_olo_dp_MOD_c0rr
// B0-routines
#define OLO_B0ccr   __avh_olo_dp_MOD_b0ccr
#define OLO_B0cc    __avh_olo_dp_MOD_b0cc
#define OLO_B0rcr   __avh_olo_dp_MOD_b0rcr
#define OLO_B0rc    __avh_olo_dp_MOD_b0rc
#define OLO_B0rrr   __avh_olo_dp_MOD_b0rrr
#define OLO_B0rr    __avh_olo_dp_MOD_b0rr
// B11-routines
#define OLO_B11ccr  __avh_olo_dp_MOD_b11ccr
#define OLO_B11cc   __avh_olo_dp_MOD_b11cc
#define OLO_B11rcr  __avh_olo_dp_MOD_b11rcr
#define OLO_B11rc   __avh_olo_dp_MOD_b11rc
#define OLO_B11rrr  __avh_olo_dp_MOD_b11rrr
#define OLO_B11rr   __avh_olo_dp_MOD_b11rr
// Bn-routines
#define OLO_BNccr   __avh_olo_dp_MOD_bnccr
#define OLO_BNcc    __avh_olo_dp_MOD_bncc
#define OLO_BNrcr   __avh_olo_dp_MOD_bnrcr
#define OLO_BNrc    __avh_olo_dp_MOD_bnrc
#define OLO_BNrrr   __avh_olo_dp_MOD_bnrrr
#define OLO_BNrr    __avh_olo_dp_MOD_bnrr
// An-routines
#define OLO_ANcr    __avh_olo_dp_MOD_ancr
#define OLO_ANc     __avh_olo_dp_MOD_an_c
#define OLO_ANrr    __avh_olo_dp_MOD_anrr
#define OLO_ANr     __avh_olo_dp_MOD_an_r
// A0-routines
#define OLO_A0cr    __avh_olo_dp_MOD_a0cr
#define OLO_A0c     __avh_olo_dp_MOD_a0_c
#define OLO_A0rr    __avh_olo_dp_MOD_a0rr
#define OLO_A0r     __avh_olo_dp_MOD_a0_r
// auxiliary routines
#define OLO_SCALE   __avh_olo_dp_MOD_olo_scale
#define OLO_ONSHELL __avh_olo_dp_MOD_olo_onshell
#define OLO_UNIT    __avh_olo_dp_MOD_olo_unit   

extern "C" {

void OLO_D0ccr( complex<double>(*)[3]
  ,complex<double>*,complex<double>*,complex<double>*,complex<double>*,complex<double>*,complex<double>*
  ,complex<double>*,complex<double>*,complex<double>*,complex<double>*
  ,double*
);
void OLO_D0cc( complex<double>(*)[3]
  ,complex<double>*,complex<double>*,complex<double>*,complex<double>*,complex<double>*,complex<double>*
  ,complex<double>*,complex<double>*,complex<double>*,complex<double>*
);
void OLO_D0rcr( complex<double>(*)[3]
  ,double*,double*,double*,double*,double*,double*
  ,complex<double>*,complex<double>*,complex<double>*,complex<double>*
  ,double*
);
void OLO_D0rc( complex<double>(*)[3]
  ,double*,double*,double*,double*,double*,double*
  ,complex<double>*,complex<double>*,complex<double>*,complex<double>*
);
void OLO_D0rrr( complex<double>(*)[3]
  ,double*,double*,double*,double*,double*,double*
  ,double*,double*,double*,double*
  ,double*
);
void OLO_D0rr( complex<double>(*)[3]
  ,double*,double*,double*,double*,double*,double*
  ,double*,double*,double*,double*
);

void OLO_C0ccr( complex<double>(*)[3]
  ,complex<double>*,complex<double>*,complex<double>*
  ,complex<double>*,complex<double>*,complex<double>*
  ,double*
);
void OLO_C0cc( complex<double>(*)[3]
  ,complex<double>*,complex<double>*,complex<double>*
  ,complex<double>*,complex<double>*,complex<double>*
);
void OLO_C0rcr( complex<double>(*)[3]
  ,double*,double*,double*
  ,complex<double>*,complex<double>*,complex<double>*
  ,double*
);
void OLO_C0rc( complex<double>(*)[3]
  ,double*,double*,double*
  ,complex<double>*,complex<double>*,complex<double>*
);
void OLO_C0rrr( complex<double>(*)[3]
  ,double*,double*,double*
  ,double*,double*,double*
  ,double*
);
void OLO_C0rr( complex<double>(*)[3]
  ,double*,double*,double*
  ,double*,double*,double*
);

void OLO_B0ccr( complex<double>(*)[3]
  ,complex<double>*
  ,complex<double>*,complex<double>*
  ,double*
);
void OLO_B0cc( complex<double>(*)[3]
  ,complex<double>*
  ,complex<double>*,complex<double>*
);
void OLO_B0rcr( complex<double>(*)[3]
  ,double*
  ,complex<double>*,complex<double>*
  ,double*
);
void OLO_B0rc( complex<double>(*)[3]
  ,double*
  ,complex<double>*,complex<double>*
);
void OLO_B0rrr( complex<double>(*)[3]
  ,double*
  ,double*,double*
  ,double*
);
void OLO_B0rr( complex<double>(*)[3]
  ,double*
  ,double*,double*
);

void OLO_b11ccr(
   complex<double>(*)[3],complex<double>(*)[3],complex<double>(*)[3],complex<double>(*)[3]
  ,complex<double>*
  ,complex<double>*,complex<double>*
  ,double*
);
void OLO_b11cc(
   complex<double>(*)[3],complex<double>(*)[3],complex<double>(*)[3],complex<double>(*)[3]
  ,complex<double>*
  ,complex<double>*,complex<double>*
);
void OLO_b11rcr(
   complex<double>(*)[3],complex<double>(*)[3],complex<double>(*)[3],complex<double>(*)[3]
  ,double*
  ,complex<double>*,complex<double>*
  ,double*
);
void OLO_b11rc(
   complex<double>(*)[3],complex<double>(*)[3],complex<double>(*)[3],complex<double>(*)[3]
  ,double*
  ,complex<double>*,complex<double>*
);
void OLO_b11rrr(
   complex<double>(*)[3],complex<double>(*)[3],complex<double>(*)[3],complex<double>(*)[3]
  ,double*
  ,double*,double*
  ,double*
);
void OLO_b11rr(
   complex<double>(*)[3],complex<double>(*)[3],complex<double>(*)[3],complex<double>(*)[3]
  ,double*
  ,double*,double*
);

void OLO_BNccr( complex<double>(*)[9][3],int*
  ,complex<double>*
  ,complex<double>*,complex<double>*
  ,double*
);
void OLO_BNcc( complex<double>(*)[9][3],int*
  ,complex<double>*
  ,complex<double>*,complex<double>*
);
void OLO_BNrcr( complex<double>(*)[9][3],int*
  ,double*
  ,complex<double>*,complex<double>*
  ,double*
);
void OLO_BNrc( complex<double>(*)[9][3],int*
  ,double*
  ,complex<double>*,complex<double>*
);
void OLO_BNrrr( complex<double>(*)[9][3],int*
  ,double*
  ,double*,double*
  ,double*
);
void OLO_BNrr( complex<double>(*)[9][3],int*
  ,double*
  ,double*,double*
);

void OLO_ANcr( complex<double>(*)[3][3],int*
  ,complex<double>*
  ,double*
);
void OLO_ANc( complex<double>(*)[3][3],int*
  ,complex<double>*
);
void OLO_ANrr( complex<double>(*)[3][3],int*
  ,double*
  ,double*
);
void OLO_ANr( complex<double>(*)[3][3],int*
  ,double*
);

void OLO_A0cr( complex<double>(*)[3]
  ,complex<double>*
  ,double*
);
void OLO_A0c( complex<double>(*)[3]
  ,complex<double>*
);
void OLO_A0rr( complex<double>(*)[3]
  ,double*
  ,double*
);
void OLO_A0r( complex<double>(*)[3]
  ,double*
);

void OLO_SCALE( double* );
void OLO_ONSHELL( double* );
void OLO_UNIT( int* );

}//extern "C"
