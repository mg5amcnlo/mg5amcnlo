__device__ void oxxxxx(const fptype p[3], fptype fmass, int nhel, int nsf,
                       cxtype fo[6]);
__device__ void omzxxx(const fptype p[3],  int nhel, int nsf,
                       cxtype fo[6]);
__device__ void opzxxx(const fptype p[3], int nhel, int nsf,
                       cxtype fo[6]);
__device__ void oxzxxx(const fptype p[3],  int nhel, int nsf,
                       cxtype fo[6]);

__device__ void sxxxxx(const fptype p[3], int nss, cxtype sc[3]);

__device__ void ixxxxx(const fptype p[3], fptype fmass, int nhel, int nsf,
                       cxtype fi[6]);
__device__ void imzxxx(const fptype p[3], int nhel, int nsf,
                       cxtype fi[6]);
__device__ void ipzxxx(const fptype p[3], int nhel, int nsf,
                       cxtype fi[6]);
__device__ void ixzxxx(const fptype p[3], int nhel, int nsf,
                       cxtype fi[6]);

//__device__ void txxxxx(fptype p[4], fptype tmass, int nhel, int nst,
//                       cxtype fi[18]);

__device__ void vxxxxx(const fptype p[3], fptype vmass, int nhel, int nsv,
                       cxtype v[6]);
