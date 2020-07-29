__device__ void oxxxxx(const double p[3], double fmass, int nhel, int nsf,
                       thrust::complex<double> fo[6]);
__device__ void omzxxx(const double p[3],  int nhel, int nsf,
                       thrust::complex<double> fo[6]);
__device__ void opzxxx(const double p[3], int nhel, int nsf,
                       thrust::complex<double> fo[6]);
__device__ void oxzxxx(const double p[3],  int nhel, int nsf,
                       thrust::complex<double> fo[6]);

__device__ void sxxxxx(const double p[3], int nss, thrust::complex<double> sc[3]);

__device__ void ixxxxx(const double p[3], double fmass, int nhel, int nsf,
                       thrust::complex<double> fi[6]);
__device__ void imzxxx(const double p[3], int nhel, int nsf,
                       thrust::complex<double> fi[6]);
__device__ void ipzxxx(const double p[3], int nhel, int nsf,
                       thrust::complex<double> fi[6]);
__device__ void ixzxxx(const double p[3], int nhel, int nsf,
                       thrust::complex<double> fi[6]);

//__device__ void txxxxx(double p[4], double tmass, int nhel, int nst,
//                       thrust::complex<double> fi[18]);

__device__ void vxxxxx(const double p[3], double vmass, int nhel, int nsv,
                       thrust::complex<double> v[6]);
