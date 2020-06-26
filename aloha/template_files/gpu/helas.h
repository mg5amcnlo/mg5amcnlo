__device__ void oxxxxx(double p[4], double fmass, int nhel, int nsf,
                       thrust::complex<double> fo[6]);

__device__ void sxxxxx(double p[4], int nss, thrust::complex<double> sc[3]);

__device__ void ixxxxx(double p[4], double fmass, int nhel, int nsf,
                       thrust::complex<double> fi[6]);

__device__ void txxxxx(double p[4], double tmass, int nhel, int nst,
                       thrust::complex<double> fi[18]);

__device__ void vxxxxx(double p[4], double vmass, int nhel, int nsv,
                       thrust::complex<double> v[6]);
