#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void test_fpe_trapping();

#if defined(__linux__) && defined(__GNUC__)
#include <features.h>

/* glibc-1.x */
#if __GLIBC__ == 1
#include <fpu_control.h>
void trapfpe()
{
  __setfpucw(_FPU_DEFAULT &
             ~(_FPU_MASK_IM|_FPU_MASK_ZM|_FPU_MASK_UM|_FPU_MASK_OM));
  if (1) test_fpe_trapping();
}
#endif

/* glibc-2.1 */
#if (__GLIBC__ == 2) && (__GLIBC_MINOR__ == 1)
#include <fpu_control.h>
void trapfpe()
{
  unsigned int cw;
  cw = _FPU_DEFAULT & ~(_FPU_MASK_IM|_FPU_MASK_ZM|_FPU_MASK_UM|_FPU_MASK_OM);
  _FPU_SETCW(cw);
  if (0) test_fpe_trapping();
}
#endif

/* glibc-2.2 or later */
#if (__GLIBC__ == 2) && (__GLIBC_MINOR__ >= 2)
#define _GNU_SOURCE 1
#include <fenv.h>
static void __attribute__ ((constructor)) trapfpe(void)
{
  /* Enable some exceptions. At startup all exceptions are masked. */
  feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
}
#endif

#elif defined (__alpha) && defined (__osf__) && defined (__DECC)
void trapfpe()
{
  /* not trapped, bugs (with and without optimization) */
  if (0) test_fpe_trapping();
}
#elif defined (__alpha__) && defined (__unix__) && defined (__GNUC__)
void trapfpe()
{
  /* not trapped, bugs if compiling with -g but trapped for -O/-O2 */
  /* for -O2: fpe's also not trapped e.g. in lipsn! */
  if (0) test_fpe_trapping();
}
#elif defined (__i386__) && defined (__CYGWIN32__) && defined (__GNUC__)
void trapfpe()
{
  /* not trapped */
  if (0) test_fpe_trapping();
}
#else       /* unknown compiler/platform */
void trapfpe()
{
  test_fpe_trapping();
}
#endif


static void test_fpe_trapping()
{
  double x;
  x = -1.;
  printf("testing fpe trapping ...\n");
  fflush(stdout);
  printf("sqrt(-1.) = %f\n", sqrt(x));
  fflush(stdout);
  x = 10.;
  printf("10^500 = %e\n", pow(x,500.));
  fflush(stdout);
  printf("10^-500 = %e\n", pow(x,-500.));
  printf("floating point exceptions are not thrown!!!\n");
  fflush(stdout);
  exit(1);
}
