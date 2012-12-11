/*

   Rob Kutschke,  Oct. 21/96 

   Add a list of 4-vectors.  

   Calling arguments,
   int *n        - input  - the number of 4-vectors in the input list.
   double *pout  - output - the sum of the input 4-vectors.
   double *pin_1 - input  - the first 4-vector in the list.
   double *pin_2 - input  - the next 4-vector in the list.
   double *pin_3 - input  - the next 4-vector in the list.
     .....

   The number of input 4-vectors can be any positive integer, so long as 
   the first argument correctly describes how many there are.

   The first argument is passed by reference in order to make this
   routine fortran callable.

   Here a 4-vector is assumed to be 4 contiguous doubles.  The
   routine sum_4vec_f_ does the same operation for floats.

   The sum is accumulated in a temporary variable so that constructions
   like the following will work:

   real*8 psum(4), p(4)
   call sum_4vec_d ( 2, psum, psum, p )

*/

#include <stdarg.h>

void sum_4vec_d_ ( int *n, double *pout,  ... ){

   va_list ap;    /* Argument pointer for variable number of input args */
   double  *pi;   /* Pointer to next input 4-vector */

   double sum[] = {0., 0., 0., 0.} ;
   int      m;

   va_start ( ap, pout );

   for ( m = *n; m>0; m-- ){

      pi = va_arg( ap, double *) ;

      sum[0] += *(pi++) ;
      sum[1] += *(pi++) ;
      sum[2] += *(pi++) ;
      sum[3] += *(pi)   ;

   }

   *(pout++) = sum[0];
   *(pout++) = sum[1];
   *(pout++) = sum[2];
   *(pout  ) = sum[3];

   va_end(ap);

}
