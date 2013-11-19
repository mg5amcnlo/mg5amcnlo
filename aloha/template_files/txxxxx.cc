#include <complex>
#include <cmath> 
#include "aloha_aux_functions.h"
using namespace std;
#include <iostream>
#include <cstdlib>

void txxxxx(double p[4], double tmass, int nhel, int nst, complex<double> tc[18])
{
	complex<double> ft[6][4], ep[4], em[4], e0[4];
	double pt, pt2, pp, pzpt, emp, sqh, sqs;
	int i, j;
	
	sqh = pow( 0.5, 0.5 );
	sqs = pow( 0.5/3, 0.5 );
	
	pt2 = p[1]*p[1] + p[2]*p[2];
	pp = min( p[0], pow( pt2+p[3]*p[3], 0.5 ) );
	pt = min( pp, pow( pt2, 0.5 ) );
	
	ft[4][0] = complex<double>( p[0]*nst, p[3]*nst );
	ft[5][0] = complex<double>( p[1]*nst, p[2]*nst );
	
	// construct eps+
	if( nhel >= 0 )
	{
		if( pp == 0 )
		{
			ep[0] = complex<double>( 0, 0 );
			ep[1] = complex<double>( -sqh, 0 );
			ep[2] = complex<double>( 0, nst*sqh );
			ep[3] = complex<double>( 0, 0 );
		}
		else
		{
			ep[0] = complex<double>( 0, 0 );
			ep[3] = complex<double>( pt/pp*sqh, 0 );
			
			if( pt != 0 )
			{
				pzpt = p[3]/(pp*pt)*sqh;
				ep[1] = complex<double>( -p[1]*pzpt, -nst*p[2]/pt*sqh );
				ep[2] = complex<double>( -p[2]*pzpt, nst*p[1]/pt*sqh );
			}
			else
			{
				ep[1] = complex<double>( -sqh, 0 );
				ep[2] = complex<double>( 0, nst*Sgn(sqh,p[3]) );
			}
		}
		
	}
	
	// construct eps-
	if( nhel <= 0 )
	{
		if( pp == 0 )
		{
			em[0] = complex<double>( 0, 0 );
			em[1] = complex<double>( sqh, 0 );
			em[2] = complex<double>( 0, nst*sqh );
			em[3] = complex<double>( 0, 0 );
		}
		else
		{
			em[0] = complex<double>( 0, 0 );
			em[3] = complex<double>( -pt/pp*sqh, 0 );
			
			if( pt != 0 )
			{
				pzpt = -p[3]/(pp*pt)*sqh;
				em[1] = complex<double>( -p[1]*pzpt, -nst*p[2]/pt*sqh );
				em[2] = complex<double>( -p[2]*pzpt,  nst*p[1]/pt*sqh );
			}
			else
			{
				em[1] = complex<double>( sqh, 0 );
				em[2] = complex<double>( 0, nst*Sgn(sqh,p[3]) );
			}
		}
	}
	
	// construct eps0
	if( fabs(nhel) <= 1 )
	{
		if( pp == 0 )
		{
			e0[0] = complex<double>( 0, 0 );
			e0[1] = complex<double>( 0, 0 );
			e0[2] = complex<double>( 0, 0 );
			e0[3] = complex<double>( 1, 0 );
		}
		else
		{
			emp = p[0]/(tmass*pp);
			e0[0] = complex<double>( pp/tmass, 0 );
			e0[3] = complex<double>( p[3]*emp, 0 );
			
			if( pt != 0 )
			{
				e0[1] = complex<double>( p[1]*emp, 0 );
				e0[2] = complex<double>( p[2]*emp, 0 );
			}
			else
			{
				e0[1] = complex<double>( 0, 0 );
				e0[2] = complex<double>( 0, 0 );
			}
		}
	}
	
	if( nhel == 2 )
	{
		for( j=0; j<4; j++ )
		{
			for( i=0; i<4; i++ ) ft[i][j] = ep[i]*ep[j];
		}
	}
	else if( nhel == -2 )
	{
		for( j=0; j<4; j++ )
		{
			for( i=0; i<4; i++ ) ft[i][j] = em[i]*em[j];
		}
	}
	else if( tmass == 0 )
	{
		for( j=0; j<4; j++ )
		{
			for( i=0; i<4; i++ ) ft[i][j] = 0;
		}
	}
	else if( tmass != 0 )
	{
		if( nhel == 1 )
		{
			for( j=0; j<4; j++ )
			{
				for( i=0; i<4; i++ ) ft[i][j] = sqh*( ep[i]*e0[j] + e0[i]*ep[j] );
			}
		}
		else if( nhel == 0 )
		{
			for( j=0; j<4; j++ )
			{
				for( i=0; i<4; i++ ) ft[i][j] = sqs*( ep[i]*em[j] + em[i]*ep[j]
					+ 2.0*e0[i]*e0[j] );
			}
		}
		else if( nhel == -1 )
		{
			for( j=0; j<4; j++ )
			{
				for( i=0; i<4; i++ ) ft[i][j] = sqh*( em[i]*e0[j] + e0[i]*em[j] );
			}
		}
		else
		{
			std::cerr << "Invalid helicity in txxxxx.\n";
			std::exit(1);
		}
	}
	
	tc[0] = ft[4][0];
	tc[1] = ft[5][0];
	
	for( j=0; j<4; j++ )
	{
		for( i=0; i<4; i++ ) tc[j*4+i+2] = ft[j][i];
	}
}
