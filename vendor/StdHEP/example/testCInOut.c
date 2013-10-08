/*         test C binding to stdhep xdr interface       */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "stdhep.h"
#include "stdlun.h"
#include "stdcnt.h"
#include "stdhep_mcfio.h"

void fill_stdhep(int i);
int write_events();
int read_events();
double calcEnergy(int j);

int main(void)
{
   int nout, nin;
   
   mcfioC_Init();
   nout = write_events();
   nin = read_events();
   printf(" %d events written and %d events read back\n",nout,nin);
   return 0;
}

int write_events()
{
    int ierr, i, j;
    int istr = 0;
    int nevt = 50;
    char fileout[80], title[100];
    char txtfile[80];
    FILE *output;

    strcpy(fileout,"testCInOut.io\0");
    strcpy(title,"test the C interface \0");
    strcpy(txtfile,"testCInOutWrite.txt\0");
    output = fopen(txtfile,"w");
    ierr = StdHepXdrWriteOpen(fileout, title, nevt, istr);
    for (i = 0; i < nevt; i++) {
        fill_stdhep(i+1);
	if( i < 40 ) {
            ierr = StdHepXdrWrite(1,istr);
	} else {
            ierr = StdHepXdrWrite(4,istr);
	}
        if (ierr == 0) {
            fprintf(output," at event %d with %d particles\n",
                     hepevt_.nevhep, hepevt_.nhep);
            j = 17 + i;
            fprintf(output,"  momentum of particle %d is %.3f %.3f %.3f %.3f %.3f\n",
                    j,hepevt_.phep[j][0],hepevt_.phep[j][1],
		    hepevt_.phep[j][2],hepevt_.phep[j][3],hepevt_.phep[j][4]);
	}
    }
    printf(" %d events written\n",i);
    fprintf(output," %d events\n",i);
    StdHepXdrEnd(istr);
    return i;
}

int read_events()
{
    int ierr, i, j, lbl;
    int istr = 0;
    int nevt = 50;
    char filein[80];
    char txtfile[80];
    FILE *output;

    strcpy(filein,"testCInOut.io\0");
    strcpy(txtfile,"testCInOutRead.txt\0");
    output = fopen(txtfile,"w");
    ierr = StdHepXdrReadOpen(filein, nevt, istr);
    for (i = 0; i < nevt; i++) {
        ierr = StdHepXdrRead(&lbl,istr);
        if (ierr == 0) {
            fprintf(output," at event %d with %d particles\n",
                     hepevt_.nevhep, hepevt_.nhep);
            j = 17 + i;
            fprintf(output,"  momentum of particle %d is %.3f %.3f %.3f %.3f %.3f\n",
                    j,hepevt_.phep[j][0],hepevt_.phep[j][1],
		    hepevt_.phep[j][2],hepevt_.phep[j][3],hepevt_.phep[j][4]);
        } else {
            printf(" unexpected end of file after %d events\n",i);
            exit(0);
            }
    }
    printf(" %d events read\n",i);
    fprintf(output," %d events\n",i);
    StdHepXdrEnd(istr);
    return i;
}

void fill_stdhep(int i)
{
    int j, k, num;

    num = i * 20;
    hepevt_.nevhep = i;
    hepevt_.nhep = num;
    for (j = 0; j < num; j++) {
        hepevt_.idhep[j] = 211;
        hepevt_.isthep[j] = 1;
        for (k = 0; k < 2; k++) {
            hepevt_.jmohep[j][k] = j - 1;
            hepevt_.jdahep[j][k] = j + 1;
            }
        for (k = 0; k < 3; k++)
            hepevt_.phep[j][k] = 10.2 * k + 0.31 * i;
	hepevt_.phep[j][4] = .511 + i * 0.04;
	hepevt_.phep[j][3] = calcEnergy(j);
        for (k = 0; k < 4; k++)
            hepevt_.vhep[j][k] = .0001;
        }
}

double calcEnergy(int j)
{
   double p2, m2, e;
   p2 = hepevt_.phep[j][0]*hepevt_.phep[j][0] +
        hepevt_.phep[j][1]*hepevt_.phep[j][1] +
        hepevt_.phep[j][2]*hepevt_.phep[j][2];
   m2 = hepevt_.phep[j][4]*hepevt_.phep[j][4];
   e = sqrt( p2 + m2 );
   return e;
}
