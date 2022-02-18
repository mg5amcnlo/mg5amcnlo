FC=gfortran
FFLAGS=-g -Wall -O2 -Wno-integer-division -Wno-unused-variable
FLD=$(FC)
all:testgrid totlumi
eepdf.f:gridgen
	./gridgen
gridgen:calcpdf.o gridpdfaux.o dgauss.o calcisr.o
eepdf.o:eepdf.f
testgrid:eepdf.o calcpdf.o gridpdfaux.o dfint.o dgauss.o kerset.o calcisr.o
totlumi:eepdf.o calcpdf.o gridpdfaux.o dfint.o dgauss.o kerset.o calcisr.o
