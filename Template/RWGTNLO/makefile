export lhapdf=True
include ../Source/make_opts

LIBDIR = ../lib/
LINKLIBS =  -L../lib/ -lpdf $(lhapdf)
LIBS = $(LIBDIR)libpdf.$(libext) 

RWGT=rwgt.f alfas_functions_lhapdf.f


rwgt2py.so: rwgt.f makefile $(LIBS)
	touch __init__.py	
	touch ../__init__.py
	$(F2PY) $(LINKLIBS) -c  $(RWGT) -m rwgt2py --fcompiler=$(FC)

$(LIBS):
	cd PDF; make



