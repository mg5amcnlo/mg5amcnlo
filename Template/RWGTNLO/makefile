include ../Source/make_opts

LIBDIR = ../lib/
LINKLIBS =  -L../lib/ -lpdf $(llhapdf)
LIBS = $(LIBDIR)libpdf.$(libext) 

#note only support lhapdf=True
RWGT=rwgt.f $(alphas_functions)

rwgt2py.so: rwgt.f makefile $(LIBS)
	touch __init__.py	
	touch ../__init__.py
	$(F2PY) $(LINKLIBS) $(LDFLAGS) -c  $(RWGT) -m rwgt2py --fcompiler=$(FC)

$(LIBS):
	cd PDF; make



