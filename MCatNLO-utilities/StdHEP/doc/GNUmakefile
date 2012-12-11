# build stdhep document

MAKE=make

TEX_SOURCE = stdmain.tex \
             intro.tex body.tex routines.tex history.tex refer.tex \
             elem.tex meson.tex baryon.tex isthep.tex

all: stdmain.ps stdmain.pdf

# just the tables
table: stdtab.pdf

stdmain.ps: $(TEX_SOURCE)
	latex stdmain
	latex stdmain
	latex stdmain
	rm -f stdmain.ps stdmain.pdf
	dvips -P pdf stdmain

stdmain.pdf: stdmain.ps
	ps2pdf stdmain.ps

stdtab.ps:  stdtab.tex elem.tex meson.tex baryon.tex 
	latex stdtab
	latex stdtab
	latex stdtab
	rm -f stdtab.ps stdtab.pdf
	dvips -P pdf stdtab

stdtab.pdf: stdtab.ps
	ps2pdf stdtab.ps

clean:
	rm -f *.aux *.log *.dvi *.toc

realclean:
	rm -f stdmain.ps stdmain.pdf
	rm -f stdtab.ps stdtab.pdf
	rm -f *.aux *.log *.dvi *.toc

