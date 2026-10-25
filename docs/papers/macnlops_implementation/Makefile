.PHONY: all paper complete atlas refresh clean

all: paper complete atlas

paper:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build paper.tex
	cp build/paper.pdf macnlops_paper.pdf

complete:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
	cp build/main.pdf macnlops_complete.pdf

atlas:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build atlas.tex
	cp build/atlas.pdf macnlops_figures.pdf

refresh:
	python3 prepare_paper.py

clean:
	latexmk -C -outdir=build paper.tex main.tex atlas.tex
