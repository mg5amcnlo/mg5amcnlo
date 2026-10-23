.PHONY: all tables clean

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

# Optional: requires the original production records. Not needed to build PDF.
tables:
	python3 make_tables.py

clean:
	latexmk -C -outdir=build main.tex
