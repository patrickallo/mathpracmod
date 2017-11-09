#!/bin/bash

jupyter nbconvert --to=latex --template=revtex_nocode.tpl pmagain.ipynb
pdflatex pmagain.tex
bibtex pmagain.aux
pdflatex pmagain.tex
pdflatex pmagain.tex

rm *.bbl *.aux *.blg *.log *.out *Notes.bib #*.tex


