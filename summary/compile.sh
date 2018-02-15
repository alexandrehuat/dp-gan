#!/bin/sh
PATH=$PATH:/Library/TeX/texbin/

# pdflatex
pdflatex --shell-escape -synctex=1 -interaction=batchmode summary.tex

# bibliography
biber summary

# glossary
makeglossaries -l summary

# pdflatex bis
pdflatex --shell-escape -synctex=1  -interaction=batchmode summary.tex
pdflatex --shell-escape -synctex=1 -interaction=batchmode summary.tex

# clean
rm summary.acn summary.acr summary.alg summary.aux summary.bbl summary.bcf summary.blg summary.glg summary.glo summary.gls summary.ist summary.log summary.out summary.run.xml summary.synctex.gz
