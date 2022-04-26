#!/usr/bin/env bash
# sudo apt install latexmk texlive-luatex texlive-extras
# sudo apt install texlive-lang-spanish texlive-lang-english texlive-fonts-recommended texlive-fonts-extra texlive-science texlive-bibtex-extra


#latexmk -pvc -pdf -outdir=build -luatex="ppluatex -q -- -shell-escape -synctex=1 -interaction=nonstopmode" main.tex
#latexmk -pvc -pdf -outdir=build -pdflatex="ppluatex -q -- %O %S" -dvi- -ps- main.tex
#latexmk -silent -pvc -pdf  -outdir=build -pdflatex="lualatex -silent  %O %S" -dvi- -ps- main.tex
#latexmk -time -silent  -halt-on-error -pvc -lualatex main.tex
latexmk -time -silent  -halt-on-error -pvc -pdf -lualatex main.tex
#latexmk -pvc -pdf -outdir=build -pdflatex="ppdflatex -q -- -shell-escape -synctex=1 -interaction=nonstopmode" main.tex



#texloganalyzer
