#!/usr/bin/env bash
lualatex --halt-on-error principal.tex && \
bibtex principal && \
lualatex principal.tex && \
lualatex principal.tex
