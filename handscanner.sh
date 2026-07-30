#!/bin/bash
# O "nohup" e o "&" garantem que rode em segundo plano sem prender o terminal
# DISPLAY=:0 garante que a janela aparece mesmo rodando em background
export DISPLAY=:0
nohup ~/Documentos/pessoal/handscanner/venv/bin/python ~/Documentos/pessoal/handscanner/main.py > /dev/null 2>&1 &
