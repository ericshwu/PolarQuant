#!/bin/bash
# python3.8,cuda 12.1,pytorch==2.1,torchaudio==2.1.0+cu121,torchvision==0.16.0+cu121

pip install jieba
pip install fuzzywuzzy
pip install rouge
pip install -U jinja2

cd /XXX/models/kivi_quant
pip install -e .

cd /XXX/public
pip install flash_attn-2.2.4+cu122torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl 
pip install triton-3.2.0+gite1697f6b-cp38-cp38-linux_x86_64.whl


