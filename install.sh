#!/bin/bash
rm -rf .venv
python3 -m virtualenv --system-site-packages .venv
source .venv/bin/activate
export PIP_CACHE_DIR=.pip_cache/
pip3 install --upgrade pip
pip3 install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip3 install -r requirements.txt
deactivate