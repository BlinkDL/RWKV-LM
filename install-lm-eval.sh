#!/bin/bash

# install lm-evaluation-harness
git clone git@github.com:EleutherAI/lm-evaluation-harness.git
pushd lm-evaluation-harness
git checkout -b v0.4.3 v0.4.3
pip install -e .    # $(pwd) = lm-evaluation-harness
popd

# install dependencies
pip install -r requirements-lm-eval.txt


