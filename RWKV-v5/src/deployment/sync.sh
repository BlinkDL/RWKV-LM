#!/bin/bash

mkdir -p models
rsync -avz xsel02.cs.virginia.edu:/data/models/orin-deployment/ models/
