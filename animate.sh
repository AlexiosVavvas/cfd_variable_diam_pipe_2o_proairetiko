#!/bin/bash
# Move files to current directory
mv animation_scripts/animate_rho.py ./
mv animation_scripts/animate_u.py ./
mv animation_scripts/animate_p.py ./

python animate_rho.py & python animate_u.py & python animate_p.py

# Move files back
mv animate_rho.py animation_scripts/
mv animate_u.py animation_scripts/
mv animate_p.py animation_scripts/
