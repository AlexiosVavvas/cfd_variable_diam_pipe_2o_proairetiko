#!/bin/bash
rm results/animation/*.png

cp animation_scripts/multi_animate.py ./
cp animation_scripts/turn_anim_frames_to_gif.py ./

python multi_animate.py
python turn_anim_frames_to_gif.py

# copy everything back
mv multi_animate.py animation_scripts/
mv turn_anim_frames_to_gif.py animation_scripts/
