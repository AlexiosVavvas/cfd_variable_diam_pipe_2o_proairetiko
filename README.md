# cfd_variable_diam_pipe_2o_proairetiko

Main file: calculateFlowSolution.py

1. Define parameters such as a, b, c, k and flow conditions such as P0, T0, Patm, ...
    Parameters in  : constantsToUse.py
    Flow conditions: calculateFlowSolution.py 
2. Run the file with:
    python calculateFlowSolution.py
3. Temp visualise results with 
    ./animate.sh
4. Generate animation frames and GIF 
    ./generate_animation_frames.sh

multi_animate.py is them main file handling the drawing of each frame
turn_anim_frames_to_gif.py does what it sais to do

Or run everything, both calculations and GIF generation by running:
	./run.sh

Libraries Needed:
    - numpy
    - matplotlib

![Flow Solution](https://github.com/AlexiosVavvas/cfd_variable_diam_pipe_2o_proairetiko/blob/1c1f451ada9649ce9d775ed44f0572d1e66efa93/results/final1/output_animation.gif)