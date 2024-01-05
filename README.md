# cfd_variable_diam_pipe_2o_proairetiko

# Flow Solution Calculation and Visualization

This guide explains how to calculate flow solutions, visualize results, and generate animations in GIF format.

## Setup

1. **Define Parameters**:
   - Modify parameters such as `a`, `b`, `c`, `k` in the file `constantsToUse.py`.
   - Flow conditions like `P0`, `T0`, `Patm`, etc., are defined in `calculateFlowSolution.py`.

2. **Install Required Libraries**:
   Ensure you have the following Python libraries installed:
   \`\`\`bash
   numpy
   matplotlib
   \`\`\`

## Running the Calculations

1. **Calculate Flow Solution**:
   Run the calculation with the following command:
   \`\`\`bash
   python calculateFlowSolution.py
   \`\`\`

2. **Temporary Visualization**:
   To quickly visualize the results, use:
   \`\`\`bash
   ./animate.sh
   \`\`\`

## Generating Animation

1. **Create Animation Frames and GIF**:
   Execute the following script to generate animation frames and convert them into a GIF:
   \`\`\`bash
   ./generate_animation_frames.sh
   \`\`\`
   - `multi_animate.py` is the main file that handles the drawing of each frame.
   - `turn_anim_frames_to_gif.py` converts the frames into a GIF.

2. **Comprehensive Run Script**:
   Alternatively, to run both the calculations and GIF generation, use:
   \`\`\`bash
   ./run.sh
   \`\`\`

Follow these steps to successfully calculate flow solutions and generate animations.


![Flow Solution](https://github.com/AlexiosVavvas/cfd_variable_diam_pipe_2o_proairetiko/blob/1c1f451ada9649ce9d775ed44f0572d1e66efa93/results/final1/output_animation.gif)